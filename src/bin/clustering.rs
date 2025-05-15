#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(cmp_minmax)]
#![feature(generic_arg_infer)]

use std::cmp::minmax;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use ordered_float::NotNan;

use clap::{Parser, ValueEnum};

use priority_queue::PriorityQueue;

use texture_to_vert_colors::manifold::{CollapsibleManifold, EdgeKind};
use texture_to_vert_colors::sym::SymMatrix3;
use texture_to_vert_colors::{F, add, dist, dot, kmul, l1dist, normalize, poly_area, sub};

use indicatif::ProgressBar;

const PI: F = std::f64::consts::PI as F;

#[derive(Clone, Parser, Debug)]
#[clap(group(
  clap::ArgGroup::new("target")
    .required(true)
    .args(&["target_num_charts", "error_bound"])
))]
pub struct Args {
    /// Input OBJ file.
    #[arg(short, long, required = true)]
    pub input: String,

    /// Output PLY file.
    #[arg(short, long, required = true)]
    pub output: String,

    /// Output PLY file, where clusters are colored by their clustering instead of average color
    #[arg(short, long, default_value_t = String::new())]
    pub cluster_vis: String,

    /// Output PLY file, to visualize the optimized eigenvalue of each cluster normalized to the
    /// range [0,1]
    #[arg(long, default_value_t = String::new())]
    pub eigen_vis: String,

    /// Target output number of charts.
    #[arg(short, long, group = "target")]
    pub target_num_charts: Option<usize>,

    #[arg(long, default_value_t = Eigenvalue::Zero)]
    eigenvalue: Eigenvalue,

    /// How long before stopping
    #[arg(long, short='e', group="target", default_value_t = F::NEG_INFINITY)]
    error_bound: F,

    /// Do not use area weighting.
    #[arg(long)]
    no_area_weight: bool,

    /// Weight for each edge
    #[arg(long, default_value_t = 1e-2)]
    edge_weight: F,

    /// How much to weigh colors
    #[arg(long, default_value_t = 0.1)]
    color_weight: F,

    /// Absolute value difference permitted between eigenvalues, before switching to color
    /// checking.
    #[arg(long, default_value_t = 1e-4)]
    eigen_eps: F,

    /// Absolute value difference permitted between colors, before switching to bd length
    /// checking.
    #[arg(long, default_value_t = 1e-6)]
    color_eps: F,

    /// Where to output stats (unused currently)
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// What ordering is preferred when constructing a cluster
    #[arg(long, default_value_t = OrderingKind::GeomColorShape)]
    ordering: OrderingKind,

    /// What metric to use when evaluating the quality of output cluster shape
    #[arg(long, default_value_t = ShapeMetric::AngleDeviation)]
    shape_metric: ShapeMetric,

    /// Prefer opposite effect of shape metric (For jokes)
    /// Mey kill perf.
    #[arg(long, hide = true)]
    invert_shape: bool,
}

pub fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if !args.output.ends_with(".ply") {
        eprintln!("[WARN]: Output will not be colored if output format is not PLY");
    }

    let scene = pars3d::load(&args.input)?;
    let mut mesh = scene.into_flattened_mesh();
    let (s, t) = mesh.normalize();
    let (cs, ct) = mesh.normalize_colors();

    let start = std::time::Instant::now();
    let (face_charts, m) =
        face_clustering(&mesh.v, &mesh.vert_colors, &mesh.f, mesh.f.len(), &args);
    println!("[INFO]: Took {:?} for clustering", start.elapsed());

    let mut remap = HashMap::new();
    let mut num_charts = 0;
    for &f in &face_charts {
        match remap.entry(f) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                v.insert(num_charts);
                num_charts += 1;
            }
        }
    }
    eprintln!("[INFO]: Output # Charts = {}", remap.len());

    let wireframe_parts = pars3d::visualization::face_segmentation_wireframes(
        |fi| mesh.f[fi].as_slice(),
        |fi| face_charts[fi],
        mesh.f.len(),
        &mesh.v,
        3e-4,
    );
    let mut wireframe_mesh = pars3d::visualization::wireframe_to_mesh(wireframe_parts);
    wireframe_mesh.denormalize(s, t);

    if !args.cluster_vis.is_empty() {
        let face_coloring = pars3d::visualization::greedy_face_coloring(
            |i| face_charts[i],
            face_charts.len(),
            |i, j| m.is_adj(i, j),
            &pars3d::coloring::HIGH_CONTRAST,
        );

        let mut colored_mesh = mesh.with_face_coloring(&face_coloring);
        colored_mesh.denormalize(s, t);
        colored_mesh.append(&mut wireframe_mesh.clone());
        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.cluster_vis, &out_scene)?;
    }

    if !args.eigen_vis.is_empty() {
        let eigenvalues = m
            .vertices()
            .map(|(vi, (q, _, _))| {
                let eigens = q.eigen_sorted().0;
                (vi, args.eigenvalue.apply(eigens))
            })
            .collect::<std::collections::BTreeMap<_, _>>();

        let mut face_eigens = (0..mesh.f.len())
            .map(|i| eigenvalues[&face_charts[i]])
            .collect::<Vec<F>>();
        let [min_e, max_e] = face_eigens
            .iter()
            .fold([F::INFINITY, F::NEG_INFINITY], |[l, h], &n| {
                [l.min(n), h.max(n)]
            });
        assert!(max_e.is_finite());
        assert!(min_e.is_finite());
        assert!(max_e >= min_e);
        let r = max_e - min_e;
        println!("[INFO] eigenvalues in range [{min_e}, {max_e}]");
        for e in &mut face_eigens {
            *e -= min_e;
            if r != 0. {
                *e /= r;
            }
        }
        let face_colors = face_eigens
            .into_iter()
            .map(pars3d::coloring::magma)
            .collect::<Vec<_>>();
        let mut colored_mesh = mesh.with_face_coloring(&face_colors);

        colored_mesh.denormalize(s, t);

        colored_mesh.append(&mut wireframe_mesh.clone());

        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.eigen_vis, &out_scene)?;
    }

    {
        let face_colors = (0..mesh.f.len())
            .map(|i| m.get(face_charts[i]).1)
            .collect::<Vec<[F; 3]>>();

        let mut colored_mesh = mesh.with_face_coloring(&face_colors);
        colored_mesh.denormalize(s, t);
        colored_mesh.denormalize_colors(cs, ct);

        colored_mesh.append(&mut wireframe_mesh);

        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.output, &out_scene)?;
    }

    Ok(())
}

pub fn face_clustering<'a, 'b>(
    vs: &'a [[F; 3]],
    vcs: &'a [[F; 3]],
    fs: &'a [pars3d::FaceKind],
    nf: usize,
    args: &Args,
) -> (
    Vec<usize>,
    CollapsibleManifold<(SymMatrix3, [F; 3], F), union_find::AtomicUnionFind>,
) {
    // face normals
    let f_n = (0..nf).map(|fi| fs[fi].normal(&vs)).collect::<Vec<_>>();

    // vertex normals
    let mut v_ns = vec![[0.; 3]; vs.len()];
    for (fi, f_n) in f_n.iter().copied().enumerate() {
        for &vi in fs[fi].as_slice() {
            v_ns[vi] = add(v_ns[vi], f_n);
        }
    }

    for v_n in v_ns.iter_mut() {
        *v_n = normalize(*v_n);
    }

    let mut face_area = vec![0.; nf];
    let mut edge_face_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();

    for (fi, f) in (0..nf).map(|fi| &fs[fi]).enumerate() {
        face_area[fi] = if !args.no_area_weight {
            poly_area(f.as_slice().iter().map(|&v| vs[v]))
        } else {
            1.
        };
        let f = f.as_slice();
        for i in 0..f.len() {
            let n = (i + 1) % f.len();
            let e = minmax(f[i], f[n]);
            edge_face_adj
                .entry(e)
                .and_modify(|old| {
                    old.insert(fi);
                })
                .or_insert_with(|| EdgeKind::Boundary(fi));
        }
    }

    let mut vertex_adj = vec![vec![]; vs.len()];
    for &[e0, e1] in edge_face_adj.keys() {
        if e0 == e1 {
            continue;
        }
        assert!(!vertex_adj[e0].contains(&e1));
        vertex_adj[e0].push(e1);
        assert!(!vertex_adj[e1].contains(&e0));
        vertex_adj[e1].push(e0);
    }

    // for each real mesh edge, track which charts it is a part of.
    // Each edge can be at most part of 2 charts.
    let mut m = CollapsibleManifold::<(SymMatrix3, [F; 3], F), _>::atomic_new_with(nf, |fi| {
        let n = f_n[fi];
        let area = face_area[fi];
        let q = SymMatrix3::outer(n);

        let f = &fs[fi];
        let per_vert_weight = area / f.len() as F;
        assert!(!f.is_empty());
        assert!(area.is_sign_positive());
        let avg_color = if area == 0. {
            [0.; 3]
        } else {
            let sum_color = f
                .as_slice()
                .iter()
                .copied()
                .map(|vi| vcs[vi])
                .map(|vc| kmul(per_vert_weight, vc))
                .fold([0.; 3], add);
            kmul(area.recip(), sum_color)
        };
        assert!(avg_color.iter().copied().all(F::is_finite));

        (q * area, avg_color, area)
    });

    for e in edge_face_adj.values() {
        let s = e.as_slice();
        // connect all adjacent faces
        for i in 0..s.len() {
            for j in i + 1..s.len() {
                assert_ne!(s[i], s[j]);
                m.add_edge(s[i], s[j]);
            }
        }
    }

    // from this point, edge_face_adj now corresponds to the chart of each mesh.
    eprintln!("[INFO]: Input # vertices = {}", vs.len());
    eprintln!("[INFO]: Input # faces = {}", nf);

    // for each chart pair (ordered minmax)
    // store edges between these charts
    // update this on every iteration before merging
    let mut shared_chart_edges: HashMap<[usize; 2], Vec<[usize; 2]>> = HashMap::new();
    for (fi, _) in m.vertices() {
        let f0 = &fs[fi];
        for f_adj in m.vertex_adj(fi) {
            assert_ne!(fi, f_adj);
            if f_adj < fi {
                continue;
            }
            let adj_f = &fs[f_adj];

            assert!(fi < f_adj);
            let shared_e = shared_chart_edges.entry([fi, f_adj]).or_default();
            assert!(shared_e.is_empty());
            shared_e.extend(f0.shared_edges(&adj_f));
        }
    }

    let mut barycentric_area = vec![];
    pars3d::geom_processing::barycentric_areas(fs, vs, &mut barycentric_area);

    macro_rules! straightness_deviation {
        ($pi: expr, $ci: expr, $ni: expr) => {{
            let [p, c, n] = [$pi, $ci, $ni].map(|vi| vs[vi]);
            let e0 = normalize(sub(p, c));
            let e1 = normalize(sub(n, c));
            let theta = dot(e0, e1).clamp(-1., 1.).acos() / PI;
            // deviation in the range [0, 1]
            let dev = 1. - theta;
            assert!((0.0..=1.0).contains(&dev));
            // close to L0
            dev //.sqrt()
        }};
    }
    // for each chart
    // store all pairs of incident edges on the boundary of that chart along with the summed
    // deviation from pi for each pair. TODO test squared distance vs abs distance
    let mut incident_angles: Vec<BTreeMap<usize, Vec<[usize; 2]>>> =
        vec![BTreeMap::new(); m.num_vertices()];
    //let mut total_incident_angle = vec![0.; m.num_vertices()];
    for (fi, _) in m.vertices() {
        let f = &fs[fi];
        for [pi, ci, ni] in f.incident_edges() {
            assert_ne!(pi, ci);
            assert_ne!(ci, ni);
            assert_ne!(pi, ni);
            let prev = incident_angles[fi].entry(ci).or_default();
            assert!(!prev.contains(&minmax(pi, ni)));
            prev.push(minmax(pi, ni));
        }
    }

    let mut pq = PriorityQueue::new();

    let mut prev_eigens = vec![0.; nf];
    for (vi, (q, _, _)) in m.vertices() {
        let eigens = q.eigen_sorted().0;
        prev_eigens[vi] = args.eigenvalue.apply(eigens);
    }

    macro_rules! cost_of_edge {
        ($e0:expr, $e1: expr) => {{
            let [e0, e1] = std::cmp::minmax($e0, $e1);

            assert!(!m.is_deleted(e0));
            assert!(!m.is_deleted(e1));
            let (q0, avg_color0, area0) = *m.get(e0);
            let (q1, avg_color1, area1) = *m.get(e1);
            // if the charts are merged together then we don't need to combine any additional
            // costs, otherwise we need to add them.
            let q_new = q0 + q1;

            // sorted eigenvalues
            let neigens = q_new.eigen_sorted().0;
            let evn = args.eigenvalue.apply(neigens);

            let ev0 = prev_eigens[e0];
            let ev1 = prev_eigens[e1];
            let [evn, ev0, ev1] = [evn, ev0, ev1].map(F::abs);
            // subtract previous values here (only penalize added deviation)?
            let cost = evn - (ev0 + ev1);

            // also need to compute deviation from constant color
            let new_area = area0 + area1;
            let total_clr = add(kmul(area0, avg_color0), kmul(area1, avg_color1));
            assert!(new_area >= 0.);
            let new_avg = if new_area == 0. {
                [0.; 3]
            } else {
                kmul(new_area.recip(), total_clr)
            };

            // TODO use a different color distance here?
            let clr_diff =
                area0 * l1dist(new_avg, avg_color0) + area1 * l1dist(new_avg, avg_color1);
            assert!(clr_diff.is_finite());
            assert!(clr_diff >= 0.);

            let shape_metric = match args.shape_metric {
              // if either of these is 0., then there will be no space for optimizing shape
              // metrics.
              _ if args.eigen_eps <= 0. || args.color_eps <= 0. => 0.,
              ShapeMetric::None => 0.,
              ShapeMetric::Convexity => todo!(),
              //ShapeMetric::Planarity => neigens[1],
              ShapeMetric::BoundaryLength => {
                // Tested here: dividing by new area seems to make it prefer rounder charts,
                // whereas just plain seems to be ok with skinny charts.
                // Maybe that's fine?
                shared_chart_edges[&[e0, e1]]
                    .iter()
                    .map(|&[e0, e1]| dist(vs[e0], vs[e1]))
                    .sum::<F>()
              }
              ShapeMetric::AngleDeviation => {
                assert!(e1 < incident_angles.len());
                assert_ne!(e0, e1);
                let [ia_e0,  ia_e1] = incident_angles.get_disjoint_mut([e0, e1]).unwrap();
                let angle_delta = ia_e0
                    .iter_mut()
                    .filter_map(|(&vi, pns)| {
                        // for vertices which are contained in both chart, compute the new
                        // straightness for each vertex.

                        // NOTE: if there was no entry before the cost is the same

                        // TODO figure out if there's a way to do this without a buffer?
                        // NOTE: the common case (1 element) probably does not need a buffer, maybe that can be
                        // optimized and the rare case can allocate
                        let opns : &mut [[usize;2]] = ia_e1.get_mut(&vi)?;
                        match (pns.as_mut_slice(), opns) {
                          (&mut [], _) | (_, &mut []) => return None,
                          (&mut [pn], &mut [opn]) => {
                            let [new_p, new_n] = merge_wedges(pn, opn)?;
                            let prev = straightness_deviation!(pn[0],vi, pn[1]) +
                              straightness_deviation!(opn[0],vi, opn[1]);
                            let new = straightness_deviation!(new_p, vi, new_n);
                            Some(new - prev)
                          }
                          (&mut [pn], mut rest) | (mut rest, &mut [pn]) => {
                            let (to_keep, new) = new_wedges(&mut rest, pn);
                            let mut new_angle = new
                              .map(|[new_p, new_n]| straightness_deviation!(new_p, vi, new_n))
                              .unwrap_or(0.);
                            for &[p,n] in &rest[0..to_keep] {
                              new_angle += straightness_deviation!(p, vi, n);
                            }
                            let prev = straightness_deviation!(pn[0], vi, pn[1])
                              + rest.iter()
                                  .map(|&[p,n]| straightness_deviation!(p, vi, n))
                                  .sum::<F>();
                            Some(new_angle - prev)
                          }
                          (_, opns) => {
                            let mut opns: Vec<[usize; 2]> = opns.to_vec();
                            let e0_angle_dev = pns.iter()
                              .map(|&[p,n]| straightness_deviation!(p, vi, n))
                              .sum::<F>();
                            let e1_angle_dev = opns.iter()
                              .map(|&[p,n]| straightness_deviation!(p, vi, n))
                              .sum::<F>();

                            for &mut pn in pns {
                              let (to_keep, new) = new_wedges(&mut opns, pn);
                              opns.truncate(to_keep);
                              if let Some(new) = new {
                                opns.push(new);
                              }
                            }

                            let new = opns.iter().map(|&[pi, ni]| straightness_deviation!(pi, vi, ni)).sum::<F>();

                            Some(new - (e0_angle_dev + e1_angle_dev))
                          },
                        }
                    })
                    .sum::<F>();
                  -angle_delta
              }
            };


            let l = if args.invert_shape {
                shape_metric
            } else {
                -shape_metric
            };

            let arr = match args.ordering {
                OrderingKind::GeomColorShape => [cost, clr_diff, l],
                OrderingKind::ColorGeomShape => [clr_diff, cost, l],
                OrderingKind::GeomShapeColor => [cost, l, clr_diff],
            };

            arr.map(|v| NotNan::new(-v).unwrap())
        }};
    }

    // This loop can be very slow for processing all components together since it is a dense
    // mesh.
    let ec = Mutex::new(&mut pq);
    m.edges()
        // since just deduplicated edges, only check when e0 < e1
        .filter(|[e0, e1]| e0 < e1)
        .for_each(|[e0, e1]| {
            let c = cost_of_edge!(e0, e1);
            ec.lock().unwrap().push(minmax(e0, e1), c);
        });

    let p = ProgressBar::new(m.num_vertices() as u64);
    p.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} (Elapsed = {elapsed_precise})")
            .unwrap(),
    );

    let mut pq2 = PriorityQueue::new();
    // fml this is some bullshit I need to rewrite
    let mut pq3 = PriorityQueue::new();
    'outer: while let Some((e, [dev, cd, len])) = pq.pop() {
        assert!(pq2.is_empty());
        pq2.push(e, [cd, dev, len]);
        while let Some((e, [cd, dev, len])) = pq2.pop() {
            assert!(pq3.is_empty());
            pq3.push(e, [len, cd, dev]);
            // hohoho kms
            while let Some(([e0, e1], [_len, cd, dev])) = pq3.pop() {
                assert!(e0 < e1);
                assert_ne!(e0, e1);
                // Termination criteria
                let reached = args
                    .target_num_charts
                    .map(|tf| m.num_vertices() <= tf)
                    .unwrap_or(false)
                    || (*dev < args.error_bound);

                if reached {
                    break 'outer;
                }
                if m.is_deleted(e0) || m.is_deleted(e1) {
                    continue;
                }

                // --- Commit

                assert!(shared_chart_edges.remove(&[e0, e1]).is_some());

                // update shared edges between e0 - adj to instead be between e1 - adj
                for adj in m.vertex_adj(e0) {
                    if adj == e1 {
                        continue;
                    }
                    let c = minmax(adj, e0);
                    let mut sce = shared_chart_edges.remove(&c).unwrap();
                    let nc = minmax(adj, e1);
                    // should not have any dupes
                    shared_chart_edges.entry(nc).or_default().append(&mut sce);
                }

                if args.shape_metric == ShapeMetric::AngleDeviation {
                    for (c, pns) in std::mem::take(&mut incident_angles[e0]) {
                        use std::collections::btree_map::Entry as BEntry;
                        match incident_angles[e1].entry(c) {
                            BEntry::Vacant(v) => {
                                v.insert(pns);
                            }
                            BEntry::Occupied(mut o) => {
                                let opn = o.get_mut();
                                for pn in pns {
                                    let (new_len, new_wedge) = new_wedges(opn, pn);
                                    opn.truncate(new_len);
                                    if let Some(new_wedge) = new_wedge {
                                        opn.push(new_wedge);
                                    }
                                }
                            }
                        }
                    }
                }

                m.merge(e0, e1, |&(sa, ca, area_a), &(sb, cb, area_b)| {
                    assert!(area_a >= 0.);
                    assert!(area_b >= 0.);
                    let new_area = area_a + area_b;
                    assert!(new_area >= 0.);
                    let total_clr = add(kmul(area_a, ca), kmul(area_b, cb));
                    let new_avg = if new_area == 0. {
                        [0.; 3]
                    } else {
                        kmul(new_area.recip(), total_clr)
                    };
                    (sa + sb, new_avg, new_area)
                });

                assert!(m.is_deleted(e0));
                assert!(!m.is_deleted(e1));
                prev_eigens[e1] = args.eigenvalue.apply(m.get(e1).0.eigen_sorted().0);

                for adj in m.vertex_adj(e1) {
                    let c = cost_of_edge!(adj, e1);
                    let e = minmax(e1, adj);

                    pq2.remove(&e);
                    pq3.remove(&e);
                    pq.push(e, c);
                }

                let first_eps = args.ordering.first_eps(args.eigen_eps, args.color_eps);
                while let Some((ne, [ndev, ncd, nlen])) =
                    pq.pop_if(|_, [ndev, _, _]| approx_eq(**ndev, *dev, first_eps))
                {
                    pq2.push(ne, [ncd, ndev, nlen]);
                }

                let snd_eps = args.ordering.second_eps(args.eigen_eps, args.color_eps);
                while let Some((ne, [ncd, ndev, nlen])) =
                    pq2.pop_if(|_, [ncd, _, _]| approx_eq(**ncd, *cd, snd_eps))
                {
                    pq3.push(ne, [nlen, ncd, ndev]);
                }

                p.set_position(m.num_vertices() as u64);
            }
        }
    }

    // for each face which chart is it assigned to?
    let charts = (0..nf).map(|i| m.get_new_vertex(i)).collect::<Vec<_>>();
    (charts, m)
}

pub fn edges(f: &[usize]) -> impl Iterator<Item = [usize; 2]> + '_ {
    (0..f.len()).map(|i| [f[i], f[(i + 1) % f.len()]])
}

macro_rules! impl_display {
  ($name: ident, $($kind: ident => $disp: expr),+$(,)?) => {
    impl std::fmt::Display for $name {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use $name::*;
            let s = match self {
                $($kind => $disp,)+
            };
            write!(f, "{s}")
        }
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum Eigenvalue {
    Zero,
    One,
    Two,
}

impl Eigenvalue {
    pub fn apply(self, [e0, e1, e2]: [F; 3]) -> F {
        match self {
            Eigenvalue::Zero => e0,
            Eigenvalue::One => e1,
            Eigenvalue::Two => e2,
        }
    }
}

fn merge_wedges(a: [usize; 2], b: [usize; 2]) -> Option<[usize; 2]> {
    match [a, b] {
        [[w0, w1], [c0, c1]]
        | [[w0, w1], [c1, c0]]
        | [[w1, w0], [c0, c1]]
        | [[w1, w0], [c1, c0]]
            if w0 == c0 =>
        {
            Some(minmax(w1, c1))
        }
        _ => None,
    }
}

// TODO decide if this only works for meshes without repeat faces or not
/// Sorts this set of set wedges such that all those which will be deleted are at the end.
/// Then, the last (return value) can be removed and replaced with the optional return value.
fn new_wedges(wedges: &mut [[usize; 2]], mut curr: [usize; 2]) -> (usize, Option<[usize; 2]>) {
    assert!(curr[0] < curr[1]);
    assert!(wedges.iter().all(|[a, b]| a < b));
    let mut to_keep = wedges.len();
    while to_keep > 0 {
        let mut any = false;
        for i in 0..to_keep {
            let wedge = wedges[i];
            // if an existing wedge exactly matches, delete both of them and return None.
            if wedge == curr {
                wedges.swap(i, to_keep - 1);
                return (to_keep - 1, None);
            }

            // otherwise check if any of the values are shared, and delete them if so
            if let Some(w) = merge_wedges(wedge, curr) {
                wedges.swap(i, to_keep - 1);
                to_keep -= 1;
                curr = w;
                any = true;
                break;
            }
        }
        if !any {
            break;
        }
    }
    (to_keep, Some(curr))
}

#[test]
fn test_wedges() {
    let mut wedges = [[0, 1], [2, 3], [7, 8]];
    assert_eq!(new_wedges(&mut wedges, [1, 2]), (1, Some([0, 3])));
    assert_eq!(wedges[0], [7, 8]);

    let mut wedges = [[3, 4], [1, 2], [6, 7]];
    assert_eq!(new_wedges(&mut wedges, [1, 2]), (2, None));
    assert_eq!(wedges[2], [1, 2]);

    let mut wedges = [[3, 4], [1, 2], [6, 7]];
    assert_eq!(new_wedges(&mut wedges, [0, 10]), (3, Some([0, 10])));
}

impl_display!(Eigenvalue, Zero => "zero", One => "one", Two => "two");

/// What order to prefer when simplifying.
/// It is assumed that length is final (has no eps)
#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum OrderingKind {
    GeomColorShape,
    ColorGeomShape,
    GeomShapeColor,
}

impl OrderingKind {
    pub fn first_eps(&self, geom_eps: F, color_eps: F) -> F {
        match self {
            OrderingKind::GeomColorShape | OrderingKind::GeomShapeColor => geom_eps,
            OrderingKind::ColorGeomShape => color_eps,
        }
    }
    pub fn second_eps(&self, geom_eps: F, color_eps: F) -> F {
        match self {
            OrderingKind::GeomColorShape => color_eps,
            OrderingKind::ColorGeomShape => geom_eps,
            OrderingKind::GeomShapeColor => 0.,
        }
    }
}

impl_display!(
  OrderingKind,
  GeomColorShape => "geom-color-shape",
  ColorGeomShape => "color-geom-shape",
  GeomShapeColor => "geom-shape-color",
);

/// What metric to use to evaluate the quality of output cluster shapes.
#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum ShapeMetric {
    /// The total boundary length of each cluster
    BoundaryLength,
    /// The angle at each boundary point in the cluster's deviation from 180 degrees.
    /// (Prefer straight edges)
    AngleDeviation,
    // How flat each shape is
    //Planarity,

    /// This is a simple metric which instead measures the connected boundary of adjacent faces within each
    /// chart?
    Convexity,

    None,
}

impl_display!(
  ShapeMetric,
  BoundaryLength => "boundary-length",
  AngleDeviation => "angle-deviation",
  //Planarity => "planarity",
  Convexity => "convexity",
  None => "none",
);

pub fn approx_eq(a: F, b: F, eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < eps
}
