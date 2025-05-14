#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(cmp_minmax)]
#![feature(generic_arg_infer)]

use std::array::from_fn;
use std::cmp::minmax;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

use ordered_float::NotNan;

use clap::{Parser, ValueEnum};

use priority_queue::PriorityQueue;

use texture_to_vert_colors::manifold::{CollapsibleManifold, EdgeKind};
use texture_to_vert_colors::sym::SymMatrix3;
use texture_to_vert_colors::{F, add, cross, kmul, l1dist, normalize, poly_area, sub};

use indicatif::ProgressBar;

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
    pub eigenvalue_vis: String,

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

    /// Absolute value difference permitted between eigenvalues
    #[arg(long, default_value_t = 1e-2)]
    abs_eps: F,

    /// Where to output stats (unused currently)
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// Prioritize color first over eigenvalue
    #[arg(long)]
    color_over_geometry: bool,
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

    let (face_charts, m) = face_clustering(
        mesh.v.as_slice(),
        mesh.vert_colors.as_slice(),
        |fi| &mesh.f[fi].as_slice(),
        mesh.f.len(),
        &args,
    );

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

    if !args.eigenvalue_vis.is_empty() {
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
        pars3d::save(&args.eigenvalue_vis, &out_scene)?;
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

pub fn face_clustering<'a>(
    vs: &'a [[F; 3]],
    vcs: &'a [[F; 3]],
    fs: impl Fn(usize) -> &'a [usize] + Sync + Send,
    nf: usize,
    args: &Args,
) -> (
    Vec<usize>,
    CollapsibleManifold<(SymMatrix3, [F; 3], F), union_find::AtomicUnionFind>,
) {
    // face normals
    let f_n = (0..nf)
        .map(|fi| {
            let f = fs(fi);
            assert!(f.len() > 2);
            if f.len() == 4 {
                let [v0, v1, v2, v3] = from_fn(|i| vs[f[i]]);
                return normalize(cross(sub(v0, v2), sub(v1, v3)));
            }
            // just assume it's a flat polygon.
            let [v0, v1, v2] = from_fn(|i| vs[f[i]]);
            normalize(cross(sub(v1, v0), sub(v2, v0)))
        })
        .collect::<Vec<_>>();

    // vertex normals
    let mut v_ns = vec![[0.; 3]; vs.len()];
    for (fi, f_n) in f_n.iter().copied().enumerate() {
        for &vi in fs(fi) {
            v_ns[vi] = add(v_ns[vi], f_n);
        }
    }

    for v_n in v_ns.iter_mut() {
        *v_n = normalize(*v_n);
    }

    let mut face_area = vec![0.; nf];
    let mut edge_face_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();

    for (fi, f) in (0..nf).map(&fs).enumerate() {
        face_area[fi] = if !args.no_area_weight {
            poly_area(f.iter().map(|&v| vs[v]))
        } else {
            1.
        };
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

        let f = fs(fi);
        let per_vert_weight = area / f.len() as F;
        assert!(!f.is_empty());
        assert!(area.is_sign_positive());
        let avg_color = if area == 0. {
            [0.; 3]
        } else {
            let sum_color = f
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
    let mut shared_chart_edges: HashMap<[usize; 2], Vec<[usize; 2]>> = HashMap::new();
    for (fi, _) in m.vertices() {
        let f0 = pars3d::FaceKind::from(fs(fi));
        for f_adj in m.vertex_adj(fi) {
            assert_ne!(fi, f_adj);
            if f_adj < fi {
                continue;
            }
            let adj_f = pars3d::FaceKind::from(fs(f_adj));

            let shared_e = shared_chart_edges.entry([fi, f_adj]).or_default();
            assert!(shared_e.is_empty());
            shared_e.extend(f0.shared_edges(&adj_f));
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

            let clr_diff =
                area0 * l1dist(new_avg, avg_color0) + area1 * l1dist(new_avg, avg_color1);
            assert!(
                clr_diff.is_finite(),
                "{clr_diff} {area0} {area1} {new_area} {new_avg:?} {avg_color0:?} {avg_color1:?}"
            );
            assert!(clr_diff >= 0.);

            let arr = if args.color_over_geometry {
                [clr_diff, cost]
            } else {
                [cost, clr_diff]
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
    // fml this is bullshit
    //let mut pq3 = PriorityQueue::new();
    'outer: while let Some((e, [dev, cd])) = pq.pop() {
        assert!(pq2.is_empty());
        pq2.push(e, [cd, dev]);
        while let Some(([e0, e1], [_cd, dev])) = pq2.pop() {
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

            let mpq2 = std::sync::Mutex::new(&mut pq2);
            let mpq = std::sync::Mutex::new(&mut pq);
            use rayon::iter::ParallelIterator;
            m.par_vertex_adj(e1).for_each(|adj| {
                let c = cost_of_edge!(adj, e1);
                let e = minmax(e1, adj);

                mpq2.lock().unwrap().remove(&e);
                mpq.lock().unwrap().push(e, c);
            });

            while let Some((ne, [ndev, ncd])) =
                pq.pop_if(|_, [ndev, _]| approx_eq(**ndev, *dev, args.abs_eps))
            {
                pq2.push(ne, [ncd, ndev]);
            }

            p.set_position(m.num_vertices() as u64);
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

impl_display!(Eigenvalue, Zero => "zero", One => "one", Two => "two");

pub fn approx_eq(a: F, b: F, eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < eps
}
