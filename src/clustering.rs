use std::cmp::minmax;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use ordered_float::NotNan;

use clap::ValueEnum;

use priority_queue::PriorityQueue;

use super::manifold::{CollapsibleManifold, EdgeKind};
use super::quadric::Quadric;
use super::{F, add, dist, dot, kmul, normalize, poly_area, sub};

use indicatif::ProgressBar;

const PI: F = std::f64::consts::PI as F;

pub struct Args {
    /// Target output number of charts.
    pub target_num_charts: Option<usize>,

    pub eigenvalue: Eigenvalue,

    /// How long before stopping
    pub error_bound: F,

    /// Do not use area weighting.
    pub no_area_weight: bool,

    /// Absolute value difference permitted between eigenvalues, before switching to color
    /// checking.
    pub eigen_eps: F,

    /// Absolute value difference permitted between colors, before switching to bd length
    /// checking.
    pub color_eps: F,

    /// What ordering is preferred when constructing a cluster
    pub ordering: OrderingKind,

    /// What metric to use when evaluating the quality of output cluster shape
    pub shape_metric: ShapeMetric,

    /// Prefer opposite effect of shape metric (For jokes)
    /// Mey kill perf.
    pub invert_shape: bool,

    /// Do not use the delta cost, use the straight cost.
    pub no_delta_cost: bool,
}

pub fn face_clustering<'a>(
    vs: &'a [[F; 3]],
    vcs: &'a [[F; 3]],
    fs: &'a [pars3d::FaceKind],
    nf: usize,
    args: &Args,
) -> (
    Vec<usize>,
    Vec<(Quadric<0>, [F; 3], F)>,
    BTreeMap<usize, Vec<usize>>,
) {
    // face normals
    let f_n = (0..nf).map(|fi| fs[fi].normal(vs)).collect::<Vec<_>>();

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
        for e in f.edges_ord() {
            edge_face_adj
                .entry(e)
                .and_modify(|old| {
                    old.insert(fi);
                })
                .or_insert_with(|| EdgeKind::Boundary(fi));
        }
    }

    // normalize_face_areas
    let total_area = face_area.iter().copied().sum::<F>();
    let avg_area = total_area / face_area.len() as F;
    for a in face_area.iter_mut() {
        *a /= avg_area;
    }
    println!("[INFO]: Surface Area is {total_area:02}, Avg is {avg_area:02e}");

    let mut face_colors = vec![[0.; 3]; fs.len()];
    for (fi, f) in fs.iter().enumerate() {
        let area = face_area[fi];
        let per_vert_weight = area / f.len() as F;
        let avg_color = if area == 0. {
            [0.; 3]
        } else {
            let sum_color = f
                .as_slice()
                .iter()
                .copied()
                .map(|vi| vcs.get(vi).copied().unwrap_or([0.; 3]))
                .map(|vc| kmul(per_vert_weight, vc))
                .fold([0.; 3], add);
            kmul(area.recip(), sum_color)
        };
        assert!(avg_color.iter().copied().all(F::is_finite));
        face_colors[fi] = avg_color;
    }

    //let vert_vert_adj = pars3d::adjacency::VertexAdj::new(fs, vs.len());

    // for each real mesh edge, track which charts it is a part of.
    // Each edge can be at most part of 2 charts.
    let mut m = CollapsibleManifold::<(Quadric<0>, [F; 3], F), _>::atomic_new_with(nf, |fi| {
        let n = f_n[fi];
        let v0 = vs[fs[fi].as_slice()[0]];
        let q = Quadric::new_plane(v0, n, 1.);
        let area = face_area[fi];

        assert!(!fs[fi].is_empty());

        (q * area, face_colors[fi], area)
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
    eprintln!("[INFO]: Clustering Input # F = {}, # V = {}", nf, vs.len());

    // for each chart pair (ordered minmax)
    // store edges between these charts
    // update this on every iteration before merging
    let mut shared_chart_edges: HashMap<[usize; 2], EdgeSet> = HashMap::new();
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
            assert!(shared_e.vert_sets.is_empty());
            shared_e.extend(f0.shared_edges(&adj_f));
        }
    }

    let mut chart_boundary_edges: HashMap<usize, Vec<[usize; 2]>> = HashMap::new();
    for (fi, _) in m.vertices() {
        let f = &fs[fi];
        for e @ [e0, e1] in f.edges() {
            if !edge_face_adj[&minmax(e0, e1)].is_boundary() {
                continue;
            }
            chart_boundary_edges.entry(fi).or_default().push(e);
        }
    }

    macro_rules! straightness_deviation {
        ($pi: expr, $ci: expr, $ni: expr) => {{
            let [p, c, n] = [$pi, $ci, $ni].map(|vi| vs[vi]);
            let e0 = normalize(sub(p, c));
            let e1 = normalize(sub(n, c));
            let theta = dot(e0, e1).clamp(-1., 1.).acos() / PI;
            // deviation in the range [0, 1]
            let dev = 1. - theta;
            assert!((0.0..=1.0).contains(&dev));
            // close to L0?
            dev * dev //.sqrt()
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
            assert_ne!(pi, ci, "{f:?}");
            assert_ne!(ci, ni, "{f:?}");
            assert_ne!(pi, ni, "{f:?}");
            let prev = incident_angles[fi].entry(ci).or_default();
            assert!(!prev.contains(&[pi, ni]));
            // imporant to retain original winding order here, so we can determine what is right
            // and left.
            prev.push([pi, ni]);
        }
    }

    let mut pq = PriorityQueue::new();

    let mut prev_eigens = vec![0.; nf];
    for (vi, (q, _, _)) in m.vertices() {
        let eigens = q.a.eigen_sorted().0;
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
            let (neigens, n_evecs) = q_new.a.eigen_sorted();
            let evn = args.eigenvalue.apply(neigens);

            let ev0 = prev_eigens[e0];
            let ev1 = prev_eigens[e1];
            let [evn, ev0, ev1] = [evn, ev0, ev1].map(F::abs);
            // subtract previous values here (only penalize added deviation)?
            let cost = if args.no_delta_cost {
                evn
            } else {
                evn - (ev0 + ev1)
            };

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
                area0 * luma_dist(new_avg, avg_color0) +
                area1 * luma_dist(new_avg, avg_color1);
            assert!(clr_diff.is_finite());
            assert!(clr_diff >= 0.);

            let shape_metric = match args.shape_metric {
              // if either of these is 0., then there will be no space for optimizing shape
              // metrics.
              _ if args.eigen_eps <= 0. || args.color_eps <= 0. => 0.,
              ShapeMetric::None => 0.,
              ShapeMetric::Area => -new_area,
              ShapeMetric::MaxEuclideanDist => {
                let mut aabb = pars3d::aabb::AABB::new();
                for c in [e0, e1] {
                  for fi in m.merged_vertices(c) {
                    for &vi in fs[fi].as_slice() {
                      aabb.add_point(vs[vi]);
                    }
                  }
                }
                let center = aabb.center();

                let mut max_dist: F = 0.;
                for c in [e0,e1] {
                  for fi in m.merged_vertices(c) {
                    let ctr = fs[fi].centroid(vs);
                    max_dist = max_dist.max(dist(ctr, center));
                  }
                }
                -max_dist
              },
              ShapeMetric::MaxManhattanDist => {
                let mut aabb = pars3d::aabb::AABB::new();
                for c in [e0, e1] {
                  for fi in m.merged_vertices(c) {
                    for &vi in fs[fi].as_slice() {
                      aabb.add_point(vs[vi]);
                    }
                  }
                }
                let center = aabb.center();

                let mut max_dist: F = 0.;
                for c in [e0,e1] {
                  for fi in m.merged_vertices(c) {
                    let local = sub(fs[fi].centroid(vs), center);
                    let l_d = dot(local, n_evecs[1]).abs() + dot(local, n_evecs[0]).abs();
                    max_dist = max_dist.max(l_d);
                  }
                }
                -max_dist
              },
              ShapeMetric::SharedBoundaryLength => {
                // Tested here: dividing by new area seems to make it prefer rounder charts,
                // whereas just plain seems to be ok with skinny charts.
                // Maybe that's fine?
                shared_chart_edges[&[e0, e1]]
                    .edges()
                    .map(|[e0, e1]| dist(vs[e0], vs[e1]))
                    .sum::<F>()
              }
              ShapeMetric::BoundaryLength => {
                let mut sum = 0.;
                for c in [e0, e1] {
                  for adj_c in m.vertex_adj(c) {
                    if adj_c == e0 || adj_c == e1 {
                      continue;
                    }
                    for [e0, e1] in shared_chart_edges[&minmax(c, adj_c)].edges() {
                      sum += dist(vs[e0], vs[e1]);
                    }
                  }
                  /*
                  for fi in m.merged_vertices(c) {
                    for e in fs[fi].edges_ord() {
                      let is_bd = match &edge_face_adj[&e] {
                        EdgeKind::Boundary(..) => true,
                        &EdgeKind::Manifold([this,o] | [o,this]) if this == fi => {
                          let o_chart = m.get_new_vertex(o);
                          o_chart != e0 && o_chart != e1
                        },
                        EdgeKind::Manifold(_) => unreachable!(),
                        EdgeKind::NonManifold(fs) => {
                          fs.iter().any(|&ofi| {
                            let o_chart = m.get_new_vertex(ofi);
                            o_chart != e0 && o_chart != e1
                          })
                        },
                      };
                      if !is_bd {
                        continue;
                      }
                      let [e0, e1] = e;
                      sum += dist(vs[e0], vs[e1]);
                    }
                  }
                  */
                }
                -sum
              }
              ShapeMetric::Convexity =>  {
                let mut sum = 0.;
                for adj in m.vertex_adj(e0) {
                  if adj == e1 || !m.is_adj(e1, adj) {
                    continue;
                  }
                  let sce0 = &shared_chart_edges[&minmax(adj, e0)];
                  let sce1 = &shared_chart_edges[&minmax(adj, e1)];
                  for [pi, vi, ni] in sce0.shared_incident_edges(sce1) {
                    sum += straightness_deviation!(pi, vi, ni);
                  }
                }
                -sum
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

                            //Some(new)
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
    'outer: while let Some((e, [dev, cd, s])) = pq.pop() {
        assert!(pq2.is_empty());
        pq2.push(e, [cd, dev, s]);
        while let Some((e, [cd, dev, s])) = pq2.pop() {
            assert!(pq3.is_empty());
            pq3.push(e, [s, cd, dev]);
            // hohoho kms
            while let Some(([e0, e1], [_s, cd, dev])) = pq3.pop() {
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
                    shared_chart_edges.entry(nc).or_default().append(&mut sce);
                }

                if let Some(mut e0_bds) = chart_boundary_edges.remove(&e0) {
                    chart_boundary_edges
                        .entry(e1)
                        .or_default()
                        .append(&mut e0_bds);
                }

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
                prev_eigens[e1] = args.eigenvalue.apply(m.get(e1).0.a.eigen_sorted().0);

                for adj in m.vertex_adj(e1) {
                    let c = cost_of_edge!(adj, e1);
                    let e = minmax(e1, adj);

                    pq2.remove(&e);
                    pq3.remove(&e);
                    pq.push(e, c);
                }

                let first_eps = args.ordering.first_eps(args.eigen_eps, args.color_eps);
                while let Some((ne, [ndev, ncd, ns])) =
                    pq.pop_if(|_, [ndev, _, _]| approx_eq(**ndev, *dev, first_eps))
                {
                    pq2.push(ne, [ncd, ndev, ns]);
                }

                let snd_eps = args.ordering.second_eps(args.eigen_eps, args.color_eps);
                while let Some((ne, [ncd, ndev, ns])) =
                    pq2.pop_if(|_, [ncd, _, _]| approx_eq(**ncd, *cd, snd_eps))
                {
                    pq3.push(ne, [ns, ncd, ndev]);
                }

                p.set_position(m.num_vertices() as u64);
            }
        }
    }

    // for each face which chart is it assigned to?
    let mut charts = (0..nf).map(|i| m.get_new_vertex(i)).collect::<Vec<_>>();

    let mut remap = BTreeMap::new();
    let mut num_charts = 0;
    for &f in &charts {
        match remap.entry(f) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                v.insert(num_charts);
                num_charts += 1;
            }
        }
    }

    for f in &mut charts {
        *f = remap[f];
    }
    let mut inv_map = BTreeMap::new();
    for (&src, &dst) in remap.iter() {
        inv_map.insert(dst, src);
    }
    let /*mut*/ chart_attribs = (0..num_charts)
        .map(|i| m.data[inv_map[&i]])
        .collect::<Vec<_>>();
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    /*
    for [c0, c1] in m.edges_post_merge() {
        adj.entry(remap[&c0]).or_default().push(remap[&c1]);
        adj.entry(remap[&c1]).or_default().push(remap[&c0]);
    }
    */
    let mesh = pars3d::Mesh {
        v: vs.to_vec(),
        f: fs.to_vec(),
        ..Default::default()
    };
    fn push_uniq(dst: &mut Vec<usize>, v: usize) {
        if !dst.contains(&v) {
            dst.push(v);
        }
    }

    for ek in mesh.edge_pos_kinds().values() {
        for &fi in ek.as_slice() {
            let ci = charts[fi];
            for &fj in ek.as_slice() {
                let cj = charts[fj];
                if ci == cj {
                    continue;
                }
                push_uniq(adj.entry(ci).or_default(), cj);
                push_uniq(adj.entry(cj).or_default(), ci);
            }
        }
    }

    // also need to store per face quadrics and colors

    // Charts & their attributes
    /*
    let mut unique_charts = m.vertices().collect::<BTreeMap<_, _>>();

    macro_rules! swap_cost {
        // swap fi across edge e to the other cluster
        ($fi: expr, $e: expr) => {{
            let e = $e;
            let EdgeKind::Manifold([a, b]) = edge_face_adj[&e] else {
                continue;
            };
            if charts[a] == charts[b] {
                continue;
            }

            let (q_a, avg_color_a, area_a) = &unique_charts[&a];
            let (q_b, avg_color_b, area_b) = &unique_charts[&b];
        }};
    }
    // Enqueue all edges between charts that would decrease the total energy,
    // to account for poor choices earlier in the algorithm.
    // Use same eps as before.
    let mut pq = PriorityQueue::new();
    for (fi, f) in fs.iter().enumerate() {
        for e in f.edges_ord() {
            let cost = swap_cost!(fi, e);
            pq.push((fi, e), cost);
        }
    }

    while let Some(((fi, e), cost)) = pq.pop() {
        if cost >= 0. {
            break;
        }
    }
    */

    (charts, chart_attribs, adj)
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

fn merge_wedges([a0, a1]: [usize; 2], [b0, b1]: [usize; 2]) -> Option<[usize; 2]> {
    // TODO here handle inconsistent winding order
    //assert_ne!(a0, b0, "Shouldn't happen if winding order is consistent");
    //assert_ne!(a1, b1, "Shouldn't happen if winding order is consistent");

    let v = if a1 == b0 {
        [a0, b1]
    } else if a0 == b1 {
        [b0, a1]
    } else if a0 == b0 {
        // inconsistent winding
        [a1, b1]
    } else if a1 == b1 {
        // inconsistent winding
        [a0, b0]
    } else {
        return None;
    };
    Some(v)
}

// TODO decide if this only works for meshes without repeat faces or not
/// Sorts this set of set wedges such that all those which will be deleted are at the end.
/// Then, the last (return value) can be removed and replaced with the optional return value.
fn new_wedges(wedges: &mut [[usize; 2]], mut curr: [usize; 2]) -> (usize, Option<[usize; 2]>) {
    //assert!(curr[0] < curr[1]);
    //assert!(wedges.iter().all(|[a, b]| a < b));
    let mut to_keep = wedges.len();
    while to_keep > 0 {
        let mut any = false;
        for i in 0..to_keep {
            let wedge = wedges[i];
            // if an existing wedge exactly matches, delete both of them and return None.
            if wedge == curr || wedge == [curr[1], curr[0]] {
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
    /// Just the boundary that is lost between each cluster
    SharedBoundaryLength,

    /// The angle at each boundary point in the cluster's deviation from 180 degrees.
    /// (Prefer straight edges)
    AngleDeviation,

    /// The total boundary length of each cluster.
    BoundaryLength,

    /// Measures convexity of a shape by the amount of left turn of each edge.
    Convexity,

    Area,

    /// Maximum euclidean of any point in this chart
    MaxEuclideanDist,

    /// Maximum manhattan distance of any point in this chart
    MaxManhattanDist,

    None,
}

#[derive(Debug, Clone, Default)]
struct EdgeSet {
    vert_sets: Vec<Vec<usize>>,
}

impl EdgeSet {
    pub fn edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.vert_sets
            .iter()
            .flat_map(|vs| vs.array_windows::<2>().copied())
    }
    /*
    pub fn is_split(&self) -> bool {
        self.vert_sets.len() > 1
    }
    */
    /// Returns the vertices that are shared between these two edge sets
    pub fn shared_incident_edges<'a: 'c, 'b: 'c, 'c>(
        &'a self,
        o: &'b Self,
    ) -> impl Iterator<Item = [usize; 3]> + 'c {
        self.vert_sets.iter().flat_map(|vs| {
            assert!(!vs.is_empty());
            let vs0 = vs[0];
            let vsl = *vs.last().unwrap();
            let vs2l = vs[vs.len() - 2];
            o.vert_sets.iter().filter_map(move |ovs| {
                let ovsl = *ovs.last().unwrap();
                let triple = if vs0 == ovs[0] {
                    [vs[1], vs0, ovs[1]]
                } else if vs0 == ovsl {
                    [vs[1], vs0, ovs[ovs.len() - 2]]
                } else if vsl == ovs[0] {
                    [vs2l, vsl, ovs[1]]
                } else if vsl == ovsl {
                    [vs2l, vs0, ovs[ovs.len() - 2]]
                } else {
                    return None;
                };
                Some(triple)
            })
        })
    }
    pub fn append(&mut self, o: &mut Self) {
        if self.vert_sets.is_empty() {
            self.vert_sets.append(&mut o.vert_sets);
            return;
        }
        if o.vert_sets.is_empty() {
            return;
        }

        for ovs in o.vert_sets.drain(..) {
            let dst_i = self.vert_sets.iter().position(|vs| {
                let vs0 = *vs.first().unwrap();
                let vsl = *vs.last().unwrap();

                let ovs0 = *ovs.first().unwrap();
                let ovsl = *ovs.last().unwrap();
                vs0 == ovs0 || vs0 == ovsl || vsl == ovs0 || vsl == ovsl
            });
            let Some(dst_i) = dst_i else {
                self.vert_sets.push(ovs);
                continue;
            };

            let verts = &mut self.vert_sets[dst_i];

            let first_vert = *verts.first().unwrap();
            let o_last = *ovs.last().unwrap();
            if first_vert == ovs[0] || first_vert == o_last {
                verts.reverse();
            }
            let last_vert = *verts.last().unwrap();
            assert!(
                last_vert == ovs[0] || last_vert == o_last,
                /*
                "{:?} {:?} {:?} {:?} {:?} {:?}",
                self.verts.first(), self.verts.last(),
                o.verts.first(), o.verts.last(),
                self.verts, o.verts,
                */
            );
            if last_vert == ovs[0] {
                verts.extend_from_slice(&ovs[1..]);
            } else {
                verts.extend(ovs.iter().rev().skip(1).copied());
            }
        }
        self.consolidate();
    }
    pub fn extend(&mut self, es: impl Iterator<Item = [usize; 2]>) {
        for [e0, e1] in es {
            let dst_i = self.vert_sets.iter().position(|vs| {
                let vs0 = *vs.first().unwrap();
                let vsl = *vs.last().unwrap();

                vs0 == e0 || vs0 == e1 || vsl == e0 || vsl == e1
            });
            let Some(dst_i) = dst_i else {
                self.vert_sets.push(vec![e0, e1]);
                continue;
            };

            let verts = &mut self.vert_sets[dst_i];

            if verts[0] == e0 || verts[0] == e1 {
                verts.reverse();
            }
            let last_vert = *verts.last().unwrap();
            assert!(last_vert == e0 || last_vert == e1);
            verts.push(if last_vert == e0 { e1 } else { e0 });
        }
    }
    fn consolidate(&mut self) {
        let mut i = 0;
        while i < self.vert_sets.len() {
            let mut j = i + 1;
            assert!(!self.vert_sets[i].is_empty());

            while j < self.vert_sets.len() {
                let vsi = &self.vert_sets[i];
                let vsil = *vsi.last().unwrap();

                let vsj = &self.vert_sets[j];
                assert!(!vsj.is_empty());
                let vsjl = *vsj.last().unwrap();
                let first_match = vsi[0] == vsj[0] || vsi[0] == vsjl;
                let last_match = vsil == vsj[0] || vsil == vsjl;
                if !(first_match || last_match) {
                    j += 1;
                    continue;
                }
                if first_match {
                    self.vert_sets[i].reverse();
                }
                let vsj = self.vert_sets.swap_remove(j);
                let vsi = &mut self.vert_sets[i];
                let vsil = *vsi.last().unwrap();
                assert!(vsil == vsj[0] || vsil == *vsj.last().unwrap());
                if vsil == vsj[0] {
                    vsi.extend_from_slice(&vsj[1..]);
                } else {
                    vsi.extend(vsj.iter().rev().skip(1).copied());
                }
            }
            i += 1;
        }
    }
}

impl_display!(
  ShapeMetric,
  BoundaryLength => "boundary-length",
  SharedBoundaryLength => "shared-boundary-length",
  AngleDeviation => "angle-deviation",
  Convexity => "convexity",
  Area => "area",
  MaxEuclideanDist => "max-euclidean-dist",
  MaxManhattanDist => "max-manhattan-dist",
  None => "none",
);

pub fn approx_eq(a: F, b: F, eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < eps
}

pub fn luma(rgb: [F; 3]) -> F {
    dot([0.299, 0.587, 0.114], rgb)
}

pub fn luma_dist(rgb_a: [F; 3], rgb_b: [F; 3]) -> F {
    (luma(rgb_a) - luma(rgb_b)).abs()
}
