use std::collections::HashMap;
use std::ops::Range;

use ordered_float::NotNan;
use pars3d::FaceKind;
use priority_queue::PriorityQueue;
use union_find::UnionFind;

use super::{
    F, cross, dot, length,
    manifold::{CollapsibleManifold, EdgeKind},
    normalize,
    quadric::{AttrWeights, Quadric, QuadricAccumulator},
    sub,
};
pub struct Args {
    /// During decimation, how heavily should colors be preserved?
    pub color_weight: F,

    /// Target number of vertices after reduction
    pub target_vert_ratio: F,

    /// Target number of vertices after reduction
    pub target_num_verts: usize,

    /// Extra weight to add on each edge based on color differences
    pub color_preservation_weight: F,

    /// Minimum face area during decimation.
    pub min_face_area: F,

    /// Minimum edge weight for each edge.
    pub min_edge_weight: F,

    /// Epsilon value to use when comparing quadric errors.
    pub abs_eps: F,

    /// Threshold to stop quadric decimation at
    pub quadric_threshold: F,

    /// The weight to use for degenerate quadrics
    pub degen_quadric_weight: F,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            color_weight: 0.5,
            target_vert_ratio: 0.,
            target_num_verts: 0,
            color_preservation_weight: 10.,

            min_face_area: 1e-2,
            min_edge_weight: 1e-2,

            abs_eps: 1e-4,
            quadric_threshold: 1.,
            degen_quadric_weight: 1e-4,
        }
    }
}

/// In-place simplification of planar faces of a mesh
pub fn simplify_range_colored(
    mesh: &mut pars3d::Mesh,
    args: &Args,
    // function which indicates if a vertex is locked
    locked: impl Fn(usize) -> bool,
    face_range: Range<usize>,
    vert_range: Range<usize>,
    // output which vertices got mapped to which other vertices
    remap: &mut UnionFind<u32>,
) {
    let offset = vert_range.start;
    let num_v = vert_range.end - offset;

    assert!(!mesh.vert_colors.is_empty());
    let cw = args.color_weight;
    let attr_ws = AttrWeights { ws: [cw; 3] };

    let mut m = CollapsibleManifold::new_with_remapping(remap.subset(vert_range.clone()), |vi| {
        assert!(vert_range.contains(&(vi + offset)));
        let pos = mesh.v[vi + offset];
        let area = 1e-6;
        let mut q = Quadric::<3>::zero();
        for d in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] {
            q += Quadric::new_plane(pos, d, area);
            q.area += area;
        }
        q += Quadric::degen_attr(mesh.vert_colors[vi + offset], attr_ws) * area;
        q *= args.degen_quadric_weight;

        (q, pos)
    });

    let target_verts = args
        .target_num_verts
        .max((args.target_vert_ratio.clamp(0., 1.) * (num_v as F)) as usize);

    let mut edge_face_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();
    let mut f_n = vec![[0.; 3]; mesh.f.len()];
    let mut num_edges = 0;
    let mut avg_edge_len = 0.;
    for fi in face_range.clone() {
        let f = &mesh.f[fi];
        f_n[fi] = normalize(f.normal(&mesh.v));
        for e in f.edges_ord() {
            edge_face_adj
                .entry(e)
                .and_modify(|p| {
                    p.insert(fi);
                })
                .or_insert_with(|| EdgeKind::Boundary(fi));
            num_edges += 1;
            let [e0, e1] = e.map(|vi| mesh.v[vi]);
            avg_edge_len += length(sub(e1, e0));
        }
    }

    avg_edge_len /= num_edges as F;

    let mut color_dists = HashMap::new();
    if !mesh.vert_colors.is_empty() {
        for &[e0, e1] in edge_face_adj.keys() {
            let v = length(sub(mesh.vert_colors[e0], mesh.vert_colors[e1]));
            assert_eq!(color_dists.insert([e0, e1], v), None);
        }
    }

    for fi in face_range.clone() {
        for es in mesh.f[fi].edges() {
            let [e0, e1] = es.map(|vi| m.vertices.get_compress(vi - offset));
            m.add_edge(e0, e1);
        }
    }

    for fi in face_range.clone() {
        let f = &mesh.f[fi];
        let area = f.area(&mesh.v).max(0.) + args.min_face_area;
        let n = f_n[fi];
        if length(n) == 0. {
            // Handle this better (there will be many degenerate triangles)
            continue;
        }

        let f_slice = f.as_slice();
        for (i, &v) in f_slice.iter().enumerate() {
            let curr = mesh.v[v];
            let pi = f_slice[i.checked_sub(1).unwrap_or_else(|| f.len() - 1)];
            let prev = mesh.v[pi];
            let ni = f_slice[(i + 1) % f.len()];
            let e = std::cmp::minmax(v, ni);
            let next = mesh.v[ni];

            let interior_angle = {
                let e0 = normalize(sub(prev, curr));
                let e1 = normalize(sub(next, curr));
                dot(e0, e1).clamp(-1., 1.).acos()
            };
            let mut q = Quadric::new_plane(curr, n, area) * interior_angle;
            q.area = area;
            m.data[v - offset].0 += q;

            const PI: F = std::f64::consts::PI as F;

            macro_rules! dihedral_angle {
                ($f0: expr, $f1: expr) => {{
                    let angle = dot(f_n[$f0], f_n[$f1]);
                    assert!((-1.0001..=1.0001).contains(&angle), "{angle}");
                    let angle = angle.clamp(-1., 1.);

                    let v = angle.acos();
                    assert!((0.0..=PI).contains(&v), "{v} {angle}");
                    v
                }};
            }

            let e_w = match edge_face_adj[&e] {
                EdgeKind::Boundary(_) => 4.,
                EdgeKind::Manifold([a, b]) => dihedral_angle!(a, b) / PI,
                EdgeKind::NonManifold(_) => 4.,
            };
            let e_w = e_w.max(args.min_edge_weight);

            let edge_dir = sub(curr, next);
            let edge_len = length(edge_dir);
            let edge_len = edge_len / avg_edge_len;
            if edge_len == 0. {
                continue;
            }
            let edge_dir = normalize(edge_dir);
            let edge_quadric = Quadric::new_plane(curr, normalize(cross(n, edge_dir)), 0.);

            let colpw = color_dists.get(&e).copied().unwrap_or(0.) * args.color_preservation_weight;

            let e_w = e_w.max(colpw);

            let total_e_w = e_w * edge_len;
            let mut edge_quadric = edge_quadric * total_e_w.max(1e-4);
            edge_quadric.area = 0.;

            m.data[v - offset].0 += edge_quadric;
            m.data[ni - offset].0 += edge_quadric;
        }

        macro_rules! q_n_attribs(
          ($vis: expr) => {{
            Quadric::n_attribs(
                n,
                $vis.map(|vi| mesh.v[vi]),
                $vis.map(|vi| mesh.vert_colors[vi]),
                attr_ws,
            )
          }}
        );

        // add attributes as well
        let q_attr = match f {
            FaceKind::Tri(vis) => q_n_attribs!(vis),
            FaceKind::Quad(vis) => q_n_attribs!(vis),
            FaceKind::Poly(p) => Quadric::dyn_attribs(
                n,
                p.len(),
                |vi| mesh.v[vi],
                |vi| mesh.vert_colors[vi],
                attr_ws,
            ),
        };

        for &vi in f.as_slice() {
            m.data[vi - offset].0 += q_attr * area;
        }
    }

    let mut curr_costs = vec![0.; num_v];
    let mut pq = PriorityQueue::new();

    macro_rules! update_cost_of_edge {
        ($e0:expr, $e1: expr) => {{
            let [e0, e1] = std::cmp::minmax($e1, $e0);
            let mut q_acc = QuadricAccumulator::default();
            q_acc += m.get(e0).0;
            q_acc += m.get(e1).0;
            let p = q_acc.point();
            assert!(p.iter().copied().all(F::is_finite));
            let mut total_cost = 0.;

            let q01f = m.get(e0).0 + m.get(e1).0;
            // colors are also automatically clamped to [0., 1.].
            let attrs = q01f.attributes(p, attr_ws).map(|v| v.clamp(0., 1.));
            total_cost -=
                q01f.cost_attrib(p, attrs, attr_ws).max(0.) - curr_costs[e0] - curr_costs[e1];

            NotNan::new(total_cost).unwrap()
        }};
    }

    for [e0, e1] in m.ord_edges() {
        if locked(e0 + offset) || locked(e1 + offset) {
            continue;
        }
        pq.push([e0, e1], update_cost_of_edge!(e0, e1));
    }

    let p = indicatif::ProgressBar::new(num_v as u64);
    let mut buf = PriorityQueue::new();
    let mut recencies = HashMap::new();
    let mut did_update = vec![];
    'outer: while let Some((e, q)) = pq.pop() {
        assert!(buf.is_empty());
        buf.push(e, (0, q));
        recencies.clear();
        while let Some(([e0, e1], (rec, q_err))) = buf.pop() {
            assert!(e0 < e1);
            if m.is_deleted(e0) || m.is_deleted(e1) {
                continue;
            }
            if -*q_err >= args.quadric_threshold || m.num_vertices() < target_verts {
                break 'outer;
            }

            let mut q_acc = QuadricAccumulator::default();
            q_acc += m.get(e0).0;
            q_acc += m.get(e1).0;
            let pos = q_acc.point();

            if let Some(adj_faces) = edge_face_adj.get(&[e0, e1]) {
                for &af in adj_faces.as_slice() {
                    let f = &mesh.f[af];
                    let Some([q0, q1]) = f.quad_opp_edge(e0, e1) else {
                        continue;
                    };
                    let r = recencies.entry(std::cmp::minmax(q0, q1)).or_insert(rec);
                    *r += 1000;
                }
            };

            m.merge(e0, e1, |(q0, _), (q1, _)| {
                let q01 = *q0 + *q1;
                curr_costs[e1] = q01
                    .cost_attrib(pos, q01.attributes(pos, attr_ws), attr_ws)
                    .max(0.);
                (q01, pos)
            });

            did_update.clear();
            let e_dst = m.get_new_vertex(e1);
            for adj in m.vertex_adj(e_dst) {
                if locked(adj + offset) {
                    continue;
                }
                let prio = update_cost_of_edge!(e_dst, adj);
                let adj_e = std::cmp::minmax(e_dst, adj);
                buf.remove(&adj_e);
                pq.push(adj_e, prio);
                did_update.push(adj_e);
            }

            for adj in m.vertex_adj(e_dst) {
                if locked(adj + offset) {
                    continue;
                }
                for adj2 in m.vertex_adj(adj) {
                    if locked(adj2 + offset) {
                        continue;
                    }
                    let adj_e = std::cmp::minmax(adj, adj2);
                    if adj2 == e_dst || did_update.contains(&adj_e) {
                        continue;
                    }

                    did_update.push(adj_e);
                    let prio = update_cost_of_edge!(adj, adj2);
                    let recency = recencies.get(&adj_e).copied().unwrap_or(0);

                    if !approx_eq(*prio, *q_err, args.abs_eps) {
                        buf.remove(&adj_e);
                        pq.push(adj_e, prio);
                        continue;
                    }
                    let changed = buf.change_priority(&adj_e, (recency, prio)).is_some();
                    if !changed {
                        pq.push(adj_e, prio);
                    }
                }
            }

            while let Some((_e, nq_err)) = pq.peek()
                && approx_eq(**nq_err, *q_err, args.abs_eps)
            {
                let (e, nq_err) = pq.pop().unwrap();
                let recency = recencies.get(&e).copied().unwrap_or(0);
                buf.push(e, (recency, nq_err));
            }

            p.set_position(m.num_vertices() as u64);
        }
    }

    for (vi, &(q, p)) in m.vertices() {
        let vi = vi + offset;

        let attrs = q.attributes(p, attr_ws);
        assert!(!attrs.is_empty());
        assert!(attrs.len() == 3);

        mesh.v[vi] = p;
        mesh.vert_colors[vi] = attrs.map(|c| c.clamp(0., 1.));
    }
}

fn approx_eq(a: F, b: F, abs_eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < abs_eps
}

/// rotates f so that the minimum value is in front
pub fn consistent_face_ordering(f: &mut [usize]) {
    if f.is_empty() {
        return;
    }
    let min_idx = f.iter().enumerate().min_by_key(|(_, idx)| **idx).unwrap().0;
    f.rotate_left(min_idx);
}
