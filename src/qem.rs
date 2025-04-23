use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use ordered_float::NotNan;
use pars3d::FaceKind;
use priority_queue::PriorityQueue;
use union_find::UnionFind;

use super::{
    F, add, cross, dist, dot, kmul, length,
    manifold::{CollapsibleManifold, EdgeKind},
    normalize,
    quadric::{AttrWeights, Quadric, QuadricAccumulator},
    sub,
};
pub struct Args {
    /// During decimation, how heavily should colors be preserved?
    pub color_weight: F,

    /// Extra weight to add on each edge based on color differences
    pub color_preservation_weight: F,

    /// Minimum face area during decimation.
    pub min_face_area: F,

    /// Minimum edge weight for each edge.
    pub min_edge_weight: F,

    /// Epsilon value to use when comparing quadric errors.
    pub abs_eps: F,

    /// The weight to use for degenerate quadrics
    pub degen_quadric_weight: F,

    /// Check if any faces would invert when an edge is collapsed
    pub no_check_face_inversion: bool,

    pub max_degree: usize,

    /// What percentage of verts should be retained at the end?
    pub target_vert_ratio: F,

    /// What percentage of faces should be retained at the end?
    pub target_tri_ratio: F,
    pub target_tri_num: usize,

    /// If progress should be displayed
    pub display_progress: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            //color_weight: 0.5,
            color_weight: 1e-2,
            color_preservation_weight: 1.,

            min_face_area: 1e-2,
            min_edge_weight: 5e-3,

            abs_eps: 0.,
            degen_quadric_weight: 1e-3,

            no_check_face_inversion: false,

            max_degree: 100,

            target_vert_ratio: 0.,

            display_progress: false,

            target_tri_ratio: 0.,
            target_tri_num: 100,
        }
    }
}

#[derive(Default)]
pub struct QEMBuffers {
    edge_face_adj: HashMap<[usize; 2], EdgeKind>,
    pq: PriorityQueue<[usize; 2], NotNan<F>>,
    snd_pq: PriorityQueue<[usize; 2], (u32, NotNan<F>)>,
    recencies: BTreeMap<[usize; 2], u32>,
    did_update: Vec<[usize; 2]>,
    face_normals: Vec<[F; 3]>,
}

impl QEMBuffers {
    pub fn clear(&mut self) {
        self.edge_face_adj.clear();
        self.pq.clear();
        self.snd_pq.clear();
        self.recencies.clear();
        self.did_update.clear();
        self.face_normals.clear();
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

    bufs: &mut QEMBuffers,
) {
    bufs.clear();

    let offset = vert_range.start;
    let num_v = vert_range.end - offset;

    assert!(!mesh.vert_colors.is_empty());
    let cw = args.color_weight;
    let attr_ws = AttrWeights { ws: [cw; 3] };

    let target_verts =
        (remap.curr_len() as F * args.target_vert_ratio.clamp(0., 1.)).floor() as usize;
    // ensure the faces are correct at this stage
    for f in &mut mesh.f[face_range.clone()] {
        f.remap(|vi| remap.get_compress(vi));
    }
    let num_tris = mesh.f[face_range.clone()]
        .iter()
        .map(|f| f.num_tris())
        .sum::<usize>();
    let target_num_tris = (num_tris as F * args.target_tri_ratio).floor() as usize;
    let target_num_tris = target_num_tris.max(args.target_tri_num);
    if num_tris <= target_num_tris {
        return;
    }

    // normalize all vertices to [-1., 1]
    use std::array::from_fn;
    // Normalize the geometry of this mesh to lay in the unit box.
    let present_verts = vert_range.clone().filter(|&v| remap.is_root(v));
    let [l, h] = present_verts
        .clone()
        .map(|vi| mesh.v[vi])
        .fold([[F::INFINITY; 3], [F::NEG_INFINITY; 3]], |[l, h], n| {
            [from_fn(|i| l[i].min(n[i])), from_fn(|i| h[i].max(n[i]))]
        });
    let center = kmul(0.5, add(l, h));
    for vi in present_verts.clone() {
        mesh.v[vi] = sub(mesh.v[vi], center);
    }
    macro_rules! rescale_attr {
        ($attr: expr) => {{
            let largest_val = present_verts
                .clone()
                .map(|vi| $attr[vi])
                .fold(0. as F, |m, [v0, v1, v2]| m.max(v0).max(v1).max(v2));
            let scale = if largest_val == 0. {
                1.
            } else {
                largest_val.recip()
            };
            for v in present_verts.clone() {
                $attr[v] = kmul(scale, $attr[v]);
            }
            scale
        }};
    }
    let pos_scale = rescale_attr!(mesh.v);

    // also normalize colors
    let [l, h] = present_verts
        .clone()
        .map(|vi| mesh.vert_colors[vi])
        .fold([[F::INFINITY; 3], [F::NEG_INFINITY; 3]], |[l, h], n| {
            [from_fn(|i| l[i].min(n[i])), from_fn(|i| h[i].max(n[i]))]
        });
    let mid_col = kmul(0.5, add(l, h));
    for vi in present_verts.clone() {
        mesh.vert_colors[vi] = sub(mesh.vert_colors[vi], mid_col);
    }
    let col_scale = rescale_attr!(mesh.vert_colors);

    let mut m = CollapsibleManifold::new_with_remapping(remap.subset(vert_range.clone()), |vi| {
        assert!(vert_range.contains(&(vi + offset)));
        let pos = mesh.v[vi + offset];
        let area = 1e-6;
        let mut q = Quadric::<3>::zero();
        for d in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] {
            q += Quadric::new_plane(pos, d, area);
            q.area += area;
        }
        let vc = mesh.vert_colors[vi + offset];
        q += Quadric::degen_attr(vc, attr_ws) * area;
        q *= args.degen_quadric_weight;

        (q, pos)
    });
    if args.display_progress {
        println!("[INFO(QEM)]: target tris: {num_tris} -> {target_num_tris}");
    }

    let mut num_edges = 0;
    let mut avg_edge_len = 0.;

    // faces for each vertex
    let mut face_verts = vec![vec![]; num_v];
    bufs.face_normals.resize(mesh.f.len(), [0.; 3]);
    for fi in face_range.clone() {
        let f = &mesh.f[fi];
        bufs.face_normals[fi] = normalize(f.normal(&mesh.v));

        for e in f.edges_ord() {
            bufs.edge_face_adj
                .entry(e)
                .and_modify(|p| {
                    p.insert(fi);
                })
                .or_insert_with(|| EdgeKind::Boundary(fi));
            num_edges += 1;
            let [e0, e1] = e.map(|vi| mesh.v[vi]);
            avg_edge_len += length(sub(e1, e0));
        }

        for vi in f.as_slice() {
            face_verts[m.vertices.get_compress(vi - offset)].push(fi);
        }
    }
    for fv in &mut face_verts {
        fv.sort_unstable();
        fv.dedup();
    }

    avg_edge_len /= num_edges as F;

    for f in &mesh.f[face_range.clone()] {
        for es in f.edges() {
            let [e0, e1] = es.map(|vi| m.vertices.get_compress(vi - offset));
            m.add_edge(e0, e1);
        }
    }

    for fi in face_range.clone() {
        let f = &mesh.f[fi];
        if f.is_empty() {
            continue;
        }
        let area = f.area(&mesh.v).max(0.);
        let area = area + args.min_face_area;
        let n = normalize(bufs.face_normals[fi]);

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
                    let fn0 = normalize(bufs.face_normals[$f0]);
                    let fn1 = normalize(bufs.face_normals[$f1]);
                    let angle = dot(fn0, fn1);
                    assert!((-1.0001..=1.0001).contains(&angle), "{angle}");
                    let angle = angle.clamp(-1., 1.);

                    let v = angle.acos();
                    assert!((0.0..=PI).contains(&v), "{v} {angle}");
                    v
                }};
            }

            let e_w = match bufs.edge_face_adj[&e] {
                EdgeKind::Boundary(_) => 256.,
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

            let colpw =
                dist(mesh.vert_colors[v], mesh.vert_colors[ni]) * args.color_preservation_weight;

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

    macro_rules! update_cost_of_edge {
        ($e0:expr, $e1: expr) => {{
            let [e0, e1] = std::cmp::minmax($e1, $e0);
            let mut q_acc = QuadricAccumulator::default();
            let &(q0, p0) = m.get(e0);
            q_acc += q0;
            let &(q1, p1) = m.get(e1);
            q_acc += q1;
            let p = q_acc.point_with_volume_opt().unwrap_or_else(|| kmul(0.5, add(p0, p1)));
            assert!(p.iter().copied().all(F::is_finite));
            let mut total_cost = 0.;

            let q01f = m.get(e0).0 + m.get(e1).0;
            // colors are also automatically clamped to [0., 1.].
            let attrs = q01f.attributes(p, attr_ws);
            total_cost -=
                q01f.cost_attrib(p, attrs, attr_ws).max(0.) - curr_costs[e0] - curr_costs[e1];

            NotNan::new(total_cost).unwrap()
        }};
    }

    for [e0, e1] in m.ord_edges() {
        bufs.pq.push([e0, e1], update_cost_of_edge!(e0, e1));
    }

    let mut curr_tris = mesh.f[face_range.clone()]
        .iter()
        .map(|f| f.num_tris())
        .sum::<usize>();
    let p = args
        .display_progress
        .then(|| indicatif::ProgressBar::new(curr_tris as u64));

    let mut edge_counts = BTreeMap::new();
    let mut seen_faces = vec![];
    let mut adj_verts = vec![];
    'outer: while let Some((e, q)) = bufs.pq.pop() {
        debug_assert!(bufs.snd_pq.is_empty());
        bufs.snd_pq.push(e, (0, q));
        bufs.recencies.clear();
        'inner: while let Some(([e0, e1], (rec, q_err))) = bufs.snd_pq.pop() {
            assert!(e0 < e1);
            if m.is_deleted(e0) || m.is_deleted(e1) {
                continue;
            }
            if locked(e0 + offset) || locked(e1 + offset) {
                continue;
            }
            if m.num_vertices() < target_verts {
                break 'outer;
            }
            if curr_tris < target_num_tris {
                break 'outer;
            }

            // check if an output edge would be non-manifold
            edge_counts.clear();
            seen_faces.clear();
            adj_verts.clear();
            for e in [e0, e1] {
                for &adj_fi in &face_verts[e] {
                    if seen_faces.contains(&adj_fi) {
                        continue;
                    }
                    seen_faces.push(adj_fi);
                    let mut adj_f = mesh.f[adj_fi].clone();
                    adj_f.remap(|vi| {
                        if vi == e0 + offset || vi == e1 + offset {
                            e1 + offset
                        } else {
                            m.vertices.get_compress(vi - offset) + offset
                        }
                    });
                    if adj_f.canonicalize() {
                        continue;
                    }
                    for e in adj_f.all_pairs_ord() {
                        if e[0] == e[1] {
                            continue;
                        }
                        *edge_counts.entry(e).or_insert(0) += 1;
                    }
                    for e in adj_f.edges() {
                        if e[0] == e1 + offset {
                            adj_verts.push(e[1]);
                        } else if e[1] == e1 + offset {
                            adj_verts.push(e[0]);
                        }
                    }
                }
            }
            // there is a non-manifold edge introduced
            if edge_counts.values().any(|&v| v > 2) {
                continue;
            }

            let mut q_acc = QuadricAccumulator::default();
            let &(q0, p0) = m.get(e0);
            let &(q1, p1) = m.get(e1);
            q_acc += q0;
            q_acc += q1;
            let pos = q_acc.point_with_volume_opt().unwrap_or_else(|| kmul(0.5, add(p0, p1)));
            let q01 = q0 + q1;
            let attr = q01.attributes(pos, attr_ws);

            macro_rules! check_normal_orientation {
              ($e: expr) => {{
                for &adj_fi in &face_verts[$e] {
                    let mut adj_f = mesh.f[adj_fi].clone();
                    adj_f.remap(|vi| {
                        if vi == e0 + offset || vi == e1 + offset {
                            e1 + offset
                        } else {
                            m.vertices.get_compress(vi - offset) + offset
                        }
                    });
                    if adj_f.canonicalize() {
                        continue;
                    }
                    let new_n = adj_f.normal_with(|vi| if vi == e1 { pos } else { m.get(vi - offset).1 });
                    let prev_n = bufs.face_normals[adj_fi];
                    if dot(prev_n, new_n) < 0. {
                        continue 'inner;
                    }
                }
              }}
            }

            if !args.no_check_face_inversion {
                check_normal_orientation!(e0);
                check_normal_orientation!(e1);
            }

            // -- Commit

            if let Some(adj_faces) = bufs.edge_face_adj.get(&[e0, e1]) {
                for &af in adj_faces.as_slice() {
                    let f = &mesh.f[af];
                    let Some([q0, q1]) = f.quad_opp_edge(e0, e1) else {
                        continue;
                    };
                    let r = bufs
                        .recencies
                        .entry(std::cmp::minmax(q0, q1))
                        .or_insert(rec);
                    *r += 100;
                }
            };

            m.merge(e0, e1, |_, _| {
                curr_costs[e1] = q01.cost_attrib(pos, attr, attr_ws).max(0.);
                (q01, pos)
            });
            debug_assert!(m.is_deleted(e0));
            debug_assert!(!m.is_deleted(e1));

            let [ef0, ef1] = face_verts.get_disjoint_mut([e0, e1]).unwrap();
            ef1.append(ef0);
            ef1.sort_unstable();
            ef1.dedup();
            ef1.retain(|&fi| {
                mesh.f[fi].remap(|vi| m.get_new_vertex(vi - offset) + offset);
                !mesh.f[fi].canonicalize()
            });
            for &mut fi in ef1 {
                bufs.face_normals[fi] =
                    normalize(mesh.f[fi].normal_with(|vi| m.get(vi - offset).1));
            }

            bufs.did_update.clear();
            let e_dst = m.get_new_vertex(e1);
            for adj in m.vertex_adj(e_dst) {
                let prio = update_cost_of_edge!(e_dst, adj);
                let adj_e = std::cmp::minmax(e_dst, adj);
                bufs.snd_pq.remove(&adj_e);
                bufs.pq.push(adj_e, prio);
                bufs.did_update.push(adj_e);
            }

            for adj in m.vertex_adj(e_dst) {
                for adj2 in m.vertex_adj(adj) {
                    let adj_e = std::cmp::minmax(adj, adj2);
                    if adj2 == e_dst || bufs.did_update.contains(&adj_e) {
                        continue;
                    }

                    bufs.did_update.push(adj_e);
                    let prio = update_cost_of_edge!(adj, adj2);
                    let recency = bufs.recencies.get(&adj_e).copied().unwrap_or(0);

                    if !approx_eq(*prio, *q_err, args.abs_eps) {
                        bufs.snd_pq.remove(&adj_e);
                        bufs.pq.push(adj_e, prio);
                        continue;
                    }
                    let changed = bufs
                        .snd_pq
                        .change_priority(&adj_e, (recency, prio))
                        .is_some();
                    if !changed {
                        bufs.pq.push(adj_e, prio);
                    }
                }
            }

            while let Some((_e, nq_err)) = bufs.pq.peek()
                && approx_eq(**nq_err, *q_err, args.abs_eps)
            {
                let (e, nq_err) = bufs.pq.pop().unwrap();
                let recency = bufs.recencies.get(&e).copied().unwrap_or(0);
                bufs.snd_pq.push(e, (recency, nq_err));
            }

            // can assume it's usually 2 since it's manifold
            curr_tris -= 2;
            if let Some(ref p) = p {
                p.set_position(curr_tris as u64);
            }
        }
    }

    for (vi, &(q, p)) in m.vertices() {
        let vi = vi + offset;

        mesh.v[vi] = p;
        let c = q.attributes(p, attr_ws);
        mesh.vert_colors[vi] = c;
    }

    // denormalize all output vertices
    let present_verts = vert_range.clone().filter(|&v| remap.is_root(v));
    let inv_pos_scale = pos_scale.recip();
    let inv_col_scale = col_scale.recip();
    for vi in present_verts {
        mesh.v[vi] = add(kmul(inv_pos_scale, mesh.v[vi]), center);
        mesh.vert_colors[vi] =
            add(kmul(inv_col_scale, mesh.vert_colors[vi]), mid_col).map(|c| c.clamp(0., 1.));
    }
}

fn approx_eq(a: F, b: F, abs_eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < abs_eps
}

//fn denormalize(s: F, t: [F;3],
