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

    pub max_degree: usize,
    /// What percentage of faces should be retained at the end?
    pub target_tri_ratio: F,
    pub target_tri_num: usize,

    /// If progress should be displayed
    pub display_progress: bool,

    /// Stop all simplification if this difference in color is exceeded
    pub color_diff_threshold: F,

    pub check_bd: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            color_weight: 1.,
            color_preservation_weight: 5.,

            // XXX IMPORTANT this must be zero. Because there are many degenerate faces
            // if they are not zero, they will have an outsized impact on the final result.
            min_face_area: 0.,
            min_edge_weight: 1e-2,

            abs_eps: 1e-5,

            max_degree: 100,

            display_progress: false,

            target_tri_ratio: 0.,
            target_tri_num: 100,

            color_diff_threshold: F::INFINITY,

            check_bd: true,
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
    face_verts: Vec<Vec<usize>>,
    curr_costs: Vec<F>,
    v0_adj: Vec<usize>,
    v1_adj: Vec<usize>,
}

impl QEMBuffers {
    pub fn clear(&mut self) {
        self.edge_face_adj.clear();
        self.pq.clear();
        self.snd_pq.clear();
        self.recencies.clear();
        self.did_update.clear();
        self.face_normals.clear();
        // face verts handled separately
        self.curr_costs.clear();
        self.v0_adj.clear();
        self.v1_adj.clear();
    }
}

/// In-place simplification of planar faces of a mesh.
/// Returns how many faces were removed
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
) -> usize {
    bufs.clear();

    let offset = vert_range.start;
    let num_v = vert_range.end - offset;

    let mut bd_edges: BTreeMap<usize, Vec<[usize; 2]>> = BTreeMap::new();
    if args.check_bd {
        for [e0, e1] in mesh.boundary_edges() {
            if !vert_range.contains(&e0) || !vert_range.contains(&e1) {
                continue;
            }
            let e = std::cmp::minmax(e0 - offset, e1 - offset);
            for v in e {
                let be = bd_edges.entry(v).or_default();
                if !be.contains(&e) {
                    be.push(e);
                }
            }
        }
    }

    assert!(!mesh.vert_colors.is_empty());

    let cw = args.color_weight;
    let attr_ws = AttrWeights { ws: [cw; 3] };

    // ensure the faces are correct at this stage
    for f in &mut mesh.f[face_range.clone()] {
        f.remap(|vi| remap.get_compress(vi));
    }
    let num_tris = mesh.f[face_range.clone()]
        .iter()
        .map(FaceKind::num_tris)
        .sum::<usize>();
    let target_num_tris = (num_tris as F * args.target_tri_ratio).floor() as usize;
    let target_num_tris = target_num_tris.max(args.target_tri_num);
    if num_tris <= target_num_tris {
        return 0;
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
        let vc = mesh.vert_colors[vi + offset];
        (Quadric::zero(), pos, vc)
    });
    if args.display_progress {
        println!("[INFO(QEM)]: target tris: {num_tris} -> {target_num_tris}");
    }

    let mut num_edges = 0;
    let mut avg_edge_len = 0.;

    // faces for each vertex
    bufs.face_verts.resize_with(num_v, Vec::new);
    bufs.face_verts.iter_mut().for_each(Vec::clear);
    let face_verts = &mut bufs.face_verts;
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
    for fv in face_verts.iter_mut() {
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

    bufs.curr_costs.resize(num_v, 0.);
    let curr_costs = &mut bufs.curr_costs;

    macro_rules! update_cost_of_edge {
        ($e0:expr, $e1: expr) => {{
            let [e0, e1] = std::cmp::minmax($e1, $e0);
            let mut q_acc = QuadricAccumulator::default();
            let &(q0, p0, a0) = m.get(e0);
            q_acc += q0;
            let &(q1, p1, a1) = m.get(e1);
            q_acc += q1;
            let p = q_acc
                .point_with_volume_opt()
                .unwrap_or_else(|| kmul(0.5, add(p0, p1)));
            debug_assert!(p.iter().copied().all(F::is_finite));

            let q01f = q0 + q1;
            // colors are also automatically clamped to [0., 1.].
            let attrs = q01f.attributes_opt(p, attr_ws);
            let attrs = std::array::from_fn(|i| {
                if let Some(a) = attrs[i] {
                    a
                } else {
                    (a0[i] + a1[i]) / 2.
                }
            });
            let total_cost =
                q01f.cost_attrib(p, attrs, attr_ws).max(0.) - curr_costs[e0] - curr_costs[e1];

            NotNan::new(-total_cost).unwrap()
        }};
    }

    for [e0, e1] in m.ord_edges() {
        bufs.pq.push([e0, e1], update_cost_of_edge!(e0, e1));
    }

    let mut curr_tris = mesh.f[face_range.clone()]
        .iter()
        .map(|f| f.num_tris())
        .sum::<usize>();
    let init_tris = curr_tris;
    let p = args
        .display_progress
        .then(|| indicatif::ProgressBar::new(curr_tris as u64));

    let v0_adj = &mut bufs.v0_adj;
    let v1_adj = &mut bufs.v1_adj;
    'outer: while let Some((e, q)) = bufs.pq.pop() {
        assert!(bufs.snd_pq.is_empty());
        bufs.snd_pq.push(e, (0, q));
        bufs.recencies.clear();
        while let Some(([e0, e1], (rec, q_err))) = bufs.snd_pq.pop() {
            assert!(e0 < e1);
            assert!(vert_range.contains(&(e0 + offset)));
            assert!(vert_range.contains(&(e1 + offset)));
            if m.is_deleted(e0) || m.is_deleted(e1) {
                continue;
            }
            if locked(e0 + offset) || locked(e1 + offset) {
                continue;
            }
            if curr_tris <= target_num_tris {
                break 'outer;
            }

            let is_bd = bd_edges
                .get(&e0)
                .map(|bd_es| {
                    bd_es.iter().any(|bd_e| {
                        let [v0, v1] = bd_e.map(|v| m.get_new_vertex(v));
                        std::cmp::minmax(v0, v1) == [e0, e1]
                    })
                })
                .unwrap_or(false);

            if is_bd {
                assert!(bd_edges.contains_key(&e1));
            }

            // link condition
            // https://github.com/cnr-isti-vclab/vcglib/blob/88c881d8393929c8e09b9df765ce8582bf386499/vcg/simplex/face/topology.h#L460
            macro_rules! all_adj_verts {
                ($dst: expr, $v: expr) => {{
                    $dst.clear();
                    for &adj_fi in &face_verts[$v] {
                        let iter = mesh.f[adj_fi]
                            .as_triangle_fan()
                            .map(|t| t.map(|vi| vi - offset))
                            .map(|t| t.map(|vi| m.get_new_vertex(vi)))
                            .filter(|t| t.contains(&$v))
                            // remove degenerate tris
                            .filter(|[t0, t1, t2]| t0 != t1 && t0 != t2 && t1 != t2)
                            .flat_map(|t| t.into_iter())
                            .filter(|&v| v != $v);
                        $dst.extend(iter);
                    }
                    $dst.sort_unstable();
                    $dst.dedup();
                }};
            }
            all_adj_verts!(v0_adj, e0);
            all_adj_verts!(v1_adj, e1);

            let cnt = v1_adj.iter().filter(|v1a| v0_adj.contains(&v1a)).count();

            if is_bd && cnt != 1 {
                continue;
            } else if !is_bd && cnt != 2 {
                continue;
            }

            let mut q_acc = QuadricAccumulator::default();
            let &(q0, p0, a0) = m.get(e0);
            let &(q1, p1, a1) = m.get(e1);
            q_acc += q0;
            q_acc += q1;
            let pos = q_acc
                .point_with_volume_opt()
                .unwrap_or_else(|| kmul(0.5, add(p0, p1)));
            let q01 = q0 + q1;
            let attr = q01.attributes_opt(pos, attr_ws);
            let attr = std::array::from_fn(|i| {
                if let Some(a) = attr[i] {
                    a
                } else {
                    (a0[i] + a1[i]) / 2.
                }
            });
            if dist(attr, a0).max(dist(attr, a1)) > args.color_diff_threshold {
                continue;
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
                    *r += 1;
                }
            };

            m.merge(e0, e1, |_, _| {
                curr_costs[e1] = q01.cost_attrib(pos, attr, attr_ws).max(0.);
                (q01, pos, attr)
            });
            debug_assert!(m.is_deleted(e0));
            debug_assert!(!m.is_deleted(e1));

            macro_rules! remap_bd {
                ($adj_bds: expr) => {{
                    for adj_bd in $adj_bds.iter_mut() {
                        let [e0, e1] = adj_bd.map(|v| m.get_new_vertex(v));
                        *adj_bd = std::cmp::minmax(e0, e1);
                    }
                }};
            }
            match (bd_edges.remove(&e0), bd_edges.remove(&e1)) {
                (None, None) => {}
                (Some(mut adj_bds), None) => {
                    remap_bd!(adj_bds);
                    bd_edges.insert(e1, adj_bds);
                }
                (None, Some(mut adj_bds)) => {
                    remap_bd!(adj_bds);
                    bd_edges.insert(e1, adj_bds);
                }
                (Some(mut bd0s), Some(mut bd1s)) => {
                    remap_bd!(bd0s);
                    remap_bd!(bd1s);
                    // TODO delete shared between bd0s, bd1s to not allocate a new vector
                    let mut new = vec![];
                    for &bd0 in &bd0s {
                        if !bd1s.contains(&bd0) {
                            new.push(bd0);
                        }
                    }
                    for &bd1 in &bd1s {
                        if !bd0s.contains(&bd1) {
                            new.push(bd1);
                        }
                    }
                    new.sort_unstable();
                    new.dedup();
                    bd_edges.insert(e1, new);
                }
            }

            let [ef0, ef1] = face_verts.get_disjoint_mut([e0, e1]).unwrap();
            let ef1_len = ef1.len();
            for f in std::mem::take(ef0) {
                if !ef1[0..ef1_len].contains(&f) {
                    ef1.push(f);
                }
            }
            let prev_tri = ef1.iter().map(|&fi| mesh.f[fi].num_tris()).sum::<usize>();
            ef1.retain(|&fi| {
                mesh.f[fi].remap(|vi| m.get_new_vertex(vi - offset) + offset);
                // important to not rotate here otherwise the ordering may change
                // leading to non-manifold edge introduction
                let retain = !mesh.f[fi].canonicalize_no_rotate();
                if !retain {
                    // necessary for triangle counting
                    mesh.f[fi] = FaceKind::empty();
                }
                retain
            });
            let new_tri = ef1.iter().map(|&fi| mesh.f[fi].num_tris()).sum::<usize>();
            curr_tris -= prev_tri - new_tri;

            for &mut fi in ef1 {
                bufs.face_normals[fi] =
                    normalize(mesh.f[fi].normal_with(|vi| m.get(vi - offset).1));
            }

            // TEMPORARY check that face verts is correct

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

            if let Some(ref p) = p {
                p.set_position(curr_tris as u64);
            }
        }
    }

    for (vi, &(_, p, a)) in m.vertices() {
        let vi = vi + offset;

        mesh.v[vi] = p;
        mesh.vert_colors[vi] = a;
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

    init_tris - curr_tris
}

/*
fn sigmoid(x: F) -> F {
    1. / (1. + (-x).exp())
}
fn inv_sigmoid(y: F) -> F {
    y.ln() - (1. - y).ln()
}
*/

fn approx_eq(a: F, b: F, abs_eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < abs_eps
}
