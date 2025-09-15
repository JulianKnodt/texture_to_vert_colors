use std::cmp::minmax;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use ordered_float::NotNan;
use pars3d::FaceKind;
use priority_queue::PriorityQueue;
use union_find::UnionFind;

use super::{
    F, add, cross, dist_sq, dot, kmul, length,
    manifold::{CollapsibleManifold, EdgeKind},
    normalize,
    quadric::{AttrWeights, Quadric, QuadricAccumulator},
    sub,
};
pub struct Args {
    /// During decimation, how heavily should colors be preserved?
    pub color_weight: F,

    /// Additional scalar weighting to use for all edges.
    pub edge_weight: F,

    /// What percentage of faces should be retained at the end?
    pub target_tri_ratio: F,
    pub target_tri_num: usize,

    /// If progress should be displayed
    pub display_progress: bool,

    /// Absolute epsilon to use when switching to quad ordering
    pub abs_eps: F,

    /// Stop all simplification if this difference in color is exceeded
    pub color_diff_threshold: F,

    pub check_bd: bool,

    pub check_manifold: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            color_weight: 1.,
            //color_weight: 1e-3,
            edge_weight: 1.,

            display_progress: false,

            target_tri_ratio: 0.,
            target_tri_num: 100,

            //abs_eps: 5e-5, // has an outsized impact on the vertex colors, need to be careful
            // prefer 5e-7 usually
            abs_eps: 5e-7,
            color_diff_threshold: F::INFINITY,

            check_bd: true,

            check_manifold: true,
        }
    }
}

#[derive(Default)]
pub struct QEMBuffers {
    edge_face_adj: HashMap<[usize; 2], EdgeKind>,
    pq: PriorityQueue<[usize; 2], NotNan<F>>,
    // priority queue for recency
    snd_pq: PriorityQueue<[usize; 2], (u32, NotNan<F>)>,
    recency: HashMap<[usize; 2], u32>,
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
        self.did_update.clear();
        self.face_normals.clear();
        self.recency.clear();
        self.snd_pq.clear();
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
            let e = minmax(e0 - offset, e1 - offset);
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
        let n = normalize(bufs.face_normals[fi]);

        if length(n) == 0. {
            continue;
        }

        let f_slice = f.as_slice();
        for (i, &v) in f_slice.iter().enumerate() {
            let curr = mesh.v[v];
            let pi = f_slice[(i + f.len() - 1) % f.len()];
            let prev = mesh.v[pi];
            let ni = f_slice[(i + 1) % f.len()];
            let e = minmax(v, ni);
            let next = mesh.v[ni];

            let interior_angle = {
                let e0 = normalize(sub(prev, curr));
                let e1 = normalize(sub(next, curr));
                dot(e0, e1).clamp(-1., 1.).acos()
            };
            let mut q = Quadric::new_plane(curr, n, area) * interior_angle;
            q.area = area;
            m.data[v - offset].0 += q;

            if args.edge_weight <= 0. {
                continue;
            }

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

            let edge_dir = sub(curr, next);
            let edge_len = length(edge_dir);
            let edge_len = edge_len / avg_edge_len;
            if edge_len == 0. {
                continue;
            }
            let edge_dir = normalize(edge_dir);
            let edge_quadric = Quadric::new_plane(curr, normalize(cross(n, edge_dir)), 0.);

            let total_e_w = e_w * edge_len * args.edge_weight;
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
            let [e0, e1] = minmax($e1, $e0);
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
            let attrs = std::array::from_fn(|i| attrs[i].unwrap_or_else(|| (a0[i] + a1[i]) / 2.));
            let prev_cost =
                unsafe { *curr_costs.get_unchecked(e0) + *curr_costs.get_unchecked(e1) };
            let total_cost = q01f.cost_attrib(p, attrs, attr_ws).max(0.) - prev_cost;

            unsafe { NotNan::new_unchecked(-total_cost) }
        }};
    }

    for [e0, e1] in m.ord_edges() {
        bufs.pq.push([e0, e1], update_cost_of_edge!(e0, e1));
    }

    macro_rules! update_edge_face_adj {
        ($e0: expr, $e_dst: expr, $e0_adj:expr) => {{
            let e0 = $e0;
            let e_dst = $e_dst;
            debug_assert!(e0 <= e_dst);
            let adj = $e0_adj;

            debug_assert!(!m.is_deleted(adj));
            let e = minmax(adj, e0);

            let Some(prev_ef) = bufs.edge_face_adj.remove(&e) else {
                continue;
            };
            let new_e = minmax(adj, e_dst);

            use std::collections::hash_map::Entry;
            match bufs.edge_face_adj.entry(new_e) {
                Entry::Vacant(v) => {
                    v.insert(prev_ef);
                }
                Entry::Occupied(mut o) => {
                    // only keep faces which have 4 or more vertices
                    let faces_culled = prev_ef
                        .as_slice()
                        .into_iter()
                        .chain(o.get().as_slice())
                        .filter_map(|&fi| {
                            mesh.f[fi].remap(|v| match v == e0 {
                                true => e_dst,
                                false => m.get_new_vertex(v),
                            });
                            (!mesh.f[fi].is_degenerate()/*&& mesh.f[fi].len() > 4 */).then_some(fi)
                        });
                    match EdgeKind::from_iter(faces_culled) {
                        Some(mut ek) => {
                            ek.dedup_by_key(|fi| mesh.f[fi].as_slice());
                            o.insert(ek)
                        }
                        None => o.remove(),
                    };
                }
            }

            if let Some(prev) = bufs.recency.remove(&e) {
                bufs.recency
                    .entry(new_e)
                    .and_modify(|p| *p += prev)
                    .or_insert(prev);
            }
        }};
    }

    let mut curr_tris = mesh.f[face_range.clone()]
        .iter()
        .map(FaceKind::num_tris)
        .sum::<usize>();
    let init_tris = curr_tris;
    let p = args
        .display_progress
        .then(|| indicatif::ProgressBar::new(curr_tris as u64));

    let v0_adj = &mut bufs.v0_adj;
    let v1_adj = &mut bufs.v1_adj;
    'outer: while let Some(([e0, e1], q_err)) = bufs.pq.pop() {
        assert!(bufs.snd_pq.is_empty());
        bufs.recency.clear();
        bufs.snd_pq.push([e0, e1], (0, q_err));
        while let Some(([e0, e1], (rec, q_err))) = bufs.snd_pq.pop() {
            debug_assert!(e0 < e1);
            debug_assert!(vert_range.contains(&(e0 + offset)));
            debug_assert!(vert_range.contains(&(e1 + offset)));
            if m.is_deleted(e0) || m.is_deleted(e1) {
                continue;
            }

            if locked(e0 + offset) || locked(e1 + offset) {
                continue;
            }
            if curr_tris <= target_num_tris {
                break 'outer;
            }

            let is_bd = args.check_bd
                && bd_edges
                    .get(&e0)
                    .map(|bd_es| {
                        bd_es.iter().any(|bd_e| {
                            let [v0, v1] = bd_e.map(|v| m.get_new_vertex(v));
                            minmax(v0, v1) == [e0, e1]
                        })
                    })
                    .unwrap_or(false);

            debug_assert!((!is_bd) ^ bd_edges.contains_key(&e1));

            // link condition
            // https://github.com/cnr-isti-vclab/vcglib/blob/88c881d8393929c8e09b9df765ce8582bf386499/vcg/simplex/face/topology.h#L460
            macro_rules! all_adj_verts {
                ($dst: expr, $v: expr) => {{
                    $dst.clear();
                    let adj_verts = face_verts[$v].iter().flat_map(|&adj_fi| {
                        unsafe { mesh.f.get_unchecked(adj_fi) }
                            .as_triangle_fan()
                            .map(|t| t.map(|vi| m.get_new_vertex(vi - offset)))
                            .filter(|t| t.contains(&$v))
                            // remove degenerate tris
                            .filter(|[t0, t1, t2]| t0 != t1 && t0 != t2 && t1 != t2)
                            .flat_map(|t| t.into_iter())
                            .filter(|&v| v != $v)
                    });
                    $dst.extend(adj_verts);
                    $dst.sort_unstable();
                    $dst.dedup();
                }};
            }
            if args.check_manifold {
                all_adj_verts!(v0_adj, e0);
                all_adj_verts!(v1_adj, e1);

                let cnt = v1_adj.iter().filter(|v1a| v0_adj.contains(v1a)).count();

                if is_bd && cnt != 1 {
                    continue;
                } else if !is_bd && cnt != 2 {
                    continue;
                }
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
            let attr = std::array::from_fn(|i| attr[i].unwrap_or_else(|| (a0[i] + a1[i]) / 2.));
            if dist_sq(attr, a0).max(dist_sq(attr, a1)).sqrt() > args.color_diff_threshold {
                continue;
            }

            // -- Commit

            if let Some(adj_faces) = bufs.edge_face_adj.get(&[e0, e1]) {
                for &af in adj_faces.as_slice() {
                    let Some([q0, q1]) = mesh.f[af].quad_opp_edge(e0, e1) else {
                        continue;
                    };
                    *bufs.recency.entry(minmax(q0, q1)).or_insert(rec) += 1;
                }
            };

            for adj in m.vertex_adj(e0) {
                if adj == e1 {
                    continue;
                }
                update_edge_face_adj!(e0, e1, adj);

                let prev_e = std::cmp::minmax(e0, adj);
                bufs.pq.remove(&prev_e);
                bufs.snd_pq.remove(&prev_e);
            }

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
            use std::collections::btree_map::Entry;
            match (bd_edges.remove(&e0), bd_edges.entry(e1)) {
                (None, Entry::Vacant(_)) => {}
                (Some(mut adj_bds), Entry::Vacant(v)) => {
                    remap_bd!(adj_bds);
                    v.insert(adj_bds);
                }
                (None, Entry::Occupied(mut adj_bds)) => {
                    remap_bd!(adj_bds.get_mut());
                }
                (Some(mut bd0s), Entry::Occupied(mut bd1s)) => {
                    let bd1s = bd1s.get_mut();
                    remap_bd!(bd0s);
                    remap_bd!(bd1s);

                    /* // Old version with allocation
                    let mut new = vec![];
                    for &bd0 in &bd0s {
                        if !bd1s[0..og_bd1_len].contains(&bd0) {
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
                    */
                    // new version no alloc
                    let og_bd1_len = bd1s.len();
                    for &bd0 in &bd0s {
                        if !bd1s[0..og_bd1_len].contains(&bd0) {
                            bd1s.push(bd0);
                        }
                    }
                    let mut i = 0;
                    bd1s.retain(|v| {
                        if i >= og_bd1_len {
                            return true;
                        }
                        let keep = !bd0s.contains(v);
                        i += 1;
                        keep
                    });

                    bd1s.sort_unstable();
                    bd1s.dedup();
                }
            }

            let [ef0, ef1] = face_verts.get_disjoint_mut([e0, e1]).unwrap();
            let ef1_len = ef1.len();
            for f in std::mem::take(ef0) {
                if !ef1[0..ef1_len].contains(&f) {
                    ef1.push(f);
                }
            }
            let prev_tri = ef1
                .iter()
                .map(|&fi| unsafe { mesh.f.get_unchecked(fi) }.num_tris())
                .sum::<usize>();

            ef1.retain(|&fi| {
                let prev_f = unsafe { mesh.f.get_unchecked_mut(fi) };
                prev_f.remap(|vi| m.get_new_vertex(vi - offset) + offset);
                // important to not rotate here otherwise the ordering may change
                // leading to non-manifold edge introduction
                let retain = !prev_f.canonicalize_no_rotate();
                if !retain {
                    // necessary for triangle counting
                    *prev_f = FaceKind::empty();
                }
                retain
            });
            let new_tri = ef1
                .iter()
                .map(|&fi| unsafe { mesh.f.get_unchecked(fi) }.num_tris())
                .sum::<usize>();
            curr_tris -= prev_tri - new_tri;

            for &mut fi in ef1 {
                let f = unsafe { mesh.f.get_unchecked(fi) };
                *unsafe { bufs.face_normals.get_unchecked_mut(fi) } =
                    normalize(f.normal_with(|vi| m.get(vi - offset).1));
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
                    let rec = bufs.recency.get(&adj_e).copied().unwrap_or(0);

                    if !approx_eq(*prio, *q_err, args.abs_eps) {
                        bufs.snd_pq.remove(&adj_e);
                        bufs.pq.push(adj_e, prio);
                        continue;
                    }

                    let changed = bufs.snd_pq.change_priority(&adj_e, (rec, prio)).is_some();
                    if !changed {
                        bufs.pq.push(adj_e, prio);
                    }
                }
            }

            while let Some((ne, nq_err)) = bufs
                .pq
                .pop_if(|_, nq_err| approx_eq(**nq_err, *q_err, args.abs_eps))
            {
                let rec = bufs.recency.get(&ne).copied().unwrap_or(0);
                bufs.snd_pq.push(ne, (rec, nq_err));
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

fn approx_eq(a: F, b: F, abs_eps: F) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() < abs_eps
}
