#![feature(cmp_minmax)]
#![feature(let_chains)]

use clap::Parser;
use pars3d::image::{self, DynamicImage, GenericImageView};
use pars3d::{FaceKind, edge::EdgeKind, quad_area};
use std::collections::BTreeMap;

use texture_to_vert_colors::{F, U, add, dot, kmul, length, normalize, sub};

/// A utility for converting a mesh with texture into a mesh with vertex colors without
/// drop in visual quality.
#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh path.
    #[arg(long, short)]
    input: String,
    /// Output mesh path (PLY).
    #[arg(long, short)]
    output: String,

    /// Path to alternative texture to use.
    #[arg(long, short, default_value = "")]
    diffuse_img: String,

    #[arg(long, short = 'k', default_value_t = SampleKind::Exact)]
    sample_kind: SampleKind,

    /// Do not add corners
    #[arg(long, hide = true)]
    no_corners: bool,

    /// Display some extra colors for debuggin
    #[arg(long, hide = true)]
    debug_colors: bool,
}

pub fn main() {
    let args = Args::parse();
    let scene = pars3d::load(&args.input).expect("Failed to parse input");
    let mut out_scene = scene.clone();
    for (mi, mesh) in scene.meshes.iter().enumerate() {
        assert!(!mesh.uv[0].is_empty());
        let new_mesh = texture_to_vert_colors(mesh, &scene, &args);
        out_scene.meshes[mi] = new_mesh;
    }
    pars3d::save(args.output, &out_scene).expect("Failed to save output");
}

pub fn texture_to_vert_colors(
    mesh: &pars3d::Mesh,
    scene: &pars3d::Scene,
    args: &Args,
) -> pars3d::Mesh {
    let mut out = pars3d::Mesh::default();

    // Copy vertex colors for each input UV
    // TODO remove this assumption of a single material and copy per face with duplication
    let diff_img = if args.diffuse_img.is_empty() {
        let mati = mesh
            .single_mat()
            .expect("More than 1 material for this mesh");
        let mat = &scene.materials[mati];
        let diff_tex = mat
            .textures_by_kind(pars3d::mesh::TextureKind::Diffuse)
            .next()
            .expect("No diffuse texture?");
        diff_tex.image.as_ref().expect("No diffuse image?").flipv()
    } else {
        pars3d::image::open(&args.diffuse_img)
            .expect("Failed to open diffuse image")
            .flipv()
    };

    let mut edge_map = BTreeMap::new();
    let mut corner_map = BTreeMap::new();

    for (fi, f) in mesh.f.iter().enumerate() {
        /*
        let Some(mati) = mesh.mat_for_face(fi) else {
            continue;
        };
        let mat = &scene.materials[mati];
        let Some(diff_tex) = mat
            .textures_by_kind(pars3d::mesh::TextureKind::Diffuse)
            .next()
        else {
            // TODO handle other kinds of textures as well.
            continue;
        };
        let Some(ref diff_img) = diff_tex.image else {
            continue;
        };
        */

        match args.sample_kind {
            SampleKind::Approx => sample(
                &mesh,
                f,
                &diff_img,
                &mut out.f,
                &mut out.v,
                &mut out.vert_colors,
            ),
            SampleKind::Exact => sample_exact(
                &mesh,
                f,
                fi,
                &diff_img,
                &mut out.f,
                &mut out.v,
                &mut out.vert_colors,
                &mut corner_map,
                &mut edge_map,
                &args,
            ),
        }
    }

    let mut edge_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();

    // Zip edges together
    for ([e0_key, e1_key], face_verts) in edge_map {
        // nothing to be done in this case
        if face_verts.len() <= 1 {
            continue;
        }
        assert_eq!(face_verts.len(), 2, "TODO handle non-manifold case");
        // Non manifold case may just be ok to ignore, not sure what to do there exactly.

        let e0 = e0_key.map(F::from_bits);
        let e1 = e1_key.map(F::from_bits);

        let (f0, _) = &face_verts[0];
        let swapped = mesh.f[*f0]
            .edges()
            .any(|e| e.map(|vi| mesh.v[vi]) == [e1, e0]);

        let ([e0, e1], [e0_key, e1_key]) = if !swapped {
            ([e1, e0], [e1_key, e0_key])
        } else {
            ([e0, e1], [e0_key, e1_key])
        };

        let e_dir = sub(e1, e0);
        let e_len_sq = dot(e_dir, e_dir);
        assert!(e_len_sq > 1e-3, "{e_len_sq}");

        let compute_t = |&vi: &usize| {
            let t = dot(e_dir, sub(out.v[vi], e0)) / e_len_sq;
            assert!(t.is_finite(), "{t}");
            assert!((0.0..=1.0).contains(&t), "{t}");
            (vi, t)
        };

        macro_rules! add_key {
            ($dst: expr, $key: expr, $face: expr, $l: expr) => {{
                if let Some(fv) = corner_map.get($key)
                    && let Some(&(_, corner)) = fv.iter().find(|fv| fv.0 == *$face)
                {
                    $dst.push((corner, $l));
                    Some(corner)
                } else {
                    None
                }
            }};
        }

        let (f0, verts0) = &face_verts[0];
        let mut v0s = verts0.iter().map(compute_t).collect::<Vec<_>>();

        let v00 = add_key!(v0s, &e0_key, f0, 0.);
        let v01 = add_key!(v0s, &e1_key, f0, 1.);

        v0s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v0s.dedup_by_key(|v| v.0);

        let (f1, verts1) = &face_verts[1];
        let mut v1s = verts1.iter().map(compute_t).collect::<Vec<_>>();

        let v10 = add_key!(v1s, &e0_key, f1, 0.);
        let v11 = add_key!(v1s, &e1_key, f1, 1.);

        v1s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v1s.dedup_by_key(|v| v.0);

        let mut ins = |src, dst, idx| {
            assert_ne!(src, dst);
            let v = edge_adj.entry(src).or_insert([usize::MAX; 2]);
            assert!(
                v[idx] == dst || v[idx] == usize::MAX,
                "{src} {} {dst} {v:?}",
                v[idx]
            );
            v[idx] = dst;
        };
        if let Some(e0) = v00
            && let Some(e1) = v10
        {
            ins(e0, e1, 0);
            ins(e1, e0, 1);
        }

        if let Some(e0) = v01
            && let Some(e1) = v11
        {
            ins(e0, e1, 1);
            ins(e1, e0, 0);
        }

        // iterate over both and add faces with the front
        let mut v0s = v0s.iter().copied().peekable();
        let mut v1s = v1s.iter().copied().peekable();

        let mut v0_front: (usize, F) = v0s.next().unwrap();
        let mut v1_front: (usize, F) = v1s.next().unwrap();

        while let [Some(&(p0n, t0n)), Some(&(p1n, t1n))] = [v0s.peek(), v1s.peek()] {
            debug_assert!(t0n >= v0_front.1);
            debug_assert!(t1n >= v1_front.1);
            let t0_delta = (v0_front.1 - t1n).abs();
            let t1_delta = (v1_front.1 - t0n).abs();

            let new_face = if (t0_delta - t1_delta).abs() < 5e-2 {
                let new_face = FaceKind::Quad([v0_front.0, p0n, p1n, v1_front.0]);
                v0_front = (p0n, t0n);
                v1_front = (p1n, t1n);
                assert!(v0s.next().is_some());
                assert!(v1s.next().is_some());
                new_face
            } else if t0_delta >= t1_delta {
                let new_face = FaceKind::Tri([v0_front.0, p0n, v1_front.0]);
                v0_front = (p0n, t0n);
                assert!(v0s.next().is_some());
                new_face
            } else {
                let new_face = FaceKind::Tri([v0_front.0, p1n, v1_front.0]);
                v1_front = (p1n, t1n);
                assert!(v1s.next().is_some());
                new_face
            };
            out.f.push(new_face);
        }

        while let Some((p0n, t)) = v0s.next() {
            out.f.push(FaceKind::Tri([v0_front.0, p0n, v1_front.0]));
            v0_front = (p0n, t);
        }
        while let Some((p1n, t)) = v1s.next() {
            out.f.push(FaceKind::Tri([v0_front.0, p1n, v1_front.0]));
            v1_front = (p1n, t);
        }
    }

    // Zip corner faces together
    for (_, fvs) in corner_map.iter() {
        let face = match fvs.len() {
            0..3 => continue,
            // Order doesn't matter here (except if it should flip)
            3 => FaceKind::Tri(std::array::from_fn(|i| fvs[i].1)),

            4 => {
                let mut out = [fvs[0].1, 0, 0, 0];
                out[1] = edge_adj[&out[0]][0];
                for i in 2..4 {
                    let n = edge_adj[&out[i - 1]];
                    out[i] = *n.iter().find(|&&v| v != out[i - 2]).unwrap();
                }
                FaceKind::Quad(out)
            }
            n => {
                let mut poly = vec![fvs[0].1];
                poly.reserve(n);
                poly.push(edge_adj[&poly[0]][0]);
                for i in 2..n {
                    let n = edge_adj[&poly[i - 1]];
                    poly.push(*n.iter().find(|&&v| v != poly[i - 2]).unwrap());
                }
                FaceKind::Poly(poly)
            }
        };
        out.f.push(face);
    }

    out
}

pub fn sample_exact(
    mesh: &pars3d::Mesh,
    f: &FaceKind,
    fi: usize,
    diff_img: &DynamicImage,
    out_faces: &mut Vec<FaceKind>,
    out_verts: &mut Vec<[F; 3]>,
    out_colors: &mut Vec<[F; 3]>,
    // map from edge -> (original idx, face -> vertices), which stores vertices along every half edge
    corner_map: &mut BTreeMap<[U; 3], Vec<(usize, usize)>>,
    edge_map: &mut BTreeMap<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,

    args: &Args,
) {
    const CHAN: usize = 0;
    let mut aabb = AABB::<F, 2>::new();
    let f_slice = f.as_slice();
    for uv in f_slice.iter().map(|&vi| mesh.uv[CHAN][vi]) {
        aabb.add_point(uv);
    }

    // TODO should these have -1?
    let (w, h) = diff_img.dimensions();
    aabb.scale_by(w as F, h as F);
    let mut iaabb = aabb.round_to_i32();
    iaabb.expand_by(1);
    let uv_f = f.map_kind(|v| mesh.uv[0][v]);
    let v_f = f.map_kind(|v| mesh.v[v]);
    // face normal for projecting bary
    let f_n = normalize(v_f.normal());
    assert!(length(f_n) > 1e-3);

    // for each pixel, what vertices are associated with it?
    let mut pixel_map: BTreeMap<_, [usize; 4]> = BTreeMap::new();

    let get_rgb = |u: F, v: F| {
        let u = u % 1.;
        let u = if u < 0. { 1. + u } else { u };
        let v = v % 1.;
        let v = if v < 0. { 1. + v } else { v };
        let rgba = image::imageops::sample_bilinear(diff_img, u as f32, v as f32).unwrap();
        let [r, g, b, _a] = rgba.0.map(|c| c as F / 255.);
        [r, g, b]
    };

    let start = out_verts.len();
    let start_f = out_faces.len();
    for c in iaabb.iter_coords() {
        let [u, v] = c.map(|v| v as F);

        const DELTA: F = 0.999;
        let cfs = [
            [u + (1. - DELTA), v + (1. - DELTA)],
            [u + DELTA, v + (1. - DELTA)],
            [u + DELTA, v + DELTA],
            [u + (1. - DELTA), v + DELTA],
        ];
        let cfs = cfs.map(|[u, v]| [u / w as F, v / h as F]);

        // check if any of the pixel corners are in the face, if not skip
        let mut num_in_face = 0;
        for cf in cfs {
            num_in_face += uv_f
                .barycentric(cf)
                .iter()
                .all(|&v| (0.0..=1.0).contains(&v)) as usize;
        }
        match num_in_face {
            0 => continue,
            _ => {}
        }

        let bary = uv_f.barycentric([(u + 0.5) / w as F, (v + 0.5) / h as F]);
        let [tex_u, tex_v] = uv_f.from_barycentric(bary);
        let rgb = get_rgb(tex_u, tex_v);

        let pos_barys = cfs.map(|cf| {
            let bary = uv_f.barycentric(cf);
            (v_f.from_barycentric(bary), bary)
        });

        let mut failed = false;
        const B_DELTA: F = 1e-4;
        let new_verts: [_; 4] = std::array::from_fn(|i| {
            let (pos, bary) = pos_barys[i];
            let Some(_ni) = bary.iter().position(|&b| b < B_DELTA) else {
                return pos;
            };
            let new_bary = bary.map(|v| v.max(B_DELTA));
            let s = new_bary.iter().sum::<F>();
            let new_bary = new_bary.map(|v| v / s);
            let proj_pos = v_f.from_barycentric(new_bary);

            let next = pos_barys[(i + 1) % 4].0;
            let prev = pos_barys[(i + 3) % 4].0;

            // project to line defined by next and prev

            let dir = sub(next, prev);
            let e_len_sq = dot(dir, dir);

            let t = dot(dir, sub(proj_pos, prev)) / e_len_sq;
            failed = failed || !(1e-8..=(1.0 - 1e-8)).contains(&t);
            proj_pos
        });
        if failed {
            continue;
        }

        if quad_area(new_verts) < 1e-9 {
            continue;
        }

        let new_verts = std::array::from_fn(|i| {
            let new_vert = new_verts[i];
            let vi = out_verts.len();
            out_verts.push(new_vert);
            out_colors.push(rgb);

            vi
        });

        assert_eq!(pixel_map.insert(c, new_verts), None);
        out_faces.push(FaceKind::Quad(new_verts));
    }

    // If no faces were created, use the original corners.
    if out_verts.len() == start {
        let mut verts = Vec::with_capacity(f_slice.len());
        for &vi in f_slice {
            let new_vi = out_verts.len();
            verts.push(new_vi);
            let pos = mesh.v[vi];
            let [u, v] = mesh.uv[CHAN][vi];
            out_verts.push(pos);
            out_colors.push(get_rgb(u, v));
            corner_map
                .entry(pos.map(F::to_bits))
                .or_default()
                .push((fi, new_vi));
        }
        for ([vi, ni], [new_vi, new_ni]) in f.edges().zip(pars3d::edges(&verts)) {
            let [v, n] = [vi, ni].map(|i| mesh.v[i].map(F::to_bits));
            let e = std::cmp::minmax(v, n);
            let fvs = edge_map.entry(e).or_default();
            assert!(!fvs.iter().any(|fv| fv.0 == fi));
            fvs.push((fi, vec![new_vi, new_ni]));
        }
        let f = match verts.len() {
            0..3 => unreachable!(),
            3 => FaceKind::Tri(std::array::from_fn(|i| verts[i])),
            4 => FaceKind::Quad(std::array::from_fn(|i| verts[i])),
            _ => FaceKind::Poly(verts),
        };
        out_faces.push(f);
        return;
    }

    // --- Compute correspondence between original vertices and a single pixel corner.
    // This is only a vector since there are at most 3 vertices.
    let mut corner_verts = vec![];
    for &og_vi in f_slice {
        if args.no_corners {
            continue;
        }
        let vert_pos = mesh.v[og_vi];
        let [u, v] = mesh.uv[CHAN][og_vi];

        let u = ((u % 1.) * w as F - 0.5).floor() as i32;
        let v = ((v % 1.) * h as F - 0.5).floor() as i32;

        let mut nearest = ([0; 2], 0);
        let mut best_dist = F::INFINITY;
        let range = [-2, -1, 0, 1, 2];
        for i in range {
            for j in range {
                let nu = u + i;
                let nu = if nu >= 0 { nu } else { w as i32 + nu };
                let nv = v + j;
                let nv = if nv >= 0 { nv } else { h as i32 + nv };
                let p = [nu, nv];
                let Some(verts) = pixel_map.get(&p) else {
                    continue;
                };
                for (i, &vi) in verts.iter().enumerate() {
                    let v = out_verts[vi];
                    let d = length(sub(v, vert_pos));
                    if d < best_dist {
                        nearest = (p, i);
                        best_dist = d;
                    }
                }
            }
        }
        assert_ne!(best_dist, F::INFINITY);

        // pixel & nearest vert idx
        let (uv, i) = nearest;

        let pv = &pixel_map[&uv];
        let fv = corner_map.entry(vert_pos.map(F::to_bits)).or_default();
        assert!(!fv.iter().any(|fv| fv.0 == fi));
        fv.push((fi, pv[i]));
        corner_verts.push((og_vi, pv[i]));

        // Pull to a corner to make it tight
        const T: F = 0.5;
        out_verts[pv[i]] = add(kmul(1. - T, mesh.v[og_vi]), kmul(T, out_verts[pv[i]]));

        if args.debug_colors {
            out_colors[pv[i]] = [1.; 3];
        }
    }

    // --- Adding faces in between pixel quads
    for (&[u, v], &[l, r, ur, ul]) in pixel_map.iter() {
        // check bottom right first

        if !pixel_map.contains_key(&[u + 1, v + 1])
            && let Some(&[_, _, _, a]) = pixel_map.get(&[u + 1, v])
            && let Some(&[_, b, _, _]) = pixel_map.get(&[u, v + 1])
        {
            // TODO here add tri face.
            out_faces.push(FaceKind::Tri([ur, a, b]));
        }
        // TODO turn these into checked subs?
        let left = pixel_map.get(&[u - 1, v]);
        let up = pixel_map.get(&[u, v - 1]);
        if let Some(&[_, or, our, _]) = left {
            out_faces.push(FaceKind::Quad([ul, our, or, l]));
        }
        if let Some(&[_, _, our, oul]) = up {
            out_faces.push(FaceKind::Quad([oul, our, r, l]));
        }
        let upleft = pixel_map.get(&[u - 1, v - 1]);
        let corner_face = match (upleft, up, left) {
            (Some(a), Some(b), Some(c)) => FaceKind::Quad([l, c[1], a[2], b[3]]),
            (None, Some(b), Some(c)) => FaceKind::Tri([l, c[1], b[3]]),
            (Some(a), None, Some(c)) => FaceKind::Tri([l, c[1], a[2]]),
            (Some(a), Some(b), None) => FaceKind::Tri([l, a[2], b[3]]),
            (Some([_, _, _shared, _]), None, None) => {
                todo!(
                    r#"It might be necessary to add triangles to adjacent
                    faces here? Not sure
                    if this will be ever hit (it likely shouldn't be)."#
                );
            }

            /* Definitely no faces to add */
            (None, None, None) => continue,

            _ => continue,
        };
        out_faces.push(corner_face);
    }

    let mut edge_face_adj: BTreeMap<[usize; 2], EdgeKind> = BTreeMap::new();
    // compute adjacent boundary edges and trace from the corners
    for (fi, f) in out_faces.iter().enumerate().skip(start_f) {
        for [e0, e1] in f.edges() {
            edge_face_adj
                .entry(std::cmp::minmax(e0, e1))
                .and_modify(|v| {
                    v.insert(fi);
                })
                .or_insert(EdgeKind::Boundary(fi));
        }
    }
    let mut vert_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
    for (&[ef0, ef1], ek) in edge_face_adj.iter() {
        if !ek.is_boundary() {
            continue;
        }

        let mut ins = |a, b| {
            *vert_adj
                .entry(a)
                .or_insert([usize::MAX; 2])
                .iter_mut()
                .find(|v| **v == usize::MAX)
                .unwrap() = b;
        };
        ins(ef0, ef1);
        ins(ef1, ef0);
    }
    let check = vert_adj
        .values()
        .all(|&[a0, a1]| a0 != usize::MAX && a1 != usize::MAX);
    assert!(check);

    let mut labels: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
    for &(og_vi, new_vi) in &corner_verts {
        let [l, r] = vert_adj[&new_vi];
        let mut iter = |mut curr: usize, mut prev: usize| {
            while !corner_verts.iter().any(|v| v.1 == curr) {
                let label = labels.entry(curr).or_insert([usize::MAX; 2]);
                assert!(label.iter().any(|&v| v == usize::MAX), "{label:?} {og_vi}");
                *label
                    .iter_mut()
                    .find(|v| **v == usize::MAX || **v == og_vi)
                    .unwrap() = og_vi;
                assert!(vert_adj[&curr].iter().any(|&v| v == prev));
                let next = *vert_adj[&curr].iter().find(|v| **v != prev).unwrap();
                prev = curr;
                curr = next;
            }
        };
        iter(l, new_vi);
        iter(r, new_vi);
    }
    let check = labels
        .values()
        .all(|&[l0, l1]| l0 != usize::MAX && l1 != usize::MAX);
    assert!(check);

    for (new_vi, ogs) in labels {
        if args.debug_colors && out_colors[new_vi] != [1.; 3] {
            out_colors[new_vi] = [0.; 3];
        }
        let [og0, og1] = ogs.map(|vi| mesh.v[vi].map(F::to_bits));
        let fvs = edge_map.entry(std::cmp::minmax(og0, og1)).or_default();
        if let Some(fv) = fvs.iter_mut().find(|fv| fv.0 == fi) {
            fv.1.push(new_vi);
        } else {
            fvs.push((fi, vec![new_vi]));
        }
    }
}

pub fn sample(
    mesh: &pars3d::Mesh,
    f: &FaceKind,
    diff_img: &DynamicImage,
    out_faces: &mut Vec<FaceKind>,
    out_verts: &mut Vec<[F; 3]>,
    out_colors: &mut Vec<[F; 3]>,
) {
    let mut aabb = AABB::<F, 2>::new();
    for uv in f.as_slice().iter().map(|&vi| mesh.uv[0][vi]) {
        aabb.add_point(uv);
    }
    // TODO should these have -1?
    let (w, h) = diff_img.dimensions();
    aabb.scale_by(w as F, h as F);
    let mut iaabb = aabb.round_to_i32();
    iaabb.expand_by(1);
    let uv_f = f.map_kind(|v| mesh.uv[0][v]);
    let v_f = f.map_kind(|v| mesh.v[v]);
    let mut pixel_map = BTreeMap::new();

    for c in iaabb.iter_coords() {
        let cf = c.map(|v| v as F + 0.5);
        let cf = [cf[0] / w as F, cf[1] / h as F];
        let bary = uv_f.barycentric(cf);
        // small epsilon to handle points which are very close to edges.
        const EPS: F = 1e-2;
        if !bary.iter().all(|v| (-EPS..=(1. + EPS)).contains(v)) {
            // outside the triangle
            continue;
        }
        let [u, v] = uv_f.from_barycentric(bary);
        // compute color
        let u = u % 1.;
        let u = if u < 0. { 1. + u } else { u };
        let v = v % 1.;
        let v = if v < 0. { 1. + v } else { v };
        let rgba = image::imageops::sample_bilinear(diff_img, u as f32, v as f32).unwrap();
        let [r, g, b, _a] = rgba.0.map(|c| c as F / 255.);

        // this is 0-1 so vertices align across edges
        let clamped_bary = bary.map(|v| v.clamp(0., 1.));
        let sum = clamped_bary.into_iter().sum::<F>();
        let clamped_bary = clamped_bary.map(|v| v / sum);
        let new_vert = v_f.from_barycentric(clamped_bary);

        let vi = out_verts.len();
        out_verts.push(new_vert);
        out_colors.push([r, g, b]);
        assert!(pixel_map.insert(c, vi).is_none());
    }

    // compute faces for each new vertex
    for (&[u, v], &vi) in pixel_map.iter() {
        // TODO determine if this is +1 or -1?
        let up = pixel_map.get(&[u, v + 1]).copied();
        let left = pixel_map.get(&[u + 1, v]).copied();
        let upleft = pixel_map.get(&[u + 1, v + 1]).copied();
        let new_face = match (up, upleft, left) {
            (None, None, None) => {
                /* TODO should form triangle with corners? */
                continue;
            }
            (Some(u), Some(ul), Some(l)) => FaceKind::Quad([l, ul, u, vi]),
            // All triples are a triangle
            (Some(u), None, Some(l)) => FaceKind::Tri([l, u, vi]),
            (Some(u), Some(ul), None) => FaceKind::Tri([ul, u, vi]),
            (None, Some(ul), Some(l)) => FaceKind::Tri([l, ul, vi]),
            // All doubles are steep triangles along the edge (handled elsewhere)
            (Some(_), None, None) | (None, Some(_), None) | (None, None, Some(_)) => continue,
        };
        out_faces.push(new_face);
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SampleKind {
    Approx,
    Exact,
}

impl_display!(SampleKind, Approx => "approx", Exact => "exact");

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB<T, const N: usize> {
    min: [T; N],
    max: [T; N],
}

impl<const N: usize> AABB<F, N> {
    pub fn new() -> Self {
        Self {
            min: [F::INFINITY; N],
            max: [F::NEG_INFINITY; N],
        }
    }
    pub fn add_point(&mut self, p: [F; N]) {
        for i in 0..N {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }
    pub fn round_to_i32(&self) -> AABB<i32, N> {
        AABB {
            min: self.min.map(|i| i.floor() as i32),
            max: self.max.map(|i| i.ceil() as i32),
        }
    }
    pub fn scale_by(&mut self, x: F, y: F) {
        self.min[0] *= x;
        self.max[0] *= x;

        self.min[1] *= y;
        self.max[1] *= y;
    }
}

impl AABB<i32, 2> {
    pub fn iter_coords(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        let [lx, ly] = self.min;
        let [hx, hy] = self.max;
        (ly..=hy).flat_map(move |y| (lx..=hx).map(move |x| [x, y]))
    }
    pub fn expand_by(&mut self, v: i32) {
        self.min = self.min.map(|val| val - v);
        self.max = self.max.map(|val| val + v);
    }
}
