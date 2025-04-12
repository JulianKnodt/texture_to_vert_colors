#![feature(cmp_minmax)]
#![feature(let_chains)]

use clap::Parser;
use pars3d::image::{self, DynamicImage, GenericImageView};
use pars3d::{FaceKind, quad_area};
use std::collections::BTreeMap;

use texture_to_vert_colors::{F, U, dot, length, sub};

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

    #[arg(long, short = 'k', default_value_t = SampleKind::Exact)]
    sample_kind: SampleKind,
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
    let mati = mesh
        .single_mat()
        .expect("More than 1 material for this mesh");
    let mat = &scene.materials[mati];
    let diff_tex = mat
        .textures_by_kind(pars3d::mesh::TextureKind::Diffuse)
        .next()
        .expect("No diffuse texture?");
    let diff_img = diff_tex.image.as_ref().expect("No diffuse image?").flipv();

    let mut edge_map = BTreeMap::new();
    let mut corner_map = BTreeMap::new();
    let mut out_bary = vec![];

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
                &mut out_bary,
                &mut corner_map,
                &mut edge_map,
            ),
        }
    }

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
        assert!(e_len_sq > 0., "{e_len_sq}");

        let compute_t = |&vi: &usize| {
            let p = out.v[vi];
            let s = dot(e_dir, sub(p, e0));
            let t = s / e_len_sq;
            assert!(t.is_finite(), "{t}");
            assert!((0.0..=1.0).contains(&t), "{t}");
            (vi, t.clamp(0., 1.))
        };
        macro_rules! add_key {
            ($dst: expr, $key: expr, $face: expr, $l: expr) => {{
                if let Some(fv) = corner_map.get($key) {
                    if let Some((_, corner)) = fv.iter().find(|fv| fv.0 == *$face) {
                        $dst.push((*corner, $l));
                    }
                }
            }};
        }

        let (f0, verts0) = &face_verts[0];
        let mut v0s = verts0.iter().map(compute_t).collect::<Vec<_>>();

        add_key!(v0s, &e0_key, f0, 0.);
        add_key!(v0s, &e1_key, f0, 1.);

        v0s.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v0s.dedup_by_key(|v| v.0);

        let (f1, verts1) = &face_verts[1];
        let mut v1s = verts1.iter().map(compute_t).collect::<Vec<_>>();

        add_key!(v1s, &e0_key, f1, 0.);
        add_key!(v1s, &e1_key, f1, 1.);

        v1s.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v1s.dedup_by_key(|v| v.0);

        // iterate over both and add faces with the front
        let mut v0s = v0s.iter().copied().peekable();
        let mut v1s = v1s.iter().copied().peekable();

        let mut v0_front: (usize, F) = v0s.next().unwrap();
        let mut v1_front: (usize, F) = v1s.next().unwrap();

        // TODO replace this accumulation with picking the front that is closer to the opposite
        // previous front.
        while let [Some(&(p0n, t0n)), Some(&(p1n, t1n))] = [v0s.peek(), v1s.peek()] {
            debug_assert!(t0n >= v0_front.1);
            debug_assert!(t1n >= v1_front.1);
            let t0_delta = (v0_front.1 - t1n).abs();
            let t1_delta = (v1_front.1 - t0n).abs();

            let new_face = if t0_delta >= t1_delta {
                let new_face = FaceKind::Tri([v0_front.0, p0n, v1_front.0]);
                v0_front = (p0n, t0n);
                v0s.next().unwrap();
                new_face
            } else {
                let new_face = FaceKind::Tri([v0_front.0, p1n, v1_front.0]);
                v1_front = (p1n, t1n);
                v1s.next().unwrap();
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
    out_bary: &mut Vec<[F; 3]>,
    // map from edge -> (original idx, face -> vertices), which stores vertices along every half edge
    corner_map: &mut BTreeMap<[U; 3], Vec<(usize, usize)>>,
    edge_map: &mut BTreeMap<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
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

    // for each pixel, what vertices are associated with it?
    let mut pixel_map: BTreeMap<_, [usize; 4]> = BTreeMap::new();

    let _curr_start = out_verts.len();
    for c in iaabb.iter_coords() {
        let [u, v] = c.map(|v| v as F);

        const DELTA: F = 0.95;
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
            let bary = uv_f.barycentric(cf);
            num_in_face += bary.iter().all(|&v| 0. <= v && v <= 1.) as usize;
        }
        match num_in_face {
            0 => continue,
            _ => {}
        }

        let bary = uv_f.barycentric([(u + 0.5) / w as F, (v + 0.5) / h as F]);
        let rgb = {
            let [tex_u, tex_v] = uv_f.from_barycentric(bary);
            // compute color
            let tex_u = tex_u % 1.;
            let tex_u = if tex_u < 0. { 1. + tex_u } else { tex_u };
            let tex_v = tex_v % 1.;
            let tex_v = if tex_v < 0. { 1. + tex_v } else { tex_v };
            let rgba = image::imageops::sample_bilinear(diff_img, tex_u, tex_v).unwrap();
            let [r, g, b, _a] = rgba.0.map(|c| c as F / 255.);
            [r, g, b]
        };

        let barys = cfs.map(|cf| clamp_bary_to_tri(uv_f.barycentric(cf)));
        let new_verts = barys.map(|bary| v_f.from_barycentric(bary));

        if quad_area(new_verts).abs() < 1e-8 {
            continue;
        }

        let new_verts = std::array::from_fn(|i| {
            let new_vert = new_verts[i];
            let vi = out_verts.len();
            out_verts.push(new_vert);
            out_colors.push(rgb);
            out_bary.push(barys[i]);

            vi
        });
        assert_eq!(pixel_map.insert(c, new_verts), None);
        out_faces.push(FaceKind::Quad(new_verts));
    }

    // --- utility for nearest edge storage
    let nearest_edge = |bary: [F; 3]| {
        let i = bary.iter().position(|&v| v < 0.).unwrap();
        let [e0, e1] = std::array::from_fn(|j| f_slice[(j + i + 1) % f_slice.len()])
            .map(|vi| mesh.v[vi].map(F::to_bits));
        std::cmp::minmax(e0, e1)
    };

    let mut insert_to_edge_map = |vi /* new vertex */, e /* original edge */| {
        let face_verts = edge_map.entry(e).or_default();
        if let Some(fv) = face_verts.iter_mut().find(|fv| fv.0 == fi) {
            if !fv.1.contains(&vi) {
                fv.1.push(vi);
            }
        } else {
            face_verts.push((fi, vec![vi]));
        }
    };

    // --- Compute correspondence between original vertices and a single pixel corner.
    // This is only a vector since there are at most 3 vertices.
    let mut corner_pixs = vec![];
    for (f_idx, &vi) in f_slice.iter().enumerate() {
        let vert_pos = mesh.v[vi];
        let [u, v] = mesh.uv[CHAN][vi];

        let u = ((u % 1.) * w as F - 0.5).floor() as i32;
        let v = ((v % 1.) * h as F - 0.5).floor() as i32;

        let mut nearest = ([0; 2], 0);
        let mut best_dist = F::INFINITY;
        for i in [-1, 0, 1] {
            for j in [-1, 0, 1] {
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
        assert!(best_dist.is_finite());

        let (uv, i) = nearest;
        let pv = &pixel_map[&uv];
        let fv = corner_map.entry(vert_pos.map(F::to_bits)).or_default();
        assert!(!fv.iter().any(|fv| fv.0 == fi));
        fv.push((fi, pv[i]));
        corner_pixs.push(uv);

        // also mark previous and next verts
        let next = pv[(i + 1) % 4];

        let [n_e0, n_e1] =
            [vi, f_slice[(f_idx + 1) % f_slice.len()]].map(|vi| mesh.v[vi].map(F::to_bits));
        let n_e = std::cmp::minmax(n_e0, n_e1);
        insert_to_edge_map(next, n_e);

        let prev = pv[(i + 3) % 4];
        let [p_e0, p_e1] = [
            f_slice[if f_idx == 0 {
                f_slice.len() - 1
            } else {
                f_idx - 1
            }],
            vi,
        ]
        .map(|vi| mesh.v[vi].map(F::to_bits));
        let p_e = std::cmp::minmax(p_e0, p_e1);
        insert_to_edge_map(prev, p_e);

        // if opp is near to any edges, then also need to include that one.
        let opp = pv[(i + 2) % 2];
        let bary = v_f.barycentric(out_verts[opp]);
        let Some(_near_idx) = bary.iter().position(|&b| b < 1e-3) else {
            continue;
        };
        todo!();
    }

    // --- Computing nearest edge (if any) to each pixel
    for (&[u, v], verts) in pixel_map.iter() {
        if corner_pixs.contains(&[u, v]) {
            continue;
        }
        // for each pixel need to check if any of its boundaries is on an edge
        let cc_pixels = [
            [u - 1, v],
            [u - 1, v - 1],
            [u, v - 1],
            [u + 1, v - 1],
            [u + 1, v],
            [u + 1, v + 1],
            [u, v + 1],
            [u - 1, v + 1],
        ];
        // check if any of the adjacent pixels are negative?
        for i in 0..cc_pixels.len() {
            let c = cc_pixels[i];
            let n = cc_pixels[(i + 1) % cc_pixels.len()];

            if pixel_map.contains_key(&c) || pixel_map.contains_key(&n) {
                continue;
            }
            let bary_of_uv = |[u, v]: [i32; 2]| {
                uv_f.barycentric([(u as F + 0.5) / w as F, (v as F + 0.5) / h as F])
            };
            let c_edge = nearest_edge(bary_of_uv(c));
            let n_edge = nearest_edge(bary_of_uv(n));

            let prev = if i == 0 { 3 } else { ((i - 1) / 2) % 4 };
            let idxs = [i / 2, ((i + 1) / 2) % 4, prev];

            for i in idxs {
                insert_to_edge_map(verts[i], c_edge);
                if n_edge != c_edge {
                    insert_to_edge_map(verts[i], n_edge);
                }
            }
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
            (Some([_, _, a, _]), Some([_, _, _, b]), Some([_, c, _, _])) => {
                FaceKind::Quad([l, *c, *a, *b])
            }
            (None, Some([_, _, _, b]), Some([_, c, _, _])) => FaceKind::Tri([l, *c, *b]),
            (Some([_, _, a, _]), None, Some([_, c, _, _])) => FaceKind::Tri([l, *c, *a]),
            (Some([_, _, a, _]), Some([_, _, _, b]), None) => FaceKind::Tri([l, *a, *b]),
            (Some([_, _, _shared, _]), None, None) => {
                todo!(
                    r#"It might be necessary to add triangles to adjacent
                    faces here? Not sure
                    if this will be ever hit."#
                );
            }
            (None, None, None) => {
                /* No faces to add */
                continue;
            }
            _ => continue,
        };
        out_faces.push(corner_face);
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
        let rgba = image::imageops::sample_bilinear(diff_img, u, v).unwrap();
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

/// Returns the barycentric coordinates in the triangle closest to bs
fn clamp_bary_to_tri(bs: [F; 3]) -> [F; 3] {
    if bs.iter().all(|b| (0.0..=1.0).contains(b)) {
        return bs;
    }
    let new_bs = bs.map(|bs| bs.clamp(0., 1.));
    let s = new_bs.iter().sum::<F>();
    new_bs.map(|b| b / s)
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
