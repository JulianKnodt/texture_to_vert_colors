#![feature(cmp_minmax)]

use clap::Parser;
use pars3d::FaceKind;
use pars3d::image::{self, DynamicImage, GenericImageView};
use std::collections::BTreeMap;

use texture_to_vert_colors::{F, U};

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
    let mut out = pars3d::Mesh::new_geometry(mesh.v.clone(), vec![]);
    out.n = mesh.n.clone();

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
    for &[u, v] in &mesh.uv[0] {
        let rgb = image::imageops::sample_bilinear(&diff_img, u, v).unwrap();
        let [r, g, b, _a] = rgb.0.map(|c| c as F / 255.);
        out.vert_colors.push([r, g, b]);
    }

    let mut locked_buf = vec![true; out.v.len()];
    let mut edge_map = BTreeMap::new();

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
                &mut locked_buf,
            ),
            SampleKind::Exact => sample_exact(
                &mesh,
                f,
                fi,
                &diff_img,
                &mut out.f,
                &mut out.v,
                &mut out.vert_colors,
                &mut locked_buf,
                &mut edge_map,
            ),
        }
    }

    // Zip edges together
    for ([e0, e1], face_verts) in edge_map {
        // nothing to be done in this case
        if face_verts.len() <= 1 {
            continue;
        }
        assert_eq!(face_verts.len(), 2, "TODO handle non-manifold case");
        // Non manifold case may just be ok to ignore, not sure what to do there exactly.

        let e0 = e0.map(F::from_bits);
        let e1 = e1.map(F::from_bits);
        let (dim, delta) = (0..3)
            .map(|i| (i, e1[i] - e0[i]))
            .max_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();

        let compute_t = |&vi: &usize| {
            let t = (mesh.v[vi][dim] - e0[dim]) / delta;
            (vi, t.clamp(0., 1.))
        };

        let (_f0, [ei00, ei01], verts0) = &face_verts[0];
        let mut v0s = vec![(*ei00, 0.), (*ei01, 1.)];
        v0s.extend(verts0.iter().map(compute_t));
        v0s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (_f1, [ei10, ei11], verts1) = &face_verts[1];
        let mut v1s = vec![(*ei10, 0.), (*ei11, 1.)];
        v1s.extend(verts1.iter().map(compute_t));
        v1s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // iterate over both and add faces with the front
        let mut v0s = v0s.iter().copied().peekable();
        let mut v1s = v1s.iter().copied().peekable();

        let mut v0_front: (usize, F) = v0s.next().unwrap();
        let mut v1_front: (usize, F) = v1s.next().unwrap();

        while let [Some(&(p0n, t0n)), Some(&(p1n, t1n))] = [v0s.peek(), v1s.peek()] {
            assert!(t0n >= v0_front.1);
            assert!(t1n >= v1_front.1);
            // pick the nearest of the two
            let t0_delta = t0n - v0_front.1;
            let t1_delta = t1n - v1_front.1;

            if t0_delta <= t1_delta {
                out.f.push(FaceKind::Tri([p0n, v0_front.0, v1_front.0]));
                v0_front = (p0n, t0n);
            } else {
                out.f.push(FaceKind::Tri([p1n, v0_front.0, v1_front.0]));
                v1_front = (p1n, t1n);
            }
        }
    }

    out
}

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

pub fn sample(
    mesh: &pars3d::Mesh,
    f: &FaceKind,
    diff_img: &DynamicImage,
    out_faces: &mut Vec<FaceKind>,
    out_verts: &mut Vec<[F; 3]>,
    out_colors: &mut Vec<[F; 3]>,
    lock_buf: &mut Vec<bool>,
) {
    let mut aabb = AABB::<F, 2>::new();
    for uv in f.as_slice().iter().map(|&vi| mesh.uv[0][vi]) {
        aabb.add_point(uv);
    }
    // For each vert, lock it if it's on a triangle boundary;
    // TODO should these have -1?
    let (w, h) = diff_img.dimensions();
    aabb.scale_by(w as F, h as F);
    let mut iaabb = aabb.round_to_i32();
    iaabb.expand_by(1);
    let uv_f = f.map_kind(|v| mesh.uv[0][v]);
    let v_f = f.map_kind(|v| mesh.v[v]);
    let mut vert_map = BTreeMap::new();

    let _curr_start = out_verts.len();
    for c in iaabb.iter_coords() {
        let cf = c.map(|v| v as F + 0.5);
        let cf = [cf[0] / w as F, cf[1] / h as F];
        let bary = uv_f.barycentric(cf);
        // small epsilon to handle points which are very close to edges.
        const EPS: F = 8e-3;
        if !bary.iter().all(|v| (-EPS..=(1. + EPS)).contains(v)) {
            // outside the triangle
            continue;
        }
        let locked = bary.iter().any(|v| !(EPS..=(1. - EPS)).contains(v));
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
        assert!(vert_map.insert(c, vi).is_none());
        lock_buf.push(locked);
    }

    // compute faces for each new vertex
    for (&[u, v], &vi) in vert_map.iter() {
        // TODO determine if this is +1 or -1?
        let up = vert_map.get(&[u, v + 1]).copied();
        let left = vert_map.get(&[u + 1, v]).copied();
        let upleft = vert_map.get(&[u + 1, v + 1]).copied();
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

pub fn sample_exact(
    mesh: &pars3d::Mesh,
    f: &FaceKind,
    fi: usize,
    diff_img: &DynamicImage,
    out_faces: &mut Vec<FaceKind>,
    out_verts: &mut Vec<[F; 3]>,
    out_colors: &mut Vec<[F; 3]>,
    lock_buf: &mut Vec<bool>,
    // map from edge -> (original idx, face -> vertices), which stores vertices along every half edge
    edge_map: &mut BTreeMap<[[U; 3]; 2], Vec<(usize, [usize; 2], Vec<usize>)>>,
) {
    let mut aabb = AABB::<F, 2>::new();
    let f_slice = f.as_slice();
    for uv in f_slice.iter().map(|&vi| mesh.uv[0][vi]) {
        aabb.add_point(uv);
    }
    // For each vert, lock it if it's on a triangle boundary;
    // TODO should these have -1?
    let (w, h) = diff_img.dimensions();
    aabb.scale_by(w as F, h as F);
    let mut iaabb = aabb.round_to_i32();
    iaabb.expand_by(1);
    let uv_f = f.map_kind(|v| mesh.uv[0][v]);
    let v_f = f.map_kind(|v| mesh.v[v]);

    // for each pixel, what vertices are associated with it?
    let mut pixel_map: BTreeMap<_, [usize; 4]> = BTreeMap::new();

    const EPS: F = 1e-2;
    let _curr_start = out_verts.len();
    for c in iaabb.iter_coords() {
        let [u, v] = c.map(|v| v as F);

        const DELTA: F = 0.999;
        let cfs = [
            [u, v],
            [u + DELTA, v],
            [u + DELTA, v + DELTA],
            [u, v + DELTA],
        ];
        let cfs = cfs.map(|[u, v]| [u / w as F, v / h as F]);

        // check if any of the pixel corners are in the face, if not skip
        let mut num_on_edge = 0;
        let mut num_in_face = 0;
        for cf in cfs {
            let bary = uv_f.barycentric(cf);
            num_in_face += bary.iter().all(|&v| 0. <= v && v <= 1.) as usize;
            num_on_edge += bary.iter().any(|&v| v <= EPS || v >= 1. - EPS) as usize;
        }
        match (num_in_face, num_on_edge) {
            (0, _) => continue,
            (_, _) => {}
        }

        let bary = uv_f.barycentric([(u + 0.5) / w as F, (v + 0.5) / h as F]);
        let [r, g, b] = {
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

        let new_verts = cfs.map(|cf| {
            let bary = uv_f.barycentric(cf);
            let bary = clamp_bary_to_tri(bary);
            let on_edge = bary.iter().any(|v| (EPS..=(1. - EPS)).contains(v));

            let (bary, nearest_edge) = if !on_edge /* TODO temp */ || true {
                (bary, None)
            } else {
                // TODO compute nearest point on edge in perpendicular direction
                /*
                // nearest edge idx
                let nei = bary
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.total_cmp(&b));
                let e = [f_slice[nei], f_slice[(nei+1) % f_slice.len()]];
                */
                todo!();
                #[allow(unreachable_code)]
                (bary, Some([0, 0]))
            };

            let new_vert = v_f.from_barycentric(bary);

            let vi = out_verts.len();
            out_verts.push(new_vert);
            out_colors.push([r, g, b]);
            // if on edge, lock it for later simplification
            lock_buf.push(on_edge);
            if let Some([nei0, nei1]) = nearest_edge {
                let [ne0, ne1] = [nei0, nei1].map(|vi| (mesh.v[vi].map(F::to_bits), vi));
                let [(ne0, nei0), (ne1, nei1)] = std::cmp::minmax(ne0, ne1);
                let face_verts = edge_map.entry([ne0, ne1]).or_default();
                if let Some(fv) = face_verts.iter_mut().find(|fv| fv.0 == fi) {
                    fv.2.push(vi);
                } else {
                    face_verts.push((fi, [nei0, nei1], vec![vi]));
                }
            }
            vi
        });
        assert_eq!(pixel_map.insert(c, new_verts), None);
        out_faces.push(FaceKind::Quad(new_verts));
    }

    for (&[u, v], &[l, r, _, ul]) in pixel_map.iter() {
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
                /*
                todo!(
                    r#"It might be necessary to add triangles to adjacent
                    faces here? Not sure
                    if this will be ever hit."#
                );
                */
                continue;
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
