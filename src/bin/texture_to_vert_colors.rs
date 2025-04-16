#![feature(cmp_minmax)]
#![feature(let_chains)]

use std::collections::BTreeMap;
use std::io::Write;

use clap::Parser;
use pars3d::image::{self, DynamicImage, GenericImageView};
use pars3d::{FaceKind, edge::EdgeKind};

use texture_to_vert_colors::{F, U, add, cross, dot, kmul, len_sq, length, normalize, sub};
use texture_to_vert_colors::{manifold::CollapsibleManifold, quadric::Quadric};

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

    /// Do not add corners (for debugging)
    #[arg(long, hide = true)]
    no_corners: bool,

    /// Do not fill space between pixel quads (for debugging)
    #[arg(long, hide = true)]
    no_fill: bool,

    /// Display some extra colors (for debugging)
    #[arg(long, hide = true)]
    debug_colors: bool,

    /// Do not normalize the mesh before processing (for debugging)
    #[arg(long, hide = true)]
    no_normalize: bool,

    /// Output stats to this file
    #[arg(long, default_value = "")]
    stats: String,

    /// How much to separate each pixel by, useful for debugging.
    #[arg(long, default_value_t = 0.999, hide = true)]
    pixel_sep: F,

    /// How much to pull each vertex associated with an edge toward it.
    #[arg(long, default_value_t = 0.5)]
    edge_pull: F,

    /// How much to pull each vertex associated with a vertex toward it.
    #[arg(long, default_value_t = 0.5)]
    vertex_pull: F,

    /// Do not simplify the output mesh.
    #[arg(long)]
    no_simplify: bool,

    /// During decimation, how heavily should colors be preserved?
    #[arg(long)]
    color_weight: F,
}

pub fn main() {
    let args = Args::parse();
    let mut scene = pars3d::load(&args.input).expect("Failed to parse input");
    let mut out_scene = scene.clone();
    let start = std::time::Instant::now();
    for (mi, mesh) in scene.meshes.iter_mut().enumerate() {
        let (s, t) = if args.no_normalize {
            (1., [0.; 3])
        } else {
            mesh.normalize()
        };
        assert!(!mesh.uv[0].is_empty());
        let new_mesh = texture_to_vert_colors(mesh, &scene.materials, &args);
        let mut new_mesh = if args.no_simplify {
            new_mesh
        } else {
            simplify_colored(new_mesh, &args)
        };
        new_mesh.denormalize(s, t);
        out_scene.meshes[mi] = new_mesh;
    }
    let elapsed = start.elapsed();
    println!("[INFO]: Resampling took {elapsed:?}");

    pars3d::save(&args.output, &out_scene).expect("Failed to save output");

    if !args.stats.is_empty() {
        let mut stat_file = std::fs::File::create(&args.stats).expect("Failed to open stats file");
        writeln!(
            stat_file,
            r#"{{
  "num_faces": {},
  "num_vertices": {},
  "input_num_faces": {},
  "input_num_vertices": {},
  "num_boundary_edges": {},
  "input_num_boundary_edges": {}
}}"#,
            out_scene.num_faces(),
            out_scene.num_vertices(),
            scene.num_faces(),
            scene.num_vertices(),
            out_scene
                .meshes
                .iter()
                .map(|m| m.num_boundary_edges())
                .sum::<usize>(),
            scene
                .meshes
                .iter()
                .map(|m| m.num_boundary_edges())
                .sum::<usize>(),
        )
        .expect("Failed to write stats");
    }
}

pub fn texture_to_vert_colors(
    mesh: &pars3d::Mesh,
    materials: &[pars3d::mesh::Material],
    args: &Args,
) -> pars3d::Mesh {
    let mut out = pars3d::Mesh::default();

    // Copy vertex colors for each input UV
    // TODO remove this assumption of a single material and copy per face with duplication
    let diff_img = if args.diffuse_img.is_empty() {
        let mati = mesh
            .single_mat()
            .expect("No material or more than 1 material for this mesh");
        let mat = &materials[mati];
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
    let mut labels = BTreeMap::new();

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

        let ok = match args.sample_kind {
            SampleKind::Approx => sample(
                mesh,
                f,
                &diff_img,
                &mut out.f,
                &mut out.v,
                &mut out.vert_colors,
            ),
            SampleKind::Exact => sample_exact(
                mesh,
                f,
                fi,
                &diff_img,
                &mut out.f,
                &mut out.v,
                &mut out.vert_colors,
                &mut corner_map,
                &mut edge_map,
                &mut labels,
                args,
            ),
        };
        if !ok {
            pars3d::save("curr_error.ply", &out.into_scene()).expect("Failed to save error scene");
            eprintln!("Exiting after saved erroneous mesh");
            std::process::exit(1);
        }
    }

    if args.no_corners {
        return out;
    }

    let mut edge_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();

    // Zip edges together
    for ([e0_key, e1_key], face_verts) in edge_map {
        // nothing to be done in this case
        if face_verts.len() != 2 {
            continue;
        }

        // Non manifold case may just be ok to ignore, not sure what to do there exactly.
        let e0 = e0_key.map(F::from_bits);
        let e1 = e1_key.map(F::from_bits);

        let (f0, _) = &face_verts[0];
        let swapped = mesh.f[*f0]
            .edges()
            .any(|e| e.map(|vi| mesh.v[vi]) == [e1, e0]);

        let [e0_key, e1_key] = if !swapped {
            [e1_key, e0_key]
        } else {
            [e0_key, e1_key]
        };
        assert_ne!(e0_key, e1_key, "temporary check for degenerate edges");

        macro_rules! add_key {
            ($dst: expr, $key: expr, $face: expr, $l: expr) => {{
                let fv = corner_map.get($key).unwrap();
                let &(_, corner) = fv.iter().find(|fv| fv.0 == *$face).unwrap();
                $dst.push((corner, $l));
                corner
            }};
        }

        let ordering = |vi: &usize| {
            let [(t0, ei0_key), (t1, ei1_key)] = labels[&vi];
            let t = if ei0_key == e0_key {
                assert_eq!(ei1_key, e1_key);
                t0
            } else {
                assert_eq!(ei0_key, e1_key);
                assert_eq!(ei1_key, e0_key);
                t1
            };
            (*vi, t as F)
        };

        let (f0, verts0) = &face_verts[0];
        let mut v0s = verts0.iter().map(ordering).collect::<Vec<_>>();

        let v0_max = v0s.iter().map(|v0| v0.1).max_by(F::total_cmp).unwrap_or(1.);
        for (_, v0t) in v0s.iter_mut() {
            *v0t = *v0t / (v0_max + 1.);
        }

        let v00 = add_key!(v0s, &e0_key, f0, 0.);
        let v01 = add_key!(v0s, &e1_key, f0, 1.);

        v0s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v0s.dedup_by_key(|v| v.0);

        let (f1, verts1) = &face_verts[1];
        let mut v1s = verts1.iter().map(ordering).collect::<Vec<_>>();
        let v1_max = v1s.iter().map(|v1| v1.1).max_by(F::total_cmp).unwrap_or(1.);
        for (_, v1t) in v1s.iter_mut() {
            *v1t = *v1t / (v1_max + 1.);
        }

        let v10 = add_key!(v1s, &e0_key, f1, 0.);
        let v11 = add_key!(v1s, &e1_key, f1, 1.);

        v1s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v1s.dedup_by_key(|v| v.0);

        let mut ins = |src, dst, idx| {
            assert_ne!(src, dst);
            assert_ne!(src, usize::MAX);
            assert_ne!(dst, usize::MAX);

            let v = edge_adj.entry(src).or_insert([usize::MAX; 2]);
            assert!(
                v[idx] == dst || v[idx] == usize::MAX,
                "{src} {} {dst} {v:?}",
                v[idx]
            );
            v[idx] = dst;
        };
        ins(v00, v10, 0);
        ins(v10, v00, 1);

        ins(v01, v11, 1);
        ins(v11, v01, 0);

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

        for (p0n, t) in v0s {
            out.f.push(FaceKind::Tri([v0_front.0, p0n, v1_front.0]));
            v0_front = (p0n, t);
        }
        for (p1n, t) in v1s {
            out.f.push(FaceKind::Tri([v0_front.0, p1n, v1_front.0]));
            v1_front = (p1n, t);
        }
    }

    // Zip corner faces together
    for (_, fvs) in corner_map.iter() {
        let first = fvs
            .iter()
            .find(|fv| edge_adj[&fv.1].iter().any(|&v| v == usize::MAX))
            .copied()
            .unwrap_or_else(|| fvs[0])
            .1;

        let face = match fvs.len() {
            0..3 => continue,
            // Order doesn't matter here (except if it should flip)
            3 => FaceKind::Tri(std::array::from_fn(|i| fvs[i].1)),

            4 => {
                let mut quad = [first, 0, 0, 0];
                quad[1] = *edge_adj[&first].iter().find(|&&v| v != usize::MAX).unwrap();
                assert_ne!(quad[0], usize::MAX);
                for i in 2..4 {
                    assert!(
                        edge_adj.contains_key(&quad[i - 1]),
                        "{} {:?}",
                        quad[i - 1],
                        quad
                    );
                    let n = edge_adj[&quad[i - 1]];
                    quad[i] = *n.iter().find(|&&v| v != quad[i - 2]).unwrap();
                }
                FaceKind::Quad(quad)
            }
            n => {
                let mut poly = vec![first];
                poly.reserve(n);
                let next = *edge_adj[&first].iter().find(|&&v| v != usize::MAX).unwrap();
                poly.push(next);
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
    // map from edge -> (original face idx, vertices), which stores vertices along every half edge
    corner_map: &mut BTreeMap<[U; 3], Vec<(usize, usize)>>,
    // map from edge -> (original face idx, vertices on face)
    edge_map: &mut BTreeMap<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
    // map from new vertex index to original vertex position and
    labels: &mut BTreeMap<usize, [(u32, [U; 3]); 2]>,

    args: &Args,
) -> bool {
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
    //assert!(v_f.area() > 1e-5, "{}", v_f.area());
    /*
    assert!(
        uv_f.area() > 1e-8,
        "TODO handle near 0 area uv face separately {}",
        uv_f.area()
    );
    */

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

        let delta = args.pixel_sep;
        let cfs = [
            [u + (1. - delta), v + (1. - delta)],
            [u + delta, v + (1. - delta)],
            [u + delta, v + delta],
            [u + (1. - delta), v + delta],
        ];
        let cfs = cfs.map(|[u, v]| [u / w as F, v / h as F]);

        let bary = uv_f.barycentric([(u + 0.5) / w as F, (v + 0.5) / h as F]);
        let [tex_u, tex_v] = uv_f.from_barycentric(bary);
        let rgb = get_rgb(tex_u, tex_v);

        let new_verts = cfs.map(|cf| {
            let bary = uv_f.barycentric(cf);
            let pos = v_f.from_barycentric(bary);
            if !bary.iter().any(|&b| b < 0.) {
                return pos;
            };
            nearest_point_on_tri(v_f.tri().unwrap(), pos)
        });

        // commit to this new pixel
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

    macro_rules! save_bad_mesh {
        ($label: expr) => {{
            let mut tmp = pars3d::Mesh::new_geometry(
                f_slice.iter().map(|vi| mesh.v[*vi]).collect(),
                vec![FaceKind::Tri([0, 1, 2])],
            );

            tmp.uv[CHAN] = f_slice.iter().map(|vi| mesh.uv[CHAN][*vi]).collect();
            pars3d::save("failing_tri.obj", &tmp.into_scene()).expect("Failed to save temp error");

            let error_verts = out_verts[start..].to_vec();
            let error_colors = out_colors[start..].to_vec();
            let mut error_faces = out_faces[start_f..].to_vec();
            for f in &mut error_faces {
                f.offset(-(start as i32))
            }
            *out_verts = error_verts;
            *out_colors = error_colors;
            *out_faces = error_faces;
            eprintln!($label);

            return false;
        }};
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
        return true;
    }

    // --- Adding faces in between pixel quads
    for (&[u, v], &[l, r, ur, ul]) in pixel_map.iter() {
        if args.no_fill {
            break;
        }
        // check bottom right first
        if !pixel_map.contains_key(&[u + 1, v + 1])
            && let Some(&[_, _, _, a]) = pixel_map.get(&[u + 1, v])
            && let Some(&[_, b, _, _]) = pixel_map.get(&[u, v + 1])
        {
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
                eprintln!(
                    r#"It might be necessary to add triangles to adjacent
                    faces here? Not sure
                    if this will be ever hit (it likely shouldn't be)."#
                );
                continue;
            }

            /* Definitely no faces to add */
            (None, None, None) => continue,

            // Handled earlier, no special cases to add
            (None, Some(_), None) | (None, None, Some(_)) => continue,
        };
        out_faces.push(corner_face);
    }

    let mut edge_face_adj: BTreeMap<[usize; 2], EdgeKind> = BTreeMap::new();
    // compute adjacent boundary edges and trace from the corners
    for (fi, f) in out_faces.iter().enumerate().skip(start_f) {
        for [e0, e1] in f.edges() {
            assert_ne!(e0, e1);
            edge_face_adj
                .entry(std::cmp::minmax(e0, e1))
                .and_modify(|v| assert!(v.insert(fi)))
                .or_insert(EdgeKind::Boundary(fi));
        }
    }

    // Compute adjacent vertices to each boundary vertex
    let mut vert_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
    for (&[ef0, ef1], ek) in edge_face_adj.iter() {
        assert_ne!(ef0, ef1);
        assert!(!ek.is_nonmanifold());
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
    debug_assert!(check);

    // --- Compute correspondence between original vertices and a single pixel vertex, which
    // must be a boundary.
    // This is only a vector since there are at most 3 vertices.
    let mut corner_verts: Vec<(usize, usize)> = vec![/* (og, new) */];
    for &og_vi in f_slice {
        if args.no_corners {
            break;
        }
        let vert_pos = mesh.v[og_vi];
        let [u, v] = mesh.uv[CHAN][og_vi];

        let u = ((u % 1.) * w as F - 0.5).floor() as i32;
        let v = ((v % 1.) * h as F - 0.5).floor() as i32;

        let mut nearest = 0;
        let mut best_dist = F::INFINITY;
        // TODO do this check more efficiently
        let range = [0, -1, 1, -2, 2];
        for i in range {
            for j in range {
                let nu = u + i;
                let nu = if nu >= 0 { nu } else { w as i32 + nu };
                let nu = if nu >= w as i32 { nu - w as i32 } else { nu };
                let nv = v + j;
                let nv = if nv >= 0 { nv } else { h as i32 + nv };
                let nv = if nv >= h as i32 { nv - h as i32 } else { nv };
                let p = [nu, nv];
                let Some(verts) = pixel_map.get(&p) else {
                    continue;
                };
                for &vi in verts.iter() {
                    // only boundary vertices
                    if !vert_adj.contains_key(&vi) {
                        continue;
                    }
                    // unique mapping
                    if corner_verts.iter().any(|&(_, p)| p == vi) {
                        continue;
                    }
                    let d = length(sub(out_verts[vi], vert_pos));
                    if d < best_dist {
                        nearest = vi;
                        best_dist = d;
                    }
                }
            }
        }
        if !best_dist.is_finite() {
            // naive search thru everything
            for (&new_vi, _) in vert_adj.range(start..) {
                if corner_verts.iter().any(|&(_, p)| p == new_vi) {
                    continue;
                }
                let d = length(sub(out_verts[new_vi], vert_pos));
                if d < best_dist {
                    nearest = new_vi;
                    best_dist = d;
                }
            }
        }
        if !best_dist.is_finite() {
            println!("{:?}", labels.range(start..).count());
            save_bad_mesh!("No nearby points");
        }
        assert_ne!(best_dist, F::INFINITY);

        let fv = corner_map.entry(vert_pos.map(F::to_bits)).or_default();
        assert!(!fv.iter().any(|fv| fv.0 == fi));
        // check that each corner vert corresponds to a single original vertex
        let check = corner_verts.iter().any(|&(_, new_vi)| new_vi == nearest);
        if check {
            save_bad_mesh!("One vertex maps to multiple original corners");
        }
        assert!(!check);

        fv.push((fi, nearest));
        corner_verts.push((og_vi, nearest));

        // Pull to a corner to make it tight (larger T is tighter)
        let t = args.vertex_pull;
        out_verts[nearest] = add(kmul(t, mesh.v[og_vi]), kmul(1. - t, out_verts[nearest]));

        if args.debug_colors {
            out_colors[nearest] = [1.; 3];
        }
    }
    if !args.no_corners {
        assert_eq!(corner_verts.len(), 3);
    }

    const INVALID_POS: [U; 3] = [U::MAX; 3];
    for &(og_vi, new_vi) in &corner_verts {
        let [l, r] = vert_adj[&new_vi];
        let mut iter = |mut curr: usize, mut prev: usize| {
            let mut c = 1;
            while !corner_verts.iter().any(|v| v.1 == curr) {
                let label = labels.entry(curr).or_insert([(0, INVALID_POS); 2]);
                /*
                assert!(
                    label.iter().any(|&v| v.1 == usize::MAX),
                    r#"label = {label:?} og_vi = {og_vi} new_vi = {new_vi} corner_verts = {corner_verts:?}
                    #verts = {:?} vert_adj = {:?} curr = {curr} c = {c:?} vert_adj[new_vi] = {:?}"#,
                    out_verts[start..].len(), vert_adj[&curr], vert_adj[&new_vi]
                );
                */
                /*
                if let Some(p) = label.iter_mut().find(|v| v.1 == og_vi.map(F::to_bits)) {
                    assert!(false);
                    p.0 = p.0.min(c);
                } else {
                */
                *label.iter_mut().find(|v| v.1 == INVALID_POS).unwrap() =
                    (c, mesh.v[og_vi].map(F::to_bits));
                //}
                assert!(vert_adj[&curr].iter().any(|&v| v == prev));
                let next = *vert_adj[&curr].iter().find(|v| **v != prev).unwrap();
                prev = curr;
                curr = next;
                c += 1;
            }
        };

        iter(l, new_vi);
        iter(r, new_vi);
    }
    let check = labels
        .range(start..)
        .all(|(_, v)| v[0].1 != INVALID_POS && v[1].1 != INVALID_POS);
    assert!(check);

    for (&new_vi, ogs) in labels.range(start..) {
        if args.debug_colors && out_colors[new_vi] != [1.; 3] {
            out_colors[new_vi] = [0.; 3];
        }
        let [og0_key, og1_key] = ogs.map(|vi| vi.1);
        let fvs = edge_map
            .entry(std::cmp::minmax(og0_key, og1_key))
            .or_default();
        if let Some(fv) = fvs.iter_mut().find(|fv| fv.0 == fi) {
            fv.1.push(new_vi);
        } else {
            fvs.push((fi, vec![new_vi]));
        }

        let ogs = [og0_key, og1_key].map(|k| k.map(F::from_bits));
        let t = nearest_on_line(out_verts[new_vi], ogs);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        let tgt_pos = add(ogs[0], kmul(t, sub(ogs[1], ogs[0])));
        // pull edge verts to the edge (larger is closer to edge)
        let t = args.edge_pull;
        out_verts[new_vi] = add(kmul(1. - t, out_verts[new_vi]), kmul(t, tgt_pos));
    }

    true
}

pub fn sample(
    mesh: &pars3d::Mesh,
    f: &FaceKind,
    diff_img: &DynamicImage,
    out_faces: &mut Vec<FaceKind>,
    out_verts: &mut Vec<[F; 3]>,
    out_colors: &mut Vec<[F; 3]>,
) -> bool {
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
    true
}

/// Computes the value `t` such that `s + (s-e)t = nearest point to p on line`
pub fn nearest_on_line(p: [F; 3], [s, e]: [[F; 3]; 2]) -> F {
    let dir = sub(e, s);
    dot(dir, sub(p, s)) / dot(dir, dir)
}

/// Returns the nearest point to p on tri.
pub fn nearest_point_on_tri([v0, v1, v2]: [[F; 3]; 3], p: [F; 3]) -> [F; 3] {
    let v10 = sub(v1, v0);
    let v21 = sub(v2, v1);
    let v02 = sub(v0, v2);

    let [p0, p1, p2] = [v0, v1, v2].map(|v| sub(p, v));

    let nrm = cross(v10, v21);
    let c = dot(cross(v10, nrm), p0) > 0.
        && dot(cross(v21, nrm), p1) > 0.
        && dot(cross(v02, nrm), p2) > 0.;
    if c {
        return sub(p, kmul(dot(nrm, p0) / len_sq(nrm), nrm));
    }

    // compute point on each edge
    let [q0, q1, q2] = [(v0, p0, v10), (v1, p1, v21), (v2, p2, v02)].map(|(v, p, e)| {
        let t = (dot(p, e) / len_sq(e)).clamp(0., 1.);
        add(v, kmul(t, e))
    });

    // pick nearest one
    let [ds0, ds1, ds2] = [q0, q1, q2].map(|q| len_sq(sub(q, p)));
    if ds0 < ds1 && ds0 < ds2 {
        q0
    } else if ds1 < ds2 {
        q1
    } else {
        q2
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

impl<const N: usize> Default for AABB<F, N> {
    fn default() -> Self {
        Self::new()
    }
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

pub fn simplify_colored(mesh: pars3d::Mesh, args: &Args) -> pars3d::Mesh {
    let color_weight = args.color_weight;
    let start_time = std::time::Instant::now();
    let mut m = CollapsibleManifold::new_with(mesh.v.len(), |vi| {
        (Quadric::<6>::zero(), mesh.v[vi], mesh.vert_colors[vi])
    });
    println!("[INFO]: Took {:?} for decimation", start_time.elapsed());

    let mut out = pars3d::Mesh::default();
    out
}
/*
*/
