#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]
#![feature(let_chains)]

use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use clap::Parser;
use ordered_float::NotNan;
use pars3d::image::{self, DynamicImage, GenericImageView};
use pars3d::{FaceKind, edge::EdgeKind};
use priority_queue::PriorityQueue;

use texture_to_vert_colors::{
    F, U, add, cross, cross_2d, dot, kmul, len_sq, length, normalize, sub,
};
use texture_to_vert_colors::{
    aabb::AABB,
    manifold::CollapsibleManifold,
    quadric::{AttrWeights, Quadric, QuadricAccumulator},
};

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
    #[arg(long, default_value_t = 0.995)]
    pixel_sep: F,

    /// How much to pull each vertex associated with an edge toward it.
    #[arg(long, default_value_t = 0.5)]
    edge_pull: F,

    /// How much to pull each vertex associated with a vertex toward it.
    #[arg(long, default_value_t = 0.5)]
    vertex_pull: F,

    /*
    /// Do not simplify the output mesh.
    #[arg(long)]
    simplify: bool,

    /// During decimation, how heavily should colors be preserved?
    #[arg(long, default_value_t = 1e-4)]
    color_weight: F,

    /// Extra weight to add on each edge based on color differences
    #[arg(long, default_value_t = 0.1)]
    color_preservation_weight: F,

    /// Minimum face area during decimation.
    #[arg(long, default_value_t = 1e-2)]
    min_face_area: F,

    /// Minimum edge weight for each edge.
    #[arg(long, default_value_t = 1e-2)]
    min_edge_weight: F,

    /// Epsilon value to use when comparing quadric errors.
    #[arg(long, default_value_t = 1e-5)]
    abs_eps: F,

    /// Threshold to stop quadric decimation at
    #[arg(long, default_value_t = 1e-4)]
    quadric_threshold: F,
    */
    /// Area below which faces can be deleted
    #[arg(long, default_value_t = 1e-12)]
    area_threshold: F,

    /// Distance below which colors are considered similar
    #[arg(long, default_value_t = 1e-4)]
    color_diff_threshold: F,
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
        let mut new_mesh = texture_to_vert_colors(mesh, &scene.materials, &args);
        let deleted = delete_degenerate_faces(&mut new_mesh, &args);
        println!("[INFO]: Deleted {deleted} degenerate faces");
        let del_vert = new_mesh.delete_unused_vertices();
        println!("[INFO]: Deleted {del_vert} unused vertices");

        /*
        let mut new_mesh = if !args.simplify {
            new_mesh
        } else {
            simplify_colored(new_mesh, &args)
        };
        */
        new_mesh.denormalize(s, t);
        out_scene.meshes[mi] = new_mesh;
    }
    let elapsed = start.elapsed();
    println!("[INFO]: Resampling took {elapsed:?}");
    println!(
        "[INFO]: Output #F = {}, #V = {}",
        out_scene.num_faces(),
        out_scene.num_vertices()
    );

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

        //let start = out.v.len();
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
                &mut out.n,
                &mut corner_map,
                &mut edge_map,
                &mut labels,
                args,
            ),
        };
        // Add a simplification step here, as some faces are relatively similar, and we want to
        // delete degenerate faces.
        if !ok {
            pars3d::save("curr_error.ply", &out.into_scene()).expect("Failed to save error scene");
            eprintln!("Exiting after saved erroneous mesh");
            std::process::exit(1);
        }
    }

    if args.no_corners {
        return out;
    }

    // map from (new vertex -> adjacent vertices that share the same corner)
    let mut edge_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();

    // Zip edges together
    for ([e0_key, e1_key], face_verts) in edge_map {
        macro_rules! add_key {
            ($dst: expr, $key: expr, $face: expr, $l: expr) => {{
                let fv = &corner_map[$key];
                let corner = fv.iter().find(|fv| fv.0 == *$face).unwrap().1;
                assert_eq!(fv.iter().filter(|fv| fv.0 == *$face).count(), 1);
                $dst.push((corner, $l));
                corner
            }};
        }
        assert!(!face_verts.is_empty());

        // Only add boundary edges here
        let f0 = face_verts[0].0;

        let e0 = e0_key.map(F::from_bits);
        let e1 = e1_key.map(F::from_bits);

        let swapped = mesh.f[f0]
            .edges()
            .any(|e| e.map(|vi| mesh.v[vi]) == [e1, e0]);

        let [e0_key, e1_key] = if !swapped {
            [e1_key, e0_key]
        } else {
            [e0_key, e1_key]
        };

        if face_verts.len() == 1 {
            for key in [e0_key, e1_key] {
                let fv = &corner_map[&key];
                let corner = fv.iter().find(|fv| fv.0 == f0).unwrap().1;
                edge_adj.entry(corner).or_insert([usize::MAX; 2]);
            }

            continue;
        }
        assert_eq!(face_verts.len(), 2);
        assert_ne!(
            e0_key, e1_key,
            "temporary check for degenerate edges {e0:?} {e1:?}"
        );

        // Non manifold case may just be ok to ignore (for now), not sure what to do there exactly.

        let ordering = |vi: &usize| {
            let [(t0, ei0_key), (t1, ei1_key)] = labels[vi];
            let t = if ei0_key == e0_key {
                debug_assert_eq!(ei1_key, e1_key);
                t0
            } else {
                debug_assert_eq!(ei0_key, e1_key);
                debug_assert_eq!(ei1_key, e0_key);
                t1
            };
            (*vi, t as F)
        };

        let (f0, verts0) = &face_verts[0];
        let mut v0s = verts0.iter().map(ordering).collect::<Vec<_>>();

        let v0_max = v0s.iter().map(|v0| v0.1).max_by(F::total_cmp).unwrap_or(1.);
        for (_, v0t) in v0s.iter_mut() {
            *v0t /= v0_max + 1.;
        }

        let v00 = add_key!(v0s, &e0_key, f0, 0.);
        let v01 = add_key!(v0s, &e1_key, f0, 1.);

        v0s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v0s.dedup_by_key(|v| v.0);

        let (f1, verts1) = &face_verts[1];
        let mut v1s = verts1.iter().map(ordering).collect::<Vec<_>>();
        let v1_max = v1s.iter().map(|v1| v1.1).max_by(F::total_cmp).unwrap_or(1.);
        for (_, v1t) in v1s.iter_mut() {
            *v1t /= v1_max + 1.;
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
        assert!(fvs.iter().all(|fv| fv.1 < out.v.len()));

        if fvs.iter().any(|fv| !edge_adj.contains_key(&fv.1)) {
            out.f.push(FaceKind::from_iter(fvs.iter().map(|fv| fv.1)));
            continue;
        };
        let first = fvs
            .iter()
            .find(|fv| edge_adj[&fv.1].iter().any(|&v| v == usize::MAX))
            .copied()
            .unwrap_or_else(|| fvs[0])
            .1;
        assert_ne!(first, usize::MAX);

        let face = match fvs.len() {
            0..3 => continue,
            // Order doesn't matter here (except if it should flip)
            3 => FaceKind::Tri(std::array::from_fn(|i| fvs[i].1)),

            4 => {
                let mut quad = [first, 0, 0, 0];
                let Some(&next) = edge_adj[&first].iter().find(|&&v| v != usize::MAX) else {
                    out.f
                        .push(FaceKind::Quad(std::array::from_fn(|i| fvs[i].1)));
                    continue;
                };
                quad[1] = next;
                assert_ne!(quad[0], usize::MAX);
                for i in 2..4 {
                    let Some(n) = edge_adj.get(&quad[i - 1]) else {
                        quad = std::array::from_fn(|i| fvs[i].1);
                        break;
                    };
                    let Some(&next) = n.iter().find(|&&v| v != quad[i - 2] && v != usize::MAX)
                    else {
                        quad = std::array::from_fn(|i| fvs[i].1);
                        break;
                    };
                    quad[i] = next;
                }
                FaceKind::Quad(quad)
            }
            n => {
                let mut poly = vec![first];
                poly.reserve(n);
                let Some(&next) = edge_adj[&first].iter().find(|&&v| v != usize::MAX) else {
                    out.f
                        .push(FaceKind::Poly(fvs.iter().map(|fv| fv.1).collect()));
                    continue;
                };
                poly.push(next);
                for i in 2..n {
                    let Some(n) = edge_adj.get(&poly[i - 1]) else {
                        poly = fvs.iter().map(|fv| fv.1).collect();
                        break;
                        // not sure why this happens, but ok to ignore for now?
                    };
                    let Some(&next) = n.iter().find(|&&v| v != poly[i - 2] && v != usize::MAX)
                    else {
                        poly = fvs.iter().map(|fv| fv.1).collect();
                        break;
                    };
                    poly.push(next);
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
    out_normals: &mut Vec<[F; 3]>,
    // map from edge -> (original face idx, vertices), which stores vertices along every half edge
    corner_map: &mut BTreeMap<[U; 3], Vec<(usize, usize)>>,
    // map from edge -> (original face idx, vertices on face)
    edge_map: &mut BTreeMap<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
    // map from new vertex index to original vertex position and distance
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
    let iaabb = aabb.round_to_i32();
    let uv_f = f.map_kind(|vi| mesh.uv[0][vi]);
    let v_f = f.map_kind(|vi| mesh.v[vi]);
    let n_f = if !mesh.n.is_empty() {
        Some(f.map_kind(|vi| mesh.n[vi]))
    } else {
        None
    };
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

        /*
        let pix = AABB::from([cfs[0], cfs[1]]);
        if !pix.intersects_tri(uv_f.tri().unwrap()) {
            continue;
        }
        */

        let barys = cfs.map(|cf| uv_f.barycentric(cf));

        // TODO this needs to correspond to the distance of the pixel on the triangle
        let check = (0..3).any(|i| barys.iter().all(|bary| bary[i] < -0.3));
        if check {
            continue;
        }

        let bary = uv_f.barycentric([(u + 0.5) / w as F, (v + 0.5) / h as F]);
        let [tex_u, tex_v] = uv_f.from_barycentric(bary);
        let rgb = get_rgb(tex_u, tex_v);

        let raw_pos = barys.map(|bary| v_f.from_barycentric(bary));

        let new_verts = std::array::from_fn(|i| {
            let bary = barys[i];
            let pos = raw_pos[i];
            let normal = n_f.as_ref().map(|n_f| n_f.from_barycentric(bary));
            if !bary.iter().any(|&b| b < 0.) {
                return (pos, normal);
            };

            let tri = v_f.tri().unwrap();
            let new_pos = nearest_point_on_tri(tri, pos);
            let new_normal = n_f
                .as_ref()
                .map(|n_f| n_f.from_barycentric(v_f.barycentric(new_pos)));
            assert!(
                v_f.barycentric(new_pos)
                    .iter()
                    .all(|v| (0.0..=1.0).contains(v))
            );
            (new_pos, new_normal)
        });

        // commit to this new pixel
        let new_verts = new_verts.map(|(new_vert, normal)| {
            let vi = out_verts.len();
            out_verts.push(new_vert);
            out_colors.push(rgb);
            if let Some(normal) = normal {
                out_normals.push(normal);
            }

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

    if args.no_corners {
        return true;
    }

    // --- Compute correspondence between original vertices and a single pixel vertex, which
    // must be a boundary.
    // This is only a vector since there are at most 3 vertices.
    let mut corner_verts: [(usize, usize); 3] = [(usize::MAX, usize::MAX /* og, new) */); 3];
    for (ci, &og_vi) in f_slice.iter().enumerate() {
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
        corner_verts[ci] = (og_vi, nearest);

        // Pull to a corner to make it tight (larger T is tighter)
        let t = args.vertex_pull;
        out_verts[nearest] = add(kmul(t, mesh.v[og_vi]), kmul(1. - t, out_verts[nearest]));

        if args.debug_colors {
            out_colors[nearest] = [1.; 3];
        }
    }

    const INVALID_POS: [U; 3] = [U::MAX; 3];
    for &(og_vi, new_vi) in &corner_verts {
        assert!(vert_adj.contains_key(&new_vi), "{new_vi} {corner_verts:?}");
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

/// Returns the nearest point to p on tri, assuming the tri, point and direction all lie in the
/// same plane
pub fn nearest_point_on_tri_in_dir(
    vs: [[F; 3]; 3],
    p: [F; 3],
    n: [F; 3],
    dir: [F; 3],
) -> Option<[F; 3]> {
    // the plane is centered at p, with normal `n`, and x-axis defined by dir
    assert!(dot(n, dir).abs() < 1e-6);
    let dir = normalize(dir);
    let tan = cross(dir, n);

    let vs = vs.map(|v| {
        let local = sub(v, p);
        [dir, tan].map(|d| dot(d, local))
    });

    (0..3)
        // compute length along line of intersection
        .map(|i| {
            let t = cross_2d(vs[i], [1., 0.]) / cross_2d(vs[(i + 1) % 3], vs[i]);
            (i, t)
        })
        // check if between endpoints
        .filter(|(_, t)| (0.0..=1.0).contains(t))
        // ensure that the ray is only in the positive direction
        .filter_map(|(i, t)| {
            let pos = add(vs[i], kmul(t, sub(vs[(i + 1) % 3], vs[i])));
            let ray_extent = dot(pos, [1., 0.]);
            if ray_extent < 0. {
                return None;
            }
            Some((ray_extent, pos))
        })
        // take nearest point
        .min_by(|(a, _), (b, _)| a.partial_cmp(&b).unwrap())
        // remap back to 3d.
        .map(|(_, pos)| add(kmul(pos[0], dir), kmul(pos[1], tan)))
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

/*
pub fn simplify_colored(mesh: pars3d::Mesh, args: &Args) -> pars3d::Mesh {
    let start_time = std::time::Instant::now();

    let mut m =
        CollapsibleManifold::new_with(mesh.v.len(), |vi| (Quadric::<3>::zero(), mesh.v[vi]));

    let attr_ws = AttrWeights {
        ws: [args.color_weight; 3],
    };

    let mut edge_face_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();
    let mut f_n = vec![[0.; 3]; mesh.f.len()];
    let mut num_edges = 0;
    let mut avg_edge_len = 0.;
    for (fi, f) in mesh.f.iter().enumerate() {
        f_n[fi] = f.normal(&mesh.v);
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
    for &[e0, e1] in edge_face_adj.keys() {
        let v = length(sub(mesh.vert_colors[e0], mesh.vert_colors[e1]));
        assert_eq!(color_dists.insert([e0, e1], v), None);
    }

    for f in mesh.f.iter() {
        m.add_face(f.as_slice());
    }

    for (fi, f) in mesh.f.iter().enumerate() {
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
                dot(e0, e1).clamp(-1., 1.).acos() / std::f64::consts::PI as F
            };
            let mut q = Quadric::new_plane(curr, n, area) * interior_angle;
            q.area = area;
            m.data[v].0 += q;
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
                EdgeKind::Boundary(_) => 2.,
                EdgeKind::Manifold([a, b]) => dihedral_angle!(a, b) / PI,
                EdgeKind::NonManifold(_) => todo!(),
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

            m.data[v].0 += edge_quadric;
            m.data[ni].0 += edge_quadric;
        }

        /*
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
            m.data[vi].0 += q_attr * area;
        }
        */
    }

    let mut curr_costs = vec![0.; m.num_vertices()];
    let mut pq = PriorityQueue::new();

    macro_rules! update_cost_of_edge {
        ($e0:expr, $e1: expr) => {{
            let [e0, e1] = std::cmp::minmax($e1, $e0);
            let mut q_acc = QuadricAccumulator::default();
            for e in [e0, e1] {
                q_acc += m.get(e).0;
            }
            let p = q_acc.point_with_volume();
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
        pq.push([e0, e1], update_cost_of_edge!(e0, e1));
    }

    let p = indicatif::ProgressBar::new(m.num_vertices() as u64);
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
            if *q_err >= args.quadric_threshold {
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
                let prio = update_cost_of_edge!(e_dst, adj);
                let adj_e = std::cmp::minmax(e_dst, adj);
                buf.remove(&adj_e);
                pq.push(adj_e, prio);
                did_update.push(adj_e);
            }

            for adj in m.vertex_adj(e_dst) {
                for adj2 in m.vertex_adj(adj) {
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

    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut new_positions = vec![];
    let mut new_colors = vec![];

    for (curr_vi, (vi, &(q, p))) in m.vertices().enumerate() {
        let prev = remap.insert(vi, curr_vi);
        assert_eq!(prev, None);
        new_positions.push(p);

        let attrs = q.attributes(p, attr_ws);
        assert!(!attrs.is_empty());
        assert!(attrs.len() == 3);
        new_colors.push(attrs.map(|c| c.clamp(0., 1.)));
    }

    let mut face_set = std::collections::BTreeSet::new();
    let mut vertex_buf = vec![];
    for (fi, f) in mesh.f.iter().enumerate() {
        vertex_buf.clear();
        vertex_buf.extend_from_slice(f.as_slice());
        for v in vertex_buf.iter_mut() {
            *v = remap[&m.get_new_vertex(*v)];
        }
        vertex_buf.dedup();
        while !vertex_buf.is_empty() && vertex_buf.first() == vertex_buf.last() {
            vertex_buf.pop();
        }
        if vertex_buf.len() < 3 {
            continue;
        }
        consistent_face_ordering(&mut vertex_buf);
        let face = match vertex_buf.as_slice() {
            &[] | &[_] | &[_, _] => unreachable!(),
            &[a, b, c] => FaceKind::Tri([a, b, c]),
            &[a, b, c, d] => FaceKind::Quad([a, b, c, d]),
            x => FaceKind::Poly(x.to_vec()),
        };
        face_set.insert((
            face,
            mesh.face_mesh_idx.get(fi).copied().unwrap_or(0),
            mesh.mat_for_face(fi),
        ));
    }

    let mut fs = vec![];
    let mut og_mis = vec![];
    let mut og_mats = vec![];
    for (f, mi, mat) in face_set.into_iter() {
        fs.push(f);
        og_mis.push(mi);
        og_mats.push(mat);
    }
    println!("{:?}", fs.len());
    let face_set = fs;

    println!("[INFO]: Took {:?} for decimation", start_time.elapsed());
    pars3d::Mesh {
        v: new_positions,
        uv: std::array::from_fn(|_| vec![]),
        n: vec![],
        f: face_set,
        face_mesh_idx: og_mis,
        face_mat_idx: pars3d::mesh::convert_opt_usize(&og_mats),

        joint_weights: vec![],
        joint_idxs: vec![],
        vert_colors: new_colors,
        name: String::new(),
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
*/

pub fn delete_degenerate_faces(mesh: &mut pars3d::Mesh, args: &Args) -> usize {
    let mut deleted = 0;
    let mut remap = HashMap::new();

    let mut deletable = vec![true; mesh.v.len()];
    for f in &mesh.f {
        for [e0, e1] in f.edges() {
            let [vc0, vc1] = [e0, e1].map(|vi| mesh.vert_colors[vi]);
            if length(sub(vc0, vc1)) > args.color_diff_threshold {
                deletable[e0] = false;
                deletable[e1] = false;
            }
        }
    }

    // for each vertex need to compute whether it can be deleted,
    // based on whether the 1 ring of the vertex all have similar values.
    mesh.f.retain(|f| {
        let f_s = f.as_slice();
        if f_s.iter().any(|vi| remap.contains_key(vi)) {
            return true;
        }
        let area = f.area(&mesh.v).abs();
        if area > args.area_threshold {
            return true;
        }
        let n = f_s.len().max(1) as F;
        let avg_color = f_s
            .iter()
            .map(|&vi| mesh.vert_colors[vi])
            .fold([0.; 3], add)
            .map(|v| v / n);
        if !f_s.iter().all(|&vi| deletable[vi]) {
            return true;
        }

        let new_vert = f_s
            .iter()
            .map(|&vi| mesh.v[vi])
            .fold([0.; 3], add)
            .map(|v| v / n);
        let new_vi = mesh.v.len();
        mesh.v.push(new_vert);
        mesh.vert_colors.push(avg_color);

        for &vi in f_s {
            assert_eq!(remap.insert(vi, new_vi), None);
        }

        deleted += 1;
        false
    });

    mesh.f.retain_mut(|f| {
        f.remap(|vi| remap.get(&vi).copied().unwrap_or(vi));
        !f.canonicalize()
    });

    deleted
}
