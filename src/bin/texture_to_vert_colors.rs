#![feature(cmp_minmax)]
#![feature(let_chains)]
#![feature(assert_matches)]

use std::assert_matches::assert_matches;
use std::cmp::minmax;
use std::collections::{BTreeMap, HashSet};
use std::io::Write;

use clap::Parser;
use pars3d::image::{self, DynamicImage, GenericImageView};
use pars3d::{FaceKind, edge::EdgeKind};
use union_find::UnionFind;

use texture_to_vert_colors::aabb::AABB;
use texture_to_vert_colors::{
    F, U, add, cross, cross_2d, dot, kmul, len_sq, length, normalize, sub,
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

    /// Do not fill gaps between triangles (DEBUGGING)
    #[arg(long)]
    no_gap_fill: bool,

    /// Do not fill space between pixel quads (DEBUGGING)
    #[arg(long)]
    no_fill: bool,

    /// Display some extra colors (DEBUGGING)
    #[arg(long, hide = true)]
    debug_colors: bool,

    /// Do not normalize the mesh before processing (DEBUGGING)
    #[arg(long, hide = true)]
    no_normalize: bool,

    /// Output stats to this file
    #[arg(long, default_value = "")]
    stats: String,

    /// How much to separate each pixel by (DEBUGGING)
    #[arg(long, default_value_t = 0.995)]
    pixel_sep: F,

    /// How much to pull each vertex associated with an edge toward it.
    #[arg(long, default_value_t = 0.5)]
    edge_pull: F,

    /// How much to pull each vertex associated with a vertex toward it.
    #[arg(long, default_value_t = 0.5)]
    vertex_pull: F,

    /// Area below which faces can be deleted
    #[arg(long, default_value_t = 1e-14)]
    area_threshold: F,

    /// Distance below which colors are considered similar
    #[arg(long, default_value_t = 1e-4)]
    color_diff_threshold: F,

    /// Do not delete degenerate faces in the mesh (ABLATION)
    #[arg(long)]
    no_delete_degen: bool,
}

pub fn main() {
    let args = Args::parse();
    let mut scene = pars3d::load(&args.input).expect("Failed to parse input");
    let input_bd_edges = scene
        .meshes
        .iter()
        .map(|m| m.num_boundary_edges())
        .sum::<usize>();
    println!(
        "[INFO]: Input #F = {}, #V = {}, bd edges = {input_bd_edges}",
        scene.num_faces(),
        scene.num_vertices()
    );

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
        new_mesh.denormalize(s, t);
        out_scene.meshes[mi] = new_mesh;
    }
    let elapsed = start.elapsed();
    let output_bd_edges = out_scene
        .meshes
        .iter()
        .map(|m| m.num_boundary_edges())
        .sum::<usize>();
    println!("[INFO]: Resampling took {elapsed:?}");
    println!(
        "[INFO]: Output #F = {}, #V = {}, bd edges = {output_bd_edges}",
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
  "num_boundary_edges": {output_bd_edges},
  "input_num_boundary_edges": {input_bd_edges}
}}"#,
            out_scene.num_faces(),
            out_scene.num_vertices(),
            scene.num_faces(),
            scene.num_vertices(),
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
    println!(
        "[INFO]: Diffuse W = {}, H = {}",
        diff_img.width(),
        diff_img.height()
    );

    let mut edge_map = BTreeMap::new();
    let mut corner_map = BTreeMap::new();
    let mut labels = BTreeMap::new();
    let mut face_labels = vec![];
    //let mut to_del = vec![];

    use indicatif::ProgressIterator;
    for (fi, f) in mesh.f.iter().enumerate().progress() {
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
                &mut out,
                &mut corner_map,
                &mut edge_map,
                &mut labels,
                &mut face_labels,
                args,
            ),
        };
        assert_eq!(out.f.len(), face_labels.len());
        // For now, do not store normals since they are not well supported
        out.n.clear();
        // Add a simplification step here, as some faces are relatively similar, and we want to
        // delete degenerate faces.
        if !ok {
            pars3d::save("curr_error.ply", &out.into_scene()).expect("Failed to save error scene");
            eprintln!("Exiting after saved erroneous mesh");
            std::process::exit(1);
        }
    }

    if args.no_gap_fill {
        return out;
    }

    // map from (new vertex -> adjacent vertices that share the same corner)
    let mut edge_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();

    // Zip edges together
    for ([e0_key, e1_key], face_verts) in edge_map {
        macro_rules! add_key {
            ($dst: expr, $key: expr, $face: expr, $l: expr) => {{
                let fv = &corner_map[$key];
                let corner = fv.iter().find(|fv| fv.0 == $face).unwrap().1;
                assert_eq!(fv.iter().filter(|fv| fv.0 == $face).count(), 1);
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
        if face_verts.len() != 2 {
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
        let f0 = *f0;
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
        let f1 = *f1;
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

            let (new_face, label) = if (t0_delta - t1_delta).abs() < 2e-2 {
                let new_face = FaceKind::Quad([v0_front.0, p0n, p1n, v1_front.0]);
                let label = FaceLabel::GapFill([v0_front.0, p0n], [v1_front.0, p1n]);
                v0_front = (p0n, t0n);
                v1_front = (p1n, t1n);
                assert!(v0s.next().is_some());
                assert!(v1s.next().is_some());
                (new_face, label)
            } else if t0_delta >= t1_delta {
                let new_face = FaceKind::Tri([v0_front.0, p0n, v1_front.0]);
                let label = FaceLabel::GapFill([v0_front.0, p0n], [v1_front.0, usize::MAX]);
                v0_front = (p0n, t0n);
                assert!(v0s.next().is_some());
                (new_face, label)
            } else {
                let new_face = FaceKind::Tri([v0_front.0, p1n, v1_front.0]);
                let label = FaceLabel::GapFill([v0_front.0, usize::MAX], [v1_front.0, p1n]);
                v1_front = (p1n, t1n);
                assert!(v1s.next().is_some());
                (new_face, label)
            };
            face_labels.push(label);
            out.f.push(new_face);
        }

        for (p0n, t) in v0s {
            face_labels.push(FaceLabel::GapFill(
                [v0_front.0, p0n],
                [v1_front.0, usize::MAX],
            ));
            out.f.push(FaceKind::Tri([v0_front.0, p0n, v1_front.0]));
            v0_front = (p0n, t);
        }
        for (p1n, t) in v1s {
            face_labels.push(FaceLabel::GapFill(
                [v0_front.0, usize::MAX],
                [v1_front.0, p1n],
            ));
            out.f.push(FaceKind::Tri([v0_front.0, p1n, v1_front.0]));
            v1_front = (p1n, t);
        }
    }

    // Zip corner faces together
    for (_, fvs) in corner_map.iter() {
        assert!(fvs.iter().all(|fv| fv.1 < out.v.len()));

        if fvs.iter().any(|fv| !edge_adj.contains_key(&fv.1)) {
            println!("aborting1");
            face_labels.push(FaceLabel::GapCorner);
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
                    println!("aborting2");
                    out.f
                        .push(FaceKind::Quad(std::array::from_fn(|i| fvs[i].1)));
                    continue;
                };
                quad[1] = next;
                assert_ne!(quad[0], usize::MAX);
                for i in 2..4 {
                    let Some(n) = edge_adj.get(&quad[i - 1]) else {
                        println!("aborting3");
                        quad = std::array::from_fn(|i| fvs[i].1);
                        break;
                    };
                    let Some(&next) = n.iter().find(|&&v| v != quad[i - 2] && v != usize::MAX)
                    else {
                        println!("aborting4");
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
                        poly.clear();
                        poly.extend(fvs.iter().map(|fv| fv.1));
                        break;
                        // not sure why this happens, but ok to ignore for now?
                    };
                    let Some(&next) = n.iter().find(|&&v| v != poly[i - 2] && v != usize::MAX)
                    else {
                        poly.clear();
                        poly.extend(fvs.iter().map(|fv| fv.1));
                        break;
                    };
                    poly.push(next);
                }
                FaceKind::Poly(poly)
            }
        };
        face_labels.push(FaceLabel::GapCorner);
        out.f.push(face);
    }

    assert_eq!(face_labels.len(), out.f.len());

    // compute statistics for degenerate mesh
    let mut degen_degen = 0;
    let mut degen_pixel = 0;
    let mut degen_bridge = 0;
    let mut degen_bridge_corner = 0;
    let mut degen_fill = 0;

    for (fi, f) in out.f.iter().enumerate() {
        let area = f.area(&out.v);
        let counter = match face_labels[fi] {
            FaceLabel::Degen => &mut degen_degen,
            FaceLabel::Pixel => &mut degen_pixel,
            FaceLabel::Bridge(_, _) => &mut degen_bridge,
            FaceLabel::BridgeCorner => &mut degen_bridge_corner,
            FaceLabel::GapFill(_, _) | FaceLabel::GapCorner => &mut degen_fill,
        };
        if area < args.area_threshold {
            *counter += 1;
        }
    }
    macro_rules! count_face_label {
        ($p: pat) => {{ face_labels.iter().filter(|c| matches!(c, $p)).count() }};
    }
    println!(
        r#"
      Degen Degen   : {degen_degen} / {}
      Degen Pixel   : {degen_pixel} / {}
      Degen Bridge  : {degen_bridge} / {}
      Degen BridgeC : {degen_bridge_corner} / {}
      Degen Fill    : {degen_fill} / {}
    "#,
        count_face_label!(FaceLabel::Degen),
        count_face_label!(FaceLabel::Pixel),
        count_face_label!(FaceLabel::Bridge(_, _)),
        count_face_label!(FaceLabel::BridgeCorner),
        count_face_label!(FaceLabel::GapFill(_, _) | FaceLabel::GapCorner),
    );

    let init_f = out.f.len();
    let init_v = out.v.len();
    println!("[INFO]: Initial generated mesh has {init_f} faces & {init_v} vertices");

    if !args.no_delete_degen {
        println!("[INFO]: Starting degenerate face deletion");
        delete_degenerate_faces(&mut out, args, &face_labels);
        out.delete_unused_vertices();
        println!(
            "[INFO]: Cleaned mesh has {} faces (-{}) & {} vertices (-{})",
            out.f.len(),
            init_f - out.f.len(),
            out.v.len(),
            init_v - out.v.len(),
        );
    }

    out
}

pub fn sample_exact(
    mesh: &pars3d::Mesh,
    f: &FaceKind,
    fi: usize,
    diff_img: &DynamicImage,

    // destination for all attributes
    out: &mut pars3d::Mesh,

    // map from edge -> (original face idx, vertices), which stores vertices along every half edge
    corner_map: &mut BTreeMap<[U; 3], Vec<(usize, usize)>>,
    // map from edge -> (original face idx, vertices on face)
    edge_map: &mut BTreeMap<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
    // map from new vertex index to original vertex position and distance
    labels: &mut BTreeMap<usize, [(u32, [U; 3]); 2]>,

    // label for what kind of face is being inserted into the mesh, useful for cleaning up faces
    // later
    face_labels: &mut Vec<FaceLabel>,

    args: &Args,
) -> bool {
    const CHAN: usize = 0;
    let mut aabb = AABB::<F, 2>::new();
    let f_slice = f.as_slice();

    let get_rgb = |u: F, v: F| {
        let u = u % 1.;
        let u = if u < 0. { 1. + u } else { u };
        let v = v % 1.;
        let v = if v < 0. { 1. + v } else { v };
        let rgba = image::imageops::sample_bilinear(diff_img, u as f32, v as f32).unwrap();
        let [r, g, b, _a] = rgba.0.map(|c| c as F / 255.);
        [r, g, b]
    };

    for uv in f_slice.iter().map(|&vi| mesh.uv[CHAN][vi]) {
        aabb.add_point(uv);
    }

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
    /*
    assert!(
        uv_f.area() > 1e-8,
        "TODO handle near 0 area uv face separately {}",
        uv_f.area()
    );
    */

    // for each pixel, what vertices are associated with it?
    let mut pixel_map: BTreeMap<_, [usize; 4]> = BTreeMap::new();
    let mut pix_fi: BTreeMap<_, usize> = BTreeMap::new();

    let iarea = iaabb.area();
    assert_ne!(iarea, 0);

    if iarea == 1 {
        todo!();
    }

    let start = out.v.len();
    let start_f = out.f.len();
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

        let barys = cfs.map(|cf| uv_f.barycentric(cf));

        // seems to work fine but a bit sus about it because it took so long to figure out if it
        // worked.
        if (0..3).any(|i| barys.iter().all(|bary| bary[i] < -1e-4)) {
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

            let next = raw_pos[(i + 1) % 4];
            let prev = raw_pos[(i + 3) % 4];
            let tri = v_f.tri().unwrap();

            let mut new_pos = nearest_point_on_tri(tri, pos);

            // stupid way to project to the nearest point on the line of the tri (but seems to
            // work fine...)
            for _ in 0..4 {
                let a = nearest_pt_on_line(new_pos, [next, pos]);
                let b = nearest_pt_on_line(new_pos, [prev, pos]);
                new_pos = if dist(a, new_pos) < dist(b, new_pos) {
                    a
                } else {
                    b
                };
                new_pos = nearest_point_on_tri(tri, new_pos);
            }
            let new_normal = n_f
                .as_ref()
                .map(|n_f| n_f.from_barycentric(v_f.barycentric(new_pos)));
            debug_assert!(
                v_f.barycentric(new_pos)
                    .iter()
                    .all(|v| (0.0..=1.0).contains(v))
            );
            (new_pos, new_normal)
        });

        // TODO check if this is alright?
        /*
        if pars3d::quad_area(new_verts.map(|vn| vn.0)) == 0. {
          continue;
        }
        */

        // commit to this new pixel
        let new_verts = new_verts.map(|(new_vert, normal)| {
            let vi = out.v.len();
            out.v.push(new_vert);
            out.vert_colors.push(rgb);
            if let Some(normal) = normal {
                out.n.push(normal);
            }

            vi
        });

        let prev = pixel_map.insert(c, new_verts);
        assert_eq!(prev, None);
        pix_fi.insert(c, out.f.len());
        face_labels.push(FaceLabel::Pixel);
        out.f.push(FaceKind::Quad(new_verts));
    }

    // hohoho what a funny name
    macro_rules! triagram {
        () => {{
            for h in iaabb.height_range() {
                for w in iaabb.width_range() {
                    let c = if pixel_map.contains_key(&[w, h]) {
                        "x"
                    } else {
                        "-"
                    };
                    print!("{c} ");
                }
                println!();
            }
        }};
    }

    macro_rules! save_bad_mesh {
        ($label: expr) => {{
            let mut tmp = pars3d::Mesh::new_geometry(
                f_slice.iter().map(|vi| mesh.v[*vi]).collect(),
                vec![FaceKind::Tri([0, 1, 2])],
            );

            println!("--- Bad Triangle Diagram ---");
            triagram!();

            tmp.uv[CHAN] = f_slice.iter().map(|vi| mesh.uv[CHAN][*vi]).collect();
            pars3d::save("failing_tri.obj", &tmp.into_scene()).expect("Failed to save temp error");

            let error_verts = out.v[start..].to_vec();
            let error_colors = out.vert_colors[start..].to_vec();
            let mut error_faces = out.f[start_f..].to_vec();
            for f in &mut error_faces {
                f.offset(-(start as i32))
            }
            out.v = error_verts;
            out.vert_colors = error_colors;
            out.f = error_faces;
            eprintln!($label);

            return false;
        }};
    }
    // sanity check
    for [u, v] in iaabb.iter_coords() {
        if !pixel_map.contains_key(&[u, v]) {
            continue;
        }
        let nbrs = [[u - 1, v], [u + 1, v], [u, v - 1], [u, v + 1]];
        let has_nbr = nbrs.iter().any(|nbr| pixel_map.contains_key(nbr));
        if !has_nbr {
            save_bad_mesh!("Each pixel in a tri should have at least one nbr");
        }
    }

    // --- Adding faces in between pixel quads
    for (&[u, v], &[l, r, ur, ul]) in pixel_map.iter() {
        if args.no_fill {
            break;
        }
        let own_fi = pix_fi[&[u, v]];
        // check bottom right first
        if !pixel_map.contains_key(&[u + 1, v + 1])
            && let Some(&[_, _, _, a]) = pixel_map.get(&[u + 1, v])
            && let Some(&[_, b, _, _]) = pixel_map.get(&[u, v + 1])
        {
            face_labels.push(FaceLabel::BridgeCorner);
            out.f.push(FaceKind::Tri([ur, a, b]));
        }
        // TODO turn these into checked subs?
        let left = pixel_map.get(&[u - 1, v]);
        let up = pixel_map.get(&[u, v - 1]);
        if let Some(&[_, or, our, _]) = left {
            face_labels.push(FaceLabel::Bridge(own_fi, pix_fi[&[u - 1, v]]));
            out.f.push(FaceKind::Quad([ul, our, or, l]));
        }
        if let Some(&[_, _, our, oul]) = up {
            face_labels.push(FaceLabel::Bridge(own_fi, pix_fi[&[u, v - 1]]));
            out.f.push(FaceKind::Quad([oul, our, r, l]));
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
                todo!();
                //continue;
            }

            /* Definitely no faces to add */
            (None, None, None) => continue,

            // Handled earlier, no special cases to add
            (None, Some(_), None) | (None, None, Some(_)) => continue,
        };
        face_labels.push(FaceLabel::BridgeCorner);
        out.f.push(corner_face);
    }

    let mut edge_face_adj: BTreeMap<[usize; 2], EdgeKind> = BTreeMap::new();
    // compute adjacent boundary edges and trace from the corners
    for (fi, f) in out.f.iter().enumerate().skip(start_f) {
        for [e0, e1] in f.edges() {
            assert_ne!(e0, e1);
            edge_face_adj
                .entry(minmax(e0, e1))
                .and_modify(|v| {
                    let did_ins = v.insert(fi);
                    assert!(did_ins);
                })
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

        let mut ins = |a: usize, b: usize| {
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

    // This is only an array since there are exactly 3 vertices.
    let mut corner_verts: [(usize, usize); 3] = [(usize::MAX, usize::MAX /* og, new) */); 3];
    for (ci, &og_vi) in f_slice.iter().enumerate() {
        let vert_pos = mesh.v[og_vi];

        let [og_u, og_v] = mesh.uv[CHAN][og_vi];
        let u = (og_u.fract().abs() * w as F - 0.5).floor() as i32;
        let v = (og_v.fract().abs() * h as F - 0.5).floor() as i32;
        assert!(u >= 0, "{og_u} {u} {v}");
        assert!(v >= 0, "{og_u} {u} {v}");

        let mut nearest = 0;
        let mut best_dist = F::INFINITY;
        // TODO do this check more efficiently
        let range = [0, -1, 1, -2, 2, -3, 3];
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
                    let d = dist(out.v[vi], vert_pos);
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
                let d = dist(out.v[new_vi], vert_pos);
                if d < best_dist {
                    nearest = new_vi;
                    best_dist = d;
                }
            }
        }
        if !best_dist.is_finite() {
            save_bad_mesh!("No nearby points");
        }
        assert_ne!(best_dist, F::INFINITY);

        let fv = corner_map.entry(vert_pos.map(F::to_bits)).or_default();
        debug_assert!(!fv.iter().any(|fv| fv.0 == fi));
        // check that each corner vert corresponds to a single original vertex
        let check = corner_verts.iter().any(|&(_, new_vi)| new_vi == nearest);
        assert!(!check);

        fv.push((fi, nearest));
        corner_verts[ci] = (og_vi, nearest);

        // Pull to a corner to make it tight (larger T is tighter)
        let t = args.vertex_pull;
        out.v[nearest] = add(kmul(t, mesh.v[og_vi]), kmul(1. - t, out.v[nearest]));

        if args.debug_colors {
            out.vert_colors[nearest] = [1.; 3];
        }
    }

    if args.no_gap_fill {
        return true;
    }

    const INVALID_POS: [U; 3] = [U::MAX; 3];
    for &(og_vi, new_vi) in &corner_verts {
        assert!(vert_adj.contains_key(&new_vi), "{new_vi} {corner_verts:?}");
        let [l, r] = vert_adj[&new_vi];
        assert_ne!(l, r);
        assert_ne!(l, new_vi);
        assert_ne!(r, new_vi);
        let mut iter = |mut curr: usize, mut prev: usize| {
            let mut c = 1;
            while !corner_verts.iter().any(|v| v.1 == curr) {
                let label = labels.entry(curr).or_insert([(0, INVALID_POS); 2]);
                *label.iter_mut().find(|v| v.1 == INVALID_POS).unwrap() =
                    (c, mesh.v[og_vi].map(F::to_bits));
                assert!(vert_adj[&curr].iter().any(|&v| v == prev));
                let next = *vert_adj[&curr].iter().find(|v| **v != prev).unwrap();
                prev = curr;
                curr = next;
                c += 1;
            }
            curr
        };

        // add to the edge map here, to ensure that even if the edge doesn't have any
        // intermediate vertices it will be labeled.
        let end_pt = iter(l, new_vi);
        let next_og_vi = corner_verts.iter().find(|v| v.1 == end_pt).unwrap().0;
        let [e0_key, e1_key] = [og_vi, next_og_vi].map(|vi| out.v[vi].map(F::to_bits));
        let fvs = edge_map.entry(minmax(e0_key, e1_key)).or_default();
        if fvs.iter_mut().find(|fv| fv.0 == fi).is_none() {
            fvs.push((fi, vec![]));
        }
        iter(r, new_vi);
    }
    let check = labels
        .range(start..)
        .all(|(_, v)| v[0].1 != INVALID_POS && v[1].1 != INVALID_POS);
    assert!(check);

    for (&new_vi, ogs) in labels.range(start..) {
        if args.debug_colors && out.vert_colors[new_vi] != [1.; 3] {
            out.vert_colors[new_vi] = [0.; 3];
        }
        let [og0_key, og1_key] = ogs.map(|vi| vi.1);
        let fvs = edge_map.entry(minmax(og0_key, og1_key)).or_default();
        if let Some(fv) = fvs.iter_mut().find(|fv| fv.0 == fi) {
            fv.1.push(new_vi);
        } else {
            fvs.push((fi, vec![new_vi]));
        }

        let ogs = [og0_key, og1_key].map(|k| k.map(F::from_bits));
        let t = nearest_on_line(out.v[new_vi], ogs);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        let tgt_pos = add(ogs[0], kmul(t, sub(ogs[1], ogs[0])));
        // pull edge verts to the edge (larger is closer to edge)
        let t = args.edge_pull;
        out.v[new_vi] = add(kmul(1. - t, out.v[new_vi]), kmul(t, tgt_pos));
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
    aabb.expand_by(1e-3);
    let iaabb = aabb.round_to_i32();

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

/// Computes the nearest point on the line
pub fn nearest_pt_on_line(p: [F; 3], [s, e]: [[F; 3]; 2]) -> [F; 3] {
    let t = nearest_on_line(p, [s, e]).clamp(0., 1.);
    add(s, kmul(t, sub(e, s)))
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
        .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
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

/// Deletes degenerate faces by merging all vertices of the face together
pub fn delete_degenerate_faces(
    mesh: &mut pars3d::Mesh,
    args: &Args,
    labels: &[FaceLabel],
) -> usize {
    let mut deleted = 0;
    let mut remap = UnionFind::new(mesh.v.len());

    let mut fixed = HashSet::new();
    for f in &mesh.f {
        for [e0, e1] in f.edges() {
            let [vc0, vc1] = [e0, e1].map(|vi| mesh.vert_colors[vi]);
            let lock = dist(vc0, vc1) > args.color_diff_threshold;
            if lock {
                fixed.insert(e0);
                fixed.insert(e1);
            }

            // TODO here also check planarity
        }
    }

    // First, delete degenerate fill faces, they can be taken as the average of all of their
    // components.

    // for each vertex need to compute whether it can be deleted,
    // based on whether the 1 ring of the vertex all have similar values.
    /*
    for (fi, f) in mesh.f.iter_mut().enumerate() {
        if f.is_empty() {
            continue;
        }
        if !matches!(labels[fi], FaceLabel::GapFill(_, _) | FaceLabel::GapCorner) {
            continue;
        }

        let f_s = f.as_slice();
        if f_s.iter().any(|vi| fixed.contains(vi)) {
            continue;
        }
        let area = f.area(&mesh.v);
        assert!(area >= 0.);
        assert!(area.is_finite());
        if area > args.area_threshold {
            continue;
        }
        let n = f_s.len().max(1) as F;
        let avg_color = f_s
            .iter()
            .map(|&vi| mesh.vert_colors[remap.get_compress(vi)])
            .fold([0.; 3], add)
            .map(|v| v / n);

        let new_v = f_s
            .iter()
            .map(|&vi| mesh.v[remap.get_compress(vi)])
            .fold([0.; 3], add)
            .map(|v| v / n);

        let new_vi = f_s[0];
        mesh.v[new_vi] = new_v;
        mesh.vert_colors[new_vi] = avg_color;

        for &vi in f_s {
            remap.set(vi, new_vi);
        }

        *f = FaceKind::empty();
        deleted += 1;
    }
    */

    // --- Bridge Face Deletion
    for fi in 0..mesh.f.len() {
        let f = &mesh.f[fi];
        if f.len() != 4 {
            continue;
        }
        let FaceLabel::Bridge(fi0, fi1) = labels[fi] else {
            continue;
        };
        assert_matches!(labels[fi0], FaceLabel::Pixel);
        assert_matches!(labels[fi1], FaceLabel::Pixel);

        let [f, f0, f1] = mesh.f.get_disjoint_mut([fi, fi0, fi1]).unwrap();
        let [e00, e01] = f.shared_edge(f0).unwrap().map(|v| remap.get_compress(v));
        let [e10, e11] = f.shared_edge(f1).unwrap().map(|v| remap.get_compress(v));

        // e00 - e11 is a paired edge
        // e01 - e10 is a paired edge

        if dist(mesh.vert_colors[e00], mesh.vert_colors[e11]) > args.color_diff_threshold {
            continue;
        }
        if dist(mesh.vert_colors[e01], mesh.vert_colors[e10]) > args.color_diff_threshold {
            continue;
        }

        // Commit to collapsing this face
        let mut combine = |a: usize, b: usize| {
            let a = remap.get_compress(a);
            let b = remap.get_compress(b);
            let new_v = kmul(0.5, add(mesh.v[a], mesh.v[b]));
            let new_vc = kmul(0.5, add(mesh.vert_colors[a], mesh.vert_colors[b]));
            let new_vi = a;

            mesh.v[a] = new_v;
            mesh.vert_colors[a] = new_vc;
            remap.set(a, new_vi);
            remap.set(b, new_vi);
        };
        combine(e00, e11);
        combine(e10, e01);

        *f = FaceKind::empty();
        deleted += 1;
    }

    mesh.f.retain_mut(|f| {
        if f.is_empty() {
            return false;
        }

        f.remap(|vi| remap.get_compress(vi));
        !f.canonicalize()
    });

    deleted
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaceLabel {
    /// A triangle representing a face which is too small for any pixels
    Degen,
    /// New pixel from a given face
    Pixel,
    /// Bridge between two pixels within a given face
    Bridge(usize, usize),
    /// Fills in gaps between bridges
    BridgeCorner,
    /// A face intended to cover the boundary between two faces
    GapFill([usize; 2], [usize; 2]),
    /// A face which covers the corner between many triangles
    GapCorner,
}

fn dist<const N: usize>(a: [F; N], b: [F; N]) -> F {
    length(sub(a, b))
}
