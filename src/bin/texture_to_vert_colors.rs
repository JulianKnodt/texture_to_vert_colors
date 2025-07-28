#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]
#![feature(duration_millis_float)]

use std::cmp::minmax;
use std::collections::BTreeMap;
use std::io::Write;
use std::ops::Range;
use std::time::Duration;

type Map<K, V> = BTreeMap<K, V>;
const CHAN: usize = 0;

use clap::Parser;
use pars3d::image::{self, DynamicImage, GenericImageView};
use pars3d::{FaceKind, Mesh, edge::EdgeKind};
use union_find::UnionFind;

use texture_to_vert_colors::aabb::AABB;
use texture_to_vert_colors::qem::{Args as QEMArgs, QEMBuffers, simplify_range_colored};
use texture_to_vert_colors::{
    F, U, add, cross, dist, dist_sq, dot, kmul, len_sq, length, normalize, orthogonal, sub,
};

/// A utility for converting a mesh with texture into a mesh with vertex colors without
/// changing visual quality.
#[derive(Debug, Clone, PartialEq, Parser)]
#[clap(group(
  clap::ArgGroup::new("target")
    .required(true)
    .args(&["target_tri_ratio", "target_tri_num"])
))]
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
    #[arg(long, default_value_t = 0.999)]
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
    #[arg(long, default_value_t = 5e-2)]
    color_diff_threshold: F,

    /// Do not delete degenerate faces in the mesh (ABLATION)
    #[arg(long)]
    delete_degen: bool,

    /// Do not delete faces in the mesh incrementally, only perform a single deletion at the
    /// end (ABLATION).
    #[arg(long)]
    no_incremental_delete: bool,

    /// Perform QEM incrementally, on top of a single QEM at the end (ABLATION)
    #[arg(long)]
    incremental_qem: bool,

    /// Do not perform QEM at the end (ABLATION)
    #[arg(long)]
    no_final_qem: bool,

    /// Target tri ratio
    #[arg(long, short = 'r', default_value_t = 0.0, group = "target")]
    target_tri_ratio: F,

    /// Target number of final tris
    #[arg(long, short = 't', default_value_t = 0, group = "target")]
    target_tri_num: usize,

    /// Where to store correspondence between input and output
    #[arg(long, default_value_t = String::new())]
    correspondence_json: String,

    /// When deciding whether to fill gaps with quads or tris, the permitted distance along
    /// edges when choosing to use quads.
    #[arg(long, default_value_t = 0.)]
    gap_fill_dist: F,

    /// Resize this image to a fraction of its input.
    #[arg(long, short, default_value_t = 1.)]
    image_size_frac: F,

    /// How many pixels to resize the image to. 0 indicates no resize.
    #[arg(long, default_value_t = 0)]
    image_size_px: u32,

    /// Triangulate the output mesh
    #[arg(long)]
    triangulate: bool,

    /// Triangulate the input mesh
    #[arg(long)]
    triangulate_input: bool,

    /// Do not use direct remeshing even with constant colors.
    #[arg(long)]
    no_adaptive: bool,
}

pub fn main() {
    let args = Args::parse();
    if !(0.0..=1.0).contains(&args.target_tri_ratio) {
        eprintln!(
            "Target tri ratio is outside of valid range, expected in [0,1], got {}",
            args.target_tri_ratio
        );
        return;
    }
    let mut scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input {}", &args.input));
    let mut input_bd_edges = 0;
    let mut input_non_manifold_edges = 0;
    for m in &scene.meshes {
        let (_, bd, nm) = m.num_edge_kinds_by_position();
        input_bd_edges += bd;
        input_non_manifold_edges += nm;
    }
    println!(
        "[INFO]: Input ({}) #F = {}, #Δ = {}, #V = {}, #BdE = {}, #NmE = {}",
        args.input,
        scene.num_faces(),
        scene.num_tris(),
        scene.num_vertices(),
        input_bd_edges,
        input_non_manifold_edges,
    );

    // Copy vertex colors for each input UV
    // TODO remove this assumption of a single material and copy per face with duplication
    let mut img_res = vec![];
    let diff_img = if !args.diffuse_img.is_empty() {
        let img = pars3d::image::open(&args.diffuse_img)
            .expect(&format!(
                "Failed to open diffuse image from {}",
                &args.diffuse_img
            ))
            .flipv();

        let (w, h) = img.dimensions();
        println!("[INFO]: Diffuse W = {w}, H = {h}");
        let img = if args.image_size_frac != 1. || args.image_size_px != 0 {
            let [nw, nh] = if args.image_size_px != 0 {
                [args.image_size_px.min(w), args.image_size_px.min(h)]
            } else {
                let nw = (w as F * args.image_size_frac).ceil() as u32;
                let nh = (h as F * args.image_size_frac).ceil() as u32;
                [nw, nh]
            };
            let img = img.resize(nw, nh, image::imageops::FilterType::Triangle);
            let (nw, nh) = img.dimensions();
            println!(
                "[INFO]: Resized image ({}) from {w}, {h} to {nw}, {nh}",
                args.diffuse_img
            );
            img
        } else {
            img
        };
        img_res.push([img.width(), img.height()]);
        Some(img)
    } else {
        for texture in scene.textures.iter_mut() {
            let Some(txt) = texture.image.as_mut() else {
                continue;
            };
            *txt = txt.flipv();
            if args.image_size_frac != 1. || args.image_size_px != 0 {
                let (w, h) = txt.dimensions();
                let [nw, nh] = if args.image_size_px != 0 {
                    [args.image_size_px.min(w), args.image_size_px.min(h)]
                } else {
                    let nw = (w as F * args.image_size_frac).ceil() as u32;
                    let nh = (h as F * args.image_size_frac).ceil() as u32;
                    [nw, nh]
                };
                *txt = txt.resize(nw, nh, image::imageops::FilterType::Triangle);
                let (nw, nh) = txt.dimensions();
                println!(
                    "[INFO]: Resized image ({}) from {w}, {h} to {nw}, {nh}",
                    texture.original_path
                );
            }
            img_res.push([txt.width(), txt.height()]);
        }
        None
    };

    let mut out_scene = scene.clone();
    let start = std::time::Instant::now();
    let mut remesh_times = vec![];
    let mut simplification_times = vec![];
    let mut before_simplify_faces = 0;
    for (mi, mesh) in scene.meshes.iter_mut().enumerate() {
        let (s, t) = if args.no_normalize {
            (1., [0.; 3])
        } else {
            mesh.normalize()
        };

        if args.triangulate_input {
            mesh.triangulate();
        } else {
            let num_split = mesh.split_non_planar_faces(3e-3);
            if num_split > 0 {
                println!("[WARN]: Split {num_split} non-planar polygonal faces");
            }
            let num_split = mesh.split_self_intersecting_uv_poly(CHAN);
            if num_split > 0 {
                println!("[WARN]: Split {num_split} self-intersecting UV polygons");
            }
        }
        assert!(!mesh.uv[0].is_empty());
        assert_eq!(mesh.uv[0].len(), mesh.v.len());
        let Outputs {
            mesh: mut new_mesh,
            remesh_time,
            simplification_time,
            before_simplify_num_faces,
        } = texture_to_vert_colors(
            mesh,
            |fi| {
                if let Some(diff_img) = diff_img.as_ref() {
                    return diff_img;
                }
                let mi = mesh.mat_for_face(fi).expect(&format!(
                    "No material found in {:?} for {fi}",
                    mesh.face_mat_idx
                ));
                let &ti = scene.materials[mi]
                    .textures
                    .iter()
                    .find(|&&ti| scene.textures[ti].kind == pars3d::mesh::TextureKind::Diffuse)
                    .expect("No diffuse textures for this material");
                scene.textures[ti].image.as_ref().unwrap()
            },
            &args,
        );
        new_mesh.denormalize(s, t);
        new_mesh.n.clear();
        out_scene.meshes[mi] = new_mesh;
        remesh_times.push(remesh_time.as_millis_f64());
        simplification_times.push(simplification_time.as_millis_f64());
        before_simplify_faces += before_simplify_num_faces;
    }
    let elapsed = start.elapsed();
    let mut output_bd_edges = 0;
    let mut output_non_manifold_edges = 0;
    for m in &out_scene.meshes {
        let (_, bd, nm) = m.num_edge_kinds();
        output_bd_edges += bd;
        output_non_manifold_edges += nm;
    }
    println!("[INFO]: Resampling took {elapsed:?}");
    println!(
        "[INFO]: Output #F = {}, #Δ = {}, #V = {}, #BdE = {}, #NmE = {}",
        out_scene.num_faces(),
        out_scene.num_tris(),
        out_scene.num_vertices(),
        output_bd_edges,
        output_non_manifold_edges,
    );
    out_scene.materials.clear();
    out_scene.textures.clear();

    pars3d::save(&args.output, &out_scene).expect("Failed to save output");
    println!("[INFO]: Saved to {}", args.output);

    if !args.stats.is_empty() {
        let mut stat_file = std::fs::File::create(&args.stats).expect("Failed to open stats file");
        writeln!(
            stat_file,
            r#"{{
  "num_faces": {out_faces},
  "num_tris": {out_tris},
  "num_vertices": {out_vertices},
  "before_simplify_tris": {before_simplify_faces},
  "input_num_faces": {},
  "input_num_tris": {},
  "input_num_vertices": {},
  "remesh_times_ms": {remesh_times:?},
  "simplification_times_ms": {simplification_times:?},
  "total_time_ms": {total_time_ms},
  "image_resolutions": {img_res:?},
  "num_boundary_edges": {output_bd_edges},
  "input_num_boundary_edges": {input_bd_edges},
  "num_non_manifold_edges": {output_non_manifold_edges},
  "input_non_manifold_edges": {input_non_manifold_edges}
}}"#,
            scene.num_faces(),
            scene.num_tris(),
            scene.num_vertices(),
            total_time_ms = elapsed.as_millis_f64(),
            out_faces = out_scene.num_faces(),
            out_tris = out_scene.num_tris(),
            out_vertices = out_scene.num_vertices(),
        )
        .expect("Failed to write stats");
    }
}

/// Textures to be used when upscaling the output
#[derive(Clone, Debug)]
pub struct SourceTextures<'a> {
    diff_img: &'a DynamicImage,
}

impl<'a> SourceTextures<'a> {
    pub fn push_to_mesh(&self, m: &mut Mesh, u: F, v: F) {
        let u = u % 1.;
        let u = if u < 0. { 1. + u } else { u };
        let v = v % 1.;
        let v = if v < 0. { 1. + v } else { v };
        let Some(rgba) = image::imageops::sample_bilinear(self.diff_img, u as f32, v as f32) else {
            panic!("{u} {v}");
        };
        let [r, g, b, _a] = rgba.0.map(|c| c as F / 255.);
        m.vert_colors.push([r, g, b]);
        /*
        if let Some(hm) = self.height_map.as_ref() {
            let Some(h) = image::imageops::sample_bilinear(hm, u as f32, v as f32) else {
                panic!("{u} {v}");
            };
            let h = h.0[0] as F / 255.;
            m.vertex_attrs.height.push(h);
        }
        */
    }
    pub fn get_value(&self, u: F, v: F) -> [F; 3] {
        let u = u % 1.;
        let u = if u < 0. { 1. + u } else { u };
        let v = v % 1.;
        let v = if v < 0. { 1. + v } else { v };
        let Some(rgba) = image::imageops::sample_bilinear(self.diff_img, u as f32, v as f32) else {
            panic!("{u} {v}");
        };
        let [r, g, b, _a] = rgba.0.map(|c| c as F / 255.);
        [r, g, b]
    }
}

pub struct Outputs {
    mesh: Mesh,
    remesh_time: Duration,
    simplification_time: Duration,
    before_simplify_num_faces: usize,
}

pub fn texture_to_vert_colors<'a>(
    mesh: &Mesh,
    f_mat: impl Fn(usize) -> &'a DynamicImage,
    args: &Args,
) -> Outputs {
    let mut out = Mesh::default();
    // always at least as many faces as the input
    out.f.reserve(mesh.f.len());
    out.v.reserve(3 * mesh.f.len());
    out.vert_colors.reserve(3 * mesh.f.len());

    let start = std::time::Instant::now();

    let surface_area = mesh.f.iter().map(|f| f.area(&mesh.v)).sum::<F>();

    let mut edge_map = Map::new();
    let mut corner_map = Map::new();
    // This is required to be a BTreeMap for the range function.
    let mut labels = BTreeMap::new();
    let mut face_labels = vec![];
    let mut remap = UnionFind::new_u32(0);

    use indicatif::ProgressIterator;

    macro_rules! canonicalize {
        ($range: expr) => {{
            for f in &mut out.f[$range] {
                f.remap(|vi| remap.get_compress(vi));
                if f.canonicalize() {
                    *f = FaceKind::empty();
                }
            }
        }};
        () => {
            canonicalize!(..)
        };
    }

    let target_tri_ratio = if args.no_final_qem ^ (!args.incremental_qem) {
        args.target_tri_ratio
    } else {
        args.target_tri_ratio.sqrt()
    };
    let mut qem_buf = QEMBuffers::default();
    let mut del_by_qem = 0;
    for (fi, f) in mesh.f.iter().enumerate().progress() {
        let diff_img = f_mat(fi);
        let src_txs = SourceTextures { diff_img };

        let curr_f = out.f.len();
        let curr_v = out.v.len();
        let ok = match args.sample_kind {
            SampleKind::Exact => sample_exact(
                mesh,
                f,
                fi,
                &src_txs,
                &mut out,
                &mut corner_map,
                &mut edge_map,
                &mut labels,
                &mut face_labels,
                args,
            ),
            SampleKind::Approx => sample_approx(
                mesh,
                f,
                fi,
                &src_txs,
                &mut out,
                &mut corner_map,
                &mut edge_map,
                &mut labels,
                &mut face_labels,
                args,
            ),
            SampleKind::Direct => sample_direct(
                mesh,
                f,
                fi,
                &src_txs,
                &mut out,
                &mut face_labels,
                &mut corner_map,
                &mut edge_map,
                //&mut labels,
            ),
        };
        // Add a simplification step here, as some faces are relatively similar, and we want to
        // delete degenerate faces.
        if !ok {
            pars3d::save("curr_error.ply", &out.into_scene()).expect("Failed to save error scene");
            eprintln!("Exiting after saved erroneous mesh");
            std::process::exit(1);
        }
        assert_eq!(out.f.len(), face_labels.len());

        let new_f = out.f.len();
        let new_v = out.v.len();
        remap.extend_by(new_v - curr_v);

        if !args.no_incremental_delete && args.sample_kind == SampleKind::Exact {
            del_degen_bridges(
                &mut remap,
                &mut out,
                |vi| labels.contains_key(&vi),
                args,
                &face_labels,
                curr_f..new_f,
            );
            canonicalize!(curr_f..new_f);
        }
        // Then perform edge reduction here of just edges which are internal to the this
        // triangle.
        if args.incremental_qem {
            let qem_args = if args.target_tri_num != 0 {
                let frac_area = mesh.f[fi].area(&mesh.v) / surface_area;
                let target_tri_num = (args.target_tri_num as F * frac_area).ceil() as usize;
                QEMArgs {
                    target_tri_num,
                    check_bd: false,
                    color_diff_threshold: args.color_diff_threshold,
                    ..QEMArgs::default()
                }
            } else {
                QEMArgs {
                    target_tri_ratio,
                    check_bd: false,
                    color_diff_threshold: args.color_diff_threshold,
                    ..QEMArgs::default()
                }
            };
            del_by_qem += simplify_range_colored(
                &mut out,
                &qem_args,
                |vi| labels.contains_key(&vi),
                curr_f..new_f,
                curr_v..new_v,
                &mut remap,
                &mut qem_buf,
            );

            canonicalize!(curr_f..new_f);
        }
    }

    if del_by_qem != 0 {
        println!("[INFO]: Incremental QEM deleted {del_by_qem} faces");
    }

    if args.no_gap_fill {
        // this is only for debugging so leave 0 values for most results
        return Outputs {
            mesh: out,
            remesh_time: std::time::Duration::from_secs(0),
            simplification_time: std::time::Duration::from_secs(0),
            before_simplify_num_faces: 0,
        };
    }

    // map from (new vertex -> adjacent vertices that share the same corner on these two faces)
    // TODO make this a small vec of size 2
    let mut edge_adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    // between two faces, which way is the edge going
    let mut corner_edge_dir: BTreeMap<[usize; 2], [[usize; 2]; 2]> = BTreeMap::new();

    // Zip edges together
    let mut v0s = vec![];
    let mut v1s = vec![];
    for ([e0_key, e1_key], face_verts) in edge_map {
        macro_rules! add_key {
            ($dst: expr, $key: expr, $face: expr, $l: expr) => {{
                assert!(corner_map.contains_key($key));
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
        let correct = mesh.f[f0]
            .edges()
            .any(|e| e.map(|vi| mesh.v[vi]) == [e0, e1]);

        assert!(
            swapped ^ correct,
            "TODO colinear polygon (more than 3 verts) input faces {swapped} {correct} \n\
            face = {:?} \n\
            face index = {f0} \n\
            verts in face = {face_verts:?} \n\
            edge 0 = {e0:?} \n\
            edge 1 = {e1:?} \n\
            area = {:?} \n\
            face positions = {:?}",
            mesh.f[f0],
            mesh.f[f0].area(&mesh.v),
            mesh.f[f0].map_kind(|vi| mesh.v[vi]),
        );
        let [e0_key, e1_key] = if !swapped {
            [e1_key, e0_key]
        } else {
            [e0_key, e1_key]
        };

        assert_ne!(
            e0_key, e1_key,
            "temporary check for degenerate edges {e0:?} {e1:?}"
        );

        let mut handle_pair = |(f0, verts0): &(usize, Vec<usize>),
                               (f1, verts1): &(usize, Vec<usize>)| {
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

            let f0 = *f0;
            v0s.clear();
            v0s.extend(verts0.iter().map(ordering));

            let v0_max = v0s.iter().map(|v0| v0.1).max_by(F::total_cmp).unwrap_or(1.);
            for (_, v0t) in v0s.iter_mut() {
                *v0t /= v0_max + 1.;
            }

            let v00 = add_key!(v0s, &e0_key, f0, 0.);
            let v01 = add_key!(v0s, &e1_key, f0, 1.);
            // insert these items so that later the order is known when adding corner faces

            v0s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            v0s.dedup_by_key(|v| v.0);

            let f1 = *f1;
            v1s.clear();
            v1s.extend(verts1.iter().map(ordering));
            let v1_max = v1s.iter().map(|v1| v1.1).max_by(F::total_cmp).unwrap_or(1.);
            for (_, v1t) in v1s.iter_mut() {
                *v1t /= v1_max + 1.;
            }

            let v10 = add_key!(v1s, &e0_key, f1, 0.);
            let v11 = add_key!(v1s, &e1_key, f1, 1.);

            v1s.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            v1s.dedup_by_key(|v| v.0);

            let mut ins = |src: usize, dst: usize| {
                assert_ne!(src, dst);
                assert_ne!(src, usize::MAX);
                assert_ne!(dst, usize::MAX);

                let v = edge_adj.entry(src).or_insert_with(Vec::new);
                if !v.contains(&dst) {
                    v.push(dst);
                }
            };
            corner_edge_dir.insert(std::cmp::minmax(f0, f1), [[v01, v11], [v10, v00]]);

            ins(v00, v10);
            ins(v10, v00);

            ins(v01, v11);
            ins(v11, v01);

            // iterate over both and add faces with the front
            let mut v0s = v0s.iter().copied().peekable();
            let mut v1s = v1s.iter().copied().peekable();

            let mut v0_front: (usize, F) = v0s.next().unwrap();
            let mut v1_front: (usize, F) = v1s.next().unwrap();

            while let [Some(&(p0n, t0n)), Some(&(p1n, t1n))] = [v0s.peek(), v1s.peek()] {
                debug_assert_ne!(v0_front.0, p0n);
                debug_assert_ne!(v1_front.0, p1n);
                debug_assert_ne!(p0n, p1n);
                debug_assert_ne!(v0_front.0, v1_front.0);

                debug_assert!(t0n >= v0_front.1);
                debug_assert!(t1n >= v1_front.1);
                let t0_delta = (v0_front.1 - t1n).abs();
                let t1_delta = (v1_front.1 - t0n).abs();

                let (new_face, label) = if (t0_delta - t1_delta).abs() <= args.gap_fill_dist {
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
        };
        match face_verts.as_slice() {
            [] => unreachable!(),
            [fv0] => {
                for key in [e0_key, e1_key] {
                    let fv = &corner_map[&key];
                    let corner = fv.iter().find(|fv| fv.0 == fv0.0).unwrap().1;
                    edge_adj.entry(corner).or_insert_with(Vec::new);
                }

                continue;
            }
            [fv0, fv1] => {
                handle_pair(fv0, fv1);
                continue;
            }
            // if there are 3 non-manifold edges, just need to do all pairs
            [fv0, fv1, fv2] => {
                eprintln!("[WARN]: Handling non-manifold input (n = 3)");
                handle_pair(fv0, fv1);
                handle_pair(fv1, fv2);
                handle_pair(fv2, fv0);
                continue;
            }
            _ => {}
        }

        eprintln!(
            "[WARN]: Handling non-manifold input (n = {})",
            face_verts.len()
        );

        // break input face verts into pairs
        let e = normalize(sub(e1, e0));
        let tan = normalize(orthogonal(e));
        let bit = normalize(cross(e, tan));
        // for each face store, tangent direction orthogonal to e, and use winding around the
        // angle to order
        let mut angular_ord = face_verts
            .iter()
            .enumerate()
            .map(|(i, fv)| {
                let (_, opt_dir) = mesh.f[fv.0]
                    .edges()
                    .map(|e| {
                        let [e0, e1] = e.map(|vi| mesh.v[vi]);
                        normalize(sub(e1, e0))
                    })
                    .map(|dir| (dot(dir, e), dir))
                    .min_by(|(a, _), (b, _)| a.abs().partial_cmp(&b.abs()).unwrap())
                    .unwrap();
                let [x, y] = normalize([dot(opt_dir, tan), dot(opt_dir, bit)]);
                (y.atan2(x), i)
            })
            .collect::<Vec<_>>();
        angular_ord.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..angular_ord.len() {
            handle_pair(
                &face_verts[angular_ord[i].1],
                &face_verts[angular_ord[(i + 1) % angular_ord.len()].1],
            );
        }
    }

    // ------------- Zip corner faces together
    'outer: for (_key, mut fvs) in corner_map.into_iter() {
        assert!(fvs.iter().all(|fv| fv.1 < out.v.len()));
        if let Some(fv) = fvs.iter().find(|fv| !edge_adj.contains_key(&fv.1)) {
            assert!(false, "{fv:?} {fvs:?}");
        };
        if fvs.len() < 3 {
            continue;
        }

        let Some([e0, e1]) = (0..fvs.len()).find_map(|i| {
            ((i + 1)..fvs.len()).find_map(|j| {
                corner_edge_dir
                    .get(&std::cmp::minmax(fvs[i].0, fvs[j].0))
                    .copied()
            })
        }) else {
            // wut
            let poly = fvs.iter().map(|fv| fv.1).collect::<Vec<_>>();
            out.f.push(FaceKind::Poly(poly));
            face_labels.push(FaceLabel::GapCorner);
            continue;
        };

        let face = match fvs.len() {
            0..3 => unreachable!(),
            // Order doesn't matter here (except if it should flip)
            3 => {
                let t = FaceKind::Tri(std::array::from_fn(|i| fvs[i].1));
                if t.edges().any(|e| e == e0 || e == e1) {
                    FaceKind::Tri(std::array::from_fn(|i| t.as_slice()[2 - i]))
                } else {
                    t
                }
            }

            4 => {
                let first = fvs
                    .iter()
                    .filter(|fv| !edge_adj[&fv.1].is_empty())
                    .min_by_key(|fv| edge_adj[&fv.1].len())
                    .copied()
                    .unwrap_or_else(|| fvs[0])
                    .1;
                assert_ne!(first, usize::MAX);
                let mut quad = [first, usize::MAX, usize::MAX, usize::MAX];
                for i in 1..4 {
                    let n = &edge_adj[&quad[i - 1]];
                    let Some(&next) = n.iter().find(|&&v| !quad.contains(&v)) else {
                        println!(
                            "here {i:?} {:?}",
                            fvs.iter()
                                .map(|fv| (fv.1, edge_adj[&fv.1].clone()))
                                .collect::<Vec<_>>()
                        );
                        // here abort with the face as is?
                        if i == 3 {
                            out.f.push(FaceKind::Tri([quad[0], quad[1], quad[2]]));
                            face_labels.push(FaceLabel::GapCorner);
                        }
                        for fv in fvs.iter() {
                            out.vert_colors[fv.1] = [0., 0., 1.];
                        }
                        continue 'outer;
                    };
                    quad[i] = next;
                }
                let f = FaceKind::Quad(quad);
                if f.edges().any(|e| e == e0 || e == e1) {
                    FaceKind::Quad(std::array::from_fn(|i| quad[3 - i]))
                } else {
                    f
                }
            }
            _ => {
                // sometimes these can be split into multiple polygons? (for example if an input
                // vertex is non-manifold)
                while let Some(ci) = fvs
                    .iter()
                    .enumerate()
                    .filter(|(_, fv)| !edge_adj[&fv.1].is_empty())
                    .min_by_key(|(_, fv)| edge_adj[&fv.1].len())
                {
                    let mut curr = fvs.swap_remove(ci.0).1;
                    let mut curr_poly = vec![curr];
                    while let Some(&next) =
                        edge_adj[&curr].iter().find(|&&v| !curr_poly.contains(&v))
                    {
                        curr_poly.push(next);
                        curr = next;
                    }
                    curr_poly.reverse();

                    let mut curr = *curr_poly.last().unwrap();
                    while let Some(&next) =
                        edge_adj[&curr].iter().find(|&&v| !curr_poly.contains(&v))
                    {
                        curr_poly.push(next);
                        curr = next;
                    }

                    fvs.retain(|fv| !curr_poly.contains(&fv.1));

                    if curr_poly.len() < 3 {
                        continue;
                    }

                    let mut new_face = FaceKind::Poly(curr_poly);
                    if new_face.edges().any(|e| e == e0 || e == e1)
                        && let FaceKind::Poly(c) = &mut new_face
                    {
                        c.reverse();
                    }
                    assert!(!new_face.canonicalize());
                    assert!(new_face.len() > 2);
                    out.f.push(new_face);
                    face_labels.push(FaceLabel::GapCorner);
                }

                continue;
            }
        };
        face_labels.push(FaceLabel::GapCorner);
        out.f.push(face);
    }

    let remesh_time = start.elapsed();

    let before_simplify_num_faces = out.num_tris();
    // -------- Simplification of mesh

    assert_eq!(face_labels.len(), out.f.len());

    // compute statistics for degenerate mesh
    macro_rules! mesh_stats {
        ($label: expr) => {{
            macro_rules! count_face_label {
                ($p: pat) => {{
                    face_labels
                        .iter()
                        .enumerate()
                        .filter(|&(fi, c)| {
                            !out.f[fi].is_empty() && !out.f[fi].is_degenerate() && matches!(c, $p)
                        })
                        .count()
                }};
            }
            assert_eq!(face_labels.len(), out.f.len());
            let (_, bd_e, nm_e) = out.num_edge_kinds();
            println!(
                r#"--- {}
      Num Pixel   : {}
      Num Bridge  : {}, BridgeCorner : {}
      Num GapFill : {}, GapCorner    : {}
      Total       : {} (#Tris = {})
      Boundary Edges: {bd_e}, Non-Manifold Edges: {nm_e}"#,
                $label,
                count_face_label!(FaceLabel::Pixel),
                count_face_label!(FaceLabel::Bridge(_, _)),
                count_face_label!(FaceLabel::BridgeCorner),
                count_face_label!(FaceLabel::GapFill(_, _)),
                count_face_label!(FaceLabel::GapCorner),
                count_face_label!(_),
                out.f.iter().map(|f| f.num_tris()).sum::<usize>()
            );
        }};
    }
    mesh_stats!("Initial Generation");

    let init_f = out.f.iter().filter(|v| v.len() > 2).count();
    let num_tris = out.num_tris();
    let init_v = out.num_used_vertices();
    println!(
        "[INFO]: Initial generated mesh has {init_f} faces (Tris = {num_tris}) & {init_v} vertices"
    );

    //canonicalize!();

    let simplify_start = std::time::Instant::now();

    if args.delete_degen {
        del_degen_gap_fill(&mut remap, &mut out, args, &face_labels);
        canonicalize!();

        mesh_stats!("After deleting gap fill");
    }
    if args.triangulate {
        out.triangulate();
    }

    let final_qem = !args.no_final_qem
        && (args.target_tri_ratio < 1.
            || (args.target_tri_num > 0 && args.target_tri_num < out.num_tris()));
    if final_qem {
        let qem_args = QEMArgs {
            target_tri_ratio,
            target_tri_num: args.target_tri_num,
            display_progress: true,
            ..QEMArgs::default()
        };

        let num_f = out.f.len();
        let num_v = out.v.len();
        simplify_range_colored(
            &mut out,
            &qem_args,
            |_| false,
            0..num_f,
            0..num_v,
            &mut remap,
            &mut qem_buf,
        );
        if !args.triangulate {
            mesh_stats!("After QEM");
        }
    }

    out.f.retain_mut(|f| {
        if f.is_empty() {
            return false;
        }

        f.remap(|vi| remap.get_compress(vi));
        !f.canonicalize()
    });

    out.delete_unused_vertices();

    let simplification_time = simplify_start.elapsed();

    //out.remove_doublets();
    if init_f != out.f.len() {
        println!(
            "[INFO]: Cleaned mesh has {} faces (-{}, Tris = {}) & {} vertices (-{})",
            out.f.len(),
            init_f.saturating_sub(out.f.len()),
            out.num_tris(),
            out.v.len(),
            init_v.saturating_sub(out.v.len()),
        );
    }

    Outputs {
        mesh: out,
        remesh_time,
        simplification_time,
        before_simplify_num_faces,
    }
}

pub fn sample_exact(
    mesh: &Mesh,
    f: &FaceKind,
    fi: usize,
    src: &SourceTextures,

    // destination for all attributes
    out: &mut Mesh,

    // map from original vertex -> (original face idx, vertex)
    corner_map: &mut Map<[U; 3], Vec<(usize, usize)>>,
    // map from edge -> (original face idx, vertices on face)
    edge_map: &mut Map<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
    // map from new vertex index to original vertex position and distance
    labels: &mut BTreeMap<usize, [(u32, [U; 3]); 2]>,

    // label for what kind of face is being inserted into the mesh, useful for cleaning up faces
    // later
    face_labels: &mut Vec<FaceLabel>,

    args: &Args,
) -> bool {
    let mut aabb = AABB::<F, 2>::new();
    let f_slice = f.as_slice();

    for uv in f_slice.iter().map(|&vi| mesh.uv[CHAN][vi]) {
        aabb.add_point(uv);
    }

    let (w, h) = src.diff_img.dimensions();
    aabb.scale_by(w as F, h as F);
    let iaabb = aabb.round_to_i32();
    let uv_f = f.map_kind(|vi| mesh.uv[CHAN][vi]);
    let v_f = f.map_kind(|vi| mesh.v[vi]);
    let n_f = if !mesh.n.is_empty() {
        Some(f.map_kind(|vi| mesh.n[vi]))
    } else {
        None
    };

    // face normal for projecting bary
    let f_n = normalize(v_f.normal());
    if length(f_n) < 1e-3 {
        assert!(v_f.area() < 1e-12);
        // degenerate face (0 area), just use direct sampling
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    // for each pixel, what vertices are associated with it?
    let mut pixel_map: BTreeMap<_, [usize; 4]> = BTreeMap::new();
    let mut pix_fi: BTreeMap<_, usize> = BTreeMap::new();

    let iarea = iaabb.area();
    if iarea == 0 {
        // there is no area to this triangle, difficult to sample pixels so just skip it.
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    // simple check for uniformity, if all are equal drop to sample_direct
    let mut color_iter = iaabb.iter_coords().map(|c| {
        let [u, v] = c.map(|v| v as F + 0.5);
        let [u, v] = [u / w as F, v / h as F];
        src.get_value(u, v)
    });
    let first_col = color_iter.next().unwrap();
    if !args.no_adaptive && color_iter.all(|v| dist(v, first_col) < 1e-8) {
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    let start = out.v.len();
    let start_f = out.f.len();
    let mut isect_buf = vec![vec![]; 1];
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
        let mut cf_aabb = pars3d::aabb::AABB::<F, 2>::new();
        for cf in cfs {
            cf_aabb.add_point(cf);
        }

        let barys = cfs.map(|cf| uv_f.barycentric(cf));

        if isect_buf.len() < uv_f.num_tris() {
            isect_buf.resize_with(uv_f.num_tris(), Vec::new);
        }
        let all_outside = uv_f.as_triangle_fan().enumerate().all(|(ti, uv_t)| {
            cf_aabb.intersects_tri_poly(uv_t, &mut isect_buf[ti]);
            isect_buf[ti].is_empty()
        });

        if all_outside {
            continue;
        }

        let center_uv = [(u + 0.5) / w as F, (v + 0.5) / h as F];
        let bary = uv_f.barycentric(center_uv);
        let [tex_u, tex_v] = uv_f.from_barycentric(bary);
        assert!(tex_u.is_finite() && tex_v.is_finite(), "{bary:?} {uv_f:?}");

        let center_rgb = src.get_value(tex_u, tex_v);

        let raw_pos = barys.map(|bary| v_f.from_barycentric(bary));

        let new_verts = std::array::from_fn(|i| {
            let bary = barys[i];
            let pos = raw_pos[i];
            let (ti, bs) = bary.tri_idx_and_coords();
            if !bs.iter().any(|&b| b < -1e-3) {
                let normal = n_f.as_ref().map(|n_f| n_f.from_barycentric(bary));
                return (pos, normal);
            };

            // find nearest point in UV in isect buf and use that to compute new barycentric and
            // position
            let isects = &isect_buf[ti];
            let (new_pos, new_bary) = if isects.is_empty() {
                let tri = v_f.as_triangle_fan().nth(ti).unwrap();
                let new_pos = nearest_point_on_tri(tri, pos);
                let bary_coord = pars3d::barycentric_3d(new_pos, tri);

                let new_bary = pars3d::face::Barycentric::new(&v_f, ti, bary_coord);
                (new_pos, new_bary)
            } else {
                let cf = cfs[i];
                let (nearest_isect, _) = isects
                    .iter()
                    .map(|&i_uv| (i_uv, dist(cf, i_uv)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();

                let uv_tri = uv_f.as_triangle_fan().nth(ti).unwrap();
                let bary_coord = pars3d::barycentric_2d(nearest_isect, uv_tri);

                let new_bary = pars3d::face::Barycentric::new(&uv_f, ti, bary_coord);
                let new_pos = v_f.from_barycentric(new_bary);
                (new_pos, new_bary)
            };

            let new_normal = n_f.as_ref().map(|n_f| n_f.from_barycentric(new_bary));
            debug_assert!(
                v_f.barycentric(new_pos)
                    .coords()
                    .iter()
                    .all(|v| (-1e-3..=1.001).contains(v)),
                "{:?}",
                v_f.barycentric(new_pos).coords(),
            );
            (new_pos, new_normal)
        });

        // commit to this new pixel
        let new_verts = new_verts.map(|(new_vert, normal)| {
            let vi = out.v.len();
            out.v.push(new_vert);
            /*
            out.vert_colors.push(if bary.tri_idx() == 0 {
              [1., 0., 0.]
            } else {
              [0., 0., 1.]
            });
            */
            out.vert_colors.push(center_rgb);
            if let Some(normal) = normal {
                out.n.push(normal);
            }

            vi
        });

        let prev = pixel_map.insert(c, new_verts);
        assert_eq!(prev, None);
        let new_fi = out.f.len();
        pix_fi.insert(c, new_fi);
        face_labels.push(FaceLabel::Pixel);
        out.f.push(FaceKind::Quad(new_verts));
    }
    if out.f.len() == start_f {
        face_labels.truncate(start_f);
        out.f.truncate(start_f);
        out.vert_colors.truncate(start);
        out.v.truncate(start);
        out.n.truncate(start);
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    let cardinal_dirs = |[u, v]: [i32; 2]| [[u + 1, v], [u, v + 1], [u - 1, v], [u, v - 1]];
    let diags = |[u, v]: [i32; 2]| {
        [
            [u + 1, v + 1],
            [u + 1, v - 1],
            [u - 1, v + 1],
            [u - 1, v - 1],
        ]
    };
    let mut any_isolated = false;
    for &uv in pixel_map.keys() {
        let has_nbr = cardinal_dirs(uv)
            .into_iter()
            .chain(diags(uv).into_iter())
            .any(|[nu, nv]| pixel_map.contains_key(&[nu, nv]));
        any_isolated = any_isolated || !has_nbr;
    }

    if any_isolated {
        out.v.truncate(start);
        out.n.truncate(start);
        out.vertex_attrs.truncate(start);
        out.f.truncate(start_f);
        face_labels.truncate(start_f);
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    // hohoho what a funny name
    macro_rules! triagram {
        () => {{
            print!("  ");
            print!(
                "{} {} ",
                iaabb.width_range().start,
                iaabb.width_range().start + 1
            );
            println!();
            for h in iaabb.height_range() {
                print!("{h: >4}");
                for w in iaabb.width_range() {
                    let c = if pixel_map.contains_key(&[w, h]) {
                        "x"
                    } else {
                        "-"
                    };
                    print!("{c: >4}");
                }
                println!();
            }
        }};
    }

    macro_rules! save_bad_mesh {
        ($label: expr) => {{
            let mut vi = 0;
            let mut f = f.clone();
            f.remap(|_| {
                let curr = vi;
                vi += 1;
                curr
            });
            let mut tmp =
                Mesh::new_geometry(f_slice.iter().map(|vi| mesh.v[*vi]).collect(), vec![f]);

            println!(
                "--- Bad Poly (fi = {fi}, {} vertices) Diagram ---",
                f_slice.len()
            );
            triagram!();

            tmp.uv[CHAN] = f_slice.iter().map(|vi| mesh.uv[CHAN][*vi]).collect();
            pars3d::save("failing_tri.obj", &tmp.into_scene()).expect("Failed to save temp error");

            let error_verts = out.v[start..].to_vec();
            let error_colors = out.vert_colors[start..].to_vec();
            let mut error_faces = out.f[start_f..].to_vec();
            for f in &mut error_faces {
                f.offset(-(start as i32))
            }
            out.n.clear();
            out.v = error_verts;
            out.vert_colors = error_colors;
            out.f = error_faces;
            eprintln!($label);

            return false;
        }};
    }
    // sanity check
    // if there is ever a pixel which doesn't have a direct nbr, we just delete it.
    if iaabb.area() > 1 {
        for [u, v] in iaabb.iter_coords() {
            if !pixel_map.contains_key(&[u, v]) {
                continue;
            }
            let nbrs = [[u - 1, v], [u + 1, v], [u, v - 1], [u, v + 1]];
            let diags = [
                [u - 1, v - 1],
                [u - 1, v + 1],
                [u + 1, v - 1],
                [u + 1, v + 1],
            ];
            let has_nbr = nbrs.iter().any(|nbr| pixel_map.contains_key(nbr));
            let has_diag = diags.iter().any(|d| pixel_map.contains_key(d));
            if !has_nbr && has_diag {
                //save_bad_mesh!("Each pixel in a tri should have at least one direct nbr");
                let pix_verts = FaceKind::Quad(pixel_map.remove(&[u, v]).unwrap());
                // this is rare, so it's ok for it to be expensive
                let del_f = out.f[start_f..]
                    .iter()
                    .position(|f| f == &pix_verts)
                    .unwrap();
                out.f[start_f..][del_f] = FaceKind::empty();
                face_labels[start_f..][del_f] = FaceLabel::Deleted;
            }
        }
    }

    // --- Adding faces in between pixel quads
    for (&[u, v], &[l, r, ur, ul]) in pixel_map.iter() {
        if args.no_fill {
            break;
        }
        let own_fi = pix_fi[&[u, v]];
        let down_right = pixel_map.get(&[u + 1, v + 1]);
        // check bottom right first
        if down_right.is_none()
            && let Some(&[_, _, _, a]) = pixel_map.get(&[u + 1, v])
            && let Some(&[_, b, _, _]) = pixel_map.get(&[u, v + 1])
        {
            face_labels.push(FaceLabel::BridgeCorner);
            out.f.push(FaceKind::Tri([ur, a, b]));
        }
        if let Some(a) = pixel_map.get(&[u + 1, v - 1])
            && !pixel_map.contains_key(&[u + 1, v])
            && !pixel_map.contains_key(&[u, v - 1])
        {
            face_labels.push(FaceLabel::BridgeCorner);
            // still looks kind of odd but it works sort of
            //out.f.push(FaceKind::Quad([r, ur, a[0], a[3]]));
            //out.f.push(FaceKind::Quad([l, a[0], a[3], r]));
            out.f.push(FaceKind::Quad([r, a[3], a[0], l]));
        }
        // TODO turn these into checked subs?
        let left = pixel_map.get(&[u - 1, v]);
        if let Some(&[_, or, our, _]) = left {
            face_labels.push(FaceLabel::Bridge(own_fi, pix_fi[&[u - 1, v]]));
            out.f.push(FaceKind::Quad([ul, our, or, l]));
        }
        let up = pixel_map.get(&[u, v - 1]);
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
            (Some(a), None, None) => {
                FaceKind::Quad([l, a[2], a[1], r])
                //face_labels.push(FaceLabel::BridgeCorner);
                //out.f.push();
                //face_labels.push(FaceLabel::BridgeCorner);
                //out.f.push(FaceKind::Quad([l, ul, a[3], a[2]]));
                //save_bad_mesh!("Corner weirdness");
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
        for [e0, e1] in f.edges_ord() {
            assert_ne!(e0, e1);
            edge_face_adj
                .entry([e0, e1])
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
        assert!(!ek.is_non_manifold());
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
    assert!(check);
    let check = vert_adj.values().all(|&[a0, a1]| a0 != a1);
    assert!(check);

    // --- Compute correspondence between original vertices and a single pixel vertex, which
    // must be a boundary.

    let mut corner_verts = f.map_kind(|_| (usize::MAX, usize::MAX) /* (og, new) */);
    let corner_verts_s = corner_verts.as_mut_slice();
    for (ci, &og_vi) in f_slice.iter().enumerate() {
        let vert_pos = mesh.v[og_vi];

        let [og_u, og_v] = mesh.uv[CHAN][og_vi];
        let u = (og_u.fract().abs() * w as F + 0.5).floor() as i32;
        let v = (og_v.fract().abs() * h as F + 0.5).floor() as i32;
        assert!(u >= 0, "{og_u} {u} {v}");
        assert!(v >= 0, "{og_u} {u} {v}");

        let mut nearest = 0;
        let mut best_dist = F::INFINITY;
        // TODO do this check more efficiently
        // naive search thru everything to ensure that the nearest point is really nearest
        for (&new_vi, _) in vert_adj.range(start..) {
            if !vert_adj.contains_key(&new_vi) {
                continue;
            }
            if corner_verts_s.iter().any(|&(_, p)| p == new_vi) {
                continue;
            }
            let d = dist(out.v[new_vi], vert_pos);
            if d < best_dist {
                nearest = new_vi;
                best_dist = d;
            }
        }
        if !best_dist.is_finite() {
            save_bad_mesh!("No nearby points");
        }
        assert_ne!(best_dist, F::INFINITY);

        // check that each corner vert corresponds to a single original vertex
        let check = corner_verts_s.iter().all(|&(_, new_vi)| new_vi != nearest);
        assert!(check);

        corner_verts_s[ci] = (og_vi, nearest);
    }

    // --- cleaning up winding order

    let (&start, adjs) = vert_adj.first_key_value().unwrap();
    let mut prev: usize = start;
    let mut curr: usize = adjs[0];
    let mut og_order = vec![];
    if let Some(&(og_vi, _)) = corner_verts_s.iter().find(|c| c.1 == curr) {
        og_order.push(og_vi);
    }
    while curr != start {
        let next = *vert_adj[&curr].iter().find(|&&n| n != prev).unwrap();
        if let Some(&(og_vi, _)) = corner_verts_s.iter().find(|c| c.1 == next) {
            og_order.push(og_vi);
        }
        prev = curr;
        curr = next;
    }

    if og_order.len() != f.len() {
        save_bad_mesh!("There was an island");
    }
    assert_eq!(og_order.len(), f.len(), "{og_order:?} {corner_verts:?}");
    while og_order[0] != f_slice[0] {
        og_order.rotate_left(1);
    }

    if f_slice != og_order {
        for (fi, f) in out.f.iter_mut().enumerate().skip(start_f) {
            if face_labels[fi] != FaceLabel::Pixel {
                continue;
            }
            f.as_mut_slice().reverse();
        }
    }

    // --- End clean up winding order section

    for &(og_vi, new_vi) in corner_verts.as_slice() {
        let og_pos = mesh.v[og_vi];
        let fv = corner_map.entry(og_pos.map(F::to_bits)).or_default();
        assert!(!fv.iter().any(|fv| fv.0 == fi));
        fv.push((fi, new_vi));

        // Pull to a corner to make it tight (larger T is tighter)
        let t = args.vertex_pull;
        out.v[new_vi] = add(kmul(t, mesh.v[og_vi]), kmul(1. - t, out.v[new_vi]));

        if args.debug_colors {
            out.vert_colors[new_vi] = [1.; 3];
        }
    }

    if args.no_gap_fill {
        return true;
    }

    const INVALID_POS: [U; 3] = [U::MAX; 3];
    let corner_verts = corner_verts.as_slice();
    for &(og_vi, new_vi) in corner_verts {
        assert!(vert_adj.contains_key(&new_vi), "{new_vi} {corner_verts:?}");
        let [l, r] = vert_adj[&new_vi];
        assert_ne!(l, r);
        assert_ne!(l, new_vi);
        assert_ne!(r, new_vi);
        let mut iter = |mut curr: usize, mut prev: usize| {
            let mut c = 1;
            while !corner_verts.iter().any(|v| v.1 == curr) {
                let label = labels.entry(curr).or_insert([(0, INVALID_POS); 2]);
                let new_pos = mesh.v[og_vi].map(F::to_bits);
                *label.iter_mut().find(|v| v.1 == INVALID_POS).unwrap() = (c, new_pos);
                assert!(vert_adj[&curr].iter().any(|&v| v == prev));
                let next = *vert_adj[&curr].iter().find(|v| **v != prev).unwrap();
                prev = curr;
                curr = next;
                c += 1;
            }
            curr
        };

        let mut fix_up_side = |v| {
            let end_pt = iter(v, new_vi);
            if end_pt == new_vi {
                save_bad_mesh!("Mesh had some weirdness (maybe an isolated island)");
            }
            assert_ne!(end_pt, new_vi, "{start_f} {}", out.f.len());
            let next_og_vi = corner_verts.iter().find(|v| v.1 == end_pt).unwrap().0;
            let [e0, e1] = [og_vi, next_og_vi].map(|vi| mesh.v[vi]);
            let [e0_key, e1_key] = [e0, e1].map(|v| v.map(F::to_bits));

            // add to the edge map here, to ensure that even if the edge doesn't have any
            // intermediate vertices it will be labeled.
            assert_ne!(e0_key, e1_key);
            let fvs = edge_map.entry(minmax(e0_key, e1_key)).or_default();
            if fvs.iter_mut().find(|fv| fv.0 == fi).is_none() {
                fvs.push((fi, vec![]));
            }
            true
        };

        // important to go both ways since sometimes vert_adj is not oriented correctly.
        if !fix_up_side(l) || !fix_up_side(r) {
            return false;
        }
    }
    let check = labels
        .range(start..)
        .all(|(_, v)| v[0].1 != INVALID_POS && v[1].1 != INVALID_POS);
    assert!(check);
    let check = labels.range(start..).all(|(_, v)| v[0].1 != v[1].1);
    assert!(check);

    for (&new_vi, ogs) in labels.range(start..) {
        if args.debug_colors && out.vert_colors[new_vi] != [1.; 3] {
            out.vert_colors[new_vi] = [0.; 3];
        }
        let [og0_key, og1_key] = ogs.map(|vi| vi.1);
        assert_ne!(og0_key, og1_key);
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

pub fn sample_approx(
    mesh: &Mesh,
    f: &FaceKind,
    fi: usize,
    src: &SourceTextures,
    out: &mut Mesh,

    // map from edge -> (original face idx, vertices), which stores vertices along every half edge
    corner_map: &mut Map<[U; 3], Vec<(usize, usize)>>,
    // map from edge -> (original face idx, vertices on face)
    edge_map: &mut Map<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
    // map from new vertex index to original vertex position and distance
    labels: &mut BTreeMap<usize, [(u32, [U; 3]); 2]>,
    face_labels: &mut Vec<FaceLabel>,

    args: &Args,
) -> bool {
    let mut aabb = AABB::<F, 2>::new();
    let f_slice = f.as_slice();
    for uv in f_slice.iter().map(|&vi| mesh.uv[CHAN][vi]) {
        aabb.add_point(uv);
    }

    let (w, h) = src.diff_img.dimensions();
    aabb.scale_by(w as F, h as F);
    let iaabb = aabb.round_to_i32();
    if iaabb.area() == 0 {
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    let uv_f = f.map_kind(|vi| mesh.uv[CHAN][vi]);
    let v_f = f.map_kind(|vi| mesh.v[vi]);
    let n_f = if !mesh.n.is_empty() {
        Some(f.map_kind(|vi| mesh.n[vi]))
    } else {
        None
    };

    let mut pixel_map = BTreeMap::new();

    // if there is at most one line, just abort
    assert_ne!(iaabb.width(), 0);
    assert_ne!(iaabb.height(), 0);
    if iaabb.width() == 1 || iaabb.height() == 1 {
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    if !args.no_adaptive {
        let mut color_iter = iaabb.iter_coords().map(|c| {
            let [u, v] = c.map(|v| v as F + 0.5);
            let [u, v] = [u / w as F, v / h as F];
            src.get_value(u, v)
        });
        let Some(first_col) = color_iter.next() else {
            return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
        };
        if color_iter.all(|v| dist_sq(v, first_col) < (1e-8 as F).sqrt()) {
            return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
        }
    }

    let start = out.v.len();
    let start_f = out.f.len();
    for c in iaabb.iter_coords() {
        let [u, v] = c.map(|v| v as F);
        let cf = [(u + 0.5) / w as F, (v + 0.5) / h as F];
        let delta = args.pixel_sep;
        let cfs = [
            [u + (1. - delta), v + (1. - delta)],
            [u + delta, v + (1. - delta)],
            [u + delta, v + delta],
            [u + (1. - delta), v + delta],
        ];
        let cfs = cfs.map(|[u, v]| [u / w as F, v / h as F]);
        // small epsilon to handle points which are very close to edges.
        let outside_all = uv_f.as_triangle_fan().all(|uv_t| {
            let t_bary = cfs.map(|cf| pars3d::barycentric_2d(cf, uv_t));
            (0..3).any(|i| t_bary.iter().all(|b| b[i] < 0.) || t_bary.iter().all(|b| b[i] > 1.))
        });
        if outside_all {
            continue;
        }

        let bary = uv_f.barycentric(cf);
        let (ti, bs) = bary.tri_idx_and_coords();
        let pos = v_f.from_barycentric(bary);
        let (p, rgb, bary) = if !bs.iter().any(|&b| b < 0.) {
            (pos, src.get_value(cf[0], cf[1]), bary)
        } else {
            let tri = unsafe { v_f.as_triangle_fan().nth(ti).unwrap_unchecked() };
            let new_pos = nearest_point_on_tri(tri, pos);
            let new_bary = v_f.barycentric(new_pos);
            let [tu, tv] = uv_f.from_barycentric(new_bary);
            (new_pos, src.get_value(tu, tv), new_bary)
        };
        let n = n_f.as_ref().map(|n_f| n_f.from_barycentric(bary));

        let vi = out.v.len();
        out.v.push(p);
        out.vert_colors.push(rgb);
        if let Some(n) = n {
            out.n.push(n);
        }
        let prev = pixel_map.insert(c, vi);
        debug_assert_eq!(prev, None);
    }

    macro_rules! triagram {
        () => {{
            print!("  ");
            print!(
                "{} {} ",
                iaabb.width_range().start,
                iaabb.width_range().start + 1
            );
            println!();
            for h in iaabb.height_range() {
                print!("{h} ");
                for w in iaabb.width_range() {
                    let c = if let Some(c) = pixel_map.get(&[w, h]) {
                        let c = c % 10000;
                        format!("{c}")
                    } else {
                        String::from("-")
                    };
                    print!("{c: >4} ");
                }
                println!();
            }
        }};
    }
    assert_eq!(start_f, out.f.len());

    macro_rules! truncate {
        () => {{
            out.v.truncate(start);
            out.vert_colors.truncate(start);
            out.n.truncate(start);
            out.vertex_attrs.truncate(start);
            out.f.truncate(start_f);
            face_labels.truncate(start_f);
        }};
    }

    // There aren't even enough vertices to make a triangle,
    // Fall back to using direct sampling
    let num_verts = out.v.len() - start;
    if num_verts < 3 || num_verts < f.len() {
        truncate!();
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    // Can be represented as direct.
    let first_color = out.vert_colors[start];
    let all_same_color = out
        .vert_colors
        .iter()
        .skip(start + 1)
        .all(|&vc| dist_sq(first_color, vc) < (1e-8 as F).sqrt());
    if !args.no_adaptive && all_same_color {
        truncate!();
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    // also check if all pixel maps are only 1 in width
    let mut has_width = false;
    let mut has_height = false;
    for &[u, v] in pixel_map.keys() {
        has_height = has_height
            || pixel_map.contains_key(&[u + 1, v])
            || pixel_map.contains_key(&[u.wrapping_sub(1), v]);

        has_width = has_width
            || pixel_map.contains_key(&[u, v + 1])
            || pixel_map.contains_key(&[u, v.wrapping_sub(1)]);

        if has_width && has_height {
            break;
        }
    }
    if !has_width || !has_height {
        truncate!();
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    // check that no vertex will become non-manifold
    let mut any_non_manifold = false;
    for &[u, v] in pixel_map.keys() {
        let has_lr = pixel_map.contains_key(&[u + 1, v]) && pixel_map.contains_key(&[u - 1, v]);
        let has_one_ud = pixel_map.contains_key(&[u + 1, v + 1])
            || pixel_map.contains_key(&[u + 1, v - 1])
            || pixel_map.contains_key(&[u - 1, v + 1])
            || pixel_map.contains_key(&[u - 1, v - 1])
            || pixel_map.contains_key(&[u, v + 1])
            || pixel_map.contains_key(&[u, v - 1]);
        // if it has left-right, it must also have at least one up or down
        if has_lr && !has_one_ud {
            any_non_manifold = true;
            break;
        }

        let has_ud = pixel_map.contains_key(&[u, v + 1]) && pixel_map.contains_key(&[u, v - 1]);
        let has_one_lr = pixel_map.contains_key(&[u + 1, v])
            || pixel_map.contains_key(&[u - 1, v])
            || pixel_map.contains_key(&[u + 1, v + 1])
            || pixel_map.contains_key(&[u - 1, v + 1])
            || pixel_map.contains_key(&[u + 1, v - 1])
            || pixel_map.contains_key(&[u - 1, v - 1]);
        if has_ud && !has_one_lr {
            any_non_manifold = true;
            break;
        }
    }
    // if there is exactly one non-manifold item, possibly may be ok, but need to add a more
    // complex check.
    if any_non_manifold {
        truncate!();
        return sample_exact(
            mesh,
            f,
            fi,
            src,
            out,
            corner_map,
            edge_map,
            labels,
            face_labels,
            args,
        );
    }
    //triagram!();

    // also check if a vertex only has collinear neighbors

    let any_collinear = pixel_map.keys().any(|&uv| {
        let lr = |[u, v]: [i32; 2]| [[u + 1, v], [u - 1, v]];
        let above_below = |[u, v]: [i32; 2]| {
            [
                [u + 1, v + 1],
                [u, v + 1],
                [u - 1, v + 1],
                [u + 1, v - 1],
                [u, v - 1],
                [u - 1, v - 1],
            ]
        };

        let collinear = lr(uv).iter().all(|v| pixel_map.contains_key(v))
            && above_below(uv).iter().all(|v| !pixel_map.contains_key(v));
        if collinear {
            return true;
        }

        let ud = |[u, v]: [i32; 2]| [[u + 1, v], [u - 1, v]];
        let left_right = |[u, v]: [i32; 2]| {
            [
                [u + 1, v + 1],
                [u + 1, v],
                [u + 1, v - 1],
                [u - 1, v + 1],
                [u - 1, v],
                [u - 1, v - 1],
            ]
        };

        ud(uv).iter().all(|v| pixel_map.contains_key(v))
            && left_right(uv).iter().all(|v| !pixel_map.contains_key(v))
    });

    if any_collinear {
        truncate!();
        // TODO may be better to use sample_exact for this one.
        return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
    }

    let cardinal_dirs = |[u, v]: [i32; 2]| [[u + 1, v], [u, v + 1], [u - 1, v], [u, v - 1]];
    let diags = |[u, v]: [i32; 2]| {
        [
            [u + 1, v + 1],
            [u + 1, v - 1],
            [u - 1, v + 1],
            [u - 1, v - 1],
        ]
    };
    for (uv, _) in pixel_map.iter() {
        let any_nbr = cardinal_dirs(*uv)
            .into_iter()
            .any(|cd| pixel_map.contains_key(&cd));
        let any_diag = diags(*uv).into_iter().any(|cd| pixel_map.contains_key(&cd));
        if any_diag || any_nbr {
            continue;
        }
        truncate!();
        return sample_exact(
            mesh,
            f,
            fi,
            src,
            out,
            corner_map,
            edge_map,
            labels,
            face_labels,
            args,
        );
    }

    // compute faces for each new vertex (these are all pixels)
    for (&[u, v], &vi) in pixel_map.iter() {
        let up = pixel_map.get(&[u, v + 1]).copied();
        let left = pixel_map.get(&[u + 1, v]).copied();
        let upleft = pixel_map.get(&[u + 1, v + 1]).copied();

        // handle case where vertex may be non-manifold
        // No vertex on left-right case
        if !pixel_map.contains_key(&[u - 1, v]) && !pixel_map.contains_key(&[u + 1, v]) {
            if let Some(&up) = pixel_map.get(&[u, v + 1]) {
                if let Some(&dl) = pixel_map.get(&[u - 1, v - 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, up, dl]));
                }
                if let Some(&dr) = pixel_map.get(&[u + 1, v - 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, dr, up]));
                }
            }
            if let Some(&down) = pixel_map.get(&[u, v - 1]) {
                if let Some(&ul) = pixel_map.get(&[u - 1, v + 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, ul, down]));
                }
                if let Some(&ur) = pixel_map.get(&[u + 1, v + 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, down, ur]));
                }
            }
        }

        if !pixel_map.contains_key(&[u, v - 1]) && !pixel_map.contains_key(&[u, v + 1]) {
            if let Some(&r) = pixel_map.get(&[u + 1, v]) {
                if let Some(&ul) = pixel_map.get(&[u - 1, v + 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, r, ul]));
                }
                if let Some(&dl) = pixel_map.get(&[u - 1, v - 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, dl, r]));
                }
            }
            if let Some(&l) = pixel_map.get(&[u - 1, v]) {
                if let Some(&ur) = pixel_map.get(&[u + 1, v + 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, ur, l]));
                }
                if let Some(&dr) = pixel_map.get(&[u + 1, v - 1]) {
                    face_labels.push(FaceLabel::Pixel);
                    out.f.push(FaceKind::Tri([vi, l, dr]));
                }
            }
        }

        if !pixel_map.contains_key(&[u - 1, v - 1])
            && let Some(&r) = pixel_map.get(&[u - 1, v])
            && let Some(&d) = pixel_map.get(&[u, v - 1])
        {
            face_labels.push(FaceLabel::Pixel);
            out.f.push(FaceKind::Tri([vi, r, d]));
        }
        let new_face = match (up, upleft, left) {
            (Some(u), Some(ul), Some(l)) => FaceKind::Quad([l, ul, u, vi]),
            (Some(u), None, Some(l)) => FaceKind::Tri([l, u, vi]),
            (Some(u), Some(ul), None) => FaceKind::Tri([ul, u, vi]),
            (None, Some(ul), Some(l)) => FaceKind::Tri([l, ul, vi]),
            _ => continue,
        };
        face_labels.push(FaceLabel::Pixel);
        out.f.push(new_face);
    }

    // wind around the outer vertices, and connect them in boundary order

    let mut edge_face_adj: BTreeMap<[usize; 2], EdgeKind> = BTreeMap::new();
    // compute adjacent boundary edges and trace from the corners
    for (fi, f) in out.f.iter().enumerate().skip(start_f) {
        for [e0, e1] in f.edges_ord() {
            assert_ne!(e0, e1);
            edge_face_adj
                .entry([e0, e1])
                .and_modify(|v| {
                    let did_ins = v.insert(fi);
                    assert!(did_ins);
                })
                .or_insert(EdgeKind::Boundary(fi));
        }
    }
    if edge_face_adj.values().any(EdgeKind::is_non_manifold) {
        truncate!();
        return sample_exact(
            mesh,
            f,
            fi,
            src,
            out,
            corner_map,
            edge_map,
            labels,
            face_labels,
            args,
        );
    }

    // Compute adjacent vertices to each boundary vertex
    let mut vert_adj: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
    for (&[ef0, ef1], ek) in edge_face_adj.iter() {
        assert_ne!(ef0, ef1);
        if !ek.is_boundary() {
            continue;
        }

        let mut ins = |a: usize, b: usize| {
            let adj = vert_adj
                .entry(a)
                .or_insert([usize::MAX; 2])
                .iter_mut()
                .find(|v| **v == usize::MAX);
            let Some(adj) = adj else {
                return false;
            };
            *adj = b;
            true
        };
        if !ins(ef0, ef1) || !ins(ef1, ef0) {
            truncate!();
            return sample_direct(mesh, f, fi, src, out, face_labels, corner_map, edge_map);
        };
    }

    let check = vert_adj
        .values()
        .all(|&[a0, a1]| a0 != usize::MAX && a1 != usize::MAX);
    debug_assert!(check);

    let mut corner_verts = f.map_kind(|_| (usize::MAX, usize::MAX) /* (og, new) */);
    let corner_verts_s = corner_verts.as_mut_slice();
    for (ci, &og_vi) in f_slice.iter().enumerate() {
        let vert_pos = mesh.v[og_vi];

        let [og_u, og_v] = mesh.uv[CHAN][og_vi];
        let u = (og_u.fract().abs() * w as F + 0.5).floor() as i32;
        let v = (og_v.fract().abs() * h as F + 0.5).floor() as i32;
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
                let Some(&vi) = pixel_map.get(&p) else {
                    continue;
                };
                // only boundary vertices
                if !vert_adj.contains_key(&vi) {
                    continue;
                }
                // unique mapping
                if corner_verts_s.iter().any(|&(_, p)| p == vi) {
                    continue;
                }
                let d = dist(out.v[vi], vert_pos);
                if d < best_dist {
                    nearest = vi;
                    best_dist = d;
                }
            }
        }
        if !best_dist.is_finite() {
            // naive search thru everything
            for (&new_vi, _) in vert_adj.range(start..) {
                if corner_verts_s.iter().any(|&(_, p)| p == new_vi) {
                    continue;
                }
                let d = dist(out.v[new_vi], vert_pos);
                if d < best_dist {
                    nearest = new_vi;
                    best_dist = d;
                }
            }
        }
        if best_dist == F::INFINITY {
            triagram!();
        }
        assert_ne!(best_dist, F::INFINITY);

        // check that each corner vert corresponds to a single original vertex
        let check = corner_verts_s.iter().any(|&(_, new_vi)| new_vi == nearest);
        assert!(!check);

        corner_verts_s[ci] = (og_vi, nearest);
    }

    // --- fix up winding order of corner verts (FIXME)
    let (&start, adjs) = vert_adj.first_key_value().unwrap();
    let mut prev: usize = start;
    let mut curr: usize = adjs[0];
    let mut og_order = vec![];
    if let Some(&(og_vi, _)) = corner_verts_s.iter().find(|c| c.1 == curr) {
        og_order.push(og_vi);
    }
    while curr != start {
        let next = *vert_adj[&curr].iter().find(|&&n| n != prev).unwrap();
        //assert_eq!(next, vert_adj[&curr][1]);
        if let Some(&(og_vi, _)) = corner_verts_s.iter().find(|c| c.1 == next) {
            og_order.push(og_vi);
        }
        prev = curr;
        curr = next;
    }

    assert_eq!(og_order.len(), f.len());

    while og_order[0] != f_slice[0] {
        og_order.rotate_left(1);
    }
    /*
     */

    if f_slice != og_order {
        //println!("{og_order:?} {f_slice:?}");
        og_order.reverse();
        og_order.rotate_right(1);
        if f_slice == og_order {
            for v in vert_adj.values_mut() {
                v.reverse();
            }
        }
    }

    for &(og_vi, new_vi) in corner_verts.as_slice() {
        let og_pos = mesh.v[og_vi];
        let fv = corner_map.entry(og_pos.map(F::to_bits)).or_default();
        assert!(!fv.iter().any(|fv| fv.0 == fi));
        fv.push((fi, new_vi));

        if args.debug_colors {
            out.vert_colors[new_vi] = [1.; 3];
        }
    }

    // The following is identical to sample_exact, no modifications were made.

    if args.no_gap_fill {
        return true;
    }

    const INVALID_POS: [U; 3] = [U::MAX; 3];
    let corner_verts = corner_verts.as_slice();
    for &(og_vi, new_vi) in corner_verts {
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
        let mut fix_up_side = |v| {
            let end_pt = iter(v, new_vi);
            assert_ne!(end_pt, new_vi, "{start_f} {}", out.f.len());
            let next_og_vi = corner_verts.iter().find(|v| v.1 == end_pt).unwrap().0;
            let [e0_key, e1_key] = [og_vi, next_og_vi].map(|vi| mesh.v[vi].map(F::to_bits));

            // add to the edge map here, to ensure that even if the edge doesn't have any
            // intermediate vertices it will be labeled.
            assert_ne!(e0_key, e1_key);
            let fvs = edge_map.entry(minmax(e0_key, e1_key)).or_default();
            if fvs.iter_mut().find(|fv| fv.0 == fi).is_none() {
                fvs.push((fi, vec![]));
            }
            true
        };

        // important to go both ways since sometimes vert_adj is not oriented correctly.
        if !fix_up_side(l) || !fix_up_side(r) {
            return false;
        }
    }
    let check = labels
        .range(start..)
        .all(|(_, v)| v[0].1 != INVALID_POS && v[1].1 != INVALID_POS);
    assert!(check);

    for (&new_vi, ogs) in labels.range(start..) {
        if args.debug_colors && out.vert_colors[new_vi] != [1.; 3] {
            out.vert_colors[new_vi] = [1., 0., 0.];
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

pub fn sample_direct(
    mesh: &Mesh,
    f: &FaceKind,
    fi: usize,
    src: &SourceTextures,
    out: &mut Mesh,
    face_labels: &mut Vec<FaceLabel>,

    // map from original vertex -> (original face idx, vertex)
    corner_map: &mut Map<[U; 3], Vec<(usize, usize)>>,
    // map from edge -> (original face idx, vertices on face)
    edge_map: &mut Map<[[U; 3]; 2], Vec<(usize, Vec<usize>)>>,
    // map from new vertex index to original vertex position and distance
    //labels: &mut BTreeMap<usize, [(u32, [U; 3]); 2]>,
) -> bool {
    let new_face = f.map_kind(|og_vi| {
        let og_pos = mesh.v[og_vi];
        let fv = corner_map.entry(og_pos.map(F::to_bits)).or_default();
        if let Some(prev_fv) = fv.iter().find(|fv| fv.0 == fi) {
            return prev_fv.1;
        }
        let [u, v] = mesh.uv[CHAN][og_vi];

        src.push_to_mesh(out, u, v);
        let new_vi = out.v.len();
        out.v.push(og_pos);

        fv.push((fi, new_vi));
        new_vi
    });

    for og_e in f.edges() {
        let [e0_key, e1_key] = og_e.map(|e| mesh.v[e].map(F::to_bits));
        let fvs = edge_map.entry(minmax(e0_key, e1_key)).or_default();
        if fvs.iter_mut().find(|fv| fv.0 == fi).is_none() {
            fvs.push((fi, vec![]));
        }
    }

    face_labels.push(FaceLabel::Pixel);
    out.f.push(new_face);
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

pub fn del_degen_bridges(
    remap: &mut UnionFind<u32>,
    mesh: &mut pars3d::Mesh,
    locked: impl Fn(usize) -> bool,
    args: &Args,
    labels: &[FaceLabel],
    face_range: Range<usize>,
) -> usize {
    let mut del = 0;
    let cd_thresh = args.color_diff_threshold;
    for fi in face_range.clone() {
        let f = &mesh.f[fi];
        if f.len() != 4 {
            continue;
        }
        let FaceLabel::Bridge(fi0, fi1) = labels[fi] else {
            continue;
        };
        debug_assert_eq!(labels[fi0], FaceLabel::Pixel);
        debug_assert_eq!(labels[fi1], FaceLabel::Pixel);

        let f0 = &mesh.f[fi0];
        let f1 = &mesh.f[fi1];
        let [e00, e01] = f.shared_edge(f0).unwrap();
        let [e10, e11] = f.shared_edge(f1).unwrap();

        let all = [e00, e01, e10, e11];
        if all.iter().any(|&v| locked(v)) {
            continue;
        }
        /*
        if all.iter().any(|&v| !remap.is_root(v)) {
            continue;
        }
        */
        // TODO how to check for degenerate faces here?
        /*
        let [e00, e01, e10, e11] = all.map(|v| remap.get_compress(v));
        // Is this good enough to spot degenerate faces
        if e00 == e11 || e01 == e10 {
            continue;
        }
        */

        // e00 - e11 is a paired edge
        // e01 - e10 is a paired edge

        if dist(mesh.vert_colors[e00], mesh.vert_colors[e11]) > cd_thresh {
            continue;
        }
        if dist(mesh.vert_colors[e01], mesh.vert_colors[e10]) > cd_thresh {
            continue;
        }

        // Commit to collapsing this face
        let mut combine = |a: usize, b: usize| {
            let a = remap.get_compress(a);
            let b = remap.get_compress(b);
            let new_v = kmul(0.5, add(mesh.v[a], mesh.v[b]));
            let new_vc = kmul(0.5, add(mesh.vert_colors[a], mesh.vert_colors[b]));

            mesh.v[a] = new_v;
            mesh.vert_colors[a] = new_vc;
            remap.set(b, a);
        };

        combine(e00, e11);
        combine(e10, e01);

        mesh.f[fi] = FaceKind::empty();
        del += 1;
    }

    // BridgeCorner faces
    for fi in face_range {
        let f = &mesh.f[fi];
        if labels[fi] != FaceLabel::BridgeCorner {
            continue;
        }
        if f.is_empty() {
            continue;
        }
        let mean_color = f.centroid(&mesh.vert_colors);
        let all_near = f
            .as_slice()
            .iter()
            .all(|&vi| dist(mesh.vert_colors[vi], mean_color) < cd_thresh);
        if !all_near {
            continue;
        }

        let mean_pos = f.centroid(&mesh.v);

        let root = f.as_slice()[0];
        mesh.v[root] = mean_pos;
        mesh.vert_colors[root] = mean_color;
        for &vi in &f.as_slice()[1..] {
            remap.set(vi, root);
        }
        mesh.f[fi] = FaceKind::empty();
        del += 1;
    }

    del
}

/// Deletes degenerate faces by merging all vertices of the face together
pub fn del_degen_gap_fill(
    remap: &mut UnionFind<u32>,
    mesh: &mut Mesh,
    args: &Args,
    labels: &[FaceLabel],
) -> usize {
    let mut del = 0;

    // vertex -> face_adj
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (fi, f) in mesh.f.iter().enumerate() {
        for e in f.edges() {
            for vi in e {
                let vi = remap.get_compress(vi);
                let adj_fs = adj.entry(vi).or_default();
                if !adj_fs.contains(&fi) {
                    adj_fs.push(fi);
                }
            }
        }
    }

    fn num_shared(a_s: &[usize], b_s: &[usize]) -> usize {
        let mut total = 0;
        for a in a_s {
            for b in b_s {
                if a == b {
                    total += 1;
                }
            }
        }
        total
    }

    'outer: for fi in 0..mesh.f.len() {
        let FaceLabel::GapFill([e00, e01], [e10, e11]) = labels[fi] else {
            continue;
        };
        let all = [e00, e01, e10, e11];
        if !all.iter().all(|&v| remap.is_root(v)) {
            break;
        }
        mesh.f[fi].remap(|vi| remap.get_compress(vi));
        if !mesh.f[fi].canonicalize() {
            mesh.f[fi] = FaceKind::empty();
            continue;
        }
        if mesh.f[fi].len() != 4 {
            continue;
        }
        let get_mapped = |v: usize| {
            if v == usize::MAX {
                v
            } else {
                remap.get_compress(v)
            }
        };
        let [e00, e01, e10, e11] = [e00, e01, e10, e11].map(get_mapped);

        macro_rules! is_valid {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {{
                let f = &mesh.f[fi];
                if $a == $b || $c == $d || $a == $c || $a == $d || $b == $c || $b == $d || $c == $d
                {
                    continue;
                }
                for exp in [minmax($a, $b), minmax($c, $d)] {
                    let check = f
                        .edges()
                        .map(|e| e.map(|vi| remap.get_compress(vi)))
                        .map(|[e0, e1]| minmax(e0, e1))
                        .any(|e| e == exp);
                    if check {
                        continue 'outer;
                    }
                }
                let [a, b, c, d] = [$a, $b, $c, $d].map(|vi| mesh.vert_colors[vi]);
                if dist(a, c) > args.color_diff_threshold {
                    continue;
                }
                if dist(b, d) > args.color_diff_threshold {
                    continue;
                }
                let [a, b, c, d] = [$a, $b, $c, $d].map(|vi| mesh.v[vi]);
                if dist(a, c) > 5e-3 {
                    continue;
                }
                if dist(b, d) > 5e-3 {
                    continue;
                }
                if num_shared(&adj[&$a], &adj[&$c]) > 2 || num_shared(&adj[&$b], &adj[&$d]) > 2 {
                    continue;
                }
                [$a, $b, $c, $d]
            }};
        }

        let [a, b, c, d] = match [e01, e11] {
            [usize::MAX, usize::MAX] => unreachable!(),

            [e01, usize::MAX] => is_valid!(e00, e01, e10, e10),
            [usize::MAX, e11] => is_valid!(e00, e00, e10, e11),

            [e01, e11] => is_valid!(e00, e01, e10, e11),
        };
        // Commit to collapsing this face
        let mut combine = |a: usize, b: usize| {
            let a = remap.get_compress(a);
            let b = remap.get_compress(b);
            let new_v = kmul(0.5, add(mesh.v[a], mesh.v[b]));
            let new_vc = kmul(0.5, add(mesh.vert_colors[a], mesh.vert_colors[b]));
            let new_vi = a;

            mesh.v[a] = new_v;
            mesh.vert_colors[a] = new_vc;
            remap.set(b, new_vi);
            if let Some(b_adj) = adj.remove(&b) {
                let adj_fs_a = adj.entry(a).or_default();
                for b in b_adj {
                    if !adj_fs_a.contains(&b) {
                        adj_fs_a.push(b);
                    }
                }
                adj_fs_a.retain(|&fi| {
                    let f = &mut mesh.f[fi];
                    f.remap(|vi| remap.get_compress(vi));
                    !f.canonicalize()
                });
            }
        };
        combine(a, c);
        combine(b, d);

        mesh.f[fi] = FaceKind::empty();
        del += 1;
    }

    del
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaceLabel {
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

    /// A face which was removed because it didn't have any neighbors when it should've.
    Deleted,
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

/// How to sample the input mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SampleKind {
    /// Compute the dual of exact (one vertex per pixel)
    Approx,

    /// 4 vertices (1 quad) per pixel, with a large number of scaffolding faces.
    Exact,

    /// Use the original vertices directly. There will still be additional faces introduced at
    /// UV chart boundaries.
    Direct,
}

impl_display!(SampleKind, Approx => "approx", Exact => "exact", Direct => "direct");
