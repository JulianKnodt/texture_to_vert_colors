#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use clap::Parser;
use pars3d::F;
use pars3d::geom_processing::subdivision;
use std::io::Write;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    #[arg(long, short)]
    input: String,

    #[arg(long, short)]
    output: String,

    /// How many levels of subdivision to perform before sampling
    #[arg(long, short, default_value_t = 0)]
    subdivision_levels: usize,

    /// Explicit path to image to use for sampling.
    #[arg(long, short, default_value_t = String::new())]
    diffuse_image: String,

    /// Stores information about the output mesh at this path
    #[arg(long, default_value_t = String::new())]
    stats: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    let img = if !args.diffuse_image.is_empty() {
        pars3d::image::open(&args.diffuse_image).expect("Failed to open given image path")
    } else {
        let Some(mat) = m.single_mat() else {
            eprintln!(
                "Currently only single mat supported, got {:?}",
                m.face_mat_idx
            );
            return Ok(());
        };
        let Some(mat) = scene.materials.get(mat) else {
            eprintln!("Referenced material not found [Internal Error, please report]");
            return Ok(());
        };
        assert!(!mat.textures.is_empty());
        let tex = &scene.textures[mat.textures[0]];
        let Some(img) = tex.image.as_ref() else {
            eprintln!("There is no texture image");
            return Ok(());
        };
        img.clone()
    };

    m.triangulate();
    let in_num_v = m.v.len();
    let in_num_t = m.f.len();
    let mut tris =
        m.f.drain(..)
            .map(|f| f.as_tri().unwrap())
            .collect::<Vec<_>>();
    let mut barys = vec![];

    for _ in 0..args.subdivision_levels {
        let (bary, new_tri) = subdivision::loop_subdivision(&tris);
        barys.push(bary);
        tris = new_tri;
    }

    m.f.extend(tris.into_iter().map(pars3d::FaceKind::Tri));

    if !barys.is_empty() {
        let new_bary = barys
            .into_iter()
            .reduce(|p, n| subdivision::compose_barycentric_repr(&n, &p).collect::<Vec<_>>())
            .unwrap();
        let new_pos = new_bary.iter().map(|b| b.eval(&m.v)).collect();
        let new_uv = new_bary.iter().map(|b| b.eval(&m.uv[0])).collect();
        m.v = new_pos;
        m.uv[0] = new_uv;
    }

    assert_eq!(m.v.len(), m.uv[0].len());

    for &uv in &m.uv[0] {
        let [u, v] = uv.map(|mut c| {
            while c < 0. {
                c += 1.;
            }
            c % 1.
        });
        let v = 1. - v;
        let rgba = pars3d::image::imageops::sample_bilinear(&img, u as f32, v as f32);
        let Some(pars3d::image::Rgba(rgba)) = rgba else {
            m.vert_colors.push([0.; 3]);
            continue;
        };
        let [r, g, b, _a] = rgba.map(|v| v as F / 255.);
        m.vert_colors.push([r, g, b]);
    }

    m.n.clear();
    let out_num_t = m.f.len();
    let out_num_v = m.v.len();
    let new_scene = m.into_scene();
    pars3d::save(&args.output, &new_scene).expect("Failed to save scene");

    if args.stats.is_empty() {
        return Ok(());
    }

    use std::fs::File;
    let mut s = File::create(args.stats)?;
    writeln!(
        s,
        r#"{{
  "input_num_tris": {in_num_t},
  "input_num_vertices": {in_num_v},
  "output_num_tris": {out_num_t},
  "output_num_vertices": {out_num_v}
}}"#
    )
}
