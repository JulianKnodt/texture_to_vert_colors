use std::io::{self, BufWriter, Write};

use clap::Parser;
use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, length};

/// Output a directional field
#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    #[arg(long, short)]
    input: String,

    /// CSV to output
    #[arg(long, short)]
    output: String,

    /// Kind of weighting to use for diffusion
    #[arg(long, default_value_t = WeightingKind::Laplacian)]
    diffusion_weighting: WeightingKind,

    /// Number of iterations for diffusion
    #[arg(long, short, default_value_t = 5)]
    diffusion_iters: usize,

    /// Cull all gradients below this threshold
    #[arg(long, default_value_t = 1e-3)]
    thresh: F,

    /// Do not area weight
    #[arg(long)]
    no_area_weight: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    m.triangulate(0);
    m.f.retain_mut(|f| !f.canonicalize());
    m.normalize();

    use texture_to_vert_colors::vector_field::{dir_field_relaxation, texture_grad_field};
    let per_face_grad = texture_grad_field(&m, !args.no_area_weight);

    let mut per_vertex_grad = vec![[0.; 3]; m.v.len()];
    for (fi, f) in m.f.iter().enumerate() {
        let grad = per_face_grad[fi];
        for &vi in f.as_slice() {
            per_vertex_grad[vi] = add(per_vertex_grad[vi], grad);
        }
    }

    let mut total = 0;
    for g in per_vertex_grad.iter_mut() {
        if length(*g) < args.thresh {
            *g = [0.; 3];
            continue;
        }
        //*g = normalize(*g);
        total += 1;
    }
    if total == 0 {
        eprintln!("No elements found, lower threshold");
        return Ok(());
    }
    println!("[INFO]: {total} elements used for relaxation");

    let vv_adj = args
        .diffusion_weighting
        .vertex_weights(&m, PosColorNorm::Add, 0.)
        .unwrap();

    if args.diffusion_iters > 0 {
        dir_field_relaxation(&mut per_vertex_grad, vv_adj, args.diffusion_iters);
    }

    /*
    for g in per_vertex_grad.iter_mut() {
        *g = normalize(*g);
    }
    */

    let f = std::fs::File::create(&args.output).expect("Failed to open output");
    let mut f = BufWriter::new(f);
    for [x, y, z] in per_vertex_grad {
        writeln!(f, "{x},{y},{z}")?;
    }

    Ok(())
}
