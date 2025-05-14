#![allow(unused)]
use clap::Parser;
use pars3d::visualization::{colored_wireframe, wireframe_to_mesh};
use std::collections::HashSet;
use texture_to_vert_colors::{F, add, kmul, sub};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh with per vertex offsets stored in the R channel of the vertex colors
    #[arg(long, short)]
    input: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,

    /// How wide are the output wires?
    #[arg(long, default_value_t = 1e-3)]
    vis_width: F,
}

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    let (s, t) = m.normalize();

    let edges = m.edges().map(|(e, _)| e).collect::<HashSet<_>>();

    let mut new_mesh = wireframe_to_mesh(colored_wireframe(
        edges.into_iter(),
        |vi| m.v[vi],
        |[e0, e1]| {
            let delta = sub(m.vert_colors[e1], m.vert_colors[e0]);
            let sharp = delta.map(|v| v.abs().sqrt());
            sharp
            //kmul(0.5, add(sharp, [1.; 3]))
        },
        1e-3,
    ));

    new_mesh.denormalize(s, t);
    let s = new_mesh.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for visualization", start.elapsed());
}
