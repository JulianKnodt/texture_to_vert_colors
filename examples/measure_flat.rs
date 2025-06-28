#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use clap::Parser;

use texture_to_vert_colors::F;
use texture_to_vert_colors::clustering::Eigenvalue;

#[derive(Clone, Parser, Debug)]
pub struct Args {
    /// Input OBJ file.
    #[arg(short, long, required = true)]
    pub input: String,

    /// Output PLY file, where clusters are colored by their clustering instead of average color
    #[arg(short, long, default_value_t = String::new())]
    pub output: String,

    /// Output PLY file, to visualize the optimized eigenvalue of each cluster normalized to the
    /// range [0,1]
    #[arg(long, default_value_t = String::new())]
    pub eigen_vis: String,

    /// Which eigenvalue to visualize
    #[arg(long, default_value_t = Eigenvalue::Zero)]
    eigenvalue: Eigenvalue,

    /// Do not use area weighting.
    #[arg(long)]
    no_area_weight: bool,

    /// Output stats to this file
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// Eigen to regularize by when outputing eigenvalues. If not set will use the max.
    #[arg(long, default_value_t = -1.)]
    max_eigen: F,

    /// Do not emit a wireframe
    #[arg(long)]
    no_wireframe: bool,
}

pub fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if !args.output.ends_with(".ply") {
        eprintln!("[WARN]: Output will not be colored if output format is not PLY");
    }

    let scene = pars3d::load(&args.input).expect(&format!("Failed to load input {}", &args.input));
    let mut mesh = scene.into_flattened_mesh();
    mesh.clear_vertex_normals();

    let ff_adj = mesh.face_face_adj();
    let (face_charts, num_charts) = ff_adj.connected_components();
    eprintln!("[INFO]: # Charts = {num_charts}");

    use texture_to_vert_colors::measure_flat as mf;
    mf::measure_flat(
        &mut mesh,
        |fi| face_charts[fi] as usize,
        num_charts as usize,
        &mf::Args {
            eigen_vis: args.eigen_vis,
            eigenvalue: args.eigenvalue,
            no_wireframe: args.no_wireframe,
            max_eigen: args.max_eigen,
            stats: args.stats,
            cluster_vis: args.output,
        },
    )
}
