use clap::Parser;
use texture_to_vert_colors::{add, kmul};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh with per vertex offsets stored in the R channel of the vertex colors
    #[arg(long, short)]
    input: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,
}

fn main() {
    let args = Args::parse();
    let mut scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));

    let start = std::time::Instant::now();
    for m in scene.meshes.iter_mut() {
        if m.n.len() != m.v.len() {
            pars3d::Mesh::vertex_normals(&m.f, &m.v, &mut m.n, Default::default());
        }
        for vi in 0..m.v.len() {
            let offset = m.vert_colors[vi][0];
            m.v[vi] = add(m.v[vi], kmul(offset, m.n[vi]));
        }
    }

    println!("Elapsed {:?}", start.elapsed());
    pars3d::save(&args.output, &scene).expect(&format!("Failed to save output to {}", args.output));
}
