use clap::Parser;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    #[arg(long, short)]
    input: String,

    /// The mesh to copy to the UV of the input
    #[arg(long, short, default_value_t = String::new())]
    uv_mesh: String,

    #[arg(long, short)]
    output: String,

    /// Channel to store UV in
    #[arg(long, short, default_value_t = 0)]
    channel: usize,

    /// Remove the Y channel instead of Z
    #[arg(long)]
    y_zero: bool,

    /// Unused
    #[arg(long, default_value_t = String::new())]
    stats: String,
}

fn main() -> std::io::Result<()> {
    let mut args = Args::parse();
    if args.uv_mesh.is_empty() {
      args.uv_mesh.clone_from(&args.input);
    }
    let args = args;

    let mut scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();

    let uv_scene =
        pars3d::load(&args.uv_mesh).expect(&format!("Failed to parse input from {}", args.input));
    let uv_m = uv_scene.into_flattened_mesh();
    assert_eq!(m.v.len(), uv_m.v.len());
    m.uv[args.channel].clear();
    for &[x,y,z] in &uv_m.v {
        let [u,v] = if args.y_zero {
          assert_eq!(y, 0., "{x} {y} {z}");
          [x,z]
        } else {
          assert_eq!(z, 0., "{x} {y} {z}");
          [x,y]
        };
        m.uv[args.channel].push([(u + 1.) / 2., (v + 1.) / 2.]);
    }

    m.repopulate_scene(&mut scene);
    pars3d::save(&args.output, &scene, false)
}
