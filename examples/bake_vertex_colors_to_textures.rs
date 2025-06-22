use clap::Parser;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    #[arg(long, short)]
    input: String,

    #[arg(long, short)]
    output: String,

    /// Path to output image
    #[arg(long, short)]
    bake_texture: String,

    /// Resolution of output baked image
    #[arg(long, default_value_t = 1024)]
    bake_res: u32,

    /// Use approximate rebaking instead of exact
    #[arg(long)]
    approx_rebake: bool,

    /// UV channel to use.
    #[arg(long, default_value_t = 0)]
    uv_channel: usize,

    #[arg(long, default_value_t = String::new())]
    stats: String,
}

fn main() {
    let args = Args::parse();

    let mut scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let m = scene.into_flattened_mesh();

    assert!(!m.vert_colors.is_empty());
    let uvc = args.uv_channel;
    assert!(!m.uv[uvc].is_empty());
    use pars3d::coloring as col;
    let mut img = if args.approx_rebake {
        col::bake_vertex_colors_to_texture([args.bake_res; 2], &m.uv[uvc], &m.f, &m.vert_colors)
    } else {
        col::bake_vertex_colors_to_texture_exact(
            [args.bake_res; 2],
            &m.v,
            &m.uv[uvc],
            &m.f,
            &m.vert_colors,
        )
    };

    pars3d::image::imageops::flip_vertical_in_place(&mut img);
    let nf = m.f.len();
    let new_texture = pars3d::mesh::Texture {
        kind: pars3d::mesh::TextureKind::Diffuse,
        mul: [1.; 4],
        image: Some(img.into()),
        original_path: args.bake_texture,
    };
    let ti = scene.textures.len();
    scene.textures.push(new_texture);
    let new_mat = pars3d::mesh::Material {
        textures: vec![ti],
        name: format!("BakedVertexColors0"),
        path: args.output[..args.output.len() - 4].to_string() + ".mtl",
    };
    let mi = scene.materials.len();
    scene.materials.push(new_mat);
    scene.meshes[0].face_mat_idx = vec![((0..nf), mi)];
    pars3d::save(&args.output, &scene).expect("Failed to save output");
    println!(
        "[INFO]: Baked vertex colors to textures for {}",
        args.output
    )
}
