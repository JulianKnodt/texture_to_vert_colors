#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]

use clap::Parser;
use texture_to_vert_colors::{F, add, dist, kmul, length, normalize};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh
    #[arg(long, short)]
    input: String,

    /// Output mesh
    #[arg(long, short)]
    output: String,

    /// How large to make the output image
    #[arg(long, default_value_t = 2048)]
    output_tex_size: u32,

    /// Unused currently
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// Which image to use for strokes
    #[arg(long)]
    stroke_image: String,
}

fn main() {
    let args = Args::parse();

    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    m.f.retain_mut(|f| !f.canonicalize());
    let (s, t) = m.normalize();

    use texture_to_vert_colors::vector_field::{dir_field_relaxation, texture_grad_field};
    let mut per_face_grad = texture_grad_field(&m, true);
    let ff_adj = m.face_face_pos_adj();
    let ff_adj = ff_adj.map(|_, fi0, fi1, ()| {
        m.f[fi0]
            .shared_edges(&m.f[fi1])
            .map(|[ei0, ei1]| dist(m.v[ei0], m.v[ei1]))
            .sum::<F>()
    });
    dir_field_relaxation(&mut per_face_grad, ff_adj, |fi| m.f[fi].normal(&m.v), 50);

    let mut new_mesh = pars3d::Mesh::default();

    let _stroke_img = imageproc::image::open(&args.stroke_image).expect("Failed to load image");
    let mut output_diffuse =
        imageproc::image::RgbaImage::new(args.output_tex_size, args.output_tex_size);

    use indicatif::ProgressIterator;
    for (fi, f) in m.f.iter().enumerate().progress() {
        if length(per_face_grad[fi]) < 1e-3 {
            continue;
        }
        let mean_color = f.centroid(&m.vert_colors);
        let centroid = f.centroid(&m.v);

        let dir = normalize(per_face_grad[fi]);
        let uv_f = f.map_kind(|vi| m.uv[0][vi]);
        let v_f = f.map_kind(|vi| m.v[vi]);

        let barycentric = v_f.barycentric(centroid);
        let tri = barycentric.tri(f).map(|vi| m.v[vi]);
        let bary_dir = pars3d::dir_to_barycentric(dir, tri);
        let uv0 = uv_f.from_barycentric(barycentric);
        let uv1 = add(uv0, kmul(2e-3, bary_dir)).map(|f| f * args.output_tex_size as F);
        let uv0 = uv0.map(|f| f * args.output_tex_size as F);
        let [r, g, b] = mean_color.map(|c| (c * 255.) as u8);
        let [uv0x, uv0y] = uv0.map(|vi| vi as i32);
        let [uv1x, uv1y] = uv1.map(|vi| vi as i32);
        imageproc::drawing::draw_antialiased_line_segment_mut(
            &mut output_diffuse,
            (uv0x, uv0y),
            (uv1x, uv1y),
            imageproc::image::Rgba([r, g, b, 255]),
            imageproc::pixelops::interpolate,
        );

        /*
        let num_v = nv.len();
        let mut wireframe = pars3d::visualization::wireframe_to_mesh((
            nv.to_vec(),
            vec![mean_color; num_v],
            nf.to_vec(),
        ));
        new_mesh.append(&mut wireframe)
        */
    }
    println!(
        "[INFO]: Output #F = {}, Output #V = {}",
        new_mesh.f.len(),
        new_mesh.v.len()
    );

    let _ = output_diffuse.save("output_diffuse.png");

    new_mesh.denormalize(s, t);
    let s = new_mesh.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for hatching", start.elapsed());
}

pub fn uv_tri_per_tex(m: &pars3d::Mesh, res: usize) -> Vec<Vec<Vec<usize>>> {
    let mut out = vec![vec![vec![]; res]; res];

    for (fi, f) in m.f.iter().enumerate() {
        let aabb = f.aabb(&m.uv[0]).round_to_i32();
        for [i, j] in aabb.iter_coords() {
            out[i as usize][j as usize].push(fi);
        }
    }

    out
}
