#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]

use clap::Parser;
use texture_to_vert_colors::{F, add, dist, length};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh
    #[arg(long, short)]
    input: String,

    /// Output mesh
    #[arg(long, short)]
    output: String,

    /// Unused currently
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// Compute the gradient field on vertices
    #[arg(long)]
    vertex: bool,
}

fn main() {
    let args = Args::parse();

    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    m.triangulate();
    m.f.retain_mut(|f| !f.canonicalize());
    let (s, t) = m.normalize();

    use texture_to_vert_colors::vector_field::{dir_field_relaxation, texture_grad_field};
    let mut per_face_grad = texture_grad_field(&m, true);

    let mut new_mesh = pars3d::Mesh::default();
    if args.vertex {
        let mut per_vertex_grad = vec![[0.; 3]; m.v.len()];
        for (fi, f) in m.f.iter().enumerate() {
            let grad = per_face_grad[fi];
            for &vi in f.as_slice() {
                per_vertex_grad[vi] = add(per_vertex_grad[vi], grad);
            }
        }

        frustum_per(
            &mut new_mesh,
            m.v.len(),
            |vi| m.v[vi],
            |vi| m.vert_colors[vi],
            &per_vertex_grad,
        );
    } else {
        let ff_adj = m.face_face_pos_adj();
        let ff_adj = ff_adj.map(|_, fi0, fi1, ()| {
            m.f[fi0]
                .shared_edges(&m.f[fi1])
                .map(|[ei0, ei1]| dist(m.v[ei0], m.v[ei1]))
                .sum::<F>()
        });
        dir_field_relaxation(&mut per_face_grad, ff_adj, 50);

        frustum_per(
            &mut new_mesh,
            m.f.len(),
            |fi| m.f[fi].centroid(&m.v),
            |fi| m.f[fi].centroid(&m.vert_colors),
            &per_face_grad,
        );
    }
    println!(
        "[INFO]: Output #F = {}, Output #V = {}",
        new_mesh.f.len(),
        new_mesh.v.len()
    );

    new_mesh.denormalize(s, t);
    let s = new_mesh.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for hatching", start.elapsed());
}

pub fn frustum_per(
    new_mesh: &mut pars3d::Mesh,
    n_elem: usize,
    pos: impl Fn(usize) -> [F; 3],
    color: impl Fn(usize) -> [F; 3],
    grads: &[[F; 3]],
) {
    for i in 0..n_elem {
        let grad = grads[i];
        if length(grad) < 1e-4 {
            continue;
        }
        let mean_color = color(i);
        let centroid = pos(i);
        let (nv, nf) = sdfs::to_mesh::frustum_to_quad_mesh(
            5,
            centroid,
            grad,
            1e-2,
            1e-3,
            0.,
            length(grad) + 5.,
        );
        let num_v = nv.len();
        let mut wireframe = pars3d::visualization::wireframe_to_mesh((
            nv.to_vec(),
            vec![mean_color; num_v],
            nf.to_vec(),
        ));
        new_mesh.append(&mut wireframe)
    }
}
