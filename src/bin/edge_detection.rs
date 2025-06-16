#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]
#![feature(let_chains)]
#![feature(more_float_constants)]

use std::collections::BTreeSet;

use clap::Parser;
use pars3d::{self, FaceKind, Mesh};

use texture_to_vert_colors::quadric::{AttrWeights, Quadric};
use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, dist, dot, kmul, length, normalize, sub};

const FRAC_1_SQRT_2_PI: F = std::f64::consts::FRAC_1_SQRT_2PI as F;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh path.
    #[arg(long, short)]
    input: String,

    /// Output mesh path.
    #[arg(long, short)]
    output: String,

    /// How the weight between vertices is computed
    #[arg(long, short, default_value_t = WeightingKind::Laplacian)]
    weighting: WeightingKind,

    /// The sigma for the gaussian smoothing step
    #[arg(long, short, default_value_t = 0.1)]
    gaussian_sigma: F,

    /// How many iterations to apply smoothing for
    #[arg(long, short, default_value_t = 2)]
    smoothing_iters: u32,

    /// Return immediately after smoothing (DEBUGGING)
    #[arg(long)]
    return_smoothed: bool,

    /*
    /// How to combine position and color when computing norms.
    #[arg(long, default_value_t = PosColorNorm::Add)]
    pos_color_norm: PosColorNorm,

    /// How much to weigh color compared to geometry
    #[arg(long, default_value_t = 0.)]
    color_weight: F,
    */
    /// Unused for now
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// Minimum value to use for canny edge detection gradient
    #[arg(long, short = 'l', default_value_t = 2e-4)]
    min_val: F,

    /// Maximum value to use for canny edge detection gradient
    #[arg(long, short = 'l', default_value_t = 3e-4)]
    max_val: F,
}

fn main() {
    let args = Args::parse();

    let mut scene = pars3d::load(&args.input).expect("Failed to parse input");

    let start = std::time::Instant::now();
    for m in scene.meshes.iter_mut() {
        // if the input mesh is not triangular, copy the input faces then reapply them later
        let og_faces = if m.f.len() != m.num_tris() {
            Some(m.f.clone())
        } else {
            None
        };
        let mut new_edges = BTreeSet::new();
        m.triangulate_with_new_edges(|[e0, e1]| {
            new_edges.insert(std::cmp::minmax(e0, e1));
        });
        let (s, t) = m.normalize();
        //m.normalize_colors();

        edge_detection(m, new_edges, &args);

        m.denormalize(s, t);
        if let Some(og_faces) = og_faces {
            m.f = og_faces;
        }
    }
    println!(
        "[INFO]: Took {:?} for edge detection for {}",
        start.elapsed(),
        args.input,
    );
    pars3d::save(&args.output, &scene).expect("Failed to save output");
    println!("[INFO]: Saved to {}", args.output);
}

pub fn edge_detection(mesh: &mut Mesh, new_edges: BTreeSet<[usize; 2]>, args: &Args) {
    let mut vert_adj = args
        .weighting
        .vertex_weights(&mesh, PosColorNorm::Add, 0.) /*args.pos_color_norm, args.color_weight)*/
        .expect("Failed to construct vertex adjacency");
    // remove influence of introduced edges with length or uniform weighting
    if matches!(
        args.weighting,
        WeightingKind::Length | WeightingKind::Uniform
    ) {
        for vi in 0..mesh.v.len() {
            let (adj_vs, adj_ds) = vert_adj.adj_data_mut(vi);
            for i in 0..adj_vs.len() {
                let e = std::cmp::minmax(vi, adj_vs[i] as usize);
                if new_edges.contains(&e) {
                    adj_ds[i] = 0.;
                }
            }
        }
    }

    // compute gaussian smoothing
    for _ in 0..args.smoothing_iters {
        let smoothed_vert_colors = mesh
            .vert_colors
            .iter()
            .enumerate()
            .map(|(vi, vc)| {
                let mut total_w = FRAC_1_SQRT_2_PI / args.gaussian_sigma; //dist is 0 here
                let mut new_color = kmul(total_w, *vc);
                for (adj, _w) in vert_adj.adj_data(vi) {
                    let adj = adj as usize;
                    let d = dist(mesh.v[vi], mesh.v[adj]);
                    let w = gaussian_dist(d, args.gaussian_sigma);
                    total_w += w;
                    new_color = add(new_color, kmul(w, mesh.vert_colors[adj]));
                }
                assert!(total_w.is_finite(), "{}", args.gaussian_sigma);
                kmul(total_w.recip(), new_color).map(|v| v.clamp(0., 1.))
            })
            .collect::<Vec<_>>();
        mesh.vert_colors = smoothed_vert_colors;
    }

    if args.return_smoothed {
        return;
    }

    let luminance = mesh
        .vert_colors
        .iter()
        .copied()
        .map(luma)
        .collect::<Vec<_>>();

    // gradient determination
    let mut face_gradients = vec![[0.; 3]; mesh.f.len()];
    // gradient for each face (gradient at each vertex is the sum of gradient of adjacent faces)
    for (fi, f) in mesh.f.iter().enumerate() {
        let area = f.area(&mesh.v);
        if area < 1e-20 {
            continue;
        }
        let n = f.normal(&mesh.v);
        macro_rules! q_n_attribs(
          ($vis: expr) => {{
            Quadric::n_attribs(
                n,
                $vis.map(|vi| mesh.v[vi]),
                $vis.map(|vi| [luminance[vi]]),
                AttrWeights { ws: [1.] },
            )
          }}
        );

        // add attributes as well
        let q_attr = match f {
            FaceKind::Tri(vis) => q_n_attribs!(vis),
            FaceKind::Quad(vis) => q_n_attribs!(vis),
            FaceKind::Poly(p) => Quadric::dyn_attribs(
                n,
                p.len(),
                |vi| mesh.v[vi],
                |vi| [luminance[vi]],
                AttrWeights { ws: [1.] },
            ),
        };

        // Should this be area weighted?
        face_gradients[fi] = kmul(area, q_attr.g[0]);
        //face_gradients[fi] = kmul(area.sqrt(), q_attr.g[0]);
        //face_gradients[fi] = q_attr.g[0];
    }

    let mut vertex_gradients = vec![[0.; 3]; mesh.v.len()];
    for (fi, f) in mesh.f.iter().enumerate() {
        for &vi in f.as_slice() {
            vertex_gradients[vi] = add(vertex_gradients[vi], face_gradients[fi]);
        }
    }

    // gradient suppresion
    let mut is_edge = vec![true; mesh.v.len()];
    for (vi, &v) in mesh.v.iter().enumerate() {
        let mut max_w = 0.;
        let mut max_dir = [0.; 3];
        let mut min_w = 0.;
        let mut min_dir = [0.; 3];

        let g = normalize(vertex_gradients[vi]);
        let strength = length(vertex_gradients[vi]);

        let cos45deg = std::f64::consts::SQRT_2 as F / 2.;

        for &adj in vert_adj.adj(vi) {
            let adj = adj as usize;
            let align = dot(normalize(sub(mesh.v[adj], v)), g);
            let w = align.abs();
            if w < cos45deg {
                continue;
            }
            let w_grad = kmul(w, vertex_gradients[adj]);
            if align.is_sign_positive() {
                max_w += w;
                max_dir = add(max_dir, w_grad);
            } else if align.is_sign_negative() {
                min_w += w;
                min_dir = add(min_dir, w_grad);
            }
        }
        // is the gradient at this vertex greater than its two neighbors
        is_edge[vi] = (max_w <= 1e-12 || strength >= length(max_dir) / max_w)
            && (min_w <= 1e-12 || strength >= length(min_dir) / min_w);
    }

    let range = vertex_gradients
        .iter()
        .copied()
        .map(length)
        .fold([F::INFINITY, F::NEG_INFINITY], |[l, h], n| {
            [l.min(n), h.max(n)]
        });
    println!("Range of vertex gradients is {range:?}");

    // double thresholding
    let mut is_strong = vec![false; mesh.v.len()];
    for (vi, &vg) in vertex_gradients.iter().enumerate() {
        if !is_edge[vi] {
            continue;
        }
        let strength = length(vg);
        if strength < args.min_val {
            is_edge[vi] = false;
            continue;
        }
        is_strong[vi] = strength > args.max_val;
    }

    let mut strong_verts = (0..mesh.v.len())
        .filter(|&vi| is_strong[vi])
        .collect::<Vec<_>>();
    while let Some(n) = strong_verts.pop() {
        for &adj in vert_adj.adj(n) {
            let adj = adj as usize;
            let is_weak = is_edge[adj] && !is_strong[adj];
            if !is_weak {
                continue;
            }
            is_strong[adj] = true;
            strong_verts.push(adj);
        }
    }

    for (vi, vc) in mesh.vert_colors.iter_mut().enumerate() {
        *vc = if is_strong[vi] { [1.; 3] } else { [0.; 3] };
    }
}

fn gaussian_dist(dist: F, sigma: F) -> F {
    (FRAC_1_SQRT_2_PI / sigma) * (-0.5 * dist * dist / (sigma * sigma)).exp()
}

fn luma(rgb: [F; 3]) -> F {
    // let lum_chan = [0.2126, 0.7152, 0.0722];
    let lum_chan = [0.299, 0.587, 0.114];
    dot(lum_chan, rgb)
}
