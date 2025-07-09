#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]
#![feature(more_float_constants)]

use clap::Parser;
use pars3d::{self, FaceKind, Mesh};

use texture_to_vert_colors::quadric::{AttrWeights, Quadric};
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

    /// The sigma for the gaussian smoothing step
    #[arg(long, short, default_value_t = 0.1)]
    gaussian_sigma: F,

    /// How many iterations to apply smoothing for
    #[arg(long, short, default_value_t = 2)]
    smoothing_iters: u32,

    /// Return immediately after smoothing (DEBUGGING)
    #[arg(long, default_value_t = ReturnKind::None)]
    return_kind: ReturnKind,

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

    /// Cone to consider when measuring if there is a color gradient (in degrees)
    #[arg(long, default_value_t = 30.)]
    cone_angle_degrees: F,

    /// Apply a uniform normalization step as preprocessing.
    #[arg(long)]
    no_normalize_colors: bool,

    /// Perform edge detection on faces instead of on vertices.
    #[arg(long)]
    face: bool,

    /// Include area weighting
    #[arg(long)]
    no_area_weight: bool,

    /// Show colors for debugging (red is weak rejected, green is weak accept)
    #[arg(long)]
    debug_colors: bool,

    /// Cull triangles below a certain area (no gradient)
    #[arg(long, default_value_t = 1e-8)]
    cull_area_below: F,
}

fn main() {
    let args = Args::parse();

    let mut scene = pars3d::load(&args.input).expect("Failed to parse input");

    let start = std::time::Instant::now();
    for m in scene.meshes.iter_mut() {
        // if the input mesh is not triangular, copy the input faces then reapply them later
        let og_faces = if m.f.len() != m.num_tris() && !args.face {
            Some(m.f.clone())
        } else {
            None
        };
        m.triangulate();
        let (s, t) = m.normalize();
        if !args.no_normalize_colors {
            m.normalize_colors();
        }

        edge_detection(m, &args);

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

pub fn edge_detection(mesh: &mut Mesh, args: &Args) {
    let vert_adj = mesh.vertex_vertex_adj();

    // compute gaussian smoothing
    for _ in 0..args.smoothing_iters {
        let smoothed_vert_colors = mesh
            .vert_colors
            .iter()
            .enumerate()
            .map(|(vi, vc)| {
                let mut total_w = FRAC_1_SQRT_2_PI / args.gaussian_sigma; //dist is 0 here
                let mut new_color = kmul(total_w, *vc);
                for &adj in vert_adj.adj(vi) {
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

    if args.return_kind == ReturnKind::Smoothed {
        return;
    }

    let adj = if args.face {
        mesh.face_face_adj()
    } else {
        vert_adj
    };

    let luminance = if args.face {
        mesh.f
            .iter()
            .map(|f| f.centroid(&mesh.vert_colors))
            .map(luma)
            .collect::<Vec<_>>()
    } else {
        mesh.vert_colors
            .iter()
            .copied()
            .map(luma)
            .collect::<Vec<_>>()
    };

    // gradient determination
    let mut face_gradients = vec![[0.; 3]; mesh.f.len()];
    // gradient for each face (gradient at each vertex is the sum of gradient of adjacent faces)
    for (fi, f) in mesh.f.iter().enumerate() {
        let area = f.area(&mesh.v);
        if area < args.cull_area_below {
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

        face_gradients[fi] = if args.no_area_weight {
            q_attr.g[0]
        } else {
            kmul(area, q_attr.g[0])
        }
    }

    let gradients = if args.face {
        face_gradients
    } else {
        let mut vertex_gradients = vec![[0.; 3]; mesh.v.len()];
        for (fi, f) in mesh.f.iter().enumerate() {
            for &vi in f.as_slice() {
                vertex_gradients[vi] = add(vertex_gradients[vi], face_gradients[fi]);
            }
        }
        vertex_gradients
    };

    if args.return_kind == ReturnKind::Grad {
        let mut colors = gradients
            .into_iter()
            .map(length)
            .map(|l| [l; 3])
            .collect::<Vec<_>>();
        /*
        let max_len = colors
            .iter()
            .copied()
            .map(|l| l[0])
            .max_by(F::total_cmp)
            .unwrap();
        */
        let l = colors.len();
        let (_, &mut median_len, _) =
            colors.select_nth_unstable_by(l / 2, |a, b| a[0].total_cmp(&b[0]));
        println!("{median_len:?}");
        let ml_recip = median_len[0].recip();
        for vc in colors.iter_mut() {
            *vc = kmul(ml_recip, *vc);
        }
        if args.face {
            *mesh = mesh.with_face_coloring(&colors);
        } else {
            mesh.vert_colors = colors;
        }
        return;
    }

    let cone_angle = args.cone_angle_degrees.to_radians().cos();
    let get_src = |i: usize| {
        if args.face {
            mesh.f[i].centroid(&mesh.v)
        } else {
            mesh.v[i]
        }
    };

    let src_len = if args.face {
        mesh.f.len()
    } else {
        mesh.v.len()
    };
    // gradient suppresion
    let mut is_edge = vec![false; src_len];
    for i in 0..src_len {
        let mut max_w = 0.;
        let mut max_dir = [0.; 3];
        let mut min_w = 0.;
        let mut min_dir = [0.; 3];

        let src_p = get_src(i);

        let g = normalize(gradients[i]);
        let strength = length(gradients[i]);

        for &adj in adj.adj(i) {
            let adj = adj as usize;
            let align = dot(normalize(sub(get_src(adj), src_p)), g);
            let w = align.abs();
            if w < cone_angle {
                continue;
            }
            let w_grad = kmul(w, gradients[adj]);
            if align.is_sign_positive() {
                max_w += w;
                max_dir = add(max_dir, w_grad);
            } else if align.is_sign_negative() {
                min_w += w;
                min_dir = add(min_dir, w_grad);
            }
        }
        // is the gradient at this vertex greater than its two neighbors
        is_edge[i] = (max_w <= 1e-12 || strength >= length(max_dir) / max_w)
            && (min_w <= 1e-12 || strength >= length(min_dir) / min_w);
    }

    let range = gradients
        .iter()
        .copied()
        .map(length)
        .fold([F::INFINITY, F::NEG_INFINITY], |[l, h], n| {
            [l.min(n), h.max(n)]
        });
    println!("Range of gradients is {range:?}");

    // double thresholding
    let mut is_strong = vec![false; src_len];
    for (vi, &vg) in gradients.iter().enumerate() {
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

    let og_weaks = (0..src_len)
        .map(|vi| is_edge[vi] && !is_strong[vi])
        .collect::<Vec<_>>();
    let mut weaks = (0..src_len)
        .filter(|&vi| is_edge[vi] && !is_strong[vi])
        .collect::<Vec<_>>();
    while let Some(n) = weaks.pop() {
        if !is_edge[n] || is_strong[n] {
            continue;
        }
        let mut num_strong = 0;
        for &adj in adj.adj(n) {
            let adj = adj as usize;
            if !is_strong[adj] {
                continue;
            }
            //let d2 = texture_to_vert_colors::dist_sq(mesh.v[n], mesh.v[adj]);
            //num_strong += 1./ d2;
            num_strong += 1;
        }
        if num_strong > 0 {
            is_strong[n] = true;
            for &adj in adj.adj(n) {
                weaks.push(adj as usize);
            }
        }
    }

    let col = |vi| {
        if !args.debug_colors {
            [if is_strong[vi] { 1. } else { 0. }; 3]
        } else {
            match (is_strong[vi], og_weaks[vi]) {
                (true, false) => [1.; 3],
                (true, true) => [0., 1., 0.],
                (false, true) => [1., 0., 0.],
                (false, false) => [0.; 3],
            }
        }
    };
    if args.face {
        let new_face_colors = (0..is_strong.len()).map(col).collect::<Vec<_>>();
        *mesh = mesh.with_face_coloring(&new_face_colors);
    } else {
        for (vi, vc) in mesh.vert_colors.iter_mut().enumerate() {
            *vc = col(vi);
        }
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

macro_rules! impl_display {
  ($name: ident, $($kind: ident => $disp: expr),+$(,)?) => {
    impl std::fmt::Display for $name {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use $name::*;
            let s = match self {
                $($kind => $disp,)+
            };
            write!(f, "{s}")
        }
    }
  }
}

/// How to sample the input mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ReturnKind {
    None,
    Smoothed,
    Grad,
}

impl_display!(ReturnKind, None => "none", Smoothed => "smoothed", Grad => "grad");
