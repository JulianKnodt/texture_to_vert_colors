#![allow(unused)]
use clap::Parser;
use ordered_float::NotNan;
use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, dist, dot, kmul, sub};

use pars3d::adjacency::Adj;
use priority_queue::PriorityQueue;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh with per vertex offsets stored in the R channel of the vertex colors
    #[arg(long, short)]
    input: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,

    /// How to weigh which vertices get how much diffusion
    #[arg(long, short, default_value_t = WeightingKind::Uniform)]
    weighting: WeightingKind,

    /// How to evaluate color and position
    #[arg(long, default_value_t = PosColorNorm::Add)]
    pos_color_norm: PosColorNorm,

    #[arg(long, default_value_t = OrderKind::Nearest)]
    order: OrderKind,

    /// Use a different palette for dithering (Default = [0., 1.])
    #[arg(long, short)]
    palette: Vec<F>,

    /// Color weight for distances
    #[arg(long, default_value_t = 0.)]
    color_weight: F,

    /// Unused
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// Instead of performing dithering per vertex, perform dithering per face.
    #[arg(long)]
    face: bool,

    /// Maximum number of iterations to perform before stopping
    #[arg(long, default_value_t = 10000000)]
    max_iters: usize,

    /// How to diffuse errors to adjacent faces
    #[arg(long, default_value_t = ErrorDiffusionKind::Exact)]
    diffusion: ErrorDiffusionKind,

    /// Do not diffuse error if it is below this threshold.
    #[arg(long, default_value_t = 0.)]
    thresh: F,

    /// Diffuse only a fraction of the quantization error.
    #[arg(long, default_value_t = 0.5)]
    error_diffused: F,
}

fn main() {
    let mut args = Args::parse();
    if args.palette.is_empty() {
        args.palette = vec![0., 1.];
    }
    args.palette.sort_unstable_by(F::total_cmp);
    let args = args;
    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    assert_eq!(m.vert_colors.len(), m.v.len());
    let og_f = m.f.clone();
    if !args.face {
        m.triangulate();
    }
    let (s, t) = m.normalize();

    let mut target_chan = if args.face {
        m.f.iter()
            .map(|f| {
                let sum = f
                    .as_slice()
                    .iter()
                    .map(|&vi| m.vert_colors[vi])
                    .fold([0.; 3], add);
                luma(kmul((f.len() as F).recip(), sum))
            })
            .collect::<Vec<_>>()
    } else {
        m.vert_colors
            .iter()
            .map(|&rgb| luma(rgb))
            .collect::<Vec<_>>()
    };

    let adj = if args.face {
        let ff_adj = m.face_face_adj();
        ff_adj.map(|_, f0, f1, ()| {
            let edge_len = m.f[f0]
                .shared_edges(&m.f[f1])
                .map(|[e0, e1]| dist(m.v[e0], m.v[e1]))
                .sum::<F>();
            edge_len + args.color_weight * (target_chan[f0] - target_chan[f1]).abs()
        })
    } else {
        args.weighting
            .vertex_weights(&m, args.pos_color_norm, args.color_weight)
            .expect("Failed to construct weighting")
    };

    let mut elem_weights = if args.face {
        m.f.iter().map(|f| f.area(&m.v)).collect::<Vec<_>>()
    } else {
        let mut bary_areas = vec![];
        pars3d::geom_processing::barycentric_areas(&m.f, &m.v, &mut bary_areas);
        bary_areas
    };

    for ew in elem_weights.iter_mut() {
        *ew += 1e-12;
    }

    dither(&adj, &elem_weights, &mut target_chan, &args.palette, &args);

    if args.face {
        let face_colors = target_chan.into_iter().map(|l| [l; 3]).collect::<Vec<_>>();
        m = m.with_face_coloring(&face_colors);
    } else {
        for (i, &g) in target_chan.iter().enumerate() {
            assert!((0.0..=1.0).contains(&g), "{g}");
            m.vert_colors[i] = [g; 3];
        }
    }

    m.denormalize(s, t);
    if !args.face {
        m.f = og_f;
    }
    let s = m.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for visualization", start.elapsed());
}

pub fn dither(adj: &Adj<F>, elem_weights: &[F], channel: &mut [F], palette: &[F], args: &Args) {
    // enqueue each vertex into a queue, and take the vertex which has the least difference with
    // quantized result and convert it to quantized value
    let mut pq: PriorityQueue<usize, _> = PriorityQueue::new();

    macro_rules! nearest_palette_and_cost {
        ($color: expr) => {{
            let c = $color;
            palette
                .iter()
                //.map(|&p| (p, dist(p, c)))
                .map(|&p| (p, (p - c).abs()))
                .min_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
                .unwrap()
        }};
    }

    for (i, &c) in channel.iter().enumerate() {
        let (_, dist) = nearest_palette_and_cost!(c);

        let p = match args.order {
            OrderKind::Nearest => NotNan::new(-dist),
            OrderKind::Random => NotNan::new(0.),
            OrderKind::Farthest => NotNan::new(dist),
            OrderKind::Index => NotNan::new(-(i as F)),
        }
        .unwrap();
        pq.push(i, p);
    }

    let mut curr_iter = 0;
    let mut all_ws = vec![];
    while let Some((vi, _)) = pq.pop() {
        let curr_color = channel[vi];
        let nearest = nearest_palette_and_cost!(curr_color).0;

        // RGB error to be diffused to other vertices
        let err = (nearest - curr_color) * elem_weights[vi] * args.error_diffused;
        channel[vi] = nearest;

        curr_iter += 1;

        if err.abs() < args.thresh {
            continue;
        }

        let mut total_w = 0.;
        all_ws.clear();
        for (adj_vi, w) in adj.adj_data(vi) {
            let adj_vi = adj_vi as usize;
            // do not push errors on to vertices which have already been quantized (?)
            // maybe there is some version where this can be recursively applied
            if palette.contains(&channel[adj_vi]) {
                continue;
            }

            // uniform for now
            total_w += w;
            all_ws.push(w);
        }
        if total_w <= 0. {
            continue;
        }
        all_ws.sort_unstable_by(F::total_cmp);

        for (adj_vi, w) in adj.adj_data(vi) {
            let adj_vi = adj_vi as usize;
            if palette.contains(&channel[adj_vi]) {
                continue;
            }
            // uniform for now
            let frac = match args.diffusion {
                ErrorDiffusionKind::Exact => w / total_w,
                ErrorDiffusionKind::Uniform => 1. / all_ws.len() as F,
                ErrorDiffusionKind::LUT => {
                    lut(all_ws.len(), all_ws.iter().position(|&ow| ow == w).unwrap())
                }
            };
            channel[adj_vi] -= frac * err / elem_weights[vi];
            let (_, dist) = nearest_palette_and_cost!(channel[adj_vi]);
            let p = match args.order {
                OrderKind::Nearest => NotNan::new(-dist),
                OrderKind::Random => NotNan::new(0.),
                OrderKind::Farthest => NotNan::new(dist),
                OrderKind::Index => NotNan::new(-((curr_iter + channel.len()) as F)),
            }
            .unwrap();
            pq.push(adj_vi, p);
        }
    }
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

pub fn lut(num_adj: usize, ord: usize) -> F {
    const TAB2: [u32; 2] = [3, 1];
    const TAB3: [u32; 3] = [4, 1, 3];
    const TAB4: [u32; 4] = [7, 1, 5, 3];
    const TAB5: [u32; 5] = [15, 2, 7, 3, 5];
    match num_adj {
        0 => 0.,
        1 => 1.,
        2 => TAB2[ord] as F / 4.,
        3 => TAB3[ord] as F / 8.,
        4 => TAB4[ord] as F / 16.,
        5 => TAB5[ord] as F / 32.,
        _ => 1. / num_adj as F,
        _ => todo!("{num_adj}"),
    }
}

/// The order to take for iteration
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum OrderKind {
    Nearest,
    Random,
    Farthest,
    Index,
}

impl_display!(
  OrderKind,
  Random => "random",
  Nearest => "nearest",
  Farthest => "farthest",
  Index => "index",
  //Snaking => "snaking"
);

/// How to diffuse errors to neighboring element
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum ErrorDiffusionKind {
    Exact,
    LUT,
    Uniform,
}

impl_display!(
  ErrorDiffusionKind,
  Exact => "exact",
  LUT => "lut",
  Uniform => "uniform",
);
