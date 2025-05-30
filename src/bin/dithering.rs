#![allow(unused)]
use clap::Parser;
use ordered_float::NotNan;
use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, dist, dot, kmul, sub};

use pars3d::adjacency::VertexAdj;
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
    #[arg(long, default_value_t = PosColorNorm::PosOnly)]
    pos_color_norm: PosColorNorm,

    #[arg(long, default_value_t = OrderKind::Nearest)]
    order: OrderKind,

    // use a different palette
    /// Use RGB instead of grayscale.
    #[arg(long)]
    rgb: bool,

    /// Use a different palette for dithering (Default = [0., 1.])
    #[arg(long, short)]
    palette: Vec<F>,

    /// Unused
    #[arg(long)]
    stats: String,
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
    m.triangulate();
    let (s, t) = m.normalize();

    let vv_adj = args
        .weighting
        .vertex_weights(&m, args.pos_color_norm)
        .expect("Failed to construct weighting");

    if args.rgb {
        for c in 0..3 {
            let mut color_chan = m.vert_colors.iter().map(|rgb| rgb[c]).collect::<Vec<_>>();
            dither(&vv_adj, &mut color_chan, &args.palette, &args);

            for (i, &g) in color_chan.iter().enumerate() {
                m.vert_colors[i][c] = g;
            }
        }
    } else {
        let lum_chan = [0.2126, 0.7152, 0.0722];
        //let lum_chan = [0.299, 0.587, 0.114];
        let mut vert_grayscale = m
            .vert_colors
            .iter()
            .map(|&rgb| dot(rgb, lum_chan))
            .collect::<Vec<_>>();

        dither(&vv_adj, &mut vert_grayscale, &args.palette, &args);

        for (i, &g) in vert_grayscale.iter().enumerate() {
            assert!((0.0..=1.0).contains(&g), "{g}");
            m.vert_colors[i] = [g; 3];
        }
    }

    m.denormalize(s, t);
    m.f = og_f;
    let s = m.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for visualization", start.elapsed());
}

pub fn dither(vv_adj: &VertexAdj<F>, channel: &mut [F], palette: &[F], args: &Args) {
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

    for (vi, &vc) in channel.iter().enumerate() {
        let (_, dist) = nearest_palette_and_cost!(vc);

        let p = match args.order {
            OrderKind::Nearest => NotNan::new(-dist).unwrap(),
            OrderKind::Random => NotNan::new(0.).unwrap(),
            OrderKind::Farthest => NotNan::new(dist).unwrap(),
        };
        pq.push(vi, p);
    }

    while let Some((vi, _)) = pq.pop() {
        let curr_color = channel[vi];
        let nearest = nearest_palette_and_cost!(curr_color).0;

        // RGB error to be diffused to other vertices
        let err = nearest - curr_color;
        channel[vi] = nearest;

        let mut total_w = 0.;
        for (adj_vi, w) in vv_adj.adj_data(vi) {
            let adj_vi = adj_vi as usize;
            // do not push errors on to vertices which have already been quantized (?)
            // maybe there is some version where this can be recursively applied
            if palette.contains(&channel[adj_vi]) {
                continue;
            }

            // uniform for now
            total_w += w;
        }
        if total_w <= 0. {
            continue;
        }
        for (adj_vi, w) in vv_adj.adj_data(vi) {
            let adj_vi = adj_vi as usize;
            if palette.contains(&channel[adj_vi]) {
                continue;
            }
            // uniform for now
            let frac = w / total_w;
            channel[adj_vi] -= frac * err;
            let (_, dist) = nearest_palette_and_cost!(channel[adj_vi]);
            let p = match args.order {
                OrderKind::Nearest => NotNan::new(-dist).unwrap(),
                OrderKind::Random => NotNan::new(0.).unwrap(),
                OrderKind::Farthest => NotNan::new(dist).unwrap(),
            };
            let prev = pq.change_priority(&adj_vi, p);
            assert_ne!(prev, None);
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum OrderKind {
    Nearest,
    Random,
    Farthest,
}

impl_display!(
  OrderKind,
  Random => "random",
  Nearest => "nearest",
  Farthest => "farthest",
);
