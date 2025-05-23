#![allow(unused)]
use clap::Parser;
use ordered_float::NotNan;
use texture_to_vert_colors::{F, add, dist, kmul, sub};

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
    #[arg(long, short, default_value_t = WeightKind::Uniform)]
    weight_kind: WeightKind,
    // error diffusion kind (uniform, laplacian, distance)

    // use a different palette

    #[arg(long, default_value_t = OrderKind::Nearest)]
    order: OrderKind,
}

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    assert_eq!(m.vert_colors.len(), m.v.len());
    let (s, t) = m.normalize();

    let mut vert_grayscale = m
        .vert_colors
        .iter()
        .map(|[r, g, b]| (r + g + b) / 3.)
        .collect::<Vec<_>>();

    let mut barycentric_areas = vec![];
    pars3d::geom_processing::barycentric_areas(&m.f, &m.v, &mut barycentric_areas);
    // normalize barycentric areas?

    let palette = [0., 1.];

    // enqueue each vertex into a queue, and take the vertex which has the least difference with
    // quantized result and convert it to quantized value
    let mut pq: PriorityQueue<usize, _> = PriorityQueue::new();

    macro_rules! nearest_palette_and_cost {
        ($color: expr) => {{
            let c = $color;
            palette
                .iter()
                //.map(|&p| (p, dist(p, c)))
                .map(|&p| (p, (p-c).abs()))
                .min_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
                .unwrap()
        }};
    }

    for (vi, &vc) in vert_grayscale.iter().enumerate() {
        let (_, dist) = nearest_palette_and_cost!(vc);

        let p = match args.order {
          OrderKind::Nearest => NotNan::new(-dist).unwrap(),
          OrderKind::Random => NotNan::new(0.).unwrap(),
          OrderKind::Farthest => NotNan::new(dist).unwrap(),
        };
        pq.push(vi, p);
    }

    let vv_adj = m.vertex_vertex_adj();

    while let Some((vi, _)) = pq.pop() {
        let curr_color = vert_grayscale[vi];
        let nearest = nearest_palette_and_cost!(curr_color).0;

        // RGB error to be diffused to other vertices
        let err = nearest - curr_color;
        vert_grayscale[vi] = nearest;

        let adj_verts = vv_adj.adj(vi);
        let mut total_w = 0.;
        for &adj_vi in adj_verts {
            let adj_vi = adj_vi as usize;
            // do not push errors on to vertices which have already been quantized (?)
            // maybe there is some version where this can be recursively applied
            if palette.contains(&vert_grayscale[adj_vi]) {
                continue;
            }

            // uniform for now
            total_w += match args.weight_kind {
                WeightKind::Uniform => 1.,
                WeightKind::BarycentricArea => barycentric_areas[adj_vi],
                WeightKind::Laplacian => todo!(),
            };
        }
        if total_w <= 0. {
            continue;
        }
        for &adj_vi in adj_verts {
            let adj_vi = adj_vi as usize;
            if palette.contains(&vert_grayscale[adj_vi]) {
                continue;
            }
            let w = match args.weight_kind {
                WeightKind::Uniform => 1.,
                WeightKind::BarycentricArea => barycentric_areas[adj_vi],
                WeightKind::Laplacian => todo!(),
            };
            // uniform for now
            let frac = w / total_w;
            vert_grayscale[adj_vi] -= frac * err;
            let (_, dist) = nearest_palette_and_cost!(vert_grayscale[adj_vi]);
            let p = match args.order {
              OrderKind::Nearest => NotNan::new(-dist).unwrap(),
              OrderKind::Random => NotNan::new(0.).unwrap(),
              OrderKind::Farthest => NotNan::new(dist).unwrap(),
            };
            let prev = pq.change_priority(&adj_vi, p);
            assert_ne!(prev, None);
        }
    }

    for (i, &g) in vert_grayscale.iter().enumerate() {
      assert!((0.0..=1.0).contains(&g), "{g}");
      m.vert_colors[i] = [g; 3];
    }

    m.denormalize(s, t);
    let s = m.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for visualization", start.elapsed());
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
pub enum WeightKind {
    Uniform,
    BarycentricArea,
    Laplacian,
}

impl_display!(
  WeightKind,
  Uniform => "uniform",
  BarycentricArea => "barycentric-area",
  Laplacian => "laplacian",
);

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
