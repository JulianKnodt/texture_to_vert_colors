#![allow(unused)]
use clap::Parser;
use ordered_float::NotNan;
use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, dist, kmul, length, sub};

use priority_queue::PriorityQueue;

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
    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    assert_eq!(m.vert_colors.len(), m.v.len());
    let og_f = m.f.clone();
    m.triangulate();
    let (s, t) = m.normalize();

    let mut bary_areas = vec![];
    pars3d::geom_processing::barycentric_areas(&m.f, &m.v, &mut bary_areas);

    let mut grays = m
        .vert_colors
        .iter()
        .copied()
        .map(|[r, g, b]| (r + g + b) / 3.)
        .collect::<Vec<_>>();
    let vv_adj = m.vertex_vertex_adj();
    for f in &m.f {
        let area = f.area(&m.v);
        if area < 1e-8 {
            continue;
        }
        let [vi0, vi1, vi2] = f.as_tri().unwrap();
        let [v0, v1, v2] = [vi0, vi1, vi2].map(|vi| m.v[vi]);
        let es = [[v1, v0], [v2, v1], [v0, v2]].map(|[va, vb]| sub(va, vb));
        let els = es.map(length);

        let mut longest = [0, 1, 2];
        longest.sort_by(|&a, &b| els[a].total_cmp(&els[b]));
        let ord = match longest {
            [0, 1, 2] => [vi1, vi2, vi0],
            [0, 2, 1] => [vi0, vi2, vi1],

            [1, 0, 2] => [vi1, vi0, vi2],
            [1, 2, 0] => [vi2, vi0, vi1],

            [2, 0, 1] => [vi0, vi1, vi2],
            [2, 1, 0] => [vi2, vi1, vi0],
            _ => unreachable!(),
        };
        //let bary_total = ord.iter().copied().map(|vi| bary_areas[vi]).sum::<F>();
        for (oi, o) in ord.into_iter().enumerate() {
            let w = 1. / (vv_adj.degree(oi) as F);
            //let bw =  bary_areas[oi] / bary_total;
            grays[o] += w * (oi - 1) as F / 3.;
        }
    }

    for (vi, vc) in m.vert_colors.iter_mut().enumerate() {
        *vc = [if grays[vi] > 0.5 { 1. } else { 0. }; 3];
    }

    m.denormalize(s, t);
    m.f = og_f;
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
