#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]

use std::collections::BTreeSet;

use clap::Parser;
use pars3d::{self, Mesh};

use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, dist, kmul, sub};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh path.
    #[arg(long, short)]
    input: String,

    /// Output mesh path.
    #[arg(long, short)]
    output: String,

    #[arg(long, short)]
    weighting: WeightingKind,

    /// How to combine position and color when computing norms.
    #[arg(long, default_value_t = PosColorNorm::Add)]
    pos_color_norm: PosColorNorm,

    /// Unused for now
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// How many iterations to do for the lazy tutte.
    #[arg(long, default_value_t = 10000)]
    iters: usize,

    /// Alternate between expanding and shrinking
    #[arg(long)]
    taubin: bool,

    /// Also smooth the colors of the input mesh
    #[arg(long)]
    smooth_colors: bool,

    /// What values to update for each vertex
    #[arg(long, default_value_t = TargetKind::Pos)]
    target_properties: TargetKind,
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
        m.triangulate_with_new_edges(
            |[e0, e1]| {
                new_edges.insert(std::cmp::minmax(e0, e1));
            },
            0,
        );
        let (s, t) = m.normalize();
        let (sc, tc) = m.normalize_colors();
        smoothing(m, new_edges, &args);
        m.denormalize(s, t);
        m.denormalize_colors(sc, tc);
        if let Some(og_faces) = og_faces {
            m.f = og_faces;
        }
    }
    println!(
        "[INFO]: Took {:?} for smoothing with {} ({}) for {}",
        start.elapsed(),
        args.weighting,
        args.pos_color_norm,
        args.input,
    );

    pars3d::save(&args.output, &scene, true).expect("Failed to save output");
}

pub fn smoothing(mesh: &mut Mesh, new_edges: BTreeSet<[usize; 2]>, args: &Args) {
    let mut vert_adj = args
        .weighting
        .vertex_weights(&mesh, args.pos_color_norm, 1.)
        .expect("Failed to construct vertex adjacency");
    // remove influence of introduced edges
    for vi in 0..mesh.v.len() {
        let (adj_vs, adj_ds) = vert_adj.adj_data_mut(vi);
        for i in 0..adj_vs.len() {
            let e = std::cmp::minmax(vi, adj_vs[i] as usize);
            if new_edges.contains(&e) {
                adj_ds[i] = 0.;
            }
        }
    }
    let vert_adj = vert_adj;

    // lock boundary positions
    let (_, bd_loops) = vert_adj.boundary_loops(&mesh.f);

    let mut vs = &mut mesh.v;
    let mut buf = vs.clone();
    let mut vcs = &mut mesh.vert_colors;
    let mut bufvc = vcs.clone();

    use indicatif::ProgressIterator;
    for it in (0..args.iters).progress() {
        match args.target_properties {
            TargetKind::Pos => {
                buf.fill([0.; 3]);
            }
            TargetKind::Color => {
                bufvc.fill([0.; 3]);
            }
            TargetKind::Both => {
                buf.fill([0.; 3]);
                bufvc.fill([0.; 3]);
            }
        }

        use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

        let max_delta = if matches!(args.target_properties, TargetKind::Pos | TargetKind::Both) {
            buf.par_iter_mut()
                .enumerate()
                .map(|(vi, dst)| {
                    let mut total_w = 0.;
                    let own_pos = vs[vi];
                    let mut delta = [0.; 3];
                    for (adj, w) in vert_adj.adj_data(vi) {
                        total_w += w;
                        delta = add(delta, kmul(w as F, sub(vs[adj as usize], own_pos)));
                    }
                    if total_w == 0. {
                        *dst = vs[vi];
                        return 0.;
                    }
                    *dst = if args.taubin {
                        let mult = if it % 2 == 1 { -0.10001 } else { 0.1 };
                        add(own_pos, kmul(mult * total_w.recip(), delta))
                    } else {
                        add(own_pos, kmul(total_w.recip(), delta))
                    };
                    dist(*dst, vs[vi])
                })
                .max_by(|a, b| F::partial_cmp(a, b).unwrap())
                .unwrap()
        } else {
            0.
        };

        let max_delta_c = if matches!(args.target_properties, TargetKind::Color | TargetKind::Both)
        {
            bufvc
                .par_iter_mut()
                .enumerate()
                .map(|(vi, dst)| {
                    let mut total_w = 0.;
                    let own_col = vcs[vi];
                    let mut delta = [0.; 3];
                    for (adj, w) in vert_adj.adj_data(vi) {
                        total_w += w;
                        delta = add(delta, kmul(w as F, sub(vcs[adj as usize], own_col)));
                    }
                    if total_w == 0. {
                        *dst = vcs[vi];
                        return 0.;
                    }
                    *dst = if args.taubin {
                        let mult = if it % 2 == 1 { -0.1 } else { 0.09 };
                        add(own_col, kmul(mult * total_w.recip(), delta))
                    } else {
                        add(own_col, kmul(total_w.recip(), delta))
                    };
                    dist(*dst, vcs[vi])
                })
                .max_by(|a, b| F::partial_cmp(a, b).unwrap())
                .unwrap()
        } else {
            0.
        };
        let max_delta = max_delta.max(max_delta_c);

        for (&b, _) in &bd_loops {
            buf[b] = vs[b];
            bufvc[b] = vcs[b];
        }
        if max_delta < 1e-12 {
            break;
        }
        match args.target_properties {
            TargetKind::Pos => {
                std::mem::swap(&mut buf, &mut vs);
            }
            TargetKind::Color => {
                std::mem::swap(&mut bufvc, &mut vcs);
            }
            TargetKind::Both => {
                std::mem::swap(&mut buf, &mut vs);
                std::mem::swap(&mut bufvc, &mut vcs);
            }
        }
    }
}

/// What values to update
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TargetKind {
    Pos,
    Color,
    Both,
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

impl_display!(TargetKind, Pos => "pos", Color => "color", Both => "Both");
