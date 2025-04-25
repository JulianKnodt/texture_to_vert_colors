#![feature(cmp_minmax)]

use clap::{Parser, ValueEnum};
use pars3d::{self, Mesh};
use std::collections::BTreeMap;

use texture_to_vert_colors::{F, add, dist, dot, kmul, normalize, sub};

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

    /// Which UV channel to store the tutte parameterization into.
    #[arg(long, default_value_t = 0)]
    target_uv: usize,
}

fn main() {
    let args = Args::parse();

    let mut scene = pars3d::load(&args.input).expect("Failed to parse input");

    let start = std::time::Instant::now();
    for m in scene.meshes.iter_mut() {
        let (s, t) = m.normalize();
        assert_eq!(
            m.num_tris(),
            m.f.len(),
            "Assumes the input mesh is entirely triangles"
        );
        tutte_param(m, &args);
        m.denormalize(s, t);
    }
    println!(
        "[INFO]: Took {:?} for tutte parameterization",
        start.elapsed()
    );

    pars3d::save(&args.output, &scene).expect("Failed to save output");
}

pub fn tutte_param(mesh: &mut Mesh, args: &Args) {
    let vert_adj = mesh.vertex_adj();
    let (num_loops, bd_loops) = vert_adj.boundary_loops(&mesh);
    assert_eq!(num_loops, 1);

    let mut bd = vec![];
    let (&first, &[last, mut next]) = bd_loops.first_key_value().unwrap();
    let mut prev = first;
    bd.push((prev, 0.));
    let mut curr_len = 0.;
    loop {
        let [_, n] = bd_loops[&next];
        curr_len += dist(mesh.v[next], mesh.v[prev]);
        bd.push((next, curr_len));
        prev = next;
        next = n;
        if next == first {
            break;
        }
    }
    assert_eq!(next, first);

    curr_len += dist(mesh.v[first], mesh.v[last]);
    assert_ne!(curr_len, 0.);
    for (_, l) in bd.iter_mut() {
        *l /= curr_len;
        assert!((0.0..=1.0).contains(l));
        *l *= std::f64::consts::TAU as F;
    }

    mesh.uv[args.target_uv].clear();
    mesh.uv[args.target_uv].resize(mesh.v.len(), [0.5; 2]);
    let mut uvs = &mut mesh.uv[args.target_uv];

    for (vi, l) in bd {
        uvs[vi] = [(l.cos() + 1.) / 2., (l.sin() + 1.) / 2.];
    }
    let mut next = uvs.clone();

    let mut edge_face_adj = BTreeMap::new();
    for (fi, f) in mesh.f.iter().enumerate() {
        for e in f.edges_ord() {
            let slot = edge_face_adj
                .entry(e)
                .or_insert([usize::MAX; 2])
                .iter_mut()
                .find(|v| **v == usize::MAX)
                .unwrap();
            *slot = fi;
        }
    }

    let per_face_info = match args.weighting {
        WeightingKind::Uniform => vec![[0.; 3]; mesh.f.len()],
        WeightingKind::ColoredMeanValue | WeightingKind::MeanValue => mesh
            .f
            .iter()
            .map(|f| {
                let [v0, v1, v2] = f.as_tri().unwrap().map(|vi| mesh.v[vi]);
                let tan_ang = |r, a, b| {
                    let ar = normalize(sub(a, r));
                    let br = normalize(sub(b, r));
                    let cos_ang = dot(ar, br);
                    assert!((1. + cos_ang).abs() > 1e-8);
                    let v = (1. - cos_ang) / (1. + cos_ang);
                    assert!(v >= 0.);
                    v.sqrt()
                };
                [
                    tan_ang(v0, v1, v2),
                    tan_ang(v1, v2, v0),
                    tan_ang(v2, v0, v1),
                ]
            })
            .collect::<Vec<_>>(),
        WeightingKind::Laplacian => todo!(),
    };

    macro_rules! mean_value {
        ($v0: expr, $v1: expr) => {{
            let v0 = $v0;
            let v1 = $v1;
            let [f0, f1] = edge_face_adj[&std::cmp::minmax(v0, v1)];
            if f0 == usize::MAX || f1 == usize::MAX {
                // This means it's a boundary edge.
                return 0.;
            }

            let get_val = |fi: usize| {
                let idx = mesh.f[fi]
                    .as_slice()
                    .iter()
                    .position(|&vi| vi == v0)
                    .unwrap();
                per_face_info[fi][idx]
            };
            (get_val(f0) + get_val(f1))
        }};
    }
    // Compute per vertex weights
    let vert_adj = vert_adj.map(|adj, v0, v1, ()| match args.weighting {
        WeightingKind::Uniform => 1. / adj.degree(v0) as F,
        WeightingKind::MeanValue => {
            let d = dist(mesh.v[v0], mesh.v[v1]);
            let mv = mean_value!(v0, v1);
            d.recip() * mv
        }
        WeightingKind::ColoredMeanValue => {
            let d = dist(mesh.v[v0], mesh.v[v1]);
            let cd = dist(mesh.vert_colors[v0], mesh.vert_colors[v1]);
            let mv = mean_value!(v0, v1);
            (d * cd).recip() * mv
        }
        _ => todo!(),
    });

    for _ in 0..10000 {
        for vi in 0..mesh.v.len() {
            if bd_loops.contains_key(&vi) {
                continue;
            }
            next[vi].fill(0.);

            let mut total_w = 0.;
            for (adj, w) in vert_adj.adj_data(vi) {
                total_w += w;
                next[vi] = add(next[vi], kmul(w as F, uvs[adj as usize]));
            }
            next[vi] = next[vi].map(|c| c / total_w);
        }

        std::mem::swap(&mut next, &mut uvs);
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

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum WeightingKind {
    Uniform,
    Laplacian,
    //ColoredUniform,
    //ColoredLaplacian,
    MeanValue,
    ColoredMeanValue,
}

impl_display!(
    WeightingKind,
    Uniform => "uniform",
    Laplacian => "laplacian",
    MeanValue => "mean-value",

    ColoredMeanValue => "colored-mean-value",
);
