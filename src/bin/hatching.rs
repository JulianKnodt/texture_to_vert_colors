#![feature(cmp_minmax)]
#![allow(unused)]
use clap::Parser;
use texture_to_vert_colors::quadric::Quadric;
use texture_to_vert_colors::{F, add, dist, dot, kmul, length, normalize, sub};

use pars3d::adjacency::VertexAdj;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh with per vertex offsets stored in the R channel of the vertex colors
    #[arg(long, short)]
    input: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,

    /// Use grayscale instead of RGB.
    #[arg(long)]
    grayscale: bool,

    /// How wide to trace curves
    #[arg(long, default_value_t = 8e-4)]
    width: F,

    /// How long to trace each curve on the surface.
    /// Likely needs to be tuned per model.
    #[arg(long, default_value_t = 0.01)]
    length: F,

    /// How bendy to make each line
    #[arg(long, default_value_t = 0.)]
    bend_amt: F,

    /// Color threshold above which to draw curves
    #[arg(long, default_value_t = 0.05)]
    color_thresh: F,

    /// Distance threshold above which to draw curves
    #[arg(long, default_value_t = 5e-3)]
    dist_thresh: F,

    /// Direction along which to draw hatches
    #[arg(long, default_value_t = DirKind::MaxCurvature)]
    dir: DirKind,

    /// Unused currently
    #[arg(long, default_value_t = String::new())]
    stats: String,
}

fn main() {
    let args = Args::parse();

    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    assert_eq!(m.vert_colors.len(), m.v.len());
    m.triangulate();
    let (s, t) = m.normalize();

    let vv_adj = m.vertex_vertex_adj();

    let mut new_mesh = if !args.grayscale {
        edge_hatching(&vv_adj, &m, &m.vert_colors, &args)
    } else {
        let lum_chan = [0.2126, 0.7152, 0.0722];
        //let lum_chan = [0.299, 0.587, 0.114];
        let mut vert_grayscale = m
            .vert_colors
            .iter()
            .map(|&rgb| dot(rgb, lum_chan))
            .map(|l| [l; 3])
            .collect::<Vec<_>>();
        edge_hatching(&vv_adj, &m, &vert_grayscale, &args)
    };
    println!(
        "[INFO]: New mesh has #F = {} #V = {}",
        new_mesh.f.len(),
        new_mesh.v.len()
    );

    new_mesh.denormalize(s, t);
    let s = new_mesh.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for hatching", start.elapsed());
}

pub fn edge_hatching(
    vv_adj: &VertexAdj<()>,
    m: &pars3d::Mesh,
    vc: &[[F; 3]],
    args: &Args,
) -> pars3d::Mesh {
    let v = &m.v;
    let edge_adj = m.edge_kinds();

    use pars3d::func::ScalarFn;
    use pars3d::tracing::{Curve, trace_curve_from_mid};
    let mut out = pars3d::Mesh::default();
    let ne = vv_adj.all_pairs_ord().count();
    let prog = indicatif::ProgressBar::new(ne as u64);

    // which direction should be followed
    let mut quadrics = vec![Quadric::<0>::zero(); v.len()];
    for f in &m.f {
        let n = f.normal(v);
        let area = f.area(v);
        let v0 = v[f.as_slice()[0]];
        let mut q = Quadric::new_plane(v0, n, area);
        q.area = area;
        for &vi in f.as_slice() {
            quadrics[vi] += q;
        }
    }

    for (i, vis) in vv_adj.all_pairs_ord().map(|e| e.0).enumerate() {
        prog.set_position(i as u64);
        let e = std::cmp::minmax(vis[0], vis[1]);

        let [v0, v1] = vis.map(|vi| v[vi]);
        let el = dist(v0, v1);
        if el < args.dist_thresh {
            continue;
        }
        let [vc0, vc1] = vis.map(|vi| vc[vi]);
        let cd = (luma(vc0) - luma(vc1)).abs();
        if cd < args.color_thresh {
            continue;
        }

        // commit
        let [q0, q1] = vis.map(|vi| quadrics[vi]);
        let q01 = q0 + q1;
        let ([_, k0, k1], [_, curv0, curv1]) = q01.a.eigen_sorted();

        let midpoint = kmul(0.5, add(v0, v1));
        let fi = edge_adj[&e].as_slice()[0];

        let f = &m.f[fi];
        let start = f.map_kind(|vi| v[vi]).barycentric(midpoint);

        let tri = start.tri(f);
        let (d, dir) = match args.dir {
            DirKind::Edge => (el, normalize(sub(v1, v0))),
            DirKind::MaxCurvature => (k0, curv0),
            DirKind::MinCurvature => (k1, curv1),
            // TODO also some kind of lerp between the two?
        };
        let direction = pars3d::dir_to_barycentric(dir, tri.map(|vi| v[vi]));

        // draw multiple curves which are evenly spaced
        let curve = Curve {
            start,
            start_face: fi,
            direction,
            width: ScalarFn::Linear([args.width], [0.25 * args.width]),

            length: args.length,

            // TODO make bend amt random?
            bend_amt: args.bend_amt, //25.,

            color: ScalarFn::Linear(vc0.map(|v| v * v), vc1.map(|v| v.sqrt())),
        };

        let (wv, wvc, wf) = trace_curve_from_mid(
            v,
            &m.f,
            |[e0, e1]| edge_adj[&std::cmp::minmax(e0, e1)].as_slice(),
            curve,
        );
        let mut wf = pars3d::visualization::wireframe_to_mesh((wv, wvc, wf));
        out.append(&mut wf);
    }
    out
}

fn luma([r, g, b]: [F; 3]) -> F {
    0.299 * r + 0.587 * g + 0.114 * b
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
pub enum DirKind {
    Edge,
    MaxCurvature,
    MinCurvature,
}

impl_display!(
  DirKind,
  Edge => "edge",
  MaxCurvature => "max-curvature",
  MinCurvature => "min-curvature",
);
