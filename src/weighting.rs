use crate::{F, cross, dist, dist_sq, dot, length, normalize, sub};
use pars3d::adjacency::VertexAdj;
use std::collections::BTreeMap;

/// How to weigh relative importance of different vertices.
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum WeightingKind {
    Uniform,
    Laplacian,

    Length,
    ColorLength,

    //ColoredLaplacian,
    MeanValue,
    ColoredMeanValue,
}

/// Possible errors when constructing vertex adjacencies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    /// edge, 2 previous faces and face being added
    NonManifoldEdge([usize; 2], [usize; 2], usize),
}

impl WeightingKind {
    pub fn vertex_weights(
        &self,
        mesh: &pars3d::Mesh,
        pos_color_norm: PosColorNorm,
    ) -> Result<VertexAdj<F>, Error> {
        let mut edge_face_adj = BTreeMap::new();
        for (fi, f) in mesh.f.iter().enumerate() {
            for e in f.edges_ord() {
                let slots = edge_face_adj.entry(e).or_insert([usize::MAX; 2]);
                let slot = slots.iter_mut().find(|v| **v == usize::MAX || **v == fi);
                let Some(slot) = slot else {
                    return Result::Err(Error::NonManifoldEdge(e, *slots, fi));
                };
                *slot = fi;
            }
        }
        let vert_adj = mesh.vertex_adj();
        let per_face_info = match self {
            WeightingKind::Uniform | WeightingKind::Length | WeightingKind::ColorLength => vec![],
            WeightingKind::ColoredMeanValue | WeightingKind::MeanValue => mesh
                .f
                .iter()
                .map(|f| {
                    let vis = f.as_tri().unwrap();
                    let [v0, v1, v2] = vis.map(|vi| mesh.v[vi]);
                    let tan_ang = |r, a, b| {
                        let ar = normalize(sub(a, r));
                        let br = normalize(sub(b, r));
                        let cos_ang = dot(ar, br).clamp(-1., 1.);
                        let v = (1. - cos_ang) / (1. + cos_ang + 1e-4);
                        assert!(v.is_finite(), "{v} {cos_ang} {vis:?}");
                        assert!(v >= 0., "{v:?} {cos_ang} {vis:?}");
                        v.sqrt()
                    };
                    [
                        tan_ang(v0, v1, v2),
                        tan_ang(v1, v2, v0),
                        tan_ang(v2, v0, v1),
                    ]
                })
                .collect::<Vec<_>>(),
            WeightingKind::Laplacian => mesh
                .f
                .iter()
                .map(|f| {
                    let [v0, v1, v2] = f.as_tri().unwrap().map(|vi| mesh.v[vi]);
                    let cot_ang = |r, a, b| {
                        let ar = normalize(sub(a, r));
                        let br = normalize(sub(b, r));
                        let cos = dot(ar, br);
                        let sin = length(cross(ar, br));
                        // numerical stability
                        let sin = (sin.abs() + 1e-4).copysign(sin);
                        let cot = cos / sin;
                        assert!(cot.is_finite(), "{cot:?}");
                        cot
                    };
                    [
                        cot_ang(v0, v1, v2),
                        cot_ang(v1, v2, v0),
                        cot_ang(v2, v0, v1),
                    ]
                })
                .collect::<Vec<_>>(),
        };

        let per_vert_weights = if matches!(self, WeightingKind::Laplacian) {
            let mut vw = vec![0.; mesh.v.len()];
            for (fi, f) in mesh.f.iter().enumerate() {
                let vis = f.as_tri().unwrap();
                let vs = vis.map(|vi| mesh.v[vi]);
                let cots = per_face_info[fi];
                for i in 0..3 {
                    let n = (i + 1) % 3;
                    let nn = (n + 1) % 3;
                    let a = dist_sq(vs[n], vs[i]) * cots[nn] + dist_sq(vs[nn], vs[i]) * cots[n];
                    vw[vis[i]] += a / 8.;
                }
            }
            vw
        } else {
            vec![]
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
        let va = vert_adj.map(|_, v0, v1, ()| match self {
            WeightingKind::Uniform => 1.,
            WeightingKind::Length => dist(mesh.v[v0], mesh.v[v1]).max(1e-3).recip(),
            WeightingKind::ColorLength => pos_color_norm
                .apply(
                    dist(mesh.v[v0], mesh.v[v1]),
                    dist(mesh.vert_colors[v0], mesh.vert_colors[v1]),
                )
                .max(1e-3)
                .recip(),
            WeightingKind::MeanValue => {
                let d = dist(mesh.v[v0], mesh.v[v1]);
                let mv = mean_value!(v0, v1);
                d.max(1e-3).recip() * mv
            }
            WeightingKind::ColoredMeanValue => {
                let d = dist(mesh.v[v0], mesh.v[v1]);
                let cd = dist(mesh.vert_colors[v0], mesh.vert_colors[v1]);
                let w = pos_color_norm.apply(d, cd);
                assert!(w.is_finite());
                w.max(1e-3).recip() * mean_value!(v0, v1)
            }
            WeightingKind::Laplacian => {
                let [f0, f1] = edge_face_adj[&std::cmp::minmax(v0, v1)];
                if f0 == usize::MAX || f1 == usize::MAX {
                    return 0.;
                }
                let get_val = |fi: usize| {
                    let idx = mesh.f[fi]
                        .as_slice()
                        .iter()
                        .position(|&vi| vi != v0 && vi != v1)
                        .unwrap();
                    per_face_info[fi][idx]
                };
                let w = get_val(f0) + get_val(f1);
                let voronoi_area = per_vert_weights[v0];
                assert_ne!(voronoi_area, 0.);
                let w = w * (2. * voronoi_area).recip();
                w.max(0.) + 5.
            }
        });
        Ok(va)
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

impl_display!(
    WeightingKind,
    Uniform => "uniform",

    Length => "length",
    ColorLength => "color-length",

    Laplacian => "laplacian",
    MeanValue => "mean-value",

    ColoredMeanValue => "colored-mean-value",
);

#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum PosColorNorm {
    Add,
    Mul,
    Min,
    Max,
    GeometricMean,
}

impl_display!(
  PosColorNorm,
  Add => "add",
  Mul => "mul",
  Min => "min",
  Max => "max",
  GeometricMean => "geometric-mean"
);

impl PosColorNorm {
    pub fn apply(self, pos: F, color: F) -> F {
        use PosColorNorm::*;
        match self {
            Add => pos + color,
            Mul => pos * color,
            Min => pos.min(color),
            Max => pos.max(color),
            GeometricMean => (pos * color).sqrt(),
        }
    }
}
