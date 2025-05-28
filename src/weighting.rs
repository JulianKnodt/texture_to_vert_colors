use crate::{F, cross, dist, dot, length, normalize, sub};
use pars3d::adjacency::VertexAdj;
use std::collections::BTreeMap;

// NOTE USEFUL:
// https://doc.cgal.org/latest/Weights/group__PkgWeightsRefAnalytic.html

// TODO remove the difference between color and no color approaches.
/// How to weigh relative importance of different vertices.
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum WeightingKind {
    Uniform,

    Length,

    MeanValue,

    Laplacian,
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
        if matches!(self, WeightingKind::MeanValue | WeightingKind::Laplacian) {
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
        }
        let vert_adj = mesh.vertex_vertex_adj();
        let per_face_info = match self {
            WeightingKind::Uniform | WeightingKind::Length => vec![],
            WeightingKind::MeanValue => mesh
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
                    let [v0, v1, v2] = f.as_tri().unwrap();
                    let cot_ang = |ri, ai, bi| {
                        let [r, a, b] = [ri, ai, bi].map(|vi| mesh.v[vi]);
                        let ar = normalize(sub(a, r));
                        let br = normalize(sub(b, r));
                        let cos = dot(ar, br).clamp(-1., 1.);
                        let sin = length(cross(ar, br)).clamp(-1., 1.);
                        //assert!((sin - (1. - cos * cos).max(0.).sqrt()).abs() < 1e-6);
                        // numerical stability
                        let sin = (sin.abs() + 1e-4).copysign(sin);
                        let cot = cos / sin;
                        assert!(cot.is_finite(), "{cot:?}");
                        let dist = if mesh.vert_colors.is_empty() {
                            dist(a, b)
                        } else {
                            pos_color_norm
                                .apply(dist(a, b), dist(mesh.vert_colors[ai], mesh.vert_colors[bi]))
                        };
                        cot * dist.max(1e-2)
                    };
                    [
                        cot_ang(v0, v1, v2),
                        cot_ang(v1, v2, v0),
                        cot_ang(v2, v0, v1),
                    ]
                })
                .collect::<Vec<_>>(),
        };

        let per_vert_weights = match self {
            WeightingKind::Laplacian => {
                let mut vw = vec![0.; mesh.v.len()];
                for f in mesh.f.iter() {
                    let area = f.area(&mesh.v) / 3.;
                    for &vi in f.as_slice() {
                        vw[vi] += area;
                    }
                }
                vw
            }
            _ => vec![],
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
            WeightingKind::Length => pos_color_norm
                .apply(
                    dist(mesh.v[v0], mesh.v[v1]),
                    dist(mesh.vert_colors[v0], mesh.vert_colors[v1]),
                )
                .max(1e-3)
                .recip(),
            WeightingKind::MeanValue => {
                let d = dist(mesh.v[v0], mesh.v[v1]);
                let cd = dist(mesh.vert_colors[v0], mesh.vert_colors[v1]) + 3e-3;
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
                let voronoi_area = per_vert_weights[v0] + 1e-4;
                //assert_ne!(voronoi_area, 0.);
                softplus(w / (2. * voronoi_area))
            }
        });
        Ok(va)
    }
}

pub fn softplus(x: F) -> F {
    if x > 20. { x } else { (1. + x.exp()).ln() }
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

    MeanValue => "mean-value",

    Laplacian => "laplacian",
);

#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum PosColorNorm {
    Add,
    Mul,
    Min,
    Max,
    Concat,
    /// Similar to mul, except take the sqrt. Non-linearly affects the output.
    GeometricMean,
    /// ||Color||
    ColorOnly,
    /// ||Pos||
    PosOnly,

    /// Gaussian(pos) * Gaussian(color)
    Bilateral,

    /// Something for me to test anything with
    Tester,
}

impl_display!(
  PosColorNorm,
  Add => "add",
  Mul => "mul",
  Min => "min",
  Max => "max",
  Concat => "concat",
  GeometricMean => "geometric-mean",
  ColorOnly => "color-only",
  PosOnly => "pos-only",
  Bilateral => "bilateral",

  Tester => "tester",
);

impl PosColorNorm {
    pub fn apply(self, pos: F, color: F) -> F {
        use PosColorNorm::*;
        match self {
            Add => pos + color,
            Mul => pos * color,
            Min => pos.min(color),
            Max => pos.max(color),
            Concat => ((pos * pos) + (color * color)).sqrt(),
            GeometricMean => (pos * color).sqrt(),
            ColorOnly => color,
            PosOnly => pos,
            Bilateral => gaussian(pos) * gaussian(color),

            Tester => pos * (1. + color),
        }
    }
}

pub fn gaussian(x: F) -> F {
    let k: F = 1. / std::f64::consts::TAU.sqrt() as F;
    k * (-0.5 * x * x).exp()
}
