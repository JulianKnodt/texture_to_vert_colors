use crate::{F, dist};
use pars3d::adjacency::VertexAdj;
use std::collections::BTreeMap;

// NOTE USEFUL:
// https://doc.cgal.org/latest/Weights/group__PkgWeightsRefAnalytic.html

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
        color_weight: F,
    ) -> Result<VertexAdj<F>, Error> {
        let cw = color_weight;
        let mut edge_face_adj = BTreeMap::new();
        if matches!(self, WeightingKind::MeanValue) {
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
            WeightingKind::Uniform | WeightingKind::Length | WeightingKind::Laplacian => vec![],
            WeightingKind::MeanValue => mesh
                .f
                .iter()
                .map(|f| {
                    let [v0, v1, v2] = f.as_tri().unwrap();
                    let dist_fn = |a, b| {
                        let d = dist(mesh.v[a], mesh.v[b]);
                        if mesh.vert_colors.is_empty() {
                            return d;
                        }
                        let cd = dist(mesh.vert_colors[a], mesh.vert_colors[b]);
                        pos_color_norm.apply(d, cd, cw)
                    };
                    let opp_es = [dist_fn(v1, v2), dist_fn(v0, v2), dist_fn(v1, v0)];
                    let cos_s = pars3d::cosine_angles(opp_es).map(|c| c.clamp(-1., 1.));

                    let out = cos_s.map(|c| (1. - c) / (1. + c + 1e-5)).map(F::sqrt);
                    assert!(out.iter().copied().all(F::is_finite));
                    out
                })
                .collect::<Vec<_>>(),
        };

        //const EPS: F = 1e-5; // this is too high?
        //const EPS: F = 1e-6; // this works fine
        const EPS: F = 2e-7;
        let mut per_edge_weights = BTreeMap::new();
        if matches!(self, WeightingKind::Laplacian) {
            for f in &mesh.f {
                let dist_fn = |a, b| {
                    //dist(mesh.v[a], mesh.v[b])
                    let d = dist(mesh.v[a], mesh.v[b]);
                    if mesh.vert_colors.is_empty() {
                        return d;
                    }
                    let cd = color_dist(mesh.vert_colors[a], mesh.vert_colors[b]);
                    pos_color_norm.apply(d, cd, cw)
                };
                assert!(f.is_tri());
                for [pi, vi, ni] in f.incident_edges() {
                    let a = dist_fn(pi, vi);
                    let b = dist_fn(vi, ni);
                    let c = dist_fn(pi, ni);
                    let area = pars3d::herons_area([a, b, c]);
                    let v = a * a + b * b - c * c;
                    let cot_c = v / (4. * area + EPS);
                    assert!(cot_c.is_finite(), "{cot_c:?} {area:?} {v:?}");
                    let ew = per_edge_weights
                        .entry(std::cmp::minmax(pi, ni))
                        .or_insert(0.);
                    *ew += cot_c;
                }
            }
        }

        let per_vert_weights = match self {
            WeightingKind::Laplacian => {
                let mut vw = vec![0.; mesh.v.len()];
                for f in mesh.f.iter() {
                    let [v0, v1, v2] = f.as_tri().unwrap();
                    let dist_fn = |a, b| {
                        let d = dist(mesh.v[a], mesh.v[b]);
                        if mesh.vert_colors.is_empty() {
                            return d;
                        }
                        let cd = color_dist(mesh.vert_colors[a], mesh.vert_colors[b]);
                        pos_color_norm.apply(d, cd, cw)
                    };
                    let es = [dist_fn(v0, v1), dist_fn(v1, v2), dist_fn(v2, v0)];
                    let area = pars3d::herons_area(es) / 3.;
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
                get_val(f0) + get_val(f1)
            }};
        }

        // Compute per vertex weights
        let va = vert_adj.map(|_, v0, v1, ()| match self {
            WeightingKind::Uniform => 1.,
            WeightingKind::Length => {
                let d = dist(mesh.v[v0], mesh.v[v1]);
                let d = if mesh.vert_colors.is_empty() {
                    d
                } else {
                    let cd = color_dist(mesh.vert_colors[v0], mesh.vert_colors[v1]);
                    pos_color_norm.apply(d, cd, cw)
                };
                (d + 1e-4).recip().min(10.)
            }
            WeightingKind::MeanValue => {
                let d = dist(mesh.v[v0], mesh.v[v1]);
                let cd = dist(mesh.vert_colors[v0], mesh.vert_colors[v1]);
                let w = pos_color_norm.apply(d, cd, cw);
                assert!(w.is_finite());
                /*(d + 1e-8).recip() * */
                mean_value!(v0, v1)
            }
            WeightingKind::Laplacian => {
                let voronoi = per_vert_weights[v0];
                let l = per_edge_weights[&std::cmp::minmax(v0, v1)];
                assert!(l.is_finite());
                assert!(voronoi >= 0.);
                softplus(l) / (2. * voronoi + EPS)
            }
        });
        Ok(va)
    }
}

pub fn softplus(x: F) -> F {
    if x > 1. { x } else { (1. + x.exp()).ln() }
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
    Max,
    GeometricMean,
    Concat,
    /// ||Color||
    ColorOnly,
    /// Gaussian(pos) * Gaussian(color)
    Bilateral,

    /// Something for me to test anything with
    Tester,
}

impl_display!(
  PosColorNorm,
  Add => "add",
  Max => "max",
  GeometricMean => "geometric-mean",

  Concat => "concat",
  ColorOnly => "color-only",
  Bilateral => "bilateral",

  Tester => "tester",
);

impl PosColorNorm {
    pub fn apply(self, pos: F, color: F, color_weight: F) -> F {
        let cw = color_weight;
        use PosColorNorm::*;
        match self {
            Add => pos + cw * color,
            Max => pos.max(cw * color),
            GeometricMean => (pos * color).sqrt(),
            Concat => ((pos * pos) + cw * (color * color)).sqrt(),
            ColorOnly => color,
            Bilateral => gaussian(pos) * gaussian(color),

            Tester => todo!(),
        }
    }
}

pub fn gaussian(x: F) -> F {
    let k: F = 1. / std::f64::consts::TAU.sqrt() as F;
    k * (-0.5 * x * x).exp()
}

pub fn color_dist(a: [F; 3], b: [F; 3]) -> F {
    dist(a, b)
    //super::sub(a, b).map(F::abs).into_iter().sum()
    //let s = [0.299, 0.587, 0.114];
    //(super::dot(s,a) - super::dot(s,b)).abs()
}
