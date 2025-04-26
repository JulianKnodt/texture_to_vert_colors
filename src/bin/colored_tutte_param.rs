#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]

use clap::{Parser, ValueEnum};
use pars3d::{self, Mesh};
use std::collections::BTreeMap;

use texture_to_vert_colors::{F, add, cross, dist, dot, kmul, length, normalize, sub};

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

    /// How to combine position and color when computing norms.
    #[arg(long, default_value_t = PosColorNorm::Mul)]
    pos_color_norm: PosColorNorm,

    /// Save SVG of UV to this destination. If empty will not save.
    #[arg(long, default_value_t = String::new())]
    uv_svg: String,

    /// Bake vertex colors to a texture
    #[arg(long, default_value_t = String::new())]
    bake_texture: String,

    /// Unused for now
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// How many iterations to do for the lazy tutte.
    #[arg(long, default_value_t = 15000)]
    iters: usize,
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
        m.triangulate();
        let (s, t) = m.normalize();
        let (sc, tc) = m.normalize_colors();
        tutte_param(m, &args);
        m.denormalize(s, t);
        m.denormalize_colors(sc, tc);
        if let Some(og_faces) = og_faces {
            m.f = og_faces;
        }
    }
    println!(
        "[INFO]: Took {:?} for tutte parameterization",
        start.elapsed()
    );

    if !args.uv_svg.is_empty() {
        assert!(
            args.uv_svg.ends_with(".svg"),
            "Only SVG export is supported, incorrect extension for {}",
            args.uv_svg
        );
        let m0 = &scene.meshes[0];
        if let Err(e) = pars3d::svg::save_uv(args.uv_svg, &m0.uv[args.target_uv], &m0.f, 0.0003) {
            eprintln!("Failed to save SVG due to {e:?}");
        }
    }

    if !args.bake_texture.is_empty() {
        let m0 = &scene.meshes[0];
        let mut img = pars3d::coloring::bake_vertex_colors_to_texture(
            [512, 512],
            &m0.uv[args.target_uv],
            &m0.f,
            &m0.vert_colors,
        );
        pars3d::image::imageops::flip_vertical_in_place(&mut img);
        let nf = m0.f.len();
        let new_texture = pars3d::mesh::Texture {
            kind: pars3d::mesh::TextureKind::Diffuse,
            mul: [1.; 4],
            image: Some(img.into()),
            original_path: args.bake_texture,
        };
        let ti = scene.textures.len();
        scene.textures.push(new_texture);
        let new_mat = pars3d::mesh::Material {
            textures: vec![ti],
            name: format!("BakedVertexColors0"),
            path: args.output[..args.output.len() - 4].to_string() + ".mtl",
        };
        let mi = scene.materials.len();
        scene.materials.push(new_mat);
        scene.meshes[0].face_mat_idx = vec![((0..nf), mi)];
    }
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
            let slots = edge_face_adj.entry(e).or_insert([usize::MAX; 2]);
            let slot = slots.iter_mut().find(|v| **v == usize::MAX || **v == fi);
            let Some(slot) = slot else {
                for &fi in slots.iter() {
                    println!("Previous {:?}", mesh.f[fi]);
                }
                println!("To be inserted {:?}", mesh.f[fi]);
                panic!("Failed to find slot for {fi} in {slots:?} with edge {e:?}");
            };
            *slot = fi;
        }
    }

    let per_face_info = match args.weighting {
        WeightingKind::Uniform => vec![],
        WeightingKind::ColoredMeanValue | WeightingKind::MeanValue => mesh
            .f
            .iter()
            .map(|f| {
                let [v0, v1, v2] = f.as_tri().unwrap().map(|vi| mesh.v[vi]);
                let tan_ang = |r, a, b| {
                    let ar = normalize(sub(a, r));
                    let br = normalize(sub(b, r));
                    let cos_ang = dot(ar, br);
                    let v = (1. - cos_ang) / (1. + cos_ang);
                    assert!(v.is_finite());
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
                    let cot = cos / sin;
                    assert!(cot.is_finite());
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

    let per_vert_weights = if matches!(args.weighting, WeightingKind::Laplacian) {
        let mut vw = vec![0.; mesh.v.len()];
        for (fi, f) in mesh.f.iter().enumerate() {
            let vis = f.as_tri().unwrap();
            let vs = vis.map(|vi| mesh.v[vi]);
            let cots = per_face_info[fi];
            for i in 0..3 {
                let n = (i + 1) % 3;
                let nn = (n + 1) % 3;
                vw[vis[i]] += (dist(vs[n], vs[i]) * cots[nn] + dist(vs[nn], vs[i]) * cots[n]) / 8.;
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
    let vert_adj = vert_adj.map(|adj, v0, v1, ()| match args.weighting {
        WeightingKind::Uniform => 1. / adj.degree(v0) as F,
        WeightingKind::MeanValue => {
            let d = dist(mesh.v[v0], mesh.v[v1]) + 1e-6;
            let mv = mean_value!(v0, v1);
            d.recip() * mv
        }
        WeightingKind::ColoredMeanValue => {
            let d = dist(mesh.v[v0], mesh.v[v1]);
            let cd = dist(mesh.vert_colors[v0], mesh.vert_colors[v1]) + 3e-3;
            let w = args.pos_color_norm.apply(d, cd);
            assert!(w.is_finite());
            w.recip() * mean_value!(v0, v1)
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
            w * per_vert_weights[v0]
        }
    });

    for _ in 0..args.iters {
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

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
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
