#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(cmp_minmax)]
#![feature(let_chains)]

use std::collections::BTreeSet;

use clap::Parser;
use pars3d::{self, Mesh};

use texture_to_vert_colors::weighting::{PosColorNorm, WeightingKind};
use texture_to_vert_colors::{F, add, dist, kmul, length, normalize};

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
    #[arg(long, default_value_t = PosColorNorm::Add)]
    pos_color_norm: PosColorNorm,

    /// Save SVG of UV to this destination. If empty will not save.
    #[arg(long, default_value_t = String::new())]
    uv_svg: String,

    /// Bake vertex colors to a texture
    #[arg(long, default_value_t = String::new())]
    bake_texture: String,

    /// Use the more exact form of rebaking.
    #[arg(long)]
    approx_rebake: bool,

    /// Resolution of baked texture
    #[arg(long, default_value_t = 1024)]
    bake_res: u32,

    /// Unused for now
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// How many iterations to do for the lazy tutte.
    #[arg(long, default_value_t = 10000)]
    iters: usize,

    /// How much to weigh color compared to geometry
    #[arg(long, default_value_t = 0.)]
    color_weight: F,

    /// How much to clamp the maximum weight of each influence
    #[arg(long, default_value_t = F::INFINITY)]
    max_w: F,
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
        m.triangulate_with_new_edges(|[e0, e1]| {
            new_edges.insert(std::cmp::minmax(e0, e1));
        });
        let (s, t) = m.normalize();
        let (sc, tc) = m.normalize_colors();
        tutte_param(m, new_edges, &args);
        m.denormalize(s, t);
        m.denormalize_colors(sc, tc);
        if let Some(og_faces) = og_faces {
            m.f = og_faces;
        }
    }
    println!(
        "[INFO]: Took {:?} for tutte param with {} ({}) for {}",
        start.elapsed(),
        args.weighting,
        args.pos_color_norm,
        args.input,
    );
    println!("[INFO]: Saved to {}", args.output);

    if !args.uv_svg.is_empty() {
        assert!(
            args.uv_svg.ends_with(".svg"),
            "Only SVG export is supported, incorrect extension for {}",
            args.uv_svg
        );
        let m0 = &scene.meshes[0];
        if let Err(e) = pars3d::svg::save_uv(args.uv_svg, &m0.uv[args.target_uv], &m0.f, 0.4) {
            eprintln!("Failed to save SVG due to {e:?}");
        }
    }

    if !args.bake_texture.is_empty() {
        let m0 = &scene.meshes[0];
        let mut img = if args.approx_rebake {
            pars3d::coloring::bake_vertex_colors_to_texture(
                [args.bake_res, args.bake_res],
                &m0.uv[args.target_uv],
                &m0.f,
                &m0.vert_colors,
            )
        } else {
            pars3d::coloring::bake_vertex_colors_to_texture_exact(
                [args.bake_res, args.bake_res],
                &m0.v,
                &m0.uv[args.target_uv],
                &m0.f,
                &m0.vert_colors,
            )
        };
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

pub fn tutte_param(mesh: &mut Mesh, new_edges: BTreeSet<[usize; 2]>, args: &Args) {
    let mut vert_adj = args
        .weighting
        .vertex_weights(&mesh, args.pos_color_norm, args.color_weight)
        .expect("Failed to construct vertex adjacency");
    // remove influence of introduced edges with length or uniform weighting
    if matches!(
        args.weighting,
        WeightingKind::Length | WeightingKind::Uniform
    ) {
        for vi in 0..mesh.v.len() {
            let (adj_vs, adj_ds) = vert_adj.adj_data_mut(vi);
            for i in 0..adj_vs.len() {
                let e = std::cmp::minmax(vi, adj_vs[i] as usize);
                if new_edges.contains(&e) {
                    adj_ds[i] = 0.;
                }
            }
        }
    }
    let vert_adj = vert_adj;
    let (num_loops, bd_loops) = vert_adj.boundary_loops(&mesh);
    assert_eq!(num_loops, 1, "{bd_loops:?}");

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
    mesh.uv[args.target_uv].resize(mesh.v.len(), [0.; 2]);
    let mut uvs = &mut mesh.uv[args.target_uv];
    for (i, uv) in uvs.iter_mut().enumerate() {
        let i = i as F;
        // stupid randomness but it probably works better than setting everything to 0.5
        let rand_x = (i * 238471.32 + 11.45).sin();
        let rand_y = (i * 15437.65 + 2.13).cos();
        assert!(rand_x.is_finite() && rand_y.is_finite());
        assert!((-1.0..=1.0).contains(&rand_x));
        assert!((-1.0..=1.0).contains(&rand_y));
        let r = length([rand_x, rand_y]).min(1.);
        let [rand_x, rand_y] = kmul(0.99 * r, normalize([rand_x, rand_y]));
        assert!((-1.0..=1.0).contains(&rand_x));
        assert!((-1.0..=1.0).contains(&rand_y));
        *uv = [rand_x, rand_y].map(|v| (v + 1.) / 2.);
    }

    for (vi, l) in bd {
        uvs[vi] = [(l.cos() + 1.) / 2., (l.sin() + 1.) / 2.];
    }
    let mut buf = uvs.clone();

    /*
    let mut triplets = vec![];
    let nv = mesh.v.len();
    for vi in 0..nv {
        if bd_loops.contains_key(&vi) {
            triplets.push(([vi, vi], 1.));
            continue;
        }
        let mut total_w = 0.;
        for (adj, w) in vert_adj.adj_data(vi) {
            assert_ne!(adj as usize, vi);
            triplets.push(([vi, adj as usize], w));
            total_w += w;
        }
        assert!(total_w.is_finite());
        assert_ne!(total_w, 0.);
        triplets.push(([vi, vi], -total_w));
    }
    triplets.sort_unstable_by_key(|a| a.0);
    triplets.dedup_by_key(|a| a.0);
    let csc =
        sparse_lu::Csc::from_triplets(nv, nv, &mut triplets).expect("Failed to construct csc?");
    println!("Starting decomposition");
    let lu = sparse_lu::LeftLookingLUFactorization::new(&csc);

    let t = std::time::Instant::now();
    println!("Starting to solve");
    for i in 0..1 {
        lu.solve_arr(uvs, &mut buf);
    }
    if true {
        return;
    }
    */

    // TODO here set up array instead of using stupid iterations
    use indicatif::ProgressIterator;
    for _it in (0..args.iters).progress() {
        buf.fill([0.; 2]);
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
        buf.par_iter_mut().enumerate().for_each(|(vi, dst)| {
            let mut total_w = 0.;
            for (adj, w) in vert_adj.adj_data(vi) {
                let w = w.min(args.max_w);
                // negative values cause this to explode
                total_w += w;
                //assert!(w.is_finite(), "{w}");
                *dst = add(
                    *dst,
                    kmul(w as F, unsafe { *uvs.get_unchecked(adj as usize) }),
                );
            }
            debug_assert_ne!(total_w, 0.);
            *dst = kmul(total_w.recip(), *dst);
            debug_assert!(dst[0].is_finite());
            debug_assert!(dst[1].is_finite(), "{total_w:?} {:?}", *dst);
        });
        for (&b, _) in &bd_loops {
            buf[b] = uvs[b];
        }

        std::mem::swap(&mut buf, &mut uvs);
    }

    assert!(
        uvs.iter()
            .copied()
            .all(|uv| uv[0].is_finite() && uv[1].is_finite()),
        "{:?}",
        uvs.iter()
            .copied()
            .filter(|uv| !(uv[0].is_finite() && uv[1].is_finite()))
            .collect::<Vec<_>>()
    );
}
