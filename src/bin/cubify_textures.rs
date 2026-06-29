#![allow(unused)]

use clap::Parser;
use pars3d::adjacency::Adj;
use pars3d::{FaceKind, Mesh};
use sparse_lu::{Csc, LeftLookingLUFactorization};
use texture_to_vert_colors::{
    F, add, kmul, sub,
    weighting::{PosColorNorm, WeightingKind},
};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh with per vertex offsets stored in the R channel of the vertex colors
    #[arg(long, short)]
    input: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,

    /// Weight to use for cubeness (in [0,1])
    #[arg(long, default_value_t = 0.2)]
    cubeness: F,
}

fn main() {
    let args = Args::parse();
    let mut scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));

    let start = std::time::Instant::now();
    for m in scene.meshes.iter_mut() {
        let (constraint, v) = {
            // XXX TEMP delete this
            let highest_idx =
                m.v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1[1].partial_cmp(&b.1[1]).unwrap())
                    .unwrap()
                    .0;
            (highest_idx, [1., 1., 0.])
        };
        let (s, t) = m.normalize();
        let og_f = m.f.clone();
        m.triangulate(0);
        let (lu, lapl) = cube_style_precompute(m);
        let vertex_face_adj = m.vertex_face_adj();
        let mut rots = vec![ident(); m.v.len()];
        let mut rhs = vec![[0.; 3]; m.v.len()];
        let mut buf = vec![[0.; 3]; m.v.len()];
        for _ in 0..100 {
            arap_rhs(&rots, &m.v, &lapl, &mut rhs);
            lu.solve_arr(&mut rhs, &mut buf);
            println!("{:?}\n{:?}", &m.v[0..10], &rhs[0..10]);
            m.v = rhs;
            //m.v[constraint] = v;
            break;
        }
        m.f = og_f;
        m.denormalize(s, t);
    }

    println!("Elapsed {:?}", start.elapsed());
    pars3d::save(&args.output, &scene, true)
        .expect(&format!("Failed to save output to {}", args.output));
}

pub fn cube_style_precompute(m: &mut Mesh) -> (LeftLookingLUFactorization<F>, Adj<F>) {
    let nv = m.v.len();

    m.vertex_normals(Default::default());
    let lapl = WeightingKind::Laplacian
        .vertex_weights(m, PosColorNorm::Add, 0.)
        .expect("Failed to compute?");

    let mut triplets = lapl
        .all_pairs()
        .map(|([i, j], d)| ([i as usize, j as usize], d))
        .chain(
            lapl.all_adj_data()
                .map(|(i, _, d)| ([i as usize, i as usize], -d.iter().copied().sum::<F>())),
        )
        .collect::<Vec<_>>();
    #[allow(non_snake_case)]
    let L =
        Csc::from_triplets(nv, nv, &mut triplets).expect("Failed to construct Laplacian matrix");
    (LeftLookingLUFactorization::new(&L), lapl)
}

fn arap_rhs(rots: &[Mat3], vs: &[[F; 3]], lapl: &Adj<F>, dst: &mut Vec<[F; 3]>) {
    dst.resize(vs.len(), [0.; 3]);
    dst.fill([0.; 3]);
    for i in 0..vs.len() {
        let src_rot = rots[i];
        for (adj, w) in lapl.adj_data(i) {
            let adj = adj as usize;
            let r01 = add_mat(&src_rot, &rots[adj]);
            dst[i] = add(dst[i], kmul(w / 2., vec_mul(&r01, sub(vs[i], vs[adj]))));
        }
    }
}

type Mat3 = [[F; 3]; 3];
fn vec_mul(m: &Mat3, v: [F; 3]) -> [F; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[i][j] * v[j]).sum())
}
fn add_mat(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn ident() -> Mat3 {
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
}
