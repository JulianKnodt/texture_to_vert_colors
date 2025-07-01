use super::{F, add, dot, kmul, length, normalize, sub};
use crate::clustering::luma;
use crate::quadric::{AttrWeights, Quadric};
use pars3d::FaceKind;

/// For each face on an input mesh, compute a vector field
pub fn texture_grad_field(m: &pars3d::Mesh, area_weight: bool) -> Vec<[F; 3]> {
    let attr_ws = AttrWeights { ws: [1.] };
    assert!(!m.vert_colors.is_empty());

    let mut out = vec![[0.; 3]; m.f.len()];
    for (fi, f) in m.f.iter().enumerate() {
        let area = if area_weight { f.area(&m.v) } else { 1. };
        if area < 1e-20 {
            continue;
        }
        let n = f.normal(&m.v);
        macro_rules! q_n_attribs(
          ($vis: expr) => {{
            Quadric::n_attribs(
                n,
                $vis.map(|vi| m.v[vi]),
                $vis.map(|vi| [luma(m.vert_colors[vi])]),
                attr_ws,
            )
          }}
        );

        // add attributes as well
        let q_attr = match f {
            FaceKind::Tri(vis) => q_n_attribs!(vis),
            FaceKind::Quad(vis) => q_n_attribs!(vis),
            FaceKind::Poly(p) => Quadric::dyn_attribs(
                n,
                p.len(),
                |vi| m.v[vi],
                |vi| [luma(m.vert_colors[vi])],
                attr_ws,
            ),
        };

        let grad = q_attr.g[0];
        let grad = kmul(area, grad);
        out[fi] = grad;
    }

    out
}

pub fn dir_field_relaxation(
    dir_field: &mut [[F; 3]],
    adj: pars3d::adjacency::Adj<F>,
    normals: impl Fn(usize) -> [F; 3],
    iters: usize,
) {
    for (i, d) in dir_field.iter_mut().enumerate() {
        let n = normalize(normals(i));
        let ortho = sub(*d, kmul(dot(n, *d), n));
        assert!(dot(ortho, n).abs() < 1e-3);
        *d = normalize(ortho);
    }
    let zero_mask = dir_field
        .iter()
        .map(|&v| length(v) == 0.)
        .collect::<Vec<_>>();
    let mut buf = vec![[0.; 3]; dir_field.len()];
    for _ in 0..iters {
        for (i, d) in dir_field.iter().enumerate() {
            if !zero_mask[i] {
                buf[i] = *d;
                continue;
            }
            let mut dir = [0.; 3];
            let n = normalize(normals(i));
            for (adj, w) in adj.adj_data(i) {
                if w == 0. {
                    continue;
                }
                let curr_dir = dir_field[adj as usize];
                //let ortho = sub(curr_dir, kmul(dot(n, curr_dir), n));
                dir = add(dir, kmul(w, curr_dir));
            }

            let dir = normalize(dir);
            let dir = sub(dir, kmul(dot(n, dir), n));
            assert!(dot(dir, n) < 1e-3, "{dir:?} {n:?}");
            buf[i] = normalize(dir);
        }
        dir_field.copy_from_slice(&buf);
        buf.fill([0.; 3]);
    }
}
