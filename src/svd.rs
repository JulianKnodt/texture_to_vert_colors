use super::sym::SymMatrix3;
use super::{F, kmul, quat_to_mat};

fn approx_givens_quat(a11: F, a12: F, a22: F) -> [F; 2] {
    const G: F = 3. + 2. * std::f64::consts::SQRT_2 as F;
    //let G = 3. + (8. as F).sqrt();

    let ch = 2. * (a11 - a22);
    let sh = a12;

    if ch == 0. && sh == 0. {
        return [1., 0.];
    }

    let sh2 = sh * sh;
    let ch2 = ch * ch;
    if G * sh2 < ch2 {
        let w = 1. / (sh2 + ch2).sqrt();
        [w * ch, w * sh]
    } else {
        const PI: F = std::f64::consts::PI as F;
        [(PI / 8.).cos(), (PI / 8.).sin()]
    }
}

fn jacobi_conjugation(x: usize, y: usize, z: usize, s: &mut [F; 6], q: [F; 4]) -> [F; 4] {
    let [ch, sh] = approx_givens_quat(s[0], s[1], s[2]);

    let scale = ch * ch + sh * sh;
    let a = (ch * ch - sh * sh) / scale;
    let b = (2. * ch * sh) / scale;

    let ns = conj_sym(s, a, b);

    let tmp = kmul(sh, super::to_3(q));
    let sh = sh * q[3];
    let mut q = kmul(ch, q);

    q[z] += sh;
    q[3] -= tmp[z];
    q[x] += tmp[y];
    q[y] -= tmp[x];

    let [s11, s21, s22, s31, s32, s33] = ns;
    *s = [s22, s32, s33, s21, s31, s11];
    q
}

fn conj_sym(&[s11, s21, s22, s31, s32, s33]: &[F; 6], a: F, b: F) -> [F; 6] {
    let cse1 = -b * s11 + a * s21;
    let cse2 = -b * s21 + a * s22;
    [
        a * (a * s11 + b * s21) + b * (a * s21 + b * s22),
        a * cse1 + b * cse2,
        -b * cse1 + a * cse2,
        a * s31 + b * s32,
        -b * s31 + a * s32,
        s33,
    ]
}

fn jacobi_eigen(mut s: [F; 6]) -> [F; 4] {
    let mut q = [0., 0., 0., 1.].map(F::from);
    for _ in 0..15 {
        let nq = jacobi_conjugation(0, 1, 2, &mut s, q);
        let nq = jacobi_conjugation(1, 2, 0, &mut s, nq);
        let nq = jacobi_conjugation(2, 0, 1, &mut s, nq);
        q = nq;
    }
    q
}

/// Computes the eigenvectors of a symmetric matrix using jacobi iterations.
pub fn eigen_jacobi(s: SymMatrix3) -> [[F; 3]; 3] {
    let [s00, s01, s02, s11, s12, s22] = s.data;
    let alt_order = [s00, s01, s11, s02, s12, s22];
    super::transpose3(quat_to_mat(jacobi_eigen(alt_order)))
}

#[test]
fn test_jacobi_eigen() {
    let eye = SymMatrix3::ident();
    let [e0, e1, e2] = eigen_jacobi(eye);
    assert_eq!(e0, [1., 0., 0.]);
    assert_eq!(e1, [0., 1., 0.]);
    assert_eq!(e2, [0., 0., 1.]);

    let s = SymMatrix3::new([1., 2., 3., 4., 5., 6.]);
    let vs = eigen_jacobi(s);
    use super::normalize;
    for v in vs {
        assert!(super::dot(v, normalize(s.vec_mul(v))).abs() > 0.9999);
    }

    let degen = SymMatrix3::outer(normalize([0.5, 0.5, 0.]));
    let degen_vs = eigen_jacobi(degen);
    for v in degen_vs {
        let d = super::dot(v, normalize(degen.vec_mul(v))).abs();
        assert!(d > 0.9999 || d == 0., "{d}");
    }
}
