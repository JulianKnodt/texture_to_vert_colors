use super::F;

use std::array::from_fn;

#[inline]
pub fn cross([x, y, z]: [F; 3], [a, b, c]: [F; 3]) -> [F; 3] {
    [y * c - z * b, z * a - x * c, x * b - y * a]
}

pub fn cross_2d([x, y]: [F; 2], [a, b]: [F; 2]) -> F {
    x * b - y * a
}

#[test]
fn test_cross() {
    assert_eq!(cross([0., 0., 1.], [1., 0., 0.]), [0., 1., 0.]);
    assert_eq!(cross([0., 0., 2.], [1., 0., 0.]), [0., 2., 0.]);
    assert_eq!(cross([1., 0., 0.], [0., 0., 1.]), [0., -1., 0.]);
}

#[inline]
pub fn add<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    from_fn(|i| a[i] + b[i])
}

#[inline]
pub fn add_assign<const N: usize>(a: &mut [F; N], b: [F; N]) {
    for i in 0..N {
        a[i] += b[i];
    }
}

#[inline]
pub fn sub<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    from_fn(|i| a[i] - b[i])
}

/// L-2 norm of a vector
pub fn length<const N: usize>(v: [F; N]) -> F {
    dot(v, v).sqrt()
}

/// Squared L-2 norm of a vector
pub fn len_sq<const N: usize>(v: [F; N]) -> F {
    dot(v, v)
}

/// Euclidean distance between vectors
pub fn dist<const N: usize>(a: [F; N], b: [F; N]) -> F {
    length(sub(a, b))
}

/// Squared Euclidean distance between vectors
pub fn dist_sq<const N: usize>(a: [F; N], b: [F; N]) -> F {
    len_sq(sub(a, b))
}

pub fn norm_inf<const N: usize>(v: [F; N]) -> F {
    v.into_iter().map(F::abs).max_by(F::total_cmp).unwrap()
}

pub fn to_3<const N: usize>(v: [F; N]) -> [F; 3] {
    assert!(N >= 3);
    [v[0], v[1], v[2]]
}

/*
pub fn concat<const N: usize, const M: usize>(a: [F; N], b: [F; M]) -> [F; N + M]
where
    [(); N + M]:,
{
    std::array::from_fn(|i| if i < N { a[i] } else { b[i - N] })
}
*/

#[inline]
pub fn normalize<const N: usize>(v: [F; N]) -> [F; N] {
    let sum: F = v.iter().map(|v| v * v).sum();
    if sum < 1e-20 {
        return [0.; N];
    }
    let s = sum.sqrt().recip();
    v.map(|v| v * s)
}

#[inline]
pub fn dot<const N: usize>(a: [F; N], b: [F; N]) -> F {
    (0..N).map(|i| a[i] * b[i]).sum()
}

#[inline]
pub fn kmul<const N: usize>(k: F, xyz: [F; N]) -> [F; N] {
    xyz.map(|v| v * k)
}

#[inline]
pub fn minmax<const N: usize>(vs: impl Iterator<Item = [F; N]>) -> [[F; N]; 2] {
    vs.fold([[F::INFINITY; N], [F::NEG_INFINITY; N]], |[min, max], n| {
        use std::array::from_fn;
        [from_fn(|i| min[i].min(n[i])), from_fn(|i| max[i].max(n[i]))]
    })
}

/// Computes the conjugate for inverse rotation of a quaternion.
#[inline]
pub fn conj([x, y, z, w]: [F; 4]) -> [F; 4] {
    [-x, -y, -z, w]
}

/// Multiplies two quaternions together
pub fn quat_mul([r1, r2, r3, r0]: [F; 4], [s1, s2, s3, s0]: [F; 4]) -> [F; 4] {
    [
        r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2,
        r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1,
        r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0,
        r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3,
    ]
}

pub fn quat_rot([x, y, z]: [F; 3], quat: [F; 4]) -> [F; 3] {
    let v = [x, y, z, 0.];
    let [a, b, c, _] = quat_mul(quat_mul(quat, v), conj(quat));
    [a, b, c]
}

#[inline]
pub fn tri_area(a: [F; 3], b: [F; 3], c: [F; 3]) -> F {
    length(cross(sub(a, b), sub(a, c))) / 2.
}

#[inline]
pub fn poly_area(mut v: impl Iterator<Item = [F; 3]>) -> F {
    let mut acc = 0.;
    let Some(fst) = v.next() else {
        return 0.;
    };
    let Some(mut v0) = v.next() else {
        return 0.;
    };
    for v1 in v {
        acc += tri_area(fst, v0, v1);
        v0 = v1;
    }
    acc
}

pub fn orthogonal(v: [F; 3]) -> [F; 3] {
    assert!(v.iter().any(|&v| v != 0.));
    let [x, y, z] = v.map(F::abs);

    let other = if x <= y && x <= z {
        [1., 0., 0.]
    } else if y <= x && y <= z {
        [0., 1., 0.]
    } else {
        [0., 0., 1.]
    };
    cross(v, other)
}

#[inline]
pub fn quat_from_to(s: [F; 3], t: [F; 3]) -> [F; 4] {
    let ns = normalize(s);
    let d = dot(ns, normalize(t));
    // opposite directions
    if d < -1. + 1e-5 {
        let [ox, oy, oz] = normalize(orthogonal(ns));
        return [ox, oy, oz, 0.];
    }

    let v = cross(t, s);
    normalize([v[0], v[1], v[2], 1. + d])
}

#[inline]
pub fn quat_from_axis_angle(axis: [F; 3], angle: F) -> [F; 4] {
    let s = (angle / 2.).sin();
    let [x, y, z] = axis.map(|v| v * s);
    [x, y, z, (angle / 2.).cos()]
}

/// Computes rotation from the standard xyz basis to this basis, where fwd and up are orthogonal
/// and normalized.
pub fn quat_from_standard(fwd: [F; 3], up: [F; 3]) -> [F; 4] {
    assert!(dot(fwd, up).abs() < 1e-4);
    let r0 = quat_from_to([1., 0., 0.], fwd);
    let r1 = quat_from_to(quat_rot([0., 1., 0.], r0), up);
    quat_mul(r1, r0)
}

pub fn quat_from_basis(fwd: [F; 3], up: [F; 3], b0: [F; 3], b1: [F; 3]) -> [F; 4] {
    assert!(dot(fwd, up).abs() < 1e-4);
    let r0 = quat_from_to(b0, fwd);
    let r1 = quat_from_to(quat_rot(b1, r0), up);
    quat_mul(r1, r0)
}

/// returns each row of the matrix representing a quaternion
pub fn quat_to_mat([x, y, z, w]: [F; 4]) -> [[F; 3]; 3] {
    let qxx = x * x;
    let qyy = y * y;
    let qzz = z * z;
    let qxz = x * z;
    let qxy = x * y;
    let qyz = y * z;
    let qwx = w * x;
    let qwy = w * y;
    let qwz = w * z;

    [
        [1. - 2. * (qyy + qzz), 2. * (qxy - qwz), 2. * (qxz + qwy)],
        [2. * (qxy + qwz), 1. - 2. * (qxx + qzz), 2. * (qyz - qwx)],
        [2. * (qxz - qwy), 2. * (qyz + qwx), 1. - 2. * (qxx + qyy)],
    ]
}

pub fn transpose3(s: [[F; 3]; 3]) -> [[F; 3]; 3] {
    let [[a, b, c], [d, e, f], [g, h, i]] = s;
    [[a, d, g], [b, e, h], [c, f, i]]
}

#[test]
fn test_quat() {
    let q = quat_from_to([1., 0., 0.], [0., 1., 0.]);
    let rot = quat_rot([1., 0., 0.], q);
    assert!(length(sub(rot, [0., 1., 0.])) < 1e-3);
}

#[test]
fn test_quat_from_to_parallel() {
    let e0 = [1., 0., 0.];
    let rot = quat_from_to(e0, e0);
    assert_eq!(quat_rot(e0, rot), e0);
    use core::ops::Neg;
    let neg_e0 = e0.map(Neg::neg);
    let opp_rot = quat_from_to(e0, neg_e0);
    assert_ne!(length(opp_rot), 0.);
    assert_eq!(quat_rot(e0, opp_rot), neg_e0, "{opp_rot:?}");
}

#[test]
pub fn test_quat_basis() {
    let tgt = normalize([0., 0.5, 0.5]);
    let up = [1., 0., 0.];

    let q = quat_from_standard(tgt, up);

    let r0 = quat_rot([1., 0., 0.], q);
    let r1 = quat_rot([0., 1., 0.], q);
    assert!(length(sub(r0, tgt)) < 1e-4);
    assert!(length(sub(r1, up)) < 1e-4);
}

#[test]
fn test_quat_from_standard() {
    let fwd = [0., 1., 0.];
    let up = [1., 0., 0.];

    let q = quat_from_standard(fwd, up);
    assert!((length(q) - 1.).abs() < 1e-4, "{q:?}");
}

#[test]
fn test_identity_quat() {
    let n = normalize([1.; 3]);
    let q = quat_from_to(n, n);
    assert_eq!(quat_rot(n, q), n);
}
