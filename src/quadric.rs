use super::sym::SymMatrix3;
use super::{F, add, dot, kmul, sub};

use core::array::from_fn;
use core::ops::{Add, AddAssign, Mul, MulAssign};

pub const N_ATTRIB: usize = 3 + 3;

/// Weights for each attribute. Joints are kept separate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttrWeights<const N: usize> {
    pub ws: [F; N],
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QuadricAccumulator {
    a: SymMatrix3,
    b: [F; 3],
    area: F,

    ggt: SymMatrix3,
    gd: [F; 3],

    // volume constraints
    nv: [F; 3],
    dv: F,
}

impl<const N: usize> AddAssign<Quadric<N>> for QuadricAccumulator {
    fn add_assign(&mut self, q: Quadric<N>) {
        self.a = self.a + q.a;
        self.b = add(self.b, q.b);
        self.area += q.area;

        for i in 0..N {
            self.ggt = self.ggt + SymMatrix3::outer(q.g[i]);
            self.gd = add(self.gd, kmul(q.d[i], q.g[i]));
        }

        self.nv = add(self.nv, q.nv);
        self.dv += q.dv;
    }
}

impl QuadricAccumulator {
    pub fn point(&self) -> [F; 3] {
        if self.area == 0. {
            return [0.; 3];
        }
        let inv_a = self.area.recip();
        debug_assert!(inv_a.is_finite(), "{}", self.area);

        let a = self.a - self.ggt * inv_a;
        let b = sub(self.b, kmul(inv_a, self.gd));

        let ([e0, e1, e2], [v0, v1, v2]) = a.eigen();
        [(e0, v0), (e1, v1), (e2, v2)]
            .into_iter()
            .map(|(e, v)| {
                if e.abs() < 1e-8 {
                    return [0.; 3];
                }
                kmul(-dot(b, v) / e, v)
            })
            .fold([0.; 3], add)
    }
    pub fn invert(&self) -> [F; 3] {
        if self.area < F::EPSILON {
            return [0.; 3];
        }
        let inv_a = self.area.recip();
        assert!(inv_a.is_finite(), "{}", self.area);

        let a = self.a - self.ggt * inv_a;
        let b = sub(self.b, kmul(inv_a, self.gd));

        invert_quadric(a, b).unwrap_or_else(|| {
            let ([e0, e1, e2], [v0, v1, v2]) = a.eigen();
            [(e0, v0), (e1, v1), (e2, v2)]
                .into_iter()
                .map(|(e, v)| {
                    if e.abs() < 1e-8 {
                        return [0.; 3];
                    }
                    kmul(-dot(b, v) / e, v)
                })
                .fold([0.; 3], add)
        })
    }
    pub fn point_with_volume(&self) -> [F; 3] {
        if self.area < F::EPSILON {
            return [0.; 3];
        }
        self.point_with_volume_opt().unwrap_or_else(|| self.point())
    }
    pub fn point_with_volume_opt(&self) -> Option<[F; 3]> {
        if self.area < F::EPSILON {
            return None;
        }
        let inv_a = self.area.recip();
        assert!(inv_a.is_finite(), "{}", self.area);
        let a = self.a - self.ggt * inv_a;

        let b = sub(self.b, kmul(inv_a, self.gd));
        invert_quadric_volume(a, b, self.nv, self.dv)
    }
}

fn invert_quadric(a: SymMatrix3, [r0, r1, r2]: [F; 3]) -> Option<[F; 3]> {
    let [a, b, c, d, e, f] = a.data;

    let ad = a * d;
    let ae = a * e;
    let af = a * f;
    let bc = b * c;
    let be = b * e;
    let bf = b * f;
    let df = d * f;
    let ce = c * e;
    let cd = c * d;

    let be_cd = be - cd;
    let bc_ae = bc - ae;
    let ce_bf = ce - bf;

    let inv_denom = a * df + 2. * b * ce - ae * e - bf * b - cd * c;
    const EPS: F = 1e-4;
    if inv_denom < EPS {
        return None;
    }
    assert!(inv_denom >= EPS, "{inv_denom}");
    let denom = inv_denom.recip();
    let numer = [
        r0 * (df - e * e) + r1 * ce_bf + r2 * be_cd,
        r0 * ce_bf + r1 * (af - c * c) + r2 * bc_ae,
        r0 * be_cd + r1 * bc_ae + r2 * (ad - b * b),
    ];

    Some(kmul(denom, numer))
}

fn invert_quadric_volume(
    a: SymMatrix3,
    [r0, r1, r2]: [F; 3],
    [vx, vy, vz]: [F; 3],
    dv: F,
) -> Option<[F; 3]> {
    let [mxx, mxy, mxz, myy, myz, mzz] = a.data;

    let det2_01_01 = mxx * myy - mxy * mxy;
    let det2_01_02 = mxx * myz - mxz * mxy;
    let det2_01_12 = mxy * myz - mxz * myy;
    let det2_01_03 = mxx * vy - vx * mxy;
    let det2_01_13 = mxy * vy - vx * myy;
    let det2_01_23 = mxz * vy - vx * myz;

    // 3x3 sub-determinants required to calculate 4x4 determinant
    let invx = mzz * det2_01_13 - myz * det2_01_23 - vz * det2_01_12;
    let invy = mxz * det2_01_23 - mzz * det2_01_03 + vz * det2_01_02;
    let invz = myz * det2_01_03 - mxz * det2_01_13 - vz * det2_01_01;

    let det = invx * vx + invy * vy + invz * vz;

    if det < 1e-6 {
        return None;
    }

    let denom = det.recip();

    // remaining 2x2 sub-determinants
    let det2_03_02 = mxx * vz - mxz * vx;
    let det2_03_12 = mxy * vz - mxz * vy;
    let det2_13_12 = myy * vz - myz * vy;

    let det2_03_03 = -vx * vx;
    let det2_03_13 = -vx * vy;
    let det2_03_23 = -vx * vz;

    let det2_13_13 = -vy * vy;
    let det2_13_23 = -vy * vz;

    // remaining 3x3 sub-determinants
    let imxx = mzz * det2_13_13 - myz * det2_13_23 - vz * det2_13_12;
    let imxy = myz * det2_03_23 - mzz * det2_03_13 + vz * det2_03_12;
    let imyy = mzz * det2_03_03 - mxz * det2_03_23 - vz * det2_03_02;

    let imxz = vy * det2_01_23 - vz * det2_01_13;
    let imyz = vz * det2_01_03 - vx * det2_01_23;
    let imzz = vx * det2_01_13 - vy * det2_01_03;

    let numer = [
        r0 * imxx + r1 * imxy + r2 * imxz - invx * dv,
        r0 * imxy + r1 * imyy + r2 * imyz - invy * dv,
        r0 * imxz + r1 * imyz + r2 * imzz - invz * dv,
    ];
    Some(kmul(denom, numer))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadric<const N: usize = N_ATTRIB> {
    pub a: SymMatrix3,
    pub b: [F; 3],
    c: F,

    g: [[F; 3]; N],
    d: [F; N],

    pub area: F,

    nv: [F; 3],
    dv: F,
}

impl Quadric<0> {
    pub fn cost(&self, p: [F; 3]) -> F {
        let quadratic = dot(p, self.a.vec_mul(p));
        quadratic + 2. * dot(self.b, p) + self.c
    }
    pub fn point_no_attrib(&self) -> [F; 3] {
        let ([e0, e1, e2], [v0, v1, v2]) = self.a.eigen();
        [(e0, v0), (e1, v1), (e2, v2)]
            .into_iter()
            .map(|(e, v)| {
                if e.abs() < 1e-8 {
                    return [0.; 3];
                }
                kmul(-dot(self.b, v) / e, v)
            })
            .fold([0.; 3], add)
    }
}

impl<const N: usize> Quadric<N> {
    pub fn new_plane(v: [F; 3], n: [F; 3], area: F) -> Self {
        let a = SymMatrix3::outer(n);
        let dist = -dot(n, v);
        let b = kmul(dist, n);
        let c = dist * dist;

        Self {
            a,
            b,
            c,
            g: [[0.; 3]; N],
            d: [0.; N],

            area: 1.,

            nv: kmul(area / 3., n),
            dv: area * c / 3.,
        }
    }
    pub fn new_from_bary([b0, b1]: [F; 2], area: F) -> Self {
        let b2 = 1. - b0 - b1;
        Self::new_plane([b0, b1, b2], [0., 0., 1.], area)
    }

    /// For a single vertex, add a constant `d` value so that the attributes default to that
    /// value.
    pub fn degen_attr(attrib: [F; N], weights: AttrWeights<N>) -> Self {
        let d = from_fn(|i| attrib[i] * weights.ws[i].max(0.));
        Self {
            a: SymMatrix3::zero(),
            b: [0.; 3],
            c: d.into_iter().map(|d| d * d).sum::<F>(),
            g: [[0.; 3]; N],
            d,
            area: 0.,
            nv: [0.; 3],
            dv: 0.,
        }
    }
    pub fn n_attribs<const V: usize>(
        [nx, ny, nz]: [F; 3],
        points: [[F; 3]; V],
        attribs: [[F; N]; V],

        weights: AttrWeights<N>,
    ) -> Self
    where
        [(); V + 1]:,
    {
        assert!(V > 2);
        let pn: [[F; 4]; V + 1] = from_fn(|i| {
            if i == V {
                return [nx, ny, nz, 0.];
            }
            let [x, y, z] = points[i];
            [x, y, z, 1.]
        });
        let (q, r) = least_sq::mgs_qr(pn);
        debug_assert!(q.iter().all(|col| col.iter().all(|v| v.is_finite())));
        debug_assert!(r.iter().all(|col| col.iter().all(|v| v.is_finite())));
        //assert!(q.iter().any(|col| col.iter().any(|&v| v > 0.)));
        //assert!(r.iter().any(|col| col.iter().any(|&v| v > 0.)));

        let gd: [[F; 4]; N] = from_fn(|i| {
            let w = weights.ws[i];
            if w <= 0. {
                return [0.; 4];
            }
            let a_s = from_fn(|pi| {
                if pi == V {
                    return 0.;
                }
                w * attribs[pi][i]
            });
            debug_assert!(a_s.iter().copied().all(F::is_finite));
            least_sq::qr_solve(q, r, a_s)
        });
        let g = gd.map(|[g0, g1, g2, _]| [g0, g1, g2]);
        let d = gd.map(|[_, _, _, d]| d);

        let a = g
            .into_iter()
            .map(SymMatrix3::outer)
            .fold(SymMatrix3::zero(), Add::add);
        let b = (0..N).map(|i| kmul(d[i], g[i])).fold([0.; 3], add);
        let c = d.into_iter().map(|d| d * d).sum::<F>();

        Self {
            a,
            b,
            c,
            g,
            d,
            area: 0.,
            nv: [0.; 3],
            dv: 0.,
        }
    }

    /// Construct a quadric for a set of attributes
    pub fn dyn_attribs(
        [nx, ny, nz]: [F; 3],
        np: usize,
        points: impl Fn(usize) -> [F; 3],
        attribs: impl Fn(usize) -> [F; N],
        weights: AttrWeights<N>,
    ) -> Self {
        assert!(np > 2);
        assert!(np > 4, "Internal error, expected > 4, got: {np}");
        // TODO maybe make these a small vec of size 4?
        let mut q_buf = vec![];
        let mut pn = vec![[0.; 4]; np + 1];
        for pi in 0..np {
            let [x, y, z] = points(pi);
            pn[pi] = [x, y, z, 1.];
        }
        pn[np] = [nx, ny, nz, 0.];
        let r = least_sq::dyn_mgs_qr(&mut pn, &mut q_buf);
        assert!(
            q_buf
                .iter()
                .all(|col| col.iter().copied().all(F::is_finite))
        );
        assert!(r.iter().all(|col| col.iter().copied().all(F::is_finite)));

        let gd: [[F; 4]; N] = from_fn(|i| {
            let w = weights.ws[i];
            if w <= 0. {
                return [0.; 4];
            }
            least_sq::dyn_qr_solve(
                &q_buf,
                r,
                |pi| if pi == np { 0. } else { w * attribs(pi)[i] },
            )
        });
        let g = gd.map(|[g0, g1, g2, _]| [g0, g1, g2]);
        let d = gd.map(|[_, _, _, d]| d);

        let a = g
            .into_iter()
            .map(SymMatrix3::outer)
            .fold(SymMatrix3::zero(), Add::add);
        let b = (0..N).map(|i| kmul(d[i], g[i])).fold([0.; 3], add);
        let c = d.into_iter().map(|d| d * d).sum::<F>();

        Self {
            a,
            b,
            c,
            g,
            d,
            area: 0.,

            nv: [0.; 3],
            dv: 0.,
        }
    }
    pub fn zero() -> Self {
        Self {
            a: SymMatrix3::zero(),
            b: [0.; 3],
            c: 0.,
            g: [[0.; 3]; N],
            d: [0.; N],
            area: 0.,
            nv: [0.; 3],
            dv: 0.,
        }
    }
    /// Compute cost with attributes
    pub fn cost_attrib(&self, v: [F; 3], attrs: [F; N], ws: AttrWeights<N>) -> F {
        let mut a_v = self.a.vec_mul(v);
        for i in 0..N {
            a_v = sub(a_v, kmul(attrs[i] * ws.ws[i], self.g[i]));
        }
        let mut vt_a_v = dot(a_v, v);

        let mut bt_v = dot(v, self.b);
        for i in 0..N {
            let s_i = attrs[i] * ws.ws[i];
            let t = self.area * s_i - dot(self.g[i], v);
            vt_a_v += s_i * t;
            bt_v -= self.d[i] * s_i;
        }

        vt_a_v + 2. * bt_v + self.c
    }

    pub fn invert(&self) -> Option<[F; 3]> {
        let [a, b, c, d, e, f] = self.a.data;
        let [r0, r1, r2] = self.b;

        let ad = a * d;
        let ae = a * e;
        let af = a * f;
        let bc = b * c;
        let be = b * e;
        let bf = b * f;
        let df = d * f;
        let ce = c * e;
        let cd = c * d;

        let be_cd = be - cd;
        let bc_ae = bc - ae;
        let ce_bf = ce - bf;

        let inv_denom = a * df + 2. * b * ce - ae * e - bf * b - cd * c;
        const EPS: F = 1e-6;
        if inv_denom < EPS {
            return None;
        }
        assert!(inv_denom >= EPS, "{inv_denom}");
        let denom = inv_denom.recip();
        let numer = [
            r0 * (df - e * e) + r1 * ce_bf + r2 * be_cd,
            r0 * ce_bf + r1 * (af - c * c) + r2 * bc_ae,
            r0 * be_cd + r1 * bc_ae + r2 * (ad - b * b),
        ];

        Some(kmul(denom, numer))
    }

    /*
    pub fn attributes_opt(&self, p: [F; 3], ws: AttrWeights<N>) -> [Option<F>; N] {
        from_fn(|i| {
            let w = ws.ws[i];
            if w <= 0. {
                return 0.;
            }
            let s = dot(self.g[i], p) + self.d[i];
            debug_assert!(s.is_finite(), "{p:?} {:?} {:?}", self.g[i], self.d[i]);
            let denom = w * self.area;
            if denom.abs() < 1e-4 {
                return None;
            }
            assert!(
                denom > 1e-14,
                "Expected non-degenerate denom, denom = {denom} w = {w} area = {}",
                self.area
            );
            let out = s / denom;
            assert!(out.is_finite(), "{s}/{denom}");
            Some(out)
        })
    }
    */
    pub fn attributes(&self, p: [F; 3], ws: AttrWeights<N>) -> [F; N] {
        from_fn(|i| {
            let w = ws.ws[i];
            if w <= 0. {
                return 0.;
            }
            let s = dot(self.g[i], p) + self.d[i];
            debug_assert!(s.is_finite(), "{p:?} {:?} {:?}", self.g[i], self.d[i]);
            let denom = w * self.area;
            if denom.abs() < 1e-8 {
                todo!();
                return 0.;
            }
            assert!(
                denom > 1e-14,
                "Expected non-degenerate denom, denom = {denom} w = {w} area = {}",
                self.area
            );
            let out = s / denom;
            assert!(out.is_finite(), "{s}/{denom}");
            out
        })
    }
}

#[test]
fn test_local_quadric() {
    let q =
        Quadric::<0>::new_plane([0.; 3], [0., 1., 0.]) + Quadric::new_plane([0.; 3], [1., 0., 0.]);
    assert_eq!(q.cost([0., 0., 1.]), 0.);
    assert_eq!(q.cost([0., 0., -1.]), 0.);

    assert_ne!(q.cost([1., 0., 0.]), 0.);
    assert_ne!(q.cost([-1., 0., 0.]), 0.);

    assert_ne!(q.cost([0., 1., 0.]), 0.);
}

impl<const N: usize> Add for Quadric<N> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            a: self.a + o.a,
            b: add(self.b, o.b),
            c: self.c + o.c,

            g: from_fn(|i| add(self.g[i], o.g[i])),
            d: add(self.d, o.d),

            area: self.area + o.area,

            nv: add(self.nv, o.nv),
            dv: self.dv + o.dv,
        }
    }
}

impl<const N: usize> Mul<F> for Quadric<N> {
    type Output = Self;
    fn mul(self, o: F) -> Self {
        Self {
            a: self.a * o,
            b: kmul(o, self.b),
            c: o * self.c,

            g: from_fn(|i| kmul(o, self.g[i])),
            d: kmul(o, self.d),

            area: self.area * o,

            nv: self.nv,
            dv: o * self.dv,
        }
    }
}

impl<const N: usize> AddAssign for Quadric<N> {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl<const N: usize> MulAssign<F> for Quadric<N> {
    fn mul_assign(&mut self, o: F) {
        *self = *self * o;
    }
}

#[test]
fn test_quadric_attr() {
    let ps = [[1., 0., 0.], [1., 0., 1.], [0., 0., 1.]];
    let ws = [0.5];
    let q_attr = Quadric::<1>::n_attribs::<3>([0., 1., 0.], ps, [[0.], [1.], [0.5]], ws) * 2.;
    let q_attr = q_attr + q_attr;

    let out_attr = q_attr.attributes(ps[0], ws);
    assert!(out_attr[0].abs() < 1e-8);

    let out_attr = q_attr.attributes(ps[1], ws);
    assert!((out_attr[0] - 1.).abs() < 1e-8);

    let out_attr = q_attr.attributes(ps[2], ws);
    assert!((out_attr[0] - 0.5).abs() < 1e-8);
}
