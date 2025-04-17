use super::F;
use super::{normalize, sub};

pub(crate) fn cross_2d([x, y]: [F; 2], [a, b]: [F; 2]) -> F {
    x * b - y * a
}

const EPS: F = 1e-2;
pub fn tri_2d_contains(p: [F; 2], t: [[F; 2]; 3]) -> bool {
    pars3d::barycentric_2d(p, t)
        .iter()
        .all(|v| (EPS..=(1. - EPS)).contains(v))
}

// positive if cw, negative if ccw
#[inline]
pub fn orient_2d(a: [F; 2], b: [F; 2], p: [F; 2]) -> F {
    assert_ne!(a, b);
    // in case they are exactly equal just output 0.
    if a == p || b == p {
        return 0.;
    }
    let ba = normalize(sub(b, a));
    let pa = normalize(sub(p, a));
    cross_2d(ba, pa)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB<T, const N: usize> {
    min: [T; N],
    max: [T; N],
}

impl<const N: usize> Default for AABB<F, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> AABB<F, N> {
    pub fn new() -> Self {
        Self {
            min: [F::INFINITY; N],
            max: [F::NEG_INFINITY; N],
        }
    }
    pub fn add_point(&mut self, p: [F; N]) {
        for i in 0..N {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }
    pub fn round_to_i32(&self) -> AABB<i32, N> {
        AABB {
            min: self.min.map(|i| i.floor() as i32),
            max: self.max.map(|i| i.ceil() as i32),
        }
    }
    pub fn scale_by(&mut self, x: F, y: F) {
        self.min[0] *= x;
        self.max[0] *= x;

        self.min[1] *= y;
        self.max[1] *= y;
    }
    #[inline]
    pub fn contains_point(&self, p: [F; N]) -> bool {
        p.iter()
            .enumerate()
            .all(|(dim, &c)| self.within_dim(dim, c))
    }
    #[inline]
    fn within_dim(&self, dim: usize, v: F) -> bool {
        ((self.min[dim] + EPS)..=(self.max[dim] - EPS)).contains(&v)
    }
}

impl From<[[F; 2]; 2]> for AABB<F, 2> {
    fn from([a, b]: [[F; 2]; 2]) -> Self {
        AABB {
            min: std::array::from_fn(|i| a[i].min(b[i])),
            max: std::array::from_fn(|i| a[i].max(b[i])),
        }
    }
}

impl AABB<F, 2> {
    #[inline]
    pub fn corners(&self) -> [[F; 2]; 4] {
        [
            self.min,
            [self.min[0], self.max[1]],
            [self.min[1], self.max[0]],
            self.max,
        ]
    }
    #[inline]
    pub fn intersects_tri(&self, [v0, v1, v2]: [[F; 2]; 3]) -> bool {
        // Check if the aabb contains any of the triangle's vertices
        // (if the triangle is contained entirely within the box)
        let tri_in_box =
            self.contains_point(v0) || self.contains_point(v1) || self.contains_point(v2);
        if tri_in_box {
            return true;
        }

        // Check if the box is entirely contained within the triangle
        let box_in_tri = self
            .corners()
            .into_iter()
            .any(|c| tri_2d_contains(c, [v0, v1, v2]));
        if box_in_tri {
            return true;
        }

        // https://stackoverflow.com/questions/99353/how-to-test-if-a-line-segment-intersects-an-axis-aligned-rectange-in-2d
        // (if they partially overlap)
        self.intersects_line(v0, v1) || self.intersects_line(v2, v1) || self.intersects_line(v0, v2)
    }
    pub fn intersects_line(&self, a: [F; 2], b: [F; 2]) -> bool {
        for d in [0, 1] {
            if a[d] > self.max[d] + EPS && b[d] > self.max[d] + EPS {
                return false;
            }
            if a[d] < self.min[d] - EPS && b[d] < self.min[d] - EPS {
                return false;
            }
        }
        let cs = self.corners().map(|c| {
            let o = orient_2d(a, b, c);
            (o, o.signum() as i8)
        });
        assert!(!cs.iter().any(|c| c.1 == 0));

        cs[1..].iter().any(|&c| c.1 != cs[0].1)
    }
}

impl AABB<i32, 2> {
    pub fn iter_coords(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        let [lx, ly] = self.min;
        let [hx, hy] = self.max;
        (ly..hy).flat_map(move |y| (lx..hx).map(move |x| [x, y]))
    }
    pub fn expand_by(&mut self, v: i32) {
        self.min = self.min.map(|val| val - v);
        self.max = self.max.map(|val| val + v);
    }
}
