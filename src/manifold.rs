use super::inv_map::InverseMap;
use union_find::{AtomicUnionFind, UnionFind, UnionFindOp};

/// A mesh representation which is suitable for collapsing vertices.
/// It can associate data with each vertex, and each edge.
/// Associated edge data is oriented.
#[derive(Debug, Clone)]
pub struct CollapsibleManifold<T, UF: UnionFindOp = UnionFind<u32>> {
    pub(crate) vertices: UF,

    pub edges: Vec<Vec<u32>>,

    inv_map: InverseMap,

    pub data: Vec<T>,
}

impl<T> CollapsibleManifold<T, UnionFind<u32>> {
    pub fn new_with(size: usize, f: impl Fn(usize) -> T) -> Self {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(f(i));
        }
        Self {
            vertices: UnionFind::new_u32(size),
            edges: vec![vec![]; size],
            inv_map: InverseMap::new(size),

            data,
        }
    }

    #[inline]
    pub fn new(size: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            vertices: UnionFind::new_u32(size),
            edges: vec![],
            data: vec![T::default(); size],
            inv_map: InverseMap::new(size),
        }
    }
}

impl<T> CollapsibleManifold<T, AtomicUnionFind> {
    pub fn atomic_new_with(size: usize, f: impl Fn(usize) -> T) -> Self {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(f(i));
        }
        Self {
            vertices: AtomicUnionFind::new(size),
            edges: vec![vec![]; size],
            inv_map: InverseMap::new(size),

            data,
        }
    }
}

impl<T, UF: UnionFindOp> CollapsibleManifold<T, UF> {
    pub(crate) fn new_with_remapping(remap: UF, f: impl Fn(usize) -> T) -> Self {
        let size = remap.capacity();
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(f(i));
        }

        Self {
            vertices: remap,
            edges: vec![vec![]; size],
            inv_map: InverseMap::new(size),

            data,
        }
    }
    pub fn get_new_vertex(&self, old: usize) -> usize {
        self.vertices.find(old)
    }
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
    pub fn vertices(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        (0..self.vertices.capacity())
            .filter(|&vi| !self.is_deleted(vi))
            .map(|vi| (vi, &self.data[vi]))
    }
    pub fn is_deleted(&self, vi: usize) -> bool {
        !self.vertices.is_root(vi)
    }

    /// Adds an edge. For faces, should call `add_face`.
    pub fn add_edge(&mut self, v0: usize, v1: usize) {
        if v0 == v1 {
            return;
        }
        self.edges[v0].push(v1 as u32);
        self.edges[v1].push(v0 as u32);

        // note that this is not using the mapping since edges should only be added ahead of
        // time.
        self.edges[v0].sort_unstable_by_key(|&dst| dst);
        self.edges[v1].sort_unstable_by_key(|&dst| dst);

        self.edges[v0].dedup_by_key(|&mut dst| dst);
        self.edges[v1].dedup_by_key(|&mut dst| dst);
    }

    pub fn add_face(&mut self, face: &[usize]) {
        let n = face.len();
        for i in 0..n {
            let e0 = face[i];
            let e1 = face[(i + 1) % n];
            self.add_edge(e0, e1);
        }
    }

    pub fn degree(&self, v: usize) -> usize {
        self.edges[v].len()
    }

    /// Returns adjacent vertices (should always be in sorted order)
    pub fn vertex_adj(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        self.edges[v]
            .iter()
            .map(|&dst| self.vertices.find(dst as usize))
    }
    /// Returns adjacent vertices (should always be in sorted order)
    pub fn par_vertex_adj(&self, v: usize) -> impl rayon::iter::ParallelIterator<Item = usize> + '_
    where
        UF: Sync,
        T: Sync,
    {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        self.edges[v]
            .par_iter()
            .map(|&dst| self.vertices.find(dst as usize))
    }

    /// Returns whether two vertices v0 and v1 are adjacent.
    /// v0 and v1 can be merged into other vertices.
    #[inline]
    pub fn is_adj(&self, v0: usize, v1: usize) -> bool {
        let v0 = self.vertices.find(v0);
        let v1 = self.vertices.find(v1);
        self.edges[v0]
            .iter()
            .any(|&dst| self.vertices.find(dst as usize) == v1)
    }

    pub fn merged_vertices(&mut self, v: usize) -> impl Iterator<Item = usize> + Clone + '_ {
        self.inv_map.merged(v).map(|v| v as usize)
    }

    /// Merges v0 into v1.
    pub fn merge(&mut self, v0: usize, v1: usize, mut merge: impl FnMut(&T, &T) -> T)
    where
        T: Clone,
    {
        debug_assert_ne!(v0, v1);
        let [src, dst] = std::cmp::minmax(v0, v1);
        debug_assert!(!self.is_deleted(src));
        debug_assert!(!self.is_deleted(dst));
        debug_assert!(self.is_adj(src, dst));

        self.vertices.union(src, dst);

        self.inv_map.merge(src, dst);

        let [data_dst, data_src] = unsafe { self.data.get_disjoint_unchecked_mut([src, dst]) };
        let new_data = merge(&data_dst, &data_src);
        *data_src = new_data.clone();
        *data_dst = new_data;
        // data_src should no longer be accessed

        let [src_e, dst_e] = unsafe { self.edges.get_disjoint_unchecked_mut([src, dst]) };
        let pos = dst_e.iter().position(|&v| v == src as u32).unwrap();
        dst_e.swap_remove(pos);

        for v in dst_e.iter_mut() {
            *v = self.vertices.find(*v as usize) as u32;
        }
        let mut src_e = std::mem::take(src_e);
        let pos = src_e.iter().position(|&v| v == dst as u32).unwrap();
        src_e.swap_remove(pos);

        let curr_dst_len = dst_e.len();
        for v in src_e {
            let v = self.vertices.find(v as usize) as u32;
            if !dst_e[0..curr_dst_len].contains(&v) {
                dst_e.push(v);
            }
        }
        //src_e.retain(|&v| !dst_e.contains(&v));

        //dst_e.append(&mut src_e);

        let tmp = std::mem::take(&mut self.edges[dst]);
        for &adj in &tmp {
            let adj = self.vertices.find(adj as usize);
            debug_assert_ne!(adj, dst);
            let adj_e = unsafe { self.edges.get_unchecked_mut(adj) };
            if let Some(i) = adj_e.iter().position(|&v| v == src as u32) {
                adj_e.swap_remove(i);
            }
            if !adj_e.contains(&(dst as u32)) {
                adj_e.push(dst as u32);
            }
        }
        self.edges[dst] = tmp;
    }

    pub fn get(&self, v: usize) -> &T {
        unsafe { self.data.get_unchecked(self.vertices.find(v)) }
    }

    /// All the edges of this manifold mesh, with v0-v1 in original order
    pub fn edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges
            .iter()
            .enumerate()
            .flat_map(|(src, dsts)| dsts.iter().map(move |&dst| [src, dst as usize]))
    }

    #[inline]
    pub fn edges_post_merge(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges.iter().enumerate().flat_map(move |(src, dsts)| {
            let src = self.vertices.find(src);
            dsts.iter()
                .map(move |&dst| [src, self.vertices.find(dst as usize)])
                .filter(|[a, b]| a < b)
        })
    }

    /// All edges in this manifold mesh with v0-v1 in sorted order.
    pub fn ord_edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges.iter().enumerate().flat_map(|(src, dsts)| {
            dsts.iter()
                .filter(move |&&dst| src < dst as usize)
                .map(move |&dst| [src, dst as usize])
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind {
    Boundary(usize),
    Manifold([usize; 2]),
    NonManifold(smallvec::SmallVec<[usize; 3]>),
}

impl EdgeKind {
    pub fn insert(&mut self, v: usize) -> bool {
        use EdgeKind::*;
        *self = match self {
            &mut Boundary(a) if a == v => return false,
            &mut Manifold([a, _] | [_, a]) if a == v => return false,
            &mut Boundary(a) => Manifold([a, v]),
            &mut Manifold([a, b]) => NonManifold(smallvec::smallvec![a, b, v]),

            NonManifold(vs) if vs.contains(&v) => return false,
            NonManifold(vs) => {
                vs.push(v);
                return true;
            }
        };
        true
    }
    pub fn is_boundary(&self) -> bool {
        matches!(self, EdgeKind::Boundary(_))
    }
    pub fn as_slice(&self) -> &[usize] {
        use EdgeKind::*;
        match self {
            Boundary(f) => std::slice::from_ref(f),
            Manifold(fs) => fs.as_slice(),
            NonManifold(fs) => fs.as_slice(),
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        use EdgeKind::*;
        match self {
            Boundary(f) => std::slice::from_mut(f),
            Manifold(fs) => fs.as_mut_slice(),
            NonManifold(fs) => fs.as_mut_slice(),
        }
    }
    /// Constructs an edge kind from an iterator of items
    pub fn from_iter(mut v: impl Iterator<Item = usize>) -> Option<Self> {
        let mut curr = Self::Boundary(v.next()?);
        for v in v {
            curr.insert(v);
        }
        Some(curr)
    }
    pub fn dedup_by_key<T: Ord + Eq>(&mut self, k: impl Fn(usize) -> T) {
        use EdgeKind::*;
        match self {
            Boundary(_) => {}
            &mut Manifold([a, b]) if k(a) == k(b) => *self = Boundary(a),
            Manifold(_) => {}
            NonManifold(fs) => {
                fs.sort_unstable_by_key(|&v| k(v));
                fs.dedup_by_key(|&mut v| k(v));
                assert!(!fs.is_empty());
                if let &[f] = fs.as_slice() {
                    *self = Boundary(f);
                } else if let &[a, b] = fs.as_slice() {
                    *self = Manifold([a, b])
                }
            }
        }
    }
}
