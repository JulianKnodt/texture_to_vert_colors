use union_find::{UnionFind, UnionFindOp};

/// A mesh representation which is suitable for collapsing vertices.
/// It can associate data with each vertex, and each edge.
/// Associated edge data is oriented.
#[derive(Debug, Clone)]
pub struct CollapsibleManifold<T, UF: UnionFindOp> {
    pub(crate) vertices: UF,

    pub edges: Vec<Vec<usize>>,

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
        self.edges[v0].push(v1);
        self.edges[v1].push(v0);

        // note that this is not using the mapping since edges should only be added ahead of
        // time.
        self.edges[v0].sort_unstable_by_key(|&dst| dst);
        self.edges[v1].sort_unstable_by_key(|&dst| dst);

        self.edges[v0].dedup_by_key(|&mut dst| dst);
        self.edges[v1].dedup_by_key(|&mut dst| dst);
    }

    /*
    /// Adds a face to this `CollapsibleManifold`.
    pub fn add_const_face<const N: usize>(&mut self, face: [usize; N]) {
        for i in 0..N {
            self.add_edge(face[i], face[(i + 1) % N]);
        }
    }
    */

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
        self.edges[v].iter().map(|&dst| self.vertices.find(dst))
    }
    /// Fix up a one ring to not contain duplicates
    pub fn dedup_one_ring(&mut self, v: usize) {
        self.dedup(v);
        let nbrs = std::mem::take(&mut self.edges[v]);
        for &adj in &nbrs {
            assert_ne!(adj, v);
            self.dedup(adj);
        }
        self.edges[v] = nbrs;
    }
    pub fn dedup(&mut self, v: usize) {
        self.edges[v].sort_unstable_by_key(|&v| self.vertices.find(v));
        self.edges[v].dedup_by_key(|&mut v| self.vertices.find(v));
    }

    /// Returns whether two vertices v0 and v1 are adjacent.
    /// v0 and v1 can be merged into other vertices.
    #[inline]
    pub fn is_adj(&self, v0: usize, v1: usize) -> bool {
        let v0 = self.vertices.find(v0);
        let v1 = self.vertices.find(v1);
        self.edges[v0]
            .iter()
            .any(|&dst| self.vertices.find(dst) == v1)
    }

    /// An iterator over the shared one ring of v0 and v1.
    /// Does not contain v0 or v1.
    pub fn shared_one_ring(&self, v0: usize, v1: usize) -> impl Iterator<Item = usize> + '_ {
        assert!(!self.is_deleted(v0));
        assert!(!self.is_deleted(v1));

        let v0_adj = self.vertex_adj(v0).peekable();
        let v1_adj = self.vertex_adj(v1).peekable();

        super::merge::Merge::new(v0_adj, v1_adj)
            .map(|v| v.into_inner())
            .filter(move |&v| v != v0 && v != v1)
    }
    pub fn shared_one_ring_deg(&self, v0: usize, v1: usize) -> usize {
        self.shared_one_ring(v0, v1).count()
    }
    /// An iterator over the shared one ring of v0 and v1.
    /// Does not contain v0 or v1.
    pub fn sided_shared_one_ring(
        &self,
        v0: usize,
        v1: usize,
    ) -> impl Iterator<Item = super::merge::Side<usize>> + '_ {
        assert!(!self.is_deleted(v0));
        assert!(!self.is_deleted(v1));

        let v0_adj = self.vertex_adj(v0).peekable();
        let v1_adj = self.vertex_adj(v1).peekable();

        super::merge::Merge::new(v0_adj, v1_adj)
            .filter(move |&v| *v.inner() != v0 && *v.inner() != v1)
    }
    /*
    pub fn par_vertex_adj(&self, v: usize) -> impl ParallelIterator<Item = usize> + '_ {
        let vs = &self.vertices;
        self.edges[v].par_iter().map(|&dst| vs.find(dst))
    }
    */

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

        let [data_dst, data_src] = unsafe { self.data.get_disjoint_unchecked_mut([src, dst]) };
        let new_data = merge(&data_dst, &data_src);
        *data_src = new_data.clone();
        *data_dst = new_data;
        // data_src should no longer be accessed

        let [src_e, dst_e] = unsafe { self.edges.get_disjoint_unchecked_mut([src, dst]) };
        let mut src_e = std::mem::take(src_e);
        /*
        for e in dst_e.iter_mut() {
            *e = self.vertices.find(*e);
        }
        let curr_dst_len = dst_e.len();
        for e in src_e {
            let e = self.vertices.find(*e);
            if e == dst || dst_e[0..curr_dst_len].contains(e) {
                continue;
            }
            dst_e.push(e);
        }
        */
        dst_e.append(&mut src_e);
        dst_e.retain_mut(|e1| {
            let new_v = self.vertices.find(*e1);
            *e1 = new_v;
            new_v != dst
        });
        dst_e.sort_unstable();
        dst_e.dedup();

        let tmp = std::mem::take(&mut self.edges[dst]);
        for &adj in &tmp {
            let adj = self.vertices.find(adj);
            debug_assert_ne!(adj, dst);
            let adj_e = unsafe { self.edges.get_unchecked_mut(adj) };
            for e in adj_e.iter_mut() {
                *e = self.vertices.find(*e);
            }
            adj_e.sort_unstable();
            adj_e.dedup();
        }
        self.edges[dst] = tmp;
    }

    pub fn get(&self, v: usize) -> &T {
        unsafe { self.data.get_unchecked(self.vertices.find(v)) }
    }
    pub fn set(&mut self, v: usize, t: T) {
        self.data[self.vertices.find(v)] = t;
    }

    /// All the edges of this manifold mesh, with v0-v1 in original order
    pub fn edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges
            .iter()
            .enumerate()
            .flat_map(|(src, dsts)| dsts.iter().map(move |&dst| [src, dst]))
    }

    #[inline]
    pub fn edges_post_merge(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges.iter().enumerate().flat_map(move |(src, dsts)| {
            let src = self.vertices.find(src);
            dsts.iter()
                .map(move |&dst| [src, self.vertices.find(dst)])
                .filter(|[a, b]| a < b)
        })
    }

    /// All edges in this manifold mesh with v0-v1 in sorted order.
    pub fn ord_edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges.iter().enumerate().flat_map(|(src, dsts)| {
            dsts.iter()
                .filter(move |&&dst| src < dst)
                .map(move |&dst| [src, dst])
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
