use crate::inv_map::InverseMap;
use union_find::UnionFind;

/// A mesh representation which is suitable for collapsing vertices.
/// It can associate data with each vertex, and each edge.
/// Associated edge data is oriented.
#[derive(Debug, Clone)]
pub struct CollapsibleManifold<T> {
    vertices: UnionFind,

    pub edges: Vec<Vec<usize>>,

    inv_map: InverseMap,

    pub data: Vec<T>,
}

impl<T> CollapsibleManifold<T> {
    #[inline]
    pub fn new(size: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            vertices: UnionFind::new(size),
            edges: vec![],
            inv_map: InverseMap::new(size),
            data: vec![T::default(); size],
        }
    }
    pub fn new_with(size: usize, f: impl Fn(usize) -> T) -> Self {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(f(i));
        }
        Self {
            vertices: UnionFind::new(size),
            edges: vec![vec![]; size],
            inv_map: InverseMap::new(size),

            data,
        }
    }
    pub fn get_new_vertex(&self, old: usize) -> usize {
        self.vertices.get_compress(old)
    }
    pub fn num_vertices(&self) -> usize {
        self.vertices.curr_len()
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

    pub fn quad_parallel_edges(
        &self,
        e0: usize,
        e1: usize,
    ) -> impl Iterator<Item = [usize; 2]> + '_ {
        assert!(self.is_adj(e0, e1));
        let e0_nbrs = self
            .vertex_adj(e0)
            .filter(move |&e0_adj| e0_adj != e1 && !self.is_adj(e0_adj, e1))
            .filter(move |&e0_adj| self.degree(e0_adj) == 4)
            .filter(move |&e0_adj| {
                self.vertex_adj(e0_adj)
                    .filter(|&nbr| nbr != e0)
                    .all(|e0_aa| !self.is_adj(e0_aa, e1))
            })
            .map(move |e0_adj| [e0, e0_adj]);
        let e1_nbrs = self
            .vertex_adj(e1)
            .filter(move |&e1_adj| e1_adj != e0 && !self.is_adj(e1_adj, e0))
            .filter(move |&e1_adj| self.degree(e1_adj) == 4)
            .filter(move |&e1_adj| {
                self.vertex_adj(e1_adj)
                    .filter(|&nbr| nbr != e1)
                    .all(|e1_aa| !self.is_adj(e1_aa, e0))
            })
            .map(move |e1_adj| [e1, e1_adj]);
        e0_nbrs.chain(e1_nbrs)
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

    /// Returns adjacent vertices (should always be in sorted order)
    pub fn vertex_adj(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        self.edges[v]
            .iter()
            .map(|&dst| self.vertices.get_compress(dst))
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
    #[inline]
    /// the degree of a given vertex
    pub fn degree(&self, v: usize) -> usize {
        self.edges[v].len()
    }
    pub fn dedup(&mut self, v: usize) {
        self.edges[v].sort_unstable_by_key(|&v| self.vertices.get_compress(v));
        self.edges[v].dedup_by_key(|&mut v| self.vertices.get(v));
    }

    /// Returns whether two vertices v0 and v1 are adjacent.
    /// v0 and v1 can be merged into other vertices.
    #[inline]
    pub fn is_adj(&self, v0: usize, v1: usize) -> bool {
        let v0 = self.vertices.get_compress(v0);
        let v1 = self.vertices.get_compress(v1);
        self.edges[v0]
            .iter()
            .any(|&dst| self.vertices.get_compress(dst) == v1)
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
        self.edges[v].par_iter().map(|&dst| vs.get_compress(dst))
    }
    */

    /// Merges v0 into v1.
    pub fn merge(&mut self, v0: usize, v1: usize, mut merge: impl FnMut(&T, &T) -> T)
    where
        T: Clone,
    {
        assert_ne!(v0, v1);
        let [src, dst] = std::cmp::minmax(v0, v1);
        assert!(!self.is_deleted(src));
        assert!(!self.is_deleted(dst));
        assert!(self.is_adj(src, dst));

        self.vertices.set(src, dst);

        self.inv_map.merge(src, dst);

        self.data[dst] = merge(&self.data[dst], &self.data[src]);
        self.data[src] = self.data[dst].clone();

        let mut src_e = std::mem::take(&mut self.edges[src]);
        self.edges[dst].append(&mut src_e);
        self.edges[dst].retain(|&e1| self.vertices.get_compress(e1) != dst);
        self.edges[dst].sort_by_key(|&e1| self.vertices.get(e1));
        self.edges[dst].dedup_by_key(|&mut e1| self.vertices.get(e1));

        let tmp = std::mem::take(&mut self.edges[dst]);
        for &adj in &tmp {
            let adj = self.vertices.get_compress(adj);
            assert_ne!(adj, dst);
            self.edges[adj].sort_unstable_by_key(|&v| self.vertices.get_compress(v));
            self.edges[adj].dedup_by_key(|&mut v| self.vertices.get(v));
        }
        self.edges[dst] = tmp;
    }
    pub fn merged_vertices(&self, v0: usize) -> impl Iterator<Item = usize> + Clone + '_ {
        self.inv_map.merged(v0)
    }

    pub fn get(&self, v: usize) -> &T {
        &self.data[self.vertices.get(v)]
    }
    pub fn set(&mut self, v: usize, t: T) {
        self.data[self.vertices.get(v)] = t;
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
            let src = self.vertices.get_compress(src);
            dsts.iter()
                .map(move |&dst| [src, self.vertices.get_compress(dst)])
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
