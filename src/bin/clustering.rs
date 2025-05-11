#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(cmp_minmax)]
#![feature(generic_arg_infer)]

use std::array::from_fn;
use std::cmp::minmax;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

use ordered_float::NotNan;

use clap::{Parser, ValueEnum};

use priority_queue::PriorityQueue;

use texture_to_vert_colors::manifold::{CollapsibleManifold, EdgeKind};
use texture_to_vert_colors::quadric::{AttrWeights, Quadric};
use texture_to_vert_colors::sym::SymMatrix3;
use texture_to_vert_colors::{F, add, cross, dist, dot, length, normalize, poly_area, sub};

use indicatif::ProgressBar;

#[derive(Clone, Parser, Debug)]
#[clap(group(
  clap::ArgGroup::new("target")
    .required(true)
    .args(&["target_num_charts", "error_bound"])
))]
pub struct Args {
    /// Input OBJ file.
    #[arg(short, long, required = true)]
    pub input: String,

    /// Output PLY file.
    #[arg(short, long, required = true)]
    pub output: String,

    /// Target output number of charts.
    #[arg(short, long, group = "target")]
    pub target_num_charts: Option<usize>,

    #[arg(long, default_value_t = 1e-2)]
    pub vis_width: F,

    #[arg(long, default_value_t = Eigenvalue::One)]
    eigenvalue: Eigenvalue,

    /// How long before stopping
    #[arg(long, short='e', group="target", default_value_t = F::NEG_INFINITY)]
    error_bound: F,

    /// Do not use area weighting.
    #[arg(long)]
    no_area_weight: bool,

    /// Weight for each edge
    #[arg(long, default_value_t = 1e-2)]
    edge_weight: F,

    /// How much to weigh colors
    #[arg(long, default_value_t = 0.1)]
    color_weight: F,
}

pub fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if !args.output.ends_with(".ply") {
        eprintln!("[WARN]: Output will not be colored if not output as PLY");
    }

    let scene = pars3d::load(&args.input)?;
    let mut mesh = scene.into_flattened_mesh();
    let (s, t) = mesh.normalize();
    mesh.normalize_colors();
    //mesh.geometry_only();

    let (face_charts, m) = face_clustering(
        mesh.v.as_slice(),
        mesh.vert_colors.as_slice(),
        |fi| &mesh.f[fi].as_slice(),
        mesh.f.len(),
        &args,
    );

    let mut remap = HashMap::new();
    let mut num_charts = 0;
    for &f in &face_charts {
        match remap.entry(f) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                v.insert(num_charts);
                num_charts += 1;
            }
        }
    }
    eprintln!("[INFO]: Output # Charts = {}", remap.len());

    let face_coloring = pars3d::visualization::greedy_face_coloring(
        |i| face_charts[i],
        face_charts.len(),
        |i, j| m.is_adj(i, j),
        &pars3d::coloring::HIGH_CONTRAST,
    );

    let mut colored_mesh = mesh.with_face_coloring(&face_coloring);
    colored_mesh.denormalize(s, t);

    let out_scene = colored_mesh.into_scene();
    pars3d::save(&args.output, &out_scene)
}

pub fn face_clustering<'a>(
    vs: &'a [[F; 3]],
    vcs: &'a [[F; 3]],
    fs: impl Fn(usize) -> &'a [usize] + Sync + Send,
    nf: usize,
    args: &Args,
) -> (Vec<usize>, CollapsibleManifold<SymMatrix3>) {
    let cw = args.color_weight;
    let attr_ws = AttrWeights { ws: [cw; 3] };
    // face normals
    let f_n = (0..nf)
        .map(|fi| {
            let f = fs(fi);
            assert!(f.len() > 2);
            if f.len() == 4 {
                let [v0, v1, v2, v3] = from_fn(|i| vs[f[i]]);
                return normalize(cross(sub(v0, v2), sub(v1, v3)));
            }
            // just assume it's a flat polygon.
            let [v0, v1, v2] = from_fn(|i| vs[f[i]]);
            normalize(cross(sub(v1, v0), sub(v2, v0)))
        })
        .collect::<Vec<_>>();

    // vertex normals
    let mut v_ns = vec![[0.; 3]; vs.len()];
    for (fi, f_n) in f_n.iter().copied().enumerate() {
        for &vi in fs(fi) {
            v_ns[vi] = add(v_ns[vi], f_n);
        }
    }

    for v_n in v_ns.iter_mut() {
        *v_n = normalize(*v_n);
    }

    let mut face_area = vec![0.; nf];
    let mut edge_face_map: HashMap<[usize; 2], EdgeKind> = HashMap::new();
    let mut total_edge_len = 0.;
    let mut edge_num = 0;

    for (fi, f) in (0..nf).map(&fs).enumerate() {
        face_area[fi] = if !args.no_area_weight {
            poly_area(f.iter().map(|&v| vs[v]))
        } else {
            1.
        };
        for i in 0..f.len() {
            let n = (i + 1) % f.len();
            let e = minmax(f[i], f[n]);
            edge_face_map
                .entry(e)
                .and_modify(|old| {
                    old.insert(fi);
                })
                .or_insert_with(|| EdgeKind::Boundary(fi));
            total_edge_len += dist(vs[f[i]], vs[f[n]]);
            edge_num += 1;
        }
    }

    // TODO use this
    let avg_edge_len = total_edge_len / edge_num as F;
    assert_ne!(avg_edge_len, 0.);

    let mut edge_dihedral_angle = HashMap::new();

    let mut face_adj_quadrics = HashMap::new();

    let mut vertex_adj = vec![vec![]; vs.len()];
    for &[e0, e1] in edge_face_map.keys() {
        if e0 == e1 {
            continue;
        }
        assert!(!vertex_adj[e0].contains(&e1));
        vertex_adj[e0].push(e1);
        assert!(!vertex_adj[e1].contains(&e0));
        vertex_adj[e1].push(e0);

        let e = minmax(e0, e1);

        let (dihedral_angle, fs) = match edge_face_map[&e] {
            // no need to associate any cost with boundary edges.
            EdgeKind::Boundary(_) => continue,
            EdgeKind::Manifold([f0, f1]) => {
                let cos_sim = dot(f_n[f0], f_n[f1]).clamp(-1., 1.);
                (cos_sim.acos(), [f0, f1])
            }
            // TODO add multiple output pairs for each face with large values
            // Pairs of two faces across a non-manifold edge can be merged together.
            EdgeKind::NonManifold(_) => continue,
        };
        let prev = edge_dihedral_angle.insert(e, dihedral_angle);
        assert_eq!(prev, None);

        let e_w = dihedral_angle;

        let edge_dir = sub(vs[e0], vs[e1]);
        let edge_len = length(edge_dir) / avg_edge_len;
        if edge_len == 0. {
            continue;
        }
        assert_ne!(edge_len, 0.);

        let edge_dir = normalize(edge_dir);

        // TODO max weight here with symmetry weight

        let edge_quadric = SymMatrix3::outer(edge_dir) * edge_len * e_w;

        let prev = face_adj_quadrics.insert(fs, edge_quadric);
        assert_eq!(prev, None);
    }

    // for each real mesh edge, track which charts it is a part of.
    // Each edge can be at most part of 2 charts.
    let mut m = CollapsibleManifold::<SymMatrix3>::new_with(nf, |fi| {
        let n = f_n[fi];
        let q = SymMatrix3::outer(n);
        if vcs.is_empty() || attr_ws.is_zero() {
            return q;
        }

        macro_rules! q_n_attrib {
            ($vis: expr) => {{ Quadric::n_attribs(n, $vis.map(|vi| vs[vi]), $vis.map(|vi| vcs[vi]), attr_ws) }};
        }

        let q_attr = match fs(fi) {
            [] | [_] | [_, _] => SymMatrix3::zero(),
            &[a, b, c] => q_n_attrib!([a, b, c]).a,
            &[a, b, c, d] => q_n_attrib!([a, b, c, d]).a,
            p => Quadric::dyn_attribs(n, p.len(), |vi| vs[vi], |vi| vcs[vi], attr_ws).a,
        };
        let mut q = (q + q_attr) * face_area[fi];
        for [e0, e1] in edges(fs(fi)) {
            let e = std::cmp::minmax(e0, e1);
            let Some(adj_q) = face_adj_quadrics.get(&e) else {
                continue;
            };
            q += *adj_q;
        }
        q
    });

    for e in edge_face_map.values() {
        let s = e.as_slice();
        // connect all adjacent faces
        for i in 0..s.len() {
            for j in i + 1..s.len() {
                assert_ne!(s[i], s[j]);
                m.add_edge(s[i], s[j]);
            }
        }
    }

    // from this point, edge_face_map now corresponds to the chart of each mesh.
    eprintln!("[INFO]: Input # vertices = {}", vs.len());
    eprintln!("[INFO]: Input # faces = {}", nf);

    let mut pq = PriorityQueue::new();

    macro_rules! cost_of_edge {
        ($e0:expr, $e1: expr) => {{
            let [e0, e1] = std::cmp::minmax($e0, $e1);

            assert!(!m.is_deleted(e0));
            assert!(!m.is_deleted(e1));
            let q0 = *m.get(e0);
            let q1 = *m.get(e1);
            // if the charts are merged together then we don't need to combine any additional
            // costs, otherwise we need to add them.
            let q_new = q0 + q1;

            // sorted eigenvalues
            let [evn0, evn1, evn2] = q_new.eigen_sorted().0;
            let [ev00, ev01, ev02] = q0.eigen_sorted().0;
            let [ev10, ev11, ev12] = q1.eigen_sorted().0;

            // TODO may need to subtract previous values here?
            let [evn, ev0, ev1] = match args.eigenvalue {
                Eigenvalue::Zero => [evn0, ev00, ev10],
                Eigenvalue::One => [evn1, ev01, ev11],
                Eigenvalue::Two => [evn2, ev02, ev12],
            };
            let [evn, ev0, ev1] = [evn, ev0, ev1].map(F::abs).map(F::sqrt);
            NotNan::new(-(evn - (ev0 + ev1))).unwrap()
        }};
    }

    // This loop can be very slow for processing all components together since it is a dense
    // mesh.
    let ec = Mutex::new(&mut pq);
    m.edges()
        // since just deduplicated edges, only check when e0 < e1
        .filter(|[e0, e1]| e0 < e1)
        .for_each(|[e0, e1]| {
            let c = cost_of_edge!(e0, e1);
            ec.lock().unwrap().push(minmax(e0, e1), c);
        });

    let p = ProgressBar::new(m.num_vertices() as u64);
    p.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} (Elapsed = {elapsed_precise})")
            .unwrap(),
    );

    while let Some(([e0, e1], n)) = pq.pop() {
        // Termination criteria
        let reached = args
            .target_num_charts
            .map(|tf| m.num_vertices() <= tf)
            .unwrap_or(false)
            || (*n < args.error_bound);

        if reached {
            break;
        }
        if m.is_deleted(e0) || m.is_deleted(e1) {
            continue;
        }

        m.merge(e0, e1, |a, b| *a + *b);

        assert!(m.is_deleted(e0));
        assert!(!m.is_deleted(e1));

        for adj in m.vertex_adj(e1) {
            let c = cost_of_edge!(adj, e1);
            let e = minmax(e1, adj);

            pq.push(e, c);
        }

        p.set_position(m.num_vertices() as u64);
    }

    // for each face which chart is it assigned to?
    let charts = (0..nf).map(|i| m.get_new_vertex(i)).collect::<Vec<_>>();
    (charts, m)
}

pub fn edges(f: &[usize]) -> impl Iterator<Item = [usize; 2]> + '_ {
    (0..f.len()).map(|i| [f[i], f[(i + 1) % f.len()]])
}

macro_rules! impl_display {
  ($name: ident, $($kind: ident => $disp: expr),+$(,)?) => {
    impl std::fmt::Display for $name {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use $name::*;
            let s = match self {
                $($kind => $disp,)+
            };
            write!(f, "{s}")
        }
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum Eigenvalue {
    Zero,
    One,
    Two,
}

impl_display!(Eigenvalue, Zero => "zero", One => "one", Two => "two");
