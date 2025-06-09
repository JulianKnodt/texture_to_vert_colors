#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(generic_arg_infer)]

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use clap::Parser;

use texture_to_vert_colors::F;
use texture_to_vert_colors::clustering::{
    Args as ClusterArgs, Eigenvalue, OrderingKind, ShapeMetric,
};

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

    /// Output PLY file, where clusters are colored by their clustering instead of average color
    #[arg(short, long, default_value_t = String::new())]
    pub cluster_vis: String,

    /// Output PLY file, to visualize the optimized eigenvalue of each cluster normalized to the
    /// range [0,1]
    #[arg(long, default_value_t = String::new())]
    pub eigen_vis: String,

    /// Target output number of charts.
    #[arg(short, long, group = "target")]
    pub target_num_charts: Option<usize>,

    #[arg(long, default_value_t = Eigenvalue::Zero)]
    eigenvalue: Eigenvalue,

    /// How long before stopping
    #[arg(long, short='e', group="target", default_value_t = F::NEG_INFINITY)]
    error_bound: F,

    /// Do not use area weighting.
    #[arg(long)]
    no_area_weight: bool,

    /// Absolute value difference permitted between eigenvalues, before switching to color
    /// checking.
    #[arg(long, default_value_t = 1e-4)]
    eigen_eps: F,

    /// Absolute value difference permitted between colors, before switching to bd length
    /// checking.
    #[arg(long, default_value_t = 1e-6)]
    color_eps: F,

    /// Where to output stats (unused currently)
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// What ordering is preferred when constructing a cluster
    #[arg(long, default_value_t = OrderingKind::GeomColorShape)]
    ordering: OrderingKind,

    /// What metric to use when evaluating the quality of output cluster shape
    #[arg(long, default_value_t = ShapeMetric::AngleDeviation)]
    shape_metric: ShapeMetric,

    /// Prefer opposite effect of shape metric (For jokes)
    /// Mey kill perf.
    #[arg(long, hide = true)]
    invert_shape: bool,

    /// Store data to be passed to XAtlas here. For now should be an OBJ file which specifies
    /// a separate mesh for each cluster
    #[arg(long, default_value_t = String::new())]
    xatlas_output: String,
}

impl From<Args> for ClusterArgs {
    fn from(a: Args) -> Self {
        Self {
            target_num_charts: a.target_num_charts,
            eigenvalue: a.eigenvalue,
            error_bound: a.error_bound,
            no_area_weight: a.no_area_weight,
            eigen_eps: a.eigen_eps,
            color_eps: a.color_eps,
            ordering: a.ordering,
            shape_metric: a.shape_metric,
            invert_shape: a.invert_shape,
        }
    }
}

pub fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if !args.output.ends_with(".ply") {
        eprintln!("[WARN]: Output will not be colored if output format is not PLY");
    }

    let scene = pars3d::load(&args.input).expect(&format!("Failed to load input {}", &args.input));
    let mut mesh = scene.into_flattened_mesh();
    let (s, t) = mesh.normalize();
    let (cs, ct) = mesh.normalize_colors();

    let start = std::time::Instant::now();
    /*
    let (face_charts, m) =
        face_clustering(&mesh.v, &mesh.vert_colors, &mesh.f, mesh.f.len(), &args);
        */
    let (face_charts, m) = texture_to_vert_colors::clustering::face_clustering(
        &mesh.v,
        &mesh.vert_colors,
        &mesh.f,
        mesh.f.len(),
        &args.clone().into(),
    );
    println!("[INFO]: Took {:?} for clustering", start.elapsed());

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

    let wireframe_parts = pars3d::visualization::face_segmentation_wireframes(
        |fi| mesh.f[fi].as_slice(),
        |fi| face_charts[fi],
        mesh.f.len(),
        &mesh.v,
        3e-4,
    );
    let mut wireframe_mesh = pars3d::visualization::wireframe_to_mesh(wireframe_parts);
    wireframe_mesh.denormalize(s, t);

    if !args.cluster_vis.is_empty() {
        let face_coloring = pars3d::visualization::greedy_face_coloring(
            |i| face_charts[i],
            face_charts.len(),
            |i, j| m.is_adj(i, j),
            &pars3d::coloring::HIGH_CONTRAST,
        );

        let mut colored_mesh = mesh.with_face_coloring(&face_coloring);
        colored_mesh.denormalize(s, t);
        colored_mesh.append(&mut wireframe_mesh.clone());
        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.cluster_vis, &out_scene)?;
    }

    let eigenvalues = m
        .vertices()
        .map(|(vi, (q, _, _))| {
            let eigens = q.a.eigen_sorted().0;
            (vi, args.eigenvalue.apply(eigens))
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let [min_e, max_e] = eigenvalues
        .values()
        .fold([F::INFINITY, F::NEG_INFINITY], |[l, h], &n| {
            [l.min(n), h.max(n)]
        });
    assert!(max_e.is_finite());
    assert!(min_e.is_finite());
    assert!(max_e >= min_e);
    println!("[INFO] eigenvalues in range [{min_e}, {max_e}]");

    if !args.eigen_vis.is_empty() {
        let mut face_eigens = (0..mesh.f.len())
            .map(|i| eigenvalues[&face_charts[i]])
            .collect::<Vec<F>>();
        let r = max_e - min_e;
        for e in &mut face_eigens {
            *e -= min_e;
            if r != 0. {
                *e /= r;
            }
        }
        let face_colors = face_eigens
            .into_iter()
            .map(pars3d::coloring::magma)
            .collect::<Vec<_>>();
        let mut colored_mesh = mesh.with_face_coloring(&face_colors);

        colored_mesh.denormalize(s, t);

        colored_mesh.append(&mut wireframe_mesh.clone());

        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.eigen_vis, &out_scene)?;
    }

    {
        let face_colors = (0..mesh.f.len())
            .map(|i| m.get(face_charts[i]).1)
            .collect::<Vec<[F; 3]>>();

        let mut colored_mesh = mesh.with_face_coloring(&face_colors);
        colored_mesh.denormalize(s, t);
        colored_mesh.denormalize_colors(cs, ct);

        colored_mesh.append(&mut wireframe_mesh);

        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.output, &out_scene)?;
    }

    if !args.xatlas_output.is_empty() {
        // for each chart, output a separate mesh
        let mut out_scene = pars3d::Scene::default();
        for ci in 0..num_charts {
            let mut new_mesh = pars3d::Mesh::default();
            new_mesh.v.clone_from(&mesh.v);
            let faces_in_chart = mesh
                .f
                .iter()
                .enumerate()
                .filter(|&(fi, _)| remap[&face_charts[fi]] == ci)
                .map(|(_, f)| f.clone());
            new_mesh.f.extend(faces_in_chart);

            new_mesh.delete_unused_vertices();
            new_mesh.denormalize(s, t);
            // for trimesh to generate a separate mesh for each (but not actually used)
            new_mesh.assign_single_mat(ci);
            out_scene.materials.push(Default::default());
            out_scene.meshes.push(new_mesh);
        }
        pars3d::save(&args.xatlas_output, &out_scene)?;
    }

    Ok(())
}
