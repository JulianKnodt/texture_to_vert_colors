#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(generic_arg_infer)]

use clap::Parser;

use std::io::BufRead;

use texture_to_vert_colors::F;
use texture_to_vert_colors::clustering::{
    Args as ClusterArgs, Eigenvalue, OrderingKind, ShapeMetric,
};

#[derive(Clone, Parser, Debug)]
#[clap(group(
  clap::ArgGroup::new("target")
    .required(true)
    .args(&["target_num_charts", "error_bound", "match_json"])
))]
pub struct Args {
    /// Input OBJ file.
    #[arg(short, long, required = true)]
    pub input: String,

    /// Output PLY file.
    #[arg(short, long, default_value_t = String::new())]
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

    /// Output stats to this file
    #[arg(long, default_value_t = String::new())]
    stats: String,

    /// What ordering is preferred when constructing a cluster
    #[arg(long, default_value_t = OrderingKind::GeomColorShape)]
    ordering: OrderingKind,

    /// What metric to use when evaluating the quality of output cluster shape
    #[arg(long, default_value_t = ShapeMetric::MaxManhattanDist)]
    shape_metric: ShapeMetric,

    /// Prefer opposite effect of shape metric (For jokes)
    /// Mey kill perf.
    #[arg(long, hide = true)]
    invert_shape: bool,

    /// Store data to be passed to XAtlas here. For now should be an OBJ file which specifies
    /// a separate mesh for each cluster
    #[arg(long, default_value_t = String::new())]
    xatlas_output: String,

    /// First convert the mesh to a geometry only representation, stripping off colors and UVs
    #[arg(long)]
    geometry_only: bool,

    /// Eigen to regularize by when outputing eigenvalues. If not set will use the max.
    #[arg(long, default_value_t = -1.)]
    max_eigen: F,

    /// Match the `num_charts` field in this json instead of passing an explicit value
    #[arg(long, default_value_t = String::new())]
    match_json: String,

    /// Do not include the wireframe in the output.
    #[arg(long)]
    no_wireframe: bool,

    /// Do not use the delta cost function
    #[arg(long)]
    no_delta_cost: bool,
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
            no_delta_cost: a.no_delta_cost,
        }
    }
}

pub fn main() -> std::io::Result<()> {
    let mut args = Args::parse();
    if !args.output.ends_with(".ply") {
        eprintln!("[WARN]: Output will not be colored if output format is not PLY");
    }

    let scene = pars3d::load(&args.input).expect(&format!("Failed to load input {}", &args.input));
    let mut mesh = scene.into_flattened_mesh();
    mesh.f.retain_mut(|f| !f.canonicalize());
    let (s, t) = mesh.normalize();
    //let (cs, ct) = mesh.normalize_colors();
    if args.geometry_only {
        mesh.geometry_only();
    }

    if !args.match_json.is_empty() {
        if !std::fs::exists(&args.match_json)? {
            eprintln!("No source for {}", args.match_json);
            return Ok(());
        }
        let f = std::fs::File::open(&args.match_json)
            .expect(&format!("Failed to load match json {}", args.match_json));
        let r = std::io::BufReader::new(f);
        for l in r.lines() {
            let l = l?;
            if !l.contains("num_charts") {
                continue;
            }
            let num_charts = l.split_whitespace().nth(1).unwrap();
            let num_charts = if num_charts.ends_with(",") {
                num_charts.strip_suffix(",").unwrap()
            } else {
                num_charts
            };
            println!("[INFO]: Target number of charts is {num_charts}");
            args.target_num_charts = Some(num_charts.parse::<usize>().unwrap());
            break;
        }
    }

    let start = std::time::Instant::now();
    /*
    let (face_charts, m) =
        face_clustering(&mesh.v, &mesh.vert_colors, &mesh.f, mesh.f.len(), &args);
        */
    let (face_charts, chart_attribs, _chart_adj) =
        texture_to_vert_colors::clustering::face_clustering(
            &mesh.v,
            &mesh.vert_colors,
            &mesh.f,
            mesh.f.len(),
            &args.clone().into(),
        );
    println!("[INFO]: Took {:?} for clustering", start.elapsed());

    let num_charts = chart_attribs.len();
    eprintln!("[INFO]: Output # Charts = {num_charts}");

    mesh.denormalize(s, t);

    let mut wireframe_mesh = if args.no_wireframe {
        pars3d::Mesh::default()
    } else {
        let wireframe_parts = pars3d::visualization::face_segmentation_wireframes(
            |fi| mesh.f[fi].as_slice(),
            |fi| face_charts[fi],
            mesh.f.len(),
            &mesh.v,
            3e-4,
        );
        pars3d::visualization::wireframe_to_mesh(wireframe_parts)
    };
    if !args.output.is_empty() {
        let face_colors = (0..mesh.f.len())
            .map(|i| chart_attribs[face_charts[i]].1)
            .collect::<Vec<[F; 3]>>();

        let mut colored_mesh = mesh.with_face_coloring(&face_colors);
        //colored_mesh.denormalize_colors(cs, ct);

        colored_mesh.append(&mut wireframe_mesh);

        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.output, &out_scene)?;
    }

    use texture_to_vert_colors::measure_flat as mf;
    mf::measure_flat(
        &mut mesh,
        |fi| face_charts[fi],
        num_charts,
        &mf::Args {
            eigen_vis: args.eigen_vis,
            eigenvalue: args.eigenvalue,
            no_wireframe: args.no_wireframe,
            max_eigen: args.max_eigen,
            stats: args.stats,
            cluster_vis: args.cluster_vis,
        },
    )?;

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
                .filter(|&(fi, _)| face_charts[fi] == ci)
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
