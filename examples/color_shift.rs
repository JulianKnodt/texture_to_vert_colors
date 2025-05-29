#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(unused)]

use clap::Parser;
use ordered_float::NotNan;
use texture_to_vert_colors::quadric::{AttrWeights, Quadric};
use texture_to_vert_colors::{F, dot};

use priority_queue::PriorityQueue;

use pars3d::FaceKind;
use pars3d::coloring::rgb_to_yiq;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh to apply analogy to
    #[arg(long, short)]
    input: String,

    /// Mesh which has analogy in it (both geom and texture to reuse)
    #[arg(long, short)]
    analogy: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,

    /// How important to weight luminance when preserving color.
    #[arg(long, default_value_t = 0.5)]
    luma_weight: F,

    /// Drop positions from each quadric (ABLATION)
    #[arg(long)]
    drop_positions: bool,

    /// How many elements to perform a linear search for
    #[arg(long, default_value_t = 500)]
    range: usize,

    /// Enable two ring for quadrics (ABLATION)
    #[arg(long)]
    two_ring: bool,

    /// Do not update adjacent elements in the priority queue (ABLATION)
    #[arg(long)]
    no_update: bool,

    /// Stop earlier for debugging to check which vertices have been populated or not
    #[arg(long, hide = true, default_value_t = 0)]
    stop_at: usize,

    /// Visualize the clusters of the analogy mesh here
    #[arg(long, default_value_t = String::new())]
    analogy_cluster_vis: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let start = std::time::Instant::now();

    let attr_ws = AttrWeights {
        ws: [args.luma_weight],
    };

    let analogy_scene =
        pars3d::load(&args.analogy).expect(&format!("Failed to read from {}", args.analogy));
    let mut ana_m = analogy_scene.into_flattened_mesh();
    assert_eq!(ana_m.vert_colors.len(), ana_m.v.len());
    let (a_s, a_t) = ana_m.normalize();
    println!(
        "[INFO]: Analogy Mesh has #F = {}, #V = {}",
        ana_m.f.len(),
        ana_m.v.len()
    );

    use texture_to_vert_colors::clustering::{
        Args as ClusterArgs, Eigenvalue, OrderingKind, ShapeMetric, face_clustering,
    };
    let (face_charts, clusters) = face_clustering(
        &ana_m.v,
        &ana_m.vert_colors,
        &ana_m.f,
        ana_m.f.len(),
        &ClusterArgs {
            // TODO pass this as an argument
            target_num_charts: Some(500),
            eigenvalue: Eigenvalue::One,
            error_bound: F::NEG_INFINITY,
            no_area_weight: false,
            eigen_eps: 1e-6,
            color_eps: 1e-5,
            ordering: OrderingKind::GeomColorShape,
            shape_metric: ShapeMetric::BoundaryLength,
            invert_shape: false,
        },
    );

    if !args.analogy_cluster_vis.is_empty() {
        let wireframe_parts = pars3d::visualization::face_segmentation_wireframes(
            |fi| ana_m.f[fi].as_slice(),
            |fi| face_charts[fi],
            ana_m.f.len(),
            &ana_m.v,
            3e-4,
        );
        let mut wireframe_mesh = pars3d::visualization::wireframe_to_mesh(wireframe_parts);
        wireframe_mesh.denormalize(a_s, a_t);

        let face_colors = (0..ana_m.f.len())
            .map(|i| clusters.get(face_charts[i]).1)
            .collect::<Vec<[F; 3]>>();
        let mut colored_mesh = ana_m.with_face_coloring(&face_colors);
        colored_mesh.denormalize(a_s, a_t);
        colored_mesh.append(&mut wireframe_mesh);
        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.analogy_cluster_vis, &out_scene)?;
    }

    let clusters = clusters
        .vertices()
        .map(|(_, (q, avg_color, _))| (*q, *avg_color))
        .collect::<Vec<_>>();

    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    if m.vert_colors.is_empty() {
        m.vert_colors.resize(m.v.len(), [1.; 3]);
    }
    let (s, t) = m.normalize();
    println!(
        "[INFO]: Input Mesh has #F = {}, #V = {}",
        m.f.len(),
        m.v.len()
    );

    let range = args.range;
    macro_rules! find_nearest {
        ($vi: expr, $pos: expr) => {{
            let vi = $vi;
            let pos = $pos;

            let own_luma = rgb_to_yiq(m.vert_colors[vi])[0];
            let (best, dist) = clusters
                .iter()
                .enumerate()
                .map(|(i, &(q, _avg_color))| (i, (rgb_to_yiq(_avg_color)[0] - own_luma).abs()))
                .min_by_key(|(_, cost)| NotNan::new(*cost).unwrap())
                .unwrap();

            (best, dist)
        }};
    }

    // for each vertex in the input mesh, find quadric which most closely satisfies
    let mut pq = PriorityQueue::new();
    use indicatif::ProgressIterator;
    for (vi, pos) in m.v.iter().copied().enumerate().progress() {
        let (best, dist) = find_nearest!(vi, pos);
        pq.push(vi, (NotNan::new(-dist).unwrap(), best));
    }

    let vv_adj = m.vertex_vertex_adj();

    let p = indicatif::ProgressBar::new(pq.len() as u64);
    // sort texture elements by item with least difference
    while let Some((tgt_vi, (_, cluster_idx))) = pq.pop() {
        m.vert_colors[tgt_vi] = clusters[cluster_idx].1;
        p.set_position(pq.len() as u64);
    }

    // for when exiting early to see which vertices were colored or not
    for (tgt_vi, _) in pq.drain() {
        m.vert_colors[tgt_vi] = [1.; 3];
    }

    m.denormalize(s, t);
    let s = m.into_scene();
    pars3d::save(&args.output, &s).expect("Failed to save output");
    println!("[INFO]: Took {:?} for visualization", start.elapsed());

    Ok(())
}
