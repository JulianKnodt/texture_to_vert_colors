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
}

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();

    let attr_ws = AttrWeights {
        ws: [args.luma_weight],
    };

    let analogy_scene =
        pars3d::load(&args.analogy).expect(&format!("Failed to read from {}", args.analogy));
    let mut ana_m = analogy_scene.into_flattened_mesh();
    assert_eq!(ana_m.vert_colors.len(), ana_m.v.len());
    ana_m.normalize();
    println!(
        "[INFO]: Analogy Mesh has #F = {}, #V = {}",
        ana_m.f.len(),
        ana_m.v.len()
    );

    macro_rules! q_n_attribs(
      ($m: expr, $n: expr, $vis: expr) => {{
        Quadric::n_attribs(
            $n,
            $vis.map(|vi| $m.v[vi]),
            $vis.map(|vi| [rgb_to_yiq($m.vert_colors[vi])[0]]),
            attr_ws,
        )
      }}
    );

    macro_rules! m_quadric {
        ($m:expr, $f: expr) => {{
            let m = $m;
            let f = $f;
            let area = f.area(&m.v);
            let n = f.normal(&m.v);
            let mut q = Quadric::new_plane(m.v[f.as_slice()[0]], n, area);
            q.area = area;
            // add attributes as well
            let q_attr = match f {
                FaceKind::Tri(vis) => q_n_attribs!(m, n, vis),
                FaceKind::Quad(vis) => q_n_attribs!(m, n, vis),
                FaceKind::Poly(p) => Quadric::dyn_attribs(
                    n,
                    p.len(),
                    |vi| m.v[vi],
                    |vi| [rgb_to_yiq(m.vert_colors[vi])[0]],
                    attr_ws,
                ),
            };
            q + q_attr * area
        }};
    }

    let ana_face_quadrics = ana_m
        .f
        .iter()
        .map(|f| m_quadric!(&ana_m, f))
        .collect::<Vec<_>>();

    // 1-ring
    let mut ana_quadrics = (0..ana_m.v.len())
        .map(|vi| (vi, Quadric::<1>::zero()))
        .collect::<Vec<_>>();
    for (fi, f) in ana_m.f.iter().enumerate() {
        for &vi in f.as_slice() {
            ana_quadrics[vi].1 += ana_face_quadrics[fi];
        }
    }

    let analogy_vv_adj = ana_m.vertex_vertex_adj();
    let analogy_vf_adj = ana_m.vertex_face_adj();

    let mut buf = vec![];
    if args.two_ring {
        for vi in 0..ana_m.v.len() {
            buf.clear();
            analogy_vv_adj.two_ring_faces(vi, &analogy_vf_adj, &mut buf);

            for &aa_fi in &buf {
                ana_quadrics[vi].1 += ana_face_quadrics[aa_fi];
            }
        }
    }
    /*
     */

    if args.drop_positions {
        for v in ana_quadrics.iter_mut() {
            v.1.drop_positions();
        }
    }

    ana_quadrics.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap());

    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    if m.vert_colors.is_empty() {
        m.vert_colors.resize(m.v.len(), [0.5; 3]);
    }
    let (s, t) = m.normalize();
    println!(
        "[INFO]: Input Mesh has #F = {}, #V = {}",
        m.f.len(),
        m.v.len()
    );

    let vert_face_adj = m.vertex_face_adj();
    let vert_vert_adj = m.vertex_vertex_adj();

    let range = args.range;
    macro_rules! find_nearest {
        ($vi: expr, $pos: expr) => {{
            let vi = $vi;
            let pos = $pos;
            let mut in_q = Quadric::<1>::zero();
            for &fi in vert_face_adj.adj(vi) {
                in_q += m_quadric!(&m, &m.f[fi as usize]);
            }

            if args.two_ring {
                buf.clear();
                vert_vert_adj.two_ring_faces(vi, &vert_face_adj, &mut buf);
                let b = buf.len();
                buf.sort_unstable();
                buf.dedup();
                assert_eq!(buf.len(), b);

                for &aa_fi in &buf {
                    ana_quadrics[vi].1 += m_quadric!(&m, &m.f[aa_fi as usize]);
                }
            }

            if args.drop_positions {
                in_q.drop_positions();
            }

            let new_luma = in_q.attributes_opt(pos, attr_ws)[0]
                .unwrap_or_else(|| rgb_to_yiq(m.vert_colors[vi])[0]);
            //let new_luma = in_q.attributes(pos, attr_ws)[0]
            //let new_luma = rgb_to_yiq(m.vert_colors[vi])[0];
            let new_luma = [new_luma];

            let nearest =
                ana_quadrics.binary_search_by(|probe| probe.1.partial_cmp(&in_q).unwrap());
            let nearest = match nearest {
                Ok(n) => n,
                Err(n) => n.min(ana_quadrics.len() - 1),
            };

            // TODO here maybe extract the value from the quadric then use that to query the
            // input source quadrics instead of the original position?

            //let mut dist = ana_quadrics[nearest].1.cost_attrib(pos, new_luma, attr_ws);
            //let mut best = nearest;

            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let (best, dist) = (nearest.saturating_sub(range)
                ..=(nearest + range).min(ana_quadrics.len() - 1))
                .into_par_iter()
                .map(|j| {
                    let nd = ana_quadrics[j].1.cost_attrib(pos, new_luma, attr_ws);
                    (j, nd)
                })
                .min_by_key(|&(_, d)| NotNan::new(d).unwrap())
                .unwrap();
            (best, dist)
        }};
    }

    // for each vertex in the input mesh, find quadric which most closely satisfies
    let mut pq = PriorityQueue::new();
    use indicatif::ProgressIterator;
    for (vi, pos) in m.v.iter().copied().enumerate().progress() {
        let (best, dist) = find_nearest!(vi, pos);
        pq.push(vi, (NotNan::new(-dist).unwrap(), ana_quadrics[best].0));
    }

    let vv_adj = m.vertex_vertex_adj();

    let p = indicatif::ProgressBar::new(pq.len() as u64);
    // sort texture elements by item with least difference
    while let Some((tgt_vi, (_, src_vi))) = pq.pop() {
        m.vert_colors[tgt_vi] = ana_m.vert_colors[src_vi];
        p.set_position(pq.len() as u64);

        if args.stop_at >= pq.len() {
            break;
        }

        if args.no_update {
            continue;
        }

        // update adjacent vertices now to be similar color?
        for &adj_vi in vv_adj.adj(tgt_vi) {
            let adj_vi = adj_vi as usize;
            if pq.get(&adj_vi).is_none() {
                continue;
            }
            let pos = m.v[adj_vi];

            let (best, dist) = find_nearest!(adj_vi, pos);
            pq.push(adj_vi, (NotNan::new(-dist).unwrap(), ana_quadrics[best].0));
        }
    }

    // for when exiting early to see which vertices were colored or not
    for (tgt_vi, _) in pq.drain() {
        m.vert_colors[tgt_vi] = [1.; 3];
    }

    m.denormalize(s, t);
    let s = m.into_scene();
    pars3d::save(&args.output, &s, true).expect("Failed to save output");
    println!("[INFO]: Took {:?} for visualization", start.elapsed());
}

fn sqr(x: F) -> F {
    x * x
}
