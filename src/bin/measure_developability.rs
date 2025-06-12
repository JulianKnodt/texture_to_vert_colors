use clap::Parser;
use texture_to_vert_colors::quadric::Quadric;
use texture_to_vert_colors::{F, length};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input mesh with per vertex offsets stored in the R channel of the vertex colors
    #[arg(long, short)]
    input: String,

    /// Output mesh with each vertex offset in the direction of the normal by the height.
    #[arg(long, short)]
    output: String,

    /// How wide are the output wires?
    #[arg(long, default_value_t = 1e-3)]
    vis_width: F,
}

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();
    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    m.normalize();
    let ff_adj = m.face_face_adj();
    let (face_comp, num_comps) = ff_adj.connected_components();

    let mut quadrics = (0..num_comps)
        .map(|_| Quadric::<0>::zero())
        .collect::<Vec<_>>();

    for (fi, f) in m.f.iter().enumerate() {
        let area = f.area(&m.v);
        if area < 1e-20 {
            continue;
        }
        let normal = f.normal(&m.v);
        if length(normal) < 1e-3 {
            return;
        }
        let mut q = Quadric::new_plane(m.v[f.as_slice()[0]], normal, area);
        q.area = area;
        quadrics[face_comp[fi] as usize] += q;
    }

    let mut sum_e0 = 0.;
    let mut sum_e1 = 0.;
    let [min_e0, min_e1, max_e0, max_e1] = quadrics.iter().fold(
        [F::INFINITY, F::INFINITY, F::NEG_INFINITY, F::NEG_INFINITY],
        |[l_e0, l_e1, h_e0, h_e1], q| {
            let [e0, e1, _] = q.a.eigen_sorted().0;
            sum_e0 += e0;
            sum_e1 += e1;
            [l_e0.min(e0), l_e1.min(e1), h_e0.max(e0), h_e1.max(e1)]
        },
    );
    println!("Average Developability {}", sum_e0 / quadrics.len() as F);
    println!("Min/Max Developability {min_e0} {max_e0}");
    println!("Average Planarity {}", sum_e1 / quadrics.len() as F);
    println!("Min/Max Planarity {min_e1} {max_e1}");

    println!("Took {:?} for measuring developability", start.elapsed());
}
