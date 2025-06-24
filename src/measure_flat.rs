use super::F;
use super::clustering::Eigenvalue;
use super::quadric::Quadric;
use std::collections::BTreeMap;

fn push_uniq(dst: &mut Vec<usize>, v: usize) {
    if !dst.contains(&v) {
        dst.push(v)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub cluster_vis: String,
    pub eigenvalue: Eigenvalue,
    pub eigen_vis: String,
    pub stats: String,
    pub max_eigen: F,
    pub no_wireframe: bool,
}

/// Shared utility for measuring developability
pub fn measure_flat(
    mesh: &mut pars3d::Mesh,
    face_charts: impl Fn(usize) -> usize,
    num_charts: usize,
    args: &Args,
) -> std::io::Result<()> {
    let (s, t) = mesh.normalize();
    let mut per_chart_quadric = vec![Quadric::<0>::zero(); num_charts as usize];
    for (fi, f) in mesh.f.iter().enumerate() {
        let area = f.area(&mesh.v);
        let normal = f.normal(&mesh.v);
        let q = Quadric::<0>::new_plane(mesh.v[f.as_slice()[0]], normal, 1.);
        per_chart_quadric[face_charts(fi)] += q * area
    }

    // construct chart adjacency
    mesh.geometry_only();
    let edge_adj = mesh.edge_pos_kinds();
    let mut chart_adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for ek in edge_adj.values() {
        let ek = ek.as_slice();
        for &fi in ek {
            let ci = face_charts(fi);
            for &fj in ek {
                let cj = face_charts(fj);
                if ci == cj {
                    continue;
                }
                push_uniq(chart_adj.entry(ci).or_default(), cj);
                push_uniq(chart_adj.entry(cj).or_default(), ci);
            }
        }
    }

    let wireframe_mesh = if args.no_wireframe {
        pars3d::Mesh::default()
    } else {
        let wireframe_parts = pars3d::visualization::face_segmentation_wireframes(
            |fi| mesh.f[fi].as_slice(),
            |fi| face_charts(fi),
            mesh.f.len(),
            &mesh.v,
            3e-4,
        );
        let mut wireframe_mesh = pars3d::visualization::wireframe_to_mesh(wireframe_parts);
        wireframe_mesh.denormalize(s, t);
        wireframe_mesh
    };

    if !args.cluster_vis.is_empty() {
        let face_coloring = pars3d::visualization::greedy_face_coloring(
            |i| face_charts(i),
            mesh.f.len(),
            |i, j| chart_adj[&i].contains(&j),
            &pars3d::coloring::HIGH_CONTRAST,
        );

        let mut colored_mesh = mesh.with_face_coloring(&face_coloring);
        colored_mesh.append(&mut wireframe_mesh.clone());
        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.cluster_vis, &out_scene)?;
    }

    let max_planarity = per_chart_quadric
        .iter()
        .map(|q| q.a.eigen_sorted().0[1])
        .max_by(F::total_cmp)
        .unwrap();
    println!("[INFO]: Max planarity is {max_planarity:e}");

    let avg_planarity = per_chart_quadric
        .iter()
        .map(|q| q.a.eigen_sorted().0[1])
        .sum::<F>()
        / per_chart_quadric.len() as F;

    let max_developability = per_chart_quadric
        .iter()
        .map(|q| q.a.eigen_sorted().0[0])
        .max_by(F::total_cmp)
        .unwrap();
    assert!(max_planarity >= max_developability);
    println!("[INFO]: Max developability is {max_developability:e}");

    let avg_developability = per_chart_quadric
        .iter()
        .map(|q| q.a.eigen_sorted().0[0])
        .sum::<F>()
        / per_chart_quadric.len() as F;

    let eigenvalues = per_chart_quadric
        .iter()
        .map(|q| args.eigenvalue.apply(q.a.eigen_sorted().0))
        .collect::<Vec<_>>();

    let [min_e, max_e] = eigenvalues
        .iter()
        .fold([F::INFINITY, F::NEG_INFINITY], |[l, h], &n| {
            [l.min(n), h.max(n)]
        });
    assert!(max_e.is_finite());
    assert!(min_e.is_finite());
    assert!(max_e >= min_e);
    println!("[INFO] eigenvalues in range [{min_e:e}, {max_e:e}]");

    if !args.eigen_vis.is_empty() {
        let mut face_eigens = (0..mesh.f.len())
            .map(|i| eigenvalues[face_charts(i)])
            .collect::<Vec<F>>();
        let low = min_e.max(1e-20).ln();
        let r = max_e.ln() - low;
        let div = if args.max_eigen > 0. {
            args.max_eigen.ln()
        } else {
            r
        };
        for e in &mut face_eigens {
            *e = e.max(1e-20).ln() - low;
            if div == 0. {
                continue;
            }
            *e = *e / div;
            *e = e.clamp(0., 1.);
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

    /*
    {
        let face_colors = (0..mesh.f.len())
            .map(|i| chart_attribs[face_charts[i]].1)
            .collect::<Vec<[F; 3]>>();

        let mut colored_mesh = mesh.with_face_coloring(&face_colors);
        colored_mesh.denormalize(s, t);

        colored_mesh.append(&mut wireframe_mesh);

        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.output, &out_scene)?;
    }
    */

    if !args.stats.is_empty() {
        use std::io::{BufWriter, Write};

        let f = std::fs::File::create(&args.stats)?;
        let mut f = BufWriter::new(f);
        writeln!(
            f,
            r#"{{
  "eigenvalue_max": {max_e},
  "max_planarity": {max_planarity},
  "avg_planarity": {avg_planarity},
  "max_developability": {max_developability},
  "avg_developability": {avg_developability},
  "num_charts": {num_charts}
}}"#
        )?;
    }

    Ok(())
}
