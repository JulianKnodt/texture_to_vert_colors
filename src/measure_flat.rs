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
    let edge_adj = mesh.edge_kinds();
    let mut chart_adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    let mut bd_len = 0.;
    for (&[vi0, vi1], ek) in edge_adj.iter() {
        let ek = ek.as_slice();
        for (fo, &fi) in ek.iter().enumerate() {
            let ci = face_charts(fi);
            for &fj in &ek[0..fo] {
                let cj = face_charts(fj);
                if ci == cj {
                    continue;
                }
                push_uniq(chart_adj.entry(ci).or_default(), cj);
                push_uniq(chart_adj.entry(cj).or_default(), ci);
                bd_len += super::dist(mesh.v[vi0], mesh.v[vi1]);
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
            |gi| chart_adj.get(&gi).map_or(&[], Vec::as_slice),
            &pars3d::coloring::HIGH_CONTRAST,
        );

        let mut colored_mesh = mesh.with_face_coloring(&face_coloring);
        colored_mesh.denormalize(s, t);
        colored_mesh.append(&mut wireframe_mesh.clone());
        let out_scene = colored_mesh.into_scene();
        pars3d::save(&args.cluster_vis, &out_scene)?;
    }

    let planarity = per_chart_quadric
        .iter()
        .map(|q| q.a.eigen_sorted().0[1])
        .collect::<Vec<_>>();
    let max_planarity = planarity.iter().copied().max_by(F::total_cmp).unwrap();
    println!("[INFO]: Max planarity is {max_planarity:e}");

    let avg_planarity = planarity.iter().copied().sum::<F>() / per_chart_quadric.len() as F;
    let devs = per_chart_quadric
        .iter()
        .map(|q| q.a.eigen_sorted().0[0])
        .collect::<Vec<_>>();
    let max_developability = devs.iter().copied().max_by(F::total_cmp).unwrap();
    assert!(max_planarity >= max_developability);
    println!("[INFO]: Max developability is {max_developability:e}");

    let avg_developability = devs.iter().copied().sum::<F>() / per_chart_quadric.len() as F;

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

    if !args.stats.is_empty() {
        use std::io::{BufWriter, Write};

        let f = std::fs::File::create(&args.stats)?;
        let mut f = BufWriter::new(f);
        writeln!(
            f,
            r#"{{
  "boundary_len": {bd_len},
  "eigenvalue_max": {max_e:e},
  "max_planarity": {max_planarity:e},
  "avg_planarity": {avg_planarity:e},
  "max_developability": {max_developability:e},
  "avg_developability": {avg_developability:e},
  "num_charts": {num_charts},
  "developability": {devs:?},
  "planarity": {planarity:?}
}}"#
        )?;
    }

    Ok(())
}
