#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use clap::Parser;
use texture_to_vert_colors::{F, dist};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    #[arg(long, short)]
    input: String,
}

fn main() {
    let args = Args::parse();

    let scene =
        pars3d::load(&args.input).expect(&format!("Failed to parse input from {}", args.input));
    let mut m = scene.into_flattened_mesh();
    m.triangulate(0);

    for f in m.f {
        let [v0, v1, v2] = f.as_tri().unwrap();
        let mut e_cd = [[v0, v1], [v1, v2], [v2, v0]].map(|[e0, e1]| {
            let d = dist(m.v[e0], m.v[e1]);
            let cd = dist(m.vert_colors[e0], m.vert_colors[e1]);
            d + cd
        });
        e_cd.sort_unstable_by(F::total_cmp);
        assert!(
            e_cd[2] - 1e-10 <= e_cd[1] + e_cd[0],
            "{e_cd:?} {:?}",
            e_cd[1] + e_cd[0]
        );
    }
    println!("OK");
}
