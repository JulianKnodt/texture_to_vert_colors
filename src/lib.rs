#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(cmp_minmax)]
#![feature(let_chains)]
#![feature(array_windows)]

pub type F = f64;
pub type U = u64;

mod vec;
pub use vec::*;

pub mod svd;

pub mod sym;

// represent each vertex as 2D barycentric and merge them together
pub mod quadric;

pub mod inv_map;

pub mod merge;

pub mod manifold;

pub mod aabb;

pub mod qem;

pub mod weighting;

pub mod clustering;
