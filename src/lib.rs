#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(cmp_minmax)]

pub type F = f64;
pub type U = u64;

mod vec;
pub use vec::*;

pub mod svd;

pub mod sym;

// represent each vertex as 2D barycentric and merge them together
pub mod quadric;

pub mod union_find;

pub mod inv_map;

pub mod merge;

pub mod manifold;
