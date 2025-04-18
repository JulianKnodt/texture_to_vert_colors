#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(cmp_minmax)]

#[cfg(feature = "f64")]
pub type F = f64;
#[cfg(feature = "f64")]
pub type U = u64;

#[cfg(not(feature = "f64"))]
pub type F = f32;
#[cfg(not(feature = "f64"))]
pub type U = u32;

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

pub mod aabb;
