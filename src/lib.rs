#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

pub type F = f32;

mod vec;
pub use vec::*;

pub mod svd;

pub mod sym;

// represent each vertex as 2D barycentric and merge them together
pub mod quadric;
