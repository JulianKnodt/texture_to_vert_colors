#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

pub type F = f32;

mod vec;
pub use vec::*;

pub mod svd;

pub mod sym;

// pub mod quadric;

