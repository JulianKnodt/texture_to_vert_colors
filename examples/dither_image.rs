#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(unused)]

use clap::Parser;
use ordered_float::NotNan;
use texture_to_vert_colors::quadric::{AttrWeights, Quadric};
use texture_to_vert_colors::{F, dot};

use pars3d::image::{self, GenericImageView};

use priority_queue::PriorityQueue;

use pars3d::FaceKind;
use pars3d::coloring::rgb_to_yiq;

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input image to dither
    #[arg(long, short)]
    input: String,

    /// Output image destination
    #[arg(long, short)]
    output: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let start = std::time::Instant::now();

    let img = image::open(args.input).expect("Failed to load image");

    let mut grayscale =
        image::ImageBuffer::from_fn(img.width(), img.height(), |i, j| luma(img.get_pixel(i, j)));
    image::imageops::dither(&mut grayscale, &image::imageops::BiLevel);
    let mut grayscale_alpha = image::ImageBuffer::from_fn(img.width(), img.height(), |i, j| {
        let image::Rgba([_, _, _, a]) = img.get_pixel(i, j);
        let &image::Luma([l]) = grayscale.get_pixel(i, j);
        image::LumaA([l, a])
    });

    grayscale_alpha
        .save(args.output)
        .expect("Failed to save image");

    Ok(())
}

fn luma(image::Rgba(rgba): image::Rgba<u8>) -> image::Luma<u8> {
    let [r, g, b, _] = rgba.map(|v| v as F);
    let l = (r * 0.299 + g * 0.587 + b * 0.114);
    image::Luma([l as u8])
}
