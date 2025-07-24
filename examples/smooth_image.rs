use clap::Parser;

use pars3d::image::{self};

#[derive(Debug, Clone, PartialEq, Parser)]
pub struct Args {
    /// Input image to dither
    #[arg(long, short)]
    input: String,

    /// Output image destination
    #[arg(long, short)]
    output: String,

    /// Sigma to use when blurring the image with a gaussian kernel
    #[arg(long, short)]
    sigma: f32,

    /// Unused.
    #[arg(long, default_value_t = String::new())]
    stats: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let img = image::open(args.input).expect("Failed to load image");

    let img = image::imageops::blur(&img, args.sigma);

    img.save(args.output).expect("Failed to save image");

    Ok(())
}
