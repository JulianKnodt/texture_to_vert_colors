# Texture Aware Remeshing

Various binaries for first converting a mesh with UV and texture into a mesh with vertex colors,
then processing it in various ways.

To reproduce all experiments in the paper, see `bin/experiments.py`. The data can be made
available on request, but should be entirely reproducible from meshes on Sketchfab.

## Dependencies:

[Rust](https://rust-lang.org/tools/install/)
[UV](https://docs.astral.sh/uv/getting-started/installation/)

## Usage:

Convert UV textured mesh to vertex color mesh:
```
cargo build --release --all-targets
./target/release/texture_to_vert_colors -i <INPUT_MESH> -o <OUTPUT PLY or OBJ> \
  -t <TARGET TRI NUM>
```

If the input mesh is a quad mesh, the above may fail if there are non-planar quads, unless
`--triangulate` is passed.

---

Chartify vertex color mesh:
```
./target/release/clustering -i <INPUT> -o <OUTPUT> -c <CLUSTER_VIS> \
  -t <# CHARTS> --color-eps <COLOR EPS>
```

---

Tutte parameterize vertex color mesh:
```
uv run bin/tutte_param.py -i <INPUT> -o <OUTPUT> --color-weight <FLOAT = 1e-4>
```

---

Remesh vertex color mesh based on texture is actually in a separate repo:

```
git clone git@github.com:JulianKnodt/pars3d.git
cd pars3d
cargo build --release --examples --features rand
./target/release/examples/quad_remesh -i <INPUT> -o <OUTPUT> --scale <FLOAT> \
  --color-field
```
