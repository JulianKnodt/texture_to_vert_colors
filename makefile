samply:
	cargo build --release
	samply record target/release/texture_to_vert_colors -i data/scan_vase.obj -o tmp.ply -d \
    data/scan_vase_texture.jpg --target-tri-ratio 1 --sample-kind approx

samply_clustering:
	cargo build --release
	samply record target/release/clustering -i ablations/jar_with_dragon_design.ply -o tmp.ply \
    -t 500 --eigenvalue zero --abs-eps 5e-5
