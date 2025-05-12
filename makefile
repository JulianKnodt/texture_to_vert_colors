samply:
	cargo build --release
	samply record target/release/texture_to_vert_colors -i data/scan_vase.obj -o tmp.ply -d \
    data/scan_vase_texture.jpg --target-tri-num 1000000
