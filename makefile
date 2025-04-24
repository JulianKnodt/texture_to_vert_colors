samply:
	cargo build --release
	samply record target/release/texture_to_vert_colors -i data/shiba.fbx -o tmp.ply -d \
    data/shiba_texture.png --target-tri-ratio 0.4
