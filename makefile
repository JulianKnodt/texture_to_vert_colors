samply:
	cargo build --release
	samply record target/release/texture_to_vert_colors -i data/shiba.obj -o tmp.ply -d \
    data/shiba_texture.png --target-tri-num 500000 --no-incremental-qem
