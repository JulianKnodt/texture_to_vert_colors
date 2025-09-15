import os
import argparse
import sys
import time
import json
from itertools import chain

bin_file = "target/release/texture_to_vert_colors"
tutte_bin = "target/release/tutte_param"
tutte_py_bin = f"{sys.executable} bin/tutte_param.py"
smooth_bin = "target/release/smoothing"
clustering_bin = "target/release/clustering"

hatching_bin = "target/release/hatching"
dithering_bin = "target/release/dithering"
edge_detection_bin = "target/release/edge_detection"

bake_tex_to_vert_colors_bin = "target/release/examples/bake_textures_to_vertex_colors"
bake_vert_colors_to_tex = "target/release/examples/bake_vertex_colors_to_textures"
copy_mesh_to_uv = "target/release/examples/copy_mesh_to_uv"
measure_flat = "target/release/examples/measure_flat"
pars3d_dist_bin = "../pars3d/target/release/distance"
color_smoothing = f"{sys.executable} bin/color_smoothing.py"
gaussian_blur = "target/release/examples/smooth_image"
stripe_pattern = f"{sys.executable} bin/stripes.py"
color_csv = f"{sys.executable} bin/color_csv.py"
pars3d_quad_remesh = f"../pars3d/target/release/examples/quad_remesh"

args = None

abl_dir = "ablations"
cl_dir = "cluster_outputs"

def run(src, dst, flags, out_dir=abl_dir, src_dir="data", bin=bin_file, eval=True, missing_only=False):
  if dst.endswith(".csv"): eval=False
  def cb():
    nonlocal missing_only
    if "run" not in args.stages: return []
    if args.match_output is not None and args.match_output not in dst: return []
    if args.force: missing_only=False
    out_json = f"{out_dir}/{dst[:-4]}.json"
    if missing_only and os.path.exists(out_json) and not args.force:
      print(f"Skipping {src} -> {dst}, destination results {out_json} already exists")
      return []
    out_file = f"{out_dir}/{dst}"
    if missing_only and (not eval) and os.path.exists(out_file):
      print(f"Skipping {src} -> {out_file}, destination already exists")
      return []
    cmds = [f"{bin} -i {src_dir}/{src} -o {out_dir}/{dst} {flags} --stats {out_json}"]
    print(cmds[0])
    if args.eval_only: cmds.pop()
    if (not args.no_eval) and eval:
      if eval == "color":
        cmds.append(f"{pars3d_dist_bin} data/{src} {out_file} \
          --stat {out_json} 1000000 --color-weight 0.01");
      elif ".fbx" in src:
        cmds.append('echo "FBX is not currently supported for Hausdorff"')
      else:
        cmds.append(f"{sys.executable} bin/hausdorff.py -o data/{src} -n {out_file} --stats {out_json}")

    return cmds
  return cb

def eval(src, dst, out_dir=abl_dir, missing_only=False):
  def cb():
    if "run" not in args.stages: return []
    if args.match_output is not None and args.match_output not in dst: return []
    out_json = f"{out_dir}/{dst[:-4]}.json"
    out_file = f"{out_dir}/{dst}"
    if missing_only and os.path.exists(out_json) and not args.force: return []
    return [
      f"{sys.executable} bin/hausdorff.py -o data/{src} -n {out_file} \
        --stats {out_json}"
    ]

  return cb

def render(
  i, cy,cz, ly,lz,out, w=1024, h=1024, cx=0, fy=0, lx=0, rz=45,
  extras="",
  missing_only = False
):
  out = os.path.join(os.getcwd(), out)
  def cb():
    if "render" not in args.stages: return []
    if args.match_output is not None and args.match_output not in out: return []
    if (not args.force) and missing_only and os.path.exists(out): return []

    cmd = f"{sys.executable} bin/render.py \
      --mesh {i} \
      --cam-x {cx} --rot-z {rz} \
      --cam-y {cy} --cam-z {cz} --lookat-y {ly:f} --lookat-z {lz} \
      -o {out} --width {w} --floor-y {fy} --lookat-x {lx} --height {h} {extras} "
    if not args.debug_render: cmd += " --final-render --samples 256"
    return [cmd]
  return cb

def runnable_cmds(cmds, output_name="", stage_kind="run", missing_only=False):
  def cb():
    nonlocal missing_only
    if stage_kind not in args.stages: return []
    if args.match_output is not None and args.match_output not in output_name: return []
    if missing_only and not args.force:
      assert(output_name != ""), "Specify output name for runnable cmd to skip missing"
      if os.path.exists(output_name):
        print(f"Skipping cmds -> {output_name}, destination already exists")
        return []
    print(output_name)
    missing_only = "" if not args.missing_only else " --missing-only "
    return [ c + missing_only for c in cmds ]
  return cb

def tutte(lapl_kind, norm, w):
  if norm == "color-only":
    return (lapl_kind, norm, w, "color_only")
  if w == 0.:
    return (lapl_kind, norm, w, "pos_only")
  return (lapl_kind, norm, w, f"{norm}_{w:0.0e}")

dataset = [
  #("vietnam_lantern.fbx", "vietnam_lantern_small.jpeg", 1000000, None),
  #("cabbage.obj", "cabbage_diffuse.jpg", 500000, None),
  #("shiba.obj", "shiba_texture.png", 1000000, None),
  #("watercolor_girl.obj", "", 4000000, None),
  #("watercolor_cake.obj", "watercolor_cake.jpg", 1000000, None),
  #("silent_ash.obj", "silent_ash_texture.png", 10000000, None),
  #("strawberry.obj", "strawberry_textures/diffuse.png", 1000000, None),
  #("ding_censer.obj", "ding_censer_textures/diffuse.jpg", 2000000, 0.75),
  #("musashi_panels.obj", "musashi_panels_textures/diffuse.jpg", 1000000, None),
  #("tiger_lily.obj", "tiger_lily.jpeg", 2000000, None),
  #("shiny_fish.fbx", "shiny_fish_textures/Fishka_2_G_Fish_BaseColor2.jpg", 1000000, None),
  #("japanese_tray.obj", "japanese_tray_textures/diffuse.png", 1000000, None),
  #("jar_with_dragon_design.obj", "jar_with_dragon_design.png", 1000000, None),
  #("japanese_tea_cup.obj", "japanese_tea_cup_texture.png", 500000, None),
  #("eyeball.fbx", "eyeball_base_color.png", 200000, None),

  #("musk_melon.obj", "", 4000000, 0.27555/2),
  #("fire_bellied_newt.obj", "fire_bellied_newt_diffuse.jpg", 1000000, 0.2733/2.),
  #("lychee.obj", "lychee_textures/lychee.jpg", 500000, 0.25),
  #("officebot.obj", "officebot_textures/diffuse.png", 1000000, 0.5),
  #("spot_triangulated.obj", "spot_texture.png", 1000000, None),

  #("building_front.obj", "building_front.jpg", 2000000, None),
  #("japanese_toro.obj", "japanese_toro_textures/japanese_toro_small.png", 900000, None),
  #("breakfast_still_life.obj", "", 1000000, 0.5),
  #("ibis.obj", "", 1000000, 0.5),


  #("chozuya.obj", "", 4000000, None),
  #("flowers_in_vase.obj", "flowers_in_vase.jpg", 2000000, None),
  #("millers-falls-drill.fbx", "millers-falls-drill-textures/diffuse.png", 1000000, None),
  #("garlic_knight.obj", "", 1000000, 0.5),
  #("private_detective.obj", "", 2000000, 1024),
  #("classic_detective_hat.obj", "", 1000000, 1024),
  #("deku_mask.obj", "", 1000000, 0.75),
  #("umbrella_gold.obj", "", 1000000, 0.5),
  #("inari_mask.obj", "", 300000, 0.5),
  #("longevity_buns.obj", "", 1000000, 0.5),

  #("half_life_crate.obj", "", 1000000, 1.),
  #("tiger_butterfly.obj", "tiger_butterfly_diffuse.jpg", 2000000, 2048),

  #("origami_crane.obj", "", 2000000, 0.75),
  #("hot_air_balloon.obj", "", 1000000, 0.5),
  #("dish_with_maple_leaves.obj", "", 2000000, 0.25),
  #("milk_carton.obj", "", 1000000, None),
  #("bag_with_floral_pattern.obj", "", 500000, 0.5),
  #("old_teapot.obj", "", 500000, 0.4),
  #("vase.obj", "vase_2k.png", 150000, 0.5),
  #("teacup.obj", "", 1500000, 0.5),
  #("juice_box.obj", "", 2000000, 0.75),
  #("wet_floor_sign.obj", "", 1000000, 1),
  #("gollyongpo.obj", "", 4000000, 1),
  #("yazd_dome", "", 2000000, 1),
  #("scan_vase.obj", "scan_vase_texture.jpg", 1000000, None),
  #("eye_of_ra.obj", "", 2000000, 0.5),
  #("violin.obj", "", 2000000, 0.25),
  ("momo-gata-teacup.obj", "", 2000000, 0.5),

  ## need to rerun this one and see what the problem is
  #("watermelon.obj", "watermelon.jpg", 2000000, 0.5),

  ## very expensive
  #("wanderers.obj", "", 4000000, 3072),
  # very expensive but doable?
  #("meadowsweet.obj", "meadowsweet_diffuse.jpeg", 500000, 0.5),
  #("apothecary_syrup_vessel.obj", "apothecary_syrup_vessel_diffuse.png", 3000000, 0.5),

  # keeps panicking
  #("green_lamp.obj", "", 2000000, 1.),
]

dataset_direct = [
  #("baking_scallop.obj", "", 700000),
  #("chinese_sacred_lily.obj", "", 250000),
  #("takifugu.obj", "", 250000),
  #("musk_melon.obj", "", 2000000),
  #("oshima_cherry.obj", "", 2000000),
  #("mango.obj", "", 1000000),
  #("nishiki_utsugi.obj", "", 1000000),
]

experiments = {
  "test-render": [
    render(
      "data/cube.obj",
      8, -30, -2, 0, fy=-10.,
      out="test_render.png",
      extras="--flip-light",
    ),
  ],

  "tri-example": [
    render(
      "data/basic_tri.obj",
      22, -0.01, 0, 0, rz=0, fy=-1000,
      out="ablations/basic_tri_input.png",
      missing_only=True,
    ),
    *[
      run("basic_tri.obj", f"basic_tri_{k}.ply", f"-r 1. --sample-kind {k}")
      for k in ["exact", "approx", "direct"]
    ],
    *[
      runnable_cmds([
        f"../pars3d/target/release/examples/wireframe \
          ablations/basic_tri_{k}.ply ablations/basic_tri_{k}_wireframe.ply --width 3e-3"
      ])
      for k in ["exact", "approx", "direct"]
    ],
    *[
      render(
        f"ablations/basic_tri_{k}.ply",
        22, -0.01, 0, 0, rz=0, fy=-1000,
        out=f"ablations/basic_tri_{k}.png",
        extras=f"--wireframe ablations/basic_tri_{k}_wireframe.ply",
        missing_only=True,
      )
      for k in ["exact", "approx", "direct"]
    ],
    render(
      f"data/basic_tri.obj",
      8, 1.49, 0, 1.5, rz=0, fy=-1000,h=512,
      out=f"ablations/basic_tri_input_inset.png",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/basic_tri_{k}.ply",
        8, 1.49, 0, 1.5, rz=0, fy=-1000,h=512,
        out=f"ablations/basic_tri_{k}_inset.png",
        extras=f"--wireframe ablations/basic_tri_{k}_wireframe.ply",
        #missing_only=True,
      )
      for k in ["exact", "approx", "direct"]
    ],
  ],

  # basic test of a cube
  "basic-cube": [
    *[
      run("cube.obj", f"cube_{k}.ply", f"-d data/uv_grid.png -t 100000 --sample-kind {k}")
      for k in ["exact"]#["exact", "approx", "direct"]
    ]
  ],
  "sphere": [
    *[
      run("sphere.obj", f"sphere_{k}.ply", f"-d data/uv_grid.png -t 100000 --sample-kind {k}")
      for k in ["exact", "approx", "direct"]
    ]
  ],
  # test when the input has non-manifold edges
  "non-manifold": [
    run(
      "non_manifold.obj", "non_manifold.ply",
      "-d data/uv_grid.png --target-tri-ratio 0.5 --sample-kind exact",
    ),
  ],

  # robustness tests
  "thin-tri": [
    *[
      run(
        "thin_tri.obj", f"thin_tri_{k}.ply",
        f"-d data/uv_grid.png --target-tri-ratio 1. --sample-kind {k}",
      ) for k in ["approx", "exact"]
    ]
  ],

  # testing non-axis aligned vertices
  "rot-uv": [
    *[
      run(
        "cube_rotated_uv.obj", f"cube_rot_uv_{k}.ply",
        f"-d data/uv_grid.png --target-tri-ratio 1 --sample-kind {k}",
      ) for k in ["approx"]#["exact", "approx"]
    ]
  ],

  "spot": [
    *[
      run(
        "spot_triangulated.obj", f"spot_{k}.ply",
        f"-d data/spot_texture.png --target-tri-ratio 1. \
        --sample-kind {k}",
      ) for k in ["direct", "exact"]
    ]
  ],

  "teaser2": [
    #render(
    #  f"data/space_suit_geometry.obj",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_geometry.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3 \
    #  --wireframe-thickness 6e-3",
    #  missing_only=True,
    #),

    #render(
    #  f"ablations/space_suit.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_approx.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3 \
    #  --wireframe-thickness 6e-3",
    #  missing_only=True,
    #),

    #run(
    #  "../ablations/space_suit.ply",
    #  "space_suit_constant_color.ply",
    #  f"-t 1000 --eigenvalue one --cluster-vis ablations/space_suit_clusters.ply \
    #  --eigen-eps 1e-4 --color-eps 1e-6 --eigen-vis ablations/space_suit_eigen.ply \
    #  --shape-metric max-manhattan-dist",
    #  bin=clustering_bin, eval=False,
    #),

    #run(
    #  "../ablations/space_suit.ply",
    #  "space_suit_shape_only_colors.ply",
    #  f"-t 1000 --eigenvalue one --cluster-vis ablations/space_suit_shape_only_clusters.ply \
    #  --eigen-eps 1e-7 --color-eps 10000 --eigen-vis ablations/space_suit_shape_only_eigen.ply \
    #  --shape-metric max-manhattan-dist",
    #  bin=clustering_bin, eval=False,
    #),

    #render(
    #  f"ablations/space_suit_shape_only_colors.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_shape_only_colors.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    #),
    #render(
    #  f"ablations/space_suit_shape_only_clusters.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_shape_only_clusters.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    #),

    #render(
    #  f"ablations/space_suit_constant_color.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_constant_color.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    #),
    #render(
    #  f"ablations/space_suit_clusters.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_clusters.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    #),

    #run(
    #  "../ablations/space_suit.ply",
    #  "space_suit_edges.ply",
    #  "--smoothing-iters 0 --min-val 2e-3 --max-val 3e-3 --cone-angle-degrees 30",
    #  bin=edge_detection_bin, eval=False,
    #),
    #render(
    #  f"ablations/space_suit_edges.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_edges.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    #),

    #render(
    #  "ablations/space_suit.ply",
    #  3, -3, 3, 0, fy=-1000, rz=0, cx=0,lx=0,
    #  out="ablations/space_suit_zoom.png",
    #  extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3 \
    #  --wireframe-thickness 5e-3",
    #  missing_only=True,
    #),

    #run(
    #  "../ablations/space_suit.ply",
    #  "space_suit_color_smoothed.ply",
    #  "--weight 0.05 --color-weight 0.5 --color-kind add",
    #  bin=color_smoothing, eval=False,
    #),
    #render(
    #  f"ablations/space_suit_color_smoothed.ply",
    #  1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
    #  out=f"ablations/space_suit_color_smoothed.png",
    #  extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    #),

    # --- TUTTE PARAM
    runnable_cmds([
      f"{sys.executable} bin/tutte_param.py -i ablations/space_suit.ply \
        -o ablations/space_suit_pos_only.ply \
        --color-weight 0 --color-kind add",
      f"{copy_mesh_to_uv} -i ablations/space_suit.ply \
        -u ablations/space_suit_pos_only.ply \
        -o ablations/space_suit_pos_only.ply",
      f"{bake_vert_colors_to_tex} -i ablations/space_suit_pos_only.ply \
      -o ablations/space_suit_pos_only.obj \
      --bake-res 1200 \
      --bake-texture space_suit_pos_only.png",
      f"rm ablations/space_suit_pos_only.ply",
    ], output_name=f"ablations/space_suit_pos_only.obj"),

    render(
      f"ablations/space_suit_pos_only.obj",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit_pos_only_3d.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    render(
      f"ablations/space_suit_pos_only.obj",
      0.3, -1., 0.3, 0, fy=-1000, rz=30, cx=0.75,lx=0.5,
      out=f"ablations/space_suit_pos_only_zoom_3d.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    runnable_cmds([
      f"{sys.executable} bin/tutte_param.py -i ablations/space_suit.ply \
        -o ablations/space_suit_color_tutte.ply \
        --color-weight 2e-1 --color-kind add",
      f"{copy_mesh_to_uv} -i ablations/space_suit.ply \
        -u ablations/space_suit_color_tutte.ply \
        -o ablations/space_suit_color_tutte.ply",
      f"{bake_vert_colors_to_tex} -i ablations/space_suit_color_tutte.ply \
      -o ablations/space_suit_color_tutte.obj \
      --bake-res 1200 \
      --bake-texture space_suit_color_tutte.png",
      f"rm ablations/space_suit_color_tutte.ply",
    ], output_name=f"ablations/space_suit_color_tutte.obj"),

    render(
      f"ablations/space_suit_color_tutte.obj",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit_color_tutte_3d.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    render(
      f"ablations/space_suit_color_tutte.obj",
      0.3, -1., 0.3, 0, fy=-1000, rz=30, cx=0.75,lx=0.5,
      out=f"ablations/space_suit_color_tutte_zoom_3d.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    # --- END TUTTE PARAM
  ],

  "teaser": [
    # --- Input
    render(
      "data/wanderers.obj",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="outputs/wanderers_input.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
      missing_only=True,
    ),
    render(
      "data/wanderers.obj",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="outputs/wanderers_input_bright.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 0 \
        --ambient-light 50 --wireframe-thickness 5e-3",
      missing_only=True,
    ),

    # Direct remesh render
    render(
      "outputs/wanderers_approx.ply",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="outputs/wanderers_remesh.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
      missing_only=True,
    ),
    render(
      "outputs/wanderers_approx.ply",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="outputs/wanderers_remesh_bright.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 0 \
        --ambient-light 50 --wireframe-thickness 5e-3",
      missing_only=True,
    ),
    # ---

    # --- Zoom
    render(
      "data/wanderers.obj",
      1, -0.5, 1.2, 0, fy=-1000, rz=0, cx=2.5,lx=2.3,
      out="outputs/wanderers_input_bright_zoom.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 0 \
        --ambient-light 50 --wireframe-thickness 5e-3",
      missing_only=True,
    ),
    render(
      "outputs/wanderers_approx.ply",
      1, -0.5, 1.2, 0, fy=-1000, rz=0, cx=2.5,lx=2.3,
      out="outputs/wanderers_remesh_zoom.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
      missing_only=True,
    ),
    render(
      "outputs/wanderers_approx.ply",
      1, -0.5, 1.2, 0, fy=-1000, rz=0, cx=2.5,lx=2.3,
      out="outputs/wanderers_remesh_bright_zoom.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 0 \
        --ambient-light 50 --wireframe-thickness 5e-3",
      missing_only=True,
    ),

    # ---

    run(
      "../outputs/wanderers_approx.ply",
      "wanderers_constant_colors.ply",
      f"-t 2000 --eigenvalue zero --cluster-vis ablations/wanderers_clusters.ply \
      --eigen-eps 100000 --color-eps 1e-6 --eigen-vis ablations/wanderers_eigen.ply \
      --shape-metric boundary-length",
      bin=clustering_bin, eval=False,
    ),

    *[
      render(
        f"ablations/wanderers_{k}.ply",
        1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
        out=f"ablations/wanderers_{k}.png",
        extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
      ) for k in ["constant_colors", "clusters", "eigen"]
    ],

    run(
      "space_suit.obj", "space_suit.ply",
      "-d data/space_suit.png --target-tri-num 500000 \
      --sample-kind approx --triangulate --image-size-frac 1",
      missing_only=True,
    ),

    run(
      "space_suit.obj", "space_suit.obj",
      "-d data/space_suit.png --target-tri-num 500000 \
      --sample-kind approx --triangulate --image-size-frac 1",
      missing_only=True,
    ),


    run(
      "../ablations/space_suit.ply",
      f"space_suit_tutte.obj",
      f"--weighting laplacian --bake-texture space_suit_tutte_texture.png \
        --uv-svg ablations/space_suit_tutte.svg --iters 500000 \
        --pos-color-norm add --color-weight 3e-3 --bake-res 2048",
      bin=tutte_bin, eval=False,
    ),

    render(
      f"data/space_suit.obj",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit_input.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    render(
      f"ablations/space_suit_tutte.obj",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    runnable_cmds([
      "inkscape ablations/space_suit_tutte.svg --export-pdf=ablations/space_suit_tutte.pdf",
      "convert ablations/space_suit_tutte.pdf -resize 1024x1024 ablations/space_suit_tutte.png",
    ], output_name="space_suit_tutte"),

    run(
      "../outputs/wanderers_approx.ply",
      "wanderers_dither.ply",
      "--weighting laplacian --color-weight 0.5 --order nearest --face",
      bin=dithering_bin, eval=False,
      #missing_only=True,
    ),

    render(
      "ablations/wanderers_dither.ply",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="ablations/wanderers_dither.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    # --- Edge Detection
    run(
      "../outputs/wanderers_approx.ply",
      "wanderers_edges.ply",
      "--smoothing-iters 0 --min-val 9e-4 --max-val 1e-3 --cone-angle-degrees 30",
      bin=edge_detection_bin, eval=False,
    ),

    render(
      "ablations/wanderers_edges.ply",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="ablations/wanderers_edges.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),
    # --- Texture Smoothing
    run(
      "../outputs/wanderers_approx.ply",
      "wanderers_color_smoothed.ply",
      "--weight 0.2 --color-weight 64 --color-kind add",
      bin=color_smoothing, eval=False,
    ),

    render(
      "ablations/wanderers_color_smoothed.ply",
      1, -5.5, 2, 0, fy=-1000, rz=0, cx=2.5,lx=1,
      out="ablations/wanderers_color_smoothed.png",
      extras="--flip-light --light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),
    # --- Vector Fields

    run(
      "../ablations/space_suit.ply",
      "space_suit_vector_field.ply",
      f"--dist-thresh 1e-6 --color-thresh 0.0 --width 5e-4 --length 0.02 --face-hatching",
      bin=hatching_bin, eval=False,
    ),

    render(
      "ablations/space_suit_vector_field.ply",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit_vector_field.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),

    run(
      "../ablations/space_suit.ply",
      "space_suit_remesh.obj",
      "--color-field --scale 0.008 --subdivisions 1 --orient-iters 50 \
        --save-grid ablations/space_suit_grid.ply \
        --save-arrows ablations/space_suit_gradient.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),

    render(
      "ablations/space_suit_remesh.obj",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit_remesh.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3 \
        --wireframe ablations/space_suit_grid.ply",
    ),

    render(
      "ablations/space_suit_remesh.obj",
      1.5, -3, 1.5, 0, fy=-1000, rz=30, cx=1,lx=-1,
      out=f"ablations/space_suit_remesh_zoom.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3 \
        --wireframe ablations/space_suit_grid.ply",
    ),

    render(
      "ablations/space_suit_gradient.ply",
      1.5, -10., 1.5, 0, fy=-1000, rz=30, cx=2.5,lx=-1,
      out=f"ablations/space_suit_remesh_grad.png",
      extras="--light-z -50 --roughness 0.8 --light-strength 18 --ambient-light 3",
    ),
  ],

  "spot-clustering": [
    #run(
    #  "spot_triangulated.obj",
    #  "spot.ply",
    #  f"-d data/spot_texture.png --target-tri-num 50000 --no-incremental-qem",
    #),
    run(
      "../ablations/spot.ply",
      "spot_constant_colors.ply",
      f"-t 50 --eigenvalue zero --cluster-vis ablations/spot_clusters.ply --eigen-eps 1e-2 \
      --color-eps 0. --eigen-vis ablations/spot_eigen.ply",
      bin=clustering_bin, eval=False,
    ),
  ],
  "dragon_jar_clustering": [
    #run(
    #  "jar_with_dragon_design.obj",
    #  "jar_with_dragon_design.ply",
    #  f"-d data/jar_with_dragon_design.png --target-tri-num 300000 --no-incremental-qem",
    #),
    run(
      "../outputs/jar_with_dragon_design.ply",
      "jar_with_dragon_design_colors.ply",
      f"-t 500 --eigenvalue zero --cluster-vis ablations/jar_with_dragon_design_clusters.ply \
      --abs-eps 1e-4",
      bin=clustering_bin, eval=False,
    ),
  ],
  "angelfish_clustering": [
    #run(
    #  "angelfish.obj",
    #  "angelfish.ply",
    #  f"-d data/angelfish_texture.jpg --target-tri-num 1000000 --no-incremental-qem \
    #  --image-size-px 2048 --sample-kind approx",
    #  missing_only=True,
    #),
    run(
      "angelfish.obj",
      "angelfish.ply",
      f"-d data/angelfish_texture.jpg --target-tri-num 600000 \
      --sample-kind direct",
      missing_only=True,
    ),
    run(
      "../ablations/angelfish.ply",
      "angelfish_colors.ply",
      f"-t 250 --eigenvalue zero --cluster-vis ablations/angelfish_clusters.ply \
      --eigen-eps 5e-4 --color-eps 1e-6 --eigen-vis ablations/angelfish_eigen.ply",
      bin=clustering_bin, eval=False,
    ),
    render(
      "data/angelfish.obj",
      0, -26, 0, 0, fy=-9.9, rz=0,cx=1.5,lx=1.5, h=640,
      out="ablations/angelfish_input.png",
      extras="--light-z -80",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/angelfish_{l}.ply",
        0, -26, 0, 0, fy=-9.9, rz=0,cx=1.5,lx=1.5, h=640,
        out=f"ablations/angelfish_{l}.png",
        extras="--light-z -80 --roughness 1",
        #missing_only=True,
      ) for l in ["colors", "clusters", "eigen"]
    ],
  ],
  "nanchan-clustering": [
    run(
      "nanchan.obj", "nanchan.ply",
      f"-d data/nanchan_textures/diffuse.png --target-tri-num 800000 --sample-kind exact",
      missing_only=True,
    ),
    run(
      "../ablations/nanchan.ply",
      "nanchan_colors.ply",
      f"-t 250 --eigenvalue one --cluster-vis ablations/nanchan_clusters.ply \
      --eigen-eps 1e-4 --color-eps 1e-6 --shape-metric convexity \
      --eigen-vis ablations/nanchan_eigens.ply",
      bin=clustering_bin, eval=False,
    ),
    #run(
    #  "../ablations/nanchan.ply",
    #  "nanchan_colors.ply",
    #  f"-t 200 --eigenvalue one --cluster-vis ablations/nanchan_clusters.ply \
    #  --eigen-eps 0. --eigenvalue-vis ablations/nanchan_eigens.ply",
    #  bin=clustering_bin, eval=False,
    #),
    render(
      "data/nanchan.obj",
      -3.5, -18, -3.5, 0, fy=-9.9, rz=-15, w=840,
      out="ablations/nanchan_input.png",
      extras="--light-z -80",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/nanchan_{l}.ply",
        -3.5, -18, -3.5, 0, fy=-9.9, rz=-15, w=840,
        out=f"ablations/nanchan_{l}.png",
        extras="--light-z -80",
      )
      for l in ["clusters", "eigens", "colors"]
    ]
  ],
  "origami-crane-clustering": [
    run(
      "../outputs/origami_crane_approx.ply",
      "origami_crane_colors.ply",
      f"-t 250 --eigenvalue zero --cluster-vis ablations/origami_crane_clusters.ply \
      --eigen-eps 3e-4 --color-eps 0 --shape-metric boundary-length \
      --eigen-vis ablations/origami_crane_eigens.ply",
      bin=clustering_bin, eval=False,
    ),
    render(
      "data/origami_crane.obj",
      8, -19, -4, 4.5, fy=-7, rz=-35, lx=2.5, cx=2.5,
      out="ablations/origami_crane_input.png",
      extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/origami_crane_{l}.ply",
        8, -19, -4, 4.5, fy=-7, rz=-35, lx=2.5, cx=2.5,
        out=f"ablations/origami_crane_{l}.png",
        extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      )
      for l in ["colors", "clusters", "eigens"]
    ]
  ],

  "ablate-bd-cleanliness": [
    *[
      run(
        "prestige_stool_geometry.obj",
        f"prestige_stool_{lbl}.ply",
        f"-t 100 --eigenvalue one --cluster-vis ablations/prestige_stool_{lbl}_clusters.ply \
        --eigen-eps {0 if k == 'none' else 1e-7} --color-eps 10000 --shape-metric {k} \
        --eigen-vis ablations/prestige_stool_{lbl}_eigens.ply",
        bin=clustering_bin, eval=False,
      )
      for lbl, k in [
        ("bd", "boundary-length"),
        ("mh", "max-manhattan-dist"),
        ("me", "max-euclidean-dist"),
        ("none", "none"),
        ("ad", "angle-deviation"),
        ("sbd", "shared-boundary-length"),
        ("cvx", "convexity"),
        ("area", "area"),
      ]
    ],

    render(
      "data/prestige_stool.obj",
      13.5, -15.5, 5.5, 0, fy=0, rz=-45, w=800, cx=0.5, lx=0.5,
      out="ablations/prestige_stool_input.png",
      extras="--light-z -60 --roughness 1 --ambient-light 0.1 --wireframe-thickness 3e-3",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/prestige_stool_{lbl}_clusters.ply",
        13.5, -15.5, 5.5, 0, fy=0, rz=-45, w=800, cx=0.5, lx=0.5,
        out=f"ablations/prestige_stool_{lbl}_clusters.png",
        extras="--light-z -60 --roughness 1 --ambient-light 0.1",
        missing_only=True,
      )
      for lbl, _ in [
        ("bd", "boundary-length"),
        ("mh", "max-manhattan-dist"),
        ("none", "none"),
        ("me", "max-euclidean-dist"),
        ("ad", "angle-deviation"),
        ("sbd", "shared-boundary-length"),
        ("cvx", "convexity"),
        ("area", "area"),
      ]
    ],

    render(
      f"data/prestige_stool.obj",
      7, -9, 7, 0, fy=0, rz=-90,h=512,
      out=f"ablations/prestige_stool_inset.png",
      extras="--light-z -60 --roughness 1 --ambient-light 0.1",
      missing_only=True,
    ),

    *[
      render(
        f"ablations/prestige_stool_{lbl}_clusters.ply",
        7, -9, 7, 0, fy=0, rz=-90,h=512,
        out=f"ablations/prestige_stool_{lbl}_inset.png",
        extras="--light-z -60 --roughness 1 --ambient-light 0.1",
        missing_only=True,
      )
      for lbl, _ in [
        ("bd", "boundary-length"),
        ("mh", "max-manhattan-dist"),
        ("none", "none"),
      ]
    ],
  ],

  "ablate-cluster-eigen": [
    *[
      run(
        "terpsichore_lyran.obj",
        f"terpsichore_lyran_{lbl}.ply",
        f"-t 600 --eigenvalue {k} --cluster-vis ablations/terpsichore_lyran_{lbl}_clusters.ply \
        --eigen-eps {3e-8 if k == 'two' else 5e-12} --color-eps 10000 --geometry-only --shape-metric max-manhattan-dist \
        --eigen-vis ablations/terpsichore_lyran_{lbl}_eigens.ply",
        bin=clustering_bin, eval=False,
      ) for lbl, k in [
        ("dev", "zero"),
        ("planar", "one"),
        ("equiareal", "two"),
      ]
    ],

    render(
      "data/terpsichore_lyran.obj",
      10.5, -16, 6, 0, fy=-0.1, rz=150, w=512,
      out="ablations/terpsichore_lyran_input.png",
      extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      missing_only=True,
    ),

    *[
      render(
        f"ablations/terpsichore_lyran_{lbl}_clusters.ply",
        10.5, -16, 6, 0, fy=-0.1, rz=150, w=512,
        out=f"ablations/terpsichore_lyran_{lbl}_clusters.png",
        extras="--light-z -80 --roughness 1 --ambient-light 0.1",
        #missing_only=True,
      ) for lbl in [ ("dev"), ("planar"), ("equiareal") ]
    ],

    # Insets
    #render(
    #  "data/terpsichore_lyran.obj",
    #  18, -0.1, 1.2, 0, fy=0.15, rz=0, h=360, cx=-0.5, lx=-0.5,
    #  out="ablations/terpsichore_lyran_input_inset.png",
    #  extras="--light-z -80 --roughness 1 --ambient-light 0.1",
    #  missing_only=True,
    #),

    #*[
    #  render(
    #    f"ablations/terpsichore_lyran_{lbl}_clusters.ply",
    #    18, -0.1, 1.2, 0, fy=0.15, rz=0, h=360, cx=-0.5, lx=-0.5,
    #    out=f"ablations/terpsichore_lyran_{lbl}_clusters_inset.png",
    #    extras="--light-z -80 --roughness 1 --ambient-light 0.1",
    #    #missing_only=True,
    #  ) for lbl in [ ("dev"), ("planar"), ("equiareal") ]
    #],
  ],

  "ablate-delta-cost": [
    *[
      run(
        "stern_of_ss_rifle.obj",
        f"stern_of_ss_rifle_{lbl}.ply",
        f"-t 200 --eigenvalue one --cluster-vis ablations/stern_of_ss_rifle_{lbl}_clusters.ply \
        --eigen-eps 1e-9 --color-eps 10000 --geometry-only --shape-metric max-manhattan-dist \
        --eigen-vis ablations/stern_of_ss_rifle_{lbl}_eigens.ply {k}",
        bin=clustering_bin, eval=False,
      )
      for lbl, k in [
        ("delta_cost", ""),
        ("no_delta_cost", "--no-delta-cost"),
      ]
    ],

    render(
      "data/stern_of_ss_rifle.obj",
      10.5, -18.5, 1.2, 0, fy=0.15, rz=30, h=720, cx=1.8,lx=1.8,
      out="ablations/stern_of_ss_rifle_input.png",
      extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      missing_only=True,
    ),

    *[
      render(
        f"ablations/stern_of_ss_rifle_{lbl}_clusters.ply",
        10.5, -18.5, 1.2, 0, fy=0.15, rz=30, h=720, cx=1.8,lx=1.8,
        out=f"ablations/stern_of_ss_rifle_{lbl}_clusters.png",
        extras="--light-z -80 --roughness 1 --ambient-light 0.1",
        #missing_only=True,
      ) for lbl in [ "delta_cost", "no_delta_cost" ]
    ],

    # Insets
    render(
      "data/stern_of_ss_rifle.obj",
      18, -0.1, 1.2, 0, fy=0.15, rz=90, h=360, cx=-0.5, lx=-0.5,
      out="ablations/stern_of_ss_rifle_input_inset.png",
      extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      missing_only=True,
    ),

    *[
      render(
        f"ablations/stern_of_ss_rifle_{lbl}_clusters.ply",
        18, -0.1, 1.2, 0, fy=0.15, rz=90, h=360, cx=-0.5, lx=-0.5,
        out=f"ablations/stern_of_ss_rifle_{lbl}_clusters_inset.png",
        extras="--light-z -80 --roughness 1 --ambient-light 0.1",
        #missing_only=True,
      ) for lbl in [ "delta_cost", "no_delta_cost" ]
    ],
  ],

  "clustering-dish-with-maple-leaves": [
    run(
      "../outputs/dish_with_maple_leaves_approx.ply",
      "dish_with_maple_leaves_colors.ply",
      f"-t 350 --eigenvalue one --cluster-vis ablations/dish_with_maple_leaves_clusters.ply \
      --eigen-eps 1e-2 --color-eps 1e-6 --shape-metric boundary-length \
      --eigen-vis ablations/dish_with_maple_leaves_eigens.ply",
      bin=clustering_bin, eval=False,
    ),
    render(
      "data/dish_with_maple_leaves.obj",
      24.5, -21.5, 0, 0, fy=-3, rz=135, h=800,
      out="ablations/dish_with_maple_leaves_input.png",
      extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      #missing_only=True,
    ),
    *[
      render(
        f"ablations/dish_with_maple_leaves_{l}.ply",
        24.5, -21.5, 0, 0, fy=-3, rz=135, h=800,
        out=f"ablations/dish_with_maple_leaves_{l}.png",
        extras="--light-z -80 --roughness 1 --ambient-light 0.1",
      )
      for l in ["colors", "clusters", "eigens"]
    ],
  ],

  "ablate-clustering": [
    *[
      cmd
      for (mesh, clusters, eigen_eps, color_eps, label, e_max) in [
        #("dish_with_maple_leaves", 75, 1e-10, 10000, "planar"),
        #("dish_with_maple_leaves", 75, 1000, 0, "color_only"),
        #("dish_with_maple_leaves", 75, 1e-7, 0, "mixed"),

        ("milk_carton", 22, 1e-8, 10000, "planar", 5),
        ("milk_carton", 22, 1000, 0, "color_only", 5),
        ("milk_carton", 22, 1e-4, 0, "mixed", 5),
      ]
      for cmd in [
        run(
          f"../outputs/{mesh}_approx.ply",
          f"{mesh}_{label}_colors.ply",
          f"-t {clusters} --eigenvalue one \
          --cluster-vis ablations/{mesh}_{label}_clusters.ply \
          --eigen-eps {eigen_eps} --color-eps {color_eps} \
          --shape-metric boundary-length \
          --eigen-vis ablations/{mesh}_{label}_eigen.ply \
          --max-eigen {e_max}",
          bin=clustering_bin, eval=False,
          missing_only=True,
        ),
      ]
    ],
    render(
      "data/milk_carton.obj",
      8, -18, 6, 0, fy=0, rz=45, w=640,
      out=f"ablations/milk_carton_input.png",
      extras="--light-z -80 --light-x 20",
      missing_only=True,
    ),
    *[
      cmd
      for vis in ["colors", "clusters", "eigen"]
      for cmd in [
        render(
          f"ablations/milk_carton_{k}_{vis}.ply",
          8, -18, 6, 0, fy=0, rz=45, w=640,
          out=f"ablations/milk_carton_{k}_{vis}.png",
          extras="--light-z -80 --light-x 20",
        ) for k in ["planar", "color_only", "mixed"]
      ]
    ]
  ],

  "dense-sphere-smooth-boundaries": [
    #run("dense_sphere.obj", "dense_sphere.ply", "-d data/hokusai.jpg \
    #--sample-kind approx -t 500000")
  ],

  "ablate-diff-texture": [
    *[
      run(
        "bust_of_roza_loewenfeld.obj",
        f"bust_of_roza_loewenfeld_{lbl}.ply",
        f"-d data/bust_of_roza_loewenfeld_textures/{tex} --sample-kind approx -t 1000000 \
        --image-size-px 1500",
        missing_only=True,
        eval=False,
      ) for (lbl, tex) in [
        ("diffuse", "diffuse.jpg"),
        ("normals", "normals.png"),
        #("roughness", "roughness.png"),
      ]
    ],

    render(
      "data/bust_of_roza_loewenfeld.obj",
      1.25, -30, 1.25, 0, fy=-8.7, rz=0, w=660, cx=-0.25,lx=-0.25,
      out="ablations/bust_of_roza_loewenfeld_input.png",
      extras="--light-z -80",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/bust_of_roza_loewenfeld_{lbl}.ply",
        1.25, -30, 1.25, 0, fy=-8.7, rz=0, w=660, cx=-0.25,lx=-0.25,
        out=f"ablations/bust_of_roza_loewenfeld_{lbl}.png",
        extras="--light-z -80",
        missing_only=True,
      ) for lbl in ["diffuse", "normals"]
    ]
  ],

  # Test case for smoothing
  "plane-smoothing": [
    #run(
    #  "plane.obj", "hokusai.ply",
    #  "-d data/a_hollyhock_herman_saftleven.jpg --target-tri-ratio 0.03"
    #),
    *[
      run(
        "../ablations/hokusai.ply",
        f"hokusai_{label}.ply",
        f"--weighting {weighting} --iters 50 \
        --pos-color-norm {norm}",
        bin=smooth_bin, eval=False,
      )
      for (weighting, norm, label) in [
        ("uniform", "pos-only", "uniform"),

        ("length", "pos-only", "len"),
        ("length", "geometric-mean", "clen"),

        ("mean-value", "pos-only", "mv"),
        ("mean-value", "geometric-mean", "cmv"),

        ("laplacian", "pos-only", "lpl"),
        ("laplacian", "geometric-mean", "clpl"),

        ("laplacian", "bilateral", "bilpl"),
      ]
    ],
  ],
  "hokusai": [ run("plane.obj", "hokusai_plane.ply", "-d data/hokusai.jpg --target-tri-ratio 0.3") ],
  "vase": [
    run("vase.fbx", "vase.ply", "-d data/vase_2k.png", False),
  ],
  "flowers-in-vase": [
    run(
      "flowers_in_vase.obj", "flowers_in_vase.ply",
      "-d data/flowers_in_vase.jpg --no-final-qem --no-incremental-delete --target-tri-ratio 0.5", False,
    ),
  ],

  "tutte-param-example": [
    #run("open_top_box.obj", "open_top_box.ply", "-d data/spot_texture.png --target-tri-ratio 0.5"),
    #run("open_top_box.obj", "open_top_box.ply", "-d data/hokusai.jpg --target-tri-ratio 0.3"),
    #run("open_top_box.obj", "open_top_box.ply", "-d data/uv_grid.png --target-tri-ratio 0.5"),
    *[
      run(
        "../ablations/open_top_box.ply",
        f"open_top_box_{label}.obj",
        f"--weighting {weighting} --bake-texture open_top_box_{label}.png \
          --uv-svg ablations/open_top_box_{label}.svg --iters 100000 \
          --pos-color-norm {norm} --color-weight {cw} --bake-res 512",
        bin=tutte_bin, eval=False,
      )
      for (weighting, norm, cw, label) in [
        #("uniform", "add", 0., "uniform"),

        #("length", "add", 0., "len_pos_only"),
        #("length", "add", 0.1, "len_add_0_1"),
        #("length", "add", 1., "len_add_1"),
        #("length", "add", 10., "len_add_10"),

        ("length", "concat", 0.1, "len_concat_0_1"),
        ("length", "concat", 1., "len_concat_1"),
        ("length", "concat", 10., "len_concat_10"),


        ("laplacian", "add", 0., "lpl_pos_only"),

        ("laplacian", "max", 0.05, "lpl_max_0_05"),
        ("laplacian", "max", 0.1, "lpl_max_0_1"),
        ("laplacian", "max", 1., "lpl_max_1"),

        ("laplacian", "concat", 0.05, "lpl_concat_0_05"),
        ("laplacian", "concat", 0.1, "lpl_concat_0_1"),
        ("laplacian", "concat", 1, "lpl_concat_1"),

        ("laplacian", "add", 0.1, "lpl_add_0_1"),
        ("laplacian", "add", 1, "lpl_add_1"),
        ("laplacian", "add", 10, "lpl_add_10"),
      ]
    ],
  ],

  "tutte-param": [
    cmd
    for (model, ratio, sample_kind, triangulate, img_frac, bake_res, w_mul) in [
      #("scroll.obj", 0.05, "approx", True, 0.5, 1024),
      #("scroll_constant.obj", 0.15, "approx", True, 1, 2048, 1),
      #("jar_with_dragon_design_boundary.obj", 0.4, "approx", True, 1., 512, 8e-2),
      #("ogre.obj", 0.1, "approx", True, 0.5, 512, 5e-2),
      #("longevity_buns.obj", 0.09, "approx", True, 0.5, 512, 4e-1),
      #("japanese_lantern.obj", 0.025, "approx", True, 1., 512, 1.),
      #("perfume_bottle.obj", 0.15, "approx", True, 0.25, 1024, 1),
      #("yazd_dome.obj", 0.4, "approx", True, 0.25, 1024, 0.1),
      ("yongchok.obj", 0.05, "approx", True, 0.5, 2048, 0.01),
    ]
    for cmd in [
      run(
        model, model[:-4] + ".ply",
        f"--target-tri-ratio {ratio} --sample-kind {sample_kind} \
        {'--triangulate' if triangulate else ''} \
        --image-size-frac {img_frac}",
        missing_only=True,
      ),
      *[
        runnable_cmds([
          f"{sys.executable} bin/tutte_param.py -i ablations/{model[:-4]}.ply \
            -o ablations/{model[:-4]}_{label}.ply \
            --color-weight {cw} \
            {f'--color-kind {norm}' if norm != 'uniform' else '--uniform'}",
          f"{copy_mesh_to_uv} -i ablations/{model[:-4]}.ply \
            -u ablations/{model[:-4]}_{label}.ply \
            -o ablations/{model[:-4]}_{label}.ply",
          f"{bake_vert_colors_to_tex} -i ablations/{model[:-4]}_{label}.ply \
          -o ablations/{model[:-4]}_{label}.obj \
          --bake-res {bake_res} \
          --bake-texture {model[:-4]}_{label}.png",
          f"rm ablations/{model[:-4]}_{label}.ply",

          f"{sys.executable} bin/hausdorff.py -o data/{model} \
            -n ablations/{model[:-4]}_{label}.obj \
            --stats ablations/{model[:-4]}_{label}.json",
        ], output_name=f"ablations/{model[:-4]}_{label}.obj", missing_only=True)
        #run(
        #  f"../ablations/{model[:-4]}.ply",
        #  f"{model[:-4]}_{label}.obj",
        #  f"--weighting {w} --pos-color-norm {norm} \
        #    --uv-svg ablations/{model[:-4]}_{label}.svg --bake-texture \
        #    {model[:-4]}_{label}.png --iters 2000000 --color-weight {cw} \
        #    --bake-res {bake_res} --use-longest-loop",
        #  bin=tutte_bin, eval=False, missing_only=True,
        #)
        for (w, norm, cw, label) in [
          #("uniform", "add", 0., "uniform"),

          ("laplacian", "add", 0., "pos_only"),
          #("laplacian", "color-only", 0., "lpl_color_only"),

          tutte("laplacian", "add", 3e-2 * w_mul),
          tutte("laplacian", "add", 1e-1 * w_mul),
          tutte("laplacian", "add", 3e-1 * w_mul),

          tutte("laplacian", "concat", 3e-2 * w_mul),
          tutte("laplacian", "concat", 1e-1 * w_mul),
          tutte("laplacian", "concat", 3e-1 * w_mul),

          tutte("laplacian", "max", 3e-2 * w_mul),
          tutte("laplacian", "max", 1e-1 * w_mul),
          tutte("laplacian", "max", 3e-1 * w_mul),

          # experiment
          #("laplacian", "add", 10, "lpl_add_10"),
        ]
      ],
    ]
  ],

  "tutte-param-render": [
    render(
      "data/scroll_constant.obj",
      0, -26, 0, 0, fy=-4, rz=0, cx=2,lx=2, h=560,
      out="ablations/scroll_constant.png",
      extras="--light-z -80",
      missing_only=True,
    ),
    render(
      "ablations/scroll_constant_pos_only.obj",
      0, -26, 0, 0, fy=-4, rz=0, cx=2,lx=2, h=560,
      out="ablations/scroll_constant_pos_only_3d.png",
      extras="--light-z -80 --roughness 1 --shade-flat",
    ),
    render(
      "ablations/scroll_constant_add_3e-01.obj",
      0, -26, 0, 0, fy=-4, rz=0, cx=2,lx=2, h=560,
      out="ablations/scroll_constant_add_3e-01_3d.png",
      extras="--light-z -80 --roughness 1 --shade-flat",
    ),
    render(
      "ablations/scroll_constant_concat_3e-01.obj",
      0, -26, 0, 0, fy=-4, rz=0, cx=2,lx=2, h=560,
      out="ablations/scroll_constant_concat_3e-01_3d.png",
      extras="--light-z -80 --roughness 1 --shade-flat",
    ),

    # insets
    render(
      "data/scroll_constant.obj",
      0, -13, 0, 0, fy=-4, rz=0, cx=0,lx=0, h=560,
      out="ablations/scroll_constant_inset.png",
      extras="--light-z -80",
      missing_only=True,
    ),
    render(
      "ablations/scroll_constant_pos_only.obj",
      0, -13, 0, 0, fy=-4, rz=0, cx=0,lx=0, h=560,
      out="ablations/scroll_constant_pos_only_3d_inset.png",
      extras="--light-z -80 --roughness 1 --shade-flat",
    ),
    render(
      "ablations/scroll_constant_add_3e-01.obj",
      0, -13, 0, 0, fy=-4, rz=0, cx=0,lx=0, h=560,
      out="ablations/scroll_constant_add_3e-01_3d_inset.png",
      extras="--light-z -80 --roughness 1 --shade-flat",
    ),
    render(
      "ablations/scroll_constant_concat_3e-01.obj",
      0, -13, 0, 0, fy=-4, rz=0, cx=0,lx=0, h=560,
      out="ablations/scroll_constant_concat_3e-01_3d_inset.png",
      extras="--light-z -80 --roughness 1 --shade-flat",
    ),
  ],

  "tutte-param-render-longevity-buns": [
    render(
      "data/longevity_buns.obj",
      15.5, -20, 5.5, 0, fy=4, rz=20, h=800,
      out="ablations/longevity_buns_input.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/longevity_buns_pos_only.obj",
      15.5, -20, 5.5, 0, fy=4, rz=20, h=800,
      out="ablations/longevity_buns_pos_only_3d.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/longevity_buns_add_1e-02.obj",
      15.5, -20, 5.5, 0, fy=4, rz=20, h=800,
      out="ablations/longevity_buns_add_1e-02_3d.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),

    # insets
    render(
      "data/longevity_buns.obj",
      5.5, -10, 5.5, 0, fy=4, rz=10, h=512,
      out="ablations/longevity_buns_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/longevity_buns_pos_only.obj",
      5.5, -10, 5.5, 0, fy=4, rz=10, h=512,
      out="ablations/longevity_buns_pos_only_3d_inset.png",
      extras="--light-z -80 --roughness 1",
    ),
    render(
      "ablations/longevity_buns_add_1e-02.obj",
      5.5, -10, 5.5, 0, fy=4, rz=10, h=512,
      out="ablations/longevity_buns_add_1e-02_3d_inset.png",
      extras="--light-z -80 --roughness 1",
    ),
  ],

  "tutte-param-render-perfume-bottle": [
    render(
      "data/perfume_bottle.obj",
      2.5, -27, -2.5, 0, fy=-11, rz=0, w=800,
      out="ablations/perfume_bottle_input.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/perfume_bottle_pos_only.obj",
      2.5, -27, -2.5, 0, fy=-11, rz=0, w=800,
      out="ablations/perfume_bottle_pos_only_3d.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/perfume_bottle_add_1e-01.obj",
      2.5, -27, -2.5, 0, fy=-11, rz=0, w=800,
      out="ablations/perfume_bottle_add_1e-01_3d.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),

    # insets
    render(
      "data/perfume_bottle.obj",
      -1, -13, -1, 0, fy=-11, rz=0, h=512,
      out="ablations/perfume_bottle_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/perfume_bottle_pos_only.obj",
      -1, -13, -1, 0, fy=-11, rz=0, h=512,
      out="ablations/perfume_bottle_pos_only_3d_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/perfume_bottle_add_1e-01.obj",
      -1, -13, -1, 0, fy=-11, rz=0, h=512,
      out="ablations/perfume_bottle_add_1e-01_3d_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
  ],

  "tutte-param-render-yazd-dome": [
    render(
      "data/yazd_dome.obj",
      21.5, -23.5, -0.5, 0, fy=-0.05, rz=0,h=960,
      out="ablations/yazd_dome_input.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/yazd_dome_pos_only.obj",
      21.5, -23.5, -0.5, 0, fy=-0.05, rz=0,h=960,
      out="ablations/yazd_dome_pos_only_3d.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/yazd_dome_add_3e-02.obj",
      21.5, -23.5, -0.5, 0, fy=-0.05, rz=0,h=960,
      out="ablations/yazd_dome_add_3e-02_3d.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),

    # insets
    render(
      "data/yazd_dome.obj",
      10.5, -13.5, -0.5, 0, fy=-0.05, rz=0,h=480,
      out="ablations/yazd_dome_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/yazd_dome_pos_only.obj",
      10.5, -13.5, -0.5, 0, fy=-0.05, rz=0,h=480,
      out="ablations/yazd_dome_pos_only_3d_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
    render(
      "ablations/yazd_dome_add_3e-02.obj",
      10.5, -13.5, -0.5, 0, fy=-0.05, rz=0,h=480,
      out="ablations/yazd_dome_add_3e-02_3d_inset.png",
      extras="--light-z -80 --roughness 1",
      missing_only=True,
    ),
  ],

  "tutte-param-hyperparam-ablation": [
    cmd
    for (model, ratio, sample_kind, triangulate, img_frac, bake_res) in [
      ("scroll_constant.obj", 0.15, "approx", True, 1, 2048),
    ]
    for cmd in [
      #run(
      #  model, model[:-4] + ".ply",
      #  f"--target-tri-ratio {ratio} --sample-kind {sample_kind} \
      #  {'--triangulate' if triangulate else ''} \
      #  --image-size-frac {img_frac}",
      #  missing_only=True,
      #),
      *[
        runnable_cmds([
          f"{sys.executable} bin/tutte_param.py -i ablations/{model[:-4]}.ply \
            -o ablations/{model[:-4]}_{label}.ply \
            --color-weight {cw} \
            {f'--color-kind {norm}' if norm != 'uniform' else '--uniform'}",
          f"{copy_mesh_to_uv} -i ablations/{model[:-4]}.ply \
            -u ablations/{model[:-4]}_{label}.ply \
            -o ablations/{model[:-4]}_{label}.ply",
          f"{bake_vert_colors_to_tex} -i ablations/{model[:-4]}_{label}.ply \
          -o ablations/{model[:-4]}_{label}.obj \
          --bake-res {bake_res} \
          --bake-texture {model[:-4]}_{label}.png",
          f"rm ablations/{model[:-4]}_{label}.ply",

          f"{sys.executable} bin/hausdorff.py -o data/{model} \
            -n ablations/{model[:-4]}_{label}.obj \
            --stats ablations/{model[:-4]}_{label}.json",
        ], output_name=f"ablations/{model[:-4]}_{label}.obj", missing_only=True)
        for (w, norm, cw, label) in [
          s
          for w in [5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.2, 0.5, 1, 2]
          for s in [
            tutte("laplacian", "add", w),
            tutte("laplacian", "concat", w),
            tutte("laplacian", "max", w),
          ]
        ]
      ],
    ]
  ] + [
    runnable_cmds([f"{sys.executable} bin/ablate_tutte_params.py"]),
  ],

  "tutte-param-rebake-ablation": [
    *[
      cmd
      for (model, bake_res) in [("scroll.obj", 1024)]
      for cmd in [
        *[
          run(
            f"../ablations/{model[:-4]}.ply",
            f"{model[:-4]}_{label}.obj",
            f"--weighting {w} --pos-color-norm {norm} --bake-texture \
              {model[:-4]}_{label}.png --iters 500000 --color-weight 0 \
              --bake-res {bake_res} {'--approx-rebake' if approx_bake else ''}",
            bin=tutte_bin, eval=False,
          )
          for (w, norm, approx_bake, label) in [
            ("laplacian", "add", True, "rebake_approx"),
            ("laplacian", "add", False, "rebake_exact"),
          ]
        ],
      ]
    ],

    *[
      render(
        f"ablations/scroll_rebake_{k}.obj",
        0, -13, 0, 0, fy=-4, rz=0, cx=-2,lx=-2, h=560,
        out=f"ablations/scroll_rebake_{k}_3d.png",
        extras="--light-z -80 --roughness 1 --shade-flat",
      ) for k in ["approx", "exact"]
    ],
  ],

  "japanese_toro": [
    run(
      "japanese_toro.obj", "japanese_toro.ply",
      "-d data/japanese_toro_textures/japanese_toro_small.png \
      --target-tri-num 1000000",
    ),
  ],

  "smoothing": [
    #run(
    #  "tiger_lily.obj", "tiger_lily.ply",
    #  "-d data/tiger_lily.jpeg --target-tri-ratio 0.2 --sample-kind direct"
    #),
    *[
      run(
        "../ablations/tiger_lily.ply",
        f"tiger_lily_{label}_{target}.ply",
        f"--weighting {weighting} --pos-color-norm {norm} --iters 2000 \
        --target-properties {target}",
        bin=smooth_bin, eval=False,
      )
      for (weighting, norm, label, target) in [
        #("uniform", "pos-only", "uniform", "pos"),
        #("mean-value", "pos-only", "mv", "pos"),
        #("mean-value", "geometric-mean", "cmv", "pos"),
        #("length", "pos-only", "len", "pos"),
        #("length", "geometric-mean", "clen", "pos"),
        ("laplacian", "pos-only", "lpl", "color"),
        ("laplacian", "geometric-mean", "clpl", "color"),
      ]
    ],
  ],

  "vase": [
    run(
      "baluster_vase.obj", "baluster_vase.ply",
      "-d data/baluster_vase_textures/diffuse.jpg --target-tri-num 400000",
    ),
  ],
  "cubify-musk-melon": [
    runnable_cmds([
      f"{sys.executable} bin/cubify.py -i outputs/musk_melon_direct.ply \
        -o ablations/cube_musk_melon.ply --cubeness 50 --lr 1e-3 --scale-luma"
    ]),
    #runnable_cmds([
    #  "uv run bin/cubify.py -i outputs/musk_melon_direct.ply \
    #    -o ablations/cube_musk_melon_color_only.ply --cubeness 0 --color-cubeness 500 --lr 1e-3"
    #]),
    #runnable_cmds([
    #  "uv run bin/cubify.py -i outputs/musk_melon_direct.ply \
    #    -o ablations/cube_musk_melon_colored.ply --cubeness 2 --color-cubeness 500 --lr 1e-3"
    #]),
  ],
  "breakfast-still-life-line-art": [
    #run(
    #  "../outputs/breakfast_still_life_approx.ply",
    #  "breakfast_still_life_line_art.ply",
    #  f"--dist-thresh 0. --color-thresh 0.2 --dir min-curvature --width 5e-4 --length 0.01 \
    #  --bend-amt 1",
    #  bin=hatching_bin, out_dir="outputs",
    #  eval=False,
    #),
    run(
      "../outputs/breakfast_still_life_approx.ply",
      "breakfast_still_life_line_art.ply",
      f"--dist-thresh 3e-3 --color-thresh 0.1 --dir edge --width 1e-3 --length 0.01 \
        --bend-amt 5",
      bin=hatching_bin, out_dir="outputs",
      eval=False,
    ),
    render(
      "outputs/breakfast_still_life_line_art.ply",
      15, -25, 0, 0, fy=-20, rz=-5,
      out="outputs/breakfast_still_life_line_art.png",
    ),
  ],
  "strawberry-line-art": [
    run(
      "../outputs/strawberry_approx.ply",
      "strawberry_line_art.ply",
      f"--dist-thresh 3e-3 --color-thresh 0.2 --dir edge --width 1e-3 --length 0.1 \
        --bend-amt 3",
      bin=hatching_bin, out_dir="outputs",
      eval=False,
    ),
    render(
      "outputs/strawberry_line_art.ply",
      6, -22, -4, 0, fy=-10,
      out="outputs/strawberry_line_art.png",
      extras="--flip-light --light-z 20",
    ),
  ],

  "nishiki-utsugi-line-art": [
    run(
      "../outputs/nishiki_utsugi_direct.ply",
      "nishiki_utsugi_line_art_max.ply",
      f"--dist-thresh 4e-3 --color-thresh 0.02 --dir max-curvature --width 1e-3 --length 0.01 \
        --bend-amt 5",
      bin=hatching_bin, out_dir="outputs",
      eval=False,
    ),
    render(
      "outputs/nishiki_utsugi_line_art_max.ply",
      10, -27, 0, 0, fy=-7,
      out="outputs/nishiki_utsugi_line_art_max.png",
    ),
    run(
      "../outputs/nishiki_utsugi_direct.ply",
      "nishiki_utsugi_line_art_edge.ply",
      f"--dist-thresh 4e-3 --color-thresh 0.02 --dir edge --width 1e-3 --length 0.01 \
        --bend-amt 5",
      bin=hatching_bin, out_dir="outputs",
      eval=False,
    ),
    render(
      "outputs/nishiki_utsugi_line_art_edge.ply",
      10, -27, 0, 0, fy=-7,
      out="outputs/nishiki_utsugi_line_art_edge.png",
    ),
  ],

  "inari-mask-vector-field": [
    run(
      "../outputs/inari_mask_approx.ply",
      "inari_mask_max_curvature_field.ply",
      f"--dist-thresh 6e-3 --color-thresh 0.0 --dir max-curvature --width 1e-3 --length 0.03",
      bin=hatching_bin, out_dir="outputs",
      eval=False,
    ),

    run(
      "../outputs/inari_mask_approx.ply",
      "inari_mask_color_grad_field.ply",
      f"--dist-thresh 1e-5 --color-thresh 0.0 --dir max-curvature --width 1e-3 --length 0.03 \
        --face-hatching",
      bin=hatching_bin, out_dir="outputs",
      eval=False,
    ),
    render(
      "data/inari_mask.obj",
      6, -28, 0, 0, fy=-8, rz=0, w=800,
      out="outputs/inari_mask_input.png",
      extras="--light-z -155 --light-x 1",
      missing_only=True,
    ),
    render(
      "outputs/inari_mask_max_curvature_field.ply",
      6, -28, 0, 0, fy=-8, rz=0, w=800,
      out="outputs/inari_mask_max_curv_field.png",
      extras="--light-z -155 --light-x 1",
    ),
    render(
      "outputs/inari_mask_color_grad_field.ply",
      6, -28, 0, 0, fy=-8, rz=0, w=800,
      out="outputs/inari_mask_color_grad_field.png",
      extras="--light-z -155 --light-x 1",
    ),

    # insets

    render(
      "data/inari_mask.obj",
      4, -16, 2, 0, fy=-8, rz=0, h=512, w=800,
      out="outputs/inari_mask_input_inset.png",
      extras="--light-z -155 --light-x 1",
      missing_only=True,
    ),
    render(
      "outputs/inari_mask_max_curvature_field.ply",
      4, -16, 2, 0, fy=-8, rz=0, h=512, w=800,
      out="outputs/inari_mask_max_curv_field_inset.png",
      extras="--light-z -155 --light-x 1",
    ),
    render(
      "outputs/inari_mask_color_grad_field.ply",
      4, -16, 2, 0, fy=-8, rz=0, h=512, w=800,
      out="outputs/inari_mask_color_grad_field_inset.png",
      extras="--light-z -155 --light-x 1",
    ),
  ],

  "stripes": [
    run(
      "plane.obj",
      "plane_black_circle.obj",
      f"--triangulate-input --triangulate --target-tri-num 20000 --sample-kind approx \
        --image-size-frac 1 --no-adaptive -d data/black_circle.png",
      #missing_only=True,
      eval=False,
    ),
    run(
      "plane.obj",
      "plane_black_square.obj",
      f"--triangulate --target-tri-ratio 1. --sample-kind approx \
        --image-size-frac 0.4 --no-adaptive -d data/black_square.jpg",
      #missing_only=True,
      eval=False,
    ),
    run(
      "plane.obj",
      "plane_hokusai.obj",
      f"--triangulate --target-tri-num 150000 --sample-kind approx \
        --image-size-frac 1 --no-adaptive -d data/hokusai.jpg",
      #missing_only=True,
      eval=False,
    ),
    run(
      "dense_sphere.obj",
      "dense_sphere_black_circle.obj",
      f"--triangulate-input --triangulate --target-tri-num 50000 --sample-kind approx \
        --image-size-frac 1 --no-adaptive -d data/black_circle.png",
      #missing_only=True,
      eval=False,
    ),
    run(
      "inari_mask_front.obj",
      "inari_mask_front_approx_manifold.obj",
      f"--triangulate-input --triangulate --target-tri-num 100000 --sample-kind approx \
        --image-size-frac 0.25 --no-adaptive",
      #missing_only=True,
      eval=False,
    ),
    run(
      "gothic_armchair.obj",
      "gothic_armchair_manifold.obj",
      f"--triangulate --target-tri-num 200000 --sample-kind approx \
        --image-size-frac 0.15 --no-adaptive", # originally 400,000 and 0.25
      #missing_only=True,
      eval=False,
    ),

    run(
      "momo-gata-teacup.obj",
      "momo-gata-teacup-manifold.obj",
      f"--triangulate-input --triangulate --target-tri-num 300000 --sample-kind direct \
        --no-adaptive",
      eval=False,
    ),

    run(
      "striped_shirt.obj",
      "striped_shirt_manifold.obj",
      f"--triangulate --target-tri-num 400000 --sample-kind direct \
        --no-adaptive",
      eval=False,
    ),
  ],

  "stripe-pattern-gen": [
    #run(
    #  "../ablations/inari_mask_front_approx_manifold.obj",
    #  "../../../oss/stripes/build/direction_field.csv",
    #  "--thresh 1e-4",
    #  bin=stripe_pattern,
    #  eval=False,
    #),

    #run(
    #  "../ablations/plane_hokusai.obj",
    #  "../../../oss/stripes/build/direction_field.csv",
    #  "--thresh 6e-3",
    #  bin=stripe_pattern,
    #  eval=False,
    #),
    #run(
    #  "../ablations/gothic_armchair_manifold.obj",
    #  "../../../oss/stripes/build/direction_field.csv",
    #  "--thresh 5e-4",
    #  bin=stripe_pattern,
    #  eval=False,
    #),
    #run(
    #  "../ablations/momo-gata-teacup-manifold.obj",
    #  "../../../oss/stripes/build/color.csv",
    #  "",
    #  bin=color_csv,
    #  eval=False,
    #),

    run(
      "../ablations/striped_shirt_manifold.obj",
      "../../../oss/stripes/build/color.csv",
      "",
      bin=color_csv,
      eval=False,
    ),
  ],

  "render-original-stripes": [
    #render(
    #  f"ablations/plane_hokusai.obj",
    #  22, 0, 0, 0.0001, fy=-100, rz=0,
    #  out=f"ablations/plane_hokusai.png",
    #  extras="--light-z -155 --light-x -30 --roughness 1 --with-vertex-colors",
    #  missing_only=True,
    #),
    render(
      f"data/gothic_armchair.obj",
      6, -26.8, 1.3, 0, fy=-1000., rz=0, h=770,#cx=-0.25, lx=-0.25,
      out=f"outputs/gothic_armchair.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),
  ],

  "quad-remesh-wagara": [
    run(
      "plane.obj",
      "plane_wagara.obj",
      f"--triangulate --target-tri-ratio 1. --sample-kind approx \
        --image-size-frac 0.6 --no-adaptive -d data/wagara.jpg",
      missing_only=True,
      eval=False,
    ),
    run(
      "../ablations/plane_wagara.obj",
      "plane_wagara_grads.csv",
      "--thresh 1e-3",
      #"--thresh 1e-2",
      missing_only=True,
      eval=False,
      bin=stripe_pattern,
    ),
    run(
      "../ablations/plane_wagara.obj",
      "plane_wagara_remesh.obj",
      "--field ablations/plane_wagara_grads.csv --scale 0.008 --subdivisions 1 \
        --save-grid ablations/plane_wagara_grid.ply --orient-iters 5 \
        --save-arrows ablations/plane_wagara_arrows.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),
    run(
      "../ablations/plane_wagara.obj",
      "plane_wagara_remesh_no_grad.obj",
      "--scale 0.008 --subdivisions 1 \
        --save-grid ablations/plane_wagara_grid_no_grad.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),
    # insets
    render(
      "ablations/plane_wagara_arrows.ply",
      5, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara_arrows_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1 --ambient-light 1.5 \
        --with-vertex-colors",
    ),
    render(
      "ablations/plane_wagara.obj",
      5, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1 --ambient-light 1.5 \
        --with-vertex-colors",
    ),
    render(
      "ablations/plane_wagara_remesh.obj",
      5, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara_remesh_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/plane_wagara_grid.ply --ambient-light 1",
    ),
    render(
      "ablations/plane_wagara_remesh_no_grad.obj",
      5, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara_remesh_no_grad_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/plane_wagara_grid_no_grad.ply --ambient-light 1",
    ),
    # ---

    render(
      "ablations/plane_wagara.obj",
      22, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara.png",
      extras="--light-z -155 --light-x -30 --roughness 1 --ambient-light 1.5 \
        --with-vertex-colors",
    ),
    render(
      "ablations/plane_wagara_remesh.obj",
      22, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara_remesh.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/plane_wagara_grid.ply --ambient-light 1",
    ),
    render(
      "ablations/plane_wagara_remesh_no_grad.obj",
      22, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/plane_wagara_remesh_no_grad.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/plane_wagara_grid_no_grad.ply --ambient-light 1",
    ),
  ],
  "quad-remesh-rot-cube-approx": [
    run(
      "../ablations/cube_rot_uv_approx.ply",
      "cube_rot_uv_approx_remesh.obj",
      "--color-field --scale 0.015 --subdivisions 1 --orient-iters 5 \
        --save-grid ablations/cube_rot_uv_approx_grid.ply \
        --save-arrows ablations/cube_rot_uv_approx_arrows.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),
  ],
  "quad-remesh-cushion": [
    run(
      "cushion.obj",
      "cushion_approx.ply",
      f"--target-tri-ratio 1 --sample-kind approx \
        --image-size-frac 0.5",
      missing_only=True,
      eval=False,
    ),
    run(
      "../ablations/cushion_approx.ply",
      "cushion_approx_remesh.obj",
      "--color-field --scale 0.02 --subdivisions 0 --orient-iters 100 \
        --save-grid ablations/cushion_approx_grid.ply \
        --save-arrows ablations/cushion_approx_arrows.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),

    render(
      "ablations/cushion_approx.ply",
      4.5, -25, 0, -0.9, fy=-5.8, rz=-15, lx=-0.8,
      out=f"ablations/cushion_approx.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
    ),
    render(
      "ablations/cushion_approx_arrows_before_smooth.ply",
      4.5, -25, 0, -0.9, fy=-5.8, rz=-15, lx=-0.8,
      out=f"ablations/cushion_approx_remesh.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
      --wireframe ablations/cushion_approx_grid.ply",
    ),
    render(
      "data/cushion.obj",
      #"ablations/cushion_approx.ply",
      4.5, -15, 0, 1, fy=-5.8, rz=-35, lx=-1.5, h=720,
      out=f"ablations/cushion_approx_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1 --ambient-light 1",
    ),
    render(
      "ablations/cushion_approx_arrows_before_smooth.ply",
      4.5, -15, 0, 1, fy=-5.8, rz=-35, lx=-1.5, h=720,
      out=f"ablations/cushion_approx_remesh_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
      --wireframe ablations/cushion_approx_grid.ply --ambient-light 1",
    ),
  ],

  "quad-remesh-komoganari": [
    run(
      "komoganari.obj",
      "komoganari_approx.ply",
      f"--target-tri-ratio 1 --sample-kind approx \
        --image-size-frac 0.5 --no-adaptive",
      missing_only=True,
      eval=False,
    ),
    run(
      "../ablations/komoganari_approx.ply",
      "komoganari_approx_remesh.obj",
      "--color-field --scale 0.01 --subdivisions 0 --orient-iters 100 \
        --save-grid ablations/komoganari_approx_grid.ply \
        --save-arrows ablations/komoganari_approx_arrows.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),

    render(
      "data/komoganari.obj",
      18.5, -22.5, 1.5, 0, fy=-5, rz=-15, lx=0.7, cx=0.7, h=800,
      out=f"outputs/komoganari_input.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),

    render(
      "ablations/komoganari_approx_remesh.obj",
      18.5, -22.5, 1.5, 0, fy=-5, rz=-15, lx=0.7, cx=0.7, h=800,
      out=f"ablations/komoganari_remesh.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/komoganari_approx_grid.ply",
    ),

    render(
      "data/komoganari.obj",
      5, -17, 4, 0, fy=-5, rz=-15, lx=1, cx=1, h=512,
      out=f"outputs/komoganari_input_zoom.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),

    render(
      "ablations/komoganari_approx_grad.ply",
      5, -17, 4, 0, fy=-5, rz=-15, lx=1, cx=1, h=512,
      out=f"ablations/komoganari_remesh_zoom.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/komoganari_approx_grid.ply",
    ),
  ],

  "quad-remesh-checkered-origami-ball": [
    run(
      "checkered_origami_ball.obj",
      "checkered_origami_ball.obj",
      f"--target-tri-ratio 1 --sample-kind direct \
        --image-size-frac 1",
      missing_only=True,
      eval=False,
    ),
    run(
      "../ablations/checkered_origami_ball.obj",
      "checkered_origami_ball_remesh.obj",
      "--color-field --scale 0.025 --subdivisions 1 --orient-iters 25 \
        --save-arrows ablations/checkered_origami_ball_arrows.ply \
        --save-grid ablations/checkered_origami_ball_grid.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),
  ],
  "quad-remesh-matcha-bowl": [
    run(
      "japanese_matcha_bowl.obj",
      "japanese_matcha_bowl.obj",
      f"--target-tri-ratio 0.5 --sample-kind approx \
        --image-size-frac 0.25",
      missing_only=True,
      eval=False,
    ),
    run(
      "../ablations/japanese_matcha_bowl.obj",
      "japanese_matcha_bowl_remesh.obj",
      "--color-field --scale 0.0125 --subdivisions 0 --orient-iters 50 \
        --save-arrows ablations/japanese_matcha_bowl_arrows.ply \
        --save-grid ablations/japanese_matcha_bowl_arrows_grid.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),
  ],

  "quad-remesh-yazd-dome": [
    run(
      "yazd_quarter_dome.obj",
      "yazd_quarter_dome.obj",
      f"--target-tri-ratio 1. --sample-kind approx \
        --image-size-frac 0.25 --no-adaptive",
      missing_only=True,
      eval=False,
    ),
    run(
      "../ablations/yazd_quarter_dome.obj",
      "yazd_quarter_dome_remesh.obj",
      "--color-field --scale 0.007 --subdivisions 1 --orient-iters 100 \
        --save-grid ablations/yazd_quarter_dome_grid.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),
    run(
      "../ablations/yazd_quarter_dome.obj",
      "yazd_quarter_dome_remesh_no_tex_grad.obj",
      "--scale 0.007 --subdivisions 1 \
        --save-grid ablations/yazd_quarter_dome_grid_no_tex_grad.ply",
      eval=False,
      bin=pars3d_quad_remesh,
    ),

    render(
      "ablations/yazd_quarter_dome.obj",
      6, -20, 0.5, 0, fy=-1000, rz=-45,
      out=f"ablations/yazd_quarter_dome.png",
      extras="--light-z -155 --light-x -30 --roughness 1 --ambient-light 1 \
        --with-vertex-colors",
      missing_only=True,
    ),
    render(
      "ablations/yazd_quarter_dome_remesh.obj",
      6, -20, 0.5, 0, fy=-1000, rz=-45,
      out=f"ablations/yazd_quarter_dome_remesh.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/yazd_quarter_dome_grid.ply --ambient-light 1",
    ),
    render(
      "ablations/yazd_quarter_dome_remesh_no_tex_grad.obj",
      22, 0, 0, -0.0001, fy=-1000, rz=180,
      out=f"ablations/yazd_dome_remesh_no_tex_grad.png",
      extras="--light-z -155 --light-x -30 --roughness 1 \
        --wireframe ablations/yazd_quarter_dome_grid_no_tex_grad.ply \
        --ambient-light 1",
    ),

    # insets
    #render(
    #  "ablations/yazd_dome.obj",
    #  5, 0, 0, -0.0001, fy=-1000, rz=180,
    #  out=f"ablations/yazd_dome_inset.png",
    #  extras="--light-z -155 --light-x -30 --roughness 1 --ambient-light 1.5 \
    #    --with-vertex-colors",
    #),
    #render(
    #  "ablations/yazd_dome_remesh.obj",
    #  5, 0, 0, -0.0001, fy=-1000, rz=180,
    #  out=f"ablations/yazd_dome_remesh_inset.png",
    #  extras="--light-z -155 --light-x -30 --roughness 1 \
    #    --wireframe ablations/yazd_dome_grid.ply --ambient-light 1",
    #),
    #render(
    #  "ablations/yazd_dome_remesh_no_grad.obj",
    #  5, 0, 0, -0.0001, fy=-1000, rz=180,
    #  out=f"ablations/yazd_dome_remesh_no_grad_inset.png",
    #  extras="--light-z -155 --light-x -30 --roughness 1 \
    #    --wireframe ablations/yazd_dome_grid_no_grad.ply --ambient-light 1",
    #),
    # ---

  ],

  "color-smoothing-takifugu": [
    *[
      run(
        "../outputs/takifugu_direct.ply",
        f"takifugu_smoothing_cw_{cw:08.03f}.ply",
        f"--weight 0.05 --color-weight {cw} --color-kind add",
        out_dir="ablations",
        bin=color_smoothing,
        missing_only=True,
      ) for cw in [0, 1, 4, 16]
    ],
    render(
      f"data/takifugu.obj",
      5, -15, -1, 0, fy=-3, rz=50,cx=-1.5,lx=-1.5,h=800,
      out=f"outputs/takifugu_input.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),
    render(
      f"ablations/takifugu_smoothing_cw_0000.000.ply",
      5, -15, -1, 0, fy=-3, rz=50,cx=-1.5,lx=-1.5,h=800,
      out=f"ablations/takifugu_smoothing_cw_0000.000.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),
    render(
      f"ablations/takifugu_smoothing_cw_0004.000.ply",
      5, -15, -1, 0, fy=-3, rz=50,cx=-1.5,lx=-1.5,h=800,
      out=f"ablations/takifugu_smoothing_cw_0004.000.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),

    runnable_cmds([
      "cp -R data/takifugu* tmp/",
      f"{gaussian_blur} -i tmp/takifugu_textures/takifugu_1.jpeg \
        -o tmp/takifugu_textures/takifugu_1.jpeg --sigma 12",
      f"{gaussian_blur} -i tmp/takifugu_textures/takifugu_2.jpeg \
        -o tmp/takifugu_textures/takifugu_2.jpeg --sigma 12"
    ]),
    render(
      f"tmp/takifugu.obj",
      5, -15, -1, 0, fy=-3, rz=50,cx=-1.5,lx=-1.5,h=800,
      out=f"ablations/takifugu_uv_smooth.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
    ),
  ],

  "color-smoothing": [
    #*[
    #  run(
    #    "../outputs/bag_with_floral_pattern_approx.ply",
    #    f"smoothing_bag_with_floral_pattern_cw_{cw:08.03f}.ply",
    #    f"--weight 1. --color-weight {cw} --color-kind add",
    #    out_dir="ablations",
    #    bin=color_smoothing,
    #    missing_only=True,
    #  ) for cw in [0, 1/64, 1/16, 1/4, 1,4,16, 64, 256, 1024]
    #],

    *[
      run(
        "../outputs/old_teapot_approx.ply",
        f"smoothing_old_teapot_cw_{cw:08.03f}.ply",
        f"--weight 0.05 --color-weight {cw} --color-kind add",
        out_dir="ablations",
        bin=color_smoothing,
        missing_only=True,
      ) for cw in [0, 1/64, 1/16, 1/4, 1,4,16, 64, 256, 1024]
    ],

    render(
      f"data/old_teapot.obj",
      11, -16, 4, 0, fy=-1, rz=-110, cx=-0.25, lx=-0.25,
      out=f"outputs/old_teapot_input.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),
    render(
      f"data/old_teapot.obj",
      6, -10, 4.5, 0, fy=-1, rz=-135, cx=-0.25, lx=-0.25, h=512,
      out=f"outputs/old_teapot_input_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),

    *[
      render(
        f"ablations/smoothing_old_teapot_cw_{cw:08.03f}.ply",
        11, -16, 4, 0, fy=-1, rz=-110, cx=-0.25, lx=-0.25,
        out=f"ablations/smoothing_old_teapot_cw_{cw:08.03f}.png",
        extras="--light-z -155 --light-x -30 --roughness 1",
        missing_only=True,
      ) for cw in [0, 1/64, 1/16, 1/4, 1,4,16, 64, 256, 1024]
    ],
    *[
      render(
        f"ablations/smoothing_old_teapot_cw_{cw:08.03f}.ply",
        6, -10, 4.5, 0, fy=-1, rz=-135, cx=-0.25, lx=-0.25, h=512,
        out=f"ablations/smoothing_old_teapot_cw_{cw:08.03f}_inset.png",
        extras="--light-z -155 --light-x -30 --roughness 1",
        missing_only=True,
      ) for cw in [0, 1/64, 1/16, 1/4, 1,4,16, 64, 256, 1024]
    ],

    runnable_cmds([
      "cp data/old_teapot* tmp/",
      f"{gaussian_blur} -i tmp/old_teapot_albedo.jpg \
        -o tmp/old_teapot_albedo.jpg --sigma 12"
    ]),
    render(
      f"tmp/old_teapot.obj",
      11, -16, 4, 0, fy=-1, rz=-110, cx=-0.25, lx=-0.25,
      out=f"ablations/smoothing_old_teapot_uv_blur.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
    ),
    render(
      f"tmp/old_teapot.obj",
      6, -10, 4.5, 0, fy=-1, rz=-135, cx=-0.25, lx=-0.25, h=512,
      out=f"ablations/smoothing_old_teapot_uv_blur_inset.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
    ),

    #*[
    #  run(
    #    "../outputs/takifugu_direct.ply",
    #    f"smoothing_takifugu_direct_cw_{cw:08.03f}.ply",
    #    f"--weight 0.1 --color-weight {cw} --color-kind add",
    #    out_dir="ablations",
    #    bin=color_smoothing,
    #    missing_only=True,
    #  ) for cw in [0, 1/64, 1/16, 1/4, 1,4,16, 64, 256, 1024]
    #],

  ],

  "lod-comparison": [
    *[
      run(
        "japanese_tea_cup.obj", f"japanese_tea_cup_res_{tri_num}.ply",
        f"-r {tri_num} --sample-kind approx",
        out_dir="outputs",
        missing_only=True,
      )
      for tri_num in [1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015, 0.008]
    ],
    render(
      f"data/japanese_tea_cup.obj",
      6, -18.5, 6, 0, fy=0, rz=0, w=660,
      out=f"outputs/japanese_tea_cup_input.png",
      extras="--light-z -155 --light-x -30",
      missing_only=True,
    ),
    *[
      render(
        f"outputs/japanese_tea_cup_res_{tri_num}.ply",
        6, -18.5, 6, 0, fy=0, rz=0, w=660,
        out=f"outputs/japanese_tea_cup_res_{tri_num}.png",
        extras="--light-z -155 --light-x -30",
        missing_only=True,
      )
      for tri_num in [1, 0.25, 0.0625, 0.015]
    ],
    *[
      render(
        f"outputs/japanese_tea_cup_res_{tri_num}.ply",
        5.5, -9, 5.5, 0, fy=0, rz=0, w=800,
        out=f"outputs/japanese_tea_cup_res_{tri_num}_inset.png --wireframe-thickness 6e-3",
        extras="--light-z -155 --light-x -30",
        missing_only=True,
      )
      for tri_num in [1, 0.25, 0.0625, 0.015]
    ],
    render(
      f"data/japanese_tea_cup.obj",
      5.5, -9, 5.5, 0, fy=0, rz=0, w=800,
      out=f"outputs/japanese_tea_cup_input_inset.png",
      extras="--light-z -155 --light-x -30 --wireframe-thickness 6e-3",
      missing_only=True,
    ),
  ],

  "image-lod-comparison": [
    *[
      run(
        "incense_burner.obj", f"incense_burner_res_{img_size_frac}.ply",
        f"-t 2000000 --sample-kind approx --image-size-frac {img_size_frac}",
        out_dir="ablations",
        missing_only=True,
      )
      for img_size_frac in [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
    ],
    render(
      f"data/incense_burner.obj",
      9.5, -18, 5.5, 0, fy=0, rz=0, w=840,
      out=f"outputs/incense_burner_input.png",
      extras="--light-z -155 --light-x -30 --roughness 1",
      missing_only=True,
    ),
    *[
      render(
        f"ablations/incense_burner_res_{img_size_frac}.ply",
        9.5, -18, 5.5, 0, fy=0, rz=0, w=840,
        out=f"ablations/incense_burner_res_{img_size_frac}.png",
        extras="--light-z -155 --light-x -30 --roughness 1",
        missing_only=True,
      ) for img_size_frac in [1, 1/2, 1/4, 1/8, 1/16]
    ],

    render(
      f"data/incense_burner.obj",
      9.3, 8, 9.3, 0, fy=0, rz=-22, h=512,
      out=f"outputs/incense_burner_input_inset.png",
      extras="--light-z -80 --light-x -30 --roughness 1 --wireframe-thickness 4e-3 \
      --ambient-light 1",
    ),
    *[
      render(
        f"ablations/incense_burner_res_{img_size_frac}.ply",
        9.3, 8, 9.3, 0, fy=0, rz=-22, h=512,
        out=f"ablations/incense_burner_res_{img_size_frac}_inset.png",
        extras="--light-z -80 --light-x -30 --roughness 1 --wireframe-thickness 4e-3 \
        --ambient-light 1",
        missing_only=True,
      ) for img_size_frac in [1, 1/2, 1/4, 1/8, 1/16]
    ],

  ],

  "adaptive-eyeball": [
    run(
      "eyeball.fbx", "eyeball_adaptive.ply",
      f"-d data/eyeball_base_color.png --sample-kind approx --target-tri-num 100000", eval=False,
    ),
    render(
      "ablations/eyeball_adaptive.ply",
      0, -35, 0, 0, fy=-11, rz=30,
      out="ablations/eyeball_adaptive_render.png",
      extras="--light-z -80 --swap-xy --shade-flat --roughness 1",
    ),
    runnable_cmds([
      "../pars3d/target/release/examples/wireframe --width 6e-4 \
        ablations/eyeball_adaptive.ply ablations/eyeball_wireframe.ply",
    ], output_name="eyeball_wireframe.ply"),

    render(
      "ablations/eyeball_adaptive.ply",
      0, -35, 0, 0, fy=-11, rz=30,
      out="ablations/eyeball_adaptive_wireframe.png",
      extras="--light-z -80 --swap-xy --shade-flat \
        --roughness 1 --wireframe ablations/eyeball_wireframe.ply",
    ),
  ],

  "officebot-dithering": [
    #run(
    #  "../outputs/officebot_approx.ply",
    #  "officebot_uniform_dithering.ply",
    #  "--weighting uniform",
    #  bin=dithering_bin, out_dir="outputs", eval=False
    #),
    run(
      "../outputs/officebot_approx.ply",
      "officebot_length_dithering.ply",
      "--weighting length",
      bin=dithering_bin, out_dir="outputs", eval=False
    ),
    render(
      "outputs/officebot_length_dithering.ply",
      11, -19, 5, 0, fy=0.5, rz=-45,
      out="outputs/officebot_length_dithering.png",
      extras="--flip-light --light-z 200",
    ),
  ],

  "violin-dithering": [
    run(
      "violin.obj",
      "violin.ply",
      "--sample-kind approx --image-size-frac 0.5 --target-tri-num 5000000 --no-adaptive",
      missing_only=True,
    ),
    run(
      "../ablations/violin.ply",
      "violin_dithering.ply",
      "--weighting laplacian --color-weight 5 --face --order index \
        --error-diffused 0.35 --diffusion lut",
      bin=dithering_bin, out_dir="ablations", eval=False
    ),
    render(
      "data/violin.obj",
      21, -3, 0, -3.5, fy=-1.5, rz=-90, cx=-3,lx=-3, w=480,
      out="ablations/violin_input.png",
      extras="--flip-light --light-z 200 --roughness 1",
    ),
    render(
      "ablations/violin_dithering.ply",
      21, -3, 0, -3.5, fy=-1.5, rz=-90, cx=-3,lx=-3, w=480,
      out="ablations/violin_dithering.png",
      extras="--flip-light --light-z 200 --roughness 1",
    ),

    runnable_cmds([
      "cargo run --release --example dither_image -- -i ablations/violin_input.png \
        -o ablations/violin_input_screen_dither.png"
    ], output_name="screen_dither"),
  ],

  "flat-dithering": [
    run(
      "plane.obj",
      "hokusai_plane.ply",
      "-d data/hokusai.jpg --target-tri-ratio 1 --sample-kind approx",
      missing_only=True,
    ),

    run(
      "plane.obj",
      "hokusai_plane_reduced.ply",
      "-d data/hokusai.jpg --target-tri-ratio 0.5 --sample-kind approx",
      missing_only=True,
    ),
  ],
  "watercolor-cake-dithering": [
    run(
      "../outputs/watercolor_cake_approx.ply",
      "watercolor_cake_dithering.ply",
      "--weighting length --color-weight 1e-2",
      bin=dithering_bin, out_dir="outputs", eval=False
    ),
    #render(
    #  "data/watercolor_cake.obj",
    #  8, 24, 5, 0, fy=0.2, cx=-2,lx=-2,rz=-90,
    #  out="outputs/watercolor_cake_input.png",
    #  extras="--light-z 80 --light-x 40",
    #),
    render(
      "outputs/watercolor_cake_dithering.ply",
      8, 24, 5, 0, fy=0.2, cx=-2,lx=-2,rz=-90,
      out="outputs/watercolor_cake_dithering.png",
      extras="--light-z 80 --light-x 40",
    ),
  ],
  "baking-scallop-dithering": [
    run(
      "../outputs/baking_scallop_direct.ply",
      "baking_scallop_dither.ply",
      "--weighting laplacian --color-weight 0 --face --order index --diffusion lut",
      bin=dithering_bin, out_dir="ablations", eval=False
    ),
  ],
  "watercolor-girl-dithering": [
    run(
      "watercolor_girl.obj",
      "watercolor_girl_approx.ply",
      "-t 3000000 --sample-kind approx --no-adaptive",
      out_dir="ablations", eval=False,
      missing_only=True,
    ),

    run(
      "../ablations/watercolor_girl_approx.ply",
      "watercolor_girl_dithering.ply",
      "--weighting laplacian --color-weight 0 --face --order random \
        --error-diffused 0.85 --diffusion lut",
      bin=dithering_bin, out_dir="outputs", eval=False
    ),

    runnable_cmds([
      "cp data/watercolor_girl.obj outputs/watercolor_girl_dithered.obj",
      "cp data/watercolor_girl.mtl outputs",
      "cargo run --release --example dither_image -- -i data/watercolor-girl-albedo.jpg \
        -o outputs/watercolor-girl-albedo.jpg"
    ], output_name="watercolor-girl-albedo.jpg"),

    # render full distance
    render(
      "data/watercolor_girl.obj",
      6.5, -13, 6.5, 0, fy=0, cx=0,lx=0,rz=0, w=720,
      out="ablations/watercolor_girl_input.png",
      extras="--light-z 80 --roughness 1 --light-strength 8",
      missing_only=True,
    ),
    render(
      "outputs/watercolor_girl_dithering.ply",
      6.5, -13, 6.5, 0, fy=0, cx=0,lx=0,rz=0, w=720,
      out="ablations/watercolor_girl_dithered.png",
      extras="--light-z 80 --roughness 1 --light-strength 8",
    ),
    render(
      "outputs/watercolor_girl_dithered.obj",
      6.5, -13, 6.5, 0, fy=0, cx=0,lx=0,rz=0, w=720,
      out="ablations/watercolor_girl_texture_dither.png",
      extras="--light-z 80 --roughness 1 --light-strength 8",
    ),
    runnable_cmds([
      "cargo run --release --example dither_image -- -i ablations/watercolor_girl_input.png \
        -o ablations/watercolor_girl_output_dither.png"
    ], output_name="watercolor_girl_output_dither.png", stage_kind="render"),
    # inset renders
    render(
      "data/watercolor_girl.obj",
      8, -4, 8, 0, fy=0, cx=0,lx=0,rz=36, w=720, h=512,
      out="ablations/watercolor_girl_input_inset.png",
      extras="--light-z 80 --roughness 1 --light-strength 8",
      missing_only=True,
    ),
    render(
      "outputs/watercolor_girl_dithering.ply",
      8, -4, 8, 0, fy=0, cx=0,lx=0,rz=36, w=720, h=512,
      out="ablations/watercolor_girl_dithered_inset.png",
      extras="--light-z 80 --roughness 1 --light-strength 8",
    ),
    render(
      "outputs/watercolor_girl_dithered.obj",
      8, -4, 8, 0, fy=0, cx=0,lx=0,rz=36, w=720, h=512,
      out="ablations/watercolor_girl_texture_dither_inset.png",
      extras="--light-z 80 --roughness 1 --light-strength 8",
    ),
    runnable_cmds([
      "cargo run --release --example dither_image -- -i ablations/watercolor_girl_input_inset.png \
        -o ablations/watercolor_girl_output_dither_inset.png"
    ], output_name="watercolor_girl_output_dither_inset.png", stage_kind="render"),
  ],

  "classic-detective-hat-dithering": [
    run(
      "classic_detective_hat.obj",
      "classic_detective_hat.ply",
      "--sample-kind approx --image-size-frac 0.5 --target-tri-num 1000000 --no-adaptive",
      missing_only=True,
    ),
    run(
      "../ablations/classic_detective_hat.ply",
      "classic_detective_hat_dithering.ply",
      "--weighting laplacian --color-weight 0 --face --order nearest \
        --error-diffused 0.9 --diffusion lut",
      bin=dithering_bin, out_dir="outputs", eval=False
    ),
    # for private detective
    #  10.1, -5.25, 10.1, 0, fy=-100., cx=0,lx=0, rz=0,
    render(
      "data/classic_detective_hat.obj",
      15, -20, -1, 0, fy=-100., cx=0,lx=0, rz=-45,h=800,
      out="outputs/classic_detective_hat_input.png",
      extras="--light-z -80 --light-x -20 --ambient-light 1 --roughness 1",
      missing_only=True,
    ),
    render(
      "outputs/classic_detective_hat_dithering.ply",
      15, -20, -1, 0, fy=-100., cx=0,lx=0, rz=-45,h=800,
      out="outputs/classic_detective_hat_dithering.png",
      extras="--light-z -80 --light-x -20 --ambient-light 1 --roughness 1",
    ),
  ],

  "edge-detection-butterfly": [
    run(
      "../outputs/tiger_butterfly_approx.ply",
      "tiger_butterfly_edges.ply",
      #"--smoothing-iters 10 --min-val 0 --max-val 0",
      "--smoothing-iters 4 --min-val 1e-4 --max-val 5e-4 \
        --cone-angle-degrees 30 --no-normalize-colors --no-area-weight",
      #"--smoothing-iters 4 --min-val 1e-5 --max-val 3e-4 --cone-angle-degrees 30",
      bin=edge_detection_bin, eval=False,
    ),

    render(
      "data/tiger_butterfly.obj",
      2, -27, 0, 0, fy=-7, rz=0, h=720,
      out="ablations/tiger_butterfly_input.png",
      extras="--roughness 0.8 --light-z -50",
      missing_only=True,
    ),

    render(
      "data/tiger_butterfly.obj",
      5, -11, 3, 0, fy=-7, rz=0, h=360, cx=5.5,lx=5.5,
      out="ablations/tiger_butterfly_input_inset.png",
      extras="--roughness 0.8 --light-z -50",
      missing_only=True,
    ),

    render(
      "ablations/tiger_butterfly_edges.ply",
      2, -27, 0, 0, fy=-7, rz=0, h=720,
      out="ablations/tiger_butterfly_edges.png",
      extras="--roughness 0.8 --light-z -50",
    ),

    render(
      "ablations/tiger_butterfly_edges.ply",
      5, -11, 3, 0, fy=-7, rz=0, h=360, cx=5.5,lx=5.5,
      out="ablations/tiger_butterfly_edges_inset.png",
      extras="--roughness 0.8 --light-z -50",
      missing_only=True,
    ),

    runnable_cmds([
      "cp data/tiger_butterfly* tmp/",
      f"{sys.executable} bin/canny_edge.py -i tmp/tiger_butterfly_diffuse.jpg \
        -o tmp/tiger_butterfly_diffuse.jpg --min 50 --max 130"
    ]),

    render(
      "tmp/tiger_butterfly.obj",
      2, -27, 0, 0, fy=-7, rz=0, h=720,
      out="ablations/tiger_butterfly_uv_space.png",
      extras="--roughness 0.8 --light-z -50",
    ),

    render(
      "tmp/tiger_butterfly.obj",
      5, -11, 3, 0, fy=-7, rz=0, h=360, cx=5.5,lx=5.5,
      out="ablations/tiger_butterfly_uv_space_inset.png",
      extras="--roughness 0.8 --light-z -50",
      missing_only=True,
    ),

    runnable_cmds([
      f"{sys.executable} bin/canny_edge.py -i ablations/tiger_butterfly_input.png \
        -o ablations/tiger_butterfly_rendered_edges.png --min 80 --max 180"
    ]),

    runnable_cmds([
      f"{sys.executable} bin/canny_edge.py -i ablations/tiger_butterfly_input_inset.png \
        -o ablations/tiger_butterfly_rendered_edges_inset.png --min 80 --max 180"
    ],output_name="rendered_edges_inset.png", stage_kind="render"),
  ],

  "edge-detection-eye-of-ra": [
    run(
      "../outputs/eye_of_ra_approx.ply",
      "eye_of_ra_approx_edges.ply",
      "--smoothing-iters 1 --min-val 80 --max-val 110 \
        --cone-angle-degrees 30 --no-normalize-colors --cull-area-below 2e-7 --no-area-weight",
      bin=edge_detection_bin, eval=False,
    ),

    render(
      "data/eye_of_ra.obj",
      28, -0.9, 0, -0.901, fy=-1, rz=180, cx=-0.5, lx=-0.5, h=840,
      out="ablations/eye_of_ra_input.png",
      extras="--roughness 1 --light-z -50 --wireframe-thickness 5e-3",
      missing_only=True,
    ),
    render(
      "data/eye_of_ra.obj",
      28, -0.9, 0, -0.901, fy=-1, rz=180, cx=-0.5, lx=-0.5, h=840,
      out="ablations/eye_of_ra_no_wireframe.png",
      extras="--roughness 1 --light-z -50",
      missing_only=True,
    ),

    render(
      "ablations/eye_of_ra_approx_edges.ply",
      28, -0.9, 0, -0.901, fy=-1, rz=180, cx=-0.5, lx=-0.5, h=840,
      out="ablations/eye_of_ra_approx_edges.png",
      extras="--roughness 1 --light-z -50",
      missing_only=True,
    ),

    runnable_cmds([
      "cp -r data/eye_of_ra* tmp/",
      f"{sys.executable} bin/canny_edge.py -i tmp/eye_of_ra_textures/EyeOfRah_Diffuse_02.png \
        -o tmp/eye_of_ra_textures/EyeOfRah_Diffuse_02.png --min 45 --max 70"
    ], output_name="eye_of_ra_uv_space.png"),

    render(
      "tmp/eye_of_ra.obj",
      28, -0.9, 0, -0.901, fy=-1, rz=180, cx=-0.5, lx=-0.5, h=840,
      out="ablations/eye_of_ra_uv_space.png",
      extras="--roughness 1 --light-z -50",
    ),

    runnable_cmds([
      f"{sys.executable} bin/canny_edge.py -i ablations/eye_of_ra_no_wireframe.png \
        -o ablations/eye_of_ra_rendered_edges.png --min 80 --max 180"
    ], output_name="eye_of_ra_rendered_edges.png"),
  ],

  # ---

  "edge-detection-bag-with-floral-pattern": [
    run(
      "../outputs/bag_with_floral_pattern_approx.ply",
      "bag_with_floral_pattern_edges.ply",
      "--smoothing-iters 0 --min-val 1e-3 --max-val 1.2e-3 \
        --cone-angle-degrees 45 --no-normalize-colors \
        --cull-area-below 2e-7",
      bin=edge_detection_bin, eval=False,
    ),

    render(
      "data/bag_with_floral_pattern.obj",
      9, -17.5, 5.5, 0, fy=0, rz=70, w=800,lx=0.5,
      out="ablations/bag_with_floral_pattern_input.png",
      extras="--roughness 1 --light-z -50 --wireframe-thickness 1e-2",
      missing_only=True,
    ),
    render(
      "data/bag_with_floral_pattern.obj",
      9, -17.5, 5.5, 0, fy=0, rz=70, w=800,lx=0.5,
      out="ablations/bag_with_floral_pattern_input_no_wireframe.png",
      extras="--roughness 1 --light-z -50",
      missing_only=True,
    ),

    render(
      "ablations/bag_with_floral_pattern_edges.ply",
      9, -17.5, 5.5, 0, fy=0, rz=70, w=800,lx=0.5,
      out="ablations/bag_with_floral_pattern_edges.png",
      extras="--roughness 1 --light-z -50",
    ),

    runnable_cmds([
      "cp data/bag_with_floral_pattern* tmp/",
      f"{sys.executable} bin/canny_edge.py -i tmp/bag_with_floral_pattern_diffuse.png \
        -o tmp/bag_with_floral_pattern_diffuse.png --min 30 --max 50"
    ], output_name="bag_with_floral_pattern_uv_space.png"),

    render(
      "tmp/bag_with_floral_pattern.obj",
      9, -17.5, 5.5, 0, fy=0, rz=70, w=800,lx=0.5,
      out="ablations/bag_with_floral_pattern_uv_space.png",
      extras="--roughness 1 --light-z -50",
    ),

    runnable_cmds([
      f"{sys.executable} bin/canny_edge.py -i ablations/bag_with_floral_pattern_input_no_wireframe.png \
        -o ablations/bag_with_floral_pattern_rendered_edges.png --min 80 --max 180"
    ], output_name="bag_with_floral_pattern_rendered_edges.png"),
  ],

  "compare-subdiv": [
    run(
      "officebot.obj",
      "officebot_subdiv.ply",
      "-s 5 -d data/officebot_textures/diffuse.png",
      bin=bake_tex_to_vert_colors_bin,
    ),

    # KEEP THESE COMMENTED OUT, TOO DENSE TO RENDER
    #runnable_cmds([
    #  f"../pars3d/target/release/examples/wireframe \
    #    data/officebot.obj ablations/officebot_input_wireframe.ply --width 3e-3"
    #]),
    #runnable_cmds([
    #  f"../pars3d/target/release/examples/wireframe \
    #    ablations/officebot_subdiv.ply \
    #    ablations/officebot_subdiv_wireframe.ply --width 3e-3"
    #]),

    #runnable_cmds([
    #  f"../pars3d/target/release/examples/wireframe \
    #    outputs/officebot_approx.ply \
    #    ablations/officebot_approx_wireframe.ply --width 3e-3"
    #]),
    #runnable_cmds([
    #  f"../pars3d/target/release/examples/wireframe \
    #    outputs/officebot_exact.ply \
    #    ablations/officebot_exact_wireframe.ply --width 3e-3"
    #]),

    render(
      "data/officebot.obj",
      5.8, -18, 5.8, 0, fy=0.55, rz=-45, w=840,
      out="ablations/officebot_input.png",
      extras="--roughness 0.8 --light-z -50 --wireframe-thickness 1e-2",
      missing_only=True,
    ),
    render(
      "ablations/officebot_subdiv.ply",
      5.8, -18, 5.8, 0, fy=0.55, rz=-45, w=840,
      out="ablations/officebot_subdiv.png",
      extras="--roughness 0.8 --light-z -50 --wireframe-thickness 1e-2",
    ),
    render(
      "outputs/officebot_approx.ply",
      5.8, -18, 5.8, 0, fy=0.55, rz=-45, w=840,
      out="ablations/officebot_approx.png",
      extras="--roughness 0.8 --light-z -50 --wireframe-thickness 1e-2"
    ),
    render(
      "outputs/officebot_exact.ply",
      5.8, -18, 5.8, 0, fy=0.55, rz=-45, w=840,
      out="ablations/officebot_exact.png",
      extras="--roughness 0.8 --light-z -50 --wireframe-thickness 1e-2"
    ),

    #render(
    #  "data/officebot.obj",
    #  8, -8, 8, 0, fy=0.55, rz=-90, w=840, lx=0,cx=0, h=512,
    #  out="ablations/officebot_input_inset.png",
    #  extras="--roughness 0.8 --light-z -50",
    #  missing_only=True,
    #),
    #render(
    #  "ablations/officebot_subdiv.ply",
    #  8, -8, 8, 0, fy=0.55, rz=-90, w=840, lx=0,cx=0, h=512,
    #  out="ablations/officebot_subdiv_inset.png",
    #  extras="--roughness 0.8 --light-z -50",
    #),
    #render(
    #  "outputs/officebot_approx.ply",
    #  8, -8, 8, 0, fy=0.55, rz=-90, w=840, lx=0,cx=0, h=512,
    #  out="ablations/officebot_approx_inset.png",
    #  extras="--roughness 0.8 --light-z -50",
    #),
    render(
      "outputs/officebot_exact.ply",
      8, -8, 8, 0, fy=0.55, rz=-90, w=840, lx=0,cx=0, h=512,
      out="ablations/officebot_exact_inset.png",
      extras="--roughness 0.8 --light-z -50",
    ),

  ],

  "vietnam-lantern-comparison": [
    eval(
      "vietnam_lantern.obj",
      "vietnam_lantern_instant_mesh_240k.obj",
      out_dir="data",
      missing_only=True,
    ),
    eval(
      "vietnam_lantern.obj",
      "vietnam_lantern_instant_mesh_2m.obj",
      out_dir="data",
      missing_only=True,
    ),

    render(
      "data/vietnam_lantern.obj",
      -1, -16, -1, 0, fy=-11, rz=0, w=750,
      out="ablations/vietnam_lantern_input.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),

    render(
      "data/vietnam_lantern_instant_mesh_2m.obj",
      -1, -16, -1, 0, fy=-11, rz=0, w=750,
      out="ablations/vietnam_lantern_instant_mesh_2m.png",
      extras="--roughness 1 --light-z -50 --light-x 20 --with-vertex-colors",
      missing_only=True,
    ),

    render(
      "data/vietnam_lantern_instant_mesh_240k.obj",
      -1, -16, -1, 0, fy=-11, rz=0, w=750,
      out="ablations/vietnam_lantern_instant_mesh_550k.png",
      extras="--roughness 1 --light-z -50 --light-x 20 --with-vertex-colors",
      missing_only=True,
    ),
    # inset
    render(
      "data/vietnam_lantern.obj",
      -0.5, -6, -0.5, 0, fy=-11, rz=0, w=750, h=512,
      out="ablations/vietnam_lantern_input_zoom.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),

    render(
      "data/vietnam_lantern_instant_mesh_2m.obj",
      -0.5, -6, -0.5, 0, fy=-11, rz=0, w=750, h=512,
      out="ablations/vietnam_lantern_instant_mesh_2m_zoom.png",
      extras="--roughness 1 --light-z -50 --light-x 20 --with-vertex-colors",
      missing_only=True,
    ),

    render(
      "data/vietnam_lantern_instant_mesh_240k.obj",
      -0.5, -6, -0.5, 0, fy=-11, rz=0, w=750, h=512,
      out="ablations/vietnam_lantern_instant_mesh_550k_zoom.png",
      extras="--roughness 1 --light-z -50 --light-x 20 --with-vertex-colors",
      missing_only=True,
    ),
    # comparison to instant meshes
    run(
      "vietnam_lantern.obj",
      f"vietnam_lantern_2m.ply",
      f"-t 2068000 --sample-kind approx --image-size-frac 0.75",
      #missing_only=True,
    ),
    run(
      "vietnam_lantern.obj",
      f"vietnam_lantern_550k.ply",
      f"-t 550000 --sample-kind approx --image-size-frac 0.75",
      #missing_only=True,
    ),

    render(
      "ablations/vietnam_lantern_2m.ply",
      -1, -16, -1, 0, fy=-11, rz=0, w=750,
      out="ablations/vietnam_lantern_2m.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),

    render(
      "ablations/vietnam_lantern_550k.ply",
      -1, -16, -1, 0, fy=-11, rz=0, w=750,
      out="ablations/vietnam_lantern_550k.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),

    # inset
    render(
      "ablations/vietnam_lantern_2m.ply",
      -0.5, -6, -0.5, 0, fy=-11, rz=0, w=750, h=512,
      out="ablations/vietnam_lantern_2m_zoom.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),

    render(
      "ablations/vietnam_lantern_550k.ply",
      -0.5, -6, -0.5, 0, fy=-11, rz=0, w=750, h=512,
      out="ablations/vietnam_lantern_550k_zoom.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),
  ],

  "uvatlas-clustering": [
    *[
      cmd
      for f in os.listdir("data")
      if ".obj" in f and all(v not in f for v in
        ["basic", "cube", "plane", "sphere", "takifugu", "meadowsweet", "nishiki", "mango",
        "chozuya", "watercolor_cake", "angelfish", "musk_melon", "officebot", "breakfast",
        "oshima", "maple_leaves", "tiger", "ibis", "scan_vase", "newt", "flowers_in_vase",
        "watermelon", "thin_tri", "non_manifold", "open_top_box", "boundary", "scroll_constant"])
      for cmd in [
        runnable_cmds([
          f"{sys.executable} bin/run_uvatlas.py -i data/{f} -o {cl_dir}/{f[:-4]}_uvatlas.obj",
        ], missing_only=True, output_name=f"{cl_dir}/{f[:-4]}_uvatlas.obj"),
        run(
          f"../{cl_dir}/{f[:-4]}_uvatlas.obj", f"{f[:-4]}_uvatlas.obj",
          flags="",
          out_dir=cl_dir,
          bin=measure_flat, eval=False,
        )
      ]
    ],
  ],

  "our-clustering-match-uvatlas": [
    *[
      cmd
      for f in os.listdir("data")
      if ".obj" in f and all(v not in f for v in ["basic", "cube", "plane", "sphere"])
      for cmd in [
        run(
          f,
          f"{f[:-4]}_match_uvatlas_{label}.ply",
          f"--match-json {cl_dir}/{f[:-4]}_uvatlas.json --eigenvalue {egv} --geometry-only \
            --eigen-eps 1e-8 --color-eps 100000 --no-wireframe \
            --shape-metric max-manhattan-dist --no-delta-cost \
            --cluster-vis {cl_dir}/{f[:-4]}_match_uvatlas_{label}.ply",
          out_dir=cl_dir,
          bin=clustering_bin, eval=False,
          missing_only=True,
        ) for (label, egv) in [("planar", "one"), ("dev", "zero")]
      ]
    ],
  ],

  "xatlas-clustering": [
    *[
      cmd
      for f in os.listdir("data")
      if ".obj" in f and all(
        v not in f for v
        in ["basic", "cube", "plane", "sphere", "thin_tri", "non_manifold", "open_top_box",
          "boundary", "scroll_constant"]
      )
      for cmd in [
        runnable_cmds([
          f"{sys.executable} bin/run_xatlas.py -i data/{f} -o {cl_dir}/{f[:-4]}_xatlas.obj",
        ], missing_only=True, output_name=f"{cl_dir}/{f[:-4]}_xatlas.obj"),
        run(
          f"../{cl_dir}/{f[:-4]}_xatlas.obj", f"{f[:-4]}_xatlas.obj",
          flags="",
          out_dir=cl_dir,
          bin=measure_flat, eval=False,
        )
      ]
    ],
  ],

  "our-clustering-match-xatlas": [
    *[
      cmd
      for f in os.listdir("data")
      if ".obj" in f and all(v not in f for v in ["basic", "cube", "plane", "sphere"])
      for cmd in [
        run(
          f,
          f"{f[:-4]}_match_xatlas_{label}.ply",
          f"--match-json {cl_dir}/{f[:-4]}_xatlas.json --eigenvalue {egv} --geometry-only \
          --eigen-eps 1e-8 --color-eps 100000 --no-wireframe \
          --shape-metric max-manhattan-dist --no-delta-cost \
          --cluster-vis {cl_dir}/{f[:-4]}_match_xatlas_{label}.ply",
          bin=clustering_bin, eval=False, out_dir=cl_dir,
          missing_only=True,
        ) for (label, egv) in [("planar", "one"), ("dev", "zero")]
      ]
    ],
  ],

  "render-ours-xatlas": [
    render(
      "data/baluster_vase.obj",
      2, -34, 0, 0, fy=-11.5, rz=0, w=450,
      out="ablations/baluster_vase_input.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
      missing_only=True,
    ),
    render(
      "cluster_outputs/baluster_vase_xatlas.obj",
      2, -34, 0, 0, fy=-11.5, rz=0, w=450,
      out="ablations/baluster_vase_xatlas.png",
      extras="--roughness 1 --light-z -50 --light-x 20 --with-vertex-colors",
    ),
    render(
      "cluster_outputs/baluster_vase_match_xatlas_planar.ply",
      2, -34, 0, 0, fy=-11.5, rz=0, w=450,
      out="ablations/baluster_vase_match_xatlas_planar.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
    ),
    render(
      "cluster_outputs/baluster_vase_match_xatlas_dev.ply",
      2, -34, 0, 0, fy=-11.5, rz=0, w=450,
      out="ablations/baluster_vase_match_xatlas_dev.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
    ),

    run(
      "baluster_vase.obj",
      "baluster_vase_cmp_planar.ply",
      f"-t 32 --eigenvalue one --cluster-vis ablations/baluster_vase_cmp_planar.ply \
      --eigen-eps 1e-10 --geometry-only --shape-metric max-manhattan-dist",
      bin=clustering_bin, eval=False,
    ),
    render(
      "ablations/baluster_vase_cmp_planar.ply",
      2, -34, 0, 0, fy=-11.5, rz=0, w=450,
      out="ablations/baluster_vase_cmp_planar.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
    ),

    run(
      "baluster_vase.obj",
      "baluster_vase_cmp_dev.ply",
      f"-t 32 --eigenvalue zero --cluster-vis ablations/baluster_vase_cmp_dev.ply \
      --eigen-eps 1e-10 --geometry-only --shape-metric boundary-length",
      bin=clustering_bin, eval=False,
    ),
    render(
      "ablations/baluster_vase_cmp_dev.ply",
      2, -34, 0, 0, fy=-11.5, rz=0, w=450,
      out="ablations/baluster_vase_cmp_dev.png",
      extras="--roughness 1 --light-z -50 --light-x 20",
    ),
  ],

  "render-ours-uvatlas": [
    render(
      "data/vase.obj",
      6, -17, 6, 0, fy=0, rz=0, w=600,
      out="ablations/vase_input.png",
      extras="--roughness 1 --light-z -50",
      missing_only=True,
    ),
    render(
      "cluster_outputs/vase_uvatlas.obj",
      6, -17, 6, 0, fy=0, rz=0, w=600,
      out="ablations/vase_uvatlas.png",
      extras="--roughness 1 --light-z -50 --with-vertex-colors",
    ),
    render(
      "cluster_outputs/vase_match_uvatlas_planar.ply",
      6, -17, 6, 0, fy=0, rz=0, w=600,
      out="ablations/vase_match_uvatlas_planar.png",
      extras="--roughness 1 --light-z -50",
    ),
    render(
      "cluster_outputs/vase_match_uvatlas_dev.ply",
      6, -17, 6, 0, fy=0, rz=0, w=600,
      out="ablations/vase_match_uvatlas_dev.png",
      extras="--roughness 1 --light-z -50",
    ),

    #run(
    #  "../outputs/vase_approx.ply",
    #  "vase_color.ply",
    #  f"-t 150 --eigenvalue one --cluster-vis ablations/vase_clusters.ply \
    #    --eigen-eps 1e-3 --shape-metric max-euclidean-dist --color-eps 0",
    #  bin=clustering_bin, eval=False,
    #),
    #render(
    #  "ablations/vase_clusters.ply",
    #  6, -17, 6, 0, fy=0, rz=0, w=600,
    #  out="ablations/vase_clusters.png",
    #  extras="--roughness 1 --light-z -50",
    #),
    #render(
    #  "ablations/vase_color.ply",
    #  6, -17, 6, 0, fy=0, rz=0, w=600,
    #  out="ablations/vase_color.png",
    #  extras="--roughness 1 --light-z -50",
    #),

    #run(
    #  "vase.obj",
    #  "vase_cmp_planar.ply",
    #  f"-t 64 --eigenvalue one --cluster-vis ablations/vase_cmp_planar.ply \
    #  --eigen-eps 1e-11 --geometry-only --shape-metric boundary-length",
    #  bin=clustering_bin, eval=False,
    #),
    #run(
    #  "vase.obj",
    #  "vase_cmp_dev.ply",
    #  f"-t 64 --eigenvalue zero --cluster-vis ablations/vase_cmp_dev.ply \
    #  --eigen-eps 1e-11 --geometry-only --shape-metric boundary-length",
    #  bin=clustering_bin, eval=False,
    #),
    #render(
    #  "ablations/vase_cmp_planar.ply",
    #  6, -17, 6, 0, fy=0, rz=0, w=600,
    #  out="ablations/vase_cmp_planar.png",
    #  extras="--roughness 1 --light-z -50",
    #),
    #render(
    #  "ablations/vase_cmp_dev.ply",
    #  6, -17, 6, 0, fy=0, rz=0, w=600,
    #  out="ablations/vase_cmp_dev.png",
    #  extras="--roughness 1 --light-z -50",
    #),
  ],

  "geom-comparison-plots": [
    runnable_cmds([
      f"{sys.executable} bin/compare_measure.py --comparison uvatlas --our-kind dev",
      f"{sys.executable} bin/compare_measure.py --comparison uvatlas --our-kind planar",
      f"{sys.executable} bin/compare_measure.py --comparison xatlas --our-kind dev",
      f"{sys.executable} bin/compare_measure.py --comparison xatlas --our-kind planar",
    ]),
  ],

  "compare-remeshing": [
    *[
      run("traffic_light_with_stickers.obj", f"traffic_light_with_stickers_{k}.ply",
        f"-t 2000000 --sample-kind {k} --image-size-frac 0.5 --triangulate-input")
      for k in ["exact", "approx", "direct"]
    ],

    render(
      "data/traffic_light_with_stickers.obj",
      6, -17, 6, 0, fy=0, rz=-20,w=400,
      out="ablations/traffic_light_with_stickers_input.png",
      extras="--roughness 1 --light-z -50 --light-x 30 --texture-res 2048",
      missing_only=True,
    ),
    render(
      "data/traffic_light_with_stickers.obj",
      10, -3, 10, 0, fy=0, rz=-60,w=400,
      extras="--roughness 1 --light-z -50 --light-x 30",
      out="ablations/traffic_light_with_stickers_input_inset.png --texture-res 2048 \
       --wireframe-thickness 2e-3",
      missing_only=True,
    ),

    *[
      render(
        f"ablations/traffic_light_with_stickers_{k}.ply",
        6, -17, 6, 0, fy=0, rz=-20,w=400,
        out=f"ablations/traffic_light_with_stickers_{k}.png",
        extras="--roughness 1 --light-z -50 --light-x 30",
      )
      for k in ["exact", "approx", "direct"]
    ],
    *[
      render(
        f"ablations/traffic_light_with_stickers_{k}.ply",
        10, -3, 10, 0, fy=0, rz=-60,w=400,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"ablations/traffic_light_with_stickers_{k}_inset.png",
      )
      for k in ["exact", "approx", "direct"]
    ],
  ],

  "render-direct": [
      render(
        f"outputs/baking_scallop_direct.ply",
        0, -27.5, 0, 0, fy=-10,rz=0,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/baking_scallop_direct.png",
        missing_only=True,
      ),
      render(
        f"data/baking_scallop.obj",
        0, -27.5, 0, 0, fy=-10,rz=0,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/baking_scallop_input.png",
        missing_only=True,
      ),

      render(
        f"outputs/chinese_sacred_lily_direct.ply",
        -1.5, -20, -1.5, 0, fy=-9,rz=-45,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/chinese_sacred_lily_direct.png",
        missing_only=True,
      ),
      render(
        f"data/chinese_sacred_lily.obj",
        -1.5, -20, -1.5, 0, fy=-9,rz=-45,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/chinese_sacred_lily_input.png",
        missing_only=True,
      ),

      render(
        f"outputs/musk_melon_direct.ply",
        0.8, -26, 0.8, 0, fy=-8,rz=0,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/musk_melon_direct.png",
        missing_only=True,
      ),
      render(
        f"data/musk_melon.obj",
        0.8, -26, 0.8, 0, fy=-8,rz=0,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/musk_melon_input.png",
        missing_only=True,
      ),

      render(
        f"outputs/nishiki_utsugi_direct.ply",
        0.2, -24, 0.2, 0, fy=-8,rz=0,cx=-1.5,lx=-1.5,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/nishiki_utsugi_direct.png",
        missing_only=True,
      ),
      render(
        f"data/nishiki_utsugi.obj",
        0.2, -24, 0.2, 0, fy=-8,rz=0,cx=-1.5,lx=-1.5,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/nishiki_utsugi_input.png",
        missing_only=True,
      ),

      #("takifugu.obj", "", 250000),
      #("oshima_cherry.obj", "", 2000000),
      #("mango.obj", "", 1000000),
  ],

  "ablate-degen-bridges": [
      run(
        "tiny_koi_teapot.obj",
        f"tiny_koi_teapot.ply",
        f"--target-tri-ratio 1 --sample-kind exact --image-size-frac 0.5"
      ),
      run(
        "tiny_koi_teapot.obj",
        f"tiny_koi_teapot_no_inc_delete.ply",
        f"--target-tri-ratio 1 --sample-kind exact --image-size-frac 0.5 \
          --no-incremental-delete"
      ),

      render(
          f"data/tiny_koi_teapot.obj",
          7, -25, -2, 0, fy=-8,rz=15,
          extras="--roughness 1 --light-z -50 --light-x 30",
          out=f"outputs/tiny_koi_teapot_input.png",
          missing_only=True,
      ),

      render(
          f"ablations/tiny_koi_teapot.ply",
          7, -25, -2, 0, fy=-8,rz=15,
          extras="--roughness 1 --light-z -50 --light-x 30",
          out=f"ablations/tiny_koi_teapot.png",
          missing_only=True,
      ),

      render(
          f"ablations/tiny_koi_teapot_no_inc_delete.ply",
          7, -25, -2, 0, fy=-8,rz=15,
          extras="--roughness 1 --light-z -50 --light-x 30",
          out=f"ablations/tiny_koi_teapot_no_inc_delete.png",
          missing_only=True,
      ),
  ],

  "render-approx": [
    render(
        f"data/teacup.obj",
        13, -21, 3, 0, fy=-1,rz=45,cx=1.5,lx=1.5,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/teacup_input.png",
        missing_only=True,
    ),
    render(
        f"outputs/teacup_approx.ply",
        13, -21, 3, 0, fy=-1,rz=45,cx=1.5,lx=1.5,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/teacup_approx.png",
        missing_only=True,
    ),

    render(
        f"data/umbrella_gold.obj",
        2, -25, -1.5, 0, fy=-8,rz=-75,cx=1.5,lx=1.5,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/umbrella_gold_input.png",
        #missing_only=True,
    ),
    render(
        f"outputs/umbrella_gold_approx.ply",
        2, -25, -1.5, 0, fy=-8,rz=-75,cx=1.5,lx=1.5,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/umbrella_gold_approx.png",
        #missing_only=True,
    ),
  ],

  "render-exact": [
    render(
        f"data/juice_box.obj",
        8, -16, 2, 0, fy=-1,rz=225,cx=-3,lx=-3,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/juice_box_input.png",
        missing_only=True,
    ),
    render(
        f"outputs/juice_box_exact.ply",
        8, -16, 2, 0, fy=-1,rz=225,cx=-3,lx=-3,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/juice_box_exact.png",
        missing_only=True,
    ),
    render(
        f"data/wet_floor_sign.obj",
        8, -28.5, -1.5, 0, fy=-9.5,rz=45,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/wet_floor_sign_input.png",
        missing_only=True,
    ),
    render(
        f"outputs/wet_floor_sign_exact.ply",
        8, -28.5, -1.5, 0, fy=-9.5,rz=45,
        extras="--roughness 1 --light-z -50 --light-x 30",
        out=f"outputs/wet_floor_sign_exact.png",
        missing_only=True,
    ),
  ],

  "dataset-exact": [
    *[
      run(
        model, model[:-4] + "_exact.ply",
        f"{f'-d data/{texture}' if len(texture) else ''} -t {tri_num} \
          --sample-kind exact \
          {'' if img_size_frac is None else f'--image-size-frac {img_size_frac}'}",
        out_dir="outputs",
      )
      for (model, texture, tri_num, img_size_frac) in dataset
    ],
  ],
  "dataset-approx": [
    *[
      run(
        model, model[:-4] + "_approx.ply",
        f"{f'-d data/{texture}' if len(texture) else ''} -t {tri_num} \
          --sample-kind approx \
          {'' if img_size_frac is None else (f'--image-size-frac {img_size_frac}' if type(img_size_frac) == float else f'--image-size-px {img_size_frac}')}",
        out_dir="outputs",
      )
      for (model, texture, tri_num, img_size_frac) in dataset
    ],
  ],
  "dataset-direct": [
    *[
      run(
        model, model[:-4] + "_direct.ply",
        f"{f'-d data/{texture}' if len(texture) else ''} -t {tri_num} --sample-kind direct",
        out_dir="outputs",
      )
      for (model, texture, tri_num) in dataset_direct
    ],
  ],
}


def arguments():
  a = argparse.ArgumentParser()
  a.add_argument(
    "-e", "--experiments",
    default=list(experiments.keys()),
    nargs="*",
    choices=list(experiments.keys()),
  )
  a.add_argument(
    "--stages", default=["run", "render", "plot"], nargs="+", choices=["run", "render", "plot"],
    help="What steps of testing to run"
  )
  a.add_argument(
    "--debug-render", action="store_true", help="Faster debug render instead of final version"
  )
  a.add_argument("--missing-only", action="store_true", help="Run complete dataset for only missing files")
  a.add_argument("--first-only", action="store_true", help="Run one command then exit")
  a.add_argument("--skip-to", default=None, choices=list(experiments.keys()), help="skip to this experiment")
  a.add_argument("--match-output", default=None, help="Only match render outputs with this")
  a.add_argument(
    "--no-eval", action="store_true",
    help="Do not evaluate similarity of input and output mesh",
  )
  a.add_argument(
    "--eval-only", action="store_true",
    help="Only run evaluation for output",
  )
  a.add_argument(
    "--force", action="store_true",
    help="Force run all meshes, even if missing_only = True was specified",
  )
  a.add_argument("--no-build", action="store_true", help="Do not build (CAUTION: Use wisely)")
  return a.parse_args()

args = arguments()
now = time.asctime(time.localtime())

experiment_timestamps = {}

exp_file = "experiment_log.json"

if len(args.experiments) > 0 and not args.no_build:
  assert(not os.system("cargo build --release --bins --examples"))

if args.no_build:
  print("[CAUTION]: `--no-build` FLAG SHOULD BE USED WITH CAUTION, MAY NOT REFLECT CURRENT RESULTS")

for exp in args.experiments:
  if args.skip_to is not None:
    if args.skip_to == exp: args.skip_to = None
    else: continue
  print("-================================================-")
  print(f"\tStarting {exp}")
  print("-================================================-")

  if os.path.exists(exp_file):
    with open(exp_file, "r") as f:
      experiment_timestamps = json.load(f)

  for cmd_list in experiments[exp]:
    for cmd in cmd_list():
      assert(not os.system(cmd)), cmd
      if args.first_only: exit()

  print("-================================================-")
  print(f"\tFinished {exp}!")
  print("-================================================-")
  experiment_timestamps[exp] = {
    "time": now,
    "os": os.name,
  }

  # write after finishing each experiment so that even if stopped halfway then it will stop.
  with open(exp_file, "w") as f:
    json.dump(experiment_timestamps, f, indent=2)

