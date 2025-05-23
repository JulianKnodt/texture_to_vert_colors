import os
import argparse
import sys
import time
import json
from itertools import chain

bin_file = "target/release/texture_to_vert_colors"
tutte_bin = "target/release/tutte_param"
smooth_bin = "target/release/smoothing"
clustering_bin = "target/release/clustering"

args = None

out_dir = lambda is_ablation=False: "ablations" if is_ablation else "outputs"

def run(src, dst, flags, is_abl=True, src_dir="data", bin=bin_file, eval=True, missing_only=False):
  def cb():
    if "run" not in args.stages: return []
    if args.match_output is not None and args.match_output not in dst: return []
    out_json = f"{out_dir(is_abl)}/{dst[:-4]}.json"
    if missing_only and os.path.exists(out_json): return []
    cmds = [
      f"{bin} -i {src_dir}/{src} -o {out_dir(is_abl)}/{dst} {flags} --stats {out_json}",
    ]
    print(cmds[0])
    if (not args.no_eval) and eval:
      if ".fbx" in src:
        cmds.append('echo "FBX is not currently supported for Hausdorff"')
      else:
        cmds.append(
          f"{sys.executable} bin/hausdorff.py -o data/{src} -n {out_dir(is_abl)}/{dst} --stats {out_json}"
        )

    return cmds
  return cb

def render(
  i, cy,cz, ly,lz,out, w=1024, h=1024, cx=0, fy=0, lx=0, rz=45,
  extras=""
):
  out = os.path.join(os.getcwd(), out)
  def cb():
    if "render" not in args.stages: return []
    if args.match_output is not None and args.match_output not in out: return []

    cmd = f"{sys.executable} bin/render.py \
      --mesh {i} \
      --cam-x {cx} --rot-z {rz} \
      --cam-y {cy} --cam-z {cz} --lookat-y {ly} --lookat-z {lz} \
      -o {out} --width {w} --floor-y {fy} --lookat-x {lx} --height {h} {extras} "
    if not args.debug_render: cmd += " --final-render --samples 256"
    return [cmd]
  return cb

def runnable_cmds(cmds, stage_kind="run"):
  def cb():
    if stage_kind not in args.stages: return []
    missing_only = "" if not args.missing_only else " --missing-only "
    return [ c + missing_only for c in cmds ]
  return cb

dataset = [
  #("vietnam_lantern.fbx", "vietnam_lantern_small.jpeg", 1000000, None),
  #("cabbage.obj", "cabbage_diffuse.jpg", 300000, None),
  #("shiba.obj", "shiba_texture.png", 1000000, None),
  #("watercolor_girl.fbx", "watercolor-girl-albedo.jpg", 1000000, None),
  #("watercolor_cake.obj", "watercolor_cake.jpg", 1000000, None),
  #("silent_ash.obj", "silent_ash_texture.png", 10000000, None),
  #("scan_vase.obj", "scan_vase_texture.jpg", 500000, None),
  #("strawberry.obj", "strawberry_textures/diffuse.png", 1000000, None),
  #("ding_censer.obj", "ding_censer_textures/diffuse.jpg", 2000000, None),
  #("musashi_panels.obj", "musashi_panels_textures/diffuse.jpg", 1000000, None),
  #("tiger_lily.obj", "tiger_lily.jpeg", 2000000, None),
  #("shiny_fish.fbx", "shiny_fish_textures/Fishka_2_G_Fish_BaseColor2.jpg", 1000000, None),
  #("japanese_tray.obj", "japanese_tray_textures/diffuse.png", 1000000, None),
  #("jar_with_dragon_design.obj", "jar_with_dragon_design.png", 1000000, None),
  #("japanese_tea_cup.obj", "japanese_tea_cup_texture.png", 500000, None),

  #("musk_melon.obj", "", 4000000, 0.27555/2),
  #("fire_bellied_newt.obj", "fire_bellied_newt_diffuse.jpg", 1000000, 0.2733/2.),
  #("lychee.obj", "lychee_textures/lychee.jpg", 500000, 0.25),
  #("officebot.obj", "officebot_textures/diffuse.png", 500000, 0.5),

  #("building_front.obj", "building_front.jpg", 1000000, None),
  #("japanese_toro.obj", "japanese_toro_textures/japanese_toro_small.png", 850000, None),
  #("breakfast_still_life.obj", "", 1000000, 0.5),
  #("ibis.obj", "", 1000000, 0.5),


  # need to rerun this one and see what the problem is
  ("watermelon.obj", "watermelon.jpg", 1000000, None),

  ("chozuya.obj", "", 2000000, None),
  ("flowers_in_vase.obj", "flowers_in_vase.jpg", 500000, None),
  ("millers-falls-drill.fbx", "millers-falls-drill-textures/diffuse.png", 1000000, None),
  ("garlic_knight.obj", "", 1000000, None),

  # very expensive but doable?
  #("meadowsweet.obj", "meadowsweet_diffuse.jpeg", 500000, 0.5),
  #("apothecary_syrup_vessel.obj", "apothecary_syrup_vessel_diffuse.png", 2000000),
]

dataset_direct = [
  ("takifugu.obj", "", 1000000),
  ("musk_melon.obj", "", 100000),
  ("oshima_cherry.obj", "", 2000000),
  #("mango.obj", "", 1000000),
]

experiments = {
  # basic test of a cube
  "basic-cube": [
    *[
      run("cube.obj", f"cube_{k}.ply", f"-d data/uv_grid.png -t 100000 --sample-kind {k}")
      for k in ["approx"]#["exact", "approx", "direct"]
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
      "-d data/uv_grid.png --target-tri-ratio 0.5 --sample-kind exact --no-incremental-qem",
      is_abl=True
    ),
  ],

  # robustness tests
  "thin-tri": [
    *[
      run(
        "thin_tri.obj", f"thin_tri_{k}.ply",
        f"-d data/uv_grid.png --target-tri-ratio 1. --sample-kind {k}",
        is_abl=True
      ) for k in ["approx", "exact"]
    ]
  ],

  # testing non-axis aligned vertices
  "rot-uv": [
    *[
      run(
        "cube_rotated_uv.obj", f"cube_rot_uv_{k}.ply",
        f"-d data/uv_grid.png --target-tri-ratio 0.5 --sample-kind {k}",
      ) for k in ["exact", "approx"]
    ]
  ],

  "spot": [
    *[
      run(
        "spot_triangulated.obj", f"spot_{k}.ply",
        f"-d data/spot_texture.png --target-tri-ratio 1. \
        --sample-kind {k} --no-incremental-qem",
      ) for k in ["direct", "exact"]
    ]
  ],

  "spot_clustering": [
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
    #  f"-d data/angelfish_texture.jpg --target-tri-num 200000 --no-incremental-qem \
    #  --sample-kind direct",
    #),
    run(
      "../ablations/angelfish.ply",
      "angelfish_colors.ply",
      f"-t 250 --eigenvalue zero --cluster-vis ablations/angelfish_clusters.ply \
      --eigen-eps 1e-4 --color-eps 1e-5 --eigen-vis ablations/angelfish_eigen.ply",
      bin=clustering_bin, eval=False,
    ),
  ],
  "nanchan_clustering": [
    #run(
    #  "nanchan.obj",
    #  "nanchan.ply",
    #  f"-d data/nanchan_texture.png --target-tri-num 500000 --sample-kind exact \
    #  --no-incremental-qem",
    #),
    run(
      "../ablations/nanchan.ply",
      "nanchan_colors.ply",
      f"-t 200 --eigenvalue zero --cluster-vis ablations/nanchan_clusters.ply \
      --eigen-eps 1e-4 --color-eps 1e-6 --shape-metric angle-deviation \
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
  ],
  "dense-sphere": [
    #run("dense_sphere.obj", "dense_sphere.ply", "-d data/hokusai.jpg --no-incremental-qem \
    #--sample-kind approx -t 500000")
    run(
      "../ablations/dense_sphere.ply",
      "dense_sphere_colors.ply",
      f"-t 500 --eigenvalue zero --cluster-vis ablations/dense_sphere_clusters.ply \
      --eigen-eps 1e-4 --color-eps 1e-4 --eigen-vis ablations/dense_sphere_eigen.ply",
      bin=clustering_bin, eval=False
    ),
  ],

  "dense-sphere-smooth-boundaries": [
    #run("dense_sphere.obj", "dense_sphere.ply", "-d data/hokusai.jpg --no-incremental-qem \
    #--sample-kind approx -t 500000")
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
  "watercolor_cake": [
    run(
      "watercolor_cake.fbx", "watercolor_cake.ply",
      "-d data/watercolor_cake.tif --target-tri-ratio 0.1", False
    ),
  ],
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
    run("open_top_box.obj", "open_top_box.ply", "-d data/uv_grid.png --target-tri-ratio 0.5"),
    *[
      run(
        "../ablations/open_top_box.ply",
        f"open_top_box_{label}.obj",
        f"--weighting {weighting} --bake-texture open_top_box_{label}.png \
          --uv-svg ablations/open_top_box_{label}.svg --iters 100000 \
          --pos-color-norm {norm}",
        bin=tutte_bin, eval=False,
      )
      for (weighting, norm, label) in [
        #("uniform", "pos-only", "uniform"),

        #("length", "pos-only", "len"),
        #("length", "geometric-mean", "clen"),

        #("mean-value", "pos-only", "mv"),
        #("mean-value", "geometric-mean", "cmv"),

        ("laplacian", "pos-only", "lpl"),
        ("laplacian", "geometric-mean", "clpl")
      ]
    ],
  ],

  # testing different ways to weigh distance versus color
  "tutte-param-ogre": [
    #run(
    #  "ogre.obj", "ogre.ply",
    #  "-d data/ogre.png --target-tri-ratio 1. --no-incremental-delete --no-delete-degen \
    #  --sample-kind direct"
    #),
    *[
      run(
        "../ablations/ogre.ply",
        f"ogre_{label}.obj",
        f"--weighting color-length --pos-color-norm {norm} --bake-texture ogre_{label}.png \
          --uv-svg ablations/ogre_{label}.svg --iters 100000",
        bin=tutte_bin, eval=False,
      )
      for (norm, label) in [
        ("tester", "tester"),
        #("add", "add"),
        #("mul", "mul"),
        #("min", "min"),
        #("max", "max"),
        #("geometric-mean", "geom_mean"),
        #("color-only", "color_only"),
      ]
    ],
  ],

  "japanese_toro": [
    run(
      "japanese_toro.obj", "japanese_toro.ply",
      "-d data/japanese_toro_textures/japanese_toro_small.png --no-incremental-qem \
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

  "dataset-exact": [
    *[
      run(
        model, model[:-4] + "_exact.ply",
        f"{f'-d data/{texture}' if len(texture) else ''} -t {tri_num} \
          --no-incremental-qem --sample-kind exact",
        is_abl=False,
      )
      for (model, texture, tri_num, img_size_frac) in dataset
    ],
  ],
  "dataset-approx": [
    *[
      run(
        model, model[:-4] + "_approx.ply",
        f"{f'-d data/{texture}' if len(texture) else ''} -t {tri_num} \
          --no-incremental-qem --sample-kind approx \
          {'' if img_size_frac is None else f'--image-size-frac {img_size_frac}'}",
        is_abl=False,
      )
      for (model, texture, tri_num, img_size_frac) in dataset
    ],
  ],
  "dataset-direct": [
    *[
      run(
        model, model[:-4] + "_direct.ply",
        f"{f'-d data/{texture}' if len(texture) else ''} -t {tri_num} \
          --no-incremental-qem --sample-kind direct",
        is_abl=False,
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
  return a.parse_args()

args = arguments()
now = time.asctime(time.localtime())

experiment_timestamps = {}

exp_file = "experiment_log.json"

if len(args.experiments) > 0:
  assert(not os.system("cargo build --release"))

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

