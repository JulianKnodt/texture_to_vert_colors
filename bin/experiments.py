import os
import argparse
import sys
import time
import json
from itertools import chain

bin_file = "target/release/texture_to_vert_colors"
tutte_bin = "target/release/tutte_param"
smooth_bin = "target/release/smoothing"

args = None

out_dir = lambda is_ablation=False: "ablations" if is_ablation else "outputs"

def run(src, dst, flags, is_abl=True, src_dir="data", bin=bin_file, eval=True):
  def cb():
    if "run" not in args.stages: return []
    if args.match_output is not None and args.match_output not in dst: return []
    out_json = f"{out_dir(is_abl)}/{dst[:-4]}.json"
    cmds = [
      f"{bin} -i {src_dir}/{src} -o {out_dir(is_abl)}/{dst} {flags} --stats {out_json}",
    ]
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
  #("vietnam_lantern.fbx", "vietnam_lantern_small.jpeg", 1000000),
  #("cabbage.obj", "cabbage_diffuse.jpg", 300000),
  #("shiba.obj", "shiba_texture.png", 1000000),
  #("watercolor_girl.fbx", "watercolor-girl-albedo.jpg", 1000000),
  #("scan_vase.obj", "scan_vase_texture.jpg", 0.3),
  ("silent_ash.obj", "silent_ash_texture.png", 10000000),
  #("strawberry.obj", "strawberry_textures/diffuse.png", 500000),
  #("ding_censer.obj", "ding_censer_textures/diffuse.png", 500000),
  #("musashi_panels.obj", "musashi_panels_texture.png", 1000000),
  #("tiger_lily.obj", "tiger_lily.jpeg", 1000000),
  #("shiny_fish.fbx", "shiny_fish_textures/Fishka_2_G_Fish_BaseColor2.jpg", 1000000),
  #("apothecary_syrup_vessel.obj", "apothecary_syrup_vessel_diffuse.png", 500000),
  #("japanese_tray.obj", "japanese_tray_textures/diffuse.png", 500000),
]

experiments = {
  # basic test of a cube
  "basic-cube": [
    *[
      run("cube.obj", f"cube_{k}.ply", f"-d data/uv_grid.png -t 100000 --sample-kind {k}")
      for k in ["exact", "approx", "direct"]
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

  "dataset": [
    *[
      run(
        model, model[:-4] + ".ply",
        f"-d data/{texture} -t {tri_num} --no-incremental-qem",
        is_abl=False,
      )
      for (model, texture, tri_num) in dataset
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

