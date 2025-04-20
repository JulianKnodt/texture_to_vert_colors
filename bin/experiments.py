import os
import argparse
import sys
import time
import json
from itertools import chain

if os.name == "nt":
  bin_file = ...
else:
  bin_file = "target/release/texture_to_vert_colors"

args = None

out_dir = lambda is_ablation=False: "ablations" if is_ablation else "outputs"

def run(src, dst, flags, is_abl=True, src_dir="data"):
  def cb():
    if "run" not in args.stages: return []
    if args.match_output is not None and args.match_output not in dst: return []
    out_json = f"{out_dir(is_abl)}/{dst[:-4]}.json"
    cmds = [
      f"{bin_file} -i {src_dir}/{src} -o {out_dir(is_abl)}/{dst} {flags} --stats {out_json}",
    ]
    if not args.no_eval:
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

experiments = {
  "basic-cube": [
    run("cube.obj", "cube.ply", "-d data/colors.png", False),
  ],
  "sphere": [
    run("sphere.obj", "sphere.ply", "-d data/uv_grid.png", False),
  ],
  "rot-uv": [
    run("cube_rotated_uv.obj", "cube_rot_uv.ply", "-d data/colors.png", False),
  ],
  "thin-tri": [
    run("thin_tri.obj", "thin_tri.ply", "-d data/uv_grid.png", False),
  ],
  "spot": [
    run(
      "spot_triangulated.obj", "spot_triangulated.ply",
      "-d data/spot_texture.png", False
    ),
  ],
  "planar": [
    run("plane.obj", "plane.ply", "-d data/vase_2k.png", False),
  ],
  "shiba": [
    run("shiba.obj", "shiba.ply", "-d data/uv_grid.png", False),
  ],
  "watercolor_cake": [
    run("watercolor_cake.obj", "watercolor_cake.ply", "-d data/watercolor_cake.tif"),
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

