from argparse import ArgumentParser
import numpy as np
import json
import os

def arguments():
  a = ArgumentParser()
  a.add_argument("-o", "--original-mesh", required=True, help="Original mesh")
  a.add_argument("-n", "--new-mesh", required=True, help="New mesh")
  a.add_argument("--stats", default=None, help="File to write statistics to")
  a.add_argument("--num-random-samples", default=100000, type=int, help="Number of random samples to use")
  a.add_argument("--original-image", default=None, help="Texture image of original model")
  a.add_argument("--new-image", default=None, help="Texture image of new model")
  return a.parse_args()

def main():
  args = arguments()
  print("[INFO]: Computing distance between meshes")

  import trimesh

  og_mesh = trimesh.load(args.original_mesh, force="mesh")
  new_mesh = trimesh.load(args.new_mesh, force="mesh")

  if type(new_mesh) == trimesh.PointCloud or type(new_mesh) == list:
    print(f"No faces left in {args.new_mesh}")
    return

  og_bb_diag = np.linalg.norm(og_mesh.vertices.max(axis=0) - og_mesh.vertices.min(axis=0))
  if len(new_mesh.vertices) == 0:
    print(f"No vertices left in {args.new_mesh}")
    return
  new_bb_diag = np.linalg.norm(new_mesh.vertices.max(axis=0) - new_mesh.vertices.min(axis=0))
  bb_diag = max(og_bb_diag, new_bb_diag)


  import igl

  N = args.num_random_samples
  nv = new_mesh.vertices
  # TODO here also concatenate colors
  if N > 0:
    b,fis,new_positions = igl.random_points_on_mesh(N, new_mesh.vertices, new_mesh.faces)
    nv = np.concatenate([nv, new_positions])

  new_to_og,_,_ = igl.point_mesh_squared_distance(
    nv,
    og_mesh.vertices,
    og_mesh.faces,
  )
  new_to_og = np.sqrt(new_to_og)/bb_diag

  ov = og_mesh.vertices
  if N > 0:
    bary,fis,new_positions = igl.random_points_on_mesh(N, og_mesh.vertices, og_mesh.faces)
    ov = np.concatenate([ov, new_positions])

  og_to_new,_,_ = igl.point_mesh_squared_distance(
    ov,
    new_mesh.vertices,
    new_mesh.faces,
  )
  og_to_new = np.sqrt(og_to_new)/bb_diag

  hausdorff = max(new_to_og.max(), og_to_new.max())
  chamfer = new_to_og.mean() + og_to_new.mean()


  print()
  print(f"hausdorff(new to original) = {new_to_og.max()}")
  print(f"hausdorff(original to new) = {og_to_new.max()}")
  print(f"hausdorff = {hausdorff}")
  print()
  print(f"chamfer(new to original) = {new_to_og.mean()}")
  print(f"chamfer(original to new) = {og_to_new.mean()}")
  print(f"chamfer = {chamfer}")
  print()
  stats = {}
  if args.stats is not None and os.path.exists(args.stats):
    with open(args.stats, "r") as f:
      try:
        stats = json.load(f)
      except:
        print("Failed to json decode: ", f.read(), " from ", args.stats)
        exit(1)

  stats["hausdorff_new_to_original"] = new_to_og.max()
  stats["hausdorff_original_to_new"] = og_to_new.max()
  stats["hausdorff"] = hausdorff
  stats["chamfer"] = chamfer
  stats["chamfer_new_to_original"] = new_to_og.mean()
  stats["chamfer_original_to_new"] = og_to_new.mean()

  if args.stats is not None:
    with open(args.stats, "w") as f:
      json.dump(stats, f, indent=2)


if __name__ == "__main__":
  main()
