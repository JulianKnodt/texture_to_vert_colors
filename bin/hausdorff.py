from argparse import ArgumentParser
import numpy as np
import json
import os

import igl

def arguments():
  a = ArgumentParser()
  a.add_argument("-o", "--original-mesh", required=True, help="Original mesh")
  a.add_argument("-n", "--new-mesh", required=True, help="New mesh")
  a.add_argument("--stats", default=None, help="File to write statistics to")
  a.add_argument("--num-random-samples", default=100000, type=int, help="Number of random samples to use")
  return a.parse_args()

def get_color_per_vertex(mesh):
  try:
    return mesh.visual.vertex_colors[:, :3]
  except:
    dims = np.array(mesh.visual.material.image.size)[None, :]
    uv = np.copy(mesh.visual.uv)
    uv[:, 1] = 1 - uv[:, 1]
    coords = uv * dims
    img = mesh.visual.material.image
    img_data = img.getdata()
    img = np.array(img_data).reshape(img.size[0], img.size[1], len(img.getbands()))
    img = np.moveaxis(img, 1, 0)
    coords = coords.astype(int)
    u = coords[:, 0]
    v = coords[:, 1]
    return img[u,v,:3]

def get_color(mesh, face_idxs, barys=None, pos=None):
  a,b,c = [
    mesh.vertices[vi.squeeze(-1)]
    for vi in np.split(mesh.faces[face_idxs], 3,axis=-1)
  ]
  if barys is None:
    assert(pos is not None)
    barys = igl.barycentric_coordinates(pos, a, b, c)
    barys[~np.isfinite(barys)] = 0
  b0, b1, b2 = [barys[:, i][:, None] for i in [0,1,2]]

  try:
    ac,bc,cc = [
      mesh.visual.vertex_colors[vi.squeeze(-1)]
      for vi in np.split(mesh.faces[face_idxs], 3,axis=-1)
    ]
    interp_colors = ac * b0 + bc * b1 + cc * b2
    return interp_colors[:, :3]
  except:
    dims = np.array(mesh.visual.material.image.size)[None, :]
    auv, buv, cuv = [
      mesh.visual.uv[vi.squeeze(-1)]
      for vi in np.split(mesh.faces[face_idxs], 3,axis=-1)
    ]
    interp_uv = auv * b0 + buv * b1 + cuv * b2
    interp_uv[:, 1] = 1 - interp_uv[:, 1]
    coords = interp_uv * dims
    img = mesh.visual.material.image
    img_data = img.getdata()
    img = np.array(img_data).reshape(img.size[0], img.size[1], len(img.getbands()))
    img = np.moveaxis(img, 1, 0)
    coords = coords.astype(int)
    u = coords[:, 0]
    v = coords[:, 1]
    return img[u,v,:3]

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

  N = args.num_random_samples
  nv = new_mesh.vertices
  nvc = get_color_per_vertex(new_mesh)

  if N > 0:
    b,fis,new_pos = igl.random_points_on_mesh(N, new_mesh.vertices, new_mesh.faces)
    nv = np.concatenate([nv, new_pos])
    new_colors = get_color(new_mesh, fis, b)
    nvc = np.concatenate([nvc, new_colors])


  new_to_og,og_face_idxs,og_pts = igl.point_mesh_squared_distance(
    nv,
    og_mesh.vertices,
    og_mesh.faces,
  )
  og_nearest_color = get_color(og_mesh, og_face_idxs, pos=og_pts)
  #import matplotlib.pyplot as plt
  #fig = plt.figure()
  #ax = fig.add_subplot(projection='3d')
  #ax.scatter(og_pts[:,0], og_pts[:,1], og_pts[:,2], c=og_nearest_color/255)
  #plt.show()
  #exit()
  avg_new_to_og_color = np.mean(np.linalg.norm((nvc - og_nearest_color)/255., axis=-1))

  new_to_og = np.sqrt(new_to_og)/bb_diag

  ov = og_mesh.vertices
  ovc = get_color_per_vertex(og_mesh)
  if N > 0:
    bary,fis,new_positions = igl.random_points_on_mesh(N, og_mesh.vertices, og_mesh.faces)
    ov = np.concatenate([ov, new_positions])
    og_colors = get_color(og_mesh, fis, bary)
    ovc = np.concatenate([ovc, og_colors])

  og_to_new,new_face_idxs,new_pts = igl.point_mesh_squared_distance(
    ov,
    new_mesh.vertices,
    new_mesh.faces,
  )
  new_nearest_color = get_color(new_mesh, new_face_idxs, pos=new_pts)
  avg_og_to_new_color = np.mean(np.linalg.norm((ovc - new_nearest_color)/255., axis=-1))

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
  print(f"avg color(new to original) = {avg_new_to_og_color}")
  print(f"avg color(original to new) = {avg_og_to_new_color}")
  print(f"avg color diff = {(avg_new_to_og_color + avg_og_to_new_color)/2}")
  print()
  if args.stats is None: return;
  stats = {}
  if os.path.exists(args.stats):
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

  stats["avg_color_new_to_original"] = avg_new_to_og_color
  stats["avg_color_original_to_new"] = avg_og_to_new_color
  stats["avg_color_diff"] = (avg_og_to_new_color + avg_new_to_og_color) / 2

  with open(args.stats, "w") as f:
    json.dump(stats, f, indent=2)


if __name__ == "__main__":
  main()
