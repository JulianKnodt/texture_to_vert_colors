import trimesh
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import random
import math

def arguments():
  a = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  a.add_argument("-i", "--input", required=True, help="Input mesh")
  a.add_argument("-o", "--output", required=True, help="Output mesh")
  a.add_argument("--uniform", action="store_true", help="Use uniform weighting instead")
  a.add_argument("--color-kind", choices=["max", "concat", "add", "mul", "color-only", "bilateral"], default="add")
  a.add_argument("--color-weight", default=0, type=float, help="How much to weigh color")
  a.add_argument("--debug-colors", action="store_true", help="Render debug colors as output")
  a.add_argument("--weight", default=1e-2, type=float, help="Weight of smoothing")
  a.add_argument("--stats", help="Currently unused", default=None)
  return a.parse_args()

def main():
  args = arguments()
  print("Loading mesh")
  mesh = trimesh.load_mesh(args.input, process=False)
  vert_colors = mesh.visual.vertex_colors.astype(float) / 255.
  vert_colors = vert_colors[:, :3]
  V = len(mesh.vertices)
  # compute laplacian
  kept_verts = set(
    i for i in range(V)
    #if random.random() > 0.25
    #if luma(vert_colors[i]) < 0.5
  )
  if args.debug_colors:
    for vi in range(V):
      if vi in kept_verts: continue
      mesh.visual.vertex_colors[vi] = 0

    mesh.export(args.output, encoding="ascii")
    return


  M = [0] * V
  for vis in tqdm(mesh.faces, leave=False):
    vi,vj,vk = [mesh.vertices[idx] for idx in vis]
    vci,vcj,vck = [vert_colors[idx][:3] for idx in vis]
    a = dist_fn(vi,vci, vj,vcj, args.color_weight, args.color_kind)
    b = dist_fn(vj,vcj, vk,vck, args.color_weight, args.color_kind)
    c = dist_fn(vi,vci, vk,vck, args.color_weight, args.color_kind)
    area = herons(a,b,c) / 3
    for vi in vis: M[vi] += area

  M = np.array(M)


  EPS = 3e-3 # are higher values ok?
  rows = []
  cols = []
  data = []
  totals = [0] * V
  for [vi0,vi1,vi2] in tqdm(mesh.faces, leave=False):
    edges = [
      [vi0,vi1,vi2],
      [vi1,vi2,vi0],
      [vi2,vi0,vi1],
    ]

    for ijk in edges:
      vi,vj,vk = [mesh.vertices[idx] for idx in ijk]
      vci,vcj,vck = [vert_colors[idx][:3] for idx in ijk]
      a = dist_fn(vi,vci, vj,vcj, args.color_weight, args.color_kind)
      b = dist_fn(vj,vcj, vk,vck, args.color_weight, args.color_kind)
      c = dist_fn(vi,vci, vk,vck, args.color_weight, args.color_kind)
      area = herons(a,b,c)
      v = a * a + b * b - c * c
      cot_c = v / (4 * area + EPS)
      vals = [
        (ijk[0], ijk[2], 1. if args.uniform else cot_c),
        (ijk[2], ijk[0], 1. if args.uniform else cot_c),
      ]
      for r, c, val in vals:
        val /= (2. * M[r] + EPS)
        val = -val
        totals[r] += val
        rows.append(r)
        cols.append(c)
        data.append(val * args.weight)

  for vi in range(V):
    rows.append(vi)
    cols.append(vi)
    data.append(-totals[vi] * args.weight)

  # add in kept verts
  for b in range(V):
    rows.append(b)
    cols.append(b)
    data.append(1)

  csr = sp.csr_matrix((data, (rows, cols)), shape=(V, V))

  r, g, b = [np.squeeze(c) for c in np.split(vert_colors, 3, axis=1)]

  new_r = sp.linalg.spsolve(csr, r)
  new_g = sp.linalg.spsolve(csr, g)
  new_b = sp.linalg.spsolve(csr, b)

  new_rgb = np.stack([new_r, new_g, new_b], axis=-1)
  mesh.visual.vertex_colors = (np.clip(new_rgb, 0,1) * 255).astype(int)

  mesh.export(args.output, encoding="ascii")

def luma(rgb):
  return np.sum(np.array([0.299, 0.587, 0.114]) * rgb[:3])

def dist_fn(va,vca, vb, vcb, color_weight=1e-4, kind="add"):
  geom = np.linalg.norm(va - vb)
  if color_weight == 0. and kind != "color-only": return geom
  color = abs(luma(vca) - luma(vcb))
  #color = np.linalg.norm(vca - vcb)
  if kind == "add":
    result = geom + color_weight * color
  elif kind == "max":
    result = np.maximum(geom, color_weight * color)
  elif kind == "concat":
    result = np.linalg.norm(
      np.concatenate([va, color_weight * vca]) - \
      np.concatenate([vb, color_weight * vcb])
    )
  elif kind == "color-only": return color + 1
  elif kind == "mul": return geom * (color ** color_weight)
  elif kind == "bilateral":
    combined = gaussian(geom, 1.) * \
      (gaussian(color, 1.) ** color_weight),
    return inv_gaussian(combined)
  else: raise NotImplementedError(kind)
  #return result
  return result / (1 + color_weight)

def herons(e0, e1, e2):
  s = (e0 + e1 + e2) / 2
  #return np.sqrt(s * (s - e0) * (s - e1) * (s - e2))
  f = lambda v: np.log(max(v, 1e-12))
  return np.exp(0.5 * (f(s) + f(s - e0) + f(s - e1) + f(s - e2)))

def minmax(a, b): return [min(a,b), max(a,b)]

def gaussian(x, sigma):
  return 1/(sigma * math.sqrt(math.tau)) * math.exp(-0.5 * x * x / (sigma * sigma))

def inv_gaussian(y, sigma):
  #assert(y > 0.),y
  y = max(y, 1e-6)
  x2 = math.log(y * sigma * math.sqrt(math.tau)) * -2 * sigma * sigma
  assert(x2 >= -1e-3), x2
  return math.sqrt(max(x2, 1e-6))

if __name__ == "__main__": main()
