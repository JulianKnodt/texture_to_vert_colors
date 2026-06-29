import trimesh
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import trange
import torch.optim as optim
import math

torch.autograd.set_detect_anomaly(True)

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument("-i", "--input", required=True, help="Input mesh")
  a.add_argument("--uv", help="Mesh of texture coordinates", default=None)
  a.add_argument("-o", "--output", required=True, help="Output mesh")
  a.add_argument("--color-weight", default=1e-4, type=float, help="How much to weigh color")
  a.add_argument("--iters", type=int, default=500, help="Number of optimization iters")
  a.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
  return a.parse_args()

def dot(a, b, dim=-1): return torch.sum(a*b, dim=dim)
def norm(v, dim=-1): return torch.linalg.norm(v, ord=2, dim=dim)

def log_tex_tri_areas(uv, f):
  vis = f.split([1,1,1], dim=-1)
  vt0, vt1, vt2 = [uv[ti.squeeze(-1)] for ti in vis]

  return herons_log_area(norm(vt1 - vt0), norm(vt0 - vt2), norm(vt2 - vt1))

# for degenerate geometric triangles, just assign the UV coordinates to all be the same
def degen_reg(v, uv, f):
  vis = f.split([1,1,1], dim=-1)
  vt0, vt1, vt2 = [uv[ti.squeeze(-1)] for ti in vis]
  et0 = vt0 - vt1
  et1 = vt2 - vt0
  v0, v1, v2 = [v[vi.squeeze(-1)] for vi in vis]
  e0 = v0 - v1
  e1 = v2 - v0
  tri_areas = torch.cross(e0, e1, dim=-1).norm(dim=-1).div(2)

  centroid = (vt0 + vt1 + vt2) / 3

  return torch.where(
    tri_areas < 1e-12,
    norm(centroid - vt0) + norm(centroid - vt1) + norm(centroid - vt2),
    torch.zeros_like(tri_areas),
  )

def bijective_mapping_energy(v, vc, uv, f, color_weight: float = 0.):
  vis = f.split([1,1,1], dim=-1)
  vt0, vt1, vt2 = [uv[ti.squeeze(-1)] for ti in vis]
  et0 = vt0 - vt1
  et1 = vt2 - vt0

  v0, v1, v2 = [
    concat(v[vi.squeeze(-1)], vc[vi.squeeze(-1)], color_weight)
    for vi in vis
  ]

  e0 = v0 - v1
  e1 = v2 - v0

  log_tri_areas = herons_log_area(norm(e0), norm(e1), norm(v1 - v2))

  dirichlet_energy = ((norm(et1)*norm(e0)).square() + (norm(et0)*norm(e1)).square())/2
  dirichlet_energy = dirichlet_energy - (dot(et0, et1)*dot(e0, e1))

  valid = dirichlet_energy > 0
  assert(valid.all()), f"{dirichlet_energy[~valid]}, {tri_areas[~valid]}, {vt0[~valid]}"
  valid = dirichlet_energy.isfinite()
  assert(valid.all()), f"{dirichlet_energy[~valid]}, {tri_areas[~valid]}, {vt0[~valid]}"

  log_dirichlet_energy = dirichlet_energy.log() - math.log(2) - 2 * log_tri_areas

  valid = (dirichlet_energy >= 0) & (dirichlet_energy.isfinite())
  assert(valid.all()), f"{dirichlet_energy[~valid]}, {tri_areas[~valid]}"

  # this is exp(log(a^2/t^2)), simplified to reduce numerical instability
  ltta = log_tex_tri_areas(uv, f) + math.log(16)

  log_diff = 2 * (log_tri_areas - ltta)
  assert(log_diff.isfinite().all())
  sq_ratio = log_diff.exp()
  valid = sq_ratio.isfinite()
  assert(valid.all()), f"{log_diff[~valid]}, {dirichlet_energy[~valid]}, {tri_areas[~valid]}, {ltta[~valid]}"

  raw = (1 + sq_ratio) * log_dirichlet_energy.exp()
  return raw.clamp(min=4) - 4
  #return log_tri_areas.exp() * (raw.clamp(min=4) - 4)

def main():
  args = arguments()
  print("Loading mesh")
  mesh = trimesh.load_mesh(args.input, process=False)
  v = torch.from_numpy(mesh.vertices)
  mesh.visual.vertex_colors = mesh.visual.vertex_colors.astype(float) / 255.
  vc = torch.from_numpy(mesh.vertices)

  if hasattr(mesh.visual, "uv"):
    uv = mesh.visual.uv
  else:
    uv_mesh = trimesh.load_mesh(args.uv, process=False)
    uv = uv_mesh.vertices[:, :2] + 1

  assert(vc.shape[0] == v.shape[0])
  assert(uv.shape[0] == v.shape[0])
  uv = torch.from_numpy(uv).requires_grad_(True)

  f = torch.from_numpy(mesh.faces)

  opt = optim.Adam(params=[uv], lr=args.lr)
  t = trange(args.iters)
  for i in t:
    opt.zero_grad()
    e = bijective_mapping_energy(v, vc, uv, f, args.color_weight)
    assert((e >= 0).all())
    loss = e.mean()

    #reg = degen_reg(v, uv, f).mean()
    reg = 0
    # for triangles where the area is less than a threshold, force the uv coordinates to be
    # their mean vaue

    t.set_postfix(L=f"{loss:.02e}", R=f"{reg:.02e}")
    total = loss + reg
    total.backward()

    opt.step()

  uv = uv.detach().numpy()
  mesh.vertices[:,0] = uv[:, 0]
  mesh.vertices[:,1] = uv[:, 1]
  mesh.vertices[:,2] = 0
  mesh.export(args.output, encoding="ascii")

def luma(rgb):
  return np.sum(np.array([0.299, 0.587, 0.114]) * rgb)

def concat(v, vc, color_weight:float=1e-4, dim=-1):
  return torch.concat([v, color_weight * vc], dim=dim)

def herons_log_area(e0, e1, e2):
  s = (e0 + e1 + e2) / 2
  #return np.sqrt(s * (s - e0) * (s - e1) * (s - e2))
  f = lambda v: v.clamp(min=1e-14).log()
  return 0.5 * (f(s) + f(s - e0) + f(s - e1) + f(s - e2))

if __name__ == "__main__": main()
