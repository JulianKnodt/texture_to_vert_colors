import trimesh
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import trange, tqdm

import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import robust_laplacian

#torch.set_anomaly_enabled(True)

def arguments():
  a = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter
  )
  a.add_argument("-i", "--input", required=True, help="Path to input mesh")
  a.add_argument("-o", "--output", required=True, help="Path to saved mesh")
  a.add_argument(
    "--spokes-and-rims", action="store_true",
    help="Instead of just using spokes, use spokes-and-rims"
  )
  a.add_argument(
    "--device", default="cpu",
    help="Device to run on",
  )
  a.add_argument(
    "--cubeness", type=float, default=1.,
    help="How cube-y to make the output"
  )
  a.add_argument(
    "--color-cubeness", type=float, default=0.,
    help="How much to weigh the straightness of color gradients"
  )
  a.add_argument(
    "--opt-color", action="store_true",
    help="Optimize colors as well as vertex positions (TODO this needs to be figured out more)",
  )
  a.add_argument(
    "--iters", type=int, default=75,
    help="Iterations to run of cubification",
  )
  a.add_argument(
    "--lr", type=float, default=3e-3,
    help="Learning rate",
  )
  return a.parse_args()

def main():
  args = arguments()
  mesh = trimesh.load_mesh(args.input)

  #TODO here need to do mesh normalization
  device = args.device
  if not torch.cuda.is_available() and device == "cuda":
    print("[WARN]: CUDA is not available, defaulting to CPU")
    device = "cpu"

  v = torch.from_numpy(mesh.vertices).float().to(device)
  og_v = v.clone().detach()
  v.requires_grad_(True)

  V = v.shape[0]
  print(f"[INFO]: Input #F = {mesh.faces.shape[0]}, #V = {V}")

  vc = None
  if args.color_cubeness > 0:
    assert(hasattr(mesh.visual, "vertex_colors")), "Input mesh does not have vertex colors"
    # TODO here emit an error if this does not have vertex colors
    vc = torch.from_numpy(mesh.visual.vertex_colors).div(255).to(device)[:, :3]
    vc.requires_grad_(True)
    assert(vc.shape == v.shape)

  L, M = robust_laplacian.mesh_laplacian(np.array(mesh.vertices), np.array(mesh.faces))
  L_map = {}
  for ci in range(V):
    for ind in range(L.indptr[ci], L.indptr[ci+1]):
        row = int(L.indices[ind])
        data = L.data[ind]
        L_map[(row, ci)] = data

  print("[INFO]: Starting precomputation...")
  # for each vertex, construct neighborhood edges of spokes-and-rims
  nbr_edges = [set() for _ in range(V)]
  bary_area = torch.zeros(V,requires_grad=False, device=device)

  area_weighted_normal = torch.zeros_like(v, requires_grad=False)
  luma_prim_dir = torch.zeros_like(v, requires_grad=False)

  all_edges = set()
  with torch.no_grad():
    # discrete set of edges
    for vi0, vi1, vi2 in tqdm(mesh.faces, leave=False):
      vi0, vi1, vi2 = [int(vi) for vi in [vi0, vi1, vi2]]

      edge_set = [(vi0, vi1), (vi1, vi2), (vi2, vi0)]
      nbr_edges[vi0].update(edge_set)
      nbr_edges[vi1].update(edge_set)
      nbr_edges[vi2].update(edge_set)
      all_edges.update(
        [(min(e0, e1), max(e0, e1)) for (e0, e1) in edge_set]
      )

    # Tensors for each edge set
    vis = [[], [], []]
    for vi_ in mesh.faces:
      for i in range(3):
        vis[i].append(int(vi_[i]))

    vi0, vi1, vi2 = [torch.tensor(vi, device=device) for vi in vis]
    v0 = v[vi0]
    v1 = v[vi1]
    v2 = v[vi2]

    if vc is not None:
      l0 = luma(vc[vi0])
      l1 = luma(vc[vi1])
      l2 = luma(vc[vi2])

      normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
      area = normals.norm(dim=-1) / 2.
      for vi in [vi0, vi1, vi2]:
        bary_area[vi] += area / 3.
        area_weighted_normal[vi] += F.normalize(normals, dim=-1) * area[:, None]

      area_weighted_normal = F.normalize(area_weighted_normal, dim=-1)


      # given area weighted normals, compute linear functional G for luma of each point?
      P = torch.stack([
        append_val(v0),
        append_val(v1),
        append_val(v2),
        append_val(F.normalize(normals, dim=-1), one=True),
      ], dim=-2)
      S = torch.stack([
        l0, l1, l2, torch.zeros_like(l0)
      ], dim=-1)
      # TODO in theory could manually implement the inversion here
      GD = torch.linalg.lstsq(P, S, rcond=1e-5, driver="gelsd").solution

      assert(GD.isfinite().all())
      G = GD[:, :3]

      # store G into each vertex
      for vi in [vi0, vi1, vi2]:
        # TODO normalize G[vi]?
        luma_prim_dir[vi] += G[vi] * area[:, None]

  #ei0, ei1 = [
  #  torch.tensor([e[i] for e in all_edges])
  #  for i in [0,1]
  #]
  #luma_delta = (luma(vc[ei0]) - luma(vc[ei1])).abs()


  print("[INFO]: Starting Laplacian computation")
  laplacians = []
  for es in tqdm(nbr_edges, leave=False):
    lapl_diag = []
    for e0, e1 in es:
      lapl_diag.append(-L_map.get((e0, e1), 0))
      #lapl_diag.append(-L[(e0, e1)]) # do not index directly, otherwise it will be slow
    laplacians.append(torch.tensor(lapl_diag, dtype=torch.float, device=device))

  print("[INFO]: Constructing acceleration structures...")
  max_degree = max(len(nes) for nes in nbr_edges) + 1
  # edge tensors
  nbr_ets = [ [] for _ in range(max_degree) ]
  # indices
  nbr_eis = [ [] for _ in range(max_degree) ]
  nbr_ls  = [ [] for _ in range(max_degree) ]
  for vi in range(V):
    nes = nbr_edges[vi]
    # NOTE: theoretically this iteration is a bit sketchy since the set order isn't guaranteed,
    # but in practice seems to be fine since the sets aren't modified
    nes = torch.stack([
      torch.tensor([ne[i] for ne in nes], device=device) for i in [0,1]
    ], dim=-1)

    N = nes.shape[0]
    nbr_ets[N].append(nes)
    nbr_eis[N].append(vi)
    nbr_ls[N].append(laplacians[vi])

  nbr_ets = [ torch.stack(nbr_et) for nbr_et in nbr_ets if len(nbr_et) > 0 ]
  nbr_eis = [ torch.tensor(nbr_ei, device=device) for nbr_ei in nbr_eis if len(nbr_ei) > 0 ]
  nbr_ls  = [ torch.stack(nbr_ls) for nbr_ls in nbr_ls if len(nbr_ls) > 0 ]

  # DONE COMPUTING ACCELERATION ---

  params=[v]
  if args.opt_color:
    assert(vc is not None)
    params.append(vc)
  opt = optim.Adam(params=[v], lr=3e-3)

  t_outer = trange(args.iters)

  quats = torch.zeros(
    V, 4,
    device=device,
    requires_grad = True,
  )
  with torch.no_grad(): quats[:, -1] = 1.

  rot_opt = optim.Adam(params=[quats], lr=3e-3)
  for i in t_outer:
    # START QUAT OPTIMIZATION --------

    vd = v.clone().detach().requires_grad_(False)
    t = trange(25, leave=False)
    for it in t:
      rot_opt.zero_grad()

      mats = quat_to_mat(F.normalize(quats, dim=-1))
      loss = 0
      for i in range(len(nbr_ets)):
        idxs = nbr_eis[i]
        ei0, ei1 = [ei.squeeze(-1) for ei in nbr_ets[i].split([1,1], dim=-1)]
        R = mats[idxs]

        D = og_v[ei1] - og_v[ei0]
        Dp = vd[ei1] - vd[ei0]
        # extra weighting term
        #Li = (luma(vc[ei1]) - luma(vc[ei0])).abs()
        #Li = (1. - Li)[..., None]

        z = (R @ D.mT) - Dp.mT

        Ls = nbr_ls[i][:, None]
        arap = 0.5 * torch.vmap(torch.trace)((z * Ls) @ z.mT)

        A = bary_area[idxs]
        ni = area_weighted_normal[idxs, :, None]
        cubeness = 0
        if args.cubeness > 0:
          cubeness = args.cubeness * A * (R @ ni).abs().squeeze(dim=-1).sum(dim=-1)

        loss = loss + (arap + cubeness).sum()
      t.set_postfix(L=f"{loss.item():.3f}")
      loss.backward()
      rot_opt.step()
      quats.data = F.normalize(quats.data, dim=-1)

      ## flat version kept for posterity
      #for vi in range(V):
      #  R = mats[vi]
      #  D = torch.stack(
      #    [og_v[e1] - og_v[e0] for [e0, e1] in nbr_edges[vi]],
      #    dim=-1,
      #  )
      #  Dp = torch.stack(
      #    [vd[e1] - vd[e0] for [e0, e1] in nbr_edges[vi]],
      #    dim=-1,
      #  )
      #  z = (R @ D) - Dp
      #  arap = 0.5 * torch.trace((z * laplacians[vi][None]) @ z.T)
      #  cubeness = args.cubeness * bary_area[vi] * (R @ area_weighted_normal[vi]).abs().sum()
      #  loss = loss + arap + cubeness

    mats = quat_to_mat(quats).detach().requires_grad_(False)
    # END QUAT OPTIMIZATION ----------

    opt.zero_grad()
    loss = 0

    ## flat version kept for posterity
    #for vi in range(V):
    #  R = mats[vi]
    #  D = torch.stack(
    #    [og_v[e1] - og_v[e0] for [e0, e1] in nbr_edges[vi]],
    #    dim=-1,
    #  )
    #  Dp = torch.stack(
    #    [v[e1] - v[e0] for [e0, e1] in nbr_edges[vi]],
    #    dim=-1,
    #  )
    #  z = (R @ D) - Dp
    #  arap = 0.5 * torch.trace((z * laplacians[vi][None]) @ z.T)
    #  cubeness = args.cubeness * bary_area[vi] * (R @ area_weighted_normal[vi]).abs().sum()
    #  loss = loss + arap + cubeness

    for i in range(len(nbr_ets)):
      idxs = nbr_eis[i]
      ei0, ei1 = [ei.squeeze(-1) for ei in nbr_ets[i].split([1,1], dim=-1)]
      R = mats[idxs]
      D = og_v[ei1] - og_v[ei0]
      Dp = v[ei1] - v[ei0]
      z = (R @ D.mT) - Dp.mT

      Ls = nbr_ls[i][:, None]

      arap = 0.5 * torch.vmap(torch.trace)((z * Ls) @ z.mT)

      A = bary_area[idxs]
      ni = area_weighted_normal[idxs, :, None]
      cubeness = 0
      if args.cubeness > 0:
        cubeness = args.cubeness * A * (R @ ni).abs().squeeze(dim=-1).sum(dim=-1)

      loss = loss + (arap + cubeness).sum()

    t_outer.set_postfix(L=f"{loss.item():.3f}")
    loss.backward()
    opt.step()
  mesh.vertices = v.detach().numpy()
  mesh.export(args.output)

def quat_to_mat(q, dim=-1):
  x,y,z,w = q.split([1,1,1,1], dim=-1)
  qxx = x * x
  qyy = y * y
  qzz = z * z
  qxz = x * z
  qxy = x * y
  qyz = y * z
  qwx = w * x
  qwy = w * y
  qwz = w * z

  return torch.cat([
    torch.stack([1. - 2. * (qyy + qzz), 2. * (qxy - qwz), 2. * (qxz + qwy)], dim=dim),
    torch.stack([2. * (qxy + qwz), 1. - 2. * (qxx + qzz), 2. * (qyz - qwx)], dim=dim),
    torch.stack([2. * (qxz - qwy), 2. * (qyz + qwx), 1. - 2. * (qxx + qyy)], dim=dim),
  ], dim=dim-1)

#t = F.normalize(torch.tensor([0.3,0.2,-0.5,0.7071]), dim=-1)
#print(quat_to_mat(torch.tensor([t.tolist(), [0., 0., 0., 1.]])))
#exit()

def luma(rgb, dim=-1,keepdim=False):
  r,g,b = rgb.split([1,1,1], dim=dim)
  l = 0.299 * r + 0.587 * g + 0.114 * b
  if keepdim: return l
  return l.squeeze(dim=dim)

def append_val(v, one:bool=False, dim=-1):
  s = list(v.shape)
  s[dim] = 1
  func = torch.zeros if one else torch.ones
  return torch.cat([v, func(s, device=v.device, dtype=v.dtype)], dim=dim)

if __name__ == "__main__": main()
