import trimesh
import argparse
import numpy as np
import scipy.sparse as sp

def arguments():
  a = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  a.add_argument("-i", "--input", required=True, help="Input mesh")
  a.add_argument("-o", "--output", required=True, help="Output mesh")
  a.add_argument("--uniform", action="store_true", help="Use uniform weighting instead")
  a.add_argument("--color-kind", choices=["max", "concat", "add", "color-only"], default="add")
  a.add_argument("--color-weight", default=1e-4, type=float, help="How much to weigh color")
  return a.parse_args()

def main():
  args = arguments()
  print("Loading mesh")
  mesh = trimesh.load_mesh(args.input, process=False)
  mesh.visual.vertex_colors = mesh.visual.vertex_colors.astype(float) / 255.
  V = len(mesh.vertices)
  # compute laplacian
  [bd_verts] = boundary(mesh)

  delta = [0.]
  total_len = 0
  prev = None
  for vi in bd_verts:
    v = mesh.vertices[vi]
    if prev is not None:
      seg_len = np.linalg.norm(v - prev)
      total_len += seg_len
      delta.append(total_len)
    prev = v
  delta = [float(d) / float(total_len) for d in delta]

  M = [0] * V
  for vis in mesh.faces:
    vi,vj,vk = [mesh.vertices[idx] for idx in vis]
    vci,vcj,vck = [mesh.visual.vertex_colors[idx][:-1]/255. for idx in vis]
    a = dist_fn(vi,vci, vj,vcj, args.color_weight, args.color_kind)
    b = dist_fn(vj,vcj, vk,vck, args.color_weight, args.color_kind)
    c = dist_fn(vi,vci, vk,vck, args.color_weight, args.color_kind)
    area = herons(a,b,c) / 3
    for vi in vis:
      M[vi] += area
  M = np.array(M)


  EPS = 0
  rows = []
  cols = []
  data = []
  totals = [0] * V
  for [vi0,vi1,vi2] in mesh.faces:
    edges = [
      [vi0,vi1,vi2],
      [vi1,vi2,vi0],
      [vi2,vi0,vi1],
    ]

    for ijk in edges:
      vi,vj,vk = [mesh.vertices[idx] for idx in ijk]
      vci,vcj,vck = [mesh.visual.vertex_colors[idx][:-1]/255. for idx in ijk]
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
        if r in bd_verts: continue
        val /= (2. * M[r] + EPS)
        totals[r] += val
        rows.append(r)
        cols.append(c)
        data.append(val)

  for vi in range(V):
    rows.append(vi)
    cols.append(vi)
    data.append(-totals[vi])
  # add in boundary verts
  for b in bd_verts:
    rows.append(b)
    cols.append(b)
    data.append(1.)

  csr = sp.csr_matrix((data, (rows, cols)), shape=(V, V))

  u = np.array([0.] * V)
  v = np.array([0.] * V)
  for i, b in enumerate(bd_verts):
    t = delta[i] * np.pi * 2
    #t = 2 * np.pi * i / len(bd_verts)
    u[b] = np.cos(t)
    v[b] = np.sin(t)

  new_u = sp.linalg.spsolve(csr, u)
  new_v = sp.linalg.spsolve(csr, v)
  uv = np.stack([new_u, new_v, np.zeros_like(new_u)], axis=-1)
  mesh.vertices[:,0] = new_u
  mesh.vertices[:,1] = new_v
  mesh.vertices[:,2] = 0
  mesh.export(args.output, encoding="ascii")

def luma(rgb):
  return np.sum(np.array([0.299, 0.587, 0.114]) * rgb)

def dist_fn(va,vca, vb, vcb, color_weight=1e-4, kind="add"):
  geom = np.linalg.norm(va - vb)
  if color_weight == 0. and kind != "color-only": return geom
  color = abs(luma(vca) - luma(vcb))
  #color = np.linalg.norm(vca - vcb)
  if kind == "add":
    return geom + color_weight * color
  elif kind == "max":
    return np.maximum(geom, color_weight * color)
  elif kind == "concat":
    return np.linalg.norm(
      np.concatenate([va, color_weight * vca]) - \
      np.concatenate([vb, color_weight * vcb])
    )
  elif kind == "color-only": return color + 1
  else: raise NotImplementedError(kind)

def herons(e0, e1, e2):
  s = (e0 + e1 + e2) / 2
  #return np.sqrt(s * (s - e0) * (s - e1) * (s - e2))
  f = lambda v: np.log(max(v, 1e-12))
  return np.exp(0.5 * (f(s) + f(s - e0) + f(s - e1) + f(s - e2)))

def minmax(a, b): return [min(a,b), max(a,b)]

def boundary(mesh):
    # Set of all edges and of boundary edges (those that appear only once).
    edge_set = set()
    boundary_edges = set()

    # Iterate over all edges, as tuples in the form (i, j) (sorted with i < j to remove ambiguities).
    # For each edge, three cases are possible:
    # 1. The edge has never been visited before. In this case, we can add it to the edge set and as a boundary
    #    candidate as well.
    # 2. The edge had already been visited once. We want to keep it into the set of all edges but remove it from the
    #    boundary set.
    # 3. The edge had already been visited at least twice. This is generally an indication that there is an issue with
    #    the mesh. More precisely, it is not a manifold, and boundaries are not closed-loops.
    for e in map(tuple, mesh.edges_sorted):
        if e not in edge_set:
            edge_set.add(e)
            boundary_edges.add(e)
        elif e in boundary_edges:
            boundary_edges.remove(e)
        else:
            raise RuntimeError(f"The mesh is not a manifold: edge {e} appears more than twice.")

    # Given all boundary vertices, we create a simple dictionary that tells who are their neighbours.
    neighbours = {}
    for v1, v2 in boundary_edges:
        neighbours[v1] = []
        neighbours[v2] = []
    for v1, v2 in boundary_edges:
        neighbours[v1].append(v2)
        neighbours[v2].append(v1)

    # We now look for all boundary paths by "extracting" one loop at a time. After obtaining a path, we remove its edges
    # from the "boundary_edges" set. The algorithm terminates when all edges have been used.
    boundary_paths = []

    while len(boundary_edges) > 0:
        # Given the set of remaining boundary edges, get one of them and use it to start the current boundary path.
        # In the sequel, v_previous and v_current represent the edge that we are currently processing.
        v_previous, v_current = next(iter(boundary_edges))
        boundary_vertices = [v_previous]

        # Keep iterating until we close the current boundary curve (the "next" vertex is the same as the first one).
        while v_current != boundary_vertices[0]:
            # We grow the path by adding the vertex "v_current".
            boundary_vertices.append(v_current)

            # We now check which is the next vertex to visit.
            v1, v2 = neighbours[v_current]
            if v1 != v_previous:
                v_current, v_previous = v1, v_current
            elif v2 != v_previous:
                v_current, v_previous = v2, v_current
            else:
                # This line should be un-reachable. I am keeping it only to detect bugs in case I made a mistake when
                # designing the algorithm.
                raise RuntimeError(f"Next vertices to visit ({v1=}, {v2=}) are both equal to {v_previous=}.")

        # "Convert" the vertices from indices to actual Cartesian coordinates.
        boundary_paths.append(boundary_vertices)

        # Remove all boundary edges that were added to the last path.
        boundary_edges = set(e for e in boundary_edges if e[0] not in boundary_vertices)

    # Return the list of boundary paths.
    return boundary_paths

if __name__ == "__main__": main()
