import trimesh
import argparse
import numpy as np
import scipy.sparse as sp
import potpourri3d as pp3d
from tqdm import tqdm

def arguments():
  a = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  a.add_argument("-i", "--input", required=True, help="Input mesh")
  a.add_argument("-o", "--output", default=None, help="Output CSV file of directions")

  a.add_argument("--thresh", help="Threshold for gradient else it is removed", default=1e-3, type=float)

  a.add_argument("--uniform", action="store_true")
  a.add_argument("--color-kind", choices=["max", "concat", "add", "color-only"], default="add")
  a.add_argument("--color-weight", default=0., type=float, help="How much to weigh color")
  return a.parse_args()

def main():
  args = arguments()
  mesh = trimesh.load_mesh(args.input, process=False)
  mesh.visual.vertex_colors = mesh.visual.vertex_colors.astype(float) / 255.
  V = len(mesh.vertices)

  M = [0] * V
  V_g = np.zeros([V, 3])
  for vis in tqdm(mesh.faces):
    vi,vj,vk = [mesh.vertices[idx] for idx in vis]
    vci,vcj,vck = [mesh.visual.vertex_colors[idx][:-1]/255. for idx in vis]
    a = dist_fn(vi,vci, vj,vcj, args.color_weight, args.color_kind)
    b = dist_fn(vj,vcj, vk,vck, args.color_weight, args.color_kind)
    c = dist_fn(vi,vci, vk,vck, args.color_weight, args.color_kind)
    area = herons(a,b,c)
    third_area = area / 3
    n = np.cross(vj-vi, vk-vi)
    n /= np.linalg.norm(n) + 1e-8
    pn = np.array([
      [vi[0], vi[1], vi[2], 1],
      [vj[0], vj[1], vj[2], 1],
      [vk[0], vk[1], vk[2], 1],
      [n[0], n[1], n[2], 0]
    ])
    attrs = np.array([luma(vci), luma(vcj), luma(vck), 0.])
    grad_offset = np.linalg.lstsq(pn, attrs)[0]
    grad = grad_offset[:3]
    for i in vis:
      M[i] += third_area
      V_g[i] += area * grad

  M = np.array(M)

  # clamp all grads below a certain_value to 0
  mask = np.linalg.norm(V_g, axis=-1) > args.thresh
  if not np.any(mask):
    print("[INFO]: Lower threshold for gradient")

  solver = pp3d.MeshVectorHeatSolver(mesh.vertices, mesh.faces)
  (x_basis, y_basis, _normal) = solver.get_tangent_frames()
  mask = mask \
    & np.all(np.isfinite(x_basis), axis=-1) \
    & np.all(np.isfinite(y_basis), axis=-1)
  if not mask.any():
    print("[INFO]: No valid basis for all elements that passed threshold")
    exit()
  else:
    print("[INFO]: number of elements used is", mask.sum())

  grads = V_g[mask]
  def dot(a,b, dim=-1): return np.sum(a * b,axis=dim)
  grads_tan = np.stack([
    dot(x_basis[mask], grads),
    dot(y_basis[mask], grads),
  ], axis=-1)
  assert(np.isfinite(grads_tan).all())

  # per vertex gradient function
  out = solver.transport_tangent_vectors(np.nonzero(mask)[0], grads_tan)
  out /= np.linalg.norm(out,axis=-1,keepdims=True) + 1e-8
  if args.output is not None:
    np.savetxt(args.output, out, delimiter=',')

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
