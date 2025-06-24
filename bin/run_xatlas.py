import xatlas
import trimesh
from argparse import ArgumentParser
import numpy as np

def arguments():
  a = ArgumentParser()
  a.add_argument("-o", "--output", required=True)
  a.add_argument("-i", "--input", required=True)
  return a.parse_args()

def main():
  args = arguments()
  scene = trimesh.load_scene(args.input)
  atlas = xatlas.Atlas()
  meshes = []
  for g in scene.geometry:
    g = scene.geometry[g]
    meshes.append(g)
    atlas.add_mesh(g.vertices, g.faces)

  # TODO here need to add more parameters
  co = xatlas.ChartOptions()
  co.fix_winding = True
  co.use_input_mesh_uvs = False
  #co.texture_seam_weight = 10.
  #co.max_cost = 200.
  #co.normal_deviation_weight = 0.
  atlas.generate()

  # TODO accumulate all vertices, faces, uvs here
  all_vertices = []
  v_cnt =  0
  all_idxs = []
  all_uvs = []
  for i in range(atlas.mesh_count):
    vmapping, idxs, uvs = atlas[i]
    assert(vmapping.shape[0] == uvs.shape[0])
    idxs += v_cnt

    vertices=meshes[i].vertices[vmapping]
    v_cnt += vmapping.shape[0]

    all_vertices.append(vertices)
    all_idxs.append(idxs)
    all_uvs.append(uvs)


  xatlas.export(
    args.output,
    np.vstack(all_vertices),
    np.vstack(all_idxs),
    np.vstack(all_uvs),
  )

if __name__ == "__main__": main()
