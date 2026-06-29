import trimesh
import argparse
import numpy as np
import scipy.sparse as sp
import potpourri3d as pp3d
import robust_laplacian
from tqdm import tqdm

def arguments():
  a = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  a.add_argument("-i", "--input", required=True, help="Input mesh")
  a.add_argument("-o", "--output", default=None, help="Output CSV file of directions")
  a.add_argument("--stats", default=None, help="Unused")
  return a.parse_args()

def main():
  args = arguments()
  mesh = trimesh.load_mesh(args.input, process=False)
  vc = mesh.visual.vertex_colors.astype(float) / 255.
  np.savetxt(args.output, vc[:, :3], delimiter=',')

if __name__ == "__main__": main()
