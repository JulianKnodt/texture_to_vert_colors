import open3d as o3d
from argparse import ArgumentParser
import json

def arguments():
  a = ArgumentParser()
  a.add_argument("-i", "--input", help="Input mesh to parameterize with UVAtlas",required=True)
  a.add_argument("-o", "--output", help="Destination path",required=True)
  a.add_argument("--size", type=int, default=1024, help="Size of texture")
  a.add_argument("--stats", default=None, help="Where to store stats of running this")
  a.add_argument("--max-stretch", default=0.3, type=float, help="Stretch to use")
  return a.parse_args()

def main():
  args = arguments()
  mesh = o3d.t.io.read_triangle_mesh(args.input, enable_post_processing=True)
  stretch, num_charts, _partitions = mesh.compute_uvatlas(size=args.size, max_stretch=args.max_stretch)
  o3d.t.io.write_triangle_mesh(args.output, mesh)

  print("Stretch =", stretch)
  print("Num Charts =", num_charts)

  if args.stats is None: return
  data = {}
  if os.path.exists(args.stats):
    with open(args.stats,"r") as f:
      data = json.load(f)

  data["stretch"] = stretch
  data["num_charts"] = num_charts

  with open(args.stats, "w") as f:
    json.dump(data, f, indent=2)

if __name__ == "__main__": main()
