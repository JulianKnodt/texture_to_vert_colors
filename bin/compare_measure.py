import os
import argparse
import json

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--comparison", choices=["uvatlas", "xatlas"], required=True)
  a.add_argument("--our-kind", choices=["planar", "dev"], required=True)
  return a.parse_args()

args = arguments()
our_suffix = f"match_{args.comparison}_{args.our_kind}"

base = "cluster_outputs"
our_results = [
  f
  for f in os.listdir(base) if our_suffix in f and ".json" in f
]

comp = [
  f.replace(our_suffix, args.comparison)
  for f in our_results
]

ours_better = 0
total = 0
for (o, c) in zip(our_results, comp):
  name = o
  o = os.path.join(base, o)
  c = os.path.join(base, c)
  assert(os.path.exists(o))
  assert(os.path.exists(c))
  with open(o, "r") as fo:
    o = json.load(fo)
  with open(c, "r") as fc:
    c = json.load(fc)
  assert(o["num_charts"] == c["num_charts"]), (name, o["num_charts"], c["num_charts"])
  total += 1
  #ours_better += o["max_planarity"] < c["max_planarity"]
  ours_better += o["avg_developability"] < c["avg_developability"]
print(ours_better, total)
