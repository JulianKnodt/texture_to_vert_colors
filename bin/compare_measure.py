import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--comparison", choices=["uvatlas", "xatlas"], required=True)
  a.add_argument("--our-kind", choices=["planar", "dev"], required=True)
  return a.parse_args()

args = arguments()
our_suffix = f"match_{args.comparison}_{args.our_kind}"

base = "cluster_outputs"
our_results = sorted([
  f for f in os.listdir(base) if our_suffix in f and ".json" in f
])

comp = [
  f.replace(our_suffix, args.comparison) for f in our_results
]

keys = [
  "max_developability", "avg_developability", "median_developability",
  "max_planarity", "avg_planarity", "median_planarity",
]

data_keys = ["planarity", "developability"]

our_vals = {k: {} for k in keys}
cmp_vals = {k: {} for k in keys}

our_data = {k: {} for k in data_keys}
cmp_data = {k: {} for k in data_keys}

cmp = {k: 0 for k in keys}
total = {k: 0 for k in keys}

for (o, c) in zip(our_results, comp):
  base_idx = o.index("_match")
  name = o[:base_idx]
  o = os.path.join(base, o)
  c = os.path.join(base, c)
  assert(os.path.exists(o))
  assert(os.path.exists(c))
  with open(o, "r") as fo:
    o = json.load(fo)
  with open(c, "r") as fc:
    c = json.load(fc)
  if o["num_charts"] != c["num_charts"]:
    print(f"Need to fix up {name}")
    continue

  assert(o["num_charts"] == c["num_charts"]), (name, o["num_charts"], c["num_charts"])
  if o["num_charts"] == 1: continue

  for k in keys:
    our_vals[k][name] = o[k]
    cmp_vals[k][name] = c[k]
    if abs(o[k] - c[k]) < 1e-8: continue
    if o[k] == c[k]: continue
    total[k] += 1
    cmp[k] += o[k] < c[k]

  for k in data_keys:
    if k not in c: continue
    our_data[k][name] = o[k]
    cmp_data[k][name] = c[k]

for k in keys:
  #plt.clf()
  #plt.bar(list(our_vals[k].keys()), list(our_vals[k].values()), log=True, alpha=0.8)
  #plt.bar(list(cmp_vals[k].keys()), list(cmp_vals[k].values()), log=True, alpha=0.8)
  #plt.xticks(rotation=90)
  #plt.title(k)
  #plt.savefig(f"tmp_{k}.pdf", bbox_inches="tight")
  print(k, cmp[k], total[k])

for k in data_keys:
  for name in our_data[k]:
    plt.clf()
    o_vals = our_data[k][name]
    c_vals = cmp_data[k][name]
    #keys = pd.Series(["ours", "cmps"
    cmp_key = "XAtlas" if args.comparison == "xatlas" else "UVAtlas"
    df = pd.DataFrame(data={
      "Ours": o_vals,
      cmp_key: c_vals,
    })
    df.columns = df.columns.astype("category")
    sns.boxenplot(df, palette="pastel")#, flierprops=flierprops)
    plt.savefig(f"plots/{name}_{args.comparison}_{k}.pdf")
