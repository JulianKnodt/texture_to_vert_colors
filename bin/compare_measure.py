import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import numpy as np

plt.style.use(["science", "ieee", "no-latex"])

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--comparison", choices=["uvatlas", "xatlas"], required=True)
  a.add_argument("--our-kind", choices=["planar", "dev"], required=True)
  return a.parse_args()

args = arguments()
our_suffix = f"match_{args.comparison}_{args.our_kind}"

base = "cluster_outputs"
our_results = sorted([
  f for f in os.listdir(base)
  if our_suffix in f and ".json" in f and not ".swp" in f
])

comp = [
  f.replace(our_suffix, args.comparison) for f in our_results
]

keys = [
  "max_developability", "avg_developability",
  "max_planarity", "avg_planarity",
  #"boundary_len",
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
  assert(os.path.exists(c)), c
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
    if o[k] == c[k]: continue
    total[k] += 1
    cmp[k] += o[k] < c[k]

  for k in data_keys:
    if k not in c: continue
    our_data[k][name] = o[k]
    cmp_data[k][name] = c[k]

cmp_key = "XAtlas" if args.comparison == "xatlas" else "UVAtlas"
width = 0.8
our_label = "Ours (Planar)" if args.our_kind == "planar" else "Ours (Devel.)"
use_log = True
for k in keys:
  if "avg" in k: continue
  print(k, cmp[k], total[k])
  plt.clf()
  N = len(cmp_vals[k].keys())
  plt.bar(np.arange(N) * 2 - width/2, list(our_vals[k].values()), log=use_log, alpha=1,
    label="Ours", color="green", width=width)
  plt.bar(np.arange(N) * 2 + width/2, list(cmp_vals[k].values()), log=use_log, alpha=1,
    label=cmp_key, color="red", width=width)
  plt.xticks(
    ticks=np.arange(N) * 2, labels=[k.title().replace("_", " ") for k in
    cmp_vals[k].keys()],rotation=90,
    fontsize=12, minor=False,
  )
  plt.gcf().set_size_inches(12, 4)
  plt.minorticks_off()
  plt.tick_params(top=False, right=False)
  plt.ylabel("Log Radians/Unit Length$^\downarrow$")
  title = k.title()\
    .replace("_", " ")\
    .replace("Planarity", "Planar")\
    .replace("Developability", "Developable")\
    .replace("Max", "Least")
  if "max" in k: title += " Chart"
  plt.title(title)
  frame = plt.legend(frameon=True, framealpha=0.6, edgecolor="white",ncols=2)
  plt.savefig(f"plots/overall_{args.comparison}_{args.our_kind}_{k}.pdf", bbox_inches="tight")

exit()
for k in data_keys:
  for name in our_data[k]:
    plt.clf()
    o_vals = our_data[k][name]
    c_vals = cmp_data[k][name]
    #keys = pd.Series(["ours", "cmps"
    df = pd.DataFrame(data={
      "Ours": o_vals,
      cmp_key: c_vals,
    })
    df.columns = df.columns.astype("category")
    sns.boxenplot(df, palette="pastel")#, flierprops=flierprops)
    plt.savefig(f"plots/{name}_{args.comparison}_{k}.pdf")
