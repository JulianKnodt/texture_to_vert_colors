import os
import matplotlib.pyplot as plt
from matplotlib import colors
import json
import seaborn as sns
import scienceplots
import numpy as np
import math
from argparse import ArgumentParser

a = ArgumentParser()
fig, ax = plt.subplots()
scs = []
names = []
choices = ["exact", "approx"]
all_results = {}
for kind in choices:
  all_results[kind] = {}
  args = a.parse_args()
  base = "outputs"
  for b in os.listdir(base):
    if kind not in b: continue
    if ".swp" in b: continue
    if ".git" in b: continue
    if ".json" not in b: continue
    with open(os.path.join(base, b), "r") as f:
      try: data = json.load(f)
      except: continue

    if "remesh_times_ms" not in data: continue
    if "chozuya" in b: print(b, sum(data["remesh_times_ms"]))
    data["name"] = b
    if "image_resolutions" not in data: continue
    if sum(math.prod(ir) for ir in data['image_resolutions']) < 100:
      print(b)
      exit()
    if "input_num_tris" not in data:
      print("Missing input num tris in", b)
      continue
    all_results[kind][b] = data

min_value = math.inf
max_value = -math.inf
for kind in choices:
  results = all_results[kind]
  max_ts = max(sum(r["remesh_times_ms"]) for r in results.values())
  min_ts = min(sum(r["remesh_times_ms"]) for r in results.values())
  max_value = max(max_ts, max_value)
  min_value = min(min_ts, min_value)

print(min_value, max_value)
norm = colors.Normalize(vmin=min_value, vmax=max_value)

for kind in choices:
  v = {
    "exact": "Exact",
    "approx": "Dual",
    "direct": "Direct",
  }[kind]
  names.append(v)
  marker = {
    "exact": "o",
    "approx": "^",
    "direct": "*",
  }[kind]

  x = [
    sum(math.prod(ir) for ir in r["image_resolutions"]) for r in results
  ]
  faces = [
    r["input_num_tris"] for r in results
  ]
  times = [
    sum(r["remesh_times_ms"]) for r in results
  ]

  plt.style.use(['science','ieee', 'no-latex'])

  sc = ax.scatter(x=x, y=faces, c=times, cmap="magma", marker=marker, norm=norm, alpha=0.5)
  scs.append(sc)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Total #Pixels in Images")
  ax.set_ylabel("#Input Tris")
  plt.title(f"{v} Remeshing Timings")
  #ax.set_zlabel("Time (ms)")
  #ax.set_xticks(np.arange(0, int(1.1e6), int(2e5)), np.arange(0, int(1.1e6), int(2e5)))
  #ax.grid(True)
  #ax.ticklabel_format(useOffset=False, style="plain")
plt.colorbar(sc, label="Time (ms)")
plt.legend(handles=scs, labels=names)
fig.tight_layout()
plt.show()
#plt.savefig(f"plots/remesh_{v.lower()}_time.pdf")
