import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scienceplots
import numpy as np
import math
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument(
  "--kind", choices=["exact", "approx", "direct"],
  required=True
)
args = a.parse_args()
v = {
  "exact": "Exact",
  "approx": "Dual",
  "direct": "Direct",
}[args.kind]

base = "outputs"
results = []
for b in os.listdir(base):
  if args.kind not in b: continue
  if ".swp" in b: continue
  if ".git" in b: continue
  if ".json" not in b: continue
  with open(os.path.join(base, b), "r") as f:
    try: data = json.load(f)
    except: continue

  if "remesh_times_ms" not in data: continue
  data["name"] = b
  if "image_resolutions" not in data: continue
  if sum(math.prod(ir) for ir in data['image_resolutions']) < 100:
    print(b)
    exit()
  if "input_num_tris" not in data:
    print("Missing input num tris in", b)
    continue
  results.append(data)

x = [
  sum(math.prod(ir) for ir in r["image_resolutions"]) for r in results
]
faces = [
  r["input_num_tris"] for r in results
]
times = [
  r["before_simplify_tris"] for r in results
]

plt.style.use(['science','ieee', 'no-latex'])

fig, ax = plt.subplots()
sc = ax.scatter(x=x, y=faces, c=times, cmap="viridis")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Total #Pixels in Images")
ax.set_ylabel("#Input Tris")
plt.colorbar(sc, label="Output Tris")
plt.title(f"{v} Remeshing Output Tris")
#ax.set_zlabel("Time (ms)")
#ax.set_xticks(np.arange(0, int(1.1e6), int(2e5)), np.arange(0, int(1.1e6), int(2e5)))
#ax.grid(True)
#ax.ticklabel_format(useOffset=False, style="plain")
fig.tight_layout()
#plt.show()
plt.savefig(f"plots/output_tris_{v.lower()}.pdf")
