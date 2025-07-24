import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scienceplots
import numpy as np
import math

base = "outputs"
results = []
for b in os.listdir(base):
  if ".swp" in b: continue
  if ".git" in b: continue
  if ".json" not in b: continue
  with open(os.path.join(base, b), "r") as f:
    try: data = json.load(f)
    except: continue

  if "remesh_times_ms" not in data: continue
  data["name"] = b
  if "image_resolutions" not in data: continue
  results.append(data)

x = [
  sum(math.prod(ir) for ir in r["image_resolutions"]) for r in results
]
y = [
  sum(t for t in r["remesh_times_ms"]) for r in results
]

plt.style.use(['science','ieee', 'no-latex'])

fig, ax = plt.subplots()
sns.regplot(x=x, y=y, line_kws={"color": "orange"})
#ax.scatter(x, y)
ax.set_xlabel("Total #Pixels in Images")
ax.set_ylabel("Time (ms)")
#ax.set_xticks(np.arange(0, int(1.1e6), int(2e5)), np.arange(0, int(1.1e6), int(2e5)))
#ax.grid(True)
#ax.ticklabel_format(useOffset=False, style="plain")
fig.tight_layout()
#plt.show()
plt.savefig("plots/remesh_time.pdf")
