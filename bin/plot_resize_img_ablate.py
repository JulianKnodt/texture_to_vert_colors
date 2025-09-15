import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "no-latex"])

base = "ablations"
o = sorted([
  f for f in os.listdir(base)
  if "incense" in f and ".json" in f
  and "approx" not in f
])

res = []
avg_color_diffs = []
for f in o:
  with open(os.path.join(base, f), "r") as f:
    data = json.load(f)
  res.append(data["image_resolutions"][0][0])
  avg_color_diff = data["avg_color_diff"]
  avg_color_diffs.append(avg_color_diff)

mpl.rcParams.update({'font.size': 20})
plt.scatter(res, avg_color_diffs, marker='o', c=avg_color_diffs, cmap="magma", s=200)
plt.plot(res, avg_color_diffs, alpha=0.5)
plt.xlabel("Image Resolution")
fig = plt.gcf()
fig.set_size_inches(4.4, 10)
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.yticks(rotation=-90)
plt.ylabel("Nearest Point Avg. Color Dist.$^\\downarrow$", rotation=-90, labelpad=24)
plt.savefig("plots/incense_burner_color_diff.pdf", bbox_inches="tight")
