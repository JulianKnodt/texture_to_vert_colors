import os
import matplotlib.pyplot as plt
import json
import scienceplots
plt.rcParams.update({'font.size': 26})

plt.style.use(["science", "ieee", "no-latex"])

base = "ablations"
comps = [f for f in os.listdir(base) if "scroll_constant" in f and ".json" in f]

pos_only = None
concats = {}
adds = {}
maxs = {}

for og_c in comps:
  c = os.path.join(base, og_c)
  with open(c, "r") as f:
    data = json.load(f)
  if "pos_only" in c:
    pos_only = data["avg_color_diff"]
    continue
  try:
    res = float(og_c.split("_")[-1][:5])
  except:
    print("Skipping", og_c)
    continue
  if "concat" in og_c: d = concats
  elif "max" in og_c: d = maxs
  elif "add" in og_c: d = adds
  d[res] = data["avg_color_diff"]

plt.yticks(rotation=90)
plt.hlines(pos_only, 0, 2, label="Geometry Only", color="teal")
for (name, res) in [
  ("$\\sqrt{\\bullet^2 + \\bullet^2}$", concats),
  ("$\\bullet+\\bullet$", adds),
  ("$\\max(\\bullet,\\bullet)$", maxs),
]:
  keys = sorted(list(res.keys()))
  values = [res[k] for k in keys]
  plt.plot(keys, values, label=name, marker="o")
plt.ylim(None, 0.11)
plt.xscale("log")
plt.legend(loc="lower left")
plt.xlabel("$\lambda$ (unitless)")
plt.ylabel("Average Color Difference$^\\downarrow$")
plt.gcf().set_size_inches(6,3/2 * 2.5)
plt.savefig("plots/ablate_tutte_params.pdf", bbox_inches="tight")
#plt.show()
