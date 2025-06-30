import os
import json

choices = ["_direct", "_approx", "_exact"]

skip = [
  "non_manifold",
  "thin_tri",
  "cube",
  "open_top_box",
  "sphere",
  "cube",
  "basic",
  "constant",
  "tri",
]

data = sorted([f for f in os.listdir("data") if ".obj" in f or ".fbx" in f or ".ply" in f])
for f in data:
  if any(s in f.lower() for s in skip): continue
  num_faces = "\\todo{Missing Num Faces}"
  for c in choices:
    json_file = f"outputs/{f[:-4]}{c}.json"
    if not os.path.exists(json_file): continue
    with open(json_file, "r") as file:
      try:
        data = json.load(file)
        num_faces = data["input_num_faces"]
      except: continue
  mesh_name = f[:-4].title().replace("_", " ")
  print(mesh_name, "&", num_faces, "&", "\\todo{Missing tex size}", "&", "\\todo{Missing Source}",  "\\\\\\hline")
