import os
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument("-i", "--image", required=True)
a.add_argument("--duration", type=int, default=2)
a.add_argument("--height", type=int, default=None)
a.add_argument("--width", type=int, default=None)
a.add_argument("-o", "--output", required=True)
args = a.parse_args()

W = args.width
H = args.height
if H is not None: assert(W is not None)
if W is not None: assert(H is not None)

img = args.image
dur = args.duration
output = args.output

scale = ""
if args.height is not None:
  scale = f"-vf scale={W}:{H}"
os.system(f"ffmpeg -loop 1 -i {args.image} -c:v libx264 -t {dur} -pix_fmt yuv420p {scale} {output} -y")

