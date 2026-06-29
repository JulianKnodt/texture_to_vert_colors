import os
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument("-i", "--image-match-string", required=True)
a.add_argument("--frame-rate", type=int, default=30)
a.add_argument("-o", "--output", required=True)
args = a.parse_args()

ms = args.image_match_string
fps = args.frame_rate
output = args.output
os.system(
  f"cat {ms} | ffmpeg -f image2pipe -r {fps} -i - -c:v libx264 -pix_fmt yuv420p {output} -y"
)

