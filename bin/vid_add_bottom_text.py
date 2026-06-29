import os
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument("-i", "--input", required=True)
a.add_argument("--caption", type=str, required=True)
a.add_argument("-x", type=int, default=1500)
a.add_argument("-y", type=int, default=1030)
a.add_argument("-o", "--output", required=True)
a.add_argument("--font-size", type=int, default=24)
args = a.parse_args()

output = args.output
cap = f"drawtext:font='Times New Roman':\
text=\'{args.caption}\':fontcolor=white:fontsize={args.font_size}:x={args.x}:y={args.y}"

os.system(f'ffmpeg -i {args.input} -vf "{cap}" {output} -y')

