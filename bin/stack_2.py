import os
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument("-l", "--left", required=True)
a.add_argument("-r", "--right", required=True)

a.add_argument("--l-label", type=str)
a.add_argument("--r-label", type=str)

a.add_argument("-o", "--output", required=True)
args = a.parse_args()

T = "tmp_stack_2.mp4"
output = args.output
os.system(f"ffmpeg -y -i {args.left} -i {args.right} -filter_complex hstack=inputs=2 {T}")


l_label_vf = ""
if args.l_label is not None:
  l_label_vf = f",\
  drawtext=fontfile=/path/to/font.ttf:text=\'{args.l_label}\':fontcolor=white:fontsize=32:x=210:y=80"

r_label_vf = ""
if args.r_label is not None:
  r_label_vf = f",\
  drawtext=fontfile=/path/to/font.ttf:text=\'{args.r_label}\'\
:fontcolor=white:fontsize=32:x={440+64}:y=80"

os.system(f'ffmpeg -i {T} -vf "pad=width=960:height=540:x=210:y=135:color=black{l_label_vf}{r_label_vf}" \
  {output} -y')

os.remove(T)

