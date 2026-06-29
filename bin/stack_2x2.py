import os
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument("-tl", "--top-left", required=True)
a.add_argument("-tr", "--top-right", required=True)
a.add_argument("-bl", "--bot-left", required=True)
a.add_argument("-br", "--bot-right", required=True)

a.add_argument("--tl-label", type=str)
a.add_argument("--tr-label", type=str)
a.add_argument("--bl-label", type=str)
a.add_argument("--br-label", type=str)

a.add_argument("-o", "--output", required=True)
args = a.parse_args()

def copy_to_label(og, label, dst):
  if label is None:
    os.system(f"cp {og} {dst}")
    return dst

  os.system(
  f'ffmpeg -i {og} -vf "drawtext=fontfile=/path/to/font.ttf:text=\'{label}\'\
:fontcolor=white:fontsize=64:x=30:y=30" -codec:a copy {dst} -y'
  )
  return dst

tl = copy_to_label(args.top_left, args.tl_label, "tl_tmp.mp4")
tr = copy_to_label(args.top_right, args.tr_label, "tr_tmp.mp4")

bl = copy_to_label(args.bot_left, args.bl_label, "bl_tmp.mp4")
br = copy_to_label(args.bot_right, args.br_label, "br_tmp.mp4")

os.system(f"ffmpeg -y -i {tl} -i {bl} -filter_complex vstack=inputs=2 l.mp4")
os.system(f"ffmpeg -y -i {tr} -i {br} -filter_complex vstack=inputs=2 r.mp4")
os.system(f"ffmpeg -y -i l.mp4 -i r.mp4 -filter_complex hstack=inputs=2 {args.output}")

os.remove("l.mp4")
os.remove("r.mp4")

os.remove("tl_tmp.mp4")
os.remove("tr_tmp.mp4")

os.remove("bl_tmp.mp4")
os.remove("br_tmp.mp4")
