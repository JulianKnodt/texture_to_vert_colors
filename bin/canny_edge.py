import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument("-i", "--input", required=True)
a.add_argument("-o", "--output", required=True)
a.add_argument("--min", default=100, type=float)
a.add_argument("--max", default=200, type=float)
args = a.parse_args()

img = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,args.min,args.max)

cv.imwrite(args.output, edges)
