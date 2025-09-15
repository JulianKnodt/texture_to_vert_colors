#generate a colorbar
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

a = argparse.ArgumentParser()
a.add_argument("--units", required=True, help="Units to display")
a.add_argument("--max", default=1, type=float, help="Max unit")
a.add_argument("--min", default=0, type=float, help="Min unit")
a.add_argument("--vertical", action="store_true")
a.add_argument("--output", default="plots/colorbar.pdf")
args = a.parse_args()

mpl.rcParams.update({'font.size': 24})

fig, ax = plt.subplots(figsize=(3, 12) if args.vertical else (6, 1), layout='constrained')

cmap = mpl.cm.magma
norm = mpl.colors.Normalize(vmin=args.min, vmax=args.max)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax,
             orientation='vertical' if args.vertical else 'horizontal',
             label=args.units)
plt.ylabel(args.units, fontsize=64)
plt.yticks(fontsize=50)

plt.savefig(args.output, bbox_inches="tight")
