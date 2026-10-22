"""Figure: 16-step error of the steppers of exp72 against the constant background of the evaluation trajectories (their
exact Galilean boosts), median over seeds 42-44, per augmentation arm, from results/utility/summary.json.

Run:  python make_fig_augment.py [--results ../results] [--out fig_augment.pdf]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

EQS = (("ks", "Kuramoto–Sivashinsky"), ("kdv", "KdV"), ("bo", "Benjamin–Ono"))
SEEDS = (42, 43, 44)
BG = ((-2.0, "-2.00"), (-1.0, "-1.00"), (0.0, "in"), (1.0, "+1.00"), (2.0, "+2.00"))
# arm, legend, color, line style, marker, filled
ARMS = (("none", "no augmentation", fs.SECOND, "-", "o", True),
        ("shift", "constant without drift", fs.MUTED, (0, (3, 1.5)), "X", True),
        ("fins", "span of FINS", fs.BLUE, "-", "o", True),
        ("oracle", "true algebra", fs.ORANGE, (0, (3, 1.5)), "s", False),
        ("wide", "real backgrounds within $\\pm3$", fs.AQUA, (0, (1, 1.2)), "^", True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--out", default="fig_augment.pdf")
    args = ap.parse_args()
    fs.setup()
    err = json.load(open(os.path.join(args.results, "utility", "summary.json")))["errors"]
    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.15))
    x = np.array([b for b, _ in BG])
    handles = []
    for ax, (pde, title) in zip(axes, EQS):
        ax.axvspan(-0.5, 0.5, color=fs.BAND, lw=0, zorder=0)
        for arm, legend, col, ls, mk, filled in ARMS:
            y = 100 * np.median([[err[f"{pde} {s}"][arm][key][1] for _, key in BG] for s in SEEDS], axis=0)
            h, = ax.plot(x, y, color=col, ls=ls, lw=1.0, marker=mk, ms=3.0, mfc=col if filled else "white",
                         mec="white" if filled else col, mew=0.3 if filled else 0.7, zorder=3, label=legend)
            if ax is axes[0]:
                handles.append(h)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["−2", "−1", "0", "1", "2"])
        ax.set_xlim(-2.3, 2.3)
        ax.set_title(title, pad=3, color=fs.INK)
        ax.set_xlabel("background $b$")
        fs.tidy(ax)
    axes[0].set_ylabel("16-step error, %")
    handles.append(plt.Rectangle((0, 0), 1, 1, color=fs.BAND, lw=0, label="backgrounds of the training data"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=2.0,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
