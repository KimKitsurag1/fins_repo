"""Figure: time-scaling exponent read from the span against the true exponent alpha of fractional Burgers equations
(exp57), every seed (short gray bars: the true value): the weak form of FINS through the exact solver (seeds 42-51)
and the FNO steppers (42-44), and weak SINDy in the authors' configuration followed by Lie's algorithm (hard and soft
readouts). A span without a
direction of the scaling plane at least half inside it (containment below 0.5) is drawn in the 'no scaling' strip.

Run:  python make_fig_fburgers.py [--results ../results] [--out fig_fburgers.pdf]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

ALPHA = (("fb125", 1.25), ("fb150", 1.5), ("fb175", 1.75), ("fb200", 2.0))
NONE_Y, CONTAIN = 0.95, 0.5
# label in exponent.json, legend, color, marker, filled, x offset
ROWS = {"clean": (("FINS weak", "FINS, weak, exact solver", fs.BLUE, "o", True, -0.06),
                  ("FINS weak, learned operator", "FINS, weak, learned operator", fs.BLUE, "^", False, -0.02),
                  ("WSINDy, evaluation data, soft", "two-stage, soft readout", fs.SECOND, "D", False, 0.02),
                  ("WSINDy, evaluation data, hard (own count)", "two-stage, hard readout", fs.SECOND, "s", True,
                   0.06)),
        "n1e-2": (("FINS weak", "FINS, weak, exact solver", fs.BLUE, "o", True, -0.06),
                  ("WSINDy, evaluation data, soft", "two-stage, soft readout", fs.SECOND, "D", False, 0.02),
                  ("WSINDy, evaluation data, hard (own count)", "two-stage, hard readout", fs.SECOND, "s", True,
                   0.06))}
TITLES = {"clean": "clean data", "n1e-2": "noise 1%"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--out", default="fig_fburgers.pdf")
    args = ap.parse_args()
    fs.setup()
    ex = json.load(open(os.path.join(args.results, "fburgers", "exponent.json")))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDTH, 2.3), sharey=True)
    handles = {}
    for ax, cond in zip(axes, ("clean", "n1e-2")):
        ax.axhspan(0.88, 1.02, color=fs.BAND, lw=0, zorder=0)
        for _, alpha in ALPHA:
            ax.plot([alpha - 0.1, alpha + 0.1], [alpha, alpha], color=fs.MUTED, lw=0.8, zorder=1)
        for label, legend, col, mk, filled, dx in ROWS[cond]:
            for pde, alpha in ALPHA:
                runs = ex[cond].get(pde, {}).get(label, {})
                if not runs:
                    continue
                a = np.array([v[0] if v[1] >= CONTAIN else np.nan for v in runs.values()])
                y = np.where(np.isnan(a), NONE_Y + rng.uniform(-0.03, 0.03, len(a)), a)
                xs = alpha + dx + rng.uniform(-0.008, 0.008, len(a))
                h, = ax.plot(xs, y, mk, ms=3.0, color=col, mfc=col if filled else "white", mec="white" if filled
                             else col, mew=0.3 if filled else 0.6, ls="none", zorder=3, alpha=0.9)
                handles[legend] = h
        ax.set_xlim(1.12, 2.13)
        ax.set_xticks([a for _, a in ALPHA])
        ax.set_ylim(0.85, 2.15 if cond == "n1e-2" else 2.15)
        ax.set_yticks([NONE_Y, 1.25, 1.5, 1.75, 2.0])
        ax.set_yticklabels(["none", "1.25", "1.50", "1.75", "2.00"])
        ax.set_xlabel(r"true exponent $\alpha$")
        ax.set_title(TITLES[cond], pad=3, color=fs.INK)
        fs.tidy(ax, grid="x")
    axes[0].set_ylabel(r"exponent read from the span, $\hat a$")
    order = ("FINS, weak, exact solver", "FINS, weak, learned operator", "two-stage, hard readout",
             "two-stage, soft readout")
    fig.legend([handles[o] for o in order], order, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1),
               handlelength=1.2, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
