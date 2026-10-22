"""Figure: missing dimensions of the finite algebra and of its non-trivial generators against the noise level (rows of
the paper's table tab:noise), means over the seeds each row has, from the summaries of exp51 (heat, KdV), exp52
(Benjamin-Ono) and exp53 (Kuramoto-Sivashinsky).

Run:  python make_fig_noise.py [--results ../results] [--out fig_noise.pdf]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

CONDS = (("clean", "0"), ("n1e-3", "0.1"), ("n1e-2", "1"), ("n3e-2", "3"), ("n5e-2", "5"), ("n1e-1", "10"))
PANELS = (("heat", "noise_weak", "heat", 6, 4), ("KdV", "noise_weak", "kdv", 4, 2),
          ("Benjamin–Ono", "bo", "bo", 4, 2), ("Kuramoto–Sivashinsky", "ks", "ks", 3, 1))
SCORES = (("finite", "finite algebra"), ("non-trivial", "non-trivial generators"))
# label in the summaries, legend, color, line style, marker, filled, z-order (FINS drawn on top)
ROWS = (("Kreider et al., equation surface, soft", "Kreider et al., surface", fs.MUTED, (0, (1, 1.2)), "D", True, 2),
        ("LieNLSD_RHS@eval", "LieNLSD, exact RHS$^\\dagger$", fs.SECOND, "-", "P", True, 2),
        ("two-stage, H 0.05, hard (own count)", "two-stage, $H=0.05$", fs.SECOND, (0, (3, 1.5)), "v", True, 2),
        ("two-stage, H 0.2, hard (own count)", "two-stage, $H=0.2$", fs.SECOND, (0, (5, 1.2, 1, 1.2)), "<", True,
         2),
        ("WSINDy, evaluation data, hard (own count)", "two-stage, reference", fs.AQUA, "-", "^", True, 3),
        ("FINS pointwise", "FINS, pointwise", fs.ORANGE, "-", "s", True, 4),
        ("FINS weak, learned operator", "FINS, weak, learned", fs.BLUE, (0, (3, 1.5)), "o", False, 5),
        ("FINS weak", "FINS, weak", fs.BLUE, "-", "o", True, 6))
ORDER = ("FINS, weak", "FINS, weak, learned", "FINS, pointwise", "two-stage, reference", "two-stage, $H=0.2$",
         "two-stage, $H=0.05$", "LieNLSD, exact RHS$^\\dagger$", "Kreider et al., surface")


def mean_missing(summ, cond, pde, label, score):
    m = summ.get(cond, {}).get(pde, {}).get(label, {}).get(score, {}).get("missing", {})
    return float(np.mean(list(m.values()))) if m else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--out", default="fig_noise.pdf")
    args = ap.parse_args()
    fs.setup()
    summ = {d: json.load(open(os.path.join(args.results, d, "summary.json"))) for d in ("noise_weak", "bo", "ks")}
    fig, axes = plt.subplots(2, 4, figsize=(fs.WIDTH, 3.5))
    x = np.arange(len(CONDS))
    handles = {}
    for r, (score, name) in enumerate(SCORES):
        for c, (title, study, pde, k, kn) in enumerate(PANELS):
            ax, top = axes[r, c], (k if score == "finite" else kn)
            for label, legend, col, ls, mk, filled, z in ROWS:
                y = np.array([mean_missing(summ[study], cd, pde, label, score) for cd, _ in CONDS])
                ok = ~np.isnan(y)
                if not ok.any():
                    continue
                if pde == "bo" and label.startswith("WSINDy"):
                    filled = False                         # spurious generators or the wrong exponent
                h, = ax.plot(x[ok], y[ok], color=col, ls=ls, lw=1.0, marker=mk, ms=3.0,
                             mfc=col if filled else "white", mec="white" if filled else col,
                             mew=0.3 if filled else 0.6, zorder=z, label=legend)
                if r == 0 and c == 0:
                    handles[legend] = h
            ax.set_ylim(-0.04 * top - 0.05, top * 1.04)
            ax.set_yticks(range(0, top + 1))
            ax.set_xticks(x)
            ax.set_xticklabels([lab for _, lab in CONDS])
            ax.set_xlim(-0.4, len(CONDS) - 0.6)
            if r == 0:
                ax.set_title(title, pad=3, color=fs.INK)
            fs.tidy(ax)
        axes[r, 0].set_ylabel(f"missing dimensions,\n{name}")
    for ax in axes[1]:
        ax.set_xlabel("noise, %")
    fig.legend([handles[o] for o in ORDER], ORDER, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06),
               handlelength=2.2, columnspacing=0.9)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.3)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
