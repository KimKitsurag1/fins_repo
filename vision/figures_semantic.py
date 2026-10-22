"""Figure: contrastive same-image specificity of the 24 single-class operators, ordered by superclass, one panel per
network (from the analyse summaries of the confirmatory run)."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUES = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABELS = {"domestic_cat": "domestic cats", "dog": "dogs", "big_cat": "big cats", "bird": "birds",
          "vehicle": "vehicles", "building": "buildings"}


def load(path):
    s = json.load(open(path))
    names = s["groups"]
    S = np.array(s["results"]["order1_contrastive"]["all"]["specificity"])
    ext = [i for i, n in enumerate(names) if ":" in n]
    S = 0.5 * (S + S.T)[np.ix_(ext, ext)]
    return S, [names[i].split(":")[0] for i in ext], s["results"]["confirm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", nargs="+", required=True, help="confirm_<network>_summary.json")
    ap.add_argument("--titles", nargs="+", required=True)
    ap.add_argument("--out", default="fig_semantic_blocks.pdf")
    args = ap.parse_args()
    data = [load(p) for p in args.summaries]
    vals = np.concatenate([S[~np.eye(len(S), dtype=bool)] for S, _, _ in data])
    vmin, vmax = min(0.0, vals.min()), vals.max()
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad("#e4e3df")                                  # diagonal (an operator with itself)
    fig, axes = plt.subplots(1, len(data), figsize=(2.3 * len(data) + 0.6, 2.6), squeeze=False)
    fig.subplots_adjust(wspace=0.06)
    for j, (ax, (S, sc, conf), title) in enumerate(zip(axes[0], data, args.titles)):
        M = S.copy()
        np.fill_diagonal(M, np.nan)
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        order = list(dict.fromkeys(sc))
        bounds = np.cumsum([sc.count(s) for s in order])
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color="white", lw=1.6)
            ax.axvline(b - 0.5, color="white", lw=1.6)
        centres = bounds - np.array([sc.count(s) for s in order]) / 2 - 0.5
        ax.set_xticks(centres)
        ax.set_xticklabels([LABELS[s] for s in order], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(centres)
        ax.set_yticklabels([LABELS[s] for s in order] if j == 0 else [], fontsize=6)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        g4, g5 = conf["G4_specificity"], conf["G5_specificity_readout_regressed_out"]
        ax.set_title(f"{title}: between - within {g4['between_minus_within']:+.2f}\n"
                     f"(head geometry removed {g5['between_minus_within']:+.2f})", fontsize=6.5)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("specificity (missing dimensions of 8)", fontsize=6)
    cb.outline.set_visible(False)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
