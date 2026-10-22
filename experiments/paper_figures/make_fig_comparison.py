"""Figure: missing dimensions of the finite algebra through learned operators, per method and equation (the values of
the paper's tables tab:main and tab:new-learned, means over seeds 42-44; trained and untrained baselines; random
subspaces of the class where measured).

Run:  python make_fig_comparison.py [--out fig_comparison.pdf]
"""

import argparse

import matplotlib.pyplot as plt

import figstyle as fs

# panel: title, k, random subspaces (Gaussian coefficients, uniform) or None
PANELS = (("Burgers", 5, (2.76, 4.17)), ("KdV", 4, (2.85, 3.46)), ("heat, finite", 6, (2.91, 4.99)),
          ("heat, full class", 10, (6.36, 8.33)), ("Benjamin–Ono", 4, None), ("Kuramoto–Sivashinsky", 3, None))
METHODS = (("FINS", fs.BLUE, "o"), ("LieNLSD", fs.ORANGE, "o"), ("Kreider et al., surface", fs.AQUA, "D"),
           ("Kreider et al., solutions", fs.AQUA, "s"), ("LieGAN", fs.SECOND, "o"), ("LaLiGAN", fs.SECOND, "o"),
           ("Augerino", fs.SECOND, "o"), ("LIG (uses the equation)", fs.SECOND, "^"))
# trained, untrained (None: no untrained version); tab:main and tab:new-learned
VALUES = {
    "Burgers": ((0.44, None), (0.55, 1.07), (1.34, None), (3.40, None), (2.98, 2.54), (4.63, 5.00), (1.96, 1.96),
                (2.11, 2.03)),
    "KdV": ((0.01, None), (0.90, 0.98), (1.68, None), (2.95, None), (2.84, 2.05), (3.05, 4.00), (0.90, 0.90),
            (2.90, 2.64)),
    "heat, finite": ((0.14, None), (0.32, 1.14), (1.61, None), (3.51, None), (4.06, 3.57), (4.10, 6.00),
                     (2.00, 2.00), (3.45, 3.23)),
    "heat, full class": ((1.59, None), (0.43, 2.29), (4.37, None), (6.81, None), (6.66, 6.31), (7.39, 10.00),
                         (6.00, 6.00), (6.89, 6.56)),
    "Benjamin–Ono": ((0.018, None), (1.354, None), (1.75, None), (3.10, None), (2.70, 2.18), (3.86, 4.00),
                     (0.87, 0.87), (None, None)),
    "Kuramoto–Sivashinsky": ((0.000, None), (0.000, None), (0.90, None), (2.08, None), (2.59, 2.05), (2.98, 3.00),
                             (0.93, 0.93), (1.56, 2.06)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fig_comparison.pdf")
    args = ap.parse_args()
    fs.setup()
    fig, axes = plt.subplots(2, 3, figsize=(fs.WIDTH, 3.25), sharey=True)
    ys = list(range(len(METHODS)))[::-1]
    for ax, (title, k, rnd) in zip(axes.flat, PANELS):
        if rnd:
            ax.axvspan(*rnd, color=fs.BAND, lw=0, zorder=0)
        for y, (name, col, mk), (tr, un) in zip(ys, METHODS, VALUES[title]):
            if tr is None:
                ax.text(0.15, y, "not run", color=fs.MUTED, fontsize=5.5, va="center")
                continue
            if un is not None:
                ax.plot([tr, un], [y, y], color=fs.MUTED, lw=0.6, zorder=1)
                ax.plot(un, y, mk, ms=4.0, mfc="white", mec=col, mew=0.8, zorder=2)
            ax.plot(tr, y, mk, ms=4.0, color=col, mec="white", mew=0.4, zorder=3)
        ax.set_xlim(-0.15 * k / 4, k * 1.04)
        ax.set_xticks(range(0, k + 1, 1 if k <= 6 else 2))
        ax.set_title(title, pad=3, color=fs.INK)
        fs.tidy(ax, grid="x")
        ax.tick_params(axis="y", length=0)
    for ax in axes[1]:
        ax.set_xlabel("missing dimensions")
    axes[0, 0].set_yticks(ys)
    axes[0, 0].set_yticklabels([m[0] for m in METHODS])
    h_tr, = axes[0, 0].plot([], [], "o", ms=4.0, color=fs.SECOND, mec="white", mew=0.4)
    h_un, = axes[0, 0].plot([], [], "o", ms=4.0, mfc="white", mec=fs.SECOND, mew=0.8)
    h_rn = plt.Rectangle((0, 0), 1, 1, color=fs.BAND, lw=0)
    fig.legend([h_tr, h_un, h_rn], ["trained", "untrained (chance level)", "random subspaces of the class"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.035), handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.8, w_pad=1.3)
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
