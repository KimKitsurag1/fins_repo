"""Figures of the image section.

planted: the planted generator and its projection on the k lowest eigenvectors of the network trained with it
and of the control, as fields on a probe image (arrows xi, eta; background the characteristic).
encoders: rank of each named generator in the spectrum of each encoder (1 = most invariant).

Run:  python figures_image.py planted --json planted_fmnist.json --images fmnist.npz --out fig_planted.pdf
      python figures_image.py planted --json planted_cifar10.json --images ./data_cifar \
          --groups rotation,projective,brightness,saturation,hue --out fig_planted_cifar.pdf
      python figures_image.py encoders --json encoders.json --out fig_encoders.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision.datasets as tvd

from fins_image import make_class, named, solve

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "font.family": "serif"})


def field(c, fields, u, xs, ys):
    """xi, eta [H, W] and phi [C, H, W] = offsets + channel mixings of coefficient vector c; u is [C, H, W]."""
    xi, eta, phi = np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(u)
    for ci, (k, a, b, ch, d) in zip(c, fields):
        if k == "xi":
            xi += ci * xs ** a * ys ** b
        elif k == "eta":
            eta += ci * xs ** a * ys ** b
        elif k == "off":
            phi[ch] += ci
        else:
            phi[ch] += ci * u[d]
    return xi, eta, phi


def characteristic_image(c, fields, u, xs, ys):
    xi, eta, phi = field(c, fields, u, xs, ys)
    C, H, W = u.shape
    uy, ux = np.gradient(u, axis=(1, 2))
    return xi, eta, phi - xi * ux * (W - 1) / 2 - eta * uy * (H - 1) / 2


def show(ax, img, extent=(-1, 1, 1, -1)):
    ax.imshow(img, extent=extent, interpolation="nearest")


def draw(ax, c, fields, u, xs, ys, title, lim, amax, step=3):
    """Background: the characteristic Q = phi - xi u_x - eta u_y on the image -- diverging map for grey images,
    0.5 + Q / (2 lim) as RGB for colour (grey = no change); arrows (xi, eta). lim and amax are shared along a
    row, so a projection shows how much of the generator it keeps."""
    xi, eta, Q = characteristic_image(c, fields, u, xs, ys)
    if len(u) == 1:
        ax.imshow(Q[0], cmap=DIVERGING, vmin=-lim, vmax=lim, extent=(-1, 1, 1, -1), interpolation="nearest")
        ax.imshow(u[0], cmap="Greys", alpha=0.12, extent=(-1, 1, 1, -1), interpolation="nearest")
    else:
        show(ax, np.clip(0.5 + Q.transpose(1, 2, 0) / (2 * lim), 0, 1))
    if amax > 0:
        s = slice(1, None, step)
        ax.quiver(xs[s, s], ys[s, s], xi[s, s] / amax, eta[s, s] / amax, color=INK, angles="xy", scale_units="xy",
                  scale=5.0, width=0.012, headwidth=3.5, headlength=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1, 1); ax.set_ylim(1, -1)
    ax.set_title(title, fontsize=6.5, pad=2)


def probe_image(path, index):
    """[C, H, W] in [0, 1]: Fashion-MNIST npz (Xt), an npz of colour images (X, [N, 3, H, W] uint8), or a
    torchvision CIFAR-10 root (test split)."""
    if path.endswith(".npz"):
        z = np.load(path)
        if "Xt" in z:
            return z["Xt"][index][None].astype(float) / 255
        return z["X"][index].astype(float) / 255
    return np.asarray(tvd.CIFAR10(path, train=False, download=True).data[index]).transpose(2, 0, 1) / 255


def projection(N, D, g, k):
    """Planted generator g (D-normalised) projected on the k lowest eigenvectors; share left outside."""
    lam, C = solve(N, D)
    g = g / np.sqrt(g @ D @ g)
    Ck = C[:, :k]
    p = Ck @ (Ck.T @ (D @ g))
    r = g - p
    return g, p, float(np.sqrt(max(r @ D @ r, 0.0)))


def planted(args):
    d = json.load(open(args.json))
    u = probe_image(args.images, args.image)
    C, H, W = u.shape
    fields = make_class(C)
    gts = named(fields, C)
    ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    groups = args.groups.split(",")
    rows = {r["group"]: r for r in d["results"]
            if (r["arch"], r["output"], r["seed"], r.get("sigma", 0.0)) == (args.arch, args.output, args.seed, args.sigma)
            and r["group"] in groups}
    fig, axes = plt.subplots(len(groups), 3, figsize=(2.9, 1.0 * len(groups)), squeeze=False)
    for i, grp in enumerate(groups):
        pen = rows[grp]["pencils"]
        Nt, Dt = (np.array(a) for a in pen[grp])
        Nc, Dc = (np.array(a) for a in pen["none"])
        gv = gts["grey_offset" if (grp == "brightness" and C == 3) else grp]
        g, pt, st = projection(Nt, Dt, gv, args.k)
        gc, pc, sc = projection(Nc, Dc, gv, args.k)
        xi, eta, Q = characteristic_image(g, fields, u, xs, ys)
        lim = 1.5 * np.abs(Q).max()
        amax = np.sqrt(xi ** 2 + eta ** 2).max()
        draw(axes[i, 0], g, fields, u, xs, ys, "", lim, amax)
        draw(axes[i, 1], pt, fields, u, xs, ys, f"{st:.2f} outside", lim, amax)
        draw(axes[i, 2], pc, fields, u, xs, ys, f"{sc:.2f} outside", lim, amax)
        axes[i, 0].set_ylabel(grp, fontsize=7)
    for j, head in enumerate(("planted", "trained with it", "control")):
        axes[0, j].annotate(head, (0.5, 1.22), xycoords="axes fraction", ha="center", fontsize=7, weight="bold")
    fig.subplots_adjust(wspace=0.06, hspace=0.3, left=0.06, right=0.99, top=0.92, bottom=0.01)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


ENC_LABEL = {"dinov2_vits14": "DINOv2 ViT-S/14", "resnet50_v1": "ResNet-50 V1", "vit_random": "ViT-S/14 untrained"}
GEN_ORDER = [("photometric", ["grey_offset", "intensity", "saturation", "hue", "colour_offset_rg", "colour_offset_by",
                              "colour_cast"]),
             ("geometric", ["tx", "ty", "scaling", "stretch", "rotation", "shear", "projective", "projective_y"])]
GEN_LABEL = {"grey_offset": "grey offset", "colour_offset_rg": "offset R-G", "colour_offset_by": "offset B-Y",
             "colour_cast": "colour cast", "projective_y": "projective (y)"}


def encoders(args):
    d = json.load(open(args.json))
    res = {r["encoder"]: r for r in d["results"] if r.get("sigma", 0.0) == args.sigma}
    names = [n for _, grp in GEN_ORDER for n in grp]
    P = len(d["fields"])
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    y = np.arange(len(names))[::-1]
    for e, (enc, r) in enumerate(res.items()):
        ranks = [r["score_all"]["per"][n]["rank"] for n in names]
        ax.scatter(ranks, y + (1 - e) * 0.22, s=22, marker=MARKERS[e], color=SERIES[e], edgecolor="#fcfcfb",
                   linewidth=0.6, zorder=3, label=ENC_LABEL.get(enc, enc))
    ax.set_yticks(y)
    labels = []
    for n in names:
        mark = ("D" if n in res.get("dinov2_vits14", {}).get("expected", []) else "") + \
               ("R" if n in res.get("resnet50_v1", {}).get("expected", []) else "")
        labels.append(GEN_LABEL.get(n, n) + (f" [{mark}]" if mark else ""))
    ax.set_yticklabels(labels)
    split = len(GEN_ORDER[0][1])
    ax.axhline(y[split - 1] - 0.5, color=GRID, lw=0.6)
    ax.set_xlim(0.5, P + 0.5)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel(f"rank in the spectrum (1 = most invariant, of {P})")
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=7, handletextpad=0.2,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("planted")
    p.add_argument("--json", required=True)
    p.add_argument("--images", required=True, help="Fashion-MNIST npz, colour-image npz or CIFAR-10 root")
    p.add_argument("--arch", default="mlp")
    p.add_argument("--output", default="logp")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--groups", default="rotation,scaling,projective,brightness")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--image", type=int, default=0)
    p.add_argument("--out", default="fig_planted.pdf")
    e = sub.add_parser("encoders")
    e.add_argument("--json", required=True)
    e.add_argument("--sigma", type=float, default=0.0)
    e.add_argument("--out", default="fig_encoders.pdf")
    a = ap.parse_args()
    planted(a) if a.cmd == "planted" else encoders(a)


if __name__ == "__main__":
    main()
