"""Structure of the learned CIFAR-10 generators (diagnostic; no training, no encoder).

For every generator checkpoint saved by fins_cifar.py, evaluates the slots on the first --n-probe images
of the subset they were fitted on (the images the chance levels were matched on) and splits the energy
of each field:
  (xi, eta)  translation (per-image spatial mean; "fixed" is its part common to all images), rotation,
             dilation, shear, the non-affine rest, and the image-dependent part (deviation from the
             mean field over images);
  phi        grey offset (brightness), colour offset, per-channel contrast (phi proportional to the
             centred image), the rest, and the image-dependent part.
Shares are energy shares within (xi, eta) or within phi; over slots they are energy-weighted. geo_frac
is the balance ratio of fins_cifar.py, E_geo / (E_geo + E_phi), averaged over slots. shift_px is the RMS
translation at tau = 1 in pixels, offset the RMS grey offset at tau = 1.
null_init (the generator network at initialisation, rescaled slot by slot to the learned RMS, as in
fins_cifar.py) is analysed on the same images, and so are --n-init further initialisations, rescaled the
same way: "pct" is the fraction of them below the learned value, the chance level of each share.

Usage, from the directory fins_cifar.py ran in (./data_cifar):
    python generator_structure.py --ckpt-dir <out_dir>/sym_gen_checkpoints --out-json structure.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neuralop.models import FNO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = re.compile(r"sym_gen_seed(\d+)(?:_train(\d+)_fit(\d+))?\.pt$")
GEO = ("transl", "transl_fixed", "rot", "dil", "shear", "geo_imgdep")
PHI = ("grey", "colour", "contrast", "phi_imgdep")


class MultiSymmetryFNO(nn.Module):
    """As in fins_cifar.py, which runs its experiment on import."""
    def __init__(self, hidden_channels=16, n_modes=8, n_layers=3, N_sym=6):
        super().__init__()
        self.N_sym = N_sym
        self.fno = FNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                       in_channels=5, out_channels=5 * N_sym, n_layers=n_layers,
                       use_channel_mlp=True, channel_mlp_dropout=0.0)

    def forward(self, x_grid, y_grid, img):
        out = self.fno(torch.cat([x_grid, y_grid, img], dim=1))
        return [(out[:, 5 * i:5 * i + 1], out[:, 5 * i + 1:5 * i + 2], out[:, 5 * i + 2:5 * i + 5])
                for i in range(self.N_sym)]


def create_normalized_grids(B, H, W, device):
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    xg, yg = torch.meshgrid(x, y, indexing="xy")
    return xg.view(1, 1, H, W).expand(B, 1, -1, -1), yg.view(1, 1, H, W).expand(B, 1, -1, -1)


def parse_name(path):
    """(seed, training-set size, fit size); the untagged name is the (50 000, 5 000) run."""
    m = CKPT.search(os.path.basename(path))
    seed = int(m.group(1))
    return (seed, 50000, 5000) if m.group(2) is None else (seed, int(m.group(2)), int(m.group(3)))


def probe_images(ds, seed, n_train, n_fit, n_probe):
    """The first n_probe images of the generator's fit subset, drawn as in fins_cifar.py."""
    n_all = len(ds)
    train_idx = (np.random.RandomState(0).permutation(n_all)[:n_train] if n_train < n_all
                 else np.arange(n_all))
    fit = np.random.RandomState(seed).permutation(len(train_idx))[:n_fit]
    return torch.stack([ds[int(i)][0] for i in train_idx[fit[:n_probe]]])


def load_generator(path, n_sym):
    G = MultiSymmetryFNO(N_sym=n_sym).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    state.pop("_metadata", None)
    G.load_state_dict(state)
    return G.eval()


def slot_rms(fields):
    return [(float((xi ** 2 + eta ** 2).mean().sqrt()), float((phi ** 2).mean().sqrt())) for xi, eta, phi in fields]


def init_fields(learned, xg, yg, img, init_seed, n_sym):
    """The network at initialisation (torch seed init_seed), rescaled slot by slot to the learned RMS of
    (xi, eta) and of phi on the same images; init_seed = seed + 1_000_003 is null_init of fins_cifar.py."""
    torch.manual_seed(init_seed)
    base = MultiSymmetryFNO(N_sym=n_sym).to(DEVICE).eval()
    got = base(xg, yg, img)
    out = []
    for (t_geo, t_phi), (g_geo, g_phi), (xi, eta, phi) in zip(slot_rms(learned), slot_rms(got), got):
        s, q = t_geo / max(g_geo, 1e-12), t_phi / max(g_phi, 1e-12)
        out.append((xi * s, eta * s, phi * q))
    return out


def affine_basis(x, y):
    """Vector fields on the grid, orthonormal: two translations, rotation, dilation, two shears."""
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    raw = {"tx": (one, zero), "ty": (zero, one), "rot": (-y, x), "dil": (x, y), "sh1": (x, -y), "sh2": (y, x)}
    out = {}
    for k, (u, v) in raw.items():
        n = (u * u + v * v).sum().sqrt()
        out[k] = (u / n, v / n)
    return out


def slot_energies(xi, eta, phi, img, basis):
    """Energies (sums over images and pixels) of the parts of one slot's fields."""
    xi, eta = xi[:, 0], eta[:, 0]
    B, H, W = xi.shape
    c = {k: (xi * u + eta * v).sum((1, 2)) for k, (u, v) in basis.items()}
    t = torch.stack([xi.mean((1, 2)), eta.mean((1, 2))], 1)
    m = phi.mean((2, 3))
    ic = img - img.mean((2, 3), keepdim=True)
    icn = (ic ** 2).sum((2, 3)).clamp(min=1e-12)
    beta = ((phi - m[:, :, None, None]) * ic).sum((2, 3)) / icn
    return {
        "n": B * H * W, "B": B,
        "geo": float((xi ** 2 + eta ** 2).sum()),
        "transl": float((c["tx"] ** 2 + c["ty"] ** 2).sum()),
        "transl_fixed": float(B * H * W * (t.mean(0) ** 2).sum()),
        "rot": float((c["rot"] ** 2).sum()),
        "dil": float((c["dil"] ** 2).sum()),
        "shear": float((c["sh1"] ** 2 + c["sh2"] ** 2).sum()),
        "geo_imgdep": float(((xi - xi.mean(0)) ** 2 + (eta - eta.mean(0)) ** 2).sum()),
        "shift2": float((t ** 2).sum()),
        "phi": float((phi ** 2).sum()),
        "grey": float(3 * H * W * (m.mean(1) ** 2).sum()),
        "colour": float(H * W * ((m - m.mean(1, keepdim=True)) ** 2).sum()),
        "contrast": float((beta ** 2 * icn).sum()),
        "phi_imgdep": float(((phi - phi.mean(0)) ** 2).sum()),
        "offset2": float((m.mean(1) ** 2).sum()),
    }


def shares(e, W):
    """Shares within (xi, eta) and within phi, and magnitudes at tau = 1, from summed energies."""
    g, p = max(e["geo"], 1e-30), max(e["phi"], 1e-30)
    out = {k: e[k] / g for k in GEO}
    out["affine_rest"] = 1 - (e["transl"] + e["rot"] + e["dil"] + e["shear"]) / g
    out.update({k: e[k] / p for k in PHI})
    out["phi_rest"] = 1 - (e["grey"] + e["colour"] + e["contrast"]) / p
    out["rms_geo"] = (e["geo"] / e["n"]) ** 0.5
    out["rms_phi"] = (e["phi"] / (3 * e["n"])) ** 0.5
    out["shift_px"] = (e["shift2"] / e["B"]) ** 0.5 * (W - 1) / 2
    out["offset"] = (e["offset2"] / e["B"]) ** 0.5
    return out


@torch.no_grad()
def analyse(fields, img):
    """Per-slot shares and their energy-weighted aggregate."""
    B, _, H, W = img.shape
    xg, yg = create_normalized_grids(B, H, W, img.device)
    basis = affine_basis(xg[0, 0], yg[0, 0])
    per = [slot_energies(xi, eta, phi, img, basis) for xi, eta, phi in fields]
    slots = [shares(e, W) for e in per]
    for s in slots:
        s["geo_frac"] = s["rms_geo"] ** 2 / (s["rms_geo"] ** 2 + s["rms_phi"] ** 2 + 1e-8)
    total = shares({k: sum(e[k] for e in per) for k in per[0]}, W)
    total["geo_frac"] = float(np.mean([s["geo_frac"] for s in slots]))
    return {"slots": slots, "all": total}


COLS = [("rms_geo", "rms_g", "{:.3f}"), ("transl", "transl", "{:.3f}"), ("transl_fixed", "fixed", "{:.3f}"),
        ("rot", "rot", "{:.3f}"), ("dil", "dil", "{:.3f}"), ("shear", "shear", "{:.3f}"),
        ("affine_rest", "rest", "{:.3f}"), ("geo_imgdep", "imgdep", "{:.3f}"), ("shift_px", "shift", "{:.2f}"),
        ("rms_phi", "rms_p", "{:.4f}"), ("geo_frac", "g_frac", "{:.3f}"), ("grey", "grey", "{:.3f}"),
        ("colour", "colour", "{:.3f}"), ("contrast", "contr", "{:.3f}"), ("phi_rest", "rest", "{:.3f}"),
        ("phi_imgdep", "imgdep", "{:.3f}"), ("offset", "offset", "{:.4f}")]
HEAD = "".join(f"{h:>8s}" for _, h, _ in COLS[:9]) + "  |" + "".join(f"{h:>8s}" for _, h, _ in COLS[9:])
MATCHED = ("rms_geo", "rms_phi", "geo_frac")   # equal to the learned values by construction


def row(label, s, pct=False):
    vals = ["-" if pct and k in MATCHED else ("{:.2f}" if pct else f).format(s[k]) for k, _, f in COLS]
    return f"{label:<22s}" + "".join(f"{v:>8s}" for v in vals[:9]) + "  |" + "".join(f"{v:>8s}" for v in vals[9:])


def main():
    ap = argparse.ArgumentParser(description="Structure of the learned CIFAR-10 generators")
    ap.add_argument("--ckpt-dir", required=True, help="sym_gen_checkpoints of fins_cifar.py")
    ap.add_argument("--data-root", default="./data_cifar")
    ap.add_argument("--n-probe", type=int, default=256, help="fit-subset images (fins_cifar.py matches nulls on 256)")
    ap.add_argument("--n-sym", type=int, default=6)
    ap.add_argument("--n-init", type=int, default=100, help="initialisations for the chance level of each share")
    ap.add_argument("--no-slots", action="store_true", help="aggregate rows only")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    files = sorted((f for f in glob.glob(os.path.join(args.ckpt_dir, "sym_gen_seed*.pt"))
                    if CKPT.search(os.path.basename(f))), key=lambda f: parse_name(f)[::-1])
    if not files:
        raise SystemExit(f"no generator checkpoints in {args.ckpt_dir}")
    ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
    print(f"Device {DEVICE}; {len(files)} checkpoints; shares of energy within (xi, eta) | within phi")
    print(f"{'':<22s}" + HEAD)

    results, last = [], None
    for f in files:
        seed, n_train, n_fit = parse_name(f)
        if (n_train, n_fit) != last:
            print(f"--- training set {n_train}, generators fitted on {n_fit} ---")
            last = (n_train, n_fit)
        img = probe_images(ds, seed, n_train, n_fit, args.n_probe).to(DEVICE)
        B, _, H, W = img.shape
        xg, yg = create_normalized_grids(B, H, W, DEVICE)
        with torch.no_grad():
            learned = load_generator(f, args.n_sym)(xg, yg, img)
            null = init_fields(learned, xg, yg, img, seed + 1_000_003, args.n_sym)
            ref = [analyse(init_fields(learned, xg, yg, img, 3_000_000 + j, args.n_sym), img)["all"]
                   for j in range(args.n_init)]
        r = {"file": os.path.basename(f), "seed": seed, "n_train": n_train, "n_fit": n_fit, "n_probe": B,
             "learned": analyse(learned, img), "null_init": analyse(null, img)}
        if ref:
            r["init"] = {k: {"pct": float(np.mean([x[k] < r["learned"]["all"][k] for x in ref])),
                             "q05_q50_q95": [float(q) for q in np.quantile([x[k] for x in ref], [0.05, 0.5, 0.95])]}
                         for k in ref[0]}
        results.append(r)
        if not args.no_slots:
            for i, s in enumerate(r["learned"]["slots"]):
                print(row(f"s{seed} slot {i}", s))
        print(row(f"s{seed} learned, all", r["learned"]["all"]))
        print(row(f"s{seed} null_init, all", r["null_init"]["all"]))
        if ref:
            print(row(f"s{seed} pct of {len(ref)} inits", {k: v["pct"] for k, v in r["init"].items()}, pct=True))

    sizes = sorted({(r["n_train"], r["n_fit"]) for r in results}, key=lambda s: s[::-1])
    groups = [[r for r in results if (r["n_train"], r["n_fit"]) == s] for s in sizes]
    print("\nSummary over seeds: learned mean +/- std (null_init mean); columns are training set / fit set")
    print(f"{'':<14s}" + "".join(f"{f'{a}/{b} ({len(g)} seeds)':>28s}" for (a, b), g in zip(sizes, groups)))
    for k in ("rms_geo", "transl", "transl_fixed", "rot", "dil", "shear", "geo_imgdep", "shift_px",
              "rms_phi", "geo_frac", "grey", "colour", "contrast", "phi_imgdep", "offset"):
        line = f"{k:<14s}"
        for g in groups:
            a = [r["learned"]["all"][k] for r in g]
            b = np.mean([r["null_init"]["all"][k] for r in g])
            line += f"{f'{np.mean(a):.4f} +/- {np.std(a):.4f} ({b:.4f})':>28s}"
        print(line)
    for (a, b), g in zip(sizes, groups):
        print(f"{a}/{b}, seeds {' '.join(str(r['seed']) for r in g)}:")
        for k in ("transl", "grey", "contrast"):
            print(f"  {k:<9s} learned/null_init" + (" (pct among inits)" if "init" in g[0] else "") + "  "
                  + "  ".join(f"{r['learned']['all'][k]:.3f}/{r['null_init']['all'][k]:.3f}"
                              + (f" ({r['init'][k]['pct']:.2f})" if "init" in r else "") for r in g))

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
