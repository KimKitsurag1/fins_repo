"""FINS on images: Lie point symmetries of a frozen network from a finite generator class.

The operator is a frozen network f (class log-probabilities, or L2-normalised features of an encoder).
A generator v = xi d_x + eta d_y + phi d_u acts on an image u through its characteristic
Q = phi - xi u_x - eta u_y, and f is invariant along v where J_f(u) Q = 0. With a basis Q_1..Q_P of the
class, N_ij = E <J_f Q_i, J_f Q_j> and D_ij = E <Q_i, Q_j>; the generalised eigenproblem N c = lambda D c
orders the class from the most to the least invariant direction. Class: xi, eta polynomials of degree
<= 2 in (x, y) on [-1, 1]^2; phi affine in the colour (offsets and channel mixing).

Modes:
  planted   classifiers trained with augmentation by one known group; the planted generator is the ground
            truth; the same architecture without augmentation and at initialisation are the references;
            LieGG (Moskalev et al., NeurIPS 2022; polarization matrix of github.com/amoskalev/liegg) as
            the baseline
  encoder   pretrained encoders; the continuous part of their documented training augmentations is the
            expected invariant subspace

Usage:
  python fins_image.py planted --data fmnist --arch mlp,cnn --seeds 100,101,102 --out planted_fmnist.json
  python fins_image.py encoder --encoders dinov2_vits14,resnet50_v1,vit_random --out encoders.json
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as tvd
import torchvision.models as tvm
import torchvision.transforms as T
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONO = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
LUMA = (0.299, 0.587, 0.114)


# ---- generator class ----
def make_class(C, degree=2):
    """Fields as (kind, a, b, c_out, c_in): 'xi'/'eta' with monomial x^a y^b; 'off' offset on channel c_out;
    'mix' phi_{c_out} = u_{c_in}."""
    mono = [m for m in MONO if sum(m) <= degree]
    fields = [("xi", a, b, -1, -1) for a, b in mono] + [("eta", a, b, -1, -1) for a, b in mono]
    fields += [("off", 0, 0, c, -1) for c in range(C)]
    fields += [("mix", 0, 0, c, d) for c in range(C) for d in range(C)]
    return fields


def field_name(f):
    k, a, b, c, d = f
    if k in ("xi", "eta"):
        return f"{k}:" + ("1" if a + b == 0 else "x" * a + "y" * b)
    return f"off:{c}" if k == "off" else f"mix:{c}<{d}"


def named(fields, C):
    """Coefficient vectors of named generators in the class."""
    idx = {field_name(f): i for i, f in enumerate(fields)}
    P = len(fields)

    def vec(coef):
        v = np.zeros(P)
        for k, val in coef.items():
            v[idx[k]] = val
        return v

    g = {"tx": vec({"xi:1": 1}), "ty": vec({"eta:1": 1}),
         "rotation": vec({"xi:y": -1, "eta:x": 1}), "scaling": vec({"xi:x": 1, "eta:y": 1}),
         "stretch": vec({"xi:x": 1, "eta:y": -1}), "shear": vec({"xi:y": 1, "eta:x": 1})}
    if all(k in idx for k in ("xi:xx", "eta:xy", "xi:xy", "eta:yy")):
        g["projective"] = vec({"xi:xx": 1, "eta:xy": 1})
        g["projective_y"] = vec({"xi:xy": 1, "eta:yy": 1})
    if C == 1:
        g["brightness"] = vec({"off:0": 1})
        g["intensity"] = vec({"mix:0<0": 1})
        return g
    off = [idx[f"off:{c}"] for c in range(C)]
    mix = np.array([[idx[f"mix:{c}<{d}"] for d in range(C)] for c in range(C)])

    def from_matrix(M):
        v = np.zeros(P)
        for c in range(C):
            for d in range(C):
                v[mix[c, d]] = M[c, d]
        return v

    w = np.array(LUMA)
    one = np.ones(C) / np.sqrt(C)
    K = np.array([[0, -one[2], one[1]], [one[2], 0, -one[0]], [-one[1], one[0], 0]])
    g["grey_offset"] = np.zeros(P); g["grey_offset"][off] = 1
    g["colour_offset_rg"] = np.zeros(P); g["colour_offset_rg"][off] = [1, -1, 0]
    g["colour_offset_by"] = np.zeros(P); g["colour_offset_by"][off] = [-0.5, -0.5, 1]
    g["intensity"] = from_matrix(np.eye(C))
    g["saturation"] = from_matrix(np.eye(C) - np.outer(np.ones(C), w))
    g["hue"] = from_matrix(K)
    g["colour_cast"] = from_matrix(np.diag([1.0, -1.0, 0.0]))
    return g


def derivatives(u):
    """Central differences in the normalised coordinates of [-1, 1]^2 (replicated border)."""
    H, W = u.shape[-2:]
    up = F.pad(u, (1, 1, 1, 1), mode="replicate")
    ux = (up[..., 1:-1, 2:] - up[..., 1:-1, :-2]) * (W - 1) / 4
    uy = (up[..., 2:, 1:-1] - up[..., :-2, 1:-1]) * (H - 1) / 4
    return ux, uy


def characteristic(f, u, ux, uy, xs, ys):
    k, a, b, c, d = f
    if k == "xi":
        return -(xs ** a * ys ** b) * ux
    if k == "eta":
        return -(xs ** a * ys ** b) * uy
    q = torch.zeros_like(u)
    q[:, c] = 1.0 if k == "off" else u[:, d]
    return q


def pencil(op, images, fields, batch=64, sigma=0.0, samples=1, seed=0):
    """N_ij = mean <J Q_i, J Q_j>, D_ij = mean <Q_i, Q_j> over the images (tensor [n, C, H, W] in [0, 1]).
    sigma > 0: J is the Jacobian of the Gaussian-smoothed operator E_e f(u + e), e ~ N(0, sigma^2 I), estimated
    by the mean of J(u + e_m) Q over `samples` fixed draws; Q stays the characteristic of the clean image. The
    criterion then measures invariance at the scale sigma and stays quadratic in the coefficients."""
    P = len(fields)
    N, D = np.zeros((P, P)), np.zeros((P, P))
    H, W = images.shape[-2:]
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=DEVICE), torch.linspace(-1, 1, W, device=DEVICE),
                            indexing="ij")
    for s in range(0, len(images), batch):
        u = images[s:s + batch].to(DEVICE)
        ux, uy = derivatives(u)
        J, Q = [], []
        g = torch.Generator(device=DEVICE).manual_seed(seed + s)
        points = [u] if sigma == 0 else [u + sigma * torch.randn(u.shape, generator=g, device=DEVICE)
                                         for _ in range(samples)]
        for f in fields:
            q = characteristic(f, u, ux, uy, xs, ys)
            acc = 0
            for v in points:
                with sdpa_kernel(SDPBackend.MATH):              # fused attention kernels lack forward-mode AD
                    acc = acc + jvp(op, (v,), (q,))[1].detach()
            J.append((acc / len(points)).flatten(1).double())
            Q.append(q.flatten(1).double())
        J, Q = torch.stack(J), torch.stack(Q)
        N += torch.einsum("pbk,qbk->pq", J, J).cpu().numpy()
        D += torch.einsum("pbn,qbn->pq", Q, Q).cpu().numpy()
    return N / len(images), D / len(images)


def solve(N, D, ridge=1e-9):
    s = np.sqrt(np.diag(D))
    Dn = D / np.outer(s, s) + ridge * np.eye(len(D))
    lam, Y = sla.eigh(N / np.outer(s, s), Dn)
    return lam, Y / s[:, None]                                      # eigenvectors D-orthonormal


def score(N, D, gts, k=None):
    """Per ground-truth generator: rank, loss, share outside the k lowest eigenvectors (D-norm); missing
    dimensions of the ground-truth subspace in the k lowest; chance level of a random k-subspace."""
    lam, Cv = solve(N, D)
    G = np.stack(list(gts.values()), 1)
    kG = np.linalg.matrix_rank(G)
    k = kG if k is None else k
    Ck = Cv[:, :k]
    tol = 1e-9 * np.abs(lam).max()
    per = {}
    for name, g in gts.items():
        loss = float(g @ N @ g / (g @ D @ g))
        r = g - Ck @ (Ck.T @ (D @ g))
        cos2 = (Cv.T @ (D @ g)) ** 2 / (g @ D @ g)
        per[name] = {"rank": int((lam < loss - tol).sum()) + 1, "loss": loss,
                     "share_out": float(np.sqrt(max(0.0, r @ D @ r) / (g @ D @ g))),
                     "best": int(np.argmax(cos2)) + 1, "best_cos2": float(cos2.max())}
    Lg = np.linalg.cholesky(G.T @ D @ G + 1e-15 * np.eye(G.shape[1]))
    Gd = G @ np.linalg.inv(Lg).T                                   # D-orthonormal basis of the subspace
    cos = np.linalg.svd(Gd.T @ D @ Ck, compute_uv=False)
    missing = float(kG - (cos[:kG] ** 2).sum())
    return {"k": int(k), "k_gt": int(kG), "missing": missing, "chance_missing": float(kG * (1 - k / len(lam))),
            "lam": lam.tolist(), "per": per}


def readout(N, D, fields, n=3, top=3):
    lam, Cv = solve(N, D)
    names = [field_name(f) for f in fields]
    out = []
    for j in range(n):
        v = Cv[:, j] / np.abs(Cv[:, j]).max()
        out.append(" ".join(f"{names[i]} {v[i]:+.2f}" for i in np.argsort(-np.abs(v))[:top]))
    return out


# ---- LieGG baseline (github.com/amoskalev/liegg, src/liegg.py: polarization_matrix_2, symmetry_metrics) ----
def liegg(net, images):
    """GL(2) generator of least symmetry variance from the polarization matrix of LieGG: gradient of the sum
    of L2-normalised outputs, forward differences, coordinates in their convention (row, column); colour
    channels summed. Returns the 2x2 generator and its symmetry variance."""
    u = images.to(DEVICE).clone().requires_grad_(True)
    out = F.normalize(net(u), dim=1)
    out.backward(torch.ones_like(out))
    dF = u.grad[..., :-1, :-1]
    d = images.to(DEVICE)
    gx = d[..., 1:, :-1] - d[..., :-1, :-1]
    gy = d[..., :-1, 1:] - d[..., :-1, :-1]
    dI = torch.stack([gx, gy], -1)                                  # [B, C, h, w, 2]
    h = dI.shape[2]
    H = images.shape[-2]
    xy = torch.stack(torch.meshgrid(torch.arange(h, device=DEVICE), torch.arange(dI.shape[3], device=DEVICE),
                                    indexing="ij"), -1) / (H // 2) - 1
    Cm = (dF[..., None, None] * dI[..., None] * xy[None, None, :, :, None, :]).sum((1, 2, 3)).reshape(len(d), -1)
    _, S, Vh = torch.linalg.svd(Cm.detach().double(), full_matrices=False)
    return Vh[-1].reshape(2, 2).cpu().numpy(), float(S[-1] ** 2 / len(d))


def liegg_share(Hm, planted):
    """Share of the planted GL(2) generator outside the LieGG generator (Frobenius); 1 outside GL(2).
    LieGG's (row, column) coordinates: rotation is skew, isotropic scaling the identity."""
    ref = {"rotation": np.array([[0.0, -1.0], [1.0, 0.0]]), "scaling": np.eye(2),
           "stretch": np.array([[-1.0, 0.0], [0.0, 1.0]]), "shear": np.array([[0.0, 1.0], [1.0, 0.0]])}
    if planted not in ref:
        return 1.0
    G = ref[planted]
    c = float(np.sum(G * Hm) / (np.linalg.norm(G) * np.linalg.norm(Hm)))
    return float(np.sqrt(max(0.0, 1 - c * c)))


# ---- planted benchmark ----
class MLP(nn.Module):
    def __init__(self, C, H, W, n=10):
        super().__init__()
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(C * H * W, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                               nn.Linear(512, n))

    def forward(self, x):
        return self.f(x)


class CNN(nn.Module):
    def __init__(self, C, H, W, n=10, w=32):
        super().__init__()
        c = lambda i, o: nn.Conv2d(i, o, 3, padding=1)
        self.f = nn.Sequential(c(C, w), nn.ReLU(), c(w, w), nn.ReLU(), nn.MaxPool2d(2),
                               c(w, 2 * w), nn.ReLU(), c(2 * w, 2 * w), nn.ReLU(), nn.MaxPool2d(2),
                               c(2 * w, 4 * w), nn.ReLU())
        self.fc = nn.Linear(4 * w, n)

    def forward(self, x):
        return self.fc(self.f(x).mean((2, 3)))


class LogProb(nn.Module):
    def __init__(self, net, mode="logp"):
        super().__init__()
        self.net, self.mode = net, mode

    def forward(self, x):
        z = self.net(x)
        return F.log_softmax(z, 1) if self.mode == "logp" else F.normalize(z, dim=1)


RANGE = {"none": 0.0, "rotation": np.pi / 6, "scaling": 0.3, "projective": 0.3, "brightness": 0.3,
         "intensity": 0.5, "saturation": 0.5, "hue": 0.5}
PLANTED_GT = {"rotation": "rotation", "scaling": "scaling", "projective": "projective", "brightness": "brightness",
              "intensity": "intensity", "saturation": "saturation", "hue": "hue"}


def augment(u, kind, s):
    """The planted one-parameter group at parameter s (per image)."""
    if kind == "none":
        return u
    B, C, H, W = u.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=u.device), torch.linspace(-1, 1, W, device=u.device),
                            indexing="ij")
    X, Y, s3 = xs.expand(B, H, W), ys.expand(B, H, W), s.view(B, 1, 1)
    warp = lambda gx, gy: F.grid_sample(u, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="zeros",
                                        align_corners=True)
    s4 = s.view(B, 1, 1, 1)
    if kind == "rotation":
        c, sn = torch.cos(s3), torch.sin(s3)
        return warp(c * X - sn * Y, sn * X + c * Y)
    if kind == "scaling":
        return warp(torch.exp(s3) * X, torch.exp(s3) * Y)
    if kind == "projective":
        d = 1 - s3 * X
        return warp(X / d, Y / d)
    if kind == "brightness":
        return u + s4
    if kind == "intensity":
        return u * torch.exp(s4)
    w = torch.tensor(LUMA, device=u.device).view(1, 3, 1, 1)
    grey = (u * w).sum(1, keepdim=True)
    if kind == "saturation":
        return grey + torch.exp(s4) * (u - grey)
    if kind == "hue":
        one = torch.ones(3, device=u.device) / 3 ** 0.5
        K = torch.tensor([[0, -one[2], one[1]], [one[2], 0, -one[0]], [-one[1], one[0], 0]], device=u.device)
        R = torch.matrix_exp(s.view(B, 1, 1) * K)                  # rotation about the grey axis
        return torch.einsum("bcd,bdhw->bchw", R, u)
    raise ValueError(kind)


def load_data(name, root, fmnist_npz=None):
    if name == "fmnist":
        if fmnist_npz and os.path.exists(fmnist_npz):
            z = np.load(fmnist_npz)
            X, y, Xt, yt = z["X"], z["y"], z["Xt"], z["yt"]
        else:
            tr, te = tvd.FashionMNIST(root, train=True, download=True), tvd.FashionMNIST(root, train=False, download=True)
            X, y, Xt, yt = tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()
        f = lambda a: torch.tensor(a, dtype=torch.float32)[:, None] / 255
        return f(X), torch.tensor(y, dtype=torch.long), f(Xt), torch.tensor(yt, dtype=torch.long)
    tr, te = tvd.CIFAR10(root, train=True, download=True), tvd.CIFAR10(root, train=False, download=True)
    f = lambda a: torch.tensor(a, dtype=torch.float32).permute(0, 3, 1, 2) / 255
    return f(tr.data), torch.tensor(tr.targets), f(te.data), torch.tensor(te.targets)


def train(arch, kind, X, y, seed, epochs, lr=1e-3, bs=128):
    torch.manual_seed(seed)
    C, H, W = X.shape[1:]
    net = (MLP if arch == "mlp" else CNN)(C, H, W).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            u, t = X[idx].to(DEVICE), y[idx].to(DEVICE)
            s = (torch.rand(len(idx), device=DEVICE) * 2 - 1) * RANGE[kind]
            loss = F.cross_entropy(net(augment(u, kind, s)), t)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def accuracy(net, X, y, bs=1000):
    return float(np.mean([(net(X[i:i + bs].to(DEVICE)).argmax(1).cpu() == y[i:i + bs]).float().mean()
                          for i in range(0, len(X), bs)]))


def run_planted(args):
    X, y, Xt, yt = load_data(args.data, args.data_root, args.fmnist_npz)
    X, y = X[:args.n_train], y[:args.n_train]
    probe = Xt[:args.n_probe]
    C = X.shape[1]
    fields = make_class(C, args.degree)
    gts = named(fields, C)
    groups = args.groups.split(",") if args.groups else (
        ["none", "rotation", "scaling", "projective", "brightness", "intensity"] + (["saturation", "hue"] if C == 3 else []))
    results = []
    print(f"{args.data}: {len(X)} training images, {len(probe)} probe images, {len(fields)} fields; device {DEVICE}")
    for arch in args.arch.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            torch.manual_seed(10_000 + seed)
            init = (MLP if arch == "mlp" else CNN)(*X.shape[1:]).to(DEVICE).eval()
            nets = {"init": init}
            for kind in groups:
                t0 = time.time()
                nets[kind] = train(arch, kind, X, y, seed, args.epochs)
                print(f"  trained {arch} {kind} s{seed}: clean accuracy {accuracy(nets[kind], Xt, yt):.3f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            lieggs = {}
            for mode, sigma in [(m, float(sg)) for m in args.outputs.split(",") for sg in args.sigmas.split(",")]:
                pens = {k: pencil(LogProb(n, mode), probe, fields, sigma=sigma, samples=args.samples, seed=seed)
                        for k, n in nets.items()}
                for kind in groups:
                    if kind == "none":
                        continue
                    gname = "grey_offset" if (kind == "brightness" and C == 3) else PLANTED_GT[kind]
                    gt = {gname: gts[gname]}
                    row = {"arch": arch, "seed": seed, "output": mode, "sigma": sigma, "group": kind,
                           "accuracy": accuracy(nets[kind], Xt, yt)}
                    for ref in ("trained", "none", "init"):
                        N, D = pens[kind if ref == "trained" else ref]
                        sc = score(N, D, gt, k=1)
                        sc3 = score(N, D, gt, k=3)
                        p = sc["per"][gname]
                        row[ref] = {"rank": p["rank"], "share_out_1": p["share_out"],
                                    "share_out_3": sc3["per"][gname]["share_out"], "best": p["best"],
                                    "best_cos2": p["best_cos2"], "loss": p["loss"], "lam1": sc["lam"][0],
                                    "readout": readout(N, D, fields, 1)[0]}
                    if gname in ("rotation", "scaling"):
                        idx = [i for i, f in enumerate(fields) if f[0] in ("xi", "eta") and f[1] + f[2] == 1]
                        row["fins_gl2"] = {}
                        for ref in ("trained", "none"):
                            N, D = pens[kind if ref == "trained" else ref]
                            sc = score(N[np.ix_(idx, idx)], D[np.ix_(idx, idx)], {gname: gts[gname][idx]}, k=1)
                            row["fins_gl2"][ref] = sc["per"][gname]["share_out"]
                    if kind not in lieggs:
                        lieggs[kind] = (liegg(nets[kind], probe), liegg(nets["none"], probe)[0])
                    (Hm, sv), Hc = lieggs[kind]
                    row["liegg"] = {"generator": Hm.round(3).tolist(), "variance": sv,
                                    "share_out": liegg_share(Hm, gname)}
                    row["liegg_control_share"] = liegg_share(Hc, gname)
                    row["pencils"] = {k: [pens[k][0].tolist(), pens[k][1].tolist()] for k in (kind, "none", "init")}
                    results.append(row)
                    pl, no = row["trained"], row["none"]
                    print(f"  {arch:<4s} s{seed} {mode:<4s} sigma {sigma:<4g} {kind:<11s} planted net: rank {pl['rank']:>2d}, closest eigenvector "
                          f"{pl['best']} (cos2 {pl['best_cos2']:.2f}), share outside 1/3 lowest {pl['share_out_1']:.2f}/"
                          f"{pl['share_out_3']:.2f} | control: rank {no['rank']:>2d}, {no['share_out_1']:.2f}/"
                          f"{no['share_out_3']:.2f} | init: rank {row['init']['rank']:>2d} | "
                          f"LieGG share out {row['liegg']['share_out']:.2f} (control {row['liegg_control_share']:.2f})"
                          + (f" | FINS in GL(2) {row['fins_gl2']['trained']:.2f} (control {row['fins_gl2']['none']:.2f})"
                             if "fins_gl2" in row else ""), flush=True)
    summarise_planted(results, len(fields))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"fields": [field_name(f) for f in fields], "results": results}, fh)
        print(f"saved {args.out}")


def summarise_planted(results, P):
    print(f"\nSummary: planted generator, median over seeds; chance share outside the lowest 1 / 3 of {P}: "
          f"{np.sqrt(1 - 1 / P):.2f} / {np.sqrt(1 - 3 / P):.2f}")
    keys = sorted({(r["arch"], r["output"], r.get("sigma", 0.0), r["group"]) for r in results})
    for arch, mode, sigma, kind in keys:
        rs = [r for r in results if (r["arch"], r["output"], r.get("sigma", 0.0), r["group"]) == (arch, mode, sigma, kind)]
        f = lambda ref, k: np.median([r[ref][k] for r in rs])
        ranks = " ".join(str(r["trained"]["rank"]) for r in rs)
        best = " ".join(str(r["trained"]["best"]) for r in rs)
        print(f"  {arch:<4s} {mode:<4s} sigma {sigma:<4g} {kind:<11s} ranks {ranks:<9s} closest {best:<7s} share out 1/3 "
              f"{f('trained', 'share_out_1'):.2f}/"
              f"{f('trained', 'share_out_3'):.2f} | control rank {f('none', 'rank'):.0f}, {f('none', 'share_out_1'):.2f}/"
              f"{f('none', 'share_out_3'):.2f} | LieGG {np.median([r['liegg']['share_out'] for r in rs]):.2f} "
              f"(control {np.median([r['liegg_control_share'] for r in rs]):.2f})"
              + (f" | FINS in GL(2) {np.median([r['fins_gl2']['trained'] for r in rs]):.2f} (control "
                 f"{np.median([r['fins_gl2']['none'] for r in rs]):.2f})" if "fins_gl2" in rs[0] else ""))


# ---- encoders ----
EXPECTED = {
    "dinov2_vits14": ["tx", "ty", "scaling", "stretch", "intensity", "saturation", "hue"],
    "resnet50_v1": ["tx", "ty", "scaling", "stretch"],
    "vit_random": [],
}



def load_dinov2(name, **kw):
    """DINOv2 from a local clone (env DINOV2_REPO) or the hub cache, without GitHub API validation."""
    repo = os.environ.get("DINOV2_REPO", "")
    if os.path.isdir(repo):
        return torch.hub.load(repo, name, source="local", **kw)
    return torch.hub.load("facebookresearch/dinov2", name, skip_validation=True, trust_repo=True, **kw)

class Encoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if name == "dinov2_vits14":
            self.m = load_dinov2("dinov2_vits14")
        elif name == "vit_random":
            torch.manual_seed(0)
            self.m = load_dinov2("dinov2_vits14", pretrained=False)
        elif name == "resnet50_v1":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            m.fc = nn.Identity()
            self.m = m
        else:
            raise ValueError(name)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def forward(self, x):
        return F.normalize(self.m((x - self.mean) / self.std), dim=1)


def load_images(name, root, n, size, seed=0):
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    if name == "imagenette":
        ds = tvd.Imagenette(root, split="val", size="160px", download=not os.path.exists(os.path.join(root, "imagenette2-160")),
                            transform=tf)
    else:
        ds = tvd.CIFAR10(root, train=False, download=True, transform=tf)
    idx = np.random.RandomState(seed).permutation(len(ds))[:n]
    return torch.stack([ds[int(i)][0] for i in idx])


def run_encoder(args):
    images = load_images(args.images, args.data_root, args.n_images, args.size)
    fields = make_class(3, args.degree)
    gts = named(fields, 3)
    print(f"{args.images}: {len(images)} images at {args.size} px, {len(fields)} fields; device {DEVICE}")
    results = []
    for name, sigma in [(n, float(sg)) for n in args.encoders.split(",") for sg in args.sigmas.split(",")]:
        t0 = time.time()
        N, D = pencil(Encoder(name).to(DEVICE), images, fields, batch=args.batch, sigma=sigma, samples=args.samples)
        exp = EXPECTED.get(name, [])
        sc = score(N, D, {g: gts[g] for g in exp}) if exp else None
        allg = score(N, D, gts, k=len(exp) if exp else 3)
        lam = allg["lam"]
        print(f"== {name}, sigma {sigma:g} ({time.time() - t0:.0f} s): lowest eigenvalues " + " ".join(f"{v:.2e}" for v in lam[:8])
              + f" ... median {np.median(lam):.2e}")
        if sc:
            print(f"   expected {', '.join(exp)}: missing {sc['missing']:.2f} of {sc['k_gt']} in the {sc['k']} lowest "
                  f"(random subspace {sc['chance_missing']:.2f})")
        print("   rank / closest eigenvector / share outside the k lowest: " + "  ".join(
            f"{g} {p['rank']}/{p['best']}/{p['share_out']:.2f}" for g, p in sorted(allg["per"].items(),
                                                                              key=lambda t: t[1]["rank"])))
        for j, line in enumerate(readout(N, D, fields, 8, 4)):
            print(f"   eig {j}: {line}")
        results.append({"encoder": name, "sigma": sigma, "expected": exp, "score_expected": sc, "score_all": allg,
                        "pencil": [N.tolist(), D.tolist()]})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"fields": [field_name(f) for f in fields], "results": results}, fh)
        print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser(description="FINS on images")
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("planted")
    p.add_argument("--data", default="fmnist", choices=["fmnist", "cifar10"])
    p.add_argument("--data-root", default="./data_img")
    p.add_argument("--fmnist-npz", default=None, help="local Fashion-MNIST arrays X, y, Xt, yt")
    p.add_argument("--arch", default="mlp,cnn")
    p.add_argument("--groups", default=None)
    p.add_argument("--seeds", default="100,101,102")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-probe", type=int, default=500)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--outputs", default="logp,l2")
    p.add_argument("--sigmas", default="0", help="smoothing scales of the operator, comma-separated (0: plain)")
    p.add_argument("--samples", type=int, default=8, help="noise draws per image for sigma > 0")
    p.add_argument("--out", default=None)
    e = sub.add_parser("encoder")
    e.add_argument("--encoders", default="dinov2_vits14,resnet50_v1,vit_random")
    e.add_argument("--images", default="imagenette", choices=["imagenette", "cifar10"])
    e.add_argument("--data-root", default="./data_img")
    e.add_argument("--n-images", type=int, default=256)
    e.add_argument("--size", type=int, default=224)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--degree", type=int, default=2)
    e.add_argument("--sigmas", default="0", help="smoothing scales of the operator, comma-separated (0: plain)")
    e.add_argument("--samples", type=int, default=8, help="noise draws per image for sigma > 0")
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    run_planted(args) if args.mode == "planted" else run_encoder(args)


if __name__ == "__main__":
    main()
