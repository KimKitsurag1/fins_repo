"""Class-conditional symmetries of a frozen classifier.

For a group S of classes the operator is the log-odds f_S(u) = LSE(z_S) - LSE(z_rest) of the logits z; a field
v is a symmetry of S when v[f_S] = J_{f_S} Q_v = 0 on images of S (Olver's criterion; the transport condition
with horizon zero). Fields, linear in their coefficients:
  order 1 (Lie point fields on (x, y, u)): xi, eta in tensor cubic B-splines (geo); xi, eta = luminance L(u)
    times bilinear B-splines (geo_u); local grey offset and local intensity in B-splines (photo_local); chroma
    offsets and traceless channel mixings (photo_global);
  order 2 (generalised symmetries): bilinear B-splines times u_xx, u_xy, u_yy, Gaussian derivatives (order2).
Per image the scalar JVPs of every group's log-odds along every field are stored, so N_S of any image subset
is a sum of outer products; pooled over all images, M = mean Gram of the JVPs of the penultimate features (the
network's own metric) and D = mean Gram of the characteristics (pixel metric). Discovery: N_S c = lambda M c
(relative: change of the class score per unit of representational change) or N_S c = lambda D c (plain).
Order-1 results are the order-2 pencils restricted to the order-1 fields.

  pencils  (GPU)  frozen classifier on Oxford-IIIT Pet (cats, dogs) and Imagenette classes -> npz; --pool fresh:
                  the images the design did not use; --operators extended: single-class operators of six
                  superclasses, compared on the same images
  analyse  (CPU)  spectra, family composition, split-half reliability and class specificity, both orders;
                  same-image comparisons (raw and reliability-corrected); contrastive pencil; confirmatory tests
  flow           finite flows along discovered fields on held-out images; figure
  readout        control: the same comparisons for the log-odds gradients in feature space (no fields)
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as tvd
import torchvision.models as tvm
import torchvision.transforms as T
from scipy.interpolate import BSpline
from scipy.stats import spearmanr
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LUMA = (0.299, 0.587, 0.114)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# ImageNet class groups; Imagenette classes by WordNet id
IMAGENET_GROUPS = {"cat": list(range(281, 286)), "dog": list(range(151, 269)), "springer": [217], "tench": [0],
                   "garbage_truck": [569], "church": [497]}
IMAGENETTE_WNID = {"springer": "n02102040", "tench": "n01440764", "garbage_truck": "n03417042",
                   "church": "n03028079"}
CHROMA = ((1.0, -1.0, 0.0), (-0.5, -0.5, 1.0))
PET_CATS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon", "Persian",
            "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}
# extended operators: single ImageNet classes in six superclasses, compared on the common pool (no images of their own)
SUPERCLASSES = {"domestic_cat": (281, 282, 283, 284), "dog": (207, 235, 162, 254), "big_cat": (289, 291, 292, 293),
                "bird": (9, 11, 15, 22), "vehicle": (407, 555, 779, 817), "building": (483, 663, 668, 698)}


# ---- field class ----
def bspline(K, n):
    """K clamped uniform B-splines of degree min(3, K - 1) on [-1, 1] at n points: [K, n] (a partition of unity
    whose span contains every polynomial of that degree)."""
    p = min(3, K - 1)
    t = np.r_[[-1.0] * (p + 1), np.linspace(-1, 1, K - p + 1)[1:-1], [1.0] * (p + 1)]
    x = np.linspace(-1, 1, n)
    return np.stack([BSpline(t, np.eye(K)[i], p)(x) for i in range(K)])


def tensor_basis(K, H, W):
    By, Bx = bspline(K, H), bspline(K, W)
    return torch.tensor(np.einsum("ih,jw->ijhw", By, Bx).reshape(K * K, H, W), dtype=torch.float32)


def mixing_basis():
    """Traceless 3 x 3 channel mixings: the six off-diagonal units, diag(1, -1, 0), diag(0, 1, -1)."""
    out = []
    for c in range(3):
        for d in range(3):
            if c != d:
                m = np.zeros((3, 3)); m[c, d] = 1.0; out.append(m)
    out += [np.diag([1.0, -1.0, 0.0]), np.diag([0.0, 1.0, -1.0])]
    return out


def gaussian_blur(u, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=u.dtype, device=u.device)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    C = u.shape[1]
    u = F.conv2d(F.pad(u, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(u, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def derivatives(u, sigma2):
    """First derivatives by central differences; second derivatives of the image smoothed at sigma2 pixels;
    all in the normalised coordinates of [-1, 1]^2."""
    H, W = u.shape[-2:]
    hx, hy = (W - 1) / 2, (H - 1) / 2
    p = F.pad(u, (1, 1, 1, 1), mode="replicate")
    ux = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) * hx / 2
    uy = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) * hy / 2
    s = gaussian_blur(u, sigma2) if sigma2 > 0 else u
    q = F.pad(s, (1, 1, 1, 1), mode="replicate")
    uxx = (q[..., 1:-1, 2:] - 2 * s + q[..., 1:-1, :-2]) * hx ** 2
    uyy = (q[..., 2:, 1:-1] - 2 * s + q[..., :-2, 1:-1]) * hy ** 2
    uxy = (q[..., 2:, 2:] - q[..., 2:, :-2] - q[..., :-2, 2:] + q[..., :-2, :-2]) * hx * hy / 4
    return {"ux": ux, "uy": uy, "uxx": uxx, "uxy": uxy, "uyy": uyy}


class FieldClass:
    """Fields as (family, kind, index); characteristics Q_i(u) for images u [B, C, H, W] in [0, 1]."""

    def __init__(self, C, H, W, kg=5, ku=2, kp=3, k2=2, sigma2=1.0):
        self.C, self.H, self.W, self.sigma2 = C, H, W, sigma2
        self.cfg = dict(kg=kg, ku=ku, kp=kp, k2=k2, sigma2=sigma2)
        self.basis = {"geo": tensor_basis(kg, H, W), "geo_u": tensor_basis(ku, H, W),
                      "photo_local": tensor_basis(kp, H, W),
                      "order2": tensor_basis(k2, H, W) if k2 else torch.zeros(0, H, W)}   # k2 = 0: order 1 only
        self.mix = [torch.tensor(m, dtype=torch.float32) for m in mixing_basis()]
        f = [("geo", "xi", k) for k in range(kg * kg)] + [("geo", "eta", k) for k in range(kg * kg)]
        f += [("geo_u", "xi", k) for k in range(ku * ku)] + [("geo_u", "eta", k) for k in range(ku * ku)]
        f += [("photo_local", "grey", k) for k in range(kp * kp)] + [("photo_local", "int", k) for k in range(kp * kp)]
        if C == 3:
            f += [("photo_global", "chroma", m) for m in range(2)] + [("photo_global", "mix", m) for m in range(8)]
        f += [("order2", d, k) for d in ("xx", "xy", "yy") for k in range(k2 * k2)]
        self.fields = f

    def to(self, device):
        self.basis = {k: v.to(device) for k, v in self.basis.items()}
        self.mix = [m.to(device) for m in self.mix]
        return self

    def prepare(self, u):
        d = derivatives(u, self.sigma2)
        if self.C == 3:
            d["L"] = (u * torch.tensor(LUMA, device=u.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        else:
            d["L"] = u
        return d

    def char(self, i, u, d):
        fam, kind, k = self.fields[i]
        if fam == "photo_global":
            if kind == "chroma":
                return torch.tensor(CHROMA[k], device=u.device).view(1, 3, 1, 1).expand_as(u).contiguous()
            return torch.einsum("cd,bdhw->bchw", self.mix[k], u)
        b = self.basis[fam][k]
        if fam == "geo":
            return -b * (d["ux"] if kind == "xi" else d["uy"])
        if fam == "geo_u":
            return -b * d["L"] * (d["ux"] if kind == "xi" else d["uy"])
        if fam == "photo_local":
            return b.expand_as(u).contiguous() if kind == "grey" else b * u
        return b * d["u" + kind]

    def maps(self, c):
        """Coefficient maps of the field with coefficients c (over all fields)."""
        H, W, dev = self.H, self.W, self.basis["geo"].device
        m = {k: torch.zeros(H, W, device=dev) for k in ("xi", "eta", "xiL", "etaL", "grey", "int", "xx", "xy", "yy")}
        m["chroma"] = torch.zeros(self.C, device=dev)
        m["mix"] = torch.zeros(self.C, self.C, device=dev)
        for i, ci in enumerate(c):
            if ci == 0.0:
                continue
            fam, kind, k = self.fields[i]
            ci = float(ci)
            if fam == "geo":
                m[kind] += ci * self.basis[fam][k]
            elif fam == "geo_u":
                m[kind + "L"] += ci * self.basis[fam][k]
            elif fam == "photo_local":
                m[kind] += ci * self.basis[fam][k]
            elif fam == "order2":
                m[kind] += ci * self.basis[fam][k]
            elif kind == "chroma":
                m["chroma"] += ci * torch.tensor(CHROMA[k], device=dev)
            else:
                m["mix"] += ci * self.mix[k]
        return m

    def combine(self, m, u, d):
        """Characteristic Q(u) of the field with coefficient maps m."""
        Q = m["grey"] + m["int"] * u - (m["xi"] + m["xiL"] * d["L"]) * d["ux"] - (m["eta"] + m["etaL"] * d["L"]) * d["uy"]
        Q = Q + m["xx"] * d["uxx"] + m["xy"] * d["uxy"] + m["yy"] * d["uyy"]
        if self.C == 3:
            Q = Q + m["chroma"].view(1, 3, 1, 1) + torch.einsum("cd,bdhw->bchw", m["mix"], u)
        return Q

    def speed(self, c):
        """Upper bounds of the advection speed (pixels per unit s), of the second-order coefficient and of the
        photometric rate of the field c, for choosing integration steps."""
        H, W = self.H, self.W
        xi = torch.zeros(H, W, device=self.basis["geo"].device); eta = torch.zeros_like(xi)
        a2 = torch.zeros_like(xi); ph = torch.zeros_like(xi)
        for i, ci in enumerate(c):
            if ci == 0.0:
                continue
            fam, kind, k = self.fields[i]
            if fam in ("geo", "geo_u"):
                (xi if kind == "xi" else eta).add_(abs(float(ci)) * self.basis[fam][k])
            elif fam == "order2":
                a2.add_(abs(float(ci)) * self.basis[fam][k])
            elif fam == "photo_local":
                ph.add_(abs(float(ci)) * self.basis[fam][k])
            else:
                ph.add_(abs(float(ci)))
        px = float(torch.sqrt(xi ** 2 + eta ** 2).max()) * (max(H, W) - 1) / 2
        return px, float(a2.max()) * ((max(H, W) - 1) / 2) ** 2, float(ph.max())


def to_unit(x):
    """Images stored as uint8 (to save memory) or float, as float in [0, 1] on the device."""
    x = x.to(DEVICE)
    return x.float() / 255 if x.dtype == torch.uint8 else x


# ---- classifier ----
class Classifier(nn.Module):
    """Frozen classifier: images in [0, 1] -> [logits | penultimate features]."""

    def __init__(self, body, head, n_classes, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        super().__init__()
        self.body, self.head, self.n_classes = body, head, n_classes
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        f = self.body((x - self.mean) / self.std)
        return torch.cat([self.head(f), f], 1)


def imagenet_classifier(name):
    if name in ("resnet50_v1", "resnet50_random"):
        torch.manual_seed(0)
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if name == "resnet50_v1" else None)
        head = m.fc
        m.fc = nn.Identity()
    elif name == "vit_b16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        head = m.heads.head
        m.heads = nn.Identity()
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        head = m.classifier[2]                          # features: LayerNorm2d + Flatten, 768
        m.classifier[2] = nn.Identity()
    else:
        raise ValueError(name)
    return Classifier(m, head, 1000, IMAGENET_MEAN, IMAGENET_STD)


# ---- pencils ----
def log_odds_weights(z, masks):
    """Log-odds of each group and the weights w with d(log-odds) = sum_c w_c dz_c: [B, G], [B, G, classes]."""
    f, w = [], []
    for m in masks:
        zs, zr = z.masked_fill(~m, -torch.inf), z.masked_fill(m, -torch.inf)
        f.append(torch.logsumexp(zs, 1) - torch.logsumexp(zr, 1))
        w.append(torch.softmax(zs, 1) - torch.softmax(zr, 1))
    return torch.stack(f, 1), torch.stack(w, 1)


def compute_pencils(model, fc, images, image_group, group_classes, batch=16, log_every=20):
    """images [n, C, H, W] in [0, 1]; image_group [n] index of each image's group; group_classes: class lists.
    Returns G [n, P, groups] JVPs of the log-odds, f0 [n, groups], top1 [n], per image group the sums of the
    feature Gram (M) and the characteristic Gram (D), with counts, and the mean gradient of each group's log-odds
    with respect to the features (readout_mean [groups, features], the readout-geometry control)."""
    P, nG, n = len(fc.fields), len(group_classes), len(images)
    masks = torch.zeros(nG, model.n_classes, dtype=torch.bool)
    for g, cl in enumerate(group_classes):
        masks[g, cl] = True
    masks = masks.to(DEVICE)
    G, f0, top1 = np.zeros((n, P, nG)), np.zeros((n, nG)), np.zeros(n, dtype=int)
    groups = sorted(set(int(g) for g in image_group))
    M = {g: np.zeros((P, P)) for g in groups}
    D = {g: np.zeros((P, P)) for g in groups}
    wsum = torch.zeros(nG, model.n_classes, dtype=torch.float64, device=DEVICE)
    t0 = time.time()
    for b, s in enumerate(range(0, n, batch)):
        u = to_unit(images[s:s + batch])
        d = fc.prepare(u)
        with torch.no_grad():
            z = model(u)[:, :model.n_classes]
        fz, wz = log_odds_weights(z, masks)
        wsum += wz.double().sum(0)
        f0[s:s + batch] = fz.double().cpu().numpy()
        top1[s:s + batch] = z.argmax(1).cpu().numpy()
        Tf, Qf = [], []
        for i in range(P):
            q = fc.char(i, u, d)
            with sdpa_kernel(SDPBackend.MATH):                  # fused attention kernels lack forward-mode AD
                dout = jvp(model, (u,), (q,))[1]
            G[s:s + batch, i] = torch.einsum("bgc,bc->bg", wz, dout[:, :model.n_classes]).double().cpu().numpy()
            Tf.append(dout[:, model.n_classes:].double())
            Qf.append(q.flatten(1))
        Tf, Qf = torch.stack(Tf), torch.stack(Qf)
        gb = np.asarray(image_group[s:s + batch])
        for g in set(gb.tolist()):
            sel = torch.tensor(gb == g, device=DEVICE)
            Ts, Qs = Tf[:, sel], Qf[:, sel].reshape(P, -1)
            M[g] += torch.einsum("pbf,qbf->pq", Ts, Ts).cpu().numpy()
            D[g] += (Qs @ Qs.T).double().cpu().numpy()
        if log_every and (b + 1) % log_every == 0:
            print(f"    {s + len(u)} / {n} images, {time.time() - t0:.0f} s", flush=True)
    counts = {g: int((np.asarray(image_group) == g).sum()) for g in groups}
    readout_mean = ((wsum / n) @ model.head.weight.detach().double()).cpu().numpy()
    return dict(G=G, f0=f0, top1=top1, M=np.stack([M[g] for g in groups]), D=np.stack([D[g] for g in groups]),
                count=np.array([counts[g] for g in groups]), groups_present=np.array(groups), readout_mean=readout_mean)


def save_pencils(path, res, fc, group_names, group_classes, image_group, meta):
    np.savez_compressed(path, **res, image_group=np.asarray(image_group), group_names=np.array(group_names),
                        group_classes=json.dumps(group_classes), fields=json.dumps(fc.fields),
                        field_cfg=json.dumps(fc.cfg), meta=json.dumps(meta))


# ---- data ----
def load_groups(root, n_species, n_imagenette, size, holdout=200, flow=False, seed=0,
                imagenette=("springer", "tench", "garbage_truck", "church"), pool="design"):
    """Oxford-IIIT Pet cats and dogs (species labels, trainval and test pooled) and Imagenette classes (train and
    val pooled), resized and centre-cropped. The first `holdout` Pet images per species of a fixed permutation are
    kept for flows (flow=True returns them); pencils use the next n_species. pool="fresh": every image of these
    sources that the design (same seed, holdout, n_species, n_imagenette) did not use, flows' holdout included."""
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    rng = np.random.RandomState(seed)
    pets = [tvd.OxfordIIITPet(root, split=sp, target_types="category", download=True, transform=tf)
            for sp in ("trainval", "test")]
    items = [(d, i) for d in pets for i in range(len(d))]
    species = np.concatenate([np.array(d._bin_labels) if getattr(d, "_bin_labels", None)
                              else np.array([0 if d.classes[c] in PET_CATS else 1 for c in d._labels])
                              for d in pets])                     # 0 cat, 1 dog
    names, imgs, gidx = [], [], []
    for g, name in enumerate(("cat", "dog")):
        perm = rng.permutation(np.flatnonzero(species == g))
        if pool == "fresh":
            idx = np.sort(perm[holdout + n_species:])
        else:
            idx = perm[:holdout] if flow else perm[holdout:holdout + n_species]
        imgs += [(items[i][0][items[i][1]][0] * 255).round().to(torch.uint8) for i in idx]
        gidx += [g] * len(idx)
        names.append(name)
    if n_imagenette and not flow:
        sets = [tvd.Imagenette(root, split=sp, size="160px", transform=tf,
                               download=(sp == "train" and not os.path.exists(os.path.join(root, "imagenette2-160"))))
                for sp in ("train", "val")]
        items = [(d, i) for d in sets for i in range(len(d))]
        wnid = np.concatenate([np.array([d.wnids[s[1]] for s in d._samples]) for d in sets])
        for name in imagenette:
            perm = rng.permutation(np.flatnonzero(wnid == IMAGENETTE_WNID[name]))
            idx = np.sort(perm[n_imagenette:]) if pool == "fresh" else perm[:n_imagenette]
            imgs += [(items[i][0][items[i][1]][0] * 255).round().to(torch.uint8) for i in idx]
            gidx += [len(names)] * len(idx)
            names.append(name)
    return torch.stack(imgs), np.array(gidx), names


# ---- analysis ----
def solve(N, M, ridge=1e-9):
    s = np.sqrt(np.maximum(np.diag(M), 1e-300))
    Mn = M / np.outer(s, s) + ridge * np.eye(len(M))
    lam, Y = sla.eigh(N / np.outer(s, s), Mn)
    return lam, Y / s[:, None]


def orthonormal(V, M):
    """Columns of V made M-orthonormal (symmetric orthonormalisation)."""
    G = V.T @ M @ V
    w, U = np.linalg.eigh(G)
    return V @ U @ np.diag(1 / np.sqrt(np.maximum(w, 1e-300))) @ U.T


def missing(Va, Vb, M):
    """k - ||Va' M Vb||_F^2 for M-orthonormal k-column bases."""
    return float(Va.shape[1] - ((Va.T @ M @ Vb) ** 2).sum())


def class_pencil(G, idx, I, g):
    X = G[idx][:, I, g]
    return X.T @ X / len(idx)


def lowest(N, M, k):
    lam, V = solve(N, M)
    return lam, orthonormal(V[:, :k], M)


KINDS = {"geometric": ("geo", "geo_u"), "photometric": ("photo_local", "photo_global"), "order2": ("order2",)}


def composition(V, M, fam):
    """Share of an M-orthonormal k-subspace that lies in the span of each kind of field (geometric,
    photometric, second order): (1/k) ||Q_F' M V||_F^2 with Q_F an M-orthonormal basis of the kind's span."""
    out = {}
    for kind, fams in KINDS.items():
        J = [i for i, x in enumerate(fam) if x in fams]
        if not J:
            continue
        E = np.zeros((len(fam), len(J)))
        E[J, range(len(J))] = 1.0
        w, U = np.linalg.eigh(E.T @ M @ E)
        keep = w > 1e-12 * w.max()
        Q = E @ U[:, keep] / np.sqrt(w[keep])
        out[kind] = float(((Q.T @ M @ V) ** 2).sum() / V.shape[1])
    return out


def split_distances(G, image_group, groups, I, Mden, k, splits, m, seed):
    """Within-group distances of half-sample subspaces and between-group distances of halves, per split."""
    rng = np.random.RandomState(seed)
    W = {g: [] for g in groups}
    B = {(a, b): [] for i, a in enumerate(groups) for b in groups[i + 1:]}
    for _ in range(splits):
        half = {}
        for g in groups:
            idx = rng.permutation(np.flatnonzero(image_group == g))[:2 * m]
            half[g] = [lowest(class_pencil(G, idx[:m], I, g), Mden, k)[1],
                       lowest(class_pencil(G, idx[m:], I, g), Mden, k)[1]]
            W[g].append(missing(half[g][0], half[g][1], Mden))
        for (a, b) in B:
            B[a, b].append(0.5 * (missing(half[a][0], half[b][0], Mden) + missing(half[a][1], half[b][1], Mden)))
    return W, B


def compare_halves(D, names, splits, label):
    """D[(a, b)]: per split, distance of operator a on one half of a pool to operator b on the other half (a = b: the
    operator's own split-half distance). Whether the Pet operators are closer to springer's than to the others', by
    the raw distance and by the specificity (distance - mean of the two operators' own split-half distances; removes
    the advantage of an operator that is merely more reliable)."""
    nG = len(names)
    sym = lambda a, b: 0.5 * (np.array(D[(a, b)]) + np.array(D[(b, a)]))
    spec = lambda a, b: sym(a, b) - 0.5 * (np.array(D[(a, a)]) + np.array(D[(b, b)]))
    res = {"matrix": [[float(np.mean(D[(a, b)])) for b in range(nG)] for a in range(nG)],
           "specificity": [[0.0 if a == b else float(spec(a, b).mean()) for b in range(nG)] for a in range(nG)]}
    if "springer" in names:
        sp = names.index("springer")
        for pet in [n for n in ("cat", "dog") if n in names]:
            a = names.index(pet)
            for other in [n for n in ("tench", "garbage_truck", "church") if n in names]:
                o = names.index(other)
                raw, cor = sym(a, o) - sym(a, sp), spec(a, o) - spec(a, sp)
                res[f"{pet}: springer < {other}"] = {"wins": int((raw > 0).sum()), "margin": float(raw.mean())}
                res[f"{pet}: springer < {other} (corrected)"] = {"wins": int((cor > 0).sum()), "margin": float(cor.mean())}
    keys = [key for key in res if key not in ("matrix", "specificity")]
    print(f"  {label}: own split-half " + " ".join(f"{names[a]} {np.mean(D[(a, a)]):.2f}" for a in range(nG)))
    for tag, sel in (("raw", [x for x in keys if not x.endswith("(corrected)")]), ("corrected", [x for x in keys if x.endswith("(corrected)")])):
        if sel:
            print(f"    {tag}: " + "  ".join(f"{x.replace(' (corrected)', '')} {res[x]['wins']}/{splits} ({res[x]['margin']:+.2f})" for x in sel))
    return res


def same_images(G, image_group, names, I, Mden, k, splits, seed, pools=("all", "cat", "dog", "garbage_truck")):
    """Invariance subspaces of every class operator on the same images: operator a on one half of the pool against
    operator b on the other half; comparisons by compare_halves."""
    rng = np.random.RandomState(seed)
    out = {}
    for pool_name in pools:
        if pool_name != "all" and pool_name not in names:
            continue
        pool = np.arange(len(image_group)) if pool_name == "all" else np.flatnonzero(image_group == names.index(pool_name))
        D = {}
        for _ in range(splits):
            p = rng.permutation(pool)
            h1, h2 = p[:len(p) // 2], p[len(p) // 2:]
            V1 = [lowest(class_pencil(G, h1, I, g), Mden, k)[1] for g in range(len(names))]
            V2 = [lowest(class_pencil(G, h2, I, g), Mden, k)[1] for g in range(len(names))]
            for a in range(len(names)):
                for b in range(len(names)):
                    D.setdefault((a, b), []).append(missing(V1[a], V2[b], Mden))
        out[pool_name] = compare_halves(D, names, splits, f"same images ({pool_name}, halves {len(pool) // 2}, {splits} splits)")
    return out


def contrastive(G, image_group, names, I, fam, k, splits, seed, ref=None):
    """Contrastive pencil N_S c = lambda Nbar c, Nbar the mean of the reference groups' log-odds Grams on the same
    images (ref: the groups with images, i.e. the design groups) -- directions along which the class score changes
    less than the average class score. Per group on its own images: spectrum and composition; all images pooled:
    operators on halves, distances in the pooled Nbar metric, comparisons by compare_halves."""
    nG = len(names)
    ref = list(range(nG)) if ref is None else list(ref)
    out = {"groups": {}, "ref": [names[g] for g in ref]}
    for g in range(nG):
        idx = np.flatnonzero(image_group == g)
        if not len(idx):
            continue
        Ns = [class_pencil(G, idx, I, h) for h in range(nG)]
        Nbar = sum(Ns[h] for h in ref) / len(ref)
        lam, V = solve(Ns[g], Nbar)
        comp = composition(orthonormal(V[:, :k], Nbar), Nbar, fam)
        out["groups"][names[g]] = {"lam": lam[:k + 2].tolist(), "median_lam": float(np.median(lam)), "composition": comp}
        print(f"  {names[g]:<14s} own images: lambda_1..{k} " + " ".join(f"{v:.2f}" for v in lam[:k])
              + f", median {np.median(lam):.2f} | composition " + " ".join(f"{f} {s:.2f}" for f, s in comp.items()))
    pool = np.arange(len(image_group))
    Nall = sum(class_pencil(G, pool, I, g) for g in ref) / len(ref)
    rng = np.random.RandomState(seed)
    D = {}
    for _ in range(splits):
        p = rng.permutation(pool)
        V = []
        for h in (p[:len(p) // 2], p[len(p) // 2:]):
            Ns = [class_pencil(G, h, I, g) for g in range(nG)]
            Nbar = sum(Ns[g] for g in ref) / len(ref)
            V.append([orthonormal(solve(Ns[g], Nbar)[1][:, :k], Nall) for g in range(nG)])
        for a in range(nG):
            for b in range(nG):
                D.setdefault((a, b), []).append(missing(V[0][a], V[1][b], Nall))
    out["all"] = compare_halves(D, names, splits, f"contrastive, same images (all, halves {len(pool) // 2}, {splits} splits)")
    return out


def readout_distance(R):
    """1 - cos^2 between the rows of R (mean log-odds gradients with respect to the features)."""
    U = R / np.linalg.norm(R, axis=1, keepdims=True)
    return 1.0 - (U @ U.T) ** 2


def superclass_test(S, labels, n_perm=10000, seed=0):
    """S symmetric over classes, labels their superclasses: mean over between-superclass pairs minus mean over
    within-superclass pairs; one-sided p from random reassignments of the labels (superclass sizes kept)."""
    iu = np.triu_indices(len(labels), 1)
    x, labels = S[iu], np.asarray(labels)

    def stat(lab):
        same = lab[iu[0]] == lab[iu[1]]
        return x[~same].mean() - x[same].mean()
    obs = stat(labels)
    rng = np.random.RandomState(seed)
    null = np.array([stat(rng.permutation(labels)) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs).sum()) / (1 + n_perm))


def confirm_tests(spec, names, readout_mean, design):
    """Pre-registered confirmatory tests (README, G1-G5) from the contrastive same-image specificity: the readout
    control over the design groups (G3) and, for the extended operators, the superclass test on the specificity,
    on the readout distance, and on the specificity with the readout distance regressed out (G4, G5)."""
    spec, Rd = np.array(spec), readout_distance(readout_mean)
    S = 0.5 * (spec + spec.T)
    out = {}
    iu = np.triu_indices(len(design), 1)
    rho = spearmanr(Rd[np.ix_(design, design)][iu], S[np.ix_(design, design)][iu])
    out["G3_spearman_design"] = {"rho": float(rho.correlation), "p": float(rho.pvalue)}
    print(f"  G3: Spearman over the {len(iu[0])} design pairs, readout distance vs contrastive specificity: "
          f"rho {rho.correlation:+.2f} (p {rho.pvalue:.3f}); readout distance: " + "  ".join(
              f"{names[design[a]]}-{names[design[b]]} {Rd[design[a], design[b]]:.3f}" for a, b in zip(*iu)))
    ext = [i for i, n in enumerate(names) if ":" in n]
    if len(ext) < 4:
        return out
    labels = [names[i].split(":")[0] for i in ext]
    Se, Re = S[np.ix_(ext, ext)], Rd[np.ix_(ext, ext)]
    iu = np.triu_indices(len(ext), 1)
    A = np.c_[np.ones(len(iu[0])), Re[iu]]
    resid = np.zeros_like(Se)
    resid[iu] = Se[iu] - A @ np.linalg.lstsq(A, Se[iu], rcond=None)[0]
    for key, mat in (("G4_specificity", Se), ("G5_specificity_readout_regressed_out", resid),
                     ("control_readout_distance", Re)):
        d, p = superclass_test(mat, labels)
        out[key] = {"between_minus_within": d, "p": p}
        print(f"  {key}: between - within {d:+.3f}, permutation p {p:.4f} ({len(ext)} operators, "
              f"{len(set(labels))} superclasses)")
    animal = {"domestic_cat", "dog", "big_cat", "bird"}
    tree = np.array([[0 if a == b else (1 if (a in animal) == (b in animal) else 2) for b in labels] for a in labels])
    r = spearmanr(tree[iu], Se[iu])
    out["exploratory_tree_spearman"] = {"rho": float(r.correlation), "p": float(r.pvalue)}
    print(f"  exploratory: Spearman of the specificity with the tree distance (same superclass 0, both animals or "
          f"both artefacts 1, else 2) rho {r.correlation:+.2f}")
    print("  mean specificity within each superclass: " + "  ".join(
        f"{s} {np.mean([Se[a, b] for a in range(len(ext)) for b in range(a + 1, len(ext)) if labels[a] == labels[b] == s]):+.2f}"
        for s in dict.fromkeys(labels)))
    return out


def analyse(path, k=8, splits=20, seed=0, out=None):
    z = np.load(path, allow_pickle=True)
    G, image_group = z["G"], z["image_group"]
    names = [str(x) for x in z["group_names"]]
    fields = [tuple(f) for f in json.loads(str(z["fields"]))]
    fam_all = [f[0] for f in fields]
    total = z["count"].sum()
    Mpool, Dpool = z["M"].sum(0) / total, z["D"].sum(0) / total
    groups = list(range(len(names)))
    present = [g for g in groups if (image_group == g).sum() > 0]
    print(f"{path}: {len(image_group)} images, groups " + ", ".join(f"{names[g]} {int((image_group == g).sum())}"
                                                                   for g in present) + f", {len(fields)} fields")
    for g in present:
        idx = np.flatnonzero(image_group == g)
        acc = float(np.mean(z["f0"][idx, g] > 0))
        print(f"  {names[g]:<14s} own log-odds median {np.median(z['f0'][idx, g]):+.2f}, positive on {acc:.2f}")
    summary = {"k": k, "splits": splits, "groups": names, "results": {}}
    for order in ((1, 2) if "order2" in fam_all else (1,)):
        I = [i for i, f in enumerate(fields) if order == 2 or f[0] != "order2"]
        fam = [fam_all[i] for i in I]
        P = len(I)
        for metric, Mfull in (("relative", Mpool), ("plain", Dpool)):
            Mden = Mfull[np.ix_(I, I)]
            key = f"order{order}_{metric}"
            res = {"P": P, "chance": k * (1 - k / P), "groups": {}}
            print(f"\n== order {order} ({P} fields), {metric} pencil (N_S c = lambda "
                  f"{'M' if metric == 'relative' else 'D'} c); k = {k}, random k-subspace misses {k * (1 - k / P):.2f}")
            full = {}
            for g in present:
                idx = np.flatnonzero(image_group == g)
                lam, V = lowest(class_pencil(G, idx, I, g), Mden, k)
                full[g] = V
                comp = composition(V, Mden, fam)
                lam_all = solve(class_pencil(G, idx, I, g), Mden)[0]
                res["groups"][names[g]] = {"lam": lam_all[:k + 2].tolist(), "median_lam": float(np.median(lam_all)),
                                           "composition": comp}
                print(f"  {names[g]:<14s} lambda_1..{k} / median: " + " ".join(f"{v / np.median(lam_all):.1e}"
                                                                        for v in lam_all[:k])
                      + " | composition " + " ".join(f"{f} {s:.2f}" for f, s in comp.items()))
            mat = {}
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    mat[f"{names[a]}-{names[b]}"] = missing(full[a], full[b], Mden)
            print("  full-sample distances (missing of k): " + "  ".join(f"{p} {v:.2f}" for p, v in mat.items()))
            res["full_distances"] = mat
            sets = [("pet", [g for g in present if names[g] in ("cat", "dog")]), ("all", present)]
            for label, gs in sets:
                if len(gs) < 2 or (label == "all" and len(gs) == 2):
                    continue
                m = min(int((image_group == g).sum()) // 2 for g in gs)
                W, B = split_distances(G, image_group, gs, I, Mden, k, splits, m, seed)
                res[f"split_{label}"] = {"half": m, "within": {names[g]: W[g] for g in gs},
                                         "between": {f"{names[a]}-{names[b]}": v for (a, b), v in B.items()}}
                print(f"  split halves ({label}, {m} images each, {splits} splits): within " + "  ".join(
                    f"{names[g]} {np.mean(W[g]):.2f}+/-{np.std(W[g]):.2f}" for g in gs))
                for (a, b), v in B.items():
                    spec = np.array(v) - 0.5 * (np.array(W[a]) + np.array(W[b]))
                    print(f"    between {names[a]}-{names[b]} {np.mean(v):.2f}+/-{np.std(v):.2f}; specificity "
                          f"(between - mean within) {spec.mean():+.2f}+/-{spec.std():.2f}, > 0 on "
                          f"{int((spec > 0).sum())}/{splits}")
                if label == "all" and "springer" in names:
                    pair = lambda a, b: np.array(B[(a, b)] if (a, b) in B else B[(b, a)])
                    sp = names.index("springer")
                    hier, hier_c = {}, {}
                    for pet in [g for g in gs if names[g] in ("cat", "dog")]:
                        for other in [g for g in gs if names[g] in ("garbage_truck", "church", "tench")]:
                            d = pair(pet, other) - pair(pet, sp)
                            c = d - 0.5 * (np.array(W[other]) - np.array(W[sp]))   # specificity difference
                            hier[f"{names[pet]}: springer < {names[other]}"] = {"wins": int((d > 0).sum()),
                                                                                "margin": float(d.mean())}
                            hier_c[f"{names[pet]}: springer < {names[other]}"] = {"wins": int((c > 0).sum()),
                                                                                  "margin": float(c.mean())}
                    res["hierarchy"], res["hierarchy_corrected"] = hier, hier_c
                    print("  hierarchy (Pet group closer to springer than to ...): " + "  ".join(
                        f"{k} {v['wins']}/{splits} ({v['margin']:+.2f})" for k, v in hier.items()))
                    print("  hierarchy, specificity (distance - mean own split-half distance): " + "  ".join(
                        f"{k} {v['wins']}/{splits} ({v['margin']:+.2f})" for k, v in hier_c.items()))
            if metric == "relative" and order == 1:
                res["same_images"] = same_images(G, image_group, names, I, Mden, k, max(10, splits // 2), seed)
            summary["results"][key] = res
        print(f"\n== order {order}, contrastive pencil (N_S c = lambda Nbar c), Nbar over "
              + ", ".join(names[g] for g in present))
        con = contrastive(G, image_group, names, I, fam, k, max(10, splits // 2), seed, ref=present)
        summary["results"][f"order{order}_contrastive"] = con
        if order == 1 and "readout_mean" in z.files:
            summary["results"]["confirm"] = confirm_tests(con["all"]["specificity"], names, z["readout_mean"], present)
    if out:
        with open(out, "w") as fh:
            json.dump(summary, fh)
        print(f"saved {out}")
    return summary


# ---- finite flows ----
def integrate(fc, c, u, s):
    """du/ds = Q_c(u) from 0 to s by RK4, steps from the field's speed (<= 0.5 px, diffusion and photometric
    rates bounded per step)."""
    px, a2, ph = fc.speed(c)
    n = int(min(400, max(8, math.ceil(2 * px * abs(s)), math.ceil(4 * a2 * abs(s)), math.ceil(4 * ph * abs(s)))))
    h = s / n
    m = fc.maps(c)
    rhs = lambda v: fc.combine(m, v, fc.prepare(v))
    for _ in range(n):
        k1 = rhs(u); k2 = rhs(u + h / 2 * k1); k3 = rhs(u + h / 2 * k2); k4 = rhs(u + h * k3)
        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def flow_scores(model, fc, c, images, g_mask, s, batch=16):
    """Per image: |change of the group's log-odds| and relative change of the features at s and -s."""
    df, dg = [], []
    for st in range(0, len(images), batch):
        u = to_unit(images[st:st + batch])
        with torch.no_grad():
            o0 = model(u)
            f0 = log_odds_weights(o0[:, :model.n_classes], g_mask)[0][:, 0]
            for sg in (s, -s):
                us = integrate(fc, c, u, sg)
                o1 = model(us)
                f1 = log_odds_weights(o1[:, :model.n_classes], g_mask)[0][:, 0]
                df.append((f1 - f0).abs().cpu().numpy())
                a, b = o0[:, model.n_classes:], o1[:, model.n_classes:]
                dg.append(((b - a).norm(dim=1) / a.norm(dim=1)).cpu().numpy())
    return np.concatenate(df), np.concatenate(dg)


def flow_directions(z, group, order, k, n_random, seed=0):
    """Lowest k and highest relative eigenvectors of a group and random directions of the class (Jacobi-scaled
    Gaussian), as coefficient vectors over the fields of the given order; with the pooled metric M."""
    names = [str(x) for x in z["group_names"]]
    fields = [tuple(f) for f in json.loads(str(z["fields"]))]
    Mpool, Dpool = z["M"].sum(0) / z["count"].sum(), z["D"].sum(0) / z["count"].sum()
    g = names.index(group)
    I = [i for i, f in enumerate(fields) if order == 2 or f[0] != "order2"]
    Mden, Dden = Mpool[np.ix_(I, I)], Dpool[np.ix_(I, I)]
    lam, V = solve(class_pencil(z["G"], np.flatnonzero(z["image_group"] == g), I, g), Mden)
    rng = np.random.RandomState(seed)
    dirs = [(f"lowest {j + 1}", V[:, j]) for j in range(k)] + [("highest", V[:, -1])]
    dirs += [(f"random {r + 1}", rng.standard_normal(len(I)) / np.sqrt(np.diag(Dden))) for r in range(n_random)]
    return dirs, I, Dden, len(fields)


def image_norm(images, batch=64):
    """Root mean square norm of the images."""
    return float(np.sqrt(np.mean(np.concatenate([(to_unit(images[st:st + batch]) ** 2).flatten(1).sum(1).cpu().numpy()
                                                  for st in range(0, len(images), batch)]))))


def flow_test(model, fc, dirs, I, Dden, P, images, mask, tau, batch=16):
    """For each direction, scaled so that its first-order change of the image is tau times the image norm
    (characteristics in the pooled pixel metric D): median |change of the group's log-odds| and median relative
    change of the features over s = +-1, and their ratio."""
    unorm = image_norm(images)
    rows = []
    for label, v in dirs:
        c = np.zeros(P)
        c[I] = v * (tau * unorm / np.sqrt(v @ Dden @ v))
        df, dg = flow_scores(model, fc, c, images, mask, 1.0, batch)
        rows.append({"direction": label, "median_dlogodds": float(np.median(df)), "median_dfeat": float(np.median(dg)),
                     "ratio": float(np.median(df) / np.median(dg))})
        print(f"  {label:<10s} median |d log-odds| {np.median(df):.3f}, median relative feature change "
              f"{np.median(dg):.3f}, ratio {rows[-1]['ratio']:.3f}", flush=True)
    low = [r["ratio"] for r in rows if r["direction"].startswith("lowest")]
    rand = [r["ratio"] for r in rows if r["direction"].startswith("random")]
    if low and rand:
        print(f"  lowest: ratio median {np.median(low):.3f} (max {max(low):.3f}); random: median {np.median(rand):.3f}; "
              f"lowest / random {np.median(low) / np.median(rand):.2f}")
    return rows, unorm


def flow_figure(model, fc, dirs, I, Dden, P, images, tau, unorm, path, n_img=3):
    show = [d for d in dirs if d[0].startswith("lowest")][:3] + [d for d in dirs if d[0] == "random 1"]
    n_img = min(n_img, len(images))
    fig, axes = plt.subplots(len(show), 3 * n_img, figsize=(1.1 * 3 * n_img, 1.15 * len(show)), squeeze=False)
    u = to_unit(images[:n_img])
    for r, (label, v) in enumerate(show):
        c = np.zeros(P)
        c[I] = v * (tau * unorm / np.sqrt(v @ Dden @ v))
        with torch.no_grad():
            frames = [integrate(fc, c, u, -1.0), u, integrate(fc, c, u, 1.0)]
        for j in range(n_img):
            for t, fr in enumerate(frames):
                ax = axes[r, 3 * j + t]
                ax.imshow(fr[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(("-s", "0", "+s")[t], fontsize=7)
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"saved {path}")


# ---- finite flows of the class-specific directions: transfer between classes ----
def contrastive_directions(z, sources, k, n_random, seed=0):
    """The k lowest contrastive eigenvectors of each source operator on the whole pool (Nbar over the groups with
    images) and n_random Gaussian directions scaled by the pixel metric, as coefficient vectors over all fields;
    with the pooled pixel metric."""
    G, gi = z["G"], z["image_group"]
    names = [str(x) for x in z["group_names"]]
    P = G.shape[1]
    I = list(range(P))
    ref = [g for g in range(len(names)) if (gi == g).sum() > 0]
    pool = np.arange(len(gi))
    Ns = {g: class_pencil(G, pool, I, g) for g in set(ref) | set(sources)}
    Nbar = sum(Ns[g] for g in ref) / len(ref)
    Dpool = z["D"].sum(0) / z["count"].sum()
    dirs = []
    for a in sources:
        V = solve(Ns[a], Nbar)[1]
        dirs += [(names[a], j, V[:, j]) for j in range(k)]
    rng = np.random.RandomState(seed)
    dirs += [("random", r, rng.standard_normal(P) / np.sqrt(np.diag(Dpool))) for r in range(n_random)]
    return dirs, Dpool, names, ref


def flow_changes(model, fc, c, images, masks, batch=16):
    """Per image and operator, along the flow of c at s = +-1: mean |change of the log-odds| and mean squared
    change, [n, operators] each."""
    out_a, out_q = [], []
    for st in range(0, len(images), batch):
        u = to_unit(images[st:st + batch])
        with torch.no_grad():
            f0 = log_odds_weights(model(u)[:, :model.n_classes], masks)[0]
            a, q = torch.zeros_like(f0), torch.zeros_like(f0)
            for sg in (1.0, -1.0):
                df = log_odds_weights(model(integrate(fc, c, u, sg))[:, :model.n_classes], masks)[0] - f0
                a += 0.5 * df.abs()
                q += 0.5 * df ** 2
        out_a.append(a.double().cpu().numpy())
        out_q.append(q.double().cpu().numpy())
    return np.concatenate(out_a), np.concatenate(out_q)


def transfer_tests(path, n_perm=10000):
    """Pre-registered W1, W2 (README) from the saved flows. T[A, B] = mean squared change of B's log-odds along A's
    lowest directions (mean over directions and images) / the same along the random directions. W1: T[A, A] / mean
    of T[A, g] over the design groups; W2: superclass test on (log T[A, B] + log T[B, A]) / 2."""
    t = np.load(path, allow_pickle=True)
    Q, A_, src = t["sq"], t["changes"], [str(x) for x in t["source"]]
    names, ref = [str(x) for x in t["names"]], [int(x) for x in t["ref"]]
    ext = [i for i, n in enumerate(names) if ":" in n]
    labels = [names[i].split(":")[0] for i in ext]
    rnd = [i for i, s in enumerate(src) if s == "random"]
    own = {a: [i for i, s in enumerate(src) if s == names[a]] for a in ext}
    out = {"operators": names, "sources": [names[a] for a in ext]}
    for key, M, red in (("meansq", Q.mean(1), np.mean), ("median_abs", np.median(A_, axis=1), np.median)):
        base = red(M[rnd], axis=0)
        T = np.array([red(M[own[a]], axis=0) / base for a in ext])
        ratio = np.array([T[ia, a] / T[ia, ref].mean() for ia, a in enumerate(ext)])
        S = 0.5 * (np.log(T[:, ext]) + np.log(T[:, ext]).T)
        d, p = superclass_test(S, labels, n_perm)
        out[key] = {"T": T.tolist(), "W1_ratio": ratio.tolist(), "W1_count_below_1": int((ratio < 1).sum()),
                    "W1_median": float(np.median(ratio)), "W2_between_minus_within": d, "W2_p": p}
        tag = "primary" if key == "meansq" else "exploratory"
        print(f"  [{key}, {tag}] W1: own / design groups along the own lowest directions, below 1 on "
              f"{out[key]['W1_count_below_1']}/{len(ext)}, median {out[key]['W1_median']:.2f}; W2: log transfer, "
              f"between - within superclasses {d:+.3f}, permutation p {p:.4f}")
        if key == "meansq":
            print("    W1 per class: " + " ".join(f"{names[a].split(':')[1]} {r:.2f}" for a, r in zip(ext, ratio)))
            print("    mean log transfer within each superclass / to the others: " + "  ".join(
                f"{s} {np.mean([S[a, b] for a in range(len(ext)) for b in range(len(ext)) if a != b and labels[a] == labels[b] == s]):+.2f}/"
                f"{np.mean([S[a, b] for a in range(len(ext)) for b in range(len(ext)) if labels[a] == s != labels[b]]):+.2f}"
                for s in dict.fromkeys(labels)))
    print(f"{t['network']}: tau {float(t['tau'])}, {Q.shape[1]} test images, {Q.shape[0]} directions")
    return out


def transfer_figure(model, fc, dirs, Dpool, images, gimg, gnames, tau, unorm, path, n_img=2,
                    show=(("domestic_cat:281", "cat"), ("dog:207", "dog"), ("big_cat:291", "cat"),
                          ("building:483", "church"))):
    """Flows along the lowest contrastive direction of a few classes at -s, 0, +s on images of a related group."""
    cats = tvm.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    fig, axes = plt.subplots(len(show), 3 * n_img, figsize=(1.1 * 3 * n_img, 1.15 * len(show)), squeeze=False)
    for r, (op, grp) in enumerate(show):
        v = next(d[2] for d in dirs if d[0] == op and d[1] == 0)
        c = v * (tau * unorm / np.sqrt(v @ Dpool @ v))
        u = to_unit(images[np.flatnonzero(gimg == gnames.index(grp))[:n_img]])
        with torch.no_grad():
            frames = [integrate(fc, c, u, -1.0), u, integrate(fc, c, u, 1.0)]
        for j in range(n_img):
            for k, fr in enumerate(frames):
                ax = axes[r, 3 * j + k]
                ax.imshow(fr[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(("-s", "0", "+s")[k], fontsize=7)
        axes[r, 0].set_ylabel(cats[int(op.split(":")[1])], fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"saved {path}")


def run_transfer(args):
    z = np.load(args.pencils, allow_pickle=True)
    cfg = json.loads(str(z["field_cfg"]))
    group_classes = json.loads(str(z["group_classes"]))
    names0 = [str(x) for x in z["group_names"]]
    sources = [i for i, n in enumerate(names0) if ":" in n]
    dirs, Dpool, names, ref = contrastive_directions(z, sources, args.k, args.random, args.seed)
    model = imagenet_classifier(args.network).to(DEVICE)
    images, gimg, gnames = load_groups(args.data_root, args.n_species, args.n_imagenette, args.size,
                                       holdout=args.holdout)
    rng = np.random.RandomState(args.seed + 7)
    sel = np.concatenate([rng.choice(np.flatnonzero(gimg == g), args.n_per_group, replace=False)
                          for g in range(len(gnames))])
    images, gimg = images[sel], gimg[sel]
    fc = FieldClass(images.shape[1], images.shape[2], images.shape[3], **cfg).to(DEVICE)
    masks = torch.zeros(len(group_classes), model.n_classes, dtype=torch.bool)
    for g, cl in enumerate(group_classes):
        masks[g, cl] = True
    masks = masks.to(DEVICE)
    unorm = image_norm(images)
    print(f"transfer: {args.network}, {len(sources)} sources x {args.k} lowest + {args.random} random directions, "
          f"{len(images)} test images (design pool), first-order image change {args.tau} of the image norm", flush=True)
    t0 = time.time()
    D, Q = np.zeros((len(dirs), len(images), len(names))), np.zeros((len(dirs), len(images), len(names)))
    for i, (src, j, v) in enumerate(dirs):
        c = v * (args.tau * unorm / np.sqrt(v @ Dpool @ v))
        D[i], Q[i] = flow_changes(model, fc, c, images, masks)
        if (i + 1) % 8 == 0:
            print(f"    {i + 1} / {len(dirs)} directions, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(args.out, changes=D, sq=Q, source=np.array([d[0] for d in dirs]),
                        index=np.array([d[1] for d in dirs]), names=np.array(names), ref=np.array(ref),
                        image_group=gimg, tau=args.tau, network=args.network)
    print(f"saved {args.out}")
    with open(args.out.replace(".npz", "_summary.json"), "w") as fh:
        json.dump(transfer_tests(args.out), fh)
    if args.figure:
        transfer_figure(model, fc, dirs, Dpool, images, gimg, gnames, args.tau_figure, unorm, args.figure)


# ---- readout geometry (control for the relational comparisons) ----
def readout_gradients(model, images, group_classes, batch=64):
    """Per image the gradient of every group's log-odds with respect to the penultimate features: the head's class
    weights mixed by the log-odds weights, [n, groups, features]; and the log-odds [n, groups]."""
    masks = torch.zeros(len(group_classes), model.n_classes, dtype=torch.bool)
    for g, cl in enumerate(group_classes):
        masks[g, cl] = True
    masks = masks.to(DEVICE)
    Wh = model.head.weight.detach().float()
    R, F0 = [], []
    with torch.no_grad():
        for s in range(0, len(images), batch):
            z = model(to_unit(images[s:s + batch]))[:, :model.n_classes]
            f, w = log_odds_weights(z, masks)
            R.append(torch.einsum("bgc,cf->bgf", w.float(), Wh).cpu())
            F0.append(f.float().cpu())
    return torch.cat(R).numpy(), torch.cat(F0).numpy()


def top_subspace(X, k):
    """Leading k right singular vectors of the rows of X: [features, k]."""
    Xt = torch.as_tensor(X, dtype=torch.float64, device=DEVICE)
    U = torch.linalg.eigh(Xt.T @ Xt)[1]
    return U[:, -k:].flip(1).cpu().numpy()


def readout_analysis(R, names, Wh, group_classes, splits, seed, k=8):
    """Distances between the groups' readouts on the same images, operator a on one half of all images against b on
    the other: (mean) 1 - cos^2 of the mean log-odds gradients; (top) missing dimensions between the top-k subspaces
    of the gradients. Comparisons by compare_halves. Image-free: 1 - cos^2 of the head's group directions (mean class
    weight of the group minus that of the rest)."""
    nG, n = len(names), len(R)
    unit = lambda v: v / np.linalg.norm(v)
    rng = np.random.RandomState(seed)
    Dm, Dt = {}, {}
    for _ in range(splits):
        p = rng.permutation(n)
        halves = (p[:n // 2], p[n // 2:])
        mean = [[unit(R[h, g].mean(0)) for g in range(nG)] for h in halves]
        top = [[top_subspace(R[h, g], k) for g in range(nG)] for h in halves]
        for a in range(nG):
            for b in range(nG):
                Dm.setdefault((a, b), []).append(1.0 - float(mean[0][a] @ mean[1][b]) ** 2)
                Dt.setdefault((a, b), []).append(float(k - ((top[0][a].T @ top[1][b]) ** 2).sum()))
    out = {"mean": compare_halves(Dm, names, splits, "readout, mean gradient direction (1 - cos^2)"),
           f"top{k}": compare_halves(Dt, names, splits, f"readout, top-{k} gradient subspace (missing of {k})")}
    full = [unit(R[:, g].mean(0)) for g in range(nG)]
    out["mean_full"] = {f"{names[a]}-{names[b]}": 1.0 - float(full[a] @ full[b]) ** 2
                        for a in range(nG) for b in range(a + 1, nG)}
    head = []
    for cl in group_classes:
        rest = np.setdiff1d(np.arange(len(Wh)), cl)
        head.append(unit(Wh[cl].mean(0) - Wh[rest].mean(0)))
    out["head"] = {f"{names[a]}-{names[b]}": 1.0 - float(head[a] @ head[b]) ** 2
                   for a in range(nG) for b in range(a + 1, nG)}
    for key in ("mean_full", "head"):
        print(f"  {key} (1 - cos^2): " + "  ".join(f"{pr} {v:.3f}" for pr, v in out[key].items()))
    return out


def run_readout(args):
    images, gimg, names = load_groups(args.data_root, args.n_species, args.n_imagenette, args.size,
                                      holdout=args.holdout, imagenette=args.imagenette.split(","), pool=args.pool)
    group_classes = [IMAGENET_GROUPS[n] for n in names]
    print(f"{len(images)} images at {args.size} px ({args.pool} pool): " + ", ".join(
        f"{n} {int((gimg == g).sum())}" for g, n in enumerate(names)) + f"; device {DEVICE}")
    summary = {"groups": names, "splits": args.splits, "pool": args.pool, "results": {}}
    for net in args.networks.split(","):
        model = imagenet_classifier(net).to(DEVICE)
        t0 = time.time()
        R, f0 = readout_gradients(model, images, group_classes)
        print(f"== {net}: readout gradients {R.shape}, {time.time() - t0:.0f} s; log-odds median on all images "
              + " ".join(f"{n} {np.median(f0[:, g]):+.1f}" for g, n in enumerate(names)), flush=True)
        summary["results"][net] = readout_analysis(R, names, model.head.weight.detach().double().cpu().numpy(),
                                                   group_classes, args.splits, args.seed)
        del R
    with open(args.out, "w") as fh:
        json.dump(summary, fh)
    print(f"saved {args.out}")


def run_flow(args):
    z = np.load(args.pencils, allow_pickle=True)
    cfg = json.loads(str(z["field_cfg"]))
    dirs, I, Dden, P = flow_directions(z, args.group, args.order, args.k, args.random)
    model = imagenet_classifier(args.network).to(DEVICE)
    images, gimg, gnames = load_groups(args.data_root, 0, 0, args.size, holdout=args.n_images, flow=True)
    images = images[gimg == gnames.index(args.group)]
    fc = FieldClass(images.shape[1], images.shape[2], images.shape[3], **cfg).to(DEVICE)
    mask = torch.zeros(1, model.n_classes, dtype=torch.bool)
    mask[0, IMAGENET_GROUPS[args.group]] = True
    mask = mask.to(DEVICE)
    print(f"flows on {len(images)} held-out {args.group} images, order {args.order}, first-order image change "
          f"{args.tau} of the image norm")
    rows, unorm = flow_test(model, fc, dirs, I, Dden, P, images, mask, args.tau)
    if args.figure:
        flow_figure(model, fc, dirs, I, Dden, P, images, args.tau_figure, unorm, args.figure)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"group": args.group, "order": args.order, "tau": args.tau, "rows": rows}, fh)


def operator_groups(names, operators):
    """Class lists of the operators: the design groups, then (extended) the single-class operators of SUPERCLASSES,
    named superclass:index."""
    names, group_classes = list(names), [IMAGENET_GROUPS[n] for n in names]
    if operators == "extended":
        for sc, cl in SUPERCLASSES.items():
            names += [f"{sc}:{c}" for c in cl]
            group_classes += [[c] for c in cl]
    return names, group_classes


def run_pencils(args):
    images, gimg, names = load_groups(args.data_root, args.n_species, args.n_imagenette, args.size,
                                      holdout=args.holdout, imagenette=args.imagenette.split(","), pool=args.pool)
    print(f"{len(images)} images at {args.size} px ({args.pool} pool): " + ", ".join(
        f"{n} {int((gimg == g).sum())}" for g, n in enumerate(names)) + f"; device {DEVICE}")
    names, group_classes = operator_groups(names, args.operators)
    print(f"{len(names)} operators: " + ", ".join(names))
    for net in args.networks.split(","):
        model = imagenet_classifier(net).to(DEVICE)
        fc = FieldClass(3, args.size, args.size, kg=args.kg, ku=args.ku, kp=args.kp, k2=args.k2,
                        sigma2=args.sigma2).to(DEVICE)
        t0 = time.time()
        print(f"== {net}: {len(fc.fields)} fields", flush=True)
        res = compute_pencils(model, fc, images, gimg, group_classes, batch=args.batch)
        path = args.out.replace(".npz", f"_{net}.npz")
        save_pencils(path, res, fc, names, group_classes, gimg, {"network": net, "size": args.size, "pool": args.pool})
        print(f"   {time.time() - t0:.0f} s, saved {path}", flush=True)
        analyse(path, k=args.k, splits=args.splits, out=path.replace(".npz", "_summary.json"))


def main():
    ap = argparse.ArgumentParser(description="Class-conditional symmetries of a frozen classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pencils")
    p.add_argument("--networks", default="resnet50_v1,resnet50_random")
    p.add_argument("--data-root", default="./data_img")
    p.add_argument("--n-species", type=int, default=1800, help="Pet images per species for the pencils")
    p.add_argument("--n-imagenette", type=int, default=600, help="Imagenette images per class (0: none)")
    p.add_argument("--imagenette", default="springer,tench,garbage_truck,church")
    p.add_argument("--holdout", type=int, default=200, help="Pet images per species kept for flows")
    p.add_argument("--pool", default="design", choices=("design", "fresh"),
                   help="fresh: every image the design (these n, holdout, seed) did not use")
    p.add_argument("--operators", default="design", choices=("design", "extended"),
                   help="extended: also single-class operators of SUPERCLASSES (no images of their own)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--kg", type=int, default=5)
    p.add_argument("--ku", type=int, default=2)
    p.add_argument("--kp", type=int, default=3)
    p.add_argument("--k2", type=int, default=2)
    p.add_argument("--sigma2", type=float, default=1.0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--out", default="semantic.npz")
    a = sub.add_parser("analyse")
    a.add_argument("pencils")
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--splits", type=int, default=20)
    a.add_argument("--out", default=None)
    f = sub.add_parser("flow")
    f.add_argument("--pencils", required=True)
    f.add_argument("--network", default="resnet50_v1")
    f.add_argument("--group", default="cat")
    f.add_argument("--order", type=int, default=1)
    f.add_argument("--k", type=int, default=3)
    f.add_argument("--random", type=int, default=8)
    f.add_argument("--tau", type=float, default=0.05, help="first-order image change / image norm")
    f.add_argument("--tau-figure", type=float, default=0.15)
    f.add_argument("--n-images", type=int, default=200, help="held-out Pet images (the pencils' holdout)")
    f.add_argument("--size", type=int, default=224)
    f.add_argument("--data-root", default="./data_img")
    f.add_argument("--figure", default=None)
    f.add_argument("--out", default=None)
    r = sub.add_parser("readout", help="readout-geometry control on the pencils' images (forward passes only)")
    r.add_argument("--networks", default="resnet50_v1,resnet50_random")
    r.add_argument("--data-root", default="./data_img")
    r.add_argument("--n-species", type=int, default=1800)
    r.add_argument("--n-imagenette", type=int, default=600)
    r.add_argument("--imagenette", default="springer,tench,garbage_truck,church")
    r.add_argument("--holdout", type=int, default=200)
    r.add_argument("--pool", default="design", choices=("design", "fresh"))
    r.add_argument("--size", type=int, default=224)
    r.add_argument("--splits", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="semantic_readout.json")
    t = sub.add_parser("transfer", help="finite flows along the class-specific directions; transfer between classes")
    t.add_argument("--pencils", required=True, help="confirmatory npz (extended operators)")
    t.add_argument("--network", default="resnet50_v1")
    t.add_argument("--k", type=int, default=3, help="lowest contrastive directions per source")
    t.add_argument("--random", type=int, default=8)
    t.add_argument("--tau", type=float, default=0.05, help="first-order image change / image norm")
    t.add_argument("--tau-figure", type=float, default=0.15)
    t.add_argument("--n-per-group", type=int, default=100, help="test images per design group (design pool)")
    t.add_argument("--data-root", default="./data_img")
    t.add_argument("--n-species", type=int, default=1800)
    t.add_argument("--n-imagenette", type=int, default=600)
    t.add_argument("--holdout", type=int, default=200)
    t.add_argument("--size", type=int, default=224)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--figure", default=None)
    t.add_argument("--out", default="transfer.npz")
    args = ap.parse_args()
    {"pencils": run_pencils, "analyse": lambda a: analyse(a.pencils, a.k, a.splits, out=a.out), "flow": run_flow,
     "readout": run_readout, "transfer": run_transfer}[args.cmd](args)


if __name__ == "__main__":
    main()
