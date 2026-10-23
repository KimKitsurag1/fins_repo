"""
FINS on CIFAR-10

Reproduces the main image-domain experiments from Section 6 of the conference paper.
Downloads CIFAR-10 and DINOv2 weights automatically on first run.

Usage:
    pip install -r requirements.txt
    python fins_cifar.py

Parameters are configured in the CONFIG dict below; command-line flags override
them (python fins_cifar.py --help). Without flags the script reproduces the CONFIG
run. Auto-resumes from saved results and checkpoints if interrupted.

Controls (flags):
    --methods fins,null_init,null_smooth --paired
                                              learned generators vs two chance levels of the
                                              learned per-slot magnitudes
    --cifar-train-size N [--sym-gen-subset M] training-set size of the classifier
                                              (and of the generator fit)
    --base-aug                                random crop and flip in every arm

Outputs (under ./outputs/):
    fins_cifar_results.json     - per-seed accuracy across all methods
    sym_gen_checkpoints/*.pt    - FNO generator weights, one per seed

Runtime: ~30-45 min per seed on A100/V100; ~1.5 h on T4.

"""

# ---- Imports ----
import argparse
import os, json, math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO

# CONFIGURATION — edit these
CONFIG = {
    # Experiment scale
    "n_seeds":          1,         # 1 for smoke test, 3-5 for confidence
    "seed_start":       100,
    "smoke_test":       True,      # True = only no_aug + fins; False = + RandAug/AutoAug/TrivialAug

    # FINS hyperparameters (conference paper, Section 6.3)
    "w_olver":          1.0,       # w_1 for L_inv (infinitesimal Olver)
    "w_finite_tau":     1.0,       # w_2 for L_tau (finite-tau invariance)
    "w_balance":        10.0,      # w_4 for L_bal (geom/phi energy balance)
    "olver_eps":        0.01,
    "tau_probe":        0.8,       # for L_tau anchor
    "tau_range":        (0.5, 1.3),# augmentation sampling range
    "n_active":         3,         # generators sampled per augmented copy
    "N_sym":            6,

    # Training
    "epochs_sym_gen":   12,
    "epochs_resnet":    30,
    "sym_gen_subset":   5000,      # subset of CIFAR train for sym_gen discovery
    "n_aug":            3,         # augmented copies per image (4x total)
    "cifar_train_size": 50000,     # reduce for low-VRAM (e.g., 20000)

    # I/O
    "out_dir":          "./outputs",
    "out_json":         "fins_cifar_results.json",
    "encoder_name":     "dinov2_vits14",
    "save_sym_gen":     True,      # save sym_gen weights per seed
    "sym_micro_batch":  None,      # encoder terms in slices (same gradient, lower memory; T4: 4)
}

ALL_METHODS = ["no_aug", "fins", "null_init", "null_smooth", "randaug", "autoaug", "trivialaug"]

# Command-line overrides of CONFIG
_ap = argparse.ArgumentParser(description="FINS on CIFAR-10")
_ap.add_argument("--n-seeds", type=int, default=None)
_ap.add_argument("--seed-start", type=int, default=None)
_ap.add_argument("--full", action="store_true", help="all baselines (smoke_test off)")
_ap.add_argument("--methods", default=None, help="comma list from " + ",".join(ALL_METHODS))
_ap.add_argument("--cifar-train-size", type=int, default=None, help="classifier training images")
_ap.add_argument("--sym-gen-subset", type=int, default=None, help="generator fit images (capped at the training set)")
_ap.add_argument("--epochs-sym-gen", type=int, default=None)
_ap.add_argument("--epochs-resnet", type=int, default=None)
_ap.add_argument("--sym-micro-batch", type=int, default=None,
                 help="generator training: encoder terms in slices of this size (same gradient, lower memory)")
_ap.add_argument("--paired", action="store_true",
                 help="reseed before every arm: same classifier init, data order and augmentation draws")
_ap.add_argument("--base-aug", action="store_true", help="random crop (pad 4) and horizontal flip in every arm")
_ap.add_argument("--out-dir", default=None)
_ap.add_argument("--out-json", default=None)
_ap.add_argument("--sym-gen-dir", default=None, help="generator checkpoints (default: out_dir/sym_gen_checkpoints)")
_args, _ = _ap.parse_known_args()
for _k in ("n_seeds", "seed_start", "cifar_train_size", "sym_gen_subset", "epochs_sym_gen", "epochs_resnet",
           "sym_micro_batch", "out_dir", "out_json"):
    if getattr(_args, _k) is not None:
        CONFIG[_k] = getattr(_args, _k)
if _args.full:
    CONFIG["smoke_test"] = False
CONFIG["paired"] = _args.paired
CONFIG["base_aug"] = _args.base_aug
CONFIG["methods"] = (_args.methods.split(",") if _args.methods else
                     ["no_aug", "fins"] if CONFIG["smoke_test"] else
                     ["no_aug", "fins", "randaug", "autoaug", "trivialaug"])
if set(CONFIG["methods"]) - set(ALL_METHODS):
    raise SystemExit(f"unknown methods: {sorted(set(CONFIG['methods']) - set(ALL_METHODS))}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# Output paths
OUT_DIR = CONFIG["out_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

OUT_JSON       = os.path.join(OUT_DIR, CONFIG["out_json"])
SYM_GEN_DIR    = _args.sym_gen_dir or os.path.join(OUT_DIR, "sym_gen_checkpoints")
if CONFIG["save_sym_gen"]:
    os.makedirs(SYM_GEN_DIR, exist_ok=True)

print(f"Results JSON  : {OUT_JSON}")
if CONFIG["save_sym_gen"]:
    print(f"Sym_gen ckpts : {SYM_GEN_DIR}")
print(f"Methods       : {','.join(CONFIG['methods'])}; paired {CONFIG['paired']}; base aug {CONFIG['base_aug']}")
with open(os.path.splitext(OUT_JSON)[0] + "_config.json", "w") as _f:
    json.dump(CONFIG, _f, indent=2)


# Reproducibility
def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# Corruptions (parametric — small kernels for 32x32)
def _gaussian_kernel_1d(ksize, sigma, device):
    x = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2.0
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def apply_gaussian_blur(img, ksize=3, sigma=1.0):
    if img.dim() == 3:
        img = img.unsqueeze(0); squeeze = True
    else:
        squeeze = False
    C = img.shape[1]
    k1d = _gaussian_kernel_1d(ksize, sigma, img.device)
    kx = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    pad = ksize // 2
    out = F.conv2d(F.pad(img, (pad, pad, 0, 0), mode="reflect"), kx, groups=C)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), ky, groups=C)
    return out.squeeze(0) if squeeze else out


def apply_gaussian_noise(img, s=0.15): return torch.clamp(img + s*torch.randn_like(img), 0, 1)
def apply_shot_noise(img, s=0.15): return torch.clamp(img + s*torch.randn_like(img)*torch.sqrt(img+1e-3), 0, 1)
def apply_impulse_noise(img, s=0.10):
    m = torch.rand_like(img); out = img.clone()
    out[m < s/2] = 0.0; out[m > 1-s/2] = 1.0
    return out
def apply_contrast(img, f=0.4): m = img.mean(); return torch.clamp((img-m)*f+m, 0, 1)
def apply_brightness(img, d=0.3): return torch.clamp(img+d, 0, 1)
def apply_jpeg_compression(img, q=10):
    levels = max(2, q // 5); return (img*levels).round() / levels
def apply_pixelate(img, b=4):
    C, H, W = img.shape
    small = F.interpolate(img.unsqueeze(0), size=(H//b, W//b), mode="nearest")
    return F.interpolate(small, size=(H, W), mode="nearest").squeeze(0)


def build_corruptions_cifar(severity_scale=1.0):
    return {
        "clean":          lambda x: x,
        "gaussian_noise": lambda x: apply_gaussian_noise(x, 0.15 * severity_scale),
        "shot_noise":     lambda x: apply_shot_noise(x, 0.15 * severity_scale),
        "impulse_noise":  lambda x: apply_impulse_noise(x, 0.10 * severity_scale),
        "gaussian_blur":  lambda x: apply_gaussian_blur(x, ksize=3, sigma=1.0 * severity_scale),
        "contrast":       lambda x: apply_contrast(x, 0.4 / max(0.2, severity_scale)),
        "brightness":     lambda x: apply_brightness(x, 0.30 * severity_scale),
        "jpeg":           lambda x: apply_jpeg_compression(x, max(2, int(10 / max(0.2, severity_scale)))),
        "pixelate":       lambda x: apply_pixelate(x, max(2, int(4 * severity_scale))),
    }



def load_dinov2(name, **kw):
    """DINOv2 from a local clone (env DINOV2_REPO) or the hub cache, without GitHub API validation."""
    repo = os.environ.get("DINOV2_REPO", "")
    if os.path.isdir(repo):
        return torch.hub.load(repo, name, source="local", **kw)
    return torch.hub.load("facebookresearch/dinov2", name, skip_validation=True, trust_repo=True, **kw)

# Encoder target — DINOv2-ViT-S/14
class EncoderTarget(nn.Module):
    def __init__(self, name="dinov2_vits14", image_size=224):
        super().__init__()
        self.image_size = image_size
        self.model = load_dinov2(name)
        self.feature_dim = self.model.embed_dim
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
            img = F.interpolate(img, size=(self.image_size, self.image_size),
                                 mode="bilinear", align_corners=False)
        x = (img - self.mean) / self.std
        return self.model(x)


# Symmetry generator (FNO)
class MultiSymmetryFNO(nn.Module):
    def __init__(self, hidden_channels=16, n_modes=8, n_layers=3, N_sym=6):
        super().__init__()
        self.N_sym = N_sym
        self.fno = FNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                       in_channels=5, out_channels=5 * N_sym, n_layers=n_layers,
                       use_channel_mlp=True, channel_mlp_dropout=0.0)

    def forward(self, x_grid, y_grid, img):
        out = self.fno(torch.cat([x_grid, y_grid, img], dim=1))
        gens = []
        for i in range(self.N_sym):
            xi  = out[:, 5*i + 0:5*i + 1]
            eta = out[:, 5*i + 1:5*i + 2]
            phi = out[:, 5*i + 2:5*i + 5]
            gens.append((xi, eta, phi))
        return gens


class ScaledGenerators(nn.Module):
    """Generators of `base` with per-slot scales on (xi, eta) and on phi."""
    def __init__(self, base, s_geo, s_phi):
        super().__init__()
        self.base = base
        self.N_sym = base.N_sym
        self.register_buffer("s_geo", torch.tensor(s_geo, dtype=torch.float32))
        self.register_buffer("s_phi", torch.tensor(s_phi, dtype=torch.float32))

    def forward(self, x_grid, y_grid, img):
        return [(xi * self.s_geo[i], eta * self.s_geo[i], phi * self.s_phi[i])
                for i, (xi, eta, phi) in enumerate(self.base(x_grid, y_grid, img))]


def create_normalized_grids(B, H, W, device):
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    xg, yg = torch.meshgrid(x, y, indexing="xy")
    return xg.view(1, 1, H, W).expand(B, 1, -1, -1), yg.view(1, 1, H, W).expand(B, 1, -1, -1)


# RK4 group action
def apply_symmetry_rk4(xi, eta, phi, img, tau, n_steps=4,
                        disable_geom=False, disable_phi=False):
    B, C, H, W = img.shape
    if disable_geom: xi = torch.zeros_like(xi); eta = torch.zeros_like(eta)
    if disable_phi:  phi = torch.zeros_like(phi)

    x_px = torch.linspace(-1, 1, W, device=img.device).view(1, 1, 1, -1).expand(B, 1, H, -1)
    y_px = torch.linspace(-1, 1, H, device=img.device).view(1, 1, -1, 1).expand(B, 1, -1, W)
    grid = torch.cat([x_px, y_px], dim=1)
    dt = tau / n_steps

    def sample_v(g):
        gs = g.permute(0, 2, 3, 1)
        v_x = F.grid_sample(xi,  gs, mode="bilinear", padding_mode="border", align_corners=True)
        v_y = F.grid_sample(eta, gs, mode="bilinear", padding_mode="border", align_corners=True)
        return torch.cat([v_x, v_y], dim=1)

    for _ in range(n_steps):
        k1 = sample_v(grid)
        k2 = sample_v(grid + 0.5*dt*k1)
        k3 = sample_v(grid + 0.5*dt*k2)
        k4 = sample_v(grid + dt*k3)
        grid = grid + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    grid_s = grid.permute(0, 2, 3, 1)
    warped = F.grid_sample(img, grid_s, mode="bilinear", padding_mode="border", align_corners=True)
    return torch.clamp(warped + tau * phi, 0.0, 1.0)


# Losses
def _image_spatial_gradients(img):
    du_dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    du_dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    du_dx = F.pad(du_dx, (0, 1, 0, 0), mode="replicate")
    du_dy = F.pad(du_dy, (0, 0, 0, 1), mode="replicate")
    return du_dx, du_dy


def encoder_olver_loss(gens, encoder, img, eps=0.01):
    du_dx, du_dy = _image_spatial_gradients(img)
    loss = 0.0
    for (xi, eta, phi) in gens:
        v_im = -xi * du_dx - eta * du_dy + phi
        v_norm = v_im.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-6)
        v_dir = v_im / v_norm.view(-1, 1, 1, 1)
        x_plus  = (img + eps * v_dir).clamp(0.0, 1.0)
        x_minus = (img - eps * v_dir).clamp(0.0, 1.0)
        jvp = (encoder(x_plus) - encoder(x_minus)) / (2 * eps)
        loss = loss + ((jvp * v_norm) ** 2).sum(dim=1).mean()
    return loss / len(gens)


def encoder_finite_tau_loss(gens, encoder, img, tau_probe=0.3, rk4_steps=4):
    f_ref = encoder(img)
    loss = 0.0
    for (xi, eta, phi) in gens:
        x_plus  = apply_symmetry_rk4(xi, eta, phi, img, tau=+tau_probe, n_steps=rk4_steps)
        x_minus = apply_symmetry_rk4(xi, eta, phi, img, tau=-tau_probe, n_steps=rk4_steps)
        loss = loss + ((encoder(x_plus)  - f_ref) ** 2).sum(dim=1).mean()
        loss = loss + ((encoder(x_minus) - f_ref) ** 2).sum(dim=1).mean()
    return loss / (2 * len(gens))


def encoder_manifold_loss(gens, encoder, img, tau_probe=0.3):
    f_ref = encoder(img)
    ref_norm = f_ref.norm(dim=1, keepdim=True)
    loss = 0.0
    for (xi, eta, phi) in gens:
        x_aug = apply_symmetry_rk4(xi, eta, phi, img, tau=tau_probe, n_steps=2)
        aug_norm = encoder(x_aug).norm(dim=1, keepdim=True)
        loss = loss + ((aug_norm - ref_norm) ** 2).mean()
    return loss / len(gens)


def ortho_loss(gens):
    loss = 0.0; pairs = 0
    for i in range(len(gens)):
        for j in range(i+1, len(gens)):
            xi1, eta1, phi1 = gens[i]; xi2, eta2, phi2 = gens[j]
            ip = ((xi1*xi2).mean() + (eta1*eta2).mean() + (phi1*phi2).mean())
            loss = loss + ip**2; pairs += 1
    return loss / max(1, pairs)


def lipschitz_loss(gens):
    loss = 0.0
    for (xi, eta, phi) in gens:
        for v in (xi, eta, phi):
            loss = loss + (v[:, :, :, 2:] - v[:, :, :, :-2]).pow(2).mean()
            loss = loss + (v[:, :, 2:, :] - v[:, :, :-2, :]).pow(2).mean()
    return loss


def phi_geom_balance_loss(gens, target_ratio=0.5):
    loss = 0.0
    for (xi, eta, phi) in gens:
        e_geom = (xi**2).mean() + (eta**2).mean()
        e_phi  = (phi**2).mean()
        ratio  = e_geom / (e_geom + e_phi + 1e-8)
        loss = loss + (ratio - target_ratio)**2
    return loss / len(gens)


# Chance levels at the learned per-slot magnitudes
@torch.no_grad()
def slot_rms(gen, images, n_probe=256):
    """Per-slot RMS of (xi, eta) and of phi on `images`."""
    gen.eval()
    img = images[:n_probe].to(DEVICE)
    B, _, H, W = img.shape
    xg, yg = create_normalized_grids(B, H, W, DEVICE)
    return [(float((xi ** 2 + eta ** 2).mean().sqrt()), float((phi ** 2).mean().sqrt()))
            for xi, eta, phi in gen(xg, yg, img)]


@torch.no_grad()
def null_init(learned, images, seed):
    """The generator network at initialisation (own init stream), rescaled slot by slot to the learned
    RMS of (xi, eta) and of phi. Its fields are nearly constant: global translations and colour shifts."""
    target = slot_rms(learned, images)
    torch.manual_seed(seed + 1_000_003)
    base = MultiSymmetryFNO(N_sym=learned.N_sym).to(DEVICE).eval()
    got = slot_rms(base, images)
    s_geo = [t[0] / max(g[0], 1e-12) for t, g in zip(target, got)]
    s_phi = [t[1] / max(g[1], 1e-12) for t, g in zip(target, got)]
    print("  learned RMS of (xi, eta) / phi per slot: " + "  ".join(f"{g:.3g}/{q:.3g}" for g, q in target))
    return ScaledGenerators(base, s_geo, s_phi).to(DEVICE).eval()


class SmoothFieldGenerators(nn.Module):
    """Image-independent random smooth fields per slot: Fourier modes 1 <= |k|_inf <= kmax on [-1, 1]^2,
    amplitudes ~ 1 / (1 + |k|^2), no constant mode, scaled to the given RMS of (xi, eta) and of phi."""
    def __init__(self, rms, H, W, seed, kmax=3):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 2_000_003)
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        modes = [(kx, ky) for kx in range(kmax + 1) for ky in range(-kmax, kmax + 1) if (kx, ky) > (0, 0)]

        def field(n):
            f = torch.zeros(n, H, W)
            for kx, ky in modes:
                ph = math.pi * (kx * x + ky * y)
                a, b = torch.randn(2, n, 1, 1, generator=g)
                f += (a * torch.cos(ph) + b * torch.sin(ph)) / (1 + kx * kx + ky * ky)
            return f

        self.N_sym = len(rms)
        xs, es, ps = [], [], []
        for r_geo, r_phi in rms:
            xe = field(2)
            xe = xe * r_geo / xe.pow(2).sum(0).mean().sqrt()
            p = field(3)
            p = p * r_phi / p.pow(2).mean().sqrt()
            xs.append(xe[:1]); es.append(xe[1:]); ps.append(p)
        self.register_buffer("xi", torch.stack(xs))
        self.register_buffer("eta", torch.stack(es))
        self.register_buffer("phi", torch.stack(ps))

    def forward(self, x_grid, y_grid, img):
        B = img.shape[0]
        return [(self.xi[i].expand(B, -1, -1, -1).contiguous(), self.eta[i].expand(B, -1, -1, -1).contiguous(),
                 self.phi[i].expand(B, -1, -1, -1).contiguous()) for i in range(self.N_sym)]


def random_crop_flip(img, pad=4):
    """Random crop after zero padding and random horizontal flip, per image (standard CIFAR augmentation)."""
    B, C, H, W = img.shape
    padded = F.pad(img, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (B,), device=img.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=img.device)
    rows = (i[:, None] + torch.arange(H, device=img.device)[None])[:, None, :, None]
    cols = (j[:, None] + torch.arange(W, device=img.device)[None])[:, None, None, :]
    out = padded[torch.arange(B, device=img.device)[:, None, None, None],
                 torch.arange(C, device=img.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=img.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


# Training
def train_sym_gen(train_images, encoder, *, N_sym=6, epochs=12, batch_size=32,
                   w_olver=1.0, w_finite_tau=1.0, w_balance=10.0,
                   w_ortho=1.0, w_lips=0.01, w_manifold=0.1,
                   olver_eps=0.01, tau_probe=0.3, micro_batch=None):
    """micro_batch: encoder terms in slices of this size, backpropagated slice by slice; the batch-level
    terms (orthogonality, balance, smoothness) on the whole batch. Same gradient, lower peak memory."""
    G = MultiSymmetryFNO(N_sym=N_sym).to(DEVICE)
    opt = optim.Adam(G.parameters(), lr=1e-4)
    loader = DataLoader(TensorDataset(train_images, torch.zeros(len(train_images))),
                        batch_size=batch_size, shuffle=True, num_workers=0)
    for ep in range(epochs):
        G.train()
        ls = {"olver":0., "ftau":0., "bal":0., "tot":0.}; nb = 0
        for img, _ in loader:
            img = img.to(DEVICE)
            B, _, H, W = img.shape
            xg, yg = create_normalized_grids(B, H, W, DEVICE)
            gens = G(xg, yg, img)
            if micro_batch is None or micro_batch >= B:
                l_olv = encoder_olver_loss(gens, encoder, img, eps=olver_eps) if w_olver > 0 else torch.tensor(0., device=DEVICE)
                l_ft  = encoder_finite_tau_loss(gens, encoder, img, tau_probe=tau_probe) if w_finite_tau > 0 else torch.tensor(0., device=DEVICE)
                l_or  = ortho_loss(gens)
                l_lp  = lipschitz_loss(gens)
                l_bl  = phi_geom_balance_loss(gens)
                l_mn  = encoder_manifold_loss(gens, encoder, img, tau_probe=tau_probe)
                loss = w_olver*l_olv + w_finite_tau*l_ft + w_ortho*l_or + w_lips*l_lp + w_balance*l_bl + w_manifold*l_mn
                opt.zero_grad(); loss.backward(); opt.step()
                l_olv, l_ft, l_bl, tot = float(l_olv.detach()), float(l_ft.detach()), float(l_bl.detach()), float(loss.detach())
            else:
                opt.zero_grad()
                l_olv = l_ft = l_mn = 0.0
                for s0 in range(0, B, micro_batch):
                    sl = slice(s0, s0 + micro_batch)
                    part = [(xi[sl], eta[sl], phi[sl]) for xi, eta, phi in gens]
                    frac = img[sl].shape[0] / B
                    o = encoder_olver_loss(part, encoder, img[sl], eps=olver_eps) if w_olver > 0 else torch.tensor(0., device=DEVICE)
                    f = encoder_finite_tau_loss(part, encoder, img[sl], tau_probe=tau_probe) if w_finite_tau > 0 else torch.tensor(0., device=DEVICE)
                    m = encoder_manifold_loss(part, encoder, img[sl], tau_probe=tau_probe)
                    (frac * (w_olver*o + w_finite_tau*f + w_manifold*m)).backward(retain_graph=True)
                    l_olv += frac * float(o.detach()); l_ft += frac * float(f.detach()); l_mn += frac * float(m.detach())
                    del part, o, f, m
                l_or  = ortho_loss(gens)
                l_lp  = lipschitz_loss(gens)
                l_bl  = phi_geom_balance_loss(gens)
                rest = w_ortho*l_or + w_lips*l_lp + w_balance*l_bl
                rest.backward()
                opt.step()
                tot = w_olver*l_olv + w_finite_tau*l_ft + w_manifold*l_mn + float(rest.detach())
                l_bl = float(l_bl.detach())
            ls["olver"] += l_olv; ls["ftau"] += l_ft
            ls["bal"]   += l_bl; ls["tot"]  += tot
            nb += 1
        print(f"    epoch {ep+1:2d}/{epochs}  olver={ls['olver']/nb:.3f}  ftau={ls['ftau']/nb:.3f}  bal={ls['bal']/nb:.4f}  tot={ls['tot']/nb:.3f}")
    G.eval()
    return G


@torch.no_grad()
def augment_fins(images, sym_gen, *, n_aug=3, tau_range=(0.2, 0.5),
                           n_active=2, batch_size=128):
    sym_gen.eval()
    out = [images]
    for _ in range(n_aug):
        block = []
        for s in range(0, len(images), batch_size):
            img = images[s:s+batch_size].to(DEVICE)
            B, _, H, W = img.shape
            xg, yg = create_normalized_grids(B, H, W, DEVICE)
            gens = sym_gen(xg, yg, img)
            idxs = np.random.choice(len(gens), size=min(n_active, len(gens)), replace=False)
            cur = img.clone()
            for idx in idxs:
                xi, eta, phi = gens[idx]
                tau = float(np.random.uniform(*tau_range))
                if np.random.rand() > 0.5: tau = -tau
                cur = apply_symmetry_rk4(xi, eta, phi, cur, tau=tau, n_steps=4)
            block.append(cur.cpu())
        out.append(torch.cat(block, dim=0))
    return torch.cat(out, dim=0)


def make_resnet18_cifar(num_classes=10):
    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train_resnet18_cifar(imgs, lbls, num_classes=10, epochs=30, batch_size=128, lr=0.1, base_aug=False):
    net = make_resnet18_cifar(num_classes).to(DEVICE)
    opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(imgs, lbls), batch_size=batch_size, shuffle=True, num_workers=0)
    for ep in range(epochs):
        net.train()
        for img, lbl in loader:
            img = img.to(DEVICE)
            if base_aug:
                img = random_crop_flip(img)
            loss = F.cross_entropy(net(img), lbl.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 5 == 0:
            print(f"    epoch {ep+1}/{epochs} done")
    net.eval()
    return net


@torch.no_grad()
def evaluate_corruptions_cifar(net, test_imgs, test_lbls, corruptions, batch_size=256):
    res = {}
    for name, fn in corruptions.items():
        corrupted = torch.stack([fn(x) for x in test_imgs])
        preds = []
        for i in range(0, len(corrupted), batch_size):
            preds.append(net(corrupted[i:i+batch_size].to(DEVICE)).argmax(dim=1).cpu())
        preds = torch.cat(preds, dim=0)
        res[name] = (preds == test_lbls).float().mean().item()
    return res


def make_aug_batch(images, transform, n_aug=3):
    out = [images]
    for _ in range(n_aug):
        out.append(torch.stack([transform(x) for x in images]))
    return torch.cat(out, dim=0)


# Main per-seed driver
def run_one_seed(seed, train_imgs, train_lbls, test_imgs, test_lbls, cfg):
    seed_all(seed)
    corruptions = build_corruptions_cifar()
    methods = cfg["methods"]

    def arm(name):
        if cfg["paired"]:
            seed_all(seed)
        print(f"  arm {name}")

    # 1. Sym_gen training (on subset)
    sym_gen = sub = None
    if {"fins", "null_init", "null_smooth"} & set(methods):
        perm = np.random.RandomState(seed).permutation(len(train_imgs))
        sub = train_imgs[perm[:cfg["sym_gen_subset"]]]
        tag = "" if (len(train_imgs), len(sub)) == (50000, 5000) else f"_train{len(train_imgs)}_fit{len(sub)}"
        print(f"  loading encoder {cfg['encoder_name']}...")
        encoder = EncoderTarget(name=cfg["encoder_name"], image_size=224).to(DEVICE)

        # Check for cached sym_gen weights (resume after disconnect)
        sym_gen_path = os.path.join(SYM_GEN_DIR, f"sym_gen_seed{seed}{tag}.pt") if cfg["save_sym_gen"] else None
        if sym_gen_path and os.path.exists(sym_gen_path):
            print(f"  found cached sym_gen at {sym_gen_path} — loading...")
            sym_gen = MultiSymmetryFNO(N_sym=cfg["N_sym"]).to(DEVICE)
            # weights_only=False required for PyTorch 2.6+: NeuralOp FNO state_dict
            # contains activation function refs (gelu) that the safe loader blocks.
            _state = torch.load(sym_gen_path, map_location=DEVICE, weights_only=False)
            # Pop _metadata key (NeuralOp FNO serialization quirk — see visualize script).
            _state.pop("_metadata", None)
            sym_gen.load_state_dict(_state)
            sym_gen.eval()
        else:
            print(f"  training sym_gen on {len(sub)} CIFAR images...")
            sym_gen = train_sym_gen(sub, encoder,
                                    N_sym=cfg["N_sym"], epochs=cfg["epochs_sym_gen"],
                                    w_olver=cfg["w_olver"], w_finite_tau=cfg["w_finite_tau"],
                                    w_balance=cfg["w_balance"],
                                    olver_eps=cfg["olver_eps"], tau_probe=cfg["tau_probe"],
                                    micro_batch=cfg["sym_micro_batch"])
            if sym_gen_path:
                torch.save(sym_gen.state_dict(), sym_gen_path)
                print(f"  saved sym_gen to {sym_gen_path}")
        del encoder
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    results = {}

    # 2. no_aug
    if "no_aug" in methods:
        arm("no_aug")
        print("  training ResNet-18 no_aug...")
        net = train_resnet18_cifar(train_imgs, train_lbls, epochs=cfg["epochs_resnet"], base_aug=cfg["base_aug"])
        results["no_aug"] = evaluate_corruptions_cifar(net, test_imgs, test_lbls, corruptions)

    # 3. fins (learned generators) and the two chance levels at the learned per-slot magnitudes
    for name in ("fins", "null_init", "null_smooth"):
        if name not in methods:
            continue
        if name == "fins":
            gen = sym_gen
        elif name == "null_init":
            gen = null_init(sym_gen, sub, seed)
        else:
            gen = SmoothFieldGenerators(slot_rms(sym_gen, sub), *train_imgs.shape[-2:], seed).to(DEVICE).eval()
        arm(name)
        print(f"  building {name} augmented set...")
        aug_imgs = augment_fins(train_imgs, gen,
                                n_aug=cfg["n_aug"], tau_range=cfg["tau_range"],
                                n_active=cfg["n_active"])
        aug_lbls = train_lbls.repeat(cfg["n_aug"] + 1)
        print(f"    augmented size: {len(aug_imgs)}")
        print(f"  training ResNet-18 on {name}-augmented data...")
        net = train_resnet18_cifar(aug_imgs, aug_lbls, epochs=cfg["epochs_resnet"], base_aug=cfg["base_aug"])
        results[name] = evaluate_corruptions_cifar(net, test_imgs, test_lbls, corruptions)
        del aug_imgs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # 4. Baselines: RandAug / AutoAug / TrivialAug
    for name, policy in [
        ("randaug",    transforms.RandAugment(num_ops=2, magnitude=9)),
        ("autoaug",    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)),
        ("trivialaug", transforms.TrivialAugmentWide()),
    ]:
        if name not in methods:
            continue
        arm(name)
        tfm = transforms.Compose([transforms.ToPILImage(), policy, transforms.ToTensor()])
        print(f"  building {name}-augmented set...")
        aug_imgs = make_aug_batch(train_imgs, tfm, n_aug=cfg["n_aug"])
        aug_lbls = train_lbls.repeat(cfg["n_aug"] + 1)
        print(f"  training ResNet-18 on {name}-augmented data...")
        net = train_resnet18_cifar(aug_imgs, aug_lbls, epochs=cfg["epochs_resnet"], base_aug=cfg["base_aug"])
        results[name] = evaluate_corruptions_cifar(net, test_imgs, test_lbls, corruptions)
        del aug_imgs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return results


# Load CIFAR-10
print("\n=== Loading CIFAR-10 ===")
tfm = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root="./data_cifar", train=True, download=True, transform=tfm)
test_set  = datasets.CIFAR10(root="./data_cifar", train=False, download=True, transform=tfm)

train_imgs_all = torch.stack([train_set[i][0] for i in range(len(train_set))])
train_lbls_all = torch.tensor([train_set[i][1] for i in range(len(train_set))], dtype=torch.long)
test_imgs  = torch.stack([test_set[i][0] for i in range(len(test_set))])
test_lbls  = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)

# Optionally subset for memory-constrained environments
if CONFIG["cifar_train_size"] < len(train_imgs_all):
    perm = np.random.RandomState(0).permutation(len(train_imgs_all))[:CONFIG["cifar_train_size"]]
    train_imgs = train_imgs_all[perm]
    train_lbls = train_lbls_all[perm]
else:
    train_imgs = train_imgs_all
    train_lbls = train_lbls_all
print(f"train: {train_imgs.shape}   test: {test_imgs.shape}")


# Run experiment (with auto-resume from saved results / checkpoints)
# Check for existing results — resume from where we left off
all_results = []
if os.path.exists(OUT_JSON):
    try:
        with open(OUT_JSON) as f:
            all_results = json.load(f)
        print(f"\nFound existing results: {len(all_results)} seed(s) already completed.")
        if len(all_results) >= CONFIG["n_seeds"]:
            print(f"  All {CONFIG['n_seeds']} seeds already done. Skipping to results display.")
        else:
            next_seed = CONFIG["seed_start"] + len(all_results)
            print(f"  Resuming from seed {next_seed}.")
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"\n! Could not parse {OUT_JSON} — starting fresh.")
        all_results = []

for i in range(len(all_results), CONFIG["n_seeds"]):
    seed = CONFIG["seed_start"] + i
    print(f"\n{'='*70}\n=== Seed {seed}  ({i+1}/{CONFIG['n_seeds']}) ===\n{'='*70}")
    r = run_one_seed(seed, train_imgs, train_lbls, test_imgs, test_lbls, CONFIG)
    all_results.append(r)
    # Save after each seed so disconnect costs at most one seed
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  seed {seed} complete, results saved to {OUT_JSON}")


# Pretty results table
print("\n\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

corr_names = ["clean", "gaussian_noise", "shot_noise", "impulse_noise",
              "gaussian_blur", "pixelate", "jpeg", "contrast", "brightness"]
methods = list(all_results[0].keys())

# Average across seeds
agg = {m: {c: np.mean([r[m][c] for r in all_results]) for c in corr_names} for m in methods}
agg_std = {m: {c: np.std([r[m][c] for r in all_results]) for c in corr_names} for m in methods}
for m in methods:
    agg[m]["avg_corrupt"] = np.mean([agg[m][c] for c in corr_names if c != "clean"])
    agg_std[m]["avg_corrupt"] = np.std([np.mean([r[m][c] for c in corr_names if c != "clean"]) for r in all_results])

# Print mean ± std table
header = f"{'corruption':<16s}" + "".join(f"{m:>16s}" for m in methods)
print(header)
print("-" * len(header))
for c in corr_names + ["avg_corrupt"]:
    line = f"{c:<16s}"
    means = {m: agg[m][c] for m in methods}
    best = max(means.values())
    for m in methods:
        mu, sd = agg[m][c], agg_std[m][c]
        mark = "*" if mu >= best - 0.005 else " "
        line += f"  {mu:.3f}±{sd:.3f}{mark}"
    print(line)

# Deltas vs no_aug
if "fins" in methods and "no_aug" in methods:
    print(f"\n{'Δ vs no_aug':<16s}")
    for c in corr_names + ["avg_corrupt"]:
        d = agg["fins"][c] - agg["no_aug"][c]
        symbol = "better" if d > 0.005 else ("worse" if d < -0.005 else "tie")
        print(f"  {c:<16s}  {d:+.3f}  {symbol}")

# Controls: learned vs untrained generators (paired seeds), FINS vs the best baseline
def _avg(r):
    return float(np.mean([r[c] for c in corr_names if c != "clean"]))

for _null in ("null_init", "null_smooth"):
    if "fins" in methods and _null in methods:
        d = [_avg(r["fins"]) - _avg(r[_null]) for r in all_results]
        print(f"\nfins - {_null}, avg_corrupt per seed: " + " ".join(f"{x:+.3f}" for x in d)
              + f"; mean {np.mean(d):+.3f} +/- {np.std(d):.3f}; fins ahead on {sum(x > 0 for x in d)}/{len(d)}")
_base = [m for m in ("randaug", "autoaug", "trivialaug") if m in methods]
if "fins" in methods and _base:
    best = max(_base, key=lambda m: agg[m]["avg_corrupt"])
    g = [_avg(r["fins"]) - _avg(r[best]) for r in all_results]
    print(f"FINS - {best} (best baseline on average), avg_corrupt per seed: " + " ".join(f"{x:+.3f}" for x in g)
          + f"; mean {np.mean(g):+.3f} +/- {np.std(g):.3f}; FINS ahead on {sum(x > 0 for x in g)}/{len(g)}")

print(f"\nResults saved to {OUT_JSON}")
if CONFIG["save_sym_gen"]:
    print(f"Sym_gen weights : {SYM_GEN_DIR}")
print("\nTip: re-run the script anytime to resume — completed seeds are skipped.")
