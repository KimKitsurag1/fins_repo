"""Per-generator decomposition on CIFAR-10.

Loads a sym_gen checkpoint saved by fins_cifar.py and visualizes how each
of the six discovered generators acts on representative CIFAR-10 images.

Usage:
    pip install -r requirements.txt
    python fins_cifar.py                         # produces checkpoints
    python visualize_cifar_generators.py         # this script

By default the latest checkpoint under ./outputs/sym_gen_checkpoints/ is
used; override CONFIG["sym_gen_path"] to pick a specific seed.

Outputs (under ./outputs/cifar_viz/):
    per_generator_img{i}_{label}_tau{tau:.2f}.png   - one per image
    augmented_gallery_tau{lo}-{hi}.png              - single composed grid

"""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neuralop.models import FNO


# CONFIGURATION — edit these
CONFIG = {
    # --- sym_gen checkpoint ---
    "sym_gen_path":     None,     # None = auto-find latest in ckpt_dir below
    "ckpt_dir":         "./outputs/sym_gen_checkpoints",
    "N_sym":             6,

    # --- visualization parameters (match training) ---
    "tau":               0.8,     # for per-generator decomposition (matches tau_probe)
    "tau_range":        (0.5, 1.3), # for augmented gallery (matches augmentation)
    "n_active":          3,
    "n_aug":             5,       # augmented samples per image in gallery
    "n_images":          4,       # distinct CIFAR test images
    "seed":              42,      # for image selection (not method seed)

    # --- I/O ---
    "data_root":        "./data_cifar",
    "out_dir":          "./outputs/cifar_viz",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# Output paths + sym_gen checkpoint auto-discovery
os.makedirs(CONFIG["out_dir"], exist_ok=True)

if CONFIG["sym_gen_path"] is None:
    ckpt_dir = CONFIG["ckpt_dir"]
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"Directory {ckpt_dir} does not exist. Run fins_cifar.py first "
            f"with save_sym_gen=True, or set CONFIG['sym_gen_path'] manually.")
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No .pt files found in {ckpt_dir}. Run the training script first.")
    CONFIG["sym_gen_path"] = os.path.join(ckpt_dir, ckpts[0])
    print(f"Auto-selected checkpoint: {ckpts[0]}  ({len(ckpts)} available)")
    if len(ckpts) > 1:
        print(f"  (To use a different seed, set CONFIG['sym_gen_path'] explicitly.")
        print(f"   Available: {ckpts})")

print(f"Sym_gen checkpoint: {CONFIG['sym_gen_path']}")
print(f"Output dir:         {CONFIG['out_dir']}\n")


# Replicas of FNO + RK4 + encoder from fins_cifar.py
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


def create_normalized_grids(B, H, W, device):
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    xg, yg = torch.meshgrid(x, y, indexing="xy")
    return (xg.view(1, 1, H, W).expand(B, 1, -1, -1),
            yg.view(1, 1, H, W).expand(B, 1, -1, -1))


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
    warped = F.grid_sample(img, grid_s, mode="bilinear", padding_mode="border",
                            align_corners=True)
    return torch.clamp(warped + tau * phi, 0.0, 1.0)


# Visualization
def _to_disp(img):
    """(3, H, W) → (H, W, 3) numpy in [0, 1]."""
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _generator_energy(xi, eta, phi):
    e_geom = float((xi**2).mean() + (eta**2).mean())
    e_phi  = float((phi**2).mean())
    total = e_geom + e_phi + 1e-8
    return e_geom, e_phi, e_geom / total


def visualize_cifar_image(img: torch.Tensor, sym_gen: nn.Module,
                          tau: float, save_path: str, device: str,
                          upsample_for_display: int = 4):
    """img: (3, H, W) in [0, 1] — single CIFAR image at 32x32.
    Saves a (N_sym × 6) panel: [original | full | geom-only | phi-only | RGB diff | ξ/η arrows].
    """
    img_b = img.unsqueeze(0).to(device)
    B, _, H, W = img_b.shape
    xg, yg = create_normalized_grids(B, H, W, device)
    with torch.no_grad():
        gens = sym_gen(xg, yg, img_b)

    n_gens = len(gens)
    fig, axes = plt.subplots(n_gens, 6, figsize=(16, 2.8 * n_gens))
    if n_gens == 1:
        axes = axes[None, :]

    col_titles = ["original", "full flow", "geom-only", "phi-only",
                  "RGB diff", "ξ/η field"]

    # Upsample 32×32 to (32*4)=128 for display visibility
    def up(x_img):
        return F.interpolate(x_img.unsqueeze(0),
                              size=(H * upsample_for_display, W * upsample_for_display),
                              mode='nearest').squeeze(0)

    for k, (xi, eta, phi) in enumerate(gens):
        e_geom, e_phi, ratio = _generator_energy(xi, eta, phi)
        with torch.no_grad():
            full_img = apply_symmetry_rk4(xi, eta, phi, img_b, tau=tau)
            geom_img = apply_symmetry_rk4(xi, eta, phi, img_b, tau=tau, disable_phi=True)
            phi_img  = apply_symmetry_rk4(xi, eta, phi, img_b, tau=tau, disable_geom=True)

        axes[k, 0].imshow(_to_disp(up(img)))
        if k == 0:
            axes[k, 0].set_title(col_titles[0])
        axes[k, 0].set_ylabel(f"gen {k}\nratio={ratio:.2f}\n"
                                f"E_geom={e_geom:.4f}\nE_phi={e_phi:.4f}",
                              fontsize=8)
        axes[k, 0].set_xticks([]); axes[k, 0].set_yticks([])

        for c, x_img, title in [(1, full_img[0], col_titles[1]),
                                  (2, geom_img[0], col_titles[2]),
                                  (3, phi_img[0],  col_titles[3])]:
            axes[k, c].imshow(_to_disp(up(x_img)))
            if k == 0:
                axes[k, c].set_title(title)
            axes[k, c].axis("off")

        diff = (full_img[0] - img_b[0]).detach().cpu().permute(1, 2, 0).numpy()
        d_max = max(abs(diff.min()), abs(diff.max()), 1e-3)
        diff_disp = 0.5 + diff / (2 * d_max)
        diff_up = np.repeat(np.repeat(diff_disp, upsample_for_display, axis=0),
                            upsample_for_display, axis=1)
        axes[k, 4].imshow(np.clip(diff_up, 0, 1))
        title_text = col_titles[4] + f"\n[±{d_max:.2f}]" if k == 0 else f"[±{d_max:.2f}]"
        axes[k, 4].set_title(title_text, fontsize=8)
        axes[k, 4].axis("off")

        axes[k, 5].imshow(_to_disp(up(img)), alpha=0.4)
        stride = max(1, H // 12)
        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        u = xi[0, 0].detach().cpu().numpy()[::stride, ::stride]
        v = eta[0, 0].detach().cpu().numpy()[::stride, ::stride]
        # Scale grid coordinates to upsampled image space for visual alignment
        xs_up = xs * upsample_for_display
        ys_up = ys * upsample_for_display
        speed = np.sqrt(u**2 + v**2)
        axes[k, 5].quiver(xs_up, ys_up, u, v, speed, cmap="viridis",
                            scale=0.5, scale_units="xy")
        if k == 0:
            axes[k, 5].set_title(col_titles[5])
        axes[k, 5].set_xticks([]); axes[k, 5].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def visualize_random_augmented_gallery(images: torch.Tensor, sym_gen: nn.Module,
                                        tau_range, n_active, n_aug, save_path: str,
                                        device: str, upsample_for_display: int = 3):
    """Show grid of (n_images × (1 + n_aug)) = original + several augmented samples per image.
    Mimics what the downstream classifier actually sees as training data."""
    sym_gen.eval()
    img_b = images.to(device)
    B, _, H, W = img_b.shape
    xg, yg = create_normalized_grids(B, H, W, device)

    fig, axes = plt.subplots(B, n_aug + 1, figsize=(2.0 * (n_aug + 1), 2.0 * B))
    if B == 1:
        axes = axes[None, :]

    def up(x_img):
        return F.interpolate(x_img.unsqueeze(0),
                              size=(H * upsample_for_display, W * upsample_for_display),
                              mode='nearest').squeeze(0)

    with torch.no_grad():
        gens = sym_gen(xg, yg, img_b)

    rng = np.random.RandomState(0)
    for i in range(B):
        axes[i, 0].imshow(_to_disp(up(img_b[i])))
        axes[i, 0].set_title("original" if i == 0 else "", fontsize=10)
        axes[i, 0].axis("off")
        for j in range(n_aug):
            idxs = rng.choice(len(gens), size=n_active, replace=False)
            cur = img_b[i:i+1].clone()
            for idx in idxs:
                xi, eta, phi = gens[idx]
                tau = float(rng.uniform(*tau_range))
                if rng.rand() > 0.5: tau = -tau
                cur = apply_symmetry_rk4(xi[i:i+1], eta[i:i+1], phi[i:i+1],
                                          cur, tau=tau, n_steps=4)
            axes[i, j+1].imshow(_to_disp(up(cur[0])))
            axes[i, j+1].set_title(f"aug {j+1}" if i == 0 else "", fontsize=10)
            axes[i, j+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# Execute: load CIFAR + sym_gen checkpoint + run visualizations

# --- Load CIFAR-10 ---
print(f"Loading CIFAR-10 from {CONFIG['data_root']}...")
_tfm = transforms.Compose([transforms.ToTensor()])
_test = datasets.CIFAR10(root=CONFIG["data_root"], train=False, download=True, transform=_tfm)
_rng = np.random.RandomState(CONFIG["seed"])
_pick = _rng.choice(len(_test), size=CONFIG["n_images"], replace=False)
_images = torch.stack([_test[int(i)][0] for i in _pick])
_labels = [_test.classes[_test[int(i)][1]] for i in _pick]
print(f"  picked images: {_labels}")

# --- Load sym_gen ---
print(f"Loading sym_gen from {CONFIG['sym_gen_path']}...")
sym_gen = MultiSymmetryFNO(N_sym=CONFIG["N_sym"]).to(DEVICE)
# weights_only=False required: NeuralOp FNO state_dict stores activation function
# references (torch._C._nn.gelu) which PyTorch 2.6+ default-blocks.
_state = torch.load(CONFIG["sym_gen_path"], map_location=DEVICE, weights_only=False)
# NeuralOp FNO serialization quirk: pop _metadata key (it's supposed to be an
# attribute of the OrderedDict, not a member key — but gets serialized as one).
_state.pop("_metadata", None)
sym_gen.load_state_dict(_state)
sym_gen.eval()

# --- Per-generator decomposition (one figure per image) ---
print(f"\nGenerating per-generator decompositions at τ={CONFIG['tau']}...")
for _i, (_img, _lbl) in enumerate(zip(_images, _labels)):
    _save_path = os.path.join(CONFIG["out_dir"],
                                f"per_generator_img{_i}_{_lbl}_tau{CONFIG['tau']:.2f}.png")
    visualize_cifar_image(_img, sym_gen, CONFIG["tau"], _save_path, DEVICE)

# --- Augmented gallery (single combined figure) ---
_lo, _hi = CONFIG["tau_range"]
print(f"\nGenerating augmented gallery (n_aug={CONFIG['n_aug']}, "
      f"tau∈[{_lo}, {_hi}], n_active={CONFIG['n_active']})...")
_gallery_path = os.path.join(CONFIG["out_dir"],
                              f"augmented_gallery_tau{_lo}-{_hi}.png")
visualize_random_augmented_gallery(_images, sym_gen,
                                    tuple(CONFIG["tau_range"]),
                                    CONFIG["n_active"], CONFIG["n_aug"],
                                    _gallery_path, DEVICE)

print(f"\nDone. {CONFIG['n_images'] + 1} figures saved to {CONFIG['out_dir']}/")
