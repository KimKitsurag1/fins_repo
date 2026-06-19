"""
FINS on Oxford-IIIT Pet

Reproduces the Pet-domain ablation reported in Section 6 of the paper.
Trains a six-way breed classifier (six cat breeds, all with >=30 examples)
under five augmentation regimes (NoAug / RandAug / AutoAug / TrivialAug /
FINS) and evaluates each on nine corruption types.

Method-side parameters (FNO size, Olver-loss formulation, augmentation
strength) match the v5/v6 version used to produce the Pet ablation tables;
the CIFAR script (fins_cifar.py) uses the upgraded v7 method described in
Section 6 of the paper.

Usage:
    pip install -r requirements.txt
    python fins_pet.py

Parameters are configured in the CONFIG dict below. Auto-resumes from
saved results and checkpoints if interrupted; simply re-run to continue.

Outputs (under ./outputs/):
    fins_pet_results.json       - per-seed accuracy across all methods
    sym_gen_pet_checkpoints/*.pt - FNO generator weights, one per seed

Runtime: ~25 min per seed on A100/V100; ~1 h on T4. 10 seeds default.

"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO


# CONFIGURATION
CONFIG = {
    # Experiment scale
    "n_seeds":          10,          # 10 seeds for full ablation reported in paper
    "seed_start":       100,
    "smoke_test":       False,        # True -> only no_aug + fins (skip baselines)

    # FINS hyperparameters (Pet ablation: v5/v6 setup)
    "N_sym":            6,            # number of generators to discover
    "w_ortho":          1.0,          # orthogonality regularizer
    "w_lips":           0.01,         # spatial smoothness
    "tau_range":        (0.3, 0.8),   # augmentation tau sampling range

    # Architecture
    "image_size":       256,
    "n_breeds":         6,            # uses 6 breeds with >=30 train examples
    "fno_hidden":       32,
    "fno_modes":        12,
    "fno_layers":       4,

    # Training
    "epochs_classifier":     15,      # cat-vs-dog classifier for Olver target
    "epochs_sym_gen":        15,      # symmetry-generator training
    "epochs_breed_resnet":   15,      # downstream breed classifier
    "batch_size_train":      4,
    "batch_size_resnet":     16,
    "n_cls_train":           300,     # cats/dogs each for the binary classifier
    "n_aug":                 5,       # augmented copies per image (6x total set)

    # I/O
    "out_dir":          "./outputs",
    "out_json":         "fins_pet_results.json",
    "save_sym_gen":     True,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# Output paths
OUT_DIR = CONFIG["out_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

OUT_JSON     = os.path.join(OUT_DIR, CONFIG["out_json"])
SYM_GEN_DIR  = os.path.join(OUT_DIR, "sym_gen_pet_checkpoints")
if CONFIG["save_sym_gen"]:
    os.makedirs(SYM_GEN_DIR, exist_ok=True)

print(f"Results JSON  : {OUT_JSON}")
if CONFIG["save_sym_gen"]:
    print(f"Sym_gen ckpts : {SYM_GEN_DIR}")


# Reproducibility
def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# Corruptions (parametric — kernel sizes tuned for 256x256)
def apply_gaussian_noise(img, severity=0.15):
    return torch.clamp(img + severity * torch.randn_like(img), 0, 1)


def apply_shot_noise(img, severity=0.15):
    return torch.clamp(img + severity * torch.randn_like(img) * torch.sqrt(img + 1e-3), 0, 1)


def apply_impulse_noise(img, severity=0.10):
    mask = torch.rand_like(img)
    out = img.clone()
    out[mask < severity / 2] = 0
    out[mask > 1 - severity / 2] = 1
    return out


def apply_gaussian_blur(img, ksize=7):
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    return F.avg_pool2d(F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"),
                        ksize, stride=1).squeeze(0)


def apply_contrast(img, factor=0.4):
    m = img.mean()
    return torch.clamp((img - m) * factor + m, 0, 1)


def apply_brightness(img, delta=0.3):
    return torch.clamp(img + delta, 0, 1)


def apply_jpeg_compression(img, quality=10):
    levels = max(2, quality // 5)
    return (img * levels).round() / levels


def apply_pixelate(img, block_size=8):
    C, H, W = img.shape
    small = F.interpolate(img.unsqueeze(0), size=(H // block_size, W // block_size),
                          mode="nearest")
    return F.interpolate(small, size=(H, W), mode="nearest").squeeze(0)


CORRUPTIONS = {
    "clean":           lambda x: x,
    "gaussian_noise":  lambda x: apply_gaussian_noise(x, 0.15),
    "shot_noise":      lambda x: apply_shot_noise(x, 0.15),
    "impulse_noise":   lambda x: apply_impulse_noise(x, 0.10),
    "gaussian_blur":   lambda x: apply_gaussian_blur(x, 7),
    "contrast":        lambda x: apply_contrast(x, 0.4),
    "brightness":      lambda x: apply_brightness(x, 0.3),
    "jpeg":            lambda x: apply_jpeg_compression(x, 10),
    "pixelate":        lambda x: apply_pixelate(x, 8),
}


# Models
class CoordinateClassifier(nn.Module):
    """ResNet-18 with appended coordinate channels (x, y).
    Used as the Olver-loss target classifier."""
    def __init__(self, image_size=256):
        super().__init__()
        model = models.resnet18(weights="IMAGENET1K_V1")
        conv1 = model.conv1
        new_conv1 = nn.Conv2d(5, conv1.out_channels, kernel_size=conv1.kernel_size,
                               stride=conv1.stride, padding=conv1.padding,
                               bias=conv1.bias is not None)
        with torch.no_grad():
            new_conv1.weight[:, :3] = conv1.weight
            new_conv1.weight[:, 3:] = conv1.weight[:, :2] * 0.1
        model.conv1 = new_conv1
        model.fc = nn.Linear(model.fc.in_features, 1)
        self.model = model
        x = torch.linspace(0, 1, image_size)
        y = torch.linspace(0, 1, image_size)
        xg, yg = torch.meshgrid(x, y, indexing="xy")
        self.register_buffer("x_grid", xg.unsqueeze(0))
        self.register_buffer("y_grid", yg.unsqueeze(0))

    def forward(self, img):
        B = img.shape[0]
        x_coords = self.x_grid.expand(B, 1, -1, -1)
        y_coords = self.y_grid.expand(B, 1, -1, -1)
        return self.model(torch.cat([img, x_coords, y_coords], dim=1))


class MultiSymmetryFNO(nn.Module):
    def __init__(self, hidden_channels=32, n_modes=12, n_layers=4, N_sym=6):
        super().__init__()
        self.N_sym = N_sym
        self.fno = FNO(n_modes=(n_modes, n_modes), hidden_channels=hidden_channels,
                       in_channels=5, out_channels=5 * N_sym, n_layers=n_layers,
                       use_channel_mlp=True, channel_mlp_dropout=0.1)

    def forward(self, x_grid, y_grid, img):
        out = self.fno(torch.cat([x_grid, y_grid, img], dim=1))
        return [(out[:, 5*i + 0:5*i + 1], out[:, 5*i + 1:5*i + 2],
                  out[:, 5*i + 2:5*i + 5]) for i in range(self.N_sym)]


def create_grids_image(B, H, W, device):
    x = torch.linspace(0, 1, W, device=device)
    y = torch.linspace(0, 1, H, device=device)
    xg, yg = torch.meshgrid(x, y, indexing="xy")
    return (xg.view(1, 1, H, W).expand(B, 1, -1, -1),
            yg.view(1, 1, H, W).expand(B, 1, -1, -1))


def apply_symmetry_image(xi, eta, phi, img, tau=0.5):
    """Single-step Euler integration of the spatial flow + additive amplitude."""
    B, C, H, W = img.shape
    x_px = torch.linspace(-1, 1, W, device=img.device).view(1, 1, 1, -1).expand(B, 1, H, -1)
    y_px = torch.linspace(-1, 1, H, device=img.device).view(1, 1, -1, 1).expand(B, 1, -1, W)
    grid = torch.cat([(x_px + tau * xi).permute(0, 2, 3, 1),
                      (y_px + tau * eta).permute(0, 2, 3, 1)], dim=-1)
    img_warped = F.grid_sample(img, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
    return torch.clamp(img_warped + tau * phi, 0.0, 1.0)


# Losses
def olver_loss_cats(xi, eta, phi, classifier, img, device):
    """Olver condition via gradients of the binary cat-vs-dog classifier."""
    B, C, H, W = img.shape
    img = img.requires_grad_(True)
    x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, -1).expand(B, 1, H, -1)
    y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, -1, 1).expand(B, 1, -1, W)
    inputs = torch.cat([img, x_coords, y_coords], dim=1).requires_grad_(True)
    L = classifier.model(inputs).sum()
    grads = torch.autograd.grad(L, inputs, create_graph=True)[0]
    v_f = (xi * grads[:, 3:4]
           + eta * grads[:, 4:5]
           + (phi * grads[:, :3]).sum(dim=1, keepdim=True))
    return (v_f ** 2).mean()


def ortho_loss(generators):
    """Pairwise orthogonality between generators."""
    loss = 0.0
    pairs = 0
    for i in range(len(generators)):
        for j in range(i + 1, len(generators)):
            xi1, eta1, phi1 = generators[i]
            xi2, eta2, phi2 = generators[j]
            ip = (xi1 * xi2).mean() + (eta1 * eta2).mean() + (phi1 * phi2).mean()
            loss = loss + ip ** 2
            pairs += 1
    return loss / max(1, pairs)


def lipschitz_loss_cats(generators):
    loss = 0.0
    for (xi, eta, phi) in generators:
        v = torch.cat([xi, eta, phi], dim=1)
        loss = loss + (v[:, :, :, 2:] - v[:, :, :, :-2]).pow(2).mean()
        loss = loss + (v[:, :, 2:, :] - v[:, :, :-2, :]).pow(2).mean()
    return loss


# Dataset preparation
CAT_BREEDS = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
              "Russian_Blue", "Siamese", "Sphynx"]


def load_oxford_pet(image_size: int):
    """Load Oxford-IIIT Pet with four augmentation transforms and split into
    cat/dog index lists + per-breed index dict."""
    base_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    randaug_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
    ])
    autoaug_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
    ])
    trivialaug_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
    ])

    print("Loading Oxford-IIIT Pet ...")
    full = OxfordIIITPet(root="./data_pet", split="trainval",
                          target_types="category", transform=base_tfm,
                          download=True)
    full_rand     = OxfordIIITPet(root="./data_pet", split="trainval",
                                   target_types="category",
                                   transform=randaug_tfm, download=False)
    full_auto     = OxfordIIITPet(root="./data_pet", split="trainval",
                                   target_types="category",
                                   transform=autoaug_tfm, download=False)
    full_trivial  = OxfordIIITPet(root="./data_pet", split="trainval",
                                   target_types="category",
                                   transform=trivialaug_tfm, download=False)

    cat_indices: List[int] = []
    dog_indices: List[int] = []
    print("Sorting cat/dog indices ...")
    for i in range(len(full)):
        _, label = full[i]
        class_name = full.classes[label].replace(" ", "_")
        if class_name in CAT_BREEDS:
            cat_indices.append(i)
        else:
            dog_indices.append(i)

    breed_images: Dict[str, List[int]] = {breed: [] for breed in CAT_BREEDS}
    for i in cat_indices:
        _, label = full[i]
        class_name = full.classes[label].replace(" ", "_")
        if class_name in breed_images:
            breed_images[class_name].append(i)

    print(f"  Cats: {len(cat_indices)}, Dogs: {len(dog_indices)}")
    return (full, full_rand, full_auto, full_trivial,
            cat_indices, dog_indices, breed_images)


# Per-seed driver
def run_one_seed(seed: int, datasets_tuple, cfg: dict) -> Dict[str, Dict[str, float]]:
    (full, full_rand, full_auto, full_trivial,
     cat_indices, dog_indices, breed_images) = datasets_tuple

    seed_all(seed)
    image_size = cfg["image_size"]

    # ---- 1. Breed split (80/20) ----
    breeds_list = [b for b in breed_images.keys()
                   if len(breed_images[b]) >= 30][:cfg["n_breeds"]]

    train_data: List[torch.Tensor] = []
    train_labels: List[int] = []
    test_data_clean: List[torch.Tensor] = []
    test_labels: List[int] = []
    train_indices_for_aug: List[Tuple[int, int]] = []  # for RandAug/AutoAug/TrivialAug

    for i, breed in enumerate(breeds_list):
        indices = breed_images[breed].copy()
        np.random.shuffle(indices)
        n_train = int(len(indices) * 0.8)
        train_inds = indices[:n_train]
        test_inds  = indices[n_train:]
        for idx in train_inds:
            img, _ = full[idx]
            train_data.append(img)
            train_labels.append(i)
            train_indices_for_aug.append((idx, i))
        for idx in test_inds:
            img, _ = full[idx]
            test_data_clean.append(img)
            test_labels.append(i)

    train_images = torch.stack(train_data)
    train_labels_t = torch.tensor(train_labels)
    test_images_clean = torch.stack(test_data_clean)
    test_labels_t = torch.tensor(test_labels)
    print(f"  Train: {len(train_images)} images, Test: {len(test_images_clean)} images")

    # ---- 2. Train cat-vs-dog binary classifier (Olver-loss target) ----
    print("  Training cat-vs-dog classifier (Olver target) ...")
    n_cls_train = cfg["n_cls_train"]
    cat_shuf = cat_indices.copy()
    dog_shuf = dog_indices.copy()
    np.random.shuffle(cat_shuf)
    np.random.shuffle(dog_shuf)
    cls_train_indices = cat_shuf[:n_cls_train] + dog_shuf[:n_cls_train]
    cls_train_labels  = [1] * n_cls_train + [0] * n_cls_train
    cls_dataset = [(full[idx][0], lbl) for idx, lbl in zip(cls_train_indices, cls_train_labels)]
    cls_loader = DataLoader(cls_dataset, batch_size=cfg["batch_size_train"], shuffle=True)

    classifier = CoordinateClassifier(image_size=image_size).to(DEVICE)
    opt_cls = optim.Adam(classifier.parameters(), lr=1e-4)
    for epoch in range(cfg["epochs_classifier"]):
        classifier.train()
        for img, lbl in cls_loader:
            img = img.to(DEVICE)
            lbl = torch.tensor(lbl, dtype=torch.float32).view(-1, 1).to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(classifier(img), lbl)
            opt_cls.zero_grad(); loss.backward(); opt_cls.step()
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    # ---- 3. Train or load symmetry generator ----
    sym_gen_path = (os.path.join(SYM_GEN_DIR, f"sym_gen_pet_seed{seed}.pt")
                    if cfg["save_sym_gen"] else None)
    sym_gen = MultiSymmetryFNO(hidden_channels=cfg["fno_hidden"],
                                n_modes=cfg["fno_modes"],
                                n_layers=cfg["fno_layers"],
                                N_sym=cfg["N_sym"]).to(DEVICE)

    if sym_gen_path and os.path.exists(sym_gen_path):
        print(f"  Found cached sym_gen at {sym_gen_path} - loading ...")
        _state = torch.load(sym_gen_path, map_location=DEVICE, weights_only=False)
        _state.pop("_metadata", None)
        sym_gen.load_state_dict(_state)
        sym_gen.eval()
    else:
        print(f"  Training symmetry generator ({cfg['epochs_sym_gen']} epochs) ...")
        sym_loader = DataLoader(TensorDataset(train_images, train_labels_t),
                                  batch_size=cfg["batch_size_train"], shuffle=True)
        opt_sym = optim.Adam(sym_gen.parameters(), lr=1e-4)
        for epoch in range(cfg["epochs_sym_gen"]):
            sym_gen.train()
            for img, _ in sym_loader:
                img = img.to(DEVICE)
                B, C, H, W = img.shape
                xg, yg = create_grids_image(B, H, W, DEVICE)
                generators = sym_gen(xg, yg, img)
                l_olv = sum(olver_loss_cats(xi, eta, phi, classifier, img, DEVICE)
                            for (xi, eta, phi) in generators)
                l_or  = ortho_loss(generators)
                l_lp  = lipschitz_loss_cats(generators)
                loss = l_olv + cfg["w_ortho"] * l_or + cfg["w_lips"] * l_lp
                opt_sym.zero_grad(); loss.backward(); opt_sym.step()
        sym_gen.eval()
        if sym_gen_path:
            torch.save(sym_gen.state_dict(), sym_gen_path)
            print(f"  Saved sym_gen to {sym_gen_path}")

    # ---- 4. Helpers for downstream classifier ----
    def train_breed_classifier(imgs: torch.Tensor, lbls: torch.Tensor,
                                epochs: int) -> nn.Module:
        cls = models.resnet18(weights="IMAGENET1K_V1")
        cls.fc = nn.Linear(512, len(breeds_list))
        cls = cls.to(DEVICE)
        opt = optim.Adam(cls.parameters(), lr=1e-4)
        loader = DataLoader(TensorDataset(imgs, lbls),
                             batch_size=cfg["batch_size_resnet"], shuffle=True)
        for _ in range(epochs):
            cls.train()
            for img, lbl in loader:
                loss = F.cross_entropy(cls(img.to(DEVICE)), lbl.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        return cls

    @torch.no_grad()
    def evaluate_on_corruptions(cls: nn.Module) -> Dict[str, float]:
        cls.eval()
        res: Dict[str, float] = {}
        for name, fn in CORRUPTIONS.items():
            corrupted = torch.stack([fn(img) for img in test_images_clean])
            preds = cls(corrupted.to(DEVICE)).argmax(dim=1)
            res[name] = (preds == test_labels_t.to(DEVICE)).float().mean().item()
        res["corrupted_avg"] = float(np.mean([v for k, v in res.items() if k != "clean"]))
        return res

    # ---- 5. Train and evaluate each method ----
    results: Dict[str, Dict[str, float]] = {}

    # No Augmentation
    print("  [1/5] Training NoAug classifier ...")
    cls_no_aug = train_breed_classifier(train_images, train_labels_t,
                                          epochs=cfg["epochs_breed_resnet"])
    results["no_aug"] = evaluate_on_corruptions(cls_no_aug)

    # FINS Augmentation (composition of all generators with random tau + permutation)
    print(f"  [2/5] Building FINS augmented set (n_aug={cfg['n_aug']}) ...")
    aug_list = [train_images]
    sym_gen.eval()
    tau_lo, tau_hi = cfg["tau_range"]
    with torch.no_grad():
        for _ in range(cfg["n_aug"]):
            for i in range(len(train_images)):
                img = train_images[i:i+1].to(DEVICE)
                B, _, H, W = img.shape
                xg, yg = create_grids_image(B, H, W, DEVICE)
                generators = sym_gen(xg, yg, img)
                img_aug = img.clone()
                for idx in np.random.permutation(len(generators)):
                    xi, eta, phi = generators[idx]
                    tau = float(np.random.uniform(tau_lo, tau_hi))
                    if np.random.rand() > 0.5:
                        tau = -tau
                    img_aug = apply_symmetry_image(xi, eta, phi, img_aug, tau=tau)
                aug_list.append(img_aug.cpu())
    aug_fins_images = torch.cat(aug_list, dim=0)
    aug_fins_labels = train_labels_t.repeat(cfg["n_aug"] + 1)
    print(f"    augmented size: {len(aug_fins_images)}")
    print("  Training FINS classifier ...")
    cls_fins = train_breed_classifier(aug_fins_images, aug_fins_labels,
                                        epochs=cfg["epochs_breed_resnet"])
    results["fins"] = evaluate_on_corruptions(cls_fins)
    del aug_fins_images
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if cfg["smoke_test"]:
        return results

    # Standard image augmentations (RandAug / AutoAug / TrivialAug)
    def build_augmented_set(transformed_dataset, n_aug: int):
        out_imgs = list(train_data)
        out_lbls = list(train_labels)
        for _ in range(n_aug):
            for idx, lbl in train_indices_for_aug:
                img, _ = transformed_dataset[idx]
                out_imgs.append(img)
                out_lbls.append(lbl)
        return torch.stack(out_imgs), torch.tensor(out_lbls)

    print("  [3/5] Building RandAug augmented set ...")
    aug_imgs, aug_lbls = build_augmented_set(full_rand, cfg["n_aug"])
    print("  Training RandAug classifier ...")
    cls = train_breed_classifier(aug_imgs, aug_lbls, epochs=cfg["epochs_breed_resnet"])
    results["randaug"] = evaluate_on_corruptions(cls)
    del aug_imgs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("  [4/5] Building AutoAug augmented set ...")
    aug_imgs, aug_lbls = build_augmented_set(full_auto, cfg["n_aug"])
    print("  Training AutoAug classifier ...")
    cls = train_breed_classifier(aug_imgs, aug_lbls, epochs=cfg["epochs_breed_resnet"])
    results["autoaug"] = evaluate_on_corruptions(cls)
    del aug_imgs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("  [5/5] Building TrivialAug augmented set ...")
    aug_imgs, aug_lbls = build_augmented_set(full_trivial, cfg["n_aug"])
    print("  Training TrivialAug classifier ...")
    cls = train_breed_classifier(aug_imgs, aug_lbls, epochs=cfg["epochs_breed_resnet"])
    results["trivialaug"] = evaluate_on_corruptions(cls)
    del aug_imgs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


# Load dataset once and reuse across seeds
print("\n=== Loading Oxford-IIIT Pet ===")
datasets_tuple = load_oxford_pet(CONFIG["image_size"])


# Run experiment (with auto-resume from saved results / checkpoints)
all_results: List[Dict[str, Dict[str, float]]] = []
if os.path.exists(OUT_JSON):
    try:
        with open(OUT_JSON) as f:
            all_results = json.load(f)
        print(f"\nFound existing results: {len(all_results)} seed(s) already completed.")
        if len(all_results) >= CONFIG["n_seeds"]:
            print(f"  All {CONFIG['n_seeds']} seeds already done.")
        else:
            print(f"  Resuming from seed {CONFIG['seed_start'] + len(all_results)}.")
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"\nCould not parse {OUT_JSON} - starting fresh.")
        all_results = []

for i in range(len(all_results), CONFIG["n_seeds"]):
    seed = CONFIG["seed_start"] + i
    print(f"\n{'='*70}\n=== Seed {seed}  ({i+1}/{CONFIG['n_seeds']}) ===\n{'='*70}")
    r = run_one_seed(seed, datasets_tuple, CONFIG)
    all_results.append(r)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  seed {seed} complete, results saved to {OUT_JSON}")
    for method, accs in r.items():
        print(f"    {method:<12s}  clean={accs['clean']:.3f}  "
              f"blur={accs['gaussian_blur']:.3f}  "
              f"avg_corr={accs['corrupted_avg']:.3f}")


# Pretty results summary
print("\n\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

corr_names = ["clean", "gaussian_noise", "shot_noise", "impulse_noise",
              "gaussian_blur", "pixelate", "jpeg", "contrast", "brightness"]
methods = list(all_results[0].keys()) if all_results else []

if all_results:
    agg     = {m: {c: np.mean([r[m][c] for r in all_results]) for c in corr_names + ["corrupted_avg"]}
               for m in methods}
    agg_std = {m: {c: np.std([r[m][c] for r in all_results])  for c in corr_names + ["corrupted_avg"]}
               for m in methods}

    header = f"{'corruption':<16s}" + "".join(f"{m:>16s}" for m in methods)
    print(header)
    print("-" * len(header))
    for c in corr_names + ["corrupted_avg"]:
        line = f"{c:<16s}"
        means = {m: agg[m][c] for m in methods}
        best = max(means.values())
        for m in methods:
            mu, sd = agg[m][c], agg_std[m][c]
            mark = "*" if mu >= best - 0.005 else " "
            line += f"  {mu:.3f}+/-{sd:.3f}{mark}"
        print(line)

    if "fins" in methods and "no_aug" in methods:
        print(f"\n{'Delta vs no_aug:':<16s}")
        for c in corr_names + ["corrupted_avg"]:
            d = agg["fins"][c] - agg["no_aug"][c]
            symbol = "+" if d > 0.005 else ("-" if d < -0.005 else "~")
            print(f"  {c:<16s}  {d:+.3f}  {symbol}")

print(f"\nResults saved to {OUT_JSON}")
if CONFIG["save_sym_gen"]:
    print(f"Sym_gen weights : {SYM_GEN_DIR}")
print("\nTip: re-run the script anytime to resume - completed seeds are skipped.")
