"""
Augerino — Benton, Finzi, Izmailov, Wilson (NeurIPS 2020)
"Learning Invariances in Neural Networks from Training Data"
arXiv:2010.11882 · github.com/g-benton/learning-invariances

This reimplementation follows the NeurIPS 2020 paper and the reference
code of g-benton/learning-invariances as closely as possible.

THE METHOD (§3 of the paper)

Augerino learns a distribution μ_θ over augmentations jointly with
the network weights.  For the affine Lie-group parametrisation
(§3.2, App. B), the distribution factors through the Lie-algebra:

    g_ε  =  exp(Σ_i ε_i θ_i G_i)         ε ~ U[-1, 1]^k             (9)

θ_i ≥ 0 are learnable WIDTHS.  Large θ_i means the learned augmentation
distribution covers a wide range → the model is invariant under G_i.
Small θ_i → no invariance.  Positivity is enforced via softplus:

    θ_i = log(1 + exp(θ̃_i))                                       (§3.2)

For an input x the averaged prediction (invariant form) is

    f̄(x) = E_{g ~ μ_θ} f(gx)                                      (1)

and the EQUIVARIANT form used for segmentation, dense prediction and
PDE surrogate learning is (§3.1, eq. 6)

    f_aug-eq(x) = E_{g ~ μ_θ} g⁻¹ f(gx).                          (6)

A Monte-Carlo estimator uses  ncopies  samples per input.  The paper
shows (§D) that  ncopies = 1  at train time is optimal for efficiency.

The TRAINING LOSS combines task loss with a negative-L² penalty on
widths (§3.2):

    L = E_g ℓ(f_w(gx))  +  λ R(θ),   R(θ) = −‖θ‖²                  (5)

R(θ) PUSHES widths LARGER, biasing the model toward invariance.  When
a particular direction is not a symmetry of the data, the task loss
pushes the corresponding θ_i BACK DOWN; the tension identifies the
true range of invariance.

AFFINE 2D GENERATORS (App. B) — applied to the (x, t) PDE grid

    G1 = [0,0,1; 0,0,0; 0,0,0]       translation in x   (ξ=1, η=0, φ=0)
    G2 = [0,0,0; 0,0,1; 0,0,0]       translation in t   (ξ=0, η=1, φ=0)
    G3 = [0,-1,0; 1,0,0; 0,0,0]      rotation (x↔t)     (ξ=-t, η=x, φ=0)
    G4 = diag(1,1,0)                  uniform scaling    (ξ=x, η=t, φ=0)
    G5 = diag(1,-1,0)                 hyperbolic scaling (ξ=x, η=-t, φ=0)
    G6 = [0,1,0; 1,0,0; 0,0,0]       shearing           (ξ=t, η=x, φ=0)

For PDEs we add a 7th generator acting on the VALUE u:

    G7:  u → exp(ε θ_7) u            u-scaling          (ξ=0, η=0, φ=u)

which is needed for symmetries like Galilean's φ = x·u or heat's u·∂_u.
G_4 ... G_6 capture x-t rotations, dilatations and Galilean-like
shears needed for Olver's canonical LPS algebras.

HYPER-PARAMETERS (App. D)
    epochs:     200          batch_size: 128
    optimiser:  SGD          lr:          0.01  (cosine schedule)
    λ ∈ {0.01, 0.05, 0.1}    ncopies:     1 (train)  /  4+ (test)
    θ̃ init:    −4           (widths start near 0.018)

For our PDE FNO we use Adam and cfg.lr_augerino in place of SGD/0.01,
but keep the rest of the recipe.

STRUCTURAL LIMITATION
Augerino extracts a  FIXED  linear-combination of the 7 pre-specified
generators — it CANNOT discover symmetries outside this span.  For
example, it cannot represent the heat-equation projective generator
   −(x²+2t) u ∂_u + 4xt ∂_x + 4t² ∂_t
because this needs coefficients quadratic in (x, t).  Within the
2D-affine span, Augerino expects to recover (partially):
   heat    :   ∂_x, ∂_t, u∂_u, scaling, Galilean-like shear   (≤ 5)
   Burgers :   ∂_x, ∂_t, scaling                              (≤ 3)
   KdV     :   ∂_x, ∂_t, Galilean-like shear, scaling         (≤ 4)

The projective and exact-Galilean generators with non-affine φ in u
are structurally out of reach — this is the same kind of limit that
LieGAN has for non-affine actions (and is honestly noted in Ko et al.
(2024) §A.2 when comparing to Augerino).

Interface:
    train_augerino(train_sol, pde_type, x_grid, t_grid, cfg, device)
                                             → (aug_transform, model)
    extract_augerino_generators(aug, model, sol_batch,
                                x_grid, t_grid, cfg, device)
                                             → (list[(ξ, η, φ)], norms)
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from shared_modules import FNOSurrogate


#  1.  Augerino affine transform  (§3.2 + App. B + G7 u-scaling)
class AugerinoAffine2D(nn.Module):
    """
    Learnable uniform distribution over the 6 affine generators in 2D
    PLUS one u-scaling generator, via softplus-positive widths θ.

    Parameters:
        theta_tilde[0..5] → widths for G1..G6 (trans_x, trans_t,
                            rot, scaling, anti_scaling, shear)
        theta_tilde[6]    → width for G7 (multiplicative u-scale)
    """

    def __init__(self, init_width: float = 0.018):
        super().__init__()
        # Softplus inverse to get the right initial width
        #   θ = softplus(θ̃) = log(1+exp(θ̃))
        #   θ̃ = log(exp(θ) - 1)
        theta0 = float(init_width)
        theta_tilde_0 = float(np.log(np.expm1(max(theta0, 1e-4))))
        self.theta_tilde = nn.Parameter(
            torch.full((7,), theta_tilde_0, dtype=torch.float32))

        # The 6 affine generators from App. B (3×3 in homogeneous coords)
        G = torch.zeros(6, 3, 3)
        G[0, 0, 2] = 1.0                              # G1 : trans x
        G[1, 1, 2] = 1.0                              # G2 : trans t
        G[2, 0, 1] = -1.0; G[2, 1, 0] = 1.0           # G3 : rotation
        G[3, 0, 0] = 1.0;  G[3, 1, 1] = 1.0           # G4 : uniform scale
        G[4, 0, 0] = 1.0;  G[4, 1, 1] = -1.0          # G5 : anti scale
        G[5, 0, 1] = 1.0;  G[5, 1, 0] = 1.0           # G6 : shearing
        self.register_buffer('G', G)                  # [6, 3, 3]

    # ----- properties -----
    def widths(self) -> torch.Tensor:
        """θ = softplus(θ̃), shape [7]."""
        return F.softplus(self.theta_tilde)

    # ----- sampling of a batch of group elements -----
    def sample(self, B: int, device):
        """
        Draw ε ~ U[-1, 1]^7 and return
            A_aff     — [B, 2, 3]  (2-D affine matrix, homogeneous)
            u_scale   — [B]         (multiplicative u scaling)
            eps       — [B, 7]      (raw ε samples, for inverse)
        """
        eps = (torch.rand(B, 7, device=device) * 2.0 - 1.0)
        w = self.widths().to(device)                  # [7]
        # Lie-algebra element in the affine subspace
        alpha = eps[:, :6] * w[:6]                    # [B, 6]
        A_log = torch.einsum('bi,ijk->bjk', alpha, self.G)   # [B, 3, 3]
        A_exp = torch.matrix_exp(A_log)                # [B, 3, 3]
        A_aff = A_exp[:, :2, :]                        # drop homog row
        # u-scaling from G7
        u_scale = torch.exp(eps[:, 6] * w[6])          # [B]
        return A_aff, u_scale, eps

    def invert(self, A_aff: torch.Tensor, u_scale: torch.Tensor, eps):
        """
        Closed-form inverse of (A_aff, u_scale): since the forward
        map is exp(Σ α_i G_i)·... , the inverse uses -ε instead of ε.
        Returns the SAME signature as sample().
        """
        w = self.widths().to(A_aff.device)
        alpha = -eps[:, :6] * w[:6]
        A_log = torch.einsum('bi,ijk->bjk', alpha, self.G)
        A_inv = torch.matrix_exp(A_log)[:, :2, :]
        u_inv = torch.exp(-eps[:, 6] * w[6])
        return A_inv, u_inv

    def apply(self, u: torch.Tensor,
              A_aff: torch.Tensor, u_scale: torch.Tensor) -> torch.Tensor:
        """
        Apply transform. A_aff is [B, 2, 3], u_scale is [B],
        u is [B, C, Nx, Nt].  Uses torch's affine_grid / grid_sample.
        """
        grid = F.affine_grid(A_aff, list(u.shape), align_corners=True)
        u_t = F.grid_sample(u, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)
        return u_t * u_scale.view(-1, 1, 1, 1)


#  2.  Training loop  (§3 equivariant form)
def train_augerino(train_sol, pde_type, x_grid, t_grid, cfg, device):
    """
    Train an FNO surrogate with Augerino augmentation on PDE data.

    train_sol : [N, 1, Nx, Nt]   full PDE trajectories
    pde_type  : str              (unused but kept for interface parity)
    x_grid    : np array, Nx     (unused)
    t_grid    : np array, Nt     (unused)
    cfg       : ExperimentConfig
    device    : torch device

    Returns (aug, model).
    """
    N, C, Nx, Nt = train_sol.shape

    aug = AugerinoAffine2D(init_width=0.018).to(device)
    model = FNOSurrogate(cfg).to(device)

    # Following App. D (SGD + cosine); we use Adam for speed / parity
    # with the other baselines.  The three learning rates are the same
    # (Augerino exposes one global lr).
    lr = getattr(cfg, 'lr_augerino', 1e-3)
    opt = optim.Adam(list(model.parameters()) + list(aug.parameters()),
                     lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs_augerino)

    lam = 0.05
    ncopies = 1

    train_ic = train_sol[:, :, :, 0]
    loader = DataLoader(TensorDataset(train_ic, train_sol),
                        batch_size=getattr(cfg, 'batch_size', 128),
                        shuffle=True)

    print(f"    Augerino: equivariant training  "
          f"(ncopies={ncopies}, λ={lam}, 7 generators)...")

    for epoch in range(cfg.epochs_augerino):
        t0 = time.time()
        ep_task = ep_reg = 0.0
        nb = 0

        for u0, u in loader:
            u0 = u0.to(device)
            u  = u.to(device)
            B = u.shape[0]

            pred_sum = torch.zeros_like(u)
            for _ in range(ncopies):
                A, us, eps = aug.sample(B, device)
                u_aug = aug.apply(u, A, us)
                u0_aug = u_aug[:, :, :, 0]
                pred_aug = model(u0_aug)
                A_inv, us_inv = aug.invert(A, us, eps)
                pred_inv = aug.apply(pred_aug, A_inv, us_inv)
                pred_sum = pred_sum + pred_inv
            pred_avg = pred_sum / ncopies

            task_loss = F.mse_loss(pred_avg, u)
            w = aug.widths()
            reg = -lam * (w * w).sum()
            loss = task_loss + reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_task += task_loss.item()
            ep_reg  += reg.item()
            nb += 1

        sched.step()

        if (epoch + 1) % max(1, cfg.epochs_augerino // 10) == 0 or epoch == 0:
            ws = aug.widths().detach().cpu().numpy()
            print(f"    Augerino ep {epoch+1}/{cfg.epochs_augerino}  "
                  f"task={ep_task/nb:.4e}  reg={ep_reg/nb:+.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  "
                  f"widths=[{', '.join(f'{x:.2f}' for x in ws)}]  "
                  f"({time.time()-t0:.1f}s)")

    aug.eval(); model.eval()
    return aug, model


def extract_augerino_generators(aug, model, sol_batch,
                                x_grid, t_grid, cfg, device):
    """
    Convert each learned width θ_i into the corresponding physical-space
    vector field (ξ_i, η_i, φ_i) on the (x_grid × t_grid) mesh.

    Normalised grid coords are in [-1, 1]; we write the fields in the
    physical coordinates x ∈ x_grid, t ∈ t_grid so that they can be
    compared by cosine similarity to the analytic LPS generators in
    ground_truth.py (which are given in physical coords).

    The mapping between normalised and physical coords is
        x_n = 2 * x / L_x - 1                           (L_x = x_grid[-1])
        t_n = 2 * t / T   - 1                           (T   = t_grid[-1])

    For each affine generator G_i, exp(ε θ_i G_i) acts on (x_n, t_n) as
    (x_n, t_n) → (x_n, t_n) + ε θ_i · G_i · (x_n, t_n, 1).  The
    PHYSICAL vector field is therefore θ_i · (L/2) · (G_i row).
    Since we use cosine similarity in the downstream metric, the
    scalar factor (L/2) drops out — we just return the normalised-grid
    vector field times θ_i.  G7 acts as u → u e^{ε θ_7}, giving
    φ = θ_7 · u.
    """
    widths = aug.widths().detach().cpu().numpy()
    Nx = len(x_grid)
    Nt = len(t_grid)

    x_n = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    t_n = np.linspace(-1.0, 1.0, Nt, dtype=np.float32)
    X, T = np.meshgrid(x_n, t_n, indexing='ij')

    u_sample = sol_batch[0, 0].cpu().numpy()

    # Analytic vector field per generator (in normalised coords, scaled by its learned width)
    gens = []
    fields = [
        # (ξ, η, φ)
        (np.ones_like(X),    np.zeros_like(X), np.zeros_like(X)),  # G1
        (np.zeros_like(X),   np.ones_like(X),  np.zeros_like(X)),  # G2
        (-T,                  X,                np.zeros_like(X)),  # G3
        ( X,                  T,                np.zeros_like(X)),  # G4
        ( X,                 -T,                np.zeros_like(X)),  # G5
        ( T,                  X,                np.zeros_like(X)),  # G6
        (np.zeros_like(X),   np.zeros_like(X), u_sample),           # G7
    ]

    norms = []
    for i, (xi, eta, phi) in enumerate(fields):
        w = widths[i]
        xi_s  = (xi  * w).astype(np.float32)
        eta_s = (eta * w).astype(np.float32)
        phi_s = (phi * w).astype(np.float32)
        gens.append((xi_s, eta_s, phi_s))
        norm = float(np.sqrt(np.mean(xi_s**2 + eta_s**2 + phi_s**2)))
        norms.append(norm)

    return gens, np.array(norms)
