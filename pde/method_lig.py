"""
LIG — Ko, Kim, Lee (NeurIPS 2024)
"Learning Infinitesimal Generators of Continuous Symmetries from Data"
arXiv:2410.21853 · github.com/kogyeonghoon/learning-symmetry-from-scratch

This file is a reverse-engineered reimplementation that follows the
paper as closely as possible.  The PDF defines everything down to the
MLP width and Swish activation (App. B.1/B.2), the loss weights for
PDEs (§5.2: w_sym=1, w_ortho=3, w_lips=1), and the scale-independent
log/arccos normalisation (§4.4 end).

KEY CORRECTNESS POINTS vs. a naive implementation:

1.  Flow acts on the full (x, t, u)-space ⊆ ℝ³ (§4.3).
    The MLP shares two hidden layers of width 256 and then branches
    into per-generator heads of width 32 (App. B.1).  Swish activations.

2.  After Neural-ODE flow, (x, t, u) → (x̃, t̃, ũ) and the points are
    no longer on a rectangular grid (App. A.2).  The PDE residual Δ(ũ)
    is evaluated on this deformed grid via non-uniform finite
    differences (which replace the paper's WENO from D.1 — WENO on a
    2-D non-uniform grid is non-trivial to implement; we use the
    standard non-uniform stencils with correct metric scaling, which
    is consistent with the paper's explicit statement in App. E.1
    that bilinear-resample → discrete-FD fails).

3.  Three losses (eq. 9, 11, 13) with the scale-independent wrappers
    log and arccos (§4.4 last paragraph):
        L_sym   = Σ_a log S(ϑ_s, u_a)
        L_ortho = Σ_{a<b} arccos |⟨ sg(h̃_a), h̃_b ⟩|   (normalised h̃)
        L_lips  = Σ_a Σ_{i, j∈nbhd(i)} max(Lips − τ, 0)

4.  Integration uses torchdiffeq.odeint (adjoint ok for memory);
    we fall back to RK2 if torchdiffeq is missing.

5.  Training schedule (App. B.2):
        epochs = 50,  bs = 4,  Adam,
        lr = 1e-4 for first 25 epochs, 1e-5 for the rest,
        Sobolev regularisation switched on in the last 10 epochs.

6.  Hyper-parameters for PDEs (§5.2):
        n_sym = 4   (3 true + 1 spare),
        σ = 0.4,    τ = 3,
        w_sym = 1,  w_ortho = 3,   w_lips = 1.

ADAPTATIONS FOR OUR PIPELINE:

*   Our grids are 64 × 128 instead of 256 × 140 — this does not change
    the method, only the subsampling budget.  We do not subsample at
    all by default.
*   Our KdV is  u_t + 6 u u_x + u_xxx = 0  (matches ground_truth.py),
    whereas the paper uses u_t + u u_x + u_xxx = 0.  This is a
    coefficient rescaling u ↦ 6u and does not affect the symmetry
    algebra; we pick up the coefficient from the residual formula.
*   We keep the interface
        train_lig(train_sol, pde_type, x_grid, t_grid, cfg, device)
        extract_lig_generators(gen, sol_batch, x_grid, t_grid, cfg, device)
    so the rest of the pipeline is untouched.
"""
from __future__ import annotations
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchdiffeq import odeint_adjoint as odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    try:
        from torchdiffeq import odeint
        HAS_TORCHDIFFEQ = True
    except ImportError:
        HAS_TORCHDIFFEQ = False


class LIGMultiHeadMLP(nn.Module):
    """
    Shared trunk of two Linear(256) + Swish layers, then per-generator
    head  Linear(32) + Swish + Linear(3).  Output dim per head = 3 for
    (ξ, η, φ) on the PDE space X × U = ℝ³.

    The model behaves as a single forward:  (B, 3) → (B, n_sym, 3).
    When evaluated on a full grid of Z = X × U points, B = Nx·Nt.
    """

    def __init__(self, n_sym: int = 4, trunk_width: int = 256,
                 head_width: int = 32, input_dim: int = 3):
        super().__init__()
        self.n_sym = n_sym
        self.input_dim = input_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_width), nn.SiLU(),      # Swish ≡ SiLU
            nn.Linear(trunk_width, trunk_width), nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_width, head_width), nn.SiLU(),
                nn.Linear(head_width, input_dim),
            ) for _ in range(n_sym)
        ])

    def forward(self, xtu: torch.Tensor) -> torch.Tensor:
        """
        xtu : [..., 3] arbitrary leading shape
        returns : [..., n_sym, 3]
        """
        trunk_out = self.trunk(xtu)
        outs = [head(trunk_out) for head in self.heads]
        return torch.stack(outs, dim=-2)

    def generator_k(self, xtu: torch.Tensor, k: int) -> torch.Tensor:
        """Return only the k-th vector field (used inside ODE rhs)."""
        trunk_out = self.trunk(xtu)
        return self.heads[k](trunk_out)


class _GeneratorRHS(nn.Module):
    """Adapter making the k-th head look like a standard
    `func(t, y)` for torchdiffeq."""

    def __init__(self, mlp: LIGMultiHeadMLP, k: int, sign: float = 1.0):
        super().__init__()
        self.mlp = mlp
        self.k = k
        self.sign = sign

    def forward(self, s, y):
        return self.sign * self.mlp.generator_k(y, self.k)


def ode_flow_points(mlp: LIGMultiHeadMLP, k: int, xtu0: torch.Tensor,
                    alpha: torch.Tensor, n_steps: int = 10,
                    use_adjoint: bool = True) -> torch.Tensor:
    """
    Integrate ẏ = h_θ^(k)(y)  from s=0 to s=α  (per-batch alpha scalar).

    The paper handles α<0 by integrating -h for |α| (§4.3).  We pack
    that convention here so that `alpha` can be any real number.

    Args:
        mlp:      the multi-head MLP.
        k:        which generator slot to integrate.
        xtu0:     [B, 3] starting points (B = Nx·Nt).
        alpha:    [B] transformation scales.  Each point may have its
                  own scale — but per the paper all grid points of one
                  PDE instance share the same α.
        n_steps:  number of fixed-step integration steps when
                  torchdiffeq is used with rk4; ignored for adjoint.

    Returns:
        xtu_final : [B, 3] integrated positions.
    """
    device = xtu0.device
    alpha = alpha.view(-1, 1)

    class ScaledRHS(nn.Module):
        def __init__(self, mlp, k, alpha):
            super().__init__()
            self.mlp, self.k, self.alpha = mlp, k, alpha
        def forward(self, s, y):
            return self.alpha * self.mlp.generator_k(y, self.k)

    rhs = ScaledRHS(mlp, k, alpha)
    t_span = torch.tensor([0.0, 1.0], device=device, dtype=xtu0.dtype)

    if HAS_TORCHDIFFEQ and use_adjoint:
        y_traj = odeint(rhs, xtu0, t_span, method='rk4',
                        options={'step_size': 1.0 / n_steps})
        return y_traj[-1]
    else:
        h = 1.0 / n_steps
        y = xtu0
        for _ in range(n_steps):
            k1 = rhs(0, y)
            k2 = rhs(0, y + h * k1)
            y = y + 0.5 * h * (k1 + k2)
        return y


def _nu_du_dx(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """du/dx on a non-uniform grid.  Inputs & output: [B, Nx, Nt].
    Boundaries (i=0 and i=Nx-1) are filled by one-sided differences."""
    du_c = u[:, 2:, :] - u[:, :-2, :]
    dx_c = x[:, 2:, :] - x[:, :-2, :]
    interior = du_c / (dx_c + 1e-8)

    du0 = (u[:, 1:2, :] - u[:, 0:1, :]) / (x[:, 1:2, :] - x[:, 0:1, :] + 1e-8)
    duN = (u[:, -1:, :] - u[:, -2:-1, :]) / (x[:, -1:, :] - x[:, -2:-1, :] + 1e-8)
    return torch.cat([du0, interior, duN], dim=1)


def _nu_du_dt(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """du/dt on a non-uniform grid (t varies along last axis)."""
    du_c = u[:, :, 2:] - u[:, :, :-2]
    dt_c = t[:, :, 2:] - t[:, :, :-2]
    interior = du_c / (dt_c + 1e-8)

    du0 = (u[:, :, 1:2] - u[:, :, 0:1]) / (t[:, :, 1:2] - t[:, :, 0:1] + 1e-8)
    duN = (u[:, :, -1:] - u[:, :, -2:-1]) / (t[:, :, -1:] - t[:, :, -2:-1] + 1e-8)
    return torch.cat([du0, interior, duN], dim=-1)


def _nu_d2u_dx2(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d²u/dx² on a non-uniform grid (central, 2nd-order)."""
    h1 = x[:, 1:-1, :] - x[:, :-2, :]
    h2 = x[:, 2:, :] - x[:, 1:-1, :]
    front = (u[:, 2:, :] - u[:, 1:-1, :]) / (h2 + 1e-8)
    back  = (u[:, 1:-1, :] - u[:, :-2, :]) / (h1 + 1e-8)
    interior = 2.0 * (front - back) / (h1 + h2 + 1e-8)
    # Simple boundary continuation
    b0 = interior[:, 0:1, :]
    bN = interior[:, -1:, :]
    return torch.cat([b0, interior, bN], dim=1)


def pde_residual_deformed(u: torch.Tensor, x: torch.Tensor,
                          t: torch.Tensor, pde_type: str,
                          alpha: float = 0.01, nu: float = 0.01
                          ) -> torch.Tensor:
    """
    Evaluate Δ(ũ) on a non-uniform (x̃, t̃) grid.
    Inputs are [B, Nx, Nt].  Returns [B, Nx, Nt].
    """
    u_t  = _nu_du_dt(u, t)
    u_x  = _nu_du_dx(u, x)
    u_xx = _nu_d2u_dx2(u, x)

    if pde_type == 'heat':
        return u_t - alpha * u_xx
    if pde_type == 'burgers':
        return u_t + u * u_x - nu * u_xx
    if pde_type == 'kdv':
        u_xxx = _nu_du_dx(u_xx, x)
        return u_t + 6.0 * u * u_x + u_xxx
    raise ValueError(f"unknown pde {pde_type}")


def lsym_log(residual: torch.Tensor) -> torch.Tensor:
    """
    L_sym per instance  =  log Σ_i |Δ(ũ)_i|    (eq. 6 + log norm §4.4).
    Using ∑|·| (not ∑·²) matches eq. 6 precisely.
    """
    S = residual.abs().mean()
    return torch.log(S + 1e-8)


def lortho_arccos(vfs_on_grid: list[torch.Tensor]) -> torch.Tensor:
    """
    Normalise each vector field to unit ω(x)=1 L² norm, then
    compute  Σ_{a<b}  arccos |⟨ sg(h̃_a), h̃_b ⟩| ∈ [0, π/2].

    Each vfs_on_grid[a] has shape [B, Nx, Nt, 3] (a single generator's
    vector field evaluated on a PDE instance or mini-batch of them).
    """
    n = len(vfs_on_grid)
    if n < 2:
        return torch.tensor(0.0, device=vfs_on_grid[0].device)

    flat = [v.reshape(-1) for v in vfs_on_grid]
    norms = [f.norm() + 1e-10 for f in flat]
    normed = [f / n_ for f, n_ in zip(flat, norms)]

    loss = torch.tensor(0.0, device=flat[0].device)
    for a in range(n):
        for b in range(a + 1, n):
            inner = (normed[a].detach() * normed[b]).sum()
            inner_abs = inner.abs().clamp(max=1.0 - 1e-6)
            loss = loss + (math.pi / 2 - torch.arccos(inner_abs))
    return loss


def llips_grid(vfs_on_grid: list[torch.Tensor], xtu_grid: torch.Tensor,
               tau: float = 3.0) -> torch.Tensor:
    """
    Lipschitz loss (eq. 13) using index-adjacent grid neighbours
    along both x and t axes.

    vfs_on_grid : list of [Nx, Nt, 3] (one representative instance)
    xtu_grid    : [Nx, Nt, 3]
    """
    loss = torch.tensor(0.0, device=xtu_grid.device)
    for V in vfs_on_grid:
        dV_x = (V[1:, :, :] - V[:-1, :, :]).norm(dim=-1)
        dP_x = (xtu_grid[1:, :, :] - xtu_grid[:-1, :, :]).norm(dim=-1) + 1e-8
        lips_x = dV_x / dP_x
        loss = loss + F.relu(lips_x - tau).mean()

        dV_t = (V[:, 1:, :] - V[:, :-1, :]).norm(dim=-1)
        dP_t = (xtu_grid[:, 1:, :] - xtu_grid[:, :-1, :]).norm(dim=-1) + 1e-8
        lips_t = dV_t / dP_t
        loss = loss + F.relu(lips_t - tau).mean()
    return loss


def sobolev_penalty(vfs_on_grid: list[torch.Tensor],
                    L_spatial: float = 1.0) -> torch.Tensor:
    """
    Sobolev H²-seminorm penalty along the x-axis, evaluated in Fourier
    domain (App. B.2, eq. 23).  We exclude the zeroth Fourier mode
    because the fields are already unit-normalised.
    """
    loss = torch.tensor(0.0, device=vfs_on_grid[0].device)
    for V in vfs_on_grid:
        # V: [Nx, Nt, 3]
        Nx = V.shape[0]
        V_hat = torch.fft.fft(V, dim=0)
        n = torch.arange(Nx, device=V.device)
        n_freq = torch.minimum(n, Nx - n).float()
        w = (1.0 + (n_freq / L_spatial) ** 2) - 1.0    # App. B.2 eq. 23
        w = w.view(Nx, 1, 1)
        loss = loss + (w * V_hat.abs() ** 2).mean()
    return loss


def rescale_to_unit(u_batch: torch.Tensor,
                    x_grid: np.ndarray,
                    t_grid: np.ndarray):
    """
    Returns (x_norm, t_norm, u_norm, u_scale) such that
        x_norm, t_norm ∈ [0, 1] uniformly, and u_norm ≈ N(0, 0.29).
    x_norm/t_norm are [Nx] / [Nt] tensors; u_norm is [B, Nx, Nt].
    """
    xmin, xmax = float(x_grid.min()), float(x_grid.max())
    tmin, tmax = float(t_grid.min()), float(t_grid.max())
    x_norm = torch.tensor((x_grid - xmin) / (xmax - xmin + 1e-8),
                          dtype=torch.float32)
    t_norm = torch.tensor((t_grid - tmin) / (tmax - tmin + 1e-8),
                          dtype=torch.float32)
    u_std = u_batch.std().clamp(min=1e-6)
    u_scale = 0.29 / u_std
    u_norm = u_batch * u_scale
    return x_norm, t_norm, u_norm, u_scale


def train_lig(train_sol, pde_type, x_grid, t_grid, cfg, device):
    """
    Train LIG generators following Ko et al. (NeurIPS 2024).

    train_sol : [N, 1, Nx, Nt] torch tensor of PDE solutions
    x_grid    : numpy [Nx] physical x coords (periodic domain)
    t_grid    : numpy [Nt] physical t coords
    cfg       : ExperimentConfig (fields used: batch_size, epochs_lig,
                lr_lig, n_sym, alpha_heat, nu_burgers)
    """
    B_total, _, Nx, Nt = train_sol.shape

    n_sym = max(1, min(getattr(cfg, 'n_sym', 4), 4))        # 3+1 spare
    sigma = 0.4
    tau_lips = 3.0
    w_sym, w_ortho, w_lips = 1.0, 3.0, 1.0
    w_sobolev = 0.1                                          # App. B.2
    sobolev_from_epoch = int(0.8 * cfg.epochs_lig)           # last 20 %
    n_ode_steps = 8
    use_adjoint = HAS_TORCHDIFFEQ

    x_norm_t, t_norm_t, u_norm_all, u_scale = rescale_to_unit(
        train_sol[:, 0], x_grid, t_grid)                     # [N, Nx, Nt]
    x_norm_t = x_norm_t.to(device)
    t_norm_t = t_norm_t.to(device)
    phys_alpha = cfg.alpha_heat
    phys_nu = cfg.nu_burgers
    L_x = float(x_grid.max() - x_grid.min() + 1e-8)
    L_t = float(t_grid.max() - t_grid.min() + 1e-8)
    alpha_norm = phys_alpha * L_t / (L_x * L_x)
    nu_norm = phys_nu * L_t / (L_x * L_x)

    mlp = LIGMultiHeadMLP(n_sym=n_sym).to(device)
    lr0 = getattr(cfg, 'lr_lig', 1e-4)
    opt = optim.Adam(mlp.parameters(), lr=lr0)

    def lr_for_epoch(ep):
        return lr0 if ep < cfg.epochs_lig // 2 else 0.1 * lr0

    bs = min(getattr(cfg, 'batch_size', 4), 4)
    loader = DataLoader(TensorDataset(u_norm_all.unsqueeze(1)),
                        batch_size=bs, shuffle=True)

    xg = x_norm_t.view(1, Nx, 1).expand(1, Nx, Nt)
    tg = t_norm_t.view(1, 1, Nt).expand(1, Nx, Nt)

    print(f"    LIG/Ko-et-al: n_sym={n_sym}, σ={sigma}, τ={tau_lips}, "
          f"w=(sym={w_sym}, ortho={w_ortho}, lips={w_lips}), "
          f"adjoint={use_adjoint}, grid={Nx}×{Nt}")

    for ep in range(cfg.epochs_lig):
        for pg in opt.param_groups:
            pg['lr'] = lr_for_epoch(ep)

        t0 = time.time()
        ep_sym = ep_ortho = ep_lips = ep_sob = 0.0
        nb = 0

        for (u_batch,) in loader:
            u = u_batch.to(device)
            B = u.shape[0]
            x_all = xg.expand(B, Nx, Nt).contiguous()
            t_all = tg.expand(B, Nx, Nt).contiguous()
            u_all = u[:, 0]

            l_sym_total = torch.tensor(0.0, device=device)
            for k in range(n_sym):
                alpha_k = (2 * torch.rand(B, device=device) - 1) * sigma
                alpha_pts = alpha_k.view(B, 1, 1).expand(B, Nx, Nt)

                pts = torch.stack([x_all, t_all, u_all], dim=-1)
                pts_flat = pts.reshape(-1, 3)
                alpha_flat = alpha_pts.reshape(-1)

                pts_flow = ode_flow_points(mlp, k, pts_flat, alpha_flat,
                                           n_steps=n_ode_steps,
                                           use_adjoint=use_adjoint)
                pts_flow = pts_flow.view(B, Nx, Nt, 3)

                x_tilde = pts_flow[..., 0]
                t_tilde = pts_flow[..., 1]
                u_tilde = pts_flow[..., 2]
                res = pde_residual_deformed(
                    u_tilde, x_tilde, t_tilde, pde_type,
                    alpha=alpha_norm, nu=nu_norm)
                l_sym_total = l_sym_total + lsym_log(res)
            pts0 = torch.stack([x_all, t_all, u_all], dim=-1)
            all_vfs = mlp(pts0)
            vfs_per_gen = [all_vfs[..., a, :] for a in range(n_sym)]
            l_ortho = lortho_arccos(vfs_per_gen)
            pts0_one = pts0[0]
            vfs_one = [all_vfs[0, ..., a, :] for a in range(n_sym)]
            l_lips = llips_grid(vfs_one, pts0_one, tau=tau_lips)

            loss = (w_sym * l_sym_total
                    + w_ortho * l_ortho
                    + w_lips * l_lips)

            l_sob = torch.tensor(0.0, device=device)
            if ep >= sobolev_from_epoch:
                l_sob = sobolev_penalty(vfs_one, L_spatial=1.0)
                loss = loss + w_sobolev * l_sob

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()

            ep_sym   += l_sym_total.item()
            ep_ortho += l_ortho.item()
            ep_lips  += l_lips.item()
            ep_sob   += l_sob.item() if torch.is_tensor(l_sob) else float(l_sob)
            nb += 1

        if (ep + 1) % 1 == 0:
            dt = time.time() - t0
            print(f"    LIG ep {ep+1}/{cfg.epochs_lig} lr={lr_for_epoch(ep):.0e}"
                  f" | L_sym={ep_sym/nb:+.3f}  L_ortho={ep_ortho/nb:.3f}"
                  f"  L_lips={ep_lips/nb:.3f}  L_sob={ep_sob/nb:.3f}"
                  f"  ({dt:.1f}s)")

    mlp.eval()
    # Stash the normalisation for use at extraction time
    mlp._x_range = (float(x_grid.min()), float(x_grid.max()))
    mlp._t_range = (float(t_grid.min()), float(t_grid.max()))
    mlp._u_scale = float(u_scale.item() if torch.is_tensor(u_scale) else u_scale)
    return mlp


@torch.no_grad()
def extract_lig_generators(mlp: LIGMultiHeadMLP, sol_batch,
                           x_grid, t_grid, cfg, device):
    """
    Evaluate every learned infinitesimal generator on a _single_ PDE
    sample (the first element of sol_batch), returning the fields in
    _physical_ coordinates so that the rest of the pipeline
    (ground_truth.py metrics) sees the same quantities as the other
    baselines.
    """
    mlp.eval()
    u_sample = sol_batch[0, 0].to(device)
    Nx, Nt = u_sample.shape
    xmin, xmax = mlp._x_range
    tmin, tmax = mlp._t_range
    u_scale = mlp._u_scale

    x_n = torch.tensor((x_grid - xmin) / (xmax - xmin + 1e-8),
                       dtype=torch.float32, device=device)
    t_n = torch.tensor((t_grid - tmin) / (tmax - tmin + 1e-8),
                       dtype=torch.float32, device=device)
    u_n = u_sample * u_scale

    X, T = torch.meshgrid(x_n, t_n, indexing='ij')
    pts = torch.stack([X, T, u_n], dim=-1)
    vfs = mlp(pts)
    L_x = xmax - xmin
    L_t = tmax - tmin

    generators_np, norms = [], []
    for a in range(vfs.shape[-2]):
        xi = (vfs[..., a, 0] / (L_x + 1e-8)).cpu().numpy()
        eta = (vfs[..., a, 1] / (L_t + 1e-8)).cpu().numpy()
        phi = (vfs[..., a, 2] / (u_scale + 1e-8)).cpu().numpy()

        norm = float(np.sqrt(np.mean(xi**2) + np.mean(eta**2) + np.mean(phi**2)))
        norms.append(norm)
        generators_np.append((xi, eta, phi))

    return generators_np, np.array(norms)
