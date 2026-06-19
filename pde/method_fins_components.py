"""
FINS-components — Building blocks for FINS sequential deflation discovery.

This module provides the architectural and algorithmic primitives used by
method_fins.py:

  - SingleGeneratorFNO: per-generator FNO architecture
  - deflate_generator:  Gram–Schmidt orthogonal projection in the
                        tangent space
  - create_grids:       coordinate grid utilities
  - _fd_x_periodic:     spectral / finite-difference x-derivatives
  - _fd_t, fd_t_high_order: high-order finite-difference t-derivatives
  - _spectral_dx, _spectral_dt: spectral derivative routines

The module also contains an alternative training routine
(train_deflation_residual) based on a direct PDE-residual loss, retained
for ablation and diagnostics — it is NOT the FINS production training
loop (use train_fins from method_fins.py for that).

Mathematical formulation (Gram-Schmidt deflation, shared with FINS):
    Given frozen generators {v_1, ..., v_{k-1}}, define the deflated generator:
        v_k^⊥ = v_k - Σ_{i<k} <v_k, v_i> / <v_i, v_i> · v_i

    Minimize:
        L = Residual(v_k^⊥) + λ_norm · (||v_k|| - 1)²

Architecture:
    Each generator is a separate FNO with 3 input channels (x, t, u) and
    3 output channels (ξ, η, φ). This is ~3x smaller than a joint FNO with
    3*N_sym outputs, and eliminates inter-generator gradient conflicts.

"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from torch.utils.data import DataLoader, TensorDataset

from shared_modules import make_fno



class SingleGeneratorFNO(nn.Module):
    """
    One FNO that outputs a single generator (ξ, η, φ).

    3 input channels: (x, t, u)
    3 output channels: (ξ, η, φ)

    Uses the same make_fno as the joint version, so FNO hyperparameters
    (modes, hidden channels, layers) are inherited from cfg.
    """
    def __init__(self, cfg=None):
        super().__init__()
        from config import ExperimentConfig
        cfg = cfg or ExperimentConfig()
        self.fno = make_fno(3, 3, cfg)

    def forward(self, x_grid, t_grid, u):
        """
        x_grid, t_grid, u: [B, 1, Nx, Nt]
        Returns: (xi, eta, phi), each [B, 1, Nx, Nt]
        """
        inp = torch.cat([x_grid, t_grid, u], dim=1)
        out = self.fno(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]



def create_grids(x_coords, t_coords, device):
    """
    Create 2D grids from per-sample 1D coordinates.

    x_coords: [B, 1, Nx]
    t_coords: [B, 1, Nt]
    Returns:  x_grid [B, 1, Nx, Nt], t_grid [B, 1, Nx, Nt]
    """
    B, _, Nx = x_coords.shape
    Nt = t_coords.shape[-1]
    x_grid = x_coords.unsqueeze(-1).expand(B, 1, Nx, Nt)
    t_grid = t_coords.unsqueeze(2).expand(B, 1, Nx, Nt)
    return x_grid, t_grid

def _fd_x_periodic(f, dx):
    """Central difference along Nx axis with periodic wrap."""
    out = torch.zeros_like(f)
    out[:, :, 1:-1, :] = (f[:, :, 2:, :] - f[:, :, :-2, :]) / (2 * dx)
    out[:, :, 0:1,  :] = (f[:, :, 1:2, :] - f[:, :, -1:, :])  / (2 * dx)
    out[:, :, -1:,  :] = (f[:, :, 0:1, :] - f[:, :, -2:-1, :]) / (2 * dx)
    return out


def _fd_t(f, dt):
    """Central difference along Nt axis (one-sided at the ends)."""
    out = torch.zeros_like(f)
    out[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * dt)
    out[:, :, :, 0:1]  = (f[:, :, :, 1:2] - f[:, :, :, 0:1]) / dt
    out[:, :, :, -1:]  = (f[:, :, :, -1:] - f[:, :, :, -2:-1]) / dt
    return out
def fd_t_high_order(u: torch.Tensor, dt) -> torch.Tensor:
    """
    Vectorized 6th-order central finite difference along the last axis.

    u:  [B, 1, Nx, Nt]
    dt: scalar (float or 0-d tensor) — uniform temporal step

    Returns u_t with the same shape:
      - 6th-order central difference for indices 3 .. Nt-4
      - 4th-order central for indices 2 and Nt-3
      - 2nd-order central for indices 1 and Nt-2
      - one-sided 1st-order at indices 0 and Nt-1

    For synthetic PDE data with smooth solutions, the 6th-order interior
    scheme has truncation error ~ dt^6 * u^(7), which for dt ~ 0.016 and
    smooth heat solutions is below 1e-9 — effectively exact.
    """
    Nt = u.shape[-1]

    if torch.is_tensor(dt):
        dt = dt.mean().item()
    dt = float(dt)

    out = torch.zeros_like(u)

    if Nt >= 7:
        c_m3 = -1.0 / 60.0
        c_m2 =  3.0 / 20.0
        c_m1 = -3.0 / 4.0
        c_p1 =  3.0 / 4.0
        c_p2 = -3.0 / 20.0
        c_p3 =  1.0 / 60.0

        out[..., 3:Nt-3] = (
            c_m3 * u[..., 0:Nt-6]
          + c_m2 * u[..., 1:Nt-5]
          + c_m1 * u[..., 2:Nt-4]
          + c_p1 * u[..., 4:Nt-2]
          + c_p2 * u[..., 5:Nt-1]
          + c_p3 * u[..., 6:Nt]
        ) / dt

    if Nt >= 5:
        for k in (2, Nt - 3):
            out[..., k:k+1] = (
                -u[..., k+2:k+3] + 8.0 * u[..., k+1:k+2]
                - 8.0 * u[..., k-1:k] + u[..., k-2:k-1]
            ) / (12.0 * dt)

    if Nt >= 3:
        for k in (1, Nt - 2):
            out[..., k:k+1] = (u[..., k+1:k+2] - u[..., k-1:k]) / (2.0 * dt)

    out[..., 0:1] = (u[..., 1:2]  - u[..., 0:1])  / dt
    out[..., -1:] = (u[..., -1:]  - u[..., -2:-1]) / dt

    return out
from torch.func import jvp, vmap


def _spectral_dx(u, dx):
    """Spectral d/dx along dim=2 (spatial). Exact on periodic domains."""
    Nx = u.shape[2]
    k = torch.fft.fftfreq(Nx, d=dx, device=u.device) * 2 * np.pi
    u_hat = torch.fft.fft(u, dim=2)
    return torch.fft.ifft(1j * k.view(1, 1, Nx, 1) * u_hat, dim=2).real


def _spectral_dt(u, dt):
    """
    Spectral d/dt along dim=3 (temporal) with mirror extension
    to handle non-periodic time axis.
    """
    Nt = u.shape[3]
    # Mirror-extend: [u(t_0)...u(t_N)] → [u(t_0)...u(t_N)...u(t_0)]
    u_mirror = torch.cat([u, u.flip(dims=[3])[:, :, :, 1:-1]], dim=3)
    Nt_ext = u_mirror.shape[3]
    
    omega = torch.fft.fftfreq(Nt_ext, d=dt, device=u.device) * 2 * np.pi
    u_hat = torch.fft.fft(u_mirror, dim=3)
    du_hat = 1j * omega.view(1, 1, 1, Nt_ext) * u_hat
    du_mirror = torch.fft.ifft(du_hat, dim=3).real
    
    return du_mirror[:, :, :, :Nt]

def olver_loss_single_jvp(surrogate, xi, eta, phi, u, x_coords, t_coords, device):
    B, C, Nx, Nt = u.shape

    x_grid, t_grid = create_grids(x_coords, t_coords, device)
    x_grid = x_grid.contiguous()
    t_grid = t_grid.contiguous()
    u = u.contiguous()

    dx = (x_coords.max() - x_coords.min()).item() / max(1, x_coords.shape[-1] - 1)
    dt_val = (t_coords.max() - t_coords.min()).item() / max(1, t_coords.shape[-1] - 1)

    u_x = _spectral_dx(u, dx)
    u_t = _spectral_dt(u, dt_val)

    Q = phi - xi * u_x - eta * u_t

    def N_of_u(u_):
        return surrogate.generator_field(u_, x_grid, t_grid)

    _, dN_Q = torch.func.jvp(N_of_u, (u,), (Q,))
    _, u_tt_smooth = torch.func.jvp(N_of_u, (u,), (u_t,))

    u_tx = _spectral_dt(u_x, dt_val)
    xi_t  = _spectral_dt(xi, dt_val)
    eta_t = _spectral_dt(eta, dt_val)
    phi_t = _spectral_dt(phi, dt_val)
    Q_t = phi_t - xi_t * u_x - xi * u_tx - eta_t * u_t - eta * u_tt_smooth

    residual = dN_Q - Q_t

    tslice = slice(4, -4) if Nt > 8 else slice(None)
    residual_int = residual[..., tslice]

    v_sq = xi.pow(2) + eta.pow(2) + phi.pow(2)
    v_norm_sq = v_sq[..., tslice].mean(dim=(2, 3), keepdim=True) + 1e-6

    return (residual_int.pow(2) / v_norm_sq).mean()

def olver_loss_single(surrogate, xi, eta, phi, u, x_coords, t_coords, device):
    """
    Scalar-functional Olver loss for ONE generator (xi, eta, phi).

    J[u] = || u_t^fd - N_theta[u] ||^2   (PDE residual as scalar)

    residual = xi * dJ/dx + eta * dJ/dt + phi * dJ/du

    loss = mean(residual^2) / mean(|v|^2)   (Rayleigh quotient)

    All three partials come from a single backward through J — no D_t Q,
    no double finite differences, no hand-dropped prolongation terms.
    """
    B, C, Nx, Nt = u.shape

    x_grid, t_grid = create_grids(x_coords, t_coords, device)
    x_grid = x_grid.contiguous()
    t_grid = t_grid.contiguous()
    u_c = u.contiguous()

    inputs = torch.cat([u_c, x_grid, t_grid], dim=1).requires_grad_(True)
    u_att = inputs[:, 0:1]
    x_att = inputs[:, 1:2]
    t_att = inputs[:, 2:3]

    N_field = surrogate.generator_field(u_att, x_att, t_att)

    # u_t via central FD (2nd order — sufficient here because J is a scalar
    # and we only need its gradient, not the target itself).
    dt_scalar = (t_coords.max() - t_coords.min()).item() / max(1, Nt - 1)
    u_t_fd = torch.zeros_like(u_att)
    u_t_fd[..., 1:-1] = (u_att[..., 2:] - u_att[..., :-2]) / (2 * dt_scalar)

    # Temporal interior where FD is valid.
    tslice = slice(4, -4) if Nt > 8 else slice(1, -1)
    residual_field = (u_t_fd - N_field)[..., tslice]

    # Scalar functional J.
    J = residual_field.pow(2).sum()

    # Single backward pass → gradients w.r.t. (u, x, t).
    grads = torch.autograd.grad(J, inputs, create_graph=False)[0].detach()
    dJ_du = grads[:, 0:1]
    dJ_dx = grads[:, 1:2]
    dJ_dt = grads[:, 2:3]

    # Olver condition: v · ∇J = 0.
    vJ = xi * dJ_dx + eta * dJ_dt + phi * dJ_du

    # Rayleigh quotient normalization (scale-invariant).
    v_sq = xi.pow(2) + eta.pow(2) + phi.pow(2)
    v_norm_sq = v_sq[..., tslice].mean(dim=(2, 3), keepdim=True) + 1e-6

    return (vJ[..., tslice].pow(2) / v_norm_sq).mean()



def deflate_generator(xi, eta, phi, frozen_fields):
    """
    Project out the components of (xi, eta, phi) that lie in the span
    of the frozen generators.

    Uses standard Gram-Schmidt: for each frozen v_i, subtract the projection
        v_k ← v_k - <v_k, v_i> / <v_i, v_i> · v_i

    The result v_k^⊥ is the part of v_k linearly independent of all frozen.

    If v_k is collinear with any frozen generator, v_k^⊥ → 0, which makes
    the Rayleigh-quotient Olver loss → ∞. This provides a natural barrier
    against collapse WITHOUT forcing orthogonality.

    Generators with moderate cos similarity (e.g. Galilean and scaling with
    cos ≈ 0.5) are NOT penalized — only their deflated components are used
    for Olver, and the raw generators are returned for metric evaluation.
    """
    xi_d, eta_d, phi_d = xi, eta, phi

    for (xi_f, eta_f, phi_f) in frozen_fields:
        # Inner product <v_current, v_frozen> over the spatial grid.
        # Both have shape [B, 1, Nx, Nt] or [1, 1, Nx, Nt] (frozen).
        inner = (xi_d * xi_f + eta_d * eta_f + phi_d * phi_f).mean()
        norm_sq = (xi_f.pow(2) + eta_f.pow(2) + phi_f.pow(2)).mean() + 1e-8
        coeff = inner / norm_sq

        xi_d  = xi_d  - coeff * xi_f
        eta_d = eta_d - coeff * eta_f
        phi_d = phi_d - coeff * phi_f

    return xi_d, eta_d, phi_d



def train_deflation_residual(train_sol, train_x, train_t, surrogate, cfg, device,
                            n_sym: int = 6,
                            epochs_per_gen: int = 100,
                            lr: float = 1e-3,
                            lambda_norm: float = 1.0):
    """
    Sequential deflation training (PDE-residual variant — diagnostic only).

    For k = 1, ..., n_sym:
      1. Create a fresh SingleGeneratorFNO.
      2. Train it by minimizing Olver loss on the Gram-Schmidt deflated
         generator (orthogonal complement of all previously frozen).
      3. Evaluate on a reference sample, freeze its output field.
      4. Log raw/deflated norms and cos similarities to frozen generators.

    Parameters
    ----------
    train_sol : [N, 1, Nx, Nt] — PDE solution trajectories
    train_x   : [N, 1, Nx]     — spatial coordinates per sample
    train_t   : [N, 1, Nt]     — temporal coordinates per sample
    surrogate : LocalGeneratorFNO — pre-trained PDE operator N_theta
    cfg       : ExperimentConfig
    device    : torch device
    n_sym     : number of generators to discover
    epochs_per_gen : training epochs per individual generator
    lr        : learning rate for each generator's optimizer
    lambda_norm : weight of the ||v|| ≈ 1 anchor

    Returns
    -------
    trained_models : list of n_sym SingleGeneratorFNO instances
    frozen_fields  : list of (xi, eta, phi) tensors on the reference grid
    """
    B, C, Nx, Nt = train_sol.shape
    loader = DataLoader(
        TensorDataset(train_sol, train_x, train_t),
        batch_size=cfg.batch_size, shuffle=True
    )

    # Reference sample for freezing and monitoring.
    u_ref  = train_sol[:1].to(device)
    xc_ref = train_x[:1].to(device)
    tc_ref = train_t[:1].to(device)
    x_grid_ref, t_grid_ref = create_grids(xc_ref, tc_ref, device)

    trained_models: List[SingleGeneratorFNO] = []
    frozen_fields: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    print(f"    Residual DEFLATION: {n_sym} generators, "
          f"{epochs_per_gen} epochs each, lr={lr}, λ_norm={lambda_norm}")
    print(f"    Grid: {Nx}x{Nt}, batch_size={cfg.batch_size}, "
          f"train samples={B}")

    for k in range(n_sym):
        print(f"\n    {'═'*50}")
        print(f"    Generator {k+1}/{n_sym}  "
              f"(frozen: {len(frozen_fields)})")
        print(f"    {'═'*50}")

        gen_k = SingleGeneratorFNO(cfg).to(device)
        opt = optim.AdamW(gen_k.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, epochs_per_gen, eta_min=lr * 0.01
        )

        for epoch in range(epochs_per_gen):
            t0 = time.time()
            epoch_olver = 0.0
            epoch_norm  = 0.0
            n_batches   = 0

            gen_k.train()
            for u_batch, x_batch, t_batch in loader:
                u  = u_batch.to(device)
                xc = x_batch.to(device)
                tc = t_batch.to(device)

                x_grid, t_grid = create_grids(xc, tc, device)
                xi, eta, phi = gen_k(x_grid, t_grid, u)

                # Project out frozen components. The Olver loss sees only
                # the linearly independent residual.
                xi_d, eta_d, phi_d = deflate_generator(
                    xi, eta, phi, frozen_fields
                )

                l_olver = olver_loss_single_jvp(
                    surrogate, xi_d, eta_d, phi_d, u, xc, tc, device
                )

                # Prevents collapse. If we apply this to the raw xi, the network
                # simply copies a frozen generator (so xi_d = 0), making Olver = 0.
                v_d = torch.cat([xi_d, eta_d, phi_d], dim=1)
                rms_d = v_d.pow(2).mean().sqrt() + 1e-8
                l_norm = (rms_d - 1.0).pow(2)

                loss = l_olver + lambda_norm * l_norm

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_k.parameters(), 5.0)
                opt.step()

                epoch_olver += l_olver.item()
                epoch_norm  += l_norm.item()
                n_batches   += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                nb = max(n_batches, 1)
                print(f"      epoch {epoch+1}/{epochs_per_gen}: "
                      f"olver={epoch_olver/nb:.6f}, "
                      f"norm={epoch_norm/nb:.4f}, "
                      f"time={time.time()-t0:.1f}s")

        gen_k.eval()
        with torch.no_grad():
            xi_f, eta_f, phi_f = gen_k(x_grid_ref, t_grid_ref, u_ref)

            raw_norm = torch.cat([xi_f, eta_f, phi_f], dim=1) \
                            .pow(2).mean().sqrt().item()

            xi_d, eta_d, phi_d = deflate_generator(
                xi_f, eta_f, phi_f, frozen_fields
            )
            defl_norm = torch.cat([xi_d, eta_d, phi_d], dim=1) \
                             .pow(2).mean().sqrt().item()

            # Cosine similarities with frozen generators.
            cos_with_frozen = []
            v_raw = torch.cat([xi_f, eta_f, phi_f], dim=1).reshape(-1)
            for (xf, ef, pf) in frozen_fields:
                v_f = torch.cat([xf, ef, pf], dim=1).reshape(-1)
                c = torch.dot(v_raw, v_f) / (v_raw.norm() * v_f.norm() + 1e-8)
                cos_with_frozen.append(c.item())

        # Olver diagnostics NEED autograd — must be outside no_grad.
        with torch.enable_grad():
            raw_olver = olver_loss_single_jvp(
                surrogate, xi_f, eta_f, phi_f,
                u_ref, xc_ref, tc_ref, device
            ).item()

            defl_olver = olver_loss_single_jvp(
                surrogate, xi_d, eta_d, phi_d,
                u_ref, xc_ref, tc_ref, device
            ).item()

        # Store FROZEN DEFLATED field.
        # This is the actual structurally independent symmetry we found.
        frozen_fields.append((
            xi_d.detach(), eta_d.detach(), phi_d.detach()
        ))

        cos_str = ", ".join(f"{c:+.3f}" for c in cos_with_frozen) \
                  if cos_with_frozen else "—"
        print(f"      → v_{k+1}: "
              f"||v||={raw_norm:.4f}, ||v⊥||={defl_norm:.4f}, "
              f"Olver_raw={raw_olver:.6f}, Olver_defl={defl_olver:.6f}")
        print(f"        cos with frozen: [{cos_str}]")

        # Flag potential rank exhaustion.
        if k > 0 and defl_olver > 10 * raw_olver:
            print(f"Olver_defl >> Olver_raw — possible rank exhaustion")

        trained_models.append(gen_k)

    print(f"\n    {'─'*50}")
    print(f"    Olver residual progression (for rank detection):")
    for k, model in enumerate(trained_models):
        model.eval()
        with torch.no_grad():
            xi_f, eta_f, phi_f = model(x_grid_ref, t_grid_ref, u_ref)
            xi_d, eta_d, phi_d = deflate_generator(
                xi_f, eta_f, phi_f, frozen_fields[:k]
            )
        with torch.enable_grad():
            defl_o = olver_loss_single_jvp(
                surrogate, xi_d, eta_d, phi_d,
                u_ref, xc_ref, tc_ref, device
            ).item()
            raw_o = olver_loss_single_jvp(
                surrogate, xi_f, eta_f, phi_f,
                u_ref, xc_ref, tc_ref, device
            ).item()
        marker = ""
        if k > 0:
            prev_defl = _prev_defl
            if defl_o > 5 * prev_defl:
                marker = "  ← SPECTRAL GAP"
        _prev_defl = defl_o  # noqa
        print(f"      v_{k+1}: Olver_raw={raw_o:.6f}, "
              f"Olver_defl={defl_o:.6f}{marker}")

    return trained_models, frozen_fields



def extract_deflation_generators(trained_models, frozen_fields,
                                  sol_batch, x_batch, t_batch,
                                  surrogate, cfg, device):
    """
    Evaluate each trained generator on a sample and return as numpy arrays.

    Note: returns the DEFLATED generators since these form the independent basis.
    """
    u_eval  = sol_batch[:1].to(device)
    xc_eval = x_batch[:1].to(device)
    tc_eval = t_batch[:1].to(device)
    x_grid, t_grid = create_grids(xc_eval, tc_eval, device)

    extracted_gens_np = []
    norms_list  = []
    
    current_frozen = []

    for model in trained_models:
        model.eval()
        with torch.no_grad():
            xi_f, eta_f, phi_f = model(x_grid, t_grid, u_eval)
            xi_d, eta_d, phi_d = deflate_generator(
                xi_f, eta_f, phi_f, current_frozen
            )
            v_sq = xi_d.pow(2) + eta_d.pow(2) + phi_d.pow(2)
            norms_list.append(torch.sqrt(v_sq.mean()).item())
            extracted_gens_np.append((
                xi_d[0, 0].cpu().numpy(),
                eta_d[0, 0].cpu().numpy(),
                phi_d[0, 0].cpu().numpy(),
            ))
            current_frozen.append((xi_d, eta_d, phi_d))

    return extracted_gens_np, np.array(norms_list)
