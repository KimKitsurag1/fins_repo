"""
FINS — Equivariance-of-Solution-Operator discovery of Lie point symmetries.

A naive equation-free approach would enforce an Olver condition through
a learned PDE operator N_θ[u] ≈ u_t. This requires dN[u] to be accurate
in arbitrary directions (especially u_x, u_xx), which FNO approximations
of linear operators like α·u_xx do not provide — GELU-induced curvature
makes Jacobian errors ~12% even with well-trained N_θ.

FINS replaces this with a mathematically equivalent but numerically far
friendlier condition: EQUIVARIANCE OF THE LEARNED SOLUTION OPERATOR.

Let S_θ : u_0 ↦ u(·, ·) denote the trained solution operator (FNO).
A vector field v = ξ∂_x + η∂_t + φ∂_u is a Lie point symmetry iff its
characteristic Q = φ − ξ·u_x − η·u_t satisfies

    dS_θ[u_0] (Q_0)  =  Q(x, t)            ∀ t ≥ 0

where Q_0 = Q|_{t=0}. In words: perturbing the IC along the symmetry
orbit evolves under S_θ into the symmetry orbit of the full trajectory.
    from method_fins import (
        train_fins,
        extract_fins_generators,
        train_solution_operator,
        sanity_check_equivariance,
    )

    # Train S_θ (FNOSurrogate from shared_modules) ONCE
    S_theta = train_solution_operator(train_ic, train_sol,
                                       train_x, train_t, cfg, device)

    # Deflation discovery with equivariance loss
    models, frozen = train_fins(
        train_sol, train_x, train_t, S_theta, cfg, device,
        n_sym=8, epochs_per_gen=100, lr=1e-3, lambda_norm=1.0,
        loss_mode='jvp',              # or 'finite_tau'
    )
    gens_np, norms = extract_fins_generators(
        models, frozen, train_sol, train_x, train_t, S_theta, cfg, device
    )

Building blocks (SingleGeneratorFNO, deflate_generator, FD utilities)
are imported from method_fins_components.
"""

import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from method_fins_components import (
    SingleGeneratorFNO,
    deflate_generator,
    create_grids,
    _fd_x_periodic,
    _fd_t,
    fd_t_high_order,
)



def train_solution_operator(train_ic, train_sol, train_x, train_t,
                             cfg, device, epochs: Optional[int] = None,
                             verbose: bool = True,
                             equivariant: bool = True):
    """
    Train a fresh solution operator S_θ on (u_0, t_coords) → u(x, t).

    Two architectural variants:
      - equivariant=True (default, RECOMMENDED for FINS):
            FNOSurrogateEquivariant from shared_modules_v2 — receives
            only (u_0, t_coord) as channels, so translation-equivariance
            under x is architectural (through FFT).
      - equivariant=False: FNOSurrogate from shared_modules — receives
            (u_0, x_coord, t_coord). Use ONLY for v1-compatibility
            experiments. Breaks ∂_x symmetry recovery.

    Returns a frozen S_θ with requires_grad=False on its parameters.
    """
    if equivariant:
        from shared_modules_v2 import FNOSurrogateEquivariant
        surrogate = FNOSurrogateEquivariant(cfg).to(device)
        surrogate_name = "FNOSurrogateEquivariant (2 ch: u0 + t)"
    else:
        from shared_modules import FNOSurrogate
        surrogate = FNOSurrogate(cfg).to(device)
        surrogate_name = "FNOSurrogate (3 ch: u0 + x + t)"

    if verbose:
        print(f"    Surrogate class: {surrogate_name}")
    opt = optim.AdamW(surrogate.parameters(), lr=cfg.lr_surrogate,
                       weight_decay=1e-5)
    n_epochs = epochs if epochs is not None else cfg.epochs_surrogate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs,
                                                      eta_min=cfg.lr_surrogate * 0.01)
    loader = DataLoader(
        TensorDataset(train_ic, train_sol, train_x, train_t),
        batch_size=cfg.batch_size, shuffle=True,
    )

    if verbose:
        print(f"    Training solution operator S_θ ({n_epochs} epochs, "
              f"batch_size={cfg.batch_size})...")

    for epoch in range(n_epochs):
        surrogate.train()
        epoch_loss = 0.0
        t0 = time.time()

        for u0, u, xc, tc in loader:
            u0 = u0.to(device); u = u.to(device)
            xc = xc.to(device); tc = tc.to(device)

            u_pred = surrogate(u0, xc, tc)
            loss = F.mse_loss(u_pred, u)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"      S_θ epoch {epoch+1}/{n_epochs}: "
                  f"MSE={epoch_loss/len(loader):.6e}, "
                  f"time={time.time()-t0:.1f}s")

    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    if verbose:
        with torch.no_grad():
            u0_s = train_ic[:4].to(device)
            u_s = train_sol[:4].to(device)
            xc_s = train_x[:4].to(device)
            tc_s = train_t[:4].to(device)
            pred = surrogate(u0_s, xc_s, tc_s)
            rel = ((pred - u_s).pow(2).mean()
                   / (u_s.pow(2).mean() + 1e-12)).sqrt().item()
            print(f"    [S_θ sanity] relative RMSE on 4 train samples = "
                  f"{rel:.4e}")

    return surrogate



def _compute_Q_field(xi, eta, phi, u, dx, dt):
    """
    Characteristic Q = φ − ξ·u_x − η·u_t on the full space-time field.

    xi, eta, phi, u: [B, 1, Nx, Nt]
    dx, dt: scalars (uniform spacings)
    Returns: Q of shape [B, 1, Nx, Nt].

    u_x is computed spectrally (periodic BC — exact for band-limited u),
    u_t via 6th-order central FD (truncation ~ dt⁶, near machine-eps for
    smooth synthetic data).
    """
    Nx = u.shape[2]
    k = torch.fft.fftfreq(Nx, d=dx, device=u.device) * 2.0 * np.pi
    k = k.view(1, 1, Nx, 1)
    u_hat = torch.fft.fft(u, dim=2)
    u_x = torch.fft.ifft(1j * k * u_hat, dim=2).real

    u_t = fd_t_high_order(u, dt)

    return phi - xi * u_x - eta * u_t


def equivariance_loss_jvp(S_theta, xi, eta, phi,
                           u0, u_traj, x_coords, t_coords,
                           dx, dt, device,
                           t_margin: int = 2,
                           normalize: bool = True):
    """
    FINS JVP equivariance residual.

    Condition: dS_θ[u_0](Q_0) = Q(x, t) on the full trajectory, where
               Q_0 = Q(x, t=0).

    loss = ⟨ |dS_θ[u_0](Q_0) − Q(x,t)|²  ⟩ / ⟨ |v|² ⟩   (Rayleigh form)

    The t-interior [t_margin, Nt-t_margin) is used so that Q_t endpoints
    (where fd_t_high_order falls back to lower order) don't dominate.

    Parameters
    ----------
    S_theta   : FNOSurrogate — forward (u0, x_coords, t_coords) → [B,1,Nx,Nt]
    xi, eta, phi : [B, 1, Nx, Nt] — generator fields on space-time
    u0        : [B, 1, Nx]   — initial conditions
    u_traj    : [B, 1, Nx, Nt] — either ground-truth u or S_θ(u0); we
                use whichever caller provides (must match x/t grid)
    x_coords  : [B, 1, Nx]
    t_coords  : [B, 1, Nt]
    """
    from torch.func import jvp

    B, C, Nx, Nt = u_traj.shape

    Q = _compute_Q_field(xi, eta, phi, u_traj, dx, dt)

    Q_0 = Q[:, :, :, 0].contiguous()

    def S_fn(u0_):
        return S_theta(u0_, x_coords, t_coords)

    _, dS_Q = jvp(S_fn, (u0,), (Q_0,))

    if Nt > 2 * t_margin:
        tsl = slice(t_margin, Nt - t_margin)
    else:
        tsl = slice(None)

    resid = dS_Q[..., tsl] - Q[..., tsl]

    if not normalize:
        return resid.pow(2).mean()

    v_sq = (xi.pow(2) + eta.pow(2) + phi.pow(2))[..., tsl]
    v_norm_sq = v_sq.mean(dim=(2, 3), keepdim=True) + 1e-6
    return (resid.pow(2) / v_norm_sq).mean()


def equivariance_loss_finite_tau(S_theta, xi, eta, phi,
                                  u0, u_traj, x_coords, t_coords,
                                  dx, dt, device,
                                  taus=(0.01, 0.05, 0.1),
                                  t_margin: int = 2,
                                  normalize: bool = True):
    """
    Finite-τ equivariance: τ⁻² ‖S_θ(u_0 + τ Q_0) − S_θ(u_0) − τ Q(x,t)‖².

    Averages over several τ to cover both the infinitesimal regime (small τ)
    and large-τ consistency (group action stability). No JVP required —
    only three forward passes through S_θ per batch per τ.

    Equivalent to JVP version in the limit τ → 0 but cheaper per-step in
    some setups (no torch.func dependency) and more robust to JVP
    numerical noise on very deep FNOs.
    """
    B, C, Nx, Nt = u_traj.shape

    Q = _compute_Q_field(xi, eta, phi, u_traj, dx, dt)
    Q_0 = Q[:, :, :, 0].contiguous()

    u_ref = S_theta(u0, x_coords, t_coords)

    if Nt > 2 * t_margin:
        tsl = slice(t_margin, Nt - t_margin)
    else:
        tsl = slice(None)

    total = 0.0
    for tau in taus:
        u_pert = S_theta(u0 + tau * Q_0, x_coords, t_coords)
        resid = (u_pert - u_ref) / tau - Q
        resid_int = resid[..., tsl]

        if normalize:
            v_sq = (xi.pow(2) + eta.pow(2) + phi.pow(2))[..., tsl]
            v_norm_sq = v_sq.mean(dim=(2, 3), keepdim=True) + 1e-6
            total = total + (resid_int.pow(2) / v_norm_sq).mean()
        else:
            total = total + resid_int.pow(2).mean()

    return total / len(taus)


def equivariance_loss(S_theta, xi, eta, phi,
                       u0, u_traj, x_coords, t_coords,
                       dx, dt, device,
                       mode: str = 'jvp', **kwargs):
    """Dispatch to JVP or finite-τ variant."""
    if mode == 'jvp':
        return equivariance_loss_jvp(
            S_theta, xi, eta, phi, u0, u_traj,
            x_coords, t_coords, dx, dt, device, **kwargs,
        )
    elif mode == 'finite_tau':
        return equivariance_loss_finite_tau(
            S_theta, xi, eta, phi, u0, u_traj,
            x_coords, t_coords, dx, dt, device, **kwargs,
        )
    else:
        raise ValueError(f"Unknown loss mode: {mode}. "
                         f"Use 'jvp' or 'finite_tau'.")



def train_fins(train_sol, train_x, train_t,
                                S_theta, cfg, device,
                                n_sym: int = 6,
                                epochs_per_gen: int = 100,
                                lr: float = 1e-3,
                                lambda_norm: float = 1.0,
                                loss_mode: str = 'jvp',
                                use_ic_from_sol: bool = True):
    """
    Sequential deflation training with equivariance loss.

    Algorithm (per generator k = 1 ... n_sym):
      1. Spawn a fresh SingleGeneratorFNO → outputs (ξ_k, η_k, φ_k).
      2. For each batch:
         a. Forward the generator on (x_grid, t_grid, u).
         b. Gram–Schmidt deflate against frozen {v_1, ..., v_{k-1}}.
         c. Equivariance loss on v_k^⊥.
         d. Norm anchor on v_k^⊥ to prevent collapse.
      3. Evaluate on a reference sample, freeze the deflated field.
      4. Log raw/deflated norms & Olver/equivariance diagnostics.

    Parameters mirror the deflation training interface in
    method_fins_components for compatibility.
    """
    B_total, _, Nx, Nt = train_sol.shape

    if use_ic_from_sol:
        train_ic = train_sol[:, :, :, 0].contiguous()
    else:
        raise NotImplementedError("Pass IC as separate tensor if needed")

    loader = DataLoader(
        TensorDataset(train_ic, train_sol, train_x, train_t),
        batch_size=cfg.batch_size, shuffle=True,
    )

    dx = (train_x[0, 0, 1] - train_x[0, 0, 0]).item()
    dt = (train_t[0, 0, 1] - train_t[0, 0, 0]).item()

    ic_ref  = train_ic[:1].to(device)
    u_ref   = train_sol[:1].to(device)
    xc_ref  = train_x[:1].to(device)
    tc_ref  = train_t[:1].to(device)
    x_grid_ref, t_grid_ref = create_grids(xc_ref, tc_ref, device)

    trained_models: List[SingleGeneratorFNO] = []
    frozen_fields:  List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    print(f"    FINS DEFLATION: n_sym={n_sym}, loss='{loss_mode}', "
          f"epochs={epochs_per_gen}, lr={lr}, λ_norm={lambda_norm}")
    print(f"    Grid: {Nx}×{Nt}, dx={dx:.4e}, dt={dt:.4e}")

    for k in range(n_sym):
        print(f"\n    {'═'*50}")
        print(f"    Generator {k+1}/{n_sym}  (frozen: {len(frozen_fields)})")
        print(f"    {'═'*50}")

        gen_k = SingleGeneratorFNO(cfg).to(device)
        opt = optim.AdamW(gen_k.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, epochs_per_gen, eta_min=lr * 0.01,
        )

        for epoch in range(epochs_per_gen):
            t0 = time.time()
            eq_sum = 0.0
            nm_sum = 0.0
            nb = 0

            gen_k.train()
            for u0_b, u_b, xc_b, tc_b in loader:
                u0_b = u0_b.to(device)
                u_b  = u_b.to(device)
                xc_b = xc_b.to(device)
                tc_b = tc_b.to(device)

                x_grid, t_grid = create_grids(xc_b, tc_b, device)
                xi, eta, phi = gen_k(x_grid, t_grid, u_b)

                xi_d, eta_d, phi_d = deflate_generator(
                    xi, eta, phi, frozen_fields,
                )

                l_eq = equivariance_loss(
                    S_theta, xi_d, eta_d, phi_d,
                    u0_b, u_b, xc_b, tc_b,
                    dx, dt, device,
                    mode=loss_mode,
                )

                v_d = torch.cat([xi_d, eta_d, phi_d], dim=1)
                rms_d = v_d.pow(2).mean().sqrt() + 1e-8
                l_norm = (rms_d - 1.0).pow(2)

                loss = l_eq + lambda_norm * l_norm

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_k.parameters(), 5.0)
                opt.step()

                eq_sum += l_eq.item()
                nm_sum += l_norm.item()
                nb += 1

            sched.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                nbm = max(nb, 1)
                print(f"      epoch {epoch+1}/{epochs_per_gen}: "
                      f"equiv={eq_sum/nbm:.6f}, "
                      f"norm_anchor={nm_sum/nbm:.4f}, "
                      f"time={time.time()-t0:.1f}s")

        gen_k.eval()
        with torch.no_grad():
            xi_f, eta_f, phi_f = gen_k(x_grid_ref, t_grid_ref, u_ref)

            raw_norm = torch.cat([xi_f, eta_f, phi_f], dim=1) \
                            .pow(2).mean().sqrt().item()

            xi_d, eta_d, phi_d = deflate_generator(
                xi_f, eta_f, phi_f, frozen_fields,
            )
            defl_norm = torch.cat([xi_d, eta_d, phi_d], dim=1) \
                             .pow(2).mean().sqrt().item()

            cos_with_frozen = []
            v_raw = torch.cat([xi_f, eta_f, phi_f], dim=1).reshape(-1)
            for (xf, ef, pf) in frozen_fields:
                v_fz = torch.cat([xf, ef, pf], dim=1).reshape(-1)
                c = torch.dot(v_raw, v_fz) / (
                    v_raw.norm() * v_fz.norm() + 1e-8
                )
                cos_with_frozen.append(c.item())

        with torch.enable_grad():
            raw_eq = equivariance_loss(
                S_theta, xi_f, eta_f, phi_f,
                ic_ref, u_ref, xc_ref, tc_ref,
                dx, dt, device, mode=loss_mode,
            ).item()
            defl_eq = equivariance_loss(
                S_theta, xi_d, eta_d, phi_d,
                ic_ref, u_ref, xc_ref, tc_ref,
                dx, dt, device, mode=loss_mode,
            ).item()

        frozen_fields.append((xi_d.detach(), eta_d.detach(), phi_d.detach()))

        cos_str = ", ".join(f"{c:+.3f}" for c in cos_with_frozen) \
                  if cos_with_frozen else "—"
        print(f"      → v_{k+1}: "
              f"‖v‖={raw_norm:.4f}, ‖v⊥‖={defl_norm:.4f}, "
              f"Equiv_raw={raw_eq:.6f}, Equiv_defl={defl_eq:.6f}")
        print(f"        cos with frozen: [{cos_str}]")

        if k > 0 and defl_eq > 10 * raw_eq:
            print(f"        ⚠ Equiv_defl >> Equiv_raw — "
                  f"possible rank exhaustion")

        trained_models.append(gen_k)

    print(f"\n    {'─'*50}")
    print(f"    Equivariance residual progression (rank detection):")
    _prev = None
    for k, model in enumerate(trained_models):
        model.eval()
        with torch.no_grad():
            xi_f, eta_f, phi_f = model(x_grid_ref, t_grid_ref, u_ref)
            xi_d, eta_d, phi_d = deflate_generator(
                xi_f, eta_f, phi_f, frozen_fields[:k]
            )
        with torch.enable_grad():
            defl_e = equivariance_loss(
                S_theta, xi_d, eta_d, phi_d,
                ic_ref, u_ref, xc_ref, tc_ref,
                dx, dt, device, mode=loss_mode,
            ).item()
            raw_e = equivariance_loss(
                S_theta, xi_f, eta_f, phi_f,
                ic_ref, u_ref, xc_ref, tc_ref,
                dx, dt, device, mode=loss_mode,
            ).item()
        marker = ""
        if _prev is not None and defl_e > 5 * _prev:
            marker = "  ← SPECTRAL GAP"
        _prev = defl_e
        print(f"      v_{k+1}: Equiv_raw={raw_e:.6f}, "
              f"Equiv_defl={defl_e:.6f}{marker}")

    return trained_models, frozen_fields


def extract_fins_generators(trained_models, frozen_fields,
                            sol_batch, x_batch, t_batch,
                            S_theta, cfg, device):
    """
    Evaluate each trained generator on a reference sample and return
    them as (numpy arrays, norms) with the SAME signature as
    extract_deflation_generators, so evaluate_method in run_single_pde
    works unchanged.
    """
    u_eval  = sol_batch[:1].to(device)
    xc_eval = x_batch[:1].to(device)
    tc_eval = t_batch[:1].to(device)
    x_grid, t_grid = create_grids(xc_eval, tc_eval, device)

    gens_np: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    norms: List[float] = []
    current_frozen: List = []

    for model in trained_models:
        model.eval()
        with torch.no_grad():
            xi_f, eta_f, phi_f = model(x_grid, t_grid, u_eval)
            xi_d, eta_d, phi_d = deflate_generator(
                xi_f, eta_f, phi_f, current_frozen
            )
            v_sq = xi_d.pow(2) + eta_d.pow(2) + phi_d.pow(2)
            norms.append(torch.sqrt(v_sq.mean()).item())
            gens_np.append((
                xi_d[0, 0].cpu().numpy(),
                eta_d[0, 0].cpu().numpy(),
                phi_d[0, 0].cpu().numpy(),
            ))
            current_frozen.append((xi_d, eta_d, phi_d))

    return gens_np, np.array(norms)


def sanity_check_equivariance(S_theta, pde_type, train_sol, train_x, train_t,
                                cfg, device, alpha: float = 0.01,
                                loss_mode: str = 'jvp', n_samples: int = 4):
    """
    Plug analytical ground-truth generators into the equivariance loss.

    If S_θ is well-trained, each GT generator should yield a SMALL
    equivariance residual (order 1e-3 or better on normalised Rayleigh
    form). If some GT generator yields a LARGE residual, that tells you
    the surrogate S_θ is not accurate in the corresponding direction —
    a useful diagnostic before launching expensive discovery runs.
    """
    from ground_truth import GroundTruthAlgebra, get_gt_names

    u_sample = train_sol[:n_samples].to(device)
    ic_sample = u_sample[:, :, :, 0].contiguous()
    xc = train_x[:n_samples].to(device)
    tc = train_t[:n_samples].to(device)

    x_np = train_x[0, 0].cpu().numpy()
    t_np = train_t[0, 0].cpu().numpy()
    u_np = train_sol[0, 0].cpu().numpy()

    dx = (xc[0, 0, 1] - xc[0, 0, 0]).item()
    dt_scalar = (tc[0, 0, 1] - tc[0, 0, 0]).item()

    if pde_type == 'heat':
        gt_list = GroundTruthAlgebra.heat_generators(x_np, t_np, u_np,
                                                       alpha=alpha)
    elif pde_type == 'burgers':
        gt_list = GroundTruthAlgebra.burgers_generators(x_np, t_np, u_np)
    elif pde_type == 'kdv':
        gt_list = GroundTruthAlgebra.kdv_generators(x_np, t_np, u_np)
    else:
        raise ValueError(f"Unknown pde: {pde_type}")

    gt_names = get_gt_names(pde_type)

    print(f"\n    [Equivariance sanity check — {pde_type}, "
          f"mode='{loss_mode}']")
    print(f"    For each GT generator, measure equivariance residual.")
    print(f"    Small values (≲1e-2) ⇒ S_θ captures this symmetry.")

    Nx, Nt = u_sample.shape[2], u_sample.shape[3]
    results = []
    for name, (xi_np, eta_np, phi_np) in zip(gt_names, gt_list):
        # Replicate the GT field across the batch
        xi_t = torch.tensor(xi_np, dtype=torch.float32, device=device) \
                    .view(1, 1, Nx, Nt).expand(n_samples, 1, Nx, Nt)
        eta_t = torch.tensor(eta_np, dtype=torch.float32, device=device) \
                     .view(1, 1, Nx, Nt).expand(n_samples, 1, Nx, Nt)
        phi_t = torch.tensor(phi_np, dtype=torch.float32, device=device) \
                     .view(1, 1, Nx, Nt).expand(n_samples, 1, Nx, Nt)

        with torch.enable_grad():
            val = equivariance_loss(
                S_theta, xi_t, eta_t, phi_t,
                ic_sample, u_sample, xc, tc,
                dx, dt_scalar, device,
                mode=loss_mode, normalize=True,
            ).item()
        print(f"      {name:>12}: equiv = {val:.4e}")
        results.append((name, val))

    return results
