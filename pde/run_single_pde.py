"""
Run all methods on a single PDE and compute metrics.
Usage: python run_single_pde.py --pde heat --seed 42 [--quick]
"""
import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.func import jvp
from shared_modules_v2 import FNOSurrogateEquivariant

class SpectralPDESurrogate:
    """
    Exact analytical surrogate for standard PDEs via spectral differentiation.
    
    Supports:
      heat:    N[u] = α u_xx
      burgers: N[u] = -u u_x + ν u_xx
      kdv:     N[u] = -6u u_x - u_xxx
    
    All spatial derivatives computed spectrally (exact on periodic domains).
    No training needed — plug directly into deflation pipeline.
    
    Note: Burgers and KdV are NONLINEAR, so their Jacobians are nontrivial:
      heat:    dN[u](h) = α h_xx                    (linear, trivial)
      burgers: dN[u](h) = -h u_x - u h_x + ν h_xx  (product rule)
      kdv:     dN[u](h) = -6h u_x - 6u h_x - h_xxx  (product rule)
    
    JVP through these is handled automatically by torch.func.jvp because
    the forward pass uses standard torch ops (FFT, multiply, IFFT).
    """
    def __init__(self, pde_type='heat', alpha=0.01, nu=0.01):
        self.pde_type = pde_type
        self.alpha = alpha
        self.nu = nu

    def _spectral_dx(self, u, dx, order=1):
        Nx = u.shape[2]
        k = torch.fft.fftfreq(Nx, d=dx, device=u.device) * 2 * np.pi
        k = k.view(1, 1, Nx, 1)
        u_hat = torch.fft.fft(u, dim=2)
        
        # Anti-aliasing filter: exponential cutoff for high modes
        # Prevents Gibbs ringing on shocks (Burgers, KdV)
        k_max = np.pi / dx
        sigma = torch.exp(-36.0 * (k.abs() / k_max) ** 36)  # sharp but smooth cutoff
        
        deriv_hat = sigma * (1j * k) ** order * u_hat
        return torch.fft.ifft(deriv_hat, dim=2).real

    def generator_field(self, u, x_grid, t_grid):
        """
        Evaluate the PDE right-hand side: u_t = N[u].
        
        u:      [B, 1, Nx, Nt]
        x_grid: [B, 1, Nx, Nt] (used only to extract dx)
        t_grid: [B, 1, Nx, Nt] (unused for autonomous PDEs)
        
        Returns: N[u] with shape [B, 1, Nx, Nt]
        """
        dx = (x_grid[0, 0, 1, 0] - x_grid[0, 0, 0, 0]).item()

        if self.pde_type == 'heat':
            u_xx = self._spectral_dx(u, dx, order=2)
            return self.alpha * u_xx

        elif self.pde_type == 'burgers':
            u_x  = self._spectral_dx(u, dx, order=1)
            u_xx = self._spectral_dx(u, dx, order=2)
            return -u * u_x + self.nu * u_xx

        elif self.pde_type == 'kdv':
            u_x   = self._spectral_dx(u, dx, order=1)
            u_xxx = self._spectral_dx(u, dx, order=3)
            return -6.0 * u * u_x - u_xxx

        else:
            raise ValueError(f"Unknown pde_type: {self.pde_type}")
try:
    torch.serialization.add_safe_globals([torch._C._nn.gelu])
except:
    pass

from config import ExperimentConfig
from data_generation import (generate_pde_data, generate_tangent_pairs,
                             solver_rhs_ut)
from oracle_jacobian import build_oracle
from stepper_surrogate import RolloutOperator, Stepper, train_stepper
from shared_modules import FNOSurrogate, LocalGeneratorFNO
from closure_metrics import (closure_report, describe,
                             fit_affine)
from ground_truth import get_gt_generators, get_gt_dim, get_gt_names, Metrics

try:
    from shared_operators import LocalGeneratorNO
    HAS_CNO = True
except ImportError:
    HAS_CNO = False
    print("INFO: shared_operators.py not found — CNO backend unavailable.")

from method_lig import train_lig, extract_lig_generators
from method_lienlsd import run_lienlsd
from method_liegan import train_liegan, extract_liegan_generators
from method_laligan import train_laligan, extract_laligan_generators
from method_augerino import train_augerino, extract_augerino_generators

from method_fins import (
    train_fins,
    extract_fins_generators,
    train_solution_operator,
    sanity_check_equivariance,
)
from method_fins_components import fd_t_high_order

def check_surrogate_jacobian(surrogate, train_sol, train_x, train_t, cfg, device):
    """Compare dN[u](h) via JVP with analytical α·h_xx."""
    
    u = train_sol[:1].to(device)
    xc = train_x[:1].to(device)
    tc = train_t[:1].to(device)
    B, C, Nx, Nt = u.shape
    
    x_grid = xc.unsqueeze(-1).expand(B, 1, Nx, Nt)
    t_grid = tc.unsqueeze(2).expand(B, 1, Nx, Nt)
    
    dx = (xc[0, 0, 1] - xc[0, 0, 0]).item()
    alpha = cfg.alpha_heat
    
    # Test perturbation: h = u_x
    u_x = torch.zeros_like(u)
    u_x[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    u_x[:, :, 0] = (u[:, :, 1] - u[:, :, -1]) / (2 * dx)
    u_x[:, :, -1] = (u[:, :, 0] - u[:, :, -2]) / (2 * dx)
    
    # Analytical: dN[u](u_x) = α·u_xxx
    u_xxx = torch.zeros_like(u)
    k = torch.fft.fftfreq(Nx, d=dx, device=device) * 2 * np.pi
    u_hat = torch.fft.fft(u, dim=2)
    u_xxx = torch.fft.ifft(-1j * k.view(1, 1, Nx, 1)**3 * u_hat, dim=2).real
    analytical = alpha * u_xxx
    
    # JVP through surrogate
    def N_of_u(u_):
        return surrogate.generator_field(u_, x_grid, t_grid)
    
    _, dN_ux = jvp(N_of_u, (u.contiguous(),), (u_x.contiguous(),))
    
    tslice = slice(4, -4)
    err = (dN_ux[..., tslice] - analytical[..., tslice]).pow(2).mean().sqrt()
    ref = analytical[..., tslice].pow(2).mean().sqrt()
    
    print(f"  Jacobian check: ||dN[u](u_x) - α·u_xxx|| / ||α·u_xxx|| = {err/ref:.6f}")
    print(f"  ||dN[u](u_x)|| = {dN_ux[..., tslice].pow(2).mean().sqrt():.6f}")
    print(f"  ||α·u_xxx||    = {ref:.6f}")

    _, dN_u = jvp(N_of_u, (u.contiguous(),), (u.contiguous(),))
    N_val = surrogate.generator_field(u, x_grid, t_grid)
    err_u = (dN_u[..., tslice] - N_val[..., tslice]).pow(2).mean().sqrt()
    ref_u = N_val[..., tslice].pow(2).mean().sqrt()
    print(f"  Jacobian check: ||dN[u](u) - N[u]|| / ||N[u]|| = {err_u/ref_u:.6f}")
def train_local_generator_surrogate(train_sol, train_x, train_t, cfg, device,
                                     pde_type: str = 'heat'):
    """
    Pre-train a local PDE generator N_theta[u] ≈ u_t via central-difference
    teacher forcing.  The backbone is picked via `cfg.operator_backend`:
        'fno' → LocalGeneratorFNO (default)
        'cno' → LocalGeneratorNO  (Convolutional Neural Operator)
    Both expose the same API (forward, rollout, generator_rhs,
    generator_field), so the rest of the JVP-based residual pipeline (
    sanity checks, training loop) is unchanged.
    """
    backend = getattr(cfg, 'operator_backend', 'fno').lower()
    if backend == 'cno':
        if not HAS_CNO:
            raise RuntimeError(
                "cfg.operator_backend='cno' but shared_operators.py is "
                "not importable. Add shared_operators.py to PYTHONPATH.")
        print(f"    Surrogate backbone: CNO (Convolutional Neural Operator)")
        surrogate = LocalGeneratorNO(cfg).to(device)
    else:
        print(f"    Surrogate backbone: FNO (default)")
        surrogate = LocalGeneratorFNO(cfg).to(device)

    opt = optim.AdamW(surrogate.parameters(), lr=cfg.lr_surrogate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs_surrogate)
    loader = DataLoader(TensorDataset(train_sol, train_x, train_t),
                        batch_size=cfg.batch_size, shuffle=True)
    print(f"    Pre-training N_theta ({cfg.epochs_surrogate} epochs, "
          f"central-difference teacher forcing)...")

    for epoch in range(cfg.epochs_surrogate):
        surrogate.train()
        epoch_loss = 0.0
        t0 = time.time()

        for u, xc, tc in loader:
            u, xc, tc = u.to(device), xc.to(device), tc.to(device)
            B, C, Nx, Nt = u.shape

            dt_scalar = (tc[0, 0, 1] - tc[0, 0, 0]).item()
            dx = (xc[0, 0, 1] - xc[0, 0, 0]).item()
            du_dt_data = fd_t_high_order(u, dt_scalar)

            x_grid = xc.unsqueeze(-1).expand(B, 1, Nx, Nt)
            t_grid = tc.unsqueeze(-2).expand(B, 1, Nx, Nt)

            rhs = surrogate.generator_field(u, x_grid, t_grid)
            tslice = slice(3, -3)
            loss = F.mse_loss(rhs[..., tslice], du_dt_data[..., tslice])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % max(1, 10) == 0 or epoch == 0:
            print(f"    Surrogate epoch {epoch+1}/{cfg.epochs_surrogate}: "
                  f"loss={epoch_loss/len(loader):.6f}, time={time.time()-t0:.1f}s")

    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        u_test  = train_sol[:1].to(device)
        xc_test = train_x[:1].to(device)
        tc_test = train_t[:1].to(device)
        Nx, Nt = u_test.shape[2], u_test.shape[3]

        x_grid = xc_test.unsqueeze(-1).expand(1, 1, Nx, Nt)
        t_grid = tc_test.unsqueeze(-2).expand(1, 1, Nx, Nt)
        N_pred = surrogate.generator_field(u_test, x_grid, t_grid)

        dx = (xc_test[0, 0, 1] - xc_test[0, 0, 0]).item()

        u_xx = torch.zeros_like(u_test)
        u_xx[:, :, 1:-1] = (u_test[:, :, 2:] - 2*u_test[:, :, 1:-1] + u_test[:, :, :-2]) / dx**2
        u_xx[:, :, 0]    = (u_test[:, :, 1]  - 2*u_test[:, :, 0]    + u_test[:, :, -1])   / dx**2
        u_xx[:, :, -1]   = (u_test[:, :, 0]  - 2*u_test[:, :, -1]   + u_test[:, :, -2])   / dx**2

        u_x = torch.zeros_like(u_test)
        u_x[:, :, 1:-1] = (u_test[:, :, 2:] - u_test[:, :, :-2]) / (2 * dx)
        u_x[:, :, 0]    = (u_test[:, :, 1]  - u_test[:, :, -1])  / (2 * dx)
        u_x[:, :, -1]   = (u_test[:, :, 0]  - u_test[:, :, -2])  / (2 * dx)

        if pde_type == 'heat':
            target = cfg.alpha_heat * u_xx
            label  = f"alpha * u_xx (alpha={cfg.alpha_heat})"
        elif pde_type == 'burgers':
            target = -u_test * u_x + cfg.nu_burgers * u_xx
            label  = f"-u u_x + nu u_xx (nu={cfg.nu_burgers})"
        elif pde_type == 'kdv':
            u_xxx = torch.zeros_like(u_test)
            u_xxx[:, :, 1:-1] = (u_x[:, :, 2:] - u_x[:, :, :-2]) / (2 * dx)
            u_xxx[:, :, 0]    = (u_x[:, :, 1]  - u_x[:, :, -1])  / (2 * dx)
            u_xxx[:, :, -1]   = (u_x[:, :, 0]  - u_x[:, :, -2])  / (2 * dx)
            target = -6.0 * u_test * u_x - u_xxx
            label  = "-6 u u_x - u_xxx"
        else:
            target = torch.zeros_like(u_test)
            label  = "(unknown PDE, target=0)"

        tslice = slice(2, None)
        rel_err = ((N_pred[..., tslice] - target[..., tslice]).abs().mean() /
                   (target[..., tslice].abs().mean() + 1e-8)).item()
        print(f"    [Sanity] N_theta vs {label}: rel_err = {rel_err:.4f}")

    return surrogate


def evaluate_method(name, gens, norms, gt_gens, gt_names, gt_dim, dx, dt, train_time):
    if not gens:
        return {'grassmann': float('inf'), 'cosine': 0, 'bracket': float('inf'),
                'rank': 0, 'rank_error': gt_dim, 'time': train_time}

    # Boundary crop, applied to BOTH the fields and the ground truth
    # (extract_fins_generators returns full-grid fields). FINS_EVAL_CROP:
    #   unset -> M = 8 for FINS only
    #   int   -> the same crop for every method (symmetric evaluation)
    _env = os.environ.get('FINS_EVAL_CROP')
    if _env is not None:
        M = int(_env)
    else:
        M = 8 if name == 'FINS' else 0

    def _crop(triples, m):
        if m <= 0:
            return triples
        return [(xi[m:-m, m:-m], eta[m:-m, m:-m], phi[m:-m, m:-m])
                for (xi, eta, phi) in triples]

    gt_gens_eval = _crop(gt_gens, M)
    gens = _crop(gens, M)
    
    gt_matrix = Metrics.generators_to_matrix(gt_gens_eval)
    disc_matrix = Metrics.generators_to_matrix(gens)
    grassmann = Metrics.grassmann_distance(disc_matrix, gt_matrix)
    cosine = Metrics.best_cosine_similarity(
        [np.concatenate([g.flatten() for g in gen]) for gen in gens],
        [np.concatenate([g.flatten() for g in gen]) for gen in gt_gens_eval])
    bracket = Metrics.bracket_closure_error(gens, dx, dt) if len(gens) > 1 else 0.0
    rank = Metrics.spectral_rank(norms)
    matching = Metrics.per_generator_matching(gens, gt_gens_eval, gt_names, norms)
    print({
        'grassmann': grassmann, 'cosine': cosine, 'bracket': bracket,
        'rank': rank, 'rank_error': abs(rank - gt_dim), 'time': train_time,
        'norms': norms.tolist() if isinstance(norms, np.ndarray) else norms,
        'matching': matching,
    })
    return {
        'grassmann': grassmann, 'cosine': cosine, 'bracket': bracket,
        'rank': rank, 'rank_error': abs(rank - gt_dim), 'time': train_time,
        'norms': norms.tolist() if isinstance(norms, np.ndarray) else norms,
        'matching': matching,
    }



def save_generator_artifacts(outpath, pde_type, seed, gt_gens, gt_names,
                             method_fields, method_norms, x_np, t_np,
                             u_sample, extra_fields=None, extra_u=None,
                             n_sym_used=-1):
    """Serialise every method's discovered generator fields.

    This is what makes Table 1 re-scorable without re-running discovery:
    any later change to the ground truth or the metrics costs seconds via
    rescore_from_artifacts.py instead of 6-8 GPU hours per equation.
    """
    payload = dict(pde=pde_type, seed=seed, x=x_np, t=t_np,
                   u_sample=u_sample,
                   gt_names=np.array(gt_names, dtype=object),
                   gt_projective_corrected=np.array(True),
                   n_sym=np.array(n_sym_used))
    for c, cname in enumerate(("xi", "eta", "phi")):
        payload[f"gt_{cname}"] = np.stack([np.asarray(g[c])
                                           for g in gt_gens])
    for m, gens in method_fields.items():
        for c, cname in enumerate(("xi", "eta", "phi")):
            payload[f"{m}_{cname}"] = np.stack([np.asarray(g[c])
                                                for g in gens])
        payload[f"{m}_norms"] = np.asarray(method_norms[m], dtype=float)
    # multi-sample block: u_ref [n_ref, Nx, Nt] and {m}_{c}_ref
    # [n_gens, n_ref, Nx, Nt], sample 0 first, for the closure metric
    if extra_fields and extra_u and any(len(v) for v in extra_fields.values()):
        payload["u_ref"] = np.stack([np.asarray(u_sample)] + [np.asarray(v)
                                                              for v in extra_u])
        for m, gens in method_fields.items():
            if not extra_fields.get(m):
                continue
            for c, cname in enumerate(("xi", "eta", "phi")):
                per_sample = [payload[f"{m}_{cname}"]] + [
                    np.stack([np.asarray(g[c]) for g in fl])
                    for fl in extra_fields[m]]
                payload[f"{m}_{cname}_ref"] = np.stack(per_sample, axis=1)
    np.savez(outpath, **payload)
    print(f"  generator artifacts -> {outpath}")


def run_experiment(pde_type, seed, cfg):
    device = cfg.device
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  PDE: {pde_type.upper()} | Seed: {seed}")
    print(f"{'='*60}")

    gt_dim = get_gt_dim(pde_type)
    print(f"  Ground truth algebra dimension: {gt_dim}")

    print("  Generating PDE data...")
    train_ic, train_sol, train_x, train_t = generate_pde_data(pde_type, cfg, cfg.n_train, seed)
    test_ic, test_sol, test_x, test_t = generate_pde_data(pde_type, cfg, cfg.n_test, seed + 10000)
    print(f"    Train: {train_sol.shape}, Test: {test_sol.shape}")
    if pde_type == 'heat':
        L_vals = train_x[:, 0, -1].numpy()
        print(f"    Domain lengths L: {np.unique(L_vals).round(2)}")

    Nx, Nt = cfg.grid_size, cfg.time_steps
    x_np = train_x[0, 0].numpy()

    if pde_type == 'heat':
        L_sample = x_np[-1] + (x_np[1] - x_np[0])
        T_sample = cfg.T_final * (L_sample) ** 2
    elif pde_type == 'kdv':
        L_sample = x_np[-1] + (x_np[1] - x_np[0])
        scale_lambda = L_sample / (2 * np.pi)
        T_sample = cfg.T_final * (scale_lambda ** 3)
    else:
        T_sample = cfg.T_final

    t_np = np.linspace(0, T_sample, Nt)
    dx = x_np[1] - x_np[0]
    dt = t_np[1] - t_np[0]

    u_sample = train_sol[0, 0].numpy()
    gt_gens = get_gt_generators(pde_type, x_np, t_np, u_sample)
    gt_names = get_gt_names(pde_type)
    arch = getattr(cfg, 'surrogate_arch', 'fno')
    S_theta_ckpt = os.path.join(
        os.environ.get('FINS_RESULTS_DIR', '.'),
        f"S_theta_{pde_type}_s{seed}_g{cfg.grid_size}x{cfg.time_steps}"
        + ("" if arch == 'fno' else f"_{arch}") + ".pt")
    if getattr(cfg, 'oracle_operator', False):
        S_theta, (train_ic, train_sol, train_x, train_t) = build_oracle(
            pde_type, cfg, cfg.oracle_n, seed)
        S_theta = S_theta.to(device)
        print(f"    oracle operator ready; FINS trains on {train_ic.shape[0]} ICs")
    elif arch == 'stepper':
        stp = Stepper().to(device)
        if os.path.exists(S_theta_ckpt):
            _sd = torch.load(S_theta_ckpt, map_location=device, weights_only=False)
            if isinstance(_sd, dict):
                _sd = dict(_sd); _sd.pop('_metadata', None)
            stp.load_state_dict(_sd)
            print(f"    loaded stepper surrogate {S_theta_ckpt}")
        else:
            print(f"    training time-stepping surrogate "
                  f"({cfg.stepper_epochs} epochs)...")
            stp = train_stepper(train_sol, device, epochs=cfg.stepper_epochs,
                                seed=seed)
            torch.save(stp.state_dict(), S_theta_ckpt)
        # with the short-horizon loss only horizon + 1 slices are ever
        # scored, so roll out only that far: 9 steps instead of 127 at K = 8
        n_roll = (cfg.fins_horizon + 1 if getattr(cfg, 'fins_horizon', None)
                  else cfg.time_steps)
        S_theta = RolloutOperator(stp, n_roll).to(device)
        print(f"    stepper rolled out {n_roll} steps")
        S_theta.eval()
        for p in S_theta.parameters():
            p.requires_grad_(False)
    elif os.path.exists(S_theta_ckpt):
        print(f"\n  Loading S_θ from {S_theta_ckpt}...")
        S_theta = FNOSurrogateEquivariant(cfg).to(device)
        _sd = torch.load(S_theta_ckpt, map_location=device,
                         weights_only=False)
        if isinstance(_sd, dict):
            _sd = dict(_sd); _sd.pop('_metadata', None)
        S_theta.load_state_dict(_sd)
        S_theta.eval()
        for p in S_theta.parameters():
            p.requires_grad_(False)
    else:
        print(f"\n  Training S_θ (solution operator) for FINS...")
        tangent = None
        if getattr(cfg, 'surrogate_tangent', 0):
            tangent = generate_tangent_pairs(
                pde_type, train_ic, cfg, n_dirs=cfg.surrogate_tangent,
                n_ics=min(128, train_ic.shape[0]), seed=seed,
                mode=cfg.tangent_mode)
            print(f"    tangent pairs: {tangent[0].shape[0]}")
        S_theta = train_solution_operator(
            train_ic, train_sol, train_x, train_t, cfg, device,
            equivariant=True, tangent=tangent,
            lambda_tangent=getattr(cfg, 'tangent_lambda', 0.0),
        )
        torch.save(S_theta.state_dict(), S_theta_ckpt)

    results = {}

    print("\n  [1/6] FINS (equation-free, equivariance loss)...")
    t0 = time.time()

    # Sanity check on the trained S_θ before expensive discovery
    sanity_check_equivariance(
        S_theta, pde_type, train_sol, train_x, train_t, cfg, device,
        alpha=cfg.alpha_heat, loss_mode='jvp', n_samples=4,
        horizon=getattr(cfg, 'fins_horizon', None),
        seam=getattr(cfg, 'fins_seam', 0.2),
        shoot_every=getattr(cfg, 'fins_shoot_every', None),
    )

    fins_models, fins_frozen = train_fins(
        train_sol, train_x, train_t, S_theta, cfg, device,
        n_sym=cfg.n_sym,
        epochs_per_gen=getattr(cfg, 'fins_epochs', 100),
        lr=1e-3,
        lambda_norm=1.0,
        loss_mode='jvp',
        lambda_b=cfg.fins_lambda_b,
        anchor=cfg.fins_anchor,
        b_penalty=cfg.fins_b_penalty,
        channel_norm=cfg.fins_channel_norm,
        train_ut=(solver_rhs_ut(pde_type, train_sol, cfg)
                  if getattr(cfg, 'provided_ut', False) else None),
        batch_size=getattr(cfg, 'fins_batch', None),
        n_sub=getattr(cfg, 'fins_n_train', None),
        horizon=getattr(cfg, 'fins_horizon', None),
        seam=getattr(cfg, 'fins_seam', 0.2),
        shoot_every=getattr(cfg, 'fins_shoot_every', None),
        generator=getattr(cfg, 'fins_generator', 'fno'),
    )
    fins_gens, fins_norms = extract_fins_generators(
        fins_models, fins_frozen,
        train_sol, train_x, train_t, S_theta, cfg, device,
    )
    results['FINS'] = evaluate_method(
        'FINS', fins_gens, fins_norms,
        gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0,
    )

    if getattr(cfg, 'fins_only', False):
        print('\n  --fins-only: baselines skipped')
        liegan_gens = laligan_gens = lig_gens = lienlsd_gens = aug_gens = []
        liegan_norms = laligan_norms = lig_norms = lienlsd_norms = aug_norms = []
        liegan_gen = laligan_gen = enc = dec = lig_gen = aug = aug_model = None
    else:
        print("\n  [2/6] LieGAN (equation-free)...")
        t0 = time.time()
        liegan_gen = train_liegan(train_sol, pde_type, x_np, t_np, cfg, device)
        liegan_gens, liegan_norms = extract_liegan_generators(
            liegan_gen, train_sol, x_np, t_np, cfg, device)
        results['LieGAN'] = evaluate_method('LieGAN', liegan_gens, liegan_norms,
                                             gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0)


        print("\n  [3/6] LaLiGAN (equation-free)...")
        t0 = time.time()
        laligan_gen, enc, dec = train_laligan(train_sol, cfg, device)
        laligan_gens, laligan_norms = extract_laligan_generators(
            laligan_gen, enc, dec, train_sol, cfg, device)
        results['LaLiGAN'] = evaluate_method('LaLiGAN', laligan_gens, laligan_norms,
                                              gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0)


        print("\n  [4/6] LIG (PDE-aware)...")
        t0 = time.time()
        lig_gen = train_lig(train_sol, pde_type, x_np, t_np, cfg, device)
        lig_gens, lig_norms = extract_lig_generators(lig_gen, train_sol, x_np, t_np, cfg, device)
        results['LIG'] = evaluate_method('LIG', lig_gens, lig_norms,
                                          gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0)


        print("\n  [5/6] LieNLSD (PDE-aware)...")
        t0 = time.time()
        lienlsd_gens, lienlsd_norms, lienlsd_svs = run_lienlsd(
            train_sol, pde_type, x_np, t_np, cfg, device)
        results['LieNLSD'] = evaluate_method('LieNLSD', lienlsd_gens, lienlsd_norms,
                                              gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0)


        print("\n  [6/6] Augerino (equation-free)...")
        t0 = time.time()
        aug, aug_model = train_augerino(train_sol, pde_type, x_np, t_np, cfg, device)
        aug_gens, aug_norms = extract_augerino_generators(
            aug, aug_model, train_sol, x_np, t_np, cfg, device)
        results['Augerino'] = evaluate_method('Augerino', aug_gens, aug_norms,
                                               gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0)

    method_fields = {m: g for m, g in
                     (('FINS', fins_gens), ('LieGAN', liegan_gens),
                      ('LaLiGAN', laligan_gens), ('LIG', lig_gens),
                      ('LieNLSD', lienlsd_gens), ('Augerino', aug_gens))
                     if len(g)}

    # Fields on ADDITIONAL reference samples. Every extractor evaluates on
    # sample 0 of the batch it receives, so slicing the batch re-evaluates
    # the trained objects on another sample at forward-pass cost. Needed by
    # the closure metric of closure_metrics.py (rescore --ace), which recovers
    # phi = a(x,t) u + b(x,t) by regressing across samples and cannot
    # separate a*u from b from one sample.
    n_ref = int(os.environ.get('FINS_N_REF_SAMPLES', '4'))
    extra_fields = {m: [] for m in method_fields}
    extra_u = []
    for k in range(1, min(n_ref, train_sol.shape[0])):
        sl = slice(k, k + 1)
        extra_u.append(train_sol[k, 0].numpy())
        try:
            extra_fields['FINS'].append(extract_fins_generators(
                fins_models, fins_frozen, train_sol[sl], train_x[sl],
                train_t[sl], S_theta, cfg, device)[0])
            if 'LieGAN' in extra_fields:
                extra_fields['LieGAN'].append(extract_liegan_generators(
                liegan_gen, train_sol[sl], x_np, t_np, cfg, device)[0])
            if 'LaLiGAN' in extra_fields:
                extra_fields['LaLiGAN'].append(extract_laligan_generators(
                laligan_gen, enc, dec, train_sol[sl], cfg, device)[0])
            if 'LIG' in extra_fields:
                extra_fields['LIG'].append(extract_lig_generators(
                lig_gen, train_sol[sl], x_np, t_np, cfg, device)[0])
            if 'LieNLSD' in extra_fields:
                extra_fields['LieNLSD'].append(run_lienlsd(
                train_sol[sl], pde_type, x_np, t_np, cfg, device)[0])
            if 'Augerino' in extra_fields:
                extra_fields['Augerino'].append(extract_augerino_generators(
                aug, aug_model, train_sol[sl], x_np, t_np, cfg, device)[0])
        except Exception as exc:                     # never lose the run
            print(f"  [warn] extra reference sample {k} skipped: {exc}")
            break
    method_norms = {'FINS': fins_norms, 'LieGAN': liegan_norms,
                    'LaLiGAN': laligan_norms, 'LIG': lig_norms,
                    'LieNLSD': lienlsd_norms, 'Augerino': aug_norms}
    method_norms = {m: method_norms[m] for m in method_fields}
    # Closure (closure_metrics.py), computed live from the multi-sample
    # block. The 'bracket' column (ACE) is printed for comparability with the
    # published table, but it is not scale-invariant, uses a bracket that
    # drops the phi d/du term, and scores abelian or duplicated sets as
    # perfect -- so it is reported as 'ACE(legacy)' and must not be quoted
    # on its own. See closure_metrics.py.
    if extra_u:
        u_ref = np.stack([np.asarray(u_sample)]
                         + [np.asarray(v) for v in extra_u])
        for m, gens in method_fields.items():
            if m not in results or not extra_fields.get(m):
                continue
            try:
                per = []
                for gi in range(len(gens)):
                    xi = np.stack([np.asarray(gens[gi][0])]
                                  + [np.asarray(fl[gi][0])
                                     for fl in extra_fields[m]])
                    eta = np.stack([np.asarray(gens[gi][1])]
                                   + [np.asarray(fl[gi][1])
                                      for fl in extra_fields[m]])
                    phi = np.stack([np.asarray(gens[gi][2])]
                                   + [np.asarray(fl[gi][2])
                                      for fl in extra_fields[m]])
                    g, sp = fit_affine(xi, eta, phi, u_ref)
                    per.append((g, sp))
                rep = closure_report([p[0] for p in per], dx, dt,
                                     basis=(x_np, t_np))
                results[m]['escaped'] = rep['escaped']
                results[m]['structure'] = rep['structure']
                results[m]['abelian_pairs'] = rep['abelian_pairs']
                results[m]['xi_spread'] = float(np.mean([p[1] for p in per]))
                results[m]['closure_verdict'] = describe(rep)
            except Exception as exc:
                print(f"  [warn] closure for {m}: {exc}")

    save_generator_artifacts(
        os.environ.get('FINS_RESULTS_DIR', '.') +
        f"/generators_{pde_type}{getattr(cfg, 'fins_tag', '')}_seed{seed}.npz",
        pde_type, seed,
        gt_gens, gt_names, method_fields, method_norms, x_np, t_np,
        u_sample, extra_fields=extra_fields, extra_u=extra_u,
        n_sym_used=int(cfg.n_sym))

    all_method_names = ['FINS', 'LieGAN', 'LaLiGAN',
                        'LIG', 'LieNLSD', 'Augerino']
    summary_names = [n for n in all_method_names if n in results]

    print(f"\n  {'Method':>10} | {'Grassmann':>9} | {'CosSim':>6} | "
          f"{'ACE(leg)':>9} | {'Rank':>4} | {'Time':>7}"
          f"  | closure")
    print(f"  {'-'*59}")
    for name in summary_names:
        r = results[name]
        print(f"  {name:>10} | {r['grassmann']:>9.4f} | {r['cosine']:>6.4f} | "
              f"{r['bracket']:>9.2e} | {r['rank']:>4d} | {r['time']:>6.1f}s"
              + (f" | esc {r['escaped']:.3f} str {r['structure']:.1e}"
                 f" spr {r['xi_spread']:.2f}"
                 if 'escaped' in r and r['escaped'] == r['escaped'] else ""))
    print(f"  GT generators: {', '.join(gt_names)} (dim={gt_dim})")

    print(f"\n  Generator norms:")
    for name in summary_names:
        r = results[name]
        if 'norms' in r:
            norms_str = ', '.join(f'{n:.4f}' for n in r['norms'])
            print(f"  {name:>10}: [{norms_str}]")

    print(f"\n  Per-generator matching (high: cos > 0.8, mid: > 0.5, low: < 0.5):")
    for name in ['FINS', 'LIG', 'LieNLSD']:
        if name not in results:
            continue
        r = results[name]
        if 'matching' in r and r['matching']:
            print(f"  {name}:")
            for line in r['matching']['report_lines']:
                print(line)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', type=str, default='heat', choices=['heat', 'burgers', 'kdv'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--fins-lambda-b', type=float, default=0.0,
                        help='FINS-v2: constrained ansatz phi = a u + b with '
                             'this penalty on the b-share (0 = published '
                             'unconstrained method; exp6 used 30)')
    parser.add_argument('--oracle-operator', action='store_true',
                        help='replace S_theta by the exact Jacobian oracle '
                             '(precomputed from the data-generating solver). '
                             'NOT equation-free: the discovery loop\'s '
                             'ceiling, a diagnostic row like LieNLSD.')
    parser.add_argument('--oracle-n', type=int, default=128,
                        help='ICs the oracle Jacobian is precomputed for '
                             '(also the FINS training set size in that mode)')
    parser.add_argument('--provided-ut', action='store_true',
                        help="KdV only: feed FINS the solver's own "
                             "right-hand side as u_t instead of finite "
                             "differences of the stored trajectory, which "
                             "are aliased on KdV (~100 percent error). Uses "
                             "the solver -- diagnostic, not equation-free.")
    parser.add_argument('--fins-horizon', type=int, default=None,
                        help='short-horizon increment loss over this many '
                             'time slices (local; sees scaling/projective '
                             'on a torus). None = published full horizon')
    parser.add_argument('--fins-shoot-every', type=int, default=None,
                        help='multiple shooting: apply the horizon loss from '
                             'data states every this many slices (needs '
                             '--surrogate-arch stepper); use = horizon for '
                             'full coverage of the time axis')
    parser.add_argument('--fins-generator', choices=['fno', 'pointwise'],
                        default='fno',
                        help="fno: field-to-field operator (published); "
                             "pointwise: the same functions of (x, t, u) at "
                             "every point and sample -- the point-symmetry class")
    parser.add_argument('--fins-seam', type=float, default=0.2,
                        help='fraction of L excluded at each side of the '
                             'seam in the horizon loss')
    parser.add_argument('--fins-batch', type=int, default=None,
                        help='batch size for the discovery loop only. A '
                             'rollout surrogate unrolls Nt steps per '
                             'evaluation and is launch-bound, so a larger '
                             'batch costs little and cuts the number of '
                             'evaluations proportionally')
    parser.add_argument('--fins-n-train', type=int, default=None,
                        help='use this many training ICs for discovery '
                             '(the surrogate still trains on all)')
    parser.add_argument('--fins-epochs', type=int, default=100,
                        help='epochs per FINS generator slot (smoke tests)')
    parser.add_argument('--fins-only', action='store_true',
                        help='skip the baselines (they do not use S_theta)')
    parser.add_argument('--surrogate-arch', default='fno',
                        choices=['fno', 'stepper'],
                        help="'fno': one-shot u0 -> u(., t) (published); "
                             "'stepper': u(t) -> u(t + dt) rollout (exp28), "
                             "Burgers/KdV only")
    parser.add_argument('--stepper-epochs', type=int, default=400)
    parser.add_argument('--surrogate-tangent', type=int, default=0,
                        help='random-direction Jacobian targets per IC for '
                             'surrogate training (0 = off). Improves the '
                             'held-out Jacobian error but, at toy scale, did '
                             'NOT improve Galilean equivariance (exp27); '
                             'campaign-scale effect untested.')
    parser.add_argument('--tangent-lambda', type=float, default=1.0)
    parser.add_argument('--tangent-mode', default='modes',
                        choices=['fourier', 'data', 'modes'])
    parser.add_argument('--fins-channel-norm', default='none',
                        choices=['none', 'q'],
                        help="'none' divides the residual by ||v|| (the "
                             "published Rayleigh form, which penalises the "
                             "eta channel wherever u_t is large); 'q' "
                             "divides by ||Q||, measuring the equivariance "
                             "condition on its own scale")
    parser.add_argument('--fins-b-penalty', default='full',
                        choices=['full', 'nonconst'],
                        help="'full' penalises all of b (heat: every GT "
                             "generator has b = 0); 'nonconst' penalises "
                             "only its (x,t)-varying part, so the constant "
                             "Galilean boost of Burgers/KdV survives")
    parser.add_argument('--fins-anchor', default='deflated',
                        choices=['deflated', 'raw'],
                        help="norm anchor: 'deflated' reproduces the "
                             "published loss; 'raw' anchors the raw field "
                             "and floors the deflated norm (exp15)")
    parser.add_argument('--fins-tag', default='',
                        help='suffix for artifacts/json of this FINS variant')
    parser.add_argument('--n-sym', type=int, default=None,
                        help='generators emitted per method. Fix it '
                             'explicitly across a campaign: k_A enters d_G '
                             'through the |k_A - k_B| (pi/2)^2 rank penalty, '
                             'so runs with different counts are not '
                             'comparable. --quick otherwise raises it to 8.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--operator_backend', type=str, default='cno',
                        choices=['fno', 'cno'],
                        help="Backbone for diagnostic time-derivative "
                             "surrogate N_theta[u] ≈ u_t (not used by FINS "
                             "itself; FINS uses S_θ). 'fno' = spectral FNO; "
                             "'cno' = convolutional neural operator "
                             "(Raonic et al., NeurIPS 2023).")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.device = args.device if torch.cuda.is_available() else 'cpu'
    cfg.operator_backend = args.operator_backend

    if args.quick:
        cfg.grid_size = 64; cfg.time_steps = 32
        cfg.n_train = 64; cfg.n_test = 16
        cfg.epochs_surrogate = 100; cfg.epochs_symmetry = 100
        cfg.epochs_liegan = 40; cfg.epochs_laligan = 40
        cfg.epochs_lig = 30; cfg.epochs_augerino = 30
        cfg.n_sym = 8; cfg.batch_size = 2
        print("QUICK MODE: reduced parameters")
        if args.n_sym is None:
            print(f"  [warn] --quick raised n_sym to {cfg.n_sym}; d_G is not "
                  f"comparable with default runs (n_sym=6). "
                  f"Pass --n-sym to fix it.")
    if args.n_sym is not None:
        cfg.n_sym = args.n_sym
        print(f"n_sym fixed to {cfg.n_sym} by flag")
    cfg.fins_lambda_b = args.fins_lambda_b
    cfg.fins_tag = args.fins_tag
    cfg.fins_anchor = args.fins_anchor
    cfg.fins_b_penalty = args.fins_b_penalty
    cfg.fins_channel_norm = args.fins_channel_norm
    cfg.surrogate_arch = args.surrogate_arch
    cfg.stepper_epochs = args.stepper_epochs
    if args.surrogate_arch == 'stepper':
        if args.pde == 'heat':
            raise SystemExit("--surrogate-arch stepper needs a fixed time grid; "
                             "heat data are multi-scale (dt varies per sample)")
        print("SURROGATE: time-stepping rollout (exp28)")
    cfg.oracle_operator = args.oracle_operator
    cfg.oracle_n = args.oracle_n
    cfg.fins_only = args.fins_only
    cfg.fins_epochs = args.fins_epochs
    cfg.fins_batch = args.fins_batch
    cfg.fins_horizon = args.fins_horizon
    cfg.fins_seam = args.fins_seam
    cfg.fins_shoot_every = args.fins_shoot_every
    cfg.fins_generator = args.fins_generator
    if args.fins_generator != 'fno':
        print(f"FINS: {args.fins_generator} generator (point-symmetry class)")
    if args.fins_shoot_every and args.surrogate_arch != 'stepper':
        raise SystemExit("--fins-shoot-every needs --surrogate-arch stepper: "
                         "the one-shot surrogate only starts from t = 0")
    if args.fins_horizon:
        shoot = (f", multiple shooting every {args.fins_shoot_every} slices"
                 if args.fins_shoot_every else ", single start at t = 0")
        print(f"FINS: short-horizon increment loss, K = {args.fins_horizon}, "
              f"seam = {args.fins_seam} L{shoot}")
    cfg.fins_n_train = args.fins_n_train
    cfg.provided_ut = args.provided_ut
    if args.provided_ut:
        print("PROVIDED u_t: solver right-hand side (diagnostic, not equation-free)")
    if args.oracle_operator:
        print(f"ORACLE OPERATOR: exact Jacobian on {args.oracle_n} ICs "
              f"(diagnostic, not equation-free)")
    cfg.surrogate_tangent = args.surrogate_tangent
    cfg.tangent_lambda = args.tangent_lambda
    cfg.tangent_mode = args.tangent_mode
    if args.surrogate_tangent:
        print(f"surrogate: tangent matching, {args.surrogate_tangent} "
              f"dirs/IC, mode={args.tangent_mode}, lambda={args.tangent_lambda}")
    if args.fins_channel_norm != 'none':
        print(f"FINS: residual normalised by ||Q|| (channel-balanced)")
    if args.fins_anchor == 'raw':
        print("FINS: raw-generator norm anchor")
    if args.fins_lambda_b > 0:
        print(f"FINS-v2: constrained ansatz, lambda_b = {args.fins_lambda_b}")

    print(f"Device: {cfg.device}")
    print(f"Grid: {cfg.grid_size}x{cfg.time_steps}, Train: {cfg.n_train}")
    print(f"Operator backend: {cfg.operator_backend.upper()}")

    results = run_experiment(args.pde, args.seed, cfg)

    out_file = (f'results_{args.pde}{args.fins_tag}_s{args.seed}'
                f'_{cfg.operator_backend}.json')
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_file}")