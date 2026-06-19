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
# class SpectralHeatSurrogate:
#     """Exact analytical surrogate for u_t = α u_xx."""
#     def __init__(self, alpha=0.01):
#         self.alpha = alpha

#     def generator_field(self, u, x_grid, t_grid):
#         """Compute α·u_xx via spectral differentiation. Exact."""
#         B, C, Nx, Nt = u.shape
#         dx = (x_grid[0, 0, 1, 0] - x_grid[0, 0, 0, 0]).item()
#         k = torch.fft.fftfreq(Nx, d=dx, device=u.device) * 2 * np.pi
#         u_hat = torch.fft.fft(u, dim=2)
#         u_xx_hat = -(k.view(1, 1, Nx, 1) ** 2) * u_hat
#         u_xx = torch.fft.ifft(u_xx_hat, dim=2).real
#         return self.alpha * u_xx

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
from data_generation import generate_pde_data
from shared_modules import FNOSurrogate, LocalGeneratorFNO
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
    from torch.func import jvp
    
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
        'fno' → LocalGeneratorFNO (default, backward-compat)
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

    if name == 'FINS':
        M = 8
        gt_gens_eval = [(xi[M:-M, M:-M], eta[M:-M, M:-M], phi[M:-M, M:-M]) 
                        for (xi, eta, phi) in gt_gens]
    else:
        gt_gens_eval = gt_gens
    
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
    S_theta_ckpt = (f"S_theta_{pde_type}_s{seed}"
                    f"_g{cfg.grid_size}x{cfg.time_steps}.pt")
    if os.path.exists(S_theta_ckpt):
        print(f"\n  Loading S_θ from {S_theta_ckpt}...")
        from shared_modules_v2 import FNOSurrogateEquivariant
        S_theta = FNOSurrogateEquivariant(cfg).to(device)
        S_theta.load_state_dict(torch.load(S_theta_ckpt, map_location=device))
        S_theta.eval()
        for p in S_theta.parameters():
            p.requires_grad_(False)
    else:
        print(f"\n  Training S_θ (solution operator) for FINS...")
        S_theta = train_solution_operator(
            train_ic, train_sol, train_x, train_t, cfg, device,
            equivariant=True,
        )
        torch.save(S_theta.state_dict(), S_theta_ckpt)

    results = {}

    print("\n  [1/6] FINS (equation-free, equivariance loss)...")
    t0 = time.time()

    # Sanity check on the trained S_θ before expensive discovery
    sanity_check_equivariance(
        S_theta, pde_type, train_sol, train_x, train_t, cfg, device,
        alpha=cfg.alpha_heat, loss_mode='jvp', n_samples=4,
    )

    fins_models, fins_frozen = train_fins(
        train_sol, train_x, train_t, S_theta, cfg, device,
        n_sym=cfg.n_sym,
        epochs_per_gen=100,
        lr=1e-3,
        lambda_norm=1.0,
        loss_mode='jvp',
    )
    fins_gens, fins_norms = extract_fins_generators(
        fins_models, fins_frozen,
        train_sol, train_x, train_t, S_theta, cfg, device,
    )
    results['FINS'] = evaluate_method(
        'FINS', fins_gens, fins_norms,
        gt_gens, gt_names, gt_dim, dx, dt, time.time() - t0,
    )

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


    all_method_names = ['FINS', 'LieGAN', 'LaLiGAN',
                        'LIG', 'LieNLSD', 'Augerino']

    summary_names = [n for n in all_method_names if n in results]

    print(f"\n  {'Method':>10} | {'Grassmann':>9} | {'CosSim':>6} | "
          f"{'ACE':>9} | {'Rank':>4} | {'Time':>7}")
    print(f"  {'-'*59}")
    for name in summary_names:
        r = results[name]
        print(f"  {name:>10} | {r['grassmann']:>9.4f} | {r['cosine']:>6.4f} | "
              f"{r['bracket']:>9.2e} | {r['rank']:>4d} | {r['time']:>6.1f}s")
    print(f"  GT generators: {', '.join(gt_names)} (dim={gt_dim})")

    print(f"\n  Generator norms:")
    for name in summary_names:
        r = results[name]
        if 'norms' in r:
            norms_str = ', '.join(f'{n:.4f}' for n in r['norms'])
            print(f"  {name:>10}: [{norms_str}]")

    print(f"\n  Per-generator matching (★>0.8 ●>0.5 ○<0.5):")
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

    print(f"Device: {cfg.device}")
    print(f"Grid: {cfg.grid_size}x{cfg.time_steps}, Train: {cfg.n_train}")
    print(f"Operator backend: {cfg.operator_backend.upper()}")

    results = run_experiment(args.pde, args.seed, cfg)

    out_file = (f'results_{args.pde}_s{args.seed}'
                f'_{cfg.operator_backend}.json')
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_file}")