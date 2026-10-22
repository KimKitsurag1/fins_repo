"""PDE solvers and data generation with symmetry-aware diversity."""
import numpy as np
import torch
from typing import Tuple
from tqdm import tqdm
from config import ExperimentConfig


class PDESolver:
    @staticmethod
    def solve_heat(u0, alpha, T, Nt, L=1.0):
        Nx = len(u0)
        dx = L / Nx
        k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        u0_hat = np.fft.fft(u0)
        t = np.linspace(0, T, Nt)
        solution = np.zeros((Nx, Nt))
        for i, ti in enumerate(t):
            solution[:, i] = np.fft.ifft(u0_hat * np.exp(-alpha * k**2 * ti)).real
        return solution

    @staticmethod
    def solve_burgers(u0, nu, T, Nt, dt_internal=1e-4, verbose=False):
        import torch
        import numpy as np

        if isinstance(u0, torch.Tensor):
            u0_np = u0.cpu().numpy()
        else:
            u0_np = np.asarray(u0)

        is_batched = u0_np.ndim == 2
        if not is_batched:
            u0_np = u0_np[np.newaxis, :]

        Batch, Nx = u0_np.shape
        L = 2 * np.pi
        dx = L / Nx

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        k = torch.fft.fftfreq(Nx, d=dx / (2 * np.pi), device=device)
        angle_k = 2 * np.pi * k
        Lk = -nu * (angle_k ** 2)

        n_int = max(int(T / dt_internal), Nt)
        dt = T / n_int
        save_idx = np.round(np.linspace(0, n_int, Nt)).astype(int).tolist()

        u = torch.tensor(u0_np, dtype=torch.float64, device=device)
        solution = torch.zeros((Batch, Nx, Nt), dtype=torch.float64, device=device)

        E_half = torch.exp(Lk * dt / 2).unsqueeze(0)
        ik = (1j * angle_k).unsqueeze(0)

        # 2/3 dealiasing mask — kills modes above 2/3 of Nyquist
        kmax = torch.max(torch.abs(k))
        dealias = (torch.abs(k) <= (2.0 / 3.0 * kmax)).unsqueeze(0)  # [1, Nx]

        from tqdm import tqdm
        iterator = tqdm(range(n_int + 1), desc='      Burgers', leave=False) \
                if verbose else range(n_int + 1)

        sc = 0
        for step in iterator:
            if sc < Nt and step == save_idx[sc]:
                solution[:, :, sc] = u
                sc += 1
                if sc == Nt:
                    break

            # Step 1: Linear half-step
            u_hat = torch.fft.fft(u)
            u_hat = u_hat * E_half * dealias
            u = torch.fft.ifft(u_hat).real

            # Step 2: Nonlinear full-step with dealiasing
            u_hat = torch.fft.fft(u) * dealias
            ux = torch.fft.ifft(ik * u_hat).real
            u = u - dt * u * ux

            # Step 3: Linear half-step
            u_hat = torch.fft.fft(u)
            u_hat = u_hat * E_half * dealias
            u = torch.fft.ifft(u_hat).real

        res = solution.cpu().numpy().astype(np.float32)
        return res if is_batched else res[0]

    @staticmethod
    def solve_kdv(u0, T, Nt, dt_max=1e-4, u_clip_factor=10.0,
                          hyper_viscosity=1e-4, verbose=False):
        """
        KdV solver (ETDRK4) with PyTorch GPU batching.
        Returns an array of shape (Batch, Nx, Nt) or (Nx, Nt); the solution is not clipped.
        """
        import torch
        import numpy as np
        
        # Check if batched
        if isinstance(u0, torch.Tensor):
            u0_np = u0.cpu().numpy()
        else:
            u0_np = np.asarray(u0)
            
        is_batched = u0_np.ndim == 2
        if not is_batched:
            u0_np = u0_np[np.newaxis, :]
            
        Batch, Nx = u0_np.shape
        L = 2 * np.pi
        dx = L / Nx
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        k = torch.fft.fftfreq(Nx, d=dx, device=device) * 2 * torch.pi
        L_op = 1j * (k**3)
        
        # Determine dt safely
        max_u0 = float(np.max(np.abs(u0_np)))
        actual_dt_max = min(dt_max, 2e-5)
        n_int = max(int(T / actual_dt_max), Nt)
        h = T / n_int
        save_idx = np.round(np.linspace(0, n_int, Nt)).astype(int).tolist()
        
        E = torch.exp(h * L_op)
        E2 = torch.exp(h * L_op / 2)
        
        z = h * L_op
        
        mask_small = torch.abs(z) < 1.0
        z_safe = torch.where(mask_small, torch.tensor(1.0, device=device, dtype=z.dtype), z)
        
        Q_ex  = h * (torch.exp(z_safe/2) - 1) / z_safe
        f1_ex = h * (-4 - z_safe + torch.exp(z_safe)*(4 - 3*z_safe + z_safe**2)) / z_safe**3
        f2_ex = h * (2 + z_safe + torch.exp(z_safe)*(-2 + z_safe)) / z_safe**3
        f3_ex = h * (-4 - 3*z_safe - z_safe**2 + torch.exp(z_safe)*(4 - z_safe)) / z_safe**3
        
        M = 32 # Contour points
        r = torch.exp(1j * torch.pi * (torch.arange(1, M+1, device=device) - 0.5) / M)
        LR = z[:, None] + r[None, :]
        Q_co  = h * torch.mean((torch.exp(LR/2) - 1) / LR, dim=1)
        f1_co = h * torch.mean((-4 - LR + torch.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, dim=1)
        f2_co = h * torch.mean((2 + LR + torch.exp(LR)*(-2 + LR)) / LR**3, dim=1)
        f3_co = h * torch.mean((-4 - 3*LR - LR**2 + torch.exp(LR)*(4 - LR)) / LR**3, dim=1)
        
        Q  = torch.where(mask_small, Q_co, Q_ex).unsqueeze(0)
        f1 = torch.where(mask_small, f1_co, f1_ex).unsqueeze(0)
        f2 = torch.where(mask_small, f2_co, f2_ex).unsqueeze(0)
        f3 = torch.where(mask_small, f3_co, f3_ex).unsqueeze(0)
        E = E.unsqueeze(0)
        E2 = E2.unsqueeze(0)
        
        kmax = torch.max(torch.abs(k))
        mask23 = (torch.abs(k) <= (2.0/3.0 * kmax)).unsqueeze(0)
        ik = (1j * k).unsqueeze(0)
        
        def nonlin(v_hat):
            v_masked = v_hat * mask23
            u_clean = torch.fft.ifft(v_masked).real
            return -3 * ik * torch.fft.fft(u_clean**2) * mask23

        u0_t = torch.tensor(u0_np, dtype=torch.float64, device=device)
        v = torch.fft.fft(u0_t) * mask23
        
        solution = torch.zeros((Batch, Nx, Nt), dtype=torch.float64, device=device)
        
        from tqdm import tqdm
        iterator = tqdm(range(n_int + 1), desc='      KdV PyTorch Batched', leave=False) if verbose else range(n_int + 1)
        
        sc = 0
        for step in iterator:
            if sc < Nt and step == save_idx[sc]:
                solution[:, :, sc] = torch.fft.ifft(v).real
                sc += 1
                if sc == Nt:
                    break
                    
            Nv = nonlin(v)
            a = E2 * v + Q * Nv
            Na = nonlin(a)
            b = E2 * v + Q * Na
            Nb = nonlin(b)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = nonlin(c)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
            
        res = solution.cpu().numpy()
        return res if is_batched else res[0]


def solver_rhs_ut(pde_type, sols, cfg):
    """Instantaneous time derivative the data-generating solver integrates.

    For KdV the time derivative is NOT recoverable from the benchmark's
    stored trajectory: the solver keeps modes up to k ~ 21 after 2/3
    dealiasing, rotating at omega = k^3 ~ 9e3, while the save step (~0.008)
    resolves only omega < 398 -- u_t lives 97.5 percent above k = 7, and
    finite differences of the stored field are wrong by ~99 percent. It IS
    recoverable equation-free from a local stencil aligned to the solver's
    internal grid (spacing 2e-5: 0.3 percent; exp31 --ut stencil). This returns the solver's own right-hand side at
    each saved time, evaluated exactly as the integrator does
    (u_hat_t = i k^3 v_hat - 3 i k FFT(u^2), both dealiased).

    It uses the solver, so a FINS run fed with it is NOT equation-free: it
    is the diagnostic ceiling, like the oracle operator.
    """
    if pde_type != 'kdv':
        raise ValueError("solver_rhs_ut is implemented for kdv only; "
                         "heat and burgers time derivatives are resolvable "
                         "by finite differences (0.25 and 3 percent)")
    u = torch.as_tensor(sols, dtype=torch.float64)
    if u.dim() == 4:
        u = u[:, 0]
    B, Nx, Nt = u.shape
    L = 2 * np.pi
    k = torch.fft.fftfreq(Nx, d=L / Nx, dtype=torch.float64) * 2 * np.pi
    kmax = torch.abs(k).max()
    mask23 = (torch.abs(k) <= (2.0 / 3.0 * kmax)).view(1, Nx, 1)
    ik = (1j * k).view(1, Nx, 1)
    v = torch.fft.fft(u, dim=1) * mask23
    u_clean = torch.fft.ifft(v, dim=1).real
    rhs = (1j * k.view(1, Nx, 1) ** 3) * v \
        - 3 * ik * torch.fft.fft(u_clean ** 2, dim=1) * mask23
    ut = torch.fft.ifft(rhs, dim=1).real
    return ut.float().unsqueeze(1)                     # [B, 1, Nx, Nt]


def generate_tangent_pairs(pde_type, ics, cfg, n_dirs=2, n_ics=None,
                           eps=1e-3, seed=0, mode='fourier'):
    """Random-direction Jacobian targets from the SAME solver as the data.

    For a subset of initial conditions u0 and random smooth directions
    delta, returns the central-difference response of the true operator,
    R = (S(u0 + eps delta) - S(u0 - eps delta)) / (2 eps), i.e. the action
    of the true Jacobian dS[u0] on delta. Training the surrogate to match
    these on RANDOM directions constrains its Jacobian -- which is what the
    equivariance criterion actually evaluates -- without naming any
    symmetry: the directions are random Fourier fields plus a random
    constant, drawn like the ICs themselves.

    Returns (u0_t [P,1,Nx], delta [P,1,Nx], resp [P,1,Nx,Nt]) with P =
    n_ics * n_dirs, with the same ordering convention as generate_pde_data.
    """
    rng = np.random.default_rng(seed)
    n_ics = n_ics or ics.shape[0]
    idx = rng.choice(ics.shape[0], size=min(n_ics, ics.shape[0]),
                     replace=False)
    u0 = ics[idx, 0].cpu().numpy()
    M, Nx = u0.shape
    L = 2 * np.pi
    x = np.linspace(0, L, Nx, endpoint=False)
    all_ic = ics[:, 0].cpu().numpy()
    deltas = np.zeros((M, n_dirs, Nx))
    for i in range(M):
        for j in range(n_dirs):
            if mode == 'modes':
                # one Fourier mode per direction, k = 0 (the mean) treated
                # like any other; direction j cycles through
                # 1, cos x, sin x, cos 2x, sin 2x, ...
                k = (j + 1) // 2
                f = (np.ones(Nx) if j == 0 else
                     (np.cos(k * x) if j % 2 == 1 else np.sin(k * x)))
            elif mode == 'data':
                # data-manifold direction: the difference to another
                # training IC -- the variability the data actually has,
                # constant offsets and amplitude changes included, still
                # naming no symmetry
                other = all_ic[rng.integers(0, all_ic.shape[0])]
                f = other - u0[i]
                if np.sqrt(np.mean(f ** 2)) < 1e-8:
                    f = rng.standard_normal(Nx)
            else:
                f = np.zeros(Nx)
                for k in range(1, rng.integers(2, 6)):
                    f += rng.standard_normal() / k * np.sin(
                        k * x + rng.uniform(0, 2 * np.pi))
                f += rng.uniform(-0.5, 0.5)
            scale = np.sqrt(np.mean(u0[i] ** 2)) / (np.sqrt(np.mean(f ** 2)) + 1e-12)
            deltas[i, j] = f * scale
    u0_rep = np.repeat(u0[:, None, :], n_dirs, axis=1).reshape(M * n_dirs, Nx)
    dl = deltas.reshape(M * n_dirs, Nx)
    plus = np.concatenate([u0_rep + eps * dl, u0_rep - eps * dl], axis=0)
    Nt = cfg.time_steps
    if pde_type == 'heat':
        sols = np.stack([PDESolver.solve_heat(p, cfg.alpha_heat, cfg.T_final,
                                              Nt, L=L) for p in plus])
    elif pde_type == 'burgers':
        sols = PDESolver.solve_burgers(plus, cfg.nu_burgers, cfg.T_final, Nt,
                                       verbose=False)
    else:
        sols = PDESolver.solve_kdv(plus, cfg.T_final, Nt, verbose=False)
    sols = np.asarray(sols)
    resp = (sols[:M * n_dirs] - sols[M * n_dirs:]) / (2 * eps)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return (T(u0_rep).view(-1, 1, Nx), T(dl).view(-1, 1, Nx),
            T(resp).view(-1, 1, Nx, Nt))


def generate_pde_data(pde_type: str, cfg: ExperimentConfig, n_samples: int,
                      seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate PDE dataset with symmetry-aware diversity.
    Returns (ics, sols, x_coords, t_coords):
      ics: [N, 1, Nx]
      sols: [N, 1, Nx, Nt]
      x_coords: [N, 1, Nx] — physical x per sample
      t_coords: [N, 1, Nt] — physical t per sample
    """
    np.random.seed(seed)
    Nx, Nt = cfg.grid_size, cfg.time_steps
    ics = np.zeros((n_samples, 1, Nx))
    sols = np.zeros((n_samples, 1, Nx, Nt))
    x_all = np.zeros((n_samples, 1, Nx))
    t_all = np.zeros((n_samples, 1, Nt))

    if pde_type == 'heat':
        # Multi-scale: L varies, T ∝ L² (scaling symmetry: x→λx, t→λ²t)
        domain_lengths = [0.5, 0.7, 1.0, 1.4, 2.0]
        T_base = cfg.T_final  # T=1.0 at L=1.0
        n_gauss = n_samples // 3
        n_sine = n_samples // 3

        for i in tqdm(range(n_samples), desc=f'    Generating {pde_type}'):
            L = domain_lengths[i % len(domain_lengths)]
            T_i = T_base * L**2  # Scaling: T ∝ L²
            x = np.linspace(0, L, Nx, endpoint=False)
            t = np.linspace(0, T_i, Nt)
            x_all[i, 0] = x
            t_all[i, 0] = t

            if i < n_gauss:
                u0 = np.zeros(Nx)
                for _ in range(np.random.randint(1, 5)):
                    x0 = np.random.uniform(0.0, L)
                    A = np.random.uniform(0.1, 3.0)
                    w = np.random.uniform(0.02*L, 0.30*L)
                    u0 += A * np.exp(-((x - x0)/w)**2)
            elif i < n_gauss + n_sine:
                u0 = np.zeros(Nx)
                global_amp = np.random.uniform(0.3, 4.0)
                for k in range(1, np.random.randint(2, 6)):
                    A = np.random.uniform(-1,1) / np.sqrt(k)
                    u0 += A * np.sin(2*np.pi*k*x/L + np.random.uniform(0, 2*np.pi))
                u0 *= global_amp
                if np.random.rand() > 0.5:
                    u0 = u0 - u0.min() + np.random.uniform(0, 1)
            else:
                st = np.random.choice(['poly', 'step', 'combo'])
                if st == 'poly':
                    u0 = np.polyval(np.random.randn(np.random.randint(2,5)), x/L-0.5)
                    u0 = np.clip(u0 * np.random.uniform(0.5, 3.0), -10, 10)
                elif st == 'step':
                    u0 = np.zeros(Nx)
                    for _ in range(np.random.randint(1,4)):
                        l = np.random.uniform(0, 0.7*L); r = l + np.random.uniform(0.1*L, 0.3*L)
                        u0 += np.random.uniform(0.3,3.0)*0.5*(np.tanh(30*(x-l)/L)-np.tanh(30*(x-r)/L))
                else:
                    u0 = np.random.uniform(0.5,2)*np.exp(-((x-np.random.uniform(0.2*L,0.8*L))/(0.1*L))**2)
                    u0 += np.random.uniform(0.1,1)*np.sin(2*np.pi*np.random.randint(1,4)*x/L)

            if i > 0 and np.random.rand() < 0.2:
                src = np.random.randint(0, i)
                src_L = domain_lengths[src % len(domain_lengths)]
                if abs(src_L - L) < 1e-8:
                    u0 = ics[src, 0] * np.random.uniform(0.3, 3.0)

            ics[i, 0] = u0
            sols[i, 0] = PDESolver.solve_heat(u0, cfg.alpha_heat, T_i, Nt, L=L)

    elif pde_type == 'burgers':
        x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
        t = np.linspace(0, cfg.T_final, Nt)
        for i in tqdm(range(n_samples), desc=f'    Generating {pde_type} initial conditions'):
            x_all[i, 0] = x
            t_all[i, 0] = t
            u0 = np.zeros(Nx)
            ga = np.random.uniform(0.2, 2.0)
            for k in range(1, np.random.randint(3, 7)):
                u0 += np.random.uniform(-1, 1)/k * np.sin(k*x + np.random.uniform(0,2*np.pi))
            u0 *= ga
            if np.random.rand() < 0.3: u0 += np.random.uniform(-1, 1)
            ics[i, 0] = u0
        
        print(f"    Running batched PyTorch solver on GPU for {n_samples} samples...")
        sols[:, 0] = PDESolver.solve_burgers(ics[:, 0], cfg.nu_burgers, cfg.T_final, Nt, verbose=True)

    elif pde_type == 'kdv':
        # KdV Scaling symmetry: x -> \lambda x, t -> \lambda^3 t, u -> \lambda^-2 u
        domain_lengths = [1.0 * 2*np.pi, 1.0 * 2*np.pi, 1.0 * 2*np.pi, 1.0 * 2*np.pi]
        T_base = cfg.T_final
        L_base = 2*np.pi
        
        for i in tqdm(range(n_samples), desc=f'    Generating {pde_type} initial conditions'):
            L = domain_lengths[i % len(domain_lengths)]
            
            # According to KdV scaling, if x is scaled by \lambda = L / L_base, 
            # then t must scale by \lambda^3 to preserve the equation dynamics.
            scale_lambda = L / L_base
            T_i = T_base * (scale_lambda**3)
            
            x = np.linspace(0, L, Nx, endpoint=False)
            t = np.linspace(0, T_i, Nt)
            x_all[i, 0] = x
            t_all[i, 0] = t
            
            u0 = np.zeros(Nx)
            for _ in range(np.random.randint(1, 4)):
                # Base parameters on standard 2pi grid
                x0_base = np.random.uniform(0.5, 5.5)
                A_base = np.random.choice([np.random.uniform(0.1,0.5),
                                           np.random.uniform(0.5,2.0),
                                           np.random.uniform(2.0,5.0)])
                w_base = np.random.uniform(0.2, 1.0)
                
                # Apply symmetry scaling transformation to embed the exact Lie group orbits
                x0 = x0_base * scale_lambda
                A = A_base * (scale_lambda**-2)
                w = w_base * scale_lambda
                
                u0 += A / np.cosh((x - x0)/w)**2
                
            ics[i, 0] = u0
            
        print(f"    Running batched PyTorch ETDRK4 solver on GPU for {n_samples} samples...")
        sols[:, 0] = PDESolver.solve_kdv(ics[:, 0], cfg.T_final, Nt, verbose=True)

    return (torch.tensor(ics, dtype=torch.float32),
            torch.tensor(sols, dtype=torch.float32),
            torch.tensor(x_all, dtype=torch.float32),
            torch.tensor(t_all, dtype=torch.float32))