"""
LieNLSD (Hu et al., ICML 2025; github.com/hulx2002/LieNLSD).

- run_lienlsd: a wrapper that, unlike LieNLSD, substitutes the textbook equation for
  LieNLSD's fitted network, takes cross derivatives from a network fitted to u(x, t), and returns the t-component
  as xi.
- run_lienlsd_general: protocol v1 form of that wrapper (class-degree library, the data's equation, components in
  (xi, eta, phi) order, column-scaled SVD).
- run_lienlsd_published: LieNLSD as published (an MLP F fitted to u_t on finite-difference jets, F's Jacobian in
  the determining system) with the protocol's library, readout and component order; with the exact right-hand
  side in place of F it gives the oracle row.
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import copy
import os
import itertools
import sympy as sp
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from adan import Adan        # optimiser of the official LieNLSD training (github.com/sail-sg/Adan)
except ImportError:
    Adan = None

N_THETA = 10

class LieNLSDSurrogate(nn.Module):
    def __init__(self, hidden=200, n_layers=3):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Sigmoid()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Sigmoid(), nn.Linear(hidden, hidden)])
        layers.extend([nn.Sigmoid(), nn.Linear(hidden, 1)])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def theta(t, x, u):
    return np.array([1, t, x, u, t**2, x**2, u**2, t*x, t*u, x*u])

def dtheta_dx(t, x, u, ux):
    return np.array([0, 0, 1, ux, 0, 2*x, 2*u*ux, t, t*ux, u+x*ux])

def dtheta_dxdx(t, x, u, ux, uxx):
    return np.array([0, 0, 0, uxx, 0, 2, 2*ux**2+2*u*uxx, 0, t*uxx, 2*ux+x*uxx])

def dtheta_dxdxdx(t, x, u, ux, uxx, uxxx):
    return np.array([0, 0, 0, uxxx, 0, 0, 6*ux*uxx+2*u*uxxx, 0, t*uxxx, 3*uxx+x*uxxx])

def dtheta_dt(t, x, u, ut):
    return np.array([0, 1, 0, ut, 2*t, 0, 2*u*ut, x, u+t*ut, x*ut])


def build_theta_n_heat(t, x, u, ux, ut, uxx, utx):
    """Prolongation matrix for heat/burgers [4, 30]. Matches discovery.py."""
    th = theta(t, x, u)
    dx = dtheta_dx(t, x, u, ux)
    dxx = dtheta_dxdx(t, x, u, ux, uxx)
    dt = dtheta_dt(t, x, u, ut)

    M = np.zeros((4, 30))
    M[0, 20:] = th                              # phi
    M[1, :10] = -ut * dx                        # xi -> phi_x
    M[1, 10:20] = -ux * dx                      # eta -> phi_x
    M[1, 20:] = dx                              # phi -> phi_x
    M[2, :10] = -(ut * dxx + 2*utx * dx)        # xi -> phi_xx
    M[2, 10:20] = -(ux * dxx + 2*uxx * dx)      # eta -> phi_xx
    M[2, 20:] = dxx                             # phi -> phi_xx
    M[3, :10] = -ut * dt                         # xi -> eta_t
    M[3, 10:20] = -ux * dt                       # eta -> eta_t
    M[3, 20:] = dt                              # phi -> eta_t
    return M


def build_theta_n_kdv(t, x, u, ux, ut, uxx, uxxx, utx, utxx):
    """Prolongation for KdV [5, 30]."""
    th = theta(t, x, u)
    dx = dtheta_dx(t, x, u, ux)
    dxx = dtheta_dxdx(t, x, u, ux, uxx)
    dxxx = dtheta_dxdxdx(t, x, u, ux, uxx, uxxx)
    dt = dtheta_dt(t, x, u, ut)

    M = np.zeros((5, 30))
    M[0, 20:] = th
    M[1, :10] = -ut * dx
    M[1, 10:20] = -ux * dx
    M[1, 20:] = dx
    M[2, :10] = -(ut * dxx + 2*utx * dx)
    M[2, 10:20] = -(ux * dxx + 2*uxx * dx)
    M[2, 20:] = dxx
    M[3, :10] = -(ut * dxxx + 3*utx*dxx + 3*utxx*dx)
    M[3, 10:20] = -(ux * dxxx + 3*uxx*dxx + 3*uxxx*dx)
    M[3, 20:] = dxxx
    M[4, :10] = -ut * dt
    M[4, 10:20] = -ux * dt
    M[4, 20:] = dt
    return M


def compute_fd(u, dx, dt):
    """Finite differences: ux, ut, uxx, uxxx."""
    Nx, Nt = u.shape
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:]-u[:-2])/(2*dx); ux[0] = (u[1]-u[-1])/(2*dx); ux[-1] = (u[0]-u[-2])/(2*dx)
    ut = np.zeros_like(u)
    ut[:,1:-1] = (u[:,2:]-u[:,:-2])/(2*dt); ut[:,0] = (u[:,1]-u[:,0])/dt; ut[:,-1] = (u[:,-1]-u[:,-2])/dt
    uxx = np.zeros_like(u)
    uxx[1:-1] = (u[2:]-2*u[1:-1]+u[:-2])/dx**2
    uxx[0] = (u[1]-2*u[0]+u[-1])/dx**2; uxx[-1] = (u[0]-2*u[-1]+u[-2])/dx**2
    uxxx = np.zeros_like(u)
    for i in range(Nx):
        uxxx[i] = (u[(i+2)%Nx]-2*u[(i+1)%Nx]+2*u[(i-1)%Nx]-u[(i-2)%Nx])/(2*dx**3)
    return ux, ut, uxx, uxxx


def nn_cross_derivs(model, x, t, idx, device):
    """Cross-derivatives via NN autograd: utx, utxx."""
    inp = torch.tensor([[x, t, idx]], dtype=torch.float32, device=device, requires_grad=True)
    out = model(inp)
    g = torch.autograd.grad(out, inp, create_graph=True)[0]
    dudx, dudt = g[0,0], g[0,1]
    # utx = d²u/dtdx
    try:
        g2 = torch.autograd.grad(dudt, inp, create_graph=True, retain_graph=True)[0]
        utx = g2[0,0].item()
    except:
        utx = 0.0
    # utxx = d³u/dt dx²
    try:
        g3 = torch.autograd.grad(dudx, inp, create_graph=True, retain_graph=True)[0]
        dudxdx = g3[0,0]
        g4 = torch.autograd.grad(dudxdx, inp, retain_graph=True)[0]
        utxx = g4[0,1].item()
    except:
        utxx = 0.0
    return utx, utxx


def train_surrogate(solutions, x_grid, t_grid, cfg, device):
    N, Nx, Nt = solutions.shape
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
    inputs, targets = [], []
    for s in range(min(N, 20)):
        for xi in range(0, Nx, max(1, Nx//32)):
            for ti in range(0, Nt, max(1, Nt//32)):
                inputs.append([X[xi,ti], T[xi,ti], float(s)/max(N,1)])
                targets.append(solutions[s, xi, ti])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    model = LieNLSDSurrogate(hidden=cfg.lienlsd_hidden, n_layers=cfg.lienlsd_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=256, shuffle=True)
    for _ in range(min(cfg.lienlsd_epochs, 200)):
        for xb, yb in loader:
            loss = F.mse_loss(model(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def run_lienlsd(train_sol, pde_type, x_grid, t_grid, cfg, device):
    B, C, Nx, Nt = train_sol.shape
    sols = train_sol[:, 0].numpy()
    dx = x_grid[1]-x_grid[0]
    dt = t_grid[1]-t_grid[0]

    print("    LieNLSD: training NN surrogate...")
    nn_model = train_surrogate(sols, x_grid, t_grid, cfg, device)

    print("    LieNLSD: building determining system...")
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
    all_rows = []

    for s in range(min(len(sols), 10)):
        u = sols[s]
        ux, ut, uxx, uxxx = compute_fd(u, dx, dt)
        n_sample = min(200, Nx*Nt)
        indices = np.random.choice(Nx*Nt, n_sample, replace=False)

        for pt in indices:
            xi_idx, ti_idx = pt // Nt, pt % Nt
            x_v, t_v = float(x_grid[xi_idx]), float(t_grid[ti_idx])
            u_v = float(u[xi_idx, ti_idx])
            ux_v, ut_v = float(ux[xi_idx, ti_idx]), float(ut[xi_idx, ti_idx])
            uxx_v, uxxx_v = float(uxx[xi_idx, ti_idx]), float(uxxx[xi_idx, ti_idx])

            utx, utxx = nn_cross_derivs(nn_model, x_v, t_v, float(s)/max(len(sols),1), device)

            if pde_type in ['heat', 'burgers']:
                M = build_theta_n_heat(t_v, x_v, u_v, ux_v, ut_v, uxx_v, utx)
                if pde_type == 'heat':
                    row = M[3] - cfg.alpha_heat * M[2]  # ut = a*uxx
                else:
                    row = M[3] + u_v * M[1] + ux_v * M[0] - cfg.nu_burgers * M[2]
                all_rows.append(row)
            elif pde_type == 'kdv':
                M = build_theta_n_kdv(t_v, x_v, u_v, ux_v, ut_v, uxx_v, uxxx_v, utx, utxx)
                row = M[4] + 6*u_v*M[1] + 6*ux_v*M[0] + M[3]
                all_rows.append(row)

    if not all_rows:
        return [], np.array([0.0]), np.array([0.0])

    W = np.array(all_rows, dtype=np.float64)
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    S_norm = S / (S[0] + 1e-12)

    threshold = 0.01
    n_gen = max(1, min(int(np.sum(S_norm < threshold)), cfg.n_sym))

    coeffs_list = [Vh[-(i+1)] for i in range(n_gen)]

    # Convert to vector fields
    u_sample = sols[0]
    generators_np, norms = [], []
    for coeffs in coeffs_list:
        th = np.stack([np.ones_like(u_sample), T, X, u_sample,
                       T**2, X**2, u_sample**2, T*X, T*u_sample, X*u_sample], axis=-1)
        xi = (th * coeffs[:10]).sum(axis=-1)
        eta = (th * coeffs[10:20]).sum(axis=-1)
        phi = (th * coeffs[20:30]).sum(axis=-1)
        norm = np.sqrt(np.mean(xi**2)+np.mean(eta**2)+np.mean(phi**2))
        generators_np.append((xi, eta, phi))
        norms.append(norm)

    norms = np.array(norms) if norms else np.array([0.0])
    print(f"    LieNLSD: found {len(generators_np)} generators")
    return generators_np, norms, S


# ---------------------------------------------------------------------------
# General library (protocol v1). The function above keeps the published
# configuration; this one differs in three documented ways:
#  (1) library of all monomials t^a x^b u^c with a + b + c <= degree, total
#      derivatives derived symbolically;
#  (2) the determining equation of the equation the data actually satisfy
#      (eq = coefficients below), not the textbook Burgers form;
#  (3) components returned in the (xi, eta, phi) = (x, t, u) order of the
#      ground truth: in the prolongation the block multiplying -u_t is the
#      t-component, which run_lienlsd returns as xi.
# eq: heat {"alpha"}: u_t = alpha u_xx; burgers {"a", "b"}: u_t + a u u_x = b u_xx;
#     kdv {"a", "c"}: u_t + a u u_x + c u_xxx = 0.

_T, _X, _U, _UX, _UXX, _UXXX, _UXXXX, _UT = sp.symbols("t x u ux uxx uxxx uxxxx ut")


def _dx(f):
    return (sp.diff(f, _X) + sp.diff(f, _U) * _UX + sp.diff(f, _UX) * _UXX + sp.diff(f, _UXX) * _UXXX
            + sp.diff(f, _UXXX) * _UXXXX)


def library(degree):
    """Exponents (a, b, c) and numpy callables for theta, D_x theta, ..., D_x^4 theta, D_t theta."""
    exps = [m for m in itertools.product(range(degree + 1), repeat=3) if sum(m) <= degree]
    th = [_T ** a * _X ** b * _U ** c for a, b, c in exps]
    d1 = [_dx(f) for f in th]
    d2 = [_dx(f) for f in d1]
    d3 = [_dx(f) for f in d2]
    d4 = [_dx(f) for f in d3]
    dt = [sp.diff(f, _T) + sp.diff(f, _U) * _UT for f in th]
    args = (_T, _X, _U, _UX, _UXX, _UXXX, _UXXXX, _UT)
    fns = [sp.lambdify(args, sp.Matrix(v), "numpy") for v in (th, d1, d2, d3, d4, dt)]
    return exps, fns


def _lib_eval(fns, *vals):
    return [np.asarray(f(*vals), dtype=np.float64).ravel() for f in fns]


def run_lienlsd_general(train_sol, pde_type, x_grid, t_grid, cfg, device, degree, eq,
                        n_samples=10, n_points=200, threshold=0.01, n_max=None, seed=0):
    """Returns (coeffs [n_gen, 3 n] in (xi, eta, phi) block order, exps, singular values)."""
    rng = np.random.default_rng(seed)
    B, C, Nx, Nt = train_sol.shape
    sols = train_sol[:, 0].numpy().astype(np.float64)
    dx, dt = x_grid[1] - x_grid[0], t_grid[1] - t_grid[0]
    nn_model = train_surrogate(sols, x_grid, t_grid, cfg, device)
    exps, fns = library(degree)
    n = len(exps)
    rows = []
    for s in range(min(len(sols), n_samples)):
        u = sols[s]
        ux, ut, uxx, uxxx = compute_fd(u, dx, dt)
        for pt in rng.choice(Nx * Nt, min(n_points, Nx * Nt), replace=False):
            i, j = pt // Nt, pt % Nt
            x_v, t_v, u_v = float(x_grid[i]), float(t_grid[j]), float(u[i, j])
            ux_v, ut_v, uxx_v, uxxx_v = float(ux[i, j]), float(ut[i, j]), float(uxx[i, j]), float(uxxx[i, j])
            utx, utxx = nn_cross_derivs(nn_model, x_v, t_v, float(s) / max(len(sols), 1), device)
            th, d1, d2, d3, _, dtt = _lib_eval(fns, t_v, x_v, u_v, ux_v, uxx_v, uxxx_v, 0.0, ut_v)
            # prolongation rows over blocks [tau | xi | phi] (tau = t-component)
            P0 = np.concatenate([np.zeros(n), np.zeros(n), th])
            P1 = np.concatenate([-ut_v * d1, -ux_v * d1, d1])
            P2 = np.concatenate([-(ut_v * d2 + 2 * utx * d1), -(ux_v * d2 + 2 * uxx_v * d1), d2])
            P3 = np.concatenate([-(ut_v * d3 + 3 * utx * d2 + 3 * utxx * d1),
                                 -(ux_v * d3 + 3 * uxx_v * d2 + 3 * uxxx_v * d1), d3])
            Pt = np.concatenate([-ut_v * dtt, -ux_v * dtt, dtt])
            if pde_type == "heat":
                rows.append(Pt - eq["alpha"] * P2)
            elif pde_type == "burgers":
                rows.append(Pt + eq["a"] * (u_v * P1 + ux_v * P0) - eq["b"] * P2)
            else:
                rows.append(Pt + eq["a"] * (u_v * P1 + ux_v * P0) + eq["c"] * P3)
    W = np.array(rows)
    # column scaling: monomials of very different size would otherwise set the null space
    sc = np.sqrt((W ** 2).mean(0))
    sc = np.where(sc > 1e-12, sc, 1.0)
    _, S, Vh = np.linalg.svd(W / sc, full_matrices=False)
    S_norm = S / (S[0] + 1e-12)
    n_gen = max(1, min(int(np.sum(S_norm < threshold)), n_max or cfg.n_sym))
    coeffs = []
    for i in range(n_gen):
        c = Vh[-(i + 1)] / sc
        coeffs.append(np.concatenate([c[n:2 * n], c[:n], c[2 * n:]]))    # -> (xi, eta, phi)
    return np.array(coeffs), exps, S


def lienlsd_fields(coeffs, exps, x_grid, t_grid, u):
    """(xi, eta, phi) fields [Nx, Nt] of each generator on one trajectory u [Nx, Nt]."""
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
    th = np.stack([T ** a * X ** b * u ** c for a, b, c in exps], -1)
    n = len(exps)
    return [((th * c[:n]).sum(-1), (th * c[n:2 * n]).sum(-1), (th * c[2 * n:]).sum(-1)) for c in coeffs]


# ---------------------------------------------------------------------------
# LieNLSD as published (train.py, discovery.py, data_utils/*.py of hulx2002/LieNLSD): F = MLP(200 x 3, sigmoid) fitted
# to u_t as a function of (u, u_x, u_xx) (heat, Burgers) or (u, u_x, u_xx, u_xxx) (KdV), Adan lr 1e-3, batch 256, MSE,
# epochs 200 / 100 / 200; jets by central differences (periodic in x, interior times); F's Jacobian in the determining
# system. Protocol adaptations as in run_lienlsd_general (class-degree library, column-scaled SVD, (xi, eta, phi)
# order, at most k generators). In the baselines' favour: inputs and target standardised (undone in the Jacobian),
# training points from all training trajectories, checkpoint every 10 epochs chosen on held-out trajectories.

FIT_EPOCHS = {"burgers": 200, "heat": 100, "kdv": 200, "bo": 200, "ks": 200}   # bo, ks: as KdV
_FITS = {}      # fitted F by (data fingerprint, settings): variants that differ only in the system reuse one fit
JET_COLS = {"heat": ("u", "ux", "uxx"), "burgers": ("u", "ux", "uxx"), "kdv": ("u", "ux", "uxx", "uxxx"),
            "bo": ("u", "ux", "uxx", "uxxx"),          # Benjamin-Ono: nonlocal; F is fitted on jets up to u_xxx
            "ks": ("u", "ux", "uxx", "uxxx", "uxxxx")}


def _cx(f, dx):
    return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dx)


def _cxx(f, dx):
    return (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / dx ** 2


def _cxxx(f, dx):
    return (np.roll(f, -2, 1) - 2 * np.roll(f, -1, 1) + 2 * np.roll(f, 1, 1) - np.roll(f, 2, 1)) / (2 * dx ** 3)


def _cxxxx(f, dx):
    return (np.roll(f, -2, 1) - 4 * np.roll(f, -1, 1) + 6 * f - 4 * np.roll(f, 1, 1) + np.roll(f, 2, 1)) / dx ** 4


def fd_jets(u, dx, dt):
    """Official central differences: u [N, Nx, Nt] -> arrays [N, Nx, Nt - 2] on the interior times."""
    ut = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dt)
    v = u[:, :, 1:-1]
    return dict(u=v, ux=_cx(v, dx), uxx=_cxx(v, dx), uxxx=_cxxx(v, dx), uxxxx=_cxxxx(v, dx), ut=ut, utx=_cx(ut, dx),
                utxx=_cxx(ut, dx), utxxx=_cxxx(ut, dx))


def spectral_jets(u, dx, dt):
    """FINS's derivatives: spectral in x (periodic data), five-point fourth-order central in t.
    u [N, Nx, Nt] -> arrays [N, Nx, Nt - 4] on times 2 .. Nt - 3."""
    k = 2 * np.pi * np.fft.fftfreq(u.shape[1], d=dx)[None, :, None]

    def d(f, n):
        return np.real(np.fft.ifft((1j * k) ** n * np.fft.fft(f, axis=1), axis=1))

    ut = (-u[:, :, 4:] + 8 * u[:, :, 3:-1] - 8 * u[:, :, 1:-3] + u[:, :, :-4]) / (12 * dt)
    v = u[:, :, 2:-2]
    return dict(u=v, ux=d(v, 1), uxx=d(v, 2), uxxx=d(v, 3), uxxxx=d(v, 4), ut=ut, utx=d(ut, 1), utxx=d(ut, 2),
                utxxx=d(ut, 3))


JETS = {"fd": (fd_jets, 1), "spectral": (spectral_jets, 2)}     # jet function, index of its first time


def sample_jets(sols, dx, dt, cols, n_points, rng, jets="fd"):
    """n_points (jet, u_t) pairs drawn uniformly over trajectories, x and the jets' times."""
    N, Nx, Nt = sols.shape
    jf, off = JETS[jets]
    n, i, j = rng.integers(0, N, n_points), rng.integers(0, Nx, n_points), rng.integers(0, Nt - 2 * off, n_points)
    X, y = np.empty((n_points, len(cols))), np.empty(n_points)
    for s in np.unique(n):
        m = n == s
        J = jf(sols[s:s + 1], dx, dt)
        X[m] = np.stack([J[c][0, i[m], j[m]] for c in cols], -1)
        y[m] = J["ut"][0, i[m], j[m]]
    return X, y


class RHSNet(nn.Module):
    """Official LieNLSD MLP (model.py) between input standardisation and output de-standardisation."""

    def __init__(self, mx, sx, my, sy, hidden=200, n_layers=3):
        super().__init__()
        layers = [nn.Linear(len(mx), hidden), nn.Sigmoid()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Sigmoid()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, 1))
        for k, v in (("mx", mx), ("sx", sx), ("my", my), ("sy", sy)):
            self.register_buffer(k, torch.as_tensor(np.asarray(v), dtype=torch.float32))

    def forward(self, z):
        return self.net((z - self.mx) / self.sx) * self.sy + self.my


def fit_rhs(sols, x_grid, t_grid, pde, epochs, seed, n_val_traj=64, every=10, batch=256, lr=1e-3, jets="fd",
            n_fit=None, n_val=None):
    """F: jets -> u_t on n_fit points of the training trajectories (default 10 Nx Nt, ten of our trajectories; the
    official ten initial conditions hold ~1e6 points), validation on n_val (default 2 Nx Nt) points of n_val_traj
    held-out ones. Returns (net, log)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    N, Nx, Nt = sols.shape
    dx, dt = x_grid[1] - x_grid[0], t_grid[1] - t_grid[0]
    cols = JET_COLS[pde]
    Xtr, ytr = sample_jets(sols[:N - n_val_traj], dx, dt, cols, n_fit or 10 * Nx * Nt, rng, jets)
    Xva, yva = sample_jets(sols[N - n_val_traj:], dx, dt, cols, n_val or 2 * Nx * Nt, rng, jets)
    net = RHSNet(Xtr.mean(0), Xtr.std(0) + 1e-12, ytr.mean(), ytr.std() + 1e-12)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)[:, None]
    if Adan is None and not os.environ.get("LIENLSD_ALLOW_ADAM"):
        # the published optimiser, or Adam by explicit opt-in
        raise RuntimeError("LieNLSD as published trains F with Adan: pip install git+https://github.com/sail-sg/Adan.git "
                           "(or set LIENLSD_ALLOW_ADAM=1)")
    opt = Adan(net.parameters(), lr) if Adan is not None else optim.Adam(net.parameters(), lr)

    def val():
        with torch.no_grad():
            return float(((net(Xv) - yv) ** 2).mean() / (yv ** 2).mean())

    log = dict(optimizer=type(opt).__name__, n_train=len(Xtr), n_val=len(Xva), epochs=epochs, val=[(0, val())])
    best, best_state = log["val"][0][1], copy.deepcopy(net.state_dict())
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), batch):
            idx = perm[b:b + batch]
            loss = F.mse_loss(net(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % every == 0 or ep + 1 == epochs:
            v = val()
            log["val"].append((ep + 1, v))
            if v < best:
                best, best_state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    log["best_epoch"] = min(log["val"], key=lambda e: e[1])[0]
    log["val_rel_mse"] = best
    return net, log


def exact_jacobian(pde, eq, P):
    """dF / d(u, u_x, u_xx[, u_xxx]) of the equation the data satisfy at jets P [n, ncols]."""
    u, ux, one = P[:, 0], P[:, 1], np.ones(len(P))
    if pde == "bo":                                     # no jet-space right-hand side: diagnostics only (nan)
        return np.full((len(P), 4), np.nan)
    if pde == "ks":                                     # u_t = -u u_x - u_xx - u_xxxx
        return np.stack([-ux, -u, -one, 0 * u, -one], -1)
    if pde == "heat":
        return np.stack([0 * u, 0 * u, eq["alpha"] * one], -1)
    if pde == "burgers":
        return np.stack([-eq["a"] * ux, -eq["a"] * u, eq["b"] * one], -1)
    return np.stack([-eq["a"] * ux, -eq["a"] * u, 0 * u, -eq["c"] * one], -1)


def exact_rhs(pde, eq, P):
    if pde == "bo":
        return np.full(len(P), np.nan)
    if pde == "ks":
        return -P[:, 0] * P[:, 1] - P[:, 2] - P[:, 4]
    if pde == "heat":
        return eq["alpha"] * P[:, 2]
    if pde == "burgers":
        return -eq["a"] * P[:, 0] * P[:, 1] + eq["b"] * P[:, 2]
    return -eq["a"] * P[:, 0] * P[:, 1] - eq["c"] * P[:, 3]


def net_jacobian(net, P):
    z = torch.tensor(P, dtype=torch.float32, requires_grad=True)
    g, = torch.autograd.grad(net(z).sum(), z)
    return g.detach().numpy().astype(np.float64)


def net_rhs(net, P):
    with torch.no_grad():
        return net(torch.tensor(P, dtype=torch.float32))[:, 0].numpy().astype(np.float64)


def run_lienlsd_published(train_sol, pde_type, x_grid, t_grid, degree, n_max, eq, epochs=None, eval_u=None,
                          n_samples=10, n_points=200, threshold=0.01, seed=0, jets="fd", n_fit=None, n_val=None,
                          system=None):
    """epochs None: exact right-hand side (oracle); epochs >= 0: F fitted for that many epochs (0: at initialisation).
    jets: "fd" as published, "spectral" FINS's derivatives (sensitivity check). system: (u [N, Nx, Nt], t grid) of
    the trajectories the determining system is built on (default the training trajectories, as published).
    Returns (coeffs [n_max, 3 n] in (xi, eta, phi) order, ascending singular value; n_own; exps; S; diag)."""
    if epochs is None and pde_type == "bo":
        raise ValueError("Benjamin-Ono has no jet-space right-hand side: LieNLSD needs a fitted F")
    sols = train_sol[:, 0].numpy().astype(np.float64)
    dx = x_grid[1] - x_grid[0]
    cols = JET_COLS[pde_type]
    diag = {}
    net = None
    if epochs is not None:
        key = (pde_type, epochs, seed, jets, n_fit, n_val, sols.shape, float(sols.sum()), float((sols ** 2).sum()),
               float(t_grid[1] - t_grid[0]))
        if key not in _FITS:
            _FITS[key] = fit_rhs(sols, x_grid, t_grid, pde_type, epochs, seed, jets=jets, n_fit=n_fit, n_val=n_val)
        net, diag["fit"] = _FITS[key]
    sys_u, sys_t = (sols, t_grid) if system is None else (np.asarray(system[0], dtype=np.float64), system[1])
    N, Nx, Nt = sys_u.shape
    dt = sys_t[1] - sys_t[0]
    rng = np.random.default_rng(seed)
    exps, fns = library(degree)
    n = len(exps)
    jf, off = JETS[jets]
    nt = Nt - 2 * off
    pts = []
    for s in range(min(N, n_samples)):
        J = jf(sys_u[s:s + 1], dx, dt)
        for pt in rng.choice(Nx * nt, min(n_points, Nx * nt), replace=False):
            i, j = pt // nt, pt % nt
            pts.append((float(sys_t[j + off]), float(x_grid[i])) + tuple(float(J[c][0, i, j]) for c in
                                                                         ("u", "ux", "uxx", "uxxx", "uxxxx", "ut",
                                                                          "utx", "utxx", "utxxx")))
    pts = np.array(pts)
    P = pts[:, 2:2 + len(cols)]
    Jex = exact_jacobian(pde_type, eq, P)
    Jac = Jex if net is None else net_jacobian(net, P)
    diag["jac_rel"] = float(np.linalg.norm(Jac - Jex) / np.linalg.norm(Jex))
    diag["jac_rel_cols"] = (np.linalg.norm(Jac - Jex, axis=0) / np.linalg.norm(Jex)).tolist()
    rows = []
    for (t_v, x_v, u_v, ux_v, uxx_v, uxxx_v, uxxxx_v, ut_v, utx, utxx, utxxx), jr in zip(pts, Jac):
        th, d1, d2, d3, d4, dtt = _lib_eval(fns, t_v, x_v, u_v, ux_v, uxx_v, uxxx_v, uxxxx_v, ut_v)
        P0 = np.concatenate([np.zeros(n), np.zeros(n), th])
        P1 = np.concatenate([-ut_v * d1, -ux_v * d1, d1])
        P2 = np.concatenate([-(ut_v * d2 + 2 * utx * d1), -(ux_v * d2 + 2 * uxx_v * d1), d2])
        Pt = np.concatenate([-ut_v * dtt, -ux_v * dtt, dtt])
        row = Pt - jr[0] * P0 - jr[1] * P1 - jr[2] * P2
        if len(cols) >= 4:
            P3 = np.concatenate([-(ut_v * d3 + 3 * utx * d2 + 3 * utxx * d1),
                                 -(ux_v * d3 + 3 * uxx_v * d2 + 3 * uxxx_v * d1), d3])
            row = row - jr[3] * P3
        if len(cols) == 5:
            # fourth prolongation: phi^(xxxx) = D_x^4 Q + xi u_xxxxx + tau u_txxxx (Leibniz, the top terms cancel)
            P4 = np.concatenate([-(ut_v * d4 + 4 * utx * d3 + 6 * utxx * d2 + 4 * utxxx * d1),
                                 -(ux_v * d4 + 4 * uxx_v * d3 + 6 * uxxx_v * d2 + 4 * uxxxx_v * d1), d4])
            row = row - jr[4] * P4
        rows.append(row)
    W = np.array(rows)
    sc = np.sqrt((W ** 2).mean(0))
    sc = np.where(sc > 1e-12, sc, 1.0)
    _, S, Vh = np.linalg.svd(W / sc, full_matrices=False)
    n_own = max(1, min(int(np.sum(S / (S[0] + 1e-12) < threshold)), n_max))
    coeffs = []
    for i in range(n_max):
        c = Vh[-(i + 1)] / sc
        coeffs.append(np.concatenate([c[n:2 * n], c[:n], c[2 * n:]]))
    if eval_u is not None:
        Je = jf(np.asarray(eval_u, dtype=np.float64), dx, dt)
        Pe = np.stack([Je[c].ravel() for c in cols], -1)
        ute = Je["ut"].ravel()
        diag["rhs_rel_exact"] = float(np.linalg.norm(exact_rhs(pde_type, eq, Pe) - ute) / np.linalg.norm(ute))
        if net is not None:
            diag["rhs_rel_fit"] = float(np.linalg.norm(net_rhs(net, Pe) - ute) / np.linalg.norm(ute))
    diag["n_own"] = n_own
    diag["sv_rel"] = (S[::-1] / S[0])[:n_max + 4].tolist()
    return np.array(coeffs), n_own, exps, S, diag
