"""
LieNLSD (Hu et al., ICML 2025) — Corrected implementation.
Hardcoded prolongation per PDE (matching discovery.py in hulx2002/LieNLSD).
Function library: 10 monomials {1, t, x, u, t^2, x^2, u^2, tx, tu, xu}.
NN surrogate for cross-derivatives. SVD on assembled system.
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
