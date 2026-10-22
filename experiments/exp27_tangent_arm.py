import numpy as np, torch, sys, io, contextlib, time
sys.path.insert(0, '.'); sys.path.insert(0, '../experiments')
from torch.func import jvp
from config import ExperimentConfig
from method_fins import train_solution_operator, equivariance_loss
lam = float(sys.argv[1]); epochs = int(sys.argv[2])
d = torch.load('/tmp/tangent_data.pt', weights_only=False)
u0, u, x, t = d['u0'], d['u'], d['x'], d['t']
n, Nx, Nt = u.shape
cfg = ExperimentConfig(); cfg.device='cpu'
cfg.grid_size, cfg.time_steps, cfg.batch_size = Nx, Nt, 2
cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers = 8, 16, 2
Tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
ics, sols = Tt(u0).view(n,1,Nx), Tt(u).view(n,1,Nx,Nt)
xs, ts = Tt(np.tile(x,(n,1))).view(n,1,Nx), Tt(np.tile(t,(n,1))).view(n,1,Nt)
dx, dt = xs[:,0,1]-xs[:,0,0], ts[:,0,1]-ts[:,0,0]
Z = torch.zeros_like(sols); X, Tm = np.meshgrid(x, t, indexing='ij')
gal = (Tt(np.tile(Tm,(n,1,1,1))).view(n,1,Nx,Nt), Z, torch.ones_like(sols))
ref = float(sols.pow(2).mean())
torch.manual_seed(0); t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    S = train_solution_operator(ics, sols, xs, ts, cfg, 'cpu', epochs=epochs, verbose=False,
                                tangent=d[sys.argv[3]] if lam > 0 else None, lambda_tangent=lam)
for p in S.parameters(): p.requires_grad_(True)
ru = np.sqrt(float(equivariance_loss(S, Z, Z, sols, ics, sols, xs, ts, dx, dt, 'cpu', mode='jvp', channel_norm='q')))
rg = np.sqrt(float(equivariance_loss(S, *gal, ics, sols, xs, ts, dx, dt, 'cpu', mode='jvp', channel_norm='q')))
with torch.no_grad(): e = np.sqrt(float((S(ics, xs, ts)-sols).pow(2).mean())/ref)
# held-out Jacobian error on directions never used in training
hu0, hdl, hre = d['held']
xh, th = xs[:1].expand(hu0.shape[0],-1,-1), ts[:1].expand(hu0.shape[0],-1,-1)
_, jv = jvp(lambda z: S(z, xh, th), (hu0,), (hdl,))
jac_err = float(((jv-hre).pow(2).mean(dim=(1,2,3))/hre.pow(2).mean(dim=(1,2,3))).sqrt().mean())
print(f"lambda={lam:<4} epochs={epochs}: rel L2 {e:.3f} | held-out Jacobian err {jac_err:.3f} | "
      f"R(Galilean) {rg:.3f} | R(u d_u) {ru:.3f}   [{time.time()-t0:.0f}s]")
