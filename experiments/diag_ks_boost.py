"""Exploratory diagnostic (not registered): how a KS stepper moves a constant, the Galilean boost's phi. For tangents
q = 1 and q = -u_x (translation) at evaluation states (data 1000 + s, starts 16, 48, 80): relative error of the
learned J^k q - q against the exact solver's (central differences), k = 1, 4, 16.

Run:  python diag_ks_boost.py <seed> [attractor|mixed]
"""
import os
import sys
import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp39_localised as e39
import ks
from stepper_surrogate import Stepper
from window_stepper import to_float64

R = os.environ.get("FINS_RESULTS_DIR", "results")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
data = sys.argv[2] if len(sys.argv) > 2 else "attractor"
x = np.arange(128) * ks.L / 128
t = np.linspace(0, ks.T, 128)
u, _, ts = e39.evaluation_data("ks", "periodic", 4, 1000 + seed, x, t, 0.002)
stp = Stepper(modes=64, positional_embedding=None)
sd = torch.load(os.path.join(R, f"S_theta_ks_s{seed}_{data}_stepper.pt"), map_location="cpu", weights_only=True)
stp.load_state_dict({k: v for k, v in sd.items() if torch.is_tensor(v)})
stp = to_float64(stp).eval()
for p in stp.parameters():
    p.requires_grad_(False)
starts = [16, 48, 80]
u0 = np.concatenate([u[:, :, j] for j in starts])
k = 2 * np.pi * np.fft.fftfreq(128, d=ks.L / 128)
ux = np.real(np.fft.ifft(1j * k * np.fft.fft(u0, axis=1), axis=1))
K = 16
dt = t[1] - t[0]
for name, q in (("constant", np.ones_like(u0)), ("-u_x", -ux)):
    # exact: central differences through the solver
    eps = 1e-4 * np.linalg.norm(u0, axis=1, keepdims=True) / np.linalg.norm(q, axis=1, keepdims=True)
    sol = ks.solve(np.concatenate([u0 + eps * q, u0 - eps * q]), K * dt, K + 1)
    B = len(u0)
    ex = (sol[:B] - sol[B:]) / (2 * eps[:, :, None])
    s, tg = torch.tensor(u0)[:, None], torch.tensor(q)[:, None]
    errs = []
    for kk in range(1, K + 1):
        s, tg = jvp(stp, (s,), (tg,))
        le = tg[:, 0].numpy()
        e = ex[:, :, kk]
        errs.append(np.linalg.norm(le - e) / np.linalg.norm(e - q))
    print(f"s{seed} {data} {name:9s} rel. error of J^k q - q (learned vs exact): k=1 {errs[0]:.3f}, k=4 {errs[3]:.3f}, "
          f"k=16 {errs[15]:.3f}; |J^16 q - q| / |q| exact {np.linalg.norm(ex[:, :, 16] - q) / np.linalg.norm(q):.3f}")
