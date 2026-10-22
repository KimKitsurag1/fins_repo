"""exp48 learned arm, diagnostic (exploratory, not registered): how a KdV stepper propagates modes that carry only noise.

On the protocol's periodised solitons the energy above k = 32 is ~1e-19 of the field: there the snapshots of a noisy
data set carry noise only. The exact solver transports such modes dispersively, with gain ~1; a stepper trained on
noisy pairs may learn to damp them. Gain after k steps: ||P dS^k z|| / ||z|| for z white noise restricted to the band
P (|k| > --kc), at the evaluation states FINS uses (exp48 windows), through each condition's stepper and, as a
reference, through the exact solver (central differences, relative step 1e-6). Also the same gain in the band
0 < |k| <= --kc for comparison.

Run:  python exp48_gain.py --conds clean,clean-rt,n1e-4,n1e-3,n1e-2 --seeds 42,43,44 --json gain.json
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pde"))
import exp31_polynomial_discovery as e31                   # noqa: E402
import exp39_localised as e39                              # noqa: E402
import corrupt as cr                                       # noqa: E402

CONDS = {"clean": (0.0, 1, ""), "clean-rt": (0.0, 1, "_clean-rt"), "n1e-4": (1e-4, 1, "_n1e-4"),
         "n3e-4": (3e-4, 1, "_n3e-4"), "n1e-3": (1e-3, 1, "_n1e-3"), "n1e-2": (1e-2, 1, "_n1e-2"),
         "t2": (0.0, 2, "_t2"), "t4": (0.0, 4, "_t4"), "t8": (0.0, 8, "_t8")}


def band_noise(n, Nx, kc, high, rng):
    k = np.abs(np.fft.fftfreq(Nx, d=1.0 / Nx))
    m = (k > kc) if high else ((k > 0) & (k <= kc))
    z = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal((n, Nx)), axis=1) * m, axis=1))
    return z / np.linalg.norm(z, axis=1, keepdims=True), m


def project(v, m):
    return np.real(np.fft.ifft(np.fft.fft(v, axis=-1) * m, axis=-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conds", default="clean,clean-rt,n1e-4,n3e-4,n1e-3,n1e-2")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--kc", type=int, default=32)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--draws", type=int, default=4, help="noise vectors per start")
    ap.add_argument("--exact", action="store_true", help="also the exact solver's gain (slower)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = e31.ExperimentConfig()
    T, Nx, Nt = 0.05, 128, 128
    L, x, t = e39.grid("kdv", cfg, Nx, Nt, T)
    out = {}
    for s in [int(v) for v in args.seeds.split(",")]:
        dseed = 1000 + s
        u_all, _, _ = e39.evaluation_data("kdv", "periodic", args.n, dseed, x, t, 0.002)
        for c in args.conds.split(","):
            sigma, thin, tag = CONDS[c]
            path = e39.stepper_path(SimpleNamespace(pde="kdv", seed=s, train_ic="mixed", tag=tag, stepper=None))
            if not os.path.exists(path):
                print(f"s{s} {c}: no stepper at {path}")
                continue
            ns = SimpleNamespace(pde="kdv", seed=s, train_ic="mixed", tag=tag, stepper=None, modes=64,
                                 train_if_missing=False)
            stp = e39.load_or_train(ns, x, T, Nt)
            _, u = cr.corrupt(u_all, sigma, thin, dseed, cr.EVAL)
            K = 16 // thin
            starts = list(range(2, u.shape[2] - K - 2, K))
            u0 = np.concatenate([u[:, :, j] for j in starts])
            rng = np.random.default_rng(s)
            res = {}
            for high in (True, False):
                gk = {1: [], K: []}
                for _ in range(args.draws):
                    z, m = band_noise(len(u0), Nx, args.kc, high, rng)
                    st, tg = torch.tensor(u0)[:, None], torch.tensor(z)[:, None]
                    for k in range(1, K + 1):
                        st, tg = jvp(stp, (st,), (tg,))
                        if k in gk:
                            gk[k].append(np.linalg.norm(project(tg[:, 0].numpy(), m), axis=1))
                res["high" if high else "low"] = {str(k): float(np.mean(v)) for k, v in gk.items()}
            if args.exact:
                z, m = band_noise(len(u0), Nx, args.kc, True, rng)
                eps = 1e-6 * np.linalg.norm(u0, axis=1, keepdims=True)
                sol = e39.solve("kdv", np.concatenate([u0 + eps * z, u0 - eps * z]), K * (t[1] - t[0]) * thin,
                                K * thin + 1, 0.002)
                dS = (sol[:len(u0), :, -1] - sol[len(u0):, :, -1]) / (2 * eps)
                res["high_exact"] = {str(K): float(np.mean(np.linalg.norm(project(dS, m), axis=1)))}
            out[f"s{s} {c}"] = res
            print(f"s{s} {c:8s} K={K:2d}  gain |k|>{args.kc}: 1 step {res['high']['1']:.3f}, K steps "
                  f"{res['high'][str(K)]:.3f}" + (f" (exact {res['high_exact'][str(K)]:.3f})" if args.exact else "")
                  + f";  0<|k|<={args.kc}: 1 step {res['low']['1']:.3f}, K steps {res['low'][str(K)]:.3f}",
                  flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
