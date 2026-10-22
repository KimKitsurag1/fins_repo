"""Corrupted data for the noise and thinning experiment (exp48; README_EXPERIMENTS.md, "Noise and temporal thinning").

thin: every s-th snapshot. noise: i.i.d. Gaussian, sigma times each trajectory's standard deviation. The standard
normal draws are made on the full grid from (data seed, role) only -- role EVAL for evaluation trajectories, TRAIN for
training ones -- so every method, noise level and thinning sees the same draws.
"""

import numpy as np

EVAL, TRAIN = 0, 1


def corrupt(u, sigma=0.0, s=1, seed=0, role=EVAL):
    """u [n, Nx, Nt] -> (clean, corrupted), both [n, Nx, ceil(Nt / s)]."""
    clean = np.ascontiguousarray(u[:, :, ::s])
    if not sigma:
        return clean, clean.copy()
    z = np.random.default_rng([int(seed), int(role)]).standard_normal(u.shape)[:, :, ::s]
    sd = clean.reshape(len(clean), -1).std(1)[:, None, None]
    return clean, clean + sigma * sd * z


def corrupted(args):
    return bool(args.noise) or args.thin > 1
