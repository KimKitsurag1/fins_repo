"""Weak SINDy for PDEs as published by Messenger and Bortz (J. Comput. Phys. 443, 2021), ported from the authors'
MATLAB code (github.com/MathBioCU/WSINDy_PDE, files named in the comments): a library of d_x^k (u^j) terms and a
constant, separable piecewise-polynomial test functions whose half-support and degree come from the corner of the
data's Fourier spectrum (findcorners.m), query points on a strided grid (at most K_max rows), the authors' scaling
(toggle_scale = 2, get_scales.m) and the modified sequential thresholding (sparsifyDynamics.m) with the threshold
chosen by the authors' loss over lambda = 10^linspace(-4, 0, 100) (wsindy_pde_RGLS_seq.m).

Extension used only as an oracle: Fourier-multiplier columns M(u^j) on periodic data, M given by its symbol; tested
like any other column, i.e. moved onto the test function by Parseval.
"""

from math import comb

import numpy as np
from scipy.optimize import brentq

TAU = 1e-10
TAUHAT = 2.0
LAMBDAS = 10.0 ** np.linspace(-4, 0, 100)


# ---------------------------------------------------------------------------------------------------------------------
# test functions (findcorners.m, phi_int_weights.m)

def corner_index(U, x, axis):
    """findcornerpts: changepoint of the cumulative mean |FFT| along `axis` of U [..]; returns the wavenumber index k
    (distance from the zero mode) that findcorners uses."""
    Uf = np.abs(np.fft.fftshift(np.fft.fft(np.moveaxis(U, axis, 0), axis=0), axes=0))
    Uf = Uf.reshape(Uf.shape[0], -1).mean(axis=1)
    L = len(x)
    wn = (np.arange(L) - L // 2) * 2 * np.pi / (x[-1] - x[0])
    xx = wn[:int(np.ceil(L / 2))]
    NN = len(xx)
    C = np.cumsum(Uf)[:int(np.ceil(L / 2))]
    errs = np.zeros(NN - 2)
    for k in range(2, NN):                                       # MATLAB k = 2..NN-1, 1-based subinds 1:k and k:NN
        a1, a2 = C[:k], C[k - 1:]
        x1, x2 = xx[:k], xx[k - 1:]
        m1 = np.ptp(a1) / np.ptp(x1)
        m2 = np.ptp(a2) / np.ptp(x2)
        L1 = a1.min() + m1 * (x1 - x1[0])
        L2 = a2.max() + m2 * (x2 - x2[-1])
        errs[k - 2] = np.sqrt(np.sum(((L1 - a1) / a1) ** 2) + np.sum(((L2 - a2) / a2) ** 2))
    # MATLAB's min skips NaN (0 / 0 where the spectrum's first cumulative value is exactly 0) and returns index 1 when
    # every entry is NaN; np.argmin would return the first NaN
    tstarind = 1 if np.isnan(errs).all() else int(np.nanargmin(errs)) + 1
    return max(NN - tstarind - 3, 1)


def _l(m, k, N):
    return np.log((2 * m - 1) / m ** 2) * (4 * np.pi ** 2 * k ** 2 * m ** 2 - 3 * N ** 2 * TAUHAT ** 2) \
        - 2 * N ** 2 * TAUHAT ** 2 * np.log(TAU)


def test_params(kx, Nx, kt, Nt, max_dx, max_dt, raw=False, m=None):
    """findcorners, phi_class 1: half-supports m_x, m_t (raw: the unrounded roots) and degrees p_x, p_t; with m given,
    only the rounding and the degrees."""
    if m is None:
        m1 = np.sqrt(3) / np.pi * Nx / 2 / kx * TAUHAT
        m2 = 1 / np.pi * TAUHAT * (Nx / 2) / kx * np.sqrt(np.log(np.e ** 3 / TAU ** 8))
        mx = min(brentq(_l, m1, m2, args=(kx, Nx)), Nx / 2 - 1)
        mt = min(brentq(_l, 1 + 1e-12, 2 / np.sqrt(TAU), args=(kt, Nt)), Nt / 2 - 1)
        if raw:
            return mx, mt
    else:
        mx, mt = m
    mx = int(min(np.floor((Nx - 1) / 2), np.ceil(mx)))
    mt = int(min(np.floor((Nt - 1) / 2), np.ceil(mt)))
    px = int(max(max_dx + 1, np.ceil(np.log(TAU) / np.log(1 - (1 - 1 / mx) ** 2))))
    pt = int(max(max_dt + 1, np.ceil(np.log(TAU) / np.log(1 - (1 - 1 / mt) ** 2))))
    return mx, mt, px, pt


def phi_weights(m, d, p):
    """phi_int_weights, phi_class 1: values of phi = (1 - s^2)^p and its first d derivatives (in s) at s = j / m,
    j = -m..m: [d + 1, 2 m + 1]."""
    t = np.arange(m + 1) / m
    q = np.arange(p, p - d - 1, -1)                              # fliplr(p-d:p)
    tL = (1 + t[None, :]) ** q[:, None]
    tR = (1 - t[None, :]) ** q[:, None]
    tL[:, m] = 0.0                                               # MATLAB fills columns 1..m only: the edge stays 0
    tR[:, m] = 0.0
    ps = np.ones(d + 1)
    for j in range(1, d + 1):
        ps[j] = (p - j + 1) * ps[j - 1]
    tL = ps[:, None] * tL
    tR = (((-1.0) ** np.arange(d + 1)) * ps)[:, None] * tR
    C = np.zeros((d + 1, 2 * m + 1))
    C[0] = np.concatenate([(tL[0] * tR[0])[::-1], (tL[0] * tR[0])[1:]])
    for k in range(1, d + 1):
        acc = np.zeros(m + 1)
        for j in range(k + 1):
            acc += comb(k, j) * tL[k - j] * tR[j]
        C[k] = np.concatenate([(-1) ** k * acc[::-1], acc[1:]])
    return C


# ---------------------------------------------------------------------------------------------------------------------
# library and linear system (build_fcn_lib_tags.m, get_scales.m, get_lib_columns.m, convNDfft.m)

def library(polys, max_dx, multipliers=(), mult_polys=None):
    """Terms (j, k, mult): d_x^k (u^j) for j in polys (j = 0 only as the constant), k = 0..max_dx; multiplier terms
    M_i(u^j) (k = 0, mult = i; multipliers are (symbol, order) pairs) for j in mult_polys (default: polys). The
    left-hand side d_t u is separate."""
    terms = [(0, 0, -1)]
    for j in polys:
        if j == 0:
            continue
        for k in range(max_dx + 1):
            terms.append((j, k, -1))
        if mult_polys is None or j in mult_polys:
            for i in range(len(multipliers)):
                terms.append((j, 0, i))
    return terms


def scale_u(U, P):
    """toggle_scale = 2."""
    v = U.ravel()
    v = v / np.linalg.norm(v, 2) ** (1 / P)
    s = np.linalg.norm(v, 2 * P) ** (P / max(P - 1, 1))
    return float(min(max(s, np.finfo(float).eps), 1 / np.finfo(float).eps))


def scale_xt(p, m, h, max_d):
    """get_scales, phi_class 1: scale of one coordinate."""
    if max_d == 0:
        return 1.0
    num = np.prod([p - i for i in range(int(np.floor(max_d / 2)))]) if max_d >= 2 else 1.0
    den = np.prod(np.arange(1, int(np.ceil(max_d / 2)) + 1))
    return float((num / den * np.prod(np.arange(1, max_d + 1))) ** (1 / max_d) / (m * h))


def conv_valid(X, cols, strides):
    """convNDfft (ver 2) over the two axes of X [Nx, Nt]: circular FFT convolution with each axis' column (placed at the
    end of a zero vector, as the authors do), kept at indices 0, s, ..., N - 2 m - 1 (their 1:s:N-2m)."""
    for ax, (col, s) in enumerate(zip(cols, strides)):
        N = X.shape[ax]
        n = len(col)
        h = np.zeros(N)
        h[N - n:] = col
        Xf = np.fft.ifft(np.fft.fft(h)[(slice(None),) + (None,) * (1 - ax)] * np.fft.fft(X, axis=ax), axis=ax)
        idx = np.arange(0, N - n + 1, s)
        X = np.take(Xf, idx, axis=ax)
    return np.real(X)


class WSINDy:
    """Weak SINDy over trajectories U [N, Nx, Nt] on a uniform grid (x [Nx], t [Nt]); periodic in x for the multiplier
    columns, which take symbols m(kappa) on the rfft wavenumbers."""

    def __init__(self, polys=(0, 1, 2, 3, 4), max_dx=6, multipliers=(), K_max=5000, mult_polys=None, terms=None):
        self.polys, self.max_dx, self.multipliers, self.K_max = tuple(polys), max_dx, tuple(multipliers), K_max
        self.mult_polys = mult_polys
        self.fixed_terms = None if terms is None else list(terms)

    def prepare(self, U, x, t, rows_total=None):
        """Test functions, scales and query strides from the data (findcorners, get_scales); returns the left-hand side
        rows b [K] (scaled) and the left-hand side's scale."""
        N, Nx, Nt = U.shape
        dx, dt = x[1] - x[0], t[1] - t[0]
        # findcorners: a half-support per data set (here per trajectory), averaged, then rounded up
        ks = [(corner_index(u, x, 0), corner_index(u, t, 1)) for u in U]
        mxs = [test_params(kx, Nx, kt, Nt, self.max_dx, 1, raw=True) for kx, kt in ks]
        mx, mt, px, pt = test_params(None, Nx, None, Nt, self.max_dx, 1,
                                     m=(np.mean([a for a, _ in mxs]), np.mean([b for _, b in mxs])))
        kx, kt = float(np.mean([a for a, _ in ks])), float(np.mean([b for _, b in ks]))
        P = max(self.polys)
        su = scale_u(U, P)
        sx = scale_xt(px, mx, dx, self.max_dx)
        st = scale_xt(pt, mt, dt, 1)
        Cx, Ct = phi_weights(mx, self.max_dx, px), phi_weights(mt, 1, pt)
        Cx = Cx * (mx * dx * sx) ** (-np.arange(self.max_dx + 1))[:, None]
        Ct = Ct * (mt * dt * st) ** (-np.arange(2))[:, None]
        K_per = self.K_max if rows_total is None else max(1, rows_total // N)
        s_x = max(int(np.ceil((Nx - 2 * mx) / K_per ** 0.5)), 1)
        s_t = max(int(np.ceil((Nt - 2 * mt) / K_per ** 0.5)), 1)
        self.prep = dict(Cx=Cx, Ct=Ct, s_x=s_x, s_t=s_t, su=su, sx=sx, Nx=Nx, dx=dx)
        b = np.concatenate([conv_valid(u / su, (Cx[0], Ct[1]), (s_x, s_t)).ravel() for u in U])
        self.info = dict(kx=kx, kt=kt, mx=mx, mt=mt, px=px, pt=pt, su=su, sx=sx, st=st, s_x=s_x, s_t=s_t,
                         rows=len(b))
        return b, su ** (-1) * st ** (-1)

    def columns(self, U, terms, multipliers=()):
        """G [K, len(terms)] (scaled) of terms (j, k, mi) and their scales M_scale (before the left-hand side's), on the
        test functions of the last prepare; mi indexes multipliers ((symbol, order) pairs)."""
        p = self.prep
        kap = 2 * np.pi * np.fft.rfftfreq(p["Nx"], d=p["dx"])
        syms = [np.asarray(f(kap)) * p["sx"] ** (-order) for f, order in multipliers]
        Gs = []
        for u in U:
            us = u / p["su"]
            cols = []
            for j, k, mi in terms:
                f = us ** j
                if mi >= 0:
                    f = np.fft.irfft(syms[mi][:, None] * np.fft.rfft(f, axis=0), n=p["Nx"], axis=0)
                cols.append(conv_valid(f, (p["Cx"][k], p["Ct"][0]), (p["s_x"], p["s_t"])).ravel())
            Gs.append(np.stack(cols, 1))
        # M_scale = prod(scales.^(-tags)): u^j d_x^k -> su^-j sx^-k; a multiplier of order r counts as d_x^r
        Mcol = np.array([p["su"] ** (-j) * p["sx"] ** (-(k if mi < 0 else multipliers[mi][1])) for j, k, mi in terms])
        return np.concatenate(Gs), Mcol

    def build(self, U, x, t, rows_total=None):
        """G [K, J], b [K] (scaled variables), the column scales M [J] and the test-function parameters."""
        b, Mlhs = self.prepare(U, x, t, rows_total)
        self.terms = (self.fixed_terms if self.fixed_terms is not None
                      else library(self.polys, self.max_dx, self.multipliers, self.mult_polys))
        G, Mcol = self.columns(U, self.terms, self.multipliers)
        return G, b, Mcol / Mlhs

    @staticmethod
    def stls(G, b, lam, M, maxits=np.inf):
        """sparsifyDynamics with the column scales M: the modified thresholding bounds [lam max(1, bnd), min(1, bnd) /
        lam], bnd = |b| / |G_i| M_i; returns coefficients in the original units (M times the scaled ones)."""
        nn = G.shape[1]
        Xi = M * np.linalg.lstsq(G, b, rcond=None)[0]
        bnds = np.linalg.norm(b) / np.linalg.norm(G, axis=0) * M
        LB, UB = lam * np.maximum(1, bnds), np.minimum(1, bnds) / lam
        small = np.zeros(nn, bool)
        its = 0
        while its < maxits:
            new = (np.abs(Xi) < LB) | (np.abs(Xi) > UB)
            if new.sum() == nn or (new == small).all():
                return Xi
            small = new
            Xi = np.where(small, 0.0, Xi)
            Xi[~small] = M[~small] * np.linalg.lstsq(G[:, ~small], b, rcond=None)[0]
            its += 1
        return Xi

    def fit(self, U, x, t, rows_total=None, lambdas=LAMBDAS):
        """MSTLS: the threshold minimising |G (W - W_ls)| / |G W_ls| + nnz(W) / J (alpha = 1/2). Returns the
        coefficients of self.terms in the original units, the chosen lambda and the relative residual."""
        G, b, M = self.build(U, x, t, rows_total)
        return self.mstls(G, b, M, lambdas)

    def mstls(self, G, b, M, lambdas=LAMBDAS, nterms=None):
        """MSTLS on a built system (G, b, M); as fit. nterms: the library size J in the loss (default the columns')."""
        W_ls = np.linalg.lstsq(G, b, rcond=None)[0]
        GW = np.linalg.norm(G @ W_ls)
        J = G.shape[1] if nterms is None else nterms
        loss = []
        for lam in lambdas:
            W = self.stls(G, b, lam, M) / M
            loss.append(np.linalg.norm(G @ (W - W_ls)) / GW + np.count_nonzero(W) / J)
        lam = lambdas[int(np.argmin(loss))]
        W = self.stls(G, b, lam, M)
        resid = float(np.linalg.norm(b - G @ (W / M)) / np.linalg.norm(b))
        self.info.update(lam=float(lam), loss=float(min(loss)), resid=resid)
        return W, lam, resid
