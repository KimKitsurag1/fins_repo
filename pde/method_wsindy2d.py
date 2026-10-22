"""Weak SINDy for PDEs in two space dimensions, the authors' algorithm as method_wsindy.py (separable piecewise-polynomial
test functions whose supports and degrees come from the data's spectrum along each axis, query points on a strided grid
with at most K_max rows per trajectory, the authors' scales, MSTLS). Library: d_x^p d_y^q (u^j) with p + q <= max_d,
j in polys (j = 0 only as the constant), plus Fourier-multiplier columns M(u^j) on periodic data (symbols m(kx, ky))."""

import numpy as np

import method_wsindy as mw


def conv_nd(X, cols, strides):
    """convNDfft over every axis of X: circular FFT convolution with each axis' column (placed at the end of a zero
    vector), kept at indices 0, s, ..., N - 2 m - 1."""
    nd = X.ndim
    for ax, (col, s) in enumerate(zip(cols, strides)):
        N, n = X.shape[ax], len(col)
        h = np.zeros(N)
        h[N - n:] = col
        shape = [1] * nd
        shape[ax] = N
        X = np.fft.ifft(np.fft.fft(h).reshape(shape) * np.fft.fft(X, axis=ax), axis=ax)
        X = np.take(X, np.arange(0, N - n + 1, s), axis=ax)
    return np.real(X)


def library(polys, max_d):
    """Terms (j, p, q): the constant (0, 0, 0) and d_x^p d_y^q (u^j), p + q <= max_d, for j in polys (j >= 1)."""
    terms = [(0, 0, 0)]
    for j in polys:
        if j == 0:
            continue
        for p in range(max_d + 1):
            for q in range(max_d + 1 - p):
                terms.append((j, p, q))
    return terms


class WSINDy2D:
    """Weak SINDy over trajectories U [N, Nx, Ny, Nt] on uniform grids x, y, t."""

    def __init__(self, polys=(0, 1, 2, 3), max_d=4, K_max=5000):
        self.polys, self.max_d, self.K_max = tuple(polys), max_d, K_max
        self.terms = library(self.polys, self.max_d)

    def prepare(self, U, x, y, t, rows_total=None):
        """Test functions, scales and query strides; returns the left-hand side rows b [K] (scaled) and its scale."""
        N, Nx, Ny, Nt = U.shape
        dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
        ks = [(mw.corner_index(u, x, 0), mw.corner_index(u, y, 1), mw.corner_index(u, t, 2)) for u in U]
        raw = [(mw.test_params(kx, Nx, kt, Nt, self.max_d, 1, raw=True), mw.test_params(ky, Ny, kt, Nt, self.max_d, 1,
                                                                                         raw=True)) for kx, ky, kt in ks]
        mx, mt, px, pt = mw.test_params(None, Nx, None, Nt, self.max_d, 1,
                                        m=(np.mean([a[0][0] for a in raw]), np.mean([a[0][1] for a in raw])))
        my, _, py, _ = mw.test_params(None, Ny, None, Nt, self.max_d, 1,
                                      m=(np.mean([a[1][0] for a in raw]), np.mean([a[0][1] for a in raw])))
        P = max(self.polys)
        su = mw.scale_u(U, P)
        sx, sy = mw.scale_xt(px, mx, dx, self.max_d), mw.scale_xt(py, my, dy, self.max_d)
        st = mw.scale_xt(pt, mt, dt, 1)
        Cx = mw.phi_weights(mx, self.max_d, px) * (mx * dx * sx) ** (-np.arange(self.max_d + 1))[:, None]
        Cy = mw.phi_weights(my, self.max_d, py) * (my * dy * sy) ** (-np.arange(self.max_d + 1))[:, None]
        Ct = mw.phi_weights(mt, 1, pt) * (mt * dt * st) ** (-np.arange(2))[:, None]
        K_per = self.K_max if rows_total is None else max(1, rows_total // N)
        s_x = max(int(np.ceil((Nx - 2 * mx) / K_per ** (1 / 3))), 1)
        s_y = max(int(np.ceil((Ny - 2 * my) / K_per ** (1 / 3))), 1)
        s_t = max(int(np.ceil((Nt - 2 * mt) / K_per ** (1 / 3))), 1)
        self.prep = dict(Cx=Cx, Cy=Cy, Ct=Ct, s=(s_x, s_y, s_t), su=su, sx=sx, sy=sy, Nx=Nx, Ny=Ny, dx=dx, dy=dy)
        b = np.concatenate([conv_nd(u / su, (Cx[0], Cy[0], Ct[1]), (s_x, s_y, s_t)).ravel() for u in U])
        self.info = dict(kx=float(np.mean([k[0] for k in ks])), ky=float(np.mean([k[1] for k in ks])),
                         kt=float(np.mean([k[2] for k in ks])), mx=mx, my=my, mt=mt, px=px, py=py, pt=pt, su=su,
                         sx=sx, sy=sy, st=st, s_x=s_x, s_y=s_y, s_t=s_t, rows=len(b))
        return b, su ** (-1) * st ** (-1)

    def symbols_grid(self):
        p = self.prep
        kx = 2 * np.pi * np.fft.fftfreq(p["Nx"], d=p["dx"])
        ky = 2 * np.pi * np.fft.fftfreq(p["Ny"], d=p["dy"])
        return np.meshgrid(kx, ky, indexing="ij")

    def columns(self, U, terms, multipliers=()):
        """G [K, len(terms)] (scaled) of local terms (j, p, q, -1) and multiplier terms (j, 0, 0, mi); multipliers are
        (symbol(KX, KY), order) pairs; returns G and the column scales (before the left-hand side's)."""
        p = self.prep
        KX, KY = self.symbols_grid()
        syms = [np.asarray(f(KX, KY)) * p["sx"] ** (-order) for f, order in multipliers]
        Gs = []
        for u in U:
            us = u / p["su"]
            cols = []
            for j, px_, qy, mi in terms:
                f = us ** j if j else np.ones_like(us)
                if mi >= 0:
                    f = np.real(np.fft.ifft2(syms[mi][:, :, None] * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))
                cols.append(conv_nd(f, (p["Cx"][px_], p["Cy"][qy], p["Ct"][0]), p["s"]).ravel())
            Gs.append(np.stack(cols, 1))
        M = np.array([p["su"] ** (-j) * (p["sx"] ** (-px_) * p["sy"] ** (-qy) if mi < 0
                                           else p["sx"] ** (-multipliers[mi][1])) for j, px_, qy, mi in terms])
        return np.concatenate(Gs), M

    def mstls(self, G, b, M, lambdas=mw.LAMBDAS, nterms=None):
        w = mw.WSINDy()
        w.info = {}
        W, lam, resid = w.mstls(G, b, M, lambdas, nterms)
        self.info.update(w.info)
        return W, lam, resid
