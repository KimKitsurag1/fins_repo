"""Symbolic determining equations for u_t + a u u_x = b u_xx.

The symmetry condition D_t Q = L_u[Q] on solutions, with Q = phi - xi u_x
- eta u_t and L_u[Q] = -a (Q u_x + u D_x Q) + b D_x^2 Q, is LINEAR in the
generator. With xi, eta, phi expanded on a monomial basis in (x, t, u) it
becomes M c = 0, M computed exactly by sympy: no derivative of any data
field is taken. ker M is the symmetry algebra inside the basis; ||M c||
measures how far a given generator is from being a symmetry.
"""

import numpy as np
import sympy as sp

MONO = ("1", "x", "t", "u", "xt", "x2", "t2", "tu", "xu")


def _monomial_exprs(x, t, u):
    return [sp.Integer(1), x, t, u, x * t, x ** 2, t ** 2, t * u, x * u]


def determining_matrix(a, b):
    """M (rows: independent monomials of the residual) acting on
    c = [c_xi (9), c_eta (9), c_phi (9)]."""
    x, t, u = sp.symbols("x t u")
    ux, uxx, uxxx, uxxxx = sp.symbols("u_x u_xx u_xxx u_xxxx")
    cs = sp.symbols("c0:27")
    mono = _monomial_exprs(x, t, u)
    xi = sum(cs[i] * mono[i] for i in range(9))
    eta = sum(cs[9 + i] * mono[i] for i in range(9))
    phi = sum(cs[18 + i] * mono[i] for i in range(9))
    ut = -a * u * ux + b * uxx
    uxt = -a * (ux ** 2 + u * uxx) + b * uxxx
    uxxt = -a * (3 * ux * uxx + u * uxxx) + b * uxxxx
    jet = (u, ux, uxx, uxxx)
    nxt_x = (ux, uxx, uxxx, uxxxx)
    nxt_t = (ut, uxt, uxxt, None)

    def Dx(f):
        return sp.diff(f, x) + sum(n * sp.diff(f, j) for j, n in zip(jet, nxt_x))

    def Dt(f):
        return sp.diff(f, t) + sum(n * sp.diff(f, j)
                                   for j, n in zip(jet, nxt_t) if n is not None)

    Q = phi - xi * ux - eta * ut
    DxQ = Dx(Q)
    res = sp.expand(Dt(Q) - (-a * (Q * ux + u * DxQ) + b * Dx(DxQ)))
    poly = sp.Poly(res, x, t, u, ux, uxx, uxxx, uxxxx)
    rows = [[float(sp.expand(coef).coeff(c)) for c in cs] for coef in poly.coeffs()]
    return np.array(rows)


def gt_coefficients(a):
    """The corrected Burgers algebra in the monomial basis (xi | eta | phi)."""
    def vec(xi={}, eta={}, phi={}):
        v = np.zeros(27)
        for blk, d in ((0, xi), (9, eta), (18, phi)):
            for m, c in d.items():
                v[blk + MONO.index(m)] = c
        return v
    return np.stack([vec(xi={"1": 1}), vec(eta={"1": 1}),
                     vec(xi={"t": a}, phi={"1": 1}),
                     vec(xi={"x": 1}, eta={"t": 2}, phi={"u": -1}),
                     vec(xi={"xt": 1}, eta={"t2": 1}, phi={"x": 1 / a, "tu": -1})]).T
