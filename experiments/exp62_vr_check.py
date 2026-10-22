"""V_r = ((e^(rt) - 1) / r) d_t - e^(rt) psi d_psi for F = Lap psi_t + J(psi, Lap psi) + r Lap psi: F'[Q] = -2 e^(rt) F - tau F_t."""
import sympy as sp

t, x, y, r, e = sp.symbols("t x y r epsilon", real=True)
psi = sp.Function("psi")(t, x, y)
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
jac = lambda a, b: sp.diff(a, x) * sp.diff(b, y) - sp.diff(a, y) * sp.diff(b, x)
F = lambda p: lap(sp.diff(p, t)) + jac(p, lap(p)) + r * lap(p)
tau = (sp.exp(r * t) - 1) / r
Q = -sp.exp(r * t) * psi - tau * sp.diff(psi, t)
lin = sp.diff(F(psi + e * Q), e).subs(e, 0)
print(sp.simplify(sp.expand(lin - (-2 * sp.exp(r * t) * F(psi) - tau * sp.diff(F(psi), t)))))
