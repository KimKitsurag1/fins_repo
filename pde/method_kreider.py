"""Model-free symmetry discovery of Kreider, Harlim and Huang, "A model-free method for discovering symmetry in
differential equations" (arXiv 2511.09779), after the authors' MATLAB code (Fig_3c.m, Fig_3d.m).

Pipeline on samples (t, x, u) of one solution: K nearest neighbours, local SVD for the tangent frame of the 2D graph,
J GMLS steps with a local polynomial of degree ell (no constant term, the point itself left out), derivatives of the
next jet level from the refined tangent vectors, repeated in the prolonged space up to jet order p; normals of the
prolonged graph; rows S . pr^(p) v for a polynomial generator class; A = sum p'p; SVD of A.
Jet coordinates in the authors' order (t, x, u, u_t, u_x, u_tt, u_tx, u_xx, u_ttt, u_ttx, u_txx, u_xxx, ...); mixed
derivatives as theirs: u_{t^a x^(b+1)} = d_x u_{t^a x^b}, u_{t^(k+1)} = d_t u_{t^k}.
pooled_normals: codimension-q frames of a pooled point cloud of any dimension (same SVD + GMLS, tangent dimension
D - q).
"""

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

CHUNK = 4096
MEM_BYTES = 2.5e8
COMPONENTS = ("xi_t", "xi_x", "phi")


def multi_indices(p):
    """(a, b) of u_{t^a x^b} for orders 0..p: by order, then a descending."""
    return [(k - b, b) for k in range(p + 1) for b in range(k + 1)]


def jet_labels(p):
    """Names of the jet coordinates up to order p."""
    return ["t", "x"] + ["u" + ("_" + "t" * a + "x" * b if a + b else "") for a, b in multi_indices(p)]


def jet_dim(p):
    return 2 + (p + 1) * (p + 2) // 2


def compositions(deg, d):
    """Exponent tuples of length d summing to deg, lexicographically ascending."""
    if d == 1:
        return [(deg,)]
    return [(a,) + r for a in range(deg + 1) for r in compositions(deg - a, d - 1)]


def gmls_exponents(d, ell):
    """GMLS monomials s^alpha, 1 <= |alpha| <= ell; for d = 2 the authors' column order (s1^pp s2^(deg - pp))."""
    return [e for deg in range(1, ell + 1) for e in compositions(deg, d)]


def _monomial_plan(exps):
    """Per exponent: (index of the exponent less e_j, or -1 for the constant; j, its first nonzero coordinate)."""
    pos = {e: i for i, e in enumerate(exps)}
    plan = []
    for e in exps:
        j = next(i for i, a in enumerate(e) if a)
        plan.append((pos.get(tuple(a - (i == j) for i, a in enumerate(e)), -1), j))
    return plan


def _vandermonde(s, plan):
    """Monomial columns [B, n, m] of local coordinates s [B, n, d]."""
    V = np.empty(s.shape[:2] + (len(plan),))
    for i, (par, j) in enumerate(plan):
        V[..., i] = s[..., j] if par < 0 else V[..., par] * s[..., j]
    return V


def _lstsq(V, Y):
    """Least-squares solutions [B, m, r] of V [B, n, m] X = Y [B, n, r] (full column rank): Householder QR of the
    column-equilibrated V."""
    c = np.linalg.norm(V, axis=1)
    c = np.where(c > 0, c, 1.0)
    Q, R = np.linalg.qr(V / c[:, None, :])
    return np.linalg.solve(R, np.swapaxes(Q, 1, 2) @ Y) / c[:, :, None]


def _complement(T):
    """Orthonormal basis [B, D, D - d] of the orthogonal complement of span T [B, D, d] (MATLAB's null(T'))."""
    Vh = np.linalg.svd(np.swapaxes(T, 1, 2), full_matrices=True)[2]
    return np.swapaxes(Vh[:, T.shape[2]:, :], 1, 2)


def _frames(Ms, Mg, d, plan, lin, J):
    """Tangent [B, D, d] and normal [B, D, D - d] frames from neighbour differences Ms (SVD), Mg (GMLS) [B, K, D],
    the point itself first. GMLS step: local coordinates s = M T, heights h = M S, least-squares polynomial h(s) over
    the neighbours but the point itself, T_j <- T_j + sum_k (dh_k / ds_j)(0) S_k, each T_j normalised (not
    re-orthogonalised), S <- null(T')."""
    T = np.linalg.svd(np.swapaxes(Ms, 1, 2), full_matrices=False)[0][:, :, :d]
    S = _complement(T)
    for _ in range(J):
        P = Mg[:, 1:, :] @ np.concatenate([T, S], 2)
        beta = _lstsq(_vandermonde(P[..., :d], plan), P[..., d:])
        T = T + S @ np.swapaxes(beta[:, lin, :], 1, 2)
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        S = _complement(T)
    return T, S


def local_frames(Y, d, K, ell, J, k_gmls=None, chunk=CHUNK, idx=None):
    """Tangent [n, D, d] and normal [n, D, D - d] frames of a d-dimensional point cloud Y [n, D]: SVD of the
    differences to the K nearest neighbours (the point itself included), then J GMLS steps of degree ell over the
    k_gmls (default K) nearest neighbours. idx [n, >= max(K, k_gmls)]: precomputed neighbour lists (the point
    itself first) in place of the search in Y."""
    Y = np.asarray(Y, dtype=float)
    n, D = Y.shape
    kg = K if k_gmls is None else k_gmls
    exps = gmls_exponents(d, ell)
    if J and kg - 1 < len(exps):
        raise ValueError(f"GMLS needs k_gmls - 1 >= {len(exps)} neighbours for d = {d}, ell = {ell}")
    plan = _monomial_plan(exps)
    lin = [exps.index(tuple(int(i == j) for i in range(d))) for j in range(d)]
    kq = max(K, kg) if J else K
    idx = cKDTree(Y).query(Y, k=kq, workers=-1)[1] if idx is None else np.asarray(idx)[:, :kq]
    B = int(max(1, min(chunk, MEM_BYTES // (8 * kq * (3 * D + 2 * len(exps))))))
    T, S = np.empty((n, D, d)), np.empty((n, D, D - d))
    for a in range(0, n, B):
        sl = slice(a, min(n, a + B))
        M = Y[idx[sl]] - Y[sl, None, :]
        T[sl], S[sl] = _frames(M[:, :K], M[:, :kg], d, plan, lin, J)
    return T, S


def pooled_normals(Y, q, K, ell, J, k_gmls=None, chunk=CHUNK):
    """Codimension-q frames of a pooled point cloud Y [n, D] (e.g. the equation surface u_t = F(...) in jet space,
    sampled by many solutions): tangent [n, D, D - q] and normal [n, D, q] frames by the local SVD and J GMLS steps
    of degree ell in the D - q intrinsic coordinates."""
    Y = np.asarray(Y, dtype=float)
    return local_frames(Y, Y.shape[1] - q, K, ell, J, k_gmls, chunk)


def next_jet_level(Z, T, k):
    """Order-(k + 1) columns from the refined tangents T [n, D, 2] of the order-k graph Z [n, D]: each order-k
    coordinate c gives [d_t c, d_x c] = A^-1 [T1(c), T2(c)], A = [T1(t) T1(x); T2(t) T2(x)]; returned
    [d_t u_{t^k}, d_x u_{t^k}, d_x u_{t^(k-1) x}, ..., d_x u_{x^k}]."""
    D = Z.shape[1]
    G = np.linalg.solve(np.swapaxes(T[:, :2, :], 1, 2), np.swapaxes(T[:, D - k - 1:, :], 1, 2))
    return np.concatenate([G[:, 0, :1], G[:, 1, :]], 1)


def jet_frames(t, x, u, p, K, ell, J, chunk=CHUNK):
    """The authors' pipeline on samples of one solution: frames of the graph in (t, x, u), derivatives of the next
    level, repeated up to order p. Returns the estimated jets Z [n, jet_dim(p)] and the tangent [n, D, 2] and normal
    [n, D, D - 2] frames of the order-p graph."""
    Z = np.stack([np.ravel(t), np.ravel(x), np.ravel(u)], 1).astype(float)
    for k in range(p + 1):
        T, S = local_frames(Z, 2, K, ell, J, chunk=chunk)
        if k < p:
            Z = np.concatenate([Z, next_jet_level(Z, T, k)], 1)
    return Z, T, S


class PolynomialGenerators:
    """Generators v = xi^t d_t + xi^x d_x + phi d_u with components in span{tau^a chi^b ups^c : a + b + c <= degree},
    tau = (t - t0) / T, chi = (x - x0) / Lx, ups = (u - u0) / s (shift = (t0, x0, u0), scale = (T, Lx, s)).
    Coefficient k = component * n_mono + monomial; components (xi^t, xi^x, phi); monomials by degree, then
    lexicographic in (t, x, u) descending (degree 1: 1, t, x, u, the authors' order). pr^(p) v by
    phi^J = D_J(phi - xi^t u_t - xi^x u_x) + xi^t u_{J,t} + xi^x u_{J,x}, computed with sympy once, lambdified."""

    def __init__(self, p, degree=1, shift=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        self.p, self.degree = p, degree
        self.shift, self.scale = tuple(float(v) for v in shift), tuple(float(v) for v in scale)
        self.monomials = [e for deg in range(degree + 1) for e in sorted(compositions(deg, 3), reverse=True)]
        self.n_mono, self.dim = len(self.monomials), jet_dim(p)
        self.n_basis = 3 * self.n_mono
        self.labels = [f"{c}[{self._mono_name(e)}]" for c in COMPONENTS for e in self.monomials]
        t, x = sp.symbols("t x")
        U = {J: sp.Symbol("u" + ("_" + "t" * J[0] + "x" * J[1] if sum(J) else "")) for J in multi_indices(p + 1)}
        jets = multi_indices(p)
        tau, chi, ups = self._normalised(t, x, U[(0, 0)])

        def Dt(f):
            return sp.expand(sp.diff(f, t) + sum(sp.diff(f, U[J]) * U[(J[0] + 1, J[1])] for J in jets))

        def Dx(f):
            return sp.expand(sp.diff(f, x) + sum(sp.diff(f, U[J]) * U[(J[0], J[1] + 1)] for J in jets))

        self.exprs = []
        for comp in range(3):
            for e in self.monomials:
                m = sp.expand(tau ** e[0] * chi ** e[1] * ups ** e[2])
                xt, xx, ph = [m if c == comp else sp.Integer(0) for c in range(3)]
                DQ = {(0, 0): sp.expand(ph - xt * U[(1, 0)] - xx * U[(0, 1)])}
                for a, b in jets[1:]:
                    DQ[(a, b)] = Dt(DQ[(a - 1, b)]) if a else Dx(DQ[(a, b - 1)])
                col = [xt, xx, ph] + [sp.expand(DQ[J] + xt * U[(J[0] + 1, J[1])] + xx * U[(J[0], J[1] + 1)])
                                      for J in jets[1:]]
                self.exprs.append(col)
        syms = [t, x] + [U[J] for J in jets]
        bad = set().union(*(ex.free_symbols for col in self.exprs for ex in col)) - set(syms)
        if bad:
            raise RuntimeError(f"prolongation depends on {sorted(map(str, bad))}")
        entries = [(i, k, ex) for k, col in enumerate(self.exprs) for i, ex in enumerate(col) if ex != 0]
        self._entries = [(i, k) for i, k, _ in entries]
        self._f = sp.lambdify(syms, [ex for _, _, ex in entries], modules="numpy", cse=True)

    @staticmethod
    def _mono_name(e):
        f = [v + (f"^{a}" if a > 1 else "") for v, a in zip("txu", e) if a]
        return "*".join(f) if f else "1"

    def _normalised(self, t, x, u):
        (t0, x0, u0), (T_, L_, s_) = self.shift, self.scale
        return (t - t0) / T_, (x - x0) / L_, (u - u0) / s_

    def prolongation(self, Z):
        """pr^(p) of every basis generator at the jet points Z [n, D]: L [n, D, n_basis]."""
        Z = np.asarray(Z, dtype=float)
        if Z.shape[1] != self.dim:
            raise ValueError(f"jet points have {Z.shape[1]} coordinates, order {self.p} needs {self.dim}")
        L = np.zeros((len(Z), self.dim, self.n_basis))
        for (i, k), val in zip(self._entries, self._f(*Z.T)):
            L[:, i, k] = val
        return L

    def rows(self, Z, S):
        """Rows S_j . pr v [n, q, n_basis] for normals S [n, D, q] at the jet points Z [n, D]."""
        return np.swapaxes(S, 1, 2) @ self.prolongation(Z)

    def gram(self, Z, S, chunk=CHUNK):
        """A = sum over points and normals of p'p [n_basis, n_basis]."""
        A = np.zeros((self.n_basis, self.n_basis))
        for a in range(0, len(Z), chunk):
            P = self.rows(Z[a:a + chunk], S[a:a + chunk]).reshape(-1, self.n_basis)
            A += P.T @ P
        return A

    def coefficients(self, xi_t, xi_x, phi):
        """Coefficient vector of the generator whose components are polynomials (sympy expressions or strings) in the
        raw t, x, u."""
        t, x, u = sp.symbols("t x u")
        tau, chi, ups = sp.symbols("tau chi ups")
        (t0, x0, u0), (T_, L_, s_) = self.shift, self.scale
        sub = {t: T_ * tau + t0, x: L_ * chi + x0, u: s_ * ups + u0}
        out = np.zeros(self.n_basis)
        for comp, f in enumerate((xi_t, xi_x, phi)):
            g = sp.Poly(sp.expand(sp.sympify(f).subs(sub, simultaneous=True)), tau, chi, ups)
            for e, cf in g.terms():
                if cf == 0:
                    continue
                if e not in self.monomials:
                    raise ValueError(f"monomial {e} of component {COMPONENTS[comp]} exceeds degree {self.degree}")
                out[comp * self.n_mono + self.monomials.index(e)] = float(cf)
        return out

    def from_authors_affine(self, c12):
        """Coefficients in this class of the authors' 12-vector [xi^t (1, t, x, u), xi^x (1, t, x, u),
        phi (1, t, x, u)]."""
        t, x, u = sp.symbols("t x u")
        c = [sp.Float(v) for v in np.asarray(c12, dtype=float)]
        return self.coefficients(*[c[4 * j] + c[4 * j + 1] * t + c[4 * j + 2] * x + c[4 * j + 3] * u
                                   for j in range(3)])


def null_space_svd(A):
    """Singular values (descending) and right singular vectors (columns) of A, as [~, sv, V] = svd(A, 'econ')."""
    _, s, Vh = np.linalg.svd(A)
    return s, Vh.T


def _orth(A):
    A = np.asarray(A, dtype=float).reshape(len(A), -1)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    return U[:, s > tol]


def sin_subspace(A, B):
    """sin of the largest principal angle between the column spans of A and B (MATLAB's sin(subspace(A, B)))."""
    A, B = _orth(A), _orth(B)
    if A.shape[1] < B.shape[1]:
        A, B = B, A
    return float(min(1.0, np.linalg.norm(B - A @ (A.T @ B), 2)))


def discover(t, x, u, gens, K, ell, J, chunk=CHUNK):
    """Full pipeline at the generator class' jet order: estimated jets Z, normals S, A, singular values sv
    (descending) and right singular vectors V (columns; V[:, -1] the smallest)."""
    Z, _, S = jet_frames(t, x, u, gens.p, K, ell, J, chunk)
    A = gens.gram(Z, S, chunk)
    sv, V = null_space_svd(A)
    return dict(Z=Z, S=S, A=A, sv=sv, V=V)
