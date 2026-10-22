"""Experiment 17 — symbolic ground truth: determining equations and closure.

Two things the numerical audit cannot settle by itself:

A. Symbolic verification of every tabulated generator against the
   linearised equation, on the jet space with time derivatives eliminated
   by the PDE. This turns exp13's numerical finding about the Burgers
   projective generator into an exact statement, and checks the remaining
   generators of all three equations.

B. Lie-bracket closure of the ESSENTIAL (torus) subalgebras. Claiming
   "the essential rank is 3" requires those three fields to close under
   the bracket; otherwise they span a subspace, not a subalgebra. The
   structure constants also give the ACE column its exact target.

Method: jet-space coordinates (x, t, U0, U1, ...) with U_k = d^k u / dx^k,
total derivatives D_x and D_t defined so that u_t is replaced by the
evolution expression, and brackets computed in the (x, t, u) point-symmetry
sense.

Run:  python exp17_symbolic_gt.py
"""

import sympy as sp

NORD = 7                                   # jet truncation: U0 .. U6
x, t, nu, al = sp.symbols("x t nu alpha", real=True, positive=True)
U = sp.symbols(f"U0:{NORD}", real=True)
u = U[0]

EVOL = {                                   # u_t as a jet expression
    "heat": al * U[2],
    "burgers": nu * U[2] - U[0] * U[1],
    "kdv": -6 * U[0] * U[1] - U[3],
}


def Dx(f):
    return sp.diff(f, x) + sum(U[k + 1] * sp.diff(f, U[k])
                               for k in range(NORD - 1))


def Dt(f, pde):
    F = EVOL[pde]
    out = sp.diff(f, t)
    Fk = F
    for k in range(NORD - 1):
        out += Fk * sp.diff(f, U[k])
        Fk = Dx(Fk)
    return out


def linearised(Q, pde):
    """M[Q] = d/deps N[u + eps Q] for each equation's N."""
    if pde == "heat":
        return Dt(Q, pde) - al * Dx(Dx(Q))
    if pde == "burgers":
        return Dt(Q, pde) + Q * U[1] + U[0] * Dx(Q) - nu * Dx(Dx(Q))
    if pde == "kdv":
        return Dt(Q, pde) + 6 * (Q * U[1] + U[0] * Dx(Q)) + Dx(Dx(Dx(Q)))
    raise ValueError(pde)


# (name, xi, eta, phi) in the (x, t, u) point-symmetry convention
GENS = {
    "heat": [
        ("d_x", 1, 0, 0),
        ("d_t", 0, 1, 0),
        ("u d_u", 0, 0, u),
        ("Galilean", 2 * al * t, 0, -x * u),
        ("scaling", x, 2 * t, 0),
        ("projective", 4 * al * t * x, 4 * al * t ** 2,
         -(x ** 2 + 2 * al * t) * u),
    ],
    "burgers": [
        ("d_x", 1, 0, 0),
        ("d_t", 0, 1, 0),
        ("Galilean", t, 0, 1),
        ("scaling", x, 2 * t, -u),
        ("projective PAPER", x * t, t ** 2, -(x + t * u)),
        ("projective CORRECTED", x * t, t ** 2, x - t * u),
    ],
    "kdv": [
        ("d_x", 1, 0, 0),
        ("d_t", 0, 1, 0),
        ("Galilean", t, 0, sp.Rational(1, 6)),
        ("scaling", x, 3 * t, -2 * u),
    ],
}

ESSENTIAL = {"heat": ["d_x", "d_t", "u d_u"],
             "burgers": ["d_x", "d_t", "Galilean"],
             "kdv": ["d_x", "d_t", "Galilean"]}


def characteristic(xi, eta, phi, pde):
    return sp.expand(phi - xi * U[1] - eta * EVOL[pde])


def bracket(a, b):
    """[a, b] for point vector fields with coefficients in (x, t, u)."""
    def act(v, f):
        return (v[0] * sp.diff(f, x) + v[1] * sp.diff(f, t)
                + v[2] * sp.diff(f, u))
    return tuple(sp.simplify(act(a, b[i]) - act(b, a[i])) for i in range(3))


def in_span(vec, basis):
    """Express vec as a CONSTANT-coefficient combination of basis, or None."""
    cs = sp.symbols(f"c0:{len(basis)}")
    eqs = []
    for i in range(3):
        expr = sp.expand(vec[i] - sum(c * b[i] for c, b in zip(cs, basis)))
        poly = sp.Poly(expr, x, t, u)
        eqs += list(poly.coeffs()) if poly.total_degree() >= 0 else [expr]
    sol = sp.solve(eqs, cs, dict=True)
    if not sol:
        return None
    s = sol[0]
    return [sp.simplify(s.get(c, c)) for c in cs]


def main():
    print("=" * 78)
    print("EXP 17A - determining equation M[Q] = 0 on the jet space")
    print("=" * 78)
    for pde, gens in GENS.items():
        print(f"\n  {pde}:")
        for name, xi, eta, phi in gens:
            Q = characteristic(xi, eta, phi, pde)
            M = sp.simplify(sp.expand(linearised(Q, pde)))
            ok = "SYMMETRY" if M == 0 else "NOT a symmetry"
            print(f"    {name:<22} {ok}")
            if M != 0:
                print(f"        defect M[Q] = {sp.factor(M)}")

    print("\n" + "=" * 78)
    print("EXP 17B - bracket closure of the essential (torus) subalgebras")
    print("=" * 78)
    for pde in GENS:
        names = ESSENTIAL[pde]
        basis = [(sp.sympify(g[1]), sp.sympify(g[2]), sp.sympify(g[3]))
                 for g in GENS[pde] if g[0] in names]
        print(f"\n  {pde}: essential basis {names}")
        closed, lines = True, []
        for i in range(3):
            for j in range(i + 1, 3):
                br = bracket(basis[i], basis[j])
                if all(c == 0 for c in br):
                    lines.append(f"    [{names[i]}, {names[j]}] = 0")
                    continue
                co = in_span(br, basis)
                if co is None:
                    closed = False
                    lines.append(f"    [{names[i]}, {names[j]}] = {br} "
                                 f"NOT IN SPAN")
                else:
                    terms = " + ".join(f"{c}*{n}" for c, n in
                                       zip(co, names) if c != 0)
                    lines.append(f"    [{names[i]}, {names[j]}] = {terms}")
        print("\n".join(lines))
        print(f"    closed subalgebra: {closed}   "
              f"(exact ACE target = 0)")


if __name__ == "__main__":
    main()
