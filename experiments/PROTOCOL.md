# FINS evaluation protocol, frozen v1

Fixed before the final tables. Every choice below was settled by an earlier experiment
(README_EXPERIMENTS.md); none is tuned on the final runs. Seeds s = 42, 43, 44.

## 1. Data and operators

| equation | data | operator | training data |
|---|---|---|---|
| Burgers | benchmark initial data, solver saves aligned to the internal grid (`burgers_coarse`), read on four-period line windows | CNO1d (channel multiplier 16, 2 residual blocks, 106k parameters), same-size, unroll 4, train margin 6, 400 epochs | 1024 trajectories, seed s |
| heat | benchmark initial data low-passed (`--smooth`), evolved exactly, one domain length L = 1 (`--only-L 1`), four-period windows | as Burgers | 1024 smooth trajectories at L = 1, seed s |
| KdV | periodised two-soliton states (`--ic periodic`), T = 0.05, aligned three-point stencil for u_t | torus FNO stepper, 64 modes, no positional embedding, 400 epochs | 1024 trajectories, half periodised solitons on constant backgrounds, half broadband humps (`--train-ic mixed`), seed s |

Evaluation trajectories: 10, data seed 1000 + s, never seen in training. No reliability filter:
every method is scored on the same ten trajectories; per-trajectory one-step errors are printed.

## 2. Band check (equation-free, before discovery)

Energy share above k = Nx / 4 of u and of u_t on the evaluation data, printed by exp39 / exp40.
Flag when the u_t share exceeds 1e-2: the criterion would probe modes the operator has barely seen.

| data | u | u_t | flag |
|---|---|---|---|
| Burgers aligned | 2.4e-5 | 4.9e-3 | no |
| heat benchmark | 2.2e-4 | 0.69 | yes |
| heat smooth | 1.6e-32 | 6.6e-28 | no |
| KdV broadband benchmark | 1.0e-5 | 0.40 | yes |
| KdV solitons, plain profiles | 9.1e-9 | 3.3e-2 | yes |
| KdV solitons, periodised | 1.1e-19 | 8.8e-17 | no |

Flagged sets are exactly those where the time translation failed or was weakest through a
learned operator. A flag is reported, not silently acted on.

## 3. Criterion

Polynomial class of degree d in (x, t, u), d fixed per equation: Burgers 2, KdV 2, heat 3 (the
degree rule of exp33 recovered Burgers and KdV but not heat, so d is a stated choice). K = 16,
starts every 16, mu_rel = 1. Windows: x measured from the window centre, scored region the central
W - 12 K points. Torus (KdV): seam band 0.2. Printed per run: band check, relative change of the
characteristics over the horizon, rollout errors at 1 and K steps, edge leak, eigenvalues.

## 4. Readout

k = dimension of the algebra in the class (Burgers 5, KdV 4, heat 10). Blind readout by varimax at
m = k, marked "m from the ground-truth dimension": no rule recovers the rank through a learned
operator (Burgers: S2 near-symmetry; heat: no gap). The rank from the largest eigenvalue gap is
reported beside it.

## 5. Metrics (exp42; ground truth used only here)

All methods: generators as (xi, eta, phi) fields on the ten evaluation trajectories, torus band
[0.2, 0.8] L, times 8 .. Nt - 8, components scaled xi / L, eta / T, phi / sigma_u.
- missing dimensions = k_B - ||Q_A' Q_B||_F^2 (ground-truth directions not captured) and spurious
  dimensions = rank_A - ||Q_A' Q_B||_F^2 (discovered directions outside the ground truth); both
  basis-free, no ad hoc rank penalty. Primary.
- novel containment of each ground-truth generator: the part of it orthogonal to the other
  ground-truth generators, its share inside the discovered span; recovered = novel containment
  >= 0.9. Needed because on a band of x the fields 1, x, x^2, x^3 (and, at alpha = 0.01, Galilean
  vs x u d_u, projective vs x^2 u d_u) are nearly dependent: plain containment credited LIG's four
  generators with eight of heat's ten.
- Grassmann angle part and rank penalty (LieNLSD's metric, the earlier campaign), Hungarian cosine:
  secondary, for comparability.
FINS criterion columns: all members below lambda_{k+1}, worst margin, lambda_{k+1} / lambda_k.
Two ground-truth conventions: the finite algebra (Burgers 5, KdV 4, heat 6) and the full algebra in
the class (heat 10: the finite six and the heat polynomials of degree <= 3; under the finite
convention FINS's superpositions count as spurious).
Reference rows: the exact operator on the same trajectories (heat kernel on windows, spectral
solver on the torus); Burgers has no exact window operator.

## 6. Baselines (exp41)

LieNLSD, LieGAN, LaLiGAN, Augerino, LIG (repository implementations, pde/method_*.py), each trained
on the same training trajectories of seed s (cfg.n_sym = k), fields evaluated on the ten evaluation
trajectories. Equation-aware methods, marked in the tables: LIG and LieNLSD build their objective
from the PDE; they are references, not equation-free competitors. Corrections, all toward the
baselines' benefit and each isolated behind an option so the published configuration still runs:
- LieNLSD (`run_lienlsd_general`): library of all monomials of degree <= d (degree 2 missed heat's
  projective x^2 u and x^3); the determining equation of the equation the data satisfy (Burgers
  u_t + 2 pi u u_x = 4 pi^2 nu u_xx, not the textbook form); components returned in the ground
  truth's (x, t, u) order -- `run_lienlsd` returns the t-component as xi; column-scaled SVD.
- LIG (`cfg.lig_eq`): advection and dispersion coefficients carried into LIG's normalised
  variables (the published residual used u_t + u u_x - nu u_xx and u_t + 6 u u_x + u_xxx in
  normalised x, t, u, correct only for heat).
LIG needs ~6 GB at batch 1 on the 128 x 128 KdV grid without torchdiffeq; install torchdiffeq
(adjoint) and use --batch-size 1 for KdV. exp41 saves after every method.
Not implemented, optional: AtlasD (reports heat), SDBA; own-class nulls (table1.py --span).

## 7. Files (FINS_RESULTS_DIR)

fins_{pde}_s{s}.npz (exp40 / exp39 --save), exact_{pde}_s{s}.npz (heat, KdV), baselines_{pde}_s{s}.npz
(exp41); exp42 --dir reads them and prints the tables.

## 8. Post-run notes (v1 unchanged; found in the tables, README part 3)

- Augerino returns its fixed dictionary (six affine fields and u d_u) scaled by learned widths; the
  span metrics are width-blind, so its row scores the dictionary. exp42 adds a post hoc row with the
  generators whose width grew past 0.05 (final widths are either < 0.025 or >= 0.15).
- method_lig caps n_sym at 4 (Ko et al.'s PDE setting), so LIG ran with 4 generators on Burgers
  and heat, not k as section 6 says. exp42 prints the floor k_B - rank.
- Hungarian cosine is basis-dependent (exact operator: 0.57-0.83); novel containment under-credits
  a generator found alone. Missing and spurious dimensions remain primary.
- Chance levels (README part 4): an untrained smooth network already spans much of the low-order
  field space, so missing dimensions are read against nulls: each method untrained (exp41
  --untrained, files untrained_{pde}_s{s}.npz) and random subspaces of FINS's class (exp42,
  uniform and Gaussian coefficients). LIG_k: LIG with n_sym = k (Burgers).

## 9. Known limits, stated in the paper

Burgers' margin is bounded by the inviscid near-symmetry S2 = t d_t - u d_u (loss grows as nu^2).
Heat: the necessary condition holds, full discovery through a learned operator does not (its
characteristics move 13 percent over the horizon; its nearest non-symmetries lie at 1e-5 .. 1e-3
through the exact operator; the operator's floor lifts the whole spectrum into that band).
Blind readout of d_t is never clean through a learned operator when the bottom losses lie within
one or two decades of each other.

## 10. Amendment (04.10): baseline provenance, LieNLSD as published

pde/method_*.py are our reimplementations of the baselines, not the authors' code. LieNLSD's official code fits an
MLP to u_t on finite-difference jets and uses its Jacobian; our wrapper substituted the equation, so the v1 LieNLSD
row is an oracle variant (exact right-hand side). The bugs listed in section 6 were bugs of our wrappers. Added rows
(README, entry of 04.10, pre-registered): LieNLSD_F (as published, with the protocol's library and readout),
LieNLSD_RHS (same pipeline, exact right-hand side), F at initialisation as its own-class null; readouts m = k and
own count.
