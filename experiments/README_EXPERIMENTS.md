# FINS post-review experiment suite

Seven experiment scripts addressing the three meta-review concerns of the
Discovery Science 2026 rejection (Proposition 1 insufficiently proven;
ground-truth algebras possibly inconsistent with the periodic experimental
setup; main result overstated), plus a verified patch for a bias in the
repository's sanity check. Scripts live in `experiments/` inside the FINS
repository and import from `../pde` unchanged; the only repository
modification proposed is the patch file.

Environment of these runs: CPU-only, torch 2.13, numpy/scipy; `neuraloperator`
absent, so `make_fno` falls back to `SimpleFNO2d` (the repository's own
guard). Experiments 1-4 and 8 are oracle-level (no surrogate) and the
numbers below are final. Experiments 5-7 were run in `--smoke` mode and 9
with a small CPU surrogate, to validate mechanics; paper-scale numbers
require the full config on GPU.

## Files

```
experiments/
  exp_common.py                  shared: oracle solvers (exact heat, dealiased
                                 ETDRK4 with save grid snapped to the output
                                 times), batched FD-JVP, per-sample GT
                                 characteristics, repo-form Rayleigh residual,
                                 IC families mirroring data_generation.py,
                                 per-sample fixed sanity check (torch)
  exp1_oracle_torus_audit.py     ground-truth audit on the torus + wide
                                 controls (heat / Burgers / KdV)
  exp2_null_model.py             null-model floor via pde/ground_truth.Metrics
  exp3_domain_geometry.py        residual vs L/w curve, heat, exact operator
  exp4_symmetry_breaking.py      controlled breaking, detection curve, exact
                                 linear split-step operator
  exp5_delta_sweep.py            surrogate-quality sweep + fixed-vs-repo
                                 sanity comparison (torch)
  exp6_trivial_subalgebra.py     constrained deflation phi = a(x,t)u + b(x,t)
                                 with penalty on the b-share; per-slot
                                 checkpointing and resume (torch)
  exp6_analyze.py                scores the saved exp6 slots against the
                                 ground truth (per-slot cos, energy split)
  exp15_rank_statistic.py        screens replacement rank statistics; the
                                 --oracle mode runs deflation against the
                                 exact operator (delta_J = 0) to separate
                                 statistic, surrogate and generator faults
  exp16_generator_acceptance.py  decomposes the slot residual floor into
                                 expressivity (eps_fit), amplification
                                 (R/eps^2) and optimisation (R_scratch)
  exp17_symbolic_gt.py           sympy: determining equation for every
                                 tabulated generator + bracket closure of
                                 the essential subalgebras
  exp18_windowed_rank.py         detectable rank per protocol: global vs
                                 seam-windowed, all three equations
  exp19_windowed_discovery.py    the discovery loop under the window, with
                                 shift averaging and a raw-generator anchor
  exp19_analyze.py               subspace view of exp19 output: Gram,
                                 effective dimension, GT containment
  exp20_null_floors.py           band-limited and shuffled-GT null floors
                                 for all three equations, both spaces
  exp21_olver_protocol.py        Olver residual, exact operator, single
                                 scale: spectral vs seam-aware local FD;
                                 also lints LieSym's coded ground truth
  exp22_single_scale_discoverability.py
                                 linear-ansatz null space: what is exactly
                                 discoverable at single scale per protocol
  exp23_single_scale_discovery.py
                                 neural discovery loop at single scale;
                                 coords x seam 2x2 with subspace analysis
  exp23_stats.py                 aggregates discovery npz into paired
                                 within-seed contrasts with bootstrap CI
  exp24_surrogate_protocol.py    the same protocol comparison through a
                                 TRAINED surrogate, incl. the multi-scale
                                 cell; GPU
  exp25_resolution_floor.py      data floor vs Nx for Burgers and KdV:
                                 which GT generators the data admits
  exp26_joint_training.py        controlled test: joint surrogate +
                                 equivariance training manufactures the
                                 injected symmetry at no L2 cost
  exp27_tangent_arm.py           one arm of the tangent-training test;
                                 negative result at toy scale
  exp28_timestepper.py           one-shot vs time-stepping surrogate:
                                 5-7x better fit (its Galilean column is
                                 withdrawn: wrong GT for repo-solver data)
  exp29_verify_generators.py     FINS generators: symbolic fit, ANALYTIC
                                 determining-equation split (primary),
                                 local and flow residuals (secondary)
  exp30_loss_spectrum.py         spectrum of the increment loss on the
                                 polynomial class: what it can identify
                                 through a given operator, no training
  pde/cno1d.py                   the authors' CNO1d (MIT), batch norm off
  exp43_transport_error.py       Proposition 2 measured: transport error of
                                 the learned operator against the exact one
                                 on the protocol data (heat, KdV)
  exp42_tables.py                protocol tables: missing / spurious
                                 dimensions, novel containment, class nulls,
                                 untrained rows, Augerino widths
  exp41_baselines.py             every baseline on the protocol data
                                 (--untrained: zero epochs; LIG_k)
  exp40_window.py                discovery through a boundary-free window
                                 operator on line windows (pde/window_stepper.py)
  exp45_neural_2d.py             neural symmetry generators of a 2D PDE
                                 (u_t = div(u grad u)): point-field MLPs
                                 trained by the Ky Fan loss of the transport
                                 pencil, exact discrete operator or a learned
                                 FNO2d (--op fno); polynomial class and nulls
  figures_neural2d.py            figure of exp45 (training curves, spectrum
                                 inside the learned span)
  exp45_span_analysis.py         a trained exp45 span: its directions against
                                 the algebra and phi(x, y) quadratics; its
                                 spectrum against the evaluation horizon
  exp45_fno_errors.py            rollout errors of saved exp45 FNO2d steppers
                                 on several data sets, per trajectory
  exp45_criteria.py              criteria beyond the eigenvalue (2D): operator
                                 sensitivity, affine class aff(4), mu
                                 sensitivity, closure-gated rank
  exp44_closure_rank.py          closure-gated rank of saved 1D FINS runs
  exp45_flows.py                 finite flows exp(s v) of exp45 generators
                                 (algebra, polynomial eigenvectors, near-
                                 symmetries with tau removed, random, trained
                                 network directions) through the exact
                                 operator: defect of the finite transformation
  figures_flows.py               figure of the finite flows (one seed)
  exp45_tau_class.py             the tau = tau(t) class (V5) as sub-pencils
  exp46_jet_symmetries.py        generalised symmetries in a jet-polynomial
                                 class (KdV, porous medium, heat)
  exp46_jet_flows.py             finite flows of jet characteristics
                                 (adaptive Lawson RK4; KdV in |k| < 32)
  exp47_lienlsd_2d.py            LieNLSD (fitted F, exact RHS; central or
                                 spectral jets) on the 2D equation of exp45,
                                 scored on exp45's probe
  ../pde/method_lienlsd.py       run_lienlsd_published: LieNLSD as published
                                 (exp41 methods LieNLSD_F, LieNLSD_RHS, *_spec;
                                 suffix @eval: system on the evaluation data)
  corrupt.py                     noise and temporal thinning of trajectories
                                 (same draws for every method; exp39/40/41
                                 --noise --thin)
  exp48_noise_thinning.py        noise and thinning, perfect models: FINS
                                 through the exact operators against LieNLSD
                                 (exact RHS, fitted F); runs, claims, summary
  exp39_localised.py             localised vs broadband data, same equation;
                                 exact operator or a torus stepper trained on
                                 localised trajectories (--op stepper)
                                 and grid, exact operators
  exp38_cone.py                  influence cone of an operator from the seam
  exp37_dirichlet_heat.py        non-periodic data: heat between Dirichlet
                                 walls, interior scoring
  exp36_horizon.py               horizon growth of each direction's loss
  exp35_periodic_superposition.py  heat's periodic superpositions in a
                                 Fourier-in-x dictionary
  exp34_normalization.py         absolute term in the eigenproblem's
                                 denominator
  exp33_degree.py                spectrum vs class degree, cross-validation
                                 and fold agreement over trajectories
  exp32_blind_readout.py         the low eigen-subspace rotated to its
                                 sparsest form and printed as formulas,
                                 no ground truth
  exp31_polynomial_discovery.py  FINS discovery as a generalised
                                 eigenproblem in a polynomial generator
                                 class: no training, equation-free
  determining.py                 exact determining equations of
                                 u_t + a u u_x = b u_xx as a linear map
  exp7_receptive_field.py        sensitivity map d out(x_c)/d u(x') for the
                                 FNO generator vs a pointwise MLP (torch)
  exp8_seam_window.py            seam-windowed (local) equivariance test on
                                 the torus; domain-of-dependence hierarchy
  exp9_surrogate_locality.py     locality of dS_theta/du0 vs the exact heat
                                 kernel — validity check for exp8 (torch)
  exp10_leak_vs_delta.py         exp5 x exp8 x exp9 on one surrogate per
                                 budget: delta, leak and windowed residual
                                 measured together (torch)
  exp11_delta_j_probe.py         delta_J = Jacobian error over random probes:
                                 the variable the stability lemma actually
                                 needs (torch)
  exp12_short_horizon.py         falsifiable short-horizon predictions for
                                 Burgers projective and KdV wrap-around
  exp13_projective_correction.py the Burgers projective generator of the
                                 paper is not a symmetry; derivation + proof
                                 by determining equation + windowed JVP
  exp14_nonlinear_breaking.py    eps^2 detection law on a nonlinear operator
                                 (forced Burgers)
  patch_burgers_projective.diff  OPT-IN fix of the Burgers v5 generator in
                                 ground_truth.py (changes all Burgers
                                 scoring; apply consciously)
  patch_sanity_per_sample.diff   fix for method_fins.sanity_check_equivariance
                                 and alpha propagation in get_gt_generators
  results/*.csv                  outputs of these runs
```

## Headline results (oracle level, final)

**exp1.** On the periodic domains the experiments actually use, the
essential symmetry algebra has rank 3 for all three PDEs, not (6, 5, 4):

| PDE | at floor (torus, resolved grid) | above 1e-2 threshold |
|---|---|---|
| heat | d_x, d_t, u d_u (<= 1e-27) | Galilean 1.2e-1, scaling 1.5, projective 2.3e-1 |
| Burgers | d_x, d_t, Galilean (<= 5e-13) | scaling 6.3e-1, projective 4.1 |
| KdV | d_x, d_t, Galilean | scaling 1.0 (Q-normalised) |

Wide-domain controls: heat collapses to full rank 6 at machine precision;
Burgers reaches 4 of 5 (projective has Q0 = -x and is never
periodic-compatible; note it nevertheless scores 4.5e-3 "below threshold"
in the ||v||^2-normalised Rayleigh form — an example of the threshold's
normalisation fragility); KdV scaling collapses in the repo form (2.9e-4)
but keeps 3.8e-2 in the scale-invariant Q form because dispersive radiation
wraps the domain within T = 1.

Data-resolution findings on the paper's own grid (Nx = 64): Burgers shocks
at nu = 0.01 are unresolved, and on the repo pipeline even the Galilean
boost fails (0.87) because the 2/3 dealiasing mask breaks commutation with
shift-plus-constant on under-resolved fields; for KdV, solitons cross more
than one cell per saved frame, so the fd6 time derivative — and with it the
d_t characteristic — cannot be represented (0.34, Q-normalised). A material
share of the paper's Table 3 residuals is therefore a data-resolution
floor, not surrogate error.

**exp2.** Null model = 6 random directions of the trivial superposition
subalgebra v_beta = beta(x,t) d_u, scored with the repository `Metrics`:
d_G = 3.011 +/- 0.056, mean cos = 0.263 +/- 0.048 (20 trials). Per
generator, the null set reaches |cos| = 0.67-0.71 against u d_u, Galilean,
projective and exactly 0 against d_x, d_t, scaling. Consequences: the
per-generator cosines in Table 2 split into a null-immune triple and a
null-inflatable triple; LaLiGAN's d_G = 3.39 is worse than the random
floor; FINS (2.17 / 0.50) sits above it.

**exp3.** Residual vs domain geometry, centred localised data, exact
operator: at L/w = 4 the x-weighted generators are already at the 1e-2
threshold, at L/w = 8 near 1e-8, from L/w = 16 machine zero. Control: a
single bump ON the seam at L/w = 128 lifts the residual back by ~28 orders
(Galilean 3.9e-3). The operative condition for the rank-6 regime is data
support bounded away from the seam, not domain length per se.

**exp4.** Controlled breaking with an exact linear operator
(u_t = alpha u_xx + eps m u, split-step validated to 6.4e-14): log-log
slope of the broken generator's residual is 2.00/2.02 as predicted by the
Rayleigh form, kept generators stay <= 1e-19 across four decades of eps,
detection limits eps* (residual = 1e-2) are 3.8e-2 for f(x)-breaking of
d_x and 1.1e-1 for g(t)-breaking of d_t. This is the positive control and
the calibration point a stability lemma must reproduce.

**exp8 — seam-windowed test (oracle level, final).** The x-weighted
generators fail on the torus for one reason: xi = x is not periodic, so Q0
carries a jump at the seam. Lie POINT symmetries are local; the FINS
condition is global only because S is. Restricting the residual to a window
W = [m, L - m] that excludes the seam is therefore legitimate up to the
contamination the seam propagates into W within T, which the equation's
domain of dependence bounds.

| setting | m = 0 (full domain) | m = 0.05 L | m = 0.10 L |
|---|---|---|---|
| heat L = 2pi, Nx = 64, repo ICs, scaling | 2.1e-1 | 7.0e-5 | 2.2e-5 |
| heat L = 2pi, Galilean / projective | 3.5e-3 / 5.3e-3 | 1.7e-7 / 2.7e-7 | 1.7e-8 / 3.0e-8 |
| Burgers L = 2pi, Nx = 512, scaling | 1.8e-1 | 7.5e-2 | 6.8e-5 |
| Burgers projective | 9.0e-1 | 9.4e-1 | 1.0e+0 |
| KdV L = 2pi, Nx = 128, scaling | 1.25e+1 | 1.28e+1 | 1.32e+1 |

Heat reaches the rank-6 regime on the PAPER'S OWN grid and data with a 5
percent margin — no new data, no architectural change, one mask in the
loss. Burgers scaling is rescued at 10 percent and plateaus; Burgers
projective is not rescued at any margin (unresolved: it was borderline
4.5e-3 on the wide domain at nu = 0.1, so a nu sweep is needed before
claiming anything). KdV does not improve at all and drifts upward, exactly
as the dispersive case predicts — group velocity ~ -3k^2 is unbounded on
the grid, so there is no finite domain of dependence to localise into. The
hierarchy heat > Burgers > KdV was predicted from the domain of dependence
before measurement, not fitted after.

**exp9 — validity of exp8 on a surrogate (first pass, small model).** The
window removes the seam from the residual but not from the input:
dS_theta[u0](Q0) inside W still reads Q0 at the seam. The exact operator
bounds that by the Green kernel; an FNO has no domain of dependence by
construction. Measured rows of dS_theta/du0 against the exact heat kernel,
at test rel-L2 delta = 0.167 (small CPU model, 96 train samples, 60
epochs), on a sample with L = 0.7:

| t | sigma = sqrt(2 alpha t) | r95 exact | r95 learned | ratio | leak beyond exact 4 sigma |
|---|---|---|---|---|---|
| 0.064 | 0.0358 | 0.066 | 0.241 | 3.7x | 0.103 |
| 0.128 | 0.0506 | 0.098 | 0.241 | 2.4x | 0.066 |
| 0.256 | 0.0715 | 0.142 | 0.241 | 1.7x | 0.027 |
| 0.490 | 0.0990 | 0.197 | 0.241 | 1.2x | 0.000 |

Two things to read here. The learned r95 is CONSTANT in t: the surrogate's
Jacobian has a fixed width set by the architecture, not by the diffusion
length, so it does not reproduce the parabolic cone. At early times 10
percent of its mass sits outside the exact 4-sigma cone, and the profile is
ragged rather than kernel-like. The t = 0.490 row showing zero leak is an
artefact of L = 0.7: the exact 4-sigma radius already exceeds the half
domain, so nothing can be outside it. Whether the leak shrinks with
surrogate quality is the open question and requires the delta sweep of exp5
crossed with this diagnostic; at delta = 0.167 the surrogate is far from
the paper's 0.37 percent on heat.

**exp10 — the three quantities on one surrogate (smoke).** Single-domain
heat at L = 2 pi (the exp8 setting), two CPU budgets:

| n_train | delta | leak | R(0.05L) Galilean / scaling / projective |
|---|---|---|---|
| 48 | 1.269 | 0.100 | 3.3e-1 / 5.4e-1 / 3.1e-1 |
| 96 | 0.052 | 0.267 | 2.2e-3 / 3.2e-3 / 2.7e-3 |

Two decades of delta buy two decades of windowed residual, and at delta =
0.052 all three x-weighted generators are already below the 1e-2 threshold
at m = 0.05 L — which on the torus they never are. But the gap to the
oracle (1.7e-7 / 7.0e-5 / 2.7e-7) is four to six orders, the m = 0.05 to
0.10 step buys nothing, and leak went UP while delta fell. Two budgets
cannot separate "R tracks delta" from "R is heading for a floor"; the
question needs five or six budgets spanning delta from 1e-1 to 1e-3, which
is a GPU run.

**exp13 — the Burgers projective generator of the paper is wrong (final).**
The paper and `ground_truth.py` list v5 = xt d_x + t^2 d_t - (x + tu) d_u.
Substituting its characteristic into the linearised Burgers operator
M[Q] = Q_t + (uQ)_x - nu Q_xx leaves the defect M[Q] = -2(u + x u_x) != 0;
the correct projective symmetry has phi = x - tu, for which M[Q] = -3tE -
xt(d_x E) - t^2(d_t E) = 0 identically on solutions. Numerically (local-FD
determining equation on a wide domain, two viscosities): corrected variant
2.5e-15 / 2.1e-16 against a d_x floor of 5e-13; repo variant 2.4e-2 / 2.2e-2;
the discrete defect matches -2(u + x u_x) to 3e-7. On the torus the
corrected generator is rescued by the seam window exactly like the scaling
control, while the repo variant stays O(1) at every margin — which resolves
the exp8/exp12A anomaly. Consequences: the Burgers ground-truth subspace of
Table 1 was partly wrong for ALL six methods; the v5 column of Table 3
tracked a non-symmetry, so its plateau under IC enrichment was guaranteed a
priori (the v3 orbit-coverage story is unaffected); Section 5.1's formula
needs correction. Apply `patch_burgers_projective.diff` and re-score.

**exp12 — short-horizon predictions (final).** KdV wrap-around: confirmed —
the Q-normalised scaling residual on the wide domain falls monotonically
over seven orders (6.0e-10 / 7.0e-7 / 9.0e-3 at T = 0.05 / 0.2 / 1.0),
tracking the radiation range v_g T = 17 / 68 / 341 against L = 100. Burgers
projective seam-advection hypothesis: REFUTED — the residual is flat in the
margin at every horizon, which is what exposed the wrong generator and led
to exp13.

**exp11 — delta_J is the right variable for the stability lemma (CPU,
three budgets).**

| budget | delta | delta_J | k = delta_J/delta | leak (avg) | R/delta^2 | R/delta_J^2 |
|---|---|---|---|---|---|---|
| 96:30 | 0.138 | 0.119 | 0.9x | 0.170 | 0.51 | 0.69 |
| 96:60 | 0.052 | 0.078 | 1.5x | 0.220 | 0.98 | 0.44 |
| 96:140 | 0.023 | 0.048 | 2.1x | 0.225 | 2.57 | 0.57 |

R/delta^2 drifts 5x across a 6x range of delta; R/delta_J^2 stays within
1.6x with no trend. The lemma must be stated against the Jacobian error
delta_J, not the test error delta. The amplification k GROWS as the model
improves (function error falls faster than Jacobian error), which is
exactly why delta-based thresholds cannot be portable. The sample-and-time
averaged leak does NOT fall with delta (0.17 -> 0.22 -> 0.23), supporting
the architectural reading of exp9 at these budgets.

**exp14 — detection law on a nonlinear operator (final).** Forced Burgers
u_t + u u_x = nu u_xx + eps sin(x) u: both broken generators (d_x,
Galilean) show log-log slope exactly 2.00 over four decades of eps while
the kept d_t stays pinned at the eps = 0 floor (~4e-22). Within the swept
range the broken residuals remain below 1e-2; slope-2 extrapolation puts
eps* at ~1.1 (d_x) and ~0.29 (Galilean).

**exp5 — full GPU sweep (Colab session, T4, neuralop FNO, heat multi-scale,
seed 42, rep 0; 5.4 h).** Fixed per-sample sanity vs delta:

| budget | delta | d_x | d_t | u d_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|
| 64:100 | 0.3993 | 7.5e-1 | 6.6e-3 | 3.0e-1 | 3.0e-1 | 3.0e-1 | 3.1e-1 |
| 128:150 | 0.1682 | 1.2e-1 | 1.7e-3 | 1.7e-1 | 1.8e-1 | 8.0e-2 | 2.0e-1 |
| 256:250 | 0.0976 | 3.7e-2 | 7.4e-4 | 1.5e-1 | 1.7e-1 | 5.8e-2 | 1.9e-1 |
| 512:400 | 0.0226 | 3.4e-3 | 2.3e-4 | 1.2e-2 | 4.5e-2 | 5.2e-2 | 8.1e-2 |
| 1024:600 | 0.0051 | 1.7e-4 | 3.6e-5 | 5.6e-5 | 3.8e-2 | 5.1e-2 | 7.4e-2 |
| 1024:1200 | 0.0037 | 1.9e-4 | 4.9e-5 | 2.5e-5 | 3.8e-2 | 5.2e-2 | 7.5e-2 |

Read-outs. (1) At the paper's own delta = 0.0037 the corrected sanity
reproduces the essential rank 3 with a 2-3 order gap: exact symmetries at
2e-5..2e-4, geometry-broken generators at 4e-2..7e-2; the 1e-2 threshold
falls inside the gap. (2) The broken-generator plateaus EQUAL the exact
operator's values on the same four samples: Galilean 3.8e-2 vs oracle
3.7e-2 (ratio 1.04), projective 7.5e-2 vs 6.7e-2 (1.12) — at high budget
the pipeline measures geometry, not model error. Scaling is the exception:
plateau 5.2e-2 vs oracle 5.4e-1, ten times BELOW; the fd6-vs-exact u_t
explanation was tested and ruled out (oracle unchanged under fd6), leaving
the smoothing Jacobian damping the seam-propagated singular content that
dominates the scaling oracle (consistent with exp9/exp10); needs a
per-sample decomposition. (3) The u d_u homogeneity anomaly seen at
256-512 resolved itself: 1.5e-1 -> 2.5e-5, a super-quadratic collapse
between 512 and 1024 — a mid-training transient, not an architectural
floor. (4) d_x follows slope 1.93 over the full range; between the last
two budgets d_x and d_t stagnate while u d_u keeps falling — either
last-epoch noise (single rep) or the onset of a delta_J floor; resolved by
a second repeat and the GPU exp11. (5) Broadcast bias at full budget:
scaling a deterministic 4.07x, Galilean/projective 1.4x/1.24x, and u d_u
flips sign at the floor (0.77, 0.42) — the bias is not uniformly
optimistic, it is simply wrong.

**exp6 — full GPU run (Colab session, T4, 4.05 h; verdict: prediction NOT
confirmed, analysis pending).** lambda_b = 0: deflated norms 1.000 / 1.045 /
0.218 / 0.120 / 0.116 / 0.027, b-shares 0.947 / 0.570 / 0.412 / 2.420 /
2.828 / 2.364, gap rank 2. Slots 1-2 reproduce the killed session (0.986 /
0.504) — the in-vivo dominance of the trivial subalgebra in unconstrained
discovery is a reproducible fact and stands as the empirical companion to
exp2 regardless of everything below. b-share > 1 in late slots is a
parameterisation pathology: the share is mean(b^2)/mean(phi^2), and values
above 1 mean a*u and b anticorrelate and cancel inside phi.

lambda_b = 30: b-share 0.000 in every slot — the penalty is fully
effective — but the norm sequence is non-monotone: 1.006 / 0.067 / 0.037 /
0.141 / 0.505 / 0.208. Slot 5 finding a direction with four times the
orthogonal norm of slots 2-4 means those slots MISSED available directions:
the same surrogate's sanity shows unit-norm candidates (d_x, d_t, u d_u)
with residuals <= 1e-3 that were not taken. The deflation is
optimisation-limited at 50 epochs/slot for the 4-channel constrained net.
The printed rank 6 is the no-gap fallback of `spectral_rank` on sorted
norms, not a detection of six directions.

The rank-3 prediction is therefore neither confirmed nor refuted by this
run. The discriminating step costs seconds, not GPU-hours:
`exp6_analyze.py` scores every saved slot against the alpha-correct ground
truth (per-slot best-match cosine, channel energy split, Hungarian mean,
d_G) from the npz artefacts. Next-fix candidates, to be chosen after that
analysis: per-slot multi-init screening (restarts) if slots 2-4 are
optimisation casualties; revisiting the deflated-norm anchor if slot 1
itself is junk.

**exp10 — full GPU run (Colab session, T4, 5.2 h): THE WINDOWED TEST
TRANSFERS TO THE PIPELINE.** Single-domain heat at L = 2 pi, neuralop FNO:

| n_train | delta | leak | R(0) Gal/scal/proj | R(0.05L) | R(0.10L) |
|---|---|---|---|---|---|
| 128-512 | 0.22-0.31 | 0.18-0.32 | ~2e-1 | ~2e-1 (flat) | ~2e-1 |
| 1024:600 | 0.0013 | 0.071 | 1.1e-2 / 9.4e-3 / 1.8e-2 | 1.8e-5 / 2.7e-5 / 3.9e-5 | 1.5e-5 / 1.4e-5 / 4.3e-5 |
| 1024:1200 | 0.0007 | 0.058 | 1.1e-2 / 9.3e-3 / 1.8e-2 | 8.7e-6 / 1.3e-5 / 2.1e-5 | 4.7e-6 / 4.8e-6 / 1.6e-5 |

Read-outs. (1) At delta <= 1.3e-3 a 5-percent seam margin buys THREE
orders of magnitude on every x-weighted generator (1e-2 -> 1e-5, three
orders below the 1e-2 threshold), and m = 0.10 keeps improving — the
seam-windowed equivariance condition works end to end on the paper's
architecture, which greenlights a windowed-FINS discovery variant.
(2) In the undertrained regime (delta ~ 0.2) the window buys nothing —
R is flat in m, as it must be when model error dominates the geometric
signal. (3) The Jacobian leak FALLS with delta at full scale (0.32 ->
0.058), contradicting the small-CPU-model reading of exp9/exp11 that it
was architectural; that concern is retired for neuralop FNO at paper
budgets, though the residual 5.8 percent is still finite. (4) Against the
exact-operator reference at m = 0.05 L: scaling lands BELOW oracle
(1.3e-5 vs 7.0e-5, the smoothing-Jacobian effect again), Galilean and
projective 50-80x above their oracle floors (8.7e-6 vs 1.7e-7) — far
below threshold either way. (5) The delta trajectory itself is notable:
128-512 samples plateau at 0.22-0.31 on this single-scale dataset, then
1024 collapses to 1.3e-3 — a sharp data-requirement transition worth one
sentence in the paper.

**exp9 — full GPU run (Colab session, T4, 1.25 h): THE TRAINED JACOBIAN
REPRODUCES THE PARABOLIC CONE.** neuralop FNO, n_train = 1024, epochs =
500, delta = 0.0047; probed sample L = 2.0, T = 4.0 (multi-scale draw):

| t | sigma | r95 exact | r95 learned | ratio | leak exact | leak learned |
|---|---|---|---|---|---|---|
| 0.504 | 0.100 | 0.188 | 0.219 | 1.2x | 9.4e-5 | 0.018 |
| 1.008 | 0.142 | 0.281 | 0.281 | 1.0x | 4.5e-5 | 0.010 |
| 2.016 | 0.201 | 0.406 | 0.406 | 1.0x | 7.1e-5 | 0.004 |
| 4.000 | 0.283 | 0.562 | 0.594 | 1.1x | 0 | 0.000 |

Read-outs. (1) r95 learned GROWS with t and tracks the exact kernel to
1.0-1.2x — the domain-of-dependence cone, including its sqrt(t) width, is
induced by training. The small-CPU-model finding of a t-independent,
3.7x-too-wide Jacobian is decisively overturned at scale; this is the
THIRD small-model conclusion reversed by a full-budget run (after the
leak trend in exp10 and the u d_u homogeneity floor in exp5) — Jacobian
diagnostics must be run at target scale, worth a methods sentence.
(2) Residual nonlocality is 1.8 percent of |J| mass outside the exact
4-sigma cone at the earliest time, decaying to zero — consistent with
exp10's global leak of 0.058-0.071. (3) Caveats: on this L = 2 sample the
4-sigma radius exceeds the half-domain from t ~ 2, so the last two rows
are weakly informative (leak = 0 trivially); the meaningful evidence is
the early rows, plus the visible 1e-3..1e-2 sidelobes in the profile;
single sample and centre — averaging is an optional upgrade.
(4) Consequence: the locality regulariser is demoted from necessary
component to optional robustness add-on; the windowed test's prerequisite
is now supported both structurally (this) and functionally (exp10).

**exp6_analyze on the first full run (fields survived): the fork is
resolved — optimisation lottery, not a structural failure.** Under
lambda_b = 30 the deflation DID find all three null-immune generators,
just scrambled across slots: d_x at cos 0.871 (slot 6, energy 0.76 in
xi), d_t at 0.753 (slot 2), Galilean at 0.726 (slot 4); slots 1/3/5 are
weak phi-dominated directions (best cos 0.18-0.47). A side derivation
sharpens the picture: in the constrained a-sector M[a u] = u (a_t -
alpha a_xx) - 2 alpha a_x u_x, which vanishes on all solutions only for
a = const — the a-sector has NO continuous degeneracy, so the junk slots
are delta_J-level surrogate artifacts, not remnants of an infinite
family. Under lambda_b = 0 the field analysis confirms superposition
filling directly: slots 1-3 all best-match u d_u (0.34/0.51/0.66) with
b-shares 0.95/0.57/0.41. Even on the broken deflation the constraint
improves both metrics: Hungarian mean 0.441 -> 0.526, d_G 2.583 -> 2.350
(reference sample L = 0.5; not the full Table 1 protocol). Success
criterion for the v2 rerun (multi-init): {d_x, d_t, u d_u or Galilean}
in slots 1-3 with cos >= 0.7 and a near-monotone norm sequence.

**exp11 — GPU run (small 64x24 grid, 18 min): k = 14.1x.** delta_J /
delta amplification grows 1.1x -> 1.5x -> 14.1x as delta falls to
0.0027 — bridging toward the 34-40x measured on the independent JVP
prototype, and settling why delta-based thresholds cannot be portable.
As an UPPER bound, R <= 1.3 delta_J^2 holds at every budget including
the saturated ones; as a scaling law it loosens 20x at the good budget
(R/delta_J^2 = 0.057 vs ~1.3 elsewhere) — working hypothesis: the global
delta_J counts nonlocal/seam error that the windowed residual excludes.
A windowed delta_J column was added to test exactly this (identical to
global at bad budgets in the CPU smoke, as expected — discrimination
only possible at high quality). The 512:400 point is a failed training
draw (delta = 0.536 non-monotone, delta_J +/- 0.33, last-epoch model) —
excluded from any fit. Caveat: small-model numbers have been overturned
three times in this project; the decisive k must be re-measured on the
paper grid — `--full-grid` flag added, run pending
(`--full-grid --budgets 1024:300 1024:600`, ~2.5-3 h).

**exp11 --full-grid (paper grid 64x128, neuralop, 2.3 h): THE STABILITY
LEMMA IS CONFIRMED IN delta_J.**

| budget | delta | delta_J | dJ_win | k | leak | R(0.05L) Gal/scal/proj | R/delta^2 | R/delta_J^2 |
|---|---|---|---|---|---|---|---|---|
| 1024:300 | 0.0023 | 0.0220 | 0.0224 | 9.6x | 0.053 | 3.5e-5 / 6.3e-5 / 6.1e-5 | 9.97 | 0.109 |
| 1024:600 | 0.0011 | 0.0152 | 0.0154 | 14.3x | 0.043 | 1.6e-5 / 2.7e-5 / 3.2e-5 | 21.9 | 0.108 |

R/delta_J^2 is constant to 1 percent while R/delta^2 doubles. Fitted
exponents: R ~ delta_J^2.00 and R ~ delta^1.02 — the residual is exactly
quadratic in the Jacobian error and merely linear in the test error. The
lemma is therefore

    R(v) <= C delta_J^2,   C = 0.108 measured on the paper grid,

and the discoverability threshold gets its constructive replacement:
tau(delta_J) = C delta_J^2 instead of the fixed 1e-2 calibrated on biased
sanity numbers. At the paper-grade delta_J = 0.0152 this is tau = 2.5e-5,
right at the measured residuals of the in-window symmetries.

Amplification k = delta_J/delta grows 9.6x -> 14.3x, reproducing the
small-grid 14.1x and bridging to the 34-40x of the independent JVP
prototype: delta cannot serve as a portable threshold variable because the
function error falls faster than the Jacobian error.

The windowed delta_J equals the global one to 2 percent (0.0224 vs 0.0220;
0.0154 vs 0.0152), REFUTING the working hypothesis that the small-grid
20x loosening of R/delta_J^2 came from seam error counted globally but
excluded by the window. Re-reading with the full-grid numbers in hand: the
small-grid C values of ~1.3 came from saturated budgets (delta ~ 0.3-0.5)
that were never in the asymptotic regime, while its one good budget gave
0.057 — the same order as 0.108. The lemma holds where the quadratic
regime holds, and saturated budgets simply do not belong in the fit.

Caveats: two budgets, one seed, single-scale L = 2 pi data; the constant C
is architecture- and geometry-specific, not universal. A third budget and
a second seed would turn the two-point consistency into a fit.

**exp6 v2, lambda_b = 0 (multi-init, 2.9 h): BETTER OPTIMISATION MAKES
UNCONSTRAINED FINS WORSE — the degeneracy is structural.** Norms 0.996 /
0.994 / 0.291 / 0.040 / 0.113 / 0.069; b-shares 0.965 / 0.818 / 1.483 /
1.026 / 2.354 / 0.760; slots 1-3 all best-match u d_u with phi energy
~1.00.

Comparison with the single-init v1 run, same seed and surrogate budget:

| | v1 (single init) | v2 (3-init screening) |
|---|---|---|
| b-shares, slots 1-3 | 0.95 / 0.57 / 0.41 | 0.965 / 0.818 / 1.483 |
| Hungarian mean cos | 0.441 | 0.351 |
| d_G (vs 6-dim GT) | 2.583 | 2.494 |

Screening for better optima RAISED the trivial content and LOWERED the
match to the true algebra. This is the expected behaviour once the
structure is understood: on a linear PDE the superposition directions are
EXACT global minima of the unconstrained loss (zero equivariance
residual), so a better optimiser finds more of them. The corollary for the
paper is uncomfortable and should be stated plainly: the published heat
result at lambda_b = 0 depends in part on the optimiser NOT reaching the
loss's global minima. The degeneracy is not an optimisation artefact to be
tuned away — it is a property of the functional, and the constraint (or an
equivalent factorisation) is mandatory rather than optional.

lambda_b = 30 crashed at 8 s on checkpoint load — `torch.save` preserves
the state_dict's `_metadata`, which returns as a KEY and trips
`load_state_dict`'s strict check. Fixed by popping it on both save and
load (verified through a full save-then-reload cycle); no GPU time was
lost and the Drive checkpoint is valid, so the rerun is deflation only
(~1.7 h). The decisive rank-3 test therefore remains open.

**exp6 v2, lambda_b = 30 (multi-init): criterion NOT met — and the
diagnosis moves from optimisation to instrumentation.** Slots 1-3 give
projective 0.468, d_t 0.566, u d_u 0.623 (all below the 0.7 bar), norms
0.970 / 0.056 / 0.076 / 0.244 / 0.207 / 0.436 still badly non-monotone,
d_x again landing in slot 6 (cos 0.972). Against v1: Hungarian 0.526 ->
0.533, d_G 2.350 -> 2.339 — multi-init changed nothing, so the
initialisation lottery was NOT the cause. Two instrumentation defects
found instead.

(1) The reported norm was read off ONE sample while the anchor holds the
BATCH-AVERAGE norm at 1 — and on the repo's multi-scale heat mixture that
sample is L = 0.5, the smallest domain. A single-scale smoke with
batch-averaged norms shows the size of the error directly: slot norms
1.064 / 1.152 / 1.055 batch-averaged versus 1.008 / 0.981 / 0.519 read off
sample 0. The non-monotone sequences of v1 and v2 were substantially a
measurement artefact, and the same multi-scale mismatch also affects the
deflation basis, which is built from sample-0 fields and broadcast against
batches living on other grids. Fixed: `--single-scale` (L = 2 pi, the
exp8/exp10/exp11 setting) and `--n-eval` batch-averaged norms.

(2) A deeper design point, verified against the repository: the anchor is
applied to the DEFLATED generator, and the repo comments explain why (an
anchor on the raw field lets the network copy a frozen generator, sending
v_perp and the loss to zero). But that same choice makes the rank signal
self-defeating — the optimiser is explicitly instructed to produce a
deflated norm of 1 at EVERY slot, so a norm collapse can only happen where
optimisation fails, not where directions run out. The single-scale smoke
shows exactly this: all three slots sit at norm ~1.1. The spectral-gap
rank criterion of the paper (Section 3.3, Eq. 5) therefore does not
measure what it is claimed to measure, and this affects the ORIGINAL
method, not just the constrained variant — it is the most consequential
finding of the exp6 line and needs a redesigned rank statistic
(candidates: residual at fixed norm per slot, or the achievable Rayleigh
value once the deflated norm is pinned).

**exp15 — rank-statistic screening (CPU, minutes; oracle mode): the
rank bottleneck is the GENERATOR, not the statistic and not the
surrogate.** Two candidate replacements for the paper's norm gap were
screened on the constrained deflation: the converged Rayleigh value, and
the residual re-evaluated with the deflated generator rescaled to unit
norm. Both must RISE when no genuine direction remains, since the
optimiser can always produce something orthogonal of norm 1 but cannot
make it equivariant.

Screening with a small trained surrogate was noise-limited by
construction: residuals 1.2e-2..3.4e-2 sit exactly at the lemma floor
C delta_J^2 ~ 0.027 for delta_J ~ 0.5, so no statistic could separate
anything. Rerun with an exact heat operator wrapped in the surrogate's
signature (delta_J = 0, `--oracle`):

| slot | norm | rayleigh | resid @ unit norm |
|---|---|---|---|
| 1 | 1.007 | 8.9e-5 | 2.7e-4 |
| 2 | 1.031 | 1.3e-4 | 4.1e-4 |
| 3 | 0.993 | 1.7e-4 | 5.0e-4 |
| 4 | 0.989 | 1.4e-4 | 4.2e-4 |
| 5 | 0.968 | 4.7e-4 | 1.4e-3 |
| 6 | 0.994 | 3.9e-4 | 1.2e-3 |

Two conclusions. (a) The norm is pinned at 1.0 +/- 0.04 across all six
slots even with a perfect operator — direct confirmation that the anchor
on the deflated generator destroys the paper's rank signal; this is a
property of the loss, not of surrogate quality. (b) With the EXACT
operator, slot 1 should reach machine zero if the network could represent
d_x, d_t or u d_u (all inside the constrained ansatz), and instead
plateaus at 8.9e-5, improving only ~10x from 12 to 80 epochs. The
generator network — not the surrogate, not the statistic — is what limits
every slot; the residual ordering across slots is therefore
training-noise, and no rank criterion evaluated on these runs can mean
anything.

Consequence for the revision: rank detection under the constrained ansatz
cannot be settled by a longer GPU run of exp6 in its present form, and the
planned 4.6 h rerun is NOT worth spending. The prerequisites are (i) an
anchor that does not pin the deflated norm, (ii) a generator that provably
reaches machine zero on a known symmetry with the oracle operator — a
cheap CPU acceptance test that should gate any further GPU time.

**exp16 — generator acceptance test (CPU, 1 min): the deflation floor is
a FIELD-ERROR AMPLIFICATION, not an optimisation failure.** Oracle
operator (delta_J = 0), repo `SingleGeneratorFNO`, known heat symmetries
as targets:

| target | R_analytic | eps_fit | R_fitted | R_fitted/eps_fit^2 | R_scratch |
|---|---|---|---|---|---|
| d_x | 8.3e-13 | 0.006 | 1.3e-4 | 3.5 | 4.5e-5 |
| u d_u | 9.6e-15 | 0.008 | 3.6e-4 | 5.0 | 4.5e-5 |

Read-outs. (1) The FNO fits the analytic generator to 0.6-0.8 percent —
expressivity is not the problem. (2) But that 0.7 percent field error
costs FOUR TO FIVE ORDERS in the residual: R_fitted ~ 1e-4 against an
analytic floor of ~1e-13. The residual amplifies field error
quadratically with a factor of 3.5-5.0, so the achievable residual is
R ~ A eps^2 with A ~ 4 — a THIRD amplification constant alongside the
lemma's C = 0.108 (surrogate Jacobian) and the k = 14x of exp11.
(3) Equivariance training from scratch reaches 4.5e-5, BELOW the
supervised fit's residual: the optimiser is doing its job, the acceptance
criterion passes, and the floor is not an optimisation defect.

Consequence: a deflation slot cannot resolve directions whose residual
separation is below A eps_fit^2 ~ 1e-4, no matter how good the surrogate
is. That number is the true resolution limit of the discovery loop, and it
explains every non-monotone norm sequence in exp6: on the torus the
essential directions sit at ~1e-28 and the geometry-broken ones at
~1e-2..1e-1, so THOSE are separable, but the slot-to-slot ordering among
near-degenerate candidates is decided at the 1e-4 level, i.e. by noise.
The practical route to rank detection is therefore to lower eps_fit
(bigger/better generator net, longer fits, or a coordinate-aware
parameterisation), not to lengthen deflation training.

**exp16 --full-grid (paper grid 64x128, GPU, 17 min): the generator floor
and the surrogate floor are COMPARABLE at the paper's best budget.**

| target | net | R_analytic | eps_fit | R_fitted | R/eps^2 | R_scratch |
|---|---|---|---|---|---|---|
| d_x | single | 9.6e-12 | 0.007 | 2.0e-4 | 3.65 | 6.9e-5 |
| d_x | constrained | 9.6e-12 | 0.008 | 2.5e-4 | 3.50 | 5.0e-5 |
| d_t | single | 4.1e-6 | 0.006 | 2.8e-5 | 0.76 | 6.9e-5 |
| d_t | constrained | 4.1e-6 | 0.007 | 4.6e-5 | 0.94 | 5.0e-5 |
| u d_u | single | 1.2e-14 | 0.013 | 2.6e-3 | 14.7 | 6.9e-5 |
| u d_u | constrained | 1.2e-14 | 0.008 | 2.4e-4 | 3.83 | 5.0e-5 |

(R_scratch has no target — free equivariance training finds whatever
direction it finds — so it is a per-net constant, repeated down the
column; the script's table format obscures that.)

Four read-outs.

(1) THE DISCUSSION CLAIM NEEDS QUALIFYING. The free-generator floor is
5.0e-5 (constrained) / 6.9e-5 (single) with a PERFECT operator, while the
surrogate floor at the paper's best budget is C delta_J^2 = 0.108 x
0.0152^2 = 2.5e-5 (exp11). The two are within a factor of 2-3. The paper's
third generalising finding — "the quality of the learned surrogate is the
binding constraint... effort is better spent improving S_theta than
refining the symmetry search" — holds in the undertrained regime and stops
holding at the good end, where generator and surrogate are co-limiting.
Further surrogate work buys at most a factor of 3.

(2) A THIRD FLOOR, previously invisible: the analytic d_t residual is
4.1e-6, not machine zero, because Q = -u_t for d_t is formed with the
6th-order FD time derivative of the loss. Any generator with eta != 0
inherits this discretisation floor. d_x (9.6e-12) and u d_u (1.2e-14) are
unaffected, so the three essential torus directions do NOT share a common
floor — a fact worth stating when reporting per-generator residuals.

(3) eps_fit did NOT improve from the 32x32 probe config to the paper grid
(0.006-0.013 in both): the ~1 percent supervised-fit error is a property
of the FNO generator parameterisation, not of resolution. Lowering it
needs a different parameterisation (coordinate-aware or pointwise), which
is the same conclusion the locality analysis of exp7 pointed at.

(4) The constrained ansatz helps EXPRESSIVITY on amplitude-type
generators, independently of its role against the trivial subalgebra: for
u d_u it fits to 0.008 vs 0.013 and reaches R_fitted 2.4e-4 vs 2.6e-3, a
10x improvement, because phi = a u with a = const represents u d_u
exactly. A second, independent argument for the constraint.

**exp17 — symbolic ground truth (sympy, seconds).** Every tabulated
generator checked against the linearised equation on the jet space with
time derivatives eliminated by the PDE. All heat and KdV generators pass;
all Burgers generators pass EXCEPT the paper's projective, whose defect is
exactly M[Q] = -2(u + x u_x) — the symbolic confirmation of exp13. The
corrected phi = x - tu passes.

Bracket closure of the essential (torus) subalgebras, also symbolic: heat
{d_x, d_t, u d_u} is ABELIAN; Burgers and KdV {d_x, d_t, Galilean} are
3-dimensional solvable with [d_t, Galilean] = d_x. All three are closed,
so the exact ACE target is 0 — "essential rank 3" is now a statement about
a subalgebra, not an observation about a subspace.

**exp1 Burgers, re-run with the corrected projective generator.** The
earlier Burgers rows scored a NON-symmetry, so they are superseded:
resolved torus now gives d_x 2.4e-13, d_t 3.5e-14, Galilean 5.5e-13 below
threshold and scaling 6.3e-1, projective 4.1e-1 above — the rank-3
conclusion is unchanged, but the wide-domain control improves markedly
(projective 2.9e-3, now genuinely below threshold, versus the previous
borderline reading for the wrong field). `exp_common.gt_characteristics`
was patched accordingly; exp8 and exp12 Burgers rows should be re-run for
consistency (cheap, CPU).

**exp18 — detectable rank is a property of the PROTOCOL, not the
equation.** Oracle operators, generators detected out of the classical rank:

| setting | classical rank | m/L = 0 | m/L = 0.05 | m/L = 0.10 |
|---|---|---|---|---|
| heat, the paper's own grid Nx = 64 | 6 | 5 | **6** | **6** |
| Burgers, resolved Nx = 512, T = 0.5 | 5 | 3 | 4 | **5** |
| KdV, resolved torus Nx = 128 | 4 | 3 | 3 | 3 |
| KdV, wide L = 32pi, short T = 0.2 | 4 | **4** | **4** | **4** |

This corrects the earlier "essential rank (3,3,3)" framing recorded above,
which conflated two ground truths. Rank 3 is the algebra of the periodic
BOUNDARY VALUE PROBLEM under the GLOBAL residual, and as such it is right
(and closed, exp17). But the paper's aim is the symmetry algebra of the
local law, and under the seam-windowed residual the claimed ranks (6, 5, 4)
ARE reachable: heat at 5 percent margin on the published grid (Galilean
1.7e-7, projective 2.7e-7, scaling 7.0e-5), Burgers at 10 percent on a
shock-resolving grid with the corrected v5, and KdV not by windowing at all
(dispersive, no domain of dependence) but by moving to a wide domain with a
short horizon so the radiation does not wrap. What is indefensible is the
PAIRING of "rank 6" with the global torus residual, plus attributing the
Burgers and KdV shortfalls to surrogate fidelity when they are data
resolution and topology.

**exp19 — windowed FINS DISCOVERY (smoke, oracle operator).** Detectability
(exp10, exp18) is not recovery: the discovery loop under the window had
never been run. The loop here uses a windowed Rayleigh residual (numerator
AND denominator windowed, so nothing is gained by hiding energy at the
seam), random cyclic shifts averaged per step to integrate out the origin
dependence a fixed window introduces, the constrained ansatz of exp6, and —
following exp15 — the norm anchor on the RAW generator with a floor penalty
on the deflated norm, so ||v_perp|| stays free to collapse when directions
run out.

Smoke at 32x32, 48 samples, 10 epochs/slot, exact operator, 4 slots:
recovered u d_u (0.706), projective (0.741), scaling (0.781); Galilean at
0.620 just under the bar. Hungarian mean 0.712 against a null floor of
0.263, d_G 2.595 against 3.011. Crucially the deflated norms no longer sit
pinned at 1 (1.038 / 1.144 / 0.968 / 0.667), so the rank statistic has room
to move — the exp15 defect is fixed by the anchor change.

Ablations at the same budget: removing the window (m = 0) gives the same
3/6 with d_G 2.636, and removing shift averaging drops recovery to 2/6.
Neither ablation is decisive at smoke scale — the surrogate-free residuals
are already at the 1e-4 generator floor of exp16, where slot ordering is
noise — so the window's benefit for DISCOVERY (as opposed to detection)
remains untested at scale. That is the run that matters next.

**exp23 — NEURAL discovery at single scale, coords x seam (CPU torch,
3 seeds; statistic now evaluated on ALL samples, fields serialised).**
exp22 established ADMISSIBILITY under a seam margin; this runs the actual
loop -- a coord-aware CNN generator trained against the Olver residual,
exact operator, constrained ansatz phi = a(x,t) u, anchor on the raw
generator -- and ablates the two factors LieSym bundles into MS+coord,
with the domain length NEVER varying. Containment of the component
orthogonal to the essential trio, 5 slots, all 8 evaluation samples:

SCALING (novelty in the COORDINATE channels; LieSym's headline target):

| cell | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| coords + seam-aware local | 0.081 | 0.045 | 0.188 | 0.105 +/- 0.074 |
| coords + spectral (global) | 0.012 | 0.021 | 0.011 | 0.015 +/- 0.006 |
| no coords + seam-aware | 0.022 | 0.002 | 0.051 | 0.025 +/- 0.025 |

The seam-aware residual beats the periodic global one 3/3 within seed
(+0.069, +0.024, +0.177) and averages 7x higher; without coordinate
channels the value collapses to the same range as the spectral cell.
Coordinate channels are necessary (xi = x is unrepresentable without
them); given them, the seam is what the global residual loses. Neither
factor is multi-scale -- which matches LieSym's own Figure 4 ladder
(multi-scale alone +0.04 against the coord channel's +0.23) and
reproduces exp19's channel-specific signature from the dual formulation.

Galilean and projective do NOT follow this pattern: in the no-coords cell
they read 0.043/0.038 (s42), 0.074/0.095 (s43), 0.294/0.284 (s44) -- on
s44 above the coord-aware seam cell. Their novelty sits in the amplitude
channel phi ~ (x-c) u, which the network reaches through the u channel
without coordinates. Only the scaling claim is supported; an earlier
single-seed reading that claimed both factors for all three generators is
withdrawn.

Fixing the statistic mattered. Evaluating on one reference sample instead
of all eight had inflated every number and the spread with it: scaling
under the seam protocol read 0.133 / 0.182 / 0.599 (CV 0.84) and now reads
0.081 / 0.045 / 0.188 (CV 0.71); the spectral cell's CV fell from 0.68 to
0.38, and its values tightened to 0.011-0.021. The ordering survived the
correction, the magnitudes did not -- which is the reason to quote this
statistic only as a within-seed contrast. The other known sensitivity
stands: novel containment grows with slot count (s42 scaling was 0.227 at
six slots versus 0.133 at five under the old statistic), so cells are
comparable only at equal slot count. Discovered fields are now written to
`results/exp23_{coords}_{protocol}_s{seed}.npz`, so the next change of
statistic costs seconds rather than a re-run.

**Cross-sample generator alignment (`closure_metrics.align_refs`).** The
multi-sample block is produced by re-invoking each extractor on another
sample. For five of the six methods that re-evaluates one trained object,
so the generator ORDER is stable; LieNLSD re-solves the determining system
per sample, so its order and signs are not. Regressing phi across samples
would then mix different generators and the reported closure and
xi/eta-spread would be nonsense. Every sample's set is now matched to
sample 0's by |cos| with sign correction before the affine fit. Verified
by scrambling a known set per sample: cross-sample spread 7.3e-1 before
alignment, exactly 0 after. The first campaign table was computed WITHOUT
this, so its closure and spread columns need recomputing -- the d_G,
angle and cos columns are unaffected.

**v3 campaign verdict (v2 + ||Q|| channel normalisation; baselines
identical to v2, so only FINS moved). Three equations, three answers, one
mechanism.**

| | v2 -> v3 d_G | v2 -> v3 cos | eta share (GT) | z cos vs band |
|---|---|---|---|---|
| heat | 2.04 -> 2.09 | 0.56 -> 0.64 | 0.48 -> 0.39 (0.18) | +0.08 -> +2.94 |
| Burgers | 3.39 -> 2.15 | 0.26 -> 0.67 | **0.00 -> 0.17** (0.22) | -6.47 -> +2.74 |
| KdV | 3.75 -> 3.32 | 0.03 -> 0.22 | **0.00 -> 0.00** (0.29) | -9.23 -> -6.01 |

BURGERS: the eta-collapse diagnosis was right. Normalising by ||Q|| revived
the eta channel (0.00 -> 0.17 against a GT share of 0.22; channel split now
0.65/0.17/0.23 vs GT 0.47/0.22/0.31), and FINS went from worst to best
equation-free method on both d_G (2.15 vs LieGAN 2.44) and cosine (0.67 vs
Augerino 0.62), within reach of the equation-aware LIG (2.05 / 0.72) and
LieNLSD (2.08 / 0.76). All three seeds moved the same way (cos 0.29/0.26/
0.24 -> 0.68/0.67/0.53). It clears the band-limited null on cosine
(z +2.74), sits on it for d_G (-0.42), and stays under the shuffled null
(-1.97 / -0.73). Seed spread is large (d_G +/- 0.67): seed 44 lags.

KDV: eta stayed at exactly 0.00 on every seed and phi rose to 0.86. The
normalisation removed the bias but found nothing to reveal -- exactly what
the probe predicted: through the KdV surrogate every GT generator sits at a
||Q||-normalised residual >= 0.50 (d_x 0.54, d_t 0.50, Galilean 0.93,
scaling 1.32) with fit error 8 percent, while on Burgers d_x passes at
9.7e-4 and scaling reads 0.14. There is no signal in the KdV surrogate for
any loss to find. FINS improved from below-random (0.03) to 0.22 but stays
the worst equation-free method and under both nulls. The blocker is the
surrogate, full stop.

HEAT: cosine up (0.56 -> 0.64, now z +2.94 vs band), d_G neutral within
spread, channels more balanced (spread 0.93 -> 0.83). The normalisation did
not hurt the equation it was not designed for.

Where FINS-v3 stands: best equation-free on d_G for heat and Burgers, best
equation-free on cosine for Burgers, clears the band-limited null on
cosine for both, clears the shuffled null nowhere, fails on KdV with its
surrogate. Every change from the published configuration is a principled
correction -- a spacing bug, a degeneracy penalty, an anchor that no longer
pins the deflated norm, a scale-invariant normalisation -- and none injects
symmetry knowledge. The progression v1 -> v2 -> v3, each step diagnosed
before it was applied, is the method section the revision should tell.

**Paper figures (16.10; no new runs).**
paper_figures/: figstyle.py (shared style; categorical slots 1-3 validated on white), make_fig_method.py (KdV soliton
and Galilean boost in closed form; the 12 lowest eigenvalues of weak-learned_kdv_s42, lambda5 / lambda4 = 511),
make_fig_data.py (trajectory 0 of seed 42 per equation, from the run files), make_fig_comparison.py (the values of
tab:main and tab:new-learned), make_fig_noise.py (summaries: the rows of tab:noise but LieNLSD as published and the
solutions reading of Kreider et al.), make_fig_fburgers.py (fburgers/exponent.json per seed; 'none' below containment
0.5), make_fig_augment.py (utility/summary.json, medians over seeds per background), make_fig_spectra.py (from the paper
sources; legend in percent). method, data and spectra read the run files under FINS_RESULTS_DIR, the others results/.
An independent check reproduced every plotted value (pixel-identical reruns) and checked the captions; its layout and
wording findings are applied (BO's reference two-stage drawn hollow with the caveat in the caption, H = 0.2 row added,
FINS drawn on top, t = 0 labels in the method figure). The check also found that Section 5.4 left BO seed 50 (KdV type
with an advection term, clean and at 1e-2) and seed 43 at 1e-2 (advection) unaccounted for; the text now covers them.

**Paper tables of exp71 and exp72 (13.10-14.10; no new runs).**
exp71_rows.py prints the Kreider rows of tab:main, tab:noise and tab:new-learned from the studies' summaries (equal to
the paper's). exp72_table.py prints tab:use (medians over seeds 42-44 of the 16-step errors, in percent) and the numbers
of its section from results/utility: the ratios behind UT1-UT7; the parts of FINS's spans in W lie within a principal
angle of 0.0031-0.0205 of the oracle's (missing 1.0e-5 to 4.2e-4) while the speed ratios of their boost elements span
0.854-1.121 over the horizon (BO 42 0.854-1.027, KdV 43 0.973-1.121; LieNLSD's on KS 1.003-1.007); evaluation errors
at most 2.01 (fins), 1.94 (oracle), 1.87 (lienlsd) and 2.10 (wide) times none's. Post hoc (--zero-mean, exp53's
zero-mean KS files): the boost element of FINS's part through the zero-mean steppers moves at 0.803-0.854 (42) and
0.756-0.908 (43) of the true speed over the horizon (principal cosine to the oracle's part 0.986 and 0.970); on 44 the
part in W is the translation alone. --spans on the development run's closed-loop span (fins_runs/utility_dev/
ks_mixed_n256_e150/spans_ks_s7_closedloop.npz): xi = 0.910 t - 4.9e-3 t^2, phi = 1 - 0.030 t + 1.1e-3 t^2 (phi drift
-0.17 over T = 20), speed 0.910 at t = 0, 0.859-1.005 over the horizon. Corrected below: UT7's KdV and BO ratios to
their range 1.000-1.001 (was the bound 0.995-1.005); the development entry's boost element (t^2 terms omitted, phi
drift -0.6 was the linear term alone).

**Does a symmetry FINS reports make an emulator generalise? (exp72) -- results (13.10; notebook 14 on a T4, seeds
42-44; 54 steppers, about 110 s each).**
Outputs copied unchanged to results/utility/runs (eval_<pde>_s<seed>.json, S_*.json, logs); summary and verdicts:
exp72_utility.py summary -> results/utility/summary.{md,json}, verdicts.txt. Every run as registered (512
trajectories, 300 epochs, S as fixed before the runs, parts as in the inputs table). An independent check evaluated
UT1-UT7 with its own code, recomputed S from its definition (within 1e-8), compared the copies with the originals and the
JSONs with the logs: no discrepancy.
16-step errors, range over seeds (ID: evaluation trajectories; OOD: mean over their exact boosts to backgrounds -2,
-1, 1, 2; BO's fins without seed 43, where it equals none up to GPU nondeterminism, 1e-6 relative):
| arm | KS ID | KS OOD | KdV ID | KdV OOD | BO ID | BO OOD |
|---|---|---|---|---|---|---|
| none | 2.0-3.0e-3 | 0.217-0.269 | 0.45-1.5e-3 | 0.011-0.019 | 1.0-2.8e-3 | 0.017-0.038 |
| fins | 4.0-5.0e-3 | 0.0093-0.0108 | 0.52-1.7e-3 | 0.0012-0.0071 | 1.2-1.8e-3 | 0.0047-0.0068 |
| oracle | 3.7-4.8e-3 | 0.0083-0.0096 | 0.45-1.5e-3 | 0.0009-0.0026 | 1.1-2.5e-3 | 0.0021-0.0054 |
| lienlsd | 3.6-4.9e-3 | 0.0117-0.0156 | as none | as none | as none | as none |
| shift | 2.8-3.4e-2 | 1.58-1.60 | 1.8-2.4e-3 | 0.099-0.100 | 3.2-4.5e-3 | 0.119-0.124 |
| wide | 4.3-5.3e-3 | 0.0091-0.0102 | 0.52-1.8e-3 | 0.0008-0.0029 | 1.1-2.6e-3 | 0.0019-0.0061 |
Verdicts: UT1 met (none's OOD 12.4-108 times its ID); UT2 met (oracle against wide 0.88-1.09); UT3 met 8/8 (fins
against none 0.040-0.046 on KS, 0.105-0.375 on KdV, 0.27 and 0.31 on BO 44 and 42); UT4 not met 5/6 (fins against the
oracle: KS 1.08-1.14, KdV 1.36 and 1.32 on 42 and 44, 2.71 on 43, the span whose speed ratio is 0.97-1.12 in the
inputs); UT5 met (shift against none 3.3-9.1); UT6 met (fins and oracle ID 0.88-2.01 times none's); UT7 not met 7/9
(KdV and BO: lienlsd against none 1.000-1.001; KS: lienlsd against the oracle 1.64, 1.59, 1.41 on 42, 43, 44).
Without predictions: BO's fins against the oracle 3.28 (42) and 2.04 (44); lienlsd differs from none by at most 5e-5
(KdV) and 5e-3 (BO) relative: translations alone do nothing for this translation-equivariant stepper.
Reading (as registered, UT3 met and UT4 not met): FINS's spans of the learned operators carry an emulator to
backgrounds two to four times outside its data, 22-25 times lower OOD error on KS, 3-10 on KdV, 3-4 on BO; within
1.08-1.36 times the true algebra (1.02-1.48 times real data) on KS and on KdV 42 and 44, and their error costs where
the span is less precise (KdV 43, BO). UT5 met: the gain needs the drift that comes with the background, i.e. the symmetry; more
backgrounds without it make the emulator 3-9 times worse out of distribution. UT7: on KdV and BO a span without the
boost gives nothing; on KS LieNLSD's span (speed ratio 1.004-1.007 in the inputs) lands 1.41-1.64 times above the
oracle where FINS's (0.994-1.002) lands 1.08-1.14 (post hoc: at backgrounds up to 2 on KS, where advection dominates
a step, half a percent in the speed shows; not tested).

**Kreider, Harlim and Huang as a baseline (exp71) -- results (13.10; seeds 42-51, 420 runs on the CPU).**
Runs: five equations at their studies' noise levels, ten seeds, two readings (fins_runs/run_exp71.py, two workers,
about six hours; every file carries the registered settings). Summaries regenerated (exp51 summary --pdes heat,kdv
--exp48; exp53_ks, exp52_bo, exp55_extra summary): only the Kreider rows are new (the other values within 1.1e-14,
summary.md otherwise unchanged). Verdicts: exp71_verdicts.py -> results/kreider_verdicts.txt. An independent check
rescored every file from the raw outputs (9070 values; at most 3.6e-15 from the summaries) and evaluated KR1-KR6 with
its own code: the same verdicts and numbers.
Clean, finite algebra, soft readout (m = k), mean +/- sd over seeds 42-51 (FINS learned: 42-44):
| equation | solutions | equation surface | FINS weak, exact | FINS weak, learned | LieNLSD as published (paper) |
|---|---|---|---|---|---|
| heat (6) | 3.624 +/- 0.311 | 1.712 +/- 0.242 | 0.000 | 0.010 | 0.55 |
| Burgers (5) | 3.525 +/- 0.217 | 1.311 +/- 0.203 | -- | -- | 0.55 |
| KdV (4) | 3.090 +/- 0.160 | 1.658 +/- 0.119 | 0.000 | 0.004 | 0.91 |
| BO (4) | 2.818 +/- 0.375 | 1.715 +/- 0.239 | 0.000 | 0.018 | 1.354 |
| KS (3) | 2.012 +/- 0.129 | 0.919 +/- 0.040 | 0.000 | 0.000 | 0.00 |
Equation surface under noise (mean / median): heat 2.65-2.88 / 2.59-2.91 from 1e-3 to 1e-1; KdV 1.50-1.94 /
1.45-1.94; BO 1.72-1.75 / 1.76; KS 0.63-1.12 / 0.51-0.99 (1e-2: median 0.507, five seeds below 0.5). Solutions: heat
4.33-5.02, KdV 2.81-3.49, BO 3.05-3.17, KS 2.29-2.59.
Verdicts: KR1 met (10/10 on every equation; minimum 1.845, KS); KR2 met (10/10; minimum 0.846, KS); KR3 met (10/10 on
heat, KdV, BO, KS; FINS weak at most 0.000 against at least 0.846); KR4 met (3/3; FINS learned at most 0.028 against
at least 0.875); KR5 met (medians 0.507-2.911; the narrowest margin KS at 1e-2, 0.007); KR6 met (surface means over
42-44 heat 1.614, Burgers 1.339, KdV 1.682, BO 1.747, KS 0.904 against 0.55, 0.55, 0.91, 1.354, 0.00; the main
table's heat 0.32 and KdV 0.90 change nothing).
Without predictions: hard readouts miss more (clean: solutions 2.30-5.55, surface 1.00-3.96, its own count reading
the translations alone on KdV and KS, 2.00 and 1.00 on every seed); in the non-trivial convention the surface misses
the same as in the finite one (it returns the translations exactly), the solutions reading 0.32 on KS against 1.16-2.36
on the others.
Reading (as registered): as published, the method returns the symmetries the sampled solutions share and answers a
different question than the equation's algebra; estimated locally from the authors' jets, the equation surface is
not accurate enough for the non-trivial generators at the protocol's resolution (0.9-1.7 dimensions missed clean,
more under noise on heat), where an operator (FINS: 0.000 exact, at most 0.028 learned) or a fitted model (LieNLSD
as published: 0.00-1.35) is needed.

**Does a symmetry FINS reports make an emulator generalise? (exp72) -- pre-registration (13.10; development on seed 7
only; before any run on seeds 42-44).**
Question (the reviewers' "what is a discovered generator for?"): used as data augmentation, does the span FINS reports
for a learned operator extend an emulator beyond the backgrounds of its training data, as the true algebra and real
data do? exp72_utility.py; KS, KdV, BO on the torus; seeds 42-44.
Design: every arm trains the same stepper (FNO, 64 modes, 4-step unrolled MSE, held-out increment error for the best
state; 512 trajectories, 300 epochs) on the protocol's 'mixed' data (constant backgrounds within [-0.5, 0.5]). Arms
'fins', 'oracle', 'lienlsd', 'shift' move each training window with probability 1/2 by a random unit element of a set
of fields of W at a parameter uniform in [-S, S] (S: the oracle's boost reaches background 3). W: the class's fields
xi(t) d_x + phi(t) d_u (free of x and u, tau = 0), which act on a periodic state at time t exactly,
u(x) -> u(x - s xi(t)) + s phi(t). 'fins': the part inside W of the span FINS weak reported for the learned operators
(the paper's rows, clean, own count): principal directions with cosine >= 0.99 in the scoring metric, projected onto
W; 'lienlsd': the same for LieNLSD as published (own count; KdV: exp41 with the noise study's settings at noise 0, as
for BO and KS; the paper's clean KdV row is exp48's, whose fields are not on disk); 'oracle': d_x and
t d_x + phi0 d_u; 'shift': d_x and d_u (a constant added without its drift, the naive augmentation). 'none': no
moves; 'wide': no moves, data with the same draws and backgrounds within [-3, 3] (the reference of real data; recipes
'narrow' and 'wide' scale the backgrounds of 'mixed', ks.SCALE; 'mixed' unchanged, initial conditions identical to the
committed code). Scores: one-step and 16-step errors relative to the state's deviation from its mean, over starts 8,
40, 72, 104 and the ten evaluation trajectories (data seed 1000 + seed, zero background: in distribution, ID) and over
their exact Galilean boosts to backgrounds -2, -1, 1, 2 (out of distribution, OOD: their mean). Notebook 14 (T4).
Inputs (exp72_utility.py spans -> results/utility/spans_<pde>_s<seed>.npz; the boost element of a part: its element
with xi(0) = 0 and phi(0) = phi0; speed ratio xi'(t) phi0 / phi(t), 1 for the boost):
| | KS 42 / 43 / 44 | KdV 42 / 43 / 44 | BO 42 / 43 / 44 |
|---|---|---|---|
| FINS: own count, part in W | 3, 2 / 3, 2 / 3, 2 | 4, 2 / 4, 2 / 4, 2 | 4, 2 / 1, 0 / 4, 2 |
| FINS: speed ratio at t = 0, T | 1.001-1.001 / 0.996-0.994 / 1.000-1.002 | 0.982-1.005 / 0.973-1.121 / 0.991-1.024 | 0.854-1.027 / -- / 0.928-1.002 |
| LieNLSD: own count, part in W | 3, 2 / 3, 2 / 3, 2 | 2, 1 / 4, 1 / 2, 1 | 4, 1 / 4, 1 / 4, 1 |
| LieNLSD: speed ratio at t = 0, T | 1.007-1.006 / 1.004-1.005 / 1.005-1.005 | -- | -- |
LieNLSD's parts on KdV and BO are the translation alone (second principal cosine with W 0.00-0.88). BO seed 43: FINS's
own count 1 leaves no part in W; its 'fins' arm trains as 'none'.
Development (seed 7, KS; 256 trajectories, 150 epochs; FINS run through each design's own 'none' stepper, a closed
loop):
- Base data with zero mean: FINS's boost through that stepper is slow (speed ratio 0.75; the protocol's zero-mean KS
  files 0.80 and 0.83 at t = 0, 0.80-0.85 and 0.76-0.91 over the horizon, seed 44 without a boost in W); the stepper's own drift on a background b is 0.86 b at
  b = 0.05. Arm 'fins' moved the OOD 16-step
  error at backgrounds +-0.25, +-0.5 from 0.123 to 0.101, the oracle to 0.0126: an operator that never saw a background
  cannot tell how a background moves the dynamics.
- Backgrounds within [-0.1, 0.1]: FINS's speed 0.82 (the stepper's own drift 0.95).
- Backgrounds within [-0.5, 0.5] ('mixed', the registered base): the stepper's own drift 1.003 b, FINS's boost
  element xi = 0.910 t - 4.9e-3 t^2, phi = 1 - 0.030 t + 1.1e-3 t^2 (corrected 14.10: t^2 terms were omitted): the
  readout of this stepper (256 x 150) is less precise than the stepper. The
  protocol's learned operators (1024 trajectories, 400 epochs) give the spans above (KS within 0.6 %): the arms use
  the paper's spans, not a closed loop.
- With the u-dependent fields of the class in W, the closed-loop span's flows (RK4 in the parameter) moved states
  13-26 % beyond a shift and a background (second principal cosine with that W 0.998; the small u-dependent parts grow
  with the parameter); restricted to fields free of u the moves are exact (the true torus-compatible algebra is free of
  u on all three equations).
| arm (seed 7, KS, 256 x 150) | ID one-step | ID 16-step | OOD one-step | OOD 16-step |
|---|---|---|---|---|
| none | 5.83e-04 | 7.42e-03 | 2.34e-02 | 0.328 |
| fins (closed-loop span: speed 0.91 at t = 0, 0.86-1.01 over T, phi drift -0.17 over T; corrected 14.10) | 1.18e-03 | 1.32e-02 | 4.83e-03 | 0.086 |
| oracle | 1.05e-03 | 1.28e-02 | 2.57e-03 | 0.026 |
| shift | 3.34e-03 | 5.16e-02 | 0.186 | 1.59 |
| wide | 1.24e-03 | 1.44e-02 | 2.36e-03 | 0.023 |
fins_runs/utility_dev (ks_n256_e150: zero mean; ks_narrow_n256_e150; ks_mixed_n256_e150: the registered design, the
oracle there moved by RK4 flows, 8e-5 from the exact moves).
Predictions (16-step errors; OOD the mean over the four backgrounds):
(UT1) none: OOD >= 3 x ID, on every equation and seed.
(UT2) oracle: OOD <= 1.5 x wide's OOD, on every equation and seed.
(UT3) fins: OOD <= 0.5 x none's OOD, on every equation and seed with a part in W (all but BO 43).
(UT4) fins: OOD <= 1.5 x oracle's OOD on KS and KdV, every seed.
(UT5) shift: OOD >= none's OOD, on every equation and seed.
(UT6) fins and oracle: ID <= 2.5 x none's ID, on every equation and seed.
(UT7) lienlsd: OOD <= 1.5 x oracle's on KS; OOD >= 0.8 x none's on KdV and BO (translation alone); every seed.
No prediction: one-step errors, the backgrounds one by one, fins against oracle on BO, run times.
Reading registered in advance:
- UT3 and UT4 met: the generators FINS reports for learned operators are accurate enough to use: as augmentation they
  carry an emulator to backgrounds 2-4 times outside its data as the true algebra and real data do.
- UT3 met, UT4 not met: the span helps but its error costs, as the closed-loop span in development.
- UT5 met: the gain needs the drift that comes with the background, i.e. the symmetry, not more backgrounds.
- UT7 met: a method whose span misses the boost gives no gain; the comparison of discovery methods carries over to use.
Fix before the runs (13.10, independent review of the code, no registered run yet): augment_scale read S off one
column of an arbitrary orthonormal basis of the oracle's span, so S depended on rounding (KS 1.61 and 2.10 on seeds
42 and 44, 1.54-1.78 under 1e-13 perturbations of the metric). Now S = 3 ||b||_G / phi0 for the oracle boost b, as
registered (the G-normalised boost reaches 3; KS 1.71-1.74, KdV 0.63-0.64, BO 0.37; the largest background over unit
elements 3.37 on KS, 3.01 on KdV and BO). Also cached: the grid (KdV's comes from a solve); guarded: a run whose every
validation is NaN keeps its last state. The development runs used the first version (seed 7: S 1.77 and 1.67).

**Kreider, Harlim and Huang as a baseline (exp71) -- pre-registration (13.10; development on seed 7 only; before any
run on seeds 42-51).**
Method: Kreider, Harlim, Huang, "A model-free method for discovering symmetry in differential equations" (arXiv
2511.09779; code github.com/MaxKreider/SymmetryAndScaling, MATLAB, read and not run). Reimplementation
pde/method_kreider.py: per point, K nearest neighbours, local SVD for the tangent frame of the solution's graph, J GMLS
steps of degree ell, derivatives of the next jet level from the refined tangents, repeated to the equation's order;
normals of the prolonged graph; rows S . pr v for FINS's polynomial class (prolongation by sympy total derivatives);
A = sum p'p; SVD. Checks (kreider_check.py, 8/8): rows equal the authors' local_row of Fig_3c.m and Fig_3d.m to 1e-16;
a literal per-point port of their loops agrees to 4e-15 (jets) and 3e-13 (normals). Reproduction (kreider_reproduce.py,
100 trials): Fig 3c errors 0.589, 0.226, 0.056, 0.030 and Fig 3d 0.034, 0.012, 0.0046, 0.0017 at N^2 = 3136-25600
(slopes -1.48 and -1.44 against the theory's -0.50 and -1.00; their figure's values could not be read off). Fig 3c's
affine null space has 4 dimensions (f (d_t - d_x), f in {1, t, x, u}): the method returns symmetries of the sampled
solution. fins_runs/kreider_dev/validation/.
exp71_kreider.py, two readings of the same estimator on the protocol's evaluation data (FINS's normalised (X, T, U);
periodic copies close the x-period; rows from snapshots 8 .. Nt - 8; columns of A scaled by their rms, exact null
columns kept at a typical scale; soft readout k smallest, hard readout e49.own_count; C format, tags kreider-sheet and
kreider-surface, scored by exp51.score_cond like the two-stage files):
- solutions (the authors' setting): per trajectory, frames of its graph and of its prolongations, rows from every
  normal, accumulated over the ten trajectories: the symmetries shared by the sampled solutions;
- equation surface: the jets of all ten trajectories pooled in (u, u_x, ..., u_{x^p}, u_t) (autonomous, the input
  LieNLSD's network gets), codimension-1 normals of that cloud by the same SVD + GMLS (method_kreider.pooled_normals),
  rows n . pr v: the symmetries of the locally estimated equation.
Two changes for gridded data, both needed in development: neighbours on the (t, x) grid (a search in (t, x, u, ...)
picks points along the flat direction of steep data and the 2x2 solve became singular at 93 of 8192 heat points), and
every level's coordinates scaled by their rms increment to the four nearest grid neighbours (global standardisation
left u_xx 170 % wrong on heat; with it 2.5 %; KdV u_x, u_xx, u_xxx 2.6 %, 8.9 %, 26 % at K 9, ell 2).
Development (seed 7, clean; missing dimensions of the finite algebra at m = k):
| reading, setting | heat | Burgers | KdV | BO | KS |
|---|---|---|---|---|---|
| solutions, K 9, ell 2 | 2.615 | 3.866 | 3.164 | 2.564 | 2.049 |
| solutions, K 20, ell 3 | 2.868 | 3.798 | 3.310 | -- | -- |
| surface, normals K 40, ell 1 | 1.396 | 1.434 | 1.707 | 1.485 | 0.888 |
| surface, normals K 80, ell 2 | 1.349 | 1.355 | 1.740 | 1.965 | 0.867 |
| surface, normals K 200, ell 1 | 1.093 | 1.422 | 1.067 | 1.847 | 0.931 |
| surface, normals K 200, ell 2 | 0.910 | 1.367 | 1.288 | 1.828 | 0.853 |
- The surface reading returns the translations exactly (their prolongation vanishes on the autonomous jet) and misses
  the non-trivial generators by about one dimension. Its normal is right to |cos| 0.9997 (median) on heat but the
  equation's u_xx component is 6e-4 of it, below the error; the jets' heat residual u_t - a u_xx is 4.4 %.
- Pooling 32 of the operators' training trajectories instead (KdV) gave 1.393 against 1.067: more trajectories do not
  help; 128 exceeded the memory.
Registered settings: jets K 9, ell 2, J 2 for both readings; surface normals K 200, ell 2; order as LieNLSD's jets (heat
and Burgers 2, KdV and BO 3, KS 4); class degree 3 (heat) or 2.
Runs: seeds 42-51, both readings, the noise studies' levels: heat and KdV clean, 1e-3, 1e-2, 3e-2, 5e-2, 1e-1
(noise_weak_runs); KS clean, 1e-2, 3e-2, 5e-2, 1e-1 (ks_runs); BO clean, 1e-2, 3e-2 (bo_runs); Burgers clean
(burgers_runs). Predictions (missing dimensions of the finite algebra, soft readout):
(KR1) Solutions reading, clean: >= 1.5 on every equation, on >= 9/10 seeds.
(KR2) Surface reading, clean: >= 0.5 on every equation, on >= 9/10 seeds.
(KR3) FINS weak through the exact operator (the files already on disk) misses fewer than the surface reading, clean,
on heat, KdV, BO and KS, on every seed.
(KR4) FINS weak through the learned operators (seeds 42-44) misses fewer than the surface reading, clean, on heat, KdV,
BO and KS, on every seed.
(KR5) Surface reading: the median over seeds >= 0.5 at every noise level on every equation.
(KR6) Surface reading, clean, mean over seeds 42-44: more than LieNLSD as published (the paper's values) on heat
(finite), Burgers, KdV, BO and KS.
No prediction: the own count rows, the non-trivial and full conventions, run times.
Reading registered in advance:
- KR1 met: as published, the method returns the symmetries the sampled solutions share; on diverse data it answers a
  different question than the equation's algebra.
- KR2 met: the equation surface estimated locally from the authors' jets is not accurate enough for the non-trivial
  generators at the protocol's resolution; a fitted model (LieNLSD) or an operator (FINS) is needed.
- KR2 not met on an equation: the local estimate suffices there; the paper's comparison must say so.

**PDE paper: the hidden-variables section and mock reviews (12.10; no new registered runs).**
- Written into paper_pde as Section 6 "A variable without a closed equation" (Table 7, Fig. 2), Appendix E and Tables
  18-19 (TR, UF, HL, LT, EP verdicts), with sentences in the abstract, contribution (v), related work and discussion.
  Figure and numbers: paper_pde/figures/make_fig_hidden.py, from results/tracer/summary.json,
  results/tracer/exp70_summary.json and results/pyqg/pairs_summary.json. An independent check found 9 errors and 20
  imprecise wordings, all fixed; four mock reviewers then read the build.
- Post hoc, from the committed summaries (reviewers' re-analyses, reproduced by us):
  - Rotation defect against the 8-step error (rollouts from the first snapshots of the held-out trajectories): the 18
    independent models (9 flow emulators, 3 (q, c) data arms, 3 latent data arms, 3 window data arms) lie within
    0.30-2.2 times 0.35 e_K^2, against 0.73-201 with the one-step error; e_K / e_1 is 0.77-1.02 for the full-state
    emulators, 2.3-3.0 for the latent models, 4.0-17 for the windows; the pooled log-log exponent against e_K is 1.2. The
    c-only models' excess over 0.35 e_1^2 is an effect of the one-step abscissa.
  - All nine f-plane emulators (the exp68 entry's "d_x 0.06-0.51" was on eight): d_x 0.06-0.50, d_y 0.07-0.99, D_t
    0.84-6.1, D_x 1.6-2.3 times 0.35 e^2. The (q, c) emulators: the largest observed quotient is 1.7-3.1 times the
    rotation defect; the ratio with q's error is 0.56-0.85.
  - Latent model: the separation divided by the over-breaking is 3.3-7.0 on the data (3.1-7.8 own rollouts); with z
    unweighted the boosts sit at 0.78-1.18 of the largest observed quotient (c d_c broken without z's weight).
  - Two-stage on c alone: weak residual 0.626-0.684 on every seed; the rotation is lost through a != b
    ([a d_xx + b d_yy, x d_y - y d_x] = 2(a - b) d_xy).
- Reported by the mock reviewers, not reproduced in the repository: weak SINDy on (q, c) with the transport columns
  d_x(u c), d_y(v c) (u, v from the Biot-Savart inversion of q) returns the c-equation exactly on 10/10 seeds (-1, -1,
  0.05, 0.05; residual 1.2-1.6e-6); a rotation-invariant library on c alone returns kappa-hat Lap c on 10/10 (kappa-hat
  0.041-0.049); the local (psi, c) system has 12 projectable point symmetries in the degree-2 class plus 4 vertical
  (the seven, t d_x - y d_psi, t^2 d_x - 2ty d_psi, t d_y + x d_psi, t^2 d_y + 2tx d_psi and a rotating frame; sympy);
  a zero-parameter cubic extrapolation through the window emulators' four frames has one-step error 0.010 %.
- Reviewers' scores 4 (area chair), 4.25 (mathematician), 4.5 (advocate), 5 (GFD and emulators); all four recommend a
  separate paper for Section 6 with at most two pages kept here.

**Learned operators in place of the simulator, step 2 -- a model of c alone with a learned hidden field, results (exp70;
notebook 13, Colab; 12.10; LT0-LT5 on seeds 42-44).**
Runs: fins_nb13.zip, extracted into its own new directory (fins_runs/in_fins_nb13) and merged into fins_runs/tracer70_runs
(logs in logs_nb13); results/tracer/exp70_summary.{json,md}; verdicts_0710.py. Every arm in float64 on the GPU, with the
plane readout.
Models (c only: 1024 trajectories, 10000 steps of 32, 40 modes, the loss on c over four steps; 107 min each): held-out
one-step error of c 0.0216, 0.0171, 0.0132 (8 steps 0.066, 0.042, 0.030); rotation commutator at 30 degrees 7.9e-3,
5.2e-3, 5.0e-3. Own rollouts stay within 0.030-0.038 of the data (c, 47 steps).
| seed | arm | observed algebra and S, largest (plane) | c d_c (z weight) | boosts | boosts / largest | boosts / the oracle's | boosts / the window emulator's | superpositions x, y, xy, x^2 - y^2, r^2 | smallest heat-false / largest |
|---|---|---|---|---|---|---|---|---|---|
| 42 | data | 1.5e-3 (d_t) | 1.3e-4 (+0.99) | 2.9e-2, 3.2e-2 | 19.7, 21.8 | 2.82, 3.73 | 282, 480 | 1.2e-3, 1.3e-3, 1.2e-3, 1.3e-3, 3.2e-3 | 0.78 |
| 43 | data | 1.2e-3 (d_t) | 9.7e-5 (+1.02) | 2.6e-2, 2.4e-2 | 21.0, 19.3 | 3.12, 3.44 | 515, 757 | 1.8e-4, 1.9e-4, 1.1e-4, 2.0e-4, 1.3e-3 | 0.09 |
| 44 | data | 2.9e-3 (S) | 1.2e-4 (+1.01) | 2.7e-2, 3.7e-2 | 9.4, 12.9 | 2.86, 2.96 | 96, 90 | 1.8e-3, 1.9e-3, 8.0e-4, 2.5e-3, 4.8e-3 | 0.28 |
| 42 | self | 1.3e-3 (S) | 9.1e-5 (+0.99) | 3.1e-2, 3.4e-2 | 23.2, 25.7 | 2.97, 3.93 | 297, 506 | 1.1e-3, 1.2e-3, 1.2e-3, 1.3e-3, 3.0e-3 | 0.81 |
| 43 | self | 1.3e-3 (S) | 6.0e-5 (+1.02) | 2.8e-2, 2.5e-2 | 21.4, 19.1 | 3.38, 3.62 | 558, 796 | 1.8e-4, 2.0e-4, 1.1e-4, 2.0e-4, 1.3e-3 | 0.09 |
| 44 | self | 3.1e-3 (S) | 8.8e-5 (+1.01) | 2.8e-2, 4.0e-2 | 9.2, 13.0 | 2.98, 3.15 | 100, 96 | 1.9e-3, 2.0e-3, 8.8e-4, 2.7e-3, 5.0e-3 | 0.29 |
- LT0 NOT MET (2/3): seed 42's model has 2.16 %.
- LT1 MET (2/3): on 42 and 43 both boosts stand 19.3-21.8 times above the observed algebra and S; on 44 9.4 and 12.9.
- LT2 MET (3/3): the boosts are 90-757 times the window emulators'. They are 2.8-3.7 times the oracle's: the model
  over-breaks them.
- LT3 MET (3/3): the smallest heat-false quotient is 0.09-0.78 of the largest. The superpositions x, y, xy, x^2 - y^2
  are 0.20-0.28 of the oracle's on 43 (kept, as by the window emulators) and 1.2-3.5 times it on 42 and 44; x^2 + y^2 +
  4Dt is inflated (6.7-21.6 times; data and self).
- LT4 MET (2/3): own rollouts within 0.038 of the data (3/3); both boosts at least 10 times the largest on 42 (23.2,
  25.7) and 43 (21.4, 19.1); 44 9.2 and 13.0.
- LT5 MET (3/3): a free perturbation of z brings every heat-false generator to at most 0.036-0.064 of the largest of
  the observed algebra and S (the seven fall to 3.1e-5 - 3.3e-4 of the scalar lift; d_x, d_y, d_t, the rotation and S
  to 0.009-0.20 of theirs; d_c and c d_c, whose scalar lift leaves z unperturbed, to 1.0e-4 - 3.8e-4).
- No prediction: c d_c holds with z's weight +0.99 to +1.02 (with the scalar lift it is broken, 2.8e-2 to 3.7e-2); S's z
  weight -1.84 to -2.06 (data) and -1.81 to -2.06 (self), no meaning claimed.
- Post hoc: the rotation plane (EP3's readout) is 2.4-10.1 times 0.35 e^2 (the state emulators 0.65-1.38, the window
  emulators' data arm 7-200, exp68's emulators 0.90-1.43). The span of the pencil's 7 lowest directions misses 2.0-4.0
  of the observed algebra with S (S on every run, d_t on the data arm, the rotation on the self arm).
Verdicts (exp70 step 2): LT0 not met; LT1-LT5 met.
Reading (registered): LT1-LT2 met: from data of c alone, with the hidden field lifted as a point field (the observed
algebra's own definition), FINS separates the heat closure's boosts that window emulators keep: for the boosts the
simulator is not needed (the breaking is detected on 2/3 seeds; its size is over-estimated 2.8-3.9 times). LT3 met: the
superpositions need a lower floor (as in step 1). LT5 met: a free perturbation of a
learned hidden field is not a test of symmetry; it absorbs generators that the true hidden state cannot. The readout must
keep the lift's point structure.

**Learned operators in place of the simulator, step 1 -- the state emulator, results (exp70; notebook 12,
Colab; 11.10, night; HL0-HL5 on seeds 42-44).**
Runs: fins_nb12.zip, extracted into its own new directory (fins_runs/in_fins_nb12) and merged into fins_runs/tracer70_runs
(logs in logs_nb12); results/tracer/exp70_summary.{json,md}; verdicts_0710.py. The notebook ran step 1's code (before the
plane readout); every arm in float64 on the GPU.
Emulators (1024 trajectories with q, 10000 steps of 32, 40 modes; 49 min each on a T4): held-out one-step error of c
0.0509, 0.0358, 0.0306 and of q 0.0550, 0.0390, 0.0390 (8 steps: c 0.039, 0.030, 0.024; q 0.056, 0.045, 0.046);
rotation commutator at 30 degrees (c) 0.049, 0.039, 0.045. Own rollouts stay within 0.004-0.006 of the data (c, 50
snapshots).
| seed | arm | observed algebra and S, largest | S's hidden lift | boosts (/ the oracle's) | boosts / largest | superpositions x, y, xy, x^2 - y^2, r^2 | smallest heat-false / largest |
|---|---|---|---|---|---|---|---|
| 42 | data | 1.3e-3 (d_t) | -2.002 | 1.0e-2, 8.7e-3 (0.97, 1.01) | 7.5, 6.5 | 2.7e-3, 4.4e-3, 8.4e-3, 2.0e-3, 3.9e-3 | 1.48 |
| 43 | data | 1.2e-3 (d_t) | -2.030 | 8.4e-3, 7.1e-3 (1.02, 1.03) | 7.2, 6.1 | 4.5e-3, 3.0e-3, 8.3e-3, 2.5e-3, 1.1e-2 | 2.18 |
| 44 | data | 1.1e-3 (d_t) | -2.065 | 9.4e-3, 1.2e-2 (0.99, 0.98) | 8.9, 11.6 | 3.3e-3, 4.0e-3, 8.6e-3, 2.1e-3, 1.3e-2 | 2.00 |
| 42 | self | 1.0e-3 (S) | -2.004 | 1.1e-2, 9.1e-3 (1.03, 1.06) | 10.6, 9.1 | 2.9e-3, 4.6e-3, 8.9e-3, 2.1e-3, 4.0e-3 | 2.07 |
| 43 | self | 8.2e-4 (d_c) | -2.059 | 8.8e-3, 7.3e-3 (1.07, 1.07) | 10.8, 8.9 | 4.7e-3, 3.2e-3, 8.8e-3, 2.7e-3, 1.1e-2 | 3.26 |
| 44 | self | 9.7e-4 (d_c) | -2.068 | 9.7e-3, 1.3e-2 (1.03, 1.02) | 10.1, 13.3 | 3.5e-3, 4.2e-3, 9.0e-3, 2.2e-3, 1.4e-2 | 2.29 |
- HL0 NOT MET (2/3): seed 42's emulator has 5.1 % (c) and 5.5 % (q).
- HL1 NOT MET (0/3): on the data arm the boosts sit 6.1-11.6 times above the observed algebra and S (44's boost_y 11.6,
  the rest 6.1-8.9).
- HL2 MET (3/3): S's hidden lift -2.002, -2.030, -2.065 (self -2.004, -2.059, -2.068).
- HL3 MET: the boosts are 0.97-1.03 times the oracle's (3/3) and 30-226 times the window emulators' (3/3).
- HL4 MET (3/3): the smallest heat-false quotient (x^2 - y^2 on every seed) is 1.48-2.18 times the largest of the
  observed algebra and S.
- HL5 NOT MET (1/3): own rollouts within 0.006 of the data on 3/3; both boosts at least 10 times the largest on 44 only
  (42: 10.6, 9.1; 43: 10.8, 8.9; 44: 10.1, 13.3).
- Lift arm (no prediction): through the emulator's tangent the best hidden perturbation lowers every quotient two to
  three orders: the seven to 8.0e-4 - 1.9e-2 of the scalar lift (6.2e-6 to 2.2e-4), the observed algebra and S to
  1.4e-3 - 9.5e-3 of theirs (at most 7.3e-6). The seven stay 1.75-2.30 times above the observed algebra and S (1.81-2.18
  at the scalar lift). The exact dynamics keeps 0.14-0.52 of the seven with the observed algebra and S at most 7.2e-12
  (TR8). Unlike the latent model in step 2's development, the free hidden perturbation keeps the order here.
- Post hoc: the superpositions' quotients are inflated over the oracle's (xy 17-25 times, x^2 + y^2 + 4Dt 21-59 times);
  the oracle's superpositions sit at 0.14-1.4 times the emulator's floor, so the floor hides them.
- Post hoc (not registered): the span of the pencil's 7 lowest directions misses 0.023-0.064 of the observed algebra
  with S on the self arm (3/3) and 0.045 and 0.074 on the data arm's 44 and 43 (42: 1.003); exp69's oracle 0.000, its
  window emulators 3.1-4.2. The gap above the 7th eigenvalue is small (lambda_8 / lambda_7 1.29-3.56 self, 1.06-1.69 data;
  the oracle 4300-7600), so the own count stays 1-2.
Verdicts (exp70 step 1): UF1 met, UF2 met; HL0 not met; HL1 not met; HL2 met; HL3 met; HL4 met; HL5 not met.
Reading (registered): HL3 met with HL1 not: the breaking is learned and the accuracy floor hides it; separation needs a
lower error. HL4 met: the superpositions need a lower floor than one-step errors of a few per cent give (EP3's
0.35 e^2). In numbers: with
q observed in training, a learned operator reproduces the simulator's breaking of the boosts (0.97-1.07 of it) and S's
hidden weight (-2.00 to -2.07) and holds the observed algebra in its lowest seven directions on its own rollouts, with the
boosts 6-13 times above the observed algebra and S (the simulator: 8e4-3e5 times).

**Learned operators in place of the simulator for a tracer in a hidden flow, step 2: a model of c alone with a learned
hidden field (exp70, exp70_learned_hidden.py; notebook 13) -- development and pre-registration (11.10, evening; written
after the development runs below, before any run on seeds 42-44).**
Question. Step 1 learns the full state with q observed in training. Here only c is observed, as for exp69's window
emulators: can a model with a learned hidden field replace the simulator where the windows could not?
Model. LatentModel: z_j = E(c_{j-3}, ..., c_j) (exp69's WindowFNO2d body with one output channel: a hidden field with no
target), then (z, c)_{k+1} = G((z, c)_k) (StateFNO2d with z in q's place, z's increment scale 0.01). Trained on c alone,
on exp69's window emulators' training set (1024 trajectories of data seed 6000 + s), with the loss on c over four steps
of G from the encoded window (over c's increment scale); 10000 steps of 32, 40 modes.
Arms (FINS through G's tangent, z in q's place; exp69's 61 fields with z d_z as the 61st):
- data: c from the data, z from the encoder on each window ending at the snapshot (pairs from snapshot 5).
- self: the model's own rollouts of (z, c) from the data's first four snapshots (pairs from snapshot 6).
- lift: the best perturbation of z at t0 (exp69's best hidden lift through G's tangent: CGLS from the scalar lift, 4
  pairs, 100 iterations, TR8's pairs).
Readouts: exp69's, and the plane readout (exp70 `plane_readouts`): each candidate with its best z d_z weight, the
quotient's minimum over the plane (candidate with the scalar lift, z d_z). It is exp69's S_lift for every candidate:
q's weights are known (0, and -2 for S), a learned field's are not. The registered quotients of the data and self arms
are the plane readout's.
Development (seed 7, 64^2 on CPU: 512 trajectories, 3000 steps of 16, 20 modes; step 1's state emulator and exp69's
references in the step 1 entry):
| arm (64^2) | one-step error (c) | observed algebra and S, largest (plane) | c d_c: scalar lift -> plane (z weight) | boosts x, y (plane) | superpositions x, y, xy, x^2 - y^2, r^2 (plane) |
|---|---|---|---|---|---|
| latent, data | 0.0267 | 4.5e-3 (d_t; rotation 4.5e-3) | 4.4e-2 -> 1.9e-4 (+0.97) | 4.1e-2, 5.4e-2 | 1.0e-3, 1.2e-3, 1.1e-3, 1.5e-3, 4.9e-3 |
| latent, self | 0.0267 | 4.4e-3 (rotation) | 4.2e-2 -> 4.9e-5 (+0.98) | 3.8e-2, 4.8e-2 | 6.7e-4, 8.0e-4, 7.2e-4, 1.0e-3, 3.6e-3 |
| latent, lift (best perturbation of z) | 0.0267 | 6.9e-4 (d_t; next d_x 3.2e-4) | 6.2e-2 -> 2.7e-5 (best lift) | 2.8e-5, 2.5e-5 | 3.0e-8, 4.0e-8, 4.5e-8, 4.6e-8, 1.3e-7 |
- Lifted as a point field (a scalar with a fitted weight), the learned z makes the model break the heat closure's boosts
  9.1-12.0 times (data) and 8.5-10.8 times (self) above its observed algebra, where exp69's window emulator kept them
  below its floor (1.5e-4, 2.9e-4 against 3.3e-3); 2.0-4.4 times the 64^2 oracle's. The superpositions x, y, xy, x^2 - y^2
  sit below the floor (0.15-0.33 of it), not separated from it, as in step 1 (there 0.46-1.36 of it). With the scalar
  lift instead, c d_c (4.4e-2, 4.2e-2) is the largest observed quotient and the boosts are 0.9-1.3 times it: the
  separation needs the fitted weight.
- The learned z scales with c: c d_c holds only with z's weight +0.97 (+0.98), and the scaling space's deepest direction
  is c d_c + 0.92 z d_z + 0.05 t d_t (4.7e-5, self; data c d_c + 0.75 z d_z + 0.20 t d_t, 1.3e-4). S's z weight -1.82
  (data), -1.94 (self); no meaning is claimed for it.
- A free perturbation of z absorbs the heat closure's seven: they fall to 1.6e-5 to 3.9e-4 of the scalar lift (boosts
  2.5e-5 to 2.8e-5, superpositions 3e-8 to 1.3e-7), below the largest of the observed algebra and S (d_t 6.9e-4; d_c
  and c d_c fall as far, to 4.4e-9 and 2.7e-5). The exact dynamics' hidden perturbation keeps 0.14-0.52 of them (TR8):
  the learned hidden space admits perturbations that the true hidden state does not.
- Own rollouts stay bounded (c within 0.057 of the data over 47 steps); 8-step error 0.077; rotation commutator at 30
  degrees 1.9e-2.
- On the development seed LT0, LT1 and LT4 would not be met (one-step error 0.027; boost_x 9.1 and 8.5 times the
  largest), LT2, LT3 and LT5 would. The predictions assume the registered budget lowers the floor: 6.7 times as many
  samples at 128^2 with 40 modes (exp69's window emulators went from 1.85 % in development to 0.35-0.92 %).
Predictions (seeds 42-44, 128^2, notebook 13):
(LT0) Every latent model's held-out one-step error of c <= 0.02.
(LT1) Data arm (plane readout): both boosts >= 10 times the largest of the observed algebra and S, on >= 2/3.
(LT2) Data arm (plane readout): both boosts >= 10 times exp69's window emulator's data arm (same seed), on 3/3.
(LT3) Data arm (plane readout): the seven not all separated (the smallest heat-false < 10 times the largest of the
observed algebra and S), on >= 2/3.
(LT4) Self arm (plane readout): c's own rollouts within 0.1 of the data (relative) and LT1's separation, on >= 2/3.
(LT5) Lift arm: every heat-false generator's best lift <= the largest best lift of the observed algebra and S (the free
perturbation of z absorbs them), on >= 2/3.
No prediction: S's and c d_c's z weights, the scalar-lift quotients, the rotation commutators, the 8-step errors.
Reading registered in advance:
- LT1-LT2 met: from data of c alone, with the hidden field lifted as a point field (the observed algebra's own
  definition), FINS separates the heat closure's boosts that window emulators keep: for the boosts the simulator is not
  needed.
- LT2 met with LT1 not: the learned field breaks the boosts and the error floor hides the separation.
- LT2 not met: the learned field acts as a window.
- LT3 met: the superpositions need a lower floor (as in step 1).
- LT5 met: a free perturbation of a learned hidden field is not a test of symmetry; it absorbs generators that the true
  hidden state cannot. The readout must keep the lift's point structure.

**Notebook 10 rerun: f rot s44 (exp68) -- EP3 and EP4 (11.10, evening, Colab; 36 of 36 audits).**
Run: fins_nb10_2.zip, extracted into its own new directory (fins_runs/in_fins_nb10_2) and merged into
fins_runs/pairs_runs (logs in logs_nb10_2); results/pyqg/pairs_summary.*; verdicts_0710.py. The notebook skipped the 11
emulators with all three audits and trained f rot s44 again (loss at step 10000 1.660e-3, as in the first run, whose
checkpoint was lost); the rerun's copies of the 33 earlier json files are identical to the merged ones.
f rot s44: held-out one-step error 0.0189; 30-degree commutator 2.51e-2; rotation plane 1.6e-4 (1.27 times 0.35 e^2).
- EP3 MET: rotation plane / 0.35 e^2 within 0.90-1.43 on all nine f-plane emulators (plain 1.14-1.43, grid 0.90-1.15,
  rot 1.19-1.35).
- EP4 MET: rot against plain lowers the 30-degree commutator 1.92, 1.95 and 1.88 times and the rotation plane 3.95,
  3.87 and 2.90 times (3/3); the plane ratio over the squared error ratio is 1.19, 0.98 and 0.90 (3/3 within 1.5).
Verdicts (all): EP0 not met; EP1-EP5 met.
Reading (registered): EP3 met: an FNO emulator's rotation defect follows its one-step error; read on the rotation, the
audit measures accuracy rather than a separately learned symmetry. EP4 met: rotation augmentation lowers the rotation
defect through accuracy; the audit and the test error agree.

**Learned operators in place of the simulator, step 1 -- the uniform flow, results (exp70; 11.10, evening; UF1-UF2 on
seeds 42-44; HL0-HL5 wait for notebook 12).**
Runs: fins_runs/tracer70_runs (ex70_observe_s42-44; 4 pairs, 100 CGLS iterations, exp69's TR8 pairs; logs in logs/);
results/tracer/exp70_summary.{json,md}; verdicts_0710.py. The scalar-lift quotients reproduce TR8's within 1.4e-13
(relative 7.2e-13 for every quotient above 1e-6).
| seed | gal_x, gal_y with (dq0, flow) | boosts: TR8 (dq0) -> with (dq0, flow) | superpositions x, y, xy, x^2 - y^2, r^2: TR8 -> with (dq0, flow) |
|---|---|---|---|
| 42 | 3.3e-18, 4.6e-18 | 3.6e-3, 3.4e-3 -> 3.6e-3, 3.4e-3 | 1.3e-4, 1.9e-4, 4.2e-5, 9.9e-5, 3.2e-5 -> 1.2e-4, 1.9e-4, 4.2e-5, 9.7e-5, 3.2e-5 |
| 43 | 7.3e-18, 2.8e-18 | 2.4e-2, 4.8e-3 -> 2.4e-2, 4.8e-3 | 4.8e-4, 3.5e-4, 2.0e-4, 5.5e-4, 8.2e-5 -> 4.7e-4, 3.4e-4, 2.0e-4, 5.4e-4, 7.9e-5 |
| 44 | 2.9e-17, 3.5e-18 | 5.6e-3, 3.7e-3 -> 5.6e-3, 3.7e-3 | 3.9e-4, 2.4e-4, 2.1e-4, 3.0e-4, 8.0e-5 -> 3.8e-4, 2.4e-4, 2.1e-4, 3.0e-4, 7.9e-5 |
- UF1 MET (3/3): with the flow gal_x and gal_y fall from 0.72-0.95 to at most 1.0e-16 at the scalar lift; their flow
  coefficients are 1 within 2.9e-9 on all 12 pairs.
- UF2 MET (3/3): with the flow every heat-false generator keeps 0.965-1.000 of TR8's value. The flow alone at the scalar
  lift removes 2-14 % of the boosts' quotients; with dq0 the seven keep 0.14-0.52 of the scalar lift, as in TR8. The
  observed algebra and S stay at most 6.7e-12.
Reading (registered): UF2 met: the breaking that c carries is not the hidden state's missing mean flow; it lies in the
flow's structure, so a model of c's data is not excused by a global constraint. (UF2 rules out a uniform mean flow only.)
- Corrections to the registration entry's development text (independent check; predictions unchanged): the Question's
  0.005-0.033 is the window emulators' data arm (TR6), not TR7's self arm; the uniform flow on seed 7 reads gal_x
  0.95 -> 2.7e-16 -> 6.6e-17 and gal_y 0.94 -> 3.9e-17 -> 3.6e-18, the boosts with the best flow 2.0e-2, 3.3e-2;
  exp69's development pairs share two of the four with exp70's; that x^2 + y^2 + 4Dt is inflated because the ramps
  leave the training distribution is a guess, not tested. The checks' numbers are saved in
  fins_runs/tracer_dev/check_n64.log.

**Learned operators in place of the simulator for a tracer in a hidden flow, step 1 (exp70, exp70_learned_hidden.py;
notebook 12) -- development and pre-registration (11.10, day; the question whether FINS needs a simulator for a
variable without a closed equation; written after the development runs below, before any run on seeds 42-44).**
Question. Through a simulator of the full system FINS reads the tracer's observed algebra (exp69, TR1-TR3); through
window emulators of c it does not (TR7: they keep the heat closure's boosts at 0.005-0.033 of the flow's breaking,
although c carries it, TR8). Can a learned operator replace the simulator, and is the windows' failure the missing
hidden state?
System, data, the 61 lifted fields, candidates and readouts: exp69's. References on the same seeds: exp69's oracle, its
best hidden lift (TR8) and its window emulators' data arm.
Arms:
- uniform flow (the exact dynamics; CPU): exp69's best hidden lift with a uniform background flow (dU, dV) added to
  the hidden perturbation, its c-response projected out exactly in CGLS. The Galilean t d_x with the flow changed by its
  speed is an exact symmetry of the full system; exp69's hidden state has no mean flow (periodic inversion), so there it
  is broken by a constraint on the hidden state, not by c's dynamics. The heat closure's boosts 2Dt d_x - x c d_c share
  only their small t d_x part with it (2DT / Lx = 0.031 of a unit Galilean boost in the scaled units). Asks whether the
  breaking that c carries (TR8) comes from that constraint.
- state emulator (data, q observed in training only): StateFNO2d, x_{j+1} = x_j + inc P(FNO((x_j - mean) / std)) for
  x = (q, c) (exp45's FNO2d with both fields as channels and per-field scales, no positional input), exp69's window
  recipe (one-step plus two-step loss, each field over its increment scale; 1024 trajectories of data seed 6000 + s with
  q kept, 10000 steps of 32, 40 modes). FINS through its tangent with the oracle's lift (the 60 fields in c lifted to q as
  scalars, q d_q, c compared; S's hidden lift read as in the oracle): data (the data's (q, c)), self (its own rollouts
  from the data's first state; pairs from snapshot 3), lift (exp69's best hidden lift through its tangent: 4 pairs, 100
  CGLS iterations, adjoint by autograd).
Checks (`check`): the uniform-flow tangent against central differences 3.9e-9; with a unit flow the Galilean
perturbation -t (q_x, c_x) maps to itself at the later time within 3.6e-6 (q) and 3.8e-7 (c) over three snapshots
(RK4's error); the emulator's tangent against central differences 7e-12; the state pencil with the exact step in the
emulator's place reproduces exp69's oracle terms exactly; the lift arm with the exact step reproduces exp69's best
hidden lift within 1.3e-12 (one pair, ten iterations).
Development (seed 7; emulators at 64^2 on CPU: 512 trajectories, 3000 steps of 16, 20 modes; references exp69's
64^2 oracle and window emulator; the uniform flow at 128^2):
| arm (64^2) | one-step error (c; q) | own count | observed algebra, largest | S (hidden lift) | boosts x, y | superpositions x, y, xy, x^2 - y^2, r^2 | H lowest |
|---|---|---|---|---|---|---|---|
| oracle (exp69) | - | 7 | 3.7e-8 | 1.3e-5 (-2.0004) | 9.4e-3, 2.4e-2 | 2.8e-3, 2.8e-3, 4.0e-3, 1.2e-3, 2.0e-4 | 1.9e-4 |
| window w 4, data (exp69) | 0.0185 | 1 | 3.3e-3 (d_t) | 7.3e-4 | 1.5e-4, 2.9e-4 | 1.3e-4, 1.6e-4, 1.4e-4, 3.6e-4, 2.7e-3 | 9.0e-5 |
| state, data | 0.080; 0.107 | 1 | 6.4e-3 (d_t) | 3.2e-3 (-2.035) | 9.5e-3, 2.4e-2 | 3.1e-3, 4.2e-3, 4.4e-3, 2.9e-3, 1.1e-2 | 2.6e-3 |
| state, self | 0.080; 0.107 | 1 | 2.5e-3 (rotation) | 3.3e-3 (-2.062) | 9.8e-3, 2.4e-2 | 3.3e-3, 4.4e-3, 4.5e-3, 3.1e-3, 1.2e-2 | 2.7e-3 |
- The state emulator breaks the boosts at the oracle's level (1.0-1.04 times it) where the window emulator keeps them
  (0.016 and 0.012 of it), and reads S's hidden weight (-2.035, -2.062). At 8 % one-step error in c its floor (the
  largest of the observed algebra and S 3.3e-3 to 6.4e-3; the rotation 2.5e-3 to 2.6e-3, 1.1-1.2 times exp68's
  0.35 e^2) reaches the superpositions and lies 1.5-7 times below the boosts: no separation at this accuracy. Its own
  rollouts stay within 0.009 of the data (c, 50 snapshots);
  8-step error 0.067 (c) and 0.119 (q); rotation commutator at 30 degrees 0.098 (c). x^2 + y^2 + 4Dt is inflated in
  both learned operators (2.7e-3 and 1.1e-2 against 2.0e-4): the ramps leave the training distribution at the box's
  edges.
- Uniform flow (seed 7, 4 pairs, 100 iterations; the scalar lift -> with the best flow -> with the best (dq0, flow)):
  gal_x 0.95 -> 2.7e-16 -> 6.7e-17, its flow coefficient 1.0000 on every pair (gal_y the same); boosts 2.5e-2,
  3.5e-2 -> 2.0e-2, 3.4e-2 -> 8.9e-3, 1.8e-2 (0.36 and 0.52 of the scalar lift); the superpositions keep 0.23-0.29;
  the observed algebra and S at most 6.2e-12. exp69's best hidden lift without the flow kept 0.49-0.66 of the boosts
  and 0.32-0.42 of the superpositions on this seed's development pairs (other pairs); the registered comparison is on
  42-44 with TR8's pairs.
Predictions (seeds 42-44; the uniform flow at 128^2 on CPU with 4 pairs and 100 iterations; state emulators at 128^2 on
Colab, notebook 12):
(UF1) gal_x and gal_y with the best uniform flow <= 1e-12, and their flow coefficients along their own direction within
1e-3 of 1 on every pair, 3/3 seeds (the construction's check).
(UF2) Every heat-false generator with the best (dq0, uniform flow) >= 0.5 times exp69's TR8 value without the flow (same
pairs and iterations), 3/3.
(HL0) State emulators: held-out one-step error of c <= 0.04 and of q <= 0.05, 3/3.
(HL1) Data arm: both boosts >= 10 times the largest quotient of the observed algebra and S, on >= 2/3.
(HL2) Data arm: S's hidden lift within 0.1 of -2, on >= 2/3.
(HL3) Data arm: both boosts within a factor 3 of the oracle's (same seed) on >= 2/3, and >= 10 times exp69's window
emulator's data arm (same seed) on 3/3.
(HL4) Data arm: the seven are not all separated: the smallest heat-false quotient < 10 times the largest of the observed
algebra and S, on >= 2/3.
(HL5) Self arm: c's own rollouts within 0.05 of the data (relative) and HL1's separation, on >= 2/3.
No prediction: the lift arm (reported against TR8), the rotation commutators, the 8-step errors.
Reading registered in advance:
- HL1-HL3 met: a learned operator of the full state replaces the simulator when the hidden variable is observed in
  training: FINS reads the boosts' breaking at the simulator's level and S's hidden weight. The window emulators'
  failure on the boosts is the missing hidden state, not learning as such.
- HL3 met with HL1 not: the breaking is learned and the accuracy floor hides it; separation needs a lower error.
- HL4 met: the superpositions need a lower floor than one-step errors of a few per cent give (EP3's 0.35 e^2).
- HL3 not met: the state emulator keeps the boosts too; the hidden state is not enough.
- UF2 met: the breaking c carries is not the hidden state's missing mean flow; it lies in the flow's structure, so a
  model of c's data is not excused by a global constraint. UF2 not met: part of it is that constraint.
Step 2 (a model of c alone with a learned hidden field) is registered separately after its development.

**A tracer in a hidden flow -- the learned arm (exp69; notebooks 11 and 11b, Colab; 11.10; TR0 and TR6-TR7 on
seeds 42-44).**
Runs: fins_nb11.zip and fins_nb11b.zip, each extracted into its own new directory (fins_runs/nb11_in, in_fins_nb11b) and
merged into fins_runs/tracer_runs (logs in logs_nb11, logs_nb11b); results/tracer/summary.{json,md}; verdicts_0710.py.
Notebook 11 lost the emulators of seeds 42 and 43 (Drive was full; 43 was audited first) and ran the self arm in float32
on the GPU. Notebook 11b retrained 42 and 43 (the same logged losses to four digits; 43's float32 rollouts reproduce
notebook 11's bit for bit), checked the emulators' rollouts (rollcheck) and ran the self arm in float64 (self64,
development's precision; the registered verdicts read it), the self arm with pyqg's filter after every step (selffilt,
post hoc) and seed 42's data arm. For 43 and 44 the float32 and float64 self arms agree to two digits except at the
round-off floor. The data arm ran in float32 on all three seeds (under 0.5 % by that comparison).
Emulators (1024 trajectories, 10000 steps of 32, 40 modes): held-out one-step error 0.0092, 0.0035, 0.0048 (8 steps
0.0435, 0.0139, 0.0829); rotation commutator at 30 degrees 5.5e-3, 3.1e-3, 4.0e-3.
Rollouts from the data's first four frames (frames 4-50 from the emulator): seeds 43 and 44 diverge, max |c| passing
twice the data's at frames 28 and 18 (the emulator's 25th and 15th steps) and reaching 3.3e5 and 2.9e10, identically in
float32 and float64 and with the filter after every step: an instability of the emulators' own dynamics at resolved
scales, not of precision or of the grid scale. Seed 42 stays within 0.034 of the data, its max |c| decaying from 1.91 to
1.35, then jumping 12 % to 1.53 at the last frame (after the frames the pencil uses).
Self arm (float64): on 43 and 44 it audits non-physical states (the characteristics change by 1.9 and 0.28 over K; the
translations, rotation, d_t, S and the boosts at 2e-2 to 0.84, d_c and c d_c far lower, 43's superpositions at 7e-6 to
3e-5 and its own count 7). On 42, d_t and S fail on its own rollouts (0.67 and 0.12; development: d_t 5.5e-8, S
5.5e-4) while the translations and the rotation match the data arm; post hoc reading: the rollouts are not smooth in
time, perhaps the growing mode of 43 and 44 at small amplitude (not measured). The self audit is not available for these
emulators.
Data arm (FINS's window pencil on the data: K-step transports from true windows):
| seed | observed algebra: d_x, d_y, rotation, d_c, c d_c; d_t | S | boosts x, y | superpositions x, y, xy, x^2 - y^2; r^2 | H lowest |
|---|---|---|---|---|---|
| 42 | 2.3e-4, 2.0e-4, 2.2e-4, 2.3e-4, 5.1e-5; 9.5e-4 | 2.1e-4 | 1.0e-4, 6.7e-5 | 3.5e-4, 3.5e-4, 7.9e-4, 4.6e-4; 5.9e-3 | 6.2e-5 |
| 43 | 2.0e-4, 1.2e-4, 1.2e-4, 1.6e-4, 2.5e-5; 4.8e-4 | 2.0e-4 | 5.0e-5, 3.1e-5 | 2.7e-4, 2.1e-4, 3.3e-4, 2.4e-4; 3.4e-3 | 3.0e-5 |
| 44 | 2.3e-3, 1.9e-3, 2.5e-3, 8.4e-4, 3.6e-4; 8.0e-3 | 1.7e-3 | 2.8e-4, 4.1e-4 | 1.1e-3, 1.4e-3, 3.7e-3, 1.6e-3; 1.6e-2 | 2.6e-4 |
- No separation: on every seed H's lowest lies below the observed algebra's d_x, d_y, rotation and d_c, and the
  boosts below every member of it but c d_c (on 44 boost_x is below c d_c too).
- Against the oracle on the same seed: the boosts at 0.005-0.033 of the oracle's quotients; the superpositions x and y
  at 0.25-1.21, x^2 - y^2 at 0.24-1.18, xy at 0.83-7.2, x^2 + y^2 + 4Dt at 18-71 times (seed 42's self arm: boosts
  0.017-0.022, x and y 0.42-0.49). The best hidden perturbation found (TR8) leaves the boosts 0.28-0.52 of their
  quotients and the superpositions 0.14-0.30 (upper bounds on the minimum, nearly converged for the boosts).
Verdicts: TR0 MET (3/3); TR6 MET (self and data 3/3; the self arm's numbers are not interpretable); TR7 NOT MET (0/3 in
the self arm: seed 42's x and y at 0.42 and 0.49, and 43 and 44 non-physical). The float32 self runs (43, 44) and the
filtered ones give the same verdicts.
Reading:
- TR0 and TR6 met, TR7 not: from data alone, window emulators accurate to 0.35-0.92 % per step do not separate the
  observed algebra from the heat closure's generators, though not for the registered reason. They break the
  superpositions x and y about as much as the hidden flow does (development's persistence is not reproduced at 128^2;
  r^2 18-71 times more) and keep the boosts at 0.005-0.033 of the flow's breaking, although c carries it once the hidden
  state may adjust (TR8: 0.28-0.52 left after the best adjustment found). Consistent with a window predictor that
  continues observed motion: a boost adds a drift to the window's history. Post hoc reading; not tested.
- Two of three accurate emulators diverge on their own within 15-25 steps and the third fails d_t and S on its own
  rollouts: here one-step accuracy of 0.35-0.92 % did not give usable own dynamics, and the self audit was lost to it.

**Notebook 10: emulators of pyqg with a known answer (exp68) -- results (11.10, Colab; 33 of 36 audits; f rot
s44 pending).**
Runs: fins_nb10.zip extracted into its own new directory (fins_runs/in_fins_nb10) and merged into fins_runs/pairs_runs
(logs in logs_nb10); results/pyqg/pairs_summary.{json,md}; verdicts_0710.py (epv). Drive ran out of space during the
notebook: all 12 emulators trained in the first session and printed "saved", but at the rerun 9 checkpoints (all but f
plain) were missing. Five were retrained in full (beta plain 42 and 44, f rot 43, f grid 42 and 44; their logged losses
match the first run to 3-4 digits); four crashed on a truncated .partial (EOFError: beta plain 43, f rot 42 and 44, f
grid 43). f rot s44 has no audit; the other 11 have all three (beta plain s44's error audit appears twice, with the same
output). exp68 now removes an unreadable .partial and starts over, and notebook 10 skips audited emulators.
| regime, arm, seed | held-out one-step error | rotation commutator 90 / 30 / 2 deg | FINS: rotation plane (/ 0.35 e^2) | d_x, d_y | D_x, D_t, D_beta |
|---|---|---|---|---|---|
| f plain 42 | 0.0371 | 5.6e-2 / 5.2e-2 / 9.9e-3 | 6.9e-4 (1.43) | 7.7e-5, 1.4e-4 | 1.0e-3, 5.9e-4, 9.4e-4 |
| f plain 43 | 0.0402 | 5.2e-2 / 5.0e-2 / 1.0e-2 | 7.4e-4 (1.31) | 5.6e-5, 1.8e-4 | 1.1e-3, 1.1e-3, 1.0e-3 |
| f plain 44 | 0.0340 | 4.8e-2 / 4.7e-2 / 8.1e-3 | 4.6e-4 (1.14) | 1.0e-4, 1.7e-4 | 6.5e-4, 1.4e-3, 6.7e-4 |
| f grid 42 | 0.0275 | 3.8e-2 / 3.5e-2 / 6.2e-3 | 2.5e-4 (0.94) | 1.6e-5, 1.9e-5 | 4.4e-4, 2.2e-4, 3.8e-4 |
| f grid 43 | 0.0412 | 5.1e-2 / 5.2e-2 / 9.4e-3 | 5.3e-4 (0.90) | 4.6e-5, 1.2e-4 | 9.3e-4, 9.7e-4, 9.1e-4 |
| f grid 44 | 0.0269 | 3.8e-2 / 3.5e-2 / 6.8e-3 | 2.9e-4 (1.15) | 3.5e-5, 7.4e-5 | 4.5e-4, 1.6e-3, 5.4e-4 |
| f rot 42 | 0.0204 | 3.0e-2 / 2.7e-2 / 5.6e-3 | 1.7e-4 (1.19) | 3.8e-5, 1.1e-4 | 2.8e-4, 1.5e-4, 2.4e-4 |
| f rot 43 | 0.0202 | 2.5e-2 / 2.5e-2 / 4.7e-3 | 1.9e-4 (1.35) | 2.5e-5, 7.7e-5 | 3.2e-4, 2.1e-4, 2.9e-4 |
| beta plain 42 | 0.0257 | 1.1 / 0.43 / 3.0e-2 | 1.3e-2 | 7.2e-5, 1.2e-4 | 9.9e-3, 7.3e-2, 5.0e-4 |
| beta plain 43 | 0.0161 | 1.3 / 0.48 / 3.3e-2 | 1.4e-2 | 4.7e-5, 9.6e-5 | 9.2e-3, 5.9e-2, 7.0e-4 |
| beta plain 44 | 0.0246 | 1.3 / 0.48 / 3.3e-2 | 6.7e-3 | 6.9e-5, 1.1e-4 | 4.8e-3, 4.5e-2, 4.3e-4 |
The port's 30-degree commutator on the same states is 1.2e-3 to 3.9e-3 at f and 0.44-0.49 at beta, where the flow
breaks the rotation. d_t sits at 1.6e-10 to 1.3e-9, the shift at 7.3e-4 to 4.2e-2, own count 1 everywhere, the span of k
misses 0.002-0.093 at f and 0.002-0.294 at beta. Rollouts stay within 0.002-0.009 of pyqg.
- EP0 NOT MET: f plain 0.0340-0.0402 (beta plain 0.0161-0.0257); PG8's emulators gave 0.073-0.105 and 0.035-0.055 on the
  same held-out trajectories.
- EP1 MET: the rotation contrast beta / f (same seed) is 19.1, 19.1 and 14.5 (PG8's emulators 4.1-5.7).
- EP2 MET: beta's deepest scaling direction (x d_x + y d_y, t d_t, q d_q) is (0.940, -0.945, 1), (0.932, -0.920, 1),
  (0.924, -0.922, 1): (w_t, w_u) / w_x = (-1.005, 1.064), (-0.987, 1.073), (-0.998, 1.083).
- EP3 PENDING (f rot s44): rotation plane / 0.35 e^2 within 0.90-1.43 on the other eight f-plane emulators.
- EP4 PENDING (f rot s44): on 42 and 43, rot against plain lowers the 30-degree commutator 1.92 and 1.95 times and the
  rotation plane 3.95 and 3.87 times; the plane ratio over the squared error ratio is 1.19 and 0.98 (the second part is
  met already; the first needs f rot s44 at most 3.1e-2 and 2.3e-4).
- EP5 MET: grid's 32-cell commutators are 4.0e-2 to 8.0e-2 (plain's at most 2.6e-15). FINS's d_x and d_y through grid
  are within 3 times plain's on 43 and 44; on 42 they are 4.8 and 7.5 times lower. grid's held-out errors are 0.74, 1.02
  and 0.79 of plain's.
Reading (registered):
- EP1-EP2 met with EP0 not: the contrast does not need 3 %. With PG8's two limits removed, FINS on the emulators' own
  rollouts reads which parameter breaks the rotation and beta's surviving scaling, the learned-operator row that PG8 and
  notebook 9 missed.
- EP3 so far (8 of 9 f-plane emulators): their rotation defect follows their one-step error (0.90-1.43 times 0.35 e^2).
  Read on the rotation, the audit measures accuracy rather than a separately learned symmetry. The other kept generators
  do not follow 0.35 e^2 as closely (d_x 0.06-0.51, D_t 0.84-6.1 times it at f; post hoc).
- EP5 met: the grid channels break finite translations by 4-8 % over 32 cells. FINS's infinitesimal translations at the
  data's states do not show it (0.13-0.81 of plain's), nor does the test error (0.74-1.02 of plain's): a limit of the
  local readout.

**A tracer in a hidden flow -- results of the CPU arms (exp69; 11.10, night; TR1-TR5 on seeds 42-51, TR8 on 42-44; TR0
and TR6-TR7 wait for notebook 11).**
Runs: fins_runs/tracer_runs (data in fins_runs/tracer_reg/tracer_cache); results/tracer/summary.{json,md};
verdicts_0710.py prints TR0-TR8. The machine's memory limit killed oracle s42 and the first observability attempts
(s42-s44) when they ran beside another job (the kernel's OOM log, no traceback). Oracle s42 was rerun with the
registered command by the second runner (beside the first runner's jobs); the observability runs were rerun one at a
time with two threads instead of one; otherwise unchanged.
Oracle (a simulator, c observed; seeds 42-51):
- Own count 7 on 10/10: the 7th eigenvalue 1.9e-8 to 7.4e-8, the 8th 2.1e-4 to 4.4e-4; the span of the 7 lowest misses
  at most 1.0e-5 of the observed algebra with S.
- The observed algebra's largest quotient is the rotation's on every seed (2.9e-8 to 8.6e-8, the discretization's
  aliasing). S lifted 1.6e-8 to 5.5e-8; its hidden lift -2.00038 to -1.99980; S without the lift 1.1e-2 to 2.0e-2 (2.1e5
  to 6.9e5 times).
- The heat model's seven: the smallest is sup_r2 on every seed (1.9e-4 to 3.3e-4), 2.2e3 to 8.8e3 times the largest of
  the observed algebra and S; the boosts 6.9e-3 to 2.1e-2, sup_x and sup_y 7.9e-4 to 1.5e-3, sup_xy 3.5e-4 to 7.0e-4,
  x^2 - y^2 8.9e-4 to 1.7e-3. H's lowest 1.2e-4 to 2.7e-4; c^2 d_c 1.6e-2 to 2.3e-2.
Two-stage (weak SINDy on c; seeds 42-51; weak residual 0.63-0.68 everywhere):
- 7/10 (42-44, 46, 47, 50, 51): c_t = a c_xx + b c_yy with a, b 0.040-0.056 (|a - b| 2.7-32 % of their mean); algebra of
  14: 5 of the observed algebra (the rotation lost), S, and 8 outside (H's seven and a stretched rotation).
- 3/10 keep spurious terms beside the two diffusions: s45 a drift 0.024 c_y with -0.046 (c^2)_y + 0.030 (c^3)_y; s48
  -0.012 c^2 + 0.0074 (c^3)_y; s49 -0.011 c^2. Their algebras have 3-5 dimensions: d_x, d_y, d_t, with S - (c - 0.50)
  d_c / 2 and a drift -0.0011 t d_y (s45; the minimum of the fitted flux coefficient), S - 2c d_c and a stretched
  rotation (s49); d_c and c d_c lost (s48 also S), nothing of H.
- The rotation is lost on 10/10.
Best hidden lift (exact dynamics; seeds 42-44, 4 pairs each, 100 iterations):
- The observed algebra and S fall to at most 7.2e-12 (rotation; S 9.9e-13 to 3.4e-12; with the scalar lift the rotation
  and S sat at 1.6e-8 to 1.6e-7, the aliasing and filter residue, reduced about 1e4-fold).
- After 100 iterations the heat model's seven are at least 3.2e-5 (sup_r2, s42): the boosts 3.4e-3 to 2.4e-2 (0.28-0.52
  of the scalar lift), the superpositions 3.2e-5 to 5.5e-4 (0.14-0.30). These are upper bounds on the minimum: the
  superpositions still fell 6-16 % between iterations 50 and 100 (development, one pair: sup_xy 20 % lower at 400).
Verdicts: TR1 MET (10/10), TR2 MET (10/10), TR3 MET (10/10), TR4 NOT MET (7/10), TR5 MET (10/10), TR8 MET (3/3); TR0,
TR6 and TR7 PENDING (notebook 11; its first attempt's trainer was killed by Colab's memory: the training set kept q and
concatenated copies, ~10 GB; fixed in 9b1832f, the same data bit for bit).
Reading:
- TR1-TR3 met: through a simulator, FINS reads the observed algebra of a variable that has no closed equation and how
  the hidden variable must move with it (S's -2q d_q to 4e-4), the 8th eigenvalue 3.7e3 to 1.4e4 times the 7th.
- TR4 not met, TR5 met: weak SINDy closes c as diffusion on 7/10, whose exact algebra carries H's seven and loses the
  rotation; on 3/10 its threshold keeps spurious terms, whose algebras lose d_c, c d_c and the rotation instead (s48
  also S; in s45 and s49 S survives only combined with c d_c). The identified algebra is wrong on 10/10; the registered
  form (holds H's seven) held on 7.
- TR8 met: c carries the breaking: the best hidden perturbation found over K steps does not remove it (6.6 decades
  between the smallest of the heat model's seven, sup_r2 on s42, and the largest of the observed algebra and S, the
  rotation on s44; 7.0-7.2 within each seed). The learned arm (TR0, TR6-TR7) will show whether an emulator of c alone
  reads it; in development it did not.
- Corrections to the development text of the registration entry below (independent check; predictions unchanged): weak
  SINDy at 64^2 gives c_xx 0.0413 (not 0.0414); on one pair at 400 iterations sup_xy 4.0e-5 (not 4.1e-5), sup_x 3.5e-4
  at 100 and 3.4e-4 at 400 (not 3.6e-4, 3.5e-4); the emulator's boost_x is 0.014 of the 128^2 oracle's (not 0.015); on
  the data the observed algebra spans 1.7e-4 (c d_c) to 3.3e-3 (d_t), not 1.2e-3 to 3.3e-3 (H's lowest 9.0e-5 is below
  all of it); the w = 1 self arm used 80 pairs (64 for the two w = 4 self arms).

**A tracer in a hidden flow: symmetries of a variable that is not closed (exp69, exp69_hidden_tracer.py; notebook 11) --
development and pre-registration (10.10, night; the 09.10 reviewers' partial observability, the authors' choice of system;
written after the development runs below, before any run on seeds 42-51).**
System: exp61's f-plane flow (qg2d's numerics: RK4, pyqg's filter on both fields) carrying a passive tracer,
q_t + J(psi, q) = 0, c_t + J(psi, c) = kappa Lap c, kappa = 0.05. Hidden q: exp61's two or three shielded elliptic
vortices with centres uniform in the disk of radius 2 (an isotropic prior); c: three to five positive elliptic Gaussian
blobs (widths 0.6-1.4, centres uniform in the disk of radius 2.5) times a log-uniform factor in [0.5, 2]. L = 16, scored
|x|, |y| < 4.5, 51 snapshots, dt 0.05 (two RK4 steps each), K = 8. Only c is observed (max |u| 0.38; c changes by 3.6 %
over K; the tracer's mass drifts 5e-16; half step 8.5e-5 of the change over the window).
Ground truth. The full system's point algebra on localised data: d_x, d_y, d_t, rotation, d_c, c d_c, the shift of q (a
convention of the inversion) and S = x d_x + y d_y + 2t d_t - 2q d_q (kappa ties time to length). Finite flows (64^2,
four trajectories, |c' - g c| / |c(T) - c(0)|): rotation by 30 degrees 1.4e-3, S (x -> 1.1 x, t -> 1.21 t,
q -> q / 1.21) 5.6e-4; broken by kappa: t d_t - q d_q (t -> 1.1 t, q -> q / 1.1) 6.5e-2, x d_x + y d_y alone 0.13. Those
with a point lift form the observed algebra in c: d_x, d_y, d_t, rotation, d_c, c d_c (6) and S (the 7th; its lift
scales the hidden q). A closed equation of heat type c_t = D Lap c has in exp45's class 14 generators (exact rational
determining equations; `check`): the 7 and seven that the hidden flow breaks, the boosts 2Dt d_x - x c d_c,
2Dt d_y - y c d_c and the superpositions x, y, xy, x^2 - y^2, x^2 + y^2 + 4Dt. H = span{t d_x, t d_y, (xc, yc, x, y, xy,
x^2, y^2, t) d_c} (10 fields) holds the seven for every D.
Arms:
- oracle (a simulator, c observed): the full system's exact tangent; exp45's 60 fields in c lifted to q as scalars (xi,
  eta, tau act on q too, phi on c only) and q d_q (61 fields); the pencil compares c only. Readouts: exp61's, the pencil
  restricted to H, and S's hidden lift (the q d_q coefficient of the deepest direction in the plane (S in c, q d_q)).
- two-stage: weak SINDy on c alone (exp61's reference library d_x^p d_y^q (c^j), j <= 3, p + q <= 4, and a constant, on
  the 16 FINS trajectories), then the identified equation's algebra in the class (exp50's determining system in 2D,
  exact rationals, null space below 1e-8 of the largest singular value).
- learned (data only): window emulators c_{j+1} = F(c_{j-3}, ..., c_j) (WindowFNO2d: exp45's FNO2d with the last frame
  and the three increments as channels, no positional input) trained on c alone (port trajectories of data seed
  6000 + s). FINS's window pencil: the window's perturbation is Q at its four frames, the emulator's tangent transports
  it, Q(t0 + k) is the data term; on the emulator's own rollouts from the data's first four frames (self; windows from
  frame 9, so that every frame and time-derivative stencil comes from the emulator) and on the data (data).
- best hidden lift (a diagnostic with the exact dynamics): per candidate, the minimum over a hidden perturbation dq(t0)
  of the c-part's K-step mismatch (CGLS from the scalar lift, 100 iterations; 4 pairs), over the scalar lift's pencil
  denominator: whether c carries a generator's breaking once the hidden state may adjust.
Development (seed 7; data seeds 4007, 5007, 6007; oracle, two-stage and best lift at 128^2; emulators at 64^2 on CPU:
512 trajectories, 3000 steps of 16, 20 modes; all pencils on 80 pairs, self on 64):
| arm | one-step error | own count | observed algebra, largest | S | heat-false: smallest; sup_x; boost_x | H, lowest | span of 7, missing (with S) |
|---|---|---|---|---|---|---|---|
| oracle, 128^2 | - | 7 | 3.7e-8 (rotation) | 2.7e-8 (lift -2.0002; unlifted 1.9e-2) | 1.9e-4 (sup_r2); 1.0e-3; 9.4e-3 | 1.7e-4 | 0.000 |
| oracle, 64^2 | - | 7 | 3.7e-8 | 1.3e-5 (lift -2.0004) | 2.0e-4; 2.8e-3; 9.4e-3 | 1.9e-4 | 0.000 |
| best hidden lift, 128^2 | - | - | 7.5e-12 (rotation; S 4.7e-12) | - | 1.3e-4 (sup_r2); 5.8e-4; 1.1e-2 | - | - |
| window FNO, w 4, self | 0.0185 | 1 | 7.4e-4 (rotation) | 5.5e-4 | 1.4e-4; 1.5e-4; 1.4e-4 | 8.8e-5 | 3.302 |
| window FNO, w 4, data | 0.0185 | 1 | 3.3e-3 (d_t) | 7.3e-4 | 1.3e-4; 1.3e-4; 1.5e-4 | 9.0e-5 | 4.509 |
| window FNO, w 1, self | 0.618 | 1 | 0.15 (rotation) | 0.12 | 3.1e-2; 0.17; 3.1e-2 | 2.5e-2 | 1.999 |
| advective, w 4, self | 0.193 | 1 | 0.97 (d_t) | 0.43 | 4.6e-3; 6.1e-3; 3.4e-2 | 4.4e-3 | 4.780 |
Two-stage: c_t = 0.0402 c_xx + 0.0429 c_yy at 128^2 (weak residual 0.725; 64^2: 0.0414, 0.0429, 0.717); the algebra has
14 dimensions: 5 of the observed algebra (the rotation lost to the unequal diffusivities), S, 8 outside the observed
algebra with S, 7 in H.
- The oracle's 7 lowest eigenvalues are at most 3.2e-8, the 8th 2.9e-4 (128^2). At 64^2 pyqg's filter acting on the
  non-periodic ramps inflates the superpositions' quotients (|J_8 x - x| / |x| on the block without flow 3.7e-3 at 64^2,
  3.4e-6 at 128^2; with the flow 7.6e-3 and 6.7e-3): the registered oracle runs at 128^2.
- Best hidden lift: the hidden adjustment removes 34-51 % of the boosts' quotients and 58-68 % of the superpositions'
  (4 pairs); the observed algebra and S fall to 1e-11 (the scalar lift's aliasing and filter residue). On one pair with
  400 iterations the boosts converge by 25, the superpositions keep falling slowly (sup_xy 5.1e-5 at 100, 4.1e-5 at
  400; sup_x 3.6e-4, 3.5e-4), rotation and S reach 4e-13.
- Through the w = 4 emulator the observed algebra and the heat model's seven sit together at its error floor (one-step
  error^2 3.4e-4): the emulator persists perturbations that the hidden flow would advect (its sup_x and boost_x at 0.15
  and 0.015 of the 128^2 oracle's, below the best hidden lift's 0.32 and 0.49). On the data the order inverts (H's lowest
  9.0e-5 against the observed algebra's 1.2e-3 to 3.3e-3).
- Development variants: raw frames as channels lose the history in the conditioning (increments are ~0.5 % of the field;
  loss 1.8 at step 200 against 1.2e-2 with increment channels); w = 1 (Markov in c) cannot infer the flow (0.62); a
  learned streamfunction advecting c_j by Heun steps (AdvectFNO2d) reached 0.19 one-step error with non-smooth rollouts
  (8-step error 0.79, d_t 0.97 on its own rollouts). Neither is registered. Rotation commutators over held-out windows
  (30 degrees): w 4 1.2e-2, w 1 0.58, advective 0.16.
Predictions (128^2; oracle and two-stage on seeds 42-51, the best hidden lift on 42-44 with 4 pairs, window emulators w = 4
on 42-44 with 1024 trajectories, 10000 steps of 32 and 40 modes, notebook 11):
(TR0) Every window emulator's held-out one-step error <= 0.02.
(TR1) Oracle: own count 7 and the span of the 7 lowest misses <= 0.05 of the observed algebra with S, on >= 9/10 seeds.
(TR2) Oracle: S's hidden lift within 0.02 of -2 and S without its lift >= 100 times S with it, on >= 9/10.
(TR3) Oracle: every heat-false generator's quotient >= 100 times the largest of the observed algebra's and S's, on
>= 9/10.
(TR4) Two-stage: the identified equation's algebra holds H's seven (its intersection with H has dimension 7) and >= 7
generators outside the observed algebra with S, on >= 9/10.
(TR5) Two-stage: the rotation is not in the identified equation's algebra on >= 9/10.
(TR6) Learned: the lowest eigenvalue of the pencil on H <= 3 times the largest quotient of the observed algebra (no
separation), in self and in data, on >= 2/3 seeds each.
(TR7) Learned (self) against the oracle (same seed): sup_x, sup_y, boost_x and boost_y each <= 0.3 times the oracle's on
>= 2/3.
(TR8) Best hidden lift: every heat-false generator >= 1e-5 and the observed algebra with S <= 1e-9, on 3/3 seeds.
No predictions: S, the own counts and spans of the learned arm, c^2 d_c, the emulators' commutators, the development
variants.
Reading registered in advance:
- TR1-TR3 met: through a simulator, FINS reads the algebra of a variable without a closed equation and how the hidden
  variable must move with it (S's -2q d_q).
- TR4-TR5 met: the two-stage pipeline must close c; weak SINDy closes it as diffusion, whose exact algebra carries seven
  generators the hidden flow breaks and loses the rotation: a closed model makes exact claims the data do not support.
- TR8 met: c carries the breaking: no hidden adjustment over K steps removes it (H stays >= 1e4 above the observed
  algebra).
- TR0 and TR6-TR7 met: from data alone, a window emulator accurate to 2 % separates nothing: it persists perturbations
  that the hidden flow advects, below what any hidden adjustment allows, so the learned arm certifies neither the
  observed algebra nor the closure's extras. Under partial observation the learned arm is limited by the inference of
  the hidden state, not by FINS's criterion or by the observed variable.
- TR6 not met (a separation at 128^2): FINS through a window emulator reads the observed algebra from c alone; TR7 then
  says whether the emulator's breaking is the flow's. TR0 not met: the learned rows describe a weaker emulator than
  development's.

**Emulators of pyqg with a known answer (exp68, exp68_emulator_pairs.py; notebook 10) -- development and
pre-registration (10.10, afternoon; the 09.10 reviewers' emulator audit with a Lie-derivative reference; written after
the development runs below, before any run on seeds 42-44).**
Why PG8's emulators were poor: with 24 Fourier modes the spectral path cannot carry 3.1e-3 of the one-snapshot
increment's energy (a relative floor of 0.056; 40 modes leave 2.6e-5, 48 modes 3e-8; pyqg's f-plane data at 128^2),
and 256 training trajectories leave a generalization gap (development below: one-step error 0.050 on the plain
emulator's own training trajectories against 0.112 held out).
Design (pyqg at 128^2, exp61's settings; seeds 42-44):
- Recipe: exp61's FNO2d (residual step, one-step plus two-step loss, Adam, cosine) with 40 modes, width 32, 4 layers, on
  1024 trajectories of data seed 6000 + s, 10000 steps of 32 random triples; resumable.
- Emulators (same seed, initialisation and triple sequence within a seed):
  - f plain, beta plain;
  - f rot: every training triple rotated about the box centre by a uniform random angle (a quarter turn, then three
    Fourier shears: 4e-16 on an analytic blob, 1.2e-7 for pyqg states rotated and back);
  - f grid: the coordinates x / 8, y / 8 as two more input channels (the FNO paper's default), which can break the
    translations; plain is exactly equivariant to whole-cell shifts.
- Audits: the one-step error relative to the increment on the scored block (exp45; held-out trajectories of data seed
  5000 + s); the direct reference, commutators ||F(T q) - T F(q)|| / ||F(q) - q|| over 40 held-out pyqg states for
  rotations by 90, 30, 2 degrees and shifts by 32, 8, 0.5 cells along x and y (the port, seed 42's 40 states: 3.1e-9,
  1.2e-3, 9.6e-5 for the rotations, the discretization's aliasing; below 1e-6 for the shifts); FINS on the emulator's
  own rollouts (exp65, 80 pairs, K = 8).
Development (seed 7; 64^2, width 16, 24 modes, 256 trajectories, 3000 steps of 16, on CPU; the registered runs
are larger and at 128^2):
| emulator | held-out error | commutator: rotation 30, 2 deg; shift 32, 8, 0.5 cells (x) | FINS: rotation plane; d_x, d_y; D_x, D_t | rotation plane / error^2 |
|---|---|---|---|---|
| f plain | 0.112 | 0.138, 0.022; 0, 2.7e-15, 0.019 | 4.6e-3; 1.6e-3, 1.5e-3; 1.1e-2, 4.9e-3 | 0.37 |
| f rot | 0.065 | 0.080, 0.013; 0, 2.6e-15, 0.016 | 1.4e-3; 8.0e-4, 6.8e-4; 6.2e-3, 2.7e-3 | 0.33 |
| f grid | 0.139 | 0.159, 0.026; 0.122, 0.036, 0.019 | 6.5e-3; 1.7e-3, 1.3e-3; 1.3e-2, 5.3e-3 | 0.34 |
- The augmentation lowered the rotation commutator and the test error alike (30-degree commutator / error 1.23 in
  both arms). The rotation plane scales as 0.33-0.37 times the squared one-step error in all three arms: here the
  emulator's rotation defect is set by its accuracy, and the test error shows what the audit shows.
- The grid channels break finite shifts (0.122 over 8 units, 0.036 over 2), but the infinitesimal translations at the
  data's states are at plain's level (the 0.5-cell commutator 0.019, FINS's d_x and d_y within 1.2 times), set by the
  FNO's aliasing at 64^2; the span of the 7 lowest misses 2.07 of the algebra (plain 0.08). The test error is 1.25 times
  plain's.
- On its own training trajectories (the first 16) the one-step error is 0.050 (plain), 0.055 (grid) and 0.067 (rot),
  against 0.112, 0.139 and 0.065 held out: the augmentation removes the generalization gap.
Predictions (seeds 42-44):
(EP0) Every plain emulator (f and beta) has held-out one-step error <= 0.03.
(EP1) The rotation plane of beta plain >= 10 times f plain's (same seed) on >= 2/3 seeds (EA1 with the new recipe).
(EP2) beta plain's deepest scaling direction has (w_t, w_u) / w_x within 0.1 of (-1, 1) on >= 2/3 seeds (EA2).
(EP3) Every f-plane emulator (plain, rot, grid): the rotation plane within a factor 3 of 0.35 e^2, e its held-out
one-step error.
(EP4) rot against plain (same seed): the 30-degree commutator >= 1.5 times lower and the rotation plane >= 2 times
lower on 3/3 seeds; the rotation-plane ratio within a factor 1.5 of the squared error ratio on >= 2/3.
(EP5) grid against plain: grid's 32-cell commutators (x and y) >= 1e-2 and plain's <= 1e-6 on 3/3 seeds; grid's FINS d_x
and d_y quotients within a factor 3 of plain's on >= 2/3; the held-out errors within a factor 1.5 on >= 2/3.
No predictions: the 90-degree commutators, D_x and D_t, the spans and own counts, rot and grid at beta.
Reading registered in advance:
- EP0-EP2 met: with PG8's two limits removed, FINS on the emulators' own rollouts reads which parameter breaks the
  rotation (the learned-operator row PG8 missed). EP1-EP2 met with EP0 not: the contrast needs less than 3 %.
- EP3 met: an FNO emulator's rotation defect follows its one-step error; read on the rotation, the audit measures
  accuracy rather than a separately learned symmetry.
- EP4 met: rotation augmentation lowers the rotation defect through accuracy; the audit and the test error agree.
- EP5 met: the grid channels break finite translations, a global inhomogeneity that FINS's infinitesimal translations at
  the data's states do not see and the test error does not show: a limit of the local readout. If grid's d_x and d_y
  come out at least 3 times plain's, FINS sees the position dependence at 128^2.

**Notebook 9: an audit of PG8's emulators (exp65) and LieNLSD on exp60 (exp66) -- results (10.10, midday;
Colab; 12 audit runs and 30 LieNLSD runs, none failed; fins_nb9.zip extracted into fins_runs/nb9_in, json and logs
merged into fins_runs/audit_runs and lienlsd_aniso_runs; results/pyqg/audit_summary.{json,md},
results/aniso2d/lienlsd_summary.{json,md}).**
exp65, FINS on each emulator's own rollouts (self) against PG8 on pyqg's data (PG8), and the error ratios:
| regime | seed | rollouts' distance from pyqg | translations: self; PG8 | rotation plane (self) | other generators: self; PG8 | error ratio (transformed / untransformed) |
|---|---|---|---|---|---|---|
| f | 42 | 0.013 | 4.7e-5, 1.1e-4; 3.9e-3, 3.5e-3 | 3.5e-3 | D_x 5.0e-3, D_t 3.6e-3; 5.3e-3, 8.1e-3 | rotation 0.889, D_x 1.446, D_t 1.162 |
| f | 43 | 0.019 | 1.1e-4, 1.8e-4; 4.5e-3, 5.0e-3 | 3.0e-3 | D_x 4.8e-3, D_t 7.4e-3; 6.1e-3, 1.7e-2 | rotation 1.034, D_x 1.709, D_t 1.063 |
| f | 44 | 0.017 | 9.9e-5, 1.4e-4; 3.1e-3, 3.7e-3 | 1.3e-3 | D_x 2.1e-3, D_t 6.3e-3; 3.6e-3, 1.3e-2 | rotation 1.068, D_x 1.444, D_t 1.238 |
| beta | 42 | 0.012 | 4.3e-5, 9.8e-5; 1.5e-3, 1.7e-3 | 1.4e-2 | D_beta 1.5e-3; 1.9e-3 | D_beta 1.945 |
| beta | 43 | 0.018 | 5.5e-5, 1.1e-4; 2.0e-3, 2.8e-3 | 1.6e-2 | D_beta 2.3e-3; 3.7e-3 | D_beta 1.829 |
| beta | 44 | 0.015 | 6.1e-5, 7.1e-5; 1.3e-3, 1.5e-3 | 7.5e-3 | D_beta 9.1e-4; 1.6e-3 | D_beta 1.794 |
- d_t sits at 1.5e-10 to 1.4e-9 (the emulator is autonomous); the translations' error ratios are 1.000-1.001; own
  count 1 everywhere; the span of the k lowest misses 0.019-0.043 at f (k = 7), 0.016-0.307 at beta (k = 5).
- Self-consistency moves the translations from PG8's 1.3e-3 to 5.0e-3 to 4.3e-5 to 1.8e-4, and the generators in the
  table's last quotient columns by factors 1.07-2.31; they stay at 9.1e-4 to 7.4e-3. The shift sits at 6.0e-4 to 4.4e-2
  either way (PG8 / self 0.86-1.00).
- EA1: the rotation contrast beta / f is 4.1, 5.3, 5.7 (PG8 2.6-3.7; through the port, exp61's smallest broken quotient
  at beta is 4.3e-3 against 2.8e-5 for the f-plane's largest kept one).
- EA2: beta's deepest scaling direction has (w_t, w_u) / w_x = (-1.000, 1.185), (-0.987, 1.452), (-0.955, 1.175): the
  time weight within 0.045, the q weight 0.175-0.452 off (PG8: within 0.1 and 0.16-0.56 off).
- EA3: Spearman 0.604 over the 24 pairs. Post hoc: without the translations (12 pairs) -0.34; with PG8's quotients
  0.20 over the 24 pairs and -0.43 without the translations. The rotation's defect (1.3e-3 to 3.5e-3) is near D_x's
  (2.1e-3 to 5.0e-3), but its error ratio is 0.89-1.07 against 1.44-1.71.
exp66, LieNLSD as published on exp60 (fitted F; F's relative error on the held-out trajectories 0.014-0.043):
| a | LS1: a-hat more than 0.1 off | a-hat: median (range) | LS2: missing >= 0.5 | missing (5): median (range) | eps-hat: scan; sl(2) |
|---|---|---|---|---|---|
| 1.25 | 9/10 | 1.630 (1.341-1.745) | 10/10 | 1.200 (0.646-1.574) | 0.598-0.615; 0.598-0.614 |
| 1.5 | 9/10 | 1.722 (1.587-1.871) | 8/10 | 0.848 (0.229-1.445) | 0.599-0.607; 0.599-0.606 |
| 1.75 | 4/10 | 1.789 (1.627-1.930) | 2/10 | 0.318 (0.162-0.830) | 0.601-0.604; 0.601-0.604 |
- The untrained F (relative error 1.00-1.08): scan eps 0.80-2.00, a-hat 1.33-2.01, missing 1.21-1.96; its sl(2) field
  is hyperbolic on 8/10 seeds.
VERDICTS (literal; verdicts_0710.py): (EA1) NOT MET: 0/3. (EA2) NOT MET: 0/3. (EA3) MET: 0.604. (LS1) NOT MET: worst
cell 4/10 (a = 1.75; 9/10 at 1.25 and 1.5). (LS2) NOT MET: worst cell 2/10 (a = 1.75; 10/10 at 1.25, 8/10 at 1.5).
Reading:
- EA1-EA2 not met, the registered alternative: the emulators did not learn the pattern. Read from each emulator alone,
  with no data term, f's rotation, D_x, D_t and beta's D_beta sit at 9.1e-4 to 7.4e-3; the beta-plane's rotation
  breaking stands 4.1-5.7 times above the f-plane's, not 10, and beta's surviving scaling has its q weight 0.175-0.452
  off. Without the data term EA1 and EA2 fail as PG8 did. Per quotient the emulators' own share of PG8's is 0.43-0.94
  (the data term is the larger part for f's D_t and, on seed 44, f's rotation); the data term lowered the rotation
  contrast by factors 1.16-2.22 and lifted the translations by more than an order of magnitude.
- EA3 met by its number; the registered reading (FINS's per-generator defects rank the emulator's errors on transformed
  inputs) holds only as far as the translations' quotients (at most 1.8e-4, against at least 9.1e-4) set them apart;
  their error ratios are 1.000-1.001, and f's rotation has 0.889-1.068. Among the other generators the defects do not
  rank the errors (-0.34, post hoc).
- LS1-LS2 not met at a = 1.75: LieNLSD's scan exponent is biased high for every a (1.341-1.745, 1.587-1.871,
  1.627-1.930; medians 1.630, 1.722, 1.789), so it lands within 0.1 of 1.75 on 6/10, and its five smallest directions
  miss less of the algebra as a grows (medians 1.200, 0.848, 0.318). At a = 1.25 and 1.5 the registered pattern holds.
  Not registered: LieNLSD reads eps 0.598-0.615 (scan and sl(2)). In exp60's equation eps enters the local nonlinearity
  as well as the dissipation; the exponent only the nonlocal term. FINS through the solver reads both at every a, the
  linear response (exp67, computed at a = 1.5) there.

**FINS for systems: pyqg's flow in the velocity (u, v) (exp63) -- results (10.10, midday; 55 runs on two CPU workers,
709-1078 s each, none failed; the first launch of f seed 42 died with the machine's restart after its data
step and was rerun; json and npz in fins_runs/velocity_runs; results/velocity/summary.{json,md}).**
| row | own count = k | missing (span of k) | rotation space: deepest; spin / orbital; stretch | scaling space: deep; next | boost space: lowest (meanflow: two lowest) |
|---|---|---|---|---|---|
| f, pyqg | 3/10 (4 on 7) | 1.4e-5 to 2.7e-4 | 2.2e-5 to 8.8e-5; 0.99999-1.00000; 0.99999-1.00005 | D_t 7.9e-11 to 1.1e-9, then 1.9e-5 to 4.9e-5; 0.90-1.00 | 0.91-0.96 |
| f, meanflow | 1/5 | 5.0e-6 to 2.0e-4 | 2.2e-5 to 8.8e-5; 0.99999-1.00000; 0.99999-1.00003 | 7.9e-11 to 1.1e-9, then 2.3e-5 to 4.9e-5; 0.90-1.00 | 2.0e-13 to 4.2e-11, speeds within 1.4e-7 of 1 |
| beta, pyqg | 0/10 (3) | 0.85-0.99 | 3.5e-2 to 5.8e-2; 0.9986-0.9992; 1.002-1.213 | D_beta 9.5e-4 to 1.9e-3 (1, -1.01 to -1.00, 2.20-2.39, 2.20-2.39); 3.4e-2 to 5.4e-2 | 0.72-0.77 |
| beta, meanflow | 0/5 (5 of 6) | 9.3e-4 to 1.5e-3 | 3.5e-2 to 5.8e-2; 0.9986-0.9990; 1.051-1.163 | 4.8e-4 to 5.6e-4 (1, -0.98 to -0.99, 2.06-2.13, 2.07-2.13); 3.4e-2 to 5.0e-2 | 8.4e-14 to 1.9e-11, speeds within 1.4e-7 of 1 |
| rd, pyqg | 10/10 | 6.1e-7 to 4.9e-5 | 6.1e-6 to 2.2e-5; 0.99999-1.00000; 0.99999-1.00000 | D_t 6.3e-11 to 1.1e-9; 4.4e-4 to 1.2e-3 | 0.95-0.98 |
| rd, meanflow | 4/5 | 2.0e-6 to 1.3e-4 | 6.1e-6 to 2.2e-5; 1.00000; 0.99999-1.00000 | 6.3e-11 to 1.1e-9; 2.0e-4 to 4.8e-4 | 1.1e-3 to 2.8e-3, speeds 1.16-1.20 |
| drag, pyqg | 10/10 | 1.6e-6 to 7.0e-6 | 4.2e-7 to 2.4e-6; 1.00000; 0.99999-1.00004 | D_x 2.9e-7 to 9.9e-7 (1, 0.00, 1.00, 1.00); 2.9e-3 to 5.4e-3 | 0.74-0.76 |
- The smallest of the scalar rotation, the spin alone and the opposite spin sits 1.1e4 to 1.6e6 times above the rotation
  space's deepest at f, rd and drag (16-25 times at beta); with u d_u - v d_v they are at 0.63-1.0 in every row.
- f: the two deepest scaling directions span (1, 0, 1, 1) and (0, 1, -1, -1) to 1.8e-7 (the second eigenvector is
  D_x + 1.4 D_t, roughly); the blind count stops at 4 on 7/10, as in development (the rotation and D_x above the 1e-6
  floor).
- beta through pyqg's operator: the operator drops the mean of D_beta's characteristic (by construction; on seeds 42-46
  D_beta's quotient is 2.4-3.5 times the meanflow operator's, D_x's 1.06-1.10 times, the rotation's the same), and the
  span of the 4 lowest misses 0.85-0.99 (not isolated further: the fourth direction mixes D_beta with a zonal flow,
  independent check); the meanflow operator keeps D_beta (direction within 0.13 of (-1, 2, 2), span missing at most
  1.5e-3).
- The boost space through pyqg's operator: 354-1.8e6 times the algebra's largest quotient; meanflow rd: 104-184 times.
VERDICTS (literal; verdicts_0710.py): (UV1) MET: worst cells 10/10 (pyqg) and 5/5 (meanflow). (UV2) MET: 10/10.
(UV3) MET: 10/10 and 5/5. (UV4) MET: 10/10. (UV5) MET: 5/5.
Reading, registered in advance:
- UV1-UV3 met: FINS for systems recovers pyqg's velocity algebra with the rotation acting on the components as a vector
  (spin to orbital 1, the scalar rotation and the opposite spin broken) and the scalings' velocity weights.
- UV4-UV5 met: FINS reads the Galilean boosts as a property of the operator's state space, broken by pyqg's zero-mean
  velocity, exact with a conserved mean flow, broken again by the PV gradient that a mean flow implies with rd.
- Through the exact tangent of the port these are ceiling rows; the velocity algebra is the image of q's under a linear
  map, except the boosts, which q does not see.

**exp60's controls and the linear-response baseline (exp67) -- results (10.10, morning; 20 runs on one CPU worker,
91-110 s each, none failed; json and npz in fins_runs/controls_runs; results/aniso2d/controls_summary.{json,md}; the
linear response, deterministic, is in the registration and results/aniso2d/linear_response.json).**
| case | rotation scan: eps-hat; quotient | scaling scan: (a, c); quotient | (1.5, 0.5) scaling | stretched rotation at 0.6 | d_t | ratio |
|---|---|---|---|---|---|---|
| mismatch | 0.646-0.653; 2.0e-5 to 4.3e-5 | (1.500-1.504, 0.494-0.500); 3.0e-7 to 4.4e-7 | 3.0e-7 to 4.4e-7 | 2.0e-4 to 3.4e-4 (at 0.8: 1.8e-3 to 4.0e-3) | 2.2e-10 to 2.0e-9 | rotation scan / largest kept: 51-133 |
| cubic | 0.600; 1.7e-7 to 8.3e-7 | (1.595-1.620, 0.263-0.292); 7.2e-7 to 1.6e-6 | 5.6e-5 to 8.4e-5 | 1.7e-7 to 8.3e-7 | 4.5e-9 to 4.5e-8 | (1.5, 0.5) / largest kept: 76-359 |
- The sl(2) readout agrees with the rotation scan: 0.646-0.653 (mismatch), 0.5999 (cubic).
- CT2's misses: seeds 45 and 46 at 76 and 91 times (seed 51 at 100.5). There the stretched rotation sits at 8.2e-7 and
  7.4e-7 (development 2.4e-7) and the (1.5, 0.5) scaling at 6.2e-5 and 6.7e-5 (development 9.1e-5).
- cubic: the best scaling of the scan is 1.7-4.5 times the stretched rotation's quotient and below 1e-6 on 5/10 seeds;
  the pencil's fifth eigenvalue is 1.9e-7 to 9.2e-7, the sixth 7.0e-6 to 8.3e-6. It breaks the dissipation's weight
  too: on the a = 1.5 line the best c is 0.375-0.389, at 1.9-2.9 times the scan's best and 21-29 times below
  (1.5, 0.5) (unregistered; independent check). An approximate scaling of the data, between the cubic nonlinearity and
  the order-1.5 dissipation.
VERDICTS (literal; verdicts_0710.py): (CT1) MET: 10/10. (CT2) NOT MET: 8/10 (the 100-times margin; a-hat more than
0.05 from 1.5 and eps within 0.002 of 0.6 on 10/10).
Reading: the reading registered for CT1-CT2 met is not reached (CT2). What CT1 gives: with eps 0.8 in the dissipation
and 0.6 in the nonlinearity, the whole-equation readout shows no stretched rotation (51-133 times the kept generators);
the linear response shows the two anisotropies, 0.8 and 0.6, from which the same follows by an argument about the
equation (a symmetry acting on x, y alone fixes the constant states, so it must keep both ellipse families). What CT2's
parts give: in cubic the best scaling's exponent differs from the linear order (1.595-1.620 against 1.5 on 10/10) and
the stretched rotation is read at 0.600 on the scan's grid (sl(2) 0.59992-0.59995), but the (1.5, 0.5) scaling stands
out from the kept generators by 76-359 times, under 100 on 2/10, and the best scaling is itself near the kept
generators' level. The linear response keeps 1.5 and flags the b^2 term (0.21). At a = 1.5, FINS's exponent readout
adds nothing to the linear response (deterministic, registered).

**Two-stage rows on pyqg's data without quasi-geostrophic knowledge (exp64) -- results (10.10, morning; 40 runs on one
CPU worker, 280-345 s each, none failed; json in fins_runs/pyqg_lib_runs; results/pyqg/libraries_summary.{json,md}
and, post hoc, libraries_obs_posthoc.{json,md}).**
| regime | obs | riesz-free | riesz | nested (picked) |
|---|---|---|---|---|
| f | 7/10 (by function 10/10) | 10/10, s 1.9992-2.0008 | 10/10, kappa 0 | 10/10 (riesz-free) |
| beta | 10/10 | 10/10 | 10/10 | 10/10 (riesz-free) |
| rd | 9/10 (by function 10/10); kd2-hat 1.000000 | 0/10: s 1.17-1.29, transport -0.55 to -0.56 | 10/10, s 1.9992-2.0008, kappa 1.0016 | 10/10 (riesz) |
| drag | 10/10 | 10/10 | 10/10 | 10/10 (riesz-free) |
Counts are true equations; the counts with the regime's algebra are the same in every cell. ref: 0/40, as in exp61.
- The true models' coefficients: obs within 1.4e-5 of the truth (-1, -0.5 v, -0.3 q); riesz within 9.1e-4.
- s and kappa sit at the golden section's resolution: s = 2 -+ 0.0008, kappa 1.0016 on every rd run.
- The four obs misses are the true equation in another form. The obs library holds exact identities of incompressible
  flow, d_x(u q) = d_xx(uv) + d_xy(v^2 - u^2) / 2 and d_y(v q) = -d_yy(uv) + d_xy(v^2 - u^2) / 2 for q = curl u (with rd
  only their sum, by u . grad psi = 0). MSTLS returned the stress form (minus the curl of div(u u)) on f s46 and rd s51
  and mixed forms on f s47 and s51. The registered scorer counts only the transport form, and its algebra breaks the
  rotation for any obs term without a quasi-geostrophic equivalent, so the four fail both scores (the scalings match).
  Rescored by function (post hoc, exp64_posthoc_identity.py: the right-hand side as a differential polynomial in psi,
  each of the regime's monomials within 2 % of its coefficient, no other monomial above 1e-3; the algebra not
  rescored): the equation on 40/40, largest relative deviation 5.2e-5. Development removed the identity with the
  derivatives of u and v alone and missed these.
VERDICTS (literal; verdicts_0710.py): (QL1) NOT MET: worst cell 7/10 (f; rd 9/10). (QL2) MET: 10/10 in every regime.
(QL3) MET: 10/10. (QL4) MET: 10/10.
Reading: the reading registered for QL1-QL4 met is not reached (QL1). What QL2-QL4 give: without being told
"quasi-geostrophic", the two-stage route reaches pyqg's equation and its algebra in every regime with the
generalized-SQG family, transport of q by grad^perp of q under an isotropic multiplier -(|k|^2 + kappa)^(-s/2), the
exponent and the screening profiled by the residual to their true values (40/40; exp61's quasi-geostrophic completion
fixed s = 2). The completion needs the active-scalar transport, not its quasi-geostrophic member; exp61's reading that
the quasi-geostrophic structure must be supplied is narrowed to that. Post hoc: from pyqg's own velocity in a library
local in (q, u, v), the equation on 40/40 by function (36/40 by the registered term match and algebra). The two-stage
route also returns beta and kd2, which FINS does not; FINS reads rek only in exp62's tau(t) class.

**The drag's deformed time scaling (exp62) -- results (10.10, morning; 40 runs on one CPU worker; one run, rek 0.1
seed 43, was killed out of memory beside two development runs and rerun; json and npz in fins_runs/drag_tau_runs;
results/drag_tau/summary.{json,md}).**
| row | own count = 7 | missing (span 7) | r-hat (phi) | r-hat (tau) | V_r exact | D_t | time pencil: 2nd; 3rd | 8th eigenvalue |
|---|---|---|---|---|---|---|---|---|
| drag 0.1 | 10/10 | 9.6e-7 to 2.8e-6 | 0.09995-0.09997 | 0.1022-0.1024 | 9.1e-10 to 4.3e-8 | 3.0e-2 to 4.7e-2 | 4.1e-7 to 5.1e-7; 0.77-0.83 | 1.6e-3 to 2.6e-3 |
| drag 0.3 | 10/10 | 1.1e-6 to 7.8e-6 | 0.29967-0.29969 | 0.3095-0.3098 | 2.6e-10 to 3.5e-9 | 9.1e-2 to 9.9e-2 | 2.9e-7 to 4.5e-7; 0.66-0.69 | 3.1e-4 to 8.1e-4 |
| drag 0.6 | 10/10 | 2.9e-6 to 2.6e-5 | 0.59833-0.59836 | 0.6267-0.6270 | 9.1e-10 to 1.1e-9 | 0.122-0.125 | 3.8e-7 to 5.4e-7; 0.47-0.48 | 4.5e-5 to 1.3e-4 |
| f | 9/10 | 1.8e-6 to 1.2e-5 | -0.00002 to -0.00001 | -0.0001 to 0.0000 | V_0.3: 0.25-0.43 | 2.3e-9 to 1.1e-7 | 3.6e-7 to 4.7e-7; 0.78-0.85 | 3.3e-4 to 5.8e-4 |
- The f-plane's miss (seed 51): own count 6, D_x at 2.8e-5, the space scaling at 128^2 on the same data seed (4051)
  as exp61's f-plane maximum (2.8e-5).
- The time pencil's lowest eigenvalue is 3.8e-12 to 1.1e-7 over the 40 runs.
- r-hat (phi) is low by 3.5e-5 to 4.7e-5, 3.2e-4 to 3.4e-4 and 1.64e-3 to 1.67e-3 at rek 0.1, 0.3, 0.6. Not the
  class's truncation (independent check, on the saved pencils): the exact V_r's D-projection on the 14 time fields keeps
  its quotient to three figures and reads r 0.100000, 0.300000, 0.599990, while the deepest direction with tau(0) = 0
  sits below the exact V_r (5.9e-11 to 2.7e-10 against 9.1e-10 to 1.1e-9 at rek 0.6). The pencil follows a residual
  that the exact V_r leaves on these data; not resolved further. r-hat from tau's T^2 / T coefficients is 2.2-4.5 %
  high.
- lambda8 / lambda7: 274-6185, 943-2345, 82-328 at rek 0.1, 0.3, 0.6; 13-565 on the f-plane.
VERDICTS (literal; verdicts_0710.py): (DT1) MET: worst cell 9/10 (f). (DT2) MET: 10/10 in every row. (DT3) MET: 10/10.
(DT4) MET: 10/10. (DT5) MET: 10/10.
Reading, registered in advance: in a class with tau(t) and the q-coefficient to degree 6, FINS recovers from pyqg's
data, through the port, the drag's time scaling deformed into V_r, and reads the drag rate from the generator's slope,
without an exponential candidate. exp61's degree-2 class reported one generator fewer at drag. A ceiling row (exact
tangent), validated by the point equivalence of damped and inviscid flows. The registered "Lie's algorithm on the known
equation" was not run as such; the known equation's linearized symmetry condition holds for V_r: with
F = Lap psi_t + J(psi, Lap psi) + r Lap psi and Q_r = -e^(rt) psi - tau psi_t, F'[Q_r] = -2 e^(rt) F - tau F_t
(exp62_vr_check.py, sympy).

**exp60's controls and the linear-response baseline (exp67, exp67_controls.py) -- development and pre-registration
(10.10, night; the 09.10 area chair's mandatory baseline; written after the development runs below, before any run on
seeds 42-51).**
Design (aniso2d's solver and exp60's data and settings at a = 1.5: 64^2, 16 trajectories, 160 pairs):
- Equations: aniso (exp60); mismatch: eps 0.6 in the nonlinearity, 0.8 in the dissipation (no stretched rotation of
  the whole equation; the scaling x d_x + y d_y + a t d_t + (2 - a) u d_u kept); cubic: u_t = A(u^2 / 2 + gamma u^3) -
  nu (-A)^(a/2) u, gamma 0.2 (that scaling broken; the stretched rotation kept).
- FINS: exp60's pencil through the exact tangent, its scans and the sl(2) readout.
- Linear response (no data): the one-snapshot stepper's Jacobian at the constant states b in {0.25, 0.35, 0.45} is a
  Fourier multiplier exp(dt sigma(k; b)); sigma = sigma0 - b S - b^2 S2 through the three states; the dissipation's eps
  from sigma0 (argmin over e of its variation on the ellipses kx^2 + e^2 ky^2 = r^2, r in {1, 2, 3}, cubic
  interpolation) and its order (slope of log |sigma0| against log |k|_e on 0.5 <= |k|_e <= 6 inside the mask), the
  nonlinearity's eps from S, and |b^2 S2| / |b S| at the mean b.
Linear response (deterministic; computed during development, so not predicted):
| equation | dissipation eps | order | nonlinearity eps | b^2 term |
|---|---|---|---|---|
| aniso | 0.600 | 1.500 | 0.600 | 6.1e-5 |
| mismatch | 0.800 | 1.500 | 0.600 | 6.2e-5 |
| cubic | 0.600 | 1.500 | 0.600 | 0.21 |
Linear interpolation instead of cubic gave 0.598 and 0.593; the order fit needs the masked modes excluded.
FINS development (seed 7, data seeds 100 / 101; 82-134 s per run):
- aniso: rotation scan eps 0.600 (6.7e-8), scaling scan (1.502, 0.497) (4.2e-7), sl(2) 0.5999.
- mismatch: the best stretched rotation eps 0.648 at 3.5e-5, 76 times the largest kept generator (the scaling,
  4.6e-7); the rotation at 0.6 at 3.2e-4, at 0.8 at 3.6e-3; scaling scan (1.502, 0.497). With 1.0 in the dissipation
  (development only): eps 0.702 at 1.5e-4, 270 times.
- cubic: the scaling (1.5, 0.5) at 9.1e-5, 380 times the stretched rotation (2.4e-7); the scaling scan finds
  (1.610, 0.273) at 9.4e-7, 3.9 times the rotation: over the data's amplitudes A(u^2 / 2 + 0.2 u^3) is close to a power
  law with another exponent. With gamma 1.0 (development only): (1.5, 0.5) at 3.6e-4, the scan (1.630, 0.209) at
  1.6e-6.
Predictions (FINS, seeds 42-51):
(CT1) mismatch: the rotation scan's best quotient >= 30 times the largest of d_x, d_y, d_t and the scaling (1.5, 0.5),
and the scaling scan within 0.02 of (1.5, 0.5), on >= 9/10.
(CT2) cubic: the scaling (1.5, 0.5) >= 100 times the largest of d_x, d_y, d_t and the stretched rotation (0.6), the
scaling scan's a-hat more than 0.05 from 1.5, and the rotation scan's eps within 0.002 of 0.6, on >= 9/10.
Reading registered in advance:
- The linear response reads the dissipation's anisotropy and order in all three equations, with no data and no
  operator family, and the nonlinearity's anisotropy from its b-derivative. On exp60, FINS's exponent readout adds
  nothing to it.
- CT1-CT2 met: FINS's whole-equation readout adds that mismatch has no stretched rotation (the linear response shows
  two anisotropies, from which the same follows by an argument about the equation) and that cubic's best scaling has
  an exponent other than the linear order (the linear response keeps 1.5 and flags a b^2 term).

**Two-stage rows on pyqg's data without quasi-geostrophic knowledge (exp64, exp64_qg_libraries.py) -- development and
pre-registration (10.10, night; the 09.10 advocate's and applied mathematician's baselines; written after the
development runs below, before any run on seeds 42-51).**
Design (exp61's weak SINDy settings: pyqg's data at 128^2, the 16 training trajectories of data seed 4000 + s):
- obs: the reference library in q plus u, v and d_x^p d_y^r (f), f in {u q, v q, u^2, v^2, u v}, p + r <= 2, from
  pyqg's own velocity; kd2 from q = c1 Lap psi + c0 psi (pointwise least squares on pyqg's psi, every 7th point),
  kd2-hat = -c0 / c1.
- riesz: the reference library plus the generalized SQG (alpha-turbulence) completion, transport by
  u_s = grad^perp psi_s with psi_s = -(|k|^2 + kappa)^(-s/2) q, and psi_s, psi_s,x, psi_s,y (QG: s = 2,
  kappa = kd2). s and kappa profiled by the residual on s in {0.5, 1, ..., 3} x kappa in {0, 0.5, 1, 2, 4}, then golden
  section on s within +-0.25 and on kappa between the neighbouring grid values (kept at 0 if no worse); riesz-free
  fixes kappa = 0. nested: ref -> riesz-free -> riesz by exp61's rule (residual drop > 1.25).
- Scores: the true equation (the quasi-geostrophic terms, mapped from obs (d_x(u q) -> adv_x, v -> psi_x), with
  exp61's 2 % coefficients, s within 0.02 of 2, kappa or kd2-hat within 0.05 of the truth); the identified model's
  algebra by structure (exp61_wsindy.qg_algebra with the terms' weights at s), scored on the zero-mean states: the
  rotation and the scalings; the shift is reported, not scored (the reviewers' single ground truth).
Development (seed 7, data seed 100; 165-450 s per run):
| regime | obs | riesz-free | riesz | nested |
|---|---|---|---|---|
| f | true | true (s 2.0008) | true (s 2.0008, kappa 0) | riesz-free, true |
| beta | true (-0.5 v) | true (s 2.0008, psi_x -0.5002) | true (kappa 0) | riesz-free, true |
| rd | true (kd2-hat 1.000000) | s 1.235, advection -0.558 | true (s 1.9992, kappa 1.0016) | riesz, true |
| drag | true (-0.3 q) | true (s 2.0008) | true (kappa 0) | riesz-free, true |
- The first obs library also held the derivatives of u and v alone: at drag MSTLS split -0.3 q into +0.7885 q +
  1.089 (u_y - v_x), exact by curl u = q but not the term list. Those columns are collinear with q (div u = 0,
  curl u = q + kd2 psi) and were removed; f, beta, rd unchanged.
- At drag the identified -0.3 q breaks the shift by structure (as in WS5); on the zero-mean states the algebra is the
  regime's.
Predictions (seeds 42-51, every regime unless stated):
(QL1) obs: the true equation and the regime's algebra on >= 9/10.
(QL2) riesz: the true equation and the regime's algebra on >= 9/10.
(QL3) nested: the true equation on >= 9/10.
(QL4) rd: riesz-free is not the true equation on >= 9/10.
Reading registered in advance: QL1-QL4 met: without being told "quasi-geostrophic", the two-stage route reaches
pyqg's equation and its algebra in every regime, from pyqg's own velocity with a local library, or with a generic
two-parameter family of nonlocal transports whose exponent and screening the residual profiles to their true values.
exp61's reading that the two-stage route needs the quasi-geostrophic structure supplied does not hold. It also returns
beta, kd2 and rek, which FINS does not.

**FINS for systems: pyqg's flow in the velocity (u, v) (exp63, exp63_velocity.py) -- development and pre-registration
(10.10, night; the authors' request and the 09.10 reviews' design; written after the development runs below, before any
run on seeds 42-51).**
Design:
- State (u, v) = grad^perp psi of pyqg's data at 128^2 (exp61's settings: 16 training trajectories of data seed
  4000 + s, 8 held-out of 5000 + s, 80 pairs, K = 8). Operator: (u, v) -> q (the curl; with rd q_hat = zeta_hat
  (1 + kd2 / |k|^2)), the port's step, back to (u, v); its exact tangent (against central differences 2.4e-10).
- Two operators on the same data:
  - pyqg: pyqg's state space, whose mean velocity the curl annihilates;
  - meanflow: a conserved mean flow (U0, V0) that advects q (with rd also the PV gradient kd2 (U0 y - V0 x) it
    implies), uniform q tendencies dropped as pyqg's inversion drops the mean of q.
- Class: monomials of degree <= 2 in (X, Y, T, U, V) in each of xi, eta, tau, phi_u, phi_v (105 fields), one sigma for
  u and v.
- Algebra (pyqg's operator, localised data): f: d_x, d_y, d_t, R = -y d_x + x d_y - v d_u + u d_v,
  D_x = x d_x + y d_y + u d_u + v d_v, D_t = t d_t - u d_u - v d_v (6); beta: d_x, d_y, d_t,
  D_beta = x d_x + y d_y - t d_t + 2u d_u + 2v d_v (4); rd: f's without D_x (5); drag: f's without D_t (5). No shift.
  meanflow adds the constant boosts t d_x + d_u and t d_y + d_v at f and beta (the meridional one because uniform q
  tendencies are dropped; on the plane beta breaks it); with rd the PV gradient breaks both.
- Readouts: own count (floor 1e-6); the span of the k lowest against the algebra; the pencil restricted to the
  rotation space (y d_x, x d_y, v d_u, u d_v: the spin-to-orbital ratio of the deepest direction, 1 for the vector
  rotation), the scaling space (x d_x + y d_y, t d_t, u d_u, v d_v) and the boost space (t d_x, d_u, t d_y, d_v: the
  speeds of the two deepest directions, 1 for the boosts); controls: the scalar rotation, the spin alone, the opposite
  spin, u d_u - v d_v.
Development (seed 7, data seeds 100 / 101; 497-558 s per run on one thread):
- exp61's elliptic vortices (aspect 1-1.6) have quadrupole velocity tails that reach the seam: velocity rms within 1 of
  the seam over the overall rms 1.6e-2 (q 1.9e-7). Through pyqg's operator at f the vector rotation and D_x sat at
  4.8e-4 and 4.9e-4, inside a cluster of nearly inert polynomial flows (solid-body rotation y d_u - x d_v at 3.1e-4,
  shear flows at 5e-4 to 7e-4). Circular shielded vortices (aspect 1; otherwise exp61's initial conditions) have
  exponentially small velocity at t = 0 (seam ratio 4e-9) and 2.5e-3 on average over the window. exp63 uses them.
| row | own count (k) | missing (span of k) | vector rotation | D_x / D_beta | spin / orbital | scaling: deepest; next | boost space: two lowest |
|---|---|---|---|---|---|---|---|
| f, pyqg | 4 (6) | 0.000 | 2.7e-5 | D_x 3.4e-5 | 1.0000 | D_t 1.7e-10, D_x 5.0e-5; 0.85 | 0.89, 0.90 |
| f, meanflow | 6 (8) | 0.000 | 2.7e-5 | D_x 3.4e-5 | 1.0000 | as pyqg | 4.3e-13, 2.2e-12 (speeds 1.0000000) |
| beta, pyqg | 3 (4) | 0.998 | 5.2e-2 | D_beta 1.8e-3 | 0.9990 | 1.7e-3 (1, -1.01, 2.35, 2.35); 4.8e-2 | 0.70, 0.73 |
| beta, meanflow | 5 (6) | 0.006 | 5.2e-2 | D_beta 5.7e-4 | 0.9990 | 5.6e-4 (1, -0.98, 2.08, 2.08); 4.6e-2 | 3.0e-13, 2.3e-12 (1.0000) |
| rd, pyqg | 5 (5) | 0.000 | 6.5e-6 | D_x 7.0e-4 (broken) | 1.0000 | D_t 1.4e-10; 5.1e-4 | 0.94, 0.95 |
| rd, meanflow | 5 (5) | 0.000 | 6.5e-6 | D_x 5.2e-4 (broken) | 1.0000 | as pyqg | 1.3e-3, 1.4e-3 (speeds 1.18-1.20) |
| drag, pyqg | 5 (5) | 0.000 | 1.3e-6 | D_x 1.3e-6 | 1.0000 | D_x 1.3e-6 (1, 0, 1, 1); 6.2e-3 | 0.73, 0.73 |
- The scalar rotation, the spin alone, the opposite spin and u d_u - v d_v are at 0.68-1.0 in every row; d_t at
  1.9e-7 to 4.0e-7; the translations at 1e-12 to 5e-11.
- At f the blind count stops at 4: the rotation and D_x sit 17 times above the exact generators and 7 times below the
  nearly inert flows (3.4e-4).
- At beta, Rossby waves carry velocity to the seam. Through pyqg's operator the seam-induced mean of D_beta's
  characteristic is dropped (1.8e-3, velocity weight 2.35); the meanflow operator keeps it (5.7e-4, weight 2.08).
Predictions (pyqg's operator: seeds 42-51; meanflow: seeds 42-46):
(UV1) The span of the k lowest contains the regime's algebra (missing <= 0.05) on >= 9/10 at f, rd, drag (pyqg) and on
>= 4/5 at f and beta (meanflow).
(UV2) pyqg at f, rd, drag: the rotation space's deepest eigenvalue <= 1e-4, its spin-to-orbital ratio and stretch
within 0.01 of 1, and the scalar rotation, the spin alone and the opposite spin each >= 100 times that eigenvalue, on
>= 9/10; at beta the rotation space's deepest >= 1e-2 on 10/10.
(UV3) pyqg scaling space, on >= 9/10: f: the two deepest span (1, 0, 1, 1) and (0, 1, -1, -1) (missing <= 0.05);
rd: the deepest (w_x, w_u, w_v) / w_t within 0.02 of (0, -1, -1); drag: (w_t, w_u, w_v) / w_x within 0.02 of (0, 1, 1);
the next eigenvalue >= 10 times the last deep one. meanflow at beta, on >= 4/5: the deepest (w_t, w_u, w_v) / w_x within
0.15 of (-1, 2, 2), the next >= 10 times.
(UV4) pyqg's operator: the boost space's lowest eigenvalue >= 100 times the largest quotient of the regime's algebra,
on >= 9/10 in every regime.
(UV5) meanflow at f and beta: the boost space's two lowest <= 1e-6 with both speeds within 0.01 of 1, on >= 4/5; at rd
the lowest >= 30 times the algebra's largest quotient, on >= 4/5.
No predictions: own counts; beta through pyqg's operator (span, D_beta); the nearly inert polynomial flows.
Reading registered in advance:
- UV1-UV3 met: FINS for systems recovers pyqg's velocity algebra with the rotation acting on the components as a
  vector (spin to orbital 1, the scalar rotation and the opposite spin broken) and the scalings' velocity weights.
- UV4-UV5 met: FINS reads the Galilean boosts as a property of the operator's state space, broken by pyqg's zero-mean
  velocity, exact with a conserved mean flow, broken again by the PV gradient that a mean flow implies with rd.
- Through the exact tangent of the port these are ceiling rows; the velocity algebra is the image of q's under a linear
  map, except the boosts, which q does not see.

**exp60 readouts without the scans' priors -- results (10.10, night; exp60_readouts.py on the 30 saved exact pencils
and identified models; results/aniso2d/readouts_summary.{json,md}).**
- sl(2): the deepest traceless linear field is elliptic on 30/30. eps-hat 0.59983-0.59988 (a = 1.25), 0.59991-0.59994
  (1.5), 0.59997-0.59998 (1.75); the angle of its axes within 0.0013 degrees of 0; lambda2 / lambda1 8.4e4 to 1.8e6
  (lambda1 2.2e-8 to 3.4e-7).
- Reported without verdicts:
  - the scaling family with the shift d_u admitted: the ratio of the two lowest 87-113, 56-87, 19-47 (the registered
    family 410-1054); a-hat 1.259-1.272, 1.494-1.512, 1.710-1.752 (the registered family 1.255-1.261, 1.499-1.504,
    1.748-1.750);
  - the soft readout of weak SINDy's identified models: harm-free e 0.604-0.606, 0.603-0.605, 0.602 (variation at the
    minimum 7.0e-3 to 1.5e-2), its nonlinear part 0.5988-0.6000, a on its grid value on 30/30; dir-free e 0.561-0.592
    (variation 4.8e-2 to 0.14); iso-free no elliptic level sets (variation 0.19-0.29); ell-free and nested 0.600
    (variation below 1e-15).
VERDICTS (literal; verdicts_0710.py): (AR1) MET: 30/30. (AR2) MET: 30/30.
Reading, registered in advance: the stretched rotation's eps and the orientation of its axes come from the sl(2)
pencil, a family that contains every linear area-preserving field; the aligned-axes prior of the rotation scan is not
needed. Further (not registered):
- Read the same way, weak SINDy's harmonic completion gives e within 0.006 and its nonlinear part within 0.0012: on
  the stretched rotation, FINS through the exact solver and the two-stage route with a generic completion agree.
- Admitting d_u in the scaling family moves a-hat by up to 0.040 (a = 1.75, seed 45) and lowers the margin to 19; the
  registered scan's precision depends on excluding the near-symmetry t d_t - (u + nu) d_u.

**Notebook 9: an audit of PG8's emulators (exp65) and LieNLSD on exp60 (exp66) -- pre-registration (10.10, night; the
09.10 reviews' emulator audit and jet-space baseline; no development run: the emulators are on Google Drive and
LieNLSD's fit needs a GPU; the code was checked on an untrained FNO and two epochs).**
exp65 (exp65_emulator_audit.py), PG8's six FNO2d emulators (f and beta, seeds 42-44):
- fins: FINS through each emulator on its own rollouts. The emulator is rolled out from the initial states of PG8's
  FINS data (16 trajectories of data seed 4000 + s; held-out 8 of 5000 + s), so the pencil's data term follows the
  emulator and the quotients measure its symmetry defects. In PG8 the data were pyqg's, so every quotient carried the
  emulator's error (the post hoc rule of the PG8 entry). exp61's readouts and settings (80 pairs, K = 8).
- consequence: the emulator's one-step error against the port (two RK4 steps with pyqg's filter) on 40 held-out pyqg
  states (snapshots 10, 20, ..., 50 of the 8 held-out trajectories) after finite flows of the regime's generators, over
  its error on the untransformed states. Flows: shifts by half a cell (d_x, d_y), rotation by 30 degrees, x -> 1.25 x
  (D_x), q -> 1.25 q (D_t), x -> 1.25 x with q -> 1.25 q (D_beta); cubic splines about the box centre, then pyqg's
  filter, for both the emulator and the port.
exp66 (exp66_lienlsd_aniso.py): exp47's LieNLSD-2D as published (an MLP on jets to second order, central differences,
fitted to u_t; F's Jacobian in the determining system over the 60 fields) on exp60's 16 training trajectories per seed,
a in {1.25, 1.5, 1.75}, seeds 42-51; scored on exp60's 8 held-out trajectories. Readouts: the span of the 5 smallest
right singular vectors; with N = W'W and D = diag(sc^2) (LieNLSD's column scaling), exp60's scans and the sl(2)
readout. A fitted and an untrained F.
Predictions:
(EA1) Self-consistent: the rotation plane's lowest eigenvalue at beta >= 10 times that at f (same seed) on >= 2/3 seeds.
(EA2) Self-consistent: beta's deepest scaling direction has (w_t, w_u) / w_x within 0.1 of (-1, 1) on >= 2/3 seeds.
(EA3) Over the 24 pairs (emulator, generator) -- f: d_x, d_y, rotation, D_x, D_t; beta: d_x, d_y, D_beta -- the
Spearman correlation between the self-consistent quotient and the transformed-to-untransformed error ratio is >= 0.6.
(LS1) LieNLSD fitted: the scaling scan reads a-hat more than 0.1 from a on >= 8/10 seeds at each a.
(LS2) LieNLSD fitted: the span of the 5 smallest misses >= 0.5 of exp60's five-dimensional algebra on >= 8/10 seeds at
each a.
No predictions: PG8's (pyqg-data) quotients against the ratios; the translations' self-consistent quotients (the FNO's
aliasing; an untrained FNO gave 1.1e-4); the f-plane scaling space; the rollouts' distance from pyqg; LieNLSD's eps
(scan and sl(2)), own count and the untrained F.
Reading registered in advance:
- EA1-EA2 met: the emulators keep the regime's pattern as dynamical systems, and PG8 failed because the pencil on
  pyqg's data adds the emulator's error to every quotient. Not met: the emulators did not learn the pattern.
- EA3 met: FINS's per-generator defects, read from the emulator alone, rank its errors on transformed inputs.
- LS1-LS2 met: the jet-space learner fits a local surrogate and loses the dissipation's exponent, which FINS reads
  through the solver and through FNO2d steppers. The linear response of the operator (exp67) reads it too.

**exp60 readouts without the scans' priors (exp60_readouts.py) -- development and pre-registration (09.10, night; the
reviewers' B5 and the information-matched readouts of 09.10; written after the development readout on seed 7, before
any readout of the sl(2) pencil on seeds 42-51).**
Design: from exp60's saved exact pencils (60 fields; fins_runs/aniso2d_runs), no new runs.
- sl(2): the pencil restricted to the traceless linear fields (x d_x - y d_y, y d_x, x d_y), with no aligned axes. The
  deepest field M = [[a, b], [c, -a]] (xi = a x + b y, eta = c x - a y) is elliptic when det M > 0, and its invariant
  form S = [[1, -a / c], [-a / c, -b / c]] gives eps-hat = sqrt(min / max eigenvalue of S) and the angle of the axis
  of S's smaller eigenvalue (0 for the truth).
- Reported without verdicts (post hoc; computed by the reviewers on seeds 42-44 or 42-51 and checked by us on 09.10):
  - the scaling family with the shift d_u admitted (x d_x + y d_y, t d_t, u d_u, d_u) against the registered one;
  - the soft readout of weak SINDy's identified models (eps from the nonlinear part; argmin over e of the linear
    symbol's variation on kx^2 + e^2 ky^2 = r^2, r in {0.5, 1, 2, 4}, e in [0.2, 2] step 0.001).
Development (seed 7): sl(2) eps-hat 0.5998, 0.5999, 0.6000 and angle -0.0005, -0.0003, -0.0001 degrees at a = 1.25,
1.5, 1.75; lambda2 / lambda1 3.1e5, 6.3e5, 1.4e6; elliptic. The shift-admitting family: ratio 84, 74, 46 (440, 605,
1099 without d_u), a-hat 1.275, 1.514, 1.753. Soft: harm-free e 0.606, 0.604, 0.602 (e_n 0.5989-0.5996).
Predictions (seeds 42-51, each a):
(AR1) The deepest sl(2) field is elliptic, eps-hat within 0.005 of 0.6 and its angle within 0.5 degrees of 0, on
>= 9/10 seeds.
(AR2) The sl(2) pencil's lambda2 / lambda1 >= 1000 on >= 9/10 seeds.
Reading registered in advance: AR1-AR2 met: the stretched rotation's eps and the orientation of its axes come from a
family that contains every linear area-preserving field, so the aligned-axes prior of exp60's rotation scan is not
needed. It remains a family of linear fields, read through the exact solver.

**The drag's deformed time scaling (exp62, exp62_drag_tau.py) -- development and pre-registration (09.10, night; the
applied mathematician's proposal of 09.10; written after the development runs below, before any run on seeds 42-51).**
Design:
- pyqg's drag regime of exp61 at rek in {0.1, 0.3, 0.6}, and the f-plane as the control. pyqg's data at 128^2 with
  exp61's settings (16 training trajectories of data seed 4000 + s, 8 held-out of 5000 + s, 80 pairs, K = 8); the
  port's exact tangent.
- Class: exp45's 60 fields plus tau T^3..T^6 and phi T^2 U..T^6 U (69 fields): tau(t) and the coefficient of q to
  degree 6 in t. The exact V_r characteristic (r = rek; 0.3 on the f-plane) enters the pencil as a 70th field, for its
  quotient only (in the 69-field pencil's normalisation).
- Algebra in this class: d_x, d_y, d_t, shift, rotation, D_x and V_r = ((e^(rt) - 1) / r) d_t - e^(rt) q d_q, the
  latter through its degree-6 truncation; on the f-plane D_t in place of V_r. 7 in every row.
- Readouts:
  - own count by exp49's rule (floor 1e-6, 17 eigenvalues);
  - the span of the 7 lowest against the algebra with the exact V_r characteristic (missing, held-out
    characteristics);
  - the pencil restricted to the 14 time fields (tau T^0..T^6, phi U T^0..T^6): its two deepest eigenvectors combined
    to tau(0) = 0 and normalised to phi = -q at t = 0. r-hat = -(the phi T U coefficient) / T_max, the slope of
    e^(rt). r-hat from tau's T^2 / T coefficients is reported.
Development (seed 7, data seeds 100 / 101; 69 fields, 80 pairs, 484-515 s per run on one thread):
| row | own count | V_r exact | D_t | time pencil: 2nd; 3rd | r-hat (phi) | r-hat (tau) | missing (span 7) |
|---|---|---|---|---|---|---|---|
| drag 0.1 | 7 | 2.5e-9 | 2.6e-2 | 4.6e-7; 0.74 | 0.1000 | 0.1023 | 0.000 |
| drag 0.3 | 7 | 6.5e-10 | 8.9e-2 | 4.4e-7; 0.64 | 0.2997 | 0.3097 | 0.000 |
| drag 0.6 | 7 | 1.1e-9 | 0.12 | 4.2e-7; 0.47 | 0.5983 | 0.6269 | 0.000 |
| f | 7 | V_0.3: 0.21 | 4.4e-9 | 3.7e-7; 0.75 | 0.0000 | 0.0000 | 0.000 |
- The 8th eigenvalue: 1.5e-3, 7.1e-4, 1.1e-4, 3.8e-4; the 7th: 6.6e-7, 4.4e-7, 4.2e-7, 2.2e-6 (D_x on the f-plane).
- The span of the own count gives the same r-hat to four decimals.
- A smoke run at 64^2 (2 trajectories): V_r 1.3e-5 and D_x 1.6e-3 (the grid scale, as in exp61).
Predictions (seeds 42-51, in every row unless stated):
(DT1) Own count (floor 1e-6) = 7 on >= 9/10 seeds.
(DT2) The span of the 7 lowest contains the regime's algebra (the exact V_r, or D_t on the f-plane): missing <= 0.05
on >= 9/10.
(DT3) r-hat (phi) within 0.01 of rek (of 0 on the f-plane) on >= 9/10.
(DT4) Drag rows: the exact V_r's quotient <= 1e-6 and D_t's >= 1e-2; f-plane: V_0.3's >= 1e-2; each on >= 9/10.
(DT5) Time pencil: two eigenvalues <= 1e-6 and the third >= 1e-2 on >= 9/10.
No predictions: r-hat (tau), the 8th eigenvalue.
Reading registered in advance:
- DT1-DT5 met: in a class with tau(t) and the q-coefficient to degree 6, FINS recovers from pyqg's data, through the
  port, the drag's time scaling deformed into V_r, and reads the drag rate from the generator's slope, without an
  exponential candidate. exp61's degree-2 class reported one generator fewer at drag.
- A ceiling row (exact tangent) and a validation: the point-equivalence of damped and inviscid flows (q = e^(-rt) w,
  s = (1 - e^(-rt)) / r) gives V_r, and Lie's algorithm on the known equation returns it too.

**exp60 and exp61 -- corrections from the mock reviews of 09.10 (round 2; three reviewers: ML area chair, applied
mathematician, advocate of the two-stage route), each checked by us before it is written (09.10, night; scripts outside
the repository: check_reviewer_claims.py and the reviewers' own scripts rerun; nothing registered).**
exp61 (pyqg):
- The drag deforms the time scaling; it does not remove it. The damped flow q_t + J(psi, q) = -r q is point-equivalent
  to the inviscid one (q = e^(-rt) w, tau = (1 - e^(-rt)) / r), and V_r = ((e^(rt) - 1) / r) d_t - e^(rt) q d_q is a
  symmetry. pyqg's drag -rek Lap psi leaves the mean of q alone, so the equivalence holds on the zero-mean states, which
  every trajectory keeps. Check (drag, seed 42, pyqg's data, the port's exact tangent, the 80 pairs of the registered
  run): V_r 2.6e-10 in the registered pencil's normalisation, against d_t 2.1e-7 and t d_t - q d_q 9.9e-2 (the json's
  values); with the three characteristics alone, 5.3e-10, 4.0e-7 and 0.157.
  - V_r is outside the degree-2 class, so the class reports one generator fewer at drag. Seed 42 (independent
    check): the pencil restricted to {d_t, t d_t, t^2 d_t, q d_q, t q d_q} has its second eigenvalue at 1.7e-3 (1.9e-4
    without d_t; the least-squares approximant of V_r in the class 2.0e-4); the class's seventh eigenvector is not V_r
    (|cos| 0.06, 0.17 modulo the six lowest).
  - The registered reading's "the drag breaks the time scaling" (pre-registration, qg2d.py's docstring) should read:
    the drag breaks the polynomial time scaling t d_t - q d_q; the damped flow stays point-equivalent to the inviscid
    one, and its time scaling becomes V_r, outside the degree-2 class.
- The shift d_q is given by the code, not measured: its row of the pencil is zero for any data.
  - Without rd it is a convention of the periodic inversion (uniform q carries no flow). Its counterpart on the plane is
    the rotating-frame generator, t(-y d_x + x d_y) + 2 d_q in q (Bihlo and Popovych 2009), which the periodic box
    breaks except for the d_q part.
  - With rd it is a genuine symmetry, and the stated reason was wrong there. The inversion is psi_hat = a q_hat with
    a(0) = -rd^2, so the mean of q gives a uniform psi, which carries no velocity and drops out of beta psi_x and the
    drag.
  - The measured part of the algebras is 6 / 4 / 5 / 5. The translations are near-structural on the spectral grid and
    d_t comes with any autonomous operator. The non-trivial content is: rotation, D_x, D_t at f; D_beta at beta;
    rotation, D_t at rd; rotation, D_x at drag (3 / 1 / 2 / 2).
- WS5's contrast at drag is not a FINS advantage. The identified -0.30 q and pyqg's -rek Lap psi = -0.30 (q - <q>)
  coincide on every zero-mean state, hence on all the data. FINS reads the shift from the code's k = 0 convention, not
  from the data. Through PG8's emulators (f and beta; none at drag), trained on zero-mean data too, its quotient is
  5.5e-4 to 4.2e-2 (entry below). On the zero-mean states the identified equation's algebra is the regime's. The
  contrast becomes a gauge remark.
- The regime table is a validation with ceiling rows. Which parameter breaks which generator follows from dimensional
  analysis (beta ~ 1/(L T), rd ~ L, rek ~ 1/T) and from Lie's algorithm on the known equation. The surviving scalings'
  weights are integers fixed by the same analysis: "exponents to 0.003" is a calibration of the readout (beta's
  direction is (1, -1, 1) to 0.0032, i.e. within 0.004).
- Wording:
  - "pyqg's discretization differentiated exactly" is the exact tangent of an RK4 port of pyqg's spatial
    discretization and filter (pyqg integrates with AB3);
  - "pressure" is the streamfunction;
  - the configuration (barotropic BTModel, 2-3 shielded localised vortices, unforced) is not the ML benchmark's
    (Ross et al. 2023: the two-layer model in eddy and jet regimes, where only the translations, and the code's
    shifts of q, survive). Say so in the same sentence.
- Own count by floor (from the json): the registered 1e-6 gives 39/40; exp60's registered 1e-12 would give 22/40
  (f 3/10, beta 0/10, rd 10/10, drag 9/10). The 1e-6 floor of exp61 was chosen after exp60's post hoc analysis.
exp60 (aniso2d):
- The sixth near-symmetry is t d_t - (u + nu) d_u, not t d_t - u d_u (development entry). At a = 2 the equation is
  u_t = (1/2) A((u + nu)^2), the anisotropic porous-medium equation in w = u + nu, whose second scaling t d_t - w d_w is
  exact. For a < 2 its defect, nu [(-A) - (-A)^(a/2)] w, vanishes as a -> 2.
  - Saved exact pencils, seeds 42-43: t d_t - (u + nu) d_u at 2.3e-6 to 3.0e-5 (lambda6 1.5e-6 to 2.3e-5), against
    5.5e-4 to 7.9e-4 for t d_t - u d_u.
  - The alignment of the 6-span's direction outside the algebra with t d_t - (u + c) d_u depends on the metric
    (seeds 42-46). Euclidean on the scaled coefficients: |cos| 0.984-0.996 at a = 1.75 (best c 0.28-0.32), 0.87-0.94
    at 1.5, 0.67-0.82 at 1.25, against 0.24-0.43 with c = 0. In the characteristic metric: 1.000, 0.999-1.000 and
    0.996-0.997 (best c 0.31-0.35, 0.37-0.51, 0.48-0.60), against 0.95-0.99 with c = 0. The quotients above do not
    depend on a metric.
  - It is a near-symmetry of the limit equation, as Burgers' inviscid scaling and Benjamin-Ono's projective field are.
    It predicts a band near a = 2 where the algebra jumps from 5 to 6.
- The scaling scans exclude it by construction: their family x d_x + y d_y + a t d_t + c u d_u has no d_u. With the
  shift admitted (exact pencils, seeds 42-44), the ratio of the two lowest restricted eigenvalues falls from 429-545,
  562-687 and 777-867 to 88-100, 74-77 and 32-37 at a = 1.25, 1.5 and 1.75; the deepest direction reads a-hat
  1.267-1.270, 1.506-1.510 and 1.737-1.746. Through the FNOs (generators at 1.3e-6 to 9.6e-6) this margin is not
  known: the FNO pencils were not retrieved.
- The stretched rotation is scored asymmetrically. An identified model has it only if its linear symbol is constant on
  ellipses to 1e-6 relative (model_algebra); FINS's eps-hat is an argmin. Read by the same argmin (the symbol's
  variation on kx^2 + e^2 ky^2 = r^2, r in {0.5, 1, 2, 4}, e in [0.2, 2] step 0.001), harm-free gives e 0.602-0.606
  (variation at the minimum 7.0e-3 to 1.5e-2), and its nonlinear part sqrt(c_yy / c_xx) 0.5988-0.6000; its exponent is
  on the true value on 30/30. "Harmonics: exact exponent but no rotation" is the hard readout's verdict, not the
  route's.
- model_algebra returns no rotation without testing when the nonlinear part has terms other than d_xx(u^2),
  d_yy(u^2) and undifferentiated u^j: the reference models on 20/30 runs. Their "no rotation" is "untested".
- The profile grids contain the true parameters, without refinement (exp60: e 0.3:0.05:1.0, beta 0.5:0.05:4.0;
  exp61: kd2 0:0.25:4, and qg_true_eq requires |kd2 - 1| <= 1e-9). "Exact exponent", e 0.600 (ell-free) and kd2 1.0
  are partly grid alignment.
Not corrected here (open): the completeness proofs of the exp60 algebra within the class (the applied mathematician
sketches one); the FNO pencils of exp60 for the box-edge question; the asymmetries above in the verdict code
(registered; unchanged).

**Quasi-geostrophic flow from pyqg (exp61) -- PG8, notebook 8 (09.10, night; FNO2d emulators trained in Colab by PG8's
recipe: 256 pyqg trajectories of data seed 6000 + s at 128^2, width 32, modes 24, 4 layers, 40 epochs, about 1 h each
on a GPU; json, npz and logs merged into fins_runs/pyqg_runs, logs in logs_nb8; results/pyqg/summary.{json,md}
regenerated).**
- Emulators: one-step rollout error relative to the data increment 9.4-10.4 % (f) and 4.2-5.9 % (beta) on FINS's 16
  trajectories (data seed 4000 + s, unseen by the FNO), 7.3-10.5 % and 3.5-5.5 % on the held-out ones: 17-53 times
  exp60's FNO2d (0.20-0.26 %). Final training loss 3.9e-4 to 4.3e-4 (f) and 1.5e-4 to 1.7e-4 (beta), still falling
  slowly at epoch 40; exp60's ended at 6.2e-6 to 7.9e-6 after 60 epochs on 512 trajectories.
- FINS through the FNOs' tangents (pyqg's data; the same 16 trajectories and 80 pairs as the exact rows), seeds 42-44:
  - f: the kept candidates at 3.1e-3 to 1.7e-2 (exact port at most 4.0e-6 on these seeds), D_t the largest. The
    shift at 1.3e-3 to 3.1e-2: through the FNO it is no longer structural.
  - beta: the kept d_x, d_y, d_t, D_beta at 1.3e-3 to 3.8e-3; the shift 5.5e-4 to 4.2e-2. The broken candidates sit
    at 1.08-1.28 times their exact-port quotients on the same seed: rotation 7.9e-3 to 1.6e-2, D_x 5.4e-3 to 1.1e-2,
    D_t 5.1e-2 to 7.9e-2 (D_t 18-42 times the largest kept candidate other than the shift).
  - Rotation plane, lowest eigenvalue: f 3.1e-3 to 4.2e-3 (exact 8.8e-8 to 3.8e-7), stretch 0.990-0.996; beta 7.8e-3 to
    1.55e-2 (exact 6.4e-3 to 1.4e-2). Ratio beta / f per seed 3.5, 3.7, 2.6.
  - Scaling space: f two deep directions (3.5e-3 to 5.7e-3 and 6.2e-3 to 8.3e-3; next 0.64-0.90), whose span misses
    0.009, 0.061, 0.037 of D_x and D_t. beta one deep (1.5e-3 to 2.8e-3; next 2.3e-2 to 3.6e-2), (w_t, w_u) / w_x =
    (-0.955, 1.164), (-0.919, 1.557), (-0.904, 1.156): the time weight within 0.1 of -1, the q weight 0.16-0.56 off.
  - Spectrum: the lowest eigenvalue 3.3e-4 to 1.5e-3 (f) and 2.0e-4 to 5.5e-4 (beta), below every candidate; own count
    (floor 1e-6) 9, 1, 1 (f) and 6, 1, 1 (beta); the span of the k lowest misses 0.068-1.07 (f) and 0.20-0.31 (beta).
- Post hoc, not registered: in both learned cases the kept generators' quotients (the shift aside) sit near the
  square of the one-step error. Medians 3.6e-3 to 5.5e-3 against 8.9e-3 to 1.1e-2 (f), 1.6e-3 to 3.2e-3 against
  1.8e-3 to 3.5e-3 (beta); in exp60 1.7e-6 to 2.4e-6 against 3.9e-6 to 6.5e-6. Two cases and 15 runs, over nearly four
  decades.
VERDICT (literal; verdicts_0710.py): (PG8) NOT MET: rotation plane at beta >= 10 times f on 0/3 (2.6-3.7); beta's
deepest scaling direction within 0.1 of (-1, 1) on 0/3 (the q weight); f's two deepest scaling directions span D_x and
D_t on 3/3 (missing <= 0.061).
Reading, registered in advance: "PG8 decides whether a learned emulator of pyqg keeps the pattern."
- It does not, at these emulators' accuracy. The beta-plane's breaking is read at its exact level through the FNO (the
  broken quotients 1.08-1.28 times the port's), and above beta's kept candidates (the shift aside). But the emulators
  break the generators they should keep at 1.3e-3 to 1.7e-2 themselves, the f-plane rotation at 3.1e-3 to 4.2e-3, so
  the registered contrast with the f-plane fails.
- What survives: the f-plane's two-dimensional scaling space (3/3); at beta the time weight of the surviving scaling
  and the order of the broken candidates above the kept ones (the shift aside).
- exp61 has no learned-operator result. Its FINS rows are ceiling rows through the exact tangent of the port.
- Not registered: whether emulators with errors near 1 % keep the pattern. By the post hoc rule above, a contrast of
  10 for beta's rotation (6.4e-3 to 1.4e-2) needs one-step errors below 2.5-3.7 %, and for rd's broken candidates
  (from 2.3e-3) below about 1.5 %.

**Anisotropic nonlocal dissipation in 2D (exp60) -- AN5, notebook 7 (09.10, evening; FNO2d steppers trained in Colab by
exp45's recipe, 512 trajectories of data seed 6000 + s, 60 epochs, about 50 min each; json and logs merged into
fins_runs/aniso2d_runs, logs in logs_nb7; the first Colab run had failed on import, see the results entry of 09.10
night).**
- Steppers: one-step rollout error relative to the data increment 0.20-0.26 % on FINS's 16 trajectories (data seed
  4000 + s, unseen by the FNO, which trained on 6000 + s) and 0.15-0.18 % on the held-out ones; over K = 8 steps
  0.12-0.14 % and 0.08-0.11 %.
- FINS through the FNO's tangent, pencil scans:
  - eps-hat 0.600 on 9/9;
  - a-hat 1.238-1.256 (a = 1.25), 1.482-1.500 (1.5), 1.726-1.743 (1.75): at most 0.024 off;
  - c-hat 0.741-0.770, 0.499-0.527, 0.259-0.284 (a + c 1.997-2.010).
- Spectrum through the FNO:
  - the five generators at quotients 1.3e-6 to 9.6e-6 (d_t the largest), up from at most 1.1e-6 through the exact
    solver on the same seeds. That is the band of the sixth near-symmetry, which the exact spectrum also has
    (lambda6 1.4e-6 to 2.3e-5);
  - the sixth eigenvalue 6.3e-6 to 2.5e-5, below d_t's quotient on 4/9 runs; lambda6 / lambda5 1.1-3.6 (exact 15-157);
    the seventh 2.2e-4 to 3.3e-4;
  - exp60's own count is 6 on 9/9: the algebra plus that extra direction. The span of the 6 lowest contains the algebra
    (missing <= 0.0011), and its deepest scaling direction reads a-hat 1.241-1.291, 1.517-1.544, 1.765-1.776;
  - the span of the 5 lowest, the readout of the exact entry, misses 0.007-0.019, 0.036-0.84 and 0.987-0.995 at
    a = 1.25, 1.5, 1.75 (losing d_t at 1.75). Its scaling direction is more than 0.1 off on 9/9: 0.74-0.88; 1.26, 0.73,
    38.9; 4.1-5.0.
- Not checked: whether the scans' refinement box clips some FNO runs, as it did four exact runs. The FNO pencils (npz)
  were not retrieved; c-hat 0.770 (a = 1.25, seed 44) and 0.510 (1.5, 43) may sit on a box edge. It cannot change AN5.
VERDICT (literal; verdicts_0710.py): (AN5) MET: eps-hat within 0.05 and a-hat within 0.1 on 3/3 seeds at every a (in
fact within 0.0005 and 0.025).
Reading, registered in advance:
- AN5 met: the case survives replacing the exact tangent by an FNO2d's, trained on 512 clean simulated trajectories.
  FINS reads the stretched rotation's eps to the scan's step and the dissipation's exponent within 0.025, with no
  family of candidate dissipation operators. The scans search fixed generator families that contain the true
  generators, the counterpart of ell-free's elliptic family.
- With AN1-AN4: the exponent needs an angular family and the rotation the family that contains it. FINS reads both from
  a learned operator.
- The residual still selects the elliptic family once it is offered (AN4 entry). The scans search fixed generator
  families.

**Quasi-geostrophic flow from pyqg (exp61) -- results (09.10, afternoon and evening; 70 FINS runs and 40 weak SINDy runs
on two CPU workers, none failed; json in fins_runs/pyqg_runs and pyqg_ws_runs; results/pyqg/summary.{json,md},
wsindy_summary.{json,md}; every number below recomputed from the raw json by an independent check). PG8: notebook 8,
not met (entry of 09.10, night); corrections from the mock reviews in the entry of 09.10, night.**
FINS, q, pyqg's data at 128^2, through the port's exact tangent; seeds 42-51:
| regime | own count = k | missing | kept candidates (largest) | broken candidates (smallest) | rotation plane | scaling space: deep; next | deepest direction |
|---|---|---|---|---|---|---|---|
| f (k = 7) | 9/10 | 0.000 | D_x 8.8e-7 to 2.8e-5 | none | 8.8e-8 to 3.8e-7 | 2 deep (to 3.0e-5); next 0.68-0.97 | span D_x, D_t |
| beta (5) | 10/10 | 0.000 | D_beta 7.2e-6 to 2.0e-5 | D_x 4.3e-3, rotation 6.4e-3, D_t 4.2e-2 | 6.4e-3 to 2.0e-2 | 7.2e-6 to 2.0e-5; next >= 2.88e-2 | (1, -1, 1) to 0.0032 |
| rd (6) | 10/10 | 0.000 | d_t 3.8e-7 to 4.3e-7 | D_beta 2.3e-3, D_x 2.8e-3 | 3.7e-11 to 2.1e-9 | 1.6e-9 to 9.5e-8; next >= 2.3e-4 | (0, 1, -1) to 0.0012 |
| drag (6) | 10/10 | 0.000 | d_t 2.1e-7 to 2.8e-7 | D_beta 4.5e-3, D_t 9.1e-2 | 7.3e-9 to 3.7e-8 | 1.2e-8 to 2.0e-7; next >= 6.09e-3 | (1, 0, 0) to 0.0009 |
- The shift d_q is at 0 in every run. It is structural: the port's inversion ignores the mean of q, so the shift's row
  of the pencil is zero whatever the data. (Corrected 09.10, night: this reason holds without rd; with rd the mean of q
  gives a uniform psi, which carries no flow.)
- d_x and d_y are at 2.2e-8 or below; d_t at 2.1e-7 to 4.7e-7 (pyqg's AB3 data against the RK4 map).
- Broken over the largest kept, per run: at least 525 (beta), 5928 (rd), 15853 (drag).
- f-plane, seed 51: own count 6. The seventh eigenvalue, 3.3e-5 (the space scaling), is 33 times the floor; the eighth
  is 13 times the seventh. Without pyqg's filter (filterfac 0 in data and port; independent check) its D_x falls from
  2.8e-5 to 2.4e-6 and the own count is 7.
Secondary rows:
- n = 64, f-plane: D_x 2.5e-3 to 1.2e-2, against 8.8e-7 to 2.8e-5 at 128^2; the translations 3.6e-8 to 4.8e-7 (no
  dealiasing); D_t 7.8e-6 to 4.9e-5; own count 6 on 10/10 (D_x out). The grid scale breaks D_x, pyqg's filter included:
  without the filter (seed 42; independent check) D_x falls only from 3.9e-3 to 9.7e-4, and aliasing lifts d_x, d_y and
  the rotation to 4.5e-4 to 1.3e-3.
- psi (pyqg's zero-mean gauge), f-plane: D_x = x d_x + y d_y + 2 psi d_psi 0.58 to 0.82 and D_beta 0.46 to 0.73, both
  broken by the gauge; D_t 3.1e-11 to 9.6e-11; rotation 5.3e-5 to 8.9e-5; the shift 1.0 (gauge-fixed).
- psi, rd (screened): own count 6 and missing 0.000 on 10/10; the shift d_psi at 0; D_x 1.0e-3 to 1.7e-3 against
  the largest kept 4.4e-7 (d_t).
Weak SINDy (pyqg's data at 128^2, the same 16 training trajectories per seed; seeds 42-51): true equation per row:
| regime | ref | qg-free | qg-rd | nested |
|---|---|---|---|---|
| f | 0/10 (residual 0.76-0.955) | 10/10 | 10/10 (kd2 0) | 10/10 (qg-free) |
| beta | 0/10 (0.58-0.70) | 10/10 | 10/10 (kd2 0) | 10/10 (qg-free) |
| rd | 0/10 (0.75-0.95) | 0/10 (advection -0.55 to -0.63, residual 0.16-0.25) | 10/10 (kd2 1.0) | 10/10 (qg-rd) |
| drag | 0/10 (0.19-0.27: -0.30 q alone on 9/10) | 10/10 | 10/10 (kd2 0) | 10/10 (qg-free) |
- The reference returns local surrogates of 1 to 41 terms (f, beta, rd). At drag it returns the drag term alone on
  9/10 runs; seed 44 adds +0.031 d_x(q^3). Its algebra is never the regime's.
- The true models have coefficients within 2.1e-5 relative and right-hand-side errors 1.6e-7 to 2.8e-5.
- Their algebras by structure equal the regime's at f, beta and rd (30/30). At drag the identified -0.30 q breaks the
  shift (10/10), which FINS reads at 0 from the operator: pyqg's drag -rek Lap psi leaves the mean of q alone, and data
  of zero mean cannot tell the two apart.
VERDICTS (literal; verdicts_0710.py):
- (PG1) MET: missing 0.000 in every run.
- (PG2) MET: 10/10 at beta, rd and drag; 9/10 at f (seed 51: 6).
- (PG3) MET.
- (PG4) MET.
- (PG5) MET.
- (PG6) MET.
- (PG7) MET.
- (PG8) NOT MET (notebook 8; entry of 09.10, night).
- (WS1) MET: ref 0/10 and a wrong algebra on 10/10 in every regime.
- (WS2) MET: 10/10 at f, beta and drag; 0/10 at rd.
- (WS3) MET: 10/10 in every regime.
- (WS4) MET: 10/10 in every regime.
- (WS5) MET: the regime's algebra on 30/30 at f, beta, rd; at drag the true equation breaking the shift on 10/10.
Reading, registered in advance:
- PG1-PG5 met. From pyqg's own data, through pyqg's discretization differentiated exactly, FINS recovers each regime's
  point algebra and which generators each physical parameter breaks: own count = dimension on 39/40 runs, the algebra in
  the span on 40/40. It also gives the surviving scaling's exponents to 0.003, among them the beta-plane scaling
  x -> lambda x, t -> t / lambda, q -> lambda q.
- PG6 met: at 64^2 the grid scale is a length scale FINS sees. The registered reading named pyqg's filter; the filter
  is only part of it (64^2 row above).
- PG7 met: in the pressure, the zero-mean gauge breaks the space scaling. With rd, the screened pressure has the regime's
  six generators, the shift d_psi included.
- WS1-WS5 met. The two-stage route reaches the equation only when the quasi-geostrophic structure is supplied: the
  Biot-Savart transport, the beta multiplier, and the screening with its parameter profiled. (Narrowed by exp64, 10.10:
  the generalized-SQG family with its exponent profiled suffices; pyqg's velocity in a local library too, post hoc.)
  Supplied, it is exact, and
  its algebra by structure is the regime's at f, beta and rd (30/30). At drag the fitted -0.30 q breaks the shift.
- FINS is given pyqg's discretization as an operator with its exact tangent, so it needs no term library. Its reading
  of the shift at drag comes from that operator: pyqg's inversion and drag leave the mean of q alone. The equation
  fitted to zero-mean data cannot see this.
- Not an identification win over a residual that is offered the right family. The learned rows (PG8) decide what FINS
  reads from data alone. (PG8 not met: entry of 09.10, night.)

**Quasi-geostrophic flow from pyqg (exp61), weak SINDy arm -- development and pre-registration (09.10, afternoon; the
question whether weak SINDy runs on pyqg; written after the development runs below, before any weak SINDy run on
seeds 42-51; the FINS runs of exp61 were under way).**
Design (exp61_wsindy.py): weak SINDy in 2D (pde/method_wsindy2d.py, the authors' algorithm on three axes) on the same
16 training trajectories per seed as FINS (pyqg's data, 128^2, 4624 rows). Rows:
- ref: the local library d_x^p d_y^q (q^j), j <= 3, p + q <= 4, and a constant;
- qg-free: ref plus the quasi-geostrophic completion d_x(u q), d_y(v q), psi, psi_x, psi_y, with psi = Lap^-1 q,
  u = -psi_y, v = psi_x;
- qg-rd: the same with psi = (Lap - kd2)^-1 q, kd2 profiled over 0..4 (step 0.25) by the residual;
- nested: the next row only when the residual drops more than 1.25 times.
True equation: exactly the terms -d_x(u q) - d_y(v q), plus -0.5 psi_x (beta), -0.3 q (drag), with the screening kd2 = 1
(rd, 0 elsewhere), coefficients within 2 %. Each model's algebra by structure (qg_algebra):
- rotation: equal advection coefficients, no psi_x or psi_y, isotropic local symbols;
- shift: no q^j beyond linear derivatives, divergence-free advection, no screened psi;
- scalings: the null space of the weight equations of every term against q_t.
Development (seed 7, data seed 100):
| regime | ref | qg-free | qg-rd | nested |
|---|---|---|---|---|
| f | wrong, residual 0.89 | true, eF 1.4e-5 | true (kd2 0) | qg-free |
| beta | wrong, residual 0.69 | true, -0.5 psi_x | true (kd2 0) | qg-free |
| rd | wrong, residual 0.92 | wrong, coefficients -0.60 | true (kd2 1.0) | qg-rd |
| drag | wrong | true, -0.3 q | true (kd2 0) | qg-free |
- The algebras by structure are right wherever the equation is, except at drag.
- At drag, the identified -0.3 q acts on the mean of q, so it breaks the shift. pyqg's -rek Lap psi does not act on the
  mean, and zero-mean data cannot tell the two apart. FINS reads the shift from the operator (quotient 0).
- About 200 s per run (one thread).
Predictions (seeds 42-51):
(WS1) ref: the true equation on 0/10 in every regime; its algebra differs from the regime's on >= 9/10.
(WS2) qg-free: the true equation on >= 9/10 at f, beta and drag; not at rd on >= 9/10.
(WS3) qg-rd: the true equation on >= 9/10 in every regime.
(WS4) nested: the true equation on >= 9/10 in every regime.
(WS5) nested: the true equation with the regime's algebra on >= 9/10 at f, beta and rd. At drag, the true equation
breaking the shift on >= 9/10.
Reading registered in advance:
- The two-stage route reaches the equation, and through it FINS's algebra, only when offered the quasi-geostrophic
  completion. That means the Biot-Savart transport, the beta multiplier, and the screened inversion with its parameter.
- With the local library, the route gives local surrogates.
- FINS needs none of this structure, and it reads the operator's action on the mean of q, which the data cannot show the
  equation.

**Quasi-geostrophic flow from pyqg (exp61) -- pre-registration (09.10; written after the development runs of the entry
below, before any run on seeds 42-51).**
Design as developed (qg2d.py, exp61_pyqg.py):
- Model: pyqg's barotropic model (BTModel, pyqg master from GitHub), ported to torch with the exact tangent (RK4 with
  pyqg's filter after every step). Box [-8, 8)^2, n = 128 (dx 0.125), step 0.025, snapshots every 0.05, 51 snapshots.
- Data: pyqg itself (AB3) from qg2d.initial (2-3 shielded elliptic vortices, zero circulation each, centres in [-2, 2]^2).
  Seed s: 16 training trajectories (data seed 4000 + s), 8 held-out (5000 + s).
- Regimes: f (beta = 0, no rd, no drag), beta = 0.5, rd = 1, rek = 0.3.
- FINS: exp45's polynomial class (60 fields), u = q unless stated, through the port's exact tangent; every 8th start,
  K = 8 (80 pairs).
- Readouts:
  - the eight candidates' quotients (d_x, d_y, d_t, the shift d_q, the rotation, D_x = x d_x + y d_y,
    D_t = t d_t - q d_q, D_beta = x d_x + y d_y - t d_t + q d_q);
  - own count by exp49's rule (floor 1e-6, up to 17 eigenvalues);
  - the span of the k lowest against the regime's algebra (missing, held-out characteristics);
  - the pencil restricted to the rotation plane (y d_x, x d_y) and to the scaling space (x d_x + y d_y, t d_t, q d_q):
    eigenvalues and eigenvectors.
- Kept candidates:
  - f: all eight (D_beta = D_x - D_t);
  - beta: d_x, d_y, d_t, shift, D_beta;
  - rd: d_x, d_y, d_t, shift, rotation, D_t;
  - drag: d_x, d_y, d_t, shift, rotation, D_x.
- Algebra dimension k: 7, 5, 6, 6.
- Secondary rows: q on the f-plane at n = 64 (same data seeds); the pressure psi in pyqg's gauge (zero mean) on the
  f-plane and with rd (D_x = x d_x + y d_y + 2 psi d_psi; the shift d_psi only with rd).
- Learned rows (notebook 8, Colab): FNO2d by exp45's recipe on 256 pyqg trajectories of data seed 6000 + s (n = 128,
  modes 24, 40 epochs), f and beta, seeds 42-44; FINS through its tangent on the same pyqg data.
Predictions (q, n = 128, seeds 42-51, in each of the four regimes unless stated):
(PG1) The regime's algebra lies in the span of its k lowest: missing <= 0.05 on >= 9/10 seeds.
(PG2) Own count (floor 1e-6) = k on >= 9/10 seeds.
(PG3) Every kept candidate's quotient <= 1e-4 on >= 9/10 seeds. For beta, rd and drag, every broken candidate's
quotient >= 100 times the largest kept one, on >= 9/10 seeds.
(PG4) Scaling space: the s deepest eigenvalues <= 1e-4 and the next >= 10 times the s-th, with s = 2 (f) and 1
(beta, rd, drag). The deep directions:
  - f: the two deepest span (1, 0, 0) and (0, 1, -1) (missing <= 0.05);
  - beta: (w_t, w_u) / w_x within 0.02 of (-1, 1);
  - rd: (w_x, w_u) / w_t within 0.02 of (0, -1);
  - drag: (w_t, w_u) / w_x within 0.02 of (0, 0).
  Required on >= 9/10 seeds.
(PG5) Rotation plane, lowest eigenvalue: >= 1e-3 at beta on 10/10 seeds; <= 1e-4 with stretch within 0.01 of 1 at f,
rd and drag on >= 9/10.
(PG6) n = 64, f-plane: D_x quotient >= 1e-3 on >= 9/10 seeds (pyqg's filter breaks the space scaling at 64^2).
(PG7) Pressure psi (pyqg's gauge):
  - f-plane: D_x >= 1e-2, D_t <= 1e-4 and rotation <= 1e-3, on >= 9/10 seeds;
  - rd: missing <= 0.05, own count 6, and D_x >= 100 times the largest of d_x, d_y, d_t, shift, rotation, D_t, on
    >= 9/10 seeds.
(PG8) Learned (FNO2d, seeds 42-44), each on >= 2/3 seeds:
  - the rotation plane's lowest eigenvalue at beta >= 10 times that at f (same seed);
  - beta: (w_t, w_u) / w_x of the deepest scaling direction within 0.1 of (-1, 1);
  - f: the two deepest scaling directions span D_x and D_t (missing <= 0.2).
No predictions: span of the own count, the translations' levels, psi at beta and drag, the FNOs' errors.
Reading registered in advance:
- PG1-PG5 met: from pyqg's data, through pyqg's discretization differentiated exactly, FINS finds each regime's point
  algebra and which physical parameter breaks which generator, with the exponents:
  - beta breaks the rotation and both scalings, leaving x -> lambda x, t -> t / lambda, q -> lambda q;
  - rd breaks the space scaling;
  - the drag breaks the time scaling.
- PG6 met: pyqg's grid-scale filter is a length scale at 64^2.
- PG7 met: in the pressure, the gauge breaks the space scaling (the level of psi is fixed by a global condition),
  except where rd screens the field.
- PG8 decides whether a learned emulator of pyqg keeps the pattern.

**Quasi-geostrophic flow from pyqg (exp61, qg2d.py) -- development (09.10; seed 7, data seeds 100 / 101; nothing
registered).** The authors' proposal: symmetries of pressure or velocity fields from pyqg (Abernathey et al.; the
standard testbed of ML ocean parameterizations, Ross et al. 2023).
- pyqg: release 0.7.2 does not build with Cython 3 / NumPy 2 (Python 3.13); master from GitHub does.
- The port (qg2d.py): BTModel's numerics in torch, following kernel.pyx and bt_model.py:
  - inversion psi_hat = a q_hat (a = 0 at k = 0);
  - flux-form advection without dealiasing;
  - beta psi_x and the drag rek |k|^2 psi_hat;
  - the exponential filter after every step.
  It integrates with RK4 instead of pyqg's AB3, so the snapshot map has an exact tangent.
- Checks at 64^2 (the check's default then; corrected 09.10 evening):
  - against pyqg with the same step and initial condition, the relative difference at T = 2.5 is 1.0e-4 to 2.0e-4 in the
    four regimes (2 % of one snapshot's change);
  - halving the step changes q by 2e-3 to 4e-3, because the filter acts once per step;
  - without the filter, q changes by 1.2e-2 to 2.4e-2;
  - tangent against central differences: 1.3e-10;
  - seam rms over the overall rms: 5e-4 to 7e-4, and 1.1e-2 with beta (Rossby waves).
- The same checks at 128^2, the registered grid (run after the results; by an independent check and by us):
  - against pyqg: 2.0e-5 (rd) to 6.3e-5 (drag), 0.35-0.46 % of one snapshot's change;
  - half step: 1.8e-5 to 9.0e-5; without the filter: 7.2e-5 to 3.7e-4;
  - tangent: 1.4e-10;
  - seam: 3.7e-8 to 1.3e-7, and 1.1e-2 with beta.
- Algebra in q of localised data:
  - inviscid f-plane: d_x, d_y, d_t, the shift d_q (pyqg's inversion ignores the mean of q), the rotation,
    D_x = x d_x + y d_y and D_t = t d_t - q d_q (7);
  - beta leaves D_beta = x d_x + y d_y - t d_t + q d_q in place of the rotation, D_x and D_t;
  - rd removes D_x; the drag removes D_t.
- FINS, q, exact tangent, seed 7. Quotients of the four generators any regime breaks (n = 128, 80 pairs, pyqg's data
  for f and beta, the port's for rd and drag):
| regime | rotation | D_x | D_t | D_beta | own count / k | missing (span of k) |
|---|---|---|---|---|---|---|
| f | 1.3e-7 | 2.0e-6 | 4.4e-9 | 1.6e-6 | 7 / 7 | 0.000 |
| beta | 1.1e-2 | 8.1e-3 | 4.7e-2 | 1.4e-5 | 5 / 5 | 0.000 |
| rd | 5.1e-13 | 3.6e-3 | 4.9e-9 | 2.8e-3 | 6 / 6 | 0.000 |
| drag | 2.1e-8 | 3.5e-8 | 8.9e-2 | 5.4e-3 | 6 / 6 | 0.000 |
  - With pyqg's data, d_x, d_y and the shift are at 1e-9 or below and d_t at 3e-7 (AB3 data against the RK4 map). With
    the port's own data they are at 1e-13 or below.
  - Scaling space:
    - f: two deep directions, 2.0e-9 and 2.1e-6 (then 0.55);
    - beta: (1, -1.002, 1.002) at 1.4e-5 (then 4.2e-2);
    - rd: D_t at 4.9e-9 (then 3.0e-4);
    - drag: D_x at 3.5e-8 (then 1.2e-2).
  - Rotation plane at beta: 1.1e-2.
  - With pyqg's data or the port's, the readouts agree to two digits (f, beta).
- n = 64 (dx 0.25, f-plane, port data, 160 pairs):
  - pyqg's filter breaks D_x: 9.8e-3, against 1.9e-6 at n = 128; D_t 3.5e-5;
  - aliasing breaks the continuous translations: 1.5e-8 and 1.2e-7, against 1e-14 at n = 128.
- The pressure psi:
  - pyqg's zero-mean gauge fixes the level of psi globally, which breaks D_x = x d_x + y d_y + 2 psi d_psi
    (f: 0.67) and D_beta (2.1e-2);
  - D_t survives (3.8e-12); the rotation weakens to 6.7e-5 (psi's tails reach the seam);
  - with rd the field is screened and the six generators, the shift d_psi included, are found (missing 0.000).
  - A far-field gauge (minus psi's mean on the seam frame) is worse: it breaks the translations (4e-5), since the frame
    is fixed in the box. Dropped.
  - The velocity is gauge-free but a vector field: it needs FINS for systems, not run.
Reading: through pyqg's numerics FINS reads each physical parameter's broken generators and the surviving scaling's
exponents, from pyqg's own data. pyqg's grid filter and missing dealiasing show as broken symmetries at 64^2, and the
pressure's gauge as a broken space scaling.

**Anisotropic nonlocal dissipation in 2D (exp60) -- results (09.10, night; 60 runs on two CPU workers, none failed;
json in fins_runs/aniso2d_runs; results/aniso2d/summary.{json,md}; every number below recomputed from the raw json by
an independent check). AN5 pending: notebook 7's first Colab run failed on import in every job (the notebook did not
install neuraloperator, which exp31's import chain needs; fixed 09.10, rerun on Colab).** The first launch (08.10,
22:33) lost its workers when the idle machine was reclaimed (~22:40); its two finished runs (fins at a = 1.25, seeds 42,
43) were kept. It ran with OMP_NUM_THREADS=0, which libgomp rejects. The relaunch (00:04) passes --threads 1.
Re-runs with one thread (independent check):
- fins a = 1.25 seed 42 (first launch): the npz (N, D, C, spectrum, data) bit-identical; the json identical except six
  containment values in the spans' "novel" lists, 1 ulp apart.
- regress a = 1.25 seed 45 and fins a = 1.75 seed 51: bit-identical.
FINS through the exact solver, pencil scans (seeds 42-51):
| a | eps-hat | a-hat | c-hat (2 - a) | quotient at the minimum: scaling / rotation (medians) |
|---|---|---|---|---|
| 1.25 | 0.600 on 10/10 | 1.255-1.257 | 0.740-0.743 | 8.8e-7 / 1.7e-7 |
| 1.5 | 0.600 on 10/10 | 1.500-1.505 | 0.493-0.500 | 3.4e-7 / 1.0e-7 |
| 1.75 | 0.600 on 10/10 | 1.747-1.750 | 0.250-0.254 | 1.3e-7 / 5.7e-8 |
The scaling scan's refinement (step 0.001 within +-0.01 of the coarse minimum) does not reach the quotient's minimum over
the family on four runs at a = 1.25: seeds 42, 43, 47 and 48 end on the box edge c = 0.740. The quotient's valley is
narrow across (a + c 1.997-2.001) and long along a - c. The family's exact minimum (a 3 x 3 generalized eigenproblem,
independent check) lies at a = 1.2546-1.2609, 1.4995-1.5037 and 1.7478-1.7504: errors up to 0.011, 0.004 and 0.002,
with c up to 0.015 below 2 - a. AN1 holds either way.
Spectrum (medians):
- the translations at roundoff (<= 3e-17);
- d_t 5.0-5.1e-10;
- lambda4, lambda5 4.6e-8 to 1.8e-7;
- lambda6 2.1e-5, 7.6e-6, 1.4e-6 for a = 1.25, 1.5, 1.75;
- lambda7 1.4-1.7e-4.
Secondary (no predictions):
- exp60's own-count rule (the largest of the first k + 4 ratios of consecutive eigenvalues floored at 1e-12) cuts
  inside the algebra in 30/30 runs: 2 on 25 runs, 3 on 5. The largest ratio lies either between the translations
  (floored) and d_t (25 runs; median ratio 535, range 241-1870) or between d_t and lambda4 (5 runs), not after the
  fifth eigenvalue (lambda6 / lambda5 medians 119, 63, 17). The own-count span misses 2-3 of the 5 dimensions.
  - own_count's docstring says "the first k + 4" eigenvalues; the code takes k + 5 (k + 4 ratios). The count is the
    same on all 30 runs either way.
  - Post hoc, with exp49's rule (floor 1e-6, up to 17 eigenvalues) the count is 5 on 10/10 at a = 1.25 and 6 on 10/10
    at a = 1.5 and 1.75. The sixth eigenvalue (7.6e-6, 1.4e-6) is nearer the floor than the seventh (1.4e-4).
  - The rank rule is set by its floor here; neither floor was registered for 2D.
- The span of the 5 lowest contains the algebra (missing 0.0007, 0.0014, 0.0075 medians; at most 0.044 at a = 1.75). Its
  deepest directions read eps-hat 0.598-0.602 and the exponent biased low: medians 0.10, 0.10, 0.16 low, up to 0.47 at
  a = 1.75 (1.102-1.173, 1.325-1.433, 1.280-1.666).
Weak SINDy in 2D on the 16 training trajectories of each seed (4624 rows). Each cell: true equation; exponent by
homogeneity (a-hat); stretched rotation.
| a | ref | iso-free | dir-free | harm-free | ell-free | nested |
|---|---|---|---|---|---|---|
| 1.25 | 0/10; 2 (8), none (2); none | 0/10; none (10); none | 0/10; 1.25 (2), 1.30 (8); none | 0/10; 1.250 (10); none | 10/10; 1.250; 0.600 (9) | ell-free 10/10 |
| 1.5 | 0/10; 2; none | 0/10; none (10); none | 0/10; 1.50 (1), 1.55 (9); none | 0/10; 1.500 (10); none | 10/10; 1.500; 0.600 | ell-free 10/10 |
| 1.75 | 0/10; 2; none | 0/10; none (10); none | 0/10; 1.75 (5), none (5); none | 0/10; 1.750 (10); none | 10/10; 1.750; 0.600 | ell-free 10/10 |
- The completions by row:
  - iso-free puts its order at 0.90-0.95, 1.25-1.30 and 3.95. At a = 1.25 and 1.5 a local u_xx stands beside it; at
    1.75 five local linear terms (u_xx, u_yy and the fourth derivatives) stand beside it.
  - dir-free at a = 1.75 sits at 1.95 with u_xx and u_yy on 5 seeds.
  - harm-free keeps harm0 and harm2 (cos 4 theta dropped) at the exact order.
  - ell-free returns u_t = 0.18 d_yy(u^2) + 0.5 d_xx(u^2) + 0.3 ell[0.6, a] u with relative coefficient errors at most
    4.7e-6.
- Right-hand-side errors: ref 0.029-0.15, iso-free 0.012-0.024, dir-free 0.0072-0.025, harm-free 1.2e-3 to 3.1e-3,
  ell-free 2.0e-7 to 9.9e-7.
- The nested rule picks the elliptic family by the residual alone on 30/30. Its residual is 205-567 times below the
  harmonics' and 3.4e3-1.5e4 times below the reference's.
- The rotation test (model_algebra: the linear symbol constant on the ellipses of e_n = sqrt(c_yy / c_xx) to 1e-6
  relative):
  - It finds the stretched rotation in the elliptic model on 29/30.
  - On a = 1.25, seed 45 it misses by its own tolerance: defect 1.04e-6 (> 1e-6). The fitted c_yy / c_xx = 0.36000120
    is 3.3e-6 (relative) off 0.36, the largest deviation of the 30 runs; this varies the multiplier's symbol by
    (a / 4) x 3.3e-6 on the test ellipses.
  - Its margins: ell-free <= 7.6e-7 on the 29 passing runs; harm-free >= 9.3e-3; dir-free >= 6.0e-2; iso-free >= 0.25.
    The reference reaches the test only on its 10 runs at a = 1.75 (0.037-0.061); its other models have u^3 terms,
    which model_algebra excludes before the test.
  - No registered prediction depends on the elliptic row's rotation.
  - model_algebra's true equation also requires sqrt(c_yy / c_xx) within 0.02 of eps. This is stricter than registered,
    with no effect: the largest deviation is 1e-6.
- Cost per run: fins 121-152 s; regress 413-557 s, mostly the elliptic profile over 15 x 71 points.
VERDICTS (literal; verdicts_0710.py):
- (AN1) MET: eps-hat within 0.01 and a-hat within 0.02 on 10/10 at every a.
- (AN2) MET: iso-free has no single exponent on 30/30 and no stretched rotation on 30/30.
- (AN3) MET: harm-free a-hat exact (within 0.05) on 10/10 at every a; no stretched rotation on 30/30.
- (AN4) MET: ell-free returns the true equation on 30/30.
- (AN5) PENDING: notebook 7. (Met on 09.10 evening: entry above.)
Reading, registered in advance:
- AN1 + AN2 + AN3 met. Through the exact solver FINS reads eps and the exponent without an operator family. The exponent
  needs an angular family: the isotropic single order gives none, and the directional one is up to 0.05 off (none on 5
  runs at a = 1.75). The stretched rotation needs the family that contains it.
- AN4 met. The elliptic family recovers the equation, and the residual alone selects it once it is offered: here FINS is
  not needed to identify the equation. What it adds is the symmetry parameters measured from the operator; whether that
  survives a learned operator is AN5.
Caveats for any use in a paper:
- The scans search fixed generator families: the plane (y d_x, x d_y) and the scaling's form.
- The span of the 5 lowest reads eps from FINS's polynomial class too, but only with the dimension given. exp60's
  own-count rule gives 2-3 here, and exp49's floor gives 5-6.
- A case where FINS would decide more than the residual needs a family the candidates do not anticipate (development
  idea, nothing run): anisotropy axes rotated by an unknown angle. There the stretched rotation is a general traceless
  linear field and gives (eps, angle). An axis-aligned elliptic family would fail (untested), and a rotated one costs
  one more profile dimension.

**Anisotropic nonlocal dissipation in 2D (exp60) -- pre-registration (08.10, 22:30; written after the development
runs of the entry below, before any run on seeds 42-51).**
Design as developed: aniso2d.py (eps = 0.6, nu = 0.3, a in {1.25, 1.5, 1.75}; exp45's box, bumps, 51 snapshots, dt
0.02); seed s: 16 training trajectories, data seed 4000 + s; 8 held-out, 5000 + s.
- fins (exact): exp45's polynomial class through the exact tangent on the training pairs (every 4th start, K = 8); the
  readouts are the pencil scans over the exact families: the stretched rotation -y / e d_x + e x d_y over e in
  [0.2, 2] (step 0.001) and the scaling x d_x + y d_y + a t d_t + c u d_u over (a, c) in [0.5, 3] x [-1, 2] (step 0.01,
  then 0.001). Secondary: the span's deepest directions (k and own count), missing dimensions on held-out data.
- fins (fno; notebook 7, Colab): FNO2d steppers by exp45's recipe (512 trajectories of data seed 6000 + s, 60 epochs),
  seeds 42-44, every a; the same pencil and scans through the FNO's tangent; one-step and K-step errors reported.
- regress: weak SINDy in 2D (pde/method_wsindy2d.py, the authors' algorithm on three axes; library d_x^p d_y^q (u^j),
  j <= 3, p + q <= 4, and a constant) on the same 16 training trajectories: ref; iso-free (-|k|^beta); dir-free
  (-|k_x|^beta, -|k_y|^beta); harm-free (-|k|^beta times 1, cos 2 theta, cos 4 theta); ell-free
  (-(k_x^2 + e^2 k_y^2)^(beta/2), e in 0.3..1 step 0.05, beta in 0.5..4 step 0.05); nested (the next family only when
  the weak residual drops by more than 1.25 times). A model's stretched rotation: its nonlinear part c1 d_xx(u^2) +
  c2 d_yy(u^2) gives e_n = sqrt(c2 / c1), and its linear symbol must be constant on the ellipses of e_n; its exponent by
  homogeneity (exp56.balance). True equation: d_xx(u^2), d_yy(u^2) and the elliptic multiplier only, e within 0.02,
  beta within 0.1.
Predictions:
(AN1) FINS exact, scans: |eps-hat - 0.6| <= 0.01 and |a-hat - a| <= 0.02 on >= 9/10 seeds for every a.
(AN2) iso-free: |beta - a| > 0.1 or no single exponent (no balance) on >= 8/10 seeds for every a; no stretched rotation
on 10/10.
(AN3) harm-free: |a-hat - a| <= 0.05 on >= 9/10 seeds for every a; no stretched rotation on 10/10.
(AN4) ell-free: the true equation on >= 9/10 seeds for every a.
(AN5) notebook 7, FINS through FNO2d (seeds 42-44): |eps-hat - 0.6| <= 0.05 and |a-hat - a| <= 0.1 on >= 2/3 seeds for
every a.
No predictions: the span-direction readouts, dir-free, ref, nested, right-hand-side errors.
Reading registered in advance: AN1 + AN2 + AN3 met is the case where the exponent needs an angular family and the
stretched rotation needs the family that contains it, while FINS needs neither; AN4 met concedes that a family built
around the true symmetry recovers the equation; AN5 decides whether this survives without the solver.

**Anisotropic nonlocal dissipation in 2D (exp60, aniso2d.py, pde/method_wsindy2d.py) -- development only (08.10,
night; seed 7, data seeds 100 / 101; nothing registered).** The case the applied mathematician's defence (ii) names: in
d >= 2 the translation-invariant linear operators homogeneous of degree a form an infinite-dimensional space (symbols
|k|^a Omega(k / |k|)), so no finite completion is exhaustive and a hidden continuous symmetry is what singles out a
family.
- Equation: u_t = div_A(u grad_A u) - nu (-div_A grad_A)^(a/2) u, grad_A = (d_x, eps d_y), eps = 0.6, nu = 0.3,
  a in {1.25, 1.5, 1.75}, on exp45's box ([-8, 8)^2, 64^2, exp45's bumps on a background, 51 snapshots, dt 0.02):
  an anisotropic medium with an anomalous dissipation of the same anisotropy. Algebra: d_x, d_y, d_t, the stretched
  rotation -y / eps d_x + eps x d_y, the scaling x d_x + y d_y + a t d_t + (2 - a) u d_u (exp45's second scaling
  t d_t - u d_u is broken by the dissipation: quotient 7e-4). Solver: exp45's RK4 with the multipliers; against twice
  the grid and half the step 5.6e-8; tangent against central differences 6e-11.
- FINS (exp45's polynomial class, 60 fields, relative pencil through the exact tangent, 160 training pairs, 140 s):
  the five generators at quotients 1e-27 (translations), 2.5e-10 (d_t), 3e-8 - 1.3e-7 (stretched rotation), 1e-7 -
  1.2e-6 (scaling), lambda6 1.5e-6 - 2.2e-5, lambda7 1.9e-4 - 2.4e-4; the 5-span misses 0.001-0.004 of the algebra
  (characteristics, held-out). Readouts: eps-hat 0.600 at every a (span direction and pencil scan alike); the exponent
  by the pencil scan over the exact family (a, c) 1.257, 1.502, 1.750 (c = 2 - a to 0.003); by the span's deepest
  direction 1.156, 1.418, 1.647 -- biased by about -0.1: the scaling's pencil floor (4e-7, the box; n = 96 does not
  lower it: 3.8e-7) lets the near-symmetry t d_t - u d_u tilt the eigenvector by 0.1 % of its norm, which the exponent
  amplifies. (Corrected 09.10, night: the near-symmetry is t d_t - (u + nu) d_u.) The scan is the readout to register in 2D; the span direction stays as the 1D paper's readout.
- Weak SINDy in 2D (the authors' algorithm on three axes; library d_x^p d_y^q (u^j), j <= 3, p + q <= 4, and a
  constant; the 16 training trajectories, 4624 rows), the identified model's algebra by its structure (stretched
  rotation: an elliptic linear symbol of the nonlinearity's anisotropy; scaling: homogeneity), a in 1.25 / 1.5 / 1.75:
  local library: local surrogates, exponent 2, no rotation (3/3); isotropic free order -|k|^beta: beta 0.90 / 1.25 /
  3.95 (the last with five spurious local terms), no rotation; directional -|k_x|^beta - |k_y|^beta: 1.30 / 1.55 / 1.75,
  no rotation; harmonics -|k|^beta (1, cos 2 theta, cos 4 theta): 1.25 / 1.50 / 1.75, no rotation (the surrogate's
  symbol is not elliptic); elliptic family -(k_x^2 + e^2 k_y^2)^(beta/2) over (e, beta): the true equation, e 0.600,
  beta exact, coefficients to 1e-6 (3/3, 6-9 min per run). The nested rule by the residual ends at the elliptic family.
Reading: the exponent is recovered by any completion whose angular family is rich enough (harmonics), mis-measured by
the natural single free order (isotropic: 0.25 and 2.2 off); the stretched rotation's parameter only by the family that
contains it. FINS reads both from the operator without a family, through the exact solver; no noise rows exist in 2D
(the relative pencil is the pointwise form) and the learned FNO2d rows need Colab (exp45's recipe).

**Transport by a nonlocal velocity (exp59, ccf.py) and the completions at higher noise -- development only (08.10,
late evening; not registered, nothing run on seeds 42-51).** Why: two of the three mock reviews of 08.10
proposed u_t + (Hu) u_x = -nu Lambda^alpha u (Cordoba-Cordoba-Fontelos type;
algebra d_x, d_t, the shift d_u, x d_x + alpha t d_t + (1 - alpha) u d_u, no boost) as the out-of-library case no linear
completion answers; the advocate proposed a noise sweep out of the library.
- ccf.py: fburgers.py's data and ETDRK4 with -(Hu) u_x; against 8 times the grid 2e-14 of max |u|, half the step 6e-12,
  the shift u -> u + 0.3 to 3e-15; the mean of u grows by ~0.11 over T = 1 (d/dt of the mean is the mean of u Lambda u).
- exp59_ccf.py one (seed 7, data 1007): the reference and the linear completion (exp58's prof-free) return local
  surrogates with neither shift nor scaling; the bilinear completion (the reference library, E and O at a profiled order,
  and d_x(u Hu), d_x((Hu)^2), u Lambda u, Lambda(u^2), which span (Hu) u_x = d_x(u Hu) - u Lambda u) returns the true
  equation for alpha 1.5 and 1.25 at clean, 1e-2, 3e-2 and 5e-2 (coefficients within 2 %; spurious terms at 1e-1) and
  for alpha 1.75 to 1e-2 (u_xx beside E^1.885 from 3e-2); the nested rule picks it by the residual alone.
- FINS weak through the exact solver: the shift at every alpha (t d_x share 1e-14), no boost; the scaling only
  approximate on the 2 pi torus (fourth eigenvalue 1.7-2.1e-4 against 1e-16 for the exact symmetries; own count 3; m = k
  a-hat 1.146 (containment 0.38), 1.392 (0.69), 1.632 (0.85) for alpha 1.25, 1.5, 1.75; 1.391 at 1e-2). The Hilbert
  velocity's 1/x tails reach the periodic images, so the periodic operator is not scale invariant at this box size.
- Completions at higher noise (seed 7): Benjamin-Ono true to 1e-1 (beta 2.000); fractional Burgers 1.5 true to 5e-2;
  1.75 false from 3e-2 (u_xx beside E^1.885: the integer-precedence window the completion lacked); at 1e-1 the nested
  rule falls back to the reference.
Reading: neither proposal gives a win on this design. The bilinear completion identifies the CCF equation without
FINS, and FINS's exponent is biased by the domain; the completions break at high noise mainly through a fixable
near-duplicate, while FINS through learned operators is already worse at 1e-2 (Benjamin-Ono 1.86-2.17).

**Symmetry-informed weak SINDy (exp58) -- results (08.10, evening; 280 runs on two CPU workers, none failed; json
in fins_runs/sym_runs, readouts in sym_runs/readouts.json; results/symlib/summary.{json,md}; every number below
recomputed from the raw json by an independent check, three runs re-run bit-identically).** The true equation,
seeds 42-51:
| | ref | sym | sym-ref | prof | prof-free |
|---|---|---|---|---|---|
| heat, 6 conditions | 58/60 | 60/60 | | | |
| KdV, clean to 3e-2 | 40/40 | 40/40 | 40/40 | 39/40 | 36/40 |
| KdV 5e-2 | 9/10 | 10/10 | 8/10 | 7/10 | 5/10 |
| KdV 1e-1 | 9/10 | 3/10 | 6/10 | 5/10 | 5/10 |
| KS, 5 conditions | 50/50 | 50/50 | | 0/50 | 47/50 |
| Benjamin-Ono, 3 conditions | 0/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| fractional Burgers 1.25-1.75, clean and 1e-2 | 0/60 | 60/60 | 60/60 | 60/60 | 58/60 |
| fb200, clean and 1e-2 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| all cells (periodic cells) | 186/280 (128/220) | 273/280 (213/220) | | 161/220 | 201/220 |

Out of the library (Benjamin-Ono, fractional Burgers 1.25-1.75), medians per cell: |beta - alpha| sym 0.002-0.018
(FINS's exponent), sym-ref 0.001-0.004, prof and prof-free 0.000 (the grid point); right-hand-side error ref 0.09-0.67,
sym 9e-4-1e-2, prof 2e-10-8e-4; coefficient errors (advection / linear) sym 2.5e-4-5e-3 / 1.4e-3-2.6e-2, prof <= 5e-4 /
<= 7e-4 (largest cell medians 5.02e-4 and 6.77e-4). The failures: sym at KdV 1e-1 on the seven seeds whose a-hat lies
0.20-0.54 above 3 (42, 44, 45, 46, 48, 49, 51: O at a-hat with coefficient 0.46-0.73 in place of u_xxx); prof on nine
KdV runs where the residual's minimum leaves the exponent under noise (beta 3.2 three times, 2.8 three times, 2.685,
3.445, 3.475); prof-free on KS 1e-1 (seeds 44, 45, 49: a local term and the multiplier near its order split the
coefficient, seed 44 +19.3 u_xx - 20.0 E^1.98 and six spurious terms, eF 1.37), fb175 1e-2 (seeds 42, 49: -0.14 u_xx
+ 0.28-0.29 E^1.885) and 14 KdV runs at 3e-2 to 1e-1 (u_xxx beside O: four near 3, two far from it; O alone at an
order more than 0.1 off, as prof: five; spurious terms: three). Profiles' refinement may step 0.05 past the grid's
ends (betas 0.45 and 4.05 occur). The reference's failures (exp56): sym repairs heat 3e-2 seed 49 (linear and
scaling gates: u_xx alone), heat 1e-1 seed 44 (the shift gate removes the reaction terms) and KdV 5e-2 seed 48 (the
boost gate removes (u^2)_xxx), not KdV 1e-1 seed 42 (a-hat 3.44). The boost readout measures the advection
coefficient: |c-hat / c - 1| median 5e-11 to 5e-8 clean, 6e-5 to 1e-2 under noise, largest 4.2e-2 (KdV 1e-1).
Secondary rows: sym-mk 263/280 (heat 5e-2 and 1e-1: 7/10, 4/10, the m = k span's gates; KdV 1e-1: 2/10); sym-learned
32/42 (the learned operators' spans; Benjamin-Ono 1/6: no scaling read through the steppers at 1e-2 and on seed 43
clean, a-hat 2.101 on seed 44 clean, 0.101 off); sym-pure 158/170 and sym-scal 163/170 on the cells where the scaling
is read.
VERDICTS (literal; verdicts_0710.py). (SL1) MET: 10/10 in all nine cells. (SL2) NOT MET: prof-free 8/10 at fb175 1e-2
(seeds 42, 49), 10/10 in the other eight cells; its median |beta - alpha| (0.000) at most sym's in every cell (met).
(SL3) MET: 10/10 from clean to 5e-2; at 1e-1 true on exactly 43, 47, 50. (SL4) MET: 10/10 at every condition, seeds 49
(3e-2) and 44 (1e-1) included. (SL5) NOT MET: sym 10/10 and prof 0/10 at every condition (met); prof-free 7/10 at 1e-1
(seeds 44, 45, 49). (SL6) MET.
Reading: the library of the algebra FINS measured recovers both out-of-library families on every seed and repairs
three of the reference's four in-library failures: 273/280 against the reference's 186/280. Completing the reference
library with the homogeneous nonlocal family at a residual-chosen order, without FINS, recovers the out-of-library
equations too (88/90), more precisely (the order on its grid point in 85 of the 88, 0.005 off in three noisy runs;
coefficients to 1.4e-6 on clean data, 1e-10 on fractional Burgers), but under noise its extra columns break
in-library identification (KS: near-collinear pairs; KdV: an order off by more than 0.1, split coefficients, spurious
terms; KdV 46/60, KS 47/50: 201/220 against sym's 213/220 on the same cells). FINS's exponent limits sym where it drifts (KdV 1e-1: 3/10 against the reference's 9/10). For the
PDE paper: the exp57 oracle failed at 1e-2 because of its grid of six beta columns; one free order, profiled by the
residual, recovers Benjamin-Ono and fractional Burgers from the evaluation data, an objection to the paper's
out-of-library argument that the text has to meet.

**Symmetry-informed weak SINDy: the library from the algebra FINS measured (exp58) -- design and pre-registration
(08.10, afternoon; written after the development runs listed, before any run on seeds 42-51).**
Why: FINS returns an algebra and an exponent, not an equation. Lie's inverse problem turns the measured algebra into a
small family of invariant evolution equations; weak SINDy then fits its constants. The question is what this adds to
the two-stage route, and against which baselines: the reference weak SINDy (exp56), and weak SINDy completed by the
same nonlocal family with its order chosen by the residual, without FINS.
Design:
- exp58_symmetry_library.py. Readouts from the span of FINS weak through the exact solver (own count: no knowledge of
  the algebra's dimension; secondary the m = k span and the learned operator's span), every field built from the data
  (no ground truth), exp51's band and weights: the boost plane (t d_x, d_u): its deepest direction t d_x + c d_u is the
  boost when its t d_x share (normalised basis) exceeds 1e-3, the shift d_u below (heat: share <= 2e-13; the boosts
  0.011-0.25); the scaling space (x d_x, t d_t, u d_u): two directions inside, linear (exponent of the one without
  u-weight); with the boost read the exponent of exponent_readout.py's plane, reduced against the boost read (equal to
  the exponent readouts of exp56/57 to the third decimal). Gate: containment >= 0.5; 0 < a-hat <= 6. The boost also
  gives the advection coefficient (KdV: c = 0.167 = 1/6 on every development condition).
- Library (reference: d_x^k (u^j), j <= 4, k <= 6, and a constant): linear: j = 1 only; shift: the constant and
  d_x^k u (k >= 1); boost: those and (u^2)_x; scaling: nonlinear terms within TAU = 0.2 of their own exponent
  (1 + k - j) / (2 - j), d_x^k u within TAU of a-hat, and on periodic data the multipliers E (symbol -|kappa|^a-hat)
  and O (i sgn(kappa) |kappa|^a-hat) -- the translation-invariant linear operators homogeneous of degree a-hat -- one of
  a parity left out when a local d_x^k u of that parity lies within TAU (integer orders take precedence).
- MSTLS as the reference, with the per-term price 1 / 31 in every row but ref: the authors' loss charges nnz / J, and
  with J = 3 a second term costs a third of the loss (development: KdV's dispersion was dropped at 3e-2).
- Rows: ref; sym (the gates of the own-count span); sym-pure (the multipliers at a-hat only, no local linear terms);
  sym-mk; sym-learned; sym-scal (the scaling gate alone); sym-ref (beta refined by the residual within 0.25 of a-hat);
  prof (sym's structure assumed -- boost and scaling -- without FINS, beta minimising the selected model's residual over
  0.5-4 in steps of 0.05, refined to 0.005); prof-free (the reference library plus E_beta, O_beta, beta likewise; a
  multiplier equal to a local term at integer beta left out). Profiles from the normal equations (development: the
  residuals equal the reference MSTLS's to every printed digit), the model at beta* refitted by the reference MSTLS.
- Cells: heat and KdV (exp51's conditions), Benjamin-Ono (exp52's), KS (exp53's), fractional Burgers alpha = 1.25-2
  (clean, 1e-2); seeds 42-51, the evaluation data (data seed 1000 + s); 280 runs. prof and prof-free need periodic
  data (not heat).
- Scores: the true equation (its terms and no others; linear degrees within 0.1), |beta - alpha|, the relative error of
  the identified right-hand side on the clean evaluation data (periodic equations), coefficient errors.
Development (seed 7, data 1007; fins_runs/sym_dev; FINS's development spans, weaker than the registered ones on KdV:
a-hat 2.98, 2.95, 2.86, 3.23, 2.14, 2.94 from clean to 1e-1). True equation, development seed, by row (ref / sym /
prof / prof-free): heat 6/6 / 6/6 / - / -; KdV 6/6 / 4/6 (3e-2 and 5e-2: a-hat outside the window) / 4/6 (5e-2,
1e-1: beta 2.80) / 4/6; KS 5/5 / 5/5 / 0/5 / 5/5; Benjamin-Ono (3 conditions) and fractional Burgers (alpha = 1.25,
1.5, 1.75 clean, 1.5 at 1e-2) 0/7 / 7/7 / 7/7 / 7/7; fb200 1/1 everywhere. |beta - alpha| out of the library: sym
0.003-0.024 (FINS's), prof and prof-free 0.000 (the residual minimum on the grid point), right-hand-side errors
1e-3-2e-2 against 1e-9-7e-4. The exp57 oracle's failures at 1e-2 come from its grid of six beta columns, not from the
data: with one beta at a time the residual profile finds alpha.
Readouts on seeds 42-51 (inputs, computed before any regression): boost and scaling read on every KdV, Benjamin-Ono and
fractional Burgers seed; KS the boost alone on 10/10 at every level; heat linear and scaling on 10/10 to 1e-2, 9/10 at
3e-2 (seed 44: shift and one scaling direction), 4/10 at 5e-2 and 0/10 at 1e-1 (the shift alone). KdV a-hat within 0.2
of 3 on every seed to 5e-2 (largest 3.177), on 43, 47, 50 at 1e-1 (3.010, 3.132, 3.133; 49 at 3.202).
Predictions:
(SL1) sym, out of the library -- Benjamin-Ono at clean, 1e-2, 3e-2; fractional Burgers alpha = 1.25, 1.5, 1.75 at
clean and 1e-2 -- the true equation on >= 9/10 seeds in every cell.
(SL2) prof-free on the same cells: the true equation on >= 9/10 in every cell, and its median |beta - alpha| at most
sym's in every cell (the completion without FINS does as well out of the library).
(SL3) KdV, sym: the true equation on >= 9/10 at clean, 1e-3, 1e-2, 3e-2 and 5e-2 (u_xxx inside the window on every
seed); at 1e-1 on seeds 43, 47 and 50 and on none of the other seven.
(SL4) Heat, sym: the true equation on >= 9/10 at every condition, including seed 49 at 3e-2 and seed 44 at 1e-1, where
the reference fails (exp56: reaction surrogates).
(SL5) KS: sym on >= 9/10 at every condition (the scaling gate closed); prof on 0/10 at every condition (KS has no
scaling); prof-free on >= 9/10 at every condition.
(SL6) fb200, sym: the true equation on >= 9/10 at clean and 1e-2.
No predictions: sym-pure, sym-mk, sym-learned, sym-scal, sym-ref; prof outside KS; right-hand-side and coefficient
errors; the reference's rows (exp56's, recomputed).
Paper: nothing in the PDE paper's text until the authors decide (a possible second paper); the free-order completion
bears on the PDE paper's out-of-library argument whatever the verdicts.

**Fractional Burgers (exp57, notebook 6) and operators learned from noisy data (notebook 5) -- results (08.10,
morning; exp57's exact, WSINDy and oracle runs on two CPU workers, 332 runs, none failed; notebooks 5 and 6 on
a Colab T4, merged; npz in fins_runs/fb_runs, noise_weak_runs, bo_runs, ks_runs; results/fburgers,
noise_weak, bo, ks, exponent).** Time-scaling exponent a-hat, median [min, max] over seeds (42-51; learned and training
data 42-44), missing dimensions of the finite algebra (mean) in parentheses:
| | alpha = 1.25 | 1.5 | 1.75 | 2 |
|---|---|---|---|---|
| FINS weak, exact, clean | 1.238 [1.23, 1.26] (0.073) | 1.496 [1.49, 1.53] (0.017) | 1.750 [1.74, 1.78] (0.005) | 1.999 (0.000) |
| FINS weak, exact, 1e-2 | 1.234 [1.22, 1.27] (0.432) | 1.496 [1.47, 1.51] (0.792) | 1.747 [1.74, 1.76] (1.000) | 2.000 (0.002) |
| FINS weak, own count, clean / 1e-2 | (0.011 / 0.013) | (0.001 / 0.003) | (0.000 / 0.002) | (0.000 / 0.002) |
| FINS pointwise, exact, clean | 1.098 [0.95, 1.14] (0.390) | 1.433 [1.39, 1.51] (0.134) | 1.736 [1.70, 1.81] (0.040) | 2.000 (0.000) |
| FINS weak, learned | 1.223 [1.22, 1.23] (0.682) | 1.492 [1.46, 1.50] (0.583) | 1.744 [1.74, 1.75] (0.788) | 2.006 [2.00, 2.02] (0.014) |
| FINS pointwise, learned | 1.14 (1.365) | 1.73 (1.085) | 2.33 (1.078) | 2.73 (0.675) |
| WSINDy hard, clean | none (2.000) | none 9, 2.00 1 (1.903) | 2.000 (0.004) | 2.000 (0.000) |
| WSINDy hard, 1e-2 | none (2.000) | none 8, 2.00 2 (1.61) | 2.000 (0.004) | 2.000 (0.000) |
| WSINDy soft, clean | 1.56 (0.114) | 1.77 (0.024) | none 3, 1.92-2.00 on 7 (0.748) | 2.000 (0.000) |
| WSINDy, training data, hard | none (2.000) | 2.000 (0.035) | 2.000 (0.005) | 2.000 (0.000) |
| oracle, true equation, clean / 1e-2 | 10/10 / 4/10 | 10/10 / 0/10 | 10/10 / 0/10 | 10/10 / 0/10 |

("none": no direction of the scaling plane at least half inside the span.) WSINDy's surrogates: alpha = 1.25 and 1.5,
nine seeds of ten each, damped Burgers (-0.13 to -0.19 u + 0.072-0.076 u_xx - 0.52 to -0.54 (u^2)_x at 1.25), the tenth
with cubic terms (u^3, and (u^3)_xx at 1.25); 1.75 viscous Burgers 0.103-0.105 u_xx - 0.51 to -0.53 (u^2)_x on every
seed; 2 exact (0.11111 u_xx - 0.5 (u^2)_x). At 1e-2 the same pattern (alpha = 1.5: eight damped, two viscous Burgers).
Training data: damped at 1.25, viscous Burgers at 1.5 and 1.75. The oracle on clean data: -Lambda^alpha u with nu to
five digits (0.25328, 0.19245, 0.14623) and u_xx at alpha = 2, every seed; at 1e-2 the dissipation itself is wrong on
most failing seeds (-0.011 u_xxxx alone at alpha = 1.75 and 2; Lambda^1.25 with Lambda^0.5 and u_xx in place of
Lambda^1.5). FINS weak at 1e-2: the m = k reading trades a translation for the near-symmetry above the algebra (as on
Benjamin-Ono); pointwise at 1e-2 fails (2.2-2.6 missing, a-hat -0.47). Steppers: one-step increment error 0.017-0.025 on
the evaluation data; through them the missing dimensions are mostly a translation (boost and scaling 0.21, 0.10, 0.02).
Notebook 5 (seeds 42-44; missing dimensions, finite / non-trivial): heat weak through CNOs retrained on noisy data 0.008
/ 0.007 at 1e-3 and 0.182 / 0.139 at 1e-2 (pointwise through them 1.68, 2.96); Benjamin-Ono at 1e-2 through steppers
trained on 1e-2 data weak 0.058 / 0.013 (a-hat 2.17, 1.86, 2.01; through the exact solver 1.000, the near-symmetry
displacing a translation, which the steppers lift), pointwise 2.316; Kuramoto-Sivashinsky weak 0.000 / 0.000, pointwise
1.151.
VERDICTS (literal; verdicts_0710.py). (FB1) NOT MET: a-hat within 0.1 on 10/10 for every alpha (at most 0.033 off),
but missing <= 0.05 on 4/10 at alpha = 1.25 (9/10, 10/10, 10/10 for the others). (FB2) MET: 10/10 for every alpha (at
most 0.029 off). (FB3) MET: alpha = 2 a-hat 2.00 on 10/10; alpha < 2 the right exponent absent on 10/10 at both levels.
(FB4) MET: the true equation on 10/10 for every alpha. (FB5) NOT MET: a-hat within 0.15 on 3/3 for every alpha < 2 (at
most 0.037 off), but missing 0.682, 0.583, 0.788 against <= 0.3. (LW4) MET: 0.008 (<= 0.336) and 0.182 (<= 0.592).
(LW5) MET: BO weak 0.058 / 0.013, KS 0.000 / 0.000; pointwise 2.316 and 1.151; BO a-hat 2.17, 1.86, 2.01. No verdict
pending; every earlier verdict unchanged.
Reading: FINS's weak form reads the dissipation exponent of a nonlocal term from the symmetry to within 0.04 on every
run, through the exact solver on clean data and at 1e-2 and through learned steppers (pointwise it misreads: 0.95-1.14
at alpha = 1.25, -0.47 at 1e-2, 1.04-4.18 through the steppers). The identify-then-Lie route returns surrogates
without a scaling or with the diffusive exponent 2 (the subspace score cannot tell 2 from 1.75: 0.004 missing); with
the right family in its library it recovers the equation on every clean run and on 4 of 40 noisy ones. The missing
dimensions of FINS's m = k reading fail where the near-symmetry above the algebra comes close (alpha = 1.25 clean,
every learned operator for alpha < 2), while the exponent readout is unaffected and the own count nearly so (at most
0.020 per seed through the exact solver, 0.024-0.077 through the alpha = 1.25 steppers). Side notes from the
independent check: through the BO steppers at 1e-2 the own count is 1 on 3/3 seeds, and lambda5 / lambda4 is only
2.1-9.2 there.

**Reference weak SINDy then Lie, like for like (exp56) and FINS weak through the heat CNOs (EX4, notebook 4) --
results (08.10, night; 237 runs on two CPU workers, none failed; npz in fins_runs/{noise_weak,ks,bo,burgers}_runs;
results/noise_weak, ks, bo, burgers, exponent, paired).** Missing dimensions of the finite algebra, seeds 42-51, means,
hard / soft readout (number of seeds above 0.05 in brackets where not zero):
heat      0.000 / 0.000 at clean, 1e-3, 1e-2 and 5e-2; 0.322 / 0.247 (1) at 3e-2 (seed 49: u_t = 0.034 u - 0.016 u^3, a
          reaction surrogate); 0.400 / 0.082 (1) at 1e-1 (seed 44: u_xx plus five reaction terms); the class:
          the same seeds, 0.721 / 0.645 and 0.800 / 0.329
KdV       0.000 / 0.000 to 3e-2; 0.101 / 0.090 (1) at 5e-2 (seed 48: -3.050 (u^2)_x - 0.0981 (u^2)_xxx, no u_xxx);
          0.200 / 0.000 (1) at 1e-1 (seed 42: 0.557 u - 1.015 u_xxx - 0.085 u^2 - 3.003 (u^2)_x); the other 58 runs
          u_t = -u_xxx - 3 (u^2)_x within 1.7%
KS        0.000 / 0.000 at every level; Burgers, clean, 0.000 / 0.000
training  heat, KdV, KS (clean, 1e-2), Burgers (clean): 0.000 / 0.000 on 3/3
Benjamin-Ono: clean, 9/10 seeds the advection surrogate u_t = -(4.96 to 5.30) u_x, whose algebra in the class is
22-dimensional (hard 0.003 missing, 18 spurious on those seeds; soft 2.396 (9)); seed 50 a KdV-type surrogate
(-0.607 u_x - 0.0275 u_xxx - 0.265 (u^2)_x). 1e-2: KdV-type on 8/10 (-0.055 to -0.066 u_xxx, -0.335 to -0.348
(u^2)_x), advection on seed 43, KdV-type with +0.495 u_x on seed 50: hard 0.014 (1.81 spurious), soft 0.280 (1). 3e-2:
KdV-type on 10/10, 0.015 / 0.015. Training data: KdV-type, clean (with -0.25 to -0.33 u_x) 0.016 and 1e-2 0.015, hard
and soft alike.
Exponent readout (a-hat, mean [min, max], containment; results/exponent/summary.md). Benjamin-Ono (a = 2): FINS weak
2.004 [2.00, 2.01], 2.003 and 2.003 at clean, 1e-2 and 3e-2 (containment >= 0.997 on every run); through the steppers
weak 1.989 [1.80, 2.10], pointwise 2.011 [1.94, 2.06]; WSINDy's KdV-type surrogates 3.00 on every such seed (BO's a = 2
at containment 0.985); its advection surrogates hold their own scaling x d_x + t d_t exactly (a-hat 1.00) and the
plane's a = 2 and a = 3 directions to 0.9999 (diagnostic), so the readout cannot tell the plane's directions apart;
PySINDy's surrogates no scaling (hard, containment 0; soft means 1.20-1.60); LieNLSD as published mean containment <=
0.35 (its best seed 2.47 at 0.77); Augerino 1.00 (its dictionary holds x d_x + t d_t). KdV (a = 3): FINS weak 3.010 to
1e-2, 3.030, 3.077 and 3.343 [3.07, 3.73] at 3e-2, 5e-2 and 1e-1; WSINDy 3.000 except its failed seeds; LieNLSD with the
exact right-hand side 3.05 clean, mean containment <= 0.58 under noise.
VERDICTS (literal; verdicts_0710.py). (LW1) MET: every cell >= 9/10 (heat at 3e-2 and 1e-1, KdV at 5e-2 and 1e-1:
9/10; KS and Burgers 10/10). (LW2) MET: 3/3 in every cell (heat's class, not registered: 0.000). (LW3) NOT MET: the
KdV-type surrogates read 3.00 as predicted (1/10 clean, 9/10 at 1e-2, 10/10 at 3e-2, 6/6 on training data), but the
advection surrogates (9/10 clean, 1/10 at 1e-2) fail the registered criterion: their 22-dimensional algebra contains
BO's scaling (to 0.9999) together with the rest of the plane and 18 dimensions outside BO's algebra. (EX4) NOT MET:
finite 0.010 (<= 0.3, met), class 0.537 (registered >= 1.0; per seed 0.98, 0.13, 0.49; own count 10 on 3/3): the weak
form through the protocol's CNOs recovers heat's class better than predicted (pointwise through the same CNOs 1.58).
LW4 and LW5 pending (notebook 5). Every earlier verdict unchanged on the regenerated summaries.
Reading: in its reference configuration the two-stage pipeline recovers every equation its library holds on 176 of
180 evaluation runs and on all 21 training runs. Under noise neither route dominates: it is ahead of FINS's weak form
on KdV at 1e-1 (0.20 hard against 0.91 at m = k), behind on heat at 3e-2 (0.32 against 0.013) and on heat's class at
3e-2 and 1e-1 (0.72 and 0.80 against 0.014 and 0.38). On Benjamin-Ono it returns local surrogates whose algebras hold
BO's scaling only inside 22 dimensions or carry KdV's exponent 3; the subspace metric scores both near zero
(0.003-0.016), the spurious dimensions and the exponent readout separate them from FINS's span (k = 4, a-hat 2.004).
The case against the two-stage route rests on equations outside the library, now two families (exp57).

**Fractional Burgers: the exponent of a nonlocal dissipation read from the symmetry, without the equation (exp57,
notebook 6) -- design and pre-registration (08.10; written after the development runs listed, before any run on seeds
42-51).**
Why: exp56's development showed reference weak SINDy exact wherever the equation is in its library (heat, KdV, KS,
Burgers), so FINS's lead over the two-stage pipeline rests on Benjamin-Ono alone; the mock reviews asked for a second
case FINS wins like for like, ideally a process whose equation is not known. In a family of nonlocal dissipations the
quantity of interest is the dissipation exponent (anomalous diffusion). It is the time weight of the scaling symmetry,
which FINS reads from the solution operator, while a library of local terms has no term for it.
Design:
- fburgers.py: u_t + u u_x = -nu Lambda^alpha u on [0, 2 pi), Lambda^alpha the multiplier |k|^alpha, nu = 3^-alpha
  (damping rate 1 at the humps' wavenumber 3 for every alpha: the Reynolds number at that scale, 3 |a|, is the same
  across the family); ETDRK4 on 4 times the data grid, sampled back on its 128 points (against 8 times: below 1e-8 of
  max |u| from alpha = 1.25); the Galilean shift and halving the step agree to 1e-11. Evaluation data: two Gaussian
  humps of random sign, |a| in [0.5, 1], width in [0.25, 0.4], centres at [0.30, 0.34] L and [0.48, 0.52] L, 128 x 128
  over T = 1, data seed 1000 + s. Training data 'mixed': half one or two such humps on a zero or uniform [-0.5, 0.5]
  background, half one to three broadband humps anywhere.
- Family alpha in {1.25, 1.5, 1.75, 2} (fb125 ... fb200). Algebra d_x, d_t, Galilean t d_x + d_u, scaling
  x d_x + alpha t d_t + (1 - alpha) u d_u; alpha = 2 adds the projective (finite 5). Non-trivial: Galilean, scaling
  (and projective). alpha = 1 (critical) dropped after development: its fronts are under-resolved on the 128-point grid
  (the 128-point Galerkin solution differs by 9e-3 of max |u| from the converged one), and the oracle's spectral
  Lambda^1 column then fails on the samples (it picks Lambda^0.5 and Lambda^0.75).
- exp57_fburgers.py, conditions clean and 1e-2, seeds 42-51: weak and pointwise (FINS through the exact solver, exp52's
  commands); wsindy (exp56's reference weak SINDy then Lie, hard and soft, evaluation data); oracle (exp56's library
  plus -Lambda^beta u columns, beta in {0.5, 0.75, ..., 1.75}: the identified equation and the exponent at which its
  terms balance -- no Lie stage exists for a nonlocal term); wsindy-train (training data, seeds 42-44, clean);
  weak-learned and pointwise-learned (notebook 6: FNO steppers on 1024 clean 'mixed' trajectories, exp39's recipe,
  seeds 42-44, clean).
- Metric: the exponent readout a-hat (exponent_readout.py) is primary, exp51's missing dimensions secondary. The
  development runs show the subspace metric's blind spot again: WSINDy's viscous-Burgers surrogate (a = 2) scores
  0.005 missing on fb175 clean and 0.035 on fb150 at 1e-2, while its exponent is off by 0.25 and 0.5.
Development (seed 7, data 1007; fins_runs/fb_dev; missing dimensions on the finite algebra, a-hat with containment):
- FINS weak, exact solver, clean: fb125 0.028, a-hat 1.218 (0.975); fb150 0.001, 1.489 (0.999); fb175 0.000, 1.746
  (1.000); fb200 0.000, 1.999 (1.000), the projective inside the five lowest (loss 2.4e-8, lambda6 = 2.8e-2). The
  scaling's loss lies below lambda5 by factors of 80, 170 and 360 (fb125-fb175); own count 5 there (a fifth
  near-symmetry at 4e-5 to 2e-4). At 1e-2, fb150: m = k misses 0.793 (d_t displaced by the near-symmetry, as on
  Benjamin-Ono), non-trivial 0.028, a-hat 1.505 (0.972); own count 5: 0.009, a-hat 1.492.
- FINS pointwise, exact, clean, fb150: 0.066, a-hat 1.491 (0.934).
- WSINDy then Lie, clean: fb125 and fb150 give damped Burgers surrogates (u_t = -0.18 u + 0.069 u_xx - 0.54 (u^2)_x;
  -0.11 u + 0.084 u_xx - 0.53 (u^2)_x): the hard algebra is the two translations (missing 2.000, no scaling), the soft
  readouts 0.103 and 0.019 with a-hat 1.538 and 1.745. fb175 gives the viscous Burgers surrogate 0.103 u_xx - 0.527
  (u^2)_x (hard: 0.005 missing, 1.005 spurious, a-hat 2.000); fb200 the exact equation, 0.11111 u_xx - 0.5 (u^2)_x. At
  1e-2 every alpha gives a viscous Burgers surrogate (a-hat 2.000). Training data, fb150, clean: 0.094 u_xx - 0.510
  (u^2)_x (a = 2).
- Oracle, clean: the exact equation for every alpha (fb150: 0.19245 (-Lambda^1.5) u - 0.5 (u^2)_x, nu = 3^-1.5); at
  1e-2 it fails for every alpha (fb150: 0.027 u_xx + 0.047 Lambda^0.5 u - 0.213 Lambda^1.25 u - 0.4998 (u^2)_x; no
  balance). With the columns on every power u^j (24 instead of 6) it also fails at alpha = 1 clean.
Predictions:
(FB1) FINS weak, exact solver, clean, every alpha: |a-hat - alpha| <= 0.1 on >= 9/10 seeds, and finite <= 0.05 on
>= 9/10 seeds.
(FB2) FINS weak, exact solver, 1e-2, every alpha: |a-hat - alpha| <= 0.15 on >= 8/10 seeds.
(FB3) WSINDy then Lie on the evaluation data, hard readout: alpha = 2 clean, |a-hat - 2| <= 0.05 on >= 9/10 seeds;
alpha < 2, clean and 1e-2: the right exponent absent -- |a-hat - alpha| > 0.1 or containment < 0.9 -- on >= 9/10 seeds.
(FB4) Oracle, clean, every alpha: the true dissipation alone (alpha < 2: one multiplier term, Lambda^alpha; alpha = 2:
none) and balance exponent alpha on >= 9/10 seeds.
(FB5) Notebook 6, clean, seeds 42-44, alpha in {1.25, 1.5, 1.75}: FINS weak through the steppers |a-hat - alpha| <= 0.15
on 3/3 seeds, finite <= 0.3 on average.
No predictions: the pointwise rows, WSINDy's soft readouts and missing dimensions, wsindy-train, the oracle at 1e-2,
alpha = 2 through a stepper (last in notebook 6).
Paper, fixed now: a subsection on fractional Burgers with a table of a-hat by method and alpha (clean, 1e-2, learned)
and the missing dimensions beside it; the introduction and abstract name it as the second equation outside every local
library.

**The two-stage baseline like for like: reference weak SINDy on the evaluation and on the training trajectories, a
scaling-exponent readout, learned operators under noise (exp56, notebook 5) -- design and pre-registration (08.10;
written after the development runs listed, before any run on seeds 42-51).**
Why: the mock reviews of the PDE paper (07.10) asked for the reference configuration of weak SINDy (Messenger and
Bortz 2021: test functions from the data's spectrum, MSTLS), the same data for both routes (the learned operators train
on 1024 trajectories, the pipeline saw the 10 evaluation trajectories), paired readouts (m = k against soft, own count
against hard, spurious dimensions scored) and learned-operator rows wherever FINS is compared under noise.
Design:
- pde/method_wsindy.py: the authors' algorithm ported from their MATLAB code (github.com/MathBioCU/WSINDy_PDE:
  findcorners with tau = 1e-10, tauhat = 2; phi_int_weights, piecewise polynomial; get_scales, toggle_scale 2; convNDfft;
  sparsifyDynamics; wsindy_pde_RGLS_seq with lambda = 10^linspace(-4, 0, 100)). Port check on the authors' datasets
  (burgers, KdV, KS; library as their script sets it, 2 above the true power and order): the true terms and no others at
  noise 0, 0.1 and 0.5 of the rms, coefficient errors 1e-5 to 9e-2.
- exp56_wsindy.py: one library for every equation, d_x^k (u^j) with j <= 4, k <= 6, and a constant (the authors' margin
  of 2 over the largest power and order among our equations); stage 2 as exp50 (exact rationals, hard and soft readouts).
  Data: the 10 evaluation trajectories (the authors' K_max = 5000 rows each) or the 1024 training trajectories of the
  learned operator of seed s (exp39 'mixed', exp40; corrupted as that operator's training data), 50 000 rows in all.
  exp56_jobs.py: evaluation data, seeds 42-51, every condition of exp51 (heat, KdV), exp53 (KS), exp52 (BO) and Burgers
  clean; training data, seeds 42-44, clean and 1e-2, all five equations (Burgers clean only). Scored with exp51's scorer
  (tags wsindy, wsindy-train), finite algebra, non-trivial generators, spurious dimensions.
- exponent_readout.py, a second metric introduced after the development runs: the time-scaling exponent a-hat of a span,
  the direction of the plane x d_x + a t d_t + (1 - a) u d_u (spanned by x d_x + u d_u and t d_t - u d_u) deepest inside
  the span, with its containment (a 2 x 2 generalised eigenproblem). BO a = 2, KdV 3, Burgers 2.
- Notebook 5 (Colab GPU): FINS weak through exp48's heat CNOs retrained on 1e-3 and 1e-2 data (seeds 42-44), on equally
  noisy evaluation data; FNO steppers for BO and KS trained on 1e-2-corrupted 'mixed' data (seeds 42-44), FINS weak and
  pointwise through them on equally noisy evaluation data.
Development (seed 7, data 1007; fins_runs/wsindy_dev). Evaluation data: heat 0.01 u_xx at every sigma to 1e-1 (0.010255
at 1e-1), KdV and KS exact to 1e-1 (coefficients within 0.25%), Burgers 0.394 u_xx - 3.141 (u^2)_x; every score 0.000.
Training data, clean: heat, KdV, KS, Burgers exact. Benjamin-Ono: evaluation data clean and 1e-2, u_t = -5.48 u_x
(advection; hard algebra 22-dimensional: missing 0.001, spurious 18; soft 2.69 / 1.90); 3e-2, u_t = -0.063 u_xxx -
0.349 (u^2)_x; training data clean, u_t = -0.365 u_x - 0.029 u_xxx - 0.271 (u^2)_x -- KdV-type surrogates, missing 0.015,
spurious 0.015: the subspace metric cannot tell their scaling (exponent 3) from BO's, because on BO's peaked data the
scaling field is dominated by its u d_u part (the KdV scaling lies 0.997 inside BO's span); exponent readout 2.997 for
both, advection 1.03 with containment 0.39. Readout on registered runs, looked at while building it (seed 42): BO FINS
weak exact 2.004, learned 2.068, PySINDy two-stage soft 1.59 (H 0.05) and 1.59 (H 0.2); KdV FINS weak 3.011.
Predictions:
(LW1) Reference WSINDy on the evaluation data, hard and soft: heat (finite and class), KdV and KS <= 0.05 at every sigma
on >= 9/10 seeds each; Burgers clean likewise.
(LW2) Training data (seeds 42-44), clean and 1e-2: heat, KdV, KS, Burgers (clean) <= 0.05 on 3/3, hard and soft.
(LW3) Benjamin-Ono, every WSINDy run (evaluation data at every level, training data clean and 1e-2), hard readout: no
BO scaling in the span -- exponent readout >= 2.5 or containment < 0.9 -- on every seed.
(LW4) Notebook 5, heat: FINS weak through the retrained CNOs at most a fifth of exp48's pointwise learned row (1.68,
2.96) at 1e-3 and 1e-2, means over seeds 42-44.
(LW5) Notebook 5, BO and KS at 1e-2: FINS weak <= 0.3 (finite) and <= 0.2 (non-trivial), means; FINS pointwise >= 1.0;
BO weak exponent readout within 0.2 of 2 on 3/3 seeds.
No predictions: BO's missing and spurious dimensions for WSINDy; exponent readouts of the existing rows (reported for
every method on BO, KdV and Burgers); the paired-readout tables (reported in full).
Paper, fixed now: reference WSINDy as the main two-stage row (PySINDy's configuration as a sensitivity row), its rows on
the training data, the paired readouts with spurious dimensions, the exponent readout on BO, and learned rows under noise
where notebook 5 provides them; the abstract and introduction say on which equations the pipeline is exact.

**Two-stage baseline on Burgers and with one library for every equation (exp55) -- results (07.10, late evening; CPU,
two workers after the machine restart at 21:18; npz in fins_runs/burgers_runs, noise_weak_runs, bo_runs;
results/burgers, results/noise_weak, results/bo).** Missing dimensions, finite algebra, means over seeds 42-51 (number
of seeds above 0.05 in brackets where not zero):
Burgers, clean: H 0.2 hard and soft 0.000 with either library; H 0.05 0.000 with the library to u_xxxx, 0.300 hard
(seed 44: u_t = 0.391 u_xx - 6.219 u u_x + 0.0094 u u_xxx, the spurious term costing 3) and 0.098 soft with u_xxx.
Library to u_xxxx (and to u_xxx), H 0.2, hard:
heat   0.000 at clean to 3e-2; 0.300 (1) at 5e-2 and 1e-1 -- identical to u_xxx (seed 49)
KdV    0.000 at every sigma -- identical
BO     2.000 at every level (soft 0.576 / 0.748 / 0.406 at clean / 1e-2 / 3e-2, against 0.406 / 0.406 / 0.313 with u_xxx)
H 0.05 with u_xxxx: heat hard 0.400 (1) at 1e-2, 2.86-3.90 from 3e-2 (u_xxx: 0.900, 2.82-3.60), soft 0.007-0.617
(u_xxx 0.159-0.504); KdV hard 1.80 (9) at 1e-2, 2.000 from 3e-2 (u_xxx: 2.000 from 1e-2); BO soft as with u_xxx.
VERDICTS (literal; verdicts_0710.py). (EX1) MET: every variant <= 0.05 on >= 9/10 seeds (H 0.05 with u_xxx: 9/10, the
rest 10/10). (EX2) MET: heat and KdV <= 0.05 on >= 9/10 seeds at every sigma (heat at 5e-2 and 1e-1: 9/10). (EX3) MET:
2.000 on every seed and level, both H. Every earlier verdict unchanged.
Reading: the tuned pipeline does not need the equation's order -- a library to u_xxxx scores as one fitted to each
equation, on Burgers, heat and KdV -- and the independent reviewers' expectation holds: a larger library is a tuning
question, not a case FINS wins. In Table tab:main (seeds 42-44, clean) the pipeline is exact on Burgers, KdV, heat and
heat's class, ahead of every method that learns an operator or a symmetry.

**Learned operators and the published baselines on the new equations -- results (07.10, evening; notebook 3 on a
Colab T4, merged; npz in fins_runs/bo_runs/clean, ks_runs/clean; results/bo, results/ks summary.md/json).**
Seeds 42-44, clean, evaluation data 1000 + s; missing dimensions, finite algebra / non-trivial (BO: Galilean and
scaling; KS: Galilean), means (per seed):
Benjamin-Ono (4)
FINS weak, learned operator          0.018 / 0.006   (0.019 0.028 0.005 / 0.003 0.012 0.002)
FINS pointwise, learned operator     0.098 / 0.031   (0.144 0.068 0.080 / 0.049 0.030 0.014)
FINS weak, exact solver (BO1)        0.000 / 0.000
LieNLSD as published, training / eval. system   1.354 / 1.176 (finite = non-trivial)
two-stage soft, H 0.05 / H 0.2       0.047 / 0.656 on these seeds (0.047 0.046 0.049; 0.035 0.967 0.965); hard 2.000
LieGAN                               2.702 / 0.983   (untrained 2.181 / 0.867)
LaLiGAN                              3.862 / 1.862   (untrained 3.999 / 1.999)
Augerino                             0.872 / 0.872   (untrained the same)
Kuramoto-Sivashinsky (3)
FINS weak and pointwise, learned operator ('mixed')     0.000 / 0.000 (every seed)
FINS weak, learned on zero-mean data                     0.039 / 0.039   (0.029 0.061 0.025)
FINS pointwise, learned on zero-mean data                0.042 / 0.042   (0.036 0.063 0.028)
LieNLSD as published (NL9)           0.000 / 0.000
LIG                                  1.562 / 0.461   (1.559 1.520 1.606; untrained 2.063 / 0.068)
LieGAN                               2.587 / 0.631   (untrained 2.054 / 0.512)
LaLiGAN                              2.982 / 0.982   (untrained 3.000 / 1.000)
Augerino                             0.930 / 0.930   (untrained the same)
VERDICTS (literal; verdicts_0710.py). (NL1) MET: BO learned weak 0.018 / 0.006, pointwise 0.098 / 0.031, weak below
pointwise. (NL2) MET: KS learned 0.000 / 0.000, both forms. (NL3) NOT MET: through the zero-mean steppers the boost
stays inside the three lowest; non-trivial 0.039 (weak) and 0.042 (pointwise), registered >= 0.5, though above the
'mixed' arm's 0.000 on 3/3 seeds. (NL4) MET: LieGAN 2.70 (BO) and 2.59 (KS), LaLiGAN 3.86 and 2.98. (NL5) MET: Augerino
trained = untrained on every seed (difference below 1e-10: its readout is its fixed dictionary). (NL6) MET: FINS learned, weak and pointwise, below every baseline on both
equations (BO 0.018 / 0.098 against Augerino 0.872, LieGAN 2.70, LaLiGAN 3.86; KS 0.000 against Augerino 0.930, LIG
1.56, LieGAN 2.59, LaLiGAN 2.98). (NL7) MET: LIG 1.562 (1.520-1.606 per seed), all three seeds run.
Reported without predictions. NL3's mechanism shows in the spectrum, not in the m = k readout: through the zero-mean
steppers the boost's loss rises from 3.7-5.5e-6 to 2.5-5.5e-3 (x670-1140), below lambda4 = 4.0-4.7e-2 (margins 7-17), and
the own count drops the boost on seed 44 (2 instead of 3). Exploratory (diag_ks_boost.py, data 1000 + s): the zero-mean
steppers move a constant with 12-21% error over 1-16 steps, the 'mixed' ones with 0.4-0.5% (-u_x: 0.4-0.9% for both).
Own counts: BO learned weak 4, 1, 4 (seed 43: lambda2 / lambda1 = 17.5 exceeds lambda5 / lambda4 = 9.4); KS learned 3 on
every seed. Through the BO steppers all four generators lie below lambda5 on 3/3 (margins 25-49, weak) and lambda5 =
8.1e-4 to 1.2e-3: the inviscid-Burgers near-symmetry that the exact solver puts fifth at 2.8e-6 (development seed)
moves up, so the own count reads 4 rather than 5 there. LIG holds the boost better untrained (non-trivial 0.068) than
trained (0.461). Operator errors (held-out increment error of the best state): KS 'mixed' 0.003 on every seed,
zero-mean 0.003 / 0.004 / 0.003; BO 0.009 / 0.021 / 0.013. Cross-device: the six noisy WN10 runs repeated on the GPU
agree with the CPU runs to 1.7e-7 of max |u| in the data, 1.9e-5 in the eigenvalues and to four decimals in every
score; notebook 3's KS weak files are notebook 2's, byte for byte (carried on the Drive, not recomputed).
Reading: through learned operators FINS recovers both new algebras -- BO 0.018 against LieNLSD's 1.18-1.35 and, given
the same k, below the two-stage pipeline's soft readout at either H on every seed (0.046-0.049 at H 0.05); KS exactly,
level with the two-stage pipeline and LieNLSD, ahead of LieGAN, LaLiGAN and Augerino. The zero-mean ablation shows the
training-data limit as a 670-1140-fold rise of the boost's loss that the m = k readout does not see, because KS's first
non-symmetry is 7-17 times further.

**Two-stage baseline on Burgers and with one library for every equation; FINS weak through the learned heat operators
(exp55) -- design and pre-registration (07.10, evening; written after the development runs listed, before any run on
seeds 42-51).**
Why: Table tab:main has no two-stage row on Burgers; the two-stage library was matched to each equation's order (to
u_xxx; to u_xxxx for KS), a prior a user without the equation does not have; heat's full class through the learned
CNOs is FINS's worst result (pointwise 1.59 against LieNLSD 0.43) and the weak form has not been run through them.
Design (exp55_extra.py; scores as exp51: finite algebra, non-trivial generators, heat's class):
- two-stage-o4, two-stage-h05-o4: exp50 as in exp51 (H 0.2 / 0.05, 200 subdomains, threshold 0.1) with the library to
  u_xxxx (--order 4), KS's, on heat and KdV at exp51's six conditions (noise_weak_runs) and on Benjamin-Ono at exp52's
  three (bo_runs), seeds 42-51; KS's rows already use it.
- Burgers: exp50 with exp51's library (to u_xxx) and with the library to u_xxxx, H 0.2 and 0.05, clean, seeds 42-51,
  evaluation data 1000 + s (the protocol's Burgers data; burgers_runs; scored by exp55_extra.py summary).
- weak-learned (heat): FINS weak, exp51's frozen recipe, through the protocol's heat CNOs
  (S_theta_heat_s{s}_g64x128_cno_smooth_L1.pt), seeds 42-44, clean; registered separately (EX4) once its development
  run is in.
Development (seed 7, data 1007; fins_runs/extra_dev; missing dimensions, finite algebra):
- Burgers, clean: 0.000 for H 0.2 and 0.05, library to u_xxx and to u_xxxx, hard and soft.
- Library to u_xxxx, H 0.2: heat and KdV 0.000 at clean, 1e-2, 5e-2 and 1e-1, hard and soft (the u_xxx library: the
  same). H 0.05: heat soft 0.007 / 0.110 at 5e-2 / 1e-1 (u_xxx 0.137 / 0.893), hard 2.752 / 4.000 (4.000 / 4.000); KdV
  as with u_xxx (soft 0.674 / 0.013, hard 2.000).
- Benjamin-Ono, library to u_xxxx: hard 2.000 clean and at 1e-2, both H; soft at H 0.2 0.927 / 0.953 (u_xxx 0.030 /
  0.030), at H 0.05 0.042 / 0.042 (u_xxx the same).
- Exploratory, not registered (clean, H 0.2, every s-th snapshot): weak SINDy identifies heat, KdV and KS at s = 2, 4
  and 8 (KdV at s = 8: u_t = -0.9867 u_xxx - 6.0104 u u_x; KS: -0.993 u_xx - 0.997 u_xxxx - 1.005 u u_x). The scorer's
  time margin of 8 snapshots leaves nothing to score at s = 8 (16 snapshots), so those rows are not read.
- Heat through the CNO of seed 42 on data 1007: FINS pointwise (exp40 --op cno) 0.597 finite, 1.579 class.
Predictions:
(EX1) Burgers, clean: every two-stage variant (H 0.2 and 0.05; library to u_xxx and to u_xxxx), hard and soft, misses
<= 0.05 on >= 9/10 seeds.
(EX2) Library to u_xxxx, H 0.2, hard and soft: heat and KdV <= 0.05 at every sigma on >= 9/10 seeds each.
(EX3) Library to u_xxxx, Benjamin-Ono: the hard readout misses 2.000 on every seed at every level, both H.
No predictions: H 0.05 with the library to u_xxxx on heat and KdV; the Benjamin-Ono soft readouts with it; own counts.
Paper, fixed now: Burgers' two-stage row in Table tab:main and the u_xxxx library's rows (or one sentence with their
numbers) in the noise section, whatever the verdicts.
(EX4, registered 07.10 at 22:00, before any of its runs and without a development result: the development run through
the CNO of seed 42 on data 1007 died in the machine restart at 21:18 after 2.2 h of its ~2.5 h on one CPU core.) Heat
through the protocol's CNOs, seeds 42-44, clean, FINS weak: finite algebra <= 0.3; heat's class (10) >= 1.0 on average
-- the class is limited by the operators' transport error along the heat polynomials (exp43: 1.0-1.9 times the first
non-symmetry), which the weak form does not change. Runs on a Colab T4 (FINS_nb4_heat_learned_colab.ipynb),
where the CNOs are.

**Noise study in weak form (exp49, exp50, exp51) -- results (07.10; heat and KdV, seeds 42-51; KdV's weak rows on a
Colab T4, the rest on the CPU; results/noise_weak/summary.md, summary.json from exp51 summary --exp48).**
Missing dimensions, finite algebra (heat 6, KdV 4) / non-trivial generators (heat u d_u, Galilean, scaling, projective;
KdV Galilean, scaling), means over seeds; LieNLSD_F on seeds 42-44 (clean to 1e-2: exp48's fits):
heat                           clean        1e-3         1e-2         3e-2         5e-2         1e-1
FINS weak                      0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.01 / 0.01  0.05 / 0.02  0.37 / 0.07
FINS pointwise                 0.00 / 0.00  1.71 / 0.83  2.72 / 0.99  3.01 / 1.03  3.05 / 1.06  3.12 / 1.12
two-stage H 0.2, hard          0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.30 / 0.30  0.30 / 0.30
two-stage H 0.05, hard         0.00 / 0.00  0.00 / 0.00  0.90 / 0.90  2.82 / 2.82  2.96 / 2.96  3.60 / 3.60
LieNLSD_RHS@eval               0.29 / 0.29  1.07 / 1.07  1.45 / 1.45  1.45 / 1.45  1.49 / 1.49  1.65 / 1.65
LieNLSD_F                      0.55         1.23         1.60         1.26 / 1.26  1.11 / 1.11  1.19 / 1.19
KdV
FINS weak                      0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.04 / 0.00  0.20 / 0.00  0.91 / 0.00
FINS pointwise                 0.01 / 0.01  1.14 / 0.08  2.16 / 0.17  2.17 / 0.17  2.18 / 0.18  2.18 / 0.18
two-stage H 0.2, hard          0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.00 / 0.00  0.00 / 0.00
two-stage H 0.05, hard         0.00 / 0.00  0.00 / 0.00  2.00 / 2.00  2.00 / 2.00  2.00 / 2.00  2.00 / 2.00
LieNLSD_RHS@eval               0.00 / 0.00  0.51 / 0.51  0.52 / 0.52  0.98 / 0.98  0.84 / 0.84  0.68 / 0.68
LieNLSD_F                      0.91         1.13         1.43         1.32 / 1.32  1.20 / 1.20  1.13 / 1.13
Heat's class (10), FINS weak: 0.00 / 0.00 / 0.00 / 0.01 / 0.06 / 0.38; two-stage H 0.2 hard 0.00 to 3e-2, 0.70 at 5e-2
and 1e-1; H 0.05 hard 2.10 to 7.60 from 1e-2; LieNLSD_RHS@eval 0.32 clean to 2.09 at 1e-1.
VERDICTS (literal; verdicts_0710.py). (WN1) MET: FINS weak <= 0.05 on 10/10 seeds clean and at 1e-3, both equations.
(WN2) MET: at 1e-3 and 1e-2 weak 0.000 / 0.001 (heat) and 0.000 / 0.002 (KdV) against pointwise 1.71 / 2.72 and 1.14 /
2.16. (WN3) MET: weak below pointwise at 3e-2, 5e-2, 1e-1 on both (heat 0.013 / 0.053 / 0.372 against 3.01-3.12; KdV
0.036 / 0.200 / 0.914 against 2.17-2.18). (WN4) MET: heat 0.001 at 1e-2 and 0.053 at 5e-2; KdV 0.002 and 0.200. (WN5)
MET: KdV's non-trivial missing 0.000-0.001 at every sigma <= 5e-2 (0.003 at 1e-1). (WN6) MET: H 0.2 hard and soft <=
0.05 on 10/10 seeds everywhere except heat at 5e-2 and 1e-1, 9/10 (seed 49: u_t = 0.0099 u_xx + 3.5e-4 u u_xx - 6.7e-5
u^2 u_xx at 5e-2, two spurious terms of 3.5% and 0.7% that cost 3.0 dimensions). (WN7) MET: H 0.05 hard misses >= 1 at
5e-2 and 1e-1 on 8/10 and 9/10 seeds (heat) and 10/10 (KdV, 2.00 on every seed: the identified equation keeps neither
boost nor scaling). (WN8) MET: LieNLSD_RHS@eval misses more than FINS weak at every sigma from 1e-3 to 5e-2 (heat
1.07-1.49 against 0.000-0.053; KdV 0.51-0.98 against 0.000-0.200). (WN9) MET: own count 10 (heat) and 4 (KdV) on 10/10
seeds clean, at 1e-3 and at 1e-2. (WN10) MET: through the steppers retrained on the corrupted data (seeds 42-44), KdV
weak 0.004 clean, 0.004 at 1e-3 and 0.151 at 1e-2 (0.068 / 0.259 / 0.126; bounds 0.05, 0.236 and 0.3), against exp48's
pointwise learned 1.18 and 2.20; non-trivial 0.001 / 0.001 / 0.017; own count 4 on every run but seed 43 at 1e-2 (1).
The six noisy runs ran on the CPU (about an hour each beside other jobs); notebook 3 repeats them on the GPU, a
cross-device check when it arrives.
Reported without predictions: heat's own count breaks above 1e-2 (3e-2: 10 on 7/10 seeds; 5e-2 and 1e-1: 4, the
superposition fields beta d_u, on 6/10 and 10/10), KdV's from 5e-2 (5 on 7/10 and 10/10); the weak form's worst run at
1e-1 misses 1.03 (heat) and 0.99 (KdV). LieNLSD as published at the new levels: heat 1.11-1.26, KdV 1.13-1.32 (own count
2 on KdV at every level).
Reading: the weak form removes the noise fragility of the pointwise criterion -- at 1e-2 it misses 0.001 and 0.002
against 2.72 and 2.16 -- and what noise costs it first are the translations (KdV) and, on heat at 1e-1, a third of a
dimension; the two-stage pipeline with the tuned subdomain size matches it on both equations except one heat seed whose
identified equation carries two small spurious terms, and with PySINDy's default size it fails from 1e-2 (KdV on every
seed, heat on 3/10 seeds there and on most from 3e-2). The exact right-hand side in LieNLSD's determining system is
worse than both from 1e-3 to 5e-2 (at 1e-1 it misses 0.68 on KdV against the weak form's 0.91): a generator survives
when its determining equations vanish identically in the jets (the translations, heat's superposition fields, KdV's
boost) and is lost in part when they need the equation to hold on the noisy jets (the scalings, which move t, and heat's
generators proportional to u).

**Kuramoto-Sivashinsky, exact operator (exp53), and LieNLSD on it -- results (07.10; weak rows on a Colab T4,
the rest on the CPU; npz in fins_runs/ks_runs; results/ks/summary.md, summary.json).** Seeds 42-51 (LieNLSD_F
42-44); missing dimensions, finite algebra / non-trivial (Galilean), means over seeds:
                               clean          1e-2           3e-2           5e-2           1e-1
FINS weak                      0.000 / 0.000  0.000 / 0.000  0.001 / 0.000  0.004 / 0.000  0.017 / 0.001
FINS pointwise                 0.000 / 0.000  1.275 / 0.008  2.038 / 0.078  2.330 / 0.394  2.894 / 0.895
two-stage H 0.2, hard and soft 0.000 / 0.000  0.000 / 0.000  0.000 / 0.000  0.000 / 0.000  0.000 / 0.000
two-stage H 0.05, hard         0.000 / 0.000  1.000 / 1.000  1.000 / 1.000  1.000 / 1.000  1.000 / 1.000
two-stage H 0.05, soft         0.000 / 0.000  0.999 / 0.999  0.997 / 0.997  0.990 / 0.990  0.985 / 0.985
LieNLSD_RHS@eval               0.000 / 0.000  0.000 / 0.000  0.000 / 0.000  0.000 / 0.000  0.000 / 0.000
LieNLSD_F (training system)    0.000 / 0.000  0.682 / 0.682
LieNLSD_F@eval                 0.000 / 0.000  0.701 / 0.701
VERDICTS (literal; verdicts_0710.py). (KS1) MET: FINS weak <= 0.05 on 10/10 seeds at every sigma (largest 0.033, at
1e-1). (KS2) MET: own count 3 on 10/10 seeds at every sigma. (KS3) MET: pointwise 0.000 clean, 1.28 / 2.04 / 2.33 / 2.89
from 1e-2. (KS4) MET: H 0.2 hard and soft 0.000 on 10/10 seeds at every sigma (seed 42 at 1e-1: u_t = -0.993 u_xx -
0.998 u_xxxx - 1.006 u u_x). (KS5) MET: H 0.05 hard misses 1.000 (the boost) on 10/10 seeds at every sigma >= 1e-2, its
equation wrong (weak residual 0.89-0.95 at 1e-1). (NL8) MET: LieNLSD_RHS@eval 0.000 on 10/10 seeds at every sigma. (NL9)
MET: LieNLSD_F 0.000 / 0.000 clean (F's held-out relative MSE 5e-5 to 9e-5), 0.682 / 0.701 at 1e-2 (0.52-0.92 per seed;
MSE 0.42 on every seed: differences to u_xxxx of 1% noise; own count 2, the boost lost).
Reported: FINS pointwise loses d_t from 1e-2 (novel containment 0.08-0.36, d_x kept to 0.60-0.97) and the boost from
5e-2 (non-trivial 0.39, 0.90 at 1e-1). Untrained baselines so far (exp54 null, seed 42): LieGAN 2.13 / 0.38, LaLiGAN
3.00 / 1.00, Augerino 0.93 / 0.93.
Reading: on chaotic, broadband data whose equation lies in the library, the tuned two-stage pipeline and the exact
right-hand side are exact at every level, and FINS's weak form is within 0.033 of them on every run; what separates the
methods is what they assume -- PySINDy's default subdomain size or F fitted on noisy fourth derivatives -- and FINS's
pointwise form, which reads u_t and u_x from the noisy snapshots.

**Benjamin-Ono, exact operator (exp52) -- results (CPU, 07.10; npz in fins_runs/bo_runs; results/bo/summary.md,
summary.json, fifth.json from exp52_bo.py summary and fifth).** Seeds 42-51 (LieNLSD 42-44); missing dimensions,
finite algebra / non-trivial (Galilean and scaling), means over seeds:
                               clean          1e-2           3e-2
FINS weak                      0.000 / 0.000  1.000 / 0.001  1.000 / 0.001
FINS weak, own count           0.000 / 0.000  0.301 / 0.000  1.000 / 0.001
FINS pointwise                 0.000 / 0.000  2.208 / 0.292  2.554 / 0.565
two-stage H 0.2, soft          0.406 / 0.406  0.406 / 0.406  0.313 / 0.313
two-stage H 0.2, hard          2.000 / 2.000  2.000 / 2.000  2.000 / 2.000
two-stage H 0.05, soft         0.044 / 0.044  0.046 / 0.046  0.057 / 0.057
two-stage H 0.05, hard         2.000 / 2.000  2.000 / 2.000  2.000 / 2.000
LieNLSD_F (training system)    1.354 / 1.354  1.726 / 1.726
LieNLSD_F@eval                 1.176 / 1.176  1.622 / 1.622
VERDICTS (literal; verdicts_0710.py). (BO1) MET: FINS weak and FINS pointwise <= 0.05 clean on 10/10 seeds each. (BO2)
MET: the hard readout misses 2.000 on every seed at every level, both H, finite and non-trivial. (BO3) MET: FINS weak's
non-trivial 0.000 / 0.001 / 0.001, below the hard readouts (2.000) and LieNLSD as published (1.18-1.73 on seeds 42-44,
against FINS weak's 0.000-0.001 there). (BO4) MET: 1.000 at 1e-2 and at 3e-2, one dimension on every seed. (BO5) MET:
2.208. (BO6) MET: 1.354 / 1.176 (Adan; the clean fits first ran with Adam, 1.163 / 1.267, see the run log (viii)). F's
held-out relative MSE 2.1e-3 to 3.0e-3 clean, 6.6e-2 to 8.3e-2 at 1e-2.
Reported without predictions. Two-stage soft: H 0.2 is bimodal over seeds (0.03 or 0.97 by which surrogate weak SINDy
returns: 0.97 on 4/10 seeds clean and at 1e-2, 3/10 at 3e-2); H 0.05, 0.044-0.057 on the means (0.029-0.069 per seed).
Own counts of FINS weak: 5 on 10/10 seeds clean, 5 on 7/10 and 4 on 3/10 at 1e-2, 4 on 10/10 at 3e-2. The fifth
eigenvalue (exp52 fifth): clean, lambda5 / lambda4 = 90-192 and lambda6 / lambda5 = 2.2e3-3.8e3, and the
inviscid-Burgers projective field x t d_x + t^2 d_t + (x - t u) d_u, reduced to its part orthogonal to the algebra, lies
inside the five lowest eigen-generators on every seed (share 1.000) and outside the four lowest (share 0.001-0.013).
Under noise it lies inside the four lowest (1.000 on every seed; lambda5 / lambda4 = 23-50 at 1e-2 and 23-63 at 3e-2,
lambda6 / lambda5 = 36-79 and 4.7-9.4), where it takes the place of one translation: d_x's part orthogonal to the rest
of the algebra keeps a norm share of 0.46-0.50 inside them (0.21-0.25 of its square): the floor that noise sets for the
exact generators reaches the near-symmetry's residual, which the dispersive term H u_xx alone sets. Untrained baselines
(exp54 part null, seeds 42-44; NL registration): LieGAN 2.18 / 0.87, LaLiGAN 4.00 / 2.00, Augerino 0.87 / 0.87.
Reading: on the nonlocal equation the equation-based routes see a local surrogate -- weak SINDy returns one at both H
whose exact algebra is the two translations, and LieNLSD, which fits F on local jets, misses 1.2-1.7 -- while FINS
through the solver recovers the whole algebra clean and keeps Galilean and scaling at every noise level; what noise
costs it is one translation, traded for a near-symmetry it ranks fifth when the data are clean.

**Learned operators and the published baselines on the new equations; LieNLSD on Kuramoto-Sivashinsky -- design and
pre-registration (07.10; written after the development runs listed, before any run on seeds 42-51).**
Why: on Benjamin-Ono and Kuramoto-Sivashinsky FINS has so far run through the exact solvers and been compared with the
two-stage pipeline only; the protocol's question is equation-free discovery through a learned operator against the
published equation-free baselines trained on the same data. Rows, all scored as exp51 (finite algebra; non-trivial
= Galilean and scaling on BO, Galilean on KS; missing dimensions, m = k):
- Learned arms (exp52, exp53 parts weak-learned, pointwise-learned): FINS weak (exp51's frozen recipe) and FINS
  pointwise through FNO steppers trained on 1024 'mixed' trajectories of seed s (exp39's recipe: 400 epochs, 64 modes,
  no positional embedding), evaluation data 1000 + s, seeds 42-44, clean. BO s42 was trained on the CPU, the other
  steppers train on a Colab T4 (FINS_nb3_learned_baselines_colab.ipynb). KS ablation (parts ...-attractor):
  the same rows through steppers trained on zero-mean attractor data, the steppers of FINS_exp53_ks_steppers_colab.ipynb
  (see the correction below). Mechanism: the Galilean boost adds a constant to u, and KS conserves the mean; a stepper
  that never saw a trajectory with another mean has an untrained derivative in exactly the direction the boost needs.
- exp54: LieGAN, LaLiGAN, Augerino on BO and KS, LIG on KS (the protocol's settings: 10 epochs, batch 1, torchdiffeq
  adjoint; LIG on BO is not run: its residual is evaluated by finite differences on the transformed grid, where the
  Hilbert transform is not defined), trained on the steppers' 1024 'mixed' trajectories, k = 4 (BO) and 3 (KS)
  generators as in the protocol, seeds 42-44, clean; their untrained versions (exp41 --untrained: own-class nulls,
  parts null and lig-null) on the same seeds.
- LieNLSD on KS (exp53 parts lienlsd, lienlsd-f; the prolongation in method_lienlsd.py now runs to fourth order, a
  Leibniz term per order; on heat and KdV the systems are bit for bit those of the third-order code): the exact
  right-hand side with the system on the (corrupted) evaluation data, every seed and sigma; as published, F fitted
  on jets to u_xxxx of the 1024 'mixed' training trajectories, systems on those and on the evaluation trajectories,
  seeds 42-44, clean and 1e-2 (Adan, 200 epochs).
Development (seed 7; data 1007; missing dimensions, finite algebra / non-trivial):
  BO, stepper s42 (one-step error 0.1-0.3%): FINS pointwise 0.106 / 0.058 (d_x 0.98, d_t 1.00, Galilean 0.99, scaling
  1.00; eigenvalues 1.3e-6 .. 3.2e-5, the fifth 4.1e-4); FINS weak 0.007 / 0.003 (own count 4; eigenvalues
  2.7e-7 .. 5.5e-5, the fifth 8.8e-4 -- 2.8e-6 through the exact solver).
  KS learned: none (no KS stepper on the CPU; notebook 3 prints a pointwise check of every stepper on data 1007).
  Untrained nulls:  BO LieGAN 1.97 / 0.48, LaLiGAN 4.00 / 2.00, Augerino 0.88 / 0.88;
                    KS LieGAN 2.08 / 0.26, LaLiGAN 3.00 / 1.00, Augerino 0.93 / 0.93.
  On KS the scaled Galilean field is nearly a constant phi, which a random affine LieGAN generator contains in part
  (0.78 untrained); Augerino's seven fixed fields contain d_x, d_t, t d_x and u d_u but not d_u.
  LieNLSD, KS: RHS@eval 0.000 / 0.000 at every sigma <= 1e-1 -- KS's three generators satisfy their determining
  equations identically in the jets: none has a non-constant tau, the only coefficient that brings the data's u_t
  into the system, so with the exact right-hand side noise cannot remove them (KdV's scaling, tau = 3t, it can).
  F as published on the 'mixed'
  data: clean 0.000 / 0.000 on both systems (held-out relative MSE of F 9.4e-5); 1e-2 0.93 / 0.93 on both (MSE
  0.42: jets to u_xxxx by differences of 1% noise; own count 2) -- on the zero-mean data, before the correction:
  clean 0.000, 1e-2 0.78 / 0.62 (MSE 0.50).
Predictions (means over seeds 42-44 unless stated; finite algebra unless stated):
(NL1) BO learned arm: FINS weak and FINS pointwise <= 0.3, non-trivial <= 0.2; FINS weak <= FINS pointwise.
(NL2) KS learned arm ('mixed'): FINS weak and FINS pointwise <= 0.3, non-trivial <= 0.3.
(NL3) KS ablation ('attractor'): FINS weak and FINS pointwise non-trivial >= 0.5, and above the 'mixed' arm's
      non-trivial score on every seed (same row).
(NL4) LieGAN and LaLiGAN, trained: >= 1.5 on BO and >= 1.0 on KS.
(NL5) Augerino, trained: within 0.05 of its untrained readout on every seed, both equations (its readout is its fixed
      generator set; protocol: trained = untrained).
(NL6) FINS through the learned operators ('mixed'; weak and pointwise) misses fewer than each of LieGAN, LaLiGAN and
      Augerino on both equations, and fewer than LIG on KS.
(NL7) LIG on KS: >= 1.0, on the seeds that finish (42, 43, 44 in that order; a seed not run is reported as not run).
(NL8) LieNLSD_RHS@eval on KS: <= 0.05 at every sigma, on >= 9/10 seeds.
(NL9) LieNLSD_F on KS (seeds 42-44), both systems: <= 0.1 clean, >= 0.3 at 1e-2.
No predictions: the baselines' non-trivial scores and untrained nulls, own counts, the stepper checks on data 1007
(all reported). Paper, fixed now: the BO and KS tables get the learned rows, the baselines with their untrained nulls,
and the ablation (as a limit of learned operators: a symmetry is visible only along directions the training data
probe), whatever the verdicts.

**New equations -- run log (07.10).** (i) KS training data, corrected before any run that trains on them:
ks.training_initial returned zero-mean attractor states, while the protocol's recipe (exp39 for KdV, bo.py) puts the
soliton half of its training trajectories on a constant background (zero with p = 0.5, else uniform in [-0.5, 0.5]). KS
conserves the mean, so on zero-mean data no training trajectory shows how a constant moves a solution, the direction in
which the Galilean boost acts. Now 'mixed' = attractor states on a zero (p = 0.5) or uniform [-0.5, 0.5] background (501
of 1024 zero-mean at seed 7; such a trajectory is u(x - bt, t) + b, which the solver reproduces to 8e-10 of max |u|),
and 'attractor' = zero mean. Evaluation data keep zero mean, so no registered KS row moves (weak, pointwise and
two-stage use the evaluation data only). The KS LieNLSD_F development (seed 7) was redone on the new data (old files in
fins_runs/ks_dev/old_zero_mean). The KS steppers of FINS_exp53_ks_steppers_colab.ipynb (seeds 42-44, trained under the
name 'mixed') are zero-mean ones: notebook 3 renames them 'attractor' and they become the ablation registered above;
exp39 refuses a KS 'mixed' checkpoint that lacks the new recipe tag. (ii) LieNLSD_F on BO's development data was fitted
with Adam when its pre-registration was written (1.14 / 1.22 in that table); the optimiser LieNLSD publishes, Adan,
which every registered fit uses, gives 1.60 / 1.27, both far on the same side of BO3's and BO6's thresholds. (iii) exp54
writes LIG to its own file (baselines-lig_ks_s<seed>.npz) with the protocol's settings (10 epochs, batch 1), so the
other three baselines finish and are kept whatever happens to LIG's two hours per seed. (iv) KdV's learned weak rows at
1e-3 and 1e-2 (WN10; 6 runs) run in the same notebook as the new arms, through the steppers retrained on the corrupted
data (fins_ckpt_nb3.zip: exp48's checkpoints and BO s42). (v) The machine was reclaimed at 05:26, about when it
went idle, and restarted at 10:07: the workers lost those hours (no npz was written
after 05:26; relaunched 10:10). Long CPU runs progress only while the machine is kept active. (vi) exp41 with LieNLSD_RHS@eval
alone now builds one training trajectory instead of 1024: that row never reads them, and on KS's development data at
1e-2 every array of the output is bit for bit the same (KS's training set takes ~10 min per seed on one thread). (vii)
Notebook 2 (FINS_exp53_ks_steppers_colab.ipynb, fins_exp53_steppers.zip, run 03:31-04:31 on a T4 with the zip of 02:36)
merged: the 50 KS weak runs (about 30 s each), their reservations deleted; against the local pointwise and two-stage
files of the same seeds and levels the evaluation data differ by at most 7.1e-7 of max |u| (53 pairs; the chaotic
transient amplifies the devices' roundoff). BO steppers s43, s44: held-out increment error 0.021, 0.013. KS steppers
s42-s44 were trained on zero-mean data (held-out increment error 0.003, 0.004, 0.003) and are now
S_theta_ks_s{s}_attractor_stepper.pt, the registered ablation. Their pointwise checks on data 1007, read after NL1-NL9
were written: the three lowest eigenvalues hold d_x, d_t and Galilean (containment 1.00 on every seed), the third
eigenvalue 1.5e-4 to 2.3e-4, the fourth 1.2e-2; the exact boost's loss, 1.4e-3 to 2.3e-3, is about a hundred times the
translations' (0.9e-5 to 1.6e-5) and five to nine times below the fourth eigenvalue. The zero-mean steppers carry the
boost, against NL3's mechanism. (viii) The three registered BO clean LieNLSD_F fits (seeds 42-44, 01:49-02:28, before
the first restart) ran with Adam, the code's fallback when Adan is missing; the noise study's run log (iv) said every
fit uses Adan, which holds for every other fit (97 fit logs checked, the 8 with Adam are these three, their copies in
results/ and two development runs). They were moved to fins_runs/bo_dev/adam_registered and are rerun with Adan; both
are reported, the verdicts use Adan (LieNLSD as published).

**Kuramoto-Sivashinsky, exact operator (exp53) -- design and pre-registration (07.10; written after development runs
on seed 7, data 1007, before any run on seeds 42-51).**
Why: the mock reviews ask for a chaotic case under a frozen recipe. u_t + u u_x + u_xx + u_xxxx = 0 on the torus
[0, 22) (ks.py), the smallest box studied in detail with a chaotic attractor (lambda_max ~ 0.043); point symmetries
d_x, d_t and the Galilean boost t d_x + d_u (u_xx and u_xxxx weigh differently: no scaling). The equation lies in a
polynomial jet library, so here the two-stage pipeline is expected to work; the test is broadband chaotic data.
Setup: ten evaluation trajectories per seed on the attractor (random fields of Fourier modes 1..4, a transient of 200
at h = 0.01, then 128 x 128 snapshots over T = 20, about one Lyapunov time; ETDRK4 with the full-circle contour, 2/3
dealiasing, 16 internal steps per snapshot); the band [0.2, 0.8] L scored; noise as exp51. Checks: std(u) 1.17, max
|u| 2.9, n95 = 4; a 1e-6 perturbation grows x3.4 over T. Rows (exp53_ks.py; npz in fins_runs/ks_runs, scores in
results/ks/): FINS weak with exp51's frozen recipe (KS: n95 = 4, M = 5 on 77 scored points), FINS pointwise (exp39 as
exp51's), two-stage H 0.2 and H 0.05 with the library to u_xxxx (exp50 --order 4), hard and soft. LieNLSD is not run:
the prolongation in method_lienlsd.py stops at third order. sigma = 0, 1e-2, 3e-2, 5e-2, 1e-1; seeds 42-51; scores as
exp51 (finite algebra; non-trivial = Galilean; own counts).
Development (seed 7; missing dimensions, finite algebra / non-trivial):
                               clean         1e-2          3e-2          5e-2          1e-1
FINS weak                      0.00 / 0.00   0.00 / 0.00   0.00 / 0.00   0.00 / 0.00   0.01 / 0.00
FINS pointwise                 0.00 / 0.00   1.51 / 0.01   2.03 / 0.05   2.15 / 0.21   2.85 / 0.85
two-stage H 0.2, hard and soft 0.00 / 0.00   0.00 / 0.00   0.00 / 0.00   0.00 / 0.00   0.00 / 0.00
two-stage H 0.05, hard         0.00 / 0.00   1.00 / 1.00   1.00 / 1.00   1.00 / 1.00   1.00 / 1.00
two-stage H 0.05, soft         0.00 / 0.00   1.00 / 1.00   1.00 / 1.00   0.98 / 0.98   0.99 / 0.99
Readings: FINS weak keeps a gap of two orders at every level (three lowest eigenvalues 3e-4 to 1.2e-3 at 1e-1, the
fourth 3.7e-2; own count 3 everywhere); pointwise FINS loses the translations from 1e-2 and part of Galilean from 5e-2.
At H 0.2 weak SINDy returns KS with every coefficient within 1% at every level (1e-1: u_t = -0.996 u_xx - 0.999
u_xxxx - 1.007 u u_x); at PySINDy's default H 0.05 it returns a wrong equation from 1e-2 on (weak residual 0.72-0.90)
whose algebra lacks Galilean.
Predictions (seeds 42-51; means over seeds unless stated):
(KS1) FINS weak <= 0.05 at every sigma, on >= 9/10 seeds.
(KS2) FINS weak's own count equals 3 at every sigma on >= 8/10 seeds.
(KS3) FINS pointwise <= 0.05 clean and >= 1.0 at every sigma >= 1e-2.
(KS4) Two-stage H 0.2, hard and soft: <= 0.05 at every sigma, on >= 9/10 seeds.
(KS5) Two-stage H 0.05, hard: >= 0.5 at every sigma >= 1e-2, on >= 5/10 seeds.
No predictions: non-trivial scores (reported). The weak rows run on a Colab GPU (same code; ks.py on CUDA when
available; local claims reserve those jobs). A learned-operator arm (FNO steppers on the attractor, seeds 42-44) is
registered separately after its own development run.

**Benjamin-Ono, exact operator (exp52) -- design and pre-registration (07.10; written after development runs on seed
7, data 1007, before any run on seeds 42-51).**
Why: the mock reviews ask for an equation where equation-based methods fail, under a frozen recipe; the applied-math
review names Benjamin-Ono. u_t + u u_x = H u_xx (H the Hilbert transform) is nonlocal: its right-hand side has no
expression in the jet (x, t, u, u_x, ...), so Lie's algorithm cannot be run on it as written, weak SINDy with a
polynomial library can only return a local surrogate, and LieNLSD can only fit F on local jets. Its point symmetries
are d_x, d_t, the Galilean boost t d_x + d_u and the scaling x d_x + 2 t d_t - u d_u (verified: the Hilbert transform
commutes with translations and positive dilations and annihilates constants).
Setup (bo.py): torus [0, 2 pi), periodised two-soliton evaluation trajectories (u = 4c / (1 + c^2 (x - ct)^2) summed over
images, closed form 2 sinh(1/c) / (cosh(1/c) - cos(x - x0)), an exact periodic travelling wave of speed coth(1/c); c in
[3, 5], centres at [0.30, 0.34] L and [0.48, 0.52] L), 128 x 128 snapshots over T = 0.2, the band [0.2, 0.8] L scored as
for KdV; ETDRK4 with 2/3 dealiasing and 20 internal steps per snapshot; noise as exp51. Checks: a single soliton
travels at the periodic wave's speed (c = 4: 4.085 against coth(1/4) = 4.083) and keeps its shape to 1e-3.
Solver note: bo.py and ks.py first took the ETDRK4 contour mean over the upper half circle, as kursiv.m does, which is
exact only for real L and through its real part. The error then sits in the anti-Hermitian part of the Fourier state,
which the real fields never see unless it grows: under KS's linearly unstable modes it grows like e^(t/4) and swamps
double precision near t = 160 (the run blew up; with the full circle it does not). For BO (|E| = 1) it stays bounded:
the saved fields are unchanged to 5e-13, and the repository's KdV solver (half circle for |hL| < 1) differs from a
float64 full-circle version by 1.2e-5 of max |u| over T = 0.05. No earlier result moves; the full circle is used now.
Rows (exp52_bo.py, npz in fins_runs/bo_runs, scores in results/bo/): FINS weak with exp51's frozen recipe (BO: n95 = 7,
M = 8 on 77 scored points), FINS pointwise (exp39 as exp51's pointwise rows), two-stage H 0.2 and H 0.05 hard and soft
(exp51's settings: library u^a d_x^m u to u_xxx), LieNLSD as published (F fitted on jets to u_xxx of 1024 'mixed'
training trajectories, determining system on the training and on the evaluation trajectories; seeds 42-44, clean and
1e-2). sigma = 0, 1e-2, 3e-2; seeds 42-51; scores as exp51 (finite algebra; non-trivial = Galilean and scaling, each
orthogonalised against the other generators; own counts).
Development (seed 7; missing dimensions, finite algebra / non-trivial):
                               clean         1e-2          3e-2
FINS weak                      0.00 / 0.00   1.00 / 0.00   1.00 / 0.00
FINS pointwise                 0.00 / 0.00   2.24 / 0.31   2.32 / 0.32
two-stage H 0.2, hard          2.00 / 2.00   2.00 / 2.00   2.00 / 2.00
two-stage H 0.2, soft          0.03 / 0.03   0.03 / 0.03   0.03 / 0.03
two-stage H 0.05, hard         2.00 / 2.00   2.00 / 2.00   2.00 / 2.00
two-stage H 0.05, soft         0.04 / 0.04   0.04 / 0.04   0.05 / 0.05
LieNLSD_F (training system)    1.14 / 1.14
LieNLSD_F@eval                 1.22 / 1.22
Readings. (i) Weak SINDy returns the same local surrogate at every level, u_t = -1.37 u_x - 0.12 u_xxx - 0.21 u u_x
- 0.06 u^2 u_x - 0.004 u u_xxx (H 0.2; weak residual 2.5%; H 0.05: -4.6 u_x - 0.09 u_xxx + 0.52 u u_x + ..., 5.4%);
its exact algebra is the two translations, so the hard readout misses Galilean and scaling everywhere; the soft
readout, given k = 4, picks up the surrogate's two next-smallest directions (singular values 2e-3 and 1e-2 of the
largest), which lie near them. (Two-stage rows with exp50's exact-rational stage 2; in floats the H 0.05 soft row
at 3e-2 had read 0.22 on one run and 0.05 on another.) (ii) FINS's spectrum, clean: four eigenvalues <= 2.5e-8 (weak; pointwise <= 6.6e-6),
then a fifth at 2.8e-6 (weak; pointwise 1.7e-3) -- by its coefficients the inviscid-Burgers projective field
xt d_x + t^2 d_t + (x - tu) d_u, exact for u_t + u u_x = 0 and broken only by H u_xx, which the smooth test functions
weigh little -- then 1.0e-2: the weak own count is 5. (iii) Under noise the weak form's floor (5.6e-7 to 4.7e-6 at 1e-2)
reaches that fifth direction, which then takes a place among the four lowest; the m = k reading loses d_x
(containment 0.08) and keeps Galilean and scaling (non-trivial 0.001). Pointwise FINS loses d_t, d_x and part of the
scaling (non-trivial 0.31) as on KdV. (iv) LieNLSD's F fits the local jets to 3% (held-out MSE 1.6e-3) and still
misses most of Galilean and scaling.
Predictions (seeds 42-51 unless stated; means over seeds):
(BO1) Clean: FINS weak and FINS pointwise <= 0.05 (finite algebra), on >= 9/10 seeds each.
(BO2) Two-stage, hard readout, both H: >= 1.5 missing (finite and non-trivial) at every sigma, on >= 9/10 seeds.
(BO3) Non-trivial generators: FINS weak <= 0.05 at every sigma; FINS weak below the two-stage hard readout (both H) at
every sigma and below LieNLSD as published (both systems; seeds 42-44, clean and 1e-2).
(BO4) FINS weak, finite algebra, at 1e-2 and 3e-2: <= 1.2 (at most one translation lost).
(BO5) FINS pointwise at 1e-2: >= 1.5 (finite).
(BO6) LieNLSD as published, clean: >= 0.5 (finite, both systems; seeds 42-44).
No predictions: the two-stage soft readout, own counts, the fifth eigenvalue (reported per seed: its ratio to the
fourth and the sixth, and its overlap with the projective field). A learned-operator arm (FNO steppers on BO data,
seeds 42-44) is registered separately after its own development run.

**Noise study in weak form -- run log (07.10).** (i) exp50 correction, made before any registered result was read:
the determining system was expanded in floats, and the unit-row normalisation turned rounding residues of exactly
cancelling coefficients (~1e-15 of the largest row) into arbitrary constraints; on Benjamin-Ono's development data
this changed the system from 266 to 285 rows between two runs on data equal to 5e-13. Stage 2 now works in exact
rationals (the floats' binary values), so structural zeros are zeros; on heat and KdV the systems and singular values
are unchanged (71 / 72 rows). The 18 registered two-stage runs made before the fix (clean, seeds 42-46) were deleted and
rerun. (ii) KdV's weak rows (60 runs, ~8 min each on one CPU thread) run on a Colab T4
(FINS_exp51_kdv_weak_colab.ipynb: exp51 --pdes kdv --parts weak, the same code); the local workers hold claims on
those jobs. The exact KdV solver then runs on the GPU, whose roundoff moves the evaluation data by less than the
summary's consistency tolerance, now 1e-4 of max |u| (was 1e-6); local runs of the same jobs (clean, seeds 42-46) are
kept only as a cross-device check. (iii) The 60 Colab runs arrived (fins_exp51_kdv_weak.zip, 28 s of weak blocks per run
on the T4 against ~480 s on one CPU thread) and replace the local ones (kept in noise_weak_runs/cpu_check): against
the local clean seeds 42-45 the evaluation data differ by 1.5e-7 of max |u| and the six lowest eigenvalues agree to
three digits wherever they exceed roundoff (k-th 2.2e-7 to 4.2e-7, first non-symmetry 6.9e-3 to 7.7e-3). (iv) A
machine restart (07.10, 03:14) killed the workers mid-run; no npz was truncated (390 checked); LieNLSD's optimiser
Adan had to be reinstalled (sail-sg/Adan), so every LieNLSD_F fit of this study uses Adan as published.

**Noise study in weak form (exp49, exp50, exp51) -- design and pre-registration (06.10; written after development
runs on seed 7, data 1007, before any run on seeds 42-51).**
Why: the mock reviews of 06.10 ask for a criterion that does not
differentiate noisy data, 1-10% noise, the weak-SINDy + symbolic-Lie pipeline as a baseline, the non-trivial
generators scored apart from the translations, and >= 10 seeds. exp48 showed pointwise FINS losing d_t from 1e-3 on:
every characteristic is built from finite differences of the snapshots.
Weak form (exp49_weak.py). Class tau = tau(t), xi = xi(x, t), phi = phi(x, t, u), the form every point symmetry of a
quasilinear second-order or any higher-order evolution equation takes (heat 34 of 60 basis fields, KdV 19 of 30; the
ground truth fits inside to 1e-14). Both sides of the transport criterion are tested against smooth functions:
psi_j(x) = (1 - z^2)^2 times 1, cos(pi m z), sin(pi m z), m <= M, across the scored region, and two overlapping bumps
(1 - z^2)^2 in the start time s (a start at every snapshot). Along the data J u_t(n) = d/ds S^k(u(s)), so both u_t
terms are s-derivatives and are integrated by parts onto the bumps: only S^k of the snapshots is needed. The xi u_x
term is <psi, J(xi D u)> = -<D(xi J^T psi), u> (D skew-adjoint): the noise of the spectral u_x reaches the criterion
only through a smooth adjoint field. Pencil, normaliser and readout as exp34; m = k.
Recipe, frozen here (no ground truth enters it): K = 16, a start at every snapshot, two temporal bumps, p = 2,
M = round(2 n95 R / D), n95 the smallest Fourier index of the D-point domain below which 95% of the noisy snapshots'
non-mean energy lies, R the scored points (seed 7: heat n95 = 2 on one period, M = 4; KdV n95 = 4, M = 5; the same at
every noise level). Own count (no ground truth): the largest ratio of consecutive eigenvalues among the lowest 17, the
eigenvalues (squared relative residuals) floored at 1e-6. torch's default float32 is kept, so the exact solver and the
evaluation data are bit for bit those of every other script (a float64 default had moved KdV's data by 2e-5).
How the recipe was chosen (seed 7, exact operators; fins_runs/weak_dev; finite missing dimensions): temporal bumps
1, 2, 3 (and 5, 8, 12) x p 2, 4 x M 2-4 (heat; and 6, 10): two bumps with p = 2 best at every level; heat best at
M = 2 (0.003 / 0.029 / 0.079 / 0.272 at 1e-2 / 3e-2 / 5e-2 / 1e-1), M = 4 1.5-3x above (0.009 / 0.071 / 0.167 /
0.431); KdV at 1e-2 (a start every 4th snapshot): M = 2 0.96, 3 0.26, 4 0.03, 5 0.09, 6 0.02, 8 0.04, 12 0.27; a start
at every snapshot against every 4th, KdV 3e-2, M = 6: 0.17 against 0.78. Factor 2 in the bandwidth rule is the
smallest that keeps KdV out of its M <= 3 failure; it costs heat its M = 2 optimum.
Two-stage baseline (exp50_two_stage.py). Stage 1, weak SINDy (Messenger and Bortz; PySINDy 2.1 WeakPDELibrary):
library u^a d_x^m u (a <= 2, m <= 3) and a constant, 200 subdomains per trajectory of half-width H of each extent
(test functions p = 4), on the ten evaluation trajectories FINS uses; sequential thresholding of the column-normalised
coefficients at 0.1 of the largest, least-squares refit. Stage 2: the determining system of the identified u_t = F in
FINS's normalised variables (sympy; every point generator of degree <= d, FINS's full class). Hard readout: its null
space below 1e-8 of the largest singular value (the identified equation's algebra, with its own count); soft: the k
smallest right singular vectors (the m = k reading FINS gets). Development grid (H 0.05, 0.1, 0.2; K 200, 500;
threshold 0.02, 0.05, 0.1, 0.2; heat 5e-2, 1e-1, KdV 1e-2, 5e-2; fins_runs/weak_dev/tsgrid): at H >= 0.1 the identified
equation has exactly the true terms (KdV 5e-2: u_t = -0.98 u_xxx - 6.01 u u_x) and both readouts miss 0.000, except
heat 1e-1 at H 0.1 with threshold 0.02 (hard 4.0); at PySINDy's default H = 0.05 spurious terms enter (hard algebra
dimension 1-2): hard misses 2-4 at heat 5e-2 and 1e-1 (threshold 0.2 at 5e-2: 0.03) and at KdV 5e-2; KdV 1e-2 is
still exact for thresholds >= 0.1; soft 0.03-0.90. Registered: H = 0.2 (chosen on this grid) and H = 0.05 (PySINDy's
default; a sensitivity row), K 200, threshold 0.1.
Design (exp51_noise_weak.py; npz in fins_runs/noise_weak_runs, scores in results/noise_weak/): heat (smooth, L = 1,
four-period windows, exact heat kernel) and KdV (periodised solitons, torus band [0.2, 0.8] L, exact solver);
Gaussian noise sigma = 0, 1e-3, 1e-2, 3e-2, 5e-2, 1e-1 times each trajectory's standard deviation (corrupt.py: exp48's
draws); seeds 42-51, evaluation data 1000 + s. Rows: FINS weak (and its own count); FINS pointwise (exp48's
exact-operator rows: u_t by five-point differences, windows from snapshot 2); two-stage H 0.2 and H 0.05, hard and
soft; LieNLSD with the exact right-hand side and the evaluation system (LieNLSD_RHS@eval = exp48's "LieNLSD central");
LieNLSD as published (F fitted to the corrupted training data; seeds 42-44 at 3e-2, 5e-2, 1e-1; clean, 1e-3, 1e-2 are
exp48's, same data and draws); learned arm: FINS weak through the KdV steppers retrained on the equally corrupted
training data (exp48's: clean = the protocol's, 1e-3, 1e-2), seeds 42-44. Scores: exp42's metric on the clean
evaluation trajectories (band [0.2, 0.8] L, times 8 .. Nt - 8), missing dimensions, m = k (own-count rows: their own
number) for the finite algebra (heat 6, KdV 4), heat's class (10), and new, the non-trivial generators: those of the
finite algebra other than d_x and d_t, each reduced to its part orthogonal to every other generator (heat u d_u,
Galilean, scaling, projective; KdV Galilean, scaling), which no autonomous shift-equivariant operator gives for free.
Development run of exp51 itself (seed 7; missing dimensions, finite algebra; heat / KdV):
                               clean       1e-3        1e-2        3e-2        5e-2        1e-1
FINS weak                      0.00 0.00   0.00 0.00   0.01 0.01   0.07 0.15   0.17 0.50   0.43 0.96
FINS weak, own count           0.00 0.00   0.00 0.00   0.01 0.01   0.07 0.15   0.17 0.08   5.62 0.49
FINS weak, non-trivial         0.00 0.00   0.00 0.00   0.01 0.00   0.05 0.00   0.11 0.00   0.28 0.01
FINS pointwise                 0.00 0.01   1.57 1.16   2.81 2.16   3.11 2.17   3.17 2.18   3.15 2.18
FINS pointwise, non-trivial    0.00 0.01   0.76 0.08   1.02 0.17   1.16 0.17   1.20 0.18   1.16 0.19
two-stage H 0.2, hard and soft 0.00 0.00   0.00 0.00   0.00 0.00   0.00 0.00   0.00 0.00   0.00 0.00
two-stage H 0.05, hard         0.00 0.00   0.00 0.00   0.00 0.00   0.01 3.00   4.00 3.00   4.00 2.00
two-stage H 0.05, soft         0.00 0.00   0.00 0.00   0.00 0.00   0.01 0.55   0.14 0.55   0.87 0.01
LieNLSD_RHS@eval               0.28 0.00   0.99 0.51   1.59 0.51   1.98 0.98   2.03 0.80   2.06 0.67
Own counts: heat 10 (its class) up to 5e-2, 4 at 1e-1; KdV 4 up to 3e-2, 5 at 5e-2 and 1e-1. KdV's first
non-symmetry stays at 6.7e-3 to 9.6e-3 while the k-th eigenvalue rises from 5e-7 (clean) to 3.4e-3 (1e-1). Heat's
class: FINS weak 0.00 / 0.00 / 0.01 / 0.07 / 0.17 / 0.43, two-stage H 0.2 0.00 throughout, H 0.05 hard 4.0 at 3e-2 and
8.0 above. Learned arm, development (KdV, stepper s42, data 1007): clean 0.001; 1e-2 data through the stepper
retrained on 1e-2 data 0.106, through the clean protocol stepper 0.010 (exp48's pointwise learned row at 1e-2: 2.20).
Reading of the development run: the weak form removes FINS's noise fragility (1e-2: 0.01 against 2.81 / 2.16); on
KdV what noise costs it is the translations (non-trivial 0.00-0.01 up to 1e-1); with a tuned subdomain size the
two-stage pipeline identifies both equations exactly at every level and is at or below FINS weak everywhere.
Predictions (seeds 42-51 unless stated; means over seeds; finite algebra unless stated):
(WN1) FINS weak <= 0.05 at sigma = 0 and 1e-3 on both equations, on >= 9/10 seeds each.
(WN2) At 1e-3 and 1e-2 FINS weak misses at most a tenth of what FINS pointwise misses, both equations.
(WN3) At 3e-2, 5e-2, 1e-1 FINS weak misses fewer than FINS pointwise, both equations.
(WN4) FINS weak: heat <= 0.05 at 1e-2 and <= 0.3 at 5e-2; KdV <= 0.1 at 1e-2 and <= 1.0 at 5e-2.
(WN5) KdV: FINS weak's non-trivial missing <= 0.05 at every sigma <= 5e-2 (its losses are the translations).
(WN6) Two-stage H 0.2, hard and soft: <= 0.05 at every sigma <= 1e-1 on both equations, on >= 9/10 seeds each.
(WN7) Two-stage H 0.05, hard: >= 1 missing dimension at 5e-2 and 1e-1 on both equations, on >= 5/10 seeds each.
(WN8) LieNLSD_RHS@eval misses more than FINS weak at every sigma from 1e-3 to 5e-2, both equations.
(WN9) FINS weak's own count equals k (heat 10, KdV 4) at every sigma <= 1e-2 on >= 8/10 seeds per equation.
(WN10) Learned arm (KdV, seeds 42-44): FINS weak through the retrained steppers <= 0.05 clean and <= 0.3 at 1e-3 and at
1e-2, and at 1e-3 and 1e-2 at most a fifth of exp48's pointwise learned row (1.18, 2.20).
No predictions: heat's class (reported), the own count above 1e-2, LieNLSD as published at the new levels, the
non-trivial scores of the other rows (reported).
Paper, fixed now: a subsection on the weak form (the identity, the class restriction, the recipe and how it was
chosen, its cost) and the noise table with every row above whatever the verdicts; the two-stage pipeline introduced
as the method of choice when the equation lies in a sparse polynomial library, FINS's case resting on equations and
operators outside one (Benjamin-Ono, next; learned operators).

**Operator training time for the paper's Cost paragraph (06.10; from the learned arm's runs).**
Heat CNO, protocol settings, Colab NVIDIA T4 (the runtime for FINS_exp48_heat_colab.ipynb): one run, data
generation, 400 epochs and discovery together, 53-55 min at every noise level (intervals between consecutive saved npz
files; the runs were sequential), 50 min at s = 2, 47-48 at s = 4, 24-25 at s = 8. KdV FNO stepper, protocol settings,
one CPU thread: 2808-3633 s per run, training and discovery (the "ok N s" lines of fins_runs/worker_kdv_*.log),
against 345-361 s for discovery alone through the protocol's steppers; excluded: clean-rt s42 (5783 s, its core shared
with the clean-condition chains) and the two n1e-3 runs resumed after a restart (544 s, 969 s). The protocol's own
trainings printed no time. Paper: Sec. 5.9 (Cost), the last \todo.

**Noise and temporal thinning, learned arm -- heat's retrained conditions and the complete verdicts (Colab GPU,
05-06.10; runs copied from MyDrive/fins_runs/results/noise_thin_learned_runs to fins_runs; results/noise_thin/
{condition}/learned/tables_heat.md, scores_heat.json, logs; summary_learned.md and .json).**
Runs: FINS_exp48_heat_colab.ipynb with the registered commands (exp40 --tag _smooth_L1_{c} --train-if-missing
--score-margin 96 --train-margin 6 s --unroll min(4, 16 / s) --K 16 / s --every 16 / s --interior), 24 CNO trainings
from scratch (no resume), about 50 min per run at s = 1 to 4 and 24 min at s = 8 (intervals between saved files;
training and discovery). Every log first holds one attempt that stopped at import (neuralop not yet installed; nothing
trained or written), then the complete run. The npz files carry the registered seeds, data seeds, noise, thinning and
shapes.
Missing dimensions, m = k, mean over seeds 42-44; heat finite (6):
                                 clean   clean-rt  1e-4    3e-4    1e-3    1e-2    s = 2   s = 4   s = 8
FINS, learned operator           0.14    0.11      0.15    0.66    1.68    2.96    0.09    0.19    0.11
FINS, exact operator (exp48)     0.00    --        0.32    0.63    1.62    2.95    0.00    0.00    0.00
LieNLSD as published (F)         0.55    --        0.58    0.92    1.23    1.60    0.40    0.67    0.80
Per seed, learned: clean-rt 0.18 / 0.09 / 0.05, 1e-4 0.20 / 0.10 / 0.14, 3e-4 0.83 / 0.50 / 0.65, 1e-3 1.36 / 1.80 /
1.87, 1e-2 2.99 / 2.91 / 2.97, s = 2 0.21 / 0.03 / 0.03, s = 4 0.46 / 0.05 / 0.06, s = 8 0.19 / 0.08 / 0.05; exact at
1e-4 0.91 / 0.02 / 0.02, at 3e-4 1.05 / 0.37 / 0.48. Heat's class (10), same columns: learned 1.58 / 1.50 / 1.33 /
1.38 / 2.32 / 3.25 / 1.10 / 1.11 / 1.34, LieNLSD as published 0.72 / -- / 0.71 / 1.14 / 1.62 / 2.74 / 0.56 / 0.83 /
1.02 (fewer on the mean in every condition).
Criterion (every generator below lambda11): 3/3 at clean-rt, 1e-4 and every thinning, 1/3 at 3e-4, 0/3 at 1e-3 and
1e-2. Worst margins: clean-rt 1.3-4.0x (the protocol's CNOs 1.4-2.9x), s = 2 / 4 / 8 1.6-5.1x / 1.6-4.5x / 1.7-3.7x,
1e-4 1.5-2.7x, 3e-4 0.42-1.2x, 1e-3 0.14-0.25x, 1e-2 0.08-0.12x. d_t plain containment >= 0.998 clean, clean-rt and
thinned; 0.98-0.99 / 0.55-0.75 / 0.21-0.44 / 0.04-0.13 at 1e-4 / 3e-4 / 1e-3 / 1e-2. Largest-gap rank 1-12, never
the algebra's 10 (as in the protocol). The CNOs' held-out increment error (best state, training log): clean-rt 0.031
/ 0.015 / 0.024 (the protocol's 0.029 / 0.015 / 0.022), 1e-4 0.033 / 0.019 / 0.027, 3e-4 0.047 / 0.037 / 0.044,
1e-3 0.118 / 0.106 / 0.117, 1e-2 0.462 / 0.441 / 0.467 (against noisy snapshots), s = 2 / 4 / 8 0.013-0.017 /
0.009-0.012 / 0.007-0.009 (increments over s original steps).
VERDICTS (literal; complete).
(C1) MET on both: KdV 0.01, heat 0.14.
(LC) MET on both: KdV 0.011 / 0.015 / 0.013; heat 0.11, within 0.15 of 0.14.
(LT1) MET (KdV; entry below).
(LT2) MET: heat 0.09 / 0.19 / 0.11 at s = 2 / 4 / 8, within 0.15 of clean-rt (0.11), and below LieNLSD as published
at every s (0.40 / 0.67 / 0.80), on every seed as well.
(LN1) NOT MET: at 1e-4 the learned operator misses 0.15, below exact - 0.05 = 0.27 (exact 0.32); at 3e-4 (0.66 >=
0.58) and 1e-3 (1.68 >= 1.57) it holds. The miss is seed 42's: there the exact operator loses d_t and the scaling at
1e-4 (plain containment 0.34 and 0.57), the learned one keeps them (0.99 and 0.99); on s43 and s44 the learned
operator misses more than the exact one (0.10 and 0.14 against 0.02), as the prediction assumed. The same pattern at
3e-4: s42 0.83 against 1.05, s43 and s44 0.50 and 0.65 against 0.37 and 0.48.
(LN2) NOT MET (KdV, 3e-4; entry below).
(LN3) MET: KdV parts as below; heat at 1e-2 2.96 against 1.60, more on every seed.
(LN4) MET: heat d_t containment 0.26 / 0.44 / 0.21 at 1e-3 and 0.09 / 0.04 / 0.13 at 1e-2; KdV as below.
Unregistered, heat against LieNLSD as published: fewer at 1e-4 on every seed (0.15 against 0.58); fewer at 3e-4 on
the mean (0.66 against 0.92), on s42 and s43, level on s44 (0.651 against 0.646); more at 1e-3 on every seed (1.68
against 1.23).
Reading. Thinning keeps the learned heat operator within the registered band of its clean value (0.09-0.19 against
0.11 on the means; s42 at s = 4 0.46 against 0.18; LieNLSD as published 0.40-0.80), as on KdV. Under noise the learned
heat operator follows the exact one on the means from 3e-4 on (gaps 0.03, 0.06, 0.01; per seed up to 0.22) and at
1e-4 does better on one seed of three, where the exact operator already loses d_t; KdV's extra 0.18 at 3e-4 has no
counterpart on heat. Against LieNLSD as published the crossover lies between 3e-4 and 1e-3 on both equations, as with
perfect models; heat's full class stays LieNLSD's on the means (on s43 FINS misses fewer in 5 of 8 conditions). Paper: Tables 5 and 11 (heat rows; heat's clean-rt control in the
caption of Table 11), Table 14 (LC, LT2, LN1, LN3, LN4), Sec. 5.7 "Learned models", abstract, introduction,
discussion.

**Noise and temporal thinning, learned arm -- results for KdV and heat's clean condition (CPU, 05.10;
results/noise_thin/summary_learned.md and .json, {condition}/learned/ tables_{pde}.md, scores_{pde}.json, logs;
checkpoint checks in results/noise_thin/checkpoint_check/; the high-mode gain in results/noise_thin/gain/). Heat's
retrained conditions run on Colab (FINS_exp48_heat_colab.ipynb) and are pending; exp48_learned.py summary scores each
equation into its own file and reads back the stored scores of an equation whose npz files are absent, so heat's runs
merge without KdV's npz (checked: KdV's rows read back equal the recomputed ones).**
Missing dimensions, m = k, mean over seeds 42-44 (per seed in summary_learned.json); KdV (4):
                                 clean   clean-rt  1e-4    3e-4    1e-3    1e-2    s = 2   s = 4   s = 8
FINS, learned operator           0.01    0.01      0.07    0.49    1.18    2.20    0.01    0.01    0.01
FINS, exact operator (exp48)     0.01    --        0.05    0.31    1.15    2.16    0.01    0.01    0.01
LieNLSD as published (F)         0.91    --        0.99    0.99    1.13    1.43    1.24    1.21    1.42
clean: the protocol's steppers; clean-rt and every corrupted condition: steppers retrained with the protocol's
settings (two one-thread workers, about an hour per training; the two restarts after the partial states were
introduced, at about 10:51 and 12:11, cost at most ten epochs each). Heat, clean, the protocol's CNOs: 0.14 +/- 0.06 (0.21, 0.15, 0.06), class 1.58 +/- 0.64, worst margins
1.4x / 2.9x / 1.9x -- the protocol's 0.14 and 1.59. The heat operator error of Table 1 (held-out increment error in
the protocol's Colab training log): 0.029, 0.015, 0.022.
Criterion through the KdV steppers: every generator below lambda5 on 3/3 in every condition up to 1e-3, 0/3 at 1e-2;
worst margins clean 351-884x (protocol steppers), clean-rt 375-839x, s = 2 / 4 / 8 451-805x / 495-708x / 358-630x,
1e-4 61-72x, 3e-4 8.2-9.9x, 1e-3 1.2-1.4x, 1e-2 0.2x. Largest-gap rank 4 on 3/3 clean and thinned, mostly 3 under
noise. d_t plain containment 1.00 clean and thinned, 0.96-0.98 / 0.72-0.81 / 0.35-0.38 / 0.01-0.03 at 1e-4 / 3e-4 /
1e-3 / 1e-2. Steppers' held-out increment error: clean-rt 0.006-0.011 (the protocol's steppers 0.006-0.011), thinned
0.004-0.007, at 1e-4 / 3e-4 / 1e-3 / 1e-2 0.009-0.013 / 0.021-0.022 / 0.065-0.066 / 0.35-0.36 (against noisy
snapshots: floored by the noise).
VERDICTS (literal; heat's parts wait for Colab).
(C1) MET on both: KdV 0.01 (<= 0.05), heat 0.14 (within 0.10 of 0.14), through the protocol's checkpoints.
(LC) KdV MET: clean-rt 0.011 / 0.015 / 0.013, each <= 0.05. Heat pending.
(LT1) MET: KdV 0.01 at s = 2, 4, 8 (per seed 0.01-0.02, as clean), below LieNLSD as published on every seed (0.97-1.62).
(LN2) NOT MET: within 0.15 of exact at 1e-4 (0.07 vs 0.05) and 1e-3 (1.18 vs 1.15), not at 3e-4 (0.49 vs 0.31, +0.18;
per seed 0.58 / 0.40 / 0.49 against 0.39 / 0.23 / 0.31).
(LN3) KdV part met: fewer than LieNLSD as published at 1e-4 (0.10 / 0.05 / 0.06 vs 0.98 / 1.00 / 1.00) and 3e-4
(0.58 / 0.40 / 0.49 vs 0.96 / 0.99 / 1.02) on every seed, more at 1e-2 (2.20 vs 1.43). Heat part (1e-2) pending.
(LN4) KdV part met: d_t containment 0.35-0.38 at 1e-3, 0.01-0.03 at 1e-2, 3/3 each. Heat part pending.
(LT2, LN1) heat only: pending.
Exploratory, after the results (not registered; exp48_gain.py): gain over K = 16 steps of white noise restricted to
|k| > 32, at the evaluation states FINS uses. The exact solver removes the part above its 2/3 dealiasing band (gain
0.55-0.56); the learned steppers pass it almost unchanged on clean data (0.92-0.96), at 1e-4 (0.91-0.96), 3e-4
(0.91-0.95) and 1e-3 (0.85-0.91), and remove it only at 1e-2 (0.00), where the band below is damped as well (gain on
0 < |k| <= 32: 0.79-0.87 clean, 0.75-0.78 at 3e-4, 0.57-0.58 at 1e-3, 0.35-0.37 at 1e-2). Noise that the solver
discards within the horizon is carried by the steppers at intermediate levels; a candidate for the extra 0.18 at 3e-4,
not a tested cause (at 1e-3 the steppers carry it too and cost only 0.03 more).
Reading. Through learned operators, as through the exact one, FINS keeps KdV's algebra at its clean floor under
thinning up to every eighth snapshot (0.01; LieNLSD as published 1.21-1.42), misses fewer dimensions than LieNLSD as
published up to 3e-4 (0.07 and 0.49 against 0.99), slightly more at 1e-3 (1.18 against 1.13; more on two seeds of
three, level on s43) and more at 1e-2 (2.20 against 1.43). The learned operator costs little against the exact one at 1e-4 and 1e-3 and 0.18 at 3e-4.

**exp43 for heat through the protocol's CNOs -- results (CPU, 05.10; results/exp43_recheck/heat_s{42,43,44}.log).**
exp43 --pde heat --seed s --tag _smooth_L1 (checksums 4786.927 / 4737.260 / 4735.527); exact operator, data and the
exact rows are unchanged (floor 2.3e-4 / 2.1e-4 / 2.8e-4, sigma_11 0.0346 / 0.0311 / 0.0201).
                 eta     eta_V    sigma~_11  sigma~_10/11  Weyl max  (iii) shares / bounds         eta_V / sigma_11
  heat s42       0.0703  0.0674   0.0640     0.92          0.059     0.043-0.599 / 0.271-0.848     1.95
  heat s43       0.1024  0.0306   0.0375     0.81          0.030     0.010-0.082 / 0.241-0.585     0.98
  heat s44       0.3011  0.0334   0.0323     0.94          0.030     0.024-0.573 / 0.164-0.722     1.66
  (first run, other CNOs: eta 0.070 / 0.040 / 0.041, eta_V 0.063 / 0.037 / 0.037, eta_V / sigma_11 1.82 / 1.17 / 1.82)
The learned losses of the algebra equal those of the protocol's FINS rows (s43: max 4.8e-4, lambda_11 1.4e-3), as they
must. T1 (Weyl, (iii)) and T2 (heat's sufficient condition fails, eta >= 0.02) stay met on 3/3. What changes: along
the algebra the error is 1.0-1.9 times sigma_11 (was 1.2-1.8); over the class it is 0.070-0.30 (was 0.040-0.070), so
on s43 and s44 the worst case is dominated by directions away from the algebra, as on KdV, and the sentence "on heat
the error lies mostly along the algebra" does not hold; superpositions outside the ten lowest up to 0.60 / 0.08 /
0.57 (was up to 0.33-0.61). The reading of Sec. 3.4 stands -- the ratio along the algebra separates KdV (0.04-0.10)
from heat's class (1.0-1.9) -- with the numbers of the protocol's operators. Paper: Sec. 3.4, discussion, Sec. 5.8,
Table 12 (T1-T3, Appendix G), Appendix D (a paragraph on the first run's checkpoint).

**Checkpoint provenance (05.10): heat's protocol CNOs verified; exp43's heat rows used other CNOs; where KdV's
protocol steppers live.**
Heat. Google Drive gave S_theta_heat_s{42,43,44}_g64x128_cno_smooth_L1.pt. Re-running the protocol's heat command
through them (exp40 --op cno --smooth --only-L 1 --n 10 --tag _smooth_L1 --data-seed 1000 + s, no --interior) gives
the protocol's FINS rows ("Protocol run, part 1") to every printed digit: finite six 3.0-7.1e-4 / 0.8-3.5e-4 / 0.3-3.1e-4, b 2.8-2.9e-3 /
3.3-4.8e-4 / 4.3-5.4e-4, lambda11 4.1e-3 / 1.4e-3 / 1.0e-3, worst margins 1.4x / 2.9x / 1.9x, containment below 0.9
d_x 0.90 and b=1 0.85 (s42), b=1 0.86 (s44). They are the protocol's operators (parameter checksums 4786.927 /
4737.260 / 4735.527, the sum of absolute parameters as exp43 prints it).
exp43. The Colab notebook shows that the heat rows of exp43 (Proposition 2, paper Sec. 3.4: eta 0.040-0.070,
eta_V 0.037-0.063, the (iii) shares and bounds) loaded S_theta_heat_s{s}_g64x128_cno.pt -- exp43's default name, no
--tag was passed -- with checksums 4831.577 / 4754.430 / 4756.563: CNOs other than the protocol's. The KdV rows of
exp43 loaded exp39's default, the protocol's steppers. exp43 for heat is rerun with --tag _smooth_L1
(results/exp43_recheck/); until then the paper's heat numbers in Sec. 3.4 refer to other operators than its tables.
KdV. The protocol cell (exp39 --train-ic mixed --train-if-missing) printed no training line: it loaded
FINS_RESULTS_DIR/S_theta_kdv_s{s}_mixed_stepper.pt (= MyDrive/fins_runs/results/ on Google Drive), and the
fresh-seed run of 02.10 reproduced the protocol eigenvalues through the same files (1.6e-5, 2.9e-6, 8.0e-6). The
first files sent were exp31's torus steppers (..._stepper_m64a.pt, grid positional embedding); at 02:24 followed
S_theta_kdv_s{42,43,44}_mixed_stepper.pt, which reproduce the protocol's KdV rows (protocol command, results in
fins_runs/verify_protocol_kdv): scaling 1.9e-5 / 3.9e-5 / 1.9e-5, lambda5 1.7e-2 / 1.4e-2 / 1.3e-2, margins
~890x / 360x / 680x, d_t containment 0.99 / 0.80 / 0.97 (checksums 31682.043 / 32363.521 / 31602.550). C1 is
decidable on both equations.
Runs: two machine restarts (00:18 and 01:33) killed the KdV trainings at epochs
~1 and ~240-280; stepper training now writes a partial state every 10 epochs and resumes from it (checked: a resumed
run equals an uninterrupted one bit for bit, FNO stepper and CNO), and a claim records the boot id and pid, so a
worker after a reboot takes over the dead claims.

**Noise and temporal thinning, learned arm (exp48_learned.py) -- design and pre-registration (05.10; the KdV workers
started at 00:31 MSK, this entry was written before any learned-arm result; no development seed: two-epoch smoke runs
on seed 7 checked the mechanics only).**
Why: exp48 compared the criteria with perfect models and ran LieNLSD as published (network F fitted to the corrupted
training data) on every condition; FINS's row with learned operators is missing, so the comparison of the two methods,
each with the model it learns, is incomplete.
Design (exp48_learned.py; the commands of the Colab cells "exp48"): per condition, seeds 42-44, evaluation data
1000 + s, exp48's windows (snapshots 2 .. Nt - 3) and u_t (five-point differences; for KdV in place of the solver
stencil). clean: the protocol's operators, loaded. clean-rt: operators retrained on the clean training data with the
protocol's settings, the control that separates the corruption from retraining. n1e-4, n3e-4, n1e-3, n1e-2, t2, t4,
t8: operators retrained on the equally corrupted training data (400 epochs). Heat CNO: training crop 6 s points per
step, unroll min(4, 16 / s) kept steps (at s = 8 an unroll of 4 would crop 192 points per side of the 256-point
window and leave no target; 2 kept steps there cover the same 16 original steps as 4 at s = 4), scored region 96
points from each window edge (the protocol's). KdV stepper: the protocol's settings, unroll 4. KdV runs on the CPU
(two one-thread workers, about an hour per training); heat on Colab (its CNO trains about 10 h per run on two CPU
cores). Order: clean-rt, n1e-4, t4, t8, t2, n3e-4, n1e-3, n1e-2. Generators scored on the clean evaluation
trajectories by exp42 (--t-margin 8 // s); the perfect-model and LieNLSD rows are exp48's (summary.json).
Checkpoints. Google Drive gave heat's protocol CNOs (S_theta_heat_s{42,43,44}_g64x128_cno_smooth_L1.pt); they are
checked by re-running the protocol's heat rows before use. KdV's protocol steppers (S_theta_kdv_s{s}_mixed_stepper.pt)
were not among the files (the ..._stepper_m64a.pt among them carry the grid positional embedding: exp31's torus
steppers), so KdV's half of C1 cannot be decided as registered; clean-rt is reported beside it, not as C1.
Predictions (means over seeds unless stated; m = k; finite algebra; "LieNLSD as published" = exp48's LieNLSD_F row,
heat 0.55 / 0.58 / 0.92 / 1.23 / 1.60 at clean / 1e-4 / 3e-4 / 1e-3 / 1e-2 and 0.40 / 0.67 / 0.80 at s = 2 / 4 / 8,
KdV 0.91 / 0.99 / 0.99 / 1.13 / 1.43 and 1.24 / 1.21 / 1.42; "exact" = FINS through the exact operator in exp48):
(LC) clean-rt: KdV <= 0.05 on 3/3 seeds; heat within 0.15 of the protocol's learned row (0.14).
(LT1) Thinning, KdV: <= 0.10 at s = 2, 4, 8, and fewer than LieNLSD as published on 3/3 seeds at every s.
(LT2) Thinning, heat: within 0.15 of clean-rt at s = 2, 4, 8, and fewer than LieNLSD as published at every s.
(LN1) Noise, heat: at least exact - 0.05 at 1e-4, 3e-4, 1e-3 (exact 0.32, 0.63, 1.62): the heat kernel already damps
the modes that carry only noise, so a learned operator adds its own error and removes none of the noise.
(LN2) Noise, KdV: within 0.15 of exact at 1e-4, 3e-4, 1e-3 (0.05, 0.31, 1.15): the operator's error lies far below
the first non-symmetry (margins 360-890x in the protocol), so the noise of the characteristics sets the outcome. A
stepper trained on noisy pairs may learn to damp the modes that carry only noise, which the exact solver transports
with unit modulus; that would lower the noise term of the residual, and the band allows for it.
(LN3) Against LieNLSD as published: KdV fewer at 1e-4 and 3e-4 on 3/3 seeds; both equations more at 1e-2.
(LN4) Mechanism: at 1e-3 and 1e-2 FINS's d_t has plain containment < 0.9 on >= 2/3 seeds per equation.
No predictions: heat at 1e-4 to 1e-3 against LieNLSD as published; KdV at 1e-3 against it; 1e-2 against exact (heat's
exact row sits at the Gaussian null there, 2.95 against 2.91); the operators' errors.

**Noise and temporal thinning (exp48) -- results (CPU; results/noise_thin/: summary.md, summary.json,
{condition}/tables.md, scores.json, logs/; the npz files, 1.6 GB, stay in fins_runs/noise_thin; two one-thread
workers, 2.7 h wall, 5.3 worker-hours).**
Missing dimensions, m = k, mean over seeds 42-44 (per seed and sd in summary.json / tables.md); heat finite (6) /
KdV (4):
                                 clean       1e-4        3e-4        1e-3        1e-2        s = 2       s = 4       s = 8
FINS, exact operator             0.00 0.01   0.32 0.05   0.63 0.31   1.62 1.15   2.95 2.16   0.00 0.01   0.00 0.01   0.00 0.01
LieNLSD RHS@eval, central        0.22 0.00   0.65 0.98   0.93 0.98   1.10 0.50   1.40 0.51   0.54 0.00   0.70 0.00   0.90 0.00
LieNLSD RHS@eval, spectral       0.00 0.00   0.81 0.65   1.07 0.45   1.08 0.48   1.41 0.99   0.00 0.00   0.01 0.00   0.20 0.00
LieNLSD RHS, training system     0.43 0.66   0.59 0.99   0.84 0.99   1.27 0.83   1.41 0.87   0.73 0.63   0.75 0.63   0.93 0.72
LieNLSD F (as published, 10NxNt) 0.55 0.91   0.58 0.99   0.92 0.99   1.23 1.13   1.60 1.43   0.40 1.24   0.67 1.21   0.80 1.42
LieNLSD F@eval                   0.43 0.85   0.67 1.01   0.81 0.68   1.10 1.28   1.39 1.28   0.21 1.26   0.40 1.13   0.79 1.43
Heat's class (10): FINS as on the finite algebra (0.00 ... 2.96); LieNLSD central 0.24 / 0.73 / 1.15 / 1.40 / 1.86 /
0.57 / 0.81 / 1.12, spectral 0.00 / 0.95 / 1.42 / 1.43 / 1.92 / 0.00 / 0.01 / 0.23, F 0.72 / 0.71 / 1.14 / 1.62 / 2.74
/ 0.56 / 0.83 / 1.02. "protocol" (FINS as in the protocol's exact rows): heat 0.00 / 0.00, KdV 0.01. Class nulls
(Gaussian coefficients) 2.91, 6.36-6.38, 2.85 in every condition.
(R1) NOT MET (heat, fitted F). "protocol" gives the exact rows (0.00, 0.01); LieNLSD_RHS gives 0.43 / 0.56 / 0.66 and
LieNLSD_F gives KdV 0.91 (per seed 0.97, 0.74, 1.00, the same checkpoints and held-out MSE as the first run), but heat
0.55 / 0.72 against 0.30 / 0.42. Cause, checked on seed 42: with two torch threads (the first run) the fit reproduces
the first run exactly (best epoch 30, held-out relative MSE 3.192e-4, Jacobian error 0.192); with one thread (both
exp48 workers) it ends at epoch 80, 3.040e-4, 0.167, and seed 42 misses 0.62 instead of the first run's 0.45. Seeds
43 and 44 also train differently (best epochs 100 / 100 against 70 / 80) and were not rerun with two threads. KdV's
fit is unaffected. exp48's rows are internally consistent (one thread throughout).
(R2) MET: FINS in exp48's configuration on clean data 0.00 and 0.01 on 3/3.
(T1) MET: FINS 0.00 (heat) and 0.01 (KdV) at s = 2, 4, 8 on every seed.
(T2) MET: LieNLSD central 0.54 / 0.70 / 0.90 against FINS 0.00 (0.22 with every snapshot).
(N1) NOT MET (heat): at 1e-4 FINS misses 0.32 on heat (per seed 0.91, 0.02, 0.02: seed 42 loses d_t and the scaling)
and 0.05 on KdV; fewer than LieNLSD central (0.65, 0.98) and than F@eval (0.67, 1.01) on both.
(N2) MET: 1e-3: FINS 1.62 / 1.15 against LieNLSD central 1.10 / 0.50; 1e-2: 2.95 / 2.16 against 1.40 / 0.51.
(N3) MET: LieNLSD's translations have plain containment >= 0.99 (minimum 1.0000) in every variant, condition and seed;
FINS's d_t 0.19-0.43 (heat) and 0.27-0.31 (KdV) at 1e-3, 0.02-0.12 and 0.03 at 1e-2, 3/3 each. At 1e-2 FINS also
loses d_x (0.05-0.32, 0.09-0.11) and on heat the scaling (0.03-0.07).
(N4) MET: KdV, clean: LieNLSD_RHS@eval 0.00 on 3/3 with central and with spectral jets; LieNLSD_F@eval 0.85.
Mechanism numbers. FINS: at 1e-4 the five-point difference puts noise of 1.4-1.8 % of u_t into heat's characteristics
(slow dynamics: per trajectory rms u_t / std u 0.4-3.6, mean 1.1-1.4, at dt = 7.9e-3) and 0.7-0.8 % into KdV's (mean
30-32 at dt = 3.9e-4); d_t's loss grows
~100-fold from 1e-4 to 1e-3 (heat 1.1e-3, 4.0e-4, 2.8e-4 -> 8.9e-2, 3.6e-2, 2.6e-2; KdV 2.6e-4, 1.7e-4, 1.9e-4 ->
2.4e-2, 1.6e-2, 1.9e-2), past lambda_11 on heat (1.4e-2, 1.0e-2, 5.4e-3 at 1e-3) and within a factor of two of
lambda_5 on KdV (3.4e-2, 3.0e-2, 3.1e-2). LieNLSD: the equation's residual on the evaluation jets (central
differences) is 1.3-2.4 % (heat) and 1.4 % (KdV) of u_t on clean data and 2.2-3.0 % and 4.3-4.6 % at 1e-4 (u_xxx).
Readings. (i) Thinning costs FINS nothing on either equation: the finite-difference u_t of a solution of a linear
equation is itself a solution and is transported with the characteristic (KdV's solitons are resolved in time, so
neither method loses there with the exact RHS; the fitted F, trained on the thinned broadband half, loses 0.3-0.5).
(ii) Under noise FINS misses fewer than LieNLSD with the exact RHS up to 3e-4 and more from 1e-3, losing d_t first
(then d_x and, on heat, the scaling); LieNLSD's translations, heat's superpositions and KdV's Galilean generator
(exact RHS) are structural zeros of its system, while its scalings go at 1e-4: with central differences it misses
0.98 on KdV at 1e-4 and 3e-4 and 0.50 from 1e-3, 1.10-1.40 on heat from 1e-3; its fitted F reaches 1.43 and 1.60 at
1e-2. (iii) Spectral jets help LieNLSD under thinning; under noise they help on KdV at 1e-4 and 3e-4 (0.65, 0.45
against 0.98, 0.98), not at 1e-2 (0.99 against 0.51), and not on heat (0.81-1.41 against 0.65-1.40). (iv) With the exact right-hand side
LieNLSD's KdV gap in the main table (0.66) is the training data's: on the evaluation trajectories it is 0.00; its
fitted F misses 0.85-0.91 on either. (v) Against LieNLSD as published (F) FINS through the exact operator is ahead up
to 3e-4 and behind from 1e-3 on the means (KdV by 0.02 at 1e-3, not on seed 43; against F@eval FINS is ahead at
1e-3 on KdV, 1.15 against 1.28); the learned-operator arm is the Colab run (C1 registered).
Paper: Sec. 5.7 "Noisy and sparsely sampled data" (Table 5), Table 11 (full), Table 14 (verdicts); LieNLSD paragraph
of Sec. 5.2 (KdV gap from the training data); abstract, introduction, related work, discussion (limitation: noisy
data need smoothed derivatives; KdV's solver stencil not needed through the exact operator).

**Noise and temporal thinning (exp48) -- pre-registration (04.10; written after a development run on seed 7, data
1007, before any run on seeds 42-44).**
Why: the paper says FINS's characteristics need only first derivatives of the data and LieNLSD's determining system
derivatives up to the equation's order (heat u_xx, KdV u_xxx, with u_tx, u_txx), but no experiment shows what that
buys when the data are noisy or sparse in time. Design (exp48_noise_thinning.py): a perfect model and corrupted data,
so that the criteria are compared, not learned models.
Data: the protocol's heat (smooth, L = 1, 64 x 128 snapshots over T = 1) and KdV (periodised solitons, 128 x 128 over
T = 0.05) evaluation trajectories (10, data seed 1000 + s) and training trajectories (1024, seed s; KdV half broadband
humps). Corruption (corrupt.py), the same draws for every method and for both sets: Gaussian noise, sigma = 1e-4,
3e-4, 1e-3, 1e-2 times each trajectory's standard deviation, i.i.d. over the grid; or every s-th snapshot, s = 2, 4, 8.
FINS through the exact operator: the heat-kernel window operator composed s times per kept step, the KdV solver
integrating on the original grid; K = 16 / s kept steps (the same physical horizon), starts every 16 / s.
Characteristics from the corrupted snapshots: u_x as in the protocol (heat fourth-order differences, KdV spectral), u_t
by the five-point fourth-order difference -- for KdV in place of the protocol's aligned solver stencil, which
corrupted snapshots do not provide (exp39 --ut fd); windows on snapshots 2 .. Nt - 3 (exp39 / exp40 --interior), so
every u_t is a central difference, as LieNLSD's jets are taken on interior times. Condition "protocol": FINS exactly
as in the protocol's exact rows, clean data.
LieNLSD (exp41): exact right-hand side with the determining system on the training trajectories (LieNLSD_RHS, as
published) or on the evaluation trajectories FINS uses (suffix @eval), jets by central differences (as published) or
spectral in x / fourth order in t (_spec@eval); F fitted on the corrupted training data (10 Nx Nt points of the
original grid, official epochs), system on the training (LieNLSD_F) or the evaluation trajectories (LieNLSD_F@eval,
the same fit).
Scores: exp42 on the clean evaluation trajectories (every method's generators are polynomials in x, t, u), missing
dimensions, m = k, finite algebra (heat 6, KdV 4) and heat's class (10); scored times 8 / s .. Nt - 8 / s (the
protocol's physical region); new in exp42's JSON: the plain containment of each known generator in the found span.
Development run (seed 7; missing dimensions, m = k, finite algebra; heat / KdV):
                               clean       1e-4        1e-3        1e-2        s = 2       s = 4       s = 8
FINS, exact operator           0.00 0.01   0.01 0.03   1.57 1.16   2.81 2.16   0.00 0.01   0.00 0.01   0.00 0.01
LieNLSD RHS@eval, central      0.28 0.00   0.97 0.95   0.99 0.51   1.59 0.51   0.52 0.00   0.51 0.00   0.97 0.00
LieNLSD RHS@eval, spectral     0.00 0.00   0.64 0.63   1.05 0.51   1.54 0.99   0.00 0.00   0.00 0.00   0.19 0.00
LieNLSD RHS, training system   0.97 0.99   0.96 1.00   1.06 1.00   1.51 0.97   0.80 0.99   0.95 0.60   0.95 0.99
LieNLSD F                      0.84 0.98   0.85 0.97   1.09 1.36   -           -           -           0.95 1.28
LieNLSD F@eval                 0.44 0.93   0.69 1.26   1.00 1.19   -           -           -           0.73 1.17
("protocol": 0.00, 0.01.) Readings: (i) thinning to every 8th snapshot does not move FINS: for a linear equation the
finite-difference u_t of a solution is itself a solution, so its error is transported with the characteristic, and
travelling solitons nearly so. (ii) FINS holds at 1e-4 and loses its translations and scaling from 1e-3 on (plain
containment of d_t 0.44 / 0.32 at 1e-3, 0.20 / 0.04 at 1e-2): every FINS characteristic is built from the data's
derivatives, d_t's is -u_t itself, a five-point difference that multiplies the noise by ~0.95 / dt. (iii) LieNLSD's
translations are inside its span at every level (containment 1.00), as are heat's superpositions and, with the exact
right-hand side, KdV's Galilean generator: their columns of the determining system vanish identically, or cancel within
the same jets; its scalings carry the equation's residual on the noisy jets and are lost from 1e-4 on. (iv) On KdV the
exact right-hand side with the system on the evaluation trajectories recovers the algebra (0.00), where the training
trajectories, half broadband humps whose u_t the snapshots under-resolve, leave 0.99; the fitted F does not (0.93).
Predictions (seeds 42-44; means over seeds unless stated; "LieNLSD central" = LieNLSD_RHS@eval with central
differences; m = k):
(R1) Reproduction: "protocol" gives the paper's exact rows (heat 0.00 on both columns, KdV 0.01); on clean data
LieNLSD_RHS and LieNLSD_F give the published first-run values (0.43 / 0.56 / 0.66 and 0.30 / 0.42 / 0.91; heat, heat's
class, KdV) to two decimals (same code path, same seeds).
(R2) FINS in this configuration on clean data: <= 0.05 on both equations, 3/3.
(T1) FINS under thinning: <= 0.05 at s = 2, 4, 8 on both equations, 3/3 each.
(T2) Heat: LieNLSD central misses more than FINS at every s.
(N1) sigma = 1e-4: FINS <= 0.10 on both equations, and fewer than LieNLSD central and than LieNLSD_F@eval on both.
(N2) sigma = 1e-3 and 1e-2: FINS misses more than LieNLSD central on both equations.
(N3) Mechanism: LieNLSD's translations have plain containment >= 0.99 in every variant and condition, 3/3; at
sigma >= 1e-3 FINS's d_t has containment < 0.9 on >= 2/3 per equation.
(N4) KdV, clean: LieNLSD_RHS@eval <= 0.10 on >= 2/3 for both jets; LieNLSD_F@eval >= 0.5.
No predictions: sigma = 3e-4 (between the regimes), LieNLSD under thinning on KdV, the fitted F beyond N1.
Paper, fixed now: a subsection on noise and coarse sampling with these rows (FINS exact; LieNLSD central and spectral
@eval; LieNLSD F) whatever the verdicts, saying where FINS loses; the LieNLSD paragraph adds N4 (with the exact
right-hand side LieNLSD's KdV gap is the training data's; with the fitted F it is F's).
Colab (learned operators; cells "exp48" in the notebook): "clean" through the protocol's operators in this
configuration; "n1e-4" and "t4" with operators retrained on the corrupted training data (400 epochs). (C1) clean: KdV
<= 0.05 (FINS's KdV result does not rest on the solver stencil), heat within 0.10 of the protocol's 0.14 (finite). No
prediction for the retrained operators.

**P5a / P5b per seed (exp42 --seeds 43, --seeds 44 on the Drive files, 04.10).**
LIG with 5 generators, trained / untrained: s42 1.97 / ~1.75 (untrained from the three-seed mean 1.95), s43 1.98 /
1.99, s44 1.69 / 2.11. FINS per seed: s43 0.07, s44 0.91 (s42 ~0.34 from the mean 0.44).
(P5a) MET 3/3: 1.97 > ~0.34, 1.98 > 0.07, 1.69 > 0.91.
(P5b) NOT MET (s44): |trained - untrained| = ~0.22, 0.01, 0.42 against 0.3; on s44 training moves LIG 0.42 closer to
the algebra.
Burgers per seed, LieNLSD (2^20 points) against FINS: 0.02 / ~0.34 (s42), 1.15 / 0.07 (s43), 0.47 / 0.91 (s44):
LieNLSD misses fewer on seeds 42 and 44, FINS on seed 43 and on the mean (0.44 vs 0.55).

**P5a / P5b (LIG with k = 5 generators, Burgers) -- results (exp42 on the Drive files, 04.10).**
LIG with 5 generators, 10 epochs: 1.88 +/- 0.14 over seeds 42-44 (seed 42 1.97); untrained with 5 heads 1.95 +/- 0.14.
(P5a) MET on 3/3: the per-seed values lie in [1.68, 2.08] given the mean and sd, every one above FINS's largest
per-seed value (<= 0.94; mean 0.44 +/- 0.35). (P5b) MET on the means (1.88 vs 1.95, within 0.3); the registered
per-seed check needs exp42 --seeds 43 and --seeds 44 separately (pending).

**LieNLSD with the official amount of training data -- results (CPU; results/lienlsd_bigdata/: tables.md, summary.json,
logs/; untrained nulls rerun with the same 2^20-point standardisation).**
Missing dimensions, m = k, mean +/- sd (per seed 42 / 43 / 44); Burgers 5 / KdV 4 / heat 6 / heat's class 10:
fitted F, 2^20 points     0.55 +/- 0.46 (0.02, 1.15, 0.47)  0.90 +/- 0.13 (0.97, 0.72, 1.01)  0.32 +/- 0.12 (0.16, 0.46, 0.34)
                          0.43 +/- 0.17 (0.20, 0.59, 0.49)
F at initialisation       1.07 (1.31, 1.03, 0.86)  0.98 (0.90, 1.05, 0.99)  1.14 (1.22, 1.11, 1.11)  2.29 (2.11, 2.31, 2.47)
own count                 1.38 (k 3-5)  1.57 (2-4)  0.59 (8-10)  0.93
first run, 10 Nx Nt       0.68, 0.91, 0.30, 0.42
(L11) MET: fewer than with 10 Nx Nt points on Burgers (0.55 < 0.68, lower on every seed) and on KdV (0.90 < 0.91, by
0.01, within the seed spread; per seed 0.97, 0.72, 1.01 against 0.97, 0.74, 1.00). Heat (no prediction): 0.32 and
0.43 against 0.30 and 0.42.
(L12) MET: FINS 0.01 against 0.90 on KdV.
Diagnostics: held-out relative MSE 6.5e-3 - 1.25e-2 (Burgers; 7.0e-3 - 1.7e-2 before), 2.2e-2 - 3.4e-2 (KdV; same as
before), 2.2e-4 - 2.6e-4 (heat; 3.2e-4 - 1.8e-3 before); chosen checkpoints at epochs 120-170, 100-200, 40-80; F's
error on the evaluation jets below the exact right-hand side's on 9/9; Jacobian error 0.07-0.16, 0.06-0.30, 0.21-0.47.
Trained above untrained on Burgers s43 and KdV s42, s44, as before. Cost: 1421-1445 s (heat), 2831-2877 s (Burgers,
KdV) on one CPU core each, Burgers s42 2438 s on two.
Rule registered with L11: L11 holds, so the paper's LieNLSD row is now this configuration; the first run stays in the
appendix. Against FINS (0.44, 0.01, 0.14, 1.59) nothing changes in kind: FINS leads on Burgers (narrowly, and not on
seed 42), KdV and the finite heat algebra; LieNLSD on heat's class.

**LieNLSD with the official amount of training data -- run log (04.10).** The first attempt (queued after the 2D run,
05:48) was killed when the idle machine was reclaimed, before any result (its log held only the data header).
Restarted at 09:48 with the same configuration, resumable, in the order Burgers, KdV (the columns of L11-L12), heat.

**LieNLSD on the 2D equation -- results (CPU; results/neural2d/lienlsd_2d/lienlsd_2d.json, .log).**
Missing dimensions on exp45's held-out probe, m = 6, characteristic space (field space), seeds 42 / 43 / 44:
fitted F, central differences (LieNLSD as published)  0.868, 0.400, 0.935 (1.08, 0.71, 1.28); mean 0.73
fitted F, FINS's derivatives                          0.727, 0.594, 0.130 (0.97, 0.68, 0.49); mean 0.48
exact RHS, central differences                        0.166, 0.005, 0.025 (0.40, 0.02, 0.07)
exact RHS, FINS's derivatives                         0.0007, 0.0001, 0.0003 (0.003, 0.000, 0.001)
F at initialisation (either jets)                     1.89, 1.88-1.89, 2.09
FINS's polynomial class (exp45): exact operator 2.1-2.2e-6, FNO2d 1.4e-3, 6.2e-4, 3.8e-4.
(L7) MET: LieNLSD as published misses more than FINS's class through the FNO2d on 3/3 (0.40-0.94 vs <= 1.4e-3).
(L8) MET: exact right-hand side on FINS's derivatives <= 7e-4 on 3/3.
(L9) MET: F trained below F at initialisation for both jets on 3/3.
(L10) no prediction: FINS's derivatives help the fitted F on 2 of 3 seeds (0.73 -> 0.48 on average).
The weak generators are again the scalings (novel containment with the fitted F: D1 0.72, D2 0.59, central
differences; 0.82, 0.77 spectral). F's Jacobian error against the exact one is small (0.045-0.085), but F misses the
held-out u_t by 1.5-3.7 %, above the exact right-hand side's 1.6 % (central) and 0.15 % (spectral): unlike in 1D,
the fitted F does not reach the finite-difference floor. Best checkpoints at epochs 140-190 of 200. The relative
threshold returns six on every run (at least six singular values below 0.01). Cost: 200-268 s per fitted run on two
CPU cores.
Reading: in 2D LieNLSD with its fitted network misses 0.4-0.9 of six dimensions, three orders above FINS's class
through the FNO; with the exact right-hand side it is exact up to the jets (7e-4 on FINS's derivatives). Its error
is the fitted network's, concentrated in the scalings.

**LieNLSD on FINS's derivatives -- results (CPU; results/lienlsd_spectral/: tables.md, summary.json, logs/).**
Missing dimensions, m = k, mean +/- sd (Burgers 5 / KdV 4 / heat 6 / heat's class 10):
exact RHS, FINS's derivatives   0.46 +/- 0.33  0.64 +/- 0.45  0.00 +/- 0.00  0.00 +/- 0.00   (own count 1.26, 0.67, 0.00, 0.00)
fitted F, FINS's derivatives    0.70 +/- 0.23  0.95 +/- 0.07  0.09 +/- 0.05  0.13 +/- 0.06   (own count 1.85, 1.62, 0.24, 0.45)
(for comparison, central differences: exact RHS 0.40, 0.66, 0.43, 0.56; fitted F 0.68, 0.91, 0.30, 0.42)
(L5) NOT MET (KdV): heat 0.00 and 0.00, but KdV 0.64: better jets do not close LieNLSD's KdV gap. With the exact
right-hand side the fourth singular value stays at 2.0e-2, 1.2e-3, 1.7e-2 of the largest (central differences:
1.7e-2, 8.6e-4, 1.6e-2). On the training jets of KdV (3 trajectories checked) the equation's residual is 1.8 %
(central differences) and 2.7 % (spectral x, five-point t): the time derivative of snapshots 3.9e-4 apart, not
the spatial stencil, limits the jets, and the scaling's residual in the determining system is proportional to it
(25 and 33 per unit coefficient norm against 5e5 - 1.5e6 for random vectors). Burgers does not improve either
(spectral derivatives of under-resolved fronts; equation residual on the evaluation jets 0.08-0.88).
(L6) no prediction: the fitted F improves on heat (0.30 -> 0.09, 0.42 -> 0.13; held-out relative MSE 3.6e-5 -
2.5e-4 instead of 3.2e-4 - 1.8e-3) and is unchanged on Burgers (0.68 -> 0.70) and KdV (0.91 -> 0.95; per seed 0.98,
0.85, 1.02).
Against FINS (0.44, 0.01, 0.14, 1.59): on FINS's derivatives LieNLSD's fitted F misses fewer than FINS on heat, both
columns (0.09 vs 0.14; 0.13 vs 1.59), and more on Burgers and KdV.
Per seed, published configuration (for the record): fitted F Burgers 0.06, 1.31, 0.68; KdV 0.97, 0.74, 1.00; heat
0.45, 0.21, 0.25; class 0.58, 0.25, 0.41; untrained Burgers 1.32, 1.04, 0.86; KdV 0.90, 1.05, 0.99; heat 1.23, 1.10,
1.11; class 2.10, 2.30, 2.47: trained above untrained on Burgers s43 and KdV s42, s44 (L2 holds on the means only).
Exact RHS with central differences, KdV per seed 0.97, 0.00, 1.00 (s43 recovers all four).

**LieNLSD as published with the official amount of training data -- pre-registration (04.10; written after the
1D results above and an independent check, before any run of this configuration).**
Why: the check found that "10 Nx Nt points" is ten of OUR trajectories (81 920 points for Burgers and heat, 163 840
for KdV), while the official ten initial conditions hold 978 040 / 958 080 points; at the official epoch counts F
therefore took 6-12x fewer optimiser steps, and its chosen checkpoint was at the last epochs on Burgers (200, 180,
190) and KdV s43 (200). Configuration: as LieNLSD_F (finite-difference jets, official epochs 200 / 100 / 200,
checkpoint every 10 epochs on held-out trajectories), with 2^20 = 1 048 576 training and 2^18 validation points
(exp41 --lienlsd-fit-points 1048576 --lienlsd-val-points 262144). Runs: seeds 42-44, three equations, CPU.
Predictions (missing dimensions, m = k, means over seeds):
(L11) fewer than with 10 Nx Nt points on Burgers (0.68) and KdV (0.91); no prediction on heat.
(L12) FINS (0.01) still misses fewer than LieNLSD on KdV.
The paper's LieNLSD row changes to this configuration if L11 holds on either column (the configuration closer to
the official one); otherwise both are reported.

**LieNLSD on the 2D equation (exp47_lienlsd_2d.py) -- pre-registration (04.10; written before its runs; one code
check on development data, below).**
Setting: exp45's data (16 training trajectories, data seed 4000 + s; held-out probe on 8 trajectories, 5000 + s),
FINS's class (60 point fields, degree <= 2 in X, Y, T, U) and exp45's probe scores (missing dimensions in
characteristic space, primary, and in field space). LieNLSD as its repository runs its 2D tasks: F = MLP 200 x 3,
sigmoid, inputs (U, U_X, U_Y, U_XX, U_YY, U_XY), target U_T, Adan lr 1e-3, batch 256, 200 epochs, checkpoint every
10 epochs chosen on the last two training trajectories; 2^17 training and 2^15 validation points drawn from the
scored block |x|, |y| < 4.5 on snapshots 2-48 (in LieNLSD's favour: the informative region, standardised inputs);
determining system on 200 random block points of each of 10 training trajectories; column-scaled SVD; m = 6
smallest right singular vectors (own count: below 0.01 of the largest). Jets: central differences, second order in
t (as published), or FINS's derivatives (spectral in x, y, fourth order in t). Variants: F fitted, F at
initialisation, exact right-hand side.
Code check (development data seed 7, 4 trajectories, 400 points, exact Jacobian): the translations and the rotation
satisfy the system to round-off, the scalings D1, D2 leave 0.47, 0.82 (central differences) and 0.036, 0.062
(spectral) per unit coefficient norm, random coefficient vectors 1.1e3-5.4e3; the exact right-hand side misses the
jets' u_t by 1.8e-2 (central) and 2.9e-3 (spectral).
Reference: FINS's polynomial class, six lowest on the same probe, missing (characteristics) 2.1-2.2e-6 through the
exact operator and 3.8e-4 - 1.4e-3 through the FNO2d; random subspaces 2.58-2.78.
Predictions (missing in characteristic space, m = 6, seeds 42-44):
(L7) LieNLSD as published (fitted F, central differences) misses more than FINS's polynomial class through the
FNO2d, 3/3.
(L8) Exact right-hand side on FINS's derivatives: <= 0.05 on >= 2/3.
(L9) F trained below F at initialisation, for both kinds of jets, 3/3.
(L10) No prediction for the fitted F on FINS's derivatives against central differences.

**Through the learned FNO2d (Colab): mu = 0.01 training and finite flows -- results (logs only:
results/neural2d/flows/logs_fno/; the JSON, .pt and .npz files are on Drive).**
(MF1) MET: full span missing (characteristics) 0.061, 0.090, 0.104 (<= 0.25 on 3/3); untrained 1.93, 1.75, 1.95.
(MF2) MET: lambda7 / lambda6 inside the span on held-out pairs 10.0, 5.9, 2.3 against 2.67, 3.29, 1.74 at mu = 1 (3/3).
(MF3) MET: six lowest directions (training pairs) 0.064, 0.094, 0.112 (<= 0.15 on 3/3).
The FNOs' rollout error relative to the data increment: one step 0.0022 on held-out data, K steps 0.0010-0.0011.
(XF1) NOT MET: the translations' largest E 5.1e-4 / 5.5e-4, 4.1e-4 / 5.1e-4, 6.3e-4 / 7.7e-4 (d_x / d_y), at their
first-order residuals (r1 4.4e-4 - 8.6e-4) and falling with size (E(1) / E(0.003) 0.12-0.19). The FNO has no
positional input but does not commute with continuous shifts below its own error level.
(XF2) MET: E(0.003) / r1 0.77-1.34, 0.70-1.14, 0.73-1.38 (54/54: algebra, eigenvectors 1-6, tau-removed, random).
(XF3) NOT MET: eigenvectors 7-10 break down (E(1) >= 10 E(0.003) or non-finite) on 3/4, 2/4, 4/4; on s43
eigenvectors 7 and 8 stay flat (0.52x, 0.87x).
(XF4) MET: d_t, rotation, D1, D2 and eigenvectors 3-6 finite and below tenfold growth on 8/8, 7/8, 7/8 (exceptions:
eigenvector 5 on s43, 23x; D2 on s44, 16x); the scalings grow 2.6-16x at eps = 1 (D1 3.9-4.9x, D2 2.6-16x).
(XF5) MET (mu = 0.01 networks): directions 1-6 E(1) <= 3 E(0.003) on 17/18 (direction 4 of s44 non-finite);
directions 7-8 non-finite on 6/6. mu = 1 networks, reported: 1-6 within 1.36x on 18/18; 7-8 non-finite on 5/6
(through the exact operator X6: 3/6).
P5a / P5b (LIG with k generators, Burgers s43 / s44): trained here (1712 s, 1720 s for ten epochs), but the logs hold
no scores; the verdicts need exp42 on baselines_burgers_s43/s44.npz (Drive).

**LieNLSD on FINS's derivatives -- pre-registration (04.10; after the results above, before any run of this
variant).**
Why: the smallest singular values of LieNLSD's determining system (summary.json, sv_rel) are exact zeros only for
generators whose condition does not involve the data's jets (translations; heat's superpositions; 3, 3 and 7 of them
with the exact right-hand side), while the scalings and KdV's Galilean generator sit at the level of the next
directions: sigma_(k+1) / sigma_k = 1.9, 6.3, 2.0 on KdV with the exact right-hand side (FINS through the FNO:
lambda_5 / lambda_4 = 223-1089). If second-order finite-difference jets set that level, better jets should close
LieNLSD's KdV gap. Variant (methods LieNLSD_RHS_spec, LieNLSD_F_spec; run_lienlsd_published(jets="spectral")): the
derivatives FINS uses -- spectral in x (the data are periodic), the five-point fourth-order central difference in t
(times 2 .. Nt - 3), u_tx and u_txx spectral -- for the training jets of F and for the determining system;
everything else as in LieNLSD_F / LieNLSD_RHS (checked on sin(2x) exp(-t): spectral jets exact to 3e-10, FD to
1e-2). Runs: seeds 42-44, the same data, no untrained null. Readouts m = k and own count.
Predictions (mean missing dimensions, m = k):
(L5) exact right-hand side on FINS's derivatives: <= 0.10 on KdV, heat (6) and heat's class. No prediction on
Burgers (its data are under-resolved on 64 points: finite-difference floor up to 0.62, and spectral derivatives
of under-resolved fronts are no better).
(L6) fitted F on FINS's derivatives: no prediction (its Jacobian error, 0.08-0.56 on finite-difference jets, may
dominate).
Paper: a sensitivity row for each, marked as a modification of LieNLSD; the published LieNLSD row stays.

**LieNLSD as published -- results (CPU; results/lienlsd_published/: tables.md from exp42, summary.json, logs/; the
fields, 0.6 GB, stay outside the repository in fins_runs/lienlsd_published).**
Missing dimensions, mean +/- sd over seeds 42-44; columns Burgers (5), KdV (4), heat (6), heat's class (10):
LieNLSD_F (as published), m = k     0.68 +/- 0.51  0.91 +/- 0.12  0.30 +/- 0.11  0.42 +/- 0.13
LieNLSD_F, own count                1.77 +/- 0.32  1.58 +/- 0.59  0.57 +/- 0.36  0.96 +/- 0.78   (k 3-5, 2-4, 8-10)
F at initialisation, m = k          1.07 +/- 0.19  0.98 +/- 0.06  1.14 +/- 0.06  2.29 +/- 0.15
LieNLSD_RHS (exact RHS), m = k      0.40 +/- 0.32  0.66 +/- 0.46  0.43 +/- 0.24  0.56 +/- 0.35
LieNLSD_RHS, own count              1.27 +/- 0.53  0.67 +/- 0.47  0.43 +/- 0.24  0.56 +/- 0.35
v1 wrapper, re-run                  0.85 +/- 0.21  0.67 +/- 0.47  1.12 +/- 0.30  1.40 +/- 0.47
(L1) MET: the re-run reproduces the reported 0.85, 0.67, 1.12, 1.40 to two decimals.
(L2) MET: F trained below F at initialisation on each column (1.07 -> 0.68, 0.98 -> 0.91, 1.14 -> 0.30,
2.29 -> 0.42); on KdV by 0.07 only.
(L3) MET: FINS 0.14 against 0.30 on the finite heat algebra.
(L4) no prediction: F worse than the exact right-hand side on Burgers (0.68 vs 0.40) and KdV (0.91 vs 0.66), better
on heat (0.30 vs 0.43) and heat's class (0.42 vs 0.56).
Against FINS (reported 0.44, 0.01, 0.14, 1.59): FINS misses fewer on Burgers, KdV and the finite heat algebra;
LieNLSD misses fewer on heat's full class, recovering the heat polynomials (novel containment 0.99) where FINS's
learned operators put near-symmetries. Weakest generators of LieNLSD (novel containment, F / exact RHS): scaling on
Burgers 0.72 / 0.77 and on KdV 0.32 / 0.41, KdV's Galilean generator 0.53 / 0.73; finite heat >= 0.95 with F.
Diagnostics. F's held-out relative MSE 3.2e-4 - 1.8e-3 (heat), 7.0e-3 - 1.7e-2 (Burgers), 2.1e-2 - 3.4e-2 (KdV);
chosen checkpoint at epochs 30-80, 180-200, 60-200. Jacobian error relative to the exact Jacobian at the
determining-system points 0.13-0.56 (heat, almost all in dF/du, which is zero), 0.08-0.35 (Burgers), 0.09-0.18
(KdV). On the jets of the evaluation trajectories F's relative error is below that of the exact right-hand side on
every run (heat 0.011-0.019 vs 0.013-0.024, Burgers 0.06-0.18 vs 0.08-0.62, KdV 0.010-0.014 vs 0.014): F fits the
finite-difference relation, the case L4 left open. Own counts: Burgers 3, 5, 3; KdV 2, 4, 2; heat 10, 10, 8. Cost on
two CPU cores, training included: 59-65 s (heat), 120-129 s (Burgers), 241-246 s (KdV); the v1 wrapper 49-57 s.
Reading: LieNLSD as published is the strongest baseline: far below the class nulls on every equation, ahead of FINS
on heat's full class, behind it on Burgers, KdV and the finite heat algebra. On KdV it misses 0.66 even with the
exact right-hand side, and the fitted F (0.91) is barely ahead of its untrained null (0.98): its pointwise condition
does not isolate KdV's scaling and Galilean generators on these data, which FINS recovers (novel containment 1.00).
Its own count of generators costs it 0.3-1.1 dimensions against m = k. The v1 row understated LieNLSD on heat (1.12
against 0.30) and overstated it on KdV (0.67 against 0.91).

**LieNLSD as published -- correction and pre-registration (04.10; written before any run of the new code, no
development run).**
Correction. The official repository (github.com/hulx2002/LieNLSD: train.py, discovery.py) fits an MLP F to u_t as a
function of (u, u_x, u_xx) (heat, Burgers) or (u, u_x, u_xx, u_xxx) (KdV) on finite-difference jets and builds the
determining system from F's Jacobian; no equation enters. Our wrapper (pde/method_lienlsd.py: run_lienlsd and its
protocol form run_lienlsd_general) substituted the equation for F and took the cross derivatives from a network
fitted to u(x, t, sample index). The two LieNLSD bugs of the protocol v1 entry (t-component returned as xi, textbook
Burgers) were bugs of that wrapper, not of LieNLSD: the official code orders its blocks (d_t, d_x, d_u), labels them
so (utils.vis), and uses no equation. Likewise method_lig.py, method_liegan.py, method_laligan.py and
method_augerino.py are our reimplementations from the papers (aligned with the official code where it is public),
not the authors' code, and LIG's normalisation bug was in method_lig.py. The v1 "LieNLSD" row is LieNLSD's
determining system with the exact right-hand side, an oracle variant, correctly marked as using the equation.
New code: run_lienlsd_published (method_lienlsd.py); exp41 methods LieNLSD_F (LieNLSD as published) and LieNLSD_RHS
(the same pipeline with the exact right-hand side in place of F); exp42 rows "own count".
Configuration, official (model.py, train.py, data_utils/*.py, README): F = MLP 200 x 3, sigmoid; Adan (sail-sg), lr
1e-3, batch 256, MSE; epochs 200 / 100 / 200 (Burgers / heat / KdV); jets by central differences, periodic in x, on
interior times; u_tx, u_txx by central differences of u_t. Adaptations, (a) the protocol's, as for the v1 row:
library of all monomials of the class degree (2, 2, 3), column-scaled SVD, components in (xi, eta, phi) order,
determining system on 200 random interior points of each of 10 training trajectories (official: 100 points); (b) in
LieNLSD's favour: inputs and target standardised (undone in the Jacobian); 10 Nx Nt training points drawn from 960 of
the 1024 training trajectories (the size of the official ten initial conditions on our grids); checkpoint every 10
epochs, the one with the lowest loss on 2 Nx Nt points of the other 64 trajectories kept (the official repository
ships checkpoints at epochs 140, 30, 90 of 200, 100, 200). Readouts: (m = k) the k smallest right singular vectors,
k = 5, 4, 10, the m FINS uses; (own count) those of them below 0.01 of the largest singular value, at least one, as in
the v1 row (official: an absolute threshold set per task). The official basis sparsification, a rotation inside the
null space, is skipped: the scores are basis-free.
Runs (CPU): seeds 42-44, 1024 training trajectories, the ten evaluation trajectories of data seed
1000 + s; LieNLSD_RHS and LieNLSD_F; --untrained: LieNLSD_F with F at initialisation (own-class null); the v1 wrapper
re-run as a reproduction check. Scores: exp42, missing dimensions against the finite algebra and heat's class.
Diagnostics, reported without thresholds: F's held-out relative MSE; the relative error of F's Jacobian against the
exact one at the determining-system points; the relative errors of F and of the exact right-hand side on the jets of
the evaluation trajectories (the second is the finite-difference floor).
Predictions (mean missing dimensions, seeds 42-44; columns Burgers (5), KdV (4), heat (6), heat's class (10)):
(L1) Reproduction: the re-run v1 wrapper within 0.15 of the reported 0.85, 0.67, 1.12, 1.40 on each column (its
cross-derivative network now trains on a CPU).
(L2) F trained misses fewer dimensions than F at initialisation (m = k) on each column.
(L3) FINS through the learned operator (reported 0.44, 0.01, 0.14, 1.59) misses fewer dimensions than LieNLSD_F
(m = k) on the finite heat algebra. No prediction on Burgers, KdV and heat's class.
(L4) No prediction for LieNLSD_F against LieNLSD_RHS: F is fitted to finite-difference u_t, so its Jacobian may fit
the finite-difference jets of the determining system better than the exact one.
Paper, fixed now: Table 2's LieNLSD row becomes LieNLSD_F (m = k), LieNLSD_RHS (m = k) replaces the v1 oracle row,
F at initialisation joins the chance rows; own-count rows and the v1 values go to the appendix.

**Jet flows, KdV with four times the steps -- post-hoc check (results/jets/flows_safety16/; does not change the
registered verdicts).** 16 steps per unit rate instead of 4: the largest E of the true characteristics and the three
lowest eigenvectors at any size falls from 2.7e-5, 1.2e-6, 3.9e-5 to 1.0e-7, 1.2e-8, 2.9e-6 (K5's flow: 1.0e-7,
9.6e-9, 2.9e-6); the ratio of G2 becomes 1.3e5, 3.5e5, 861; the non-symmetries' growth is unchanged (1.2-15.9x). The
KdV failures of G1 were the integrator's; G2 still fails on s44, G3 is not integrator-limited.

**Finite flows of jet characteristics -- results (CPU; results/jets/flows/{kdv,pme,heat}_s{42,43,44}.json, logs/).**
(G1) NOT MET (KdV). Largest E of the true characteristics and the r lowest eigenvectors at any size: heat 6.1e-13,
2.1e-12, 2.5e-12; porous medium 4.8e-8, 2.6e-8, 1.1e-7; KdV 2.7e-5 (K3, s42), 1.2e-6 (s43), 3.9e-5 (K3, s44) against
the registered 1e-5. K5's flow, the fifth-order KdV equation, commutes with the solver to 6.4e-6, 1.2e-6, 1.9e-5 at
every size (realised eps up to 0.17-0.25). The KdV floors are the integrator's error, not the band: on seed 42 K3's
defect at eps = 0.3 is 2.7e-5 with 4 steps per unit rate, 3.1e-8 with 16 and 1.9e-10 with 64 (eigenvector 1: 9.8e-6,
9.2e-9, 3.3e-10); the held-out data of seeds 42-44 hold 1.1e-9, 9.3e-10, 2.1e-8 of their norm above k = 32 (dev
7.2e-10).
(G2) NOT MET (KdV). E(0.003) of eigenvector r + 1 over the largest of the r lowest: heat 1.6e11, 2.1e11, 3.9e11; porous
medium 2.9e5, 6.9e5, 1.8e5; KdV 386, 5990, 80.
(G3) NOT MET (heat 4/6, KdV 3/6; porous medium 6/6). E(1) / E(0.003) of eigenvectors r + 1, r + 2: porous medium
non-finite on 6/6 (eigenvector 3, the background shift: evolutions fail, u < 0; eigenvector 4: flows stop at the
200 000-step cap, approaching a blow-up, or evolutions fail); heat 1.23, 1.63 (s42), 5.21, 3.6 (s43), non-finite,
non-finite (s44: the pointwise flow of u^2 - ... blows up; the other stops at the cap);
KdV 2.68, 15.9 (s42), 1.17, 3.72 (s43), 1.75, 1.77 (s44) -- the travelling-wave relation grows 1.2-2.7x.
Reading: exponentiated, the recovered generalised symmetries remain symmetries of the discrete evolution at every
size tested -- for KdV including the fifth-order flow of K5 -- at floors set by round-off (heat), time stepping
(porous medium) and the step of the flow integrator (KdV). The next eigenvectors start 80 to 4e11 times higher, and break down only
where the flow leaves the equation's state space or the pointwise flow blows up (porous medium on 6/6, heat on 4/6);
KdV's travelling-wave relation grows slowly instead. The jet flows confirm the generalised symmetries; as a
breakdown criterion they are reliable only for the porous-medium equation.

**Jet flows -- change during the final runs (before any porous-medium result and with heat seed 42 and KdV seen).**
The porous-medium eigenvector 4 flow on seed 42 ran for 16 CPU minutes without finishing: with the step bound
re-evaluated at the current state, a flow approaching a blow-up takes ever smaller steps. The integrator now stops an
element after 200 000 steps and marks it failed (non-finite), as for a non-finite rate; flows that finish are
unchanged (the largest finished flow so far took 23 112 steps). The porous-medium seed 42 run is repeated.

**Finite flows of jet characteristics, second attempt (exp46_jet_flows.py) -- pre-registration (written after the
development runs below, before the final seeds).**
Integrator: Lawson RK4 (the constant-coefficient linear part of Q exact in Fourier space, the rest explicit), step at
most 1 / (4 x rate) with the explicit part's largest rate re-evaluated every 16 steps at the current state, and at
most |s| / 8; flow band P: the solver's 2/3 band for heat and the porous medium, |k| < 32 for KdV. Why the band: on
the K5 flow (development seed, 4 pairs) |k| < 64 with 1 or 4 steps per unit rate diverged from eps = 0.03-0.1; |k| < 42
with 4 steps stayed finite but drifted to E = 0.86 at eps = 1; |k| < 32 kept E <= 1.5e-7 at every size. The KdV data
hold at most 7.2e-10 of their norm above k = 32 (5.8e-10 above 42).
Development (exp46's seed 0: pencils 4600, flows on 2 held-out trajectories of 5600 x starts 0, 16):
heat -- the seven linear monomials and eigenvectors 1-7: E <= 1.4e-11 at every size; eigenvector 8 (u^2 - ...): 7.1e-2
flat to eps = 0.3, non-finite at 1 (the pointwise flow w' ~ w^2 blows up); eigenvector 9: 0.107 -> 0.452.
porous medium -- u_x, u u_xx + u_x^2 and eigenvectors 1-2: <= 8e-9; eigenvector 3 (1 - 0.45 u): 1.8e-2 flat to 0.3,
non-finite at 1 (u < 0); eigenvector 4: 3.9e-2 -> 0.69.
KdV -- u_x <= 3e-9, K3 <= 5.6e-7, K5 <= 5.3e-7 (realised eps up to 0.18), eigenvectors 1-3 <= 1.9e-7 (up to 0.78);
eigenvector 4 (the travelling-wave relation): 7.1e-3 -> 1.7e-2 at eps = 1 (x2.4, no blow-up); eigenvector 5: 2.8e-2 ->
0.51.
Predictions (final: exp46's saved pencils of seeds 42-44, held-out data 5600 + s, 2 trajectories x starts 0, 16):
(G1) the true characteristics and the r lowest eigenvectors (KdV 3, porous medium 2, heat 7): E <= 1e-5 at every size,
3/3 per equation.
(G2) E(0.003) of eigenvector r + 1 >= 1000 x the largest E(0.003) among the r lowest, 3/3 per equation.
(G3) eigenvectors r + 1 and r + 2: E(1) >= 2 E(0.003) or non-finite on >= 5/6 per equation.

**Through the learned FNO2d (Colab): mu = 0.01 training and finite flows -- pre-registration (written before any of
these runs; no development run with a trained FNO, which lives on Drive; the code path was smoke-tested on the CPU with an
untrained FNO).**
(A) Training with mu = 0.01 through the FNO2d (exp45_neural_2d.py --op fno --m 8 --mu 0.01, seeds 42-44, the saved
FNOs; outputs neural2d/fno_m8_mu001_s{s}). Reference: through the FNO at mu = 1 (N6) full span 0.148, 0.076, 0.089, six
lowest 0.197, 0.216, 0.181, lambda7 / lambda6 2.67, 3.29, 1.74; through the exact operator at mu = 0.01 (T1-T3) 0.061-
0.104, 0.065-0.111, 2.3-9.65.
(MF1) full span missing (characteristics) <= 0.25 on 3/3. (MF2) lambda7 / lambda6 inside the span on held-out pairs
above its mu = 1 value through the FNO on >= 2/3. (MF3) six lowest directions (training readout) <= 0.15 on >= 2/3.
(B) Finite flows through the FNO2d (exp45_flows.py --op fno: S_K and its tangent by the FNO in double precision, the
end snapshot the FNO's own rollout, u_t inside the flows the one-sided fourth-order difference of the FNO rollout,
the eigenvectors of the FNO pencil (fno_m6); 8 held-out trajectories x starts 2, 18, all generators of the exact
runs; networks: fno_m8_s{s} and, after (A), fno_m8_mu001_s{s}). Thresholds as in F1-F6 where they carry over:
(XF1) translations: E <= 1e-4 at every size, 2/2 on 3/3 (the FNO has no positional input).
(XF2) E(0.003) / r1 in [0.5, 2] where r1 >= 1e-6 among the algebra, eigenvectors 1-6, the tau-removed eigenvectors and
the random fields, 3/3.
(XF3) eigenvectors 7-10 of the FNO pencil: E(1) >= 10 E(0.003) or non-finite on >= 3/4 per seed, 3/3.
(XF4) d_t, rotation, x d_x + y d_y + 2t d_t, t d_t - u d_u and eigenvectors 3-6: E(1) < 10 E(0.003) and finite on
>= 6/8 per seed, 3/3 (no breakdown, with room for the FNO's error on snapshots the scalings move out of its training
distribution).
(XF5) network directions, mu = 0.01 FNO networks: 1-6 E(1) <= 3 E(0.003) on >= 15/18; 7-8 E(1) >= 2 E(0.003) or
non-finite on >= 4/6. mu = 1 FNO networks: reported.

**Independent check of the new paper text against the result files (03.10).** A separate check recomputed every
number in Secs. 5.5-5.6, Tables 15-16 and the new appendix paragraphs from the JSON/log files. Corrected: jets missing
bounds (KdV 2.2e-14, porous medium 1.2e-11, heat 3.6e-15; the J1/J3 entries above said 2e-14 / 2e-15), the mu = 0.01
extra directions' ratio to the six (4.7-12x, not 4.7-9.3x), the near-symmetries' E(0.003) upper end (2.6e-2), the
mu = 0.01 gap's upper end (9.6), the translations' final-seed defect (2.2-2.8e-7; the appendix had quoted
development values), the exception for eigenvectors 1-2 (translation combinations) from the 1.8x growth bound, the
population of the 33 generators in F2, the breakdown mechanism (a sign-changing tau carrying 15-37 % of the field,
not tau's mere x, y dependence: eigenvectors 7-8 have small sign-changing tau terms and survive to eps = 0.3), the
heat gaps (round-off ratios, not gaps), and the scope of the abstract (exact operator, polynomial class). All
registered verdicts reproduced.

**Finite flows exp(s v) -- results (CPU; results/neural2d/flows/flows_s{42,43,44}.json, flows_net_{m8,m8_mu001}_s*.json,
logs/; figure figures_flows.py, seed 43). Paper labels: X1-X6 = F1-F6.**
(F1) NOT MET: 9, 10, 8 of 12 (registered 12/12 on 3/3). Largest E of the algebra and eigenvectors 1-6 per seed:
rotation 1.0e-4, 6.5e-5, 1.3e-4; x d_x + y d_y + 2t d_t 1.4e-4, 1.2e-4, 2.0e-4; eigenvector 5 8.0e-5, 6.5e-5, 1.3e-4;
eigenvector 6 2.3e-4, 1.8e-4, 2.9e-4 -- in every case at or next to the smallest size, i.e. the first-order residual
(r1 1.1-2.4e-4 for D1 and eigenvector 6, 2-3x the development seed's); no member grows by more than 1.8x over the
sizes (D2 1.5-1.7x at 6e-9 - 1e-8, eigenvector 3 1.2-1.8x at 4e-7 - 1e-6, the rest 1.00-1.04x) except the
translations and eigenvectors 1-2 (their combinations), whose 3.3e-13 - 2.5e-12 rises to 2.2-2.8e-7 with the
s-integrator's error. The breakdown signature is absent; the
registered absolute bound, calibrated on one development seed, was too tight for these floors.
(F2) MET: E(0.003) / r1 0.74-1.21 on 33/33.
(F3) MET: 4/4 on 3/3. First breakdown (E >= 10 E(0.003) or non-finite) at eps = 0.1 for eigenvector 10 (tau share
0.18-0.37; flows fail with u still positive), 0.3 for eigenvector 9 on s42, s44 (tau share 0.15, 0.16; flows fail),
1.0 for eigenvectors 7, 8 on all seeds and 9 on s43 (tau share 0.01-0.06; T_s u0 down to -0.67 ... -5.5, evolutions
fail, or E(1) / E(0.003) from 12 to 5e22). E(0.003) of 7-10: 1.06-2.65e-2.
(F4) MET: 57 cases with T_s u0 > 0, largest E / E(0.003) 1.19; tau removed raises E(0.003) to 1.9e-2 - 0.17.
(F5) MET: E(0.003) 0.105-0.152, largest growth 2.53x.
(F6) NOT MET. mu = 1 networks (2 held-out trajectories): directions 1-6 E(1) <= 2 E(0.003) on 18/18 (ratios 0.48-1.34;
E(0.003) 1.1e-3 - 1.0e-2); directions 7, 8 break down on 1, 2, 0 of 2 -- 3/6 against >= 4/6 (s42: 7 flat 0.84x, 8
non-finite; s43: 7 non-finite, 8 2.89x; s44: 1.71x, 0.90x); their E(0.003) 7.1e-3 - 1.3e-2 overlaps the six's.
mu = 0.01 networks (reported, no prediction): directions 1-6 E(0.003) 2.7e-3 - 1.6e-2, growth 0.42-2.46x; directions
7, 8 E(0.003) 5.4e-2 - 9.4e-2 (4.7-12x the largest of the six per seed; tau share 0.52-0.96) and break down on 6/6
(5 at eps = 0.3, 1 at eps = 1; failed flows or u < 0).
Reading: the finite flow of a generator of the algebra -- learned or polynomial -- is a symmetry of the discrete
evolution up to eps = 1: its defect stays within 1.8x of the first-order value (translations: the s-integrator). The
polynomial near-symmetries break down within eps <= 1, through an ill-posed flow when a tau that changes sign across
the domain (x t, y t terms) carries 15-37 % of the field (cf. the classical tau = tau(t) restriction), or through loss
of positivity when tau carries <= 6 % (eigenvectors 7-8 also have small tau terms in x^2, y^2 that change sign, and
their flows stay finite through eps = 0.3); with tau removed and u > 0 their defect stays at its first-order value (the second-order
breaking is invisible at these sizes). Through the networks the finite flow adds separation only when the loss has
already separated the extra directions (mu = 0.01); at mu = 1 they are mixtures whose flows break down on half the
runs.

**A structural class from Lie theory: tau = tau(t) (V5) -- results (CPU; results/neural2d/tau_class.json,
logs/tau_class.log).**
(V5a) MET. Six lowest missing 0.0000 x 3 (exact), 0.0004, 0.0004, 0.0002 (FNO; degree 2: 0.0014, 0.0006, 0.0004).
(V5b) NOT MET (0/6). lambda7 / lambda6: exact 391, 429, 306 against 284, 312, 224 (1.37-1.38x); FNO 41, 47, 38 against
27, 33, 31 (1.2-1.5x). lambda7 rises from 2.3-3.2e-4 to 3.1-4.4e-4 (exact) and 3.2-4.7e-4 to 4.6-7.3e-4 (FNO); the
affine class (no phi = x^2, y^2 at all): 3.3-3.8e-3.
(V5c) MET. Largest-ratio rank 6, 6, 6 through the FNO; 1, 2, 1 through the exact operator.
Reading: the tau = tau(t) restriction costs the algebra nothing and removes the ill-posed near-symmetries, but
re-optimised within the restricted class the far-field superpositions phi = x^2, y^2 find combinations almost as good
without tau(x, y) terms: their small loss is owed to phi (the static far field), not to tau. Removing tau from the
specific eigenvectors (finite flows, F4) raised their residuals 3-10x only because those vectors were optimised with
it.

**A structural class from Lie theory: tau = tau(t) (V5) -- pre-registration (written before any restricted pencil was
solved).** Point symmetries of a second-order evolution equation have tau = tau(t) (the coefficient of u_xt in the
determining equations forces tau_x = tau_y = tau_u = 0); the finite flows showed that near-symmetries carrying tau
terms in x, y are exactly those whose flows are ill-posed, and that removing tau raises their first-order residuals
3-10x. Class: the 48 fields of the degree-2 class whose tau lies in span{1, t, t^2}, as the sub-pencil of the saved
60 x 60 pencils (exact_m6 and fno_m6, training pairs), scored on the held-out probe points (exp45_tau_class.py).
(V5a) six lowest missing (characteristics) <= 0.01 through either operator, 3/3; (V5b) lambda7 / lambda6 at least 2x
the degree-2 class's on >= 4/6 runs; (V5c) largest-ratio rule among the first 10 reads 6 through the FNO on 3/3 and
falls inside the algebra through the exact operator on >= 2/3 (numerical floors, expected failure).

**Finite flows of jet characteristics (exp46_jet_flows.py) -- development only, not pursued.** The evolutionary flow
dw/ds = P Q[w] of a jet characteristic (Lawson RK4: the constant-coefficient linear part exact in Fourier space, the
rest explicit; s signed so the leading even-order part is dissipative), defect E as in exp45_flows.py; exp46's
development seed (data 4600 / 5600), 2 held-out trajectories x starts 0, 16, 1 RK4 step per unit of |s| x explicit
rate.
heat -- the seven linear monomials and eigenvectors 1-7: E <= 1.4e-11 at every size (linear flows commute with the
linear evolution); eigenvector 8 (u^2 - 0.9 ...): 7.1e-2 flat to eps = 0.3, non-finite at 1 (the pointwise flow
w' ~ w^2 blows up); eigenvector 9: 0.107 -> 0.452 (x4.2).
porous medium -- u_x and u u_xx + u_x^2 and eigenvectors 1-2: <= 8e-9; eigenvector 3 (1 - 0.45 u): 1.8e-2 flat to
0.3, non-finite at 1 (u < 0); eigenvector 4: 3.9e-2 -> 0.69 (x17).
KdV -- u_x at round-off, K3 (the KdV flow itself) 4e-9 -> 8e-7; K5 (the fifth-order KdV flow): E 2e-8 on the
trajectory with lower solitons, but on the other the explicit stage blows up from eps = 0.01 even with 4x the
steps (stiffness ~ 10 u k^3 at the band edge, cubic terms not dealiased by the 2/3 rule; at eps = 0.003 the flow
already rotates the phases of modes above k ~ 10 by more than a radian, so E(0.003) is not the first-order limit
for K5). Eigenvector 4 (the travelling-wave relation): 7.1e-3 -> 1.3e-2 at eps = 0.3; eigenvector 5: 2.8e-2 ->
6.9e-2.
Not pursued: in the two equations where the flows are cheap the generalised symmetries are point symmetries in
disguise (translations, time translation, linear superpositions), and the one genuinely generalised flow, K5's, needs
a stiff dispersive integrator (implicit or exponential with the variable-coefficient third-order term, 1/2-rule
dealiasing) that we did not build. No predictions registered; nothing from this enters the paper.

**Finite flows -- deviation from the registration (made before any network output was seen).** The network
directions (F6 and the mu = 0.01 report) run on 2 held-out trajectories x starts 2, 18 (4 pairs, the development
setting) instead of 8 x 2: through 16 pairs a single direction took ~27 min on the shared 2-core CPU (the network is
evaluated at every RK4 stage of every flow), 3.6 h per network. The polynomial generators (F1-F5) keep 8 x 2.

**Training modification: mu = 0.01 in the neural Ky Fan loss -- results (CPU; results/neural2d/exact_m8_mu001_s{42,43,44}
.json/.pt/.npz, logs/).**
(T1) MET. Full span missing (characteristics) 0.061, 0.091, 0.104 (fields 0.226, 0.219, 0.544); mu = 1: 0.135, 0.113,
0.090 (fields 0.536, 0.349, 0.189). Novel containment >= 0.95 for every generator.
(T2) NOT MET. lambda7 / lambda6 inside the span on held-out pairs 9.65, 6.75, 2.31 (lambda6 1.4e-3, 2.0e-3, 4.1e-3;
lambda7 1.3e-2, 1.4e-2, 9.4e-3; the mu = 0.01 pencil); mu = 1: 2.25, 1.64, 1.50.
(T3) MET. Six lowest directions (training readout) missing 0.065, 0.096, 0.111 (fields 0.259, 0.271, 0.601); mu = 1:
0.400, 0.287, 0.370.
Reading: with the absolute term lowered, the static far-field superpositions no longer sit next to the algebra, so
the network's six lowest directions are the algebra (3-6x fewer missing dimensions) and the gap after the sixth
widens 1.5-6x, but not to the registered 10: the sixth direction (a scaling, novel containment 0.95-0.97) is the
least accurate field and its eigenvalue rises with the smaller normalisation. Training time 1111-3170 s (CPU, shared).

**Finite flows exp(s v): do near-symmetries break down where symmetries do not? (exp45_flows.py) -- pre-registration
(written after the development runs below, before the final seeds).**
Construction: the finite transformation of a snapshot is the evolutionary flow dw/ds = P Q[w; t] (u_t := the operator's
own generator; P the solver's circular 2/3 band), RK4 in s with 2 steps per unit of |s| x (transport + diffusion
rate) plus 8, s signed to run the time part forward; coefficient functions at windowed coordinates x' = chi(x) x
(chi = 1 for |x| <= 6, 0 beyond 7.75, C-infinity between: fields without x, y dependence are untouched). Defect after
K = 8 snapshots E = ||S_K(T_s u0) - T_s u_K|| / ||T_s u_K - u_K|| on the scored block, sizes eps = ||T_s u0 - u0|| /
||u0 - mean u0|| in {0.003, 0.01, 0.03, 0.1, 0.3, 1} (nominal; the realised size is recorded); r1 = ||J_K Q0 - Q_K|| /
||Q_K|| with the exact tangent (the s -> 0 limit). Without the projection and with a C^1 cosine window on the whole
characteristic, D2 sat at 6-7e-4 (the band-limited solver freezes the flow's out-of-band part); with both changes it
sits at 6.5e-9, as without any window.
Development (pencil on data 100, flows on held-out 101, 2 trajectories x starts 2, 18):
algebra -- E at every size: d_x, d_y 4e-13 rising to 2e-7 at eps = 1 (the s-integrator's error: 2.3e-9 with 4x the
steps, 1.2e-11 with 16x), d_t 3e-8 to 8e-11, rotation 1.2-2.3e-5, D1 2.2-4.7e-5, D2 6.5-8.6e-9; discovered
eigenvectors 1-6 <= 6.8e-5 at every size.
near-symmetries -- eigenvectors 7-10 (lambda 3.0e-4 to 1.3e-3; phi ~ x^2, y^2 with tau terms in x^2, x t, y t; tau's
share of the field 0.04, 0.02, 0.07, 0.16): E(0.003) 1.25e-2, 1.25e-2, 1.55e-2, 2.20e-2 (r1 1.14e-2, 1.26e-2,
1.55e-2, 1.45e-2), flat to eps = 0.3 for 7-8 and to 0.03 for 9; E(1) non-finite, 0.207, non-finite, non-finite
(eigenvector 10 already at 0.1). Eigenvectors 9-10: the flow itself diverged (tau ~ x t, y t: a backward diffusion on
part of the domain); 7-8: T_s u0 negative (min -0.7 to -1.1 at eps = 1).
tau removed (both signs) -- r1 rises 3-10x (0.04-0.13): tau is part of these near-symmetries; E stays within 1.8x of
E(0.003) while T_s u0 > 0 and becomes non-finite only after u < 0 (eigenvector 7, s < 0, at eps = 1.6; 8, s > 0,
at 1.24).
random (tau = 0) -- E(0.003) 0.163, 0.167; E(1) 0.280, 0.283.
network (dev m = 8 run, data 100) -- directions 1-6: E(0.003) 1.7e-3 to 5.3e-3, E(1) / E(0.003) 0.78-1.40;
directions 7, 8: E(0.003) 7.9e-3, 9.3e-3, E(1) / E(0.003) 2.3 and non-finite.
Predictions (final: pencil on training data 4000 + s, flows on held-out 5000 + s, 8 trajectories x starts 2, 18,
seeds 42-44; networks exact_m8_s{s} (mu = 1) and exact_m8_mu001_s{s} (mu = 0.01)):
(F1) no breakdown: the six true generators and eigenvectors 1-6: E <= 1e-4 at every size, 12/12 on 3/3.
(F2) first-order limit: E(0.003) / r1 in [0.5, 2] for every generator with r1 >= 1e-6 among the algebra, eigenvectors
1-6, the tau-removed eigenvectors and the random fields, 3/3.
(F3) breakdown: eigenvectors 7-10: E(1) >= 10 E(0.003) or non-finite on >= 3/4 per seed, 3/3.
(F4) mechanism: eigenvectors 7-10 without tau: E <= 3 E(0.003) at every size at which T_s u0 > 0 on the whole grid,
all such cases on 3/3.
(F5) random: E(0.003) >= 0.05 and E <= 3 E(0.003) at every size, 6/6.
(F6) networks, mu = 1: directions 1-6 E(1) <= 2 E(0.003) (finite) on >= 15/18; directions 7, 8 E(1) >= 2 E(0.003) or
non-finite on >= 4/6. mu = 0.01 networks: reported, no prediction.

**Generalised symmetries in a jet-polynomial class (exp46) -- results (CPU; results/jets/{heat,pme,kdv}_s{42,43,44}
.json/.npz, logs/).**
(J1) MET. KdV: the three lowest eigenvectors contain u_x, K3, K5 on 3/3 (held-out missing <= 2.2e-14; random 3-dim
subspaces of the class 1.74-1.79); lambda4 / lambda3 2.6e4, 7.4e3, 2.4e6. Rayleigh quotients: u_x ~1e-28; K3 6.9e-17,
5.3e-16, -6.0e-18; K5 2.7e-12, 1.4e-11, 1.9e-13; lambda4 2.4e-7, 3.9e-7, 1.4e-6 (u_xx with u, u u_xx, u_x^2, u^2: the
travelling-wave relation of the solitons the humps shed).
(J2) MET. Porous medium: the two lowest contain u_x and u u_xx + u_x^2 on 3/3 (missing <= 1.2e-11; random 1.55-1.56);
lambda3 / lambda2 7.8e5, 6.3e5, 3.7e5; lambda2 is the time-translation characteristic at 1.6-3.3e-10 (time-stepping
floor); lambda3 1.1-1.2e-4 (1 - 0.41..0.48 u, a background shift); no third generalised symmetry, as for a
non-integrable equation.
(J3) MET. Heat: the seven lowest contain the seven linear monomials (missing <= 3.6e-15; random 3.04-3.10); lambda8 /
lambda7 4.4e13, 5.5e27, 1.8e28; lambda8 1.7-4.8e-3 (u^2 - 0.9 ...).
(J4) NOT MET. Largest-ratio rank: heat 7, 7, 7; porous medium 2, 1, 2 (s43: lambda2 / lambda1 = 2.5e7 above lambda3 /
lambda2 = 6.3e5 -- the time-translation floor 1.7e-10 against u_x at 7e-18); KdV 2, 2, 3 (!= 3 on 2/3, the registered
expected failure; on s44 K3 is at round-off and K5 at 1.9e-13, six orders below lambda4).
Reading: in the jet class the transport criterion returns the KdV hierarchy u_x, K3, K5 and the two generalised
symmetries of the non-integrable porous-medium equation, with gaps of 4 to 6 orders (heat's seven sit at round-off, so
its 13-28 orders are a round-off ratio), on fresh held-out data; the
largest-ratio rule fails whenever the floors of the true characteristics (round-off, time stepping) differ by more
orders than the gap -- the same failure as for point symmetries through exact operators.

**Generalised (Lie-Baecklund) symmetries in a jet-polynomial class (exp46) -- pre-registration (written after one
development seed, before the final seeds).**
Theory: a generalised symmetry in evolutionary form is a characteristic Q[u] (u and its x-derivatives) that solves
the linearised equation on solutions -- FINS's transport condition verbatim. Class: the 15 monomials of KdV weight <= 7
in (u, u_x, ..., u_5x); equations on [0, 2 pi), pseudo-spectral, integrating-factor RK4 with its exact tangent (against
central differences of the solver 3e-10 to 7e-10). Expected ranks: KdV u_t = -u_xxx - 6 u u_x 3 (u_x, K3 = u_xxx +
6 u u_x, K5 = u_5x + 10 u u_xxx + 20 u_x u_xx + 30 u^2 u_x); 1D porous medium u_t = (u u_x)_x 2 (u_x, u u_xx + u_x^2;
not integrable); heat u_t = 0.5 u_xx 7 (every linear monomial 1, u, ..., u_5x).
Development (seed 0, data 4600 / 5600): heat -- seven at <= 2e-17, lambda8 1.7e-3, held-out missing 0.0000 (random
3.10); porous medium -- two at <= 6e-11, lambda3 1.1e-4 (1 - 0.51 u, a background shift), missing 0.0000 (random 1.48);
KdV, sech^2 humps, 8 RK4 steps per snapshot -- u_x 1e-28, K3 1e-12, K5 2.7e-8, lambda4 9.3e-7 (u_xx - 0.79 u + 0.64 u^2
..., the travelling-wave relation of the solitons the humps shed), gap 11; 16 steps: K5 2.3e-9 (time-stepping
floor, x12), gap 133, missing 0.0000; random-mode data instead of humps: K5 outside the three lowest (missing 1.0;
degenerate jets on short dispersive data) -- not used. Adopted: humps, 32 RK4 steps per snapshot for KdV.
Predictions (final: seeds 42, 43, 44, data 4600 + s / 5600 + s):
(J1) KdV: the three lowest eigenvectors contain u_x, K3, K5 (held-out missing <= 0.01) on 3/3; lambda4 / lambda3 >= 100
on >= 2/3.
(J2) porous medium: the two lowest contain u_x and u u_xx + u_x^2 (missing <= 0.01) and lambda3 / lambda2 >= 1e3, 3/3.
(J3) heat: the seven lowest contain the seven linear monomials (missing <= 0.01) and lambda8 / lambda7 >= 1e6, 3/3.
(J4) largest-ratio rank over the 15 eigenvalues: heat 7 and porous medium 2 on 3/3; KdV != 3 on >= 2/3 (expected
failure: K3 sits at a lower numerical floor than K5).

**Training modification: a smaller absolute term (mu = 0.01) in the neural Ky Fan loss -- pre-registration
(written before the runs).**
V3 showed that at mu = 0.01 the static far-field superpositions lose their advantage (polynomial lambda7 3.5-4.7e-3,
gap 905-1556) while the algebra stays <= 4.5e-6. Same networks, data and defaults as the exact-operator m = 8 runs,
only --mu 0.01 (training and evaluation pencils); seeds 42-44.
(T1) full span missing (characteristics) <= 0.25 on 3/3. (T2) lambda7 / lambda6 inside the span on held-out pairs
>= 10 on >= 2/3 (mu = 1: 1.5-2.3). (T3) six lowest directions (training readout) missing <= 0.15 on >= 2/3 (mu = 1:
0.29-0.40).

**Criteria beyond the eigenvalue -- 1D learned operators: results (Colab; logs/closure_rank_1d_learned.log).**
Closure-gated rank (tau 0.1), protocol data / fresh seeds: KdV 1, 1, 4 / 4, 1, 4 -- 3/6, (V4L) KdV NOT MET (>= 4/6
registered); escaped at k = 4: 0.110, 0.162, 0.070 / 0.097, 0.189, 0.076, at k = 5: 0.321, 0.374, 0.375 / 0.393,
0.378, 0.439 (the 4 -> 5 rise is visible on 6/6, but the registered threshold is crossed at k = 4 on 3/6). Heat 1 on
6/6 -- != 10, (V4L) heat MET as the registered expected failure (escaped at k = 10: 0.20-0.30, exact 0.10). Burgers
1, 1, 1 / 1, 5, 1 -- 1/6 (escaped at k = 5: 0.236, 0.103, 0.401 / 0.205, 0.050, 0.142). Largest-ratio rank on the same
runs: KdV 4 on 6/6, Burgers 2, 5, 3 / 3, 7, 7, heat 6, 11, 3 / 7, 11, 12.
Reading: the two rules are complementary. Closure needs accurate eigenvectors: it reads the rank through exact
operators (KdV 6/6, 2D 3/3) and through the 2D FNO (3/3), whose polynomial gap is 27-33, but the 1D learned
operators' discovered spans leave 0.07-0.40 of their brackets (K2) and the threshold fails. The eigenvalue ratio needs
the floors levelled by the operator's error: it reads KdV through its learned operator (6/6) and 2D through the FNO
(3/3), never through an exact operator. No single rule reads every case; post hoc variants (e.g. the largest rise of the
escaped share) are not adopted: one reads learned KdV 6/6 but exact KdV 3 instead of 4.

**Criteria beyond the eigenvalue -- 1D learned operators, pre-registration (written before the Colab run).**
exp44_closure_rank.py on the protocol's learned-operator runs (fins_{pde}_s{42,43,44}.npz, protocol data and
fresh/): (V4L) KdV closure-gated rank 4 on >= 4/6; heat != 10 on >= 4/6 (truncated superposition ideal); Burgers
reported (its discovered spans left 0.05-0.40 of their brackets in K2).

**Criteria beyond the eigenvalue -- results (CPU; exp45_criteria.py, exp44_closure_rank.py).**
Outputs: results/neural2d/criteria.json, logs/criteria.log, logs/closure_rank_1d.log.
(V1) NOT MET. FNO quotient / exact eigenvalue for the exact eigenvectors 1-12: s42 4.5e21, 5.4e11, 8.9e5, 2.6e4, 41,
6.0 | 3.6, 4.2, 1.8, 1.5, 2.8, 1.5; s43 (exact lambda1 -1.5e-16) 2.6e21, 4.3e5, 9.3e3, 40, 8.2 | 3.7, 3.2, 2.1, 1.4,
1.5, 1.3; s44 (lambda1 -2.3e-17) 5.9e21, 4.1e5, 4.8e3, 38, 5.4 | 5.3, 3.5, 2.0, 1.3, 1.9, 1.8. The floors set by
round-off and the time difference rise by orders of magnitude, but the resolution-limited sixth direction rises only
5.4-8.2x and the near-symmetries 7-8 rise 3.2-5.3x: the FNO's error along the far-field quadratics (a background with
curvature, never in its training data) is large, so the operator's error is not uniform across directions.
(V2) MET (a, b, c). Affine (matrix) class, 20 fields: six lowest missing 0.0000, 0.0000, 0.0000 (exact), 0.0000,
0.0000, 0.0002 (FNO); lambda7 / lambda6 3325, 4135, 3374 (exact; degree 2: 284, 312, 224), 211, 330, 270 (FNO; degree
2: 27, 33, 31); lambda7 3.3-3.8e-3 through either operator (identical to 3 digits: an intrinsic loss); largest-ratio
rank 6, 6, 6 through the FNO, 1, 1, 2 through the exact operator (expected failure, numerical floors).
(V3) MET. mu = 0.01: quotients of the mu = 1 eigenvectors 7, 8 rise 12, 55x (s42), 11, 58x (s43), 18, 61x (s44);
six lowest <= 3.8e-6, 2.9e-6, 4.5e-6, held-out missing 0.0000; lambda7 / lambda6 905, 1556, 1047 (mu = 1: 284, 312,
224). The near-symmetries owe their small loss to the absolute term over the static far field.
(V4) MET. 2D, escaped share for k = 1..10, exact: (abelian) 0.07-0.15, 0.13-0.23, 0.002-0.004, 0.000, 0.002-0.004 |
0.63-0.68, 0.85-0.86, 0.83, 0.78-0.82; FNO: (abelian) 0.56-0.97, 0.68-0.85, 0.49-0.59, 0.07-0.11, 0.05-0.08 | 0.39-0.65,
0.68-0.85, 0.66-0.82, 0.77-0.79; random 6-dim 0.45-0.49, 7-dim 0.49-0.51; ground truth 1e-14. Closure-gated rank 6 on
6/6 (largest-ratio rule: 1, 2, 2 exact; 6, 6, 6 FNO). 1D exact: KdV 4 on 6/6 (escaped at k = 4 0.043-0.069, at k = 5
0.32-0.46; largest-ratio rule 2 on 6/6); heat 10, 7, 7, 7, 7, 6 (k* != 10 on 5/6 as registered; escaped at k = 8-10
0.10-0.17, the truncation level of its full ground truth; the six lowest are closed, 0.000).
Checks of the bracket code: [D1, d_x] = -d_x, [rotation, d_x] = -d_y, [D2, d_t] = -d_t; the algebra leaves 5e-15; adding
x^2 d_u alone leaves 0.76; adding all six phi(x, y) d_u of degree <= 2 leaves 4e-15 -- the far-field superpositions
form an ideal closed under the algebra, so closure rejects partial sets of them, not the complete set.
Reading: closure under the bracket is the criterion that reads the rank where the spectrum cannot (exact operators
in 1D and 2D, numerical floors) and keeps working through the learned operator (2D); the affine class and a smaller
absolute term widen the gap by excluding or penalising the static far-field superpositions; operator sensitivity is
confounded by the operator's direction-dependent error.

**Criteria beyond the eigenvalue: separating the algebra from near-symmetries -- pre-registration (written before
any of these quantities was computed).**
Motivation: the largest-ratio rule fails through exact operators (numerical floors inside the algebra) and in the
neural class (precision); the near-symmetries found next to the algebra in 2D (phi = x^2, y^2, ...) are superpositions
of the linearisation about the uniform background, transported almost exactly where u is constant. Four candidates,
none using ground truth in the criterion itself (ground truth only scores them):
(V1) operator sensitivity (2D, degree-2 class, the saved exact and FNO pencils: same data, pairs and basis): for the
12 lowest eigenvectors of the exact pencil, r = Rayleigh quotient through the FNO / exact eigenvalue. Prediction: r >= 10
for each of eigenvectors 1-6 and r <= 3 for each of 7-12, on 3/3 seeds (exact symmetries are operator-limited --
Proposition 2 -- near-symmetries intrinsic).
(V2) matrix (affine) class aff(4) on (x, y, t, u): the 20 fields of degree <= 1 (generators = 5 x 5 matrices in
homogeneous coordinates), as the sub-pencil of the saved 60 x 60 ones. Predictions: (a) six lowest contain the algebra,
held-out missing (characteristics) <= 0.01, exact and FNO, 3/3; (b) lambda7 / lambda6 above the degree-2 class's on
6/6 runs; (c) largest-ratio rule among the first 10 gives 6 through the FNO on 3/3; through the exact operator the
largest ratio falls inside the algebra on >= 2/3 (expected failure: numerical floors).
(V3) normalisation sensitivity (2D, degree-2 class, exact operator, pencils recomputed with A'A + B'B and G0 kept
apart): the Rayleigh quotients at mu = 0.01 of the mu = 1 eigenvectors 7 and 8 are >= 10x their mu = 1 eigenvalues,
3/3; the six lowest eigenvalues at mu = 0.01 stay <= 1e-5 and their held-out missing <= 0.01, 3/3 (static
near-symmetries owe their small loss to the absolute term; the algebra's characteristics move).
(V4) closure-gated rank: k* = largest k (k <= 10 in 2D; k <= k_true + 3 in 1D, the saved eigenvectors) whose k lowest
eigenvectors leave at most 0.1 of their brackets (exp44's escaped share; abelian spans count as closed).
Predictions: 2D, exact and FNO, k* = 6 on 6/6; 1D exact KdV (protocol data 1042-1044, fresh 2042-2044) k* = 4 on
>= 5/6; 1D exact heat k* != 10 on >= 4/6 (expected failure: the class truncates the superposition ideal, its full
ground truth escapes 0.10-0.13).

**Neural generators in 2D (exp45) -- m = 8 through the FNO: results (Colab GPU).**
Outputs: results/neural2d/fno_m8_s{42,43,44}.json/.npz; 600 s per seed (FNOs loaded).
Full span missing (characteristics) 0.148, 0.076, 0.089 (fields 0.591, 0.157, 0.217; exact operator 0.135, 0.113,
0.090); six lowest directions (training readout) 0.197, 0.216, 0.181 (fields 0.90, 0.65, 0.63; exact 0.400, 0.287,
0.370); untrained 1.93, 1.75, 1.95, random 8-dim subspaces 2.10, 1.97, 2.19. Span eigenvalues on held-out pairs:
lambda6 1.75e-4, 8.8e-5, 1.02e-4, lambda7 4.7e-4, 2.9e-4, 1.8e-4, lambda8 1.24e-3, 4.1e-4, 3.6e-4: lambda7 / lambda6
2.67, 3.29, 1.74 (exact 2.25, 1.64, 1.50). Novel containment of the full span: translations and rotation >= 0.998,
D1 0.954, 0.979, 0.976, D2 0.935, 0.964, 0.957.
(N6) MET (3/3, 3/3). (N6b) MET: lambda7 / lambda6 < 10 on 3/3 -- the rank is not identified in the neural class
through the learned operator either. Through the FNO the six-lowest readout is cleaner than through the exact
operator (0.18-0.22 against 0.29-0.40) and the ratio slightly larger: the FNO lifts the class's first non-symmetries
(polynomial lambda7 3.2-4.7e-4 against 2.3-3.2e-4) more than the network's least accurate scaling (lambda6 0.9-1.8e-4
against 0.7-1.7e-4).

**Neural generators in 2D (exp45) -- m = 8 through the FNO, pre-registration (written after the Colab cell was
started, before any of its output was seen).**
Same FNOs (s42-s44), same data and defaults, m = 8. The exact-operator m = 8 runs (N4, N4b) and the FNO m = 6 runs
(neural floor unchanged by the operator, algebra's own quotients <= 1.7e-5, first non-symmetries 3-5e-4) predict the
same picture:
(N6) full span missing (characteristics) <= 0.25 on 3/3; six lowest directions (training readout) <= 0.5 on >= 2/3.
(N6b) lambda7 / lambda6 inside the span on held-out pairs < 10 on >= 2/3 (rank not identified in the neural class).

**Neural generators in 2D (exp45) -- N5, learned FNO2d: results (Colab GPU, rerun after the fix).**
Outputs: results/neural2d/fno_m6_s{42,43,44}.json/.npz (FNO checkpoints on Drive). FNOs loaded (s42, s43) or
trained (s44, ~48 min); neural stage 480 s per seed (2500 steps).
Gate, one-step error relative to the data increment: held-out 0.0021570, 0.0021593, 0.0021578 (K-step 0.00096,
0.00111, 0.00100); FINS training trajectories 0.0027, 0.0025, 0.0037 (K-step 0.0015, 0.0014, 0.0017) -- met 3/3, by
13-23x. The three held-out values agree to 0.1 percent across seeds; exp45_fno_errors.py (Colab) shows this is a
coincidence of aggregation: per-trajectory one-step errors differ (s42 0.0010-0.0051, s43 0.0009-0.0033, s44
0.0011-0.0028; plain rms 0.00248, 0.00218, 0.00178), the increment weights of the trajectories differ, and the
weighted rms rebuilt from the printed per-trajectory values with locally recomputed weights gives 0.002165, 0.002155,
0.002162 (rounding of the printed values). Unseen sets 8000 + s: 0.0024, 0.0020, 0.0032 (per trajectory 0.0007-0.0053);
FINS training sets 0.0027, 0.0025, 0.0037 (0.0007-0.0062), the largest on 4044 (u max 2.12). A CPU control (small
FNO2d, width 16, 12 modes, 3 layers, 96 trajectories, 6 epochs) has errors 0.038-0.057 on the held-out sets, also
rising with amplitude.
Polynomial class through the FNO (training pairs): six lowest 1.7e-5, 9.7e-6, 1.2e-5 (worst d_t 1.7e-5, 9.2e-6,
1.3e-5; rotation 5.6e-6, 3.5e-6, 4.1e-6; D2 7.7e-7, 5.0e-7, 1.3e-6), lambda7 4.7e-4, 3.2e-4, 3.7e-4: lambda7 / lambda6
26.9, 32.5, 31.0 (exact 284, 312, 224); held-out missing 0.0014, 0.0006, 0.0004 (fields 0.009, 0.005, 0.002); the
algebra's own pencil on held-out pairs 4.7e-7 ... 1.3e-5.
Neural, m = 6, through the FNO: held-out missing 0.102, 0.091, 0.092 (fields 0.164, 0.141, 0.162), against 0.104,
0.092, 0.086 through the exact operator; novel containment d_x, d_y, d_t, rotation >= 0.998, D1 0.968, 0.975, 0.973,
D2 0.948, 0.954, 0.953; span eigenvalues on held-out pairs 5.5e-6 ... 9.7e-5, 7.0e-6 ... 1.0e-4, 4.7e-6 ... 1.0e-4
(untrained 2.5e-3 ... 0.60, 2.3e-3 ... 0.40, 1.1e-3 ... 0.37); untrained missing 2.10, 2.76, 2.54.
(N5) MET: gate 3/3; neural <= 1.0 on 3/3 and below the untrained network on 3/3; polynomial <= 0.5 on 3/3.
Reading: through a stepper trained on trajectories alone both generator classes recover the whole 2D algebra,
rotation and both scalings included. The neural fields repeat their exact-operator values to +/- 0.006 missing
dimensions, so the network's precision, not the operator, sets their floor; the FNO lifts the algebra's own quotients
from <= 1e-6 to <= 1.7e-5 while the first non-symmetries stay at 3-5e-4, so the polynomial class's gap shrinks from
~300x to ~30x. m = 8 through the FNO: not run (not registered).
Code: operator_errors now runs one trajectory at a time (the all-snapshot float64 batch needed > 6 GB on the CPU
control) and returns per-trajectory errors; aggregates unchanged.

**Neural generators in 2D (exp45) -- N5, learned FNO2d, first Colab run (partial).**
FNO2d per seed (512 trajectories of data seed 6000 + s, width 32, modes 16, 4 layers, 60 epochs, two-step loss):
2 864 s (s42), 2 860 s (s43) on the Colab GPU; final loss 1.1e-5, 1.0e-5. Gate (one-step error relative to the data
increment, scored block): s42 training 0.003, held-out 0.002 (K-step rollout 0.001, 0.001); s43 training 0.003 --
met by a factor of 15-25. Through the s42 FNO the polynomial class has six lowest 6.9e-7 ... 1.7e-5 (d_x 3.5e-6, d_y
3.8e-6, d_t 1.7e-5, rotation 5.6e-6, D1 4.2e-6, D2 7.7e-7), lambda7 4.7e-4 (x28), held-out missing 0.001
(characteristics), 0.009 (fields); the algebra's own pencil on held-out pairs 4.7e-7 ... 1.3e-5. The neural stage
then stopped at its first step: kyfan built the identity on the CPU while the pencil sat on the GPU (the device
path had been tested on CPU only). Fixed (torch.eye(..., device=D.device)); no other tensor constructor in the
script lacks a device on the GPU path. Neural verdicts of N5 pending the rerun, which loads the saved FNOs.

**Neural generators in 2D (exp45) -- results, exact operator (CPU, 2 cores).**
Final seeds 42, 43, 44 (network seed s, training data 4000 + s, held-out 5000 + s; defaults; 11-15 min per run).
Logs and outputs: results/neural2d/ (exact_m{6,8}_s{42,43,44}.json/.pt/.npz, logs/).
Polynomial class (60 fields, exact eigensolve, training pairs): six lowest 1.1e-6, 8.0e-7, 1.0e-6 (d_x, d_y at 1e-27,
d_t 0.7-2.9e-9, rotation 0.8-1.2e-7, D1 4.3-6.6e-7, D2 2-5e-12); lambda7 3.2e-4, 2.5e-4, 2.3e-4 (phi = x^2, y^2); lambda7 / lambda6 284,
312, 224; six lowest on held-out: missing 0.000 (characteristics and fields). (N1) MET 3/3.
Neural, m = 6: held-out missing (characteristics) 0.104, 0.092, 0.086 (fields 0.171, 0.137, 0.146); novel
containment d_x, d_y, d_t, rotation 1.00 on 3/3, D1 0.97, 0.98, 0.98, D2 0.95, 0.95, 0.96; eigenvalues inside the span
on held-out pairs 1.7e-6 ... 1.0e-4, 3.9e-6 ... 1.0e-4, 9.6e-7 ... 1.0e-4 (untrained 2.5e-3 ... 0.60, 2.2e-3 ... 0.40,
9.9e-4 ... 0.37). (N2) MET: <= 0.25 on 3/3, <= 0.10 on 2/3 (seed 42 0.104).
Untrained network, m = 6: 2.10, 2.76, 2.54 (fields 2.72, 3.05, 3.01); random 6-dim subspaces of the class 2.66 +/-
0.18, 2.58 +/- 0.21, 2.78 +/- 0.18. (N3) MET 3/3 (seed 42 by 0.10).
Neural, m = 8: full span 0.135, 0.113, 0.090 (fields 0.536, 0.349, 0.189); six lowest directions read out on training
pairs 0.400, 0.287, 0.370 (fields 1.04, 1.13, 0.57); untrained 1.93, 1.75, 1.95, random 8-dim 2.10, 1.97, 2.19.
(N4) MET (3/3 and 3/3). Span eigenvalues on held-out pairs: lambda6 1.7e-4, 1.0e-4, 7.4e-5, lambda7 3.9e-4, 1.6e-4,
1.1e-4, lambda8 7.7e-4, 2.5e-4, 1.5e-4: lambda7 / lambda6 2.25, 1.64, 1.50. (N4b) MET: rank not identified by the
gap in the neural class, as registered. Directions 7 and 8 of the span are 95-99 percent explained (characteristics)
by the algebra plus phi = quadratic in (x, y) (98.9/95.1, 98.5/98.8, 98.6/98.6 percent; exp45_span_analysis.py --extra): the polynomial class's own first
non-symmetries; direction 6 is 84-89 percent algebra (D2 mixed with them).
(N5) learned FNO2d: pending (Colab cells added to the notebook).
Reading: trained through the transport criterion alone, pointwise networks find the whole six-dimensional algebra
of a 2D equation, rotation and both scalings included, on trajectories they were not trained on; their precision
(scalings at 1e-4) stops short of the class's first non-symmetries (2-3e-4), so only the polynomial class can show
the dimension. Correction (03.10): by the protocol's rule (largest ratio of consecutive eigenvalues among the first
k + 4) it does so only through the learned operator -- after the 6th on 3/3 through the FNO (26.9, 32.5, 31.0); through
the exact operator the largest ratio falls inside the algebra (after #1, #2, #2: translations at round-off 1e-28 /
1e-17, d_t at 1e-9, rotation and scalings at 1e-7 - 1e-6), as in 1D, although lambda7 / lambda6 is 224-312 there. Field-space numbers drift up with training for m = 8 (0.33 -> 0.54 on seed 42) while characteristics improve:
the far-field components that no derivative constrains.

**Neural generators in 2D (exp45) -- pre-registration (written before the final seeds were run).**
u_t = div(u grad u) = Lap(u^2 / 2) on [-8, 8)^2, grid 64^2; two or three anisotropic Gaussian bumps (widths 0.9-1.6,
centres in [-2, 2]^2) on a background b0 in [0.25, 0.45]; 51 snapshots, dt 0.02; scored block |x|, |y| < 4.5; K = 8.
Algebra (Ovsiannikov; exponent 1 is not the conformal case -1 of n = 2), 6-dim: d_x, d_y, d_t, rotation
-y d_x + x d_y, D1 = x d_x + y d_y + 2t d_t, D2 = t d_t - u d_u. Operator: the exact discrete map (pseudo-spectral RK4
with the circular 2/3 mask; tangent through the same RK4 stages; against central differences of the solver 6e-11).
Pencil: exp39's relative form (mu = 1). Generators: (i) FINS's polynomial class, degree 2 in (x, y, t, u), 60 fields;
(ii) m point fields (x, y, t, u) -> (xi, eta, tau, phi), shared swish trunk 2 x 64 and a 16-unit head per generator
(the layout of Ko et al. 2024, narrower), trained by Tr((D + eps)^-1 N), the sum of the pencil's generalised
eigenvalues inside the span (mixing-invariant, no orthogonality term), 4 (trajectory, start) pairs per step.
Scores on held-out trajectories: missing dimensions k_B - ||Qa' Qb||^2 in characteristic space (primary) and field
space; nulls: Gaussian-coefficient subspaces of the polynomial class, the untrained network.
Development (training data seed 100, held-out 101, network seed 0). Chosen on development data: the grid, bump
widths and scored block (at widths 0.7-1.4 and |x| < 6 the algebra's own quotients reached 6e-7 (rotation) and
1.6e-6 (D1) against lambda7 5e-5 -- a spatial-resolution floor: n = 96 lowers them 1000x); the optimisation (plain
Ky Fan, lr 1e-3 cosine over 300 steps stalled at missing 2.04 with rotation and D1 absent; a linear model over the
60 fields trained the same way needed 500 steps for 0.34; on that linear model a log-det objective converged the
easy directions and never found rotation and D1; adopted: outputs divided by the rms of their characteristic factor, network input
x / 4.5, Adam 3e-3 with 50 warm-up steps, constant, cosine tail over the last 40 percent to 2 percent, 2500 steps).
Development results: polynomial class -- six lowest <= 5.4e-7, lambda7 3.0e-4 (phi = x^2, y^2), held-out missing
0.000; random 6-dim subspaces of the class 2.61 +/- 0.24 (characteristics); untrained network 2.86; trained m = 6:
held-out missing 0.053 (characteristics), 0.128 (fields), novel containment d_x 1.00, d_y 1.00, d_t 1.00, rotation
1.00, D1 0.99, D2 0.97; eigenvalues inside the span on held-out pairs 2.3e-6 ... 1.0e-4 (untrained 8.4e-4 ... 0.57);
m = 8: the full span 0.057, its six lowest directions (read out on training pairs) 0.183; span lambda7 / lambda6 1.4
(training pairs), 1.5 (held-out), and 1.5-1.9 at evaluation horizons K = 16, 24, 32; directions 7 and 8 are 98-99
percent explained by the algebra plus phi = quadratic in (x, y), i.e. the polynomial class's own lambda7, lambda8, while
the network's D2-like direction sits at 9e-5: the gap is limited by the network's precision, not by new neural
near-symmetries (exp45_span_analysis.py).
Predictions (final: network seed s, training data 4000 + s, held-out 5000 + s, s = 42, 43, 44; defaults):
(N1) polynomial class: six lowest eigenvalues <= 1e-5, lambda7 / lambda6 >= 100, held-out missing (characteristics) of
the six lowest <= 0.01; 3/3.
(N2) neural, m = 6: held-out missing (characteristics) <= 0.25 on 3/3 and <= 0.10 on >= 2/3.
(N3) untrained network, m = 6: missing (characteristics) >= 2.0 on 3/3.
(N4) neural, m = 8: the full span's missing (characteristics) <= 0.25 on 3/3; its six lowest directions (training
readout) <= 0.5 on >= 2/3.
(N4b) rank from the gap inside the neural span: lambda7 / lambda6 on held-out pairs < 10 on >= 2/3 -- a registered
expected failure of rank identification in the neural class at this precision (development 1.5).
(N5) learned FNO2d stepper (Colab; registered before any FNO was trained): gate -- held-out one-step error relative to
the data increment <= 0.05; if met, neural m = 6 held-out missing (characteristics) <= 1.0 on >= 2/3 and below the
untrained network on 3/3, polynomial six lowest <= 0.5 on >= 2/3.
Field-space missing: reported, not registered (components multiplying vanishing derivatives are not identified by any
data-based loss; the linear model above reached 0.34 in characteristics with 2.2 in fields).

**Physics fresh seeds and closure -- part 3: heat (Colab).**
run_fresh.py with the protocol's heat tag (--tag _smooth_L1; the untagged checkpoint is an older operator trained on
unsmoothed data): protocol eigenvalues reproduced to 3.0e-13, 4.7e-13, 5.3e-12 -- the protocol's operators.
Data 2042-2044: all ten below lambda11 on 3/3 (worst margins 1.1x, 4.2x, 2.1x; on 2042 the four heat polynomials
sit at 1.2e-2 against lambda11 1.4e-2, inside the 10 lowest 0.74-0.83, and the characteristics move little,
relative change 0.07); finite algebra missing 0.19 +/- 0.02 against the Gaussian null 3.01 +/- 0.17, full class
1.36 +/- 0.73 against 6.36 +/- 0.18; rank from the largest gap 7, 11, 12 (k = 10).
(F1) heat 3/3: MET, so F1 MET for all three equations (Burgers 3/3, heat 3/3, KdV 3/3; on the protocol data 8/9).
(F2) heat finite and full below the null on every seed: MET for all three.
Closure on fresh heat, escaped (null mean - 2 sd): 0.303 (0.423), 0.199 (0.472), 0.219 (0.469); full ground truth
0.102-0.103. (K2) MET on 18/18 runs (protocol 9, fresh 9). (K1) and (K3) as before: NOT MET.

**Finite flows (transfer) -- results; physics fresh seeds and closure -- part 2 (Colab).**
transfer, 24 sources x 3 lowest + 8 random directions, 600 design-pool test images, tau 0.05 (1 910 s ResNet-50,
1 961 s ConvNeXt-T).
(W2) ResNet-50: (log T[A, B] + log T[B, A]) / 2, mean between superclasses - mean within +0.042, permutation p 0.0001.
MET. ConvNeXt-T: +0.093, p 0.0001. MET. The semantic structure of G4 holds for finite transformations (5 percent image
change) on images not used to find the directions. First-order in-sample calibration was +0.085 / +0.122: the finite,
out-of-sample effect keeps about half (ResNet-50) and three quarters (ConvNeXt-T) of it.
Exploratory: mean log transfer within each superclass / to the others (lower = disturbed less) -- ResNet-50 domestic
cats -0.57 / -0.39, dogs -0.58 / -0.39, big cats -0.44 / -0.36, buildings -0.38 / -0.28, but birds -0.06 / -0.19 and
vehicles -0.02 / -0.18 reversed; ConvNeXt-T domestic cats -0.51 / -0.32, dogs -0.79 / -0.48, big cats -0.49 / -0.33,
vehicles -0.32 / -0.26, birds -0.14 / -0.16, buildings +0.02 / -0.13 reversed. The finite effect is carried by the
mammal superclasses; the test images show cats, dogs, a dog breed, a fish, garbage trucks and churches, no birds.
Own-class ratio (own change / the design groups', relative to random flows): below 1 on 24/24 for both, median 0.89 /
0.73 (reported, not registered: it holds by construction even in an untrained network). Median statistic: +0.013,
p 0.0019 / +0.017, p 0.042.
KdV fresh seeds (run_fresh.py --pdes kdv): protocol eigenvalues reproduced to 1.6e-5, 2.9e-6, 8.0e-6 (same operators).
Data 2042-2044: all four below lambda5 with margins 514x, 366x, 616x; lambda5 / lambda4 510, 296, 505, so the rank
from the largest gap is 4 on 3/3; missing 0.01 +/- 0.00 against the Gaussian null 2.86 +/- 0.02.
(F1) Burgers 3/3, KdV 3/3 with margin >= 100x: MET for both; heat pending. (F2) Burgers 0.24, KdV 0.01 against
nulls 2.81, 2.86: MET for both; heat pending.
Closure on fresh KdV, escaped (null mean - 2 sd): 0.097 (0.213), 0.189 (0.244), 0.076 (0.258) -- K2 3/3; K3 (<= 0.10)
2/3. Seed 43's operator is the least accurate one on both data sets (one-step error 0.006 against 0.002, bottom
eigenvalues 1e-5 against 1e-6) and gives the largest escaped share on both.
VERDICTS so far: (K2) MET on protocol 9/9 and fresh 6/6 (heat fresh pending); (K3) NOT MET (protocol 1/3, fresh 2/3).

**Physics fresh seeds and closure -- results, part 1 (Colab).**
run_fresh.py: Burgers operators reproduce the protocol eigenvalues exactly (0.0e+00) on 1042-1044. Heat s42 does
NOT (relative 0.42: lambda11 3.5e-3 against the protocol's 4.1e-3, b-polynomials 2.2-2.8e-3 against 2.8-2.9e-3): the
heat checkpoint on Drive is not the protocol's operator; the run stopped before any fresh heat or KdV run.
Burgers on fresh data 2042-2044: all five generators below lambda6 on 3/3 (largest loss 1.1e-3 / 3.1e-4 / 4.3e-4
against lambda6 1.5e-3 / 6.6e-4 / 7.9e-4), inside the 5 lowest 0.95-1.00; missing 0.24 +/- 0.23 against the
Gaussian-coefficient null 2.81 +/- 0.03. (F1, F2) Burgers MET; heat and KdV pending.
Closure, escaped (null mean - 2 sd): protocol data, learned operators -- Burgers 0.236 (0.518), 0.103 (0.330), 0.401
(0.487); KdV 0.110 (0.225), 0.162 (0.249), 0.070 (0.256); heat 0.259 (0.403), 0.217 (0.486), 0.290 (0.434), heat's
full ground truth itself 0.099-0.130. Fresh Burgers 0.205 (0.512), 0.050 (0.352), 0.142 (0.556).
(K2) MET on the protocol data: below the threshold on 3/3 KdV, 3/3 Burgers, 3/3 heat (fresh: Burgers 3/3).
(K3) NOT MET: KdV learned escaped <= 0.10 on 1/3 (0.110, 0.162, 0.070); exact KdV 0.043-0.054.

**Finite flows of the class-specific directions (transfer) and the physics fresh-seed run -- code, calibration,
pre-registration.**
fins_semantic.py transfer: for each of the 24 single-class operators of the confirmatory run, the 3 lowest
eigenvectors of its contrastive pencil on the whole fresh pool (Nbar over the six design groups, as G1-G5), and 8
random directions (Gaussian coefficients scaled by the pixel metric), shared. Each direction's flow du/ds = Q_c(u)
(RK4; 8-20 steps, at most ~10 px displacement) is scaled to a first-order image change of 5 percent of the image norm;
s = +-1. Test images: 100 per design group from the design pool (600), not used for the directions. For every
operator B, the squared change of its log-odds, mean over the two signs. T[A, B] = mean squared change of B along
A's lowest directions (mean over directions and images) / the same along the random directions.
Calibration (CPU, before any flow; first order, the stored JVPs of 1 500 fresh-pool images, in-sample): the
superclass test on (log T[A, B] + log T[B, A]) / 2 gives +0.085 (ResNet-50) and +0.122 (ConvNeXt-T), both p 0.0002;
with medians of |change| instead of mean squares the effect is a third as large and ConvNeXt-T reaches only
p 0.05, so mean squares are the primary statistic. Own-class ratio T[A, A] / mean over the design groups of
T[A, g]: median 0.80 / 0.61, below 1 for 24/24. CPU smoke test (untrained ResNet-50, 32 px noise images, the full
pipeline): own-class ratio below 1 for 23/24 (median 0.58), superclass test +0.015, p 0.33. The own-class ratio
follows from the construction of the directions even in an untrained network, so it is reported, not registered.
PRE-REGISTERED (before running; Colab, ResNet-50 and ConvNeXt-T, the confirmatory npz files):
(W2) Each network: (log T[A, B] + log T[B, A]) / 2 over the 24 classes is smaller within superclasses than between
(mean between - mean within > 0), permutation p < 0.01.
Reading: W2 met on both -- the semantic structure of G4 holds for finite transformations on images not used to find
them: flowing along one class's invariant directions disturbs semantically related classes less than others. W2
not met -- the structure is first order only, stated as such. Exploratory: the median statistic, the own-class
ratio, the flows figure (tabby, golden retriever, lion, castle at 15 percent).
Physics, experiments/run_fresh.py: F1, F2 on evaluation data seeds 2042-2044 through the protocol's trained
operators. Each operator is first rerun on the protocol data (1042-1044) and must reproduce the protocol file's
eigenvalues to 1e-3; operators are loaded, never retrained. Then exp42 --dir fresh (F1, F2) and exp44 on the
protocol and fresh folders (K2, K3). Predictions as registered earlier (F1, F2; K2, K3).

**Images, third design: confirmatory run -- results.**
Colab, fins_semantic.py pencils --pool fresh --operators extended --k2 0: 6 349 images no earlier run used (cats 371,
dogs 2 978, springer, tench, garbage truck, church 750 each), 86 order-1 fields, 30 operators; ResNet-50
IMAGENET1K_V1 and ConvNeXt-T IMAGENET1K_V1 (8 544 s). The npz files reproduce the Colab summaries exactly.
(G1) ResNet-50, contrastive, corrected: cat closer to springer than to tench 10/10 (+0.60), garbage truck 10/10
(+0.39), church 10/10 (+0.62). MET.
(G2) ConvNeXt-T: tench 0/10 (-0.45), garbage truck 7/10 (+0.07), church 2/10 (-0.13). NOT MET: the cat-springer
ordering does not carry over to the second architecture.
(G3) Spearman between the readout distance and the contrastive specificity over the 15 design pairs: -0.26
(ResNet-50), -0.10 (ConvNeXt-T). MET.
(G4) Specificity of the 24 single-class operators, mean between superclasses - mean within: +0.301, p 0.0001
(ResNet-50); +0.186, p 0.0001 (ConvNeXt-T). MET.
(G5) With the readout distance regressed out: +0.128, p 0.0006; +0.130, p 0.0001. MET.
The readout distance's own superclass statistic: +0.072, +0.028 (both p 0.0001); Spearman of the specificity with a
tree distance (same superclass, both animals or both artefacts, else) +0.45, +0.47.
Reading fixed in advance: G4 and G5 met on both networks -- the class-specific invariance subspaces of trained
classifiers carry semantic structure beyond the head's geometry, and the experiment enters the paper as a positive
result. The single cat-springer ordering (G1, G2) is network-specific, as the registration anticipated.
EXPLORATORY (after the results, from the npz files; ResNet-50 / ConvNeXt-T):
  regressed on the readout distance and on pair reliability (mean and |difference| of the own split-half distances):
    +0.121, p 0.0006 / +0.137, p 0.0001. With equal superclass sizes the reliability correction cancels in the
    statistic; the raw distance gives the same values.
  mean specificity: within superclass +0.32 / +0.48, animal-animal +0.54 / +0.56, artefact-artefact +0.64 / +0.77,
    animal-artefact +0.69 / +0.74.
  every superclass is closer within than to the others (ResNet-50: domestic cats +0.29 against +0.57, dogs +0.30 /
    +0.68, big cats +0.23 / +0.51, birds +0.51 / +0.62, vehicles +0.38 / +0.54, buildings +0.23 / +0.81; ConvNeXt-T
    within +0.34..+0.67 against +0.56..+0.77); leaving any one superclass out keeps p <= 0.0002.
  sub-pools: Pet images +0.23 / +0.14, Imagenette images +0.19 / +0.15, dog images +0.21 / +0.12, tench images (750,
    none of the 24 classes) +0.07 / +0.04 (p 0.0001 / 0.003).
  subspace dimension: k = 4 +0.17 / +0.11, k = 12 +0.43 / +0.31 (all p 0.0001).
  relative pencil (feature metric) on the same images: +0.28 / +0.21 (p 0.0001). For single-class operators the
    superclass structure is not specific to the contrastive denominator; the earlier reading that the relative
    pencil's invariances are shared by every class score holds for the design groups on the design pool only.
  contrastive spectra stay shallow (ResNet-50 cat lambda_1 0.19, median 0.62).
Caveats: the structure is statistical, over 276 pairs; per pair the subspaces are noisy (own split-half 5.3-6.8 of
chance 7.26). About half of ResNet-50's effect is shared with the head geometry (+0.30 -> +0.13), a third of
ConvNeXt-T's. Finite flows along the class-specific directions were not part of the test.
Figure: vision/figures_semantic.py -> paper/rev/figures/fig_semantic_blocks.pdf.

**Images, third design: readout control -- results; confirmatory run pre-registered.**
Colab, fins_semantic.py readout (forward passes only, the pencils' 6 000 images, 21 s per network). Trained ResNet-50,
mean log-odds gradient direction with respect to the features (1 - cos^2), operator a on one half, b on the other,
10 splits: cat closer to springer than to tench 0/10 (-0.05), garbage truck 0/10 (-0.04), church 10/10 (+0.02); dog
10/10 (+0.15), 10/10 (+0.12), 10/10 (+0.14). Full-sample distances from cat: dog 0.897, tench 0.909, garbage truck
0.918, springer 0.955, church 0.978; from dog: springer 0.802 (springer's class is in the dog group). Head directions
(mean class weight of the group minus that of the rest): cat-dog 0.967, dog-springer 0.861, every other pair
0.990-1.000. Top-8 gradient subspaces: cat 0/10 for all three (corrected 5/10), dog 10/10. Untrained: margins within
+-0.02.
(R1) NOT MET: four of six; the readout puts cat nearer tench and garbage truck than springer.
(R2) NOT MET: Spearman over the 15 pairs between the readout distance and the contrastive same-image specificity +0.26
(p 0.35); order 2 +0.31; head directions +0.13. (The relative pencil's same-image specificity follows the readout
more: +0.53, p 0.04.) By the registered rule the contrastive structure is not the head's, and a confirmatory run is
justified.
EXPLORATORY (after R1, R2; stored JVPs; trained, contrastive, corrected; cat closer to springer than to tench /
garbage truck / church):
  sub-pools: Pet images 9, 10, 10/10; Imagenette images 10, 6, 10/10; tench, truck and church images 10, 6, 10/10;
    dog images 9, 10, 10/10; cat images 8, 6, 1/10. Not produced by the cat-dog competition on Pet images; garbage
    truck is the weakest comparison.
  denominator: trace-normalised Nbar 10, 10, 8/10; Nbar without the dog group 10, 10, 10/10 (+0.35, +0.27, +0.26);
    Nbar of the other five groups gives the same subspaces by construction; trace shares of the six groups in Nbar
    0.15-0.19.
  power: pools subsampled from the 6 000 with the same composition: all three >= 9/10 on 0/8 pools of 1 000 images
    (mean margins +0.06, +0.03, -0.01), 1/8 of 2 000 (+0.13, +0.09, +0.07), 2/8 of 3 000 (+0.24, +0.20, +0.12). The
    effect needs pools of about 6 000 images.
  CPU smoke test of the new code: an untrained ResNet-50 and an untrained ConvNeXt-T on 180 noise images give cat
    closer to springer (corrected) 9, 6, 8/10 and 8, 10, 5/10. An ordering among four single classes can come from
    one draw of the head; one comparison set is not evidence of semantics.
PRE-REGISTERED before running (Colab, fins_semantic.py pencils --pool fresh --operators extended --k2 0):
Pool: every Oxford-IIIT Pet and Imagenette image of the six design sources that neither the design nor the flows used
(load_groups pool="fresh"; about 371 cats, 2 978 dogs, 750 per Imagenette class, 6 350 in all). Fields: the 86
order-1 fields. Operators: the six design groups and 24 single classes in six superclasses -- domestic cat 281-284;
dog 162, 207, 235, 254; big cat 289, 291-293; bird 9, 11, 15, 22; vehicle 407, 555, 779, 817; building 483, 663,
668, 698 -- with no images of their own. Contrastive pencil as in the exploratory analysis: Nbar the mean of the six
design groups' log-odds Grams on the same images, k = 8, distances in the pooled Nbar metric, all images pooled,
halves, 10 splits, specificity = distance - mean own split-half distance. Networks: ResNet-50 IMAGENET1K_V1 and
ConvNeXt-T IMAGENET1K_V1. Readout: mean gradient of each operator's log-odds with respect to the features over the
pool, distance 1 - cos^2.
(G1) ResNet-50: cat closer to springer than to tench, garbage truck and church, each on >= 9/10 splits.
(G2) ConvNeXt-T: the same.
(G3) Each network: Spearman over the 15 design pairs between the readout distance and the contrastive specificity
< 0.7.
(G4) Each network: the 24 single-class operators' specificity is smaller within superclasses than between (mean
between - mean within > 0), permutation p < 0.01 (10 000 reassignments of the classes to superclasses of four).
(G5) Each network: G4 on the specificity with the readout distance regressed out (linear, over the 276 pairs),
p < 0.05.
Reading fixed in advance: G4 and G5 met on both networks -- the class-specific invariances of trained classifiers
carry semantic structure beyond the head's geometry, and the experiment enters the paper as a positive result; G4
met, G5 not -- the structure is the head's; G4 not met -- the semantic experiment closes as negative. G1 and G2 test
only whether the original ordering reproduces across images and networks; alone they are not evidence of semantics.
Exploratory: dog comparisons, correlation with a tree distance, the readout's own superclass statistic.

**Images, third design: F8; comparisons corrected for reliability; contrastive pencil (exploratory); readout control
pre-registered.**
The untrained network's npz (Colab, same 6 000 images, 6 908 s). fins_semantic.py analyse (this version prints the
same-image comparisons raw and corrected, and the contrastive pencil) reproduces the Colab summaries of both networks.
The untrained logits are saturated: own log-odds median -15.4 (dog) .. -57.3 (church), positive on 0.00 of the
images; LSE is the maximum, and each readout is one class logit minus the top logit.
(F8) Untrained, relative, order 1, all 6 000 images (halves of 3 000, 10 splits), Pet group closer to springer than
to: cat tench 10/10 (+1.35), garbage truck 0/10 (-0.91), church 10/10 (+2.03); dog 10/10 (+1.28), 0/10 (-0.77),
10/10 (+2.04). Trained: 10/10 for all six (+0.44..+0.86). NOT MET: four of six at 10/10.
The untrained ordering is the readouts' saturation: the distance from the cat and dog operators to a single-class
operator grows with that class's median |log-odds| on all images -- garbage truck (34.7) 4.86 / 4.91, springer (37.8)
5.78 / 5.68, tench (47.5) 7.13 / 6.96, church (56.9) 7.81 / 7.72.
EXPLORATORY (after the results): reliability. On the pooled images the operators' own split-half distances differ
(trained: springer 4.27, dog 4.60, cat 5.03, garbage truck 5.28, tench 5.33, church 5.43), and a more reliable
subspace is closer to every other: by the raw distance every trained operator, tench, garbage truck and church
included, is nearest to springer's. Specificity = distance - mean of the two operators' own split-half distances:
  trained, same images (all): +0.03..+0.42 for all 15 pairs (largest: church's); cat closer to springer than to tench
    3/10 (-0.10), garbage truck 7/10 (+0.03), church 6/10 (+0.08); dog 8/10 (+0.03), 6/10 (+0.04), 9/10 (+0.28). On
    the same images the six class operators of the trained network share their lowest subspaces up to sampling noise.
    The reading of the same-image comparison in the trained-results entry below ("the relation is between the class
    readouts ... not produced by the image sets themselves") is withdrawn: it was the reliability of springer's
    operator.
  per-class images (F1 design, halves of 300, 20 splits): trained cat 20/20 (+0.47), 19/20 (+0.45), 17/20 (+0.16),
    dog 20/20 (+0.67), 20/20 (+0.57), 19/20 (+0.38); untrained cat 18/20 (+0.18), 17/20 (+0.20), 20/20 (+0.53), dog
    17/20 (+0.12), 18/20 (+0.22), 20/20 (+0.57). Corrected, the F1 ordering holds in the untrained network too: it
    follows the image sets (photographs of animals), not training.
Hypothesis for the shared bottom: the relative pencil divides by the change of all 2 048 features, of which a class
score reads one direction per image; fields whose feature changes miss every readout direction have small
eigenvalues for every class.
EXPLORATORY: contrastive pencil N_S c = lambda Nbar c, Nbar the mean of the six log-odds Grams on the same images (the
class score changes less than the average class score); distances in the pooled Nbar metric (random 7.26).
  trained, own images: lambda_1 0.34 (cat), 0.54 (dog), 0.14-0.22 (the others); lambda_1 / median 0.31-0.71.
  trained, same images (all): own split-half 5.98-6.78; specificity cat-dog +0.20, cat-springer +0.27, dog-springer
    +0.14, other pairs +0.45..+1.00; cat closer to springer than to tench, garbage truck, church 10/10 (+0.35, +0.22,
    +0.19), dog 10/10 (+0.82, +0.68, +0.86); order 2 10/10 (+0.37..+0.94).
  untrained: cat 10/10 (+0.59), 0/10 (-0.43), 10/10 (+0.68); dog 8/10, 7/10, 10/10; cat and dog lambda_1 0.03-0.04,
    photometric 0.98.
  Dog's group contains springer's class (217 in 151-268), so the dog-springer comparisons are partly built in; the cat
  comparisons are the clean ones. Shallow, poorly reliable, chosen after the data.
VERDICTS of the third design: F1 met as registered, but the ordering is reproduced by the untrained network once
reliability is accounted for and vanishes when the operators are compared on the same images; F2 not met; F3 met;
F4 not met; F5 met; F6 not met; F8 not met.
Reading: the trained ResNet-50 has no cat-specific symmetry in this field class at first or second order. Its
class-conditional invariances, measured against its own features, are invariances of the network (edits of the
image periphery) shared by every class score; they change with the images, not with the class.
PRE-REGISTERED before running (fins_semantic.py readout, Colab, the pencils' 6 000 images, forward passes only):
(R1) Trained, all images pooled, distance 1 - cos^2 between the groups' mean log-odds gradients with respect to the
penultimate features (operator a on one half, b on the other, 10 splits): cat and dog each closer to springer than to
tench, garbage truck and church -- six of six on >= 9/10 splits.
(R2) Spearman correlation over the 15 pairs between that distance (full sample) and the trained contrastive
same-image specificity (order 1, above) >= 0.7.
R1 and R2 met: the relational structure is the classifier head's, and the semantic experiment closes as a negative
result. R2 not met: a confirmatory contrastive run on fresh images and a second network is justified. Untrained: no
prediction.

**Images, third design: untrained ResNet-50 and flows -- results.**
Untrained ResNet-50 (same images, summary only; run with the version before the same-image comparison).
Relative pencil, order 1: reliability higher than in the trained network (Pet halves of 900: within cat 3.49,
dog 3.41 against 6.43, 5.15); cat-dog specificity +1.27 (20/20) against +0.24. Semantic test (Pet group closer to
springer than to ...): cat tench 20/20, garbage truck 0/20 (-0.48), church 20/20; dog tench 20/20, garbage truck 0/20
(-0.46), church 20/20. Full-sample distances: cat-dog 4.06, cat and dog to garbage truck 5.8, to springer 6.6-6.7,
to tench and church 7.2-7.4. Plain pencil: no ordering (1-12/20). Order 2 as order 1.
Flows on 200 held-out Pet images, ResNet-50 trained, first-order image change 5 percent of the image norm; ratio
= median |change of log-odds| / median relative feature change:
  cat, order 1: lowest 3 ratio 2.10, random 2.45 (lowest / random 0.86), highest / random 1.22; at equal image change
                the lowest change the cat log-odds 0.67 times as much as random (0.103 against 0.153)
  cat, order 2: 0.81, highest / random 1.67; absolute 0.46
  dog, order 1: 0.75, highest / random 1.18; absolute 0.53
The lowest directions edit the image borders (local brightness and small deformations near the corners); the
random ones change colour and brightness globally.
VERDICTS. (F2) NOT MET: the church and tench orderings hold in the untrained network as well (20/20); only the
garbage-truck comparison separates the networks (trained 18/20, 20/20; untrained 0/20, 0/20). (F3) MET: specificity
+0.24 trained, +1.27 untrained. (F6) NOT MET: cats 0.86 > 0.8, highest / random 1.22 < 2. (F8) NOT MET (entry
above).
EXPLORATORY, noise-corrected (specificity = between - mean within per pair, halves of 300, 20 splits; relative,
order 1): trained cat-dog +0.06, cat-springer +0.15, dog-springer +0.08 -- the three mammals' subspaces as close to
each other as two halves of one class -- while cat and dog stay apart from tench, garbage truck, church
(+0.31..+0.74) and tench-garbage truck +1.29. Untrained: cat-dog +0.56, cat-springer +0.86, dog-springer +0.86, all
pairs +0.42..+1.43. In the trained network only the related pairs reach the noise level; in the untrained one no
pair does. The reduction itself is not specific to related pairs (cat-church -1.08, dog-church -0.97 against
cat-springer -0.71, dog-springer -0.79): it is the level, not the change, that singles them out. Not pre-registered.
Reading: the classifier's class-conditional first-order invariances, measured in its own representation, are
edits of the image periphery, not object-level transformations; per class they are shallow (lambda_1 / median
0.2-0.4), unreliable between halves and only modestly better than random at a finite step. What training adds is
relational: the invariances of cat, dog and an English springer coincide within sampling noise, unlike in the
untrained network, and the garbage-truck comparison reverses. The pre-registered semantic contrast (F2) failed for
church and tench because the untrained network already separates those classes from everything. [Revised in the F8
entry above: corrected for reliability, the ordering holds in the untrained network too, and on the same images the
trained operators are indistinguishable.]

**Images, third design: trained ResNet-50 -- results; untrained network and flows pending.**
Colab, fins_semantic.py pencils, ResNet-50 IMAGENET1K_V1, 224 px; Pet cats 1 800, dogs 1 800 (pencils), Imagenette
springer, tench, garbage truck, church 600 each; 98 fields. The untrained network's file and the flows were not yet
available. Group membership on own images (top-1 inside the group): cat 0.85, dog 0.97, springer 0.72, tench 0.95,
garbage truck 0.87, church 0.38 (own log-odds median +2.08, +6.73, +0.38, +5.59, +2.64, -1.23).
Relative pencil, order 1 (86 fields, k = 8, random 7.26):
  lambda_1 / median 0.20-0.43 (shallow bottom); cat subspace 0.83 geometric, 0.17 photometric; no named global
  field (translations, rotation, scaling, stretch, shear, brightness, contrast) lies in any group's 8 lowest
  (share outside 0.89-0.99, chance 0.95): the lowest fields are local deformations and photometric edits
  concentrated at the borders and corners of the image.
  reliability, Pet halves of 900: within cat 6.43 +/- 0.13, dog 5.15 +/- 0.13; cat-dog specificity +0.24 (18/20);
  all-group halves of 300: within 5.86 (tench) - 6.95 (cat).
  full-sample distances: dog-springer 4.72, cat-dog 5.52, cat-springer 5.85; tench 6.13-6.48 from the three
  mammals; garbage truck and church 6.39-7.24 from everything (truck-church 7.24, chance 7.26).
  semantic test (Pet group closer to springer than to ...): cat: tench 20/20 (+0.30), garbage truck 18/20 (+0.28),
  church 19/20 (+0.47); dog: tench 20/20 (+0.49), garbage truck 20/20 (+0.41), church 20/20 (+0.68).
Plain pencil, order 1: within cat 3.06, dog 2.13 (halves 900); subspaces 0.90 photometric; semantic test: cat
  truck 18/20, church 3/20 (-0.11); dog truck 20/20, church 5/20 (-0.08).
Order 2 (98 fields): order-2 span takes 0.08 of the cat subspace (dog 0.03, church 0.10); semantic test cat:
  tench 20/20, truck 16/20 (+0.22), church 20/20 (+0.40); dog: 19/20, 20/20, 20/20; mean margin 0.37 (order 1 0.46).
VERDICTS. (F1) MET: 18, 19, 20, 20 of 20. (F4) NOT MET: cat-truck 16/20, mean margin smaller (0.37 < 0.46), order-2
share 0.08 < 0.15 -- the second-order fields add nothing here. (F5) MET: in the pixel metric church breaks the
ordering (3/20, 5/20). (F2, F3, F6) pending (untrained network, flows). (F7) tench: no closer to the Pet groups
than garbage truck (cat 7.03 vs 7.01, dog 6.96 vs 6.87 at halves of 300): the grouping is cat-dog-springer, not
"animals".
EXPLORATORY (after the results, CPU, from the stored JVPs): the same comparison with every operator evaluated on
the same images, operator a on one half of a pool and operator b on the other (10 splits). All 6 000 images:
cat closer to springer than to tench, garbage truck, church on 10/10 (+0.44, +0.53, +0.67), dog 10/10 (+0.56,
+0.55, +0.86); Pet cats only: 8-10/10 (+0.13..+0.34); Pet dogs only: 8-10/10 (+0.14..+0.57); garbage-truck images
only: 1-4/10 (reversed, -0.03..-0.22). The relation is between the class readouts, visible on animal images and
on the pooled set, not produced by the image sets themselves; on truck images it disappears. [Withdrawn in the F8
entry: the raw distance favours the most reliable operator, springer's.]
PRE-REGISTERED before the untrained run: (F8) untrained ResNet-50, relative, order 1, all images pooled: each of
the six "springer closer" comparisons on <= 5/10 splits. fins_semantic.py analyse now prints this comparison.

**Images, third design: class-conditional symmetries of a frozen classifier ("symmetries of cats") -- code,
CPU pilot, pre-registration.**
Idea (the original LieSym motivation): a classifier's learned notion of a class has symmetries -- point
transformations that leave "this is a cat" unchanged. vision/fins_semantic.py. Operator: the group log-odds
f_S(u) = LSE(z_S) - LSE(z_rest) of the logits (cats: ImageNet 281-285; dogs: 151-268; Imagenette classes by
their ImageNet index); Olver's criterion v[f_S] = J Q_v = 0 on images of S. Fields, linear in the coefficients
(98): order 1 = Lie point fields on (x, y, u): xi, eta in tensor cubic B-splines (5 x 5, local; the span contains
every cubic, so rotation, scaling, projective), xi, eta = luminance L(u) times bilinear B-splines (2 x 2),
local grey offset and local intensity in quadratic B-splines (3 x 3), chroma offsets (2) and traceless channel
mixings (8); order 2 = generalised symmetries, bilinear B-splines times u_xx, u_xy, u_yy (Gaussian derivatives,
1 px) -- 12. Order-1 results are the order-2 pencils restricted to the 86 order-1 fields. One JVP per field per
image gives the JVPs of every group's log-odds (stored per image, so any image subset gives N_S) and of the
penultimate features. Pencils: relative N_S c = lambda M c, M the pooled feature Gram (change of the class score
per unit of change of the network's representation; a direction invisible to the network does not count as a
symmetry), and plain N_S c = lambda D c (pixel metric). Scores, k = 8 lowest, distances = missing dimensions in
the denominator's metric (random k-subspace: 7.26 of 8 at order 1, 7.35 at order 2): split-half reliability
(within-group distance of two disjoint halves), class specificity (between-group distance of halves minus the
mean within), and a semantic test (is a Pet group's subspace closer to that of another animal than to those of
man-made objects). Finite flows: du/ds = Q_c(u) integrated by RK4 (the evolutionary form of the point
transformation; for the order-2 fields an anisotropic diffusion), scaled to a first-order image change of 5
percent of the image norm, on held-out images; score = median |change of log-odds| / median relative change of
the features.
Checks (CPU): stored JVPs equal autograd gradients times characteristics (-5.6899993e-05 both); the RK4 flow of the
rotation field written in the B-spline basis matches the exact rotation to 1.2 percent (sign: the flow at +s is
the warp at -s), the isotropic order-2 field matches a Gaussian blur of variance 2 a s to 0.2 percent; B-splines
reproduce x^3 to 7e-16; the combined characteristic equals the sum over fields to 1e-7.
PILOT (CPU; development, seen before the registration below): CIFAR-10 CNN trained on the 50 000
training images (JPEG copies from github.com/YoongiKim/CIFAR-10-images, 10 epochs, test accuracy 0.66) and the
same architecture untrained; 500 test images each of cat, dog, deer, truck (halves of 250); 20 splits.
  relative, order 1:   within (trained) cat 6.07 dog 5.78 deer 5.59 truck 4.64; (untrained) 1.75 2.27 3.00 1.64
                       cat-dog specificity trained +0.32 (20/20), untrained +2.42 (20/20)
  semantic (cat or dog closer to deer than to truck): trained, relative, order 1: cat 17/20 (+0.15), dog 20/20
                       (+0.59); order 2: 20/20 (+0.30), 20/20 (+0.65); untrained: 0/20 for both, reversed (-0.66..-0.93);
                       plain pencil, trained: cat 1/20, dog 14/20
  composition, trained cat, relative: order 1 geometric 0.43 / photometric 0.65 of the subspace; order 2 adds
                       0.26 in the order-2 span; untrained: photometric 0.98
  flows, trained, 200 held-out test images: lowest-3 / random ratio cat 0.82 (order 1), 0.76 (order 2), dog 0.62;
                       highest / random 3.2-4.3
Reading of the pilot: class specificity by itself is not semantics -- different readouts of an untrained network
have more distinct and more reliable invariance subspaces than those of the trained one; what distinguishes the
trained network is that the invariances of two animal classes are closer to each other than to those of a
vehicle, in the network's own metric, and reversed in the untrained network. The bottom of the trained
spectrum is shallow (lambda_1 / median 0.14-0.17) and the lowest directions keep the class score only modestly
better than random ones at a finite step; the top direction changes it 3-4 times more than random ones.
PRE-REGISTERED (Colab: fins_semantic.py pencils, ResNet-50 IMAGENET1K_V1 and the same architecture untrained;
Oxford-IIIT Pet cats and dogs, trainval and test pooled, 1 800 per species for the pencils (halves of 900) and 200
per species held out for flows; Imagenette train and val pooled, 600 each of English springer, tench, garbage truck,
church (all-group halves of 300); 224 px; k = 8; 20 splits):
(F1) Trained, relative, order 1: cat and dog each closer to springer than to garbage truck and than to church --
each of the four comparisons on >= 17/20 splits.
(F2) Untrained, relative, order 1: each of the four comparisons on <= 10/20 splits.
(F3) The trained network's cat-dog specificity (Pet halves) is not above the untrained one's.
(F4) Order 2, trained, relative: F1 holds; the mean margin over the four comparisons is at least the order-1 mean
margin; the order-2 span takes >= 0.15 of the cat subspace.
(F5) Plain pencil, trained, order 1: at least two of the four comparisons on <= 15/20 splits.
(F6) Flows, trained, order 1, 200 held-out Pet cats: lowest-3 / random ratio <= 0.8; highest / random >= 2.
(F7) Tench (an animal, not a mammal): no prediction.

**Images: the smoothed criterion -- results.**
Colab, log-probabilities, seeds 100-102, 8 noise draws per image; nets retrained (sigma = 0 rows reproduce the
earlier runs within seed spread: CIFAR-10 MLP rotation 0.18-0.19, projective now 0.64-0.70 against
0.48-0.69). Share of the planted generator outside the 3 lowest, trained / control, and loss ratio
control / planted, at sigma 0 -> 0.2:
  CIFAR-10 CNN rotation    0.69 0.76 0.96 / 0.77-0.83, x1.1-1.6  ->  0.65 0.68 0.68 / 0.71-0.76, x0.83 1.22 1.14
  CIFAR-10 CNN intensity   0.13-0.14 / 0.26-0.33               ->  0.51 1.00 0.69 / 0.87-1.00
  CIFAR-10 CNN grey offset 0.04-0.05                           ->  0.06-0.08 (control 0.86-1.00)
  CIFAR-10 MLP rotation    0.18-0.19 -> 0.15-0.19; grey offset 0.06-0.09; intensity 0.23-0.28 (unchanged)
  CIFAR-10 saturation, hue 0.99-1.00 at every sigma, both networks
  F-MNIST CNN rotation 0.44-0.50 -> 0.33-0.53; projective 0.26-0.62 -> 0.32-0.82; scaling 0.73-0.86 -> 0.51-0.83;
  intensity 0.41-0.87 -> 0.95-1.00. F-MNIST MLP: every share within 0.03 of sigma 0.
Encoders, missing of DINOv2's 7 / of ResNet-50's 4 (random 4.96 / 3.33), sigma 0, 0.1, 0.2:
  DINOv2            3.25 3.94 3.54  /  4.00 3.02 2.52
  ResNet-50 V1      4.22 4.73 4.63  /  3.38 2.96 2.85
  ViT-S untrained   5.71 6.29 6.02  /  3.69 3.85 3.79
DINOv2's saturation and hue fall from rank 4 to 19-22 and its intensity to 15 at sigma 0.2; its translations,
scaling and stretch rise from 10-11 to 7-9 (shares outside the 4 lowest 0.74-0.91), ResNet-50's scaling and
stretch from 9 to 7; the untrained ViT does not move.
VERDICTS. (S1) NOT MET: CIFAR-10 CNN rotation 0.65-0.68 at sigma 0.2, 0.05-0.08 below the control. (S2) NOT MET:
ratio 0.83-1.22. (S3) NOT MET: CIFAR-10 CNN intensity lost (0.51-1.00); the other recovered generators stay
(MLP rotation, grey offset, intensity; CNN grey offset; F-MNIST I2 and I5 cases on >= 2/3). (S4) MET. (S5) NOT
MET as stated: DINOv2 misses more of its 7 (3.54 > 3.25); ResNet-50 fewer of its 4 (2.85, now below chance);
untrained ViT >= 4.0.
Reading: smoothing does not repair the CIFAR-10 CNN, whose rotation robustness is not invariance of its
smoothed version either. It trades photometric for geometric invariance: pixel noise of 0.1-0.2 takes the
operator off the image distribution, where amplitude and colour invariances (intensity under log-probabilities
rests on near-homogeneity, which additive noise breaks) no longer hold, while a small part of the crop-induced
geometric invariance of both trained encoders appears (DINOv2's geometric subspace 4.00 -> 2.52 missing,
chance 3.33), not in the untrained one. Not adopted; sigma = 0 remains the method; reported as a negative
result.

**Images: the smoothed criterion -- code, pre-registration.**
Motivation: the CIFAR-10 CNN is robust to its planted rotation at finite angles but not to first order at
the identity (entry below), which the criterion measures. Change (fins_image.py --sigmas, --samples): the
operator f is replaced by its Gaussian smoothing f_sigma(u) = E f(u + e), e ~ N(0, sigma^2 I) in pixel units
of [0, 1]; J is estimated by the mean of J(u + e_m) Q over 8 fixed draws per image, Q stays the
characteristic of the clean image. The criterion measures invariance at the scale sigma and stays quadratic
in the coefficients, so the eigenproblem is unchanged; sigma = 0 is the previous method exactly. CPU check
before this registration (one CIFAR-10 CNN pair, 8 000 images, 64 probes): the control / planted loss
ratio of rotation rises 1.28, 1.54, 1.78, 2.46 at sigma 0, 0.05, 0.1, 0.2, but rotation stays at rank 8
(share outside the 3 lowest 0.87). sigma grid 0, 0.05, 0.1, 0.2; sigma = 0.2 is the primary scale, chosen
after that check.
PRE-REGISTERED (Colab: planted CIFAR-10 and Fashion-MNIST as before, log-probabilities, seeds 100-102;
encoders as before with sigma 0, 0.1, 0.2):
(S1) CIFAR-10 CNN, sigma 0.2: planted rotation within the 3 lowest (share <= 0.6) on >= 2/3 seeds and >= 0.2
below the control at the same sigma. Held unlikely (CPU: 0.87).
(S2) CIFAR-10 CNN rotation: loss ratio control / planted >= 2 at sigma 0.2 on >= 2/3 seeds and above its
sigma = 0 value on 3/3.
(S3) No harm: every generator recovered at sigma 0 (share <= 0.6; CIFAR-10 MLP rotation, projective, grey
offset, intensity; CNN grey offset, intensity; Fashion-MNIST as in I2, I5) stays <= 0.6 at sigma 0.2 on
>= 2/3 seeds.
(S4) CIFAR-10 saturation and hue stay unrecovered (> 0.9) at every sigma.
(S5) Encoders at sigma 0.2: DINOv2 misses fewer of its 7 than at sigma 0 (3.25), ResNet-50 fewer of its 4
than at sigma 0 (3.37); the untrained ViT still misses >= 4.0 of DINOv2's 7. Held uncertain.

**Figures of the image section (vision/figures_image.py).** planted: the planted generator and its
projection on the 3 lowest eigenvectors of the network trained with it and of the control, on one probe
image (arrows xi, eta; background the characteristic; Fashion-MNIST MLP, logp, s100, from the saved
pencils). encoders: rank of each named generator per encoder (drawn from the ranks printed in the
Colab log; figures_image.py encoders --json encoders.json reproduces it from the saved pencils).

**Closure of the discovered span (exp44) -- code, check, pre-registration; fresh-seed confirmation F1-F3 --
commands.**
exp44_closure.py: FINS's generators are polynomials in X = (x - x0) / Lw, T = t / T_max, U = u / sigma, so
brackets are exact polynomial computations (checked against sympy: error 0). With e_a an orthonormal basis
of the span on exp42's region and scaling, escaped = (sum ||(I - P_V)[e_a, e_b]||^2 / sum ||[e_a, e_b]||^2)^1/2,
basis-free, 0 when closed; structure = (sum ||[e_a, e_b]||^2)^1/2 flags an abelian span. References per seed:
the ground-truth algebra fitted to the class off-shell (random u; relative residual 1.8e-15 on KdV) and
random k-subspaces of the class (Gaussian coefficients, as exp42's null). Development check (CPU, exact
operators, data 1042, seen before the registration below): KdV ground truth 0.000, exact operator 0.043,
null 0.381 +/- 0.057; heat finite ground truth 0.000, heat full class (ten, the superpositions truncated at
degree 3, so not closed) 0.099, exact operator 0.099, null 0.448 +/- 0.025.
The superpositions of heat are an infinite ideal; its degree-3 truncation is not closed, so heat's
reference is the ground truth's own value, not 0.
PRE-REGISTERED (exp44 on the protocol files, s = 42-44, and on the fresh files below):
(K1) Exact operators: KdV escaped <= 0.06, heat within 0.02 of the full ground truth's, every seed.
(K2) Learned operators: FINS escaped below the class null's mean minus 2 sd on every KdV seed and on
>= 2/3 seeds of Burgers and of heat.
(K3) KdV, learned operator: escaped <= 0.10 on 3/3.
FRESH SEEDS (F1-F3, registered above): same operators, evaluation data seeds 2042-2044, files in
FINS_RESULTS_DIR/fresh, exp42 --dir on it; exact rows computed on CPU (exp39 / exp40, identical code).
RESULTS, exact operators (CPU). F3 on fresh data 2042-2044 (exp42): KdV missing 0.01,
spurious 0.01, novel containment 0.99-1.00, margins 490-507x; heat full class missing 0.00, spurious
0.00, novel 1.00, margins 18 000-28 000x -- MET. Closure, escaped (null mean +/- sd):
  KdV exact    protocol 0.043 0.043 0.054 (0.39-0.44 +/- 0.08)   fresh 0.046 0.069 0.044 (0.38-0.44)
  heat exact   protocol 0.099 0.102 0.130 = ground truth full     fresh 0.102 0.102 0.102 = ground truth
  (heat finite ground truth 0.000 everywhere; heat null 0.45-0.53 +/- 0.02-0.03)
(K1) heat MET 6/6 (equal to the ground truth to three digits); KdV MET on the protocol data (3/3),
NOT MET on fresh s43 (0.069 > 0.06). The exact KdV span leaves 4-7 percent of its brackets outside:
its scaling member carries a loss of 2e-5, a small admixture that the bracket's derivatives amplify.
K2, K3 wait for the learned files.

**Images: loss ratios on CIFAR-10, E4, and finite against first-order invariance of the CNN.**
image_summary.py on the Colab JSONs. Ratio of the planted generator's Rayleigh quotient, control / planted,
log-probabilities, seeds 100-102: MLP brightness 4.0-6.2, rotation 2.9-4.1, projective 2.2-2.9, scaling
2.1-2.5, intensity 1.7-2.3, saturation 1.5-1.8, hue 3.4-4.7; CNN brightness 9.4-17, intensity 2.7-5.2, hue
2.7-3.3, saturation 1.4-1.8, rotation 0.98-1.30, projective 0.80-1.14, scaling 0.80-1.04. The MLP's
rotation (x2.9-4.1) is recovered and its hue (x3.4-4.7) is not: what decides is whether the loss falls below
the network's lowest levels, and hue starts about fifty times above rotation. The CNN's geometric
augmentations did not lower their generators' first-order loss at all.
Encoders, missing of each expected subspace in the k lowest (random): DINOv2 3.25 of its 7 (4.96), 4.00 of
ResNet-50's 4 (3.33); ResNet-50 4.23 of DINOv2's 7, 3.37 of its 4; untrained ViT 5.71 of DINOv2's 7, 3.69 of
ResNet-50's 4. VERDICT (E4): MET, 5.71 >= 4.0.
CNN ROTATION, CPU diagnostic after the fact (CIFAR-10 test images as JPEG, 8 000 training, 2 000 test, 300
probes, 10 epochs, seed 100): the augmented CNN is robust at finite angles -- accuracy 0.405 / 0.405 / 0.378
at 0 / 15 / 30 deg (control 0.434 / 0.364 / 0.271); mean squared change of log-probabilities 0.56 / 1.36 /
7.07 at 0.02 / 0.1 / 0.5 rad (control 1.08 / 4.40 / 21.4) -- but not to first order: the rotation
generator's quotient falls 1.35x with central differences (2.23e-2 -> 1.65e-2, rank 9 -> 7, share outside
the 3 lowest 0.99 -> 0.95), 1.32-1.45x with derivatives of the image smoothed at sigma 0.7-1.5 px, and
1.11-1.37x with the secant of the augmentation at 0.05-0.25 rad in place of the characteristic (linearised
at the identity). At 0.02 rad the finite change (1.08 in the control) is 57x what the first-order term
predicts: the bilinear resampling that any s != 0 introduces dominates small angles. The network learned
invariance at the scale of the augmentation, not at the identity; an infinitesimal criterion cannot see
it. Stated as a limitation in Sec. 6 and the Discussion; no change to the method.

**Images, second design: planted symmetries on CIFAR-10 and pretrained encoders -- results; heat seeds 43,
44 rerun with checksums.**
HEAT (exp43, Colab): the three operators are distinct, parameter checksums 4831.577 (s42), 4754.430 (s43),
4756.563 (s44); eta = 0.0702509, 0.0402496, 0.0407572; eta_V = 0.0627847, 0.0365696, 0.0365709. s42
reproduces its earlier run. The equal eta_V of s43 and s44 (FLAG below) is a coincidence to four digits.
CIFAR-10 PLANTED (Colab, fins_image.py planted --data cifar10, 50 000 training images, 15 epochs, seeds
100-102, 500 probes, 24 fields, chance outside the 1 / 3 lowest 0.98 / 0.94). Clean accuracy MLP 0.48-0.53,
CNN 0.66-0.72. Share of the planted generator outside the 3 lowest, log-probabilities, per seed; control =
the same architecture trained without augmentation:
  MLP rotation     0.20 0.57 0.13    control 0.61 0.93 0.68    ranks 2 3 2 (control 5)
  MLP scaling      0.99 0.99 1.00    control 0.99 1.00 0.99    ranks 9-10 (11-12)
  MLP projective   0.69 0.48 0.53    control 0.99 0.79 0.98    ranks 6 5 4 (9)
  MLP grey offset  0.08 0.08 0.09    control 0.25 0.32 0.27    ranks 2 (3-6)
  MLP intensity    0.23 0.26 0.27    control 0.31 0.32 0.32    ranks 5 3 5 (10)
  MLP saturation   1.00 0.99 0.99    control 1.00 1.00 1.00    ranks 21 21 20 (21-22)
  MLP hue          1.00 1.00 1.00    control 1.00 1.00 1.00    ranks 20 19 19 (23)
  CNN rotation     0.75 0.74 0.89    control 0.79 0.86 0.84    ranks 7 9 8 (8-9)
  CNN scaling      0.88 0.96 0.99    control 0.98 0.94 0.95    ranks 10-12 (11-12)
  CNN projective   0.78 0.80 0.79    control 0.83 0.78 0.81    ranks 7 (7-8)
  CNN grey offset  0.05 0.05 0.05    control 0.23 0.26 0.30    ranks 2 (2-4)
  CNN intensity    0.14 0.18 0.15    control 0.36 0.28 0.32    ranks 2 (5-8)
  CNN saturation   1.00 1.00 1.00    control 1.00 1.00 1.00    ranks 17-19 (19-21)
  CNN hue          1.00 1.00 1.00    control 1.00 1.00 1.00    ranks 18-19 (22)
L2-normalised logits: intensity the lowest eigenvector of every network (cos^2 1.00, planted or not);
MLP rotation 0.22 0.34 0.14 (control 0.96 0.92 0.75). In GL(2), share of the planted generator outside the
lowest eigenvector of FINS against LieGG's generator (controls in brackets):
  MLP rotation     FINS 0.02 0.11 0.06 (0.41 0.37 0.46)    LieGG 0.34 0.18 0.34 (0.47 0.75 0.60)
  MLP scaling      FINS 1.00 1.00 1.00                      LieGG 1.00 0.98 1.00
  CNN rotation     FINS 0.84 0.61 0.77 (0.89 0.84 0.68)    LieGG 0.67 0.74 0.75 (0.70 0.65 0.66)
  CNN scaling      FINS 0.99 0.92 1.00                      LieGG 0.99 0.93 0.99
VERDICTS. (C1) NOT MET as stated: rotation 3/3 (margins 0.36-0.55), projective 2/3 (s100 0.69), grey offset
within the 3 lowest on 3/3 but at least 0.2 below the control on 1/3 only (margins 0.17, 0.24, 0.18; the
control already holds it at 0.25-0.32). (C2) NOT MET: CNN rotation and projective at the control's level.
(C3) NOT MET: saturation and hue 0.99-1.00 in both architectures. (C4) NOT MET: FINS closer on MLP rotation
3/3; CNN rotation 1/3; scaling recovered by neither (0.92-1.00). (C5) NOT MET: MLP intensity recovered,
0.23-0.27 (control 0.31-0.32).
SATURATION AND HUE, post-hoc diagnostic (CPU; CIFAR-10 test images as JPEG from
github.com/YoongiKim/CIFAR-10-images, MLP, 8 000 training / 300 probe images, 8 epochs, seed 100): squared
norm of the characteristic per image: saturation 19, hue 17, intensity 917, rotation 2 110, grey offset
3 070 -- CIFAR-10 is weakly saturated, so these flows move few pixels. Rayleigh quotient per unit of that
norm in the control: saturation 0.92, hue 0.99, colour cast 0.60, intensity 0.085, rotation 0.020, grey
offset 0.013. Planted training lowers its own quotient by 1.16 (saturation), 2.3 (hue), 2.5 (rotation);
finite change of the log-probabilities at s = 0.25: saturation 1.43 -> 1.21, hue 0.94 -> 0.41, rotation
17.2 -> 1.7. The networks respond to chroma 50 times more strongly per unit of pixel change than to
rotation, and the planted augmentation lowers that by at most 2.3 times, so saturation and hue stay among
the least invariant directions: the classifiers did not become invariant enough to be found, not a code
error (augmentation and generator agree to cosine 1.0000, unit_fi.py). A colour-whitened normaliser
(inverse channel covariance in D) did not help (planted hue rank 6 against the control's 12, saturation
16-20, rotation lost) and is not adopted. vision/image_summary.py now prints the quotient ratio control /
planted per seed from the saved pencils.
ENCODERS (Colab, fins_image.py encoder, 512 Imagenette validation images at 224 px, 24 fields):
  DINOv2 ViT-S/14   lowest 1.5e-6, 1.5e-5, 1.8e-5 (median 6.1e-5); expected 7: missing 3.25 (random 4.96).
                    rank / share outside the 7 lowest: grey offset 2 / 0.02, intensity 2 / 0.01,
                    saturation 4 / 0.22, hue 4 / 0.21; stretch 10 / 0.79, scaling 10 / 0.91, tx 10 / 0.97,
                    ty 11 / 0.88, rotation 11; colour offsets and cast 20-22. The 8 lowest eigenvectors are
                    channel mixings.
  ResNet-50 V1      lowest 1.8e-5 (median 6.3e-5); expected 4: missing 3.37 (random 3.33). Grey offset 2 /
                    0.16, intensity 2 / 0.37; geometry ranks 9-11; saturation 18, hue 22.
  random ViT-S/14   lowest 7.4e-6 (median 1.3e-5), flat; the lowest eigenvectors quadratic geometric fields;
                    rotation 4; saturation 20, intensity, grey offset, hue 22.
VERDICTS. (E1) MET: 3.25 <= 3.5. (E2) NOT MET: 3.37, at chance. (E3) MET: saturation and hue at rank 4 / 4
in DINOv2 against 18 / 22 in ResNet-50. (E4) pending: computed from encoders.json by image_summary.py.
Reading: through the same eigenproblem, a planted Lie point symmetry is recovered where training makes the
output invariant to it by a large factor (Fashion-MNIST: rotation, projective, scaling and brightness in
the MLP, rotation and projective in the CNN; CIFAR-10: rotation, projective and the amplitude generators in
the MLP, amplitude generators in the CNN) and missed where it does not (CIFAR-10 CNN geometry, saturation
and hue). In DINOv2 the photometric part of its training augmentation -- intensity, saturation, hue, which
its colour jitter and random greyscale train strongly -- occupies the lowest eigenvectors, and neither
ResNet-50 V1 nor the same architecture untrained shows it; the geometric part induced by random resized
crops is not recovered in either trained encoder.

**Images, second design: planted symmetries on Fashion-MNIST -- results; CIFAR-10 and pretrained encoders --
pre-registration.**
CPU, fins_image.py planted, Fashion-MNIST, 20 000 training images, 5 epochs, seeds 100-102,
500 test images as probes, 14 fields; clean accuracy MLP 0.80-0.86, CNN 0.72-0.81. Share of the planted
generator outside the 1 / 3 lowest eigenvectors per seed (chance 0.96 / 0.89), log-probabilities; control
= same architecture without augmentation (share outside the 3 lowest):
  MLP rotation     0.67/0.22  0.37/0.24  0.75/0.17     control 0.52 0.55 0.57
  MLP scaling      0.35/0.23  0.87/0.25  0.93/0.52     control 0.98 0.98 0.99
  MLP projective   0.25/0.21  0.33/0.30  0.28/0.27     control 0.98 1.00 0.99
  MLP brightness   0.13/0.12  0.25/0.24  0.27/0.27     control 0.92 0.94 0.94
  MLP intensity    0.81/0.78  0.84/0.83  0.96/0.91     control 0.88 0.91 0.90
  CNN rotation     0.99/0.47  0.96/0.48  0.93/0.34     control 0.79 0.81 0.72
  CNN scaling      1.00/0.84  0.93/0.76  1.00/0.81     control 0.88 0.90 0.94
  CNN projective   0.80/0.34  0.92/0.51  0.89/0.40     control 0.82 0.93 0.89
  CNN brightness   0.20/0.19  0.27/0.21  0.39/0.39     control 0.99 0.99 0.87
  CNN intensity    1.00/0.90  0.96/0.95  0.25/0.20     control 0.99 0.99 0.70
Ranks of the planted generator: MLP 2-3 (rotation, projective, brightness), 3-5 (scaling), 12
(intensity); CNN 3-5 (rotation), 4-9 (projective), 2-11 (brightness), 6-7 (scaling). Its closest
eigenvector is the lowest for projective and brightness on 3/3 seeds (cos^2 0.89-0.98), for rotation on
2/3 (0.55, 0.86): the lowest eigenvector of the MLP mixes rotation with the bending field (-y^2, xy). With
L2-normalised logits intensity is the lowest eigenvector of every MLP to two digits (cos^2 1.00, share
outside 0.04-0.07, planted or not) and nearly so of the CNN (0.12-0.24 outside); the planted rotation,
projective and brightness follow within the 3 lowest for the MLP (0.07-0.53).
In LieGG's own class (GL(2), four linear fields), the lowest eigenvector of FINS against LieGG's
generator, share of the planted generator outside, trained (control):
  MLP rotation     FINS 0.15 (0.56) 0.16 (0.58) 0.14 (0.68)    LieGG 0.12 (0.34) 0.49 (0.52) 0.39 (0.33)
  MLP scaling      FINS 0.23 (1.00) 0.20 (0.99) 0.27 (0.99)    LieGG 0.49 (0.99) 1.00 (1.00) 0.39 (0.98)
  CNN rotation     FINS 0.38 (0.33) 0.26 (0.57) 0.27 (0.55)    LieGG 0.79 (0.82) 0.63 (0.68) 0.62 (0.61)
  CNN scaling      FINS 0.29 (0.98) 0.11 (0.98) 0.11 (0.99)    LieGG 0.82 (0.92) 0.97 (0.99) 0.99 (1.00)
In the richer classes the CNN's planted scaling is overtaken by other near-invariances (affine class with
phi: 0.87-0.94 outside the lowest; full class: 0.76-0.84 outside the 3 lowest).
VERDICTS. (I1) rank 1: NOT MET as stated (ranks 2-3; see the correction above; closest eigenvector the
lowest on 3/3 for projective and brightness, 2/3 for rotation); controls at rank >= 5: met. (I2) scaling:
MET, 2/3 (ranks 3, 4 with 0.23, 0.25; s102 rank 5, 0.52). (I3) intensity not recovered: MET, 3/3 (rank 12).
(I4) L2-normalised logits: NOT MET as stated (intensity at rank 2 with cos^2 1.00; rotation and projective
at rank 3, brightness 2). (I5) CNN rotation and projective within the 3 lowest: MET, 3/3 each (0.34-0.51);
brightness, held uncertain, recovered on 3/3 (0.19-0.39). (I6) LieGG closer to the planted generator than
for the control: MET as stated on 2/3 or 3/3 per case, by margins of 0.01-0.6; its generator stays far from
the planted one for the CNN (0.62-0.99). (I7) untrained networks at rank >= 5: NOT MET, three exceptions
(MLP s102 brightness at rank 4 and, under L2, 2; MLP s101 intensity at rank 2).
Reading: the eigenproblem recovers a planted Lie point symmetry of a trained classifier as a
low-dimensional subspace (share outside the 3 lowest 0.12-0.52 against 0.52-0.99 without it), including
the non-affine projective field and an amplitude generator, which LieGG's class cannot represent; within
LieGG's class it recovers rotation and scaling where LieGG does not (CNN). It fails where augmentation
does not produce invariance of the output (intensity under log-probabilities: the decision is invariant,
the confidence scales), and where other near-invariances of the architecture are stronger than the
planted one (CNN scaling in the full class). Richer classes expose near-invariances that mix with the
planted generator (the MLP's bending field), as heat's near-symmetries do in the PDE part.
PRE-REGISTERED, Colab, fins_image.py planted --data cifar10 --arch mlp,cnn --seeds 100,101,102 --epochs 15,
50 000 training images, 24 fields (chance outside the 1 / 3 lowest 0.98 / 0.94), groups none, rotation,
scaling, projective, brightness (grey offset), intensity, saturation, hue:
(C1) MLP, log-probabilities: rotation, projective and grey offset within the 3 lowest (share outside
<= 0.6) on >= 2/3 seeds and at least 0.2 below the control's.
(C2) CNN, log-probabilities: rotation and projective as in C1.
(C3) Saturation and hue, both architectures, log-probabilities: within the 3 lowest (<= 0.6) on >= 2/3
seeds. Held uncertain.
(C4) In the GL(2) class the lowest eigenvector of FINS lies closer to the planted rotation and scaling
than LieGG's generator on >= 2/3 seeds per architecture.
(C5) MLP, log-probabilities: intensity not recovered (> 0.6 outside the 3 lowest) on >= 2/3 seeds.
PRE-REGISTERED, Colab, fins_image.py encoder, 512 Imagenette validation images at 224 px, 24 fields; the
expected subspace of an encoder is the continuous part of its documented training augmentations:
DINOv2 ViT-S/14 (random resized crop, colour jitter): translations, isotropic scaling, stretch, intensity,
saturation, hue (7); ResNet-50 IMAGENET1K_V1 (random resized crop, flip): translations, scaling, stretch
(4); a randomly initialised ViT-S/14 as the architecture's chance level.
(E1) DINOv2: missing dimensions of its expected subspace in the 7 lowest eigenvectors <= 3.5 (random
subspace 4.96). Held uncertain.
(E2) ResNet-50: missing of its 4 in the 4 lowest <= 2.0 (random 3.33). Held uncertain.
(E3) Saturation and hue rank lower in DINOv2's spectrum than in ResNet-50's.
(E4) Random ViT: missing of the DINOv2 subspace in its 7 lowest >= 4.0.

**Images, second design: the finite-class eigenproblem on a frozen network -- code, pilot, pre-registration.**
The first design (a generator network trained against DINOv2) learned image-independent constant fields
(entry below). The second applies the method of the PDE part: a finite class and one generalised
eigenproblem, no generator network. vision/fins_image.py: class xi, eta in {x^a y^b, a + b <= 2} (12
fields), phi affine in colour (grey: offset, u; colour: 3 offsets, 9 channel mixings); characteristic
Q = phi - xi u_x - eta u_y (central differences on [-1, 1]^2); operator a frozen network (class
log-probabilities, or L2-normalised logits or features); N_ij = mean <J Q_i, J Q_j> by one JVP per field,
D_ij = mean <Q_i, Q_j>; N c = lambda D c. Scores as in the PDE protocol: rank of a known generator, its
share outside the k lowest eigenvectors, missing dimensions of a known subspace, random-subspace chance
level. LieGG (Moskalev et al., NeurIPS 2022) as the baseline, from its repository (polarization matrix of
the GL(2) class with the gradient of the summed L2-normalised logits; colour channels summed here).
Checks (CPU): exactly invariant operators give their generator at rank 1 or 2 with the rest separated
(ring averages: rotation, share outside the 2 lowest 0.19, pixelation; centred image: offset, loss 0;
unit-norm image: intensity, loss 1e-17; Fourier magnitude: translation, share 0.02; luminance of a colour
image: the 8-dimensional null space exactly, saturation in it); the planted augmentations are generated
by the named generators (cosine of the finite-s derivative with the characteristic 1.0000 for rotation,
scaling, projective, brightness, intensity, saturation, hue).
PILOT (Fashion-MNIST, 20 000 training images, 5 epochs, seeds 0 and 1, 500 test images; prototype code):
MLP 784-512-512-10 with log-probabilities: the planted generator is the lowest eigenvector for rotation
(+/- 30 deg; share outside it 0.42-0.57), the projective field (x^2, xy) (0.20-0.32) and brightness
(0.19-0.24); scaling ranks 3-4 (share outside the 3 lowest 0.32-0.47); intensity (u -> e^s u) is not
recovered (ranks 11-13): the ReLU network is nearly positively homogeneous, so augmentation makes its
decision scale-invariant but not its log-probabilities. The control without augmentation ranks the same
generators 6-13. With L2-normalised logits intensity is rank 1 in every network (the same homogeneity,
now an invariance of the operator), and the planted generator follows at rank 2 (scaling 3). A small CNN
(raw logits) lifted the planted rotation from rank 9 to 3; CNNs are local and become invariant also to
bending fields that resemble a rotation near the edges. LieGG returns a skew-like generator also for the
control.
PRE-REGISTERED (fins_image.py planted --data fmnist --fmnist-npz ... --arch mlp,cnn --seeds 100,101,102
--epochs 5; 20 000 training images, 500 test images as probes; groups none, rotation, scaling,
projective, brightness, intensity):
(I1) MLP, log-probabilities: rank 1 on >= 2/3 seeds for rotation, projective and brightness; the same
generator at rank >= 5 in the control on >= 2/3 seeds.
(I2) MLP, log-probabilities: scaling at rank <= 4 with share outside the 3 lowest <= 0.6 on >= 2/3 seeds.
(I3) MLP, log-probabilities: intensity not recovered (rank > 3 on >= 2/3 seeds). Held likely.
(I4) MLP, L2-normalised logits: intensity rank 1 in every network, planted or not; the planted rotation,
projective and brightness at rank <= 2 on >= 2/3 seeds.
(I5) CNN, log-probabilities: rotation and projective within the 3 lowest (share outside them <= 0.6) on
>= 2/3 seeds; rank 1 not predicted; brightness held uncertain (the CNN is most sensitive to offsets).
(I6) LieGG: for planted rotation and scaling its generator lies closer to the planted one than the
control's on >= 2/3 seeds. Held uncertain.
(I7) Untrained networks rank the planted generators >= 5 (intensity under L2-normalised logits aside).
CORRECTION, written after the MLP results of seeds 100 and 101 had printed: the pilot's ranks above are
counts of eigenvalues below the generator's loss, i.e. rank - 1 in fins_image.py. Read correctly the
pilot put rotation, projective and brightness at rank 2, scaling at 4-5, and intensity under
L2-normalised logits at rank 2 (share outside the lowest 0.04-0.19). A generator reaches rank 1 only if it
is exactly an eigenvector (Rayleigh quotient), so the rank is conservative; the share outside the k lowest
eigenvectors is the informative score. I1 and I4 stay as registered and are judged as stated.

**Vision part: structure of the learned generators (V4, V5) and crop and flip in every arm (V6) -- results.**
generator_structure.py on the fifteen checkpoints (Colab; 256 fit images each, 100 initialisations as
the chance level). Energy-weighted over slots, mean +/- std over seeds (null_init in brackets):
                     950/950                5 000/5 000            50 000/5 000
  translation        0.9996 +/- 0.0002      0.940 +/- 0.008        0.939 +/- 0.006      (0.997)
  of it fixed        0.9995                 0.940                  0.939                (0.997)
  rot / dil / shear  <= 0.0001              <= 0.001               <= 0.001             (<= 0.001)
  non-affine rest    < 0.001                0.048-0.071            0.051-0.070          (0.001)
  image-dependent    0.0002                 0.0002                 0.0002               (0.001)
  shift at tau = 1   2.4 +/- 0.6 px         2.2 +/- 0.5 px         2.2 +/- 0.5 px
  RMS of phi         0.050                  0.005                  0.005
  grey share of phi  0.995 +/- 0.002        0.71 +/- 0.09          0.74 +/- 0.07        (0.38; 0.24)
  colour share       0.004                  0.10                   0.08                 (0.62; 0.75)
  contrast share     0.0006                 0.028                  0.026                (0.0004)
Every learned generator is, to 93-100 percent of the energy of (xi, eta), one global translation, the same
for every image: the image-dependent part is 0.02 percent at every size, and rotation, dilation and
shear are below 0.05 percent. The generator network does not use the image it is given. At 950 the
fields are more constant than at initialisation (translation share at the 87th-100th percentile of the
100 initialisations) and phi is a grey offset (99.5 percent grey; null_init 38 percent on average, 5-55
per seed). At 5 000 and 50 000 training has added a small image-independent
non-affine warp (5-7 percent; percentile 0.99-1.00) and phi has collapsed to RMS 0.005 (offset 0.004 at
tau = 1), still greyer than chance (percentile 0.91-1.00), with a small contrast component (percentile
1.00) that carries no weight at this magnitude.
VERDICTS. (V4) translation share >= 0.8 at 5 000 and 50 000 on every seed: MET, 10/10 (0.929-0.952).
(V5) at 950 grey share above the 90th percentile of the initialisations on >= 4/5 seeds and above null_init
on >= 4/5: MET, 5/5 and 5/5 (0.992-0.997 against 0.05-0.55; percentile 1.00 on every seed).
Reading: the criterion keeps what DINOv2 is nearly invariant to, sub-pixel global translation and grey
offsets, and removes colour casts; nothing spatially structured is learned. The advantage over null_init
at 950 is the direction of phi (brightness against colour casts of the same magnitude), which is where
its gain lies (brightness +0.107, contrast +0.044); once phi has collapsed (5 000, 50 000) the learned
generators are random translations in effect, and FINS ties with null_init.
Accuracy, Colab, n = 5 000, seeds 100-104, --paired, all seven arms, --base-aug (random crop, pad 4, and
flip in every arm; the 5 000 checkpoints loaded). Average over the eight corruptions: no_aug 0.390, FINS
0.520 +/- 0.016, null_init 0.536 +/- 0.020, null_smooth 0.443, randaug 0.460, autoaug 0.470, trivialaug 0.470.
FINS minus trivialaug (best baseline) +0.093 +0.061 +0.037 +0.047 +0.011, +0.050 +/- 0.027, 5/5; minus
null_init +0.027 -0.047 -0.034 -0.014 -0.014, -0.016 +/- 0.025, 1/5; minus null_smooth +0.077 +/- 0.015, 5/5.
Per corruption, FINS / null_init against the best baseline: Gaussian noise 0.481 / 0.487 vs 0.283, shot
0.582 / 0.600 vs 0.476, impulse 0.422 / 0.431 vs 0.263, blur 0.713 / 0.749 vs 0.601, pixelate 0.388 /
0.321 vs 0.255; the baselines lead on clean (0.794 vs 0.729 / 0.764), jpeg, contrast and brightness.
VERDICT. (V6) FINS minus the best baseline <= 0.01 with crop and flip in every arm: NOT MET, +0.050 (the
lead grew from +0.039). As the registered alternative said: the lead sits in the noises, blur and
pixelate, and null_init keeps it (0.536 against 0.470). Crops and flips raise clean accuracy but lower
the baselines' noise accuracy below NoAug's (0.27-0.28 against 0.42); sub-pixel translations resampled
bilinearly act as a mild low-pass augmentation that neither integer crops nor the baselines'
nearest-neighbour geometric operations provide. Adding crop and flip lowered the average corruption
accuracy of most arms (FINS 0.541 -> 0.520, null_init 0.547 -> 0.536, randaug 0.502 -> 0.460).
Consequence for the paper: the image variant recovers two classical augmentations, global translation
and brightness, as the encoder's approximate invariances; its lead over the augmentation policies on
these corruptions comes from bilinear translation, which the untrained network rescaled to the same
magnitudes provides as well. The claim that it finds semantic symmetries or useful augmentations beyond
these is not supported. Framing (removal, or a short appendix with this result) is for the authors to decide.

**Vision part: 50 000 against the chance levels -- results (V1); paired baselines at 50 000 -- pre-registration.**
Colab (RTX PRO 6000), CIFAR-10, 50 000 training images, seeds 100-104, --paired, arms fins, null_init,
null_smooth; generators refitted on 5 000 of the 50 000 images (new checkpoints sym_gen_seed10x.pt).
Average over the eight corruptions: FINS 0.602 +/- 0.006, null_init 0.591 +/- 0.019, null_smooth
0.555 +/- 0.025. FINS minus null_init per seed +0.033 +0.033 -0.029 +0.004 +0.014, mean +0.011 +/- 0.023,
ahead on 4/5; minus null_smooth +0.052 +0.047 +0.014 +0.033 +0.090, +0.047 +/- 0.025, 5/5.
Per corruption, FINS / null_init / null_smooth: clean 0.930 / 0.931 / 0.864; Gaussian noise 0.406 /
0.377 / 0.329; shot 0.587 / 0.574 / 0.478; impulse 0.368 / 0.333 / 0.288; blur 0.917 / 0.919 / 0.835;
pixelate 0.278 / 0.257 / 0.555; jpeg 0.507 / 0.510 / 0.429; contrast 0.872 / 0.874 / 0.750; brightness
0.881 / 0.885 / 0.775. On the noises null_init varies across seeds by +/- 0.043-0.056, FINS by 0.009-0.018.
VERDICT (V1): MET as stated (4/5 against null_init, 5/5 against null_smooth). The lead over null_init is
small against its spread and sits in the three noises and pixelate; on clean and the other four
corruptions the two agree within 0.004. null_smooth teaches pixelation (0.555 against 0.278) and costs
everywhere else, clean included.
Across sizes, FINS minus null_init: +0.032 +/- 0.014 (950, 5/5), -0.007 +/- 0.015 (5 000, 2/5), +0.011 +/-
0.023 (50 000, 4/5). The generators behind the last two are nearly the same: per-slot RMS at 50 000 and
5 000 agree to a few percent seed by seed (s100: 0.156/0.0074, 0.067/0.0021, 0.103/0.0018, ... against
0.157/0.0079, 0.070/0.0021, 0.103/0.0019, ...), although the fits used different 5 000-image subsets; at
950 the geometric RMS keeps the same per-seed pattern (s100: 0.172, 0.076, 0.109, 0.168, 0.171, 0.169)
and phi is five to ten times larger. What a generator becomes is set by its initialisation and the
number of steps, not by the images it is fitted on.
The earlier 50 000 run behind the paper (FINS 0.566 +/- 0.020) is not reproduced: the same arm now gives
0.602 +/- 0.006. The script differs from the earlier version only in flags, the chance levels and the
micro-batch option (same gradient; diff checked), so the difference comes from the environment (library
versions, GPU) or from that run's configuration. The paper quotes the new paired runs only, and the
baselines at 50 000 are rerun in the same setting. V2 at 50 000 is recomputed on them.
Code: vision/paired_summary.py merges result files of the same seeds (runs with --paired reseed before
every arm, so arms from separate runs are paired), checks their config sidecars (seed start, sizes,
paired, base augmentation), prints mean +/- std per arm and corruption and the reference arm minus
every other per seed. Checked on synthetic files, mismatch warnings included.
PRE-REGISTERED, Colab, n = 50 000, seeds 100-104, --paired, no_aug, randaug, autoaug, trivialaug, merged
with the run above:
(V7) FINS above the best baseline (by mean) on >= 4/5 seeds (the paper's former claim; 5/5 in the
earlier, unpaired run).
(V8) null_init above the best baseline on >= 4/5 seeds: translations and colour offsets at the learned
magnitudes carry the lead over the baselines. Held likely, given FINS minus null_init +0.011 +/- 0.023.
(V2, final) FINS minus the best baseline above +0.039, its value at 5 000. Held uncertain.

**Protocol run, part 6: transport error, heat seeds 43 and 44 -- results (Colab).**
                 eta     eta_V    floor     sigma_(k+1)  sigma~_k / sigma~_(k+1)  Weyl max  (iii) shares / bounds
  heat s43       0.0403  0.0366   0.00021   0.0311       0.0349 / 0.0373         0.035     0.010-0.334 / 0.221-0.696
  heat s44       0.0408  0.0366   0.00028   0.0201       0.0320 / 0.0360         0.032     0.016-0.614 / 0.119-0.712
Exact losses of the algebra <= 6e-8 on both; learned losses of the algebra 2e-5 to 7e-4; implemented
normaliser sigma_theta / sigma~ in [0.998, 1.002]. The largest shares outside are the heat polynomials
(s43: b = x^3 + 6axt 0.33; s44: the same 0.61, b = x 0.43).
VERDICTS. (T1) Weyl and the (iii) bound hold on every run: MET, 6/6 (KdV 3, heat 3). (T2) heat: the
sufficient condition fails and eta >= 0.02 on each seed (0.070, 0.040, 0.041): MET, 3/3.
eta_V / sigma_(k+1): heat 1.8 (s42), 1.2 (s43), 1.8 (s44); KdV 0.04-0.10. The learned spectrum has no gap
after the algebra on any heat seed (sigma~_10 / sigma~_11 = 0.97, 0.94, 0.89).
FLAG: eta_V is 3.657e-2 on both s43 and s44, to four digits, although eta, the losses and the spectra
differ. CPU check (the exact parts of the Colab runs reproduced to every printed digit above
1e-20, the operator an older s43 CNO from 27.09): eta_V moves with the data for a fixed operator (0.0227 on data
1043, 0.0177 on 1044) and with the operator (0.0227 against Colab's 0.0366 on the same data); a frozen
operator gives 0.67 on both. So the equality is a coincidence or both runs loaded operators with the same
error along the algebra. exp43 now prints the loaded checkpoint with a parameter checksum and eta, eta_V
to seven digits; heat s43 and s44 are rerun. RESOLVED (entry above): distinct operators, eta_V
0.0365696 and 0.0365709. Also fixed: the (iii) check reported False when a loss is
exactly zero (frozen operator, b = 1), now with an absolute tolerance of 1e-12; the Colab runs were
unaffected.

**Vision part: size series at 950 and 5 000 -- results; generator structure -- code and pre-registration.**
Colab, CIFAR-10, seeds 100-104, --paired, all seven arms. Average accuracy over the eight corruptions,
mean over seeds (FINS +/- std):
                no_aug   FINS            null_init   null_smooth   randaug   autoaug   trivialaug
  n = 950       0.313    0.378 +/- 0.008  0.346       0.339         0.361     0.343     0.334
  n = 5 000     0.413    0.541 +/- 0.013  0.547       0.479         0.502     0.478     0.465
Paired differences, FINS minus arm, mean +/- std over seeds (FINS ahead on):
  n = 950       null_init +0.032 +/- 0.014 (5/5); null_smooth +0.039 +/- 0.015 (5/5); randaug +0.017 +/- 0.010 (5/5)
  n = 5 000     null_init -0.007 +/- 0.015 (2/5); null_smooth +0.062 +/- 0.023 (5/5); randaug +0.039 +/- 0.018 (5/5)
Learned per-slot magnitudes (printed by null_init): at 5 000 phi has collapsed, RMS 0.0015-0.011 against
0.07-0.25 for (xi, eta), final balance loss 0.248 (0.25 means geometric share 1 against the target 0.5);
at 950 phi is alive. At 950 the gain over null_init sits mainly in brightness (0.357 against 0.250) and
contrast (0.238 against 0.194).
VERDICTS. (V3) 950: MET against both nulls (5/5, 5/5). 5 000: MET against null_smooth (5/5), NOT MET
against null_init (2/5, -0.007): generators fitted on 5 000 images do no better than the network at
initialisation rescaled to their per-slot magnitudes, whose fields are nearly constant (global
translations and colour offsets). (V2) NOT MET as stated: FINS minus the best baseline is +0.017 at 950
and +0.039 at 5 000, but +0.034 at 50 000 in the earlier run (baselines not rerun, seeds not paired), so
no growth from 5 000 to 50 000. (V1) pending, the 50 000 run is in progress.
Two facts for the paper. NoAug trains on the n originals, every other arm on 4n images (originals and
three copies) for the same 30 epochs, i.e. four times the optimiser steps: comparisons with NoAug carry
this confound, comparisons among the augmented arms do not. The balance term did not prevent collapse at
5 000; phi collapsed and (xi, eta) survived, while the paper says the term prevents collapse to
phi-dominant directions.
Code: vision/generator_structure.py (no training, no encoder). For every generator checkpoint, the six
slots on the first 256 images of the fit subset (the images null_init was matched on); energy shares of
(xi, eta): translation (per-image spatial mean, and its part common to all images), rotation, dilation,
shear, the non-affine rest, the image-dependent part; of phi: grey offset (brightness), colour offset,
per-channel contrast (phi proportional to the centred image), the rest, the image-dependent part;
shares energy-weighted over slots. The same for null_init, reconstructed exactly, and for 100 further
initialisations rescaled to the learned per-slot RMS, whose distribution is the chance level of every
share (pct: fraction of them below the learned value). CPU checks: every share exact on synthetic fields
(constant, rotation, dilation, shear, image-dependent translation, grey, colour, contrast; the parts sum
to 1); on stub checkpoints trained by fins_cifar.py the probe images, the per-slot RMS and the null_init
fields equal those of fins_cifar.py (difference 0).
PRE-REGISTERED, generator_structure.py on the fifteen checkpoints (950/950, 5 000/5 000, 50 000/5 000;
seeds 100-104):
(V4) At 5 000 and 50 000 the translation share of (xi, eta) is >= 0.8 on every seed: the generators that
tie with null_init are global translations, as it is.
(V5) At 950 phi's grey share lies above the 90th percentile of the initialisation chance level on
>= 4/5 seeds, and above null_init's on >= 4/5: training turns phi toward brightness, which the gain on
brightness over null_init would need. Held uncertain.
PRE-REGISTERED, accuracy, Colab, n = 5 000, seeds 100-104, --paired, all seven arms, --base-aug (random
crop, pad 4, and flip in every arm; the existing 5 000 checkpoints are loaded):
(V6) FINS minus the best baseline (by mean) <= 0.01: once every arm is translated, the generators fitted
on 5 000 images add nothing. Held uncertain: the bilinear resampling of a sub-pixel translation is also a
mild blur, which crops and the baselines' nearest-neighbour geometric operations do not supply; if FINS
keeps its lead, the lead should sit in blur, pixelate and the noises, and null_init should keep it too.
Paper: not changed until V1, V4-V6 are in.

**Protocol run, part 6: transport error (exp43) -- results (Colab, KdV s42-44, heat s42).**
                 eta     eta_V    floor    sigma_(k+1)  sigma~_(k+1)  Weyl max  (iii) shares / bounds
  KdV s42        0.553   0.0058   0.0102   0.132        0.130         0.049     0.001-0.007 / 0.006-0.033
  KdV s43        0.596   0.0123   0.0110   0.122        0.118         0.085     0.003-0.011 / 0.025-0.053
  KdV s44        0.569   0.0069   0.0113   0.117        0.114         0.053     0.001-0.007 / 0.008-0.038
  heat s42       0.070   0.063    0.00023  0.0346       0.059         0.056     0.03-0.54 / 0.26-0.90
Exact losses of the algebra: KdV 3e-16 / 5-7e-11 / 6-8e-16 / 2.1-2.6e-5; heat all ten <= 4e-8, lambda11
1.2e-3. Learned sigma~_k: KdV 0.0039 / 0.0079 / 0.0048, heat 0.057 (spectrum compressed). Implemented
normaliser: sigma_theta / sigma~ in [0.999, 1.094] (KdV), [0.997, 1.002] (heat).
VERDICTS. (T1) Weyl and the (iii) bound hold on every run -- MET on 4/4 (heat s43, s44 pending). (T2)
heat: the sufficient condition fails, eta 0.070 >= 0.02 -- MET on s42. (T3) KdV: the sufficient
condition holds on 3/3 -- NOT MET, 0/3: eta 0.55-0.60 against sigma'_5 / 2 = 0.058-0.066.
Reading, as pre-registered for this case: the worst case over the class is dominated by directions far
from the algebra, which the stepper transports badly (55-60 percent) but whose losses stay large
(sigma~_5 only 1.7-3.1 percent below sigma_5; the largest Weyl shifts, 0.05-0.09, sit higher in the
spectrum). Along the algebra the error is 0.006-0.012, 0.04-0.10 of sigma_5; on heat's class it is 0.063,
1.8 times sigma_11, and the learned spectrum is compressed. The per-generator bound of (iii) with the
learned sigma~_(k+1) is informative on KdV (<= 0.053, measured <= 0.011) and not on heat (0.26-0.90; b = 1
lies 54 percent outside the ten lowest). Paper: Section 3.4 reports these numbers, the discussion reads
the outcome through eta_V / sigma_(k+1) instead of the worst-case condition, the introduction states it.

**Vision part: code review, chance levels, size series -- code and pre-registration.**
Review of vision/ against Section 6 of the paper. fins_cifar.py (the v7 method) matches the text:
DINOv2-ViT-S/14, characteristic JVP, finite-tau, orthogonality, balance, norm and smoothness terms,
six generators fitted on 5 000 training images, ResNet-18 from scratch. Found: (1) the corruption set
is eight parametric corruptions at one severity plus clean, not nine types; "jpeg" is a colour
quantization; (2) every arm gets three augmented copies generated once, and no arm uses crops or
flips; (3) AutoAugment's CIFAR-10 policy was searched on reduced CIFAR-10; (4) fins_pet.py is the
v5/v6 method: a cat-vs-dog ResNet-18 with coordinate channels as the operator, the invariance
xi dL/dx_channel + eta dL/dy_channel + phi . dL/du (the coordinate-channel form, not the
characteristic), orthogonality and smoothness only, a pretrained ResNet-18 fine-tuned on six cat
breeds, five copies, tau in (0.3, 0.8), 256 px -- the paper described the v7 method for Pet too; (5)
the Pet loss ablation (L_inv + L_tau vs L_inv) cannot come from fins_pet.py, which has no L_tau; its
code is not in the repository. Paper: Section 6 rewritten (operational definition of the candidates,
earlier Pet formulation cited, corruptions and augmentation protocol described as run, size claim
replaced by the facts, "follows from Equation (8)" removed, PDE-only-baselines clause removed); Pet
marked for rerun or removal.
Code (fins_cifar.py, flags override CONFIG; without flags the CONFIG run is reproduced): --methods
from no_aug, fins, null_init, null_smooth, randaug, autoaug, trivialaug; --paired reseeds before every
arm (same classifier init, data order, and generator slots and tau draws for fins and the nulls);
--cifar-train-size, --sym-gen-subset; --base-aug (random crop, pad 4, and flip in every arm);
--out-dir, --out-json, --sym-gen-dir; generator checkpoints named by training and fit size except the
original (50 000, 5 000). Chance levels at the learned per-slot RMS of (xi, eta) and of phi: null_init,
the generator network at initialisation -- its fields are 99.8-100 percent constant, so it amounts to
global translations and colour shifts; null_smooth, image-independent random smooth fields (Fourier
modes 1 <= |k| <= 3, no constant mode). CPU checks with stub data and encoder: RMS matched to 1e-7,
paired draws identical, crop-flip exact, full run of all seven arms, resume and default mode.
Memory (DINOv2-ViT-S/14 architecture built locally, random weights, CPU, no xformers): one generator
training step needs 1.3-1.5 GB per image (about 30 encoder passes kept for one backward), so the
default batch of 32 needs tens of GB and cannot run on a T4. --sym-micro-batch M computes the encoder
terms in slices of M and backpropagates them slice by slice, the batch-level terms (orthogonality,
balance, smoothness) on the whole batch: gradients equal the full-batch ones to 1e-6 relative (stub
encoder, batch 8, slices of 2 and 3; DINOv2, batch 2, slices of 1), peak memory 1.7 GB instead of 3.0 at
batch 2. T4: --sym-micro-batch 4.
PRE-REGISTERED (Colab, CIFAR-10, seeds 100-104, paired):
(V1) At 50 000 images, FINS above null_init and above null_smooth in average corruption accuracy on
>= 4/5 seeds each. Held uncertain: translations and colour shifts are standard CIFAR augmentations,
so null_init may be strong.
(V2) The paper's former claim: FINS minus the best baseline (by mean) grows with the training-set
size, 950 < 5 000 < 50 000. Held uncertain.
(V3) V1 at 950 and 5 000 images.

**Protocol run, part 6: transport error (exp43) -- code, CPU check, pre-registration; cost.**
exp43_transport_error.py measures Proposition 2 of the paper on the protocol's evaluation data:
transport blocks through the exact operator (heat: the 13-tap moment kernel on four-period windows;
KdV: the spectral solver on periodised solitons, the finite-difference Jacobian of exp39 --op exact)
and through the learned one, with the same basis, starts, horizon and scored region. Printed:
eta = ||E D^-1/2||, eta_V (its part along the algebra), the exact floor ||M P_V D^-1/2||, exact and
learned sigma_k and sigma_(k+1), the Weyl check, the sufficient condition eta' < sigma'_(k+1) / 2
(eta' = ||(M P_V + E) D^-1/2||, sigma' the spectrum of M (I - P_V), which vanishes on the algebra),
the rescaling by the implemented normaliser D_theta, and per known generator its share outside the
k lowest learned eigenvectors against (loss / lambda_(k+1))^1/2.
CPU check (seed-7 timing operators, one epoch on 16 trajectories; data 1042, n = 3): KdV eta 1.40,
heat 0.96 -- the condition fails, as it must at this accuracy; Weyl and (iii) hold. Synthetic check
on the KdV blocks, E scaled by 1e-3 / 1e-2 / 3e-2 / 1: eta 1.4e-3 / 1.4e-2 / 4.2e-2 / 1.4, the
sufficient condition holds for the first three (angle bounds 7.1 / 11.3 / 46.9 deg), Weyl, the
normaliser rescaling and (iii) hold throughout; the per-generator bound of (iii) sits 5-10x above
the measured share (0.30-0.33 vs 0.04-0.06 at 3e-2) and far below the eta-based angle bound. Exact
KdV on data 1042 (n = 3): losses of the algebra 3e-16, 7e-11, 5e-16, 2.4e-5; floor over the algebra
1.1e-2, above the square root of every basis loss (the algebra's directions are nearly dependent in
the D-norm). Exact heat: all ten <= 1.3e-7, lambda11 1.3e-4, floor 4.3e-4.
Paper: Proposition 2 (iii) sharpened -- every c has a share outside the k lowest learned
eigenvectors of at most L_theta(c)^1/2 / sigma~_(k+1); the step uses the learned pencil only, so it
holds with the implemented normaliser, and a margin m is a certificate of share <= m^-1/2: KdV
<= 0.053 on every seed, Burgers and heat nothing useful.
PRE-REGISTERED for the Colab run (protocol checkpoints, s = 42-44, data 1000 + s, n = 10):
(T1) Weyl and the (iii) bound hold on every run (theorems; a failure is a bug).
(T2) heat: the sufficient condition fails on 3/3, with eta >= 0.02 on each seed (by (i) eta is at
least the square root of the largest learned loss of the algebra, 0.022-0.054, against
sigma'_11 / 2 ~ 0.010-0.017 from the exact lambda11).
(T3) KdV: the sufficient condition holds on 3/3 -- the paper's reading of KdV's success (nearest
non-symmetry far, learned transport error small). If it fails, that reading rests on (i) and the
per-generator bound of (iii) only, and the discussion is revised.
PRE-REGISTERED, fresh evaluation data (data seeds 2042-2044, never used in development; same
operators, same commands as the protocol rows, --save into a separate folder, exp42 --dir on it):
(F1) FINS criterion as P1: Burgers all five below lambda6 on >= 2/3, heat all ten below lambda11 on
>= 2/3, KdV all four below lambda5 with margin >= 100x on 3/3. (F2) FINS below the Gaussian-coefficient
class null on every equation and seed, heat finite and full (as P6e). (F3) Exact operators as P4:
missing and spurious <= 0.05, novel containment >= 0.95 (KdV, heat full class).
Reproduction check (CPU): exp39 --op exact --ic periodic --n 10 on data 1042-1044 and
exp42 --pdes kdv on those files reproduce the protocol's exact KdV row and both class nulls to every
printed digit (missing 0.01, margins 838.5x / 568.5x / 519.1x, ratios 1183.8 / 650.4 / 632.5; uniform
null 3.46, Grassmann 2.53, cosine 0.23; Gaussian 2.85, 2.42, 0.31) -- except the Hungarian cosine of
the exact row, 0.89 on CPU against 0.83 on Colab: its three lowest eigenvalues are 1e-16 .. 2e-10, the
basis inside that eigenspace is arbitrary, and the cosine is basis-dependent (known, section 8 of
PROTOCOL.md).
CORRECTION to part 2: LieNLSD returned fewer generators than the class dimension on two Burgers seeds,
not only s44 (exp42 k-range 4-5, mean 4.3; KdV 3-4, mean 3.3); the paper says two seeds each.
Cost, measured for the paper (PROTOCOL_RESULTS_v1.md): the discovery step on two CPU cores, two runs,
KdV 67-73 s / 0.8 GB peak resident memory, Burgers 86-91 s / 1.3 GB, heat 163 s / 2.1 GB (data
generation included, ten trajectories, seven starts, K = 16).

**Protocol run, part 5: chance levels and own-class nulls -- results (Colab, 3 x 3, LIG_k trained
on Burgers s42 so far).** Missing dimensions, finite ground truth, mean over seeds:
                               Burgers (5)  KdV (4)  heat (6)  heat, full (10)
  FINS                         0.44         0.01     0.14      1.59
  null, Gaussian coefficients  2.76         2.85     2.91      6.36
  null, uniform on the class   4.17         3.46     4.99      8.33
  LieGAN        trained        2.98         2.84     4.06      6.66
                untrained      2.54         2.05     3.57      6.31
  LaLiGAN       trained        4.63         3.05     4.10      7.39
                untrained      5.00         4.00     6.00      10.00
  Augerino      trained = untrained  1.96   0.90     2.00      6.00
  LIG, 4 gen.   trained        2.11         2.90     3.45      6.89
                untrained      2.03         2.64     3.23      6.56
  LIG, k gen.   untrained      1.95 (5)     --       1.61 (10) 4.03
                trained        1.97 (s42)
VERDICTS. (P5a) LIG_k above FINS: 1.97 vs 0.44 on s42 -- MET so far; (P5b) within 0.3 of its own
untrained value: 1.97 vs the untrained mean 1.95 +/- 0.14 -- consistent, per-seed check when s43,
s44 finish. (P6a) LieGAN trained within 0.3 of untrained -- NOT MET, and in the other direction:
training makes LieGAN WORSE by 0.44 / 0.79 / 0.49. (P6b) LIG (4) within 0.3 -- MET (+0.08 / +0.26 /
+0.22, all slightly worse after training). (P6c) LaLiGAN trained below untrained -- MET (its
untrained generators capture nothing: 5.00 / 4.00 / 6.00). (P6d) Augerino trained = untrained --
MET, identical rows. (P6e) FINS below the Gaussian-coefficient null on every equation and seed --
MET (worst seed bounded by mean + sqrt(2) std: Burgers <= 0.94 vs 2.74, heat full <= 2.48 vs 6.2).
Augerino "grown widths only", predicted in part 3: Burgers {d_t} 4.00, KdV {} 4.00, heat {u d_u}
5.00 / 9.00 -- all as predicted.
What the nulls say: no equation-free baseline gets closer to the ground truth by training. LieGAN
and LIG end slightly or clearly worse than their initialisation; LaLiGAN improves from nothing but
ends above (worse than) FINS's Gaussian-coefficient null on every equation; Augerino's score is its
dictionary. A random smooth network with k heads (LIG_k untrained) beats every trained
equation-free baseline on Burgers (Augerino's dictionary by 0.01) and heat. Trained LieGAN sits at
the Gaussian-coefficient null (2.98 / 2.84 / 4.06 vs 2.76 / 2.85 / 2.91). FINS is the only
equation-free method far below every null (0.01-0.44 against 2.76-2.91 for the Gaussian-coefficient
null and 3.46-4.99 for the uniform one); LieNLSD, given the
equation, is below them too (0.67-1.12). For the paper: missing dimensions are reported with the
two class nulls and each method's untrained value beside it; "k random directions" sets the scale.

**Protocol run, part 4: chance levels, own-class nulls, LIG with k generators --
pre-registration.** A CPU smoke test of LIG with five generators on Burgers (8 training
trajectories, one epoch at lr 1e-4, i.e. near initialisation) missed only 1.27 of 5 dimensions, the
capped one 2.23 -- a smooth untrained network already spans much of the low-order field space the
ground truth lives in, so "missing dimensions" has a chance level far below k_B and the tables need
it. Added: (1) exp41 --untrained: every method's generators at initialisation, zero epochs, same data
and extraction (own-class null; LieNLSD has no training); (2) exp42 class nulls: random k-dim
subspaces of FINS's polynomial class, uniform on its span and with Gaussian coefficients in the
normalised monomial basis (FINS's parameterisation), 100 draws per seed; checked against the
closed form k_B (1 - k / r) for the uniform one (heat: 5.00 and 8.31 vs 5.00 and 8.33, r = 60);
(3) LIG_k, LIG with n_sym = class dimension (cfg.lig_max_sym lifts Ko et al.'s cap); (4) the exact
operator row is now marked "uses the equation" (the label test missed it).
Observed on CPU, heat s42 only (finite / full missing dimensions): untrained LieGAN 4.11 / 6.92,
LaLiGAN 6.00 / 10.00, Augerino 2.00 / 6.00 (its dictionary), LIG 4 generators 3.41 / 6.58, LIG 10
generators 2.18 / 4.74; class null uniform 5.00 / 8.31, Gaussian coefficients 3.03 / 6.52. Against
the trained three-seed means: LieGAN 4.06, LIG 3.45 -- no measurable learning on heat; LaLiGAN 4.10
-- learns; FINS 0.14 / 1.59, far below both class nulls.
PRE-REGISTERED for the Colab run (untrained on 3 equations x 3 seeds; LIG_k on Burgers, 10 epochs,
3 seeds), for everything except the heat s42 numbers above:
(P5) LIG_k on Burgers: (a) missing dimensions above FINS's 0.44 (mean); (b) within 0.3 of its own
untrained value on each seed.
(P6) Finite ground truth, mean over seeds: (a) LieGAN trained within 0.3 of untrained on every
equation; (b) LIG (4 generators) likewise; (c) LaLiGAN trained below untrained on every equation;
(d) Augerino trained equal to untrained; (e) FINS below the Gaussian-coefficient class null on every
equation and seed (heat: finite and full).

**Protocol run, part 3: tables (exp42, s = 42-44, every method on the same ten unseen
trajectories, all LIG runs 10 epochs, 3 seeds everywhere).** Missing dimensions, finite ground
truth, mean +/- std over seeds (full tables: PROTOCOL_RESULTS_v1.md):
                     Burgers (5)    KdV (4)        heat (6)       heat, full class (10)
  FINS               0.44 +/- 0.35  0.01 +/- 0.00  0.14 +/- 0.05  1.59 +/- 0.63
  FINS, exact op.    --             0.01 +/- 0.00  0.00 +/- 0.00  0.00 +/- 0.00
  LieNLSD (eq.)      0.85 +/- 0.22  0.67 +/- 0.47  1.12 +/- 0.30  1.40 +/- 0.47
  LIG, 10 ep. (eq.)  2.11 +/- 0.03  2.90 +/- 0.02  3.45 +/- 0.17  6.89 +/- 0.10
  LieGAN             2.98 +/- 0.06  2.84 +/- 0.25  4.06 +/- 0.16  6.66 +/- 0.04
  LaLiGAN            4.63 +/- 0.14  3.05 +/- 0.17  4.10 +/- 0.33  7.39 +/- 0.13
  Augerino           1.96 +/- 0.01  0.90 +/- 0.00  2.00 +/- 0.00  6.00 +/- 0.00
FINS recovers (novel containment >= 0.9) 4.3 / 5 on Burgers (d_x 0.82), 4 / 4 on KdV, 6 / 6 of the
finite heat algebra (0.97-0.98) but only the finite part of the full class (heat polynomials
0.34-0.41: its other four directions are the near-symmetries, as diagnosed). The exact operator
recovers 10 / 10 on heat and 4 / 4 on KdV; on KdV the learned operator matches it.
VERDICTS. (P2) FINS has fewer missing dimensions than LieGAN, LaLiGAN and Augerino on every equation
-- MET (0.44 / 0.01 / 0.14 against the best of them, Augerino, 1.96 / 0.90 / 2.00). (P3) no
prediction; observed: FINS, equation-free, misses fewer dimensions than both equation-aware methods
on Burgers, KdV and the finite heat algebra; on heat's full class LieNLSD is level with it (1.40 +/-
0.47 vs 1.59 +/- 0.63): given the equation it gets the heat polynomials (0.99-1.00) but loses d_t
(0.60) and scaling (0.41). (P4) exact-operator rows: KdV missing 0.01, spurious 0.01, novel 0.99-1.00
-- MET; heat under the full convention 0.00 / 0.00 / 1.00 for all ten -- MET; under the finite
convention its spurious 4.00 are the four heat polynomials, by the protocol's definition.
FOUND IN THE TABLES, stated with them:
(a) Augerino's row scores its DICTIONARY, not what it learned. It returns six fixed affine fields
and u d_u, each scaled by its learned width; a span metric is blind to the widths (they never reach
zero: softplus), so the span is the dictionary on every seed (std 0.00-0.01). Checked: the same
dictionary with arbitrary widths reproduces the heat row to two decimals (2.00 / 3.00, 2.20 + 2.47,
novel 0.98 / 0.99 / 0.11 / 0.01 / 0.95 / 0.00). Its d_x, d_t and heat scaling come from the
dictionary. By Augerino's own reading (a width that grows marks an invariance) it found u d_u on
heat (width 1.2-2.2), t-translation on Burgers (0.15-0.18) and nothing on KdV (all <= 0.01). exp42
now prints the widths and a post hoc row "grown widths only". PREDICTED for that row, before
rerunning exp42: Burgers {d_t}: missing 4.00; KdV {}: 4.00; heat {u d_u}: 5.00 finite, 9.00 full.
The protocol row stays the pre-registered one; in the paper it reads as an affine-dictionary null
that FINS beats on every equation.
(b) LIG was capped at four generators: method_lig takes min(cfg.n_sym, 4), Ko et al.'s PDE setting
(three true plus one spare), although PROTOCOL.md says n_sym = k. The cap binds on Burgers (floor 1
missing) and heat (floor 2 finite, 6 full); LIG misses 1.1, 1.45 and 0.9 dimensions above those
floors, and 2.90 of 4 on KdV where the cap does not bind, so the cap is not what limits it. exp42
now prints the floor k_B - rank beside the missing dimensions. Heat s44: LIG's L_ortho stayed at
1.573 from epoch 5 on (0.008-0.03 on the other seeds) -- a degenerate run, kept.
(c) Hungarian cosine is basis-dependent: the exact operator, whose span is exact, scores 0.57
(heat, finite), 0.71 (heat, full), 0.83 (KdV); LieNLSD's sparse, near-textbook generators score
0.79-0.91. Cosine measures agreement with the textbook basis, not recovery -- why the protocol
makes the basis-free counts primary.
(d) Novel containment is strict for isolated finds: a method that returns u d_u alone on heat
scores 0.11 on it (its share orthogonal to Galilean and projective), though the missing count
credits it with one full dimension. Missing dimensions stay the primary metric.
LieNLSD, given the equation, loses scaling on every equation (novel 0.33-0.46) and returns fewer
generators than the class dimension on KdV s42/s44 and Burgers s44; exp42 now shows k as a range.
FINS criterion lines as in part 1; exact heat: largest gap after the sixth eigenvalue (the finite
algebra, lambda7 / lambda6), then after the tenth (lambda11 / lambda10 0.8-3.0e4); exact KdV: after the
second (lambda3 / lambda2); learned KdV: after the fourth.

**Protocol run, part 2: baseline training (Colab, s = 42-44, pre-registered configurations).**
Fast baselines complete on all nine runs (three equations x three seeds). Wall time per run:
LieNLSD ~65 s, LieGAN ~130 s, LaLiGAN 450-535 s, Augerino ~730 s (heat, Burgers) and 1165-1191 s
(KdV). Observed during training, reported as the baselines' behaviour, nothing retuned:
LaLiGAN's discriminator loss blows up around epochs 40-50 in most runs (GAN instability; its
generators are read out at the last epoch as the published code does; stable only on heat s43 of
the eight runs logged here, the others blow up at epoch 40 or 50). Augerino's widths: heat -- the
seventh generator's width grows to 1.2-2.2 while the other six shrink to <= 0.02; Burgers -- the
second grows to 0.15-0.18, the rest <= 0.02; KdV -- all seven shrink to <= 0.01 (no invariance
learned). LieNLSD (equation given, threshold 0.01) returns fewer generators than the class
dimension on KdV s42 and s44 (3 of 4) and Burgers s44 (4 of 5); ten of ten on heat. LIG, 10 epochs
(cap above): Burgers s42 3420 s, heat s42 3405 s, KdV s42 7184 s; L_sym flat over the last five
epochs on all three; s43 and s44 follow (~8 h more).
exp42: a seeds column (LIG rows may average fewer seeds than the others) and a guard that scores
non-finite generators as zero with a warning. P2 involves no equation-aware method and can be
decided from the fast baselines alone.

**Protocol run, part 1: FINS rows (Colab, s = 42-44, ten unseen trajectories, no filter).**
                    band u_t   losses (algebra)                                    lambda_{k+1}  worst   inside < 0.9
  Burgers s42       2.5e-2 F   d_x 9.6e-4 d_t 1.0e-3 Gal 5.2e-4 sc 6.7e-4 pr 3.4e-4  1.3e-3        1.3x    d_t 0.83
  Burgers s43       5.7e-3     d_x 3.3e-4 d_t 3.3e-4 Gal 3.3e-4 sc 3.1e-4 pr 2.6e-4  9.4e-4        2.8x    none
  Burgers s44       8.9e-3     d_x 4.2e-3 d_t 4.5e-4 Gal 1.8e-3 sc 3.3e-3 pr 1.3e-3  3.5e-3        FAIL    d_x 0.83
  heat s42 (L = 1)  1.1e-27    finite six 3.0-7.1e-4, b 2.8-2.9e-3                  4.1e-3        1.4x    d_x 0.90, b=1 0.85
  heat s43          4.6e-28    finite six 0.8-3.5e-4, b 3.3-4.8e-4                  1.4e-3        2.9x    none
  heat s44          9.2e-28    finite six 0.3-3.1e-4, b 4.3-5.4e-4                  1.0e-3        1.9x    b=1 0.86
  KdV s42-44        1.6e-16 .. 2.1e-15   as in (K1): scaling 1.9e-5 / 3.9e-5 / 1.9e-5     1.7e-2 / 1.4e-2 / 1.3e-2   890x / 360x / 680x
  exact heat 1042-1044: all ten <= 6e-8, lambda11 1.2e-3 / 9.7e-4 / 4.0e-4, blind readout of all ten textbook
  exact KdV 1042-1044: as in (K1)
Burgers s44 fails on d_x (4.2e-3 > lambda6 3.5e-3): without the filter its two steep trajectories
(one-step error 0.064, 0.092) stay in -- the data-1044 effect known since the window stepper. Burgers
data 1042 is FLAGGED by the band check (u_t share 2.5e-2 on ten trajectories; 4.9e-3 on the five
measured earlier) yet passes; the flag is a warning, not a predictor. Heat at one domain length
reads eight of ten blind on s43 (the superpositions, u d_u, Galilean, projective, and scaling
xi = 0.5 x, eta = t); d_x containment 0.90-0.96 (multi-scale: 0.58-0.84).
Verdicts (P1): Burgers all five below lambda6 on 2/3 -- MET (>= 2/3); heat all ten below lambda11 on
3/3 -- MET; KdV margin >= 100x on 3/3 -- MET. P2-P4 wait for the baselines and exp42.
LIG COST: 342 s per epoch on Colab with the torchdiffeq adjoint (Burgers s42, 1024 trajectories,
batch 4), i.e. 7.6 h for its 80 epochs, ~70 h for nine runs. PRE-REGISTERED compute cap, before any
LIG result is seen: LIG runs 10 epochs (its schedule scales: lr drop at 5, Sobolev term from 8);
on Burgers s42 its losses had flattened by then (L_sym -13.7 .. -13.97 over epochs 4-11, L_ortho
0.12 -> 0.06); reported as "LIG, 10 epochs"; seed 42 on every equation first, 43 and 44 as time
allows. exp41 now resumes: methods already in the output file are skipped, new ones are added.

**Frozen protocol v1 (PROTOCOL.md): code, smoke test, baseline bugs, pre-registration.**
Code: exp40 --only-L (heat, one domain length; the benchmark cycles five, 5 n are drawn) and --save;
exp39 --save, its data builders factored (grid, training_data, evaluation_data -- results
unchanged, checked on data 1042); band check printed by both (e31.band_shares); exp41_baselines.py
(every baseline on the protocol's trajectories, fields on the ten evaluation trajectories, saved
after every method); exp42_tables.py (one region, one scaling, one metric for every method).
Metric, changed during the smoke test: plain containment credited LIG's FOUR generators with EIGHT
of heat's ten -- on a band of x the ground-truth fields are nearly dependent (singular values of
the ten normalised fields down to 0.007, on data and on (x, t, u) probes alike: 1, x, x^2, x^3 on
[0.2, 0.8], and at alpha = 0.01 Galilean ~ x u d_u, projective ~ x^2 u d_u). Now: missing and
spurious dimensions (k_B - ||Qa'Qb||_F^2, rank_A - ||Qa'Qb||_F^2) and NOVEL containment (the part of
each generator orthogonal to the others); Grassmann and cosine kept as secondary. Checked: the
exact operator scores missing 0.00, spurious 0.00, novel containment 1.00 for all ten heat members
and 0.01 / 0.01 / 0.99-1.00 on KdV.
BASELINE BUGS found while wiring them (fixed behind options; the published configuration still
runs): (1) LieNLSD returns the t-component as xi and the x-component as eta -- in its prolongation
the block multiplying -u_t is tau; every earlier LieNLSD score was computed on swapped fields.
(2) LieNLSD and LIG used the textbook Burgers equation (known, above) and LIG's KdV and Burgers
residuals ignore its own normalisation of x, t, u (u_t + 6 u u_x + u_xxx in normalised variables);
only heat was right. (3) LieNLSD's degree-2 library cannot represent heat's projective (x^2 u) or
x^3. run_lienlsd_general: symbolic library of any degree (checked identical to the hand-written
degree-2 derivatives), the data's equation, components in ground-truth order, column-scaled SVD;
cfg.lig_eq: LIG's coefficients in its normalised variables. LIG and LieNLSD are marked "uses the
equation" in the tables.
Smoke test (CPU, 16 training trajectories, 1-2 epochs, 3 evaluation trajectories, s42): all three
equations run end to end; LIG needs ~6 GB at batch 1 on KdV's 128 x 128 grid without torchdiffeq
(killed on CPU; ran on heat and Burgers).
PRE-REGISTERED for the protocol run (Colab, s = 42-44, 10 unseen trajectories, no filter):
(P1) FINS criterion: Burgers all five below lambda6 on >= 2/3 seeds; heat (one length, smooth) all
ten below lambda11 on >= 2/3; KdV all four below lambda5 with margin >= 100x on 3/3.
(P2) Against the equation-free baselines (LieGAN, LaLiGAN, Augerino): FINS has fewer missing
dimensions than each of them on every equation (mean over seeds, finite ground truth).
(P3) No prediction against the equation-aware LieNLSD and LIG.
(P4) Exact-operator rows: missing and spurious <= 0.05, novel containment >= 0.95 for every
member (heat, KdV).

**Heat, horizon at a fixed scored region: results (seeds 42, 43; Colab session ended before s44
and before the larger CNO's K = 64 on s43).** Scored centre 192 points, no filter, smooth data.
                      rel. change  d_x      d_t      max algebra (member)   lambda10  lambda11  l11/l10  inside < 0.9
  exact  1042 K16     0.13         1e-15    1.6e-8   1.6e-8 (d_t)           1.8e-8    2.7e-5    1.5e3    none
  exact  1042 K64     0.35         2e-16    1.3e-9   1.3e-9 (d_t)           1.5e-9    1.6e-5    1.1e4    none
  exact  1043 K16     0.17         2e-15    1.3e-8   1.3e-8 (d_t)           1.5e-8    1.8e-5    1.2e3    none
  exact  1043 K64     0.45         3e-16    1.1e-9   1.1e-9 (d_t)           1.3e-9    1.0e-5    7.7e3    none
  small  s42 K16      0.13         2.7e-4   2.6e-4   1.7e-3 (b=1)           1.1e-3    1.4e-3    1.27     b=1 0.60, b=x 0.60, b=x2 0.78, b=x3 0.89, sc 0.89
  small  s42 K64      0.35         7.6e-5   3.1e-5   6.1e-4 (b=x)           4.1e-4    5.5e-4    1.34     b=1 0.60, b=x 0.61, b=x2 0.65, b=x3 0.88
  small  s43 K16      0.17         2.6e-4   1.9e-4   4.7e-4 (scaling)       7.1e-4    7.9e-4    1.11     d_x 0.59, sc 0.64
  small  s43 K64      0.45         7.0e-5   3.7e-5   1.6e-4 (scaling)       1.9e-4    3.5e-4    1.84     d_x 0.64, sc 0.68
  big    s42 K16      0.13         5.2e-5   1.2e-4   5.8e-4 (b=x2)          5.6e-4    8.3e-4    1.48     b=x 0.71, b=x2 0.68, b=x3 0.76
  big    s42 K64      0.35         1.4e-5   2.8e-5   2.9e-4 (b=x3)          2.0e-4    3.5e-4    1.75     b=x 0.76, b=x2 0.70, b=x3 0.68
  big    s43 K16      0.17         1.1e-4   4.6e-5   2.4e-4 (scaling)       3.9e-4    4.8e-4    1.23     d_x 0.58, sc 0.65
Verdicts: relative change >= 2.5x MET (2.7x, 2.6x); lambda11 / lambda10 >= 2x through a CNO FAILED
(small 1.06x, 1.66x; big s42 1.18x); containment >= 0.9 for all ten FAILED on every run; exact
lambda11 does not fall FAILED (2.7e-5 -> 1.6e-5, 1.8e-5 -> 1.0e-5). The small CNO's clauses are
decided on two seeds; the larger CNO's would need s43 and s44 both to pass.
Reading: the longer horizon moves the characteristics 2.7x further and lowers the learned
operators' algebra losses 2-4x (d_x 3.5x, d_t 5-8x), but lambda11 falls with them, so the band
barely widens. Two further observations. (1) Through the exact operator heat's nearest
non-symmetry sits at 1-3e-5 on this 192-point region (1.3e-3 on the 64-point region): the scored
region, not the horizon, moved it most. (2) Through a CNO lambda11 is 10-50x ABOVE the exact one
(3.5e-4 .. 1.4e-3 vs 1-3e-5): the operator's error lifts every direction, symmetries by five
decades and non-symmetries by one or two, compressing the whole spectrum into 1e-5 .. 1e-3.
The eleventh eigenvalue through a CNO is the operator's floor, not heat's near-symmetry. Heat is
closed as measured: the necessary condition holds on every run, full discovery through a learned
operator is out of reach at this accuracy; the horizon is not the lever.

**Heat, horizon at a fixed scored region: pre-registration (eval only, existing checkpoints).**
The torus FNO heat runs used K = 64 (the slowness was found then: "longer horizons help
monotonically"); every window run used K = 16 because of the window stepper's 6 K margin. Heat's
step is sigma^2 = alpha dt / dx^2 = 0.32 grid units per save (std 6.4 points after 64 saves), so a
32-point margin keeps the outside at > 5 sigma after 64 saves. Scored region fixed at 192 points:
exact heat kernel (r = 6 per step) on 6-period windows at K = 16 and 15-period windows at K = 64;
CNO small _smooth and _smooth_big800 on their 4-period windows with --score-margin 32 at K = 16 and
K = 64; smooth data 1042-1044, no reliability filter.
PREDICTED if slowness binds: the relative change of the characteristics rises >= 2.5x from K = 16 to
K = 64; through the CNO lambda11 / lambda10 at K = 64 >= 2x its K = 16 value and containment >= 0.9 for
all ten on >= 2/3 seeds for at least one CNO; through the exact operator lambda11 does not fall.

**What is hard about heat (diagnosis, exact operator; no learned operator involved).**
New printout in exp39/exp40: the rms relative change of the basis characteristics over the horizon,
sqrt(tr B'B / tr K Q0'Q0) -- the scale every loss is normalised by (mu_rel = 1). K = 16:
  heat (smooth, windows, data 1042) 0.13;  KdV (periodised solitons, data 1042) 0.39;  Burgers (windows, data 1042) 1.1
The same operator error therefore costs ~(1.1 / 0.13)^2 ~ 70x more in heat's loss than in Burgers',
~9x more than in KdV's.
Heat's nearest non-symmetry is cheap even through the exact operator: lambda11 = 3e-4 .. 1.3e-3 on
the smooth sets (KdV lambda5 1.4-1.7e-2). Read at m = 11, the eleventh direction (loss 3.2e-4) is
xi = 0.25 x^2, eta = xt - 0.12 x^3 - 0.14 x^2, phi = 0.25 x u^2 + ... -- the same family as the
"x^2-modulated scaling" of the CNO runs. CORRECTION: that direction is not a CNO artifact, as written
above; it is heat's own nearest non-symmetry, which the exact operator places just above the
algebra and the CNO's floor merges with it. Likely mechanism (not tested term by term): for a
linear equation on smooth, slow data, nonlinear or modulated fields violate the transport condition
only through terms like alpha u_x^2 t (phi = u^2: its defect is -2 alpha u_x^2), small over the
horizon.
Heat is linear and slow: its characteristics move little over the horizon (inflating any operator
error) and its non-symmetries are nearly transported (deflating lambda11). The exact operator
separates the two only because its algebra sits at 1e-8; the CNO's floor (1e-4 .. 1e-3) lands
between them. Capacity lowered that floor 2-5x; ~10-100x would be needed.
Horizon: K = 32 on eight-period windows (scored centre 128 points): relative change 0.21, but
lambda11 fell to 1.0e-4 (algebra <= 4e-9). Region and horizon changed together, so the test is
inconclusive; a longer horizon does not obviously widen the band.

**Heat through a more accurate CNO: results (H3, H3u).**
  s42 big800, filter kept 8/10: held-out 0.014 (small _smooth 0.022); rollout u 0.017 / 0.003, u_x 0.031 / 0.007
  (small 0.032 / 0.007, 0.057 / 0.012); edge leak 7.2e-10
  d_x 5.1e-5  d_t 1.2e-4  u d_u 2.7e-5  Gal 3.1e-5  scaling 8.2e-5  proj 3.1e-5  b=1 2.0e-4  b=x 2.5e-4  b=x2 2.2e-4  b=x3 2.1e-4
  lambda10 6.8e-4, lambda11 1.5e-3 (ratio 2.2; small 1.4); worst margin 6.0x (small 2.5x); containment >= 0.98 for all ten
  (small: d_x 0.84, b=x 0.88)
  blind: b = x^2 u, x^3, x^2, xu, u, x, 1 as dominant monomials; scaling now recognisable (xi = 0.503 x,
  eta = t, plus small terms); d_x only inside a d_t vector (eta = 1, xi = 0.546); the spurious
  x^2-modulated scaling persists (xi = 3.24 + 0.251 x^3, eta = x^2 t - ...)
  verdicts s42: held-out <= 0.012 FAILED (0.014); lambda11 / lambda10 >= 3 FAILED (2.2);
  containment >= 0.9 MET; worst margin >= 3x MET (6.0x).
  H3u s42, the same ten trajectories, no filter:
          d_x     d_t     u d_u   Gal     scaling proj    b=1     b=x     b=x2    b=x3    lambda11 l11/l10 worst  inside < 0.9
    small 2.7e-4  2.6e-4  2.1e-4  1.9e-4  5.9e-4  1.6e-4  1.7e-3  1.4e-3  1.0e-3  9.0e-4  2.7e-3   1.35    1.6x   d_x 0.47, b=1 0.72
    big   5.2e-5  1.2e-4  7.7e-5  8.5e-5  8.3e-5  9.2e-5  3.5e-4  4.2e-4  4.5e-4  1.1e-3  1.6e-3   2.0     1.45x  none (>= 0.96)
    ratio 5.2x    2.2x    2.7x    2.2x    7.1x    1.7x    4.9x    3.3x    2.2x    0.8x
  blind, big: d_x (xi = 1), scaling (xi = 0.505 x, eta = t), Galilean, projective, u d_u and the
  superpositions 1, x, x^2, x^3 as their dominant monomials -- 9 of 10; d_t only as eta = 0.55
  inside the d_x vector; the spurious x^2-modulated scaling takes the tenth slot. Small: 7 of 10.
  verdict H3u s42: d_x >= 3x lower MET (5.2x), d_t FAILED (2.2x).
  The larger operator fixes the subspace (containment >= 0.96 on the full set; small: d_x 0.47)
  and lowers most losses 2-7x, but b = x^3 + 6 alpha x t does not improve (1.1e-3) and on the
  full set sets the margin at 1.45x; the filtered margin (6.0x) comes from dropping the two
  trajectories that carry it. The margin depends on the evaluation set; the containment does not.
  s43 big800, filter kept 10/10 (filtered = unfiltered): held-out 0.013 (small 0.018); rollout u 0.009 / 0.002,
  u_x 0.017 / 0.005; d_x 1.1e-4  d_t 4.6e-5  u d_u 5.6e-5  Gal 1.2e-4  scaling 2.8e-4  proj 2.1e-4  b=1 1.3e-4
  b=x 1.8e-4  b=x2 2.5e-4  b=x3 3.4e-4; lambda10 7.3e-4, lambda11 1.0e-3 (ratio 1.4); worst 2.9x; containment d_x 0.69
  s44 big800: training diverged at epoch ~320 (train loss 7e-5, held-out 0.47) and never recovered; best state
  0.028 (small 0.021); filter kept 9/10: d_x 1.6e-4  d_t 1.3e-4  u d_u 2.9e-5  Gal 3.5e-5  scaling 1.8e-4
  proj 4.8e-5  b=1 1.9e-4  b=x 2.3e-4  b=x2 2.7e-4  b=x3 3.5e-4; lambda10 4.3e-4, lambda11 5.6e-4 (1.3); worst 1.6x;
  containment d_x 0.59
  H3u, same ten trajectories:
              d_x     d_t     u d_u   Gal     scaling proj    b=1     b=x     b=x2    b=x3    lambda11 l11/l10 worst  inside < 0.9
    s43 small 2.6e-4  1.9e-4  1.1e-4  1.4e-4  5.3e-4  1.9e-4  3.0e-4  4.1e-4  5.5e-4  6.8e-4  1.5e-3   1.36    2.2x   d_x 0.60
    s43 big   1.1e-4  4.6e-5  5.6e-5  1.2e-4  2.8e-4  2.1e-4  1.3e-4  1.8e-4  2.5e-4  3.4e-4  1.0e-3   1.4     2.9x   d_x 0.69
    s44 small 9.0e-5  1.0e-4  1.9e-5  2.4e-5  1.0e-4  3.2e-5  2.2e-4  3.9e-4  4.9e-4  5.7e-4  6.6e-4   1.14    1.16x  b=x 0.84, b=x2 0.57, b=x3 0.66
    s44 big   1.6e-4  1.3e-4  3.1e-5  4.0e-5  1.8e-4  5.9e-5  2.1e-4  3.3e-4  4.5e-4  6.0e-4  6.8e-4   1.17    1.13x  b=x2 0.86, b=x3 0.68
  blind, big: s43 seven of ten (the superpositions, u d_u, Galilean, projective; d_t mixed; d_x and scaling
  absent; two spurious slots); s44 fewer; the x^2-modulated scaling appears on every seed (heat's own
  nearest non-symmetry, see the diagnosis above).
Verdicts H3 (filtered, as pre-registered): held-out <= 0.012 FAILED 0/3 (0.014, 0.013, 0.028); lambda11 /
lambda10 >= 3 FAILED 0/3 (2.2, 1.4, 1.3); containment >= 0.9 FAILED 1/3 (s42 only); worst margin >= 3x
FAILED 1/3 (6.0x, 2.9x, 1.6x). H3u: d_x and d_t >= 3x lower FAILED (s42 d_x only, s43 d_t only, s44
neither -- the larger operator is worse there, its training diverged).
Reading: capacity is not the lever that closes heat. Where training was stable, 6.4x the parameters
and twice the epochs gave 1.4-1.6x lower held-out error and 2-5x lower losses on most members, a
correct subspace on one seed of three, and no spectral gap on any (1.3-2.2; exact 1e4). One seed
diverged (no gradient clipping). The spurious x^2-modulated scaling survives every CNO. The heat row
stays as measured with the small CNO: the necessary condition holds on every run (6/6, plus 3/3
here), discovery is partial, and the exact operator on the same data shows that the method itself
finds the whole algebra.

**Heat through a more accurate CNO: pre-registration (Colab, unseen data 1000 + s, --smooth).**
(H3) CNO with channel_multiplier 32 and 4 residual blocks (672k), 800 epochs, trained and
evaluated on smooth data (tag _smooth_big800), s = 42-44, filter 2. Rationale: on Burgers the
same change took the held-out error from 0.018 to 0.009. PREDICTED: held-out <= 0.012;
lambda11 / lambda10 >= 3 on 3/3 (small CNO 1.3-1.4); containment >= 0.9 for all ten on 3/3
(small: d_x 0.58-0.84); all ten below lambda11 with worst margin >= 3x (small 1.6-2.5x).
(H3u) the same ten trajectories for both operators (no reliability filter, fixing the protocol
defect): small _smooth and _smooth_big800 re-evaluated. PREDICTED: d_x and d_t >= 3x lower
through the larger CNO on every seed.

**Band-limited data, learned operators: results (Colab, unseen data 1000 + s).**
(K1) KdV, torus FNO stepper trained on --train-ic mixed (400 epochs), evaluated on --ic periodic, filter 2.
               held-out d_x      d_t      Gal      scaling  lambda5  margin inside d_t  blind terms d_x/Gal/scaling/d_t
  exact 1042   --       3.2e-16  6.7e-11  7.4e-16  2.1e-5   1.7e-2   810x   1.00        1 / 2 / 2 / 1
  exact 1043   --       2.7e-16  4.7e-11  6.0e-16  2.6e-5   1.5e-2   580x   1.00        1 / 2 / 2 / 1
  exact 1044   --       3.0e-16  5.6e-11  6.7e-16  2.6e-5   1.4e-2   540x   1.00        1 / 2 / 3 / 1
  s42 10/10    0.007    6.2e-7   9.5e-7   5.5e-6   1.9e-5   1.7e-2   890x   0.99        1 / 2 / 3 / 8
  s43 10/10    0.011    9.1e-6   1.3e-5   3.3e-5   3.9e-5   1.4e-2   360x   0.80        4 / 5 / 5 / 14
  s44 10/10    0.006    9.2e-7   1.3e-6   3.6e-6   1.9e-5   1.3e-2   680x   0.97        3 / 4 / 4 / 11
  scaling read blind, xi per phi = u: exact -0.495 / -0.478 / -0.505 x; stepper -0.541 / -0.492 / -0.529 x
  Galilean read blind: exact xi = 6 t; stepper 6.47 / 5.62 / 5.96 t (phi = 1)
  one-step error on the evaluation set 0.002-0.003 (s42, s44), 0.004-0.007 (s43)
Verdicts: held-out <= 0.012 MET; margin >= 100x MET 3/3; containment >= 0.9 FAILED on s43
(d_t 0.80), 2/3; readout d_x, Galilean, scaling <= 4 terms MET on s42 and s44, FAILED on s43
(Galilean and scaling 5 terms each); d_t never clean, as expected. Previous setting
(non-periodised solitons, soliton-only training): d_t 0.11-0.15, scaling 0.04, 0/3. The
stepper's scaling loss equals the exact operator's (1.9-3.9e-5 vs 2.1-2.6e-5) and its margins
match (360-890x vs 540-810x). The weakest seed is the least accurate operator (0.011 vs
0.006-0.007). d_t's readout picks up spurious xt, x, t^2 terms: the four losses lie within one
to two decades of each other, so varimax rotates within a near-degenerate bottom, whereas
through the exact operator they are separated by 5-10 decades and read clean.

(H0) exact heat-kernel window operator, --smooth data, n = 10:
  data 1042: all ten <= 1.6e-8 (d_t, time differences), lambda11 1.3e-3; data 1043: <= 1.3e-8, lambda11 5.4e-4;
  data 1044: <= 7.4e-9, lambda11 3.3e-4; gap lambda11 / lambda10 1e4 on every set; containment 1.00; blind: all ten single monomials (phi = 1, x, x^2, x^3, u, xu, x^2 u; xi = 1;
  eta = 1), scaling xi = 0.5 x, eta = t -- the whole heat algebra read blind, textbook.
(H1) CNO trained on benchmark data, evaluated on --smooth data, filter 2 (kept 9/10 each).
        d_x     d_t     u d_u   Gal     scaling proj    b=1     b=x     b=x2+.  b=x3+.  lambda11 l11/l10 worst  inside < 0.9
  s42   2.8e-4  1.9e-4  1.9e-4  1.1e-4  5.5e-4  5.5e-5  1.7e-3  1.6e-3  1.3e-3  8.3e-4  3.5e-3   1.2     2.1x   b=x2 0.85
  s43   4.0e-4  2.5e-4  1.2e-4  1.8e-4  5.6e-4  2.6e-4  3.7e-4  4.5e-4  5.7e-4  7.2e-4  1.1e-3   1.1     1.5x   d_x 0.63
  s44   6.3e-5  8.9e-5  9.0e-6  1.2e-5  1.0e-4  1.7e-5  1.7e-4  2.1e-4  2.2e-4  2.7e-4  5.1e-4   1.2     1.9x   d_x 0.61
  benchmark data (same checkpoints): d_x 1.5e-3 / 1.7e-3 / 1.2e-4, d_t 1.6e-3 / 4.1e-3 / 3.7e-4
  one-step error median 0.026 / 0.022 / 0.019; rollout u_x 0.058 / 0.056 / 0.045 (1 step)
(H2) CNO trained and evaluated on --smooth data, filter 2.
        held-out d_x     d_t     u d_u   Gal     scaling proj    b=1     b=x     b=x2+.  b=x3+.  lambda11 l11/l10 worst  inside < 0.9
  s42   0.022    2.6e-4  2.5e-4  1.8e-4  1.7e-4  5.9e-4  1.5e-4  1.6e-3  1.7e-3  1.4e-3  1.0e-3  4.2e-3   1.4     2.5x   d_x 0.84, b=x 0.88
  s43   0.018    2.6e-4  1.9e-4  1.1e-4  1.4e-4  5.3e-4  1.9e-4  2.6e-4  4.0e-4  5.5e-4  6.8e-4  1.5e-3   1.4     2.2x   d_x 0.58
  s44   0.021    9.0e-5  1.0e-4  1.6e-5  1.6e-5  1.0e-4  1.8e-5  1.7e-4  2.2e-4  2.6e-4  3.1e-4  5.0e-4   1.3     1.6x   d_x 0.78
Blind (H1, H2): b = 1, x, x^2, x^3, u d_u, Galilean (phi = xu), projective (phi = x^2 u) read as
single monomials or dominated by them (5-7 of the seven per run); translations and scaling mixed,
and an x^2-modulated scaling (xi ~ 0.19 x^3 - 0.7 x, eta ~ x^2 t - 1.3 t) appears in every CNO run
(corrected later: it is heat's own nearest non-symmetry, the exact operator's eleventh direction).
Verdicts: H0 MET (whole algebra, gap 1e4, clean readout, 3/3). H1: median one-step error <= 0.02
FAILED (1/3); d_x and d_t >= 5x lower FAILED (3 of 6; s44's benchmark losses were already low);
all ten below lambda11 on >= 2/3 MET (3/3). H2: all ten below lambda11 MET (3/3); containment
>= 0.9 for all ten FAILED (3/3, d_x 0.58-0.84).
Reading: band-limiting the evaluation data lowers the translations 2-16x and puts the whole
algebra below lambda11 on 6/6 runs; retraining on smooth data changes nothing -- the
evaluation data were the problem, as for KdV. But through the CNO the spectrum has no gap
(lambda11 / lambda10 = 1.1-1.4; exact 1e4): several non-algebra directions lie as low as the
algebra, d_x is only partly inside the ten lowest, and the rank cannot be read off. The
necessary condition holds; discovery is limited by the operator's floor (1e-5..2e-3, set by
the superpositions b and scaling), which sits a factor 1.5-2.5 below lambda11.

**Band-limited data, learned operators: pre-registration (Colab, unseen data 1000 + s).**
(K1) KdV: stepper trained on --train-ic mixed, 400 epochs, s = 42-44, evaluated on --ic periodic,
filter 2; exact operator on the same data. PREDICTED: held-out <= 0.012; all four below lambda5
with margin >= 100x on 3/3; containment >= 0.9 for all four; blind: d_x, Galilean, scaling
<= 4 terms on 3/3 (d_t not predicted clean).
(H1) heat: the existing CNO checkpoints (trained on benchmark data) evaluated on --smooth data.
PREDICTED: median one-step error <= 0.02; d_x and d_t losses >= 5x lower than on benchmark data;
all ten below lambda11 on >= 2/3. Risk: lambda11 falls too (exact heat-kernel operator on
smooth data 1042, n = 4: lambda11 8.6e-5, all ten <= 1.6e-8).
(H2) heat: CNO trained and evaluated on --smooth data (tag _smooth), 400 epochs, s = 42-44.
PREDICTED: all ten below lambda11 on 3/3; containment >= 0.9 for all ten.
(H0) reference: exact heat-kernel window operator on the same smooth data 1042-1044.

**Why KdV and heat fail through learned operators: time derivatives live where u does not.**
KdV, CPU reproduction (s42, 200 epochs, data 1042): d_t 0.16, scaling 4.8e-2, as on Colab.
(i) Jacobian per direction at two-soliton states vs the exact solver, error relative to the
exact change after 1 / 16 steps: shift of one soliton 0.065 / 0.022 and 0.028 / 0.016, global
shift 0.075 / 0.025, relative shift 0.043 / 0.017, amplitude dc 0.073 / 0.039, constant
0.061 / 0.055. Accurate: the hypothesis "wrong linearisation of relative motion or amplitude"
is REFUTED.
(ii) Wavenumber split: above k = 12, u carries 2.4e-7 of its energy, the stencil u_t 4.6e-2
(above k = 24: 2.0e-8 vs 4.3e-2). 16-step Jacobian error on u_t's k <= 12 part 0.047, on its
k > 12 part 0.99; that part also changes fastest (exact change 1.6x its norm), so it
dominates the residual.
(iii) Source: present at t = 0 and constant in time (two solitons: 3.8e-8 above k = 16; one
soliton at 0.4 L: 1.7e-9). exp39's sech^2 profiles were not periodised: the torus field jumps
by ~3e-2 at the seam and KdV radiates the jump as dispersive waves crossing the domain within
a save step (group velocity 3 k^2). The localised data carried a seam after all -- in u_t.
(iv) Evaluation data periodised (--ic periodic: the same solitons summed over images m = -2..2),
the SAME stepper: one-step error 0.006-0.009 (was 0.025-0.072); d_x 3.2e-5, d_t 8.8e-5, Gal
6.6e-5, scaling 1.0e-4, lambda5 6.0e-3: all four below, 60x. But d_t's containment is 0.19
and the readout is messy: lambda3, lambda4 = 1.6e-4, 9.1e-4 are spurious, u-independent
phi = x - 0.17 x^2 mixed with translations -- directions with a sawtooth at the seam that the
exact solver radiates and a stepper never trained on high wavenumbers does not.
Exact operator on the periodised data: d_x 3e-16, d_t 7e-11, Gal 5e-16, scaling 2.1e-5
(non-periodised 5.4e-4), lambda5 1.7e-2 (margin 800x); blind: all four clean, scaling
xi = -0.495 x per phi = u (textbook -0.5, was -0.40 with extra terms).
Heat, the same mechanism: benchmark data 1042, above k = 16 u carries 3.4e-6 of its energy,
u_t (central4) 0.44; per trajectory above k = 8: sines 0.00, Gaussians 0.53-0.77, poly / step /
combo 0.00-0.91 (seam jumps of non-periodic initial data, steps two grid points wide). The CNO's
per-trajectory one-step error follows it (sines 0.016-0.029, the rest up to 0.33). --smooth
data: u_t above k = 16 at 1e-27.
General statement: the characteristic of d_t is u_t, and time differentiation weights mode k
by omega(k) (k^2 heat, k^3 KdV), so modes invisible in u's L2 norm -- and in an L2 training
loss -- can dominate u_t. The criterion probes the operator where training is weakest. Data
requirement: the time derivative's content must lie in the band the operator resolves; initial
data incompatible with the domain (seam jumps) or under-resolved (steps at the grid scale)
violate it. Burgers (diffusive, no k^3) passes everywhere, consistent with this.
PRE-REGISTERED (CPU, s42, 200 epochs, stepper trained on --train-ic mixed: half periodised
solitons with backgrounds, half broadband humps; evaluation --ic periodic, data 1042): all four
below lambda5 with margin >= 10x; containment >= 0.9 for all four; blind: d_x, d_t, Galilean
<= 2 terms, scaling <= 4 terms.
Result: held-out 0.012, one-step error on the evaluation set 0.003-0.004, rollout 0.004 / 0.002
(1 / 16 steps); d_x 2.0e-6, d_t 2.3e-6, Gal 1.8e-5, scaling 2.0e-5, lambda5 1.7e-2 -- margin
850x (exact operator on the same data: 800x), jump x900 after the fourth eigenvalue;
containment 1.00 / 0.99 / 1.00 / 1.00. Blind: d_x xi = 1 (2 terms, the second 3e-3 x^2);
Galilean xi = 6.63 t, phi = 1; scaling xi = -0.513 x + 0.019 x^2, eta = -0.375, phi = u (4 terms,
d_t admixture); d_t eta = 1 mixed with Galilean and x terms (12 terms). Verdicts: margin MET
(850x), containment MET, readout MET for d_x, Galilean, scaling, FAILED for d_t (the four
losses lie within 1e-6..2e-5, near-degenerate, and varimax mixes d_t in).
Same seed, same budget, only the training data changed (non-periodised solitons -> half
periodised solitons, half broadband humps) and the evaluation data periodised: d_t 0.16 ->
2.3e-6, scaling 4.8e-2 -> 2.0e-5. A stepper that has seen high-wavenumber dispersion also
radiates a seam sawtooth, so the spurious phi = x directions of (iv) are gone. The KdV learned
operator now matches the exact one.

**Seam-free settings with learned operators: results (Colab, unseen data 1000 + s, filter 2).**
(1a) KdV, localised solitons, torus FNO stepper (exp39 --op stepper).
                    held-out  d_x      d_t      Gal      scaling  lambda5  inside-4 d_t / scaling  margin
  exact, data 1042  --        2.8e-16  1.5e-10  4.6e-15  5.4e-4   2.3e-2   1.00 / 1.00             43x
  exact, data 1043  --        2.5e-16  1.2e-10  3.8e-15  5.0e-4   2.3e-2   1.00 / 1.00             46x
  exact, data 1044  --        2.7e-16  1.4e-10  4.6e-15  6.0e-4   2.1e-2   1.00 / 1.00             35x
  stepper s42 10/10 0.041     8.5e-5   1.5e-1   6.6e-5   4.5e-2   2.0e-2   0.00 / 0.99             FAIL
  stepper s43 10/10 0.048     1.1e-4   1.1e-1   1.1e-4   4.1e-2   2.4e-2   0.00 / 0.99             FAIL
  stepper s44  9/10 0.040     8.1e-5   1.1e-1   6.2e-5   4.0e-2   2.2e-2   0.00 / 0.99             FAIL
Blind (m = 4). Exact: d_x, d_t, Galilean exact (1, 1, 2 terms); scaling 4-6 terms,
xi = -0.40 x per phi = u (eta = 3t lies below the display threshold: T = 0.05
makes its field small). Stepper: d_x (xi = 1) and Galilean (xi = 5.6-6.4 t,
phi = 1) clean on 3/3; no eta term in any vector (d_t not found); the other two
7-11 terms.
Verdicts: held-out <= 0.01 FAILED (0.040-0.048; broadband stepper 0.005-0.006);
all four below lambda5 FAILED 0/3 (d_t 5-7x above, scaling ~2x above); scaling
<= 3x d_x FAILED (~500x); readout: d_x, Galilean MET, d_t and scaling FAILED.
The generators that pass are the ones handed to the operator: d_x by the FNO's
shift equivariance, Galilean by the constant backgrounds in the training data.
d_t and scaling need the linearisation where solitons move relative to each
other or change amplitude. Hypothesis: a stepper trained on a ~5-parameter
family of exact soliton states fits values but not that linearisation; the
broadband stepper (random humps) passes d_t on every seed. Removing the seam
by localising the data narrowed the data below what the operator needs.

(1b) heat, four-period windows, small CNO (exp40 --pde heat --op cno), filter kept 8/10 each.
        held-out d_x     d_t     u d_u   Gal     scaling proj    b=1     b=x     b=x2+.  b=x3+.  lambda11  below
  s42   0.022    1.5e-3  1.6e-3  2.5e-5  1.4e-5  1.7e-4  7.1e-6  1.9e-4  1.9e-4  1.6e-4  1.1e-4  1.2e-3    8/10 (d_x, d_t)
  s43   0.016    1.7e-3  4.1e-3  9.7e-5  1.6e-4  5.0e-4  2.2e-4  3.1e-4  3.9e-4  4.8e-4  5.8e-4  2.6e-3    9/10 (d_t)
  s44   0.017    1.2e-4  3.7e-4  2.0e-5  2.6e-5  1.6e-4  3.3e-5  3.5e-4  4.2e-4  4.2e-4  4.9e-4  7.7e-4    10/10, 1.6x
  containment: s42 d_x 0.08, d_t 0.70; s43 d_x 0.77, d_t 0.68; s44 d_t 0.89; every other member >= 0.90
  edge leak 1.5e-11 / 7.6e-12 / 5.8e-12; one-step rollout error u 0.087 / 0.082 / 0.051, u_x 0.078 / 0.096 / 0.077
Verdicts: projective <= 3x d_x MET 3/3; b = x^3 + 6 alpha x t <= 3x d_x MET on
s42, s43 (0.07x, 0.34x), FAILED on s44 (4.1x, where d_x is 10x lower than on
the other seeds); containment >= 0.9 for all ten FAILED (translations only);
all ten below lambda11 FAILED, 1/3.
The torus pattern is inverted. The seam-bound members pass everywhere:
b = x^3 + 6 alpha x t (torus FNO: failed 2/5) 11x / 4.5x / 1.6x below lambda11,
projective 30-170x. The translations fail: through a convolutional operator a
grid shift is exact, so d_x's residual is the operator's differentiated rollout
error, and heat's small increments make it large (one-step u_x error 8-10
percent of the increment; Burgers CNO 5). The binding constraint is the
operator's accuracy on the high wavenumbers that u_x and u_t carry, not
geometry. Blind (m = 10): b = 1, x, x^2, x^3, u d_u, Galilean (phi = xu) and
projective (phi = x^2 u) come out as their dominant monomial, one or two terms
(their xi, eta scale with alpha and fall below the display threshold);
translations and scaling mixed.

(2) Burgers, larger CNO (672k), unseen data.
        held-out d_x     d_t     Gal     scaling proj    lambda6  worst  proj/d_x sc/d_x  blind projective
  s43   0.016    3.3e-4  2.1e-4  3.3e-4  2.1e-4  2.2e-4  8.6e-4   2.6x   0.67     0.64    xi = xt, eta = 0.994 t^2, phi = 0.159 x - 1.01 tu (4 terms, rest 0.02)
  s44   0.016    5.2e-4  1.7e-4  4.3e-4  5.8e-4  5.2e-4  1.0e-3   1.7x   1.00     1.11    xi = xt, eta = 0.914 t^2, phi = 0.158 x (-tu missing)
Verdicts: held-out <= 0.015 FAILED narrowly (0.016 both); all five below lambda6
and both ratios <= 3 MET on both; projective recognisable MET on s43 (textbook,
phi = x / 2 pi - tu), FAILED on s44 (-tu missing, as for the small CNO, the
window CNN and the exact torus operator on data 1044). Scaling appears as
S1 + 1.75-1.81 S2 (viscous scaling: S1 + 2 S2).

(3) Burgers, larger CNO, 800 epochs, s42, unseen data 1042, filter kept 9/10.
  held-out 0.009 (400 epochs 0.013; torus FNO 0.006-0.008); rollout u 0.016 / 0.003, u_x 0.040 / 0.008 (1 / 16 steps)
  d_x 7.9e-4  d_t 2.5e-4  Gal 3.6e-4  scaling 7.5e-4  proj 2.9e-4  lambda6 1.3e-3  worst 1.6x  proj/d_x 0.37  sc/d_x 0.95
  containment d_x 0.98  d_t 0.97  Gal 1.00  scaling 0.89  proj 0.99; edge leak 1.3e-10
  blind: xi = xt, eta = 1.03 t^2 - 0.289, phi = 0.152 x - 0.953 tu (5 terms); Galilean 3 terms; S1 + d_t admixture
Verdicts: held-out <= 0.012 MET (0.009); criterion and ratios MET. The margin did
not grow (2.1x -> 1.6x) although the operator is 30 percent more accurate and
its 16-step u_x error halved. The filter kept a trajectory the 400-epoch run
dropped, so the two runs score different sets -- PROTOCOL DEFECT: a filter
relative to the operator's own median makes the evaluation set
operator-dependent; comparisons across operators need one fixed set. On every
Burgers window run lambda6 ~ 1e-3 and the sixth readout vector carries the
inviscid time scaling S2 = t d_t - u d_u: the margin is bounded by that
near-symmetry (its loss grows as nu^2, section "split scaling"), not only by
the operator.

**Seam-free settings with learned operators for every equation: pre-registration.**
Burgers has a learned seam-free operator (CNO, window CNN); heat and KdV so far
only exact ones (heat-kernel windows, walls; localised solitons). Runs (Colab,
unseen data 1000 + s, filter 2):
(1a) KdV, localised solitons, torus FNO stepper (64 modes, no positional
embedding) trained on 1024 localised trajectories (exp39 --op stepper: one or
two solitons, c in [10, 16], central band, background 0 or U[-0.5, 0.5] --
exact travelling waves, the field stays constant near the seam), s = 42-44.
The seam is removed by the data, not the operator: on a constant field no
KdV characteristic jumps. Exact operator on the same data as reference.
PREDICTED: held-out increment error <= 0.01; all four below lambda5 on 3/3
seeds with worst margin >= 5x (broadband stepper 2.5-4.8x, set by scaling);
scaling <= 3x d_x (broadband stepper 13x); blind readout at m = 4: d_x, d_t,
Galilean with <= 2 terms each, scaling at least as clean as the exact
operator's on the same data (exact, data 42, n 6: 4 terms, rest 0.08,
xi = -0.401 x per phi = u; eta = 3 t is below the display threshold because
T = 0.05 makes its field small).
(1b) heat, four-period windows, small CNO (106k), s = 42-44. PREDICTED: no seam
penalty -- b = x^3 + 6 alpha x t and projective each <= 3x d_x on every seed;
containment >= 0.9 for all ten; all ten below lambda11 on 3/3 seeds (torus
FNO: 3/5, both failures b = x^3 + 6 alpha x t). The last is uncertain: the CNO's
floor on Burgers (3-8e-4) is close to heat's lambda11 on the torus (6e-4 to
4e-3).
(2) Burgers, larger CNO (672k), s = 43, 44. PREDICTED: held-out <= 0.015; all
five below lambda6 and projective/d_x, scaling/d_x <= 3 on both seeds; blind
projective recognisable (xt, t^2, x, tu present) on both, clean not predicted.
(3) Burgers, larger CNO, s = 42, 800 epochs. PREDICTED: held-out <= 0.012; the
criterion and the seam-free ratios as at 400 epochs.

**Operator ablation, follow-up: trained edge leak and a larger CNO (Burgers,
seed 42, unseen data 1042, filter 2).**
PRE-REGISTERED: trained edge leak <= 1e-8 for CNO and the window CNN, >= 1e-5
for the padded FNO; CNO with channel_multiplier 32 and 4 residual blocks per
level (672k parameters vs 106k) reaches held-out increment error <= 0.012 and
keeps projective/d_x and scaling/d_x <= 3.
Edge leak, trained models (rms response in the scored region to a unit impulse
at window index 0, max over K = 16 JVP steps):
  window CNN 3.6e-53   CNO 9.1e-12   padded FNO 4.5e-3
Larger CNO (`--op cno --cno-mult 32 --cno-res 4 --tag _big`):
  held-out 0.013 (small CNO 0.018); rollout error 1 step u 0.019, u_x 0.053;
  16 steps u 0.005, u_x 0.014; filter kept 8/10
  losses d_x 4.7e-4  d_t 1.4e-4  Gal 3.6e-4  scaling 3.9e-4  proj 2.5e-4;
  lambda6 1.0e-3; worst margin 2.1x (d_x); proj/d_x 0.53, sc/d_x 0.83
  containment d_x 1.00  d_t 0.99  Gal 1.00  scaling 0.93  proj 0.99
  blind: v1 xi = xt + 0.00327 x^2, eta = 1.02 t^2 - 0.455 t,
             phi = 0.158 x + 0.486 u - 0.933 tu (7 terms, rest 0.05)
         v2 xi = t, eta = 0.081 t, phi = 0.157 - 0.092 u (5 terms)
         v3 xi = x, eta = 0.79 t
Verdicts. Edge leak: MET (CNO 9e-12, window 4e-53, padded FNO 4.5e-3; nine
orders between the convolutional and the spectral global operator). Accuracy
<= 0.012: FAILED narrowly (0.013); 6.4x the parameters buys 28 percent, the
gap to the torus FNO (0.005-0.008) halves but stays. Seam-free ratios <= 3: MET
(0.53, 0.83). Criterion passes, span correct (containment >= 0.93).
Blind projective: less clean than the small CNO on the same data (4 terms).
The extra terms are -0.46 (0, t, -u), i.e. the inviscid time scaling
S2 = t d_t - u d_u (section "split scaling"); v3 = S1 + 0.79 S2 is the same
mixing. The larger operator, like the torus stepper, does not separate the
viscous scaling S1 + 2 S2 from its inviscid partner; scaling containment 0.93
is that tilt. Not a seam effect: it appears in the seam-free setting and was
known from the torus.

**Operator ablation, results (Burgers, unseen data 1042-1044, filter 2).**
                  held-out  d_x     d_t     Gal     scaling proj    lambda6  worst   proj/d_x sc/d_x  blind projective
  CNO s42         0.018     5.8e-4  2.7e-4  5.3e-4  4.5e-4  3.3e-4  1.2e-3   2.1x    0.57     0.78    xi = xt, eta = 1.04 t^2, phi = 0.155 x - 0.949 tu
  CNO s43         0.019     3.3e-4  3.3e-4  3.3e-4  3.1e-4  2.7e-4  9.4e-4   2.8x    0.82     0.94    xi = xt, eta = 0.984 t^2, phi = 0.159 x - 0.993 tu (rest 0.01)
  CNO s44         0.020     6.4e-4  2.6e-4  5.8e-4  7.8e-4  7.9e-4  1.1e-3   1.4x    1.23     1.22    xi = xt, eta = 0.897 t^2, phi = 0.157 x (-tu missing)
  FNO+pad s42     0.065     4.1e-3  6.7e-3  1.1e-3  7.8e-2  7.7e-2  2.2e-2   FAIL    19       19      not found
  FNO+pad s43     0.063     3.1e-3  3.8e-3  8.5e-4  4.4e-2  6.8e-2  1.6e-2   FAIL    22       14      not found
  FNO+pad s44     0.077     1.1e-2  5.8e-3  4.0e-3  1.7e-1  1.3e-1  3.1e-2   FAIL    12       15      not found
  window (ref.)   0.019-0.023                                                1.7x / 3.2x / 1.8x
  torus FNO, no positional embedding (exp31):  held-out 0.006 / 0.005 / 0.006 (grid: 0.008 / 0.007 / 0.008)
      d_x 6.5e-5 / 5.2e-5 / 1.6e-4 (grid 6.5e-5 / 4.6e-5 / 2.7e-4); scaling 2.1e-4 / 1.0e-4 / 6.3e-3 (grid 2.0e-4 / 1.5e-4 / 8.4e-3)
Verdicts. (A) no seam penalty: CNO 3/3, padded FNO 0/3 (12-22x) -- FAILED as
stated. (B) held-out <= 0.012 for one of them: FAILED (CNO 0.018-0.020, padded
FNO 0.063-0.077). (C) conditional on (B): not applicable; CNO's margins equal
the window stepper's. (D) d_x lower without the positional embedding on every
seed: FAILED (equal, higher, 1.7x lower); scaling within 2x: MET. Why (D) fails:
the trained grid-embedding FNO breaks shift equivariance by only 0.4-0.7
percent of its one-step increment (untrained: 3 percent) -- training suppressed
the coordinate dependence below the operator's own error. The earlier FINDING
is real for untrained models and harmless for trained ones; dropping the
embedding gains ~25 percent accuracy.
Mechanism. Non-periodicity is not enough: the zero-padded FNO has no seam yet a
seam-sized penalty. What matters is how the operator couples distant points.
Global spectral coupling (FNO on the torus, FNO with padding) spreads a
discontinuity of the x-weighted characteristics -- the wrap-around seam, or the
jump into the zero padding -- across the whole domain; convolutional operators
whose coupling decays with distance (CNO, the window CNN) do not. Edge leak
(exp40, new: rms response in the scored region to a unit impulse at the window
edge, max over K = 16 steps), 3-epoch smoke models: CNO 5e-13, padded FNO 2e-4;
exact heat kernel 0. Trained models (see follow-up): window CNN 3.6e-53, CNO
9.1e-12, padded FNO 4.5e-3. CNO's U-Net reaches the
whole window in principle, but its coupling decays like a local operator's.
Accuracy: at this budget CNO (105k parameters) matches the window stepper, not
the torus FNO (0.005-0.008). FNO's accuracy and its seam come from the same
global spectral representation; seam-free operators are so far ~3x less
accurate, and the criterion still passes with them (worst margins 1.4-3.2x).
Blind projective: CNO 2/3, the same data set (1044) failing as for the window
stepper and the exact torus operator.

**Operator ablation: periodicity vs locality (exp40 --op, exp31
--stepper-pos-emb).** The seam comes from periodic global mixing; our operators
so far confound two properties. Four corners on the same data, windows and
scored region (central 64 of a four-period window, filter 2, unseen data):
  periodic, global       FNO on the torus (exp31)
  non-periodic, global   CNO (`pde/cno1d.py`: the authors' CNO1d, Raonic et al.
                         NeurIPS 2023, github.com/camlab-ethz/ConvolutionalNeuralOperator,
                         MIT; batch norm off; port checked identical to the reference
                         with shared weights) and FNO with zero domain padding (0.25)
                         and no positional embedding -- `GlobalStepper`, same size,
                         trained on 256-point windows scored on [6k, W - 6k] at step k
  non-periodic, local    window stepper (valid convolutions, r = 6)
Checks: forward-mode JVP through CNO's antialiased bicubic activation agrees
with central differences to 6e-11 (padded FNO: 7e-7 at step 1e-4; forward
round-off ~1e-10); interior shift-equivariance error 2.5e-8 (CNO), 1.9e-4
(padded FNO, which sees the window edges globally). Module.double() leaves
neuralop's complex spectral weights in single precision: `to_float64`.
FINDING: every FNO stepper so far used neuralop's default positional_embedding
= "grid", i.e. x in [0, 1] as an input channel: shift-equivariance error on the
torus 2.6e-3 vs 7.3e-8 without it. The torus d_x results carry this.
`Stepper(positional_embedding=None)` / exp31 `--stepper-pos-emb none`.
Geometry-aware operators (Geo-FNO, GINO): in 1-D their FFT core runs on a
latent grid with padding, i.e. they reduce to the padded FNO; the geometry
part matters for irregular 2-D/3-D domains -- a later stage.

PRE-REGISTERED (Burgers, data seeds 1042-1044, filter 2, 1024 trajectories, 400
epochs): (A) CNO and padded FNO: projective and scaling <= 3x d_x's loss on every
seed. (B) At least one of them reaches held-out increment error <= 0.012
(window 0.019-0.023, torus FNO 0.008). (C) An operator meeting (B) has a larger
worst margin below lambda6 than the window stepper on every data seed (window:
1.7x, 3.2x, 1.8x). (D) Torus FNO without positional embedding (exp31, aligned,
4th order): d_x loss lower than the grid-embedding FNO on every seed (grid:
6.5e-5, 4.6e-5, 2.7e-4), at least 2x lower on two of three; scaling within 2x of
the grid FNO's (the seam floor dominates; grid: 2.0e-4, 1.5e-4, 8.4e-3).

**Unseen trajectories, Colab (data seed 1000 + s, filter 2): verdicts on the
pre-registration below.**
(1) Window stepper r = 6, Burgers (mu_rel = 1):
          kept  d_x     d_t     Gal     scaling proj    lambda6  worst   proj/d_x sc/d_x  blind projective
  s42     8/10  3.9e-4  1.7e-4  1.4e-4  2.0e-4  8.4e-5  6.5e-4   1.7x    0.22     0.51    xi = xt, eta = 1.02 t^2, phi = 0.158 x - 0.966 tu (4 terms, rest 0.06)
  s43    10/10  2.4e-4  2.0e-4  1.7e-4  2.0e-4  1.4e-4  7.7e-4   3.2x    0.58     0.83    xi = xt, eta = 0.973 t^2, phi = 0.159 x - 0.977 tu (4, 0.02)
  s44     8/10  4.1e-4  2.0e-4  2.4e-4  4.6e-4  3.1e-4  8.4e-4   1.8x    0.76     1.12    xi = xt, eta = 0.826 t^2, phi = 0.158 x (3, 0.08) -- -tu MISSING
No seam penalty (<= 3x d_x): 3/3 MET. All five below lambda6: 3/3 MET. Blind
projective with -tu within 10 percent: 2/3 (s44 FAILED). On data seed 1044 the
EXACT PERIODIC operator fails scaling (1.0e-2 vs lambda6 8.3e-3, margin 0.83x;
projective 1.1x) while the LEARNED WINDOW operator passes (scaling margin 1.8x,
projective 2.7x); scaling / d_x: 1.2e5 on the torus, 1.1 on windows. The runs
differ in normalisation (mu_rel 0 vs 1) and the window run dropped 2 of the 10
trajectories, so margins, not raw losses, are compared. A learned boundary-free
operator beats the exact periodic one on the x-weighted generators.
(2) FNO steppers:
  KdV (all four below lambda5): s42 scaling 1.1e-2 vs 9.0e-3 FAILED; s43 2.2x; s44 3.7x -> 2/3
  heat six finite below lambda11 (mu 0): 5/5 (worst margins 6.7x, 5.2x, 5.2x, 1.8x, 5.1x)
  heat all ten below lambda11 (mu 1): s43 2.0x, s45 3.2x, s46 1.35x; s42 and s44 FAILED,
      both on b = x^3 + 6 alpha x t only (1.0e-3 vs 8.8e-4; 1.7e-3 vs 1.5e-3) -> 3/5
  Burgers d_x, d_t, Galilean below lambda6: 3/3
  Burgers scaling and projective within 3x of the exact operator on the same data:
      s42 1.2x / 2.1x, s44 0.84x / 0.73x MET; s43 4.2x / 4.7x FAILED -- the exact floor
      there is 3.6e-5 / 5.1e-5, the stepper's 1.5e-4 / 2.4e-4, both >= 30x below lambda6:
      the clause fails when the geometric floor lies far below the stepper's own error,
      a badly designed criterion, reported as failed.
Every failure on the torus involves a seam-bound generator: KdV scaling, heat's
non-periodic b = x^3 + 6 alpha x t, Burgers scaling on data 1044. Translations,
d_t and Galilean pass on every unseen set. The quantitative table for the paper
should use seam-free settings: windows for Burgers, walls or windows for heat,
localised solitons for KdV (a local operator cannot represent KdV at the
benchmark's save step).
Progress bars: exp31-exp40 now set TQDM_DISABLE=1 by default before any import
that loads tqdm (torch does).

PRE-REGISTERED (Colab, unseen trajectories, data seed 1000 + s, filter 2):
(1) window stepper r = 6, Burgers s = 42, 43, 44: projective and scaling <= 3x
d_x's loss on every seed (no seam penalty); all five below lambda6 on every
seed; blind projective <= 5 terms, rest <= 0.1, its -t u coefficient within 10
percent of -1. (2) FNO steppers: KdV s = 42-44 all four below lambda5; heat
s = 42-46 six finite generators below lambda11 (exp31) and all ten below
lambda11 at mu_rel = 1 (exp34); Burgers s = 42-44: d_x, d_t, Galilean below
lambda6, and scaling and projective within 3x of the EXACT operator's loss on
the same trajectories (no claim that they pass lambda6 on the torus).

**Main criterion on UNSEEN trajectories (data seed 1000 + s), local steppers,
filter 2.** Pre-registered wording: every algebra member below lambda_{k+1}.
  Burgers s44 FNO (data 1044)   d_x 2.7e-4 d_t 4.7e-5 Gal 2.2e-4 scaling 8.4e-3 proj 5.5e-3   lambda6 9.2e-3   MET, margin 1.1x
  KdV s43 FNO (data 1043)       d_x 2.3e-4 d_t 1.4e-3 Gal 2.5e-4 scaling 8.2e-3              lambda5 1.8e-2   MET, margin 2.2x
  heat s46 FNO (data 1046)      six finite <= 8.0e-4 (lambda11 4.1e-3, 5x); b = 1, b = x at 1.0   NOT MET as worded
The heat verdict follows the letter: exp31 runs mu_rel = 0, under which b = 1
and b = x are degenerate (known, exp34); the pre-registration should have said
"six finite generators". Post hoc, exp34 on the same trajectories: mu_rel = 1
puts all ten below lambda11 (1.3e-3; worst b = x^3 + 6 alpha x t 9.6e-4, 1.35x),
containment >= 0.98.
Burgers' margin fell from 40x (seed-44 training trajectories) to 1.1x, but NOT
because the stepper fails to generalise: the EXACT operator on the same unseen
trajectories gives scaling 1.0e-2 ABOVE lambda6 8.3e-3 (projective 7.5e-3, d_x
8e-8) -- the stepper sits at the exact operator's floor. The torus seam floor of
Burgers' x-weighted generators depends on the evaluation set (exact operator:
~5e-5 on seed 42's four, 1e-2 on seed 1044's ten), as the influence-cone
measurement predicted (transport carries ~2 percent of the seam response
everywhere). Burgers scaling "passing on every seed" on the torus was partly the
evaluation sets; the seam-free routes (localised data, window operator) are the
answer, not a better stepper. The leakage did not inflate the learned
operators' results relative to the exact operator's.

**Evaluation trajectories were inside the steppers' training sets.** exp31
and exp40 trained a missing stepper on generate_pde_data(..., n_train,
seed = --seed) and evaluated on generate_pde_data(..., n, seed = --seed); the
generators seed numpy once, so the first n training ICs ARE the evaluation
ICs (checked: max difference 0.0 on Burgers seeds 42-44), and the stepper's
held-out split is the LAST 10 percent. Every exp31/exp40 "seed s" result with
a stepper trained under the same seed was scored on training trajectories.
Unaffected: the exact-operator results, exp33's seed-50 runs, the heat
cross-evaluations (seed-43 data through the seed-44 stepper and vice versa).
Now `--data-seed` in exp31 and exp40 selects the evaluation trajectories; the
default keeps the old behaviour for reproducibility.

**The window stepper's seeds 43/44 floor: the evaluation sets.** Reading rule
fixed before the six fd/spectral runs: spectral u_x lowering d_x >= 5x ->
window derivative; < 2x change with a larger K-step u_x rollout error on
43/44 -> the stepper; neither -> something else (evaluation trajectories).
Result: spectral u_x RAISES d_x 1.3-1.4x (branch 1 out); the 16-step u_x
rollout error is 0.058 / 0.064 on 43 / 44 vs 0.015 on 42, and the d_x loss
ratio to seed 42 equals the squared ratio of rollout errors (10 vs 10.7;
13 vs 14) -- for a convolutional stepper a grid shift is exact, so d_x's
residual is its differentiated rollout error, a ready equation-free accuracy
measure. But the one-step error on the evaluation windows is 0.095 / 0.094 on
43 / 44 vs the held-out 0.023 (seed 42: 0.025 vs 0.021): the stepper is bad on
THESE trajectories. Their fronts: max |u_x| of the ten evaluation trajectories
seed 42 <= 6.8; seed 43 one at 16.9; seed 44 four at 10.9-18.5 (training
median 1.5-3.7, 90th percentile ~10-14). Across seeds, steepness and the FNO's
one-step error correlate (rank 0.48, 30 trajectories); the aligned FNO's
filter dropped exactly seed 44's steepest ones. exp40 had no reliability
filter (exp31 did): now `--reliability-filter` in exp40.

PRE-REGISTERED, unseen trajectories (data seed 1000 + s), existing local
steppers, exp31, filter 2: Burgers s44 (64-mode, --aligned --ut central4),
KdV s43, heat s46 (K 64, every 32) -- every algebra member below lambda_{k+1}.

**Training the window stepper on 16-step rollouts: FAILED.** Prediction: all
five below lambda6 and d_x <= 5e-4 on seeds 42-44. Seed 44 diverged (loss
2e23 by epoch 40), seed 42 stalled (increment error 5.4), seed 43 trained
(0.025) but got WORSE than with 4-step rollouts (d_x 3.9e-3 vs 1.9e-3; d_t,
scaling above lambda6). Long rollouts from scratch are unstable (a curriculum
and gradient clipping would be needed), so on 42 and 44 the hypothesis was not
tested at all; on 43, the only fair case, it is contradicted: the
"horizon mismatch" explanation of the high translation losses is not
supported. Next, on the existing 4-step steppers: fd vs spectral u_x, and the
operator's own 1- and K-step rollout error on the scored region (printed by
exp40 now), to separate the window derivative from the operator.

**Window stepper, Burgers seeds 43 and 44 (r = 6): prediction PARTLY met.**
Increment error 0.023 / 0.022.
                d_x     d_t     Galilean scaling projective lambda6
  s43           1.9e-3  3.3e-3  6.4e-4   1.4e-3  5.1e-4     2.4e-3
  s44           2.5e-3  2.3e-3  8.5e-4   1.9e-3  5.1e-4     2.8e-3
All five below lambda6: s43 FAILED (d_t above), s44 met with 1.1x margin.
Projective and scaling <= 3x d_x: MET on both (0.27 / 0.74; 0.20 / 0.76) --
the seam penalty is gone on all three seeds. Projective blind: s44 xi = xt,
eta = 0.968 t^2, phi = 0.159 x - 0.946 tu (5 terms, MET); s43 3 terms but
the -tu term MISSING (formally within the bar, substantively FAILED). The
"rest <= 0.1" clause could not be checked: exp40 did not print it -- our
error in stating the prediction (the printout is added now). Absolute levels
on s43/s44 are 10x seed 42's at the same increment error; translations now
set the margin, d_t containment ~0.5. For a convolutional stepper a grid
shift is an exact symmetry, so d_x's residual is almost exactly the
differentiated error of the stepper's K = 16 rollout, while it was trained
and selected on 4-step rollouts. Next: train with unroll = K (`--unroll`).

**Learned window stepper, Burgers seed 42 (Colab, 1024 trajectories, 400
epochs): PRE-REGISTERED PREDICTION MET.** Condition: held-out increment error
<= 0.03 -- r = 6: 0.021, r = 10: 0.019 (FNO on the torus 0.008). Predictions:
all five below lambda6; projective and scaling <= 3x d_x's loss.
                      d_x     d_t     Galilean scaling projective lambda6  proj/d_x
  window, r = 6       1.9e-4  1.8e-4  1.4e-4   1.7e-4  1.2e-4     6.1e-4   0.63
  window, r = 10      1.7e-4  2.0e-4  1.0e-4   1.8e-4  1.4e-4     6.3e-4   0.82
  FNO, torus (ref.)   5.2e-5  9.5e-5  3.2e-5   2.0e-4  5.8e-4     4.2e-3   11
Containment 0.90-1.00. The seam penalty is gone: on the torus projective cost
11x d_x, now it costs less than d_x. Blind readout (m = 6): PROJECTIVE AS A
TEXTBOOK FORMULA through a learned operator -- r = 10: xi = xt, eta = 0.992 t^2,
phi = 0.16 x - 1.01 tu (4 terms); r = 6: the same plus a -0.005 x^2 term (on
the torus FNO: 7-9 terms); Galilean xi = t, phi = 0.16 clean; scaling again
split into the inviscid near-symmetries (x, 1.06 t, 0) = S1 + S2 and an
S2-like vector -- the true scaling lies in their plane, as found before.
Caveat: the worst margin is 3.2x (FNO 7.2x); the window stepper is 2.5x less
accurate, every generator sits at its ~1-2e-4 floor, and the translations now
set the margin. Limit of locality: at the benchmark's save step KdV's high
modes travel ~100 grid points per step (group velocity 3 k^2), so a local
window operator cannot represent KdV at this dt; heat needs kernel moments to
~1e-7 relative. A stride of several save steps per window step is the next
lever for accuracy (not for KdV).

PRE-REGISTERED (window stepper r = 6, Burgers seeds 43 and 44): all five
below lambda6; projective and scaling <= 3x d_x; blind readout at m = 6 gives
projective with <= 5 terms and rest <= 0.1 on both seeds.

**exp40 validated with a correct exact window operator: the whole heat
algebra on line windows.** A symmetric 13-tap kernel whose even moments
0..12 match the heat step (Gaussian of variance 2 alpha dt / dx^2) -- exact to
O(theta^14) at low frequencies, sum 1 to 1e-12, max |multiplier| 1.000 on
[0, pi], tails 7e-4 .. 4.5e-7: one-step error relative to the increment
1.4e-7 .. 5.5e-7 (truncated kernel: 5e-2 .. 2.4e-1); prediction <= 1e-4 MET.
Discovery on four-period windows of low-passed heat data (degree 3, K = 16,
mu_rel = 1), prediction all ten below lambda11, containment >= 0.9, a jump
after the tenth:
  u_x spectral   losses 1e-28 .. 1.6e-8 (d_t, time differences)   lambda11 3.4e-4   containment 1.00   jump x2e4
  u_x fd (window only)   losses 1e-28 .. 1.1e-7 (scaling)       lambda11 3.4e-4   containment 1.00   jump x5e3
MET. Through an operator that knows neither boundary nor periodicity the
whole heat algebra is found -- six finite generators and four heat
polynomials, the non-periodic b = x, x^2 + 2 alpha t, x^3 + 6 alpha x t
included (degenerate or seam-bound on the torus) -- with the rank read off
the spectrum, and with u_x taken from the window alone. The earlier failure
on the same run was OUR EVALUATION ERROR: the class measured x from each
window's centre, the ground truth from zero, and with heat's five domain
lengths the offset differs per sample, so the ground-truth Galilean (-x u)
differs from the class's by a per-sample multiple of u d_u and is not one
coefficient vector; the search itself had found the right ten-dimensional
kernel all along. A direct check of the linearised condition on exact
formulas (relative residual ~1e-7 for Galilean, scaling, d_x,
x^2 + 2 alpha t) located it. Lessons: on slow diffusion a local operator
must reproduce the step's kernel moments to ~1e-7 relative; the class and
the ground truth must share one coordinate.

**exp40: a boundary-free window operator -- built, not yet validated.**
`pde/window_stepper.py`: valid 1-D convolutions map a window of width W to the
central W - 2r points; K steps leave W - 2rK, the influence cone built in, no
boundary condition. exp40 reads periodic trajectories as solutions on the
line (windows of several periods, continuous x, so no characteristic jumps
anywhere, projective's x / 2 pi included), u_x by fourth-order differences
inside the window, every step scored on the central region, mu-normalised
eigenproblem. (1) CPU pilot, Burgers, 256 trajectories, 40 epochs, r = 6:
held-out increment error 0.28 (FNO 0.008); all generators above lambda6 --
undertrained, the pipeline runs end to end, nothing more. (2) Validation with
an "exact" window operator for heat -- the spectral one-step multiplier's
discrete kernel truncated at r = 6. Prediction: all ten heat generators below
lambda11 on four-period windows. FAILED, and b = x at 0.2 (it must be zero:
a heat kernel preserves linear functions) exposed the operator, not the
method. Our claim that the truncated kernel is exact to ~1e-12 was wrong: the
band-limited multiplier is still e^{-3.18} = 0.04 at Nyquist and is cut
there, so its kernel has slowly decaying oscillating tails (2.7e-3 at 3,
7e-4 at 6); truncation loses 6e-4 of the kernel's sum, a uniform decay of
the whole field. Low-passing the data (energy above Nx/4 at 4e-27) did not
help (one-step error 5e-2 .. 2.4e-1 either way), refuting "high-k content"
as the cause: heat's low-mode increment per step is ~3e-3 of the field
(alpha k^2 dt for the first harmonic), so a 6e-4 deficit is ~20 percent of
it, growing as the decay slows. On slow diffusion a local operator must
reproduce the kernel's sum and moments to better than ~1e-4 relative.
Next: a local kernel fitted to the multiplier over the data's band (a
correct exact window operator for validation), and window steppers trained
with a stride of several time steps per window step, which enlarges the
increment relative to the operator's error.

**exp39: fields localised away from the seam, exact operators.** Same
equation parameters, grid (Nx 128) and horizon for localised and broadband
initial data; only the initial data differ. Localisation removes the seam
jump of characteristic terms proportional to u, u_x, u_t; u-independent
non-periodic terms keep it. Predictions, stated before the runs: Burgers
scaling (all terms ~ u) drops >= 10x, projective (phi = x / 2 pi - t u)
drops < 3x; KdV scaling drops >= 10x; d_x, d_t, Galilean unchanged.
Burgers (nu 0.002, T 0.05 so the field stays away from the seam; field at
the seam 3.4e-5 of its rms):
  broadband   d_x 3.2e-6 d_t 2.2e-6 Galilean 3.9e-6 scaling 2.8e-3 projective 3.7e-3  lambda6 1.7e-3
  localised   d_x 9.0e-7 d_t 6.9e-7 Galilean 2.0e-6 scaling 1.4e-6 projective 8.1e-4  lambda6 5.1e-4
Scaling drops 2000x to the d_x level: MET. Projective drops 4.6x, still above
lambda6: the "< 3x" bound FAILED numerically (its u_x, u_t terms cleaned too),
the substance held -- the u-independent x / 2 pi term, a linear function of
x added to u, is incompatible with periodicity for any data; only a window
operator can help. In this regime broadband data fail scaling and projective
even through the exact operator.
KdV (T 0.05; localised: two fast narrow solitons, c 12-16, no radiation;
field at the seam 9e-3 of its rms):
  broadband   d_x 2e-17 d_t 2e-12 Galilean 3e-17 scaling 6.9e-3  lambda5 7.4e-3 (margin 1.07x)
  localised   d_x 3e-16 d_t 1e-10 Galilean 3e-15 scaling 3.4e-4  lambda5 2.5e-2 (margin 74x)
Scaling drops 20x: MET; the others stay at machine level; containment 1.00;
the spectrum jumps x29 right after the fourth direction.
Pattern: wherever boundary contamination is removed the rank reads off the
spectrum -- heat between walls x90 after the tenth, periodic heat
superpositions x700 after the ninth, localised KdV x29 after the fourth.
The rank difficulty was largely seam contamination, not the method.

**The interior band from the operator's own influence cone (exp37, exp38).**
Rule: push an impulse at the boundary point through the operator for
k = 1..K steps; the band is the distance beyond which the response stays
below tol x its peak. Ground-truth free, exact or learned operator alike.
Prediction (heat between walls, exact, tol 1e-3, fixed beforehand): band at
K = 16 within 0.1-0.25; with the automatic band the whole algebra below
lambda11 with containment >= 0.9 at K = 8, 16, 32, 64. Result: bands 0.123
/ 0.169 / 0.231 / 0.308 (x1.33-1.37 per doubling of K, ~sqrt 2: diffusion);
all ten members below lambda11 at every K, containment 1.00 -- MET. The
hand-picked 0.2 is replaced by a rule.
From the torus seam, K = 16 (exp38; max over states / rms over states):
  heat exact:      0.16 0.044 2.0e-3 3.7e-4 at d = 0.05 0.10 0.15 0.20, floor ~1e-4 (the
                   conserved mean spreading) -> band 0.156 at tol 1e-3 (walls: 0.169)
  heat FNO:        rms 0.14 0.037 3.7e-3 3.4e-3 ... floor 1-5e-3 to d = 0.5 -- the
                   global spectral layers leak 20-40x above the physical floor
  Burgers FNO:     rms 0.13 0.067 0.029 0.021 ... 0.013-0.019 to d = 0.5 (transport)
  KdV FNO:         rms 0.28-0.40 at every distance (dispersion covers the domain)
With tol tied to the operator's own measured error (heat ~0.03) the heat band
is 0.094; a leak below the operator's error is harmless, which is why the FNO
still worked with a 0.2 band. For Burgers and KdV no band exists. This
explains the x-weighted floors seen throughout, quantitatively: Burgers' ~2
percent seam response everywhere is ~4e-4 in loss -- the scaling/projective
floor (2-6e-4), passing only because spurious directions are dearer; KdV's
~30 percent is the scaling floor ~5e-3, equal for exact and learned
operators. For transport and dispersion on a torus, only two routes remain:
data whose characteristics have no jump at the seam (fields localised away
from it), or a local operator on windows that never closes the domain.

**exp36: horizon growth of each direction's loss -- FAILED as a modulation
filter.** Prediction: from K = 16 to 32 on KdV (stepper s43), t x Galilean
grows >= 2.5x, d_x, Galilean, d_t < 2x. Growth: d_x 3.2, d_t 3.6, Galilean
4.1, scaling 5.1; t x Galilean 6.4, t x d_x 4.9; eigen-directions almost
entirely outside the algebra 2.9-3.3. The modulation grows fastest, but true
symmetries grow 3-4x too: KdV's characteristic increments saturate
(dispersive phases rotate), while the stepper's Jacobian error accumulates
step by step, so a true symmetry's loss also grows nearly as K^2. On fast
dispersive dynamics horizon growth does not reveal the kind of a direction.

**exp37: non-periodic data -- heat between Dirichlet walls, scored in the
interior.** Boundary conditions break some of the equation's symmetries
(walls break translation); the criterion targets the equation, so it is
scored only where the walls cannot reach within the horizon -- the seam band
is the torus case of this rule. Exact sine-series data and propagator, u_t by
fourth-order differences, degree 3, K = 16, mu_rel = 1. Prediction (band 0.2
from each wall): all six finite generators and four heat polynomials below
lambda11, containment >= 0.9. Result: losses 4e-30 (u d_u) .. 2.2e-6 (d_t,
finite differences), lambda11 2.0e-4, containment 1.00 for all ten, and the
spectrum jumps x90 right after the tenth direction -- MET, rank readable.
d_x is found although the walls break it. Caveat on the cause: homogeneous
Dirichlet data vanish at the walls, so the u-proportional parts of the
characteristics (phi of Galilean, projective) have no jump there, unlike the
torus seam where u is not zero; "walls beat the torus" is partly a property
of this boundary condition. Band sweep: 0.1 -- all below lambda11 but 2x
margin, d_x containment 0.68; 0.05 -- d_x, scaling and the superpositions
above lambda11; 0 -- far above. The prediction that 0.05 would suffice
(one diffusion length sqrt(alpha t) = 0.035) FAILED: the contamination is the
erfc(d / sqrt(4 alpha t)) tail of the jump the sine projection makes at a
wall (~0.3 at 0.05, ~0.05 at 0.1, ~1e-4 at 0.2). The band must span several
diffusion lengths.

**Heat's superposition algebra is infinite-dimensional (linearity):
b(x, t) d_u for every solution b.** Consequences. (1) In a class of degree d
it holds d + 1 heat polynomials h_n = sum_k n!/(k!(n-2k)!) (alpha t)^k
x^(n-2k); the ground truth now generates them up to the class degree
(`algebra(..., dmax)`; h4 = x^4 + 12 alpha x^2 t + 12 alpha^2 t^2 checked).
Earlier heat readings at d = 4 counted h4 as outside the algebra. (2) Heat
has no "right degree": the algebra in the class grows with d, and the
prediction "rule v2 picks d = 3 on heat" rested on the finite part only.
(3) On the torus heat polynomials of degree >= 1 are not periodic (seam);
the periodic superpositions are Fourier modes e^{-alpha k^2 t} cos kx, sin kx
-- NOT polynomials. The polynomial class sees the wrong part of the infinite
algebra. (4) For a linear equation the criterion on phi = b(x, t) asks
whether b is carried by the flow, i.e. whether b solves the equation: this
part of symmetry discovery is solution discovery.

**exp35: periodic superpositions in a Fourier-in-x dictionary.**
phi = trig(2 pi n x / L)(t / T)^j, n <= 4, j <= 3 (36 fields, all periodic,
no seam), K = 64, mu_rel = 1. Prediction, stated before the run: 2 nmax + 1
directions well below the rest; each mode's c1/c0 within 10 percent of
-alpha k^2 T. Heat seed 42, exact propagator: exactly 9 directions (the
constant and cos/sin pairs n = 1..4, each a single mode) below a jump
4.0e-4 -> 0.28 (x700) -- the first family in the campaign whose rank reads
off the spectrum. c1/c0 vs truth: n = 1 -0.395 / -0.395, n = 2 -1.566 /
-1.579, n = 3 -3.355 / -3.553 (MET for n <= 3); n = 4 -5.297 / -6.317, 16
percent (FAILED): the cubic in t is the best fit to e^{-6.3 tau} over the
window, not its Taylor series. Through the learned 64-mode stepper (seed 46,
filter kept 6/10): the same 9 directions, the same x700 jump, the same
profiles (n = 1 -0.394 .. -0.396, n = 2 -1.564 .. -1.570, n = 3 -3.36; n = 4
limited identically by truncation) -- no degradation (predicted for high
harmonics; did not occur). The diffusion coefficient follows from the n = 1
profile without the equation: alpha = -(c1/c0) / (4 pi^2 T / L^2) = 0.0100
(exact), 0.00998-0.01003 (stepper); truth 0.01. A dictionary matched to the
domain turns a crowded, seam-bound spectrum into a clean one with a
readable rank -- the concrete case for dictionaries beyond polynomials, and
for a learned basis as their generalisation.

**Rule v2 on fresh trajectories (data seed 50, same steppers): ONE OF THREE.**
Burgers counts 5 / 10 / 8 / 7 -> d = 2: MET (stable subspace holds the whole
algebra, 0.95-1.00). KdV 3 / 9 / 9 -> d = 3: FAILED (predicted 1 or 2; the
plateau pushes the rule up; d_t only 0.14-0.26 inside the stable subspace,
scaling 0.42-0.67). heat 4 / 10 / 9 / 6 -> d = 2: FAILED (predicted 3; one
direction lost at d = 3 stops the rule and projective is lost; on the
development seed the same step was 10 -> 10). Count-based reproducibility
rules flip on +-1 direction: the algebra's membership changes by one or two
per degree, and reproducible non-symmetries move the count by as much.
With this much data the degree is not reliably selectable this way.

**exp34: an absolute term in the normalisation.** Denominator
||a||^2 + ||b||^2 + mu_rel * (tr B'B / tr K Q0'Q0) * K ||Q0||^2, Q0 the
characteristics at the window starts. Two predictions, stated before the
runs: (i) KdV, mu_rel = 0.1: t x Galilean rises above lambda5 -- FAILED
(4.6e-3 -> 4.3e-3; at mu_rel = 1, 3.0e-3, all losses shift together, ranking
and containment unchanged). The modulation's loss is ~ (K dt / t)^2 with
either normalisation; it is set by the horizon. It is not abnormally cheap:
d_t and scaling are abnormally dear through the stepper (d_t: high-mode
accuracy, 2e-11 through the exact operator; scaling: the seam). (ii) heat,
mu_rel = 0.1: b = 1 and b = x fall below lambda11 -- MET:
  mu_rel   b=1     b=x     b=x^2+2at  b=x^3+6axt  six finite   lambda11
  0        1.0     1.0     0.11       0.047       all below    4.2e-3
  0.1      1.5e-4  5.9e-4  1.3e-3     2.7e-3      all below    1.3e-3
  1        1.5e-5  5.9e-5  1.4e-4     2.9e-4      <= 4.4e-5    6.3e-4
At mu_rel = 1 the whole heat algebra, four superpositions included, lies
below lambda11 for the first time; all ten sit >= 0.97 inside the ten
lowest eigenvectors; t x d_x stays at 0.82; the near-symmetry phi = x^2
(no 2 alpha t) is 4.8x dearer than the true b = x^2 + 2 alpha t -- the loss
tells a heat solution from its frozen approximation. Burgers at mu_rel = 1
(prediction: all five below lambda6): MET, worst margin 25x (40x at 0),
containment unchanged; t x Galilean 1.2e-2 (above), S2 1.3e-3 (below: the
near-symmetry). Candidate protocol setting: mu_rel = 1 -- fixes heat,
neutral elsewhere; to be confirmed on other seeds.

**Rule v2 for the degree, and why fold agreement cannot give the rank.**
v2: compare q = min(16, P/2 - 1) lowest directions between the two halves
(no trivial intersection); count cosines >= max(0.95, chance level); raise d
while the count does not fall, stop before the first fall. Development runs
(the same three as v1):
  Burgers s44   counts 5 / 9 / 8 / 6    -> d = 2 (algebra degree 2)
  KdV s43       counts 3 / 10 / 9       -> d = 2 (algebra degree 1; one over, harmless)
  heat s46      counts 3 / 10 / 10 / 7  -> d = 3 (algebra degree 3; v1 chose 2 and lost projective)
The degree is right on all three. The count is not a rank: Burgers 9 (truth
5), KdV 10 (truth 4), heat 10 (truth 10). Ground-truth reading of the stable
subspace at the chosen degree: Burgers holds the whole algebra (0.90-1.00)
plus ~5 directions mostly outside it; KdV holds d_x and Galilean fully but
d_t 0.58 and scaling 0.69, plus ~8 directions outside; heat holds all ten
members at 0.75-0.99, the stable directions being mixtures (2-76 percent
outside); at d = 4 heat's structure collapses (d_x 0.17, scaling 0.15).
Reproducibility across trajectories is not symmetry: near-symmetries (S2),
modulations (t x Galilean) and any systematic trait of operator and data are
reproducible; only overfitting is not. Fold agreement decides d; m reduces
to a threshold on the loss -- the near-symmetry tolerance question.

PRE-REGISTERED (rule v2 fixed as above; fresh evaluation trajectories, data
seed 50, the same steppers): Burgers d = 2; KdV d = 1 or 2; heat d = 3. No
prediction for m.

**Choosing the degree d without ground truth (exp33).** Correction to the
algorithm as stated earlier: the generalised eigenproblem's denominator is
||a||^2 + ||b||^2 over normalised columns, not ||b||^2. Blocks are computed
once at the largest degree; degree d is the column restriction.
(1) Degree sweep, Burgers, exact operator, aligned + 4th order, N = 4: up to
the algebra's degree the low spectrum is clean (d = 1: the four degree-<=1
members lowest; d = 2: all five); above it spurious directions descend
below true generators (d = 4: 6.5e-5 .. 2.7e-4, under projective's 2.4e-4,
the fourth direction 49 percent outside), while true generators' losses do
not move with d (d_x 2.8e-8 at every degree). Overfitting: 105
coefficients on four trajectories.
(2) Two-fold cross-validation over trajectories confirms it: at d >= 3
spurious directions cost 10-500x more on held-out trajectories than on their
own. Choosing d by held-out loss picks d = 1 and loses projective: a
direction fitted on one half is a mixture, expensive on the other half.
(3) Rule v1 -- fold agreement: principal-angle cosines between the q = 8
lowest directions fitted on two disjoint halves; a cosine counts if >= 0.95
and above the 99th percentile for random subspaces of the same size
(simulated); d = the degree with the most such directions (ties -> smaller),
m = their count. Built on Burgers, then applied unchanged:
  Burgers stepper s44, 16 traj.   3 / 5 / 3 / 2 (d = 1..4)  -> d = 2, m = 5 (truth 5)
  KdV stepper s43, 15 traj.       2 / 4 / 4                -> d = 2, m = 4 (truth 4); predicted m = 4 or 5: MET, t x Galilean rejected as unstable
  heat stepper s46, 12 traj.      3 / 6 / 4 / 2            -> d = 2, m = 6 (truth 10 at d = 3); predicted d = 3-4, m = 8-11: FAILED
  heat stepper s46, 21 traj.      3 / 6 / 3 / 3            -> d = 2, m = 6; predicted d >= 3, m >= 7: FAILED -- "not enough data" refuted
A 12-coefficient class (d = 1) cannot be certified: random subspaces agree
by chance there, so the rule over-specifies by one degree (harmless for the
rank). Diagnosis of the heat failure: comparing q = 16 directions restores
agreement at d = 3 (11 cosines >= 0.95, 10 beyond chance) -- heat's near-null
space exceeds eight directions, and the eight lowest on each half are
arbitrary picks from it. Plausibly because at alpha = 0.01 the data sit near
the u_t ~ 0 limit, where almost any transformation in x is a symmetry: the
Burgers S2 effect, en masse. Even with q = 16 the rule ties d = 2 and d = 3
and picks d = 2; which of the ten stable directions are the algebra and which
near-symmetries is NOT checked. Fold agreement separates overfitting from
structure; it cannot separate exact symmetries from near-symmetries -- both
are reproducible properties of the data.

**Burgers, steppers retrained on aligned data (64 modes, --aligned --ut
central4, filter 2): main criterion MET, blind readout PARTLY met.** Increment
error 0.008 / 0.007 / 0.008. Worst margin below lambda6: 7.2x (s42,
projective), 8.2x (s43, scaling), 110x (s44, scaling); d_t containment
0.99-1.00 (was 0.83-0.97). Blind readout, m = 6: d_x and Galilean clean on
all three seeds; d_t recognisable (eta = 1 dominant) on all three;
projective clean on s44 (4 terms: xi = xt, eta = 1.01 t^2, phi = 0.153 x -
1.11 tu), one small term over the <= 5 bar on s42 and s43 (leading
coefficients right everywhere); scaling split on all three.

**The split scaling is a near-symmetry of the inviscid limit, and its loss
measures the viscosity that breaks it.** The split vectors are eta = t,
phi = -u and xi = x, phi = +u: the two separate scalings of INVISCID
Burgers, S1 = x d_x + u d_u and S2 = t d_t - u d_u; viscous Burgers keeps
only S1 + 2 S2 = the scaling of the algebra. span{S1, S2} =
span{scaling, S2}, and S2 is broken only by the viscous term. Losses:
                      scaling  S1      S2      S2 / scaling
  exact operator      5.4e-5   1.3e-3  9.2e-4  17x
  64-mode stepper     9.0e-4   1.3e-3  1.4e-3  1.5x
Through the stepper the true scaling and its inviscid partner are nearly
degenerate, so the sparsity rotation returns S1- and S2-like vectors.
Pre-registered: loss(S2) ~ nu^2 (ratios ~0.25 at nu = 0.005, ~4 at 0.02,
within 2x), the true scaling's loss within 2x of its nu = 0.01 value.
Exact operator, seed 42, N = 4:
  nu = 0.005   scaling 2.32e-4   S2 2.31e-4   S1 5.15e-4
  nu = 0.010   scaling 5.39e-5   S2 9.20e-4   S1 1.34e-3
  nu = 0.020   scaling 3.56e-4   S2 4.60e-3   S1 4.05e-3
S2's ratios 0.25 and 5.0: MET -- the loss of a broken symmetry measures the
breaking term, equation-free. The scaling part FAILED: its floor depends on
nu non-monotonically (ratios 4.3 and 6.6; minimum at 0.01), mechanism not
established. At nu = 0.005 scaling and S2 are EQUAL: even the exact
operator no longer separates the true scaling from its inviscid partner.
Separability of a symmetry from a near-symmetry is set by the breaking
strength against the operator's accuracy.

**The d_t floor was a data artefact: non-uniform snapshots.** Hypothesis
"d_t's loss is floored by the finite-difference ORDER" -- REFUTED on its own:
a fourth-order difference (`--ut central4`) left d_t's loss through the
exact Burgers operator unchanged (9.2e-5 -> 1.1e-4). Clue: every generator
with eta != 0 (d_t, scaling, projective) sat near 1e-4, those with eta = 0
near 4e-7. Cause: the benchmark's Burgers solver saves at times rounded to
its internal grid (10000 steps, 128 saves: 78 or 79 steps apart, +-0.6
percent of a save step), and the exact operator re-solves with its own
rounding -- the same defect found in KdV. `--aligned` re-solves with 79
internal steps per save exactly, data and exact operator alike. Exact
operator, seed 42, N = 4 (d_x, d_t, Galilean, scaling, projective):
  rounded saves, 2nd order    3.9e-7 9.2e-5 4.9e-7 9.6e-5 3.1e-4
  rounded saves, 4th order    3.9e-7 1.1e-4 4.9e-7 1.1e-4 3.3e-4
  aligned saves, 2nd order    3.2e-8 4.8e-5 9.4e-8 7.1e-5 2.6e-4
  aligned saves, 4th order    3.2e-8 2.1e-7 9.4e-8 5.4e-5 2.4e-4
Both fixes are needed; together d_t drops 440x to the level of d_x and
Galilean. The prediction that scaling and projective would drop too FAILED:
theirs is the seam floor. Blind readout, exact operator, aligned + 4th
order, m = 5: xi = 1; xi = t, phi = 0.159; eta = 1 (1 term, rest 0.02);
xi = x, eta = 2.04 t, phi = -1.06 u; xi = xt, eta = t^2, phi = 0.164 x -
0.986 tu -- THE WHOLE BURGERS ALGEBRA AS TEXTBOOK FORMULAS, blind (rank
still given). Through the 64-mode s44 stepper (trained on rounded data),
m = 6: Galilean and d_x clean; projective clean (5 terms: xi = xt - 0.087 x,
eta = 1.05 t^2, phi = 0.137 x - 1.11 tu); d_t recognisable (eta = 1
dominant, ~20 percent admixture); scaling partial (eta = t, phi = -u, its
xi = x drifted into a junk vector -- the seam-sensitive part). Main
criterion through that stepper: d_t 6.9e-4 -> 1.3e-4, worst margin 24x ->
40x, d_t containment 1.00. The benchmark's rounded saves touch every
u_t-based result, both published methods included.

PRE-REGISTERED (Burgers steppers retrained on aligned data, 64 modes,
seeds 42-44, --aligned --ut central4, filter 2): exp31 -- all five below
lambda6 on every seed; exp32 at m = 6 -- d_x, Galilean, projective clean
(<= 5 terms, rest <= 0.1), d_t at least recognisable (eta = 1 dominant),
scaling uncertain.

**KdV seeds 43 and 44: PRE-REGISTERED PREDICTION MET.** Stepper increment
error 0.005 / 0.006; filter kept 9/10 on both. Losses d_x, d_t, Galilean,
scaling; lambda5: s43 6.9e-5 2.9e-3 8.0e-5 5.3e-3, 1.9e-2 (worst margin
3.6x, scaling); s44 2.2e-4 1.6e-3 2.8e-4 7.7e-3, 1.9e-2 (2.5x, scaling). The
"algebra at the bottom of the spectrum" criterion now holds on every seed of
all three equations: Burgers 3, heat 5, KdV 3. On KdV the fourth eigen-
direction is spurious (84-88 percent outside the algebra) on every seed.

**Blind readout with the 64-mode steppers (exp32, reliability filter 2).**
Burgers seed 44, m = 6 -- prediction (made for the 64-mode stepper): Galilean,
d_x and projective clean (<= 5 terms, rest <= 0.1), scaling at least
recognisable, d_t not separated. Got: Galilean xi = t, phi = 0.153 (2 terms,
clean); d_x xi = 1 (1 term, clean); projective xi = xt - 0.386 t^2 - 0.047 x,
eta = 1.07 t^2 + 0.109, phi = 0.143 x - 1.15 tu (7 terms: recognisable, NOT
clean); scaling split across two mixed vectors (NOT recognisable); d_t not
separated. Prediction PARTLY met. A 5x more accurate stepper barely cleaned
the readout: the difficulty is not the operator's accuracy but how closely
scaling, d_t and the nearest spurious directions sit in the spectrum.
KdV seed 43, m = 4: d_x xi = 1 and Galilean xi = t, phi = 0.161 (1/6 = 0.167)
clean; d_t and scaling in a 15-term mixture; the fourth vector is
xi = t^2, phi = 0.196 t -- t x Galilean, NOT a symmetry: the short-horizon
loss's blind spot for slow time modulation, surfacing by itself in the
blind output ahead of d_t and scaling.
Heat seed 46, m = 10: mostly mixtures (5-22 terms). Recognisable: u d_u
(phi = u), phi = 1, the phi parts of Galilean (x u) and projective (x^2 u);
d_x, d_t, scaling dissolved. Causes: the ten lowest eigenvalues crowd
within 7e-6 .. 1.3e-3; the superpositions have nearly time-independent
characteristics, to which the increment normalisation is blind, and mix;
with alpha = 0.01 the xi, eta parts of Galilean and projective (0.02 t,
0.04 tx, 0.04 t^2) are tiny next to phi.

Blind readout overall: translations and Galilean come out as exact formulas
on every equation; projective recognisably on Burgers (cleanly through the
exact operator); scaling, d_t and heat's full algebra do not separate.
Obstacles, each identified: (i) d_t's loss is floored by the finite-
difference time derivative; (ii) the short-horizon loss admits slow time
modulation (t x Galilean on KdV); (iii) a crowded low spectrum, in which a
sparsity rotation mixes near-degenerate directions; (iv) heat's
superpositions are degenerate for the loss.

**What the method outputs, read BLIND (exp32), and a corrected statement
of the main result.** The exp31 criterion "each ground-truth generator's
loss below lambda_{k+1}" plugs the KNOWN formulas into the loss: it shows
that the known generators are cheap directions the learned operator
carries -- a necessary condition for discovery, not discovery. The search
itself is blind (an eigenproblem over all polynomial vector fields of
degree <= d; ground truth never enters), but choices made knowing the ground
truth: the degree (3 for heat because projective has x^2 u), k = the
algebra's dimension, and K / s / seam tuned on these equations.

exp32 rotates the subspace of the m lowest eigenvectors to its sparsest
form (varimax on RMS-weighted coefficients) and prints formulas; no ground
truth anywhere. Burgers, seed 42:
  EXACT operator, m = 5: v1 xi = 1 (d_x); v2 xi = t, phi = 0.159 (Galilean,
    1/2pi exactly); v4 xi = x, eta = 1.96 t, phi = -1.01 u (scaling);
    v5 xi = xt, eta = 0.989 t^2, phi = 0.163 x - 0.983 tu (PROJECTIVE); v3 a
    12-term mixture carrying d_t -- four of five as clean textbook formulas.
  learned 24-mode stepper, m = 6: xi = t, phi = 0.156 (Galilean, clean);
    xi = 1, eta = 0.085 (d_x, nearly clean); xi = xt, eta = 1.24 t^2,
    phi = 0.166 x - 1.1 tu + small terms (projective, recognisable, 10-30
    percent extra); a scaling fragment (eta = t not 2t, no -u); two 13-17-term
    junk vectors. m = 5: the same, dirtier.
d_t never separates: through finite-difference u_t its loss (~1e-4) equals
the nearest spurious directions'. Junk looks like junk -- true symmetries
print as 2-5 terms, spurious directions as 12-18: a candidate rank signal
without ground truth. The Colab 64-mode steppers are ~5x more accurate on
increments than the 24-mode one used here.

CORRECTED MAIN RESULT: the known algebra lies at the bottom of the learned
loss spectrum -- every generator cheaper than the first eigenvalue beyond
the algebra's dimension (Burgers 3 seeds, heat 5, KdV 1), the untrained
null two to four orders higher. Read blind, the exact operator yields clean
formulas for four of five Burgers generators including projective; a learned
stepper yields Galilean and d_x clean and projective recognisable, not
scaling or d_t. Blind rank and a clean full basis through a learned operator
are NOT yet shown. The earlier wording "recovers the full algebra" was an
overstatement.

**KdV through the learned stepper, seed 42: PRE-REGISTERED PREDICTION MET,
better than the cautious expectation.** 64-mode stepper on aligned KdV data
(1024 ICs, 400 epochs), stencil u_t, reliability filter 2 (kept 9/10), N = 10.
Losses d_x, d_t, Galilean, scaling; lambda5:
  trained stepper       4.1e-4 3.9e-3 4.1e-4 5.2e-3    2.5e-2
  exact operator        6e-16  2e-11  1e-15  6.5e-3    1.9e-2
  untrained, 3 draws    0.87-1.3 for every generator, lambda5 0.80-1.0
All four below lambda5 (d_x, Galilean 61x; d_t 6.4x; scaling 4.8x), two to
three orders below the untrained null. d_t and scaling had been held
UNCERTAIN (u_t lives in modes carrying 0.1 percent of u's energy); they
pass. Plausibly because KdV's dispersive part is a per-mode multiplier
e^{i k^3 dt} that a 64-mode spectral layer represents exactly. As on heat,
the stepper's scaling reaches the exact operator's floor (5.2e-3 vs 6.5e-3):
the limit is the geometry (the seam), not the learned operator. Caveat:
d_t's containment in the four lowest eigen-generators is 0.32 -- its loss is
low, but the fourth eigen-direction is spurious (84 percent outside the
algebra) and ranks ahead of it; the open rank problem.

THE MAIN RESULT NOW HOLDS ON ALL THREE BENCHMARK EQUATIONS: Burgers (3
seeds, projective included), heat (5 seeds, Galilean and projective
included), KdV (1 seed so far, d_t and scaling included).

PRE-REGISTERED for KdV seeds 43 and 44 (same setting): all four generators
below lambda5.

**KdV: the time derivative is recoverable equation-free; through the exact
operator the whole algebra is found.** Where u_t lives (benchmark data):
energy of u above |k| = 7 is 0.1 percent, energy of u_t above |k| = 7 is 97.5
percent (93 above 10, 77 above 15, none above the 2/3-dealiasing cutoff 21):
the factor k^3 lifts nearly invisible modes to dominance, while the save
step 1/127 resolves omega = k^3 only below |k| ~ 7. Central differences vs
the solver's own right-hand side: at the save step 0.99; with a local
stencil aligned to the solver's internal grid -- 3-point, delta = 2e-5:
0.0034; 5-point, delta = 1e-4: 0.011; 5-point, 2e-4: 0.13; 4e-4: 0.77.
CORRECTION: the earlier record "finite differences do not converge under
refinement" was an artefact -- the solver rounds save times to its internal
grid (393 or 394 steps per save), so refined snapshots sat at the wrong
times (the same effect gave 0.043 instead of 0.0034 on a first unaligned
try). For the stored benchmark data the conclusion stands (0.99); the cause
is aliasing, and it is lifted by a measurement design: a three-point
stencil around each saved time, 0.5 percent of a save step wide.

exp31 `--pde kdv`: snapshots on the internal grid (400 internal steps per
save), a delta = 2e-5 stencil for u_t (`--ut stencil`, the default for KdV;
the stencil centre becomes the snapshot), the exact operator through the
same aligned solver. Exact operator, seed 42, N = 3, degree 2, K = 16:
  u_t stencil      d_x 6e-16  d_t 1.4e-11  Galilean 2e-15  scaling 6.5e-3   lambda5 2.0e-2
  u_t stored FD    d_x 6e-16  d_t 9.8e-2   Galilean 2e-15  scaling 1.2e-2   lambda5 4.9e-2
With the stencil all four generators lie below lambda5 and d_t drops ten
orders to machine precision; with stored snapshots d_t fails. d_x and
Galilean (eta = 0) pass either way, as predicted; the prediction that
scaling would also fail with stored snapshots did not hold (worse, but
below lambda5). Scaling's floor (6.5e-3, margin 3x, containment 0.79) is
the seam made worse by dispersion: the seam discontinuity of its
characteristic radiates high-k waves with group velocity 3 k^2, which cross
the domain many times within K = 16 save steps, so a seam band barely
helps.

PRE-REGISTERED for the learned stepper (KdV, 64 modes, 1024 ICs, 400
epochs, seed 42, N = 10, reliability filter 2): d_x and Galilean below
lambda5; d_t and scaling UNCERTAIN -- u_t lives in modes 8-21 that carry
0.1 percent of u's energy, and a stepper trained on u constrains them
weakly.

**CONFIRMATION with the reliability rule fixed (--reliability-filter 2):
the pre-registered predictions HOLD on all six runs.** Worst-margin
generator vs its threshold (heat: lambda11; Burgers: lambda6):
  heat s42     kept 9/10   projective 1.5e-3 vs 4.9e-3   3.3x
  heat s45 NEW kept 6/10   projective 1.6e-3 vs 6.0e-3   3.8x   (stepper increment error 0.066)
  heat s46 NEW kept 6/10   projective 7.7e-4 vs 4.2e-3   5.5x   (0.026)
  Burgers s42  kept 9/10   projective 6.6e-4 vs 4.5e-3   6.8x
  Burgers s43  kept 10/10  d_t, scaling 1.6e-3 vs 1.6e-2 10x
  Burgers s44  kept 9/10   scaling 1.0e-3 vs 2.4e-2      24x
Every finite-algebra generator is below the threshold in every run. Seeds
45 and 46 are new data with newly trained steppers under the fixed rule --
a prospective confirmation of a rule introduced after seed 43 failed. The
rule never dropped more than 4 of 10 trajectories; on Burgers it drops one
or none and leaves the result unchanged. Containment in the lowest
eigen-generators >= 0.95 for all but d_x on heat s46 (0.79) and d_t on
Burgers (0.83-0.97).

STATUS OF THE MAIN RESULT. The localised FINS criterion (multiple-shooting
increments over short horizons, a seam band) in a finite-dimensional
polynomial generator class, solved as a generalised eigenproblem through a
learned 64-mode stepper, with an equation-free trajectory reliability
filter, recovers the full point-symmetry algebra of Burgers (3 seeds) and
the full finite algebra of heat (5 seeds), non-affine Galilean and
projective included, equation-free and without training generators; the
untrained-operator null sits two to four orders higher. Post hoc elements,
each confirmed afterwards on unseen seeds: the GT-loss metric, the 64-mode
stepper, the reliability rule. Open: the rank (no clean spectral gap even
through the exact operator), heat superpositions with time-independent
characteristics (degenerate for the increment normalisation), KdV.

**exp31 Burgers with 64-mode steppers, seeds 42-44, no reliability
filter: PRE-REGISTERED PREDICTION MET.** Held-out increment error 0.009 /
0.007 / 0.008 (24-mode steppers: ~0.04-0.06). Losses d_x, d_t, Galilean,
scaling, projective; lambda6:
  s42  7.1e-5 5.8e-4 4.7e-5 3.1e-4 9.1e-4   4.8e-3   margin 5.3x (24 modes: 4.8x)
  s43  1.2e-4 1.6e-3 8.7e-5 1.6e-3 1.3e-3   1.6e-2   margin 10x  (7.8x)
  s44  1.7e-4 6.9e-4 8.7e-5 1.0e-3 4.3e-4   2.4e-2   margin 24x  (21x)
All five below lambda6 on all three seeds, margins moderately larger than
with 24 modes, as expected (Burgers' increment is dominated by transport,
so truncation hurt it less than heat). Containment in the five lowest:
0.97-1.00 except d_t 0.83-0.97. Burgers and heat now rest on one stepper
architecture. The filtered Burgers run (the rule must not break it) is
still pending.

**Heat seed 43: the failure follows the DATA, not the stepper; an
equation-free reliability filter repairs it (post hoc, to be confirmed).**
Homogeneity error (Jacobian along u vs the exact linear step, relative to
the increment) on each seed's own data: s43 stepper 0.302, s44 stepper
0.055 -- the prediction "s43's is clearly larger" held -- but by amplitude
the s43 stepper is as homogeneous as s44's up to ~2 (0.01-0.06) and fails
only on trajectories of amplitude 2.3-5 (0.18-0.37), which seed 44's
evaluation set lacks. Cross-evaluation settles it: seed-43 data through the
s44 stepper -- Galilean 2.4e-2, projective 4.9e-2 ABOVE lambda11 6.6e-3;
seed-44 data through the s43 stepper -- all six below lambda11 (projective
8.4e-4 vs 5.2e-3). Both steppers lose homogeneity at high amplitude; two
such trajectories in seed 43's set break the generators whose phi is
proportional to u. The stepper-specific hypothesis is refuted.

Equation-free reliability: the stepper's one-step prediction error on each
trajectory relative to the data's own increment correlates with the
homogeneity error (0.79 for s43's stepper, 0.37 for s44's) -- imperfectly.
Rule (`--reliability-filter 2`): drop trajectories whose error exceeds twice
the median. Losses u d_u, Galilean, scaling, projective; lambda11:
  data 43, stepper 43, no filter      1.7e-2 6.0e-3 1.8e-3 3.9e-2   6.4e-3
  data 43, stepper 43, filter (7/10)  3.6e-5 6.4e-4 2.3e-4 1.9e-3   4.4e-3
  data 43, stepper 44, filter (6/10)  4.9e-5 9.6e-4 1.5e-4 2.4e-3   3.9e-3
  data 44, stepper 44, filter (8/10)  1.8e-5 4.8e-4 8.0e-5 8.3e-4   5.2e-3
  data 44, stepper 43, filter (9/10)  2.8e-5 4.5e-4 6.3e-5 8.4e-4   5.1e-3
All six below lambda11 in all four combinations (d_x, d_t <= 8e-5).
Expectation stated before the seed-43 filtered run -- projective and
Galilean left at the edge -- was too cautious: projective 1.6-2.3x below,
Galilean 4-7x. The rule was introduced AFTER the failure; it did not break
the passing seed 44.

PRE-REGISTERED with the rule fixed (--reliability-filter 2): heat seed 42
and new seeds 45, 46 (64-mode steppers) -- all six finite-algebra
generators below lambda11 at K = 64; Burgers seeds 42-44 with 64-mode
steppers -- all five below lambda6.

**exp31 heat, full-mode steppers on seeds 43 and 44: the pre-registered
prediction HOLDS on 44 and FAILS on 43.** Held-out increment error 0.048
(s43), 0.028 (s44). Losses d_x, d_t, u d_u, Galilean, scaling, projective;
lambda11:
  s44, K = 64   5.8e-6 1.5e-5 1.9e-5 4.6e-4 6.9e-5 8.2e-4   5.7e-3
  s44, K = 16   5.0e-5 8.7e-5 5.5e-5 2.3e-4 8.4e-5 3.8e-4   1.3e-2
  s43, K = 64   5.5e-4 1.9e-3 1.7e-2 6.0e-3 1.8e-3 3.9e-2   6.4e-3
  s43, K = 16   6.5e-3 1.0e-2 1.7e-2 8.2e-3 1.1e-2 3.7e-2   1.7e-2
Seed 44: all six below lambda11 by 7-1000x, containment >= 0.98. Seed 43:
u d_u and projective above lambda11 at both horizons, Galilean passes by a
hair at K = 64. Heat: two seeds of three. The failure is selective: the
generators whose phi is proportional to u (u d_u, phi = u; projective,
phi = -(x^2 + 2 alpha t) u; nearly Galilean, phi = -x u) fail, those with
phi = 0 (d_x, d_t, scaling) pass; u d_u on s43 is 250x worse than on s42
at nearly equal trajectory error. Hypothesis: the s43 stepper's Jacobian is
wrong along the AMPLITUDE of the state -- the exact heat step is homogeneous,
S(lambda u) = lambda S(u), a learned stepper need not be, and the increment
metric (errors along the data) cannot see it. To be tested by the
homogeneity error of the three steppers.

**exp31 heat with the full-mode stepper, seed 42: PRE-REGISTERED
PREDICTION MET.** Stepper (--stepper-modes 64, 1024 ICs, 400 epochs, abs
loss): held-out increment error 0.041 (the 24-mode stepper: ~0.4-0.7), train
loss ~1e-7 (was 5.7e-6). Losses d_x, d_t, u d_u, Galilean, scaling,
projective; lambda11:
  stepper, K = 64    6.2e-5 8.1e-5 6.9e-5 1.1e-3 1.5e-4 1.6e-3   5.5e-3
  stepper, K = 16    4.2e-4 3.4e-4 1.5e-4 6.1e-4 3.5e-4 9.4e-4   1.2e-2
  exact,   K = 64    2e-12  8e-6   5e-14  8.5e-4 1.1e-4 1.3e-3   4.7e-3
  untrained 64-mode steppers (3 draws): 0.29-1.2, lambda11 0.13-0.68
All six finite-algebra generators lie below lambda11 at K = 64 (projective
3.4x, Galilean 5x, the rest 37-89x) and at K = 16 (13-80x), two to four
orders below the untrained null. For Galilean, scaling and projective the
stepper now matches the EXACT propagator at K = 64 -- it has reached the
floor set by the problem's geometry (seam contamination on a long horizon),
not its own error. Containment in the ten lowest eigen-generators: u d_u,
Galilean, scaling, projective and the superpositions 0.96-1.00; d_x 0.78,
d_t 0.88 (K = 64). The main result holds on a second equation, non-affine
generators included. One seed so far. The 24-mode run above failed the
earlier pre-registration; both are reported, and the architectural change
was diagnosed (frequency truncation) and pre-registered before this run.

**What limits the heat stepper: frequency truncation, not the loss.**
Stepper error vs one-step increment on the data: Burgers 4-6 percent, heat
37 percent (72 percent at late times) -- the heat stepper reproduces u to
0.16 percent but barely knows its change. The LR already decays (cosine);
the logged per-epoch MSE jumps because each epoch scores RANDOM windows, so
`train_stepper` now scores a fixed held-out set of windows with the
increment error and returns the best state. Equal CPU budget (256 ICs, 60
epochs), held-out increment error and GT losses (d_x, d_t, u d_u,
Galilean, scaling, projective; lambda11):
  modes 24, abs loss   0.398   0.27 0.41 0.11 0.095 0.19 0.083   0.095
  modes 24, incr loss  0.390   0.52 0.75 0.19 0.21  0.39 0.20    0.16
  modes 64, abs loss   0.305   0.17 0.25 0.028 0.029 0.13 0.034  0.079
The increment-relative loss (`loss="incr"`, kept as an option) does NOT
help -- our diagnosis "the loss optimises the wrong quantity" was wrong. The
cause: neuralop keeps n_modes // 2 + 1 real-FFT frequencies, so modes = 24
keeps 13 of 33 on a 64-point grid (checkpoint weights (48, 48, 13)).
Heat's one-step increment is a frequency-dependent multiplier,
e^{-alpha k^2 dt} - 1, large at high frequencies (-0.41 at frequency 13,
-0.96 at 32), which the pointwise path cannot apply; Burgers' increment is
dominated by front transport, so truncation hurt it far less. All modes
lower every GT loss 1.5-4x at equal budget, and the fit was still improving
at epoch 60. `--stepper-modes` in exp31.

PRE-REGISTERED (heat, full stepper with --stepper-modes 64, 1024 ICs, 400
epochs, abs loss; K = 64, s = 32, N = 10): all six finite-algebra
generators below lambda11. The own-class null is redrawn with the same
architecture.

**exp31 heat through the FULL stepper (1024 ICs, 400 epochs, final unrolled
MSE 5.7e-6), seed 42, N = 10: the pre-registered criterion is NOT cleanly
met; the low-baseline separation is strong.** K = 64, s = 32: losses d_x
2.7e-2, d_t 1.2e-2, u d_u 2.2e-3, Galilean 7.3e-3, scaling 2.9e-2,
projective 9.8e-3 against lambda11 = 2.9e-2 -- four clearly below, d_x
barely (7 percent), scaling AT lambda11; the trained spectrum holds five or
six spurious directions at 1e-3 .. 2e-2, so the algebra does not rank first
(d_x enters the ten lowest at 0.19). Expectation stated beforehand ("partial:
Galilean, projective, u d_u likely; d_x, d_t, scaling uncertain") matched,
with d_t passing too. Three untrained steppers: every finite-algebra
generator at 1.0-1.4 (lambda11 0.5-0.7): through the trained stepper each is
40-500x lower, Galilean ~150x, projective ~110x -- the stepper carries the
whole finite algebra; its accuracy is what keeps d_x and scaling from beating
the spurious directions. K = 16 (7 starts) is worse (losses 9e-3 .. 9e-2,
lambda11 9e-2).

Exact propagator, K = 64: all six below lambda11 = 4.7e-3 (d_x 2e-12, d_t
8e-6, u d_u 5e-14, Galilean 8.5e-4, scaling 1.1e-4, projective 1.3e-3),
containment >= 0.95 -- but at K = 16 Galilean / projective were 3e-7 / 4e-7:
on a long horizon the seam contamination reaches the window. The stepper
needs a long horizon for signal, the x-weighted generators a short one to
avoid the seam: a trade-off specific to slow dynamics.

Levers, not yet tried: (1) the heat stepper's training curve oscillates
(2.7e-6 at epoch 120, 5.7e-6 final, spikes to 2.6e-5) -- LR decay / best-
checkpoint; Burgers steppers reached 5e-7. (2) d_x is 50x worse than on
Burgers; plausibly the FNO's positional embedding breaks shift
equivariance at the scale of heat's slow increments. Dropping it would make
d_x exact by construction -- then d_x must be reported as built in, not
found. (3) A coarser-stride stepper (one step = several dt) makes each
increment larger relative to the per-step error.

**The LieSym-style routine in this repository** (`train_deflation_residual`
+ `olver_loss_single_jvp`, N_theta = `LocalGeneratorFNO` ~ u_t) implements
the correct JVP criterion; the scalar `olver_loss_single` is unused. Its
time derivatives are wrong where they matter: `_spectral_dt` (FFT of a
mirror extension) forces d/dt = 0 at the time ends -- u_t error 0.049 inside
the loss window on Burgers vs 0.007 for central FD, and xi = t gets 0 at t = 0
-- and dt/dx are one scalar from the batch range, which on multi-scale heat
puts u_t 0.90 off (central FD with per-sample dt: 0.003). Details in
LieSym_audit.md.

**LieSym enhanced version: the Olver criterion is now correct in the text;
its single-scale degeneracy claim is refuted.** Section 2.3 of the enhanced
paper evaluates DN_theta[u](Q) - Q_t -- the linearised condition, the
K -> 0 limit of the multiple-shooting criterion (the eq. (5) defect was in
the earlier paper and both scripts). Its central claim -- heat scaling
(phi = 0) cannot be recovered from single-domain-length data by any
operator-loss method -- fails on exp31 with `--only-L`: exact propagator,
10 samples all at L = 1 (T = 1): scaling 7.3e-6, Galilean 7.5e-8,
projective 1.8e-7, lambda11 5.7e-3, containment 0.92; all at L = 2: scaling
3.7e-7, lambda11 3.9e-3, containment 1.00. The infinitesimal condition needs
the linearisation along trajectories, not orbit coverage; the obstruction on
a torus is the seam. Details and the other issues in LieSym_audit.md.

**On LieSym's operator.** Every available source uses u0 -> u(x, t): the
paper (p. 3: "Input is represented as u0(x) in R^64, while output has form
u(x, t) in R^{64x32}"), LieSym_NeurIPS_Experiments.py and
symmetry_multiple_exps_5.py (`F.mse_loss(fno_surrogate(u0), u)`;
`olver_loss_heat` identical in structure). A later version training
N: u -> u_t would change the measured numbers but not the defect of eq. (5):
a scalar loss differentiated in input space is not DL[u](v[u]), and unless x
and t are inputs of the operator, dL/dx = dL/dt = 0 again.

**exp31 generalised to heat** (`--pde heat`, degree 3 by default). Heat
samples have their own L and T = L^2, so on the grid all obey one equation:
one stepper serves them all, generators stay in absolute per-sample
coordinates. The class holds 10 symmetry directions: the 6-dim finite
algebra and, the equation being linear, superpositions u + beta with
beta = 1, x, x^2 + 2 alpha t, x^3 + 6 alpha x t. Exact propagator (linear,
so its Frechet derivative is itself), K = 16, N = 10: losses d_x 1e-11, d_t
5e-5, u d_u 6e-13, Galilean 3e-7, scaling 1e-6, projective 4e-7, beta=1
3e-17, beta=x^2+2at 4e-5, beta=x^3+6axt 1e-5 against lambda11 6e-3; nine
eigen-generators 99-100 percent inside the algebra. LIMITATION: beta = x
scores 1.0 -- its characteristic Q = x is time-independent, the data
increments are identically zero, the operator's nearly zero, and the ratio
normalisation turns seam leakage into 0/0 -> 1 (beta = 1 escapes only because
both are exactly zero). The increment-normalised loss cannot see symmetries
with a time-independent characteristic; beta = x^2 + 2at is nearly so.

Heat through a PREVIEW stepper (CPU: 256 ICs, 80 epochs, unrolled MSE
4e-5): fails. K = 16 / 32 / 64 -- d_x 0.33 / 0.20 / 0.094, scaling 0.23 /
0.13 / 0.054, Galilean 0.054 / 0.036 / 0.034, projective 0.038 / 0.023 /
0.021, lambda11 0.12 / 0.07 / 0.032; at K = 64 only projective lies below
lambda11. Heat is SLOW: on the grid the lowest mode changes ~0.3 percent
per step, so the preview stepper's per-step error equals the increment it
must resolve (in residual form u + F(u) it knows F to ~100 percent). Longer
horizons help monotonically; the exact propagator at K = 64 still separates
the finite algebra (<= 8e-4 vs lambda11 1.7e-3). Needs the fully trained
stepper (Colab).

PRE-REGISTERED for heat (full stepper, 1024 ICs, 400 epochs; K = 64,
s = 32; N = 10): success = all six finite-algebra generators below lambda11.
Expectation: partial -- Galilean, projective, u d_u
likely; d_x, d_t, scaling uncertain. The time-independent superpositions
are excluded as degenerate for this loss.

**u_t from the stepper does NOT help (exp31 --ut).** Seed 42, N = 10, GT
losses d_t / scaling / projective and lambda6: central difference of the
data 1.2e-3 / 1.0e-3 / 1.5e-3, 7.2e-3; forward difference 1.7e-3 / 3.3e-3 /
6.4e-3, 6.3e-3; stepper increment (Step(u) - u)/dt 1.3e-3 / 3.3e-3 / 6.6e-3,
5.8e-3 -- projective rises above lambda6. The increment behaves like a
forward difference; the hoped-for consistency gain for d_t does not appear
because the loss compares with the DATA trajectory, not with the stepper's
own rollout, whose 16-step error eats it. The same argument closes the KdV
route: the increment is a forward difference of the aliased trajectory.
Central difference stays the default.

**How close is the current FINS to LieSym?** LieSym's paper (MathAI 2026,
section 3.1) states the symmetry condition as DL[u](v[u]) = 0, the Frechet
derivative of an operator along the characteristic. With L the evolution
residual u_t - N(u) this is the linearised (Olver) condition: Q is a
tangent trajectory. The current FINS criterion -- increments of the
learned flow's Jacobian against increments of Q over short horizons from
many starts -- is exactly its integrated, discrete-time form, and tends to
it as K -> 0. So the current FINS implements LieSym's STATED principle.

It shares nothing with LieSym's IMPLEMENTED criterion. Equation (5) of the
paper and `olver_loss` in the code replace the operator derivative by the
gradient of a SCALAR loss in input space, v[L] = xi dL/dx + eta dL/dt +
phi dL/du, with L = ||u - FNO(u0)||^2. The surrogate takes u0 only, so
dL/dx and dL/dt are identically ZERO (measured: 0.0e+00): xi and eta never
enter the loss. dL/du correlates 0.97 with the surrogate residual. Measured
olver_loss: d_x 0, random xi and eta with phi = 0 also 0, u d_u (not a
symmetry) 1.8e-10 = scaling 1.8e-10, a fully random field 9.3e-10 =
Galilean 9.6e-10 -- and it tends to 0 for everything as the surrogate
improves. The published LieSym has no working symmetry criterion; its
generators are shaped by the orthogonality and Lipschitz regularisers and
the FNO generator's prior. Its reported check ("for tau = 0.1 all
generators give Delta L < 5 percent") follows from the same degeneracy.

Where the current FINS departs from the PUBLISHED FINS: the horizon (short,
multiple shooting, seam band -- the published global horizon is blind to
generators that change the period of the domain) and the generator class
(finite-dimensional, discovery as a generalised eigenproblem -- the
published FNO generator admits spurious minima even through the exact
solver). Net: the current method is the merger both papers aimed at --
FINS's Frechet derivative of a learned evolution operator, LieSym's local
(infinitesimal) condition, one criterion with a horizon parameter whose
limits are the two published methods, and a hypothesis class in which it
provably identifies the algebra through the exact operator and, confirmed
on three seeds, through a learned one.

**exp31 GT-loss on seeds 43/44: PRE-REGISTERED PREDICTION MET.** Loss of
each GT generator through the trained stepper (d_x, d_t, Galilean,
scaling, projective) and lambda6:
  seed 42  5.3e-4 1.2e-3 2.3e-4 1.0e-3 1.5e-3   lambda6 7.2e-3  (margin 4.8x)
  seed 43  3.8e-4 2.3e-3 4.1e-4 2.2e-3 1.9e-3   lambda6 1.8e-2  (margin 7.8x)
  seed 44  6.1e-4 1.1e-3 2.9e-4 1.4e-3 6.3e-4   lambda6 2.9e-2  (margin 21x)
  untrained steppers (5 draws): 1.0 for every generator.
On three seeds every generator of the Burgers algebra, the non-affine
projective included, is a low-loss direction through a learned surrogate
and ranks inside the lowest five; the metric was chosen on seed 42 and
confirmed on seeds it had not seen.

Still open. Purity: the discovered eigen-generators are 46-99 percent
inside the algebra (seed 44's fourth is 54 percent outside) -- lambda5 and
lambda6 are close enough for spurious directions to mix in. Rank: the
spectrum alone does not give 5; a threshold at the nulls' lowest eigenvalue
(2.1e-2) would give ranks 8, 6, 5 on seeds 42, 43, 44. Candidate:
stability under resampling of trajectories -- algebra directions should
persist across bootstrap draws, spurious ones rotate.

**exp31 on seeds 42/43/44 (Colab, N = 10): the containment prediction
holds, the NULL prediction fails -- and a low-baseline metric replaces it.**
Pre-registered: share >= 0.9 for d_x, Galilean, scaling, projective on
seeds 43, 44 -- MET (43: 1.00 / 1.00 / 0.98 / 1.00; 44: 1.00 / 1.00 / 0.99 /
1.00; d_t 0.79 and 0.95). Seed 42 on GPU reproduced the CPU numbers to the
digit. Exact solver, N = 10: 1.00 / 0.88 (d_t) / 1.00 / 0.99 / 1.00 -- d_t's
weakness is the finite-difference u_t, not the surrogate. Pre-registered:
every null draw <= 0.75 -- FAILED. Five untrained-stepper draws at N = 10:
d_x 0.60-0.70, d_t 0.09-0.13, Galilean 0.79-0.86, scaling 0.72-0.75,
projective 0.91-0.93 (the N = 3 null, 0.51-0.70 for projective, understated
the floor). The 30 basis fields are nearly collinear, so ANY five-dimensional
subspace of the class captures smooth generators such as projective:
forward containment has a high geometric floor, and "projective 0.98 vs
null 0.92" is a small effect.

Low-baseline metrics (introduced AFTER the null failed -- post hoc, to be
confirmed on seeds 43/44). (1) Loss of each GT generator itself through
the operator, from its exact coefficients in the basis: trained stepper
seed 42 -- d_x 5.3e-4, d_t 1.2e-3, Galilean 2.3e-4, scaling 1.0e-3,
projective 1.5e-3, all below lambda6 = 7.2e-3; three untrained steppers --
1.0 for every generator. Projective is a genuine low-loss direction through
the trained surrogate, 670x below the null, and the algebra ranks first.
(2) Share of each discovered eigen-generator OUTSIDE span(GT): trained 0.03,
0.19, 0.13, 0.30, 0.23; nulls 0.31-0.90. The five lowest are ~80 percent
algebra: lambda5 and lambda6 are only 2.5x apart, so the fourth and fifth
eigenvectors mix in spurious directions -- the open rank problem again.

PRE-REGISTERED for seeds 43 and 44 (exp31, N = 10): every GT generator's
loss through the trained stepper < lambda6 and < 1e-2.

**exp31 -- THE POSITIVE RESULT: the FINS criterion recovers the whole
Burgers algebra, projective included, equation-free and training-free,
through a learned surrogate, once the generator class is finite-
dimensional.** With xi, eta, phi spanned by polynomials in (x/L, t/T, u/std)
of total degree <= d, the multiple-shooting increment loss is a ratio of
quadratic forms; discovery is its generalised eigenproblem (no training).
Operator: the learned stepper (seed 42); u_t a finite difference of the
data; the equation is used only to score. Share of each GT generator inside
the span of the five lowest eigen-generators, on the loss window:

  stepper, d=2, K=16/s=16, N=3     d_x 1.00  d_t 1.00  Gal 1.00  scal 0.99  proj 0.99
  stepper, d=3, K=16/s=16, N=3     d_x 1.00  d_t 0.98  Gal 1.00  scal 1.00  proj 0.99
  stepper, d=2, K=8/s=8,   N=3     d_x 1.00  d_t 0.99  Gal 1.00  scal 0.99  proj 0.99
  stepper, d=2, K=16/s=16, N=6     d_x 1.00  d_t 0.67  Gal 1.00  scal 0.97  proj 0.99
  stepper, d=2, K=16/s=16, N=10    d_x 1.00  d_t 0.94  Gal 1.00  scal 0.95  proj 0.98
  exact,   d=2, K=16/s=16, N=2     d_x 1.00  d_t 0.99  Gal 1.00  scal 1.00  proj 1.00
  own-class null (untrained        d_x 0.32-0.53  d_t 0.25-0.36  Gal 0.40-0.62
    stepper, 3 seeds), N=3         scal 0.52-0.66  proj 0.51-0.70

The trained stepper's fifth eigenvalue (2.7e-3 .. 3.8e-3) lies below the
LOWEST eigenvalue of every null (1.2e-2 .. 5.9e-2). Degree 3 (60
coefficients) does not break it: the class, not its richness, is what the
network generators lacked. Weak link: d_t, whose characteristic carries the
finite-difference u_t (1.00 / 0.67 / 0.94 as N grows) -- the same equation-
free limit as throughout. Open problem: the RANK. Even through the exact
solver the spectrum has no clean gap after the fifth value (lambda6 /
lambda5 ~ 4; the largest jump is after the two generators resolved to
~1e-7, d_x and Galilean, while d_t, scaling, projective sit at the 1e-4
floor of FD u_t and front resolution). The stepper lifts the whole ladder
10-100x and keeps its shape.

Correction to exp30: its principal angles were measured in COLUMN-
NORMALISED COEFFICIENT space, where the monomials are nearly collinear;
"projective 0.16 through the stepper" was a metric artefact, and the v5b
prediction "projective not found" was built on it. exp31 measures in field
space, where generators live.

What this says about v4-v5b: their failures were failures of the NETWORK
generator class, which admits near-minima outside the algebra even through
the exact solver (mechanism not pinned down: five diagnoses in a row
failed), not of the criterion or of the surrogate.

PRE-REGISTERED for seeds 43 and 44 (stepper trained by exp31 with the
campaign defaults, d = 2, K = 16, s = 16, N = 10): share >= 0.9 for d_x,
Galilean, scaling and projective; d_t reported, may fall lower; every null
draw <= 0.75.

**nonloc on v5b: the pre-registered prediction failed in both
directions.** Slots 3, 5 nonlocal (0.205, 0.295); slots 4, 6 POINTWISE
(0.008, 0.033) yet admitted by the loss through the exact solver; the
genuine slots 1, 2 have sample-shaped deviations (0.789, 0.127 -- nonloc is
normalised by the VARYING part, so for near-constant fields it describes
their small deviations, not the whole generator). Nonlocality explains at
most half of the spurious slots, so v6 (pointwise generator) was NOT run,
per the rule set beforehand. Our supporting argument was also flawed: the
exact exp30 spectrum was computed at K = 8, not v5b's K = 16, and in a
degree-2 class that covers only half of these slots' fields.

Time-modulation test (exact solver, training units): (1+t) d_x 1.3e-2,
(1+t) d_t 6.0e-3, (1+t) Galilean 2.5e-2 -- a margin of only ~10x over true
symmetries with field noise (<= 1e-3), and the local Olver criterion is
blind to it ((1+t) d_x 0.098 vs d_x 0.097); t d_x, t^2 d_x, t^2 d_t
0.17-0.32 are rejected. But v4's polynomial caricature scores 8.2e-2 while
the slot scores 6.5e-4: its low loss lives in the part of the field the
formula does not describe (R^2 ~ 0.45).

Evaluation-region hypothesis (the v4 mismatch again, in x instead of t):
the loss sees only x in [0.2 L, 0.8 L], while exp29's default crop scores
x in [0.79, 5.50] and so includes parts of the seam bands the loss never
constrains. `--eval-seam s` scores exactly the loss window. Calibration:
scaling + junk ONLY in the seam bands -> default share 0.784, loss window
0.000 (clean scaling); junk everywhere -> 0.655 / 0.386. Against the
hypothesis: that band junk costs shoot 0.158 (transport carries it into the
window within 16 steps), so a slot at 6.5e-4 could only hide content that
does not leak. This is a post-hoc change of the evaluation region; the
principle (judge a generator where the loss constrains it) is sound, but
both evaluations are reported.

PRE-REGISTERED (exp29 --eval-seam 0.2 on v5b): if slots 4 and 6 drop below
share 0.1 the "spurious" verdict was an evaluation artefact; if they stay
>= 0.3 the hypothesis is wrong.

**v5b attribution: the spurious slots are admitted by the LOSS, not the
surrogate -- the pre-registered prediction failed.** Training loss through
the EXACT solver (exp29 `shoot`), slots 3-6: 1.9e-3, 6.5e-4, 5.6e-3, 9.3e-3 --
below the near-miss (3.5e-2), at the level of true symmetries with field
noise; slots 1-2: 5.7e-4, 2.6e-4. The local Olver criterion (0.28-0.86) and
the analytic split (0.69-0.88) still reject slots 3-6.

Mechanism. The equivariance loss, at any horizon, certifies that the
characteristic Q evolves as a tangent trajectory of the linearised flow --
necessary for a symmetry, not sufficient. Sufficiency needs xi, eta, phi to
be the SAME pointwise functions of (x, t, u) for every sample.
SingleGeneratorFNO takes the whole u field, so it can shape each sample's
characteristic into a tangent field; short shooting windows make that far
easier than one long horizon. Independent support: in the pointwise
polynomial class the exact multiple-shooting spectrum has no spurious
minima (exp30), so a slot that is spurious through the exact solver cannot
be a pointwise near-symmetry. Hypothesis to check with the nonloc column on
v1/v3 artifacts: this is also the source of the junk slots throughout the
campaign.

Fix: `--fins-generator pointwise` -- PointwiseGenerator, an MLP of 1x1
convolutions on (x, t, u): the point-symmetry class by construction, and
three times cheaper than the FNO generator.

exp29 `nonloc` column: mismatch of a generator's fields between samples at
points where their u nearly agree -- ~0 for a point symmetry. Calibration:
GT, GT + noise, random smooth (x, t) fields, fragments 0.000; phi shaped
per sample 0.473; phi = u(x + L/8) 0.452; phi = u(t + 5 dt) 0.071.

PRE-REGISTERED. (a) exp29 on v5b: slots 3-6 nonloc >= 0.2, slots 1-2
<= 0.05. (b) v6 (pointwise generator, same loss): no slot with shoot < 1e-2
and analytic share > 0.3 (no loss-admitted spurious slots), and d_x, d_t,
Galilean found by the exp29 criterion.

**v5b (same, all 1024 ICs): the optimiser reaches the minima -- and four
of six are spurious.** Sanity block in training units: true generators
7e-5 (d_x) .. 2.1e-3 (projective), u d_u 0.221. All six slots end at
1.1e-3 .. 4.4e-3, inside the true-generator band. Per-slot cos d_t 0.972,
d_x 0.970 (best of the campaign). Pre-registered prediction vs exp29:
d_x FOUND (v2, share 0.054); d_t missed narrowly (v1, 0.133); Galilean,
scaling not found; projective not found (as predicted). Slots 3-6 have
surrogate loss at the level of true d_t / scaling / projective yet are not
symmetries (shares 0.69-0.88, fields 72-92 percent outside the algebra):
through the stepper there are spurious near-symmetries at the true
x-weighted generators' floor, and deflation lands on them after the two
translations. exp30's conservative gap (< 1) warned of this; the prediction used
the subspace cosines instead and was too optimistic, and the network class
has more spurious directions than 27 monomials. Galilean's surrogate loss
(1.5e-4) is below every spurious slot, yet the optimiser did not reach it:
the polynomial-class spectrum did not predict discovery.

exp29 `shoot` column: the training loss (multiple-shooting increments)
through the EXACT solver, in the units of the run's `equiv`. Synthetic
truth: true d_x / Galilean / scaling / projective 5e-7 / 6e-7 / 1.2e-5 /
1.0e-5; the same with 1 percent field noise 2e-4 .. 1e-3; projective without
eta 1.8e-2; x t fragment, random fields, t(1-t) junk 0.18-0.52. It separates
a spurious near-symmetry of the SURROGATE (low through the stepper, high
here) from one the LOSS itself admits (low here too).

PRE-REGISTERED for exp29 on v5b: slots 3-6 score shoot >= 0.05 (surrogate-
spurious); slots 1-2 near 1e-3 (genuine). If slots 3-6 are low here, the
loss is being exploited (seam band, normalisation) and the fix is in the
loss, not the surrogate.

**v5 (stepper, multiple shooting, K = 16, s = 16, 128 ICs) FAILS the
pre-registered prediction; the optimiser stopped far from the minima.**
exp29 on seed 42: no generator below non-symmetry share 0.1 -- best d_x + d_t
at 0.226 (v3 on this seed: 0.135; v4: 0.285). Predicted d_x, d_t, Galilean
and scaling: not met; predicted "projective not found": met. Per-slot cos
d_t 0.90, d_x 0.90; best non-affine alignment 0.40 (v3/v4: 0.18-0.23).
Measurable cause: the slots' final losses are 6e-3..3e-2, while the true
generators through the same stepper at K = 16 sit at 5e-5..1.5e-3 (exp30) --
5 to 60 times above, so the optimiser never reached the minima the spectrum
describes. --fins-n-train 128 with batch 64 gave 2 steps per epoch, 200 per
slot against v4's 1600; the run took 13 min. v5b repeats the SAME
pre-registered prediction and criterion with v4's budget (all 1024 ICs, 16
steps per epoch); both results are reported.

The sanity block now scores the ground truth with the TRAINING loss
(horizon, seam, shooting) and adds u d_u as a non-symmetry control, so each
slot's final `equiv` can be read against the true-generator level and the
non-symmetry level in the same units; before, it printed a full-horizon
residual not comparable with the epoch losses. The run header names the
shooting mode.

**v4 (stepper, single start, K = 8) FAILS the pre-registered rule; the
loss was defective by design -- and exp30 shows it without training.**
exp29 on v4 seed 42: best generator d_x + d_t at non-symmetry share 0.285
(v3's best on Burgers: 0.059); slots 3-6 are non-symmetries (90-95 percent
outside the algebra) although their loss sat at the level of true ones;
non-affine containment 0.324 equals the untrained-network null 0.338. d_t
itself came out cleaner than ever (per-slot cos 0.955). Diagnosis: a single
start at t = 0 constrains only slices 1..K, while exp29 / table1 crop 8
slices per edge -- they scored exactly the region the loss never saw, where
the generator network is a free smooth extension (the fitted formulas are
t - t^2 polynomials). The deeper flaw: a generator symmetric only for
t < 0.063 is not a symmetry.

exp30_loss_spectrum.py -- the increment residual is linear in the
generator, so on the 27-dim monomial class the loss is a ratio of quadratic
forms; its generalised eigenvalues are the stationary loss values and the
lowest five eigenvectors are what an optimiser reaches in that class. The
conservative gap is lambda6 against the WORST combination inside the algebra
(combinations can score above every single generator). Principal cosines of
the five lowest against span(GT):

  single start, K = 8        exact   1.00 1.00 1.00 0.93 0.07
  shooting, K = 8, every 16  exact   1.00 1.00 1.00 0.98 0.96
  single start, K = 8        stepper 0.99 0.97 0.90 0.40 0.04
  shooting, K = 8, every 8   stepper 1.00 1.00 0.97 0.53 0.10
  shooting, K = 16, every 16 stepper 1.00 1.00 0.98 0.92 0.16

With the exact operator the single-start loss already loses one algebra
direction -- a design defect, not a surrogate one -- while multiple
shooting recovers the whole algebra, projective included. Through the
stepper the conservative gap is ~0.5 in every configuration: spurious directions
outscore some algebra combinations (partly a normalisation effect where
increments nearly cancel), and the minima capture four directions at
K = 16 -- scaling at 0.92 -- but not projective (0.16). The loss design is
now right; the limit is the surrogate on x-weighted generators, and exp30
selects surrogates by what the loss can identify, in minutes, instead of
by L^2 fit (which does not predict discovery).

MULTIPLE SHOOTING (`--fins-shoot-every s` with `--fins-horizon K`, stepper
only): the increment condition from data states at t_j = 0, s, 2s, ...;
use s = K to cover the time axis. Through the stepper at K = 8, s = 8: d_x
0.014, Galilean 0.015, scaling 0.037, d_t 0.047, projective 0.082 against
u d_u 0.451; projective-without-eta separates 2.3x from projective (1.3x
with a single start).

PRE-REGISTERED for v5 (K = 16, s = 16, stepper seed 42), from the exp30
spectrum: d_x, d_t, Galilean recovered; scaling recovered; projective NOT.
exp29 criterion per generator: non-symmetry share < 0.1 with the named
generator at >= 0.5 of the symmetry part. v5 matching this also tests
whether exp30 predicts discovery; the FNO generator class is far larger
than the 27-dim polynomial class, so the spectrum is a best case.

**Which surrogate supports the short-horizon loss: the stepper, after
all.** Probe of the one-shot MSE surrogate (seed 43) at K = 8, in sqrt
scale: d_x 0.076, d_t 0.24, Galilean 0.36 against u d_u 0.44 -- Galilean
barely separated from a non-symmetry. The stepper at K = 8: Galilean 0.026
against 0.44. Mechanism: the one-shot surrogate's field error (5 percent) is
the size of an 8-step increment, so the local criterion drowns; the stepper
is trained on increments (u + F(u)) and reproduces them. The earlier "no
stepper needed" was wrong for this loss -- the stepper is needed for
accurate SHORT increments, not for Galilean, and with the short horizon it
is cheap: `run_single_pde` now rolls it out only horizon + 1 steps (9
instead of 127 at K = 8), the jvp loss scores on the operator's length when
it returns fewer slices than the data, and the probe builds ground truth
per sample (it broadcast sample 0's fields, which misstated scaling and
projective, whose phi depends on u) and reports the horizon loss in sqrt
scale. Smoke-tested end to end.

**exp29 on the real artifacts, and the loss redesign it motivates.**
Analytic split of every FINS generator (Burgers, 3 seeds x 2 configs):
v3 finds two or three genuine near-symmetries per seed, all combinations
of d_x, Galilean and d_t (non-symmetry share 0.06-0.45; best: seed 43,
d_x + Galilean at 0.059), then three junk slots (0.59-0.83, 93-99 percent
outside the algebra). v1 finds nothing: all six generators are
non-symmetries (0.60-0.96, polynomial R^2 often < 0.1) -- its earlier
"non-affine containment" was the xt-shaped content of junk fields, as the
own-class null said. No generator in either configuration carries scaling
or projective (alignment with the non-affine direction 0.03-0.23). Exactly
what the full-horizon loss predicts.

SHORT-HORIZON INCREMENT LOSS (`--fins-horizon K --fins-seam m`). Compare
increments dS(Q0) - Q0 with Q(t) - Q0 over the first K slices, excluding a
band of width m L at the seam; normalise by the increments so the
discrimination survives K -> 0, where the condition becomes the local Olver
equation with propagation done by the operator. Exact solver (Burgers, m =
0.2): K = 8 gives d_x 0.000, d_t 0.025, Galilean 0.001, scaling 0.001,
projective 0.008 against u d_u 0.440 and random 0.848; the full horizon
fails scaling / projective at 0.34 / 0.84. Its floor is two orders below
the local finite-difference criterion (0.15-0.4), and at K = 8 it separates
even projective-without-eta (0.055 vs 0.008). Through the LEARNED stepper
surrogate with the implemented loss, K = 8: d_x 0.012, Galilean 0.026,
scaling 0.028, projective 0.093, d_t 0.090 against u d_u 0.443 (full
horizon through the same surrogate: scaling 0.73, projective 1.47). The
surrogate's higher floor weakens the near-miss separation (1.3-1.7x).

No stepper is needed: with the one-shot surrogate the loss uses the first K
slices of the same forward pass, so the cost equals v3's. This is also the
cleanest statement of the FINS / LieSym relation: one criterion with a
horizon, K -> 0 the local (LieSym) end, K = Nt the global (published FINS)
end; short horizons see symmetries of the equation that do not respect the
periodic domain, long ones only symmetries of the periodic problem.

**CORRECTION: projective and scaling ARE symmetries of the repo's Burgers
data; the flow-map test was the wrong criterion for them.** Point (1) of
the withdrawal below is retracted. Differentiating the solver tests whether
the whole flow map commutes with the generator -- a symmetry of the
PERIODIC PROBLEM. Scaling and projective are symmetries of the EQUATION
but change the period of the domain: their Q0 is discontinuous at the
seam (phi contains x), transport at ~2 pi |u| carries that discontinuity
across the whole domain within T = 1, and no seam window helps. Galilean
passed because it is not x-weighted. Criteria, in increasing rigour:

  flow      dS[u0](Q0) vs Q through the solver -- valid only for generators
            compatible with the periodic problem (d_x, d_t, Galilean, u d_u)
  local     the determining condition Q_t = L_u[Q] on the data, local
            x-differences, interior only: all five true generators at one
            floor (0.15-0.25 at m = 0.1-0.2 L) against u d_u 0.46 and random
            0.48-0.56 on the repo data -- projective is detectable. But the
            floor comes from third derivatives of under-resolved fronts, and
            it cannot separate near-misses (projective with eta removed 0.40
            vs true projective 0.35 on a campaign-like grid)
  analytic  `determining.py`: the determining equations of
            u_t + a u u_x = b u_xx are LINEAR in the generator; on the basis
            {1, x, t, u, xt, x^2, t^2, tu, xu} they become M c = 0 with M
            computed exactly by sympy. ker M is 5-dim and EQUALS the
            corrected algebra (principal cosines 1.000; GT residuals 1e-16);
            the same test rejects the textbook algebra for this data
            (Galilean 0.83, projective 0.68). No data derivative is taken.

exp29 now fits each generator symbolically and reports the share of the
fitted formula that the determining equations reject. Validated on known
answers: true projective / scaling / Galilean 0.001 / 0.001 / 0.000;
projective with eta removed 0.202; lone xi = x t 0.352; random smooth 0.744
-- near-misses the local criterion could not separate are rejected
cleanly, and the fit recovers phi ~ -tu + 0.16 x (0.16 = 1/2 pi). Scope: the
analytic verdict covers the POLYNOMIAL part of a generator, so it is always
reported with the fit R^2.

What stands of the withdrawal: point (2) alone -- FINS's non-affine
containment does not beat an untrained network of the same architecture
(z <= +2.5, none of six pairs at 3). There IS a non-affine symmetry in the
data; FINS does not find it beyond its architectural prior. And there is a
mechanism: FINS's loss is the global flow-map condition, which is
structurally blind to x-weighted generators on a torus whenever transport
spreads the seam (Burgers), while a seam window suffices when it does not
(heat, small alpha; exp19). The local criterion of LieSym sees them
regardless -- the cleanest argument yet for the merged paper: the global
criterion for symmetries of the periodic problem, the local criterion for
symmetries of the equation that do not respect the domain.

The time-stepping surrogate is not needed: its only motivation (Galilean)
was an artefact of the wrong ground truth, and as a flow map it does not
address the seam.

**THE NON-AFFINE RESULT IS WITHDRAWN -- it does not survive a null drawn
from FINS's own architecture.** Two independent findings killed it.

(1) On the repo's Burgers data the x-weighted generators are not symmetries
at all: through the EXACT solver (corrected algebra, seam window) scaling
scores 0.632 and projective 0.946, against d_x 2.5e-4, d_t 4.7e-2 and
Galilean 3.3e-4. The data are effectively a unit-domain Burgers equation
with nu = 0.01, so the shocks are far sharper than at Nx = 64 on [0, 2 pi)
(exp25, which used a correctly scaled solver, had projective at 2e-2
windowed). The DETECTABLE algebra here is {d_x, d_t, Galilean} -- entirely
affine. The non-affine direction belongs to projective, so there was
nothing non-affine to discover.

(2) Own-class null. FINS's generator is an FNO fed x, t and u as coordinate
channels, so its outputs carry x t-like structure by construction; a
band-limited null does not model that. Null = spans of six outputs of the
same architecture with RANDOM weights: 0.338 +/- 0.148 (56 draws), against
the band-limited 0.130 +/- 0.041. Measured containments give z = +2.0,
+1.7, +0.8 (v1r seeds) and +2.5, +1.3, -0.6 (v3 seeds) -- not one seed
reaches the pre-registered z = 3. The earlier z +8 .. +14 was an artefact
of the wrong null, exactly as for LieGAN (affine class) and LaLiGAN
(phi-only). Consistency required applying the same rule to FINS, whose
class is "whatever this network outputs".

So the headline claim -- FINS recovers structure outside the affine class
-- does not stand on this benchmark. What stands: the calibrated criterion
itself (exp1, exp4, exp14, exp18), the measurement protocol, and the audit
findings. A benchmark whose detectable algebras are affine cannot show an
unrestricted class paying off; demonstrating that advantage needs data
where a non-affine generator is actually detectable, which on Burgers means
a resolved domain.

**The stepper surrogate DOES carry Galilean -- the premise of exp27/exp28
was the wrong ground truth.** The stepper checkpoint (trajectory error
1.1 percent), corrected algebra, ||Q||-normalised: d_x 4.1e-2, d_t 7.1e-2,
Galilean 6.2e-2, against u d_u 1.09 and random smooth 1.22 -- a 17-fold
separation between the true Galilean and a non-symmetry. The earlier
"the surrogate cannot tell Galilean from u d_u" (0.81 vs 0.86) compared two
non-symmetries. scaling 0.73 and projective 1.47 track the exact-solver
floors (0.63, 0.95): unreachable from this data, not a surrogate failure.

**Discovery through a rollout operator is launch-bound.** Measured on a
GPU: ~2.8 s per evaluation, so the published defaults (1024 ICs,
batch 8, 100 epochs, 6 slots = 77k unrolls) exceed 10 h without finishing
one slot. `--fins-batch` and `--fins-n-train` cut the evaluation count
(the surrogate still trains on everything), train_fins prints an ETA after
the first epoch and switches to per-epoch progress when an epoch exceeds
20 s.

**LieSym code checked on its own data and solvers: the Galilean ground
truth is wrong in all three equations.** Residual through LieSym's own
solver (seam-windowed; d_x control 1e-6 .. 1e-4): heat -- paper Galilean
(2t, 0, +xu) 0.98, correct (2 alpha t, 0, -xu) 1.1e-2; paper projective
1.01, correct with alpha 1.2e-2 (heat solver itself is consistent, 7e-3).
Burgers -- the same 2 pi solver scaling as here (textbook 0.87, rescaled
0.010); paper Galilean 0.99 vs (2 pi t, 0, 1) 1.4e-4; paper projective
1.16 vs (xt, t^2, x/2pi - tu) 0.12. KdV -- solver scaled by (2 pi)^3 with the
dispersion sign reversed relative to its docstring; paper Galilean
(t, 0, 1/6) 4.63 vs (12 pi t, 0, 1) 4.8e-4. Only translations, u d_u and
scalings are right. LieSym stores metrics, not fields, so it cannot be
rescored: a rerun with serialisation is required. Details and the two fix
options in LieSym_audit.md.

**THE REPO'S BURGERS SOLVER INTEGRATES A RESCALED EQUATION -- found while
verifying FINS generators through the exact solver (exp29).**
`data_generation.solve_burgers`: `fftfreq(Nx, d=dx/(2 pi))` already yields
integer wavenumbers for L = 2 pi, and `angle_k = 2 pi k` multiplies again.
The stored data obey u_t + 2 pi u u_x = 4 pi^2 nu u_xx on the labelled
x in [0, 2 pi) -- a unit-domain Burgers equation with nu = 0.01. Measured:
PDE residual 0.85 for the textbook form, 0.035 for the rescaled one (the
FD-in-time floor). Through the solver, Galilean (t, 0, 1) -- the repo GT
-- has residual 1.22; (2 pi t, 0, 1) has 1.5e-4, the level of the d_x
control. Projective (xt, t^2, x - tu) 1.16; (xt, t^2, x/2 pi - tu) 7.2e-2
(seam-windowed; resolution floor). The heat and KdV solvers of this repo
are consistent (FD u_t vs alpha u_xx 2.5e-3; KdV via solver_rhs_ut and the
oracle).

FIX: `ground_truth.burgers_generators(a=2 pi)` -- Galilean (a t, 0, 1),
projective (x t, t^2, x/a - t u); d_x, d_t, scaling unchanged; a = 1 is the
textbook algebra (exp17 stands for the textbook equation). The data are NOT
regenerated: every method trained on the same data, so rescoring against
the algebra the data actually carry is fair and costs seconds. `table1.py`
and `rescore_from_artifacts.py` now recompute Burgers GT; the GT stored in
existing artifacts is stale.

UNAFFECTED: the Burgers non-affine test. The non-affine direction is
identical under old and corrected GT (principal cosine 1.000), and for a
one-dimensional direction the sin^2 weights cancel, so every non-affine
score and null is unchanged -- the FINS result stands. Heat and KdV
results are unaffected.

CHANGED INTERPRETATION: under the corrected algebra that direction's sine
is 0.85, not 0.25. The old projective carried an affine phi = x term (up to
2 pi in size) that no other generator cancels; with phi = x/2 pi it shrinks
2 pi-fold. The non-affine component is ~72 percent of the true projective,
not ~6 percent, so the argument "per-slot cosine is structurally blind to
it" is WITHDRAWN pending rescoring; the old repo Galilean is only 0.71-
aligned with the true one on this data.

WITHDRAWN: every Burgers-Galilean conclusion built on repo-solver data and
the textbook GT -- the probe's "the surrogate cannot tell Galilean from
u d_u" (both were non-symmetries of the data); the Galilean columns of
exp27 and exp28 (exp28's printed exact-operator line was a literal copied
from exp26, which ran on exp_common's correctly scaled solver); and the
"16 orders of degradation" reading, which compared exp25 (correct solver)
with probe numbers (repo solver). exp26 is unaffected (exp_common solver
throughout). The d_t / eta-channel diagnosis on Burgers stands (d_t is
unaffected by the rescaling).

BASELINES: LIG (method_lig.py:236) and LieNLSD (method_lienlsd.py:182)
hard-code the textbook Burgers form, so on this data they searched for
symmetries of an equation the data do not satisfy. Their Burgers results
are confounded, and the "equation-aware included" part of the non-affine
claim is withdrawn until they are rerun with a = 2 pi. Equation-free
comparisons stand.

LieSym CODE: the same construction appears in solve_burgers
(fftfreq(Nx, dx/2 pi) then (2 pi k)^2) and in solve_kdv
(Lk = -i (2 pi k)^3 with k already integer); solve_heat uses
fftfreq(Nx, 1/Nx) 2 pi with no domain length; its GT is textbook (Galilean
(t, 0, 1), projective with the original phi = -(x + t u)). To be verified
numerically on LieSym's own data before any LieSym claim is revised.

**exp29 -- verification and interpretation of FINS generators.** Per
generator in an artifact: windowed exact-solver residual, symbolic fit on
{1, x, t, u, xt, x^2, t^2, tu, xu} across the reference samples, and the
split into span(GT) plus outside share; also the combination in the span
that best aligns with the non-affine direction. Validated with the
corrected algebra on synthetic truth: GT Galilean 1.8e-4, projective at
its resolution floor 6.2e-2, projective with eta removed 0.14 (what a v1-
like generator with the right xi and phi but no eta partner should score),
a lone xi = x t fragment 1.00, random smooth fields 0.40-0.91; the symbolic
fit returns xi ~ xt, eta ~ t^2, phi ~ x - tu at R^2 1.00.

**v1 rerun (published FINS configuration, Burgers, 3 seeds): the
pre-registered prediction FAILED, and the claim got stronger.** The rerun
reproduces v1 faithfully (xi/eta/phi 0.97/0.02/0.01, eta collapsed; cos
0.246 against 0.226-0.263 in the first campaigns); seeds 43/44 ran on the
very MSE surrogates of the v3 campaign, seed 42 on a retrained one.
Prediction registered beforehand: with eta collapsed, v1 cannot reach the
t^2 part of the projective generator and should capture the non-affine
structure markedly worse than v3. Observed: 0.636 / 0.590 / 0.450, z +12.3
/ +11.2 / +7.8 -- all three seeds above 3, against v3's two.

Why the prediction failed: the non-affine direction of the Burgers algebra
is 95 percent xi-channel (xi/eta/phi 0.95/0.03/0.02) -- essentially the
bilinear xi = x t component of the projective generator. The eta part
(t^2) we expected to matter carries 3 percent of it.

Secondary mechanism check (not part of the pre-registered test; a HARSHER
null applied to a favourable result): is v1's capture explained by its
97 percent xi-concentration, as LaLiGAN's was by its phi-concentration? A
xi-only smooth null reaches 0.144 against 0.126 for the all-channel band
null -- random smooth xi-fields rarely align with the specific x t shape.
Against it v1r stays at z +11.6 / +10.5 / +7.2 and v3 at +13.3 / +9.0 /
+2.5. (Nulls here use 200 trials; the table1 run used 100, so z differs by
Monte-Carlo noise of the null, ~1-2 units at this level.)

RESULT: five of six (configuration x seed) pairs clear z = 3 under either
null. The capability is intrinsic to FINS's formulation -- the
unrestricted generator class -- and was already present in the published
configuration; the v3 corrections improved the AFFINE part (d_t via the
revived eta channel, per-slot cos 0.25 -> 0.67), orthogonally. The
published per-slot metrics hid it completely: v1's Burgers cosine sits at
z -6.9 below its floor.

SCOPE of the claim, stated precisely: FINS's span contains the component
along which the Burgers algebra leaves the affine class. It does NOT
isolate the projective generator as a whole (per-slot cos ~0.06). No
affine-class method can represent that component; no other tested method,
equation-aware ones included, recovers it robustly. On heat nobody
recovers the non-affine part; on KdV the algebra is affine.

**NON-AFFINE TEST, FINAL -- the pre-registered decision rules applied.**
Rules stated before the per-seed data: the Burgers win stands if FINS has
z > 3 on at least two of three seeds; the heat verdict flips only if the
all-slots variant exceeds z = +3. Applied identically to every method.

Burgers, FINS per seed: 0.706 (z +14.0), 0.526 (z +9.6), 0.250 (z +2.9) --
two of three above 3, the win STANDS. Unchanged under --eff-thresh 0
(FINS already has all six slots significant). The weak seed 44 is the one
that also lagged on d_G in v3: a weak seed overall, not a targeted failure
of this test. Same rule for the others: LIG exceeds 3 on one seed only
(+4.9, or +6.4 with all slots), LieNLSD on one (+3.6), LaLiGAN at the null,
LieGAN and Augerino 0 by construction.

Heat, FINS with all six slots: 0.083 / 0.027 / 0.138 against a null of
0.811, z -22.8 .. -26.5. The verdict does NOT flip: the non-affine
structure was not hiding in the low-norm slots. No method recovers it.
KdV: algebra affine, test not applicable.

RESULT: on Burgers, FINS is the only one of six methods -- equation-aware
included -- whose recovered span robustly contains the non-affine part of
the symmetry algebra, structure that affine-class methods cannot represent
at all. It is the first result of the campaign that simultaneously clears
the own-class null, holds across seeds, holds under the effective-span
threshold, and was scored with a metric frozen before the run. It is
FINS-v3's; whether the published configuration already had it is answered
by running --nonaffine on the untagged base campaign.

**First run of the non-affine test on the v3 artifacts (metric frozen
beforehand).** BURGERS -- non-affine subspace 1-dim, sine 0.25 (the
projective generator's xt, t^2, -tu part): FINS 0.526 against a band null
of 0.130, z +9.6 (ceiling +21.1), the ONLY method above chance; LieGAN and
Augerino 0 by construction, LaLiGAN -1.0, LieNLSD -0.3, LIG +2.2. The
unrestricted class pays off exactly where it should. HEAT -- non-affine
subspace 2-dim, sines 0.68 / 0.21: FINS 0.000 on every seed, z -10. Its
effective span (3 slots) is purely affine: it found d_x, d_t, u d_u,
scaling and not the non-affine Galilean (phi = -x u) or projective
(phi = -x^2 u), though the constrained ansatz a(x,t) u + b admits both --
consistent with u, x u, x^2 u being near-collinear on heat. KDV --
algebra entirely affine, no contrast possible.

Two robustness checks pending, both announced before the run: the Burgers
median carries a seed spread of 0.456, and the heat verdict rests on an
effective rank of 3 set by the 0.2-of-max norm threshold, which could hide
non-affine structure in the three low-norm slots. `table1.py` now prints
per-seed values in the --nonaffine block and exposes `--eff-thresh`
(0 = all slots). Both variants are to be reported, not chosen between.

**Non-affine structure: a coordinate-free test of FINS's one structural
advantage.** FINS's class is unrestricted; LieGAN and Augerino emit affine
fields, LaLiGAN phi-only. The question "can FINS reach what they cannot"
is well posed only on the part of the algebra outside the affine class.
Three steps, each correcting the previous one:

(1) Centring the domain (x in [-L/2, L/2)) raises the per-generator novel
fractions (heat Galilean 0.08 -> 0.90), but it is a relabelling: under
x -> x - c, Galilean -> Galilean + c u d_u, scaling -> scaling - c d_x,
projective -> projective - 2c Galilean - c^2 u d_u. The algebra's SPAN is
unchanged, so centring improves the conditioning of the attribution
metric, not what is discoverable. Proposal withdrawn.
(2) The size of the non-affine part must be measured basis-free: singular
values of a set of normalised generators change under a non-orthogonal
change of basis of the same span. Correct measure: sines of the principal
angles between span(GT) and the affine class, from an orthonormal basis --
verified invariant to 1e-8 under relabelling. On the repo's heat data:
2-dim, sines 0.68-0.98 and 0.17-0.70 across samples. Structure no affine
method can represent at all.
(3) The test compares NON-AFFINE PARTS: the non-affine subspace of the
algebra is a projection and does not lie in span(GT), so containing it
directly scored a perfect method below chance. Both spans are projected
onto the affine complement and compared there, weighting directions by
sin^2 -- unweighted, the weak direction (sine 0.24) is drowned by ~1
percent field error, since unstructured error is almost entirely
non-affine. Validation, fixed before any real artifact was scored: a
perfect method with 1 percent white or smooth error scores 0.991 (z +3.0),
3 percent 0.955 (z +2.0); affine methods score exactly 0. POWER LIMIT:
the rank-6 band null already reaches 0.894, so the test's ceiling on heat
is z +3.3 even for perfect recovery -- `table1.py --nonaffine` prints the
ceiling beside every row.

**exp28 -- time-stepping surrogate: a better operator, not a Galilean
one.** u(t) -> u(t + dt) as u + F(u), F a coordinate-free 1-D FNO, rolled
out; small nonlinear Burgers, ||Q||-normalised residuals:

| surrogate | rel L2 | R(Galilean) true | R(u d_u) spurious |
|---|---|---|---|
| one-shot FNO u0 -> u(., t) (repo) | 0.237 | 0.906 | 0.492 |
| stepper rollout, 16 train ICs | 0.044 | 1.632 | 0.713 |
| stepper, 128 / 512 train ICs | 0.039 / 0.033 | 1.600 / 1.626 | 0.721 / 0.724 |
| exact operator | -- | 0.000 | 1.244 |

5-7x lower trajectory error and a markedly more faithful Jacobian on the
non-symmetry u d_u (the one-shot map is 4x too symmetric there), but the
Galilean residual worsens and does not respond to data (16 -> 512 ICs) --
not a coverage problem. The Galilean response 1 - t u_x grows with time
and is concentrated at fronts, where accumulated positional error bites.
Because the probe does not predict discovery (tanv3), the stepper is not
judged on the probe alone: it is wired as `run_single_pde.py
--surrogate-arch stepper` (Burgers/KdV; separate checkpoint name, so it
never collides with the one-shot cache), smoke-tested end to end. The
decisive test is a FINS run through it.

**Full v3 table under own-class nulls: the metric line CLOSES, and almost
nothing survives -- for any method.** The last class restriction applied:
LaLiGAN's extractor sets xi = eta = 0 by construction (a "vertical" field,
phi = dv), so its class is phi-only and d_x, d_t, scaling are
unrepresentable for it. Its apparent Burgers capture (Galilean +4.1,
projective +3.7 against the three-channel band null) dissolves against a
phi-only smooth null: +0.9 and +0.7 -- the novel components of Galilean
and projective are 96-97 percent phi, and a random smooth phi-subspace of
rank 6 contains them at 0.77 and 0.73 by chance. `table1 --span` applies
this null automatically.

What survives own-class nulls at z >= 3, all three equations, all six
methods: LieGAN finds KdV Galilean (+4.1, affine null); FINS through the
EXACT operator finds KdV Galilean (+11.0) and d_x (+3.6) -- a diagnostic,
not equation-free. LieNLSD's Burgers Galilean / projective (+5.5 / +5.1)
and LIG's KdV Galilean (+13.9) stand only against the band null; both are
equation-aware with structured classes (monomial library; u-restricted)
whose own-class nulls are not implemented, and LieGAN's case shows such
nulls can remove most of a band-null signal.

FINS-v3, equation-free: its per-slot cosine wins -- heat 0.636, Burgers
0.669, best equation-free on both -- do NOT survive. Heat: novel
containment at null on all six generators. Burgers: d_x +1.3, d_t +1.8,
scaling +1.6 (marginal), Galilean 0.0, projective -0.2. The cosine
credited it for the dominant, correlated directions.

A RESOLVABILITY LIMIT of the test itself, measured on heat: the novel
fractions are u d_u 0.06, Galilean 0.03, projective 0.05. At alpha = 0.01
their xi and eta parts are negligible and their phi parts u, x u, x^2 u are
nearly collinear for x >= 0, so their independent components are 3-6
percent of the field -- the same order as the generator floor
(eps_fit ~ 1 percent, exp16). Individual recovery of these three is not a
well-posed measurement at this resolution for ANY method; the test is
informative only where the novel fraction clearly exceeds the method's
field error (heat d_x, d_t, scaling at 0.42-0.55; Burgers and KdV at
0.35-0.82).

Stopping rule, stated explicitly to avoid a garden of forking paths: an
own-class null is applied wherever a method's class is STRUCTURALLY
restricted and verified in code (LieGAN, Augerino: affine; LaLiGAN:
phi-only). LieNLSD and LIG remain on the band null and are flagged. No
further nulls are added in response to individual results.

**The null must come from the METHOD'S OWN hypothesis class.** LieGAN
emits affine fields, v = L (x, t, u, 1)^T with L in R^{4x4}, so each of
xi, eta, phi is c0 + c1 x + c2 t + c3 u -- a 12-dim class. All four KdV
generators lie inside it (containment 1.000 each), as do 4 of 5 Burgers
and 4 of 6 heat generators. A random element of such a class contains the
answer by construction, so a band-limited null credits the method with
discovering what its parameterisation already encodes. Demonstration: a
method emitting a RANDOM 6-dim affine subspace -- no discovery at all --
has raw containment 0.94-0.96 for every KdV generator and would show
Galilean at z +6.5 against the band null; against its own class it scores
z -1.8 .. +1.0, i.e. nothing.

Re-scored against its own class, LieGAN on KdV (r = 6): d_x +2.2
(marginal), d_t -6.6, Galilean +4.2 (genuine), scaling +0.9 (reach). The
earlier reading "LieGAN holds the independent parts of three of four
generators" was an artefact of the wrong null; the defensible claim is
that LieGAN genuinely finds Galilean, beating even a random affine
baseline. `table1.py --span` now picks the null per method -- affine class
for LieGAN and Augerino (whose fixed group is its whole output, so it can
never score above its own class), band-limited for FINS, LaLiGAN, LIG and
LieNLSD -- and labels which was used.

Known limitation: LieNLSD's monomial library and LIG's u-restricted
parameterisation are also structured classes and deserve their own nulls;
both are equation-aware and their band-null z-scores are already modest,
so this is recorded rather than implemented.

This is the quantitative form of the representational-reach argument the
LieSym paper made qualitatively for Augerino, and it generalises: any
claim that a method DISCOVERED a symmetry needs a null drawn from that
method's own hypothesis class.

**Novel containment needs a SMOOTH null -- and with it, the KdV method
ranking inverts.** The `rand` reference sqrt(r / D) printed by the first
`--span` version assumed random DIRECTIONS; GT novel components are smooth
fields, and a span of smooth fields contains them cheaply. Band-limited
null (r random triples, GT-mean amplitude spectrum and energy per
component, the exp20 construction), novel containment by chance on KdV:
r = 2 -> d_x 0.35, d_t 0.51, Galilean 0.12, scaling 0.20; r = 6 -> 0.50,
0.76, 0.29, 0.45 -- ten to forty times the random-direction level.
`table1.py --span` now computes this null at each method's own effective
rank and prints z-scores (`--null-trials`, default 100).

Re-read against it (novel containment, z at matched effective rank):

| KdV | d_x | d_t | Galilean | scaling |
|---|---|---|---|---|
| FINS oracle, FD u_t (r 6) | 0.73 (+3.8) | 0.31 | **0.99 (+11.7)** | 0.19 |
| FINS oracle, exact u_t (r 2) | 0.36 (~0) | 0.78 (+1.8) | 0.03 | 0.01 |
| FINS v3, surrogate (r 6) | 0.32 | 0.21 | 0.30 (~0) | 0.36 |
| LieGAN, v3 table (r 6) | **0.83 (+5.5)** | 0.48 | **0.95 (+11)** | 0.78 (+3.0) |

Through the exact operator FINS genuinely captures Galilean and d_x (FD
u_t) or, marginally, d_t (exact u_t); through the surrogate, nothing clears
the null. And LieGAN -- per-slot Hungarian cosine 0.363, below the null
floor -- genuinely holds the independent parts of three of the four KdV
generators. On a correlated algebra, per-slot cosine and novel containment
against a matched null RANK METHODS DIFFERENTLY.

Calibration of the metric itself: the ideal rank-2 recovery (the GT's own
top-2 singular directions) scores novel z of -2.7 / +0.9 / -0.7 / +2.3.
Not a defect -- a 2-dim span cannot hold the independent parts of four
generators, and top singular directions optimise variance, not novelty --
but it fixes the reading rule: compare novel containment only at matched
effective rank, and only against the smooth null.

Bug fixed on the way: `--span` took GT names from the first artifact of
the list rather than from the equation's own, which mislabelled the heat
block (u d_u shown as Galilean); KdV and Burgers were correct by the
coincidence of a shared name prefix.

This completes the pattern across Table 1: every metric the papers report
needed correcting -- ACE (scale, bracket, degeneracy), d_G (configuration
penalty), rank (anchor), and now per-slot cosine (blind to subspace
recovery on correlated algebras).

**KdV oracle + exact u_t: the aliasing diagnosis confirmed IN DISCOVERY,
and a metric blind spot exposed.** Exact operator, solver-RHS u_t, v3
FINS settings, seed 42. Versus the same oracle with FD u_t: eta share
0.01 -> 0.21 (GT 0.29), d_t 0.25 -> 0.78 -- the temporal-aliasing root
cause holds in the discovery loop, not only in the residual test. For the
FIRST time in the campaign the raw-anchor rank criterion produced a clean
norm gap: ||v_perp|| 1.72, 1.47, then 0.068, 0.039, 0.026, 0.036 -- two
directions, then genuine exhaustion. But Galilean fell 0.995 -> 0.37 and
d_x 0.65 -> 0.16, and aggregate cos barely moved (0.50 -> 0.54): with FD
u_t the loop took the xi-carried pair, with exact u_t an eta-including
pair; in both cases about TWO directions.

The reason is the ground truth, not the loop. On this domain the KdV
algebra is numerically near-2-dimensional: |cos| d_x~Galilean 0.854,
d_x~scaling 0.785, Galilean~scaling 0.637 (xi = 1 and xi = t are 0.87
correlated on t in [0,1]); singular values 1.60 / 1.05 / 0.51 / 0.29,
effective dimension 2.98 of 4; the best 2-dim subspace contains all four
generators at 0.93-0.98. The found slots' mixed cosines (v1: 0.77 / 0.63
/ 0.67 / 0.85 with all four) are what a vector in that dominant subspace
looks like.

METRIC BLIND SPOT: per-slot Hungarian cosine at full rank cannot credit
subspace recovery on a correlated algebra. Calibration on synthetic KdV-
like data: a method whose two strong slots are EXACTLY the GT's top-2
singular directions scores d_G 3.13 and cos 0.47 -- below the band-limited
null. Raw span containment errs the other way (correlation inflation,
exp19). `table1.py --span` now reports both span containment and
NOVEL-component containment (the part of each GT generator orthogonal to
the others) over an EFFECTIVE span of slots with norm >= 0.2 max, so
collapsed deflation slots do not inflate it; ceilings for an ideal rank-r
recovery on the real KdV data were printed when it was added.

**Autograd through the surrogate's t input, MEASURED (Burgers, where a
clean reference exists).** The earlier withdrawal argued from architecture
("t is a channel, not a coordinate"); that framing was imprecise --
autograd is defined for any input, channel or not. The question is whether
the derivative equals d_t u. Reference u_t from the same solver on a 16x
finer time grid, FD there, subsampled; surrogate trained to 24 percent L2:

| source of u_t | relative error |
|---|---|
| FD in t of the DATA (what FINS uses) | 0.033 |
| FD in t of the surrogate's OUTPUT | 0.706 |
| autograd, uniform shift of the t input | 0.979 |
| autograd, pointwise d out_j / d t_j | 0.974 |

The input-derivative carries essentially no u_t, and is markedly worse than
differencing the SAME surrogate's output (0.98 vs 0.71), so it is not
merely surrogate inaccuracy. Correct reason: for Burgers and KdV the t
channel is IDENTICAL in every training example (one fixed grid 0..T), so
the network never observes its output respond to a change of the t input;
the derivative is off the training manifold and unconstrained. A property
of the training data, not of the architecture. Training on time-shifted
trajectories would expose it, but (i) the surrogate would then learn
time-shift structure explicitly -- borderline circular for discovering
d_t -- and (ii) on KdV it still cannot recover waves above the stored
grid's Nyquist frequency, which no model trained on those samples can
reconstruct.

**Autograd u_t through the surrogate: proposal WITHDRAWN before coding.**
The surrogate takes time as an input CHANNEL mixed by 2D spectral
convolutions, so the output at t_j depends on the t-channel at every time;
d(output)/d(t-input) is the response to shifting a coordinate feature the
network never saw shifted, not d_t of the trajectory. And the surrogate is
trained on the already-aliased data, so the fast dispersive waves are not
in it to be recovered. Stronger conclusion: on KdV, u_t is not recoverable
from the stored trajectory by ANY means -- neither finite differences nor
a model trained on it. The fast component is absent, not hidden.

**Exact u_t closes the KdV diagnosis (oracle operator, ||Q||-normalised).**
`data_generation.solver_rhs_ut` returns the right-hand side the KdV solver
itself integrates (u_hat_t = i k^3 v_hat - 3 i k FFT(u^2), both 2/3-
dealiased; the solver's hyper_viscosity argument is accepted but unused, so
high modes are undamped and rotate at k^3 up to ~9e3, against a campaign
time Nyquist of ~400). GT residuals through the exact operator:

| generator | u_t by FD on the grid | u_t = solver RHS |
|---|---|---|
| d_x | 4.9e-7 | 4.9e-7 |
| d_t | **0.629** | **2.4e-5** |
| Galilean | 3.4e-5 | 3.4e-5 |
| scaling | 0.726 | 0.784 |

With the correct u_t, d_t is an exact symmetry (26000x lower residual);
d_x and Galilean carry no eta and are unaffected. So through an exact
operator with an exact time derivative the whole KdV essential trio
passes. Per-generator attribution of the KdV failure is now complete:
d_x and Galilean are blocked by the SURROGATE (the oracle found Galilean
at 0.995), d_t by the NON-RECOVERABILITY of u_t from the trajectory,
scaling by the seam under dispersion (exp18). None of the four is a
failure of the discovery loop itself -- which the next run tests directly.

Wired as `--provided-ut` (KdV only; run_single_pde and campaign), through
the train_fins loader into the jvp loss and into the post-slot
diagnostics, which previously still scored the FD target and printed
misleading residuals. Uses the solver, so a run with it is the diagnostic
ceiling, not the method. End-to-end smoke (oracle + provided u_t, KdV,
tiny budget) completes through artifacts, summary and json.

Data-requirement corollary for the paper: equation-free discovery on a
dispersive PDE needs u_t MEASURED at collection time; it cannot be
reconstructed from snapshots at any practical sampling.

**Three results from one GPU session; one prediction falsified, one root
cause found.**

(1) THE PROBE DOES NOT PREDICT DISCOVERY -- the pre-registered test
failed. v3-FINS on the tangent surrogate (Burgers, seed 42): d_G 2.05,
cos 0.674, eta 0.17, d_t 0.998 -- the same as on the MSE surrogate
(2.15 / 0.68 / 0.17), inside v3's seed spread (d_G +/- 0.3). The probe had
rated the tangent surrogate worse on EVERY generator (d_t 3x). Absolute GT
residuals through a surrogate therefore do not forecast what the loop
finds, and the recommendation to use the probe as a pre-discovery gate is
WITHDRAWN. Plausible reason: discovery is ranking, not thresholding -- the
loop takes the lowest-residual directions among candidates, and the
uniform degradation the probe measured barely reorders them. Corollary for
exp27: tangent training degrades the surrogate by the probe's measure but
is NEUTRAL for discovery; the "tangent makes it worse" reading holds only
at the probe level.

(2) KdV THROUGH THE EXACT OPERATOR (oracle, seed 42): d_G 3.41 -> 2.91,
cos 0.22 -> 0.50 versus the surrogate run. Galilean found at cos 0.995 --
through the surrogate it was never a best match, so for Galilean the
surrogate WAS the blocker. But d_t is not found (0.25) and the eta
channel stays at 0.01 even with the exact operator. Under both nulls
(cos z -1.21 band).

(3) ROOT CAUSE OF THE KdV eta COLLAPSE: u_t IS TEMPORALLY ALIASED. FINS
forms Q = phi - xi u_x - eta u_t with u_t from finite differences on the
stored time grid (dt = 0.0079). Measured against the SAME solver's
trajectory saved K times finer, with no PDE-side reference:
heat 0.25 percent, Burgers 3.0 percent, KdV 90 / 97 / 98.5 percent for
K = 4 / 16 / 64 -- and on KdV the fine references do not converge either
(K = 16 vs 64 differ by 97 percent). The solution carries linear
dispersive waves at omega ~ k^3 up to ~3e4; their amplitude in u is small
but u_t weights them by k^3, so u_t is dominated by components with
period ~2e-4, below even the finest grid tried. The time derivative of
KdV is not recoverable by finite differences from a stored trajectory at
any practical resolution, so the characteristic of every eta-carrying
generator is ~100 percent wrong regardless of the operator -- which is
exactly why eta ~ 0 in v1, v2, v3 AND the oracle. (An earlier comparison
against a spectral PDE right-hand side showed 500 percent on Burgers;
that was Gibbs noise in the reference at under-resolved shocks, and the
PDE-free test above supersedes it.)

The eta collapse therefore has two independent causes: the ||v|| bias,
fixed by ||Q|| and sufficient on Burgers where u_t is resolvable; and
temporal aliasing of u_t, KdV only, which neither normalisation nor a
better operator can touch. CORRECTION to exp25: its claim that the KdV
essential trio sits in the data at machine precision holds for d_x and
Galilean, NOT for d_t -- exp25 took u_t from the PDE right-hand side,
i.e. from equation knowledge FINS does not have.

Candidate equation-free fix: take u_t = d S(u0; x, t) / d t by autograd
through the surrogate's t input instead of finite differences on the
stored grid. The surrogate is a smooth function of t, so the derivative is
alias-free by construction, and the d_t condition becomes the surrogate's
own time-translation equivariance -- testable entirely within the learned
operator, no equation required.

**`--fins-only` smoke caught a crash that would have cost a GPU hour.**
The baselines block defined `all_method_names`; skipping it left the name
unbound, and the run died AFTER the FINS artifacts were saved but BEFORE
the results json and the summary -- in Colab, an hour of FINS training
with no table. Fixed (defined unconditionally), and the whole path now
runs end to end: oracle build -> FINS -> corrected closure -> artifacts
-> summary -> json -> table1 --fins-tag. `--fins-epochs N` added for
smoke runs. The pipeline's own sanity block, run through the oracle on
heat, reads d_x 3.6e-10, u d_u 1.3e-14, d_t 7.1e-3: the oracle works
inside the production code path, not only in isolation.

**exp27 at CAMPAIGN scale: tangent training makes the Burgers surrogate
WORSE on every generator.** Two independently retrained MSE surrogates
(the probe's own, and the v3 campaign's) agree to within a few percent on
every probe line, which gives a retraining-noise floor. The tangent-trained
surrogate (4 mode directions per IC, lambda 1) falls outside that floor on
every line, always in the bad direction (||v||-normalised probe readings):

| | MSE A | MSE B | tangent | tangent / MSE |
|---|---|---|---|---|
| fit L2 | 0.0170 | 0.0201 | 0.0335 | 1.7x worse |
| d_x | 1.98e-3 | 2.76e-3 | 1.02e-2 | 3.7x |
| d_t | 0.145 | 0.152 | 0.471 | 3.1x |
| Galilean | 0.812 | 0.811 | 0.905 | 1.12x |
| scaling | 0.428 | 0.422 | 0.464 | 1.10x |
| projective | 0.930 | 0.927 | 1.032 | 1.11x |
| u d_u (spurious) | 0.866 | 0.863 | 0.989 | -- |

The toy-scale negative result transfers and strengthens: the fit degrades,
all five GT residuals degrade -- including d_x, which the FNO carries
architecturally -- and Galilean still cannot be told from the spurious
u d_u. With exp26 (joint training manufactures injected symmetries) this
closes the data-side line on Burgers: equivariance of a learned operator
under nonlinear symmetries is not bought by symmetry-agnostic
interventions on its training. The FINS row of that run is NOT
interpretable (published FINS settings, not v3; live table uses the
asymmetric default crop). Operational note: the cached
S_theta_burgers_s42_g64x128.pt is now the TANGENT surrogate (checkpoint
names carry no tag) and must be deleted before any further Burgers run.

**`pde/oracle_jacobian.py` — an EXACT operator for the discovery loop at
surrogate speed (the KdV plan, step 1).** On KdV every GT generator fails
through the trained surrogate (all >= 0.50), so the question that decides
the next step is whether the DISCOVERY LOOP recovers the KdV algebra when
the operator is exact. Running FINS through the differentiable solver is
~100x slower than through a surrogate (10^4 sequential ETDRK4 steps per
forward). But the criterion only uses the Jacobian ACTION dS[u0](Q0) at
the training ICs, and dS[u0] is a fixed linear map per sample: precompute
J[b] (2 Nx batched central-difference solves in float64, independent of
N), wrap it as the first-order Taylor operator S(u0) = u_data + J (u0 -
u0_data) with the same (u0, xc, tc) -> u signature train_fins expects,
and the loop runs at surrogate speed with an exact operator.

Validated on heat (whose exact Jacobian is linear and known): forward at
training ICs reproduces the data to 0.0, jvp equals J delta to 0.0, and
the GT residuals through the oracle read d_x 2.7e-6, u d_u 1e-7 (machine),
Galilean / scaling / projective 0.44-0.54 -- exactly the exp1 torus pattern
(essential trio passes, seam-broken generators fail under the global
residual), which is what a correct exact operator must show. One bug
caught on the way: heat data are multi-scale, and a Jacobian taken at a
single (L, T) is the Jacobian of a different operator; now per-sample.

Wired as `run_single_pde.py --oracle-operator --oracle-n 128 --fins-only`
(the Jacobian is cached in FINS_RESULTS_DIR). NOT equation-free -- it
uses the solver -- so the resulting FINS row is the loop's ceiling, a
diagnostic like LieNLSD, never the method. KdV precompute cost on a T4:
128 batched solves of ~10^4 steps, estimated 20-60 min, one-off.

The KdV plan this feeds: (1) oracle FINS -- if the loop recovers the
essential trio with an exact operator, the surrogate is the sole blocker;
if not, KdV is closed at the method level. (2) If blocker = surrogate, the
remedy is architectural, not more data: a time-stepping surrogate
u(t) -> u(t + dt) accumulates far less dispersive phase than the one-shot
u0 -> u(., t) map -- which turns the criterion into the local Olver form,
i.e. the LieSym formulation may simply be the right tool for dispersive
PDEs. (3) Alternatively change the KdV protocol to the wide domain / short
horizon where exp18 showed 4/4 detectable. (4) Or report the class-level
failure: on KdV every equation-free method sits under both nulls.

**exp27 — tangent (Jacobian) training of the surrogate: a NEGATIVE result
at toy scale.** Hypothesis: the surrogate's failure on Burgers Galilean is
Jacobian infidelity at good L2, so matching jvp(S) to the solver's own
finite-difference response on RANDOM directions (no symmetry named) should
fix it. Controlled test, small nonlinear Burgers (nu 0.02, amplitude 2,
16 ICs, 8-mode FNO, 500 epochs, identical seeds), ||Q||-normalised
residuals through each surrogate:

| surrogate | rel L2 | held-out Jacobian err | R(Galilean) true | R(u d_u) spurious |
|---|---|---|---|---|
| MSE only | 0.283 | 0.599 | **0.382** | 0.437 |
| + tangent, random Fourier (256 pairs) | 0.273 | 0.455 | 0.651 | 0.447 |
| + tangent, IC differences (256) | 0.257 | 0.460 | 0.620 | 0.452 |
| + tangent, pure modes k=0..3 (112) | 0.278 | 0.508 | 0.476 | 0.407 |

Tangent training does what it is built for -- held-out Jacobian error
falls 15-25 percent at unchanged L2, and unlike joint training (exp26) it
does not manufacture the spurious symmetry -- but NO direction pool
improves Galilean equivariance; all three leave it worse than plain MSE.
Average Jacobian accuracy and accuracy along the specific nonlinear
coupling that the Galilean response 1 - t u_x encodes (mean -> advection
speed) are different targets, and a capacity-limited model redistributes
toward the pool. The "raise Jacobian fidelity" diagnosis was too coarse.

Scale caveat: this is a 27 percent-L2 toy against a 1.7 percent campaign
surrogate, so the only fair test of transfer is one campaign-scale
Burgers surrogate on GPU with the probe before and after (~20 min):
`run_single_pde.py --pde burgers --seed 42 --n-sym 6 --surrogate-tangent 4
--tangent-mode modes`, then `probe_surrogate.py --pde burgers --seed 42`
(delete the cached S_theta first, or the probe reloads the old one). Wired
into run_single_pde and campaign as options that default to off. FD
targets verified clean (eps vs eps/2: 2.8e-5). Reproducible harness:
exp27_tangent_arm.py (arms) over the pools built in the exp27 run.

Also refuted on the way: the orbit-coverage explanation for Burgers
Galilean (constant offsets are present in 38 percent of ICs, KdV 88).

**exp26 — joint surrogate + equivariance training MANUFACTURES
symmetries (proposal rejected).** Idea under test: EM-style alternation,
adding the Olver residual of the discovered generators to the surrogate
loss. Controlled design on small nonlinear Burgers (exact operator:
R(u d_u) = 1.24, R(Galilean) = 0): surrogate trained to plateau, then
equal-length arms from one checkpoint. Control (MSE only): L2 0.208,
R(u d_u) 0.311, R(Galilean) 0.394. Joint with the SPURIOUS u d_u term:
lambda 0.1 -> R(u d_u) 0.240 at L2 0.207 (no cost); lambda 1.0 -> 0.174
(halved) at L2 0.227. The true Galilean residual stays 0.40-0.42 in every
arm. So the joint objective bakes in whatever the current round holds,
true or not, and the data term cannot object in the Jacobian's flat
directions -- the plain surrogate is already 4x more u d_u-symmetric than
the exact operator (0.30 vs 1.24) before any joint training. One seed, CPU,
tiny model: direction, not magnitudes.

What survives: the equivariance criterion must stay DOWNSTREAM of surrogate
training. Legitimate uses -- held-out trajectories (no gradient into S),
ensemble consistency across independently trained surrogates, and a
pre-discovery GATE: if the surrogate cannot separate non-symmetries from
true ones (Burgers today: u d_u 0.87 vs Galilean 0.55), discovery against it
is not meaningful. Jacobian fidelity itself is to be fixed by tangent
training on RANDOM solver-generated directions, which improves the
operator rather than its symmetry and creates no loop.

Also refuted on the way: the orbit-coverage explanation for Burgers
Galilean. The repo's ICs carry a constant offset on 38 percent of samples
(KdV 88 percent), so the constant direction IS in the data; the surrogate
saw it and did not learn it. Pure Jacobian infidelity at good L2.

**THE ETA CHANNEL COLLAPSES ON BURGERS AND KDV -- diagnosed, measured,
fixed.** The phi-dominance hypothesis for KdV was WRONG and the channel
energy split refuted it immediately: FINS emits xi/eta/phi =
0.98/0.00/0.01 on Burgers (near-pure xi, the opposite of the prediction)
and 0.28/0.00/0.72 on KdV. What both share is eta = 0.00, against ground
truths that need 0.22 and 0.29 -- and without an eta channel d_t is
unreachable, one of the three generators exp25 proved the data holds to
machine precision. On heat, where the v2 fixes worked, eta is 0.48 and
healthy.

Mechanism. The characteristic is Q = phi - xi u_x - eta u_t, and the loss
divides by ||v||, the RAW coefficient norm. On heat u_t ~ alpha u_xx is
small (alpha = 0.01); on Burgers and KdV u_t ~ u u_x, u_xxx is large, so
any sizeable eta inflates the numerator while the denominator does not
follow. The optimiser answers by driving eta to zero -- not symmetry
search, but flight from the largest term. Measured on an exact KdV
operator perturbed by 3e-3 (standing in for surrogate error): d_t scores
6.1e-2 against d_x's 1.05e-5 under the published normalisation, a 5800x
penalty between two generators that are both exact symmetries and differ
only in which channel carries them.

Fix: normalise the residual by ||Q|| instead of ||v||, measuring the
condition dS[u0](Q0) = Q on its own scale. Same test: the d_t/d_x penalty
falls from 5800x to 48x, two orders less channel-biased. Available as
`--fins-channel-norm q` in run_single_pde and campaign; 'none' keeps the
published Rayleigh form. This is not tuning toward the answer -- it
removes a bias whose size is set by the equation's derivative magnitudes
rather than by its symmetries.

Still open: LaLiGAN reads 0.00/0.00/1.00 on all three equations, pure phi,
for which the spurious-linearity account remains live -- it just does not
apply to FINS. `probe_surrogate.py --train-if-missing` answers that
separately.

**v2 campaign verdict (heat-only fixes, 3 seeds, crop 8, full rank).**
Heat: FINS d_G 2.84 -> 2.04 and cos 0.32 -> 0.56 (part of it is the crop
convention, ~0.17 / 0.07 by the gate run; the rest is the per-sample
spacing fix plus the constrained ansatz). FINS is now the best
equation-free method on d_G (LieGAN 2.40, Augerino 2.59, LaLiGAN 2.83),
second overall behind LieNLSD 1.88, sits ON the band-limited null in
cosine (z = +0.08) and still under the shuffled one (z = -2.04).
Burgers and KdV: unchanged (3.39 / 0.26 and 3.75 / 0.03) -- expected,
since both fixes were heat-specific: the spacing bug needs multi-scale
data and lambda_b was heat-only; the raw anchor alone did nothing there.

The KdV number is the informative one: cos 0.017 / 0.060 / 0.031 across
seeds is BELOW the random null (0.57), and xi/eta sample-spread is
1.44-1.55. Below-random is structural, not noisy: the found directions are
orthogonal to the ground truth by construction. The simplest way to be
orthogonal to d_x, d_t and the xi-carried Galilean boost is to live in the
phi channel -- and a phi-only family beta(x,t) d_u is an EXACT symmetry of
any LINEAR operator. Hypothesis: the Burgers/KdV surrogates have not
learned the nonlinearity, so they carry the superposition algebra as a
spurious symmetry set, which the discovery loop finds -- exp2/exp6's heat
mechanism, now as a defect. Predicted Hungarian mean for phi-only output
against the KdV GT is ~0.05, matching the observed 0.03.

Three tools, all validated. `table1.py` now prints the xi/eta/phi energy
split per method (phi-dominance is testable from the artifacts alone).
`pde/probe_surrogate.py` loads a cached S_theta and measures its
nonlinearity ratio ||S(2u)-2S(u)||/||S(2u)|| plus the residuals of u d_u
and random beta d_u -- non-symmetries a linear surrogate would fake;
calibrated on exact KdV (nonlinearity 0.40, u d_u residual 0.43) against
its linear Airy part (0.0, 1.4e-12). And `--fins-b-penalty nonconst`
penalises only the (x,t)-varying part of b, so lambda_b can be applied to
Burgers/KdV without killing their constant Galilean boost. Surrogate
checkpoints now save to FINS_RESULTS_DIR so they survive the session.

**exp25 — the data floor on Burgers and KdV redirects the improvement
pass.** Same ICs at every resolution, exact spectral solvers, oracle
residual of each GT generator, global / windowed at m = 0.1L:

| Burgers, nu = 0.01, T = 1 | Nx 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| d_x, d_t, Galilean | ~1e-17 | ~1e-16 | ~1e-15 | ~1e-15 |
| scaling (global / window) | 6.6e-2 / 3.6e-2 | 3.9e-2 / 1.6e-2 | 1.8e-2 / 1.4e-3 | 1.6e-2 / 7.9e-6 |
| projective (global / window) | 8.9e-2 / 2.0e-2 | 7.5e-2 / 9.0e-3 | 6.2e-2 / 8.3e-4 | 6.0e-2 / 5.2e-6 |
| generators passing 1e-2 | 3 / 3 | 3 / 4 | 3 / 5 | 3 / 5 |

KdV (torus, T = 1): d_x ~1e-19, d_t 1.5e-4, Galilean 3.1e-6 at Nx = 64 --
all pass -- while scaling sits at 95 (Nx 64) and 190 (Nx 128): dispersion
wraps harder at higher resolution, and the window cannot help (exp8,
exp18); scaling is reachable only on a wide domain with a short horizon.

The conclusion inverts the plan. The essential trio {d_x, d_t, Galilean}
IS in the campaign data at Nx = 64 to machine precision on both nonlinear
equations, so FINS's failure there (Burgers cos 0.23, KdV 0.03, Galilean
never a best match) is a failure of the METHOD or the SURROGATE, not of
the data. Resolution buys only the x-weighted pair on Burgers, and only
together with the seam window at Nx >= 256; on KdV it buys nothing.

Consequences for the pass. (1) The b-share penalty must NOT be applied on
Burgers/KdV: their Galilean boost is phi = 1 and phi = 1/6 -- a PURE b
term -- so lambda_b = 30 would suppress one of the three essential
generators, and the superposition subalgebra it targets exists only for
linear PDEs. `campaign.py --fins-lambda-b-heat-only` does exactly that.
(2) The published loss anchors the DEFLATED norm, which pins every slot
near 1 and fills the tail with junk (exp15); `--fins-anchor raw` anchors
the raw field and only floors the deflated norm. Smoke, exact heat
operator, 5 slots: deflated anchor gives norms 1.72 / 1.66 / 1.70 / 0.56 /
0.54, raw gives 1.72 / 1.73 / 1.18 / 0.31 / 0.60 -- later slots are free
to fall. (3) The next decisive number is the SURROGATE floor on Burgers
and KdV: the campaign logs' sanity blocks (GT generators through the
trained S_theta) say whether the essential trio survives the surrogate at
all; if Galilean fails there, no loss change will recover it and the fix
is upstream. (4) A resolved-Burgers run (Nx = 256, window) is the only
route to the x-weighted pair and is a separate, costly experiment.

**THE CAMPAIGN VERDICT, AND WHAT IT DOES AND DOES NOT MEAN.** Full
campaign, 3 PDEs x 3 seeds, k = 6 fixed, both crop conventions, both
ground-truth ranks. Under every convention FINS is below both null floors
on d_G and cosine (z from -2.8 to -13.7), and on the angle part of d_G --
where the configuration penalty is removed -- it is the worst or
second-worst of six on all three equations (heat 2.84 vs LieGAN 2.63 and
LieNLSD 2.08 at full rank). Its xi/eta sample-spread grows 0.59 -> 0.77 ->
1.54 from heat to KdV: the discovered objects increasingly are not point
symmetries of the phi = a u + b class at all. On KdV its cosine is 0.03.

Bias audit of the scoring, prompted by that verdict. Cosine is averaged
over Hungarian-matched pairs, not divided by the emitted count (Augerino's
1.000 on essential heat proves it). The crop is symmetric -- it REMOVES an
asymmetry that favoured FINS. The rank penalty is identical for every
k = 6 method and the angle part is reported beside it. Null floors are
shared. One genuine limitation: the closure metric assumes the affine
class, which FINS's unconstrained network does not enforce, so for FINS
`escaped`/`spread` measure distance from that class rather than closure.
Nothing in the scoring is tilted against FINS.

Two things ARE tilted against it, both in the method as run, both now
fixed. (1) `_compute_Q_field` took scalar dx, dt from sample 0 for the
whole batch; the heat data is multi-scale (L in {0.5..2.0}, T ~ L^2), so
80 percent of samples had u_x inflated up to 4x and u_t up to 16x (relative
Q error 0.75 / 2.0 / 3.7 / 6.0 for L = 0.7 / 1.0 / 1.4 / 2.0). That pushes
the network toward xi, eta -> 0 -- the phi-only collapse of exp6 -- and
explains d_t at cos 0.000 on seed 43. Burgers and KdV are single-scale and
unaffected. Spacings are now per-sample throughout (training loop,
post-slot evaluation, sanity check); verified: batched per-sample Q equals
the sample-by-sample computation to 1e-4. (2) The published method runs
unconstrained (lambda_b = 0), which exp6 showed fills the slots with the
superposition subalgebra; its heat cosine 0.32 sits on the exp2 null
(0.26). The constrained ansatz is now a first-class option:
`run_single_pde.py --fins-lambda-b 30 --fins-tag _v2`, mirrored in
`campaign.py` and scored by `table1.py --fins-tag _v2`, so FINS-v2 lands in
the same table as a separate row. Smoke on multi-scale heat with an exact
operator: unconstrained slots match d_t 0.80 / scaling 0.63 / d_x 0.85,
constrained d_t 0.98 / d_x 0.98 / u d_u 0.86.

Expectation, from exp19: FINS-v2 should clear the nulls on the essential
trio and will most likely still not recover the x-weighted generators,
because that is the generator-parameterisation floor, not a configuration
issue. The framing therefore stays "a calibratable criterion", with
FINS-v2 as the method's fair representative and the published
configuration as the cautionary baseline.

**`pde/table1.py` — the benchmark table, built from artifacts.** The
campaign produces fields, not conclusions; this assembles them, aggregated
over seeds (median with max-min spread), with every correction this
campaign established wired in: d_G SPLIT into total and angle-only part so
the |k_A - k_B| configuration penalty is visible separately, symmetric
explicit `--crop`, selectable `--gt-rank full|essential`, the corrected
closure triple (escaped / structure / spread) in place of ACE, the exp20
null floors printed as rows, and z-scores against both nulls. `--latex`
emits the table body. No discovery is re-run, so switching convention
costs seconds.

One bug it surfaced and fixed on the way: `ESSENTIAL` in
`rescore_from_artifacts.py` listed ASCII names ("d_x") while
`get_gt_names` emits Unicode ("\u2202_x"), so `--gt-rank essential` had
been silently selecting an empty ground truth. Both files now use the
emitted names and raise if the filter fails to match.

Validated on synthetic artifacts in the real schema across all three
equations: the essential-rank path reorders the table sharply (on heat
LIG's total d_G drops from 2.47 to 1.73 while its angle part rises,
because the rank penalty flips sign when k_B falls from 6 to 3), which is
exactly the kind of convention sensitivity the paper has to state rather
than hide.

**Fixed: the generator count is now explicit and verified.**
`run_single_pde.py --n-sym K` sets it, applied AFTER the `--quick`
overrides so the flag always wins, and `--quick` without the flag now warns
that it raised n_sym to 8 and that d_G is therefore not comparable with
default runs. The value used is written into every artifact
(`n_sym`), and `campaign.py --n-sym K` forwards it to every run and marks
any existing artifact whose recorded value differs as STALE rather than
silently aggregating it. Tested: with `--n-sym 6`, a stored run at n_sym=8
reports "STALE: n_sym=8 but the campaign asks for 6; d_G would not be
comparable" while the n_sym=6 run reports DONE.

**THE GENERATOR COUNT IS A FREE KNOB, AND d_G CHARGES FOR IT.** The
count does not depend on the seed -- it is `cfg.n_sym`, a config constant
-- but `--quick` silently raises it from 6 to 8, and each method caps it
differently: LieGAN min(n_sym, 6), LaLiGAN min(n_sym, 8), LIG 4, Augerino
7, FINS/LieNLSD n_sym. The rank-aware distance is
d_G^2 = sum(theta_i^2) + |k_A - k_B| (pi/2)^2, and k_A is the NUMBER OF
GENERATORS EMITTED, not the estimated rank -- so a configuration choice
enters the headline metric directly. Decomposing the two real heat runs
(GT k_B = 6):

| method | full run k | penalty | observed d_G | angle part |
|---|---|---|---|---|
| FINS | 6 | 0.00 | 2.569 | 2.569 |
| LieGAN | 6 | 0.00 | 2.798 | 2.798 |
| LaLiGAN | 6 | 0.00 | 3.251 | 3.251 |
| LIG | 4 | 4.93 | 2.856 | **1.795** |
| LieNLSD | 6 | 0.00 | 2.475 | 2.475 |
| Augerino | 7 | 2.47 | 2.465 | 1.900 |

Stripping the count penalty reorders the table: LIG moves from 5th to
1st and Augerino from 1st to 2nd, while everything with k = 6 is
unchanged. And in the quick run (n_sym = 8) FINS pays a 4.93 penalty for a
flag: its total is 2.984 but its angle part is 1.993, BETTER than
LieGAN's 2.264 -- so `--quick` makes FINS look worse for a reason that has
nothing to do with discovery.

Three consequences. (1) The two surviving heat seeds are NOT comparable to
each other on d_G: the gate run used --quick (k = 8 for FINS) and the
campaign run did not (k = 6). (2) Emitting exactly k_B generators is free
points, and since k_B is the rank we are supposed to be discovering, tuning
n_sym to it is circular. (3) Any d_G quoted in either paper should be
accompanied by the per-method generator count and, better, by the angle
part reported separately -- the penalty is a statement about
configuration, the angles about discovery. `campaign.py --status` now
prints k per method so incomparable runs are visible before they are
aggregated.

**The live run now reports BOTH closure metrics.** Until this point the
corrected metric existed only post-hoc (`rescore --ace`, `exp23_stats`),
while `run_single_pde` still printed and stored the legacy ACE alone --
so anyone reading the run's own table would have quoted the broken number.
`evaluate_method`'s `bracket` column is left untouched for comparability
with the published table but is now labelled `ACE(leg)`, and the summary
line carries three corrected fields computed live from the multi-sample
block: `esc` (fraction of the average bracket leaving the span), `str`
(bracket magnitude -- near zero flags an abelian or duplicated set that
`esc` alone would call perfect) and `spr` (sample-spread of xi and eta,
which must be ~0 for a genuine point symmetry of the phi = a u + b class).
All three also go into the results json. Verified end to end on a
simulated method output that "discovers" the true 4-dim subalgebra:
esc 0.000, str 1.2e-3, spr 0.0.

**Campaign hygiene after losing most of a multi-day run: `pde/campaign.py`
+ multi-sample serialisation + a GT bug that was still live.**

* `run_single_pde.py` never checked what already existed. `campaign.py`
  treats FINS_RESULTS_DIR as the source of truth: a (pde, seed) pair is
  DONE when its npz and json both exist and load, STALE when its artifacts
  predate an invalidating change, PENDING otherwise; `--status` prints the
  table, `--run` executes only what is needed, one pair at a time, each
  pair writing its artifacts immediately so a dead session loses at most
  the pair in flight. Tested on synthetic artifacts of all three kinds.
* The Burgers projective generator was STILL WRONG in `pde/ground_truth.py`
  -- the fix had only ever been applied to exp_common. Now fixed in the
  repo copy, and artifacts written from here on carry a
  `gt_projective_corrected` marker; any Burgers artifact without it is
  flagged STALE. No Burgers run had completed, so nothing was lost.
* Artifacts now also hold fields on several reference samples
  (`FINS_N_REF_SAMPLES`, default 4): every extractor evaluates on sample 0
  of the batch it receives, so slicing the batch re-evaluates the trained
  objects at forward-pass cost, LieNLSD included. This is what the
  corrected closure metric needs; `rescore --ace` reads the `*_ref` block
  and says so plainly when an artifact lacks it.

What the two surviving heat seeds are good for: d_G / cos / rank under
both crop conventions and both GT ranks (their heat GT is correct), but
NOT the corrected closure column, which needs the multi-sample block they
do not have. Whether to re-run them depends on whether the ACE column is
kept in the paper at all.

**A REPLACEMENT CLOSURE METRIC (`pde/closure_metrics.py`), validated.**
Fixing ACE needs three separate repairs, plus one that only showed up in
testing.

The bracket is taken in the (xi, eta, a, b) representation with
phi = a(x,t) u + b(x,t), where d phi / du = a is known rather than
dropped; that class holds every generator of all three benchmark algebras
and is bracket-closed. Generators are normalised before bracketing, which
removes the s^2 scale dependence. And the result is a PAIR, because
closure and non-triviality are orthogonal:

| set | repo ACE | escaped | structure | verdict |
|---|---|---|---|---|
| true closed 6-dim heat algebra | 8.95 | 0.001 | 1.7e-3 | closed |
| the same, scaled by 0.01 | 8.9e-4 | 0.001 | 1.7e-3 | closed |
| essential trio (abelian) | 0.135 | n/a | 0 | DEGENERATE |
| exp2 null: superposition | 0.00 | n/a | 0 | DEGENERATE |
| six copies of one generator | 2e-17 | n/a | 3e-20 | DEGENERATE |
| 5 true + 1 random | 17.0 | 0.477 | 9.7e-2 | partially closed |
| pure noise | 140 | 1.000 | 0.72 | not closed |

The three sets that the published ACE calls perfect -- including the null
model that already beats FINS on d_G and cosine -- are now named as
degenerate instead of scored as ideal.

The fourth repair: on RAW fields the statistic saturates, because the
bracket differentiates and differentiation amplifies field noise by ~1/dx
while a real algebra's brackets are small (structure 1.7e-3). One percent
of field noise already sent `escaped` to 0.80. Projecting the coefficients
onto monomials x^p t^q with p+q <= 2 before bracketing -- exact for every
generator of these algebras, and the same ansatz as exp22 -- restores a
graded response: eps = 0.01 / 0.05 / 0.2 / 0.5 / 1.0 gives escaped 0.235 /
0.300 / 0.397 / 0.516 / 0.541, against 0.86 for pure noise and 0.36 for
five true generators plus one random. Raw mode is kept behind
`--ace-raw` for comparison.

Recovering a and b requires fields on at least TWO samples with different
u; a single sample cannot separate a*u from b, and `fit_affine` refuses
rather than guessing. It also returns the sample-spread of xi and eta,
which must be ~0 for a genuine point symmetry of this class, so a large
value flags a discovered object that is not one. Wired into
`rescore_from_artifacts.py --ace`; the current run_single_pde artifacts
store one reference sample, so extending that serialisation to several
samples is the remaining step before the Table 1 ACE column can be
recomputed.

**The replacement metric on REAL discovered fields (exp23 smoke, wired
into exp23_stats).** Synthetic sets prove correctness; these are the
actual outputs of a discovery loop, 4 slots, 8 samples, single seed.

| set | escaped | structure | abelian pairs | xi/eta sample-spread |
|---|---|---|---|---|
| true 4-dim reference, same grid | 0.000 | 5.2e-4 | 0/6 | 0 |
| discovered, seam-aware cell | 0.777 | 1.4e-3 | 0/6 | 0.62 |
| discovered, spectral cell | 0.818 | 1.5e-3 | 0/6 | 0.58 |

Three things this tells us that the published ACE could not. The
discovered sets are NOT degenerate -- every pair has a live bracket, so
the "perfect closure" trap does not apply here -- but they are genuinely
far from closed, roughly 80 percent of each bracket leaving the span, at a
budget where the same loop already recovers d_x, d_t and u d_u at cos
0.85-0.90. Closure is therefore a much harder target than per-generator
cosine, which is consistent with the deflation analysis: slots are
individually decent and jointly not an algebra.

The xi/eta sample-spread earns its place immediately. It is 0 for a
genuine point symmetry of this class, and reads 0.03 / 0.18 / 1.06 / 1.20
across the four slots: the first two behave like point symmetries, the
last two do not -- their xi and eta change with the sample, so they are
not (xi(x,t), eta(x,t), a u + b) objects at all. That is a per-slot
diagnostic no metric in either paper provides.

Monotonicity check, replacing discovered generators by true ones one at a
time: escaped 0.777 / 0.660 / 0.380 / 0.635 / 0.000 for 0 / 1 / 2 / 3 / 4
true. The trend is right and the endpoints are exact, but the value is not
monotone in between -- a mixed set can close worse than a more-wrong one,
because closure is a property of the SET, not a sum over members. So
`escaped` ranks a set, it does not measure "how many generators are
correct", and it should be reported next to the per-generator cosines
rather than instead of them.

**ACE IS NOT SCALE-INVARIANT — the Table 1 ACE column is unusable as
published.** Found while reading a real heat seed-43 run. The bracket is
bilinear and `bracket_closure_error` never normalises, so scaling a
generator set by s scales ACE by s^2 exactly (verified: scale 1 / 0.3 /
0.1 / 0.03 / 0.01 gives ACE 8.24 / 0.742 / 0.0824 / 0.00742 / 8.24e-4).
Two consequences for the published table:

* A method whose generators are near-zero scores a near-perfect ACE for
  free. Augerino's norms in that run are 0.003-0.038, its ACE is 1.14e-3
  -- and normalising by the mean norm gives ACE/|v|^2 = 6.31, the WORST of
  the six methods rather than the best. The paper's reading of Augerino as
  structurally closed inverts under normalisation.
* The correct reference is not 0. Under this convention the EXACTLY closed
  6-dim heat algebra scores 8.24, so FINS at 5.47 and LIG at 5.74 are
  BELOW the closed-algebra reference and cannot be called "catastrophic
  closure failure"; LIG's 17.68 and 63.04 on Burgers and KdV likewise need
  re-reading against their own reference and norms (LIG's first generator
  in this run has ||v|| = 36.4, which alone inflates ACE by ~10^3).

Normalised ACE for that run: FINS 2.66, LieGAN 1.25, LieNLSD 1.19,
Augerino 6.31, LIG 0.05, LaLiGAN 0.00 (degenerate by construction). The
ordering bears no resemblance to the raw column. Any ACE reported in the
revision must be normalised by the generator-set norm AND quoted against
the closed-algebra reference under whichever bracket convention is used.

**Heat seed 43, full run, against the exp20 null floors.** z-scores
(positive = better than the null):

| method | d_G | z vs band | z vs shuf | cos | z vs band | z vs shuf |
|---|---|---|---|---|---|---|
| FINS | 2.569 | -2.09 | -2.05 | 0.505 | -1.92 | -2.60 |
| LieGAN | 2.798 | -2.83 | -2.67 | 0.236 | -11.88 | -5.40 |
| LaLiGAN | 3.251 | -4.29 | -3.89 | 0.310 | -9.16 | -4.64 |
| LIG | 2.856 | -3.02 | -2.83 | 0.528 | -1.06 | -2.36 |
| LieNLSD | 2.475 | -1.79 | -1.80 | 0.728 | +6.35 | -0.28 |
| Augerino | 2.465 | -1.76 | -1.77 | 0.609 | +1.91 | -1.53 |

Every method is worse than BOTH nulls on d_G, and only the two
equation-aware/parameterised methods clear the band-limited floor on
cosine; nothing clears the shuffled floor. This is the in-vivo
confirmation of the exp20 prediction made from the published numbers.

Two further observations from the same run. FINS failed to recover d_t
entirely (cos = 0.000 at ||v|| = 1.74 -- the Hungarian matcher assigned a
leftover slot), while LieNLSD and Augerino both hit it at 1.000; and the
norm-gap rank estimate returned 5 here against 8 on the seed-42 quick run,
so the criterion is unstable across seeds as well as uninformative
(exp15). FINS cost 2.9 GPU-hours against LieNLSD's 65 seconds.

**exp24 + exp23_stats — the GPU protocol comparison, prepared and smoked
(pending real runs).** Everything above about the seam mechanism was
measured with an EXACT operator, so surrogate fidelity -- which is what
LieSym's claim is actually about -- was never tested. exp24 trains
N_theta[u] ~ u_t and runs discovery against it through a JVP, over five
cells: base (single scale, no coords, spectral), coord (coords,
spectral), seam (coords, seam-aware local), ms_coord (multi-scale +
coords, spectral -- LieSym's own configuration) and ms_seam (both). The
decisive comparison is seam vs ms_coord: if single-scale seam-aware
matches or beats multi-scale at its own headline target, the multi-scale
construction is redundant there, and its cost -- twice the GPU budget, the
3500-epoch surrogate, the direction-collapse instability of their section
5.3 -- is avoidable.

exp23_stats aggregates the serialised fields into reportable numbers, with
three choices forced by mistakes made earlier in this campaign: WITHIN-SEED
PAIRED CONTRASTS with a sign test as the headline (absolute levels move
with slot count and evaluation set, differences at identical settings do
not); bootstrap CIs over seeds reported with n, so the width is visible;
and BOTH metrics side by side -- per-slot best-match cosine, which LieSym
and FINS report, and novel-component containment, which survives the
correlation inflation of exp19. They disagree, so both belong in the table.

CPU smoke (6 samples, 250 surrogate epochs, 2 slots, 40 iterations):
surrogate reaches delta = 0.0095 single-scale and 0.0341 multi-scale,
slots match d_t at cos 0.90-0.99, serialisation and the aggregator both
work end to end. The smoke numbers themselves mean nothing at that budget
-- notably the 2-slot spans invert the expected ordering -- which is
exactly why the real runs need the settings in the Colab cells and at
least three seeds per cell.

**exp22 — EXACT discoverability at single scale: LieSym section 5.2 is
refuted in its strong form.** The generator is parameterised linearly
(monomials x^p t^q, p+q <= 2, times {1, u}), which contains the whole
6-dim heat algebra exactly. Q and the Olver operator are then LINEAR in
the coefficients, so the discoverable set is the null space of the design
matrix -- read off by SVD, with no optimiser, no architecture and no
training noise to confound it. Coefficients are Gram-whitened so that
"unit norm" means the field L2 norm. The DATA is single-scale throughout;
only the evaluation protocol varies. Constrained ansatz (phi = a(x,t) u,
i.e. the b-term of the superposition subalgebra dropped as in exp6):

| protocol | admissible dim | containment of d_x / d_t / u d_u / Gal / scal / proj |
|---|---|---|
| spectral (periodic, global) | 4 | 1.00 / 1.00 / 1.00 / 0.89 / 0.86 / 0.76 |
| local FD, m = 0 | 5 | 0.84 / 1.00 / 0.95 / 0.88 / 0.74 / 0.73 |
| local FD, m = 0.05L | 9 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| local FD, m = 0.10L | 9 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |

With a seam margin, ALL SIX generators -- scaling included -- lie exactly
in the admissible set on SINGLE-SCALE data. The claim that single-L data
is "physically degenerate" and that "no operator-learning method can
recover the scaling from u-fields alone regardless of architecture" is
therefore false as stated: the degeneracy belongs to the periodic-global
evaluation, not to the data. Multi-scale + coord-aware construction is one
remedy among several, and LieSym's own Figure 4 decomposition supports the
reading -- multi-scale alone contributes +0.04, the coord channel +0.23,
longer surrogate training +0.14.

Two caveats, both in the script's output. (1) The admissible
dimension EXCEEDS 6 under the local protocols (9 constrained, 12 with free
b): the extra directions are the superposition subalgebra of a linear PDE
(exp2) plus ansatz directions the discretisation cannot separate at
threshold 1e-2 -- so the protocol admits the true algebra but does not by
itself pin the rank, which is the same rank-detection gap documented in
exp15/exp16. (2) Containment is measured against a null space of dimension
9-12 in 18-24, where the random-span reference is 0.47-0.71; the 1.000
values are well above it and exact, but the bare number should always be
quoted with that reference.

**exp21 — the LOCAL (Olver) formulation on single-scale torus data (numpy,
seconds). Refutes LieSym §5.2.** Olver residual ||D_t Q - alpha D_xx Q|| /
||Q|| on exact heat data, one scale L = 2pi, Nx = 64, alpha = 0.01, exact
operator. Spectral (periodic, global) evaluation: trio ~1e-7, Galilean
0.81, scaling 0.22, projective 1.08. Local non-periodic FD with a 5 percent
seam margin: ALL SIX correct generators pass 1e-2 (1.2e-3 .. 5.5e-3,
scaling 2.9e-3) — on single-scale data. The "physical degeneracy" is a
property of the periodic-global evaluation, not of the data, so
multi-scale + coord-aware construction is one fix among others. LieSym's
mis-coded heat Galilean (2t, 0, +xu) and alpha-less projective fail under
every protocol at O(1): the residual test doubles as a ground-truth
linter. Structural note: the local formulation's seam contamination is
confined to the stencil width, whereas the global operator's spreads
through the parabolic cone (exp9) — an intrinsic advantage of the Olver
formulation neither paper states.

**LieSym audit (code: LieSym_NeurIPS_Experiments.py).**
Symbolically (exp17 machinery): heat Galilean coded as (2t, 0, +x u) is
NOT a symmetry for any alpha (defect -2 u_x (alpha+1)); heat projective
coded without alpha is a symmetry only at alpha = 1 while the config has
0.01; Burgers projective is the same wrong generator as in FINS. Same
bracket convention ("ignoring d/du for simplicity"), same spectral_rank
(returns K = 6 when no norm gap exceeds 3.0 — LieSym reports r-hat = 6 on
all three PDEs, GT 6/5/4), same grid and metrics, so exp20 null floors
apply verbatim (no equation-free method in their Table 1 beats the
shuffled floor on cos or either null on d_G). LieSym persists only a
metrics json, not the discovered fields. Full write-up:
LieSym_audit.md.

**BLOCKER FOUND IN THE PUBLIC REPO: `run_single_pde.py` crashes on the
FINS evaluation, for every configuration.** `evaluate_method` cropped the
GROUND TRUTH by M = 8 for FINS only, assuming the FINS generators arrive
pre-cropped. They do not: neither `extract_fins_generators` nor
`extract_deflation_generators` crops -- both return full-grid fields. So
disc has Nx*Nt*3 rows while gt has (Nx-16)*(Nt-16)*3, and
`grassmann_distance` raises in `Ao.T @ Bo`. Observed on --quick
(6144 vs 2304); the same mismatch holds for the paper grid
(24576 vs 16128). This is deterministic and config-independent, so the
published Table 1 FINS column cannot have come from the repository at
HEAD -- to be resolved with the first author before the revision claims
reproducibility.

Fix applied: the crop is now applied to BOTH sides, with
`FINS_EVAL_CROP` selecting the regime -- unset keeps the original intent
(M = 8, FINS only, shapes now consistent), an integer applies the SAME
crop to every method, which is the symmetric evaluation the revision
wants. Verified on 64x32 and 64x128: all three regimes run, and with
FINS_EVAL_CROP=0 every method scores identically on identical input
(4.3826 / 4.4060), confirming the asymmetry is gone. Serialised artifacts
stay UNCROPPED, so `rescore_from_artifacts.py --crop` can explore either
convention after the fact.

**The ACE column of Table 1 is also defective (found while designing the
lab-admission test).** `Metrics._lie_bracket` computes the directional
derivative as v(f) = xi df/dx + eta df/dt — the phi d/du term is absent,
and the derivatives act on ON-SHELL fields, so for phi = u the x-derivative
returns u_x where the jet convention demands 0. On the EXACTLY closed
abelian trio {d_x, d_t, u d_u} the repo ACE returns 0.271 instead of 0,
and 6.92 on the full closed 6-dim heat algebra (partly also from the
periodic x-wrap of the central difference applied to non-periodic
x-weighted coefficients). A correct bracket is well-posed for the
constrained class phi = a(x,t) u + b(x,t) (the a1 a2 terms cancel, the
class is bracket-closed): reference implementation gives 0 exactly on the
trio, 3.6e-3 (FD floor, one-sided edges) on the full algebra, 5.7 on a
deliberately broken set, and recovers [d_t, Galilean] = 2 alpha d_x to
1e-5. Consequence for the paper: the published ACE numbers (e.g. FINS
heat 1.23) mix closure with on-shell-derivative artefacts of order 0.3+
and need recomputation under the corrected convention; the rescore tool
should gain an --ace flag once the corrected module exists (this is the
lab-admission test task, with the reference implementation held back as
the grader).

**exp20 — null floors for all three equations (CPU, 2 min): mandatory
item closed, and the numbers are harsh.** Two nulls per equation, both
spaces, 21 trials pooled over 3 reference samples:

| pde | null | d_G (triple) | cos (triple) | cos (Q) |
|---|---|---|---|---|
| heat | band-limited | 1.92 +/- 0.31 | 0.557 +/- 0.027 | 0.515 |
| heat | shuffled-GT | 1.81 +/- 0.37 | 0.755 +/- 0.096 | 0.767 |
| burgers | band-limited | 2.09 +/- 0.15 | 0.548 +/- 0.044 | 0.602 |
| burgers | shuffled-GT | 1.70 +/- 0.23 | 0.719 +/- 0.069 | 0.764 |
| kdv | band-limited | 2.01 +/- 0.13 | 0.566 +/- 0.058 | 0.654 |
| kdv | shuffled-GT | 1.57 +/- 0.29 | 0.687 +/- 0.109 | 0.789 |

These are strong (statistics-informed) nulls: band-limited matches the
GT's per-component amplitude spectra and energies; shuffled-GT permutes
the GT's own components and so inherits its correlation structure — the
inflation mechanism quantified in exp19. Read against the published
Table 1: the shuffled floor's cos (0.69-0.77) EXCEEDS every published
cosine (0.28-0.50), and both nulls' d_G (1.6-2.1) beat every published
d_G (2.17-3.39). FINS heat (d_G 2.17, cos 0.50) sits below even the
band-limited floor (z ~ -2 on cos). The published metrics therefore do
not demonstrate discovery beyond matched statistics — which is precisely
why the null row must be in the table and why the revision's framing is
"a calibratable criterion", not "a winning method". The three nulls now
in hand (superposition/exp2 for heat: exact spurious symmetries;
band-limited: spectral statistics; shuffled: statistics + correlations)
answer three different questions and should all be reported.
Per-generator inflation of the band-limited null (which Table 2 columns a
null can fake): heat u d_u 0.70 / Galilean 0.67 / projective 0.63 vs d_x
0.42 / d_t 0.46 / scaling 0.56.

**run_single_pde serialisation + rescore_from_artifacts.py (parallel
track for the re-scoring item).** run_single_pde now saves every method's
generator fields plus the GT to `generators_{pde}_seed{N}.npz`
(FINS_RESULTS_DIR-aware), and `pde/rescore_from_artifacts.py` recomputes
d_G/cos/rank from the artifacts against a switchable ground truth
(--gt-rank full|essential), with an explicit symmetric --crop replacing
the original FINS-only M = 8 crop, and a --verify mode that PASS/FAILs
the artifact metrics against the live run's json at 1e-6. Validated
synthetically end-to-end (12/12 PASS at rel 0.0); the GPU --quick
end-to-end check remains for the first real run. The bachelor task's
items 1-2 accordingly shift from "implement" to "verify and execute".

**exp19 — FINAL: novel-component containment across all five runs.**
(containment of the component orthogonal to the essential trio; random
reference ~0.016)

| run | Galilean | scaling | projective |
|---|---|---|---|
| oracle, no window, s42 | 0.172 | 0.082 | 0.195 |
| oracle, window, s42 | 0.318 | 0.389 | 0.336 |
| oracle, no window, s43 | 0.514 | 0.084 | 0.545 |
| oracle, window, s43 | 0.481 | 0.276 | 0.495 |
| surrogate, window, s42 | 0.346 | 0.144 | 0.369 |

One consistent effect and one null result. SCALING: without the window its
novel component is pinned at a floor (0.082 / 0.084, identical across
seeds); the window lifts it 3-5x on BOTH seeds (0.389 / 0.276). This is
channel-specific and mechanistically clean: heat scaling is (x, 2t, 0) —
its novelty lives purely in the COORDINATE channels, whose characteristic
is seam-discontinuous, so only the window makes it admissible. GALILEAN
and PROJECTIVE: their novelty lives in the amplitude channel (phi ~
(x - c) u, (x^2 - ...) u), which the constrained ansatz phi = a(x,t) u can
overlap through junk with x-dependent a regardless of the window — and
indeed their numbers track the slots' junk fractions (s43 no-window: junk
0.83-0.93, "novel containment" 0.51-0.55) and vary wildly across seeds.
No window effect is resolvable there, and the higher values should not be
sold as recovery: they are substantially junk-mediated.

The surrogate run matches the oracle for the amplitude-channel numbers
(0.346 / 0.369 vs 0.318 / 0.336) but suppresses scaling's coordinate-
channel novelty (0.389 -> 0.144) — the same scaling-through-surrogate
anomaly seen in exp5 (plateau 10x below oracle) and attributed to the
smoothing Jacobian damping seam-propagated content.

Closing verdict for the discovery line: the loop robustly recovers the
essential subalgebra's span (0.997-1.000 everywhere); the window
demonstrably unlocks coordinate-channel novelty (scaling, 2/2 seeds) but
absolute novel capture stays weak-to-moderate (<= 0.55) and partially
junk-mediated; full-rank recovery is not achieved, with the generator
floor as the measured blocker. Detection under the window remains solid
and is a separate, stronger claim.

**exp19 — the 2x2 (window x seed) oracle grid + corrected subspace
analysis: two downgrades and one solid core.**

(1) The best-match signal "Galilean appears only under the window" did NOT
reproduce at seed 43: the windowed run recovered only d_t at cos >= 0.7
(d_x at 0.649/0.693 just under), the unwindowed run {d_t, d_x}; no
x-weighted generator on either side. The signal is seed-dependent — one of
two seeds — and must be reported as such, not as an effect.

(2) Raw span containments are CORRELATION-INFLATED. On x in [0, L) the
fields u, x u, x^2 u are strongly mutually correlated, so a span holding
the essential trio contains the x-weighted generators nearly for free:
computed baselines ||P_ess g|| are Galilean 0.971 / scaling 0.863 /
projective 0.905 at seed 42 (0.905 / 0.863 / 0.730 at seed 43) — against
observed raw containments 0.83-0.99. Back-of-envelope novel-component
containments land at 0.14-0.69, far from full recovery, and show no clear
window effect on Galilean/projective (possibly a positive one on scaling,
one comparison). The analyzer now computes the exact statistic:
containment of the component orthogonal to the essential trio, with a
random-span reference (~0.04); re-run `exp19_analyze.py` over the four
oracle npz (and the legacy surrogate npz with --n-train 1024) for the
definitive table.

(3) What survives across all four runs untouched: the essential trio is
contained at 0.997-1.000 in every span, effective dimensions 3-5, and the
generator floor remains the blocker for resolving individual x-weighted
directions. The defensible discovery claim is therefore: the loop robustly
recovers the essential subalgebra's span plus PARTIAL novel content of the
x-weighted generators; full-rank recovery is not achieved, and a window
benefit for discovery (as opposed to detection, where it is solid) is not
demonstrated at n = 2 seeds.

**ERRATUM + fix, exp19_analyze.** The analyzer regenerated the reference
sample with n = 1 while the runs used n_train >= 48, and `make_dataset`
splits n into n//2 gauss + rest sine ICs — so at n = 1 sample 0 is a SINE
field, at n >= 2 a GAUSS field. The u-DEPENDENT ground-truth triples
(u d_u, Galilean, projective) were therefore built on the wrong field, and
their containment numbers in any earlier analyzer output are invalid; the
u-independent rows (d_x, d_t, scaling — heat scaling is (x, 2t, 0)), the
Gram/duplication block and the singular spectrum are unaffected. Fixed two
ways: exp19 now embeds the reference sample into the npz (self-contained
analysis), and for legacy npz the analyzer takes `--n-train` and refuses
to guess. Corrected smoke containments: u d_u 0.487 -> 0.971, Galilean
0.998, projective 0.979 — on the smoke, the discovered span contains
substantial parts of ALL six generators. The full surrogate run's npz is
legacy: re-run `exp19_analyze.py --n-train 1024`; from its printed output,
the valid parts already are effective dimension 3 (PR 2.98, duplication up
to 0.97 between slots) and containments d_x 0.850, d_t 0.997, scaling
0.750.

**exp19 — full runs (T4; oracle 43 min each, surrogate ~3 h): the
loop recovers THREE directions, and the window's fingerprint is real but
partial.** Oracle, m = 0.1: recovered at cos >= 0.7 {d_t 0.872, u d_u
0.756, Galilean 0.858}, d_G 2.167 (best in the project; numerically equal
to the published 2.17), Hungarian 0.732. Oracle, m = 0 (ablation):
{d_t, u d_u, d_x} — NO x-weighted generator. Surrogate 1024:600, m = 0.1:
{d_t 0.873, u d_u 0.94-0.95, Galilean 0.789} — reproduces the oracle run
nearly verbatim, confirming in vivo that surrogate and generator floors
co-limit (exp16). The discriminating signal: Galilean, an x-weighted
seam-broken generator, appears ONLY under the window, in both oracle and
surrogate runs; one seed, so a consistent signal rather than proof.

Full rank 6 is NOT recovered. The blocker is per-slot field accuracy
(cos 0.7-0.95 at the eps_fit ~ 1 percent parameterisation floor):
deflation subtracts the noisy FIELD, not the true direction, so the
leftover of the same direction wins later slots (u d_u three-four times in
a row). Best-match tables also UNDERSTATE the span: `exp19_analyze.py`
computes the effective dimension and the containment of each GT generator
in the discovered span; already on the smoke npz, scaling is contained at
0.97 while never being any slot's best match. The three full-run npz
should go through it (the two oracle runs wrote to the LOCAL results dir —
to be rescued to Drive if the session still lives).

## Smoke results (mechanics validated; numbers need GPU)

**exp5.** jvp through `SimpleFNO2d` works on CPU; per-sample vs
broadcast sanity differs by up to 2.4x on a weak surrogate. The smoke also
exposed the second component of the broadcast bias: for `scaling`
(u-independent triple) the ratio is 0.39 because on the multi-scale heat
data the broadcast reuses sample 0's (x, t) grid (L = 0.5) for samples
with L up to 2.0. Full run: `python exp5_delta_sweep.py --budgets 64:100
256:200 1024:500` (add more budgets for the slope fit; expected slope 2 of
residual vs test error for exact symmetries).

**exp6.** Constrained deflation runs end to end. Even at smoke scale the
b-share of phi behaves as designed: lambda_b = 0 gives shares 0.13 -> 0.30
growing across slots (late slots leak into the trivial subalgebra);
lambda_b = 30 collapses shares to <= 0.006 with deflation intact. The
rank-6 -> rank-3 prediction on heat requires the full budget.

**exp7.** Receptive-field caveat found and documented: `SpectralConv2d`
initialises spectral weights with scale = 1/(in_ch*out_ch), so at width 16
the spectral path is ~an order weaker than the local 1x1 path and the
RANDOM-INIT map is spuriously delta-like. After a short equivariance fit
the xi channel already spreads over the whole domain at the 1e-2 level
(participation ratio 0.031 -> 0.064); the pointwise-MLP reference stays
delta by construction. For the paper-scale claim, pass a trained
checkpoint: `python exp7_receptive_field.py --checkpoint sym_gen.pt`.

## The patch

`patch_sanity_per_sample.diff` (apply from repo root: `patch -p1 <
experiments/patch_sanity_per_sample.diff`, or `git apply`):

1. `method_fins.sanity_check_equivariance` built all ground-truth fields
   from sample 0 and broadcast them across the batch. For generators whose
   phi depends on u — heat v3/v4/v6, Burgers v4/v5, KdV scaling — the
   characteristic was evaluated on the wrong trajectory; on multi-scale
   data the (x, t) grids were wrong as well. At zero model error the bias
   is ~two orders of magnitude optimistic (heat Galilean 3.5e-5 vs true
   3.6e-3; projective 7.3e-5 vs 1.4e-2). The 1e-2 discoverability
   threshold and Table 3 are calibrated on the biased numbers and need
   regeneration after the patch.
2. `ground_truth.get_gt_generators` silently ignored alpha (default 0.01);
   it now accepts and forwards it.

Verified: patch applies cleanly to the current repository head, the
patched file compiles, and the patched sanity is bit-identical to an
independent per-sample implementation on a trained surrogate.

## Repository observations (no patch)

- `train_fins` and `train_deflation_residual` take scalar dx, dt from
  sample 0; on the multi-scale heat dataset (L in {0.5..2.0}) the Q-field
  derivatives inside the discovery loss are mis-scaled by up to 4x for 80
  percent of samples. Evaluation happens on sample 0's grid, so this adds
  noise to FINS's own training rather than inflating its metrics — but it
  should be fixed for the journal runs (per-sample dx/dt, or evaluate the
  loss per L-bucket).
- Paper text vs code: Section 5.1 states L = 2 pi for all PDEs, while the
  heat data uses L in {0.5, 0.7, 1.0, 1.4, 2.0} with T proportional to
  L^2; Section 5.1's v4/v6 formulas omit alpha while `ground_truth.py`
  includes it (the code is correct).
- Hygiene before journal submission: remove `.idea/`, remove commented-out
  code blocks (in particular the alternative `grassmann_distance`
  implementation next to the active one), replace the review-only license
  in README with a real one, replace the anonymised 4open.science link
  with a permanent citable one.
