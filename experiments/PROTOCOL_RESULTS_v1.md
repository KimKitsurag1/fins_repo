# Protocol v1 results

Results of the protocol runs as recorded at the time, with the verdicts of their registered expectations. Terms and
section numbers follow the log (README_EXPERIMENTS.md, glossary at its top); sections on hidden variables (exp68-exp70)
belong to an earlier draft and are not in the submitted paper.

exp42 output (Colab, 2026-09-29), s = 42, 43, 44, ten unseen evaluation trajectories (data seed
1000 + s), no filter; all baselines trained on the same 1024 trajectories of seed s; LIG capped at
10 epochs (pre-registered) and 4 generators (method_lig, found after the run). Printed by the
version before the post-run additions (seeds, floor and k-range columns, Augerino widths and the
post hoc "grown widths only" row); rerun exp42 for those. Verdicts and notes: README_EXPERIMENTS.md,
"Protocol run, part 3".

## Verdicts

- P1 (FINS criterion): Burgers 2/3, heat 3/3, KdV 3/3 with margin >= 100x -- MET.
- P2 (fewer missing dimensions than LieGAN, LaLiGAN, Augerino, finite ground truth, mean over seeds):
  Burgers 0.44 vs 2.98 / 4.63 / 1.96; KdV 0.01 vs 2.84 / 3.05 / 0.90; heat 0.14 vs 4.06 / 4.10 / 2.00
  -- MET.
- P3 (equation-aware LieNLSD, LIG): no prediction. FINS misses fewer dimensions than both on Burgers,
  KdV and the finite heat algebra; level with LieNLSD on heat's full class (1.59 +/- 0.63 vs
  1.40 +/- 0.47).
- P4 (exact operator: missing, spurious <= 0.05, novel >= 0.95): KdV 0.01 / 0.01 / >= 0.99; heat,
  full class, 0.00 / 0.00 / 1.00 -- MET.

## burgers: 3 seeds, ground truth finite 5; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (5)

| method | k (rank) | missing dims | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 5 (5) | 0.44 +/- 0.35 | 0.44 +/- 0.35 | 0.69 + 0.00 | 4.3 / 5 | 0.65 | 0.82 | 0.96 | 0.99 | 0.98 | 0.99 |
| LieNLSD (uses the equation) | 4 (4) | 0.85 +/- 0.22 | 0.18 +/- 0.26 | 0.31 + 1.64 | 2.0 / 5 | 0.91 | 0.90 | 0.79 | 0.94 | 0.43 | 0.91 |
| LieGAN | 5 (5) | 2.98 +/- 0.06 | 2.98 +/- 0.06 | 2.31 + 0.00 | 0.3 / 5 | 0.34 | 0.36 | 0.63 | 0.57 | 0.73 | 0.69 |
| LaLiGAN | 5 (5) | 4.63 +/- 0.14 | 4.63 +/- 0.14 | 3.13 + 0.00 | 0.0 / 5 | 0.10 | 0.23 | 0.32 | 0.30 | 0.45 | 0.30 |
| Augerino | 7 (7) | 1.96 +/- 0.01 | 3.96 +/- 0.01 | 2.03 + 4.93 | 1.7 / 5 | 0.57 | 0.92 | 0.98 | 0.15 | 0.71 | 0.12 |
| LIG, 10 epochs (uses the equation) | 4 (4) | 2.11 +/- 0.03 | 1.11 +/- 0.03 | 1.54 + 2.47 | 1.3 / 5 | 0.54 | 0.86 | 0.97 | 0.08 | 0.69 | 0.06 |

FINS: all below lambda_(k+1) on 2/3 seeds; worst margin 1.3x, 2.8x, 0.8x; lambda_(k+1) / lambda_k 1.4, 2.6, 1.6; rank from the largest gap 9, 5, 3; relative change 1.10, 1.21, 1.04; band check u_t share 2.5e-02, 5.7e-03, 8.9e-03

## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | k (rank) | missing dims | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 4 (4) | 0.01 +/- 0.00 | 0.01 +/- 0.00 | 0.11 + 0.00 | 4.0 / 4 | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 |
| FINS, exact operator | 4 (4) | 0.01 +/- 0.00 | 0.01 +/- 0.00 | 0.10 + 0.00 | 4.0 / 4 | 0.83 | 1.00 | 1.00 | 1.00 | 0.99 |
| LieNLSD (uses the equation) | 3 (3) | 0.67 +/- 0.47 | 0.00 +/- 0.00 | 0.01 + 1.64 | 2.0 / 4 | 0.91 | 0.97 | 0.87 | 0.72 | 0.33 |
| LieGAN | 4 (4) | 2.84 +/- 0.25 | 2.84 +/- 0.25 | 2.37 + 0.00 | 0.0 / 4 | 0.40 | 0.25 | 0.41 | 0.36 | 0.67 |
| LaLiGAN | 4 (4) | 3.05 +/- 0.17 | 3.05 +/- 0.17 | 2.53 + 0.00 | 0.0 / 4 | 0.22 | 0.19 | 0.37 | 0.59 | 0.62 |
| Augerino | 7 (7) | 0.90 +/- 0.00 | 3.90 +/- 0.00 | 1.24 + 7.40 | 1.0 / 4 | 0.76 | 0.96 | 0.89 | 0.32 | 0.64 |
| LIG, 10 epochs (uses the equation) | 4 (4) | 2.90 +/- 0.02 | 2.90 +/- 0.02 | 2.52 + 0.00 | 0.0 / 4 | 0.35 | 0.12 | 0.83 | 0.24 | 0.31 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 904.2x, 355.8x, 683.1x; lambda_(k+1) / lambda_k 1089.0, 223.4, 567.2; rank from the largest gap 4, 4, 4; relative change 0.39, 0.46, 0.42; band check u_t share 1.6e-16, 2.1e-15, 4.1e-16
FINS, exact operator: all below lambda_(k+1) on 3/3 seeds; worst margin 838.5x, 568.5x, 519.1x; lambda_(k+1) / lambda_k 1183.8, 650.4, 632.5; rank from the largest gap 2, 2, 2; relative change 0.39, 0.46, 0.42; band check u_t share 1.6e-16, 2.1e-15, 4.1e-16

## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | k (rank) | missing dims | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 10 (10) | 0.14 +/- 0.05 | 4.14 +/- 0.05 | 0.38 + 9.87 | 6.0 / 6 | 0.65 | 0.98 | 0.98 | 0.97 | 0.97 | 0.98 | 0.97 |
| FINS, exact operator | 10 (10) | 0.00 +/- 0.00 | 4.00 +/- 0.00 | 0.00 + 9.87 | 6.0 / 6 | 0.57 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LieNLSD (uses the equation) | 10 (10) | 1.12 +/- 0.30 | 5.12 +/- 0.30 | 1.31 + 9.87 | 2.0 / 6 | 0.79 | 0.79 | 0.61 | 0.88 | 0.86 | 0.46 | 0.86 |
| LieGAN | 6 (6) | 4.06 +/- 0.16 | 4.06 +/- 0.16 | 2.83 + 0.00 | 0.0 / 6 | 0.41 | 0.35 | 0.55 | 0.34 | 0.25 | 0.46 | 0.21 |
| LaLiGAN | 8 (8) | 4.10 +/- 0.33 | 6.10 +/- 0.33 | 2.96 + 4.93 | 0.0 / 6 | 0.33 | 0.16 | 0.07 | 0.68 | 0.69 | 0.21 | 0.70 |
| Augerino | 7 (7) | 2.00 +/- 0.00 | 3.00 +/- 0.00 | 2.20 + 2.47 | 3.0 / 6 | 0.57 | 0.97 | 0.99 | 0.10 | 0.01 | 0.95 | 0.01 |
| LIG, 10 epochs (uses the equation) | 4 (4) | 3.45 +/- 0.17 | 1.45 +/- 0.17 | 1.57 + 4.93 | 0.0 / 6 | 0.62 | 0.73 | 0.51 | 0.15 | 0.11 | 0.18 | 0.09 |

### ground truth: full (10)

| method | k (rank) | missing dims | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 10 (10) | 1.59 +/- 0.63 | 1.59 +/- 0.63 | 1.67 + 0.00 | 4.0 / 10 | 0.69 | 0.98 | 0.98 | 0.77 | 0.79 | 0.96 | 0.82 | 0.41 | 0.37 | 0.34 | 0.37 |
| FINS, exact operator | 10 (10) | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 + 0.00 | 10.0 / 10 | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LieNLSD (uses the equation) | 10 (10) | 1.40 +/- 0.47 | 1.40 +/- 0.47 | 1.60 + 0.00 | 5.3 / 10 | 0.79 | 0.77 | 0.60 | 0.73 | 0.66 | 0.41 | 0.61 | 0.99 | 1.00 | 1.00 | 0.99 |
| LieGAN | 6 (6) | 6.66 +/- 0.04 | 2.66 +/- 0.04 | 2.24 + 9.87 | 0.0 / 10 | 0.62 | 0.34 | 0.54 | 0.14 | 0.05 | 0.44 | 0.04 | 0.15 | 0.06 | 0.03 | 0.01 |
| LaLiGAN | 8 (8) | 7.39 +/- 0.13 | 5.39 +/- 0.13 | 3.14 + 4.93 | 0.0 / 10 | 0.36 | 0.17 | 0.08 | 0.55 | 0.56 | 0.22 | 0.57 | 0.21 | 0.24 | 0.26 | 0.26 |
| Augerino | 7 (7) | 6.00 +/- 0.00 | 3.00 +/- 0.00 | 2.70 + 7.40 | 2.7 / 10 | 0.49 | 0.95 | 0.99 | 0.08 | 0.02 | 0.92 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| LIG, 10 epochs (uses the equation) | 4 (4) | 6.89 +/- 0.10 | 0.89 +/- 0.10 | 1.28 + 14.80 | 0.0 / 10 | 0.72 | 0.71 | 0.50 | 0.03 | 0.01 | 0.16 | 0.01 | 0.06 | 0.01 | 0.00 | 0.00 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 1.4x, 2.9x, 1.9x; lambda_(k+1) / lambda_k 1.2, 1.5, 1.1; rank from the largest gap 6, 11, 13; relative change 0.12, 0.22, 0.26; band check u_t share 1.1e-27, 4.6e-28, 9.2e-28
FINS, exact operator: all below lambda_(k+1) on 3/3 seeds; worst margin 30404.2x, 25671.9x, 6833.3x; lambda_(k+1) / lambda_k 26806.3, 29517.4, 8049.0; rank from the largest gap 6, 6, 6; relative change 0.12, 0.22, 0.26; band check u_t share 1.1e-27, 4.6e-28, 9.2e-28

## Chance levels and own-class nulls (exp41 --untrained, exp42 class nulls)

Missing dimensions, finite ground truth, mean over seeds 42-44 (README, protocol run part 5):

| | Burgers (5) | KdV (4) | heat (6) | heat, full (10) |
|---|---|---|---|---|
| FINS | 0.44 | 0.01 | 0.14 | 1.59 |
| null: Gaussian coefficients in FINS's class | 2.76 | 2.85 | 2.91 | 6.36 |
| null: uniform on FINS's class | 4.17 | 3.46 | 4.99 | 8.33 |
| LieGAN, trained / untrained | 2.98 / 2.54 | 2.84 / 2.05 | 4.06 / 3.57 | 6.66 / 6.31 |
| LaLiGAN, trained / untrained | 4.63 / 5.00 | 3.05 / 4.00 | 4.10 / 6.00 | 7.39 / 10.00 |
| Augerino, trained = untrained | 1.96 | 0.90 | 2.00 | 6.00 |
| Augerino, grown widths only (post hoc) | 4.00 | 4.00 | 5.00 | 9.00 |
| LIG 4 generators, trained / untrained | 2.11 / 2.03 | 2.90 / 2.64 | 3.45 / 3.23 | 6.89 / 6.56 |
| LIG k generators, trained / untrained | 1.97 (s42) / 1.95 | -- | -- / 1.61 | -- / 4.03 |
| LieNLSD (equation given, no training) | 0.85 | 0.67 | 1.12 | 1.40 |

Verdicts: P5a MET (s42), P5b consistent (per seed pending), P6a NOT MET (LieGAN worse after
training), P6b-e MET; Augerino grown-widths predictions all met.


## exp42 output with chance levels and untrained baselines (Colab, 2026-09-29 14:38)

Printed by the exp42 version with the seeds, floor and k-range columns, the Augerino width readout
and the class nulls (100 draws per seed). Trained rows as above; LIG with 5 generators: Burgers s42.

Missing dims (mean +/- std), spurious (mean +/- std), Grassmann angle + penalty, recovered, cosine; k (mean rank).

Burgers (5): LieNLSD k 4-5 (4.3), floor 0.7. LIG 5 generators, 10 epochs: 1.97, 1.97, 2.05 + 0.00, 1.0, 0.52 (1 seed).
  LieGAN untrained 2.54 +/- 0.31, 2.54 +/- 0.31, 2.11 + 0.00, 0.0, 0.42
  LaLiGAN untrained 5.00 +/- 0.00, 5.00 +/- 0.00, 3.50 + 0.00, 0.0, 0.00
  Augerino untrained = trained (1.96 +/- 0.01, 3.96 +/- 0.01, 2.03 + 4.93, 1.7, 0.57)
  Augerino grown widths only (post hoc) k 1: 4.00, 0.00, 0.00 + 9.87, 0.0, 1.00
  LIG 4 untrained 2.03 +/- 0.16, 1.03 +/- 0.16, 1.18 + 2.47, 0.7, 0.75
  LIG 5 untrained 1.95 +/- 0.14, 1.95 +/- 0.14, 1.87 + 0.00, 0.7, 0.54
  null uniform (5 of 30) 4.17 +/- 0.02, 4.17 +/- 0.02, 2.75 + 0.00, 0.0, 0.24
  null Gaussian (5 of 30) 2.76 +/- 0.02, 2.76 +/- 0.02, 2.23 + 0.00, 0.0, 0.45
  Augerino learned widths (G1..G7): s42 0.012 0.212 0.004 0.005 0.005 0.004 0.022; s43 0.011 0.181 0.004 0.005
  0.005 0.004 0.020; s44 0.011 0.154 0.003 0.005 0.005 0.003 0.017 -> grown past 0.05: G2 (t-translation)
KdV (4): LieNLSD k 3-4 (3.3), floor 0.7.
  LieGAN untrained 2.05 +/- 0.38, 2.05 +/- 0.38, 1.75 + 0.00, 0.0, 0.52
  LaLiGAN untrained 4.00 +/- 0.00, 4.00 +/- 0.00, 3.13 + 0.00, 0.0, 0.01
  Augerino untrained = trained (0.90, 3.90, 1.24 + 7.40, 1.0, 0.76)
  Augerino grown k 0: 4.00, 0.00, 0.00 + 9.87, 0.0, 0.00
  LIG 4 untrained 2.64 +/- 0.12, 2.64 +/- 0.12, 2.32 + 0.00, 0.0, 0.36
  null uniform (4 of 30) 3.46 +/- 0.01; null Gaussian 2.85 +/- 0.03, 2.42 + 0.00, 0.0, 0.31
  Augerino widths: 0.010 0.011 0.002 0.003 0.004 0.002 0.006 on every seed -> none grown
heat, finite (6):
  LieGAN untrained 3.57 +/- 0.41, 3.57 +/- 0.41, 2.52 + 0.00, 0.0, 0.38
  LaLiGAN untrained 6.00 +/- 0.00, 8.00 +/- 0.00, 3.81 + 4.93, 0.0, 0.01
  Augerino grown k 1: 5.00, 0.00, 0.00 + 12.34, 0.0, 1.00
  LIG 4 untrained 3.23 +/- 0.19, 1.23 +/- 0.19, 1.33 + 4.93, 0.0, 0.60
  LIG 10 untrained 1.61 +/- 0.13, 5.61 +/- 0.13, 1.62 + 9.87, 2.3, 0.67
  null uniform (10 of 60) 4.99 +/- 0.00, 8.99, 2.92 + 9.87, 0.0, 0.21; null Gaussian 2.91 +/- 0.10, 6.91 +/- 0.10, 2.15 + 9.87, 0.0, 0.51
heat, full (10):
  LieGAN untrained 6.31 +/- 0.30, 2.31 +/- 0.30, 1.96 + 9.87, 0.0, 0.56
  LaLiGAN untrained 10.00, 8.00, 4.40 + 4.93, 0.0, 0.01
  Augerino grown k 1: 9.00, 0.00, 0.00 + 22.21, 0.0, 1.00
  LIG 4 untrained 6.56 +/- 0.09, 0.56 +/- 0.09, 0.84 + 14.80, 0.0, 0.85
  LIG 10 untrained 4.03 +/- 0.28, 4.03 +/- 0.28, 2.84 + 0.00, 1.7, 0.72
  null uniform 8.33 +/- 0.02, 3.87 + 0.00, 0.0, 0.19; null Gaussian 6.36 +/- 0.12, 3.55 + 0.00, 0.0, 0.37
  Augerino widths: s42 0.012 0.022 0.003 0.004 0.004 0.003 1.208; s43 ... 1.227; s44 0.012 0.023 ... 2.205 -> G7 (u d_u)

## Wall times (Colab GPU, from the run logs)

LieNLSD 62-69 s; LieGAN 128-143 s; LaLiGAN 439-535 s; Augerino 730-736 s (Burgers, heat), 1165-1191 s (KdV);
LIG 10 epochs 3353-3426 s (Burgers, heat), 7116-7189 s (KdV), i.e. 335-719 s per epoch.
FINS discovery step on two CPU cores (ten evaluation trajectories, data generation included):
KdV 67 s (30 basis fields), Burgers 91 s (30), heat 163 s (60).
Second timing run with peak resident memory (python resource wrapper, same checkpoints, seed 7 operators,
data seed 1042): KdV 73 s / 814 MB, Burgers 86 s / 1310 MB, heat 163 s / 2103 MB. LIG: about 6 GB at
batch 1 on KdV's 128 x 128 grid without torchdiffeq (CPU smoke test, stopped).

## exp43: transport error against the exact operators (Colab, 2026-09-29)

| run | eta | eta_V | floor | sigma_(k+1) exact | sigma~_(k+1) learned | max Weyl shift | (iii) shares / bounds | sufficient condition |
|---|---|---|---|---|---|---|---|---|
| KdV s42 | 0.553 | 0.0058 | 0.0102 | 0.132 | 0.130 | 0.049 | 0.001-0.007 / 0.006-0.033 | fails |
| KdV s43 | 0.596 | 0.0123 | 0.0110 | 0.122 | 0.118 | 0.085 | 0.003-0.011 / 0.025-0.053 | fails |
| KdV s44 | 0.569 | 0.0069 | 0.0113 | 0.117 | 0.114 | 0.053 | 0.001-0.007 / 0.008-0.038 | fails |
| heat s42 | 0.070 | 0.063 | 0.00023 | 0.0346 | 0.059 | 0.056 | 0.03-0.54 / 0.26-0.90 | fails |
| heat s43 | 0.0403 | 0.0366 | 0.00021 | 0.0311 | 0.0373 | 0.035 | 0.010-0.334 / 0.221-0.696 | fails |
| heat s44 | 0.0408 | 0.0366 | 0.00028 | 0.0201 | 0.0360 | 0.032 | 0.016-0.614 / 0.119-0.712 | fails |

T1 met on 6/6 runs, T2 met on 3/3, T3 not met (0/3). eta_V / sigma_(k+1): KdV 0.04-0.10, heat 1.2-1.8.
Heat s43 and s44 rerun with parameter checksums: distinct operators (4754.430, 4756.563); eta_V 0.0365696 and
0.0365709, equal to four digits by coincidence. s42 reproduced (checksum 4831.577, eta 0.0702509).

Correction (05.10): the heat rows above loaded S_theta_heat_s{s}_g64x128_cno.pt, exp43's default name (no --tag), CNOs
other than the protocol's (the KdV rows loaded the protocol's steppers). Rerun through the protocol's CNOs
(S_theta_heat_s{s}_g64x128_cno_smooth_L1.pt, checksums 4786.927 / 4737.260 / 4735.527; CPU; logs in
results/exp43_recheck/):

| run | eta | eta_V | floor | sigma_(k+1) exact | sigma~_(k+1) learned | max Weyl shift | (iii) shares / bounds | sufficient condition |
|---|---|---|---|---|---|---|---|---|
| heat s42 | 0.0703 | 0.0674 | 0.00023 | 0.0346 | 0.0640 | 0.059 | 0.043-0.599 / 0.271-0.848 | fails |
| heat s43 | 0.1024 | 0.0306 | 0.00021 | 0.0311 | 0.0375 | 0.030 | 0.010-0.082 / 0.241-0.585 | fails |
| heat s44 | 0.3011 | 0.0334 | 0.00028 | 0.0201 | 0.0323 | 0.030 | 0.024-0.573 / 0.164-0.722 | fails |

T1 and T2 still met (Weyl and (iii) hold on 3/3; eta 0.070-0.30 >= 0.02). eta_V / sigma_(k+1): heat 0.98, 1.66, 1.95
(1.0-1.9; the first run 1.2-1.8); learned sigma~_10 / sigma~_11 0.92, 0.81, 0.94; superpositions outside the ten lowest
up to 0.60, 0.08, 0.57. Over the class the error is dominated by directions away from the algebra on s43 and s44.

## Vision: chance levels and size series on CIFAR-10 (Colab, 2026-09-30)

Average accuracy over the eight corruptions, seeds 100-104, paired arms; FINS as mean +/- std.

| training images | no_aug | FINS | null_init | null_smooth | randaug | autoaug | trivialaug |
|---|---|---|---|---|---|---|---|
| 950 | 0.313 | 0.378 +/- 0.008 | 0.346 | 0.339 | 0.361 | 0.343 | 0.334 |
| 5 000 | 0.413 | 0.541 +/- 0.013 | 0.547 | 0.479 | 0.502 | 0.478 | 0.465 |

FINS minus arm, paired, mean +/- std (FINS ahead on): 950: null_init +0.032 +/- 0.014 (5/5), null_smooth
+0.039 +/- 0.015 (5/5), randaug +0.017 +/- 0.010 (5/5); 5 000: null_init -0.007 +/- 0.015 (2/5), null_smooth
+0.062 +/- 0.023 (5/5), randaug +0.039 +/- 0.018 (5/5). At 5 000 the learned phi has collapsed (RMS
0.0015-0.011 against 0.07-0.25 for xi, eta). V3 met at 950, at 5 000 met against null_smooth only; V2 decided
on the paired 50 000 baselines; V1 below.

### 50 000 training images (fins and the chance levels only)

| training images | FINS | null_init | null_smooth |
|---|---|---|---|
| 50 000 | 0.602 +/- 0.006 | 0.591 +/- 0.019 | 0.555 +/- 0.025 |

FINS minus null_init +0.011 +/- 0.023 (4/5), minus null_smooth +0.047 +/- 0.025 (5/5): V1 met. The earlier
run behind the paper (FINS 0.566 +/- 0.020) is not reproduced; baselines at 50 000 rerun paired (V7, V8,
V2 final).

### Structure of the learned generators (generator_structure.py) and crop and flip in every arm

| fit set (training set) | translation share of (xi, eta) | image-dependent | RMS of phi | grey share of phi |
|---|---|---|---|---|
| 950 (950) | 0.9996 | 0.0002 | 0.050 | 0.995 (null_init 0.38) |
| 5 000 (5 000) | 0.940 | 0.0002 | 0.005 | 0.71 (null_init 0.24) |
| 5 000 (50 000) | 0.939 | 0.0002 | 0.005 | 0.74 (null_init 0.24) |

V4 met (10/10), V5 met (5/5, grey share at the 100th percentile of 100 initialisations). The learned
generators are image-independent global translations (2.2-2.4 px at tau = 1) and, at 950, grey offsets.

Crop and flip in every arm, 5 000 training images: no_aug 0.390, FINS 0.520 +/- 0.016, null_init 0.536 +/-
0.020, null_smooth 0.443, randaug 0.460, autoaug 0.470, trivialaug 0.470. FINS minus the best baseline +0.050
+/- 0.027 (5/5): V6 not met; the lead sits in the noises, blur and pixelate, and null_init keeps it.

## Images, second design: planted symmetries (fins_image.py, Fashion-MNIST, CPU, seeds 100-102)

Share of the planted generator outside the 3 lowest eigenvectors of the 14-field class, log-probabilities
(chance 0.89); control = same architecture without augmentation.

| planted | MLP | MLP control | CNN | CNN control |
|---|---|---|---|---|
| rotation | 0.17-0.24 | 0.52-0.57 | 0.34-0.48 | 0.72-0.81 |
| scaling | 0.23-0.52 | 0.98-0.99 | 0.76-0.84 | 0.88-0.94 |
| projective (x^2, xy) | 0.21-0.30 | 0.98-1.00 | 0.34-0.51 | 0.82-0.93 |
| brightness | 0.12-0.27 | 0.92-0.94 | 0.19-0.39 | 0.87-0.99 |
| intensity | 0.78-0.91 | 0.88-0.91 | 0.20-0.95 | 0.70-0.99 |

In LieGG's class GL(2), share outside the lowest eigenvector of FINS against LieGG's generator: MLP
rotation 0.14-0.16 vs 0.12-0.49, scaling 0.20-0.27 vs 0.39-1.00; CNN rotation 0.26-0.38 vs 0.62-0.79,
scaling 0.11-0.29 vs 0.82-0.99. Verdicts: I2, I3, I5, I6 met; I1, I4 (rank 1) and I7 not met.

## Images, second design: planted symmetries on CIFAR-10 (Colab, 50 000 images, 15 epochs, seeds 100-102)

Share outside the 3 lowest of 24 fields (chance 0.94), log-probabilities.

| planted | MLP | MLP control | CNN | CNN control |
|---|---|---|---|---|
| rotation | 0.13-0.57 | 0.61-0.93 | 0.74-0.89 | 0.79-0.86 |
| scaling | 0.99-1.00 | 0.99-1.00 | 0.88-0.99 | 0.94-0.98 |
| projective (x^2, xy) | 0.48-0.69 | 0.79-0.99 | 0.78-0.80 | 0.78-0.83 |
| grey offset | 0.08-0.09 | 0.25-0.32 | 0.05 | 0.23-0.30 |
| intensity | 0.23-0.27 | 0.31-0.32 | 0.14-0.18 | 0.28-0.36 |
| saturation | 0.99-1.00 | 1.00 | 1.00 | 1.00 |
| hue | 1.00 | 1.00 | 1.00 | 1.00 |

GL(2): MLP rotation FINS 0.02-0.11 vs LieGG 0.18-0.34; CNN rotation FINS 0.61-0.84 vs LieGG 0.67-0.75;
scaling recovered by neither. Verdicts C1-C5 all not met (C1: grey offset margin to the control < 0.2 on
2/3; C5: intensity recovered). Saturation and hue: CIFAR-10 is weakly saturated, the networks' gain along
chroma is ~50 times that along rotation, planted training lowers it by at most 2.3 times (CPU diagnostic).

## Images: pretrained encoders (Colab, 512 Imagenette validation images, 224 px, 24 fields)

| encoder | expected subspace | missing (random) | lowest eigenvectors |
|---|---|---|---|
| DINOv2 ViT-S/14 | tx, ty, scaling, stretch, intensity, saturation, hue | 3.25 of 7 (4.96) | channel mixings; intensity, grey offset rank 2, saturation, hue rank 4 |
| ResNet-50 V1 | tx, ty, scaling, stretch | 3.37 of 4 (3.33) | channel mixings; saturation 18, hue 22 |
| random ViT-S/14 | (DINOv2's 7) | 5.71 of 7 (4.96) | quadratic geometric fields; colour 20-24 |

Verdicts: E1, E3, E4 met; E2 not met. DINOv2 misses all 4 of ResNet-50's geometric dimensions (4.00,
chance 3.33); ResNet-50 misses 4.23 of DINOv2's 7.

Loss ratio control / planted (CIFAR-10, log-probabilities): MLP rotation 2.9-4.1, projective 2.2-2.9,
scaling 2.1-2.5, hue 3.4-4.7, saturation 1.5-1.8; CNN rotation 0.98-1.30, projective 0.80-1.14, scaling
0.80-1.04, brightness 9.4-17. CPU replicate: the augmented CNN is robust to rotation at finite angles
(accuracy at 30 deg 0.38 vs 0.27) but its first-order loss falls only 1.1-1.45x with any derivative scale.

## Images: smoothed criterion (Colab, sigma 0 / 0.05 / 0.1 / 0.2, 8 noise draws)

CIFAR-10 CNN rotation 0.69-0.96 -> 0.65-0.68 outside the 3 lowest (control 0.71-0.76), loss ratio 0.83-1.22;
CIFAR-10 CNN intensity lost (0.13 -> 0.51-1.00); saturation, hue 0.99-1.00 at every sigma; MLPs unchanged.
Encoders, geometric subspace (tx, ty, scaling, stretch) missing at sigma 0 -> 0.2: DINOv2 4.00 -> 2.52, ResNet-50
3.38 -> 2.85, untrained ViT 3.69 -> 3.79 (chance 3.33); DINOv2 saturation and hue rank 4 -> 19-22. S1-S3 and S5
not met, S4 met. Not adopted.

## Images: class-conditional symmetries of ResNet-50 (Colab; Pet cats and dogs, Imagenette classes)

Relative pencil (N_S c = lambda M c, M the feature Gram), order 1, k = 8 of 86 (random 7.26). Semantic test: Pet group
closer to springer than to the other class: cat tench 20/20, garbage truck 18/20, church 19/20; dog 20/20, 20/20,
20/20 (F1 met). Plain pencil: church breaks it (3/20, 5/20; F5 met). Order 2: no gain (F4 not met; order-2 share of
the cat subspace 0.08). Reliability: Pet halves of 900, within cat 6.43, dog 5.15; cat-dog specificity +0.24.
Lowest fields: local deformations and photometric edits at the image borders; no global affine or photometric
generator in any group's subspace. Same images for all operators (exploratory): by the raw distance cat and dog are
closer to springer on 10/10 (all images), but so is every other operator -- springer's subspace is the most reliable
(own split-half 4.27 against 4.60-5.43); corrected for reliability (distance - mean own split-half distance) the six
operators are indistinguishable on the same images (+0.03..+0.42) and the comparisons fall to 3-9/10.
Untrained ResNet-50: semantic test cat/dog closer to springer than to church and tench 20/20 as in the trained
network, than to garbage truck 0/20 (trained 18/20, 20/20): F2 not met; specificity +1.27 vs trained +0.24: F3 met.
Noise-corrected specificity (exploratory): trained cat-dog +0.06, cat-springer +0.15, dog-springer +0.08, untrained
+0.56, +0.86, +0.86; other pairs +0.31..+1.29 trained. Flows (trained, 200 held-out Pet images, 5 percent image
change): lowest-3 / random 0.86 (cat, order 1), 0.81 (cat, order 2), 0.75 (dog); highest / random 1.2-1.7: F6 not
met. Untrained on the same images: tench and church 10/10 (+1.28..+2.04), garbage truck 0/10: F8 not met. The
untrained logits are saturated (log-odds -15..-57), and the distance from cat and dog to a class grows with that
class's |log-odds| (truck 34.7 < springer 37.8 < tench 47.5 < church 56.9). Corrected for reliability, the F1 ordering
holds in the untrained network on the per-class images as well (17-20/20): it follows the image sets.
Reading: no cat-specific symmetry at first or second order; the class-conditional invariances of the trained network
are network invariances shared by all class scores. Exploratory contrastive pencil (class score against the mean of
the six): trained cat and dog closer to springer than to the others on the same images 10/10, but shallow (cat
lambda_1 0.34) and poorly reliable (own split-half 6.0-6.8 of 7.26). Readout-geometry control (mean log-odds
gradients in feature space, no fields): cat nearer tench and garbage truck than springer (0/10, 0/10, church 10/10;
R1 not met); Spearman with the contrastive specificity +0.26 (R2 not met): the contrastive structure is not the head's.
The ordering holds on sub-pools without cat images (dog images 9-10/10; tench, truck, church images 10, 6, 10/10) and
with other denominators; it needs pools of about 6 000 images. Confirmatory run pre-registered (G1-G5): images no
earlier run used, ResNet-50 and ConvNeXt-T, 24 extra single-class operators in six superclasses; the semantic test
proper is G4 (within-superclass specificity smaller, permutation p < 0.01) with G5 (beyond the readout, p < 0.05).
Confirmatory run (6 349 fresh images): G1 met (ResNet-50, cat closer to springer 10/10 on all three, +0.39..+0.62);
G2 not met (ConvNeXt-T: tench 0/10, garbage truck 7/10, church 2/10); G3 met (-0.26, -0.10); G4 met (between -
within +0.301, p 0.0001; +0.186, p 0.0001); G5 met (+0.128, p 0.0006; +0.130, p 0.0001). Semantic structure of the
class-specific invariances confirmed on both networks beyond the head geometry; the single cat-springer ordering is
network-specific. Exploratory: holds with pair reliability as a covariate, without any one superclass, on sub-pools
(tench images only: p 0.0001, 0.003), for k = 4 and 12, and in the relative pencil; animal-animal pairs closer
than animal-artefact pairs.


## Fresh evaluation seeds and closure (Colab, data seeds 2042-2044, the protocol's operators)

Every operator first reproduced the protocol eigenvalues on 1042-1044 (Burgers exactly, KdV to 1e-5, heat to 5e-12
with the protocol's tag _smooth_L1). F1: Burgers all five below lambda6 on 3/3, heat all ten below lambda11 on 3/3
(worst margins 1.1-4.2x), KdV all four with margins 366-616x on 3/3 -- met for all three; on the protocol data it
was 8/9. F2: missing 0.24 (Burgers), 0.01 (KdV), 0.19 (heat finite), 1.36 (heat class) against Gaussian nulls 2.81,
2.86, 3.01, 6.36 -- met. Closure (escaped share of the brackets, against the null mean minus 2 sd): K2 met on 18/18
runs; K1 not met (exact KdV 0.069 > 0.06 on fresh s43); K3 (KdV <= 0.10 on 3/3) not met: 0.110, 0.162, 0.070
protocol, 0.097, 0.189, 0.076 fresh (exact operator 0.043-0.069); seed 43's operator is the least accurate.

## Images: finite flows of the class-specific directions (W2)

3 lowest contrastive directions of each of the 24 single-class operators, 5 percent image change, 600 design-pool
images: log transfer smaller within superclasses than between, +0.042 (ResNet-50), +0.093 (ConvNeXt-T), both
p 0.0001 -- W2 met. Carried by the mammal superclasses; birds and vehicles (ResNet-50), birds and buildings
(ConvNeXt-T) show no within-superclass advantage.

## Neural generators of a 2D equation (exp45, exact operator; CPU)

u_t = div(u grad u) on [-8, 8)^2, 6-dim algebra (d_x, d_y, d_t, rotation, x d_x + y d_y + 2t d_t, t d_t - u d_u);
predictions N1-N5 registered after one development seed (README). Seeds 42-44, missing dimensions in characteristic
space on held-out trajectories. N1 met: polynomial class six lowest <= 1.2e-6 (1.13e-6, 8.0e-7, 1.0e-6), lambda7 / lambda6 284, 312, 224, missing
0.000. N2 met: point-field networks (m = 6) trained by the Ky Fan sum of the pencil's eigenvalues, 0.104, 0.092, 0.086;
containment of every generator >= 0.95. N3 met: untrained 2.10, 2.76, 2.54 (random 6-dim subspaces 2.58-2.78).
N4 met: m = 8, full span 0.135, 0.113, 0.090, six lowest 0.400, 0.287, 0.370. N4b met (registered expected failure):
lambda7 / lambda6 inside the span 2.25, 1.64, 1.50 -- directions 7, 8 are the class's phi = x^2, y^2 near-symmetries
(2-3e-4), the network's scalings sit at 1e-4. N5 met (learned FNO2d per seed, trained on 512 trajectories of its
own data seed; registered before any FNO was trained): one-step error 0.0022 of the increment on held-out data
(gate 0.05); neural m = 6 through the FNO 0.102, 0.091, 0.092 (exact operator 0.104, 0.092, 0.086; untrained
2.10-2.76); polynomial class through the FNO six lowest <= 1.7e-5, lambda7 / lambda6 27-33, missing 0.0004-0.0014.
The neural floor is the network's precision; the learned operator shrinks the polynomial gap from ~300x to ~30x.
N6 met (m = 8 through the FNO, registered before its output was seen): full span 0.148, 0.076, 0.089, six lowest
0.197, 0.216, 0.181. N6b met (expected failure): lambda7 / lambda6 2.67, 3.29, 1.74 -- no rank from the neural class.
Rank by the protocol's largest-ratio rule (first k + 4 eigenvalues of the polynomial class): 6 on 3/3 through the FNO;
through the exact operator the largest ratio falls inside the algebra (numerical floors), as in 1D.

## Criteria beyond the eigenvalue (2D and 1D exact; registered V1-V4)

V1 not met: operator sensitivity (FNO quotient / exact eigenvalue) separates round-off and time-difference floors
(10^3 - 10^21) but the resolution-limited sixth direction rises 5.4-8.2x and the near-symmetries 3.2-5.3x. V2 met:
the affine (matrix) class aff(4), 20 fields, keeps the algebra (missing <= 0.0002) with lambda7 / lambda6 3300-4100
(exact) and 210-330 (FNO), against 224-312 and 27-33 in degree 2; its largest ratio reads 6 through the FNO (3/3).
V3 met: at mu = 0.01 the near-symmetries' quotients rise 11-61x, the algebra stays <= 4.5e-6, the gap reaches
905-1556. V4 met: the closure-gated rank (largest k whose k lowest eigenvectors leave <= 0.1 of their brackets) is 6
on 6/6 2D runs (exact and FNO), 4 on 6/6 exact KdV runs (largest ratio: 2), and not 10 for heat (5/6; its class
truncates the superposition ideal).
V4L (1D learned operators, registered before the Colab run): KdV closure-gated rank 4 on 3/6 (escaped at k = 4
0.07-0.19, at k = 5 0.32-0.44) -- not met; heat != 10 on 6/6 -- met (expected failure); Burgers 5 on 1/6. The
largest ratio reads learned KdV 6/6: closure needs accurate eigenvectors, the ratio needs levelled floors.

## Generalised symmetries in a jet-polynomial class (exp46, exact solvers; CPU; registered J1-J4)

15 monomials of KdV weight <= 7 in the scaled jets (u, u_x, ..., u_5x); integrating-factor RK4 with its exact tangent;
data 4600 + s (training, 12 trajectories) and 5600 + s (held-out, 6), seeds 42-44. J1 met: KdV, the three lowest
contain u_x, K3, K5 (missing <= 2.2e-14; random 1.74-1.79), lambda4 / lambda3 2.6e4, 7.4e3, 2.4e6; K3 at 1e-16, K5 at
1.9e-13 - 1.4e-11; the fourth eigenvector (2.4e-7 - 1.4e-6) is the solitons' travelling-wave relation. J2 met: porous
medium, u_x and u u_xx + u_x^2 (missing <= 1.2e-11; random 1.55-1.56), lambda3 / lambda2 3.7-7.8e5, no third
generalised symmetry (lambda3 1.1-1.2e-4, a background shift). J3 met: heat, the seven linear monomials (missing
<= 3.6e-15), lambda8 / lambda7 4.4e13 - 1.8e28 (the seven at round-off). J4 not met: largest-ratio rank heat 7, 7, 7; porous medium 2, 1, 2; KdV
2, 2, 3 (expected failure) -- the true characteristics' floors (round-off, time stepping) differ by more orders than
the gap.

## Training with mu = 0.01 (exp45, m = 8, exact operator; CPU; registered T1-T3, paper M1-M3)

T1 met: full span missing 0.061, 0.091, 0.104 (mu = 1: 0.135, 0.113, 0.090). T2 not met: lambda7 / lambda6 inside the
span on held-out pairs 9.65, 6.75, 2.31 against >= 10 (mu = 1: 2.25, 1.64, 1.50). T3 met: six lowest directions
missing 0.065, 0.096, 0.111 (mu = 1: 0.400, 0.287, 0.370). The smaller absolute term removes the near-symmetries'
advantage, so the network's six lowest directions become the algebra; the least accurate field (a scaling) rises
with the normalisation and caps the gap.

## A class from Lie theory: tau = tau(t) (V5; CPU)

48 of the 60 degree-2 fields (tau in span{1, t, t^2}), sub-pencils of the saved exact and FNO pencils. V5a met:
missing <= 0.0004. V5b not met: lambda7 / lambda6 rises only 1.37-1.38x (exact: 391, 429, 306) and 1.2-1.5x (FNO: 41,
47, 38), against >= 2x registered; the affine class gives 3300-4100 and 210-330. V5c met: largest ratio 6, 6, 6
(FNO), 1, 2, 1 (exact). The near-symmetries owe their small loss to phi, not to tau.

## Finite flows exp(s v) through the exact 2D operator (exp45_flows.py; CPU; registered F1-F6, paper X1-X6)

Evolutionary flow dw/ds = P Q[w; t] on snapshots (u_t from the solver, 2/3 band, windowed coordinates beyond |x| = 6),
defect E = ||S_K(T_s u0) - T_s u_K|| / ||T_s u_K - u_K||, K = 8, sizes eps 0.003-1; held-out data 5000 + s.
F1 not met (9, 10, 8 of 12 below 1e-4): the algebra and eigenvectors 1-6 stay at their first-order residuals
(6e-9 to 2.9e-4) at every size, growth <= 1.8x (translations and eigenvectors 1-2: 3e-13 -> 2.2-2.8e-7, the
s-integrator); the bound was
calibrated on one development seed. F2 met: E(0.003) / r1 0.74-1.21 (33/33). F3 met: the near-symmetries
(eigenvectors 7-10, E(0.003) 1.1-2.6e-2) break down on 12/12 -- at eps = 0.1-0.3 when tau, changing sign across the
domain (x t, y t terms), carries 15-37 % of the field (ill-posed flow), at eps = 1 when it carries <= 6 % (u < 0). F4 met: with tau removed, E stays
within 1.19x of E(0.003) while u > 0 (57 cases). F5 met: random fields 0.105-0.152, growth <= 2.53x. F6 not met:
mu = 1 networks, directions 1-6 flat 18/18 but directions 7, 8 break down on 3/6 (first-order values overlap the
six's); mu = 0.01 networks (no prediction): directions 7, 8 start 4.7-12x above the six and break down on 6/6.

## Finite flows of jet characteristics (exp46_jet_flows.py; CPU; registered G1-G3)

Lawson RK4, adaptive step (1 / (4 x rate), rate re-evaluated every 16 steps, cap 200 000 steps), flow band: the
solver's for heat and the porous medium, |k| < 32 for KdV; 2 held-out trajectories x starts 0, 16, seeds 42-44.
G1 not met (KdV): the symmetries' largest E at any size is 2.5e-12 (heat), 1.1e-7 (porous medium) and 1.2e-6 to
3.9e-5 for KdV (registered 1e-5), K5's fifth-order flow included (<= 1.9e-5); the KdV floors are the flow
integrator's error (K3 on s42: 2.7e-5 with 4 steps per unit rate, 3.1e-8 with 16). G2 not met (KdV): the next eigenvector
starts 1.6-3.9e11 (heat), 1.8-6.9e5 (porous medium) and 80-5990 (KdV) times above the r lowest. G3 not met (heat 4/6,
KdV 3/6): the next two eigenvectors break down on 6/6 for the porous medium (positivity, near blow-up), 4/6 for heat;
KdV's travelling-wave relation grows only 1.2-2.7x. The jet flows confirm the generalised symmetries; as a breakdown
criterion they hold only for the porous-medium equation.
Post hoc, with four times the steps, the KdV symmetries stay below 2.9e-6 (K5's flow below 1.0e-7 on two seeds)
and G2's KdV ratios become 1.3e5, 3.5e5, 861.

## Through the learned FNO2d: mu = 0.01 and finite flows (Colab; registered MF1-MF3, XF1-XF5; logs in results/neural2d/flows/logs_fno/)

MF1-MF3 met: the mu = 0.01 networks keep the algebra in their span (0.061, 0.090, 0.104; untrained 1.75-1.95), their
six lowest directions miss 0.064, 0.094, 0.112, and lambda7 / lambda6 on held-out pairs is 10.0, 5.9, 2.3 against
2.67, 3.29, 1.74 at mu = 1. Finite flows through the FNO (end snapshot its own rollout): XF2 met (E(0.003) / r1
0.70-1.38, 54/54), XF4 met (22/24 finite with growth < 10x; the scalings grow 2.6-16x), XF5 met (mu = 0.01 network
directions 1-6 within 3x on 17/18, 7-8 non-finite on 6/6; mu = 1: 18/18 and 5/6). XF1 not met: translations at
4.1-7.7e-4, their first-order residuals (the FNO does not commute with continuous shifts below its error). XF3 not
met: eigenvectors 7-10 break down on 3/4, 2/4, 4/4. P5a / P5b (LIG_k, Burgers s43 / s44): trained, scores pending
(exp42 on the Drive files).

## LieNLSD as published, on FINS's derivatives, and in 2D (CPU; registered L1-L10; results/lienlsd_published/, results/lienlsd_spectral/, results/neural2d/lienlsd_2d/)

Correction first: pde/method_*.py are our reimplementations; LieNLSD's official code fits an MLP F to u_t on
finite-difference jets and uses its Jacobian, while our v1 wrapper used the exact equation. The v1 "LieNLSD" rows
above are therefore an oracle variant; the wrapper's bugs listed in PROTOCOL.md section 6 were ours.
Missing dimensions, m = k, means over seeds 42-44 (Burgers 5 / KdV 4 / heat 6 / heat's class 10):
- LieNLSD as published (fitted F, central differences): 0.68, 0.91, 0.30, 0.42; F at initialisation 1.07, 0.98,
  1.14, 2.29; own count (relative threshold, at most k) 1.77, 1.58, 0.57, 0.96.
- Exact right-hand side, same pipeline: 0.40, 0.66, 0.43, 0.56. v1 wrapper re-run: 0.85, 0.67, 1.12, 1.40 (L1 met).
- On FINS's derivatives: exact RHS 0.46, 0.64, 0.00, 0.00 (L5 not met: KdV); fitted F 0.70, 0.95, 0.09, 0.13.
- FINS: 0.44, 0.01, 0.14, 1.59. L2 met on means (not on 3 of 12 seed-columns), L3 met.
2D (exp47, m = 6, held-out probe, characteristic space): LieNLSD as published 0.868, 0.400, 0.935; on FINS's
derivatives 0.727, 0.594, 0.130; exact RHS 0.166, 0.005, 0.025 (central) and <= 7e-4 (FINS's derivatives); F at
initialisation 1.88-2.09; FINS's polynomial class through the FNO2d <= 1.4e-3 (L7-L9 met).
With 2^20 training points (the official data size; L11-L12): see the last item below.
- With the official amount of training data (2^20 points, L11-L12 met): fitted F 0.55, 0.90, 0.32, 0.43 (per seed
  Burgers 0.02 / 1.15 / 0.47); F at initialisation 1.07, 0.98, 1.14, 2.29; own count 1.38, 1.57, 0.59, 0.93. This is
  now the paper's LieNLSD row; FINS leads on Burgers (narrowly), KdV and the finite heat algebra, LieNLSD on heat's
  class; on Burgers LieNLSD misses fewer on seeds 42 and 44 (0.02, 0.47 vs FINS ~0.34, 0.91), FINS on seed 43 and on
  the mean. P5a met 3/3 (LIG with 5 generators 1.97, 1.98, 1.69 against FINS ~0.34, 0.07, 0.91); P5b not met (s44:
  trained 1.69 against untrained 2.11; s42 ~0.22, s43 0.01).

## Noise and temporal thinning with perfect models (exp48; CPU; registered R1-R2, T1-T2, N1-N4; results/noise_thin/)

Heat and KdV protocol data, training and evaluation trajectories corrupted alike (corrupt.py): Gaussian noise of
sigma times each trajectory's std, or every s-th snapshot. FINS through the exact operators (u_t by five-point
differences of the corrupted snapshots, windows on snapshots 2 .. Nt - 3); LieNLSD with the exact RHS, determining
system on the same evaluation trajectories (central or spectral jets) or on the training ones, and with F fitted on
10 Nx Nt points of the corrupted training data. Scores on the clean trajectories. Missing dimensions, m = k, means
over seeds 42-44, heat (6) / KdV (4):

| | clean | 1e-4 | 3e-4 | 1e-3 | 1e-2 | s = 2 | s = 4 | s = 8 |
|---|---|---|---|---|---|---|---|---|
| FINS, exact operator | 0.00 / 0.01 | 0.32 / 0.05 | 0.63 / 0.31 | 1.62 / 1.15 | 2.95 / 2.16 | 0.00 / 0.01 | 0.00 / 0.01 | 0.00 / 0.01 |
| LieNLSD exact RHS, eval. system, central | 0.22 / 0.00 | 0.65 / 0.98 | 0.93 / 0.98 | 1.10 / 0.50 | 1.40 / 0.51 | 0.54 / 0.00 | 0.70 / 0.00 | 0.90 / 0.00 |
| LieNLSD exact RHS, eval. system, spectral | 0.00 / 0.00 | 0.81 / 0.65 | 1.07 / 0.45 | 1.08 / 0.48 | 1.41 / 0.99 | 0.00 / 0.00 | 0.01 / 0.00 | 0.20 / 0.00 |
| LieNLSD exact RHS, training system | 0.43 / 0.66 | 0.59 / 0.99 | 0.84 / 0.99 | 1.27 / 0.83 | 1.41 / 0.87 | 0.73 / 0.63 | 0.75 / 0.63 | 0.93 / 0.72 |
| LieNLSD as published (F, training system) | 0.55 / 0.91 | 0.58 / 0.99 | 0.92 / 0.99 | 1.23 / 1.13 | 1.60 / 1.43 | 0.40 / 1.24 | 0.67 / 1.21 | 0.80 / 1.42 |
| LieNLSD F, eval. system | 0.43 / 0.85 | 0.67 / 1.01 | 0.81 / 0.68 | 1.10 / 1.28 | 1.39 / 1.28 | 0.21 / 1.26 | 0.40 / 1.13 | 0.79 / 1.43 |

Verdicts: R1 not met (heat, fitted F 0.55 / 0.72 against the first run's 0.30 / 0.42: one torch thread instead of two
changes the floating-point order of training; seed 42 with two threads reproduces the first run exactly; the exact
rows and KdV's F reproduce); R2, T1, T2 met; N1 not met (heat: FINS 0.32 at 1e-4, seed 42 0.91); N2, N3, N4 met.
Readings: thinning costs FINS nothing; under noise FINS misses fewer than LieNLSD with the exact RHS up to 3e-4 and
more from 1e-3, losing d_t first (its characteristic is the noisy u_t), while LieNLSD's translations are structural
zeros of its system and its scalings go at 1e-4; with the exact RHS LieNLSD's KdV gap (0.66) is the training data's (0.00 on the evaluation
trajectories), its fitted F misses 0.85-0.91 on either. Pending: the learned-operator arm (Colab; C1).

## Noise and thinning through learned operators (exp48_learned.py; registered C1, LC, LT1-LT2, LN1-LN4; results/noise_thin/summary_learned.md)

Each method with the model it learns from the corrupted data: FINS through operators retrained on the corrupted training
trajectories with the protocol's settings (clean: the protocol's checkpoints, verified by reproducing their protocol
rows; clean-rt: retrained on clean data), LieNLSD as published (F fitted to the same data, exp48). exp48's windows and
u_t; scored on the clean evaluation trajectories. KdV on a CPU, heat on the Colab GPU.
Missing dimensions, m = k, mean over seeds 42-44:

| heat (6) | clean | clean-rt | 1e-4 | 3e-4 | 1e-3 | 1e-2 | s = 2 | s = 4 | s = 8 |
|---|---|---|---|---|---|---|---|---|---|
| FINS, learned operator | 0.14 | 0.11 | 0.15 | 0.66 | 1.68 | 2.96 | 0.09 | 0.19 | 0.11 |
| FINS, exact operator | 0.00 | -- | 0.32 | 0.63 | 1.62 | 2.95 | 0.00 | 0.00 | 0.00 |
| LieNLSD as published | 0.55 | -- | 0.58 | 0.92 | 1.23 | 1.60 | 0.40 | 0.67 | 0.80 |
| heat, class (10): FINS, learned | 1.58 | 1.50 | 1.33 | 1.38 | 2.32 | 3.25 | 1.10 | 1.11 | 1.34 |
| heat, class (10): LieNLSD as published | 0.72 | -- | 0.71 | 1.14 | 1.62 | 2.74 | 0.56 | 0.83 | 1.02 |

| KdV (4) | clean | clean-rt | 1e-4 | 3e-4 | 1e-3 | 1e-2 | s = 2 | s = 4 | s = 8 |
|---|---|---|---|---|---|---|---|---|---|
| FINS, learned operator | 0.01 | 0.01 | 0.07 | 0.49 | 1.18 | 2.20 | 0.01 | 0.01 | 0.01 |
| FINS, exact operator | 0.01 | -- | 0.05 | 0.31 | 1.15 | 2.16 | 0.01 | 0.01 | 0.01 |
| LieNLSD as published | 0.91 | -- | 0.99 | 0.99 | 1.13 | 1.43 | 1.24 | 1.21 | 1.42 |

Worst margins through the heat CNOs (every generator below lambda11): clean 1.4-2.9x, clean-rt 1.3-4.0x, thinned
1.6-5.1x, 1e-4 1.5-2.7x (3/3), 3e-4 0.42-1.2x (1/3), 1e-3 0.14-0.25x, 1e-2 0.08-0.12x (0/3).
Worst margins through the KdV steppers: clean 351-884x, clean-rt 375-839x, thinned 358-805x, 1e-4 61-72x, 3e-4
8.2-9.9x, 1e-3 1.2-1.4x, 1e-2 0.2x (0/3 below lambda5).
Verdicts: C1 met (KdV 0.01, heat 0.14); LC met (KdV 0.011-0.015, heat 0.11); LT1 met; LT2 met (heat 0.09 / 0.19 / 0.11
against clean-rt 0.11, below LieNLSD as published on every seed); LN1 not met (heat 1e-4: 0.15 against exact 0.32,
bound 0.27; seed 42: exact 0.91, learned 0.20; s43, s44 learned 0.10, 0.14 against 0.02); LN2 not met (KdV 3e-4: 0.49
against exact 0.31, band 0.15); LN3 met (heat 2.96 against 1.60 at 1e-2); LN4 met (d_t containment heat 0.21-0.44 and
0.04-0.13, KdV 0.35-0.38 and 0.01-0.03 at 1e-3 and 1e-2).
Unregistered: heat against LieNLSD as published fewer at 1e-4 (every seed) and 3e-4 (two seeds of three), more at 1e-3;
heat's class LieNLSD's on the means in every condition.
Exploratory (exp48_gain.py, not registered): over the horizon the exact solver keeps 0.55-0.56 of white noise above
|k| = 32 (its 2/3 dealiasing removes the rest); the learned steppers keep 0.85-0.96 up to 1e-3 and 0.00 at 1e-2.

## Noise study in weak form (exp49 weak form, exp50 two-stage, exp51; registered WN1-WN10; results/noise_weak/)

Heat and KdV, seeds 42-51 (LieNLSD_F and the learned arm 42-44), Gaussian noise of sigma times each trajectory's std on
the evaluation data (and on the training data of the fitted models); scored on the clean trajectories. Missing
dimensions, finite algebra / non-trivial generators, means over seeds:

| heat (6) | clean | 1e-3 | 1e-2 | 3e-2 | 5e-2 | 1e-1 |
|---|---|---|---|---|---|---|
| FINS weak | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.01 | 0.05 / 0.02 | 0.37 / 0.07 |
| FINS pointwise | 0.00 / 0.00 | 1.71 / 0.83 | 2.72 / 0.99 | 3.01 / 1.03 | 3.05 / 1.06 | 3.12 / 1.12 |
| two-stage H 0.2, hard | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.30 / 0.30 | 0.30 / 0.30 |
| two-stage H 0.05, hard | 0.00 / 0.00 | 0.00 / 0.00 | 0.90 / 0.90 | 2.82 / 2.82 | 2.96 / 2.96 | 3.60 / 3.60 |
| LieNLSD exact RHS, eval. system | 0.29 / 0.29 | 1.07 / 1.07 | 1.45 / 1.45 | 1.45 / 1.45 | 1.49 / 1.49 | 1.65 / 1.65 |
| LieNLSD as published (F) | 0.55 | 1.23 | 1.60 | 1.26 / 1.26 | 1.11 / 1.11 | 1.19 / 1.19 |

| KdV (4) | clean | 1e-3 | 1e-2 | 3e-2 | 5e-2 | 1e-1 |
|---|---|---|---|---|---|---|
| FINS weak | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.04 / 0.00 | 0.20 / 0.00 | 0.91 / 0.00 |
| FINS weak, learned operator (42-44) | 0.00 / 0.00 | 0.00 / 0.00 | 0.15 / 0.02 | | | |
| FINS pointwise | 0.01 / 0.01 | 1.14 / 0.08 | 2.16 / 0.17 | 2.17 / 0.17 | 2.18 / 0.18 | 2.18 / 0.18 |
| two-stage H 0.2, hard | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| two-stage H 0.05, hard | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 2.00 | 2.00 / 2.00 | 2.00 / 2.00 | 2.00 / 2.00 |
| LieNLSD exact RHS, eval. system | 0.00 / 0.00 | 0.51 / 0.51 | 0.52 / 0.52 | 0.98 / 0.98 | 0.84 / 0.84 | 0.68 / 0.68 |
| LieNLSD as published (F) | 0.91 | 1.13 | 1.43 | 1.32 / 1.32 | 1.20 / 1.20 | 1.13 / 1.13 |

Verdicts: WN1-WN10 met. Own counts: heat 10 and KdV 4 on 10/10 seeds up to 1e-2 (KdV also at 3e-2). Independent check
(a separately written scorer): every cell within 4e-8 of summary.json per seed.

## Benjamin-Ono, exact operator (exp52; registered BO1-BO6; results/bo/)

Seeds 42-51 (LieNLSD 42-44); missing dimensions, finite algebra / non-trivial (Galilean, scaling), means:

| | clean | 1e-2 | 3e-2 |
|---|---|---|---|
| FINS weak | 0.000 / 0.000 | 1.000 / 0.001 | 1.000 / 0.001 |
| FINS pointwise | 0.000 / 0.000 | 2.208 / 0.292 | 2.554 / 0.565 |
| two-stage H 0.2, hard / soft | 2.000 / 0.406 | 2.000 / 0.406 | 2.000 / 0.313 |
| two-stage H 0.05, hard / soft | 2.000 / 0.044 | 2.000 / 0.046 | 2.000 / 0.057 |
| LieNLSD as published, training / eval. system | 1.354 / 1.176 | 1.726 / 1.622 | |

(two-stage rows: finite = non-trivial.) Verdicts: BO1-BO6 met. The weak form's fifth eigenvector is the projective field
of inviscid Burgers (share 1.000 in the five lowest on every run; lambda5 / lambda4 = 90-192 clean); under noise it
enters the four lowest and takes one translation's place. LieNLSD first fitted with Adam on the clean seeds (1.163 /
1.267), refitted with Adan as published.

## Kuramoto-Sivashinsky, exact operator (exp53; registered KS1-KS5, NL8, NL9; results/ks/)

Seeds 42-51 (LieNLSD_F 42-44); missing dimensions, finite algebra / non-trivial (Galilean), means:

| | clean | 1e-2 | 3e-2 | 5e-2 | 1e-1 |
|---|---|---|---|---|---|
| FINS weak | 0.000 / 0.000 | 0.000 / 0.000 | 0.001 / 0.000 | 0.004 / 0.000 | 0.017 / 0.001 |
| FINS pointwise | 0.000 / 0.000 | 1.275 / 0.008 | 2.038 / 0.078 | 2.330 / 0.394 | 2.894 / 0.895 |
| two-stage H 0.2, hard and soft | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| two-stage H 0.05, hard | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LieNLSD exact RHS, eval. system | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| LieNLSD as published, training / eval. system | 0.000 / 0.000 | 0.682 / 0.701 | | | |

Verdicts: KS1-KS5, NL8, NL9 met; own count 3 on every run. Pending: the learned arms and the trained baselines on BO
and KS (NL1-NL7, notebook 3).

## Learned operators and published baselines on Benjamin-Ono and Kuramoto-Sivashinsky (notebook 3; registered NL1-NL7; results/bo/, results/ks/)

FNO steppers trained on 1024 trajectories per seed (exp39's recipe), seeds 42-44, clean evaluation data 1000 + s;
baselines trained on the same trajectories with their untrained versions. Missing dimensions, finite algebra /
non-trivial (BO: boost and scaling; KS: boost), means; baselines trained / untrained:

| | Benjamin-Ono (4) | Kuramoto-Sivashinsky (3) |
|---|---|---|
| FINS weak, learned operator | 0.018 / 0.006 | 0.000 / 0.000 |
| FINS pointwise, learned operator | 0.098 / 0.031 | 0.000 / 0.000 |
| FINS weak / pointwise, learned on zero-mean data | -- | 0.039 / 0.042 (non-trivial the same) |
| two-stage, soft, H 0.05 / H 0.2 | 0.047 / 0.656 | 0.000 / 0.000 |
| two-stage, hard | 2.000 | 0.000 |
| LieNLSD as published, training / evaluation system | 1.354 / 1.176 | 0.000 / 0.000 |
| LieGAN | 2.702 / 2.181 (non-trivial 0.983 / 0.867) | 2.587 / 2.054 (0.631 / 0.512) |
| LaLiGAN | 3.862 / 3.999 (1.862 / 1.999) | 2.982 / 3.000 (0.982 / 1.000) |
| Augerino | 0.872 / 0.872 | 0.930 / 0.930 |
| LIG (uses the equation) | -- | 1.562 / 2.063 (0.461 / 0.068) |

Verdicts: NL1, NL2, NL4-NL7 met; NL3 not met: the zero-mean steppers keep the boost (its loss rises 670-1140-fold
but stays 7-17 times below lambda4); they move a constant with 12-21% error against 0.4-0.5% for the main steppers.
Operator errors (held-out increment error): KS 0.003 on every seed, BO 0.009 / 0.021 / 0.013. Cross-device: the six
noisy WN10 runs repeated on the GPU agree with the CPU runs to four decimals in every score. Independent check (own
scorer from the raw npz): every cell confirmed.

## Two-stage baseline on Burgers and with one library for every equation (exp55; registered EX1-EX3; results/burgers/)

Seeds 42-51, missing dimensions of the finite algebra, means. Burgers, clean: H 0.2 0.000 (hard and soft, library to
u_xxx or u_xxxx); H 0.05 0.000 with u_xxxx, 0.300 hard / 0.098 soft with u_xxx (one seed's spurious u u_xxx term).
Library to u_xxxx at H 0.2: heat and KdV as with u_xxx at every noise level (heat 0.300 at 5e-2 and 1e-1 from one seed,
KdV 0.000); Benjamin-Ono hard 2.000 everywhere. Verdicts: EX1-EX3 met. EX4 (FINS weak through the learned heat CNOs):
pending, notebook 4.

## Reference weak SINDy then Lie, like for like (exp56; registered LW1-LW3; EX4 from notebook 4; results/noise_weak/, ks/, bo/, burgers/, exponent/, paired/)

Weak SINDy as published (Messenger and Bortz 2021, ported from the authors' MATLAB code: spectral test functions,
MSTLS), one library d_x^k (u^j), j <= 4, k <= 6, and a constant, for every equation; Lie's algorithm on the identified
equation; seeds 42-51 on the evaluation data, 42-44 on the 1024 training trajectories of the learned operators.
Missing dimensions, finite algebra, means, hard / soft:

| | clean | 1e-3 | 1e-2 | 3e-2 | 5e-2 | 1e-1 |
|---|---|---|---|---|---|---|
| heat | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.322 / 0.247 (1 seed) | 0.000 / 0.000 | 0.400 / 0.082 (1 seed) |
| KdV | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.101 / 0.090 (1 seed) | 0.200 / 0.000 (1 seed) |
| KS | 0.000 / 0.000 | -- | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| Burgers | 0.000 / 0.000 | | | | | |
| Benjamin-Ono | 0.003 (18 spurious) / 2.396 | -- | 0.014 (1.8 spurious) / 0.280 | 0.015 / 0.015 | -- | -- |

Training data (clean, 1e-2; Burgers clean): 0.000 on heat, KdV, KS and Burgers; Benjamin-Ono 0.016 (clean) and 0.015
(1e-2), hard and soft alike (KdV-type surrogates). Benjamin-Ono's surrogates: advection u_t = -5.1 u_x on 9/10 clean
seeds (algebra 22-dimensional in the class), KdV-type -0.06 u_xxx - 0.34 (u^2)_x from 1e-2. Exponent readout (a = 2):
FINS weak 2.004 / 2.003 / 2.003, through the steppers 1.989 (weak) and 2.011 (pointwise); WSINDy's KdV-type surrogates
3.00; its advection surrogates hold their own scaling x d_x + t d_t exactly and the plane's a = 2 and 3 directions to
0.9999. Verdicts: LW1, LW2 met; LW3 not met (the advection surrogates' algebra holds BO's scaling among 22 dimensions);
176 of 180 evaluation runs exact on the equations in the library; EX4 not met in FINS's favour (finite 0.010, class
0.537 against a registered >= 1.0; pointwise through the same CNOs 1.58). Every earlier verdict unchanged.

## Fractional Burgers (exp57, notebook 6; registered FB1-FB5) and operators learned from noisy data (notebook 5; LW4, LW5; results/fburgers/, noise_weak/, bo/, ks/)

u_t + u u_x = -nu Lambda^alpha u on the torus, nu = 3^-alpha, two Gaussian humps per trajectory, 128 x 128 over T = 1;
seeds 42-51 (learned operators and training data 42-44). Time-scaling exponent a-hat, median, and missing dimensions of
the finite algebra (mean) in parentheses:

| | alpha = 1.25 | 1.5 | 1.75 | 2 |
|---|---|---|---|---|
| FINS weak, exact, clean | 1.24 (0.073) | 1.50 (0.017) | 1.75 (0.005) | 2.00 (0.000) |
| FINS weak, exact, 1e-2 | 1.23 (0.432) | 1.50 (0.792) | 1.75 (1.000) | 2.00 (0.002) |
| FINS weak, learned | 1.22 (0.682) | 1.49 (0.583) | 1.74 (0.788) | 2.01 (0.014) |
| WSINDy then Lie, hard, clean | none (2.000) | none on 9, 2.00 on 1 (1.903) | 2.00 (0.004) | 2.00 (0.000) |
| WSINDy then Lie, hard, 1e-2 | none (2.000) | none on 8, 2.00 on 2 (1.61) | 2.00 (0.004) | 2.00 (0.000) |
| oracle (-Lambda^beta u columns), true equation, clean / 1e-2 | 10/10 / 4/10 | 10/10 / 0/10 | 10/10 / 0/10 | 10/10 / 0/10 |

FINS's weak-form exponent is at most 0.033 off on clean data, 0.029 at 1e-2 and 0.037 through the steppers; its own count
misses at most 0.013 on average (0.020 per seed). WSINDy's surrogates: damped Burgers (alpha 1.25, 1.5), viscous Burgers (1.75), exact (2). Verdicts: FB2,
FB3, FB4 met; FB1 not met (missing <= 0.05 on 4/10 at alpha = 1.25); FB5 not met (missing 0.58-0.79 through the
steppers, a-hat within 0.04 on 3/3). Notebook 5, seeds 42-44: heat weak through CNOs retrained on noisy data 0.008 at
1e-3 and 0.182 at 1e-2 (LW4 met); BO at 1e-2 through steppers trained on 1e-2 data weak 0.058 / 0.013, a-hat 1.86-2.17,
pointwise 2.316; KS weak 0.000, pointwise 1.151 (LW5 met). No verdict pending.

## Symmetry-informed weak SINDy (exp58; registered SL1-SL6; results/symlib/; in neither paper yet)

The library of weak SINDy built from the algebra FINS measured, without the equation: gates read from FINS weak's
own-count span with fields built from the data (linear: u d_u and an x-t scaling inside; shift d_u; boost t d_x + c d_u,
which also measures the advection coefficient; scaling with exponent a-hat), the library cut to the terms invariant
under the gates and completed on periodic data by the translation-invariant linear operators homogeneous of degree
a-hat (symbols -|kappa|^a-hat and i sgn(kappa) |kappa|^a-hat); MSTLS fits the constants. Baselines: the reference
weak SINDy (ref); the same structure assumed without FINS, order chosen by the residual (prof); the reference library
completed by the two multipliers at a residual-chosen order (prof-free). Seeds 42-51, the evaluation data. True
equation:

| | ref | sym | prof | prof-free |
|---|---|---|---|---|
| heat, 6 conditions | 58/60 | 60/60 | | |
| KdV, clean to 5e-2 | 49/50 | 50/50 | 46/50 | 41/50 |
| KdV 1e-1 | 9/10 | 3/10 | 5/10 | 5/10 |
| KS, 5 conditions | 50/50 | 50/50 | 0/50 | 47/50 |
| Benjamin-Ono, 3 conditions | 0/30 | 30/30 | 30/30 | 30/30 |
| fractional Burgers 1.25-1.75, clean and 1e-2 | 0/60 | 60/60 | 60/60 | 58/60 |
| fractional Burgers 2, clean and 1e-2 | 20/20 | 20/20 | 20/20 | 20/20 |

Out of the library the completion without FINS is the more precise (order on its grid point, coefficients to 1e-6 or
better on clean data; FINS's exponent 0.002-0.018 off); under noise its extra columns break in-library identification
(KdV, KS). sym repairs three of the reference's four in-library failures (heat 3e-2 seed 49, heat 1e-1 seed 44, KdV 5e-2
seed 48) and loses where FINS's exponent drifts (KdV 1e-1: a-hat 0.20-0.54 above 3 on seven seeds). Verdicts: SL1, SL3,
SL4, SL6 met; SL2 not met (prof-free 8/10 at fb175 1e-2), SL5 not met (prof-free 7/10 on KS at 1e-1). Consequence for
the PDE paper: the oracle of exp57 failed at 1e-2 because of its grid of six beta columns; one free order chosen by the
residual recovers Benjamin-Ono and fractional Burgers.

## Anisotropic nonlocal dissipation in 2D (exp60; registered AN1-AN5, AR1-AR2; exp67: CT1-CT2; exp66: LS1-LS2; results/aniso2d/; in neither paper yet)

u_t = div_A(u grad_A u) - nu (-div_A grad_A)^(a/2) u, grad_A = (d_x, eps d_y), eps = 0.6, nu = 0.3, a = 1.25, 1.5,
1.75, on exp45's box. Algebra: d_x, d_y, d_t, the stretched rotation -y / eps d_x + eps x d_y, the scaling
x d_x + y d_y + a t d_t + (2 - a) u d_u. FINS: exp45's polynomial class through the exact tangent; the readouts are the
pencil scans over the stretched rotations (eps) and the scalings (a, c). Weak SINDy in 2D (local library d_x^p d_y^q
(u^j), j <= 3, p + q <= 4) is run alone (ref) and completed by one nonlocal family each, with the residual choosing the
parameters: isotropic -|k|^beta, directional -|k_x|^beta, -|k_y|^beta, harmonics -|k|^beta (1, cos 2 theta,
cos 4 theta), elliptic -(k_x^2 + e^2 k_y^2)^(beta/2). Nested: the next family only when the residual drops more than
1.25 times. Seeds 42-51.

| a | FINS eps-hat / a-hat | ref | iso-free | dir-free | harm-free | ell-free | nested |
|---|---|---|---|---|---|---|---|
| 1.25 | 0.600 / 1.255-1.257 | 0/10, a-hat 2 (8), none (2) | 0/10, no exponent | 0/10, 1.25-1.30 | 0/10, 1.250 | 10/10 | 10/10 |
| 1.5 | 0.600 / 1.500-1.505 | 0/10, 2 | 0/10, none | 0/10, 1.50-1.55 | 0/10, 1.500 | 10/10 | 10/10 |
| 1.75 | 0.600 / 1.747-1.750 | 0/10, 2 | 0/10, none | 0/10, 1.75 or none | 0/10, 1.750 | 10/10 | 10/10 |

Counts are true equations; "a-hat" is the identified model's exponent by homogeneity. By the 1e-6 ellipse test, no
completion other than the elliptic one has the stretched rotation; the elliptic one has it on 29/30 runs. The one miss (a = 1.25, seed 45) is the
structural test's 1e-6 tolerance: the fitted c_yy / c_xx is 3.3e-6 (relative) off 0.36, which varies the
multiplier's symbol by 1.04e-6 on the test ellipses.

The registered scan's refinement (+-0.01 around the coarse minimum) clips four runs at a = 1.25 (c-hat on the box edge).
The family's exact minimum (3 x 3 generalized eigenproblem) is within 0.011, 0.004 and 0.002 of a, against 0.007 for
the registered readout. AN1 holds either way.

exp60's own-count rule (floor 1e-12) cuts at 2-3 of the 5 dimensions here: its largest gap lies between the exact
translations and d_t, or between d_t and the next eigenvalue. exp49's rule (floor 1e-6; post hoc) gives 5 at a = 1.25
and 6 at a = 1.5 and 1.75. The span of the 5 lowest contains the algebra (missing at most 0.044) and reads eps to
0.002; its exponent is low by median 0.10-0.16 (up to 0.47 at a = 1.75).

Verdicts: AN1-AN5 met. AN5 is FINS through learned FNO2d steppers (notebook 7, seeds 42-44, one-step error 0.2 %):
eps-hat 0.600 on 9/9 and a-hat within 0.024, by the scans. Through the FNOs the blind readouts are weaker: the 5-span
loses d_t at a = 1.75 (missing 0.987-0.995), lambda6 / lambda5 is 1.1-3.6 (exact 15-157), and the 6-span reads a within
0.044.

Reading: FINS's scans over fixed generator families (a Lie-derivative family search over the stretched rotations and
over scalings without d_u, both containing the truth) give eps and the exponent without an operator family. Among the
completions, the harmonic and elliptic families recover the exponent exactly; the directional family is up to 0.05 off,
or has no exponent on 5 runs at a = 1.75; the isotropic single order never has one. Only the family containing the
truth (the elliptic one) recovers the equation, and the residual alone selects it once it is offered.

Corrections from the mock reviews of 09.10, checked on the saved runs:
- The sixth near-symmetry is t d_t - (u + nu) d_u, the second scaling of the a = 2 member (the anisotropic
  porous-medium equation in u + nu): quotient 2.3e-6 to 3.0e-5 (lambda6 1.5e-6 to 2.3e-5; seeds 42-43), against
  5.5e-4 to 7.9e-4 for t d_t - u d_u. Its defect vanishes as a -> 2.
- The scaling scan's family has no d_u and so excludes it. With the shift admitted the ratio of the two lowest
  restricted eigenvalues falls from 429-867 to 32-100 (lowest at a = 1.75), and a-hat reads 1.267-1.270, 1.506-1.510,
  1.737-1.746 (seeds 42-44).
- The rotation is scored asymmetrically: the identified models by a 1e-6 ellipse test, FINS by an argmin. Read by the
  same argmin, the harmonic completion gives e 0.602-0.606, and its nonlinear part 0.5988-0.6000. The reference
  models' "no rotation" is untested on 20/30 runs (nonlinear terms outside the tested pattern).
- The profile grids contain the true parameters, without refinement.

Readouts without the scans' priors (exp60_readouts.py on the 30 saved exact pencils; registered AR1-AR2): the pencil
restricted to the traceless linear fields (x d_x - y d_y, y d_x, x d_y), with no aligned axes, has an elliptic deepest
field on 30/30. Its invariant form gives eps-hat 0.59983-0.59998 and the axes' angle within 0.0013 degrees of 0;
lambda2 / lambda1 is 8.4e4 to 1.8e6. Verdicts: AR1, AR2 met (30/30). The aligned-axes prior of the rotation scan is not
needed; the family is still one of linear fields, read through the exact solver. Read by the same soft argmin, weak
SINDy's harmonic completion gives e within 0.006 and its nonlinear part within 0.0012 (not registered).

The linear response of the operator (exp67; deterministic, no data, no operator family): the one-snapshot stepper's
Jacobian at constant states b is a Fourier multiplier exp(dt sigma(k; b)), sigma = sigma0 - b S - b^2 S2. From sigma0
it reads the dissipation's eps 0.600 and order 1.500, from S the nonlinearity's eps 0.600 (computed at a = 1.5). There
FINS's exponent readout adds nothing to it. Controls, where the two differ (exp67, a = 1.5, seeds 42-51, FINS through
the exact solver):
- eps 0.8 in the dissipation, 0.6 in the nonlinearity: the linear response reads 0.800 and 0.600; FINS finds no
  stretched rotation of the whole equation (the best, eps 0.646-0.653, at 51-133 times the kept generators) and keeps the
  scaling (1.500-1.504, 0.494-0.500). CT1 met (10/10).
- A cubic term, u_t = A(u^2 / 2 + 0.2 u^3) - nu (-A)^(a/2) u: the linear response keeps the order 1.5 and flags a b^2
  term (0.21); FINS reads the stretched rotation (0.600) and breaks the scaling (1.5, 0.5) at 76-359 times the kept
  generators, while its best scaling, (a, c) = (1.595-1.620, 0.263-0.292), sits 1.7-4.5 times above the stretched
  rotation (below 1e-6 on 5/10): an approximate scaling of the data that breaks the dissipation's weight as well. CT2
  not met (the 100-times margin on 8/10; the exponent and eps parts 10/10).

LieNLSD as published (exp66, notebook 9; an MLP on jets to second order fitted to u_t, held-out relative error
0.014-0.043; exp60's 16 training trajectories, seeds 42-51): its scaling scan reads a-hat 1.341-1.745, 1.587-1.871,
1.627-1.930 at a = 1.25, 1.5, 1.75 (medians 1.630, 1.722, 1.789), and the span of its five smallest directions misses a
median 1.200, 0.848, 0.318 of the algebra. It reads eps 0.598-0.615 (scan and sl(2)); in exp60's equation eps enters the
local nonlinearity too, the exponent only the nonlocal term. LS1 (a-hat more than 0.1 off) and LS2 (missing >= 0.5) hold
at a = 1.25 and 1.5 and fail at 1.75 (4/10, 2/10): not met.

## Quasi-geostrophic flow from pyqg (exp61; registered PG1-PG8, WS1-WS5; exp64: QL1-QL4; exp65: EA1-EA3; results/pyqg/; in neither paper yet)

Data: pyqg's barotropic model itself (AB3, GitHub master), 2-3 shielded vortices on [-8, 8)^2 at 128^2, unforced
(not the ML benchmark's two-layer eddy and jet configurations, where only the translations survive). Operator: an RK4
port of pyqg's spatial discretization and filter, checked against pyqg (2-6e-5 at T = 2.5 on 128^2), with its exact
tangent. FINS: exp45's polynomial class in q (the potential vorticity) or psi (the streamfunction). Regimes: inviscid
f-plane, beta = 0.5, rd = 1, bottom drag 0.3. Seeds 42-51.

| regime | algebra found (own count = k; missing) | broken generators (smallest quotient) | kept (largest) | surviving scalings |
|---|---|---|---|---|
| f | 7 of 7 (9/10; 0.000) | none | 2.8e-5 (D_x) | x d_x + y d_y and t d_t - q d_q |
| beta | 5 of 5 (10/10; 0.000) | rotation, D_x, D_t (4.3e-3) | 2.0e-5 | x d_x + y d_y - t d_t + q d_q (weights within 0.004) |
| rd | 6 of 6 (10/10; 0.000) | D_x, D_beta (2.3e-3) | 4.3e-7 | t d_t - q d_q |
| drag | 6 of 6 (10/10; 0.000) | the polynomial D_t, D_beta (4.5e-3) | 2.8e-7 | x d_x + y d_y; V_r outside the class |

The counts include the shift d_q, which the code gives (its pencil row is zero for any data): a convention of the
periodic inversion without rd, a symmetry with rd. The measured part is 6 / 4 / 5 / 5. At drag the time scaling is
deformed, not removed: V_r = ((e^(rt) - 1) / r) d_t - e^(rt) q d_q is exact (2.6e-10 through the port in the registered
pencil's normalisation, seed 42, against 9.9e-2 for t d_t - q d_q) and lies outside the degree-2 class (exp62 recovers it in a class with tau(t) and the q-coefficient to degree 6 in t). Which parameter breaks
which generator follows from
dimensional analysis and from Lie's algorithm on the known equation, and the surviving weights are integers: the table
validates the readout (ceiling rows).

Secondary rows:
- At 64^2 the grid scale, pyqg's filter included, breaks the space scaling (2.5e-3 to 1.2e-2).
- In the streamfunction (pyqg's zero-mean gauge), the gauge breaks the space scaling (0.58-0.82), while the time
  scaling (at most 1e-10) and the rotation (at most 9e-5) survive; with rd the screened streamfunction keeps all six
  generators.

Weak SINDy on the same data:
- The local library never returns the equation: residual 0.58-0.955 at f, beta and rd; 0.19-0.27 at drag, where it
  finds the drag term alone on 9/10 runs.
- Completed by the quasi-geostrophic terms (Biot-Savart transport, the beta multiplier, the screening profiled), it
  returns the equation on 40/40 runs with coefficients within 2.1e-5, and the nested rule picks that completion.
- Its algebras match FINS's on the zero-mean states. At drag the fitted -0.30 q acts on the mean of q and pyqg's
  drag does not; the data cannot tell these apart, and FINS reads the difference from the code's k = 0 convention. A
  gauge remark, not an advantage.
- Without quasi-geostrophic knowledge (exp64): completed by the generalized-SQG family of transports,
  u_s = grad^perp psi_s with psi_s = -(|k|^2 + kappa)^(-s/2) q, exponent and screening profiled by the residual (exp61's
  completion fixed s = 2), it returns the equation and the algebra on 40/40 runs (s within 0.0008 of 2, kappa 1.0016
  at rd, coefficients within 9.1e-4). The nested rule reaches it in every regime (at f, beta and drag through the row
  with kappa fixed at 0, identical there); with kappa fixed at 0 it misses rd on 10/10. From pyqg's own velocity with
  the reference library plus u, v and the derivatives of u q, v q, u^2, v^2, u v, the equation and the algebra on
  36/40 runs by the registered scorer; the four misses are the same equation in the stress form (two) or mixed forms
  (two), by identities of incompressible flow (the equation on 40/40 rescored by function, post hoc). Verdicts: QL2-QL4
  met; QL1 not met (7/10 at f, 9/10 at rd).

Learned rows (notebook 8; FNO2d on 256 pyqg trajectories, 40 epochs; seeds 42-44): one-step errors 9.4-10.4 % (f) and
4.2-5.9 % (beta) of the data increment, 17-53 times exp60's. Through them every kept candidate sits at 1.3e-3 to
1.7e-2. At beta the broken candidates keep their exact levels (1.08-1.28 times the port's) and stay above beta's kept
ones (the shift aside), but the f-plane rotation is at 3.1e-3 to 4.2e-3 itself, so the rotation contrast beta / f is
2.6-3.7; beta's surviving scaling has its time weight within 0.1 and its q weight 0.16-0.56 off; the f-plane's
two-dimensional scaling space is kept (3/3).

The same emulators audited on their own rollouts (exp65, notebook 9), so that the pencil carries no data term: the
translations fall to 4.3e-5 to 1.8e-4 (PG8 1.3e-3 to 5.0e-3), d_t to 1.5e-10 to 1.4e-9, and f's rotation, D_x, D_t and
beta's D_beta by factors 1.07-2.31, to 9.1e-4 to 7.4e-3; the shift stays at 6.0e-4 to 4.4e-2. The rotation contrast is 4.1-5.7 (EA1 not met: 0/3) and beta's surviving scaling has its q weight 0.175-0.452
off (EA2 not met: 0/3): the emulators did not learn the pattern. Their one-step errors after finite flows of the
generators, over the untransformed error, correlate with the self-consistent defects at Spearman 0.604 over 24 pairs
(EA3 met), all of it from the translations (exact on the grid, ratio 1): without them -0.34 (post hoc).

Verdicts: PG1-PG7, WS1-WS5 met; PG8 not met (rotation contrast 0/3, beta direction 0/3, f scaling span 3/3); EA3 met,
EA1-EA2 not met.

Reading: through the exact tangent of a port of pyqg's code (ceiling rows), FINS reads from pyqg's data which physical
parameter breaks which generator of the degree-2 class, a validation against dimensional analysis and Lie's algorithm.
The equation-learning route needs the nonlocal transport in its library, either as pyqg's velocity or as the
generalized-SQG family with its exponent free (exp64); the quasi-geostrophic member need not be supplied. It returns the
equation, the same algebra, beta and kd2, which FINS does not (FINS reads rek in exp62's tau(t) class). Through FNO2d
emulators at 4-10 % one-step error the pattern is not kept: their own defects in the rotation and the scalings, read on
their own rollouts at 9.1e-4 to 7.4e-3, mask the contrast of the beta-plane's rotation breaking with the f-plane. exp61 has no learned-operator
result.

## The drag's deformed time scaling (exp62; registered DT1-DT5; results/drag_tau/; in neither paper yet)

exp61's drag regime at rek 0.1, 0.3, 0.6 and the f-plane as the control; pyqg's data at 128^2 with exp61's settings,
the port's exact tangent. Class: exp45's 60 fields plus tau T^3..T^6 and phi T^2 U..T^6 U (69 fields): tau(t) and the
coefficient of q to degree 6 in t. Algebra in this class: d_x, d_y, d_t, the shift, the rotation, D_x and the degree-6
truncation of V_r = ((e^(rt) - 1) / r) d_t - e^(rt) q d_q (D_t on the f-plane). The drag rate is read from the pencil
restricted to the 14 time fields: its two deepest eigenvectors combined to tau(0) = 0 and phi = -q at t = 0, r-hat =
-(phi's T U coefficient) / T_max. Seeds 42-51.

| row | own count = 7 | missing (span 7) | r-hat | V_r exact | D_t | 8th eigenvalue |
|---|---|---|---|---|---|---|
| drag 0.1 | 10/10 | at most 2.8e-6 | 0.09995-0.09997 | 9.1e-10 to 4.3e-8 | 3.0e-2 to 4.7e-2 | 1.6e-3 to 2.6e-3 |
| drag 0.3 | 10/10 | at most 7.8e-6 | 0.29967-0.29969 | 2.6e-10 to 3.5e-9 | 9.1e-2 to 9.9e-2 | 3.1e-4 to 8.1e-4 |
| drag 0.6 | 10/10 | at most 2.6e-5 | 0.59833-0.59836 | 9.1e-10 to 1.1e-9 | 0.122-0.125 | 4.5e-5 to 1.3e-4 |
| f | 9/10 | at most 1.2e-5 | -0.00002 to -0.00001 | V_0.3: 0.25-0.43 | 2.3e-9 to 1.1e-7 | 3.3e-4 to 5.8e-4 |

The time pencil's eigenvalues are 3.8e-12 to 1.1e-7, 2.9e-7 to 5.4e-7 and 0.47-0.85 over the 40 runs. The f-plane's
miss (seed 51) is the space scaling at 2.8e-5, on the data seed of exp61's f-plane maximum. r-hat is low by 3.5e-5 to
4.7e-5, 3.2e-4 to 3.4e-4 and 1.64e-3 to 1.67e-3; the class is not the cause: the exact V_r's projection on the time
fields keeps its quotient and reads r to 1e-5, and the class's deepest direction sits below the exact V_r's quotient
(a residual of V_r on these data). r-hat from tau's coefficients is 2.2-4.5 % high.

Verdicts: DT1-DT5 met.

Reading: in a class with tau(t) and the q-coefficient to degree 6, FINS recovers from pyqg's data the drag's time
scaling deformed into V_r and reads the drag rate from the generator's slope, with no exponential candidate. Ceiling
rows (exact tangent); V_r follows from the point equivalence of damped and inviscid flows (q = e^(-rt) w,
s = (1 - e^(-rt)) / r) and satisfies the known equation's linearized symmetry condition (exp62_vr_check.py).

## FINS for systems: pyqg's flow in the velocity (exp63; registered UV1-UV5; results/velocity/; in neither paper yet)

State (u, v) = grad^perp psi of pyqg's data (exp61's settings with circular shielded vortices); operator
(u, v) -> q -> the port's step -> (u, v) with its exact tangent; class: monomials of degree <= 2 in (X, Y, T, U, V) for
each of xi, eta, tau, phi_u, phi_v (105 fields). Two operators on the same data: pyqg's (its velocity has zero mean) and
meanflow (a conserved mean flow that advects q; with rd also the PV gradient it implies). Seeds 42-51 (meanflow 42-46).

| regime | algebra in the span of k (missing) | rotation space: deepest; spin / orbital | scaling space | boost space (t d_x, d_u, t d_y, d_v): two lowest |
|---|---|---|---|---|
| f | 10/10, 5/5 (at most 2.7e-4) | 2.2e-5 to 8.8e-5; 0.99999-1.00000 | D_t and D_x (to 1.8e-7) | pyqg 0.91-0.97; meanflow 2.0e-13 to 4.2e-11, speeds within 1.4e-7 of 1 |
| beta | meanflow 5/5 (at most 1.5e-3); pyqg 0/10 | broken (3.5e-2 to 5.8e-2) | D_beta, velocity weight 2.06-2.13 (meanflow), 2.20-2.39 (pyqg) | pyqg 0.72-0.82; meanflow 8.4e-14 to 1.9e-11, speeds within 1.4e-7 of 1 |
| rd | 10/10, 5/5 (at most 1.3e-4) | 6.1e-6 to 2.2e-5; 0.99999-1.00000 | D_t | pyqg 0.95-0.99; meanflow 1.1e-3 to 2.8e-3 |
| drag | 10/10 (at most 7.0e-6) | 4.2e-7 to 2.4e-6; 1.00000 | D_x | pyqg 0.74-0.76 |

D_x = x d_x + y d_y + u d_u + v d_v, D_t = t d_t - u d_u - v d_v, D_beta = x d_x + y d_y - t d_t + 2u d_u + 2v d_v. The
scalar rotation, the spin alone and the opposite spin sit 1.1e4 to 1.6e6 times above the vector rotation at f, rd and
drag (the smallest of the three). Through pyqg's operator the boost space's lowest eigenvalue sits 354 to 1.8e6 times
above the algebra's largest quotient (no boost direction: its speeds are near 0); with rd the mean flow's PV gradient
lifts it to 104-184 times. At beta, pyqg's operator drops the mean of D_beta's characteristic (its quotient 2.4-3.5
times the meanflow operator's on the same seeds), and the span misses the algebra; the meanflow operator keeps it. The
blind own count (floor 1e-6) reaches the dimension through pyqg's operator at rd and drag (10/10) and at f on 3/10 (the
rotation and D_x above the floor); with the mean flow at f on 1/5 and rd on 4/5; at beta on none.

Verdicts: UV1-UV5 met.

Reading: FINS for systems recovers pyqg's velocity algebra with the rotation acting on the components as a vector and
the scalings' velocity weights (at f, rd and drag through pyqg's operator, at beta only through the meanflow one), and
reads the Galilean boosts as a property of the operator's state space: broken by
pyqg's zero-mean velocity, exact with a conserved mean flow, broken again by the PV gradient that a mean flow implies
with rd. Ceiling rows (exact tangent); the velocity algebra is the image of q's under a linear map, except the boosts,
which q does not see.

## Emulators of pyqg with a known answer (exp68, notebook 10; registered EP0-EP5; results/pyqg/pairs_summary.*; in the PDE paper since 12.10: EP3-EP4 in Section 6 and Fig. 2, all EP in Table 19)

FNO2d emulators of pyqg at 128^2 with PG8's two limits removed (40 modes, 1024 training trajectories, 10000 steps): f
plain, f rot (rotation augmentation), f grid (coordinate channels), beta plain; seeds 42-44 (f rot s44 lost when Drive
ran out of space and retrained in a second session, notebook 10 rerun). Audits: held-out one-step error, commutators with finite rotations and shifts, FINS on the
emulator's own rollouts (exp65).

| arm | held-out one-step error | 30-degree commutator | FINS rotation plane (/ 0.35 e^2) | result |
|---|---|---|---|---|
| f plain | 0.0340-0.0402 | 4.7e-2 to 5.2e-2 | 4.6e-4 to 7.4e-4 (1.14-1.43) | EP0 not met (above 0.03) |
| f rot | 0.0189-0.0204 | 2.5e-2 to 2.7e-2 | 1.6e-4 to 1.9e-4 (1.19-1.35) | lower defect through accuracy: commutator 1.88-1.95 and rotation plane 2.90-3.95 times lower than plain's |
| f grid | 0.0269-0.0412 | 3.5e-2 to 5.2e-2 | 2.5e-4 to 5.3e-4 (0.90-1.15) | 32-cell shifts broken (4.0e-2 to 8.0e-2), unseen by FINS and the test error |
| beta plain | 0.0161-0.0257 | 0.43-0.48 (port 0.44-0.49) | 6.7e-3 to 1.4e-2 | contrast with f 14.5-19.1; D_beta's weights within 0.083 (q weight 1.064-1.083, time weight within 0.013) |

Verdicts: EP0 not met; EP1-EP5 met.

Reading: with PG8's two limits removed (held-out one-step errors 2.1-2.7 times lower at f, 1.8-2.2 at beta, on the same
trajectories), FINS on the emulators' own rollouts reads which parameter breaks the rotation (beta's rotation plane
14.5-19.1 times f's, against 4.1-5.7 for PG8's emulators) and beta's surviving scaling, though the f-plane plain
emulators stay above the 3 % target (3.4-4.0 %). The rotation defect follows accuracy on all nine f-plane emulators
(0.90-1.43 times 0.35 e^2), and rotation augmentation lowers it through accuracy (the plane ratio over the squared error
ratio 0.90-1.19): read on the rotation, the audit measures accuracy rather than a separately learned symmetry. The grid channels break finite translations that the
local readout does not see.

## A tracer in a hidden flow: a variable without a closed equation (exp69; registered TR0-TR8; results/tracer/; in the PDE paper since 12.10: Section 6, Table 7, Appendix E, Table 18)

exp61's f-plane flow carrying a passive tracer, c_t + J(psi, c) = kappa Lap c (kappa 0.05); only c is observed; 128^2.
The observed algebra (generators with a point lift): d_x, d_y, d_t, rotation, d_c, c d_c and S = x d_x + y d_y + 2t d_t,
whose lift scales the hidden q (-2q d_q). A heat-type closure c_t = D Lap c has seven more in the class (the boosts and
the superpositions x, y, xy, x^2 - y^2, x^2 + y^2 + 4Dt), which the hidden flow breaks; H (10 fields) spans them for
every D. Arms: the oracle (the full system's exact tangent, c observed; exp45's 60 fields in c lifted to q as scalars,
and q d_q); the two-stage route (weak SINDy on c, then the identified equation's determining equations in 2D, exact
rationals); window emulators of c (notebooks 11 and 11b; self and data arms); the best hidden lift (the exact dynamics,
a hidden perturbation fitted by CGLS).

| arm | seeds | observed algebra and S, largest | the heat model's seven, smallest | result |
|---|---|---|---|---|
| oracle | 42-51 | 2.9e-8 to 8.6e-8 (rotation) | 1.9e-4 to 3.3e-4 (sup_r2) | own count 7 on 10/10, span missing at most 1.0e-5; S's lift -2.00038 to -1.99980, unlifted 2.1e5 to 6.9e5 times |
| best hidden lift (100 CGLS iterations) | 42-44 | at most 7.2e-12 (rotation) | 3.2e-5 (sup_r2); boosts 3.4e-3 to 2.4e-2 | the hidden perturbation found keeps 0.28-0.52 of the boosts' quotients and 0.14-0.30 of the superpositions' (upper bounds: still falling slowly) |
| window emulators of c, data arm (notebooks 11, 11b) | 42-44 | 4.8e-4 to 8.0e-3 (d_t; d_x, d_y, rotation, d_c 1.2e-4 to 2.5e-3) | 3.1e-5 (boost_y, s43); boosts 0.005-0.033 of the oracle's | no separation: H's lowest below the observed algebra's d_x, d_y, rotation and d_c on every seed; one-step error 0.0035-0.0092; own rollouts divergent (43, 44), and d_t fails on 42's own rollouts, so no self audit |
| two-stage | 42-51 | - | - | diffusion on 7/10 (a, b 0.040-0.056; algebra 14, including H's seven and a stretched rotation; the rotation lost); spurious terms on 3/10 (algebra 3-5; d_c, c d_c and the rotation lost) |

Verdicts: TR0 met; TR1-TR3 met; TR4 not met (7/10); TR5 met; TR6 met; TR7 not met (0/3); TR8 met.

Reading: through a simulator, FINS reads the algebra of a variable that has no closed equation, with the hidden
variable's weight (S's -2q d_q to 4e-4); a two-stage route must close the variable, and weak SINDy's closure (diffusion,
or diffusion with spurious nonlinear terms) has a wrong exact algebra on 10/10. The breaking is in c: the best hidden
perturbation found (100 CGLS iterations, still falling slowly) leaves 6.6 decades between the heat model's seven and the
observed algebra across seeds (7.0-7.2 within each). Development (seed 7, 64^2): a window emulator of c with 1.85 %
one-step error puts the observed algebra and H at its error floor (no separation). At 128^2 (notebooks 11 and 11b)
emulators accurate to 0.35-0.92 % per step separate nothing either: they break the superpositions x and y about as much
as the flow does but keep the boosts (0.005-0.033 of the oracle's breaking, while the best hidden perturbation found
leaves 0.28-0.52 in c); two of the three diverge on their own within 15-25 steps.

## Learned operators in place of the simulator for the tracer (exp70; registered UF1-UF2, HL0-HL5, LT0-LT5; results/tracer/exp70_summary.*; in the PDE paper since 12.10: Section 6, Fig. 2, Tables 18-19)

exp69's system and readouts. Step 1: the best hidden lift with a uniform flow in the hidden perturbation (the exact
dynamics; does the breaking that c carries come from the hidden state's missing mean flow?), and an emulator of the full
state (q, c) trained on data with q observed in training only (StateFNO2d, exp69's recipe; FINS through its tangent with
the oracle's lift; notebook 12). Step 2 (registered LT0-LT5; notebook 13): a model of c alone with a
learned hidden field z (encoder of four snapshots, stepper of (z, c)), read with z lifted as a point field with a fitted
weight (the plane readout) and with a free perturbation of z.

| arm | seeds | result |
|---|---|---|
| uniform flow (exact dynamics, 4 pairs, 100 CGLS iterations) | 42-44 | the Galilean t d_x and t d_y become exact with the flow (at most 1.0e-16, coefficients 1 within 2.9e-9 on 12 pairs); every heat-false generator keeps 0.965-1.000 of TR8's value without the flow; the observed algebra and S at most 6.7e-12 |
| state emulator of (q, c), q in training only (notebook 12; one-step error of c 3.1-5.1 %, q 3.9-5.5 %) | 42-44 | the boosts at 0.97-1.07 of the oracle's and 30-235 times the window emulators' (data and self), 6-13 times above the observed algebra and S (both at least 10 times only on 44's self arm); S's hidden lift -2.00 to -2.07; the superpositions not read (the oracle's at 0.14-1.4 times the floor; the emulator's 1.5-14 times it, inflated over the oracle's); on its own rollouts the 7 lowest directions miss 0.02-0.06 of the observed algebra with S (post hoc); a free hidden perturbation lowers every quotient two to three orders and keeps the seven 1.75-2.30 times above |
| latent model of c alone, z lifted as a point field with a fitted weight (notebook 13; one-step error of c 1.3-2.2 %) | 42-44 | the boosts 19-26 times above the observed algebra and S on 42 and 43 (44: 9.2-13.0), 90-796 times the window emulators' (2.8-3.9 times the oracle's: over-broken); the superpositions x, y, xy, x^2 - y^2 below the floor (0.09-0.96 of it), x^2 + y^2 + 4Dt 1.0-2.2 times it; c d_c holds with z's weight +0.99 to +1.02; own rollouts within 0.038; a free perturbation of z absorbs the seven (to at most 0.036-0.064 of the largest observed) |

Verdicts: UF1 met; UF2 met; HL0 not met (2/3); HL1 not met (0/3); HL2 met; HL3 met; HL4 met; HL5 not met (1/3);
LT0 not met (2/3); LT1-LT5 met.

Reading: the breaking of the heat closure's seven that c carries is not the hidden state's missing mean flow (a global
constraint); it lies in the flow's structure. With q observed in training, a learned operator of the full state reproduces the simulator's
breaking of the boosts and S's hidden weight; at 3-5.5 % one-step error its floor keeps the boosts 6-13 times above the
observed algebra and S (both at least 10 times only on 44's self arm) and hides the superpositions. From c alone, a model
with a learned hidden field, read with the field lifted as a point field with a fitted weight, separates the heat
closure's boosts on 2/3 seeds (19-26 times) where window emulators keep them; it over-breaks them (2.8-3.9 times the
oracle's) and does not read the superpositions. A free perturbation of the learned field absorbs all seven, so the
readout must keep the lift's point structure.

Post hoc (12.10, mock reviewers' re-analyses of these summaries, reproduced): against the 8-step error the 18 independent
models of exp68-exp70 lie within 0.30-2.2 times 0.35 e_K^2 (one-step: 0.73-201; pooled exponent against e_K 1.2), so the
c-only models' excess over 0.35 e_1^2 comes from the one-step abscissa (their errors grow 2.3-17 times over K = 8, the
full-state emulators' 0.77-1.02 times). The latent separation divided by its over-breaking is 3.3-7.0, and with z
unweighted it vanishes (0.78-1.18); no control on closed data (a tracer without flow) was run.

## Kreider, Harlim and Huang as a baseline (exp71; registered KR1-KR6; results/noise_weak/, ks/, bo/, burgers/, kreider_verdicts.txt; MAKE paper: Tables 3-5, rows from exp71_rows.py)

Reimplementation of arXiv 2511.09779 (pde/method_kreider.py; rows equal the authors' local_row to 1e-16, their Fig 3c and
3d reproduced) in two readings on the protocol's evaluation data: the symmetries the sampled solutions share (as
published) and the equation surface estimated locally from the pooled jets. Seeds 42-51, the noise studies' levels,
soft readout (m = k), finite algebra:

| equation | solutions, clean | surface, clean | surface under noise (medians) | FINS weak exact / learned | LieNLSD as published |
|---|---|---|---|---|---|
| heat (6) | 3.624 +/- 0.311 | 1.712 +/- 0.242 | 2.59-2.91 (1e-3 to 1e-1) | 0.000 / 0.010 | 0.55 |
| Burgers (5) | 3.525 +/- 0.217 | 1.311 +/- 0.203 | -- | -- | 0.55 |
| KdV (4) | 3.090 +/- 0.160 | 1.658 +/- 0.119 | 1.45-1.94 | 0.000 / 0.004 | 0.91 |
| BO (4) | 2.818 +/- 0.375 | 1.715 +/- 0.239 | 1.76 | 0.000 / 0.018 | 1.354 |
| KS (3) | 2.012 +/- 0.129 | 0.919 +/- 0.040 | 0.51-0.99 | 0.000 / 0.000 | 0.00 |

Verdicts: KR1-KR6 met (KR5's narrowest margin: KS at 1e-2, median 0.507 against 0.5). Independent check: every score
recomputed from the raw files and the verdicts re-derived with separate code.

Reading: as published the method answers a different question (the sampled solutions' symmetries); the locally
estimated equation surface misses about one to two non-trivial dimensions where FINS misses none.

## Symmetry as augmentation (exp72; registered UT1-UT7; results/utility/; notebook 14; MAKE paper: Section 5.8, Table 7 from exp72_table.py)

The same stepper trained on data with backgrounds within +-0.5, its training windows moved by the part of a span made of
translations and background shifts (u(x) -> u(x - s xi(t)) + s phi(t)): FINS's spans of the learned operators, the true
algebra, LieNLSD's, a constant added without its drift; reference: real data with backgrounds within +-3. Scored on the
exact Galilean boosts of the evaluation trajectories to backgrounds +-1, +-2. Development (seed 7, KS, 256 x 150; OOD
16-step error): none 0.328, oracle 0.026, real data 0.023, an imprecise closed-loop FINS span 0.086, the shift 1.59.

Seeds 42-44, 16-step errors, OOD the mean over the four boosted backgrounds (range over seeds):

| arm | KS OOD | KdV OOD | BO OOD |
|---|---|---|---|
| none | 0.217-0.269 | 0.011-0.019 | 0.017-0.038 |
| fins (FINS's learned-operator spans) | 0.0093-0.0108 | 0.0012-0.0071 | 0.0047-0.0068 (42, 44; 43 has no part) |
| oracle | 0.0083-0.0096 | 0.0009-0.0026 | 0.0021-0.0054 |
| lienlsd | 0.0117-0.0156 | as none | as none |
| shift (constant without its drift) | 1.58-1.60 | 0.099-0.100 | 0.119-0.124 |
| wide (real backgrounds within +-3) | 0.0091-0.0102 | 0.0008-0.0029 | 0.0019-0.0061 |

Verdicts: UT1, UT2, UT3, UT5, UT6 met; UT4 not met (5/6: KdV 43, fins 2.71 times the oracle); UT7 not met (7/9: KS,
lienlsd 1.64 and 1.59 times the oracle on 42 and 43). Independent check: verdicts re-derived with separate code, S and
the run metadata checked against the registration.

Reading: used as augmentation, FINS's spans carry an emulator to backgrounds two to four times outside its data (OOD
error 22-25 times lower on KS, 3-10 on KdV, 3-4 on BO), within 1.08-1.36 times the true algebra (1.02-1.48 times
real data) except where the span is less precise (KdV 43, BO); a constant added without its drift makes the emulator
worse.

Accuracy of the spans (exp72_table.py; post hoc): through the protocol's steppers the parts of FINS's spans lie within a
principal angle of 0.0031-0.0205 of the true translation and boost (missing 1.0e-5 to 4.2e-4), yet their boosts move at
0.854-1.121 of the true speed over the horizon (BO 42 0.854-1.027, KdV 43 0.973-1.121; LieNLSD on KS 1.003-1.007).
Through the zero-mean KS steppers (exp53's ablation) the boost moves at 0.76-0.91 of the true speed (42, 43; on 44 the
part is the translation alone); the development closed-loop span: 0.91 at t = 0, 0.86-1.01 over the horizon.
