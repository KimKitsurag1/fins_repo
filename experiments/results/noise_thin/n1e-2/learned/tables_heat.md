
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 2.96 +/- 0.03 | 0.0 | 6.96 +/- 0.03 | 2.54 + 9.87 | 3.0 / 6 | 0.44 | 0.28 | 0.17 | 0.99 | 0.99 | 0.33 | 0.99 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 4.99 +/- 0.01 | 0.0 | 8.99 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.72 | 0.60 | 0.39 | 0.37 | 0.55 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 3.25 +/- 0.29 | 0.0 | 3.25 +/- 0.29 | 2.64 + 0.00 | 5.7 / 10 | 0.62 | 0.34 | 0.19 | 0.98 | 0.98 | 0.41 | 0.98 | 0.85 | 0.84 | 0.84 | 0.85 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.33 +/- 0.02 | 0.0 | 8.33 +/- 0.02 | 3.88 + 0.00 | 0.0 / 10 | 0.18 | 0.39 | 0.41 | 0.40 | 0.40 | 0.40 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.36 +/- 0.12 | 0.0 | 6.36 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.37 | 0.71 | 0.60 | 0.37 | 0.36 | 0.53 | 0.34 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS: all below lambda_(k+1) on 0/3 seeds; worst margin 0.1x, 0.1x, 0.1x; lambda_(k+1) / lambda_k 1.2, 1.5, 1.3; rank from the largest gap 9, 6, 7; relative change 0.32, 0.28, 0.36; band check u_t share 4.0e-01, 3.5e-01, 3.6e-01
