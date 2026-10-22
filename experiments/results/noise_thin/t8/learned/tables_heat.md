
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 0.11 +/- 0.06 | 0.0 | 4.11 +/- 0.06 | 0.32 + 9.87 | 6.0 / 6 | 0.60 | 0.98 | 0.99 | 0.97 | 0.97 | 0.99 | 0.97 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 4.99 +/- 0.01 | 0.0 | 8.99 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.73 | 0.62 | 0.39 | 0.37 | 0.57 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 1.34 +/- 0.58 | 0.0 | 1.34 +/- 0.58 | 1.44 + 0.00 | 4.0 / 10 | 0.68 | 0.97 | 0.98 | 0.82 | 0.81 | 0.96 | 0.82 | 0.40 | 0.38 | 0.40 | 0.44 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.33 +/- 0.01 | 0.0 | 8.33 +/- 0.01 | 3.88 + 0.00 | 0.0 / 10 | 0.19 | 0.40 | 0.41 | 0.39 | 0.40 | 0.41 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.38 +/- 0.12 | 0.0 | 6.38 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.36 | 0.72 | 0.61 | 0.37 | 0.35 | 0.55 | 0.33 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 1.9x, 3.7x, 1.7x; lambda_(k+1) / lambda_k 1.1, 2.6, 1.1; rank from the largest gap 12, 11, 6; relative change 0.13, 0.21, 0.15; band check u_t share 5.2e-29, 1.9e-29, 4.1e-29
