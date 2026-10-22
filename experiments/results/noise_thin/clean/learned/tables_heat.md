
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 0.14 +/- 0.06 | 0.0 | 4.14 +/- 0.06 | 0.37 + 9.87 | 6.0 / 6 | 0.69 | 0.98 | 0.99 | 0.97 | 0.97 | 0.98 | 0.97 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 5.00 +/- 0.01 | 0.0 | 9.00 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.72 | 0.60 | 0.39 | 0.37 | 0.55 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 1.58 +/- 0.64 | 0.0 | 1.58 +/- 0.64 | 1.68 + 0.00 | 4.0 / 10 | 0.71 | 0.98 | 0.99 | 0.76 | 0.78 | 0.97 | 0.81 | 0.40 | 0.36 | 0.33 | 0.36 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.33 +/- 0.02 | 0.0 | 8.33 +/- 0.02 | 3.88 + 0.00 | 0.0 / 10 | 0.19 | 0.39 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.36 +/- 0.12 | 0.0 | 6.36 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.37 | 0.71 | 0.60 | 0.37 | 0.36 | 0.53 | 0.34 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 1.4x, 2.9x, 1.9x; lambda_(k+1) / lambda_k 1.2, 1.6, 1.2; rank from the largest gap 1, 11, 6; relative change 0.12, 0.21, 0.22; band check u_t share 1.1e-27, 4.6e-28, 9.2e-28
