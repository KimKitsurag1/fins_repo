
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 0.19 +/- 0.19 | 0.0 | 4.19 +/- 0.19 | 0.40 + 9.87 | 5.0 / 6 | 0.64 | 0.97 | 0.99 | 0.92 | 0.92 | 0.98 | 0.92 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 5.00 +/- 0.01 | 0.0 | 9.00 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.73 | 0.61 | 0.39 | 0.37 | 0.56 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 1.11 +/- 0.37 | 0.0 | 1.11 +/- 0.37 | 1.41 + 0.00 | 5.0 / 10 | 0.71 | 0.96 | 0.99 | 0.90 | 0.89 | 0.97 | 0.89 | 0.42 | 0.29 | 0.33 | 0.43 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.34 +/- 0.01 | 0.0 | 8.34 +/- 0.01 | 3.88 + 0.00 | 0.0 / 10 | 0.19 | 0.40 | 0.41 | 0.39 | 0.40 | 0.41 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.37 +/- 0.12 | 0.0 | 6.37 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.37 | 0.72 | 0.60 | 0.37 | 0.35 | 0.54 | 0.34 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 1.6x, 4.5x, 1.9x; lambda_(k+1) / lambda_k 1.3, 2.0, 1.1; rank from the largest gap 6, 11, 11; relative change 0.12, 0.21, 0.17; band check u_t share 1.6e-28, 5.8e-29, 1.1e-28
