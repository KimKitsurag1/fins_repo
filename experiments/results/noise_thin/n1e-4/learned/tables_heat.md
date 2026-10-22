
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 0.15 +/- 0.04 | 0.0 | 4.15 +/- 0.04 | 0.39 + 9.87 | 6.0 / 6 | 0.57 | 0.97 | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 4.99 +/- 0.01 | 0.0 | 8.99 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.40 | 0.41 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.72 | 0.60 | 0.39 | 0.37 | 0.55 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 1.33 +/- 0.60 | 0.0 | 1.33 +/- 0.60 | 1.50 + 0.00 | 5.0 / 10 | 0.67 | 0.97 | 0.98 | 0.89 | 0.90 | 0.96 | 0.92 | 0.43 | 0.34 | 0.35 | 0.40 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.32 +/- 0.03 | 0.0 | 8.32 +/- 0.03 | 3.87 + 0.00 | 0.0 / 10 | 0.19 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.36 +/- 0.12 | 0.0 | 6.36 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.37 | 0.71 | 0.60 | 0.37 | 0.36 | 0.53 | 0.34 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS: all below lambda_(k+1) on 3/3 seeds; worst margin 1.5x, 2.7x, 2.1x; lambda_(k+1) / lambda_k 1.4, 1.6, 1.1; rank from the largest gap 5, 11, 3; relative change 0.12, 0.21, 0.22; band check u_t share 2.3e-04, 1.2e-04, 1.5e-04
