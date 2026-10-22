
## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 1.68 +/- 0.23 | 0.0 | 5.68 +/- 0.23 | 1.73 + 9.87 | 3.3 / 6 | 0.51 | 0.81 | 0.41 | 0.99 | 0.99 | 0.52 | 0.99 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 5.00 +/- 0.01 | 0.0 | 9.00 +/- 0.01 | 2.92 + 9.87 | 0.0 / 6 | 0.21 | 0.41 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 2.91 +/- 0.10 | 0.0 | 6.91 +/- 0.10 | 2.15 + 9.87 | 0.0 / 6 | 0.51 | 0.72 | 0.60 | 0.39 | 0.37 | 0.55 | 0.35 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINS | 3 | 10 (10.0) | 2.32 +/- 0.34 | 0.0 | 2.32 +/- 0.34 | 2.05 + 0.00 | 3.3 / 10 | 0.65 | 0.82 | 0.42 | 0.98 | 0.98 | 0.57 | 0.98 | 0.64 | 0.58 | 0.59 | 0.63 |
| null: uniform on the class (10 of 60) | 3 | 10 (10.0) | 8.33 +/- 0.01 | 0.0 | 8.33 +/- 0.01 | 3.87 + 0.00 | 0.0 / 10 | 0.19 | 0.40 | 0.41 | 0.40 | 0.40 | 0.40 | 0.40 | 0.41 | 0.41 | 0.41 | 0.41 |
| null: Gaussian coefficients (10 of 60) | 3 | 10 (10.0) | 6.36 +/- 0.12 | 0.0 | 6.36 +/- 0.12 | 3.55 + 0.00 | 0.0 / 10 | 0.37 | 0.71 | 0.60 | 0.37 | 0.36 | 0.53 | 0.34 | 0.16 | 0.11 | 0.09 | 0.08 |

FINS: all below lambda_(k+1) on 0/3 seeds; worst margin 0.1x, 0.3x, 0.2x; lambda_(k+1) / lambda_k 1.5, 1.6, 1.3; rank from the largest gap 3, 8, 3; relative change 0.12, 0.21, 0.22; band check u_t share 2.2e-02, 1.2e-02, 1.5e-02
