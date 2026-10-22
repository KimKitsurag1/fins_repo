
## burgers: 3 seeds, ground truth finite 5; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (5)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_F | 3 | 5 (5.0) | 0.55 +/- 0.46 | 0.0 | 0.55 +/- 0.46 | 0.82 + 0.00 | 3.7 / 5 | 0.81 | 0.95 | 0.91 | 0.95 | 0.69 | 0.88 |
| LieNLSD_F, own count | 3 | 3-5 (4.0) | 1.38 +/- 0.44 | 1.0 | 0.38 +/- 0.54 | 0.55 + 2.47 | 0.7 / 5 | 0.84 | 0.88 | 0.79 | 0.84 | 0.18 | 0.65 |
| LieNLSD_F, untrained | 3 | 5 (5.0) | 1.07 +/- 0.18 | 0.0 | 1.07 +/- 0.18 | 1.16 + 0.00 | 1.0 / 5 | 0.70 | 0.94 | 0.86 | 0.82 | 0.69 | 0.74 |
| LieNLSD_F, untrained, own count | 3 | 4-5 (4.7) | 1.24 +/- 0.16 | 0.3 | 0.91 +/- 0.38 | 1.04 + 0.82 | 1.0 / 5 | 0.74 | 0.94 | 0.84 | 0.76 | 0.69 | 0.70 |


## kdv: 3 seeds, ground truth finite 4; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (4)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | Galilean | scaling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_F | 3 | 4 (4.0) | 0.90 +/- 0.13 | 0.0 | 0.90 +/- 0.13 | 1.32 + 0.00 | 1.3 / 4 | 0.88 | 0.95 | 0.85 | 0.52 | 0.35 |
| LieNLSD_F, own count | 3 | 2-4 (2.7) | 1.57 +/- 0.60 | 1.3 | 0.24 +/- 0.34 | 0.34 + 3.29 | 1.3 / 4 | 0.99 | 0.95 | 0.84 | 0.17 | 0.24 |
| LieNLSD_F, untrained | 3 | 4 (4.0) | 0.98 +/- 0.06 | 0.0 | 0.98 +/- 0.06 | 1.15 + 0.00 | 1.3 / 4 | 0.72 | 0.97 | 0.91 | 0.69 | 0.70 |
| LieNLSD_F, untrained, own count | 3 | 4 (4.0) | 0.98 +/- 0.06 | 0.0 | 0.98 +/- 0.06 | 1.15 + 0.00 | 1.3 / 4 | 0.72 | 0.97 | 0.91 | 0.69 | 0.70 |


## heat: 3 seeds, ground truth finite 6, full in the class 10; component scaling xi / L, eta / T, phi / sigma_u; region [0.2, 0.8] L

### ground truth: finite (6)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_F | 3 | 10 (10.0) | 0.32 +/- 0.12 | 0.0 | 4.32 +/- 0.12 | 0.59 + 9.87 | 5.0 / 6 | 0.72 | 0.98 | 0.94 | 0.93 | 0.93 | 0.95 | 0.94 |
| LieNLSD_F, own count | 3 | 8-10 (9.3) | 0.59 +/- 0.42 | 0.0 | 3.93 +/- 0.56 | 0.82 + 8.22 | 3.0 / 6 | 0.70 | 0.93 | 0.85 | 0.89 | 0.89 | 0.83 | 0.90 |
| LieNLSD_F, untrained | 3 | 10 (10.0) | 1.14 +/- 0.05 | 0.0 | 5.14 +/- 0.05 | 1.32 + 9.87 | 1.0 / 6 | 0.74 | 0.80 | 0.61 | 0.89 | 0.89 | 0.49 | 0.90 |
| LieNLSD_F, untrained, own count | 3 | 10 (10.0) | 1.14 +/- 0.05 | 0.0 | 5.14 +/- 0.05 | 1.32 + 9.87 | 1.0 / 6 | 0.74 | 0.80 | 0.61 | 0.89 | 0.89 | 0.49 | 0.90 |

### ground truth: full (10)

| method | seeds | k (rank) | missing dims | floor (k_B - rank) | spurious dims | Grassmann angle part + penalty | recovered (novel >= 0.9) | cosine | ∂_x | ∂_t | u∂_u | Galilean | scaling | projective | b=1 | b=x | b=x2+2at | b=x3+6axt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LieNLSD_F | 3 | 10 (10.0) | 0.43 +/- 0.17 | 0.0 | 0.43 +/- 0.17 | 0.69 + 0.00 | 8.3 / 10 | 0.71 | 0.99 | 0.95 | 0.89 | 0.90 | 0.97 | 0.91 | 0.99 | 0.99 | 0.99 | 0.99 |
| LieNLSD_F, own count | 3 | 8-10 (9.3) | 0.93 +/- 0.78 | 0.7 | 0.26 +/- 0.24 | 0.46 + 1.64 | 6.3 / 10 | 0.71 | 0.92 | 0.85 | 0.70 | 0.71 | 0.81 | 0.73 | 0.99 | 0.97 | 0.95 | 0.93 |
| LieNLSD_F, untrained | 3 | 10 (10.0) | 2.29 +/- 0.15 | 0.0 | 2.29 +/- 0.15 | 2.02 + 0.00 | 0.0 / 10 | 0.73 | 0.80 | 0.61 | 0.81 | 0.81 | 0.51 | 0.82 | 0.57 | 0.41 | 0.29 | 0.24 |
| LieNLSD_F, untrained, own count | 3 | 10 (10.0) | 2.29 +/- 0.15 | 0.0 | 2.29 +/- 0.15 | 2.02 + 0.00 | 0.0 / 10 | 0.73 | 0.80 | 0.61 | 0.81 | 0.81 | 0.51 | 0.82 | 0.57 | 0.41 | 0.29 | 0.24 |

