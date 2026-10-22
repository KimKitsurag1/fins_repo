# Emulator pairs with and without rotation augmentation (exp68)

| regime | seed | arm | held-out error (1 step) | commutator: rotation 90 / 30 / 2 deg, d_x 8 / 0.5 cells | rotation plane (FINS) | quotients (FINS) | span of k, missing |
|---|---|---|---|---|---|---|---|
| f | 42 | plain | 0.0371 | 5.6e-02 / 5.2e-02 / 9.9e-03 / 2.4e-15 / 7.4e-03 | 6.9e-04 | d_x 7.7e-05 d_y 1.4e-04 d_t 1.3e-09 rotation 6.9e-04 D_x 1.0e-03 D_t 5.9e-04 D_beta 9.4e-04 | 0.006 |
| f | 42 | grid | 0.0275 | 3.8e-02 / 3.5e-02 / 6.2e-03 / 5.1e-03 / 4.7e-03 | 2.5e-04 | d_x 1.6e-05 d_y 1.9e-05 d_t 1.3e-09 rotation 2.5e-04 D_x 4.4e-04 D_t 2.2e-04 D_beta 3.8e-04 | 0.004 |
| f | 42 | rot | 0.0204 | 3.0e-02 / 2.7e-02 / 5.6e-03 / 2.2e-15 / 6.0e-03 | 1.7e-04 | d_x 3.8e-05 d_y 1.1e-04 d_t 1.3e-09 rotation 1.7e-04 D_x 2.8e-04 D_t 1.5e-04 D_beta 2.4e-04 | 0.013 |
| beta | 42 | plain | 0.0257 | 1.1e+00 / 4.3e-01 / 3.0e-02 / 1.6e-15 / 5.4e-03 | 1.3e-02 | d_x 7.2e-05 d_y 1.2e-04 d_t 2.6e-10 rotation 1.3e-02 D_x 9.9e-03 D_t 7.3e-02 D_beta 5.0e-04 | 0.294 |
| f | 43 | plain | 0.0402 | 5.2e-02 / 5.0e-02 / 1.0e-02 / 3.5e-15 / 1.9e-03 | 7.4e-04 | d_x 5.6e-05 d_y 1.8e-04 d_t 4.7e-10 rotation 7.4e-04 D_x 1.1e-03 D_t 1.1e-03 D_beta 1.0e-03 | 0.007 |
| f | 43 | grid | 0.0412 | 5.1e-02 / 5.2e-02 / 9.4e-03 / 1.1e-02 / 2.3e-03 | 5.3e-04 | d_x 4.6e-05 d_y 1.2e-04 d_t 4.8e-10 rotation 5.4e-04 D_x 9.3e-04 D_t 9.7e-04 D_beta 9.1e-04 | 0.093 |
| f | 43 | rot | 0.0202 | 2.5e-02 / 2.5e-02 / 4.7e-03 / 3.4e-15 / 9.6e-04 | 1.9e-04 | d_x 2.5e-05 d_y 7.7e-05 d_t 4.6e-10 rotation 1.9e-04 D_x 3.2e-04 D_t 2.1e-04 D_beta 2.9e-04 | 0.002 |
| beta | 43 | plain | 0.0161 | 1.3e+00 / 4.8e-01 / 3.3e-02 / 1.3e-15 / 9.0e-04 | 1.4e-02 | d_x 4.7e-05 d_y 9.6e-05 d_t 1.7e-10 rotation 1.4e-02 D_x 9.2e-03 D_t 5.9e-02 D_beta 7.0e-04 | 0.002 |
| f | 44 | plain | 0.0340 | 4.8e-02 / 4.7e-02 / 8.1e-03 / 2.2e-15 / 4.2e-03 | 4.6e-04 | d_x 1.0e-04 d_y 1.7e-04 d_t 4.6e-10 rotation 4.6e-04 D_x 6.5e-04 D_t 1.4e-03 D_beta 6.7e-04 | 0.018 |
| f | 44 | grid | 0.0269 | 3.8e-02 / 3.5e-02 / 6.8e-03 / 7.8e-03 / 2.3e-03 | 2.9e-04 | d_x 3.5e-05 d_y 7.4e-05 d_t 3.9e-10 rotation 2.9e-04 D_x 4.5e-04 D_t 1.6e-03 D_beta 5.4e-04 | 0.019 |
| f | 44 | rot | 0.0189 | 2.6e-02 / 2.5e-02 / 4.6e-03 / 2.4e-15 / 3.0e-03 | 1.6e-04 | d_x 6.2e-05 d_y 1.2e-04 d_t 5.2e-10 rotation 1.6e-04 D_x 2.5e-04 D_t 3.9e-04 D_beta 2.5e-04 | 0.007 |
| beta | 44 | plain | 0.0246 | 1.3e+00 / 4.8e-01 / 3.3e-02 / 1.5e-15 / 2.6e-03 | 6.7e-03 | d_x 6.9e-05 d_y 1.1e-04 d_t 1.6e-10 rotation 6.7e-03 D_x 4.8e-03 D_t 4.5e-02 D_beta 4.3e-04 | 0.018 |
