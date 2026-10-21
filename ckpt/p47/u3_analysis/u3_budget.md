# P47 U3 - uncertainty budget of the benchmark balanced accuracy (ARR, MVTec-FS formal)

## K = 1

Benchmark mean BA 71.01 %; u_c = 1.46 pp, nu_eff = 66.3, k = 2.00, U95 = 2.91 pp. Existing product-cluster bootstrap CI: [63.98, 78.15] % (also contains between-product spread).

| Component | u (pp) | nu | Share of u_c^2 |
|---|---|---|---|
| support draw S | 1.023 | 16.7 | 49.3 % |
| bank draw B | 0.117 | 4.02 | 0.6 % |
| interaction S x B | 0.380 | 139 | 6.8 % |
| query sampling Q | 0.909 | 302 | 38.9 % |
| implementation I (Type B, rectangular) | 0.305 | inf | 4.4 % |

Single deployment (one product, one support set, one bank): pooled SD 10.40 pp.

| Product | Mean BA | SD (pp) | 95 % prediction interval (clipped to [0, 100]) |
|---|---|---|---|
| bottle | 71.58 | 9.69 | [51.5, 91.7] |
| capsule | 58.47 | 9.35 | [37.5, 79.4] |
| carpet | 72.11 | 6.39 | [59.3, 84.9] |
| grid | 63.04 | 12.62 | [34.9, 91.1] |
| hazelnut | 80.81 | 7.79 | [64.9, 96.7] |
| leather | 82.34 | 7.52 | [65.8, 98.8] |
| metal_nut | 72.03 | 10.75 | [46.5, 97.5] |
| pill | 67.08 | 9.86 | [43.9, 90.2] |
| screw | 46.75 | 9.27 | [27.2, 66.3] |
| tile | 98.97 | 2.21 | [94.4, 100.0] |
| transistor | 62.20 | 21.43 | [8.4, 100.0] |
| wood | 80.93 | 8.01 | [64.4, 97.5] |
| zipper | 72.09 | 9.00 | [52.0, 92.1] |

SimpleShot (S and Q only): mean 56.92 %, u_c 1.39 pp, U95 2.75 pp.

| M | Macro BA | sigma_B (pp) | sigma_S (pp) | sigma_SB (pp) |
|---|---|---|---|---|
| 5 | 70.62 | 2.25 | 8.21 | 4.13 |
| 10 | 70.71 | 0.85 | 8.02 | 3.74 |
| 20 | 71.06 | 1.33 | 8.26 | 3.37 |
| 50 | 71.48 | 0.87 | 7.86 | 2.93 |

## K = 3

Benchmark mean BA 84.50 %; u_c = 1.18 pp, nu_eff = 299, k = 1.97, U95 = 2.32 pp. Existing product-cluster bootstrap CI: [79.56, 89.20] % (also contains between-product spread).

| Component | u (pp) | nu | Share of u_c^2 |
|---|---|---|---|
| support draw S | 0.564 | 27.2 | 22.9 % |
| bank draw B | 0.141 | 8.37 | 1.4 % |
| interaction S x B | 0.310 | 134 | 6.9 % |
| query sampling Q | 0.929 | 284 | 62.1 % |
| implementation I (Type B, rectangular) | 0.305 | inf | 6.7 % |

Single deployment (one product, one support set, one bank): pooled SD 7.24 pp.

| Product | Mean BA | SD (pp) | 95 % prediction interval (clipped to [0, 100]) |
|---|---|---|---|
| bottle | 88.34 | 7.77 | [72.1, 100.0] |
| capsule | 74.08 | 7.19 | [59.4, 88.8] |
| carpet | 82.70 | 6.79 | [68.8, 96.6] |
| grid | 78.56 | 11.31 | [53.8, 100.0] |
| hazelnut | 90.82 | 5.89 | [78.8, 100.0] |
| leather | 89.78 | 5.77 | [77.6, 100.0] |
| metal_nut | 94.63 | 4.48 | [85.6, 100.0] |
| pill | 80.46 | 6.30 | [67.1, 93.9] |
| screw | 66.88 | 7.44 | [51.8, 81.9] |
| tile | 98.97 | 2.79 | [92.6, 100.0] |
| transistor | 76.80 | 10.79 | [54.1, 99.5] |
| wood | 91.09 | 6.36 | [77.4, 100.0] |
| zipper | 86.01 | 6.86 | [71.5, 100.0] |

SimpleShot (S and Q only): mean 68.88 %, u_c 1.44 pp, U95 2.84 pp.

| M | Macro BA | sigma_B (pp) | sigma_S (pp) | sigma_SB (pp) |
|---|---|---|---|---|
| 5 | 83.25 | 2.85 | 4.28 | 2.77 |
| 10 | 84.35 | 0.87 | 4.49 | 2.63 |
| 20 | 84.44 | 1.26 | 4.09 | 2.77 |
| 50 | 84.97 | 1.17 | 4.23 | 2.34 |

## K = 5

Benchmark mean BA 87.48 %; u_c = 1.14 pp, nu_eff = 362, k = 1.97, U95 = 2.24 pp. Existing product-cluster bootstrap CI: [83.37, 91.37] % (also contains between-product spread).

| Component | u (pp) | nu | Share of u_c^2 |
|---|---|---|---|
| support draw S | 0.368 | 23.7 | 10.5 % |
| bank draw B | 0.205 | 7.51 | 3.3 % |
| interaction S x B | 0.247 | 138 | 4.7 % |
| query sampling Q | 0.981 | 258 | 74.4 % |
| implementation I (Type B, rectangular) | 0.305 | inf | 7.2 % |

Single deployment (one product, one support set, one bank): pooled SD 6.03 pp.

| Product | Mean BA | SD (pp) | 95 % prediction interval (clipped to [0, 100]) |
|---|---|---|---|
| bottle | 88.68 | 8.15 | [71.3, 100.0] |
| capsule | 75.96 | 5.76 | [64.4, 87.5] |
| carpet | 84.53 | 6.35 | [71.6, 97.4] |
| grid | 83.52 | 8.14 | [66.8, 100.0] |
| hazelnut | 92.26 | 4.91 | [82.4, 100.0] |
| leather | 93.16 | 4.64 | [83.7, 100.0] |
| metal_nut | 95.76 | 3.86 | [88.0, 100.0] |
| pill | 83.16 | 5.54 | [71.8, 94.6] |
| screw | 72.75 | 6.93 | [58.5, 87.0] |
| tile | 99.43 | 1.76 | [95.6, 100.0] |
| transistor | 85.00 | 8.43 | [67.3, 100.0] |
| wood | 94.10 | 5.23 | [83.2, 100.0] |
| zipper | 89.78 | 5.08 | [79.6, 100.0] |

SimpleShot (S and Q only): mean 74.09 %, u_c 1.43 pp, U95 2.82 pp.

| M | Macro BA | sigma_B (pp) | sigma_S (pp) | sigma_SB (pp) |
|---|---|---|---|---|
| 5 | 86.23 | 3.39 | 3.34 | 2.07 |
| 10 | 87.14 | 1.40 | 3.22 | 2.02 |
| 20 | 87.17 | 2.34 | 3.15 | 1.72 |
| 50 | 87.60 | 1.95 | 3.10 | 1.94 |

