# P40 — screening at layer 12 vs 18 on the Cable held-out split (single run)

Commit `8accee87c6f4e80b95d62947d82f604a5cba750c`. P15 Cable units reproduced exactly (5 seeds).

## Cable, K = 8 — layer18

| K | AUROC | α | FRR | Escape rate | Type acc. (ARR) | FR / 1000 | Escapes / 1000 | Wrong type / 1000 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.969 | 0.01 | 0.0% | 29.3% | 82.8% | 0 | 14.6 | 6.1 |
| 8 | 0.969 | 0.02 | 0.4% | 23.6% | 83.0% | 4 | 11.8 | 6.4 |
| 8 | 0.969 | 0.05 | 2.4% | 18.6% | 84.2% | 23 | 9.3 | 6.4 |

Stage-2 reference (SimpleShot restricted), type acc. at α=0.02: K=8 76.4%

| K | Review band (auto-pass α) | Auto FRR | Escape rate | FR / 1000 | Escapes / 1000 | Review / 1000 |
|---|---|---|---|---|---|---|
| 8 | 0.1 | 0.0% | 15.0% | 0 | 7.5 | 57 |
| 8 | 0.3 | 0.0% | 6.4% | 0 | 3.2 | 190 |
| 8 | oracle (escape ≤ 1%, test labels) | — | 0.0% | 0 | 0.0 | 326 |

## Cable, K = 8 — layer12

| K | AUROC | α | FRR | Escape rate | Type acc. (ARR) | FR / 1000 | Escapes / 1000 | Wrong type / 1000 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.977 | 0.01 | 0.0% | 64.3% | 86.3% | 0 | 32.1 | 2.5 |
| 8 | 0.977 | 0.02 | 0.0% | 50.7% | 87.0% | 0 | 25.4 | 3.2 |
| 8 | 0.977 | 0.05 | 0.8% | 19.3% | 85.7% | 8 | 9.6 | 5.7 |

Stage-2 reference (SimpleShot restricted), type acc. at α=0.02: K=8 84.7%

| K | Review band (auto-pass α) | Auto FRR | Escape rate | FR / 1000 | Escapes / 1000 | Review / 1000 |
|---|---|---|---|---|---|---|
| 8 | 0.1 | 0.0% | 10.7% | 0 | 5.4 | 107 |
| 8 | 0.3 | 0.0% | 0.7% | 0 | 0.4 | 264 |
| 8 | oracle (escape ≤ 1%, test labels) | — | 0.0% | 0 | 0.0 | 260 |

