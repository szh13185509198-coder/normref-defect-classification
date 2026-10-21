# P37 — light gate on the Cable held-out split (78 images; single run)

Commit `dbeccaf1fc92ffa876978f940a81ac8956bb9814`. P15 Cable units reproduced exactly (5 seeds).

## Cable, K = 8 — gate clip_l18

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

## Cable, K = 8 — gate dino_s

| K | AUROC | α | FRR | Escape rate | Type acc. (ARR) | FR / 1000 | Escapes / 1000 | Wrong type / 1000 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.948 | 0.01 | 0.0% | 39.3% | 84.9% | 0 | 19.6 | 4.6 |
| 8 | 0.948 | 0.02 | 0.0% | 31.4% | 83.5% | 0 | 15.7 | 5.7 |
| 8 | 0.948 | 0.05 | 2.0% | 18.6% | 84.2% | 19 | 9.3 | 6.4 |

Stage-2 reference (SimpleShot restricted), type acc. at α=0.02: K=8 80.3%

| K | Review band (auto-pass α) | Auto FRR | Escape rate | FR / 1000 | Escapes / 1000 | Review / 1000 |
|---|---|---|---|---|---|---|
| 8 | 0.1 | 0.0% | 15.7% | 0 | 7.9 | 54 |
| 8 | 0.3 | 0.0% | 10.7% | 0 | 5.4 | 155 |
| 8 | oracle (escape ≤ 1%, test labels) | — | 0.0% | 0 | 0.0 | 639 |

