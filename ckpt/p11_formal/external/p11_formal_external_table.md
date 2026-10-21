# P11 formal F2 — VisA (confirmatory) and MTD (descriptive), single run, post-hoc hypotheses

Commit `873c5dacb15aa6b27256afc81573bf654109e05c`. Integrity vs P10 external: simpleshot_cl2n max|Δ|=0.0e+00, fused_normref_clip18 max|Δ|=0.0e+00, fused_normref_clip18_m5 max|Δ|=0.0e+00

| Hypothesis | Contrast | K=1 | K=3 | K=5 | Pooled over K | Verdict |
|---|---|---|---|---|---|---|
| E4_primary_A | fused_adareg_soft_q99 − fused_normref_clip18 | +9.51 [+3.03, +15.09] (10/11) | +9.45 [+3.30, +14.61] (10/11) | +9.95 [+4.25, +15.06] (10/11) | +9.64 [+3.68, +14.53] (10/11) | CONFIRMED |
| E5_secondary_B | fused_guided_xclass_normref − fused_normref_clip18 | +3.76 [+1.71, +5.88] (10/11) | +3.88 [+1.97, +5.91] (10/11) | +5.21 [+3.06, +7.34] (9/11) | +4.28 [+2.48, +6.18] (10/11) | CONFIRMED |

| MTD contrast | Δ BA K=1 / 3 / 5 (seed mean ± SD) |
|---|---|
| fused_adareg_soft_q99 − fused_normref_clip18 | +0.34±1.07 / +6.69±3.10 / +2.99±4.76 |
| fused_guided_xclass_normref − fused_normref_clip18 | +2.17±1.32 / +5.71±1.27 / +3.26±2.47 |

| Descriptive | VisA pooled | MTD K=1 / 3 / 5 |
|---|---|---|
| fused_adareg_soft_q99_minus_simpleshot_cl2n | +15.83 [+10.32, +20.50] (10/11) | +5.44±3.52 / +15.02±5.48 / +12.26±6.45 |
| fused_guided_xclass_normref_minus_simpleshot_cl2n | +10.47 [+7.25, +14.11] (11/11) | +7.26±2.78 / +14.03±3.95 / +12.52±4.18 |
| fused_adareg_soft_q99_m5_minus_fused_normref_clip18_m5 | +7.13 [+1.74, +12.06] (9/11) | +1.42±3.07 / +6.80±2.66 / +5.52±4.98 |
| fused_guided_xclass_normref_m5_minus_fused_normref_clip18_m5 | +3.59 [+1.72, +5.56] (9/11) | +1.72±1.08 / +6.38±1.63 / +4.05±1.18 |

| Method | VisA K=1 | VisA K=3 | VisA K=5 | MTD K=1 | MTD K=3 | MTD K=5 |
|---|---|---|---|---|---|---|
| simpleshot_cl2n | 43.56 | 56.29 | 62.19 | 30.34 | 38.42 | 45.63 |
| fused_normref_clip18 | 49.17 | 63.72 | 67.73 | 35.43 | 46.75 | 54.89 |
| fused_adareg_soft_q99 | 58.68 | 73.17 | 77.67 | 35.78 | 53.44 | 57.88 |
| fused_guided_xclass_normref | 52.93 | 67.60 | 72.93 | 37.60 | 52.45 | 58.15 |
| fused_normref_clip18_m5 | 49.98 | 62.11 | 67.17 | 34.89 | 45.85 | 53.10 |
| fused_adareg_soft_q99_m5 | 55.49 | 70.23 | 74.93 | 36.31 | 52.65 | 58.62 |
| fused_guided_xclass_normref_m5 | 52.45 | 65.54 | 72.04 | 36.61 | 52.23 | 57.15 |
