# P14 — Final formal addendum: dual-branch C1 (A + region residual) and support-consensus C0

Status: **LOCKED BEFORE ANY P14 OUTPUT** (2026-09-27). Written after the P13 dev pilot
(`ckpt/p13_pilot/dev/`). **This is the last formal round**: no further method is evaluated on
the MVTec-FS formal queries, VisA or MTD after P14, whatever the outcome.

## 1. Methods (fixed by the P13 selection rule; code reused unchanged)

- **ARR** `fused_adareg_regres_soft_q99` (C1): z(S) + z(A score) + z(region-residual score);
  region residual = Σ w (t − nn_B(t)) / Σ w with A's weights (`experiments/p13/pilot.region_residual`).
  Dev: +3.46 pp vs A, 11/13. M = 5 sensitivity row `_m5`.
- **CONS** `fused_cons_soft_supq90` (C0, no normal images): support-consensus anomaly,
  τ0 = 0.90-quantile over support patches, soft weights, SimpleShot, fused with S
  (`experiments/p13/pilot.consensus_scores`). Dev: +5.95 pp vs cross-class, 12/13.
- Only these two rows are new on formal data; the other P13 variants are dev-only.
- Declared post-hoc: added after the P10–P12 formal runs; the paper lists every dev variant.

## 2. Hypotheses (BA pooled over K; CONFIRMED iff 95% CI lower bound > 0)

| ID | Data | Contrast | Bootstrap |
|---|---|---|---|
| H6 | MVTec-FS formal | ARR − A | product cluster, seed 20260813 |
| H7 | MVTec-FS formal | CONS − `fused_xclass_clip18` | product cluster |
| E6 | VisA | ARR − A | object cluster, seed 20260814 |
| E7 | VisA | CONS − `fused_xclass_clip18` | object cluster |

MTD: same contrasts, descriptive (seed mean ± SD).

**Main-method rule (fixed now):**
- C1 main method becomes ARR only if **H6 and E6 are both CONFIRMED**; otherwise A stays the
  C1 main method and ARR is reported as a supplementary row with its outcome.
- The C0 method becomes CONS if **H7 is CONFIRMED** (E7 reported alongside); otherwise
  cross-class stays the C0 method.

**Descriptive (no verdict):** ARR and CONS − SimpleShot; CONS − V1 `ours` (same condition C0)
and − `mvrec_rgb` (C0 + text); ARR − V1 / MVREC-RGB (cross-condition, labelled); M = 5 rows.

## 3. Stages (each once, lock directory, staging + rename)

- **mvtec:** cached P10 formal features; integrity — SimpleShot, `fused_normref_clip18`,
  `fused_xclass_clip18` equal P10 formal and A, A_m5 equal P11 formal (max |Δ| < 1e-9).
  Per-query predictions saved.
- **external:** VisA / MTD re-extracted exactly as P10–P12; integrity — SimpleShot, normref,
  cross-class equal P10 external, A / A_m5 equal P11 external. Normal-aware metrics
  (false rejects / escapes per 1,000 at 95 % good) for all rows, same definitions as P10 R2.
- **cable:** Cable OOF (P10 R1 pool, folds, seeds, banks); integrity — SimpleShot, normref,
  cross-class predictions equal P10 R1, A equals P12 R1'. Descriptive.

## 4. Resource profile (not a hypothesis; no test data)

Batch-1 FP32 latency (50 warm-up, 200 timed; median / mean / p90) and peak GPU memory on the
local RTX 4050, product bottle, formal support manifest K = 5 seed 111 (support-pool images),
M = 20 bank from MVTec AD `train/good`. Online per query: SimpleShot, fixed-fraction normref,
A, ARR, CONS, cross-class. Offline once per product: support and bank feature extraction,
leave-one-out calibration of τ, prototype construction. Timing image: a support-pool image.

## 5. Outputs

`ckpt/p14_formal/{mvtec,external,cable}/`, `ckpt/p14_resource/`.
