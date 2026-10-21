# P11 Formal addendum — adaptive region (A) and guided matching (B) on formal / external data

Status: **LOCKED BEFORE ANY P11 FORMAL OUTPUT** (2026-09-27). Written after the P11 dev pilot
(`ckpt/p11_pilot/`, ledger "P11 DEV PILOT").

## 1. What goes to formal (fixed by the P11 dev selection rule)

- **A:** `fused_adareg_soft_q99` — z(S) + z(SimpleShot on the soft-weighted descriptor),
  weights w(t) = max(a(t) − τ_0.99, 0), τ_0.99 = 0.99-quantile of leave-one-image-out bank
  anomalies, top-6 fallback. Dev: +6.18 pp vs `fused_normref_clip18`, 13/13.
- **B:** `fused_guided_xclass_normref` — z(S) + z(R) + z(G_xclass). Dev: +1.53 pp, 10/13.
- Both with M = 20 (primary) and M = 5 (`_m5`, sensitivity). All constants exactly as in
  `docs/P11_ADAPTIVE_REGION_AND_GUIDED_MATCHING_PILOT_PROTOCOL.md`; code
  `experiments/p11/run_pilot_ab.py` functions are reused unchanged.
- **Disclosure:** these are hypotheses added **after** the P10 formal and external runs.
  The paper states this, lists every P11 dev variant, and reports this addendum's outcome
  whatever it is. Each stage below runs once.

## 2. Stage F1 — MVTec-FS-RGB formal (530 queries, 195 units)

Same data path as P10 formal / Supplement S: cached P10 formal features
(`ckpt/p10_formal/features/`), formal P6.B support manifests, formal normal banks.

- **H4 (primary, C1):** `fused_adareg_soft_q99` − `fused_normref_clip18`.
- **H5 (secondary, C1):** `fused_guided_xclass_normref` − `fused_normref_clip18`.
- Decision: product-cluster percentile bootstrap (10,000 resamples, seed 20260813, the P10
  estimator), BA pooled over K; CONFIRMED if the 95% CI lower bound > 0. Per-K CIs and
  positive products reported. Macro-F1 reported alongside.
- Descriptive (no verdict): A and B − SimpleShot; A − `residual_mean_clip18` and
  − `fused_residual_mean_clip18` (Supplement S rows); A − V1 `ours` and − `mvrec_rgb`
  (cross-condition, labelled as such); fraction of mask-oracle headroom per K; M = 5 rows.
- **Integrity:** recomputed `simpleshot_cl2n`, `fused_normref_clip18`,
  `fused_normref_clip18_m5`, `oracle_maskpool_clip18` must equal P10 formal (max |Δ| < 1e-9),
  else the stage aborts before writing.

## 3. Stage F2 — VisA (confirmatory) and MTD (descriptive)

Same tasks, frozen class orders, support draws and bank draws as P10 external
(`experiments/p10/external_evaluate.py`: `bank_draw`, `extract`); images are re-extracted.

- **E4 (primary, VisA):** `fused_adareg_soft_q99` − `fused_normref_clip18`, object-cluster
  bootstrap (seed 20260814), BA pooled over K, CONFIRMED if CI lower bound > 0.
- **E5 (secondary, VisA):** `fused_guided_xclass_normref` − `fused_normref_clip18`.
- MTD: the same contrasts, seed-level mean ± SD per K (descriptive).
- Descriptive: A and B − SimpleShot on VisA and MTD.
- **Integrity:** recomputed `simpleshot_cl2n` and `fused_normref_clip18` VisA object-macro
  BA and MTD BA must equal P10 external (max |Δ| < 1e-9), else abort before writing.
- Per-query predictions are stored for a later normal-aware (false-reject) analysis,
  which needs its own addendum.

## 4. Single-shot mechanics

Each stage refuses to run with uncommitted changes under `experiments/p10`, `experiments/p11`,
`experiments/revision` or the P11 protocols; refuses if its output directory exists; holds an
exclusive lock directory; writes to a staging directory and renames on success.
Outputs: `ckpt/p11_formal/mvtec/`, `ckpt/p11_formal/external/`.
