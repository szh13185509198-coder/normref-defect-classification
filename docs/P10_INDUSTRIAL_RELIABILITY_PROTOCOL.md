# P10 Industrial reliability analyses — Cable OOF comparison (R1) and normal-aware metrics on VisA/MTD (R2)

Status: **LOCKED BEFORE ANY R1/R2 OUTPUT** (2026-09-26). Both analyses are post-formal and
**descriptive**; they add industrial decision metrics and one same-protocol comparison that
the author requested. They do not change any P10 method, the selected variants, or earlier
verdicts. The manuscript labels them "added during revision".

## R1 — Cable 8-fold OOF: training-free cues vs the archived V1 method

Question: is the high V1 accuracy on the Cable development pool due to the support-side
adaptation or to the frozen features / task?

- Data: the frozen Cable development pool (72 images: `good` + 8 defect classes, 8 per
  class; `dataset/metadata/MVTecAD-Cable/oof_8fold_v1`, images under
  `data/shenzehui/mvtecad-cable`). The sealed Cable formal test (79 images) is **not** read.
- Folds: the same 8 folds (train 63 = 7 per class as supports, val 9 = 1 per class as
  queries). Each image is predicted once per seed (out-of-fold).
- Seeds 111–555. The C0 methods are deterministic, so their seeds coincide; C1 normal banks
  are drawn per seed: `random.Random(f"p10cable|{seed}").sample(sorted(MVTec AD cable/train/good), 20)`.
  MVTec AD `cable/train/good` is disjoint from the pool (byte identity asserted in code).
- Methods (unchanged definitions): `fusion_proto_local`, `simpleshot_cl2n`,
  `fused_xclass_clip18` (C0); `fused_normref_clip18`, `residual_mean_clip18`,
  `fused_normcenter_mean_clip18` (C1). Reference: archived P3.F OOF predictions of V1
  (`ckpt/p3f_clean_five_seed/s*/fold_*/oof_predictions_fold_*.jsonl`).
- Metrics per seed over the 72 OOF predictions: 9-class BA, Macro-F1, Top-1; argmax-based
  normal-aware metrics (good recall, defect recall = defect predicted as any defect, binary
  BA, defect-type accuracy among detected defects). V1 is additionally shown with its
  archived tau=0.006 rule as a separate reference row (no threshold for any new method).
- V1 rows: `v1_ours_argmax` (raw argmax of the archived OOF predictions) and `v1_ours_tau`
  (archived tau=0.006 normal-aware rule; this is the paper-reported V1, e.g. 9-class BA
  81.67%). The code verifies that the tau row reproduces the P3.F audit exactly.
- Reading rule (fixed now; uses `v1_ours_argmax` because no new method uses a threshold):
  with d = V1 − `simpleshot_cl2n` (9-class BA, mean over seeds):
  d <= +2.0 pp → "Cable accuracy is attributable to the frozen features, not to adaptation";
  d >= +5.0 pp → "adaptation helps on this 8-shot, normal-inclusive task";
  otherwise "inconclusive". The same reading is reported for the best-performing new method
  versus V1 only descriptively. Caveat, always stated: V1 was designed on this pool
  (home advantage for V1), and 72 images give low resolution (1 image = 1.39 pp Top-1).

## R2 — normal-aware industrial metrics on VisA and MTD

Question: how do the methods trade off false rejection of good parts against defect escape
in normal-dominant tasks?

- Re-run of the P10 external methods with **identical** code path, manifests, banks and
  seeds, now saving per-query predictions. Integrity: every recomputed (method, task, K,
  seed) BA / Macro-F1 / Top-1 must equal `ckpt/p10_external/results/external_results.json`
  exactly; otherwise the run fails.
- Frozen methods (`ours`, `frozen_fusion_proto`, `mvrec_rgb_adaptation`) use their archived
  per-query predictions from `ckpt/external_query_grid_bundle.tar.gz`.
- Metrics per (task, K, seed), argmax decisions, no threshold: good recall (1 − false-reject
  rate), defect recall (defect predicted as any defect; 1 − escape rate), binary BA,
  defect-type accuracy among detected defects. VisA: object macro; MTD: single task; mean ±
  SD over seeds. Illustration: expected false rejects and escapes per 1,000 parts at 95%
  good prevalence (1000·0.95·FRR and 1000·0.05·escape).
- No verdicts; all methods and K are reported.

## Guards and preservation

- Each stage refuses to run with uncommitted changes in `experiments/p10`,
  `experiments/revision`, or this protocol, takes an exclusive lock directory, and writes a
  single-shot output: `ckpt/p10_cable/results/`, `ckpt/p10_external/normal_aware/`.
- After both stages, an evidence manifest (SHA256 and size of every result file of Revision
  C and P10) is written to `docs/paper/P10_EVIDENCE_MANIFEST.json` (tracked in git), and the
  result files (not the large feature caches) are archived to
  `D:\intergration\evidence_archive\`.
