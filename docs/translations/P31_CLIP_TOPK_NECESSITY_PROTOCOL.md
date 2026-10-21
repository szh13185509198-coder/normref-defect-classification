<!-- translation-header -->
> English translation of [`P31_CLIP_TOPK_NECESSITY_PROTOCOL.md`](../P31_CLIP_TOPK_NECESSITY_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `9660b6cf5dd6fb2832dfd5b577636cc0436fb0ad3ba3c7d8d6e5ae37c56236c0`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P3.1 necessity of CLIP top-K pooling: tiered OOF protocol

Date: 2026-08-09
Status: tier A (S111) completed and the mean variant rejected; P3.1 stopped; no selection used the formal test.

## 1. Question and single variable

P3.0 removed the raw Gram part, which had no observable benefit; the current reliable backbone is
`CLIP top-K + DINO mean`. P3.1 changes only `clip_pool` from `topk` to `mean`; training, DINO, random streams,
metadata, epochs and inference stay the same.

In the current code both modes use the same `CLS + patch` token set: `topk` averages, per channel, the 10% of tokens
with the highest values, and `mean` averages all tokens. The experiment therefore asks only whether the current
per-channel top-K gives a stable decision benefit; it is not a selection of a single spatial anomaly region, and no
patch-only implementation is introduced at this stage.

## 2. Tier A: full eight-fold screen on S111

The full eight folds of S111 are run first, covering all 72 training samples, with the eight folds in parallel, and
paired sample by sample with the P3.0 S111 top-K results. CLIP mean must satisfy all of:

1. defect recall ≥ 95%;
2. good recall at most 12.5 percentage points below top-K;
3. nine-class BA at most 2 percentage points below top-K;
4. binary BA at most 6.25 percentage points below top-K.

If all hold, the decision is `advance_clip_mean_to_s222_s333`. Non-inferiority criteria are used because the mean is
simpler; tier A only removes clearly failing candidates and cannot by itself support a multi-seed conclusion.

## 3. Tier B: confirmation on S222/S333

Only if tier A passes are S222 and then S333 run, each with its eight folds in parallel; the two seeds are not
started together as 16 trainings. `drop_topk_and_adopt_all_token_clip_mean` applies only if every seed passes the
tier-A safety criteria and the mean nine-class BA and binary BA over the three seeds are not below top-K; otherwise
`retain_topk_as_p3_backbone`. The formal test is not run in either case.

## 4. Execution order

1. `P3.1 Tier A S111 | CLIP mean + DINO mean`
2. `P3.1 Tier A Audit | S111 screen`
3. only if the report allows: `P3.1 Tier B S222 | CLIP mean + DINO mean`
4. then `P3.1 Tier B S333 | CLIP mean + DINO mean`
5. `P3.1 Tier B Audit | final 3-seed decision`

Outputs are fixed to `ckpt/p31_clip_mean_oof/s{seed}`. The runner reuses only completed folds whose hashes, protocol
and metadata all match; the audits read only the within-training OOF results.

## 5. Tier A result (2026-08-09)

On S111 the top-K baseline reached a nine-class BA of 80.56% and the all-token mean 77.78%; Macro-F1 fell from
77.43% to 74.80%, good recall from 75% to 62.5% and binary BA from 85.94% to 81.25%. The mean raised defect recall
from 96.88% to 100% but lost two correct nine-class predictions net: two `cable_swap` images and one Good image that
had been correct were harmed, only one Poke was corrected exactly, and another Poke was corrected only in the binary
decision.

Because the nine-class BA fell by 2.78 pp, more than the pre-specified 2 pp, the decision is
`retain_topk_and_stop_p31_after_s111_screen`, and S222/S333 are not run. The result suffices as an engineering
screen for keeping top-K, but a single seed cannot support a final claim that top-K gives a stable improvement; if
top-K were to be presented as a core contribution after the final model is frozen, a paired multi-seed ablation
would still be needed.
