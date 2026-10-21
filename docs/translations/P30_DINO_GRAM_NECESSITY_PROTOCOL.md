<!-- translation-header -->
> English translation of [`P30_DINO_GRAM_NECESSITY_PROTOCOL.md`](../P30_DINO_GRAM_NECESSITY_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `a7dcafc44ceea10d58e24ceb8cd10c0419cf67ce07f029778c6fa6b6b69a228f`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P3.0 necessity of the current DINO Gram pooling: three-seed OOF protocol

Date: 2026-08-09
Status: three-seed OOF completed and audited; no selection used the formal test.

## 1. Question and single variable

The current `dino_pool=gram` concatenates `[dino_avg, gram_max, gram_mean]`, but about `99.9993%` of the measured
energy comes from `dino_avg`; the other two parts are nearly zero. The earlier single-seed formal ablation was also
confounded by backend, initialisation and test feedback, so it cannot show an independent contribution of the Gram
part.

P3.0 keeps the P2.0 clean terminal-only backbone, CLIP top-K, 336 px, Hub DINO, losses, epochs, augmentation,
three seeds and 8 folds fixed, and changes only `dino_pool` from `gram` to `mean`. The baseline reuses the
top-K + Gram OOF results of P2.0 S111 and P2.1 S222/S333 without retraining.

## 2. Pre-specified criteria

For each seed, top-K + mean must satisfy:

1. defect recall ≥ 95%;
2. good recall at most 12.5 percentage points (one of 8 Good images) below Gram of the same seed;
3. nine-class BA at most 2 percentage points below Gram of the same seed.

Over the three seeds, also:

- all three seeds satisfy the safety and non-inferiority criteria above;
- the mean nine-class BA is not below Gram;
- the mean binary BA is not below Gram.

If all hold, the decision is `drop_unscaled_gram_and_advance_to_topk_ablation`; otherwise
`retain_gram_provisionally_and_audit_scale_before_claiming_contribution`. The second outcome would only mean that
the current concatenation cannot simply be removed; it would not show that the low-energy Gram parts carry
second-order information.

## 3. Execution and archiving

Run `P3.0 3X OOF Multi-Seed | Top-K + DINO mean`: the three seeds start together, each with 2 folds in parallel, at
most 6 processes in total. When all three `p30_run_complete.json` files exist, run
`P3.0 Audit | DINO mean vs Gram, 3 seed OOF`.

The output directory is fixed to `ckpt/p30_dino_mean_oof/s{seed}`. By default the runner reuses only complete folds
and verifies the checkpoint, metadata, prediction and aggregate hashes of each. P3.0 does not read formal test
predictions; until the audit is complete no variant may be added and no criterion changed in view of the results.

## 4. Place in the overall plan

P3.0 is the first minimal necessary ablation before architectural improvements, not a full grid search over
representations. After the audit, DINO pooling is frozen by the pre-specified decision: mean if it passes;
otherwise Gram is kept provisionally and its scale audited separately, and a scale audit would not by itself make
Gram a contribution of the paper.

Two backbone attributions follow: P3.1 compares the current per-channel CLIP top-K with the plain mean over the same
`CLS + patch` token set, and P3.2 compares the current multi-view initialisation with single-view. After these three,
work moves to the P4.0 local defect-evidence branch and P4.1 combined-semantics modelling; the remaining earlier
ablations become conditional side branches. P3/P4 development uses OOF only, and no formal test is added before the
single final candidate is frozen.

## 5. Results and decision (2026-08-09)

All 24 folds of S111, S222 and S333 completed; each of the three aggregates holds 72 distinct OOF predictions. The
audit file is `ckpt/p30_dino_mean_oof/p30_dino_mean_vs_gram_audit.json` and records `formal_test_used=false`. The
216 final class predictions of mean and Gram are identical for all three seeds:

| Metric (mean over three seeds) | Current Gram | DINO mean | Difference |
|---|---:|---:|---:|
| Nine-class BA | 80.56% | 80.56% | 0.00 pp |
| Binary BA | 88.28% | 88.28% | 0.00 pp |

All paired `changed_predictions` are empty. The cosine scores are not bit-identical, but their mean absolute change
is only about `7.3e-5 to 1.1e-4`, too small to change any class decision, which agrees with the near-zero energy of
the two Gram parts. The pre-specified decision `drop_unscaled_gram_and_advance_to_topk_ablation` therefore applies:
the candidate backbone drops the current unscaled Gram and uses the simpler 384-d DINO mean. This rejects only the
necessity of the current raw Gram implementation; it does not extend to covariance or Gram representations in
general.
