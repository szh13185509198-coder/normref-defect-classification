<!-- translation-header -->
> English translation of [`P40_LOCAL_PATCH_EVIDENCE_DIAGNOSTIC.md`](../P40_LOCAL_PATCH_EVIDENCE_DIAGNOSTIC.md), which is kept unchanged as the committed record (SHA-256 `aac54595c84dd0428891eb0f1ccf0b5fe3b5f18d709b9430644624d9ff0fd4cd`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P4.0 frozen local patch evidence: diagnostic protocol

## 1. Purpose

P4.0 trains no model and does not propose a final module for the paper. It answers two prior questions only:

1. Do frozen DINOv2 patch features contain local evidence that separates `good` from small surface defects
   (`poke_insulation`, `cut_inner_insulation`, `cut_outer_insulation`)?
2. Can the "second-strongest basic defect evidence" of a sample separate `combined` from single defects using
   vision alone?

Only if the diagnostic passes may P4.1 implement and train the corresponding branch. This avoids another costly
3-seed × 8-fold training whose input evidence is insufficient.

## 2. Data boundary

- Only image source: the 72 training images of `dataset/metadata/MVTecAD-Cable/8-shot.jsonl`;
- validation split: the locked `oof_8fold_v1/fold_XX_train.jsonl` and `fold_XX_val.jsonl`;
- each fold builds the patch bank from its 63 training images and computes evidence for its 9 held-out images;
- after merging the 8 held-out folds, every training image is diagnosed exactly once;
- segmentation annotations are not read; the segmentation fields of the metadata are explicitly ignored;
- `test-shot.jsonl`, formal test images and their predictions are not read;
- the 72 OOF predictions of P3.0 S111 are used only to compare the global boundary with the local evidence and do
  not enter any feature bank.

P4.0 is therefore an OOF diagnostic inside the training set, not a formal test, and produces no new test score.

## 3. Fixed visual features

- Backbone: frozen local `dinov2_vits14`;
- weights: `pretrained/dinov2_vits14_pretrain.pth`;
- patch policy: `full`, 576 spatial tokens at 336 px;
- input: fixed resize (336×336) and the CLIP normalisation used throughout the project;
- no back-propagation, no optimizer, no checkpoint;
- the patch features of the 72 images are computed once and cached; the cache records the hashes of the metadata,
  images, DINO weights and local hub code.

## 4. Pre-specified local evidence

For a query patch `p` and the patch bank `B_c` of class `c`:

```text
sim_c(p) = max cosine(p, b), b in B_c
e_c(p)   = sim_c(p) - sim_good(p)
S_c(x)   = mean of e_c(p) over the 10% of patches of the query image with the highest e_c(p)
```

The 10% share is fixed before the run and may not be changed in view of the results.

- Local binary score: the maximum of `S_c(x)` over all defect classes;
- small-defect diagnostic: AUROC computed only on `good` and `poke/cut_inner/cut_outer` samples;
- combined evidence: after removing the `good` and `combined` banks, the second-largest `S_c(x)` of the seven basic
  defect classes;
- non-redundancy: Pearson correlation between the local binary score and the global
  `defect_minus_good_margin` of P3.0.

P4.0 chooses no classification threshold from labels and does not fuse local and global scores after the fact.

## 5. Pre-specified pass criteria

The two candidate branches are decided independently.

### P4.1-Micro: local small-defect branch

All of:

- local-evidence AUROC over all Good/Defect samples ≥ 0.75;
- AUROC on Good versus the three small-defect classes ≥ 0.70;
- absolute Pearson correlation between the local score and the existing global margin ≤ 0.95.

Passing means only that a usable and not fully redundant local signal exists; the fusion would still have to be
pre-specified and validated again by OOF.

### P4.1-Combined: vision-only combination branch

- AUROC of the second-strongest basic defect evidence for combined versus the single defects ≥ 0.65.

Only then may a vision-only multi-defect evidence aggregation be developed; it is not text-semantic modelling.

## 6. Outputs and stopping rule

The output directory is fixed to `ckpt/p40_local_patch_diagnostic` and contains at least:

- the frozen patch-feature cache and its manifest;
- the 72 OOF local-evidence records (JSONL);
- the audit JSON with input hashes, metrics, criteria and each decision.

If both branches fail, local arbitration structures are no longer stacked on Cable; work moves to validation on
further datasets and to the paper experiments. No result of P4.0 may be used to readjust the fixed threshold of
P2.2.

## 7. Results (2026-08-09)

The local RTX 4050 completed the feature cache and the 8-fold diagnostic for all 72 images without training. A
second run reused the provenance-checked cache; the SHA-256 of the 72 evidence records was identical in both runs.

| Diagnostic | Result | Pre-specified criterion | Decision |
|---|---:|---:|---|
| Local AUROC, all Good/Defect | 0.8809 | ≥ 0.75 | pass |
| Local AUROC, Good vs Poke/Cut | 0.8333 | ≥ 0.70 | pass |
| Correlation of the local score with the P3.0 global margin | 0.6720 | `abs(r) ≤ 0.95` | pass |
| Second-evidence AUROC, combined vs single defects | 0.7433 | ≥ 0.65 | pass |

For reference, the global margin of P3.0 reaches an AUROC of 0.9844 over all Good/Defect samples and 0.9635 for
Good vs Poke/Cut, both higher than the local evidence alone. P4.0 therefore **does not show that a local branch
should replace the global classifier**; it shows that the local evidence has moderate discriminative power and is
not fully redundant with the global score.

The combined result is likewise a ranking signal, not a ready classifier. The mean second-strongest basic evidence
of `combined` is 0.0670, but `bent_wire` and `missing_cable` reach 0.0922 and 0.0746, a clear overlap. P4.1 must
use nested calibration inside the training data and conservative arbitration; no optimal threshold may be picked on
these 72 results.

Final decision: both evidence branches may proceed to P4.1. The combined vision-only arbitration comes first,
because it addresses the main structural limit of the current nine-class BA; the micro branch is second and may
only serve as a constrained complement to the global decision.
