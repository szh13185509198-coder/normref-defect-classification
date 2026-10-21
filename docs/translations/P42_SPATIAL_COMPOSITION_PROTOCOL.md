<!-- translation-header -->
> English translation of [`P42_SPATIAL_COMPOSITION_PROTOCOL.md`](../P42_SPATIAL_COMPOSITION_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `3162ea99bc52e29ba88a776325950fa5eb46ad3acda8adfea041e6b6765339fa`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P4.2 spatially separated combined evidence: protocol

## 1. Single research hypothesis

P4.1 showed that the absolute strength of the second-strongest defect response mislabels complex `bent_wire` and
`missing_cable` images as Combined. P4.2 tests one new, pre-specified physical hypothesis only:

> The multi-class responses caused by a single defect tend to concentrate on the same patches; the two strongest
> basic-defect responses of a true Combined image are more likely to come from different patch regions.

Apart from this spatial-separation evidence, no share, distance, network layer, threshold objective or other
combination feature is searched.

## 2. Fixed spatial score

The per-patch evidence maps of the seven basic defects relative to the Good bank are computed as before, and the
top-1 and top-2 basic defect classes are determined by their top-10% means. For each of the two evidence maps the
indices of the 10% most responsive patches are taken:

```text
overlap = |TopPatch(top1) ∩ TopPatch(top2)| / K
spatial_distinctness = 1 - overlap
spatial_composition_score = top2_evidence_score × spatial_distinctness
```

At 336 px, `K = ceil(576 × 0.10) = 58`. No segmentation annotation is used, and the 10% is not changed in view of
the results.

## 3. Evaluation and safety boundary

- The outer 8 folds and the inner 7-fold calibration of P4.1 are reused unchanged;
- the threshold still requires an inner combined-vs-single specificity ≥ 90%, with the same selection order;
- arbitration still allows only non-Good, non-Combined raw predictions to become Combined;
- paired with the existing P3.0 S111/S222/S333 OOF results;
- the nine multi-seed acceptance criteria of P4.1 are reused unchanged;
- no training, no segmentation annotations, no formal test, and `formal_test_allowed=false`.

If P4.2 fails, all combined post-processing based on the fixed DINO patch bank on Cable stops and no P4.2b feature
is created. The main line then moves to cross-dataset validation, analysis of the data limits, or a trained set
representation supported by additional data, rather than searching further rules on the same 72 samples.

## 4. Results (2026-08-09)

P4.2 is rejected. The spatial-separation score produced the same 5 overrides for all three seeds: one Combined image
corrected, but four correctly classified single defects harmed (1 bent, 1 cut-inner, 2 missing-cable). The mean
nine-class BA falls from `80.56%` to `76.39%` (-4.17 pp), the mean Macro-F1 from `76.53%` to `75.33%`, and the
combined recall rises by only 12.5 pp, short of the 25 pp criterion.

Compared with P4.1 the spatial evidence does recognise one more Combined image, so the difference between response
regions has local mechanistic meaning; but under the specificity constraint it still cannot offset the wrong
overrides of single defects, and the largest recall loss of a non-combined class exceeds 12.5 pp. The SHA-256 of the
rerun outer evidence is identical.

The pre-specified stopping rule applies: no P4.2b is created and no further combined rules are searched on the
fixed DINO patch bank. The common conclusion of P4.0, P4.1 and P4.2 is that a local combination signal exists but,
with 8 shots, is not sufficient for a safe scalar post-processing.
