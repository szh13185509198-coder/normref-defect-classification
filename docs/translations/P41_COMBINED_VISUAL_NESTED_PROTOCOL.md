<!-- translation-header -->
> English translation of [`P41_COMBINED_VISUAL_NESTED_PROTOCOL.md`](../P41_COMBINED_VISUAL_NESTED_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `39af757934e00aee6af427a7a650dfeaa6e3b20a37283a0bee5064895fc3039b`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P4.1 combined vision-only nested arbitration: protocol

## 1. Research question

P4.0 showed that the "second-strongest basic defect patch evidence" ranks combined against single defects with an
AUROC of 0.7433 within the training data. P4.1 tests whether this evidence, after strictly nested calibration, can
safely improve the recognition of combined defects by the existing three-seed global classifier.

P4.1 still trains no neural network and uses no text, segmentation annotation or formal test. It is a low-cost
screen of structural feasibility; a trained set representation would be discussed only after it passes.

## 2. Outer evaluation and inner calibration

- The outer level keeps the fixed 8 folds: 63 training images and 9 held-out images per fold;
- each outer training set is split again, in within-class path order, into 7 folds: 54 images build the patch bank
  and 9 produce inner calibration evidence each time;
- every outer fold thus has 63 inner OOF evidence values that never use the outer held-out images;
- the threshold is chosen only on the 7 combined and 49 single-defect images among them; Good does not take part in
  the combination threshold;
- the 9 outer images are scored against a bank built from all 63 outer training images, with the threshold locked
  for that fold.

This avoids choosing an optimal threshold directly on the 72 outer results of P4.0, and keeps the banks of other
folds from containing the current outer held-out images indirectly.

## 3. Fixed score and threshold algorithm

The local evidence is that of P4.0: for each of the seven basic defects, the top-10% patch evidence relative to the
Good bank; the combination score is fixed as the second-largest of the seven class scores.

Threshold candidates are all distinct inner OOF scores plus a conservative "above the maximum" candidate. Only
candidates with combined-vs-single specificity ≥ 90% are considered; binary balanced accuracy is maximised first,
and ties are broken by higher specificity, then higher sensitivity, then higher threshold. This order may not be
changed after the run.

## 4. Conservative arbitration

For the raw P3.0 OOF predictions of each seed:

```text
if raw_prediction is neither Good nor Combined
   and composition_score >= outer_fold_threshold:
       final_prediction = Combined
else:
       final_prediction = raw_prediction
```

Hence:

- a raw Good prediction never changes;
- a raw Combined prediction is never downgraded;
- good recall and defect recall must stay unchanged sample by sample;
- the only risk is relabelling a single defect as Combined by mistake.

## 5. Pre-specified three-seed acceptance

Paired with P3.0 S111/S222/S333; the next structure may follow only if all of these hold:

1. good recall, defect recall and the binary predictions of every seed are unchanged sample by sample;
2. the nine-class BA of every seed is not below its baseline;
3. the nine-class BA improves strictly on at least two seeds;
4. the mean nine-class BA over the three seeds rises by at least 2.00 percentage points;
5. the worst-seed nine-class BA of the candidate is not below the worst seed of the baseline;
6. the sample standard deviation of the nine-class BA across seeds increases by at most 0.50 percentage points;
7. the mean Macro-F1 over the three seeds does not fall;
8. the mean combined recall over the three seeds rises by at least 25 percentage points;
9. no non-combined defect class of any seed loses more than 12.5 percentage points of recall (at most 1 of 8).

After a failure, the specificity, the top-patch share, the threshold objective and the override condition are not
adjusted on the current results. A failure may be reported as the mechanistic finding that "local evidence exists
but a simple threshold cannot exploit it safely".

## 6. Outputs and boundary

Outputs are fixed to `ckpt/p41_combined_visual_nested`, with the nested thresholds, the per-sample paired
predictions of the three seeds, the input hashes and the overall audit. Whatever the result,
`formal_test_allowed=false`; P4.1 cannot trigger a formal test.

## 7. Results (2026-08-09)

P4.1 is clearly rejected. The changes are identical for the three seeds: each relabels 4 correctly classified
single defects as Combined and corrects no Combined image. The mean nine-class BA falls from 80.56% to 75.00%
(-5.56 pp), the mean Macro-F1 from 76.53% to 73.12%, and the combined recall gain is 0.

The harmed samples are 3 `bent_wire` and 1 `missing_cable` images; `bent_wire` recall falls by up to 37.5 pp. The
outer scores of all 8 true Combined images are below their inner thresholds. The Good/Defect decision is unchanged
sample by sample, which shows that the safe-override code works as designed; the fine-grained arbitration itself
fails.

Mechanism: the AUROC of 0.7433 in P4.0 reflects the overall ranking, while the second-strongest evidence of
`bent_wire` and `missing_cable` overlaps strongly with Combined. The stricter high-specificity inner calibration
exposes the instability of this scalar across outer folds. The SHA-256 of the rerun outer evidence is identical,
which rules out run-to-run drift.

The stopping rule applies: the specificity is not lowered and the threshold not adjusted on this result. The simple
"second-strongest score + threshold" does not enter a trained architecture or the formal test. Any further work on
Combined must first validate new spatial evidence, for example whether the responses of two classes come from
different patch regions, rather than tuning the same scalar.
