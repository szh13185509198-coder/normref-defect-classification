<!-- translation-header -->
> English translation of [`P32_MULTIVIEW_INIT_PROTOCOL.md`](../P32_MULTIVIEW_INIT_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `4c8238f15eca4a321c280867a393d380488898f686a6884d25a6eb89eabcc660`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P3.2 multi-view prototype initialisation: tiered OOF protocol

Date: 2026-08-09
Status: tier A (S111) completed and 5-view rejected; P3.2 stopped; no selection used the formal test.

## 1. Actual baseline and single variable

The clean backbone of P3.0/P3.1 actually uses `num_init_views=1`, so multi-view initialisation is not a validated
component of the current system. P3.2 keeps CLIP top-K, DINO mean, terminal-only, 20 epochs, repeat = 4, three
independent seeds and the immutable 8 folds fixed, and changes only the number of views used for the class-mean
initialisation of the prototypes from 1 to 5.

Each view uses its own order and per-sample augmentation seed derived from `prototype_init/view_idx`; the training
itself still uses the `train/epoch/repeat` namespace, so adding initialisation views does not change the data order
or augmentation trajectory of training. What changes is whether the initial prototype of each class is the mean of
`7×1` or `7×5` augmented features.

## 2. Tier A screen on S111

The eight folds of S111 run in parallel and are paired sample by sample with the one-view P3.0 S111 results. The
variant must satisfy:

1. defect recall ≥ 95%;
2. good recall at most 12.5 percentage points lower;
3. nine-class BA at most 2 percentage points lower;
4. a strict improvement in nine-class BA or binary BA.

Only then is `advance_multiview5_to_s222_s333` issued. Multi-view adds initialisation cost, so "no change" is not
enough to extend it.

## 3. Tier B three-seed decision

After tier A passes, S222 and then S333 run, each with its eight folds in parallel (never 16 at once). Final
acceptance requires:

- all three seeds pass the safety criteria;
- the mean nine-class BA over the seeds is not below one-view;
- the mean binary BA over the seeds is not below one-view;
- the worst seed of nine-class BA and of binary BA is not below one-view;
- the sample standard deviation across seeds of both metrics is not above one-view;
- nine-class BA or binary BA improves strictly on at least two seeds.

If accepted, 5-view is adopted; otherwise 1-view is kept. The formal test is not run in either case.

## 4. Run order

1. `P3.2 Tier A S111 | Multi-View5 prototype init`
2. `P3.2 Tier A Audit | S111 screen`
3. only if allowed: the S222 and S333 tier B configurations, in turn
4. `P3.2 Tier B Audit | final 3-seed decision`

## 5. Tier A result (2026-08-09)

On all 72 OOF samples of S111 the final class predictions of 5-view and 1-view are identical: nine-class BA 80.56%
and binary BA 85.94% for both, with the same per-class recalls and the same set of errors. 5-view was not inactive:
the cosine scores changed by about `0.00146` on average and at most about `0.00727`, the final training loss was
slightly lower in all 8 folds, and the final `det_proj` and prototypes differ numerically. None of these changes
moved a sample across a decision boundary.

Because the pre-specified rule requires a strict improvement in nine-class BA or binary BA, the decision is
`retain_one_view_and_stop_p32_after_s111_screen`. S222/S333 are not run and the backbone keeps
`num_init_views=1`. The conclusion is limited to the current augmentation strength, 20 epochs and the trainable
prototype protocol: the extra initialisation views are largely absorbed by the subsequent optimisation. It does not
extend to multi-view training strategies in general.
