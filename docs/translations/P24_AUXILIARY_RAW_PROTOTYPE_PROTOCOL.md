<!-- translation-header -->
> English translation of [`P24_AUXILIARY_RAW_PROTOTYPE_PROTOCOL.md`](../P24_AUXILIARY_RAW_PROTOTYPE_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `192f59916657ec1c5e3fd14de50aa365f8657ccd81110a1449b15cae10e785e0`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P2.4 auxiliary binary training + raw visual prototype inference: protocol

Date: 2026-08-09
Status: the three-seed OOF runs are complete and the branch was rejected by the pre-specified criteria; the P2.4
formal test is not to be run.

## 1. Rationale

The hard-override branch of the P2.3 binary head was rejected. The raw visual prototype branch of P2.3 reached a
nine-class BA of `81.94%` and a defect recall of `98.44%`; but with `w=0.1/0.3` the hard override corrected only one
Good image while damaging 8/6 defect images that had been classified correctly, lowering defect recall to
`87.50%/90.625%`. The failure came from the unconditional hard override at inference, not from a training collapse.

P2.4 keeps the `w=0.1` balanced BCE as an auxiliary training objective that only shapes the terminal features; at
inference the output of the binary head is neither computed nor read, and the learned visual class prototypes with
FAISS cosine retrieval are used unchanged. This protocol is not a text-prototype inference and claims no strict
zero-shot property.

## 2. Fixed training and inference configuration

- `clean_v1 + isolated_rng + terminal_only`;
- CLIP top-K, DINO Gram, 336 px resolution, Hub DINO;
- `normal_aware_head=true`, balanced BCE weight fixed at `0.1`;
- head LR multiplier fixed at `10.0`; the threshold placeholder in the training record fixed at `0.0`;
- inference mode fixed to `normal_aware_auxiliary_raw_prototype`;
- every OOF record must satisfy `predicted_class_id == raw_predicted_class_id`;
- the sidecar must state that the head took part in training but not in inference.

## 3. OOF and acceptance criteria

Seeds `111/222/333` are run, each with its immutable 8-fold leave-one-per-class-out split. Each seed is compared
with its P2.0/P2.1 baseline and must satisfy all of:

1. defect recall ≥ 95%;
2. good recall not below the baseline;
3. binary balanced accuracy strictly above the baseline;
4. nine-class balanced accuracy at most 2 percentage points below the baseline.

Over the three seeds, the mean binary BA must also rise strictly, the mean nine-class BA must not fall, and every
seed must pass its own criteria. The final decision can only be:

- `advance_p24_auxiliary_to_full_8shot_training`; or
- `reject_p24_auxiliary_and_keep_p22_as_primary`.

## 4. Concurrency and resumption

`P2.4 3X OOF Multi-Seed | w=0.1` starts the three seed runners at once; each runs 2 folds at a time, so at most 6
training/validation processes run globally. Every fold writes to its own directory; no checkpoint is shared. A seed
is aggregated only after all 8 folds have finished and passed the per-fold hash check.

The launch configuration includes `--skip_completed`. An existing fold is reused only if its checkpoint, metadata,
protocol, sidecar and per-sample raw constraint all match; any mismatch reruns the fold.

## 5. Formal test lock

The P2.4 OOF runner and evaluator do not read formal test predictions. The aggregator reads only the formal test
metadata paths, for the leakage audit (empty intersection). A full 8-shot training would be pre-specified
separately, and only after the OOF decision passes and the lead reviewer has checked it; the P2.4 formal test is
not to be run now.

## 6. Final result and decision (2026-08-09)

| Seed | Change in binary BA | Change in nine-class BA | Final prediction changes | Seed criteria |
|---:|---:|---:|---|---|
| 111 | +0.78 pp | +1.39 pp | one Poke corrected | passed |
| 222 | 0 | 0 | none | no strict improvement |
| 333 | 0 | 0 | none | no strict improvement |

Over the three seeds the mean binary BA rose from `88.281%` to `88.542%` and the mean nine-class BA from `80.556%` to
`81.019%`; but only `poke_insulation/train/images/003.png` of S111 crossed the boundary, its margin moving from
`-0.00065` to `+0.00553`. The cosine scores of S222/S333 changed by non-zero amounts, but all 144 predictions stayed
the same.

The final decision is `reject_p24_auxiliary_and_keep_p22_as_primary`. P2.4 shows that the auxiliary BCE is a safe
but weak representation regulariser that is unstable across seeds; the damage in P2.3 came mainly from the hard
override at inference, not from the auxiliary training. P2.4 does not proceed to full 8-shot training or to the
formal test, and P2.2 remains the primary branch.

This round is a diagnostic negative branch: all 24 manifests carry the same commit, but two stable diff
fingerprints arose during resumption, and the local training logs of S111 fold00/01 are to be replaced by the
original cloud files. This does not change the rejection, but these runs cannot serve as evidence in a final
positive table.
