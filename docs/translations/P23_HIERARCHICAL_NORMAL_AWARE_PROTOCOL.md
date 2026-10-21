<!-- translation-header -->
> English translation of [`P23_HIERARCHICAL_NORMAL_AWARE_PROTOCOL.md`](../P23_HIERARCHICAL_NORMAL_AWARE_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `78ae19ffdccb8403a8dc058d75d7589f15657e00699ee50ea7bdb061d9fd5479`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P2.3 hierarchical normal-aware branch: protocol

Date: 2026-08-08
Status: code and OOF execution configuration implemented; the 8-fold cloud runs have not been started; the formal
test is not to be used.

## 1. Why P2.3

The three-seed OOF results of P2.0/P2.1 showed that the binary boundary between Good and Poke is the main source of
false rejects and escapes of the current system, whereas the main problem of Combined is the semantic
decomposition into defect sub-classes, not being judged Good. The fixed threshold `tau=0.006` of P2.2 passed the
pre-specified acceptance rule on the formal test but produced only small corrections, which shows that shifting a
prototype score threshold has limited expressive power.

P2.3 therefore does not replace the existing nine-class branch but separates two questions explicitly:

1. the first level decides `Good` or `Defect`;
2. only when the first level decides `Defect` do the eight defect prototypes of the original nine-class prototypes
   decide the defect type.

This matches the industrial goal: control false rejects of good parts and escapes of defects first, then the
defect sub-class. It also keeps the defect ranking of the original model, which is already strong.

## 2. Fixed architecture and training rules

- Binary head: `Linear(1920, 1)`, 1,921 parameters, applied to the L2-normalised fused feature after the terminal
  `det_proj`.
- Initialisation: weights and bias all zero; the construction isolates the CPU RNG, so the random state of existing
  modules and of the rest of the pipeline is unchanged.
- A positive output means Defect, a non-positive output Good; the decision threshold is fixed at `0.0` before the
  experiment, with no threshold search.
- Defect sub-classification: the highest cosine score among the eight non-Good prototypes; the binary head does not
  learn the nine-class problem.
- Binary loss: weighted BCE. With `K` classes, a Good sample has weight `K/2` and each defect sample weight
  `K/[2(K-1)]`. For Cable the weights are `4.5` and `0.5625`, so that the Good class and the eight defect classes
  carry equal total weight, without changing the existing sampler or its trajectory.
- Optimizer: the head learning rate is fixed at 10 times the base LR; the terminal `det_proj`, the prototypes and
  the original nine-class loss keep the P2.0 configuration.
- P2.3 allows only `clean_v1 + isolated_rng + terminal_only + pure vision`; text guidance, the cascade, prototype
  text refinement and the P2.2 fixed threshold may not be enabled at the same time.
- Checkpoints use `lightweight_v1`; only the last epoch of each fold is saved.

## 3. Pre-specified candidates and selection rule

The first stage compares two candidates only, on the fixed 8-fold OOF split of seed 111:

- `w010`: nine-class loss + `0.1 × balanced BCE`
- `w030`: nine-class loss + `0.3 × balanced BCE`

No other weight is added. A candidate must satisfy all of:

1. OOF defect recall of at least 95%;
2. good recall strictly higher than P2.0 S111;
3. binary balanced accuracy strictly higher than P2.0 S111;
4. nine-class BA no more than 2 percentage points below P2.0 S111.

If both pass, select by binary BA, then good recall, then nine-class BA; if still tied, the smaller loss weight.
If neither passes, P2.3 has failed: the branch stops, and the formal test may not be accessed in search of a better
result.

## 4. Cloud execution order

`launch.json` contains:

1. `P2.3 2X OOF S111 | hierarchical w=0.1+w=0.3`: runs the two candidates in parallel when GPU memory allows;
   each candidate completes the training, validation and aggregation of its 8 folds in turn;
2. `P2.3 Audit S111 | select w=0.1 vs w=0.3`: runs after both candidates and applies the pre-specified acceptance
   and selection rule mechanically.

Each candidate can also be run alone with its `P2.3 OOF S111` configuration. If a cloud run is interrupted, add
`--skip_completed` to the launch arguments; the runner keeps only checkpoints or prediction files that were
completed together and continues the missing steps. Folds from other experiment directories may not be copied in as
completed.

## 5. Deliverables and next step

The whole `ckpt/p23_hierarchical_oof_s111` is returned, with at least:

- for each candidate and each of the 8 folds: `run_manifest.json`, training/OOF logs and prediction JSONL;
- both `oof_predictions_all.jsonl` files with their manifests;
- `p23_screening_audit.json`.

Only if a candidate passes on S111 is the single selected weight extended unchanged to the S222 and S333 OOF runs.
Whether to proceed to the formal test is locked separately, and only after all three seeds pass; the current code
actively prevents P2.3 from running on the non-OOF formal test.

P2.2 is a completed, independently locked fixed-threshold result. P2.3 is a new experiment with a learnable
structure; its failure does not overturn P2.2, and the P2.2 formal test result may not be used to adjust P2.3.
