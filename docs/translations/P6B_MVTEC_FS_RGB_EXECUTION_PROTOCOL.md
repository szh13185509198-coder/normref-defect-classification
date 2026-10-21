<!-- translation-header -->
> English translation of [`P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md`](../P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `380463878c33f6a1c20320eda3059a445a6b4b5e772365e6453a7e1c3b45b0ae`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P6.B MVTec-FS 13-Product RGB-Only Method Transfer Execution Protocol

> Status: **PRE-SPECIFIED — LOCKED FOR EXECUTION** (upgraded from DRAFT by ruling of the lead reviewer; all blocking
> items closed; the stage-1 production call chain and its tests were independently re-verified).
> This protocol is pre-specified and locked: any change requires a new protocol version and a new ruling.
> Execution must follow §2/§3/§5/§6/§7/§8 strictly. The stage-1 shared infrastructure is implemented and accepted
> (execution tests 15/15, feasibility 23/23, P3.F 15/15, py_compile and git diff --check all pass); stage-2 model
> implementation proceeds only after the formal manifest bundle (completed / formal_eligible=true / dirty=false)
> has passed its audit.
> This file is not the source of research facts; all ruled facts follow the rulings of the lead reviewer and
> `docs/P6B_MVTEC_FS_RGB_FEASIBILITY_RULING.md`.

> **Addendum A (PRE-SPECIFIED — LOCKED, pre-execution)**: `docs/P6B_EXECUTION_ADDENDUM_A.md` locks the mathematical
> semantics of the baseline implementations; it only removes implementation ambiguities and changes no product, K,
> seed, budget or metric. Stage-2+ evaluation artifacts must record the SHA-256 of both the main protocol and
> addendum A; any mismatch is fail-closed.

## 1. Task definition (locked)

- P6.B is an **RGB-only few-shot fine-grained multiclass method transfer**.
- It is **not anomaly detection**: MVTec-FS is a defect-only task without the Good/Defect decision of Cable; no
  section of this document may be written as anomaly detection, nominal-vs-anomalous or good-recall semantics.
- Each product is an **independent N-way defect classification task** (N = number of official defect classes of the
  product); **label spaces are not mixed across products**.
- Cable assets explicitly **not transferred**: `tau=0.006`, the P2.2 normal-aware boundary, the Cable checkpoint, the
  Cable label names and the Good class.
- No ground-truth masks, bounding boxes or pixel annotations are used; Pixel-AUROC and PRO are not used.
- Queries are used only for the final inference/evaluation and may not take part in checkpoint, epoch, threshold,
  architecture or hyperparameter selection (see §8).

### 1.1 Scope (from the feasibility ruling)

- The feasibility probe ran on the clean commit `ac76ac8` with `formal_test_used=false`; ruling
  `INCONCLUSIVE-overall / 13-products-feasible`; v2 evidence in `data/p6b_probe_output_v2/`.
- **Fixed 13 products**: bottle, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, transistor,
  wood, zipper.
- `cable` is excluded in advance because it overlaps with the Cable research assets (not for performance reasons).
- `toothbrush` fails locally because 30 images referenced by the official CONFIG are absent from the complete merged
  12-part archive and its extracted directory; the exclusion rests on **asset integrity**, not model performance.

## 2. Support sampling protocol

### 2.1 Shots, seeds, run counts

- K ∈ {1, 3, 5}; five pre-defined draw seeds: `111, 222, 333, 444, 555`.
- The sampling unit is the **unique original RGB image** defined by the feasibility probe (records folded by
  (product, RGB_SHA256)); images are drawn **without replacement** only from the **single-type compliant candidates**
  of the corresponding product, class and support pool.
- **Run counts (locked)**: per product × method = 3 K × 5 paired pipeline seeds = **15 runs**; per method in total =
  13 products × 3 K × 5 seeds = **195 runs**.
- The main scheme is a **paired pipeline**: one seed controls both the support draw and the training randomness
  (§2.5). A 5×5 Cartesian expansion draw_seed × init_seed within a K is **not allowed**, nor is any repetition beyond
  the 5 paired seeds.

### 2.2 Locked sampling mechanics

For each (product, seed, class):

1. **Candidate ordering**: the single-type compliant candidates are sorted by **`rgb_sha256` in lexicographic
   order**, giving the index sequence `indices = [0, ..., n-1]`.
2. **sub_seed**: `sub_seed = int.from_bytes(SHA256("p6b_draw_v1|{base_seed}|{product}|{class}").digest()[:8], "big")`
   (the first 8 bytes of the digest read as a big-endian unsigned integer).
3. **Permutation**: `random.Random(sub_seed).shuffle(indices)` (Python Mersenne Twister).
4. **Nested draws**: K = 1/3/5 take **the first 1/3/5 items of the same permutation**, i.e. K=1 ⊂ K=3 ⊂ K=5, which
   reduces sampling confounding in comparisons across shots.

With these rules every manifest can be recomputed deterministically from (frozen feasibility records + product +
seed + K).

### 2.3 Manifest contract

One support manifest is generated per (product, K, seed):

- `schema_version = "p6b_support_manifest_v1"` (locked).
- Fields: `product`, `K`, `seed`, `draw_scheme = "p6b_draw_v1"`, `classes` (for each official defect class
  `{class_name, items: [{rel_path, rgb_sha256}]}` in draw order), `candidate_ordering = "lexicographic_rgb_sha256"`,
  `rng_algorithm = "python_random_mersenne_twister"`, `support_query_sha_overlap` (audit result, see §8),
  `manifest_sha256`.
- **manifest_sha256 rule (locked)**: the SHA-256 of the UTF-8 bytes of the object **without the `manifest_sha256`
  field**, serialised with the fixed parameters
  `json.dumps(obj_without_sha, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + "\n"` (trailing LF).
  **The sha field never enters its own hash**; it is recomputed and compared before evaluation, and a mismatch is
  fail-closed.

### 2.4 Support/query overlap audit

For every manifest, the intersection of the support and query RGB SHA-256 sets must be empty; otherwise this
configuration of the product is fail-closed (how the audit reads data: §8).

### 2.5 Variance decomposition caveat (locked)

One seed controls both the support draw and the training randomness (P3.F `isolated_rng` semantics:
`seed == init_seed == data_seed == augment_seed`). The five repetitions therefore report only the **joint pipeline
uncertainty**; no separate decomposition into support variance and optimisation variance may be claimed.

## 3. Frozen method transfer (P3.F method-level freeze)

The method-level design and main training hyperparameters of P3.F are frozen; the trainable terminal adapter is
retrained and the prototypes rebuilt on the support set of each product. **The Cable checkpoint is not transferred**;
backbones and feature extractors stay frozen.

### 3.1 Locked fields: P3.F verified source value vs P6.B adopted value

| Field | P3.F verified source value | P6.B adopted value |
|---|---|---|
| img_size | 336 (`--img_size 336`) | adopted: 336 |
| Frozen backbones | CLIP `ViT-L-14-336` (`clipmodel`), DINOv2 `vit-s14` (`dino_extractor`); under clean_v1 all parameters `requires_grad=False` (`configure_model_trainable_parameters()`) | adopted: kept frozen |
| Trainable modules | only `image_adapter.det_proj.*` + `prototype_mem` (prototypes + logit_scale); `seg_proj` is explicitly kept out of the optimizer (`adapter_train_mode=terminal_only`) | adopted: same |
| CLIP branch | `clip_pool=topk` | adopted |
| DINO branch | `dino_pool=mean`, `dino_mask_ratio=0.0`, `dino_patch_policy=full`, `dino_backend=hub`, checkpoint `pretrained/dinov2_vits14_pretrain.pth` (fixed `EXPECTED_DINO_SHA256`) | adopted |
| Retrieval / inference | FAISS cosine, `inference_mode=raw_faiss`, `protocol_check=strict`, `--no_cascade --no_vis` | adopted |
| Prototype construction | `prototypes_per_class=1`; `prototype_init_mode="zero"` (enforced by clean_v1); class-mean initialisation (`init_prototypes_with_class_means`); `num_init_views=1`; `init_logit_scale=10.0` | adopted; prototypes rebuilt on the support set of each product |
| Optimizer | AdamW, `weight_decay=0.01` | adopted |
| Learning rate | trainable model group `lr=1e-4`; prototype group `lr × proto_lr_mult (=3.0) = 3e-4` | **LOCKED**: `lr=1e-4`, `proto_lr=3e-4` (ruling of the lead reviewer, former U1) |
| Scheduler | `CosineAnnealingLR(T_max=num_epochs)` | adopted (structure) |
| Epochs / steps | `num_epochs=20` (uses `multiclass_epoch_19.pth`); `repeat_per_epoch=4` | **LOCKED**: `epochs=20`, `repeat_per_epoch=4` (former U1) |
| Batch construction | `batch_size=8`; `gradient_accumulation_steps=2` (effective 16); `SeededEpochSampler(shuffle=True)` + `make_loader_generator(data_seed)` + `DeterministicWorkerSeeder(augment_seed)` | **LOCKED**: `batch=8`, `grad_accum=2` (former U1); sampler/worker determinism adopted |
| Augmentations | **actual Cable-domain values**: fixed `Resize((336,336), BICUBIC)` + `RandomApply(RandomRotation(3°), p=0.3)` + `RandomApply(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0), p=0.5)` → `ToTensor` → CLIP normalisation; per-sample augmentation isolated by `augment_seed` + `fork_rng` | **LOCKED to the existing non-cable branch** (former U2): `RandomResizedCrop(336, scale=(0.7,1.0))` + `RandomApply(RandomAffine(degrees=15, translate=(0.1,0.1)), p=0.3)` + the same ColorJitter (p=0.5) → `ToTensor` → CLIP normalisation. This branch is a **pre-existing code path** (MVTec-FS products do not carry the cable name and reach it by default through the data path); it is **not a query-driven adjustment**; per-sample isolation keeps `augment_seed` + `fork_rng` |
| Loss weights | `label_smoothing=0.1`; `supcon_weight=0`; `orth_weight=0`; `itc_weight=0`; `arc_margin=0` | adopted: same |
| Normal-aware / text branches | normal-aware head off; `use_text_guided=False`, `proto_text_refine='none'`, no ITC (vision-only path) | adopted: kept off |
| Initialisation / RNG | `training_protocol=clean_v1`, `rng_protocol=isolated_rng`, `seed == init_seed == data_seed == augment_seed` | adopted |
| Checkpoint rule | `early_stop_patience=0` (no early stopping), `save_every=0` (last epoch only), `checkpoint_format=lightweight`; output `multiclass_epoch_19.pth` | adopted (rule) |

Sources: `experiments/p3/run_p30_dino_mean_oof.py` `train_command()` / `validation_command()`;
`experiments/p3/run_p3f_clean_five_seed.py`; `train_multiclass.py` (`configure_model_trainable_parameters()`,
optimizer / scheduler, argparse defaults, saving branch); `dataset/__init__.py` (cable branch of the training
transform and the augmentation_seed isolation in `__getitem__`); `docs/P3F_CLEAN_FIVE_SEED_BASELINE_PROTOCOL.md` §1.

### 3.2 Not transferred (explicit)

- Cable checkpoint, Cable label names, Good class, `tau=0.006`, the P2.2 normal-aware boundary, the normal-aware
  head.
- The text-guided branches (`use_text_guided` / `proto_text_refine` / ITC) stay off.

## 4. Training budget on MVTec-FS (LOCKED)

- **LOCKED (ruling of the lead reviewer, former U1)**: `epochs=20`, `repeat_per_epoch=4`, `batch=8`, `grad_accum=2`,
  `lr=1e-4`, `proto_lr=3e-4`; the parameter rule is **the same for all products / K / seeds**.
- Products differ in the number of classes N and in K, so the number of support samples differs and with it the
  actual number of batches and optimizer steps; every run must **record its actual batches and optimizer steps**;
  this protocol **does not promise** the same absolute number of steps for different N×K configurations.
- Once locked, the budget is not adjusted; early stopping on query metrics and adjustment by product size are not
  allowed.

## 5. Implementation requirements (locked interfaces; code files NOT specified)

The former U4 (choice of training entry point) was moved from "unresolved" into this section by the lead reviewer.
This protocol **locks only the following interface contracts and specifies no code file**; the implementation (new
or reused entry points) is carried out by the implementation agent after authorisation by the lead reviewer:

1. **Dataset interface**: N-way defect classification training with the (product, K, seed) support manifest as the
   only source of training data; the candidate indexing and augmentation RNG semantics must agree with §2.2 / §3.1;
   query data are invisible to the training process (§8).
2. **Manifest interface**: produce and validate `p6b_support_manifest_v1` (§2.3), including deterministic
   recomputation and self-consistent recomputation of `manifest_sha256`, and record the support/query SHA overlap
   audit. Frozen-source gate: the actual SHA-256 of the records JSONL must equal the value recorded in
   `feasibility.output_hashes`, and both `common_products_feasible_for_all_shots` and `feasible_products_by_shot`
   (1/3/5) must equal the locked 13 products exactly; any mismatch is fail-closed.
3. **Firewall interface**: implement the three-stage data-flow boundary of §8 and prove it by an auditable record:
   overlap = 0, manifest hashes match, no query read by the training process.
4. **Artifact interface**: produce all artifacts listed in §9 (manifests, checkpoints, prototype construction
   records, prediction exports, aggregated metric tables, completion records). Bundle-level transactional write: an
   existing formal target directory is refused; all artifacts are completed in a sibling staging directory and
   published in a single atomic step; a failure part-way may leave no formal directory, old completion or partial
   artifact. The formal CLI requires a resolvable HEAD and a clean working tree; artifacts of the explicit
   diagnostic switch must be marked `formal_eligible=false`.

## 6. Baselines and fairness (LOCKED by the U3 ruling)

### 6.1 Group A: strict support-only visual group (main fairness group, locked)

Vision only, support only, no class-name text prior, no privileged supervision at training or query time:

| Baseline | Text prior | Note |
|---|---|---|
| CLIP-Proto | none (frozen CLIP visual features + class prototypes, no class names/text) | locked |
| DINO-Proto | none (frozen DINOv2 visual features + class prototypes) | locked |
| Frozen-Fusion ProtoNet-style | none (prototype distance on the frozen two-branch fused features) | locked |
| Frozen-Fusion Linear Probe | none (linear classifier on the frozen two-branch fused features) | locked |
| Ours (P3.0 Method Transfer) | none (vision-only frozen pipeline of §3) | locked |

### 6.2 Group B: text-prior-augmented group (locked, with exact constants)

| Baseline | Locked constants | Note |
|---|---|---|
| Zero-shot CLIP | prompt template `"a photo of a {class_name}."` (class_name = official label of the class universe of the product); no support samples | training-free; the query input is still plain RGB |
| Tip-Adapter (training-free) | `alpha=1.0`, `beta=5.5`, prompt `"a photo of a {class_name}."` | training-free; the KV cache is built only from the shared support manifests and follows the §8 firewall |

- Group B methods add a text prior (pre-trained image-text alignment) and are not strictly equivalent to the
  vision-only group A methods: their **supervision and resource differences must be reported separately** and they
  may not be mixed into equivalent comparisons; zero-shot CLIP uses no support samples, which is also declared
  separately.
- Once locked, these constants are not adjusted; reselecting alpha/beta/prompt by query metrics is not allowed.

### 6.3 Privileged reference (listed separately, outside the main fairness groups)

- Published methods that use query-time masks or other privileged supervision may only be listed as privileged
  references.

### 6.4 Fairness constraints (locked)

- All group A/B methods use **exactly the same support manifests** (§2.3).
- The query-time model input is RGB only; no ground-truth masks, boxes or pixel annotations.
- No tuning on query metrics; all baseline hyperparameters are locked by §6.1/§6.2 and may not be invented or added
  afterwards.
- Text is not banned wholesale: the text-prior status of every baseline is stated explicitly in §6.1/§6.2.

### 6.5 Reserved extensions (R1–R6, locked as reserved)

- Reserved extension slots **R1, R2, R3, R4, R5, R6** for baselines or diagnostic groups that may be added later.
  Any extension can only be activated by an explicit ruling of the lead reviewer, with a protocol version upgrade;
  before activation R1–R6 are bound to no method, and nothing may be added after pre-specification.

## 7. Evaluation metrics (locked)

### 7.1 Metric definitions

- **Class universe**: fixed as the **support-side official single-type defect classes** of the product (the classes
  of the single-type candidates of the support pool defined by the feasibility probe), in fixed lexicographic order.
  This resolves the DRAFT-stage conflict between §7.1 and the §8 query firewall: the feasibility JSONL rows are
  parsed, but their query label/path fields are never accessed, kept or used for any manifest, class-universe or
  training decision; the only query-side information consumed is product/source_pools/rgb_sha256 (for the overlap
  audit only), so the universe is derived from the support side alone (for all 13 feasible products, at least K
  support candidates per class guarantee that both definitions agree in practice). BA and Macro-F1 are always
  computed on this fixed universe, which neither shrinks nor grows with the predictions.
- **Top-1 accuracy**: share of queries whose prediction equals the true label.
- **Balanced accuracy**: arithmetic mean of the per-class recalls over the fixed class universe.
- **Macro-F1**: arithmetic mean of the per-class F1 over the fixed class universe, `zero_division=0`.
- Image-level AUROC, Pixel-AUROC and PRO are not reported as main metrics.

### 7.2 Aggregation order

1. `raw_metric[product, K, seed]`: metrics are first computed per (product, K, seed).
2. For each (K, seed), an **equally weighted product macro** over the 13 products.
3. Then **mean / std (ddof=1)** over the 5 seeds.
4. The **per-product** five-seed mean/std table is kept for auditing.
5. By §2.5, the std expresses only the joint pipeline uncertainty.

## 8. Query firewall (locked, fail-closed)

1. **Before training (allowed)**: only the **query SHA metadata** of the frozen feasibility JSONL
   (`data/p6b_probe_output_v2/`) is read, for the overlap audit of §2.4.
2. **Training process (not allowed)**: reading query pixels, query labels, query class counts or any query-derived
   metric, or selecting a checkpoint, epoch, threshold, architecture or any hyperparameter from them.
3. **Final evaluator (only exit)**: only the final evaluator may load query RGB and labels, and only to compute the
   metrics of §7.
4. The support/query RGB SHA-256 overlap of every manifest is audited and must be zero; every manifest records and
   recomputes `manifest_sha256` by the §2.3 rule; a mismatch is fail-closed.
5. If any firewall condition fails, that configuration of the product is marked INCONCLUSIVE; it may not be
   downgraded into a partial conclusion, and the conditions may not be relaxed afterwards.

## 9. Artifacts (planned)

Every complete run produces:

- per (product, K, seed): the support manifest (§2.3, with sha256 and overlap audit);
- per product: terminal adapter checkpoint + prototype construction record;
- per (product, K, seed): query prediction export (per-sample predictions) + prediction manifest (with references to
  the input sha values);
- aggregated tables: `raw_metric[product, K, seed]` + equally weighted product macro per (K, seed) + five-seed
  mean/std (ddof=1) + per-product five-seed mean/std;
- per-run training record: actual batches and optimizer steps (§4);
- completion record: execution HEAD commit, `formal_test_used=false`, manifest/prediction sha256, firewall audit
  results.

## 10. Fail-closed rules

- Any §8 firewall failure, manifest hash mismatch, overlap > 0, missing or duplicate seed, or an unclean working tree
  at execution: the affected unit is INCONCLUSIVE and forms no stable conclusion.
- Unfavourable results may not lead to changes of the experiment definition or criteria, or to the exclusion of
  difficult products/classes afterwards.
- Integrity failures are not downgraded into performance conclusions; data rulings may not be replaced for
  performance reasons.

## 11. Upgrade conditions (to PRE-SPECIFIED)

All of the following are satisfied, and the lead reviewer upgraded the protocol to
**PRE-SPECIFIED — LOCKED FOR EXECUTION**:

1. All scientific unresolved fields of §12 were ruled on by the lead reviewer and filled in (U3 ruled; no scientific
   field remains). Done.
2. The two-column table of §3.1 and the interface contracts of §5 passed a review against the actual implementation
   call chain (stage-1 code and its 15 tests independently re-verified). Done.
3. The text-prior status and hyperparameters of every baseline are locked (§6.1/§6.2). Done.
4. The stage-1 shared infrastructure is accepted; model training and baseline evaluation still require the formal
   manifest bundle to pass its audit before stage 2.

### 11.1 Operational note (not a scientific unresolved item; does not block pre-specification)

- Compute scheduling (195 runs per method × the cloud run order and resource allocation of each method) is an
  operational note arranged by the user when running in the cloud; it is not a scientific field that blocks
  pre-specification.

## 12. Unresolved before pre-specification

All scientific unresolved fields have been ruled on by the lead reviewer; no open item remains in this section:

- **Former U1 (transfer of the training budget) → LOCKED**, see §4.
- **Former U2 (augmentation domain adaptation) → LOCKED**, see the augmentation row of §3.1.
- **Former U3 (baseline specification) → LOCKED**, see §6.1/§6.2 (all group A/B methods and constants); reserved
  slots R1–R6 in §6.5.
- **Former U4 (choice of training entry point) → moved to §5 implementation requirements** (interfaces locked, no
  code file specified).

What remains is only the operational note of §11.1 (cloud compute scheduling), which is not a scientific field that
blocks pre-specification. The protocol was upgraded by the lead reviewer to **PRE-SPECIFIED — LOCKED FOR
EXECUTION**; any later change requires a new protocol version and a new ruling.
