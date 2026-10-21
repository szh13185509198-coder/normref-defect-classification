<!-- translation-header -->
> English translation of [`P6B_EXECUTION_ADDENDUM_A.md`](../P6B_EXECUTION_ADDENDUM_A.md), which is kept unchanged as the committed record (SHA-256 `ab492625e3341c43b2c17aee15e0dd95b56a8edc98db3920f3000c7967b4b5a3`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P6.B Execution Protocol — Addendum A: Baseline Mathematical Semantics (pre-specified, pre-execution)

> Status: **PRE-SPECIFIED — LOCKED (PRE-EXECUTION ADDENDUM A)**, part of the main protocol
> `docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md` (PRE-SPECIFIED — LOCKED FOR EXECUTION). This addendum is locked
> together with the main protocol: any change requires a new version and a new ruling.

## 0. Declarations

1. At the time this addendum is locked, **no P6.B model performance has been viewed** (no query metric of any
   baseline or of Ours has been read or used for any decision in this file).
2. This addendum **only removes ambiguities in the baseline implementations**: it changes nothing in the 13
   products, K ∈ {1,3,5}, seeds {111,222,333,444,555}, the training budget (main protocol §4), the metric definitions
   and aggregation (main protocol §7) or the query firewall (main protocol §8).
3. Every evaluation artifact produced in stage 2 and later must **record the SHA-256 of both the main protocol and
   this addendum** (e.g. the fields `protocol_sha256` and `addendum_a_sha256`); they are checked before loading, and
   any mismatch is fail-closed.

## 1. Locked mathematical semantics

### 1.1 Embeddings (all backbones frozen, `requires_grad=False` by default)

- **CLIP embedding**: the **native projected global image embedding** of the frozen CLIP `ViT-L-14-336`, then
  **L2-normalised**.
- **DINO embedding**: frozen DINOv2 `vit-s14`, **mean pooling** over the **full patch tokens** (all patch tokens of
  `dino_patch_policy=full`, without the class token), then **L2-normalised**.
- **Frozen fusion**: `concat(CLIP embedding, DINO embedding)`, **L2-normalised again** after concatenation.

### 1.2 Prototype semantics (CLIP-Proto / DINO-Proto / Frozen-Fusion ProtoNet-style)

1. **L2-normalise** every support embedding;
2. take the **mean** per class and **L2-normalise** it to obtain the class prototype;
3. classify by the **cosine argmax** between the query embedding (same representation, L2-normalised) and the
   prototypes;
4. **ties are broken by the manifest `class_order`** (the fixed lexicographic order of the class universe; the
   earliest class wins).
- CLIP-Proto, DINO-Proto and Frozen-Fusion Proto use the corresponding representation of 1.1; everything else is
  identical.

### 1.3 Frozen-fusion linear probe

- Input: the frozen-fusion embedding (frozen; no gradient into the backbones).
- Only `Linear(dim, N, bias=True)` is trained (N = size of the class universe of the product).
- Loss: cross-entropy, `label_smoothing=0.1`.
- Optimizer: AdamW, `lr=1e-4`, `weight_decay=0.01`.
- Scheduler: `CosineAnnealingLR(T_max=epochs)`.
- Budget: `20 epochs`, `repeat_per_epoch=4`, `batch=8`, `grad_accum=2`; the **final-epoch** weights are used;
  seeds are isolated (paired-pipeline semantics of main protocol §2.5).
- **Feature extraction and augmentation (locked)**: in every epoch/repeat pass the support images receive the
  **non-cable augmentation** of main protocol §3.1 (`RandomResizedCrop(336, scale=(0.7,1.0))` →
  `RandomApply(RandomAffine(degrees=15, translate=(0.1,0.1)), p=0.3)` →
  `RandomApply(ColorJitter(0.2,0.2,0.15, hue=0.0), p=0.5)`, in PIL space, isolated per sample by `augment_seed`),
  after which the **frozen fusion feature is extracted again** for training the head; training repeatedly on a
  single deterministic embedding across epochs/repeats is **not allowed**.
- **Locked constants**: epochs/repeat/batch/grad_accum/lr/weight_decay/label_smoothing are locked constants, not
  tunable hyperparameters; any inexact value is fail-closed.

### 1.4 Evaluation view

- Query inference of **all methods** uses **a single deterministic 336 evaluation view** (Resize(336,336) → CLIP
  normalisation), without random augmentation.
- The support/cache construction of the training-free methods uses the same deterministic 336 evaluation view.

### 1.5 Zero-shot CLIP

- Prompt template: `"a photo of a {class_name}."`, with **underscores in `class_name` replaced by spaces**; the text
  embeddings come from the frozen CLIP text encoder and are L2-normalised.
- logits = `logit_scale × (query @ text.T)`, `logit_scale = 100`.

### 1.6 Tip-Adapter (training-free)

- cache keys = support **CLIP embeddings** (L2-normalised); cache values = one-hot vectors of the classes.
- `affinity = exp(-5.5 × (1 − image_emb @ cache_keys.T))` (beta = 5.5, locked).
- `logits = CLIP logits + 1.0 × (affinity @ cache_values)` (alpha = 1.0, locked).
- **No parameter is searched or adjusted**; reselection by query metrics is not allowed.

### 1.7 Ours

- Ours (P3.0 method transfer) follows main protocol §3/§4 exactly; this addendum neither restates nor changes any of
  its fields.

## 2. Provenance requirement

The completion/audit JSON of stage-2+ evaluation outputs must contain:

- `protocol_relative_path` + `protocol_sha256` (main protocol);
- `addendum_relative_path` + `addendum_a_sha256` (this file);
- `stage1_completion_sha256`: the fixed pin of the stage-1 formal bundle completion,
  `5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73`, checked when the bundle is loaded; a mismatch is
  fail-closed.

All of these SHA values are checked before loading or evaluation; any missing or mismatching value → fail-closed
(INCONCLUSIVE).
