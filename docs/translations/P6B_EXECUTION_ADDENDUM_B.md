<!-- translation-header -->
> English translation of [`P6B_EXECUTION_ADDENDUM_B.md`](../P6B_EXECUTION_ADDENDUM_B.md), which is kept unchanged as the committed record (SHA-256 `13532fa9b8968db7ec685e8434af080644fb1b80cc106fa3aebc943ad64a40d9`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P6.B Execution Protocol — Addendum B: R1/R2 Reserved-Extension Activation (pre-specified, pre-execution)

> Status: **PRE-SPECIFIED — LOCKED PRE-EXECUTION**, part of the main protocol
> `docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md` (PRE-SPECIFIED — LOCKED FOR EXECUTION) and of
> `docs/P6B_EXECUTION_ADDENDUM_A.md` (PRE-SPECIFIED — LOCKED ADDENDUM A). This addendum is locked together with the
> main protocol and addendum A: any change requires a new version and a new ruling.

## 0. Declarations (facts at the time of locking)

At the time addendum B is ruled on and created:

1. **No P6.B query performance has been read**: no query pixels, query labels, query predictions, query accuracy,
   BA, F1, confusion matrix or any other indicator of model quality on the queries has been read or used for any
   decision in this file.
2. **No query metric of any baseline or of Ours has been read.**
3. **The formal grid of the original seven methods has not started** (no formal query evaluation of
   `13 products × 3 shots × 5 seeds` has taken place).
4. **The cloud diagnostic smoke of the original seven methods is paused before this protocol upgrade**; the
   experimental semantics, provenance, query firewall and support-only execution of the nine methods are frozen
   first, and only then is a support-only diagnostic smoke allowed.

This addendum **only activates two reserved extensions (R1, R2) and adds the Protocol Bundle V2 provenance**: it
changes nothing in the 13 products, K ∈ {1,3,5}, the five paired seeds {111,222,333,444,555}, the stage-1 manifests
and bundle, the support/query split, the support draw, the metric definitions and aggregation (main protocol §7), the
query firewall (main protocol §8) or the semantics of the original group A/B methods (addendum A §1).

## 1. Reserved-extension activation (main protocol §6.5)

Under main protocol §6.5, the lead reviewer explicitly activates the reserved extension slots **R1 and R2** and
upgrades to Protocol Bundle V2 (see §4). R3–R6 remain reserved, inactive and bound to no method.

- **R1 = Exact P3.0 Frozen Representation Proto** (method id `p30_frozen_proto`).
- **R2 = Tip-Adapter-F\*** (firewall-constrained; method id `tip_adapter_f`).

R1 and R2 are the last two baseline/ablation extensions of this round; no R3/R4 or any other baseline is added.

## 2. R1 — Exact P3.0 Frozen Representation Proto (locked mathematics)

Purpose: a **feature-matched frozen, no-adaptation reference**. R1 shares exactly the same frozen pre-terminal token
source as Ours (main protocol §3), but switches off the terminal adapter and classifies with non-parametric
class-mean prototypes.

R1 is **not a strict module ablation of the terminal adapter alone**: besides `image_adapter.det_proj`, Ours also
trains `prototype_mem.prototypes` and `prototype_mem.logit_scale`, and R1 trains none of the three. Ours − R1 can
therefore only be read as "the difference made by the complete support-side adaptation (terminal adapter + learned
prototype classifier) relative to a frozen pre-terminal prototype reference", and may not be attributed to the
terminal adapter alone.

### 2.1 Frozen pre-terminal representation (identical to Ours)

R1 shares the `AdaptedCLIP` forward pipeline of Ours up to the terminal adapter:

- **CLIP branch (pre-terminal tokens)**: the same frozen CLIP `ViT-L-14-336` patch embedding + positional encoding +
  `_run_clip_blocks` (`adapter_train_mode=terminal_only`, all 24 layers under `torch.no_grad`) +
  `image_encoder.ln_post` as Ours, giving the token tensor `[B, N+1, 1024]` after `ln_post` and before `det_proj`.
  **This tensor is exactly the input that Ours feeds into the terminal adapter `det_proj`.**
- **DINO branch**: the same as Ours, **mean pooling** over the full patch tokens of the frozen DINOv2 `vit-s14`
  (`dino_patch_policy=full`). The DINO branch has no terminal adapter in Ours, so it is bit-identical in R1 and Ours.

### 2.2 R1 classification semantics (terminal adaptation OFF)

```text
CLIP branch: clip_feat = L2normalize( topk_avg_pool(ln_post_tokens[B,N+1,1024], k_ratio=0.1) )   # 1024-d
DINO branch: dino_feat = L2normalize( dino_patches.mean(dim=1) )                                   # 384-d
frozen pre-terminal fusion: feature = L2normalize( concat([clip_feat, dino_feat]) )              # 1408-d
prototype   = mean of the support features of each class
prototype   = L2normalize(prototype)
query_feat  = L2normalize(query pre-terminal feature)
decision    = cosine argmax(query_feat, prototypes)
tie-break   = fixed manifest class_order (earliest class wins)
```

Compared with Ours, R1 has:
- the same CLIP `_run_clip_blocks` and `ln_post` (the same top-K pooling operator `topk_avg_pool(k_ratio=0.1)`
  applied to the pre-terminal `[B,N+1,1024]` tokens);
- the same DINO mean implementation;
- the same feature concatenation and ordering (CLIP first, DINO second);
- the same position of the pre-terminal normalisation (L2 per branch → concat → L2 again);
- the same deterministic query evaluation view (the single deterministic 336 view of addendum A §1.4);
- the same support manifests / K / seeds;
- **no text prior**;
- **no trained terminal adapter and no trained backbone**;
- prototypes = class means of the support features, then L2-normalised;
- L2-normalised query features; cosine argmax; tie-break by the fixed `class_order`.

Relative to Ours, R1 switches off both the terminal adaptation and the learned prototype-classifier adaptation
(`det_proj` is not applied, prototypes and logit scale are not trained; no trainable parameter at all).

### 2.3 Implementation and acceptance constraints of R1

- R1 must use **the same real feature-construction helper** as Ours and may not copy a separate CLIP/DINO backbone
  loader; its pre-terminal features must equal, bit for bit (within floating-point tolerance), the tensor that Ours
  feeds into `det_proj`. This is implemented with a forward pre-hook on `AdaptedCLIP.image_adapter.det_proj` that
  captures the terminal-adapter input of Ours, reusing `AdaptedCLIP.topk_avg_pool` and
  `AdaptedCLIP.dino_extractor.forward_patches`, which guarantees "the same helper, no duplicated implementation, no
  change to the P3.F code".
- `trainable_param_count == 0`: apart from the temporary prototype tensors there is no trainable module; all model
  parameters have `requires_grad=False`; no optimizer is constructed.
- The support construction of R1 may not read query pixels, query labels or query metrics.

### 2.4 Relation to the existing Frozen-Fusion Proto (important distinction)

The existing **Frozen-Fusion Proto** (method id `fusion_proto`, addendum A §1.1/§1.2) **remains an ordinary
baseline** but **may no longer be called a strict ablation of terminal adaptation**: its CLIP representation is the
**native projected global CLIP image embedding**, which differs from the pre-terminal token source shared by R1 and
Ours. R1 is therefore a feature-matched frozen reference closer to Ours than Frozen-Fusion Proto; but because Ours
also learns a prototype classifier, R1 still may not be described as a strict terminal-adapter-only ablation.

Allowed wording in the paper: `Ours outperforms a feature-matched frozen pre-terminal prototype reference`.
Not allowed: `R1 proves that the terminal adapter alone improves X%`.

## 3. R2 — Tip-Adapter-F\* (firewall-constrained; locked mathematics and constants)

A trainable CLIP few-shot baseline is added to group B, called **Tip-Adapter-F\*** internally and in the paper
(equivalently **Tip-Adapter-F (firewall-constrained)**).

Reason for the name: the official Tip-Adapter-F trains the cache keys, looks at evaluation performance every epoch,
selects the best epoch and searches alpha/beta, which conflicts with the query firewall of this project (main
protocol §8). This project therefore uses a **pre-specified, restricted version without query tuning**. Where the
official implementation conflicts with this protocol, **the pre-specified firewall of this project takes
precedence**; queries may not be read and no tuning may be added to come closer to the official results.

### 3.1 Frozen and trainable parts

Frozen (`requires_grad=False`):
- CLIP image encoder;
- CLIP text encoder;
- cache values (support one-hot);
- class text embeddings (`text_weights`).

Trainable (`requires_grad=True`):
- **cache keys only** (the only trainable parameters).

Initialisation: the cache keys are initialised from the support cache keys of plain Tip-Adapter (support **CLIP
embeddings**, L2-normalised; the same as the training-free Tip-Adapter cache of addendum A §1.6).

### 3.2 Classification semantics (as in Tip-Adapter)

```text
affinity     = exp(-beta * (1 - image_emb @ cache_keys.T))
cache_logits = affinity @ cache_values
clip_logits  = 100 * image_emb @ text_weights.T
logits       = clip_logits + alpha * cache_logits
```

Fixed constants (fail-closed; no search, query tuning or best-epoch selection):

```text
alpha = 1.0
beta  = 5.5
```

`text_weights` are the zero-shot prompts `"a photo of a {class_name}."` (underscores replaced by spaces, addendum A
§1.5) encoded by the frozen CLIP text encoder and L2-normalised; the logit scale of `clip_logits` is 100 (addendum A
§1.5).

### 3.3 Training constants of R2 (locked; faithful to the official cache-key training where compatible)

```text
trainable set          = cache_keys only
optimizer              = AdamW (parameter group exactly the cache_keys)
lr                     = 1e-3
eps                    = 1e-4
weight_decay           = 1e-2
scheduler              = CosineAnnealingLR(T_max = expected total number of optimizer steps)
batch_size             = 8
gradient_accumulation  = 1
num_epochs             = 20 (final epoch only)
repeat_per_epoch       = 1
augmentation           = non-cable augmentation of main protocol §3.1 (RandomResizedCrop(336,
                         scale=(0.7,1.0)) + RandomApply(RandomAffine(15,(0.1,0.1)),p=0.3)
                         + RandomApply(ColorJitter(0.2,0.2,0.15,0.0),p=0.5)),
                         frozen CLIP image embedding extracted again in every epoch/repeat pass
support_repeat         = one pass over the support set per epoch (repeat_per_epoch=1)
RNG                    = seed == init_seed == data_seed == augment_seed;
                         SeededEpochSampler + make_loader_generator +
                         DeterministicWorkerSeeder (as in main protocol §3.1)
loss                   = cross-entropy (logits as in §3.2; label_smoothing=0.0)
checkpoint_selection   = final_epoch_only
```

Expected optimizer steps = `num_epochs × repeat_per_epoch × ceil(support_count / batch_size) / gradient_accumulation`
(with grad_accum = 1: `num_epochs × repeat_per_epoch × ceil(support_count / batch_size)`); any other actual number of
steps is fail-closed.

Not allowed (fail-closed):
- alpha search / beta search / prompt search / validation tuning / query tuning;
- best-epoch selection / checkpoint selection / early stopping;
- reading queries (pixels / labels / metrics) during training.

The completion / audit must record:
```text
checkpoint_selection      = final_epoch_only
query_used_for_selection  = false
validation_used_for_selection = false
```

## 4. Protocol Bundle V2 provenance (fail-closed)

Support-side artifacts and completions of stage 2+ must record and verify the SHA of **three protocol documents** and
of the stage-1 completion:

```text
main_protocol_sha256      = main protocol docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md (canonical LF)
addendum_a_sha256         = docs/P6B_EXECUTION_ADDENDUM_A.md (canonical LF)
addendum_b_sha256         = docs/P6B_EXECUTION_ADDENDUM_B.md (canonical LF, this file)
stage1_completion_sha256  = fixed pin of the stage-1 formal bundle completion
                            5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73
```

- The three protocol documents are hashed after canonical-LF normalisation (CRLF → LF, bare CR rejected); they are
  verified at the start of a run and again before the completion is written.
- Code may use other field names for compatibility, but must **pin all three protocol documents**.
- Any missing or mismatching SHA:

```text
fail-closed
formal_eligible = false
```

Formal execution may not continue. The Protocol Bundle version is **V2** (V1 pinned only the main protocol and
addendum A; V2 adds addendum B).

## 5. Unchanged items (not modified by this addendum)

- 13 products (Cable excluded; toothbrush excluded because of asset integrity);
- K ∈ {1,3,5}; five paired seeds {111,222,333,444,555};
- stage-1 support manifests / formal bundle (unchanged);
- support/query split, support draw, unique-original-RGB sampling, support/query RGB SHA overlap = 0;
- metric definitions and aggregation (main protocol §7, product macro → mean/std (ddof=1) over the 5 seeds);
- query firewall (main protocol §8, fail-closed);
- semantics of the original group A/B methods (addendum A §1, including the retained Frozen-Fusion Proto);
- every field of Ours (main protocol §3/§4).

## 6. Diagnostic order of the nine methods (support-only diagnostic smoke)

```text
01 CLIP Proto                         (clip_proto)
02 DINO Proto                         (dino_proto)
03 Frozen-Fusion Proto                (fusion_proto)
04 Zero-shot CLIP                     (zero_shot_clip)
05 Tip-Adapter                        (tip_adapter)
06 Frozen-Fusion Linear Probe         (linear_probe)
07 R1 Exact P3.0 Frozen Proto         (p30_frozen_proto)
08 R2 Tip-Adapter-F*                  (tip_adapter_f)
09 Ours                               (ours)
```

The support-only diagnostic smoke of all nine methods uses bottle / K=1 / seed=111 / the same stage-1 support
manifest / diagnostic mode / `formal_eligible=false`; the smoke **verifies the engineering semantics only and may
neither produce nor read query performance**. After the smoke passes, work stops and is returned to the lead
reviewer; query evaluation may start only after the lead reviewer explicitly issues `APPROVE_FORMAL_GRID`.
