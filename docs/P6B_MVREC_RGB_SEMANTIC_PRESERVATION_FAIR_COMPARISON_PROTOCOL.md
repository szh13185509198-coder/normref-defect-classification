# Protocol A — MVREC-RGB Semantic-Preservation & Fair-Comparison Protocol

> Status: **PREREGISTERED — LOCKED BEFORE ANY MVREC-RGB PERFORMANCE**
>
> Date locked: 2026-08-13
>
> Scope: post-P6.B extension; this document does not reopen or alter the completed
> P6.B nine-method result.

## 0. Pre-performance declaration

The completed P6.B results for Ours and the original locked baselines are already
known. At the time this protocol is locked:

- no MVREC-RGB support training, query inference, prediction, accuracy, Balanced
  Accuracy, Macro-F1, confusion matrix, or product/seed performance has been read;
- no nine-product K=8 model performance has been generated or read;
- the official MVREC paper and code structure have been inspected only to identify
  supervision and data-flow dependencies;
- the decision criteria in
  `docs/P6B_V1_GO_NO_GO_DECISION_RULE.md` are fixed before MVREC-RGB performance.

Any scientific change after MVREC-RGB performance is visible requires a new,
explicitly post-result document and cannot replace this protocol.

## 1. Scientific question and naming

**Question:** Can the transferable classification design of MVREC be represented
faithfully under the same strict RGB-only, image-level supervised, mask-free and
box-free regime as P6.B, and if so, how does it compare with Ours?

The adapted method must be named exactly:

> **MVREC-RGB adaptation**

It must not be called official MVREC, MVREC without qualification, or an official
no-mask result. Official MVREC is reported only in a separate
**privileged-information reference table**.

## 2. Authoritative source and provenance

- Paper: `https://ojs.aaai.org/index.php/AAAI/article/download/32634/34789`
- Repository: `https://github.com/ShuaiLYU/MVREC`
- Official repository HEAD inspected for this ruling:
  `112c187b6de8a7400fad89ca2a201545842a5d84`
- Relevant privileged paths include
  `modules/multi_view_data.py` and
  `modules/bbox_clip/alpha_clip_backbone.py`.

The semantic audit artifact must record the exact upstream commit, local patch
SHA256, config hashes, weights hashes, environment, and every component classified
as preserved, interface-adapted, privileged-and-removed, or substantively changed.

This protocol, Protocol B, and the V1 decision rule must first be archived together
in a clean Git commit. Their canonical-LF SHA256 values and the clean execution
commit must be verified at semantic-audit start and completion. Formal execution
from an uncommitted or dirty protocol state is forbidden.

## 3. Strict RGB-only supervision firewall

Neither support training nor query inference may consume, directly or indirectly:

- GT masks or mask-derived images/features;
- GT bounding boxes or coordinates;
- bbox-centred crops;
- mask-derived crops;
- annotation-derived ROIs, offsets, scales, views, tokens, or attention priors;
- query-time localization predictions selected or tuned using query labels;
- pixel labels, object masks, defect masks, or any equivalent localization field.

Allowed information is limited to:

- original whole RGB images;
- support-side image-level product/defect class labels;
- fixed class-name text prior;
- ordinary label-independent augmentation;
- fixed/global multi-view transforms whose locations and scales are identical rules
  for every image and are not computed from annotations or model predictions;
- query RGB pixels only in the final frozen evaluator.

An all-ones/full-frame mask is permitted solely as a constant API placeholder when
required by an unchanged backbone. It must be generated independently of all
annotations and recorded as such.

## 4. Transferable semantics that should be preserved

Subject to compatibility with §3, the adaptation must retain:

1. the official backbone family and pretrained weights;
2. the official classifier/adapter and its trainable parameterization;
3. the official loss, optimizer, schedule, and training budget;
4. the official text prior and prompt construction;
5. the support-conditioned classification and inference decision rule;
6. the official feature/view aggregation when it remains meaningful for whole-image
   or fixed global views;
7. P6.B's exact support identities, paired seeds, query identities, class universe,
   and no-query-tuning firewall.

Hardware-only changes such as deterministic batch-size reduction with exactly
equivalent gradient accumulation are allowed only if the effective batch, step
count, optimizer update sequence, and learning-rate schedule remain identical.

## 5. Allowed interface adaptation

The following are allowed and must be documented:

- replacing bbox/ROI loaders with whole-image loading;
- replacing annotation-derived views with a fixed, predeclared global-view list;
- deterministic resize/pad/crop needed by the official backbone;
- field-name and label-index adapters for the frozen P6.B manifests;
- non-trainable tensor-shape adapters that do not change the classifier;
- replacing any query/validation-selected checkpoint with the final locked training
  checkpoint, without changing the training budget, when required by the firewall;
- removal of now-unused mask/bbox/ROI branches.

The following constitute a **substantive method rewrite** and are forbidden:

- adding a trainable module, new loss, or new regularizer;
- changing the backbone family or pretrained representation;
- replacing the official classifier/adapter or logit rule;
- inventing a new learned global-view or patch aggregation mechanism;
- changing optimizer, training budget, prompt, or checkpoint-selection rule to
  improve results;
- changing the support/query protocol or using query performance for any choice.

## 6. Semantic-preservation audit

The audit is support-only and performance-blind. It has five mandatory gates.

### A1 — upstream reconstruction

- Pin and reproduce the official component graph from the source in §2.
- Produce a module-level map of backbone, view/region context, adapter/classifier,
  loss, optimizer, schedule, text prior, and decision rule.
- Identify every read of mask, bbox, ROI, crop centre, offset, or localization field.

### A2 — privileged-dependency removal

- Static audit: no production adaptation path may access a forbidden field from §3.
- Dynamic poison audit: replace every forbidden metadata value with invalid/random
  sentinels while holding RGB and image label fixed. Constructed model inputs and
  support predictions must remain byte-identical where deterministic, or numerically
  identical within `atol=1e-7, rtol=1e-6`.
- Removing the forbidden fields entirely must not change the produced inputs or
  outputs.
- Query paths are not loaded during this audit.

### A3 — transferable-design preservation

Produce an explicit component table. `SEMANTIC_PASS` requires all of the following:

- backbone and pretrained weights: preserved;
- classifier/adapter architecture and trainable parameter groups: preserved;
- loss, optimizer, schedule, epoch/step budget: preserved;
- text prior and support-conditioned decision rule: preserved;
- modifications limited to the input/data interface allowed by §5;
- no new trainable aggregation or replacement classifier.

### A4 — support-only execution smoke

On one locked support manifest, run only shape, determinism, gradient-ownership,
finite-value, and completion-schema checks. Do not calculate classification metrics.
The smoke must prove that only the official intended parameter groups are trainable
and that query data are absent from the process.

### A5 — independent audit artifact

Write a machine-readable semantic audit containing every gate, hashes, component
mapping, poison-test result, trainable parameter names/count, and a final status.
Missing evidence is fail-closed.

## 7. Semantic decision

- **SEMANTIC_PASS:** all A1–A5 gates pass. MVREC-RGB formal evaluation may be
  implemented only after the implementation commit and audit artifact are pinned.
- **STOP_NOT_REPRESENTATIVE:** removal of privileged localization requires any
  substantive rewrite in §5, or the remaining code cannot reasonably represent the
  transferable MVREC classification design. Do not create a weakened surrogate;
  report official MVREC only in the privileged-information reference table.
- **INCONCLUSIVE_ENGINEERING:** missing asset, hash, or reproducibility evidence.
  Repair only the integrity issue and repeat the same audit; do not change semantics.

`STOP_NOT_REPRESENTATIVE` cannot be converted to `SEMANTIC_PASS` by inspecting
performance.

## 8. Locked fair comparison after SEMANTIC_PASS

- Products: the original locked 13 P6.B products.
- Shots: K = 1, 3, 5.
- Paired seeds: 111, 222, 333, 444, 555.
- Support: exact existing P6.B manifests and identities.
- Query: exact existing P6.B query identities and image-level units.
- Labels: the same per-product defect class universe.
- Metrics: Top-1 Accuracy, product-macro Balanced Accuracy, and Macro-F1 under the
  original P6.B definitions.
- Selection: final locked checkpoint only; no validation/query selection, prompt
  search, hyperparameter search, or per-product tuning.
- Inputs: whole-image RGB or the fixed global views approved by the semantic audit.

Primary scientific comparison is Ours versus MVREC-RGB adaptation. Frozen-Fusion is
the locked structural baseline. Official MVREC and other methods using privileged
localization are excluded from paired statistical ranking.

The formal evaluator must produce paired per-product/per-seed predictions and the
resource fields required by `docs/P6B_V1_GO_NO_GO_DECISION_RULE.md`.

## 9. Required fairness/resource table

Every reported method must disclose:

- RGB only versus mask/bbox/pixel supervision;
- support/query annotation fields;
- text prior;
- trainable and total parameter counts;
- backbone count and pretrained weights;
- support training GPU time and peak memory;
- deterministic query latency/throughput under the locked measurement environment;
- whether reported numbers are official, reproduced, or this project's adaptation.

Implementation complexity may be described, but source lines of code cannot by
itself satisfy the resource-advantage gate.

## 10. Prohibited actions

- No performance-driven semantic decision.
- No tuning MVREC-RGB against the already known Ours predictions.
- No new baseline, adapter, loss, multi-prototype rule, or query-guided choice.
- No mixing official privileged MVREC numbers into the strict RGB-only main ranking.
- No P7, P6.A, or MVTec-AD-RGB execution before the V1 decision point.
