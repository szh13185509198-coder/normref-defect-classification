# P10 Supplement S — reference conditions added after the formal run

Status: **LOCKED BEFORE ANY SUPPLEMENT-S OUTPUT** (2026-09-26). These are post-formal,
**descriptive** reference rows requested by the novelty review (closest prior work:
UniADC / MVREC use support masks; Liu et al. 2024 use normal-residual features). They are
not candidates for the proposed method, carry no verdict, and are reported regardless of
outcome. The manuscript labels them "added during revision".

## S1 — support-mask condition (C2 = C1 + GT masks on supports only)

Supports are pooled with their GT mask (mask-area-weighted mean of CLIP layer-18 patch
tokens; plain mean if a mask is empty); queries have no mask and use the C1 anomaly-
selected pooling (same M=20 normal bank). Classification: SimpleShot-CL2N on these
descriptors (`supmask_normref_clip18`) and its fusion with the global SimpleShot score
(`fused_supmask_normref_clip18`). Purpose: show where the training-free pipeline stands
when the supervision of MVREC / UniADC (support masks) is allowed.

## S2 — training-free normal-residual baseline (C1, Liu et al.-style)

Per image, each patch p is replaced by its residual r = p − n(p), where n(p) is its
nearest patch in the M=20 normal bank; descriptor = mean residual over all patches;
SimpleShot-CL2N (`residual_mean_clip18`) and fusion with global SimpleShot
(`fused_residual_mean_clip18`). Purpose: a same-information baseline that uses normal
references through residuals, without anomaly-selected pooling, meta-training, or synthetic
data (Liu et al. additionally train a classifier on synthetic defects; not reproduced).

## Execution

1. **Dev stage** (support pool, dev seeds 9001–9005, K=1/3): sanity and bug check only;
   its outcome does not gate or change the formal stage.
2. **Formal stage** (530 formal queries, formal manifests and formal-seed normal banks
   from `ckpt/p10_formal/features/`, K=1/3/5, seeds 111–555): single run, output
   `ckpt/p10_formal/supplement_s/`. Integrity check: the recomputed
   `fused_normref_clip18` and `simpleshot_cl2n` product-macro BA must equal the P10 formal
   results exactly (|diff| < 1e-9); otherwise the stage fails.
3. Reporting (formal): each S-row minus `simpleshot_cl2n` and minus `fused_normref_clip18`,
   product-cluster bootstrap as in P10 formal; levels per K. No hypothesis verdicts.

Guards: the formal stage refuses to run with uncommitted changes in `experiments/p10`,
`experiments/revision`, or this protocol, and refuses to overwrite its output directory.
It must not run concurrently with any other GPU job.
