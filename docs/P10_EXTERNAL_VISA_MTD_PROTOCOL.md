# P10 External Evaluation — VisA (confirmatory) and MTD (descriptive)

Status: **LOCKED BEFORE ANY P10 EXTERNAL OUTPUT** (2026-09-26). Single run; repeat only
after an integrity failure with identical definitions.

Prior information disclosed: P10 formal MVTec-FS results (RESEARCH_LEDGER "P10 FORMAL")
and the earlier VisA/MTD results of Ours / Frozen-Fusion Proto / MVREC-RGB
(`docs/paper/EXTERNAL_VISA_MTD_FORMAL_RESULTS.md`). No P10 method was ever run on VisA or
MTD before this protocol.

## 1. Frozen external protocol reused unchanged

Everything below is taken verbatim from `docs/EXTERNAL_RGB_FEWSHOT_TRANSFER_PROTOCOL.md`
and its manifest bundle `data/external_transfer_manifest_bundle/`: 11 retained VisA
objects + MTD six-way; normal-inclusive N-way tasks; fixed support-candidate / query
partition; nested support draws for K in {1,3,5} and seeds {111,...,555}; frozen class
order (read from the frozen query units); BA primary, Macro-F1 secondary, Top-1
descriptive; VisA equal-weight object macro; VisA object-cluster bootstrap (10,000
resamples, RNG seed 20260814); MTD seed-level paired differences only. No VisA/MTD mask,
box or ROI is used anywhere (so no mask oracle is computed on external data).

## 2. Information conditions and normal banks

- C0 methods use only the drawn supports.
- C1 adds M unlabeled normal images of the same task, drawn from the **normal class's
  support-candidate partition, excluding that seed's K5 normal supports** (hence
  disjoint from every support at every K and, by partition, from every query):
  `random.Random(f"p10ext|{seed}|{task}").sample(sorted(candidates), 20)`, M=5 = first 5.
  Normal class = `normal` (VisA) / `MT_Free` (MTD). Disjointness is asserted in code.
- Because these tasks contain a normal class, C1 normal images are extra samples of an
  existing class; this is disclosed. The normal-class support itself still has K images.

## 3. Methods (definitions identical to P10 formal, no external tuning)

C0: `fusion_proto_local`, `simpleshot_cl2n`, `local_xclass_clip18`, `fused_xclass_clip18`.
C1: `normref_pool_clip18`, `fused_normref_clip18`, `normcenter_mean_clip18`,
`fused_normcenter_mean_clip18`, `fused_normref_clip18_m5`.
Features: frozen CLIP ViT-L/14@336 native global + DINOv2 mean fusion, CLIP layer-18 patch
tokens, deterministic 336 view (non-square images are resized, as in all earlier runs);
MTD grayscale is replicated to three channels as in the frozen protocol.
Existing comparators (frozen results, not re-run): `ours`, `frozen_fusion_proto`,
`mvrec_rgb_adaptation` from `ckpt/external_query_grid_bundle.tar.gz` / `per_cell_metrics.json`.

## 4. Hypotheses (VisA; BA primary)

Estimator: paired differences per (object, K, seed); object macro; Dbar = mean over K;
95% object-cluster percentile bootstrap (all seeds and K of an object together).

- **E1 (primary, C1):** `fused_normref_clip18` − `simpleshot_cl2n`. CONFIRMED iff Dbar
  CI lower bound > 0.
- **E2 (C1 attribution):** `fused_normref_clip18` − `fused_normcenter_mean_clip18`.
  SUPPORTED iff Dbar CI lower bound > 0.
- **E3 (C0):** `fused_xclass_clip18` − `simpleshot_cl2n`. CONFIRMED iff Dbar CI lower
  bound > 0.

MTD: the same three contrasts reported as seed-level mean ± SD per K, **no verdict**.
All descriptive contrasts vs `ours`, `frozen_fusion_proto`, `mvrec_rgb_adaptation` are
reported regardless of sign (cross-hardware term disclosed). Integrity check reported:
object-macro BA of `fusion_proto_local` vs the frozen `frozen_fusion_proto`.

## 5. Guards

Refuses to run with uncommitted changes in `experiments/p10`, `experiments/revision`, or
this protocol; single-shot output `ckpt/p10_external/results/` (staging + atomic rename).
