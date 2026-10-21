# P30: deployment cost of our method on CLIP ViT-L/14@336 alone, and its FP16 check (report only)

**Status:** locked on 2026-10-01, before any P30 output exists.

## Why

- P29 made our CLIP-L/14@336-only configuration the backbone-matched comparator of CLIP-LoRA L/14@336.
  - That configuration is P17's `a_r`: appearance + deviation on the layer-18 patches, no global score, no DINOv2.
  - It leads CLIP-LoRA L/14@336 by +7.02 in BA.
- Table 11 has deployment cost for CLIP-LoRA L/14@336, but not for this configuration.
- CLIP-LoRA is timed in FP16 autocast (its official code), while our rows are FP32. A fair same-backbone latency comparison therefore needs our configuration in FP16 as well. The FP16 accuracy must then be checked on the whole formal set, because the timing run measures no accuracy.

## Part A: timing (the RTX 4090 of P26 r2)

### The new blocks

`experiments/p26/cost_profile.py` gets two new blocks, `ours_l14_336_fp32` and `ours_l14_336_fp16`.

- **Backbone:** only CLIP ViT-L/14@336 is loaded (`build_clip_backbone`). DINOv2 is not loaded.
- **Features:** layer-18 patch tokens, taken exactly as in the main pipeline (`experiments.p14.resource_profile.Pipeline`).
- **Precision:** FP16 means CUDA autocast of the CLIP forward, as in P22.
- **Timed head:** `a_r` = z(SS[h_A]) + z(SS[h_R]). Before timing, it is checked against the evaluation code (max |d| ≤ 1e-4).
- **Offline and online steps:** the same as `experiments.p18.resource_profile.profile`: warm-up, support and bank features, leave-one-out τ, support state, then end-to-end and head-only timing.
- **Other blocks:** unchanged.

A local smoke on the laptop checked only that the blocks run; its numbers are not used. Result: head check 2.4e-7, GFLOPs 381.92.

### The run

- The whole P26 suite (all 12 blocks, forward then reverse order) runs again in one run, tag `rtx4090_r3`. This keeps every row of Table 11 "same machine, same run".
- r3 replaces r2 in Table 11, supplementary Table S8.4 and Fig. 10. r2 stays archived.
- The only file uploaded to the cloud copy of the bundle is the new `cost_profile.py`. The old file is kept as `cost_profile.py.r2`.
- On any failure: stop and report. No relaunch without the author.

## Part B: FP16 accuracy (local laptop GPU, report only)

`experiments/p30/fp16_accuracy.py` does the following.

1. **Re-extraction.** It re-extracts the CLIP-18 patches of every MVTec-FS formal image: support pool, queries and normal images.
   - It uses the P10 path: same order, batch 4.
   - Each batch is extracted twice: in FP32, and in FP16 autocast.
2. **Gate 1.** The FP32 re-extraction must equal the P10 caches bitwise.
3. **Scoring.** It scores `a_r` with the P14 code path for all 13 products × 3 K × 5 seeds (M = 20): once from the cached features, once from the FP16 features.
4. **Gate 2.** The cached features must reproduce P17's `a_r` metrics exactly.
5. **Reported:**
   - FP16 − FP32 BA: product-cluster bootstrap, per K and pooled;
   - per-query prediction agreement.

**Decision rule (fixed now):**
- If the pooled |Δ| < 0.5 pp and its 95% CI contains 0: the paper states that FP16 does not change the accuracy of this configuration, and reports the FP16 latency with the FP32 accuracy.
- Otherwise: the FP16 row reports its own accuracy from this check, and the difference is stated.

## Reporting

- Table 11 gets two rows: ours, CLIP-L/14@336 only, FP32 and FP16.
- The §7.8 same-backbone paragraph gains the L/14 pair.
- Supplementary Table S8.4 and Fig. 10 use r3.
- Supplement S2 records both parts.

## Amendment A1 (2026-10-01, before any r3 output)

- **What happened:** the cloud agent stopped at the harness check. The RTX 4090 holds 54a4749 (LF sha `61bc4c82…`). That is the harness r2 actually ran. The local commit 913db22 only added a machine-info recorder after r2 and was never uploaded.
- **Change:** r3 runs 54a4749 plus the two P30 blocks only.
  - The exact file is `docs/agent/p30_cloud_cost_profile_r3.py`, LF sha `22e3ede6…`.
  - It is written on the cloud by `docs/agent/p30_cloud_patch_v2.py` (sha `ad71c8fa…`).
  - The two blocks are byte-identical to those of c3253c6. The machine-info recorder is left out, because this machine was already documented at r2.
- **Consequence:** r3 = the r2 harness + the two new blocks. Nothing else in the protocol changes.
