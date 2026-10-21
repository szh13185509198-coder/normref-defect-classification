# P32 protocol (locked before any device extraction): does the method keep its accuracy on the Jetson?

Report only. Nothing about the methods, constants or earlier results changes.

## Question
On the Jetson Orin Nano (P31 environment: L4T R36.5, PyTorch 2.8.0), do the features and the predictions of
(a) the main configuration (ARR = `fused_adareg_regres_soft_q99`: global SimpleShot + h_A + h_R) and
(b) the CLIP-L/14@336-only head (P17 `a_r`)
match the GPU results on MVTec-FS, in FP32 and in FP16 autocast?

## Procedure
1. On the device, `experiments.p32.device_extract` re-extracts, with the P10 code path (`formal_extract._extract`), the
   same images in the same order and batch size (4) as the P10 caches: support pool, formal queries, formal normal
   images (1,124 names from `ckpt/p32/normal_names.json`), for all 13 products, once in FP32 and once in FP16 autocast.
   Kept: global fusion feature and CLIP layer-18 patch tokens (L2-normalised, FP16 storage as in P10).
2. Locally, `experiments.p32.score_device` (written before the device features arrive):
   - checks that every id list equals the P10 cache order;
   - feature-level comparison with the P10 caches: max |Δ| and mean cosine, per product and group;
   - scores ARR and `a_r` for every product / K / seed (M = 20) with the unchanged evaluation code
     (`experiments.p14.final_formal` path for ARR, `experiments.p30.fp16_accuracy.score_product` for `a_r`), from the
     cached features (gate: must reproduce P14 ARR and P17 `a_r` exactly) and from the device FP32 / FP16 features;
   - paired product-cluster bootstrap (`c1_paired_bootstrap.contrast`) of device − cache, pooled over K, and the
     per-query prediction agreement.
3. Bitwise equality with the GPU features is **not** expected (different GPU, kernels and library versions).

## Decision rule (fixed now; same as P30)
For each of the four comparisons (ARR / `a_r` × device FP32 / device FP16, each against the cached GPU features):
if the pooled |Δ BA| < 0.5 pp and its 95 % CI contains 0, the paper states that the method keeps its accuracy on the
device at that precision; otherwise the paper reports the device accuracy and the difference.
