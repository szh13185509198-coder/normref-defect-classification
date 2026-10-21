# P34 protocol (locked before the run): FP16 accuracy of our method on DINOv2-B and CLIP-B/16

Report only; nothing about the methods, constants or earlier results changes.

## Why
The Jetson timing (P31 r3) adds FP16 rows for our method on DINOv2-B@336 and CLIP-B/16@224 / @336. FP16 accuracy is
already checked for the main configuration (P32), CLIP-L/14@336 alone (P30) and DINOv2-S (P22), but not for these three.

## Procedure (laptop RTX 4050, `experiments.p34.fp16_alt`)
- Backbones and layers as reported: DINOv2-B@336 layer 11 and CLIP-B/16@336 layer 8 (P18 dev selection), CLIP-B/16@224
  with the P24 layer.
- MVTec-FS formal set: support pool, 530 queries, formal normal images; P18 / P24 extraction code (`bb.extract`, batch 4),
  once in FP32 and once under CUDA autocast FP16 (whole extraction call, as the Jetson FP16 blocks).
- ARR (`final_formal.episode_scores`, M = 20) for every product / K / seed.
- Gate: the FP32 rows reproduce P18 `arr_dinov2_b`, `arr_clip_b16` and P24 `arr_clip_b16_r224` exactly.
- FP16 − FP32: paired product-cluster bootstrap pooled over K, per-K differences, per-query agreement.

## Decision rule (fixed now; same as P30 / P32)
If the pooled |Δ BA| < 0.5 pp and its 95 % CI contains 0, the paper states that FP16 keeps the accuracy of that
configuration and reports the FP16 timing with the FP32 accuracy; otherwise it reports the FP16 accuracy.
