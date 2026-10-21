# P38 — TensorRT deployment on the Jetson: end-to-end latency and accuracy (report only)

Status: **LOCKED BEFORE ANY P38 OUTPUT** (2026-10-02). Report-only deployment analysis, added after all earlier results
(disclosed in S2). No method, constant or earlier result changes.

## 1. Engines
TensorRT 10.3 FP16 engines built on the device in P35 whose outputs were validated against PyTorch FP32:
- `clip_l14_336_r2` (CLIP-L/14@336: projected embedding + layer-18 tokens);
- `dinov2_s_336_ex_r5` and `dinov2_b_336_ex_r5` (DINOv2 with explicit attention; fix of P35 round 2);
- `clip_b16_224`, `clip_b16_336_r2`.

The main configuration runs the CLIP-L/14 engine and the DINOv2-S engine on the same preprocessed tensor, as the
PyTorch pipeline does (same DINOv2-S weights as the single-backbone configuration). The heads are unchanged (PyTorch,
FP32).

## 2. Latency (bottle, K = 5, seed 111, M = 20, batch 1; P26 conventions)
- **Configurations:** main, DINOv2-B, DINOv2-S, CLIP-B/16 at 224 and 336, each with TensorRT backbones.
- **Gate:** the light-gate pass path (DINOv2-S engine + DINOv2-S bank nearest neighbour + screening score).
- **Reported:** per-task preparation, engine time, end-to-end time from the decoded image (CPU preprocessing included)
  and head time, using `cost_profile` / `p18.resource_profile` timing.
- **Runs:** one process per configuration, two passes (forward and reverse order), MAXN_SUPER with locked clocks.
- **Head check:** before timing, each head must match the evaluation code on the same TensorRT features (≤ 1e-4).
- **Feature check:** the TensorRT features of three images are compared with the PyTorch FP32 pipeline; cosine is
  reported, there is no pass/fail.

## 3. Accuracy (main configuration and CLIP-L/14 alone, MVTec-FS formal set)
- **Extraction:** the P32 image lists (support pool, queries, normal-bank images of all 13 products) go through the
  TensorRT main pipeline on the device, batch 1, giving file `trt_<product>.pt` in the P32 format.
- **Scoring:** locally with `experiments.p32.score_device`, with the unchanged code path and gates (the cached features
  must reproduce P14 / P17 exactly).
- **Comparison:** TensorRT minus cache, paired product-cluster bootstrap.
- **Decision rule:** that of P30 / P32 — "keeps accuracy" iff |Δ| < 0.5 pp and the 95 % CI contains 0; otherwise the
  TensorRT accuracy is reported alongside.

## Amendment P38-A1 (2026-10-02, before any output of these blocks)
At the user's request, one more engine: CLIP-L/14@336 truncated after block 18 (`clip_l14_l18`, exported through the P36
`ClipSplit`; its output must agree with the PyTorch FP32 layer-18 tokens like the other engines). Two more blocks, same
conventions as §2, new tag: `trt_l14_l18` (CLIP-L/14-only configuration `a_r`, head checked against the evaluation code)
and `trt_gate_l18` (two-level inspection pass path: preprocessing, truncated engine, CLIP bank nearest neighbour,
screening score; checked against the P15 formula).
