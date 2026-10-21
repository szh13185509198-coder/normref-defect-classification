# P47 — Scaling and tiling of the deployed method (report only)

Status: **LOCKED BEFORE ANY P47 OUTPUT** (2026-10-03, user decision after an external review). Sequentially added;
disclosed in S2. No method constant, earlier result or decision rule changes. Code: `experiments/p47/scale_tile.py`.

## 1. Questions
1. **Scaling**: how do the new-task time, the per-part latency and the memory grow with the number of normal images
   M, the number of classes C and the number of products (SKUs) whose state stays in memory?
2. **Resolution**: every result so far resizes the whole image to 336 px. What do accuracy and device latency become
   when the image is cut into g × g tiles, each fed at the backbone's native 336 px (g = 2: about 672 px; g = 3: about
   1008 px, close to the native MVTec AD resolution)?

## 2. Tiling (fixed here, no selection)
- **Tiles**: the image is split into g × g non-overlapping regions with integer boundaries round(i·W/g), round(j·H/g).
  Each region is preprocessed exactly as a whole image is today (local accuracy: the locked deterministic 336 view;
  device: `gpu_prep`, bicubic anti-aliased resize to 336).
- **Global score g**: unchanged, from the whole image at 336 px (CLIP-L/14 pooled + DINOv2-S patch mean).
- **Patch evidence**: the layer-18 CLIP tokens of all tiles of an image form one patch set (g²·576 patches). The normal
  bank is the union of the tile patches of the same M = 20 normal images. Everything else is the unchanged evaluation
  code applied to this patch set: leave-one-image-out τ at the 0.99 quantile (a whole image, i.e. all its tiles, is
  left out), soft weights with the top-6 fallback, h_A, h_R, z-sum.
- **Tiles stop at block 18**: the tile tokens come from blocks 1–18 only (P36 `ClipSplit`, bitwise equal to the full
  forward's layer-18 tokens).
- **Screening statistic in the tiled engine**: the mean of the top ⌈0.01·g²·576⌉ patch anomalies (the P15 top-1 %
  rule applied to the larger patch set). Timing only; screening accuracy with tiles is not evaluated.

## 3. Accuracy (local, laptop GPU, PyTorch FP32)
- **Data**: MVTec-FS formal set, 13 products × K ∈ {1, 3, 5} × 5 seeds; the P41 bank draws (M = 20).
- **Grids**: g = 1 (same code path, no tiling), then g = 2, then g = 3, in this order.
- **Gate**: g = 1 must reproduce the P14 cached ARR predictions (descriptor cosine and BA difference reported; a BA
  difference above 0.1 pp stops the round).
- **Reported**: BA per K and the paired difference g = 2 / g = 3 minus g = 1, with the product-cluster bootstrap CI
  (the P41 scoring code, unchanged).
- **Memory-only changes**: anomaly and residual computations run one image (and, for the LOO threshold, one block of
  patches) at a time. This changes memory, not the definition.
- **Reading**: descriptive. No grid is selected; whichever result comes out is reported.

## 4. Device timing (Jetson Orin Nano 8 GB, MAXN_SUPER, clocks as set by the user's agent; normal user over SSH)
- **New engines**: built on the device from ONNX exported on the device, with the same `trtexec` options as
  `main_fused_r8`:
  - `main_fused_dyn`: P41 module, bank dimension dynamic (576·5 to 576·200);
  - `tiled_g2`, `tiled_g3`: the tiled module, static shapes, M = 20.
- **Parity checks before timing**:
  - each engine against its PyTorch module on the device (maximum score difference reported);
  - `main_fused_dyn` at M = 20 against `main_fused_r8` (same scores within FP16 tolerance; latency within 3 %).
- **Harness**: P41 conventions (bottle, K = 5, seed 111, batch 1; warm-up 20, 100 timed calls, median). Two passes,
  the second in reverse part order.
- **Parts**:
  1. **M ∈ {20, 50, 100, 200}** (`main_fused_dyn`): new-task time (bank tokens, LOO τ, supports, prototypes),
     engine-only latency, end-to-end from the decoded image, peak GPU memory. The device holds 87 bottle normal
     images; M > 87 cycles through them (timing does not depend on content; disclosed).
  2. **C ∈ {2, 4, 8, 16}** (`main_fused_r8`, M = 20): engine-only latency.
  3. **SKUs S ∈ {1, 10, 40}** resident banks (M = 20, 45 MB each in FP32): memory and engine-only latency when
     consecutive parts use different SKUs.
  4. **Tiles g = 2, 3**: new-task time, engine-only latency, end-to-end from the decoded image (GPU tile preprocessing
     included).
- **Not done here**: Real-IAD on the device; screening accuracy with tiles; INT8 accuracy.

## Amendment A1 (before computing it): decision flips of the deployed engine
P41 reported a 96.6 % agreement between the engine and the cached reference. A1 splits the disagreements of the same
P41 scoring (same features, same code) into reference-correct → engine-wrong and reference-wrong → engine-correct
(the rest: both wrong, different labels), per K and in total. Descriptive; nothing else changes.

## Amendment A2 (before any g = 3 device output): g = 3 with M = 10 on the device
In run r4 the g = 3 preparation ran out of memory on the 8 GB device (PyTorch FP32 reference: 5,184 tile patches
against a 103,680-patch bank). This is reported as a result: on this device, g = 3 with M = 20 does not fit the
reference path. For device timing only, g = 3 uses the first M = 10 images of the same bank draw (51,840 patches);
everything else as §4. Local accuracy for g = 3 stays at M = 20 as locked.

## Amendment A3 (before any g = 3 output): row-blocked nearest neighbour for g = 3
The A2 attempt with M = 10 failed again on a single ~1 GiB allocation (5,184 x 51,840 FP32 similarities); r4 failed
on an allocation of the same size. For g = 3 only, the tiled head finds each patch's nearest normal patch in 4 blocks
of rows (max and index concatenated). This is the same result with a smaller peak; g = 2 and the dyn engine are
unchanged.

## Amendment A4 (before any output): screening with 2 x 2 tiles on the device
P46 repeated, unchanged except that the engine is `p47_g2.engine` (2 x 2 tiles, §2) instead of `main_fused_r8`:
- the same tasks, P15 bank draws (M = 20), calibration sets (N = 100) and queries (`p46_ext_pack`, on the device);
- per task and seed: bank = the engine's tile tokens of the 20 bank images; screening score = the engine's own
  `screen` output (top ⌈1 %⌉ of the 2,304 tile-patch anomalies);
- each image is decoded and preprocessed once; the five seeds' banks stay in memory (memory and time only);
- scored locally with the unchanged P46 code (τ_α, realized false-reject rate, AUROC, guarantee check), compared with
  P15 and with P46.
Descriptive; no selection.
