# P45 — Production-line simulation of the deployed method on the Jetson (report only)

Status: **LOCKED BEFORE ANY P45 OUTPUT** (2026-10-03, user decision). Sequentially added system-level measurement;
disclosed in S2. No method, constant or earlier result changes. Code: `experiments/p45/line_sim.py`.

## 1. What is simulated
- **Lines**: thirteen product lines, run one after another in the fixed `rc.PRODUCTS` order. Each line uses the single
  fused TensorRT engine of P41 (`main_fused_r8.engine`), MAXN_SUPER with locked clocks, headless, `drm_holder.py`
  running. Claude runs the commands over SSH as the normal user (no sudo).
- **Task state per product (changeover)**:
  - K = 5 seed-111 supports (stage-1 manifest) and the P41 seed-111 bank draw (M = 20).
  - Built through the engine as in P41; the build time is reported as the changeover time and excluded from the line
    throughput.
- **Held-out normals**: the product's normal images not in the bank. They are split deterministically by sorted name,
  alternating, into a calibration half and a stream half.
- **Screening thresholds**:
  - Split-conformal from the engine's screening scores on the calibration half: reject at α = 5 %, accept at
    α = 30 % (the P15 three-way setting with the reject level relaxed to what about 33 calibration images allow);
    k = ⌈(n + 1)(1 − α)⌉, and k > n means no automatic rejection.
  - The decision is computed in the loop so that its cost is included. Its correctness is **not** evaluated: the
    MVTec-FS normal pool is too small for screening statistics, which are reported in §7.7 on VisA / MTD / Cable.
- **Stream**: the stream-half normals plus the product's formal query images (defects), shuffled deterministically
  (`p45|<product>`), three rounds (about 250 parts per product, about 3,300 in total).

## 2. Three runs (fresh process each; write-once)
- `file_seq`: per part, read the PNG from disk and decode it on the CPU (PIL), then GPU preprocessing, the engine, host
  read-back and the three-way decision. This is the upper bound for a compressed-image input.
- `file_pipe`: as `file_seq`, but a producer thread reads and decodes the next parts into a bounded queue (depth 4)
  while the main thread runs the GPU part (CPU / GPU overlap).
- `frame_seq`: frames already decoded in memory as 8-bit RGB arrays (a camera frame buffer), then the same GPU part.
  The GPU preprocessing takes the array directly and is numerically identical to P41 `gpu_prep`.

## 3. Reported per run
- **Throughput**: parts per second over the stream (from the first read to the last decision; changeovers excluded).
- **Per-part latency** (from the start of reading the part, or taking the frame, to the decision on the host,
  including any queue wait): p50 / p90 / p99 / max.
- **Per product**: changeover time and part count.
- **Energy per part**: mean `VDD_IN` over the stream windows (`tegrastats` at 100 ms, normal user) × wall time / parts.
  Also the dynamic part, using a 30 s idle window taken before the runs.
- **Peak resident memory** of the process.

## 4. Order and stop rule
`idle` → `frame_seq` → `file_seq` → `file_pipe`. Any failure stops the series and is reported; nothing is retried.

## Amendment A1 (2026-10-03, before any A1 output): the same line for the other deployable configurations
At the user's request the line simulation is repeated for:
- CLIP-L/14@336 blocks 1–18 (`a_r`, no global score);
- DINOv2-B@336 and DINOv2-S@336;
- CLIP-B/16 at 224 and 336 px.

Pipelines: each uses its validated P38 TensorRT pipeline, unchanged (the same as Table 13's TensorRT column):
- the configuration's own preprocessing on the CPU;
- its FP16 engine;
- the evaluation head in PyTorch: dual-branch ARR, or `a_r` for CLIP-only, with the top-6 fallback.

Checks and settings:
- **Head check**: before the first stream, the per-part head must equal the evaluation code (whole-tensor
  `simpleshot` / `region_weights` / `region_residual`) within 1e-4 on a query of the first product; otherwise stop.
- **Screening score**: each configuration's own patch anomalies, top 1 % mean (the P15 statistic; for DINOv2-S this is
  the P37 light gate). As in the main run, decisions are computed but not evaluated.
- **Runs per configuration**: `frame_seq` (decoded images in memory; CPU preprocessing per part) and `file_pipe`.
  `file_seq` is not repeated: the decoding cost does not depend on the configuration and is already measured.
- **Energy**: `tegrastats` at 100 ms as before; one 30 s idle window first.
- Everything else as sections 1–4. Order: clip_l18, dino_b, dino_s, clipb16_224, clipb16_336, each frame_seq then
  file_pipe. Stop at the first failure.

## Amendment A2 (2026-10-03, before any A2 output): one hour of continuous production
At the user's request, right after P46:
- **Lines**: the fused engine (frame mode, as `frame_seq`) serves the first 12 product lines one after another, 300 s
  each (3,600 s in total, 11 changeovers in between). Each line streams its held-out normals and defect queries in
  reshuffled rounds; no calibration split is needed because no decision thresholds are reported.
- **Per-minute windows**: parts, latency p50 / p90 / p99 / max, and the GPU clock from sysfs.
- **Logging**: `tegrastats` at 1 s for power, temperatures and clocks.
- **Report**: drift of throughput and latency over the hour, maximum junction temperature, mean board power, and any
  clock change.
