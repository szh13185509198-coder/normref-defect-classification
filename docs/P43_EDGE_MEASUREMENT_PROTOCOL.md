# P43 — Edge measurements of the TensorRT deployment (report only)

Status: **LOCKED BEFORE ANY P43 OUTPUT** (2026-10-02). Sequentially added deployment measurements on the rented
Jetson Orin Nano 8 GB, after P33 15 W and before the rental ends; disclosed in S2. No method, constant or earlier result
changes. Code: `experiments/p43/edge_measure.py`, `docs/agent/p43_int8.sh`.

## 0. Conditions (all parts)
- **Device state**
  - MAXN_SUPER with clocks locked: set by the user's agent; verified by Claude from sysfs before and after each part.
  - Headless; `drm_holder.py` running.
  - No sudo: the page cache is not dropped; file reads are warm, and this is disclosed.
- **Run rules**
  - Claude runs every command over key-based SSH.
  - Each command runs once. A failure stops the part and is reported, not retried.
  - Every output is write-once.
- **Engines**: the validated P35 / P41 engines:
  - `main_fused_r8.engine`;
  - `dinov2_s_336_ex_r5`, `dinov2_b_336_ex_r5`, `clip_l14_l18_r6`;
  - the P38 `ENGINES` table.
- **Order**: B (latency, writes the task-state files) → B cold starts → A (energy) → C (single-backbone accuracy) → D (INT8).
  D comes last because engine builds have triggered the DRM crash before.

## A. Energy per image of the TensorRT pipelines
- **Configurations**: one process each, 60 s of back-to-back calls with a CUDA synchronisation after every call, 20
  warm-up calls, 10 s pause between configurations, and a 30 s idle window first.
  - `fused_engine_only`: the P41 engine on a preprocessed input, with the bank, τ and prototypes resident.
  - `fused_e2e`: from the decoded image: GPU preprocessing, the engine, and the host reading back the class index and the
    screening score (= P41 `e2e_from_pil`).
  - `gate_dino_pass`: the P38 light-gate pass path.
  - `gate_l18_pass`: the P38-A1 layer-18 pass path.
  - `trt_s336_engine` and `trt_b336_engine`: the DINOv2-S / DINOv2-B engine forward only, comparable in scope with the
    P33 "backbone forward" energy.
  - Each pass path is checked against the P15 formula before its window, as in P38.
- **Power**: `tegrastats --interval 100` runs as the normal user and logs `VDD_IN`. If `VDD_IN` is missing without root,
  stop and ask the user's agent.
- **Energy**
  - Energy per call = mean `VDD_IN` over the window × mean time per call (the P33 definition).
  - Also reported: dynamic energy = (mean `VDD_IN` − idle mean) × mean time per call.
  - The P33 rows are backbone forward only (PyTorch), so they are compared with the `*_engine` rows. End-to-end rows are
    reported as energy per inspected part.

## B. Latency distribution, cold start and product switching (fused engine)
- **B1, distribution**
  - 10,000 consecutive end-to-end calls (`fused_e2e`, bottle query, 50 warm-up calls).
  - Reported: mean, SD, min, p50 / p90 / p95 / p99 / p99.9, max, and the count above 2 × median.
  - The raw samples are kept.
- **B2, product switching**
  - The second product is fixed here: **capsule**, chosen because its class count (5) differs from bottle (3), so the
    dynamic class dimension changes on every switch.
  - Its K = 5 seed-111 support set comes from the stage-1 manifest. If the manifest is not on the device, the first
    five support-pool images per class are used instead, and this is recorded.
  - Its bank comes from the P41 seed-111 bank draw.
  - Reported:
    - the new-task time through the engine;
    - the size of each saved task state on disk;
    - the time to load a state from disk onto the GPU (10 repetitions);
    - 200 calls on the same product vs 200 calls alternating between the two products, both states resident. Class
      decisions must be identical between the two orders and screening scores within 1e-5; otherwise stop.
- **B3, cold start**
  - Three fresh processes, each launched by the shell with its launch time.
  - Milestones:
    - interpreter at the module;
    - numpy / torch / tensorrt imported;
    - CUDA context;
    - engine deserialised with its context;
    - bottle task state loaded onto the GPU from the B2 file;
    - query image decoded;
    - first decision returned.
  - Then the median of 50 steady-state calls.

## C. TensorRT accuracy of the single-backbone configurations (removes the limitation "features only")
- **Configurations**: DINOv2-S@336 (layer 12, the P22 selection) and DINOv2-B@336 (layer 11, the P18 / P22 selection).
  Both use their P35 `_ex` FP16 engines through the P38 `TrtDino` pipeline (global = layer-12 CLS, patches at the
  selected layer).
- **Device**
  - For every MVTec-FS formal product, the engine features of the support pool, the queries and the product's normal
    images (P32 lists and order).
  - Then, per bank seed (the P41 draws, M = 20): τ = `formal_ab.tau99`, and the C1 descriptors
    `experiments.p18.run.c1_parts` (PyTorch FP32 head on the device).
  - Saved: global features, soft appearance and residual descriptors, and τ.
- **Local**
  - The PyTorch FP32 reference is recomputed: P22 code path, `lightweight.build`, `backbones.extract`, batch 4.
  - **Gate**: the reference `arr_*` rows must reproduce the P22 formal rows exactly (`arr_dinov2_s_r336_fp32`,
    `arr_dinov2_b`).
  - Both feature sets are scored with `final_formal.episode_scores`. Only the ARR score is used; the support-only inputs
    are dummies, as in P34.
- **Decision rule (P30 / P32)**: "keeps accuracy" iff |Δ| < 0.5 pp and the 95 % paired product-cluster CI contains 0.
  Also reported: per-query agreement and the descriptor cosine.

## D. INT8 latency only (not used for any accuracy claim in this paper)
- **Engines**: `trtexec --int8 --fp16` without calibration data. TensorRT then uses placeholder dynamic ranges, so the
  outputs are not meaningful; only the GPU compute time is reported.
- **Configurations**: `clip_l14_336`, `dinov2_s_336_ex`, `dinov2_b_336_ex`.
- **Anchors**: the existing FP16 engines of the same graphs are re-timed in the same session with identical `trtexec`
  timing flags (warm-up 2 s, 200 iterations, spin wait, no data transfers).
- **Use**: the result is reported in the supplement as an exploratory note (whether INT8 is faster on this device) and
  feeds a possible follow-up study. It makes no claim about INT8 accuracy.

## Reporting
All parts are report only, with no selection. Supplement S8 gets the tables, S2 the run history, S2.4 the hashes. The
main text may cite A (energy per part), B1 (p99) and C (keeps accuracy or not); D stays in the supplement.

## Amendment A1 (2026-10-02, before any P43 output): unattended night chain
At the user's request, P43 runs unattended overnight, right after P33 15 W, in one root chain
(`docs/agent/p43_night_chain.sh`). The user's agent starts the chain once with sudo. The chain runs the steps below in
this order and stops at the first failure (no retry, no deletion).

1. Wait for P33 15 W (or run it if it never started) and pack its results.
2. Check MAXN_SUPER, locked clocks (`jetson_clocks --show`, MinFreq = MaxFreq for every core and the GPU) and
   `drm_holder.py` before each part.
3. Run B, then the three cold starts.
4. Run A. `tegrastats` runs as root here, so `VDD_IN` is guaranteed.
5. Run C, then pack A–C.
6. Run D and pack it.

Changes to section 0:
- The Python steps still run as the normal user.
- The page cache **is** dropped (`echo 3 > /proc/sys/vm/drop_caches`) before each cold start. B3 therefore measures a
  true cold start from disk, which replaces "warm file reads" for B3. Parts A, B1 / B2 and C are unaffected (they load
  their files once before measuring).
- Claude only uploads the code and fetches the packed results; nothing else changes.

## Amendment A2 (2026-10-03, before any A2 output): three follow-up measurements
Added at the user's request after the night chain. Report only. Claude monitors and runs the MAXN parts over SSH; the
power-mode part needs root and is started by the user's agent with `docs/agent/p43_modes.sh`.

- **B4, the 63 vs 67 ms question** (MAXN_SUPER, locked; two fresh processes `r1`, `r2`;
  `edge_measure --repeat --phases ABCDE`, 2,000 calls per phase).
  - Phases:
    - A: after setup;
    - B: the same again;
    - C: build the capsule task state (no calls);
    - D: bottle again;
    - E: bottle with its state loaded from the B2 file.
  - Each phase reports:
    - the end-to-end distribution (no inner synchronisation, as B1);
    - split medians (GPU preprocessing / engine / host read-back, synchronised between);
    - the GPU clock from sysfs at the start and end.
  - Purpose: locate the difference between B1 (67.0 ms) and the later measurements (63.3–63.7 ms). If it cannot be
    located, both levels stay reported.
- **B5, decoding** (MAXN_SUPER): CPU time to read and decode each of the 530 formal query images with the pipeline's
  PIL loader, three reads each. Reported: the first read and the repeat reads, per product with image size. The page
  cache is not dropped (no sudo).
- **A3, the fused engine in the other power modes** (25 W and then 15 W, each followed by MAXN_SUPER; root script).
  - Per mode: `jetson_clocks`, `tegrastats` at 100 ms, a 30 s idle window.
  - Then 60 s energy windows for `fused_e2e`, `fused_engine_only` and `gate_dino_pass`.
  - Then one `--repeat --phases AB` process with 2,000 calls per phase.
  - Tags: `jetson_orin_nano_25w`, `jetson_orin_nano_15w`.
- **Stop rule**: any failure stops the part and is reported; nothing is retried, and outputs are write-once.
