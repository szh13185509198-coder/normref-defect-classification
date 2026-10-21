# P48 — Calibrate what you deploy: patch-level false selection on the deployed engine (report only)

Status: **LOCKED BEFORE ANY P48 OUTPUT** (2026-10-03, user decision; to-do A11). No method constant, earlier result or
decision rule changes. Code: `experiments/p48/patch_calib.py`.

## 1. Question
The evidence threshold τ is the 0.99 quantile of leave-one-image-out patch anomalies inside the normal bank (§5.5), so
on new normal images about 1 % of patches should exceed it. Does this hold when the patch tokens come from the deployed
fused TensorRT engine (GPU preprocessing, FP16) instead of the FP32 evaluation path, and what happens when τ is
calibrated on one path and used on the other?

## 2. Data
MVTec-FS formal products (13) × the five P41 bank draws (M = 20). Held-out normals: the product's P10 normal pool
minus the 20 bank images of that seed (train/good images only; never a query or support image).

## 3. Measurements
For every product and seed, the realized false-selection rate = fraction of held-out normal patches with a > τ, for:
- **R (reference)**: FP32 cached tokens (P10 path), τ_ref from the same tokens;
- **E (deployed, recalibrated)**: engine tokens on the device, τ_eng from the engine tokens (the LOO rule unchanged);
- **X (mismatch)**: engine tokens with τ_ref (calibrated on the reference path, used on the engine).
R is computed locally, E and X on the device (τ_ref is uploaded). Reported: product macro over seeds of each rate,
the per-product range, and τ_eng − τ_ref.

## 4. Reading (descriptive)
Nominal value 1 %. No selection; whatever comes out is reported. Not done: INT8 (no speed-up on this device), other
backbones.
