# P41 — The whole method as one TensorRT engine with the normal reference as runtime input (report only)

Status: **LOCKED BEFORE ANY P41 OUTPUT** (2026-10-02). Sequentially added deployment analysis, disclosed in S2. No
method, constant or earlier result changes.

## 1. Design
**One FP16 engine** contains:
- CLIP-L/14@336, giving the projected embedding and the layer-18 tokens;
- DINOv2-S, with the explicit-attention export of P35;
- the fused global feature;
- the complete dual-branch head:
  - nearest neighbour against the normal bank;
  - patch anomaly and the calibrated soft weights, with the top-6 fallback;
  - appearance and deviation descriptors;
  - three SimpleShot scores against the class prototypes, z-normalisation and the sum;
- the P15 screening score.

**Runtime inputs, not weights:** the normal bank (M = 20 images, 11,520 tokens), the calibrated threshold τ, and the
centring vectors and prototypes of the three spaces. The number of classes is a dynamic dimension (2–16). A new defect
type or a new product changes only these inputs; the engine is never rebuilt.

**Preprocessing on the GPU:** the decoded image is uploaded as 8-bit RGB, then resized to 336 × 336 with bicubic
anti-aliased interpolation and normalised with the CLIP statistics in PyTorch on the GPU. This replaces the PIL resize
on the CPU and changes the input numerics slightly.

## 2. Checks before any timing (fail closed)
- **PyTorch module:** with the PIL-preprocessed input it must reproduce the evaluation code on the bottle timing task —
  scores within 1e-4 of `p18.resource_profile` ARR, and the screening score equal to the P15 formula.
- **ONNX export:** onnxruntime vs PyTorch, outputs within 1e-3 in absolute value, or cosine ≥ 0.9999 for the score
  vector.
- **Engine:** class scores and screening score vs onnxruntime on the timing task; reported.

## 3. Latency (Jetson, MAXN_SUPER, locked; P26 / P38 conventions, two passes)
- **End to end from the decoded image:** GPU preprocessing plus one engine call, with the host reading back the class
  index and the screening score.
- **Compared with:** P38's TensorRT main pipeline (two engines + PyTorch head + CPU preprocessing) and the PyTorch FP16
  pipeline of Table 12.
- **New task:** support and bank feature extraction through the engine, LOO calibration (PyTorch, once) and
  prototypes — no rebuild.

## 4. Accuracy (MVTec-FS formal set, main configuration)
- **On the device, per product and bank seed:**
  1. the bank tokens come from the engine;
  2. τ is calibrated (LOO, PyTorch FP32);
  3. every support and query image of the P32 lists goes through GPU preprocessing and the engine, which outputs the
     fused global feature and the two descriptors.
- **Locally:** the episodes are scored with the unchanged SimpleShot / z-sum code. The cached features must first
  reproduce P14 exactly.
- **Decision rule (P30 / P32):** "keeps accuracy" iff |Δ| < 0.5 pp and the 95 % CI contains 0 (paired product-cluster
  bootstrap, engine minus cache); otherwise the engine's accuracy is reported alongside.
