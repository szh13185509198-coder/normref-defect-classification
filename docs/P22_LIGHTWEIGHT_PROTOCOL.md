# P22 protocol — lightweight single-backbone configurations (report-only)

Locked 2026-09-28, before any P22 output exists. Code: `experiments/p22/lightweight.py`.

## Purpose
The main pipeline (CLIP ViT-L/14@336 + DINOv2-S) takes 118 ms and 2 GB per image (P18-A8). P18
showed that DINOv2-B alone at 336 px is about as accurate as the main pipeline (−1.2 / −1.9 / −0.5 pp,
all n.s.) at 42 ms. P22 maps the accuracy–cost trade-off below that point, for later edge-device
deployment (Jetson-class latency is measured separately). The method is unchanged: the same ARR head,
τ = LOO 0.99 quantile, M = 20 normals, top-6 fallback, and equal z-fusion. Only the feature extractor
changes.

## Grid (12 configurations)
- Backbone:
  - DINOv2-S/14 (the weights of the main pipeline);
  - DINOv2-B/14 (the P18 weights).
- Input: 224 / 336 / 448 px, with the main view's squash resize and CLIP normalisation.
- Precision: FP32 / FP16 (CUDA autocast of the backbone forward pass; the heads stay in FP32).
- One forward pass per image gives two features: the global feature (final normalised CLS token) and
  the patch tokens of one layer. The layer comes from {6, 8, 9, 10, 11, 12}.

## Stage dev (dev episodes only; no formal query is read)
- Data: the MVTec-FS support pool with the P10 dev episodes (seeds 9001–9005, K = 1/3/5), and the
  MVTec AD train/good normals with the P10 draws.
- Heads: SimpleShot, fixed fraction, A, ARR, and the mask oracle, per configuration and layer.
- **Integrity:** DINOv2-B / 336 / FP32 must reproduce the P18 dev rows exactly (max|Δ| < 1e-12).
- **Layer selection:** for each configuration, the layer with the highest mean dev ARR BA. Ties go to the
  shallower layer, as in P18.
- **Recommendation rule:** a configuration is eligible if its dev ARR BA ≥ that of DINOv2-B / 336 / FP32
  minus 1.0 pp. That anchor is the configuration P18 validated on the formal sets. The recommended
  configuration is the eligible one with the lowest batch-1 ARR end-to-end latency in stage profile.
- Per-product checkpoints are kept, and a stopped run continues only with `--resume`, after the user
  authorizes it.

## Stage profile
- Setting: batch-1 latency and peak memory of every selected configuration next to the main pipeline,
  on the same setting as A8 (bottle, K = 5, seed 111, M = 20), in one session, on the local RTX 4050
  laptop GPU.
- These are relative numbers; absolute edge-device numbers come later.
- Every timed head is checked against the evaluation code.

## Formal addendum (locked 2026-09-28, after dev + profile, before any formal P22 output)
- **Decision:** lightweight is a supporting result, not a contribution. The formal set is therefore
  restricted to the recommended backbone: **DINOv2-S / 336 / layer 12, in FP32 and FP16**. FP16 is
  included because edge devices run FP16.
- The other ten configurations are dev-only and are reported as such.
- **Datasets:** MVTec-FS (13 products) and VisA + MTD (the frozen P10 external bundle). Real-IAD is not
  used here.
- **Integrity:** the same code also recomputes DINOv2-B / 336 / FP32 at the P18 layer (11). It must
  reproduce the P18 formal rows exactly on both stages.
- **Declared contrasts on BA:**
  - (a) S32 ARR − S32 SimpleShot (the principle under a 22M backbone);
  - (b) S32 ARR − main ARR (accuracy cost);
  - (c) S32 ARR − DINOv2-B ARR;
  - (d) S16 ARR − S32 ARR (precision);
  - (e) S32 A − S32 fixed fraction (normal calibration).
- **Statistics:** MVTec-FS: product cluster bootstrap and exact sign-flip. VisA: object bootstrap and
  exact sign-flip. MTD: descriptive, per K.
- **Report-only:** every cell is reported and nothing is selected. Code: `experiments/p22/formal.py`,
  with per-product / per-task checkpoints and `--resume` only after authorization.

## Later
Formal MVTec-FS, VisA and MTD for all 12 selected configurations, report-only (every cell reported).
The recommended configuration may additionally be run on Real-IAD. Nothing in P22 changes the main
method or its reported results.
