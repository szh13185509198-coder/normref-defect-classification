# P44 — A recent training-free CLIP classifier (GDA, ICLR 2024) on MVTec-FS (report only)

Status: **LOCKED BEFORE ANY P44 OUTPUT** (2026-10-03, user decision: "find a training-free method to compare").
Sequentially added after all other results; disclosed in S2. No method, constant or earlier result changes.
Code: `experiments/p44/gda.py`.

## 1. Method (official algorithm, nothing tuned on the test set)
**Source**: Wang et al., "A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation", ICLR 2024; official code
`github.com/mrflogs/ICLR24`. The following parts of `main_few_shots.py` are copied verbatim:

- **Class means and shared covariance**:
  - μ_c = class means of the support features;
  - precision = d · pinv((N − 1) · Cov + tr(Cov) · I) of the class-centred features;
  - W = μ · precision;
  - b = log(1/C) − ½ μ · precision · μ (uniform prior).
- **Logits**: 100 · f · T + α · (f · W + b), where T is the zero-shot text classifier.
- **Support augmentation**: `RandomResizedCrop(n_px, scale=(0.5, 1), bicubic)` + horizontal flip, 10 augmentation
  epochs (official `augment_epoch: 10`). Features are L2-normalised. Queries use the official CLIP test preprocessing.
- **Text classifier and loader**:
  - template "a photo of a {}.", the same as Tip-Adapter-F* and CLIP-LoRA in this study;
  - class names with "_" → " ";
  - `clip_classifier` of the vendored CLIP code, which is identical to the official one;
  - CLIP loaded with `clip.load` as in our CLIP-LoRA runs (official weights; FP16 on CUDA as loaded).
- **Information setting**: support-only + class-name text (same group as CLIP-LoRA and Tip-Adapter-F*). Zero training.
- **Randomness**: the augmentation is seeded per episode (`p44|<backbone>|<product>|K<K>|<seed>`).

## 2. α (the only hyperparameter)
- The official code grid-searches α ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100} on a separate validation split. MVTec-FS has
  none, so α is selected **once per backbone on our development tasks**:
  - the support-pool pseudo-tasks of `experiments.p10.run_pilot.episodes`;
  - K = 1 and 3, dev seeds 9001–9005, 13 products;
  - the selected α maximises the mean dev BA; ties go to the smaller α.
- The formal test queries are never used for this selection. The difference from the official validation-split search
  is disclosed.

## 3. Backbones and formal run
- **Backbones**: two, each paired with existing same-backbone results.
  - CLIP ViT-B/16 @224 (official default resolution): ours P24 `arr_clip_b16_r224`, CLIP-LoRA P21, SimpleShot P24
    `ss_clip_b16_r224`.
  - CLIP ViT-L/14@336px: ours CLIP-L/14-only `a_r` (P17 / P30), CLIP-LoRA P29.
- **Formal run**: the 195 episodes of `docs/p21/P21_MVTEC_FS_EPISODES.json` (same supports, queries and class order as
  CLIP-LoRA), run once per backbone with the dev-selected α.
- **Outputs**: per-episode predictions, plus the zero-shot predictions as a check against P21 / P29 (reported).
- **Scoring**: locally, with the frozen query contract.

## 4. Declared comparisons (paired product-cluster bootstrap, exact sign flip, Holm over G1–G4)
- G1: ours CLIP-L/14@336 only (`a_r`) − GDA L/14@336 (same backbone and resolution)
- G2: ours CLIP-B/16@224 (`arr_clip_b16_r224`) − GDA B/16@224 (same backbone and resolution)
- G3: GDA B/16@224 − CLIP-LoRA B/16@224
- G4: GDA L/14@336 − CLIP-LoRA L/14@336

Descriptive only (not tested):
- GDA − zero-shot CLIP;
- GDA B/16@224 − SimpleShot B/16@224;
- the dev BA of every α;
- levels at K = 1 / 3 / 5.

## 5. Jetson timing (report only, P26 conventions, MAXN_SUPER locked)
- **Task**: bottle, K = 5 (as Table 12), both backbones, official precision (FP16 as `clip.load` gives on CUDA).
- **New task**: text classifier + 10 augmented epochs over the 15 supports + the GDA fit.
- **Inference**:
  - end to end from the decoded image: official preprocessing on the CPU, image encoder, logits, host argmax;
  - image-encoder forward on a prepared tensor.
- **Repeats**: two passes (two processes).
- The timing does not depend on any accuracy result and may run before the formal run.

## Amendment A1 (2026-10-03, before the formal run): every α on the test set as a descriptive upper bound
The formal records also store the predictions for every α of the official grid. The score step then reports, for
each backbone:
- the test BA of every α;
- the best α on the test set ("test-oracle α"): an upper bound for GDA that uses test labels.

The test-oracle α is used only to show whether the α selection could explain the outcome. It is **not** used for any
declared comparison, which keeps the dev-selected α. Descriptive comparison added: ours − GDA (test-oracle α), same
backbone.

## Amendment A2 (2026-10-03, before any timing output): pseudo-inverse on the CPU on the Jetson
On the Jetson, `torch.linalg.pinv` on CUDA fails to load (`libtorch_cuda_linalg.so`: undefined symbol
`cusolverDnXsyevBatched_bufferSize` in the installed libcusolver). The system is not changed. For the device timing
only, the pinv of the d × d matrix (d = 512 / 768) is computed on the CPU and the result moved back to the GPU; the
rest of the fit is unchanged. The local accuracy runs keep the CUDA pinv. Disclosed with the timing. It slightly
overstates GDA's new-task time compared with a working CUDA pinv.

## Amendment A3 (2026-10-03): why CUDA linear algebra fails on the Jetson, and how GDA's new-task time is reported
- **Diagnosis** (read-only checks):
  - The Jetson AI Lab PyTorch 2.8.0 wheel needs a newer cuSOLVER than JetPack 6's 11.6.4.69 (CUDA 12.6.11): the
    symbol `cusolverDnXsyevBatched_bufferSize` is missing.
  - NVIDIA's `nvidia-cusolver-cu12==11.7.1.2` aarch64 wheel was tried, loaded from an isolated directory via
    `LD_LIBRARY_PATH` (nothing installed, nothing in the system changed). It loads, but its kernels fail on the Orin
    integrated GPU (`CUSOLVER_STATUS_EXECUTION_FAILED` in `gesvdj`); it is an SBSA build.
  - The MAGMA backend is not compiled into this wheel.
  - Hence no GPU linear algebra is available without replacing PyTorch or changing the system. Our method uses no such
    operation.
- **Measured cost of the CPU pseudo-inverse** (including copies): 62.5 ms (d = 512, CLIP-B/16) and 166.4 ms (d = 768,
  CLIP-L/14).
- **Reporting**: GDA's new-task time is given as the interval [measured − CPU pinv, measured]. Any GPU
  pseudo-inverse implementation would fall inside it. Inference latency involves no pseudo-inverse and is unaffected.
  Accuracy was computed on the laptop GPU with the official CUDA path.
