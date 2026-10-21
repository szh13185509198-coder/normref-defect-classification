# P21 protocol — official CLIP-LoRA on the frozen MVTec-FS formal episodes (report-only)

Locked 2026-09-28, before any CLIP-LoRA output exists.

## Purpose
Section 4's diagnosis says that, in this setting, improving the classifier gains little and the headroom
is in local evidence. The trained comparators so far are our own designs (logistic regression,
trainable support-side adaptation), Tip-Adapter-F* (reimplemented), and MVREC-RGB (official code with
its masks removed). P21 adds a current trained method from its **official code with its official
defaults**: CLIP-LoRA (Zanella & Ben Ayed, CVPRW 2024; https://github.com/MaxZanella/CLIP-LoRA,
AGPL-3.0, unmodified checkout; the commit is recorded at clone time and in every launch record).
Information condition: **C0 + class-name text** (same group as Tip-Adapter-F* and MVREC-RGB).
It is not the main comparison, because our main comparisons are made under the same information
condition.

## Scope
MVTec-FS formal only: 13 products × K ∈ {1, 3, 5} × seeds {111, …, 555} = 195 episodes, 530 queries.
These are exactly the support manifests and queries of all earlier formal results
(`docs/p21/P21_MVTEC_FS_EPISODES.json`, SHA256 `11de94bf…1339`, exported by
`experiments/p21/export_episodes.py`; every image re-hashed). The file carries no query label.

**Extension rule (declared now).** If the BA of the main dual-branch method (ARR) minus that of CLIP-LoRA
is below 2 pp at any K, the same runner is also applied to VisA and Real-IAD before anything is
written. Otherwise CLIP-LoRA is reported on MVTec-FS only, and the paper says so.

## Method (nothing tuned)
- `lora.run_lora` unchanged, with `run_utils.get_arguments()` defaults: ViT-B/16, lr 2e-4,
  n_iters 500 (500 × K steps in total), batch 32, r 2, alpha 1, dropout 0.25, q/k/v, all layers, both
  encoders, AdamW + cosine schedule, fp16 autocast. Validation is off (as in the official code), so no
  model selection happens.
- The train transform, loaders (`datasets.utils.build_data_loader`, 8 workers), test preprocessing
  (`clip.load` preprocess, 224 px) and logit scale 100 are taken from the official `main.py`.
- Class names come from the manifest class order (underscores → spaces, as in the official code), with
  the template `"a photo of a {}."`, which is the same template as Tip-Adapter-F* in this study.
  One template, because the official training uses only `template[0]`.
- `set_random_seed(seed of the episode)`; a fresh `clip.load` for every episode.
- Prediction = argmax of the cosine scores computed exactly as the official `evaluate_lora`. A fidelity
  check compares ours against the official `evaluate_lora`, fed placeholder labels: the share of queries
  predicted as class 0 must match exactly, otherwise the run stops. CLIP zero-shot (same template,
  computed before training) is recorded as a free by-product.

## Execution
- Runs on a rented cloud GPU (`experiments/p21/cliplora_runner.py`). The runner is standalone: it
  imports nothing from this repository and verifies the SHA256 of the episode file and of every image first.
- The episodes are split across n parallel shards (episode j goes to shard j mod n). Each episode is
  written atomically as soon as it finishes. Each shard has a lock, and a shard is relaunched only with
  `--resume` after the user authorizes it. `--resume` computes only the missing episodes and never
  recomputes finished ones.
- A smoke run first measures timing on the heaviest episode. Its predictions are discarded, and there
  are no labels on the cloud to look at.
- Scoring is done locally (`experiments/p21/score.py`), with labels from the frozen query contract.

## Report (report-only: every number below is reported, whatever it is)
- BA / macro-F1 / top-1, reported as the product-macro mean ± seed SD for each K. Rows: CLIP-LoRA and
  CLIP-B/16 zero-shot, next to ARR (main backbone), ARR (CLIP-B/16, P18), SimpleShot (main and
  CLIP-B/16), Tip-Adapter-F*, MVREC-RGB, trainable adaptation, cross-class and consensus.
- Declared contrasts on BA:
  - L1 ARR(main) − CLIP-LoRA;
  - L2 ARR(CLIP-B/16) − CLIP-LoRA (the same backbone);
  - L3 CLIP-LoRA − SimpleShot;
  - L4 CLIP-LoRA − Tip-Adapter-F*;
  - L5 CLIP-LoRA − zero-shot.

  For each contrast: the product cluster bootstrap CI (10,000 resamples), the exact sign-flip p over the
  13 products, and Holm correction within L1–L5. The other contrasts are descriptive.
- Known differences are disclosed, not corrected:
  - CLIP-LoRA sees 224-px CLIP preprocessing (MVTec-FS images are square, so nothing is cropped at test
    time), while our methods see 336-px squash resizes. This is the official setting, and L2 fixes the
    backbone.
  - GPU nondeterminism.
  - CLIP-LoRA uses class-name text, which our methods do not.
