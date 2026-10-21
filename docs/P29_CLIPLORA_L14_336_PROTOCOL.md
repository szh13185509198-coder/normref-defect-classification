# P29 protocol — CLIP-LoRA with ViT-L/14@336px on the MVTec-FS formal episodes (report-only)

Locked 2026-09-30, before any ViT-L/14@336px CLIP-LoRA output exists.

## Why
Our main method uses CLIP ViT-L/14@336px (plus DINOv2-S). CLIP-LoRA was run with ViT-B/16, its official default, at 224 (P21)
and 336 px (P25), and matched against our method on CLIP-B/16 at both resolutions (P24, P18). An external review asked for
CLIP-LoRA on the backbone of our main method, so that the choice of B/16 cannot favour us. P29 runs it.

## What changes relative to P21 / P25 (and nothing else)
- **Backbone:** OpenAI ViT-L/14@336px at its native 336 px. No positional-embedding interpolation.
- **Weights:** the CLIP copy vendored by CLIP-LoRA has no `ViT-L/14@336px` entry, so the official OpenAI file is loaded by
  path. Its SHA256 (`3035c92b…`, part of OpenAI's download URL) is checked before use.
- **LoRA placement:** loralib has no `@336px` table, so the `ViT-L/14` entry is used. It has the same 24-layer layout, and
  `all` = layers 0–23. The text encoder has 12 layers, as in B/16.
- **Train crop:** 336 px, the model's native input (as in P25). The official 224-px crop cannot enter a 336-px model.
- **Test preprocessing:** the transform returned by `clip.load` (= `clip.clip._transform(336)`).
- **Everything else is P21 unchanged:**
  - `lora.run_lora` with the official defaults (r = 2, alpha = 1, dropout 0.25, q/k/v in all layers of both encoders,
    lr 2e-4, 500 × K iterations, batch 32, logit scale 100, no validation or model selection);
  - the same 195 episodes and episode file (`11de94bf…`), the same template and per-episode seeds;
  - official data loaders with 8 workers;
  - the fidelity check against the official `evaluate_lora`;
  - no query labels on the cloud.

Runner: `experiments/p21/cliplora_runner.py --backbone ViT-L/14@336px --resolution 336`. The output directory is bound
to this backbone and resolution by marker files.

## Report (every number, whatever it is)
- BA and Macro-F1 per K for CLIP-LoRA L/14@336, next to:
  - CLIP-LoRA B/16 at 224 and 336 px;
  - zero-shot CLIP L/14@336 with the same template;
  - our method: main configuration, `a_r`, and CLIP-B/16 at 224 / 336 px;
  - SimpleShot.
- Declared contrasts (product-cluster bootstrap, exact sign-flip test, Holm within N1–N4):
  - **N1** ARR(main) − CLIP-LoRA L/14@336. Same CLIP backbone and resolution; our method also uses DINOv2-S in its
    global score.
  - **N2** `a_r` − CLIP-LoRA L/14@336. **Fully backbone-matched.**
    - `a_r` is our dual-branch method without the global score: the z-scored appearance and deviation scores on CLIP
      ViT-L/14@336px layer-18 patches and the normal bank. It uses no DINOv2 and no other backbone.
    - It already exists as a report-only cell of the P17 robustness grid (`ckpt/p17/mvtec/p17_mvtec_results.json`,
      method `a_r`). It is chosen here, before any P29 output, because it is the only configuration of our method on
      exactly this backbone.
  - **N3** CLIP-LoRA L/14@336 − CLIP-LoRA B/16@336: effect of the larger backbone on CLIP-LoRA.
  - **N4** CLIP-LoRA L/14@336 − SimpleShot(main): the Section-4 diagnosis with the strongest trained comparator on the
    big backbone.
- Descriptive, not in the Holm family:
  - CLIP-LoRA L/14@336 − zero-shot CLIP L/14@336;
  - the matched pairs at 224 (P24) and 336 (P18 / P25), repeated for reference.
- Scorer: `experiments/p29/score.py`. MVTec-FS only; the P21 extension rule is not re-opened.

## Execution
- **Parallel shards:** chosen from the smoke run's measured peak reserved GPU memory (heaviest episode, training batch 30),
  so that all shards fit in GPU memory at their peaks (`docs/agent/DISPATCH_P29_2_CLOUD_RUN.md`). The shard count is never
  divisible by 3.
- **Watcher:** a watcher script packs the results when 195/195 episodes exist.
- **Failures:** a failed shard is not relaunched without the author's authorization (`--resume`).
