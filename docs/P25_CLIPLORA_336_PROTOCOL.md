# P25 protocol — CLIP-LoRA (ViT-B/16) at 336 px on the MVTec-FS formal episodes (report-only)

Locked 2026-09-29, before any 336-px CLIP-LoRA output exists.

## Why
P21 ran the official CLIP-LoRA at its default 224 px. Our methods see 336 px. The comparison matched at
224 already exists: P24 ran our method on CLIP-B/16@224. P25 supplies the pair matched at 336, CLIP-LoRA
at 336 against our method on CLIP-B/16@336 (P18). With P24, the matched comparison then exists at both
resolutions.

## The only change
The ViT-B/16 patch-grid positional embeddings are interpolated bicubically from 14 × 14 to 21 × 21 (the
class token is unchanged). This is the standard way to fine-tune a ViT at a higher resolution. Test
preprocessing is the official CLIP transform at 336 (`clip.clip._transform(336)`), and the official
train augmentation uses crop size 336. Everything else is P21 unchanged:
- `lora.run_lora` with the official defaults;
- the same 195 episodes and the same episode file;
- the same template;
- per-episode seeds;
- the fidelity check against the official `evaluate_lora`;
- no query labels on the cloud.

Runner: `experiments/p21/cliplora_runner.py --resolution 336`. Output goes to its own directory, which
is bound to 336 by a marker file.

## Report (every number, whatever it is)
- BA per K for CLIP-LoRA@336, next to CLIP-LoRA@224, our method on CLIP-B/16@336 and @224, the main
  ARR, and SimpleShot.
- Declared contrasts (product cluster bootstrap, exact sign-flip, Holm within M1–M4):
  - M1: ARR(CLIP-B/16@336) − CLIP-LoRA@336, the comparison matched at 336;
  - M2: CLIP-LoRA@336 − CLIP-LoRA@224;
  - M3: ARR(main) − CLIP-LoRA@336;
  - M4: CLIP-LoRA@336 − SimpleShot(main).
- The P24 pair at 224 is repeated for reference.
- Scorer: `experiments/p25/score.py`.
- MVTec-FS only. The P21 extension rule is not re-opened.
