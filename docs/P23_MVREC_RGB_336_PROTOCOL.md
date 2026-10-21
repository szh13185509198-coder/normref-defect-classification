# P23 protocol — MVREC-RGB adaptation at 336 px (report-only)

Locked 2026-09-28, before any 336-px output exists. Code: `experiments/p23/mvrec_rgb_res.py`.

## Why
The P6 MVREC-RGB adaptation kept MVREC's official input of 224 px (`--data.input_shape 224`, AlphaCLIP
ViT-L/14). Official MVREC crops defect-centred 256/384/512-px regions from the full image before
resizing them to 224. Without masks, the adaptation can only resize the whole image to 224, while our
methods see the whole image at 336. P22 showed that 224 px costs our own method 6–7 pp and a global
SimpleShot about 4–5 pp. P23 removes this resolution confound.

## What changes (and nothing else)
- Backbone: the official AlphaCLIP **ViT-L/14@336px, GRIT-20M** checkpoint
  `clip_l14_336_grit_20m_4xe.pth` (AlphaCLIP model zoo; the 224 model used by MVREC is
  `clip_l14_grit20m_fultune_2xe.pth`). The official MVREC `AlphaClip` class already supports
  "ViT-L/14@336px", with the checkpoint given through `MVREC_ALPHA_L14_336_CKPT`.
- Input: the whole image resized to 336 (support and query), with the same all-ones alpha placeholder.
- **Disclosed difference:** the 336 checkpoint was fine-tuned for 4 epochs, the 224 one for 2. This is
  the only 336 GRIT-20M model the authors released.
- Unchanged:
  - the P6 support pipeline: official ClipModel, Zip-Adapter-F, ft_epo 500, AdamW lr 1e-4, the text
    prior, and every P6 gate (trainable whitelist, step count, optimizer ownership);
  - the official query forward pass;
  - the 195 frozen units and the 530 queries.

  The switch is applied at run time to the four module constants that carry the resolution; the P6
  files are not edited.

## Stages
1. **repro** (224 px, the unit bottle K=1 seed 111): the wrapper must reproduce the archived P6
   predictions of that unit exactly, otherwise stop. A scratch pre-check gave 31/31 identical.
2. **formal** (336 px, 195 units): each unit is trained and its queries are predicted. Per-unit
   predictions are checkpointed, and `--resume` is used only with authorization.

## Report (every number, whatever it is)
- BA per K (product macro mean ± seed SD), next to MVREC-RGB 224, ARR and SimpleShot.
- Paired contrasts (product cluster bootstrap):
  - MVREC-RGB 336 − 224;
  - ARR − MVREC-RGB 336;
  - SimpleShot − MVREC-RGB 336;
  - consensus (C0) − MVREC-RGB 336.
- The paper reports the 336 row next to the 224 row in Table 5, together with a note on input resolution.
  Neither row replaces the other.
