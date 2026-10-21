# P6.B MVTec-FS-RGB Feasibility Ruling

> Ruling basis: formal probe on clean commit `ac76ac8`; `formal_test_used=false`.

## Decision
**INCONCLUSIVE-overall / 13-products-feasible**

The overall `INCONCLUSIVE` status is driven solely by the local asset-integrity
failure of `toothbrush`; it does **not** negate the local feasibility verdicts
of the 13 compliant products listed below. This ruling contains no model
performance conclusions.

## Feasible Products (13)
- bottle
- capsule
- carpet
- grid
- hazelnut
- leather
- metal_nut
- pill
- screw
- tile
- transistor
- wood
- zipper

## Failed Product: Toothbrush
- **Reason**: Asset missing. The official `CONFIG/` references 30 images, but they do not exist in the full 12-chunk merged archive nor in the unpacked directory (all 12 `image.tar.*` chunks successfully merged and sized correctly).
- The exclusion basis is **asset integrity**, not model performance; this is an upstream data-missing issue and does not constitute result-driven tuning.
- `cable` is excluded pre-emptively due to overlap with the Cable research assets (not a probe failure).

## Dataset Provenance Note
- `dataset_is_dirty`: True
- **Reason for dirty state**: only the merged `image.tar` under the dataset root (`data/mvtec-fs`) is untracked by Git; the `image.tar.*` chunks are not part of the dirty state. This does not affect the integrity of the read-only CSVs and images.

## Output Pointer
- V2 outputs (retained and untampered): `data/p6b_probe_output_v2/`
- JSON proof: `data/p6b_probe_output_v2/p6b_mvtec_fs_rgb_feasibility.json`
