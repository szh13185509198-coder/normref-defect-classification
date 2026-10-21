# P19 — One-shot blind test on Real-IAD (never used before)

Status: **LOCKED BEFORE ANY REAL-IAD IMAGE IS READ BY CODE** (2026-09-28). Up to this commit only the
official annotation jsons and file names / hashes were read (fetch inventory
`data/Real-IAD/realiad_inventory.json`, SHA256 `68086350…cb4c`); no image was decoded or viewed.
All methods, baselines, backbones and constants are those frozen after P18; **nothing is tuned on
Real-IAD**. The run is executed once; every result is reported whatever it shows.

## D. Data

Real-IAD (Wang et al., CVPR 2024; CC BY-NC-SA 4.0, research use), 512-px release
(`realiad_512`), official labels `realiad_jsons/realiad_jsons/<object>.json` (`anomaly_class`,
`image_path`, `mask_path`). Independent of MVTec AD / MVTec-FS / VisA / MTD (other products,
other acquisition). Only the official json entries are used: defective images whose defect is not
visible in a view are not listed as defective there and are never used; test entries labelled `OK`
that come from the `NG/` folder (defective parts seen from a view without the defect) are never
used as normal images.

## T. Task construction (labels and file names only; `experiments/p19/manifest.py`)

- **All 30 objects.** One camera view per object, v* = the view with the most defective json
  entries in `test` (ties → lower index). One view ⇒ one image per physical part, so no part is
  in both supports and queries.
- **Defect classes** = `anomaly_class` values of defective entries in view v* (all 30 objects keep
  3–4 classes; minimum class size 41 ≥ the required 30).
- **Per class split** (images sorted by path, shuffled with `random.Random(f"p19|{object}|{class}")`):
  first 10 → support pool, next ≤ 40 → queries.
- **Supports:** seeds 19001–19005; per seed and class the pool is shuffled with
  `random.Random(f"p19sup|{object}|{class}|{seed}")`; K ∈ {1, 3, 5} = the first K (nested).
- **Normal bank (C1):** M = 20 `train` images with `anomaly_class == OK` from the `OK/` folder in
  view v*, drawn per seed with `random.Random(f"p19bank|{object}|{seed}")`.
- **Normal class (secondary task only):** `test` `OK/`-folder images of view v*, shuffled with
  `random.Random(f"p19|{object}|OK")`: 10 support pool, ≤ 40 queries (disjoint from banks, which
  come from `train`).
- The complete manifest is written to `docs/p19/P19_REALIAD_MANIFEST.json` and committed with this
  protocol.

Two settings: **defect** (primary: N-way defect-type classification, as MVTec-FS) and **normal**
(secondary, descriptive: defect types + OK, as VisA).

## M. Methods (exactly as frozen after P18)

Main features (CLIP ViT-L/14@336 layer 18 patches + fusion global, 336 view): SimpleShot, fixed
fraction, cross-class, A, **ARR** (C1 main), **CONS** (C0 main); adapted baselines `dn4_fused`
(k = 3) and `crop_fused` (largest component, scale 1.0 — the P18 dev selection); mask oracle
(defect setting only; masks = official `mask_path` PNGs, evaluation only, area fraction per 24 × 24
cell). Lightweight single backbone DINOv2-B (layer 11, P18 dev selection): `ss_dinov2_b`,
`a_dinov2_b`, `arr_dinov2_b`, `nr_dinov2_b`, `xc_dinov2_b`, `cons_dinov2_b`.

## H. Hypotheses (defect setting; pooled over K = 1/3/5; 30 objects)

| ID | Contrast |
|---|---|
| B1 | ARR − SimpleShot |
| B2 | CONS − SimpleShot |
| B3 | A − fixed fraction (normal calibration beyond a fixed fraction) |
| B4 | ARR − A (deviation branch) |
| B5 | ARR − crop_fused |
| B6 | CONS − dn4_fused |

Per contrast: mean paired BA difference; 95 % object-cluster percentile bootstrap (10,000
resamples, seed 20260928; all K and seeds of an object stay together); two-sided sign-flip test
on the 30 per-object mean differences (Monte Carlo, 200,000 sign draws, numpy seed 20260928,
p = (1 + #extreme) / (1 + draws)); Holm correction over B1–B6.
**CONFIRMED** iff the bootstrap CI lower bound > 0 **and** the Holm-adjusted p < 0.05; otherwise
NOT CONFIRMED. Reported for every contrast: per-K values, positive objects / 30.

Descriptive (no verdicts): the same contrasts in the normal setting; `arr_dinov2_b − ARR`,
`cons_dinov2_b − CONS`, `arr_dinov2_b − ss_dinov2_b`; oracle-headroom recovered
(ARR − SS) / (oracle − SS) per K; levels of every method; per-object differences; normal setting:
good recall and false rejects / 1000 at 95 % good (argmax).

## R. Run rules

One launch (`python -m experiments.p19.run --device cuda`), lock + staging as P17/P18. If the
run stops before writing results, nothing is read; a relaunch needs the user's explicit
authorisation and is disclosed. No method, constant, task rule or hypothesis may change after this
commit.

## Outputs

`ckpt/p19/blind/`.

## Amendment P19-A1 (2026-09-28, after the first launch; no result existed or was read)

- **What happened.** Launch 1 (commit 4ce49d2, 02:07) printed "done" for 27 of 30 objects (the
  manifest order ends … vcpill, wooden_beads, woodstick, zipper) and then exited silently between
  05:02 and 05:05: no traceback, no error code, no OOM; the executing agent's task handle returned
  "not found", as in the P18 N2 incident (INFERENCE: the process was reaped with the agent's
  background task). Results are written only after all objects, so nothing was written (staging
  empty) and nothing was read. The lock and the empty staging were left untouched.
- **Change (runner mechanics only).** The runner now saves each object's raw rows to
  `staging/objects/<object>.json` right after the object finishes, records every launch in
  `staging/launches.json`, and accepts `--resume` (user-authorised only), which continues inside the
  existing lock and staging and reuses checkpoints written by the same commit and manifest. Tasks,
  methods, constants, hypotheses, statistics and verdict rules are unchanged. Checkpoints are not
  read by anyone before the run completes.
- **Relaunch.** Launch 2 uses `--resume`; since launch 1 left no checkpoint, it recomputes all 30
  objects from scratch with identical deterministic inputs. It is started outside the executing
  agent's background-task tooling (the user's own terminal or a detached process with a log file).
  Both launches are disclosed in the paper's supplement (S2).
