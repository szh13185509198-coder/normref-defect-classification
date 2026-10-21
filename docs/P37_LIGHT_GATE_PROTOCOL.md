# P37 — Light gate: DINOv2-S screening ahead of the CLIP-L/14 classifier

Status: **LOCKED BEFORE ANY P37 OUTPUT** (2026-10-02). Added after all earlier results; a sequentially added,
report-only deployment hypothesis, disclosed as such in supplement S2. No method, constant or earlier result changes.

## 0. Motivation and what does not change

- The two-level inspection of P15 screens with s(x), computed from the CLIP ViT-L/14@336 layer-18 patch anomalies of
  the M = 20 normal bank. On the Jetson Orin Nano the screening decision therefore needs CLIP blocks 1–18 (P36).
- The main configuration already runs DINOv2-S: its patch mean enters the global score. The **light gate** computes
  the screening score from the main configuration's own DINOv2-S patch tokens against a DINOv2-S bank of the same
  normal images. Only flagged parts run CLIP-L/14 and the dual-branch type classification; the DINOv2-S output is
  reused, so nothing is computed twice.
- The split-conformal false-reject bound holds for any score calibrated on normal images. It therefore carries over
  unchanged. What can change is detection power (escapes, review volume); latency improves when most parts are good.
- **Unchanged:**
  - all classifiers, banks, supports, seeds, calibration sets, α levels and review bands;
  - the type classifier of rejected parts (ARR restricted to the defect classes, main configuration).
  - The P15 CLIP-gate results are recomputed first and must be reproduced exactly.

## 1. Definition (no free choices)

- **Features.** `dino_extractor.forward_patches` of the main configuration (`x_norm_patchtokens`, 576 × 384), on the
  locked deterministic 336 view with CLIP normalisation, i.e. the tensor DINOv2-S already receives in the main
  pipeline. FP32, as P15 (FP16 timing is P36; FP16 accuracy of the main configuration is P32). Each token is
  L2-normalised in FP32 and then stored in half precision, exactly as `external_evaluate.extract` stores the CLIP
  tokens that P15 scores. The tokens come from the same forward pass as P15's CLIP features.
- **Bank.** The DINOv2-S tokens of the same M = 20 normal images as the unit's CLIP bank (same draws, same seeds).
- **Screening score.** Patch anomaly a_D(t) = 1 − max_b cos(t, b). The score s_D(x) is the mean of the 6 largest
  a_D(t) (`top1pct`, the statistic P15 selected on dev; it is not re-selected).
- **Thresholds.** Split-conformal on exactly the P15 calibration normals of the unit (100 images, seeds `p15dev|…`,
  `p15cal|…`), with α ∈ {0.01, 0.02, 0.05}.
- **Three-way decision.**
  - auto-reject: s_D > τ_0.01;
  - auto-pass: s_D ≤ τ_lo, with α_lo ∈ {0.10, 0.30};
  - review otherwise (assumed correct, as in P15).

## 2. Dev stage (no test data)

- **Units.** Exactly the P15 dev units:
  - normals: the 13 MVTec AD products' `train/good`, with 100 calibration and up to 100 evaluation normals per product;
  - defects: the MVTec-FS support pool;
  - banks: the P10-A dev draws (seeds 9001–9005).
- **Report.**
  - mean image AUROC of s_D, and of the P15 CLIP score, which must equal the P15 dev values exactly;
  - realized FRR at each α on the evaluation normals.
- **Stop rule (fixed now).** P37 ends after dev if the mean dev AUROC of s_D is more than 0.05 below that of the CLIP
  score. In that case only the dev results are reported and no test stage runs.

## 3. Test stages (each once; lock and staging as P15)

- **Datasets.** VisA, MTD and the Cable held-out split, with exactly the P15 units:
  - tasks, orders, supports and bank draws as P10–P14;
  - the 78 unique Cable images after amendment P15-A1;
  - the same calibration normals.
- **Reproduction gate.** Each stage first recomputes the P15 CLIP gate. It must reproduce P15's outputs exactly (scores,
  thresholds, decisions), or the stage aborts before writing anything.

## 4. Reported (all of it, whatever the outcome)

**Per dataset, paired with the CLIP gate:**
- image AUROC;
- realized FRR at each α;
- escape rate;
- type accuracy among rejected defects;
- per 1,000 parts at 95 % good: false rejects, escapes, wrong-type rejects and review items, for the two-level
  decision at each α and the two three-way bands.

**Per-part expected latency on the Jetson Orin Nano**, for the CLIP gate, the light gate and no gate:
- inputs: P36 FP16 medians of the pass and flagged paths;
- E[T] = T_pass · P(pass) + T_flagged · P(flagged);
- at 95 / 99 / 99.9 % good;
- P(pass) comes from the realized pass rates of normals and defects.

## 5. Recommendation rule (fixed now)

- **Recommend.** If the image AUROC of s_D is at least the CLIP gate's minus 0.02 on each of VisA, MTD and the Cable
  held-out split, the paper recommends the light gate as the deployment default for lines with a high good rate.
- **Otherwise.** The paper presents the light gate as a trade-off (throughput against escapes and review volume) and
  gives every operating point.
- **Either way.** The full table goes into the supplement, and S2 records P37 as the last sequentially added hypothesis.

## 6. Outputs

`ckpt/p37/dev/`, `ckpt/p37/external/`, `ckpt/p37/cable_test/` (write-once, lock and staging, as P15).
