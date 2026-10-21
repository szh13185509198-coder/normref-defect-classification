# P40 — Earlier exit of good parts: screening at a shallower CLIP layer (edge deployment, report only)

Status: **LOCKED BEFORE ANY P40 OUTPUT** (2026-10-02). Sequentially added deployment hypothesis, disclosed in S2.
No method, constant or earlier result changes.

## 0. Motivation
- The two-level inspection (P15) screens with s(x), the mean of the 6 largest layer-18 patch anomalies against the
  M = 20 normal bank. On the Jetson, good parts can stop after CLIP block 18 (P36 / P38-A1: TensorRT pass path 59.6 ms).
- A shallower layer k would let them stop earlier. The split-conformal false-reject bound holds for any score
  calibrated on normal images, so it carries over. What can change is detection power (escapes, review volume).

## 1. Score at layer k (no other free choice)
- **Score:** P15's statistic on CLIP ViT-L/14@336 layer-k patch tokens (CLS dropped, L2-normalised, stored in half as
  P15). Patch anomaly against the same M = 20 bank images at layer k; `top1pct` = mean of the 6 largest.
- **Thresholds:** the P15 calibration normals of each unit, α ∈ {0.01, 0.02, 0.05}, review bands α_lo ∈ {0.10, 0.30}.
- **Type classification of flagged parts:** unchanged (main configuration, ARR on the defect classes).
- **Candidate layers:** k ∈ {6, 9, 12, 15}, with layer 18 as the reference.

## 2. Dev stage (P15 dev units; no test data)
- **Units:** the 13 MVTec AD products; 100 calibration and up to 100 evaluation held-out normals; the MVTec-FS support
  pool as defects; the P10-A dev bank draws (seeds 9001–9005).
- **Layer-18 reproduction:** the layer-18 score is recomputed and must reproduce P15 dev exactly (per product × seed),
  from the P15 code path.
- **Selection rule (fixed now):** the shallowest k whose mean dev AUROC is at least the layer-18 mean dev AUROC − 0.01.
  If no candidate qualifies, P40 stops after dev and only the dev table is reported.

## 3. Test stages (once each; lock and staging as P15)
- **Data:** VisA, MTD and the Cable held-out split with exactly the P15 units, for the selected layer k* only.
- **Reproduction gate:** each stage first recomputes P15 (layer 18) and must reproduce its saved unit results exactly.
- **Reported, paired with layer 18:**
  - image AUROC;
  - realized FRR at each α;
  - per 1,000 parts at 95 % good: false rejects, escapes, wrong-type rejects and review items;
  - the realized pass rates of normals and defects.

## 4. Latency on the Jetson
- **Engine:** a TensorRT FP16 engine of CLIP-L/14@336 truncated after block k*, exported through `ClipSplit` (P36); its
  output must agree with PyTorch like the other engines.
- **Measurement:** the pass path (preprocessing, engine, layer-k* bank nearest neighbour, score) is timed with the P38
  harness, two passes.
- **Expected per-part latency:** at 95 / 99 / 99.9 % good, from the measured pass path, the measured flagged path
  (pass path + main-configuration TensorRT path) and the realized pass rates.

## 5. Recommendation rule (fixed now)
- **Recommend.** If the AUROC at k* is at least the layer-18 AUROC − 0.02 on each of VisA, MTD and Cable, the paper
  recommends exiting at k* as the default for the two-level inspection.
- **Otherwise.** It is reported as a trade-off with every operating point.
- **Either way.** Everything goes into the supplement; S2 records P40 as a sequentially added hypothesis.

## 6. Outputs
`ckpt/p40/dev/`, `ckpt/p40/external/`, `ckpt/p40/cable_test/`, `ckpt/p40/report/` (write-once, lock and staging).
