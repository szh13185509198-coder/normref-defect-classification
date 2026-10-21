# P37 dev — light gate (DINOv2-S) vs P15 CLIP layer-18 gate (MVTec AD held-out normals vs MVTec-FS support pool)

P15 dev reproduced exactly (65 product × seed units).

| Gate | Image AUROC | FRR @1% / 2% / 5% | Defect recall @1% / 2% / 5% |
|---|---|---|---|
| clip_l18 | 0.985 | 0.4% / 1.5% / 4.7% | 73.3% / 82.2% / 89.4% |
| dino_s | 0.975 | 0.8% / 1.9% / 5.1% | 72.8% / 81.7% / 88.7% |

Stop rule (DINOv2-S AUROC more than 0.05 below CLIP): **continue to the test stages**
