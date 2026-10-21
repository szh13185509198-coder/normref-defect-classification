# P40 dev — screening AUROC by CLIP layer (MVTec AD held-out normals vs MVTec-FS support pool)

P15 dev (layer 18) reproduced exactly (65 units).

| Layer | Image AUROC | FRR @1% / 2% / 5% | Defect recall @1% / 2% / 5% |
|---|---|---|---|
| 6 | 0.946 | 0.6% / 1.7% / 4.8% | 55.9% / 68.6% / 78.2% |
| 9 | 0.963 | 0.4% / 1.9% / 5.8% | 67.0% / 76.1% / 82.8% |
| 12 | 0.984 | 0.4% / 2.0% / 4.7% | 77.9% / 85.5% / 90.5% |
| 15 | 0.987 | 0.5% / 1.5% / 4.8% | 78.9% / 86.8% / 93.4% |
| 18 | 0.985 | 0.4% / 1.5% / 4.7% | 73.3% / 82.2% / 89.4% |

Selection (shallowest layer with AUROC ≥ layer 18 − 0.01): **layer 12**
