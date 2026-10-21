# P47 U5 - conformal type sets (jackknife; worst-case coverage 1 - 2α) and the unknown-type flag

Integrity: ARR with all supports reproduces P14 (MVTec-FS max |dBA| = 0.0e+00).

| Data | K | α | Coverage | Class cov. mean / min | Mean size | Singleton rate / acc. | Empty | Argmax acc. |
|---|---|---|---|---|---|---|---|---|
| mvtec | 3 | 0.1 | 93.5 % | 93.8 % / 81.3 % | 1.85 | 49.8 % / 93.1 % | 2.0 % | 84.0 % |
| mvtec | 3 | 0.2 | 85.7 % | 86.4 % / 64.5 % | 1.24 | 70.9 % / 90.4 % | 6.5 % | 84.0 % |
| mvtec | 5 | 0.1 | 92.3 % | 92.7 % / 78.4 % | 1.38 | 64.9 % / 95.3 % | 3.5 % | 87.1 % |
| mvtec | 5 | 0.2 | 81.4 % | 82.1 % / 56.8 % | 0.99 | 76.6 % / 92.8 % | 12.7 % | 87.1 % |
| visa | 3 | 0.1 | 97.3 % | 96.9 % / 89.3 % | 2.30 | 21.6 % / 93.9 % | 0.5 % | 84.7 % |
| visa | 3 | 0.2 | 91.9 % | 91.6 % / 75.1 % | 1.49 | 56.0 % / 93.1 % | 2.7 % | 84.7 % |
| visa | 5 | 0.1 | 96.5 % | 96.7 % / 87.9 % | 1.75 | 40.9 % / 95.8 % | 0.4 % | 87.6 % |
| visa | 5 | 0.2 | 89.5 % | 89.3 % / 68.3 % | 1.16 | 78.1 % / 92.3 % | 3.7 % | 87.6 % |

Unknown type (MVTec-FS, C ≥ 3, one class held out at a time):

| K | α | Detection | False alarm (known) | AUROC | Episodes |
|---|---|---|---|---|---|
| 3 | 0.1 | 12.6 % | 3.0 % | 0.713 | 305 |
| 3 | 0.2 | 28.8 % | 9.2 % | 0.713 | 305 |
| 5 | 0.1 | 16.5 % | 3.1 % | 0.717 | 305 |
| 5 | 0.2 | 37.8 % | 12.1 % | 0.717 | 305 |
