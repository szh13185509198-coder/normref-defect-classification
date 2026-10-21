<!-- translation-header -->
> English translation of [`P47_U3U4_RUN_REPORT_2026-10-04.md`](P47_U3U4_RUN_REPORT_2026-10-04.md), which is kept unchanged (SHA-256 `84e8a4035d9003253ba84136b89b602530ef9c1338f21742c250105a2e69a764`). The translation keeps the structure and numbers of the original.

# P47 U3/U4 local run report (2026-10-03 to 2026-10-04)

Machine: RTX 4050 laptop GPU (6 GB), not the RTX 4090 assumed in the hand-over document; the total run time was about
2–3 times the estimate of that document.

## 1. Code and commit (done at hand-over, not repeated)
- Research repository commit: `54844f2` (only the 4 code files + the hand-over document)
- The SHA-256 of the 4 files match section 7 of the hand-over document
- Both self-tests passed: `u3 selftest passed`, `u4 selftest passed`

## 2. Read-only checks of §3.2 (done at hand-over)
- All inputs present; none of the 6 output directories or their .lock/.staging exist
- 39 GB free on drive D; GPU idle
- All passed

## 3. Timeline and outcome of each command

| Stage | Start | End | Duration | Exit code |
|---|---|---|---|---|
| u3_extract | 10-03 15:25:52 | 10-03 15:33:55 | 8 min | 0 |
| u3_run | 10-03 15:33:55 | 10-03 15:44:55 | 11 min | 0 |
| u4_mvtec | 10-03 15:44:55 | 10-03 17:42:48 | 118 min | 0 |
| u4_external | 10-03 17:42:48 | 10-04 01:54:11 | 491 min (8.2 h) | 0 |
| u3_analyze | 10-04 ~01:56 | 10-04 ~01:57 | <1 min | 0 |
| u4_analyze | 10-04 ~01:57 | 10-04 ~02:00 | ~3 min | 0 |

Final state in status.txt: `ALL_DONE 2026-10-04 01:54:11`

## 4. Integrity values

- **u3_run**: all 13 products print `diagonal reproduces P14`; finally `diagonal max |dBA| vs P14 = 0.0e+00`
- **u4_mvtec**: `none vs cached agreement` = 1.0000 for all 13 products
- **u4_external**: `none vs reference agreement` = 1.0000 for all 12 tasks (11 VisA + MTD)

No value below 0.99.

## 5. Half-width of the type B implementation component
- a = 0.5280885780885779 pp (exact value): the largest absolute 95% CI bound of the pooled ARR ΔBA over the three
  device recomputations P32/P38/P41
- Sources:
  - `ckpt/p32/jetson_orin_nano+jetson_orin_nano_b/p32_scores.json` (P32 FP16 [-0.043, +0.191], FP32 [0.000, +0.026])
  - `ckpt/p32/jetson_orin_nano_trt/p32_scores.json` (P38 TensorRT [-0.138, +0.528])
  - `ckpt/p41/jetson_orin_nano/p41_scores.json` (P41 [-0.205, +0.469])

## 6. Anomalies
None: no P47Error, no traceback, no out-of-memory, no interruption. All .lock directories kept.

## 7. Output files (all present and non-empty)
```
ckpt/p47/u3_features/u3_features_info.json      (1058 B)
ckpt/p47/u3/u3_cells.jsonl                       (2.2 MB)
ckpt/p47/u3/u3_run.json                          (54 KB)
ckpt/p47/u3_analysis/u3_budget.json              (96 KB)
ckpt/p47/u3_analysis/u3_budget.md                (4.6 KB)
ckpt/p47/u4_mvtec/u4_mvtec_preds.jsonl           (1.1 MB)
ckpt/p47/u4_mvtec/u4_mvtec_run.json              (57 KB)
ckpt/p47/u4_external/u4_external_preds.jsonl     (27 MB)
ckpt/p47/u4_external/u4_external_run.json        (350 KB)
ckpt/p47/u4_analysis/u4_analysis.json            (168 KB)
ckpt/p47/u4_analysis/u4_analysis.md              (8.8 KB)
```

The console logs (status.txt and 4 sets of .out.txt/.err.txt) were written to a local run directory outside the
repository; the .out.txt files are kept in this folder.

## 8. Hand-back notes
- `ckpt/p47/u3_features/` (about 3.8 GB) is not pushed; only `u3_features_info.json` is.
- Pushing needs `git add -f` (ckpt/ is ignored by .gitignore).
- Before pushing: fast-forward with git fetch, and scan for host names, user names, IP addresses and credentials.
