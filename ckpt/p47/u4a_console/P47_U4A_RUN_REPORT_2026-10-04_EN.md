<!-- translation-header -->
> English translation of [`P47_U4A_RUN_REPORT_2026-10-04.md`](P47_U4A_RUN_REPORT_2026-10-04.md), which is kept unchanged (SHA-256 `1834e9569e7d09b9eef4ffc714d98e3b82fb613bb7efb8eb93c4eee9c4b55aee`). The translation keeps the structure and numbers of the original.

# P47 U4-A local run report (2026-10-04)

Machine: RTX 4050 laptop GPU (6 GB). Total run time about 3 h 10 min.

## 1. Code and commit (done at hand-over, not repeated)
- Research repository commit: `461ead4`
- The SHA-256 of the 3 files match section 6 of the hand-over document
- Self-test: `u4a selftest passed`
- All read-only checks before the run passed

## 2. Timeline of the commands

| Stage | Start | End | Duration | Exit code |
|---|---|---|---|---|
| u4a_mvtec | 03:18:48 | 03:52:21 | 33.5 min | 0 |
| u4a_external | 03:52:21 | 06:28:53 | 156.5 min (2.6 h) | 0 |
| u4a_analyze | 06:28:53 | 06:29:08 | 15 s | 0 |

Final state in status.txt: `ALL_DONE 2026-10-04 06:29:08`

## 3. MVTec window ratio and reference BA (13 products, all window = 0.880)

| Product | Window ratio | Reference BA in the window view (ARR) |
|---|---|---|
| bottle | 0.880 | 88.91 % |
| capsule | 0.880 | 75.89 % |
| carpet | 0.880 | 87.00 % |
| grid | 0.880 | 88.80 % |
| hazelnut | 0.880 | 89.44 % |
| leather | 0.880 | 91.11 % |
| metal_nut | 0.880 | 96.02 % |
| pill | 0.880 | 83.48 % |
| screw | 0.880 | 74.24 % |
| tile | 0.880 | 99.50 % |
| transistor | 0.880 | 91.00 % |
| wood | 0.880 | 95.86 % |
| zipper | 0.880 | 84.35 % |

## 4. VisA / MTD window ratio (12 tasks)

| Task | Window ratio | Cumulative time |
|---|---|---|
| visa__candle | 0.880 | 12.9 min |
| visa__cashew | 0.880 | 25.5 min |
| visa__chewinggum | 0.880 | 37.5 min |
| visa__fryum | 0.880 | 49.9 min |
| visa__macaroni1 | 0.880 | 62.6 min |
| visa__macaroni2 | 0.880 | 75.2 min |
| visa__pcb1 | 0.880 | 88.1 min |
| visa__pcb2 | 0.880 | 101.0 min |
| visa__pcb3 | **0.876** | 114.0 min |
| visa__pcb4 | 0.880 | 125.6 min |
| visa__pipe_fryum | 0.880 | 138.5 min |
| mtd__mtd6way | **0.777** | 156.4 min |

## 5. Anomalies
None: no P47Error, no traceback, no interruption. All `.lock` directories kept.

## 6. Output files (all present and non-empty)
```
ckpt/p47/u4a_mvtec/u4a_mvtec_preds.jsonl     (384 KB)
ckpt/p47/u4a_mvtec/u4a_mvtec_run.json         (56 KB)
ckpt/p47/u4a_external/u4a_external_preds.jsonl (8.3 MB)
ckpt/p47/u4a_external/u4a_external_run.json   (346 KB)
ckpt/p47/u4a_analysis/u4a_analysis.json       (62 KB)
ckpt/p47/u4a_analysis/u4a_analysis.md         (4.4 KB)
```

The console logs (status.txt, 3 sets of .out.txt/.err.txt and the launch script) were written to a local run
directory outside the repository; the .out.txt files are kept in this folder.

## 7. Hand-back and push notes
- Pushing needs `git add -f` (ckpt/ is ignored by .gitignore).
- Before pushing, fast-forward to the latest remote with `git fetch`.
- Before pushing, scan for host names, device user names, IP addresses and credentials; **test every regular
  expression on a small sample first** to make sure it runs correctly.
