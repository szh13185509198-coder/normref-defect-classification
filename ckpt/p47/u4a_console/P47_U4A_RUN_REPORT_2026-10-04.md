# P47 U4-A 本机运行汇报（2026-10-04）

机器：RTX 4050 笔记本 GPU（6 GB）。总耗时约 3 小时 10 分钟。

## 1. 代码与提交（交接时已完成，未重做）
- 研究仓库提交：`461ead4`
- 3 个文件 SHA256 与交付文档第 6 节一致
- 自测：`u4a selftest passed`
- 运行前只读检查全部通过

## 2. 各命令时间线

| 阶段 | 开始 | 结束 | 耗时 | 退出码 |
|---|---|---|---|---|
| u4a_mvtec | 03:18:48 | 03:52:21 | 33.5 min | 0 |
| u4a_external | 03:52:21 | 06:28:53 | 156.5 min (2.6 h) | 0 |
| u4a_analyze | 06:28:53 | 06:29:08 | 15 s | 0 |

status.txt 结束状态：`ALL_DONE 2026-10-04 06:29:08`

## 3. MVTec 视窗比例与参照 BA（13 产品，全部 window=0.880）

| 产品 | 视窗比例 | 视窗下参照 BA (ARR) |
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

## 4. VisA / MTD 视窗比例（12 任务）

| 任务 | 视窗比例 | 累计时间 |
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

## 5. 异常
无。无 P47Error、无 Traceback、无中断。所有 `.lock` 目录保留。

## 6. 输出文件（全部存在，非空）
```
ckpt/p47/u4a_mvtec/u4a_mvtec_preds.jsonl     (384 KB)
ckpt/p47/u4a_mvtec/u4a_mvtec_run.json         (56 KB)
ckpt/p47/u4a_external/u4a_external_preds.jsonl (8.3 MB)
ckpt/p47/u4a_external/u4a_external_run.json   (346 KB)
ckpt/p47/u4a_analysis/u4a_analysis.json       (62 KB)
ckpt/p47/u4a_analysis/u4a_analysis.md         (4.4 KB)
```

控制台日志：`D:\intergration\p47_u4a_run\` 下 status.txt + 3 组 .out.txt/.err.txt + run_u4a.ps1

## 7. 交回与推送提示
- 推送需 `git add -f`（ckpt/ 被 .gitignore 忽略）
- 推送前先 `git fetch` 快进到远端最新
- 推送前扫描主机名、设备用户名、IP、凭据；**正则先在小样本上试一次**，确认能正常执行
