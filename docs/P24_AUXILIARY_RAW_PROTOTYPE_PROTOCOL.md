# P2.4 二分类辅助训练 + 原始视觉原型推理协议

日期：2026-08-09  
状态：三 seed OOF 已完成并按预注册门槛拒绝；禁止 P2.4 Formal Test。

## 1. 研究依据

P2.3 的二分类 head 硬覆盖分支已经拒绝。P2.3 的原始视觉原型分支达到九分类
BA `81.94%`、Defect Recall `98.44%`；但 hard override 在 `w=0.1/0.3`
下只修复 1 张 Good，却分别伤害 8/6 张原本判断正确的缺陷，使 Defect Recall
降至 `87.50%/90.625%`。失败来自推理时的无条件硬覆盖，而不是训练崩溃。

P2.4 保留 `w=0.1` balanced BCE 作为训练期辅助目标，只用它塑造 terminal
特征；推理阶段不计算、不读取二分类 head 的输出，完全沿用学习得到的视觉类别原型
与 FAISS 余弦检索。该协议不是纯文本原型推理，也不据此宣称严格 zero-shot。

## 2. 固定训练与推理配置

- `clean_v1 + isolated_rng + terminal_only`；
- CLIP Top-K、DINO Gram、336 分辨率、Hub DINO；
- `normal_aware_head=true`，balanced BCE 权重固定为 `0.1`；
- head LR multiplier 固定为 `10.0`，训练记录中的阈值占位值固定为 `0.0`；
- 推理模式固定为 `normal_aware_auxiliary_raw_prototype`；
- 每条 OOF 记录必须满足 `predicted_class_id == raw_predicted_class_id`；
- sidecar 必须声明 head 参与训练但未参与推理。

## 3. OOF 与验收门槛

固定运行 seed `111/222/333`，每个 seed 使用不可变的 8 折 leave-one-per-class-out
划分。每个 seed 分别与对应的 P2.0/P2.1 baseline 比较，并必须同时满足：

1. Defect Recall ≥ 95%；
2. Good Recall 不低于对应 baseline；
3. Binary Balanced Accuracy 严格高于对应 baseline；
4. 九分类 Balanced Accuracy 下降不超过 2 个百分点。

三 seed 汇总还必须满足：平均 Binary BA 严格提高、平均九分类 BA 不下降，并且
三个 seed 均通过各自门槛。最终 decision 只能是：

- `advance_p24_auxiliary_to_full_8shot_training`；或
- `reject_p24_auxiliary_and_keep_p22_as_primary`。

## 4. 并发与恢复

`P2.4 3X OOF Multi-Seed | w=0.1` 同时启动三个 seed runner；每个 runner 同时
运行 2 折，因此全局最多 6 个训练/验证进程。每折写入独立目录，不共享 checkpoint。
只有该 seed 的 8 折全部完成并通过逐折哈希检查后才允许聚合。

launch 默认携带 `--skip_completed`。只有 checkpoint、metadata、协议、sidecar 与
逐样本 raw 约束全部匹配时才复用已有折；任何一项不一致都会重跑该折。

## 5. Formal Test 锁定

P2.4 OOF runner 和 evaluator 不读取 Formal Test 预测。聚合器只读取 Formal Test
metadata 路径用于交集为零的泄漏审计。只有 OOF decision 通过并经主研究 Agent
复核后，才能另行预注册完整 8-shot 训练；当前不得运行 P2.4 Formal Test。

## 6. 最终结果与裁定（2026-08-09）

| Seed | Binary BA 变化 | 九分类 BA 变化 | 最终预测变化 | Seed 门槛 |
|---:|---:|---:|---|---|
| 111 | +0.78 pp | +1.39 pp | 修正 1 张 Poke | 通过 |
| 222 | 0 | 0 | 无 | 未通过严格提升 |
| 333 | 0 | 0 | 无 | 未通过严格提升 |

三 seed 平均 Binary BA 从 `88.281%` 升至 `88.542%`，平均九分类 BA 从
`80.556%` 升至 `81.019%`；但只有 S111 的
`poke_insulation/train/images/003.png` 跨过边界，其 margin 从 `-0.00065` 变为
`+0.00553`。S222/S333 的余弦分数发生非零变化，但 144 条预测全部不变。

最终 decision 为 `reject_p24_auxiliary_and_keep_p22_as_primary`。P2.4 证明辅助 BCE
是安全但弱、且跨 seed 不稳定的表征正则；P2.3 的主要伤害来自推理期 hard override，
而不是辅助训练本身。P2.4 不进入完整 8-shot 或 Formal Test，P2.2 保持主分支。

本轮属于诊断性负分支：24 个 manifest 均为同一 commit，但恢复过程中存在两个稳定
diff fingerprint；S111 fold00/01 的本地训练日志还需以云端原文件覆盖。该问题不改变
拒绝结论，但这些运行不能作为最终正向主表证据。
