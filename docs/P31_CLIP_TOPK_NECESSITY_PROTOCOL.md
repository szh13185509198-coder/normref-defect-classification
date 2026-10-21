# P3.1 CLIP Top-K 必要性分级 OOF 协议

日期：2026-08-09
状态：Tier A S111 已完成并拒绝 Mean；P3.1 停止，未使用 Formal Test 选择。

## 1. 问题与唯一变量

P3.0 已删除无可观察收益的 Raw Gram，当前可靠主干为 `CLIP Top-K + DINO mean`。
P3.1 只把 `clip_pool` 从 `topk` 改为 `mean`，其余训练、DINO、随机流、metadata、
epoch 和推理均保持不变。

当前代码的两个模式使用同一组 `CLS + patch` token：`topk` 是逐通道取数值最高的
10% token 后求均值，`mean` 是全部 token 的均值。因此本实验只判断当前逐通道
Top-K 是否提供稳定决策收益，不把它误写成统一空间异常区域选择，也不在本阶段引入
patch-only 新实现。

## 2. Tier A：S111 完整八折筛查

先运行完整 S111 八折，覆盖全部 72 张训练样本；八折一次性并发。与 P3.0 S111 Top-K
逐样本配对。CLIP mean 必须同时满足：

1. Defect Recall ≥95%；
2. Good Recall 相比 Top-K 最多下降 12.5 个百分点；
3. 九分类 BA 下降不超过 2 个百分点；
4. Binary BA 下降不超过 6.25 个百分点。

全部满足时，decision 为 `advance_clip_mean_to_s222_s333`。这里使用非劣门槛，是因为
Mean 更简单；Tier A 只负责淘汰明显失败候选，不能单独形成论文多 seed 结论。

## 3. Tier B：S222/S333 三 Seed 确证

只有 Tier A 通过，才依次运行 S222 与 S333；每个 seed 内部八折一次性并发，禁止把
两个 seed 同时启动成 16 个训练。最终三个 seed
分别满足 Tier A 安全门槛，并且三 seed 平均九分类 BA、Binary BA 均不低于 Top-K，
才执行 `drop_topk_and_adopt_all_token_clip_mean`；否则执行
`retain_topk_as_p3_backbone`。无论结果如何都不运行 Formal Test。

## 4. 执行顺序

1. `P3.1 Tier A S111 | CLIP mean + DINO mean`
2. `P3.1 Tier A Audit | S111 screen`
3. 仅当报告允许时：先运行 `P3.1 Tier B S222 | CLIP mean + DINO mean`
4. 再运行 `P3.1 Tier B S333 | CLIP mean + DINO mean`
5. `P3.1 Tier B Audit | final 3-seed decision`

输出固定在 `ckpt/p31_clip_mean_oof/s{seed}`。Runner 可严格复用已完成且哈希、协议、
metadata 全部匹配的 fold；审计只读取训练内 OOF。

## 5. Tier A完成结果（2026-08-09）

S111 Top-K基线九分类BA为80.56%，全token Mean为77.78%；Macro-F1由77.43%降至
74.80%，Good Recall由75%降至62.5%，Binary BA由85.94%降至81.25%。Mean把
Defect Recall从96.88%提高到100%，但净损失2个九分类正确样本：两张原本正确的
`cable_swap`和一张Good被伤害，仅一张Poke得到精确修复，另一张Poke只修复二分类。

由于九分类BA下降2.78pp，超过预注册2pp门槛，decision为
`retain_topk_and_stop_p31_after_s111_screen`。禁止继续运行S222/S333。该结果足以作为
工程筛选保留Top-K，但单个seed不足以形成“Top-K稳定提升”的最终论文结论；最终模型
冻结后如需把Top-K写成核心贡献，仍须补齐多seed配对消融。
