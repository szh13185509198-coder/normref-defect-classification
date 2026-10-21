# P4.1 Combined 纯视觉嵌套仲裁协议

## 1. 研究问题

P4.0 证明“第二强基础缺陷 Patch 证据”对 Combined 与单一缺陷具有 AUROC 0.7433 的训练内排序能力。P4.1 检查该证据在严格嵌套校准后，能否安全改善现有三 Seed 全局分类器的 Combined 识别。

P4.1 仍然不训练神经网络，不使用文本、分割标注或 Formal Test。它是一个低成本的结构可行性筛查；通过后才讨论训练型集合表示。

## 2. 外层评估与内层校准

- 外层沿用固定 8 折：每折 63 张训练、9 张留出；
- 对每个外层训练集，再按类内路径顺序做 7 折：每次 54 张建立 Patch bank，9 张产生内层校准证据；
- 每个外折因此得到 63 条完全不使用外层留出图的内层 OOF 证据；
- 阈值只在其中 7 张 Combined 和 49 张单一缺陷上选择，Good 不参与组合阈值；
- 外层 9 张图使用全部 63 张外层训练图建立 bank，再应用该折已锁定阈值。

这避免了直接在 P4.0 的 72 条外层结果上挑选最优阈值，也避免其他折的 bank 间接包含当前外层留出图。

## 3. 固定分数与阈值算法

局部证据沿用 P4.0：对七个基础缺陷分别计算相对 Good bank 的 Top-10% Patch 证据，组合分数固定为七类分数中的第二大值。

阈值候选为内层 OOF 分数的所有唯一值和“高于最大值”的保守候选。只考虑 Combined-vs-single specificity ≥ 90% 的候选；先最大化二分类 Balanced Accuracy，若并列则依次选择 specificity 更高、sensitivity 更高、阈值更高的候选。运行后禁止更改该顺序。

## 4. 保守仲裁

对每个 Seed 的 P3.0 原始 OOF 预测：

```text
if raw_prediction is neither Good nor Combined
   and composition_score >= outer_fold_threshold:
       final_prediction = Combined
else:
       final_prediction = raw_prediction
```

因此：

- 原始 Good 预测永不改变；
- 原始 Combined 预测永不降级；
- Good Recall 和 Defect Recall 理论上必须逐样本完全不变；
- 唯一风险是把某个单一缺陷错误重标为 Combined。

## 5. 三 Seed 预注册验收

同时配对 P3.0 S111/S222/S333，只有全部满足才允许进入后续结构：

1. 每个 Seed 的 Good Recall、Defect Recall 与二分类预测逐样本完全不变；
2. 每个 Seed 的九分类 BA 不低于自身 baseline；
3. 至少两个 Seed 的九分类 BA 严格提高；
4. 三 Seed 平均九分类 BA 提高至少 2.00 个百分点；
5. 候选的 worst-seed 九分类 BA 不低于 baseline worst-seed；
6. 九分类 BA 的跨 Seed sample std 增量不超过 0.50 个百分点；
7. 三 Seed 平均 Macro-F1 不下降；
8. 三 Seed 平均 Combined Recall 至少提高 25 个百分点；
9. 任一 Seed、任一非 Combined 缺陷类 Recall 降幅不超过 12.5 个百分点（最多 1/8）。

失败后不在当前结果上调整 specificity、Top-Patch 比例、阈值目标或覆盖条件。可以把失败作为“局部证据存在但简单阈值不可安全利用”的机制结果。

## 6. 输出与边界

输出固定到 `ckpt/p41_combined_visual_nested`，包含嵌套阈值、逐样本三 Seed 配对预测、输入哈希和总审计。无论通过与否，`formal_test_allowed=false`；P4.1 不能触发 Formal Test。

## 7. 实际结果（2026-08-09）

P4.1 被明确拒绝。三 Seed 的改动完全相同：每组均把 4 张原本分类正确的单一缺陷错误重标为 Combined，没有修正任何 Combined。平均九分类 BA 从 80.56% 降至 75.00%（-5.56pp），平均 Macro-F1 从 76.53% 降至 73.12%，Combined Recall 增益为 0。

被伤害的样本为 3 张 `bent_wire` 和 1 张 `missing_cable`；`bent_wire` Recall 最多下降 37.5pp。8 张真实 Combined 的外层分数全部低于各自内层阈值。Good/Defect 二分类逐样本保持不变，证明安全覆盖代码按设计生效，但细分类仲裁本身失败。

机制解释：P4.0 的 0.7433 AUROC 来自总体排序，而 `bent_wire`、`missing_cable` 的第二强证据本来就与 Combined 高度重叠。更严格的内层高 specificity 校准暴露出该标量在外折上的不稳定性。重新运行的外层证据 SHA256 完全一致，排除随机运行漂移。

停止规则生效：不得在本结果上降低 specificity 或事后调整阈值。简单“第二强分数 + 阈值”不进入训练架构或 Formal Test。若继续 Combined 主线，必须先验证新的空间证据，例如两类响应是否来自不同 Patch 区域，而不是继续调整同一标量。
