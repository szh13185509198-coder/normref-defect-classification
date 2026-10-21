# P4.2 Combined 空间分离证据协议

## 1. 唯一研究假设

P4.1 证明第二强缺陷响应的绝对强度会把复杂的 `bent_wire` 和 `missing_cable` 误判为 Combined。P4.2 只检验一个新的、预注册的物理假设：

> 单一缺陷引发的多类别响应倾向于集中在同一批 Patch；真正 Combined 的前两种基础缺陷响应更可能来自不同 Patch 区域。

除这一空间分离证据外，不再搜索比例、距离、网络层、阈值目标或其他组合特征。

## 2. 固定空间分数

继续计算七种基础缺陷相对 Good bank 的逐 Patch evidence map，并按 Top-10% 均值确定 top-1 与 top-2 基础缺陷类别。对两张 evidence map 分别取响应最高的 10% Patch 索引：

```text
overlap = |TopPatch(top1) ∩ TopPatch(top2)| / K
spatial_distinctness = 1 - overlap
spatial_composition_score = top2_evidence_score × spatial_distinctness
```

336 分辨率下 `K=ceil(576×0.10)=58`。不使用分割标注，不根据结果改变 10%。

## 3. 评估与安全边界

- 完全复用 P4.1 的外 8 折、每外折内 7 折校准；
- 阈值仍要求内层 Combined-vs-single specificity ≥90%，选择顺序不变；
- 仲裁仍只允许非 Good、非 Combined 的原预测变为 Combined；
- 配对现有 P3.0 S111/S222/S333 OOF；
- 完全复用 P4.1 的九项多 Seed 验收门槛；
- 不训练、不读分割标注、不读 Formal Test，且 `formal_test_allowed=false`。

若 P4.2 失败，停止当前 Cable 上所有基于固定 DINO Patch bank 的 Combined 后处理，不再创建 P4.2b 特征。下一主线转向跨数据集验证、数据局限分析或需要额外数据支撑的训练型集合表示，而不是继续利用同一 72 张样本搜索规则。

## 4. 实际结果（2026-08-09）

P4.2 被拒绝。空间分离分数在三 Seed 产生完全相同的 5 次覆盖：正确修复 1 张 Combined，但伤害 4 张原本正确的单一缺陷（1 张 Bent、1 张 Cut-Inner、2 张 Missing-Cable）。平均九分类 BA `80.56%→76.39%`（-4.17pp），平均 Macro-F1 `76.53%→75.33%`，Combined Recall 只提高 12.5pp，未达到 25pp 门槛。

空间证据相对 P4.1 确实多识别出 1 张 Combined，说明响应区域差异具有局部机制意义；但 specificity 约束下仍无法抵消对单缺陷的误覆盖，且最大非 Combined 类 Recall 降幅超过 12.5pp。重新运行的外层证据 SHA256 完全一致。

预注册停止规则生效：不创建 P4.2b，不再在固定 DINO Patch bank 上搜索 Combined 规则。P4.0、P4.1、P4.2 的共同结论是“局部组合信号存在，但在 8-shot 下不足以形成安全的标量后处理”。
