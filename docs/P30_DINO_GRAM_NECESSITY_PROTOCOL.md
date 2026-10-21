# P3.0 当前 DINO Gram 必要性三 Seed OOF 协议

日期：2026-08-09
状态：三 Seed OOF 已完成并审计通过；未使用 Formal Test 选择。

## 1. 问题与唯一变量

当前 `dino_pool=gram` 拼接 `[dino_avg, gram_max, gram_mean]`，但实测能量约
`99.9993%` 来自 `dino_avg`，后两个分量几乎为零。旧单 seed Formal 消融还受到
后端、初始化和测试反馈混杂，不能证明 Gram 有独立贡献。

P3.0 固定 P2.0 clean terminal-only 主干、CLIP Top-K、336、Hub DINO、损失、epoch、
数据增强、三 seed 与 8 折，只把 `dino_pool` 从 `gram` 改为 `mean`。Baseline 直接
复用 P2.0 S111 与 P2.1 S222/S333 的 Top-K+Gram OOF，不重复训练。

## 2. 预注册门槛

每个 seed 的 Top-K+Mean 必须满足：

1. Defect Recall ≥95%；
2. Good Recall 相比同 seed Gram 最多下降 12.5 个百分点，即最多 1/8 Good；
3. 九分类 BA 相比同 seed Gram 下降不超过 2 个百分点。

三 seed 汇总还必须满足：

- 三个 seed 均满足上述安全/非劣门槛；
- 平均九分类 BA 不低于 Gram；
- 平均 Binary BA 不低于 Gram。

若全部满足，decision 为 `drop_unscaled_gram_and_advance_to_topk_ablation`；否则为
`retain_gram_provisionally_and_audit_scale_before_claiming_contribution`。第二种结果
只表示当前拼接整体不能直接删除，不足以证明低能量 Gram 分量本身提供二阶信息。

## 3. 执行和归档

运行 `P3.0 3X OOF Multi-Seed | Top-K + DINO mean`：三个 seed 同时启动，每个
seed 并发 2 折，总并发最多 6。三个 `p30_run_complete.json` 全部存在后运行
`P3.0 Audit | DINO mean vs Gram, 3 seed OOF`。

输出目录固定为 `ckpt/p30_dino_mean_oof/s{seed}`。runner 默认严格复用完整折，
逐折验证 checkpoint、metadata、预测与 aggregate 哈希。P3.0 不读取 Formal Test
预测，审计完成前禁止新增变体或根据结果改变门槛。

## 4. 与当前总路线的衔接

P3.0 是进入架构提升前的第一项最小必要消融，而不是完整表征网格搜索。审计完成后按
预注册 decision 冻结 DINO pooling：通过则采用 mean；未通过则暂时保留 Gram 并单独
安排尺度审计，但尺度审计不自动证明 Gram 是论文贡献。

随后只继续两项主干归因：P3.1 比较当前逐通道 CLIP Top-K 与同一 `CLS + patch`
token 集合的全均值，P3.2 比较当前
Multi-View 与 single-view。完成这三项后立即进入 P4.0 局部缺陷证据分支和 P4.1
Combined 组合语义建模；其余旧消融降为条件性支线。P3/P4 开发均只使用 OOF，唯一
最终候选冻结前不新增 Formal Test。

## 5. 完成结果与裁定（2026-08-09）

S111、S222、S333 共 24 个 fold 均完成，三组 aggregate 各含 72 条唯一 OOF 预测。
审计文件为 `ckpt/p30_dino_mean_oof/p30_dino_mean_vs_gram_audit.json`，记录
`formal_test_used=false`。Mean 与 Gram 在三个 seed 下的 216 条最终类别预测完全一致：

| 指标（三 seed 平均） | 当前 Gram | DINO Mean | 差值 |
|---|---:|---:|---:|
| 九分类 BA | 80.56% | 80.56% | 0.00pp |
| Binary BA | 88.28% | 88.28% | 0.00pp |

逐项配对的 `changed_predictions` 均为空。余弦分数并非逐位相同，但平均绝对变化仅约
`7.3e-5～1.1e-4`，不足以改变任何类别决策。这与 Gram 后两块能量接近零的诊断一致。
因此执行预注册裁定 `drop_unscaled_gram_and_advance_to_topk_ablation`：正式候选主干
删除当前未缩放 Gram，采用更简单的 384D DINO mean。该结论只否定当前 Raw Gram
实现的必要性，不外推为所有 covariance/Gram 表征均无效。
