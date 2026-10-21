# P4.0 冻结局部 Patch 证据诊断协议

## 1. 目的

P4.0 不训练新模型，也不直接提出可写入论文的最终模块。它只回答两个先验问题：

1. 冻结 DINOv2 Patch 特征是否包含区别 `good` 与微小表面缺陷（`poke_insulation`、`cut_inner_insulation`、`cut_outer_insulation`）的局部证据；
2. 一个样本中“第二强基础缺陷证据”是否能纯视觉地区分 `combined` 与单一缺陷。

只有诊断通过，P4.1 才允许实现并训练对应分支。这样避免再次进行代价较高、但输入证据本身不足的 3 Seed × 8 Fold 训练。

## 2. 数据边界

- 唯一图像来源：`dataset/metadata/MVTecAD-Cable/8-shot.jsonl` 的 72 张训练图；
- 验证划分：已经锁定的 `oof_8fold_v1/fold_XX_train.jsonl` 与 `fold_XX_val.jsonl`；
- 每折只使用该折 63 张训练图建立 Patch bank，对 9 张留出图计算证据；
- 8 折留出预测合并后，每张训练图恰好被诊断一次；
- 禁止读取分割标注，元数据中的分割标注字段被显式忽略；
- 禁止读取 `test-shot.jsonl`、Formal Test 图像或其预测；
- P3.0 S111 的 72 条 OOF 预测仅用于比较全局边界与局部证据，不参与特征 bank。

因此 P4.0 是训练集内部 OOF 诊断，不是 Formal Test，也不产生新的论文测试分数。

## 3. 固定视觉特征

- Backbone：冻结的本地 `dinov2_vits14`；
- 权重：`pretrained/dinov2_vits14_pretrain.pth`；
- Patch policy：`full`，336 分辨率下保留 576 个空间 token；
- 输入：固定 Resize(336×336) 与当前项目相同的 CLIP Normalize；
- 不反向传播、不构造 optimizer、不保存 checkpoint；
- 72 张图的 Patch 特征只计算一次并缓存，缓存必须记录元数据、图像、DINO 权重与本地 Hub 代码哈希。

## 4. 预注册局部证据

对查询 Patch `p` 和类别 Patch bank `B_c`：

```text
sim_c(p) = max cosine(p, b), b in B_c
e_c(p)   = sim_c(p) - sim_good(p)
S_c(x)   = 查询图像中 e_c(p) 最高 10% Patch 的均值
```

10% 比例在运行前固定，不允许根据结果改变。

- 局部二分类分数：所有缺陷类别 `S_c(x)` 的最大值；
- 微小缺陷诊断：只在 `good` 与 `poke/cut_inner/cut_outer` 样本上计算 AUROC；
- Combined 组合证据：排除 `good` 和 `combined` bank 后，取七种基础缺陷 `S_c(x)` 的第二大值；
- 非冗余性：局部二分类分数与 P3.0 全局 `defect_minus_good_margin` 的 Pearson 相关系数。

P4.0 不根据标签选择分类阈值，也不把局部分数与全局分数事后融合。

## 5. 预注册通过条件

两条候选分支独立裁决。

### P4.1-Micro 局部微缺陷分支

同时满足：

- 全部 Good/Defect 的局部证据 AUROC ≥ 0.75；
- Good 与三类微小缺陷子集 AUROC ≥ 0.70；
- 局部分数与现有全局 margin 的绝对 Pearson 相关系数 ≤ 0.95。

满足后只代表“存在可利用且非完全重复的局部信号”，下一步仍须预注册融合方式并重新 OOF 验证。

### P4.1-Combined 纯视觉组合分支

- Combined 对其他单一缺陷的第二强基础缺陷证据 AUROC ≥ 0.65。

满足后才允许开发纯视觉的多缺陷证据聚合；它不是文本语义建模。

## 6. 输出与停止规则

输出目录固定为 `ckpt/p40_local_patch_diagnostic`，至少包含：

- 冻结 Patch 特征缓存及 manifest；
- 72 条 OOF 局部证据 JSONL；
- 带输入哈希、指标、门槛和逐项决定的审计 JSON。

若两条分支都失败，则停止在 Cable 上继续堆叠局部仲裁结构，转向扩展数据集验证和论文实验整理。P4.0 的任何结果都不得用于回调 P2.2 的固定阈值。

## 7. 实际结果（2026-08-09）

本地 RTX 4050 在不训练的条件下完成了全部 72 张图的特征缓存和 8 折诊断。第二次运行复用了经过 provenance 校验的缓存，72 条证据文件 SHA256 前后完全一致。

| 诊断项 | 结果 | 预注册门槛 | 裁定 |
|---|---:|---:|---|
| 全部 Good/Defect 局部 AUROC | 0.8809 | ≥0.75 | 通过 |
| Good 对 Poke/Cut 局部 AUROC | 0.8333 | ≥0.70 | 通过 |
| 局部分数与 P3.0 全局 margin 相关性 | 0.6720 | `abs(r)≤0.95` | 通过 |
| Combined 对单缺陷的第二证据 AUROC | 0.7433 | ≥0.65 | 通过 |

参照值显示，P3.0 全局 margin 的全部 Good/Defect AUROC 为 0.9844，Good 对 Poke/Cut AUROC 为 0.9635，均高于单独使用局部证据。因此 P4.0 **没有证明局部分支应替换全局分类器**；它证明的是局部证据具有中等区分能力，并且与全局分数并非完全重复。

Combined 的结果也只是排序信号而非现成分类器。`combined` 的第二强基础证据均值为 0.0670，但 `bent_wire` 和 `missing_cable` 分别达到 0.0922 和 0.0746，存在明显重叠。P4.1 必须使用训练内嵌套校准和保守仲裁，不能直接在这 72 条结果上挑一个最优阈值。

最终裁定：两条证据支线均允许进入 P4.1；执行顺序优先 Combined 纯视觉组合仲裁，因为它对应当前九分类 BA 的主要结构性上限。Micro 分支保留为第二优先级，且只能作为全局判断的受约束补充。
