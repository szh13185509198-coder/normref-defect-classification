# P2.2 Formal Test（Normal-Aware）修正机械审计

本文只依据 `ckpt/p22_fixed_threshold_tau0006/p22_formal_acceptance.json` 与三组 `normal_aware_predictions.jsonl` 归档事实，不包含新的阈值搜索、因果解释或方法修改。

## A. 文件与哈希完整性

- s111、s222、s333 均包含 79 条不重复的正式预测记录。
- 三组统一使用锁定阈值 `τ=0.006`。
- 三组校准文件 SHA256 均为 `f242afff133310550edde66a795b3006a6a4d48f66fdfa7042a04679ec491929`。
- 正式 checkpoint SHA256：
  - s111：`13c2006888793b77c0910859679a2ab6b1f00899f049ed719afd2cbde7eb5567`
  - s222：`0f79f24e14f743d56f3d6dab42b7002f188f667386945d395921ab7a1c749a31`
  - s333：`1cd7ac4380a16a45216bc4f8290111f62eafdc0e47d52ce3e5b0715acaa7efb1`

## B. Raw 与 Calibrated 指标

| Seed | Overall | 九分类 BA | Macro-F1 | Good Recall | Defect Recall | Binary BA | FP | Changed / Corrected / Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 111 | 87.34% → 89.87% | 90.81% → 91.26% | 83.03% → 84.22% | 84% → 88% | 100% → 100% | 92% → 94% | 8 → 6 | 2 / 2 / 0 |
| 222 | 87.34% → 87.34% | 90.81% → 90.81% | 83.03% → 83.03% | 84% → 84% | 100% → 100% | 92% → 92% | 8 → 8 | 0 / 0 / 0 |
| 333 | 83.54% → 84.81% | 90.15% → 90.37% | 81.66% → 82.08% | 78% → 80% | 100% → 100% | 89% → 90% | 11 → 10 | 1 / 1 / 0 |

三 seed 均值：

- Overall：`86.08% → 87.34%`
- 九分类 BA：`90.59% → 90.81%`
- Macro-F1：`82.57% → 83.11%`
- Good Recall：`82% → 84%`
- Defect Recall：`100% → 100%`
- Binary BA：`91% → 92%`

## C. 三次真实预测变化

1. s111：`good/train/images/019.png`，`poke_insulation → good`，margin=`0.0019763112`
2. s111：`good/train/images/034.png`，`poke_insulation → good`，margin=`0.0027542710`
3. s333：`good/train/images/019.png`，`poke_insulation → good`，margin=`0.0007056594`

三次变化均为正确修正，没有新增缺陷漏检或其他有害修改。

## D. 锁定结论

- P2.2 的预注册验收结果为 `accepted`。
- `τ=0.006` 永久锁定，禁止根据 Formal Test 继续调整。
- 该结果证明了方向一致、无缺陷召回代价的保守改善，但修正次数少，不应描述为统计显著或大幅提升。
- 本文档不承担下一阶段的因果解释与架构选择。
