# P3.2 Multi-View原型初始化分级OOF协议

日期：2026-08-09
状态：Tier A S111已完成且拒绝5-view；P3.2停止，未使用Formal Test选择。

## 1. 真实基线与唯一变量

P3.0/P3.1的clean主干实际使用`num_init_views=1`，所以Multi-View不是当前已验证组件。
P3.2固定CLIP Top-K、DINO Mean、terminal-only、20 epoch、repeat=4、三个独立seed和
不可变8折，只把原型类别均值初始化的视角数从1改为5。

每个view使用`prototype_init/view_idx`派生的独立顺序和逐样本增强seed；正式训练仍
使用`train/epoch/repeat`命名空间，因此增加初始化view不会改变训练阶段的数据顺序或
增强轨迹。它改变的是每类原型初值由`7×1`还是`7×5`个增强特征求均值。

## 2. Tier A S111筛查

S111八折一次性并行，与P3.0 S111 one-view逐样本配对。必须满足：

1. Defect Recall ≥95%；
2. Good Recall最多下降12.5个百分点；
3. 九分类BA下降不超过2个百分点；
4. 九分类BA或Binary BA至少一项严格提高。

全部满足才输出`advance_multiview5_to_s222_s333`。Multi-View增加初始化计算，因此仅
“不变”不足以扩展。

## 3. Tier B三seed裁定

通过Tier A后依次运行S222、S333，每个seed内部八折并行，禁止16并发。最终接受必须：

- 三个seed全部满足安全门槛；
- 三seed平均九分类BA不低于one-view；
- 三seed平均Binary BA不低于one-view；
- 九分类BA与Binary BA的最坏seed均不低于one-view；
- 两项指标跨seed的sample std均不高于one-view；
- 至少两个seed的九分类BA或Binary BA严格提高。

通过时采用5-view；否则保留1-view。无论结果如何都不运行Formal Test。

## 4. 运行顺序

1. `P3.2 Tier A S111 | Multi-View5 prototype init`
2. `P3.2 Tier A Audit | S111 screen`
3. 仅在允许时依次运行S222、S333 Tier B配置
4. `P3.2 Tier B Audit | final 3-seed decision`

## 5. Tier A完成结果（2026-08-09）

5-view与1-view在S111全部72条OOF上的最终类别预测完全一致，九分类BA均为80.56%，
Binary BA均为85.94%，所有逐类Recall和错误样本集合相同。5-view不是未生效：余弦
分数平均绝对变化约`0.00146`、最大约`0.00727`；8折最终训练loss均略低，最终
`det_proj`与prototypes也有非零数值差异。但这些变化没有使任何样本跨越决策边界。

由于预注册规则要求九分类BA或Binary BA至少一项严格提高，decision为
`retain_one_view_and_stop_p32_after_s111_screen`。不运行S222/S333，后续主干保持
`num_init_views=1`。该结论限定于当前增强强度、20 epoch和可训练原型协议；它说明
额外初始化视角被后续优化基本吸收，不外推为所有Multi-View训练策略均无效。
