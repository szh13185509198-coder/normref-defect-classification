# P2.3 层次化 Normal-Aware 分支协议

日期：2026-08-08
状态：代码与 OOF 执行配置已实现，尚未运行云端 8 折实验，禁止 Formal Test。

## 1. 为什么进入 P2.3

P2.0/P2.1 的三种子 OOF 已证明，Good 与 Poke 之间的二分类边界是当前系统的主要误杀/漏检来源；Combined 的主要问题则是缺陷子类语义分解，而不是被判断为 Good。P2.2 的固定阈值 `tau=0.006` 在 Formal Test 上通过了预先锁定的验收规则，但只产生了很小的修正，说明单纯平移原型分数阈值的表达能力有限。

P2.3 因此不替换已有九分类分支，而是显式拆分两个问题：

1. 第一层判断 `Good` 还是 `Defect`；
2. 只有第一层判断为 `Defect` 时，才由原九分类原型中的八个缺陷原型决定具体缺陷类别。

这与工业目标一致：先控制良品误杀与缺陷漏检，再讨论缺陷细分类。它也保留了原模型已经较强的缺陷类别排序能力。

## 2. 固定架构与训练规则

- 二分类头：`Linear(1920, 1)`，共 1,921 个参数，输入为 terminal `det_proj` 后的 L2 归一化融合特征。
- 初始化：权重和偏置全部为零；构造过程隔离 CPU RNG，不改变旧模块或后续流程的随机数状态。
- 正值表示 Defect，非正值表示 Good；决策阈值在实验前固定为 `0.0`，不做阈值搜索。
- 缺陷细分类：取八个非 Good 原型中余弦分数最高者；二分类头不直接学习九分类。
- 二分类损失：带权 BCE。若类别数为 `K`，Good 样本权重为 `K/2`，每个缺陷样本权重为 `K/[2(K-1)]`。Cable 的权重分别为 `4.5` 与 `0.5625`，使一个 Good 类和八个缺陷类在总体上权重相等，同时不改变既有采样器及轨迹。
- 优化器：head 学习率固定为 base LR 的 10 倍；terminal `det_proj`、原型和原九分类损失保持 P2.0 配置。
- P2.3 只允许 `clean_v1 + isolated_rng + terminal_only + pure vision`，禁止文本引导、cascade、原型文本修正和 P2.2 固定阈值同时启用。
- checkpoint 仍使用 `lightweight_v1`，每折只保存最后一个 epoch。

## 3. 预注册候选与选择规则

第一阶段只在 Seed 111 的固定 8 折 OOF 上比较两个候选：

- `w010`：九分类损失 + `0.1 × balanced BCE`
- `w030`：九分类损失 + `0.3 × balanced BCE`

两者之外不临时增加权重。候选必须同时满足：

1. OOF Defect Recall 不低于 95%；
2. Good Recall 严格高于 P2.0 S111；
3. Binary Balanced Accuracy 严格高于 P2.0 S111；
4. 九分类 BA 相对 P2.0 S111 下降不超过 2 个百分点。

若两者都通过，依次按 Binary BA、Good Recall、九分类 BA 选择；仍相同时选择损失权重更小者。若都不通过，P2.3 判为失败，停止该分支并禁止为了得到好结果而访问 Formal Test。

## 4. 云端执行顺序

`launch.json` 已加入：

1. `P2.3 2X OOF S111 | hierarchical w=0.1+w=0.3`：在显存允许时并行跑两个候选，各候选内部依次完成 8 折训练、验证和聚合；
2. `P2.3 Audit S111 | select w=0.1 vs w=0.3`：两个候选完成后运行，机械执行上面的预注册验收和选择规则。

单独运行时可选择两个 `P2.3 OOF S111` 配置。若云端中断，在对应 launch 参数末尾临时添加 `--skip_completed`，执行器只保留同时已经形成的 checkpoint 或预测文件，并继续缺失步骤；不得把其他实验目录复制进来冒充完成折。

## 5. 结果交付与下一步

应传回整个 `ckpt/p23_hierarchical_oof_s111`，至少包括：

- 两个候选各 8 折的 `run_manifest.json`、训练/OOF 日志和预测 JSONL；
- 两个 `oof_predictions_all.jsonl` 及 manifest；
- `p23_screening_audit.json`。

只有 S111 有候选通过后，才把被选中的唯一权重原样扩展到 S222、S333 OOF。三种子通过后才能另行锁定是否进入 Formal Test；当前代码主动阻止 P2.3 在非 OOF Formal Test 上运行。

P2.2 是已经完成且独立锁定的固定阈值结果。P2.3 是新的可学习结构实验，失败不会推翻 P2.2，也不能用 P2.2 Formal Test 结果反向调整 P2.3。
