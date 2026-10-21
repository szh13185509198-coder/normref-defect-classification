# P6.B MVTec-FS 13-Product RGB-Only Method Transfer Execution Protocol

> Status: **PREREGISTERED — LOCKED FOR EXECUTION**（由主研究 Agent 裁定自 DRAFT
> 升级；blocking 全部关闭，Stage 1 生产调用链与测试已独立复验验收）。
> 本协议已预注册并锁定：任何修改必须新开协议版本并重新裁定。
> 执行必须严格遵循 §2/§3/§5/§6/§7/§8；Stage 1 公共基础设施已实现并验收
> （execution tests 15/15、feasibility 23/23、P3.F 15/15、py_compile、
> git diff --check 全过）；Stage 2 模型实现仅在正式 manifest bundle
> （completed / formal_eligible=true / dirty=false）通过审计后进行。
> 本文件不是研究事实来源；所有已裁定事实以主研究 Agent 裁定与
> `docs/P6B_MVTEC_FS_RGB_FEASIBILITY_RULING.md` 为准。

> **Addendum A（PREREGISTERED — LOCKED, pre-execution）**：
> `docs/P6B_EXECUTION_ADDENDUM_A.md` 锁定 baseline 实现数学语义，仅消除
> 实现歧义，不改变产品、K、seed、预算或指标；Stage 2+ 评估 artifact 必须
> 同时记录主协议与 Addendum A 的 SHA256，任一不匹配即 fail-closed。

## 1. Task Definition (Locked)

- P6.B 是 **RGB-only few-shot fine-grained multiclass method transfer**。
- **不是 anomaly detection**：MVTec-FS 是 defect-only 任务，没有 Cable 的
  Good/Defect 决策语义；本文档任何章节不得写成 anomaly detection、
  nominal-vs-anomalous 或 Good 召回语义。
- 每个 product 是一个**独立的 N-way defect classification task**
  （N = 该 product 的官方 defect class 数）；**不跨 product 混合 label space**。
- 明确**不迁移**的 Cable 资产：`tau=0.006`、P2.2 normal-aware boundary、
  Cable checkpoint、Cable label names、Good class。
- 不使用 GT masks、bounding boxes 或 pixel annotations；
  不采用 Pixel-AUROC 或 PRO。
- Query 仅用于最终 inference/evaluation，禁止参与 checkpoint、epoch、
  threshold、architecture 或 hyperparameter selection（见 §8）。

### 1.1 Scope (from Feasibility Ruling)

- Feasibility probe 基于 clean commit `ac76ac8`，`formal_test_used=false`，
  裁定 `INCONCLUSIVE-overall / 13-products-feasible`；v2 证据见
  `data/p6b_probe_output_v2/`。
- **固定 13 产品**：bottle, capsule, carpet, grid, hazelnut, leather,
  metal_nut, pill, screw, tile, transistor, wood, zipper。
- `cable` 因与 Cable 研究资产重叠而预先排除（非性能原因）。
- `toothbrush` 因官方 CONFIG 引用的 30 张图像不存在于完整 12 分片合并
  archive 和解压目录中而局部失败；排除依据是**资产完整性**，不是模型性能。

## 2. Support Sampling Protocol

### 2.1 Shots, seeds, run counts

- K ∈ {1, 3, 5}；五个预定 draw seeds：`111, 222, 333, 444, 555`。
- 抽样单位必须是 feasibility probe 定义的 **unique original RGB**
  （按 (product, RGB_SHA256) 折叠后的聚合记录）；仅从对应 product、class、
  support pool 的 **single-type 合规候选**中**无放回**抽样。
- **Run counts（locked）**：每个 product × 每个 method =
  3 K × 5 paired pipeline seeds = **15 runs**；
  全局每 method 总数 = 13 products × 3 K × 5 seeds = **195 runs**。
- 主方案为 **paired pipeline**：同一 seed 同时控制 support draw 与训练随机性
  （§2.5）。**禁止**在每个 K 内做 draw_seed × init_seed 的 5×5 Cartesian
  expansion，禁止任何超出 5 paired seeds 的额外重复。

### 2.2 Locked sampling mechanics

对每个 (product, seed, class)：

1. **Candidate ordering**：single-type 合规候选按 **`rgb_sha256` 字典序**排序，
   得到索引序列 `indices = [0, ..., n-1]`。
2. **sub_seed**：`sub_seed = int.from_bytes(SHA256(
   "p6b_draw_v1|{base_seed}|{product}|{class}").digest()[:8], "big")`
   （固定取 digest 前 8 字节，按大端解释为 unsigned integer）。
3. **Permutation**：`random.Random(sub_seed).shuffle(indices)`
   （Python Mersenne Twister）。
4. **Nested draws**：K=1/3/5 分别取**同一 permutation 的前 1/3/5 项**，
   即 K=1 ⊂ K=3 ⊂ K=5，减少 shot 比较中的抽样混杂。

上述规则使所有 manifest 可由 (冻结 feasibility 记录 + product + seed + K)
确定性复算。

### 2.3 Manifest contract

每个 (product, K, seed) 生成一份 support manifest：

- `schema_version = "p6b_support_manifest_v1"`（已锁定）。
- 字段：`product`, `K`, `seed`, `draw_scheme = "p6b_draw_v1"`，
  `classes`：每个官方 defect class 的
  `{class_name, items: [{rel_path, rgb_sha256}]}`（按抽取顺序），
  `candidate_ordering = "lexicographic_rgb_sha256"`，
  `rng_algorithm = "python_random_mersenne_twister"`，
  `support_query_sha_overlap`（审计结果，见 §8），`manifest_sha256`。
- **manifest_sha256 规则（locked）**：对**删除 `manifest_sha256` 字段后的对象**
  序列化为 UTF-8 字节后取 SHA256，序列化参数固定为
  `json.dumps(obj_without_sha, sort_keys=True, separators=(',', ':'),
  ensure_ascii=False) + "\n"`（末尾 LF）。**sha 字段不得参与自身哈希**；
  评估前必须按此规则重算并比对，不匹配即 fail-closed。

### 2.4 Support/Query overlap audit

逐 manifest 计算 support 与 query 的 RGB SHA256 交集，必须为 0；
非 0 即该产品该配置 fail-closed（审计读取方式见 §8）。

### 2.5 Variance decomposition caveat (locked)

同一 seed 同时控制 support draw 与训练随机性（P3.F `isolated_rng` 语义：
`seed == init_seed == data_seed == augment_seed`）。因此五次重复只能报告
**joint pipeline uncertainty**；不得声称独立分解 support variance 与
optimization variance。

## 3. Frozen Method Transfer (P3.F method-level freeze)

冻结 P3.F 的方法级设计与主要训练超参数；在每个 product 的 support set 上
重新训练允许训练的 terminal adapter 并重新构建 prototypes。
**不迁移 Cable checkpoint**；backbone/feature extractors 保持冻结。

### 3.1 Locked fields — two columns: P3.F verified source value vs P6.B adopted value

| 字段 | P3.F verified source value（实际来源） | P6.B adopted value |
|---|---|---|
| img_size | 336（`--img_size 336`） | 采纳：336 |
| Frozen backbones | CLIP `ViT-L-14-336`（`clipmodel`）、DINOv2 `vit-s14`（`dino_extractor`），clean_v1 下全部参数 `requires_grad=False`（`configure_model_trainable_parameters()`） | 采纳：保持冻结 |
| Trainable modules | 仅 `image_adapter.det_proj.*` + `prototype_mem`（prototypes + logit_scale）；`seg_proj` 显式禁止进入优化器（`adapter_train_mode=terminal_only`） | 采纳：相同 |
| CLIP 分支 | `clip_pool=topk` | 采纳 |
| DINO 分支 | `dino_pool=mean`、`dino_mask_ratio=0.0`、`dino_patch_policy=full`、`dino_backend=hub`，checkpoint `pretrained/dinov2_vits14_pretrain.pth`（固定 `EXPECTED_DINO_SHA256`） | 采纳 |
| Retrieval / inference | FAISS cosine，`inference_mode=raw_faiss`，`protocol_check=strict`，`--no_cascade --no_vis` | 采纳 |
| Prototype 构造 | `prototypes_per_class=1`；`prototype_init_mode="zero"`（clean_v1 强制）；class-mean 初始化（`init_prototypes_with_class_means`）；`num_init_views=1`；`init_logit_scale=10.0` | 采纳；prototypes 在每个 product 的 support set 上重建 |
| Optimizer | AdamW，`weight_decay=0.01` | 采纳 |
| Learning rate | 模型可训练组 `lr=1e-4`；prototype 组 `lr × proto_lr_mult(=3.0) = 3e-4` | **LOCKED**：`lr=1e-4`，`proto_lr=3e-4`（主研究 Agent 裁定，原 U1） |
| Scheduler | `CosineAnnealingLR(T_max=num_epochs)` | 采纳（结构） |
| Epochs / steps | `num_epochs=20`（用 `multiclass_epoch_19.pth`）；`repeat_per_epoch=4` | **LOCKED**：`epochs=20`、`repeat_per_epoch=4`（主研究 Agent 裁定，原 U1） |
| Batch construction | `batch_size=8`；`gradient_accumulation_steps=2`（有效 16）；`SeededEpochSampler(shuffle=True)` + `make_loader_generator(data_seed)` + `DeterministicWorkerSeeder(augment_seed)` | **LOCKED**：`batch=8`、`grad_accum=2`（原 U1）；sampler/worker 确定性机制采纳 |
| Augmentations | **Cable 域实际值**：固定 `Resize((336,336), BICUBIC)` + `RandomApply(RandomRotation(3°), p=0.3)` + `RandomApply(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0), p=0.5)` → `ToTensor` → CLIP Normalize；per-sample 增强经 `augment_seed` + `fork_rng` 隔离 | **LOCKED 为现有 non-cable 分支**（主研究 Agent 裁定，原 U2）：`RandomResizedCrop(336, scale=(0.7,1.0))` + `RandomApply(RandomAffine(degrees=15, translate=(0.1,0.1)), p=0.3)` + 同一 ColorJitter(p=0.5) → `ToTensor` → CLIP Normalize。该分支是**预存代码路径**（MVTec-FS product 非 cable 名，按数据路径判定默认命中），**不是 query-driven adjustment**；per-sample 隔离沿用 `augment_seed` + `fork_rng` |
| Loss 权重 | `label_smoothing=0.1`；`supcon_weight=0`；`orth_weight=0`；`itc_weight=0`；`arc_margin=0` | 采纳：相同 |
| normal-aware / text 分支 | normal-aware head 关闭；`use_text_guided=False`、`proto_text_refine='none'`、无 ITC（纯视觉路径） | 采纳：保持关闭 |
| Initialization / RNG | `training_protocol=clean_v1`，`rng_protocol=isolated_rng`，`seed == init_seed == data_seed == augment_seed` | 采纳 |
| Checkpoint rule | `early_stop_patience=0`（无早停）、`save_every=0`（仅末轮保存）、`checkpoint_format=lightweight`；产物 `multiclass_epoch_19.pth` | 采纳（规则） |

来源路径：`experiments/p3/run_p30_dino_mean_oof.py` `train_command()` /
`validation_command()`；`experiments/p3/run_p3f_clean_five_seed.py`；
`train_multiclass.py`（`configure_model_trainable_parameters()`、optimizer /
scheduler、argparse 默认值、保存分支）；`dataset/__init__.py`（训练 transform
的 cable 分支与 `__getitem__` augmentation_seed 隔离）；
`docs/P3F_CLEAN_FIVE_SEED_BASELINE_PROTOCOL.md` §1。

### 3.2 Not transferred (explicit)

- Cable checkpoint、Cable label names、Good class、`tau=0.006`、
  P2.2 normal-aware boundary、normal-aware head。
- 文本引导分支（`use_text_guided` / `proto_text_refine` / ITC）保持关闭。

## 4. Training Budget on MVTec-FS (LOCKED)

- **LOCKED（主研究 Agent 裁定，原 U1）**：`epochs=20`、`repeat_per_epoch=4`、
  `batch=8`、`grad_accum=2`、`lr=1e-4`、`proto_lr=3e-4`；参数规则对**全部
  product / K / seed 相同**。
- 不同 product 的 class 数 N 与 K 不同，support 样本总量不同，因此实际
  batches / optimizer steps 随之变化：每次运行必须**记录实际 batches 与
  optimizer steps**；本协议**不承诺**不同 N×K 配置的绝对 step 数相同。
- 预算一经锁定不得再调；禁止按 query 指标早停或按产品规模调整。

## 5. Implementation Requirements (locked interfaces; code files NOT specified)

原 U4（训练入口选择）已由主研究 Agent 裁定从 Unresolved 移入本节。
本协议**只锁定以下接口契约，不指定任何代码文件**；具体实现（新增或复用
现有入口）由 implementation agent 在主研究 Agent 授权后执行：

1. **Dataset interface**：支持以 (product, K, seed) support manifest 为唯一
   训练数据来源的 N-way defect classification 训练；候选索引与增广 RNG 语义
   必须与 §2.2 / §3.1 一致；query 数据对训练进程不可见（§8）。
2. **Manifest interface**：产出与校验 `p6b_support_manifest_v1`（§2.3），
   包括确定性复算与 `manifest_sha256` 自洽重算；支持/query SHA overlap 审计
   记录。冻结来源 gate：records JSONL 实际 SHA256 必须等于
   `feasibility.output_hashes` 记录值，且 `common_products_feasible_for_all_shots`
   与 `feasible_products_by_shot`（1/3/5）均必须精确等于锁定 13 产品；
   任一不符 fail-closed。
3. **Firewall interface**：实现 §8 三段式数据流边界，并以可审计记录证明
   （audit record）：overlap=0、manifest hash 匹配、训练进程无 query 读取。
4. **Artifact interface**：产出 §9 所列全部产物（manifest、checkpoint、
   prototype 构建记录、预测导出、聚合指标表、完成记录）。bundle 级事务写：
   正式目标目录已存在时拒绝；全部产物在同级 staging 目录完成后单次原子
   发布；中途失败不得留下正式目录、旧 completion 或部分产物。正式 CLI
   要求 HEAD 可解析且工作树干净；显式 diagnostic 开关的产物必须标记
   `formal_eligible=false`。

## 6. Baselines and Fairness (LOCKED by U3 ruling)

### 6.1 Group A — strict support-only visual group（主公平组，locked）

纯视觉、support-only、无 class-name text prior、无训练期或 query 期特权监督：

| Baseline | Text prior 状态 | 备注 |
|---|---|---|
| CLIP-Proto | 无（冻结 CLIP 视觉特征 + 类原型，不使用 class names/text） | locked |
| DINO-Proto | 无（冻结 DINOv2 视觉特征 + 类原型） | locked |
| Frozen-Fusion ProtoNet-style | 无（冻结双分支融合特征上的原型距离分类） | locked |
| Frozen-Fusion Linear Probe | 无（冻结双分支融合特征上的线性分类器） | locked |
| Ours（P3.0 Method Transfer） | 无（§3 纯视觉冻结管线） | locked |

### 6.2 Group B — text-prior-augmented group（locked，含精确常数）

| Baseline | 锁定常数 | 备注 |
|---|---|---|
| Zero-shot CLIP | 提示模板 `"a photo of a {class_name}."`（class_name 为该 product class universe 官方标签）；不使用 support 样本 | training-free；query 输入仍为纯 RGB |
| Tip-Adapter（training-free） | `alpha=1.0`、`beta=5.5`、prompt `"a photo of a {class_name}."` | training-free；KV cache 仅由共同 support manifests 构造，遵守 §8 firewall |

- Group B 方法引入额外 text prior（预训练图文对齐先验），与 Group A 纯视觉
  方法不严格等价：必须**单独报告 supervision / resource 差异**，不得混作
  等价比较；Zero-shot CLIP 不消耗 support 样本，其资源差异同样单独声明。
- 上述常数一经锁定不得再调；禁止按 query 指标重选 alpha/beta/prompt。

### 6.3 Privileged reference（单列，不入主公平组）

- 使用 query-time mask 或其他 privileged supervision 的公开方法只能列为
  privileged reference。

### 6.4 Fairness constraints (locked)

- Group A/B 所有方法使用**完全相同的 support manifests**（§2.3）。
- Query-time 模型输入仅为 RGB；不使用 GT masks/boxes/pixel annotations。
- 不根据 query metrics 调参；所有 baseline 超参数已按 §6.1/§6.2 锁定，
  不得虚构或事后新增。
- 不笼统写"禁止 text"：每个 baseline 的 text prior 状态按 §6.1/§6.2 逐方法
  明确。

### 6.5 Reserved Extensions（R1–R6，locked as reserved）

- 预留扩展位 **R1、R2、R3、R4、R5、R6**：用于后续可能追加的 baseline 或
  诊断组。任何扩展只能由主研究 Agent 显式裁定激活并升级协议版本；
  激活前 R1–R6 不绑定任何具体方法，不得在预注册后事后追加。

## 7. Evaluation Metrics (Locked)

### 7.1 Metric definitions

- **Class universe**：固定为该 product **support-side 官方 single-type defect
  classes**（feasibility probe 定义的 support pool 单类型候选所属类别），
  字典序固定。这是 DRAFT 阶段对 §7.1 与 §8 query firewall 冲突的消解：
  feasibility JSONL 行会被解析，但 query label/path 字段不被访问、保留或
  用于任何 manifest/class-universe/训练决策；query 侧唯一被消费的信息为
  product/source_pools/rgb_sha256（仅用于 overlap 审计），故 universe 仅由
  support 侧候选导出（对全部 13 个 feasible 产品，逐类 support 候选数 ≥ K
  已保证两侧定义实际一致）。BA 与 Macro-F1 一律在该固定 universe 上计算，
  不随预测结果收缩或扩展。
- **Top-1 accuracy**：query 上预测等于真值的样本比例。
- **Balanced Accuracy**：固定 class universe 上 per-class recall 的算术均值。
- **Macro-F1**：固定 class universe 上 per-class F1 的算术均值，
  `zero_division=0`。
- 不报告 Image-level AUROC、Pixel-AUROC、PRO 作为主指标。

### 7.2 Aggregation order

1. `raw_metric[product, K, seed]`：指标先逐 (product, K, seed) 计算。
2. 每个 (K, seed) 先对 13 产品做**等权 product-macro**。
3. 再对 5 seeds 报告 **mean / std(ddof=1)**。
4. 同时保留**逐产品**的五 seed mean/std 表供审计。
5. 按 §2.5，std 仅表示 joint pipeline uncertainty。

## 8. Query Firewall (Locked, fail-closed)

1. **训练前（允许）**：只读冻结 feasibility JSONL（`data/p6b_probe_output_v2/`）
   中的 **query SHA metadata** 做 §2.4 overlap audit。
2. **训练进程（禁止）**：读取 query pixels、query labels、query class counts
   或任何 query-derived metric；不得据此选择 checkpoint、epoch、threshold、
   architecture 或任何 hyperparameter。
3. **最终 evaluator（唯一出口）**：只有最终 evaluator 被允许加载 query
   RGB 与 labels，且仅用于 §7 指标计算。
4. 逐 manifest 审计 support/query RGB SHA256 overlap，必须为 0；每个 manifest
   按 §2.3 规则记录并重算 `manifest_sha256`，不匹配即 fail-closed。
5. 任一 firewall 条件失败：该 product 该配置标记 INCONCLUSIVE，不得降级为
   部分结论，不得事后放宽条件。

## 9. Artifacts (planned)

每次完整运行必须产出：

- Per (product, K, seed)：support manifest（§2.3，含 sha256 与 overlap 审计）。
- Per product：terminal adapter checkpoint + prototype 构建记录。
- Per (product, K, seed)：query 预测导出（逐样本预测）+ prediction manifest
  （含输入 sha 引用）。
- 聚合表：`raw_metric[product, K, seed]` + 每 (K, seed) 等权 product-macro +
  5-seed mean/std(ddof=1) + 逐产品五 seed mean/std。
- Per run 训练记录：实际 batches 与 optimizer steps（§4）。
- 完成记录：execution HEAD commit、`formal_test_used=false`、
  manifest/预测 sha256、firewall 审计结果。

## 10. Fail-Closed Rules

- 任一 §8 firewall 失败、manifest hash 不匹配、overlap>0、缺 seed/重复、
  或执行时工作树不干净：对应单元记 INCONCLUSIVE，不形成稳定结论。
- 结果不理想时不得事后修改实验定义、门槛或排除困难 product/class。
- 完整性失败不降级为性能结论；不得以性能原因替换数据裁定。

## 11. Upgrade Conditions (to PREREGISTERED)

以下条件已全部满足，协议由主研究 Agent 裁定升级为
**PREREGISTERED — LOCKED FOR EXECUTION**：

1. §12 全部科学 unresolved 字段已由主研究 Agent 裁定并回填（U3 已裁定，
   无剩余科学字段）。✔
2. §3.1 两列表格与 §5 接口契约通过针对实际实现调用链的复核（Stage 1
   代码与 15 项测试独立复验通过）。✔
3. 每个 baseline 的 text prior 状态与超参数已全部锁定（§6.1/§6.2）。✔
4. Stage 1 公共基础设施已验收；模型训练与 baseline 评估仍须待正式
   manifest bundle 审计通过后方可进入（Stage 2）。

### 11.1 Operational note（非科学 unresolved，不阻止预注册）

- 执行算力排程（195 runs/method × 各 method 的云端运行顺序与资源分配）为
  operational note，由用户在云端执行时安排，不属于阻止预注册的科学字段。

## 12. Unresolved Before Preregistration

全部科学 unresolved 字段已由主研究 Agent 裁定，本节不再保留任何未决项：

- **原 U1（训练预算迁移）→ LOCKED**，见 §4。
- **原 U2（增强域适配）→ LOCKED**，见 §3.1 Augmentations 行。
- **原 U3（baseline specification）→ LOCKED**，见 §6.1/§6.2（Group A/B 全部
  方法与常数）；预留扩展位 R1–R6 见 §6.5。
- **原 U4（训练入口选择）→ 移入 §5 Implementation Requirements**（只锁接口，
  不指定代码文件）。

剩余事项仅为 §11.1 operational note（云端算力排程），不属于阻止预注册的
科学字段。协议已由主研究 Agent 裁定升级为 **PREREGISTERED —
LOCKED FOR EXECUTION**；后续任何修改必须新开协议版本并重新裁定。
