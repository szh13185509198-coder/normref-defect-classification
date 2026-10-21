# P6.B Execution Protocol — Addendum A: Baseline Mathematical Semantics (Preregistered Pre-Execution)

> Status: **PREREGISTERED — LOCKED (PRE-EXECUTION ADDENDUM A)**，隶属于主协议
> `docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md`（PREREGISTERED — LOCKED FOR
> EXECUTION）。本 Addendum 与主协议同锁：修改必须新开版本并重新裁定。

## 0. Declarations

1. 截至本 Addendum 锁定时刻，**尚未查看任何 P6.B 模型性能**（无任何
   baseline 或 Ours 的 query 指标被读取或用于本文件的任何决策）。
2. 本 Addendum **仅消除 baseline 实现歧义**：不改变 13 产品、K∈{1,3,5}、
   seeds {111,222,333,444,555}、训练预算（主协议 §4）、指标定义与聚合
   （主协议 §7）、query firewall（主协议 §8）中的任何内容。
3. Stage 2 及以后产生的每个评估 artifact 必须**同时记录主协议与本
   Addendum 的 SHA256**（例如 `protocol_sha256` 与 `addendum_a_sha256`
   字段）；加载前校验，任一不匹配即 fail-closed。

## 1. Locked mathematical semantics

### 1.1 Embeddings（全部冻结 backbone，默认 `requires_grad=False`）

- **CLIP embedding**：冻结 CLIP `ViT-L-14-336` 的**原生 projected global
  image embedding**，随后 **L2 normalize**。
- **DINO embedding**：冻结 DINOv2 `vit-s14`，对 **full patch tokens**
  （`dino_patch_policy=full` 的全部 patch tokens，不含 class token）做
  **mean pooling**，随后 **L2 normalize**。
- **Frozen Fusion**：`concat(CLIP embedding, DINO embedding)`，对拼接结果
  **再 L2 normalize**。

### 1.2 Prototype 语义（CLIP-Proto / DINO-Proto / Frozen-Fusion ProtoNet-style）

1. 先对每个 support embedding **L2 normalize**；
2. 逐类求**均值**，再 **L2 normalize** 得该类原型；
3. 分类 = query embedding（同表示、已 L2 normalize）与各原型 **cosine
   argmax**；
4. **并列（ties）按 manifest `class_order`**（即固定 class universe 字典序，
   取最靠前者）。
- CLIP-Proto / DINO-Proto / Frozen-Fusion Proto 分别使用 1.1 中对应的表示，
  其余语义完全一致。

### 1.3 Frozen-Fusion Linear Probe

- 输入：Frozen Fusion embedding（冻结，不回传 backbone 梯度）。
- 仅训练 `Linear(dim, N, bias=True)`（N = 该 product class universe 大小）。
- Loss：CrossEntropy，`label_smoothing=0.1`。
- Optimizer：AdamW，`lr=1e-4`，`weight_decay=0.01`。
- Scheduler：`CosineAnnealingLR(T_max=epochs)`。
- 预算：`20 epochs`、`repeat_per_epoch=4`、`batch=8`、`grad_accum=2`；
  取**最终 epoch** 权重；seed 隔离（主协议 §2.5 paired pipeline 语义）。
- **特征提取与增广（locked）**：每个 epoch/repeat pass 对 support 图像应用
  主协议 §3.1 **non-cable augmentation**（`RandomResizedCrop(336,
  scale=(0.7,1.0))` → `RandomApply(RandomAffine(degrees=15,
  translate=(0.1,0.1)), p=0.3)` → `RandomApply(ColorJitter(0.2,0.2,0.15,
  hue=0.0), p=0.5)`，PIL 空间、per-sample 由 `augment_seed` 确定性隔离），
  随后**重新提取冻结 Fusion feature** 供 head 训练；**禁止**把单一
  deterministic embedding 跨 epoch/repeat 重复训练。
- **常数锁定**：epochs/repeat/batch/grad_accum/lr/weight_decay/
  label_smoothing 为锁定常数，不是可调超参数；任何非精确值必须
  fail-closed。

### 1.4 Evaluation view

- **所有方法**的 query inference 使用**单个确定性 336 evaluation view**
  （Resize(336,336) → CLIP Normalize），不使用随机增强。
- training-free 方法的 support/cache 构造同样使用该确定性 336
  evaluation view。

### 1.5 Zero-shot CLIP

- prompt 模板：`"a photo of a {class_name}."`，`class_name` 中**下划线转空格**；
  text embeddings 由冻结 CLIP text encoder 得到并 L2 normalize。
- logits = `logit_scale × (query @ text.T)`，`logit_scale = 100`。

### 1.6 Tip-Adapter（training-free）

- cache keys = support **CLIP embeddings**（L2 normalized）；
  cache values = 对应 class 的 one-hot。
- `affinity = exp(-5.5 × (1 − image_emb @ cache_keys.T))`（beta=5.5 锁定）。
- `logits = CLIP logits + 1.0 × (affinity @ cache_values)`（alpha=1.0 锁定）。
- **不搜索、不调整任何参数**；禁止按 query 指标重选。

### 1.7 Ours

- Ours（P3.0 Method Transfer）完全沿用主协议 §3/§4，本 Addendum 不重述、
  不修改其任何字段。

## 2. Provenance requirement

Stage 2+ 评估产物 completion/audit JSON 必须包含：

- `protocol_relative_path` + `protocol_sha256`（主协议）；
- `addendum_relative_path` + `addendum_a_sha256`（本文件）；
- `stage1_completion_sha256`：Stage 1 formal bundle completion 的固定
  pin 值
  `5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73`，
  加载 bundle 时必须校验，不匹配即 fail-closed。

加载或评估前校验上述全部 SHA；任一缺失或不匹配 → fail-closed（INCONCLUSIVE）。
