# P6.B Execution Protocol — Addendum B: R1/R2 Reserved-Extension Activation (Preregistered Pre-Execution)

> Status: **PREREGISTERED — LOCKED PRE-EXECUTION**，隶属于主协议
> `docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md`（PREREGISTERED — LOCKED FOR
> EXECUTION）与 `docs/P6B_EXECUTION_ADDENDUM_A.md`（PREREGISTERED — LOCKED
> ADDENDUM A）。本 Addendum 与主协议、Addendum A 同锁：修改必须新开版本并
> 重新裁定。

## 0. Declarations（锁定时刻的事实声明）

截至本 Addendum B 裁定/创建时刻：

1. **未读取任何 P6.B query performance**：无任何 query pixels / query labels /
   query prediction / query accuracy / BA / F1 / confusion 或任何能反映模型在
   query 上好坏的指标被读取或用于本文件的任何决策。
2. **未读取任何 baseline / Ours 的 query metric**。
3. **原七方法尚未开始正式 grid**（无 `13 products × 3 shots × 5 seeds` 的正式
   query evaluation 发生）。
4. **原七方法的 cloud diagnostic smoke 在本轮 protocol upgrade 前暂停**；本轮
   先冻结九方法的实验语义、provenance、query firewall 与 support-only
   execution，随后才允许 support-only diagnostic smoke。

本 Addendum **仅激活两个 reserved extension（R1、R2）并补充 Protocol Bundle
V2 provenance**：不改变 13 products、K∈{1,3,5}、five paired seeds
{111,222,333,444,555}、Stage 1 manifests/bundle、support/query split、support
draw、指标定义与聚合（主协议 §7）、query firewall（主协议 §8）、Group A/B 原
方法语义（Addendum A §1）中的任何内容。

## 1. Reserved extension activation（主协议 §6.5）

依据主协议 §6.5，主研究 Agent 显式裁定激活预留扩展位 **R1、R2**，并升级到
Protocol Bundle V2（见 §4）。R3–R6 仍保留、未激活、不绑定任何方法。

- **R1 = Exact P3.0 Frozen Representation Proto**（method id `p30_frozen_proto`）。
- **R2 = Tip-Adapter-F\***（firewall-constrained；method id `tip_adapter_f`）。

R1、R2 是本轮最后两个 baseline/ablation 扩展；本轮不再新增 R3/R4 或任何其他
baseline。

## 2. R1 — Exact P3.0 Frozen Representation Proto（locked math）

目的：提供一个 **feature-matched frozen/no-adaptation reference**。R1 与 Ours
（主协议 §3）共享完全一致的 frozen pre-terminal token source，但关闭 terminal
adapter，并以非参数 class-mean prototype 分类。

R1 **不是 terminal-adapter-only 的严格模块消融**：Ours 除训练
`image_adapter.det_proj` 外，还训练 `prototype_mem.prototypes` 与
`prototype_mem.logit_scale`；R1 则三者均不训练。因此 Ours−R1 只能解释为“完整
support-side adaptation（terminal adapter + learned prototype classifier）相对冻结
pre-terminal prototype reference 的差异”，不得单独归因于 terminal adapter。

### 2.1 Frozen pre-terminal representation（与 Ours 完全一致）

R1 与 Ours 共享同一 `AdaptedCLIP` 前向管线，直到 terminal adapter 之前：

- **CLIP 分支（pre-terminal tokens）**：与 Ours 相同的冻结 CLIP
  `ViT-L-14-336` patch-embedding + positional encoding + `_run_clip_blocks`
  （`adapter_train_mode=terminal_only`，24 层全部 `torch.no_grad`）+
  `image_encoder.ln_post`，得到 `ln_post` 之后、`det_proj` 之前的 token 张量
  `[B, N+1, 1024]`。**该张量即 Ours 送入 terminal adapter `det_proj` 的输入
  特征**。
- **DINO 分支**：与 Ours 相同——冻结 DINOv2 `vit-s14`（`dino_patch_policy=full`）
  的 full patch tokens 做 **mean pooling**。DINO 分支在 Ours 中本就没有
  terminal adapter，故 R1 与 Ours 的 DINO 分支逐位一致。

### 2.2 R1 分类语义（terminal adaptation OFF）

```text
CLIP 分支：clip_feat = L2normalize( topk_avg_pool(ln_post_tokens[B,N+1,1024], k_ratio=0.1) )   # 1024-d
DINO 分支：dino_feat = L2normalize( dino_patches.mean(dim=1) )                                   # 384-d
Frozen pre-terminal fusion：feature = L2normalize( concat([clip_feat, dino_feat]) )             # 1408-d
prototype   = 逐类 support feature 取均值
prototype   = L2normalize(prototype)
query_feat  = L2normalize(query pre-terminal feature)
分类        = cosine argmax(query_feat, prototypes)
tie-break   = 固定 manifest class_order（最靠前类优先）
```

R1 与 Ours 相比：
- 相同 CLIP `_run_clip_blocks` 与 `ln_post`（相同 Top-K 池化算子
  `topk_avg_pool(k_ratio=0.1)`，作用于 pre-terminal `[B,N+1,1024]` tokens）；
- 相同 DINO mean 实现；
- 相同 feature concatenation / ordering（CLIP 在前、DINO 在后）；
- 相同 pre-terminal normalization 位置（分支各自 L2 → concat → 再 L2）；
- 相同 deterministic query evaluation view（Addendum A §1.4 的单一确定性 336
  view）；
- 相同 support manifests / K / seed；
- **不使用 text prior**；
- **不训练任何 terminal adapter、不训练 backbone**；
- prototype = support feature class mean，随后 L2 normalize；
- query feature L2 normalize；cosine argmax；tie-break 用固定 `class_order`。

相对 Ours，R1 同时关闭 terminal adaptation 与 learned prototype-classifier
adaptation（不应用 `det_proj`，也不训练 prototype/logit scale；无任何可训练参数）。

### 2.3 R1 实现与验收约束

- R1 必须使用与 Ours **同一个真实 feature construction helper**，不得复制一套新的
  CLIP/DINO backbone loader；pre-terminal 特征与 Ours 送入 `det_proj` 的输入张量
  逐位一致（允许浮点 tolerance）。实现上通过对 `AdaptedCLIP.image_adapter.det_proj`
  注册 forward-pre-hook 捕获 Ours 的 terminal-adapter 输入并复用
  `AdaptedCLIP.topk_avg_pool`、`AdaptedCLIP.dino_extractor.forward_patches`，
  以保证「同一 helper、零重复实现、无需修改 P3.F 代码」。
- `trainable_param_count == 0`：除临时 prototype 张量外不得有任何 trainable
  module；所有模型参数 `requires_grad=False`；不构造任何 optimizer。
- R1 的 support 构造阶段不得读取 query pixels / query labels / query metrics。

### 2.4 与现有 Frozen-Fusion Proto 的关系（重要区分）

现有 **Frozen-Fusion Proto**（method id `fusion_proto`，Addendum A §1.1/§1.2）
**继续保留为普通 baseline**，但**不得再被称为 terminal-adaptation 的严格消融**：
其 CLIP representation 使用 **CLIP 原生 projected global image embedding**，与
R1/Ours 共享的 pre-terminal token source 不同。R1 因而是比 Frozen-Fusion Proto
更接近 Ours 的 feature-matched frozen reference；但由于 Ours 还学习 prototype
classifier，R1 仍不得表述为 terminal-adapter-only 的严格消融。

论文允许表述：`Ours outperforms a feature-matched frozen pre-terminal prototype
reference`。论文禁止表述：`R1 proves that the terminal adapter alone improves X%`。

## 3. R2 — Tip-Adapter-F\*（firewall-constrained，locked math + constants）

在 Group B 追加一个可训练 CLIP few-shot baseline，内部与论文中命名为
**Tip-Adapter-F\***（等价写法 **Tip-Adapter-F (firewall-constrained)**）。

命名理由：官方 Tip-Adapter-F 会训练 cache keys、每 epoch 查看 evaluation
performance、选择 best epoch、并搜索 alpha/beta，这些行为与本项目 query
firewall（主协议 §8）冲突。因此本项目采用一个**预注册、无 query tuning 的受限
版本**。当官方实现与本项目协议冲突时，**以本项目 preregistered firewall 为最高
优先级**；不得为「更接近官方结果」读取 query 或增加调参。

### 3.1 Frozen / trainable 划分

冻结（`requires_grad=False`）：
- CLIP image encoder；
- CLIP text encoder；
- cache values（support one-hot）；
- class text embeddings（`text_weights`）。

可训练（`requires_grad=True`）：
- **cache keys only**（唯一可训练参数）。

初始化：cache keys 从普通 Tip-Adapter 的 support cache keys 初始化（support
**CLIP embeddings**，L2 normalized；与 Addendum A §1.6 的 training-free
Tip-Adapter cache 相同）。

### 3.2 分类语义（保持 Tip-Adapter）

```text
affinity     = exp(-beta * (1 - image_emb @ cache_keys.T))
cache_logits = affinity @ cache_values
clip_logits  = 100 * image_emb @ text_weights.T
logits       = clip_logits + alpha * cache_logits
```

固定常数（fail-closed，禁止任何 search / query tuning / best-epoch 选择）：

```text
alpha = 1.0
beta  = 5.5
```

其中 `text_weights` 为 zero-shot prompts `"a photo of a {class_name}."`
（下划线转空格，Addendum A §1.5）经冻结 CLIP text encoder 得到并 L2 normalize；
`clip_logits` 的 logit scale = 100（Addendum A §1.5）。

### 3.3 R2 训练常数（locked；faithful to official cache-key training where compatible）

```text
trainable set          = cache_keys only
optimizer              = AdamW（参数组精确等于 cache_keys）
lr                     = 1e-3
eps                    = 1e-4
weight_decay           = 1e-2
scheduler              = CosineAnnealingLR(T_max = 期望 optimizer steps 总数)
batch_size             = 8
gradient_accumulation  = 1
num_epochs             = 20（final epoch only）
repeat_per_epoch       = 1
augmentation           = 主协议 §3.1 non-cable augmentation（RandomResizedCrop(336,
                         scale=(0.7,1.0)) + RandomApply(RandomAffine(15,(0.1,0.1)),p=0.3)
                         + RandomApply(ColorJitter(0.2,0.2,0.15,0.0),p=0.5)），
                         每 epoch/repeat pass 重新提取冻结 CLIP image embedding
support_repeat         = 每 epoch 遍历 support 一次（repeat_per_epoch=1）
RNG                    = seed == init_seed == data_seed == augment_seed；
                         SeededEpochSampler + make_loader_generator +
                         DeterministicWorkerSeeder（与主协议 §3.1 一致）
loss                   = CrossEntropy（logits 如 §3.2；label_smoothing=0.0）
checkpoint_selection   = final_epoch_only
```

期望 optimizer steps = `num_epochs × repeat_per_epoch ×
ceil(support_count / batch_size) / gradient_accumulation`（grad_accum=1 时即
`num_epochs × repeat_per_epoch × ceil(support_count / batch_size)`）；实际步数
不符即 fail-closed。

禁止（fail-closed）：
- alpha search / beta search / prompt search / validation tuning / query tuning；
- best epoch selection / checkpoint selection / early stopping；
- 训练期读取 query（pixels / labels / metrics）。

completion / audit 必须记录：
```text
checkpoint_selection      = final_epoch_only
query_used_for_selection  = false
validation_used_for_selection = false
```

## 4. Protocol Bundle V2 provenance（fail-closed）

Stage 2+ 的 support-side artifact / completion 必须记录并校验**三份 protocol
文档**与 Stage 1 completion 的 SHA：

```text
main_protocol_sha256      = 主协议 docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md（canonical-LF）
addendum_a_sha256         = docs/P6B_EXECUTION_ADDENDUM_A.md（canonical-LF）
addendum_b_sha256         = docs/P6B_EXECUTION_ADDENDUM_B.md（canonical-LF，本文件）
stage1_completion_sha256  = Stage 1 formal bundle completion 固定 pin
                            5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73
```

- 三份 protocol docs 使用 canonical-LF 归一化（CRLF→LF 后 hash，拒绝 bare-CR）；
  运行开始时校验、completion 写入前再次校验。
- 若代码使用不同字段命名，可保持兼容，但必须**同时 pin 三份 protocol docs**。
- 任一 SHA 缺失或不匹配：

```text
fail-closed
formal_eligible = false
```

不得继续正式执行。Protocol Bundle 版本记为 **V2**（原 V1 只 pin 主协议 +
Addendum A；V2 追加 Addendum B）。

## 5. 不变项（未被本 Addendum 修改）

- 13 products（Cable excluded；toothbrush excluded due to asset integrity）；
- K ∈ {1,3,5}；five paired seeds {111,222,333,444,555}；
- Stage 1 support manifests / formal bundle（不变）；
- support/query split、support draw、unique-original-RGB sampling、
  support/query RGB SHA overlap = 0；
- 指标定义与聚合（主协议 §7，product-macro → 5-seed mean/std(ddof=1)）；
- query firewall（主协议 §8，fail-closed）；
- Group A/B 原方法语义（Addendum A §1，含 Frozen-Fusion Proto 保留）；
- Ours（主协议 §3/§4）任何字段。

## 6. 九方法诊断顺序（support-only diagnostic smoke）

```text
01 CLIP Proto                         (clip_proto)
02 DINO Proto                         (dino_proto)
03 Frozen-Fusion Proto                (fusion_proto)
04 Zero-shot CLIP                     (zero_shot_clip)
05 Tip-Adapter                        (tip_adapter)
06 Frozen-Fusion Linear Probe         (linear_probe)
07 R1 Exact P3.0 Frozen Proto         (p30_frozen_proto)
08 R2 Tip-Adapter-F*                  (tip_adapter_f)
09 Ours                               (ours)
```

九方法 support-only diagnostic smoke 全部使用 bottle / K=1 / seed=111 / 同一
Stage 1 support manifest / diagnostic mode / `formal_eligible=false`；smoke
**只验证工程语义，不得产生或读取 query performance**。smoke 通过后停止，交回主
研究 Agent 审查；仅在主研究 Agent 明确给出 `APPROVE_FORMAL_GRID` 后才允许进入
query evaluation。
