# P6.B MVTec-FS-RGB Feasibility Protocol

> Status: **PREREGISTERED — FEASIBILITY PROBE ONLY**
> 本文件仅固化 P6.B MVTec-FS-RGB 数据构造与 probe 契约，不得预注册训练、baseline 超参或性能门槛。
> 此文档确保数据解析的完整性、防泄漏与严格的失败保护（fail-closed），尚未涉及模型结论。

## 1. 研究定义
- **正式名称**: MVTec-FS RGB-Only Cross-Category Few-Shot Multiclass Method Generalization
- **数据适配名称**: MVTec-FS-RGB
- P6.A 保持不变，方法不得命名为 AA-CLIP，预留论文名为 "P3.0 Method Transfer (Ours)"。
- 排除 product 名规范化后等于 `cable` 的数据集，但具体剩余数目不硬编码，由真实数据统计决定。
- 划分单元（split unit）是 unique original RGB image，不是 defect instance。
- 不得称为“原始官方 MVTec-FS protocol”，且本轮实验不产生任何模型性能结论。

## 2. CLI 契约
正式运行需提供以下必需参数，不得硬编码本机路径：
```bash
python -m experiments.p6.probe_p6b_mvtec_fs_rgb_feasibility \
  --dataset-root <MVTec-FS repository root> \
  --cable-reference-root <MVTecAD-Cable image root> \
  --output-dir <output dir>
```

## 3. 权威数据解析规则 (CSV 驱动)
- `dataset-root` 必须指向 MVTec-FS 仓库根目录，并断言 `image/` 和 `CONFIG/` 子目录存在。
- **权威实例索引**：官方 `CONFIG/<product>_config1/train.csv` 和 `valid.csv` 为唯一实例真理来源，每一行对应一个 defect instance。
- **Pool 映射**：
  - `train.csv` 对应 `support` pool（对应 `training` 目录）。
  - `valid.csv` 对应 `query` pool（对应 `testing` 目录）。
  - 遇到未知 pool 或无法映射的状态时，必须发生局部产品级别 fail-closed，不得静默进入 `unknown` pool 计算。
- **CSV 一致性约束**：
  - `(product, id)` 必须唯一。
  - `part` 列必须与当前解析的 `product` 一致。
  - 同一个 `img_id` 必须严格对应唯一的 `img_rel_path` 和唯一的 RGB SHA256，不得存在多义分支。
  - 若整个 `CONFIG` 目录或所需 CSV 缺失，根据缺失层级引发全局或产品级别的完整性失败。
- **JSON 交叉验证**：
  - `LabelMe` JSON 仅用于补充信息与对账。不再验证 JSON 中的 `group_id`（允许 `group_id=null`）。
  - 对账规则：对于特定的 `(img_id, label)`，CSV 中的行数必须与对应 JSON 中具有相同 `label` 的 shape 数量严格相等。任何数量或标签不匹配均触发该产品的完整性失败。

## 4. Aggregate Records

最终聚合必须跨所有 `img_id` 按 **(product, RGB_SHA256)** 折叠，汇总所有来自不同 alias path 的实例行。
- `img_id` 仅用于一致性验证：每个 img_id 必须唯一映射到一个 (normalized_rel_path, SHA256)。
- CSV 标记解析失败或被聚合时，原 `label`（哪怕只出现了 parse failure）依旧必须作为基础类别参与统计。

## 5. Output Verification

Feasibility 结果应通过 `p6b_mvtec_fs_rgb_feasibility_table.md` 交付，并在同级生成带有详细故障原因及 SHA256 指纹列表的 `p6b_mvtec_fs_rgb_feasibility.json` 和 `p6b_mvtec_fs_rgb_image_records.jsonl`：
1. **Atomic Write**: 所有输出必须确保原子级写入（写往同级 `.tmp` 文件再 `os.replace`），防止崩溃导致的中间态，并保证 LF 换行。
2. **Deterministic Output**: 对同样的输入连续运行两次，生成的 3 份文件内容必须按字节相等（相同的键序与输出排版）。
3. **Fail-Closed Gate**: 出现以下任意全局完整性失败时，不应写入正常的逐产品 feasibility 数据，并立刻设探针状态为 `INCONCLUSIVE`：
   - dataset root 下缺失 `readme.md`、`license.txt` 或 `image/`。
   - `CONFIG/` 下不存在有效的 `*_config1` 产品目录。
   - Cable 排除参考库中无受支持图像格式。
4. **Provenance**:
   - `project_provenance`: 运行此探针的 AA-CLIP 仓库的 git commit 和 dirty 状态。
   - `dataset_provenance`: MVTec-FS 数据集（dataset-root）的 git commit 和 dirty 状态，以及提取的 `readme.md`、`license.txt` 和所有的输入 CSV 的 hashes。
- 输出文件绝对禁止逃逸覆盖 `dataset-root` 内部。

## 6. Cable 排除与重叠审计
- `cable` product 必须在哈希重叠检测前**首先被剔除**；需记录排除前的相关数目（如 unique RGB、overlap 数目），保存至 `excluded_product_audit` 中，不得因为 Cable 自身的 overlap 而污染全局完整性门控。
- `cable-reference-root` 必须存在且至少包含一张图片，仅用于路径/哈希 overlap 审计。
- 对于非 Cable 的合法 product：
  - 内容 SHA256 重叠 > 0 为产品级别的 hard integrity failure。此产品失败但**不应**株连其他清白产品。
  - 路径重叠同样需报告并记为失败。
- 输出中需显式标注 `formal_test_used=false` 及 `cable_reference_used_for_overlap_only=true`。

## 7. 局部/全局错误传播与冲突
- **数据级（Global）完整性失败**：如根目录结构缺失、Cable Reference 缺失等，将阻断整个数据集，`dataset_integrity_failures` 记录错误。
- **产品级（Local）完整性失败**：如单一产品 CSV 缺失、CSV/JSON 对账失败、非 Cable 哈希碰撞等，仅令当前产品 `integrity_pass=false`。
- 整体 `probe_status` 若遇到任一 global 或 local 错误即为 `INCONCLUSIVE`，但即便如此，也必须呈现不受影响的合规产品的真实可行性。

## 8. 每产品、每类别统计
必须在每个产品下统计并输出：
- `total_instances`
- `total_unique_rgb`
- `single_type_unique_rgb`
- `same_type_multi_instance_rgb`
- `multi_type_unique_rgb_removed`
- `support_pool_unique_rgb` / `query_pool_unique_rgb`
- 过滤后数量为 0 的 class 必须保留，不得静默删除。

## 9. Shot Feasibility
对每个**未被排除**的 product 和 K ∈ {1, 3, 5}，可行为真的条件（feasible）为严格的逻辑与：
1. 全局数据 `dataset_integrity_failures` 为空。
2. 该 product 的 `integrity_pass` 为 `true`（本地无完整性错误）。
3. 每个官方 defect class 的 support unique RGB >= K。
4. 每个官方 defect class 的 query unique RGB >= 1。
5. Support/Query RGB 完全不相交（无 cross pool collision）。

## 10. 产物与状态
只允许整体输出状态为 `COMPLETE` 或 `INCONCLUSIVE`，本轮不输出 CONFIRMED/NOT CONFIRMED。
输出包含：
1. `p6b_mvtec_fs_rgb_image_records.jsonl`
2. `p6b_mvtec_fs_rgb_feasibility.json`
3. `p6b_mvtec_fs_rgb_feasibility_table.md`
保证确定性与跨次运行一致性，不修改数据集和本地文件 mtime。
