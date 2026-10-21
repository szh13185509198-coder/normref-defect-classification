#!/usr/bin/env python3
"""P21: export the frozen MVTec-FS formal episodes for the cloud CLIP-LoRA run.

The exported file lists, for every (product, K, seed), the class order and the support images
with their labels, and for every product the formal query images WITHOUT their labels. The cloud
never sees a query label; predictions are scored locally (experiments/p21/score.py).
Every listed image is re-hashed here, so the cloud runner can verify its copy byte for byte.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p21.export_episodes
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_stage3_common import file_sha256, load_query_contract  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

MANIFEST_DIR = rc.STAGE1_BUNDLE / "p6b_support_manifests"
OUT = PROJECT_ROOT / "docs" / "p21" / "P21_MVTEC_FS_EPISODES.json"
SCHEMA = "p21_mvtec_fs_episodes_v1"


def build(queries: list[dict], dataset_root: Path) -> dict:
    per_product = {p: [] for p in rc.PRODUCTS}
    for q in queries:
        per_product[q["product"]].append({"rel_path": q["relative_path"], "sha256": q["rgb_sha256"]})
    episodes = []
    for product in rc.PRODUCTS:
        for seed in rc.SEEDS:
            for k_shot in rc.K_VALUES:
                m = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                order = list(m["class_order"])
                supports = [{"rel_path": it["rel_path"], "sha256": it["rgb_sha256"], "label": c}
                            for c, cls in enumerate(order) for it in m["classes"][cls]["items"]]
                if any(len(m["classes"][cls]["items"]) != k_shot for cls in order):
                    raise rc.P6BFailClosedError(f"support count != K: {product} k{k_shot} s{seed}")
                episodes.append({"product": product, "K": k_shot, "seed": seed,
                                 "class_order": order, "supports": supports})
    for item in [s for e in episodes for s in e["supports"]] + [q for qs in per_product.values() for q in qs]:
        if file_sha256(dataset_root / item["rel_path"]) != item["sha256"]:
            raise rc.P6BFailClosedError(f"local SHA mismatch: {item['rel_path']}")
    return {"schema_version": SCHEMA, "dataset": "MVTec-FS formal (P6B contract)",
            "products": list(rc.PRODUCTS), "K_values": list(rc.K_VALUES), "seeds": list(rc.SEEDS),
            "query_labels_included": False,
            "queries": per_product, "episodes": episodes}


def main() -> int:
    queries, meta = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    doc = build(queries, rc.DATASET_ROOT)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    n_sup = sum(len(e["supports"]) for e in doc["episodes"])
    print(f"episodes {len(doc['episodes'])}, queries {meta['formal_query_count']}, support entries {n_sup}")
    print(f"{OUT.relative_to(PROJECT_ROOT)} SHA256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
