#!/usr/bin/env python3
"""P19 §T: build the Real-IAD blind-test manifest from the official jsons only (no image is read).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p19.manifest
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "Real-IAD"
JSONS = DATA / "realiad_jsons" / "realiad_jsons"
MANIFEST = PROJECT_ROOT / "docs" / "p19" / "P19_REALIAD_MANIFEST.json"
SEEDS = (19001, 19002, 19003, 19004, 19005)
POOL, MAX_QUERIES, BANK, MIN_CLASS = 10, 40, 20, 30
NORMAL = "OK"


def view_of(path: str) -> int:
    return int(re.search(r"_C(\d)_", path).group(1))


def split(items: list[str], key: str) -> tuple[list[str], list[str]]:
    items = sorted(items)
    random.Random(key).shuffle(items)
    return items[:POOL], items[POOL:POOL + MAX_QUERIES]


def build_object(obj: str, data: dict) -> dict:
    defects = [r for r in data["test"] if r["anomaly_class"] != NORMAL]
    counts = Counter(view_of(r["image_path"]) for r in defects)
    view = max(sorted(counts), key=lambda v: counts[v])
    classes = {}
    for r in defects:
        if view_of(r["image_path"]) == view:
            classes.setdefault(r["anomaly_class"], []).append(r)
    classes = {c: v for c, v in sorted(classes.items()) if len(v) >= MIN_CLASS}
    if len(classes) < 2:
        raise ValueError(f"{obj}: fewer than two defect classes in view {view}")
    mask_of = {r["image_path"]: r["mask_path"] for r in defects}
    task = {"view": view, "view_counts": dict(sorted(counts.items())), "classes": {}}
    for c, rs in classes.items():
        pool, queries = split([r["image_path"] for r in rs], f"p19|{obj}|{c}")
        task["classes"][c] = {"available": len(rs), "pool": pool, "queries": queries,
                              "masks": {p: mask_of[p] for p in pool + queries}}
    normal_test = [r["image_path"] for r in data["test"]
                   if r["anomaly_class"] == NORMAL and r["image_path"].startswith("OK/") and view_of(r["image_path"]) == view]
    pool, queries = split(normal_test, f"p19|{obj}|{NORMAL}")
    task["normal"] = {"available": len(normal_test), "pool": pool, "queries": queries}
    normal_train = sorted(r["image_path"] for r in data["train"]
                          if r["anomaly_class"] == NORMAL and r["image_path"].startswith("OK/") and view_of(r["image_path"]) == view)
    task["banks"] = {str(s): random.Random(f"p19bank|{obj}|{s}").sample(normal_train, BANK) for s in SEEDS}
    task["supports"] = {}
    for s in SEEDS:
        per = {}
        for c in list(task["classes"]) + [NORMAL]:
            base = list(task["classes"][c]["pool"] if c != NORMAL else task["normal"]["pool"])
            random.Random(f"p19sup|{obj}|{c}|{s}").shuffle(base)
            per[c] = base
        task["supports"][str(s)] = per
    used_test = {p for c in task["classes"].values() for p in c["pool"] + c["queries"]} | set(task["normal"]["pool"] + task["normal"]["queries"])
    if used_test & {p for b in task["banks"].values() for p in b}:
        raise ValueError(f"{obj}: bank overlaps test images")
    return task


def build() -> dict:
    objects = {}
    for f in sorted(JSONS.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        obj = data["meta"]["prefix"].rstrip("/")
        objects[obj] = build_object(obj, data)
    if len(objects) != 30:
        raise ValueError(f"expected 30 objects, found {len(objects)}")
    return {"schema_version": "p19_manifest_v1", "protocol": "docs/P19_REALIAD_BLIND_TEST_PROTOCOL.md#t",
            "image_root": "data/Real-IAD/realiad_512/<object>/", "seeds": list(SEEDS),
            "pool": POOL, "max_queries": MAX_QUERIES, "bank": BANK, "objects": objects}


def main() -> int:
    manifest = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    n_cls = sum(len(o["classes"]) for o in manifest["objects"].values())
    n_q = sum(len(c["queries"]) for o in manifest["objects"].values() for c in o["classes"].values())
    n_ok = sum(len(o["normal"]["queries"]) for o in manifest["objects"].values())
    print(f"objects {len(manifest['objects'])}, defect classes {n_cls}, defect queries {n_q}, normal queries {n_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
