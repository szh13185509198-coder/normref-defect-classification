#!/usr/bin/env python3
"""P47 amendment A1: split the P41 engine-vs-reference disagreements (docs/P47_SCALING_AND_TILING_PROTOCOL.md).

Runs P41's scoring unchanged on a copy of the P41 device features and records the per-query predictions it makes.
    python -m experiments.p47.flips
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p41 import fused_engine as fe  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p14.final_formal import ARR  # noqa: E402

SRC = PROJECT_ROOT / "ckpt" / "p41" / "jetson_orin_nano"
WORK = PROJECT_ROOT / "ckpt" / "p47" / "flips"


def main() -> int:
    target = WORK / "p47_flips.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    src = WORK / "p41_copy"
    src.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("fused_*.pt"):
        shutil.copy2(f, src / f.name)
    seen = []
    orig = fa.score_rows

    def recording(scores, order, truth, meta, sink=None, shas=None):
        rows = orig(scores, order, truth, meta, sink, shas)
        if sink is not None:
            seen.extend(sink[-len(scores):])
        return rows
    fa.score_rows = recording
    fe.OUT = WORK
    fe.score(argparse.Namespace(tag="p41_copy"))
    ref = json.loads((SRC / "p41_scores.json").read_text(encoding="utf-8"))
    rerun = json.loads((src / "p41_scores.json").read_text(encoding="utf-8"))
    same_as_p41 = ref["fused_minus_cache"]["agreement"] == rerun["fused_minus_cache"]["agreement"]
    by = {(u["method"], u["product"], u["K"], u["seed"]): u for u in seen}
    split = {}
    for (m, p, k, s), u in by.items():
        if m != f"{ARR}|fused":
            continue
        c = by[(f"{ARR}|cache", p, k, s)]
        d = split.setdefault(str(k), {"n": 0, "agree": 0, "right_to_wrong": 0, "wrong_to_right": 0, "wrong_to_wrong": 0})
        for t, pe, pc in zip(u["true"], u["pred"], c["pred"]):
            d["n"] += 1
            if pe == pc:
                d["agree"] += 1
            elif pc == t:
                d["right_to_wrong"] += 1
            elif pe == t:
                d["wrong_to_right"] += 1
            else:
                d["wrong_to_wrong"] += 1
    total = {key: sum(v[key] for v in split.values()) for key in next(iter(split.values()))}
    res = {"schema_version": "p47_flips_v1", "protocol": "docs/P47_SCALING_AND_TILING_PROTOCOL.md#A1",
           "agreement_equals_p41": same_as_p41, "per_K": split, "total": total,
           "note": "counts over query x episode cells (13 products x 5 seeds per K)"}
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"agreement_equals_p41": same_as_p41, "total": total, "per_K": split}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
