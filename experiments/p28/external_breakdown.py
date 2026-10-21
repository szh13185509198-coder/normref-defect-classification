#!/usr/bin/env python3
"""P28 (external) follow-up: split the VisA / MTD errors by kind, read-only.

On normal-inclusive tasks the mask oracle is not comparable on normal images: they have no mask, so the oracle pools
them uniformly while every defect image is pooled over its defect only, and the pooling itself reveals whether an
image is normal (the oracle never misclassifies a normal image). This script therefore splits the dual branch's errors
into normal -> defect, defect -> normal and wrong defect type, and reports the oracle only on the wrong-defect-type
errors of defect images, the part comparable with MVTec-FS (which has no normal class).

Inputs: ckpt/p28/p28_external_error_decomposition.json (oracle predictions, reproduced P27 exactly) and
ckpt/p14_formal/external/p14_external_predictions.jsonl (dual branch).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p28.external_breakdown
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.revision import revision_common as rc  # noqa: E402

ARR = "fused_adareg_regres_soft_q99"
NORMAL = {"visa": "normal", "mtd": "MT_Free"}
SRC = PROJECT_ROOT / "ckpt" / "p28" / "p28_external_error_decomposition.json"
P14_PRED = PROJECT_ROOT / "ckpt" / "p14_formal" / "external" / "p14_external_predictions.jsonl"
OUT = PROJECT_ROOT / "ckpt" / "p28" / "p28_external_breakdown.json"


def main() -> int:
    oracle = rc.load_json(SRC)["oracle_predictions"]
    arr = {}
    with P14_PRED.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["method"] == ARR:
                arr[f"{r['task']}|K{r['K']}|s{r['seed']}"] = dict(zip(r["rgb_sha256"], zip(r["true"], r["pred"])))
    result = {}
    for ds, normal in NORMAL.items():
        result[ds] = {}
        for k in rc.K_VALUES:
            c = Counter()
            for key, o in oracle.items():
                if not key.startswith(ds) or f"|K{k}|" not in key:
                    continue
                q = arr[key]
                for sha, po in zip(o["rgb_sha256"], o["pred"]):
                    t, pa = q[sha]
                    if t == normal:
                        c["normal_images"] += 1
                        c["arr_normal_to_defect"] += pa != t
                        c["oracle_errors_on_normal_images"] += po != t
                    else:
                        c["defect_images"] += 1
                        if pa == normal:
                            c["arr_defect_to_normal"] += 1
                        elif pa != t:
                            c["arr_wrong_defect_type"] += 1
                            c["arr_wrong_defect_type_oracle_right"] += po == t
                            c["arr_wrong_defect_type_oracle_same_wrong"] += po == pa
                        c["oracle_errors_on_defect_images"] += po != t
            errors = c["arr_normal_to_defect"] + c["arr_defect_to_normal"] + c["arr_wrong_defect_type"]
            wt = c["arr_wrong_defect_type"]
            result[ds][str(k)] = {**c, "arr_errors": errors,
                                  "share_normal_to_defect": c["arr_normal_to_defect"] / errors,
                                  "share_defect_to_normal": c["arr_defect_to_normal"] / errors,
                                  "share_wrong_defect_type": wt / errors,
                                  "wrong_defect_type_fixed_by_oracle": c["arr_wrong_defect_type_oracle_right"] / wt if wt else None}
    rc.write_json(OUT, {"schema_version": "p28_external_breakdown_v1", "provenance": rc.git_provenance(),
                        "source": str(SRC.relative_to(PROJECT_ROOT)), "by_dataset_K": result})
    lines = ["# P28 (external) — kinds of dual-branch errors; oracle only on wrong-defect-type errors", "",
             "| Data | K | normal → defect | defect → normal | wrong defect type | of which fixed by oracle | oracle errors on normal images |",
             "|---|---|---|---|---|---|---|"]
    for ds, by_k in result.items():
        for k, v in by_k.items():
            lines.append(f"| {ds} | {k} | {100 * v['share_normal_to_defect']:.0f}% | {100 * v['share_defect_to_normal']:.0f}% | "
                         f"{100 * v['share_wrong_defect_type']:.0f}% | {100 * v['wrong_defect_type_fixed_by_oracle']:.0f}% | "
                         f"{v['oracle_errors_on_normal_images']} / {v['normal_images']} |")
    text = "\n".join(lines)
    OUT.with_suffix(".md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
