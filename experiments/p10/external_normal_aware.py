#!/usr/bin/env python3
"""R2: normal-aware industrial metrics on VisA / MTD (docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md).

Re-runs the P10 external methods with the identical code path to obtain per-query
predictions, verifies every cell metric against the published P10 external results, and
computes false-reject / escape metrics for the P10 methods and the archived frozen methods.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.external_normal_aware --device cuda
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10.industrial_metrics import mean_sd, normal_aware, per_thousand  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p10_external" / "normal_aware"
PUBLISHED = ee.RESULTS_DIR / "external_results.json"
GUARDED = ("experiments/p10", "experiments/revision", "docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md")
REPORT_METHODS = ("simpleshot_cl2n", "fused_xclass_clip18", "fused_normref_clip18",
                  "normref_pool_clip18", "fused_normref_clip18_m5", "fusion_proto_local",
                  "fused_normcenter_mean_clip18") + ee.FROZEN_METHODS
METRIC_KEYS = ("good_recall", "false_reject_rate", "defect_recall", "escape_rate", "binary_ba",
               "defect_type_acc_detected")


def guard_and_lock() -> dict:
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; R2 is single-shot")
    lock = OUT.with_name(OUT.name + ".lock")
    try:
        lock.mkdir()
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    return rc.git_provenance()


def frozen_predictions() -> list[dict]:
    out = []
    with tarfile.open(ee.FROZEN_QUERY_BUNDLE) as archive:
        for member in archive.getmembers():
            if not member.name.endswith("predictions.jsonl"):
                continue
            rows = [json.loads(line) for line in archive.extractfile(member).read().decode("utf-8").splitlines() if line]
            r0 = rows[0]
            out.append({"method": r0["method"], "task": r0["task"], "K": int(r0["K"]), "seed": int(r0["seed"]),
                        "rgb_sha256": [r["rgb_sha256"] for r in rows],
                        "true": [r["true_label"] for r in rows], "pred": [r["predicted_label"] for r in rows]})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    provenance = guard_and_lock()
    device = torch.device(args.device)
    orders, _ = ee.frozen_artifacts()
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))

    sink, rows = [], []
    for task in ee.TASKS:
        with torch.no_grad():
            r, _info = ee.run_task(task, orders[task], factory, device, args.batch_size, pred_sink=sink)
        rows += r
        print(f"[R2] {task} re-run done", flush=True)

    published = {(r["method"], r["task"], int(r["K"]), int(r["seed"])): r
                 for r in rc.load_json(PUBLISHED)["raw_metrics"]}
    mismatches = [(r["method"], r["task"], r["K"], r["seed"]) for r in rows
                  if any(r[m] != published[(r["method"], r["task"], r["K"], r["seed"])][m] for m in rc.METRICS)]
    if mismatches or len(rows) != len(published):
        raise rc.P6BFailClosedError(f"re-run does not reproduce published metrics: {len(mismatches)} cells, "
                                    f"e.g. {mismatches[:3]}")

    cells = {}
    for unit in sink + frozen_predictions():
        normal = ee.NORMAL_CLASS[unit["task"].split("__")[0]]
        cells[(unit["method"], unit["task"], unit["K"], unit["seed"])] = normal_aware(unit["true"], unit["pred"], normal)

    summary = {}
    for m in REPORT_METHODS:
        summary[m] = {}
        for k in rc.K_VALUES:
            visa_per_seed = {key: [statistics.fmean(cells[(m, t, k, s)][key] for t in ee.VISA_TASKS)
                                   for s in rc.SEEDS] for key in METRIC_KEYS}
            mtd_per_seed = {key: [cells[(m, ee.MTD_TASK, k, s)][key] for s in rc.SEEDS] for key in METRIC_KEYS}
            visa = {key: mean_sd(v) for key, v in visa_per_seed.items()}
            mtd = {key: mean_sd(v) for key, v in mtd_per_seed.items()}
            summary[m][str(k)] = {
                "visa_object_macro": visa, "mtd": mtd,
                "visa_per_1000_at_95pct_good": per_thousand(visa["false_reject_rate"]["mean"], visa["escape_rate"]["mean"]),
                "mtd_per_1000_at_95pct_good": per_thousand(mtd["false_reject_rate"]["mean"], mtd["escape_rate"]["mean"])}
    result = {"schema_version": "p10_external_normal_aware_v1",
              "protocol": "docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md#r2", "provenance": provenance,
              "integrity": {"cells_rechecked": len(rows), "mismatches": 0},
              "summary": summary,
              "cells": [{"method": k[0], "task": k[1], "K": k[2], "seed": k[3], **v} for k, v in cells.items()]}

    def f(v):
        return f"{100 * v['mean']:.1f}"
    lines = ["# R2 — normal-aware industrial metrics (argmax decisions, no threshold)", "",
             f"Commit `{provenance['head']}`. Re-run reproduced all {len(rows)} published cells exactly. "
             "Good recall = 1 − false-reject rate; defect recall = 1 − escape rate; type acc. among detected defects.", ""]
    for ds, key in (("VisA (object macro)", "visa_object_macro"), ("MTD", "mtd")):
        per = "visa_per_1000_at_95pct_good" if key == "visa_object_macro" else "mtd_per_1000_at_95pct_good"
        lines += [f"## {ds}", "",
                  "| Method | K | Good recall | Defect recall | Binary BA | Type acc. | False rejects / 1000 | Escapes / 1000 |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in REPORT_METHODS:
            for k in rc.K_VALUES:
                s = summary[m][str(k)][key]
                p = summary[m][str(k)][per]
                lines.append(f"| {m} | {k} | {f(s['good_recall'])} | {f(s['defect_recall'])} | {f(s['binary_ba'])} | "
                             f"{f(s['defect_type_acc_detected'])} | {p['false_rejects_per_1000']:.0f} | {p['escapes_per_1000']:.0f} |")
        lines.append("")
    text = "\n".join(lines)
    staging = OUT.with_name(OUT.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rc.write_json(staging / "normal_aware_results.json", result)
    with gzip.open(staging / "p10_external_predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for unit in sink:
            handle.write(json.dumps(unit) + "\n")
    (staging / "normal_aware_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    staging.rename(OUT)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
