#!/usr/bin/env python3
"""P12 R2': normal-aware industrial metrics of A and B on VisA / MTD (CPU, single-shot).

Reads the saved P11 external predictions; no model is run.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.normal_aware_ab
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10.external_normal_aware import METRIC_KEYS, OUT as P10_R2  # noqa: E402
from experiments.p10.industrial_metrics import mean_sd, normal_aware, per_thousand  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import P11_EXTERNAL, single_shot  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p12" / "normal_aware"
P11_PREDICTIONS = P11_EXTERNAL.parent / "p11_formal_external_predictions.jsonl"
NEW = (fa.SS, fa.NR, fa.A, fa.B, fa.A + "_m5", fa.B + "_m5")
COPIED = ("fused_xclass_clip18", "ours", "mvrec_rgb_adaptation")


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    with single_shot(OUT) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        units = [json.loads(line) for line in P11_PREDICTIONS.open(encoding="utf-8") if line.strip()]
        published = {(r["method"], r["task"], int(r["K"]), int(r["seed"])): r
                     for r in rc.load_json(P11_EXTERNAL)["raw_metrics"]}
        mismatches = []
        cells = {}
        for u in units:
            key = (u["method"], u["task"], int(u["K"]), int(u["seed"]))
            m = classification_metrics(u["true"], u["pred"], orders[u["task"]])
            if any(m[x] != published[key][x] for x in rc.METRICS):
                mismatches.append(key)
            cells[key] = normal_aware(u["true"], u["pred"], ee.NORMAL_CLASS[u["task"].split("__")[0]])
        if mismatches or len(units) != len(published):
            raise rc.P6BFailClosedError(f"predictions do not reproduce P11 metrics: {len(mismatches)} cells")
        p10 = rc.load_json(P10_R2 / "normal_aware_results.json")
        p10_cells = {(c["method"], c["task"], int(c["K"]), int(c["seed"])): c for c in p10["cells"]}
        diff = max(abs(cells[k][x] - p10_cells[k][x]) for k in cells if k[0] in (fa.SS, fa.NR) for x in METRIC_KEYS)
        if diff >= 1e-12:
            raise rc.P6BFailClosedError(f"SimpleShot / normref normal-aware cells differ from P10 R2: {diff}")

        summary = {}
        for m in NEW:
            summary[m] = {}
            for k in rc.K_VALUES:
                visa = {x: mean_sd(statistics.fmean(cells[(m, t, k, s)][x] for t in ee.VISA_TASKS) for s in rc.SEEDS)
                        for x in METRIC_KEYS}
                mtd = {x: mean_sd(cells[(m, ee.MTD_TASK, k, s)][x] for s in rc.SEEDS) for x in METRIC_KEYS}
                summary[m][str(k)] = {"visa_object_macro": visa, "mtd": mtd,
                                      "visa_per_1000_at_95pct_good": per_thousand(visa["false_reject_rate"]["mean"], visa["escape_rate"]["mean"]),
                                      "mtd_per_1000_at_95pct_good": per_thousand(mtd["false_reject_rate"]["mean"], mtd["escape_rate"]["mean"])}
        for m in COPIED:
            summary[m] = p10["summary"][m]
        result = {"schema_version": "p12_normal_aware_ab_v1",
                  "protocol": "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md#r2",
                  "provenance": provenance, "integrity": {"cells_rechecked": len(units), "p10_r2_max_abs_diff": diff},
                  "copied_from_p10_r2": list(COPIED), "summary": summary,
                  "cells": [{"method": k[0], "task": k[1], "K": k[2], "seed": k[3], **v} for k, v in cells.items()]}

        def f(v):
            return f"{100 * v['mean']:.1f}"
        lines = ["# P12 R2' — normal-aware metrics incl. A and B (argmax decisions, no threshold)", "",
                 f"Commit `{provenance['head']}`. {len(units)} P11 cells reproduced from predictions; "
                 f"SimpleShot / normref equal P10 R2 (max|Δ| = {diff:.1e}). Rows {', '.join(COPIED)} copied from P10 R2.", ""]
        for ds, key, per in (("VisA (object macro)", "visa_object_macro", "visa_per_1000_at_95pct_good"),
                             ("MTD", "mtd", "mtd_per_1000_at_95pct_good")):
            lines += [f"## {ds}", "",
                      "| Method | K | Good recall | Defect recall | Binary BA | Type acc. | False rejects / 1000 | Escapes / 1000 |",
                      "|---|---|---|---|---|---|---|---|"]
            for m in (fa.SS, "fused_xclass_clip18", fa.NR, fa.A, fa.B, "ours", "mvrec_rgb_adaptation"):
                for k in rc.K_VALUES:
                    s, p = summary[m][str(k)][key], summary[m][str(k)][per]
                    lines.append(f"| {m} | {k} | {f(s['good_recall'])} | {f(s['defect_recall'])} | {f(s['binary_ba'])} | "
                                 f"{f(s['defect_type_acc_detected'])} | {p['false_rejects_per_1000']:.0f} | {p['escapes_per_1000']:.0f} |")
            lines.append("")
        text = "\n".join(lines)
        rc.write_json(staging / "normal_aware_ab_results.json", result)
        (staging / "normal_aware_ab_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
