#!/usr/bin/env python3
"""Supplement S10: per-1,000-part outcomes of the inspection schemes at other good-part rates.

Only re-weights per-class rates that the finished runs already stored (P14 argmax recalls,
P15 per-unit screening rates); no prediction is recomputed. VisA = object macro at K=5 (mean over
the 55 object x seed units); MTD at K=5 (mean over seeds).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p20.yield_conversion
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "docs" / "paper" / "yield_conversion"
RATES = (0.90, 0.95, 0.99, 0.999)
ARR = "fused_adareg_regres_soft_q99"
DIRECT = {"VisA": {"good_recall": 0.857, "defect_recall": 0.908},  # Table 9 (P14 normal-aware, ARR argmax, K=5)
          "MTD": {"good_recall": 0.371, "defect_recall": 0.927}}


def per_1000(pi, fr_rate, esc_rate, review_good=0.0, review_defect=0.0):
    return (1000 * pi * fr_rate, 1000 * (1 - pi) * esc_rate, 1000 * (pi * review_good + (1 - pi) * review_defect))


def main() -> int:
    d = json.loads((PROJECT_ROOT / "ckpt/p15/external/p15_external_results.json").read_text(encoding="utf-8"))
    rows = []
    for ds in ("VisA", "MTD"):
        units = [u for u in d["units"] if u["K"] == 5 and (u["task"].startswith("visa__") == (ds == "VisA"))]
        mean = lambda f: statistics.fmean(f(u) for u in units)  # noqa: E731
        schemes = {
            "direct argmax (dual-branch)": (1 - DIRECT[ds]["good_recall"], 1 - DIRECT[ds]["defect_recall"], 0.0, 0.0),
            "two-stage, α = 5 %": (mean(lambda u: u["two_level"][ARR]["0.05"]["false_reject_rate"]),
                                   mean(lambda u: u["two_level"][ARR]["0.05"]["escape_rate"]), 0.0, 0.0),
            **{f"three-way (reject α = 1 %, pass α = {a} %)": (
                mean(lambda u, k=k: u["three_way"][k]["auto_false_reject_rate"]),
                mean(lambda u, k=k: u["three_way"][k]["escape_rate"]),
                mean(lambda u, k=k: u["three_way"][k]["review_rate_good"]),
                mean(lambda u, k=k: u["three_way"][k]["review_rate_defect"])) for a, k in ((10, "0.1"), (30, "0.3"))},
        }
        for name, rates in schemes.items():
            for pi in RATES:
                fr, esc, rev = per_1000(pi, *rates)
                rows.append({"dataset": ds, "scheme": name, "good_rate": pi, "false_rejects": fr, "escapes": esc, "review": rev})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yield_conversion.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| Dataset | Scheme | good 90 % | good 95 % | good 99 % | good 99.9 % |", "|---|---|---|---|---|---|"]
    for ds in ("VisA", "MTD"):
        for name in dict.fromkeys(r["scheme"] for r in rows):
            cells = [r for r in rows if r["dataset"] == ds and r["scheme"] == name]
            lines.append(f"| {ds} | {name} | " + " | ".join(
                f"{c['false_rejects']:.0f} / {c['escapes']:.1f} / {c['review']:.0f}" for c in cells) + " |")
    (OUT / "yield_conversion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
