#!/usr/bin/env python3
"""Addendum C1: paired product-cluster bootstrap for Group A contrasts.

Reads only per-unit metrics (formal audit ``raw_metrics`` and, if present, the C2
results file). Never opens images, predictions or query labels.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.revision.c1_paired_bootstrap
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.mvrec_rgb_formal.decision_evaluator import (  # noqa: E402
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, percentile_bootstrap,
)
from experiments.revision import revision_common as rc  # noqa: E402

OURS = "ours"
FORMAL_REFERENCES = ("fusion_proto", "p30_frozen_proto", "clip_proto", "dino_proto",
                     "linear_probe")
C2_REFERENCES = ("simpleshot_cl2n", "logreg_rfs")  # fusion_proto_local is C2-internal only
PRIMARY_REFERENCE = "fusion_proto"


def contrast(grid: dict, reference: str, metric: str, target: str = OURS) -> dict:
    ours, ref = grid[target], grid[reference]
    d = {k: {(p, s): ours[(p, k, s)][metric] - ref[(p, k, s)][metric]
             for p in rc.PRODUCTS for s in rc.SEEDS} for k in rc.K_VALUES}
    per_k = {}
    for k in rc.K_VALUES:
        product_means = {p: statistics.fmean(d[k][(p, s)] for s in rc.SEEDS)
                         for p in rc.PRODUCTS}
        seed_means = {s: statistics.fmean(d[k][(p, s)] for p in rc.PRODUCTS)
                      for s in rc.SEEDS}
        boot = percentile_bootstrap({p: [d[k][(p, s)] for s in rc.SEEDS]
                                     for p in rc.PRODUCTS})
        per_k[str(k)] = {
            "D": statistics.fmean(product_means.values()),
            "ci95": [boot["ci95_lower"], boot["ci95_upper"]],
            "ci_excludes_zero": boot["ci95_lower"] > 0 or boot["ci95_upper"] < 0,
            "positive_products": sum(v > 0 for v in product_means.values()),
            "negative_products": sum(v < 0 for v in product_means.values()),
            "median_product_difference": statistics.median(product_means.values()),
            "positive_seeds": sum(v > 0 for v in seed_means.values()),
            "per_product_mean_difference": product_means,
            "per_seed_macro_difference": {str(s): v for s, v in seed_means.items()},
        }
    product_all_k = {p: statistics.fmean(d[k][(p, s)] for k in rc.K_VALUES for s in rc.SEEDS)
                     for p in rc.PRODUCTS}
    boot_bar = percentile_bootstrap({p: [d[k][(p, s)] for k in rc.K_VALUES for s in rc.SEEDS]
                                     for p in rc.PRODUCTS})
    return {
        "per_K": per_k,
        "Dbar": statistics.fmean(per_k[str(k)]["D"] for k in rc.K_VALUES),
        "Dbar_ci95": [boot_bar["ci95_lower"], boot_bar["ci95_upper"]],
        "Dbar_ci_excludes_zero": boot_bar["ci95_lower"] > 0 or boot_bar["ci95_upper"] < 0,
        "Dbar_positive_products": sum(v > 0 for v in product_all_k.values()),
        "Dbar_per_product_mean_difference": product_all_k,
    }


def markdown_table(results: dict) -> str:
    lines = [
        "# C1 paired product-cluster bootstrap (post-hoc revision analysis)",
        "",
        "Ours minus reference, percentage points; 95% product-cluster percentile CI "
        f"({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}); positive products out of 13.",
        "",
    ]
    for metric in rc.METRICS:
        lines += [f"## {metric}", "",
                  "| Reference | K=1 | K=3 | K=5 | Mean over K |",
                  "|---|---|---|---|---|"]
        for reference, by_metric in results["contrasts"].items():
            c = by_metric[metric]
            cells = []
            for k in rc.K_VALUES:
                r = c["per_K"][str(k)]
                cells.append(f"{100 * r['D']:+.2f} [{100 * r['ci95'][0]:+.2f}, "
                             f"{100 * r['ci95'][1]:+.2f}] ({r['positive_products']}/13)")
            cells.append(f"{100 * c['Dbar']:+.2f} [{100 * c['Dbar_ci95'][0]:+.2f}, "
                         f"{100 * c['Dbar_ci95'][1]:+.2f}] ({c['Dbar_positive_products']}/13)")
            lines.append(f"| {reference} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c2-results", default=str(rc.C2_RESULTS),
                        help="C2 results JSON; skipped with a note if absent")
    parser.add_argument("--output", default=str(rc.C1_RESULTS))
    parser.add_argument("--table", default=str(rc.C1_TABLE))
    args = parser.parse_args(argv)

    grid = rc.load_formal_metric_grid()
    references = list(FORMAL_REFERENCES)
    c2_path = Path(args.c2_results)
    c2_note = "absent"
    if c2_path.is_file():
        c2 = rc.load_json(c2_path)
        if c2.get("fidelity_gate_passed") is not True:
            raise rc.P6BFailClosedError("C2 results exist but fidelity gate did not pass")
        for method, method_grid in rc.metric_grid_from_rows(c2["raw_metrics"]).items():
            if method not in C2_REFERENCES:
                continue
            if method in grid:
                raise rc.P6BFailClosedError(f"C2 method name collides with formal: {method}")
            grid[method] = method_grid
        references += [m for m in C2_REFERENCES if m in grid]
        c2_note = str(c2_path)
    for method in [OURS, *references]:
        rc.require_complete(grid, method)

    results = {
        "schema_version": "revision_c1_paired_bootstrap_v1",
        "addendum": "docs/P6B_REVISION_ADDENDUM_C.md#c1",
        "label": "post-hoc revision analysis; descriptive uncertainty, no decision gate",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "cluster": "product", "ci": "two-sided 95% percentile"},
        "primary_reference": PRIMARY_REFERENCE,
        "c2_source": c2_note,
        "provenance": rc.git_provenance(),
        "product_macro_summary": {m: rc.product_macro_summary(grid, m)
                                  for m in [OURS, *references]},
        "contrasts": {ref: {metric: contrast(grid, ref, metric) for metric in rc.METRICS}
                      for ref in references},
    }
    rc.write_json(Path(args.output), results)
    table = markdown_table(results)
    Path(args.table).write_text(table + "\n", encoding="utf-8", newline="\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
