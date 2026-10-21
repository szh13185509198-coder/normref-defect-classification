#!/usr/bin/env python3
"""Seed standard deviations (ddof = 1 over the 5 paired seeds of the product / object macro BA) for Tables 5 and 8.

Reads finished result files only. Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p20.table_sd
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

C = PROJECT_ROOT / "ckpt"
OUT = PROJECT_ROOT / "docs" / "paper" / "table_sd"
MVTEC = [  # (row label, file, method)
    ("mask oracle (CLIP-18)", "p10_formal/results/formal_results.json", "oracle_maskpool_clip18"),
    ("dual-branch", "p14_formal/mvtec/p14_mvtec_results.json", "fused_adareg_regres_soft_q99"),
    ("dual-branch, M=5", "p14_formal/mvtec/p14_mvtec_results.json", "fused_adareg_regres_soft_q99_m5"),
    ("adaptive region", "p14_formal/mvtec/p14_mvtec_results.json", "fused_adareg_soft_q99"),
    ("crop-and-reencode", "p18/mvtec/p18_mvtec_results.json", "crop_fused"),
    ("fixed fraction", "p14_formal/mvtec/p14_mvtec_results.json", "fused_normref_clip18"),
    ("Liu et al. adaptation (residual mean, standalone)", "p10_formal/supplement_s/supplement_s_results.json", "residual_mean_clip18"),
    ("consensus", "p14_formal/mvtec/p14_mvtec_results.json", "fused_cons_soft_supq90"),
    ("cross-class", "p14_formal/mvtec/p14_mvtec_results.json", "fused_xclass_clip18"),
    ("DN4 adaptation", "p18/mvtec/p18_mvtec_results.json", "dn4_fused"),
    ("SimpleShot", "p14_formal/mvtec/p14_mvtec_results.json", "simpleshot_cl2n"),
    ("logistic regression (RFS)", "revision_c/c2_frozen_feature_baselines.json", "logreg_rfs"),
]
VISA = [
    ("dual-branch", "p14_formal/external/p14_external_results.json", "fused_adareg_regres_soft_q99"),
    ("adaptive region", "p14_formal/external/p14_external_results.json", "fused_adareg_soft_q99"),
    ("crop-and-reencode", "p18/external/p18_external_results.json", "crop_fused"),
    ("fixed fraction", "p14_formal/external/p14_external_results.json", "fused_normref_clip18"),
    ("consensus", "p14_formal/external/p14_external_results.json", "fused_cons_soft_supq90"),
    ("cross-class", "p14_formal/external/p14_external_results.json", "fused_xclass_clip18"),
    ("DN4 adaptation", "p18/external/p18_external_results.json", "dn4_fused"),
    ("SimpleShot", "p14_formal/external/p14_external_results.json", "simpleshot_cl2n"),
]


def rows_of(rel):
    d = json.loads((C / rel).read_text(encoding="utf-8"))
    return d["raw_metrics"] if "raw_metrics" in d else d["rows"]


def main() -> int:
    out = {"mvtec": {}, "visa": {}, "mtd": {}}
    for label, rel, m in MVTEC:
        s = rc.product_macro_summary(rc.metric_grid_from_rows(rows_of(rel)), m)
        out["mvtec"][label] = [(100 * s[str(k)]["balanced_accuracy"]["mean"], 100 * s[str(k)]["balanced_accuracy"]["std_ddof1"]) for k in rc.K_VALUES]
    # trainable adaptation / MVREC-RGB / Tip-Adapter-F* from the frozen formal grid (if present)
    frozen = rc.load_formal_metric_grid()
    from experiments.p10.formal_evaluate import mvrec_rows
    frozen = dict(frozen)
    frozen.update(rc.metric_grid_from_rows(mvrec_rows()))
    for label, m in (("trainable support-side adaptation", "ours"), ("MVREC-RGB adaptation", "mvrec_rgb"), ("Tip-Adapter-F*", "tip_adapter_f")):
        if m in frozen:
            s = rc.product_macro_summary(frozen, m)
            out["mvtec"][label] = [(100 * s[str(k)]["balanced_accuracy"]["mean"], 100 * s[str(k)]["balanced_accuracy"]["std_ddof1"]) for k in rc.K_VALUES]
        else:
            out["mvtec"][label] = f"not in the frozen grid (methods: {sorted(frozen)[:8]})"
    for label, rel, m in VISA:
        g = ee.grid_of(rows_of(rel))
        v = ee.visa_macro(g, m)
        out["visa"][label] = [(100 * v[str(k)]["mean"], 100 * v[str(k)]["sd"]) for k in rc.K_VALUES]
        out["mtd"][label] = [(100 * statistics.fmean(g[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS),
                              100 * statistics.stdev(g[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS)) for k in rc.K_VALUES]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_sd.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for ds, rows in out.items():
        print(f"== {ds}")
        for label, v in rows.items():
            print(f"{label:40s} " + (" | ".join(f"{m:.2f} ± {s:.2f}" for m, s in v) if isinstance(v, list) else v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
