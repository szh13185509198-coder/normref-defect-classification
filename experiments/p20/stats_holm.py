#!/usr/bin/env python3
"""Exact sign-flip permutation tests + Holm over the 10 pre-registered hypotheses (H1,H3,H4,H6,H7; E1,E3,E4,E6,E7).

Reads the P14 formal / external result files only (they contain every method involved, and equal
P10/P11 exactly). Per-product (MVTec-FS) or per-object (VisA) mean paired BA difference over K and
seeds; two-sided exact test over all sign patterns; Holm over the 10 hypotheses.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p20.stats_holm
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

SS, NR, XC, A, ARR, CONS = ("simpleshot_cl2n", "fused_normref_clip18", "fused_xclass_clip18",
                            "fused_adareg_soft_q99", "fused_adareg_regres_soft_q99", "fused_cons_soft_supq90")
HYP = {"H1": ("mvtec", NR, SS), "H3": ("mvtec", XC, SS), "H4": ("mvtec", A, NR), "H6": ("mvtec", ARR, A), "H7": ("mvtec", CONS, XC),
       "E1": ("visa", NR, SS), "E3": ("visa", XC, SS), "E4": ("visa", A, NR), "E6": ("visa", ARR, A), "E7": ("visa", CONS, XC)}
DESCRIPTIVE = {"ARR−SS (MVTec-FS)": ("mvtec", ARR, SS), "ARR−SS (VisA)": ("visa", ARR, SS)}
OUT = PROJECT_ROOT / "docs" / "paper" / "stats_holm"


def exact_signflip(d: list[float]) -> float:
    obs = abs(statistics.fmean(d))
    hit = sum(abs(statistics.fmean(s * x for s, x in zip(signs, d))) >= obs - 1e-15
              for signs in itertools.product((1, -1), repeat=len(d)))
    return hit / 2 ** len(d)


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[k]))
        out[k] = running
    return out


def per_unit(grids, ds, t, r) -> dict:
    if ds == "mvtec":
        c = contrast(grids["mvtec"], r, "balanced_accuracy", target=t)
        return {k: 100 * v for k, v in c["Dbar_per_product_mean_difference"].items()}, 100 * c["Dbar"], [100 * x for x in c["Dbar_ci95"]]
    c = ee.visa_contrast(grids["visa"], t, r, "balanced_accuracy")
    objs = c["per_K"]["1"]["per_object"].keys()
    per = {o: 100 * statistics.fmean(c["per_K"][str(k)]["per_object"][o] for k in rc.K_VALUES) for o in objs}
    return per, 100 * c["Dbar"], [100 * x for x in c["Dbar_ci95"]]


def main() -> int:
    grids = {"mvtec": rc.metric_grid_from_rows(rc.load_json(PROJECT_ROOT / "ckpt/p14_formal/mvtec/p14_mvtec_results.json")["raw_metrics"]),
             "visa": ee.grid_of(rc.load_json(PROJECT_ROOT / "ckpt/p14_formal/external/p14_external_results.json")["raw_metrics"])}
    res = {}
    for name, (ds, t, r) in {**HYP, **DESCRIPTIVE}.items():
        per, d, ci = per_unit(grids, ds, t, r)
        res[name] = {"dataset": ds, "target": t, "reference": r, "delta_pp": d, "ci95_pp": ci,
                     "positive": sum(v > 0 for v in per.values()), "n_units": len(per),
                     "signflip_p": exact_signflip(list(per.values())), "per_unit_pp": per}
    adj = holm({h: res[h]["signflip_p"] for h in HYP})
    for h in HYP:
        res[h]["holm_p"] = adj[h]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats_holm.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| Hypothesis | Δ pooled [95% CI] | +units | exact sign-flip p | Holm p |", "|---|---|---|---|---|"]
    for name, v in res.items():
        lines.append(f"| {name} | {v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | {v['positive']}/{v['n_units']} | "
                     f"{v['signflip_p']:.4f} | {v.get('holm_p', float('nan')):.4f} |")
    units = sorted(next(iter(v for v in res.values() if v["dataset"] == "mvtec"))["per_unit_pp"])
    lines += ["", "Per-product differences (MVTec-FS, pp):", "| Hypothesis | " + " | ".join(units) + " |", "|---|" + "---|" * len(units)]
    lines += [f"| {n} | " + " | ".join(f"{v['per_unit_pp'][u]:+.1f}" for u in units) + " |" for n, v in res.items() if v["dataset"] == "mvtec"]
    objs = sorted(next(iter(v for v in res.values() if v["dataset"] == "visa"))["per_unit_pp"])
    lines += ["", "Per-object differences (VisA, pp):", "| Hypothesis | " + " | ".join(o.replace("visa__", "") for o in objs) + " |", "|---|" + "---|" * len(objs)]
    lines += [f"| {n} | " + " | ".join(f"{v['per_unit_pp'][o]:+.1f}" for o in objs) + " |" for n, v in res.items() if v["dataset"] == "visa"]
    (OUT / "stats_holm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
