#!/usr/bin/env python3
"""P47 U3 follow-up (offline, CPU): uncertainty of the paired gain over SimpleShot, and the split of the
single-deployment SD into the deployment part (support and bank draw) and the test-sample part.

Input: ckpt/p47/u3/u3_cells.jsonl and u3_run.json, ckpt/p47/u3_analysis/u3_budget.json (the U3 outputs).
Method: the U3 functions of experiments/p47/common.py. Paired gain per cell d[i, j] = BA_ARR[i, j] - BA_SS[i];
two-way random-effects ANOVA on d; query bootstrap with the U3 key p47q|{product}|{K}, so both methods use the same
resamples; the Type B implementation term of the method itself (from u3_budget.json) is kept, conservatively.
Output (write-once): ckpt/p47/u3_paired/u3_paired.json and .md.

    python -m experiments.p47.u3_paired_gain
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from experiments.p47 import common as cm  # noqa: E402

SRC = PROJECT_ROOT / "ckpt" / "p47"
OUT = SRC / "u3_paired"
REPS = 1000


def inf(x):
    return math.inf if x is None else x


def paired_gain(cells, run, half_width_pp):
    ARR, SS = run["methods"]["ARR"], run["methods"]["SimpleShot"]
    seeds = run["design"]["cross"]["S"]
    by = {(c["product"], c["design"], c["M"], c["B"], c["S"], c["K"], c["method"]): np.asarray(c["pred"]) for c in cells}
    prods = list(run["products"])
    P, ns = len(prods), len(seeds)
    out = {}
    for k in run["design"]["cross"]["K"]:
        comps = {"S": [], "B": [], "SB": [], "Q": []}
        gains = {}
        for p in prods:
            meta = run["products"][p]
            order = meta["class_order"]
            t = np.array([order.index(x) for x in meta["true"]])
            n = len(order)
            arr = np.stack([by[(p, "cross", 20, b, s, k, ARR)] for s in seeds for b in seeds])
            ss = np.stack([by[(p, "cross", 20, seeds[0], s, k, SS)] for s in seeds])
            ya = np.array([cm.balanced_accuracy(t, x, n) for x in arr]).reshape(ns, ns)
            ys = np.array([cm.balanced_accuracy(t, x, n) for x in ss])
            d = ya - ys[:, None]
            av = cm.two_way_random(d)
            boot = cm.stratified_bootstrap_ba(t, np.concatenate([arr, ss]), n, REPS, f"p47q|{p}|{k}")
            ba, bs = boot[:, :ns * ns].reshape(REPS, ns, ns), boot[:, ns * ns:]
            diag = np.stack([ba[:, i, i] - bs[:, i] for i in range(ns)], 1).mean(1)
            comps["S"].append((av["var_S"] / ns / P ** 2, av["nu_S"]))
            comps["B"].append((av["var_B"] / ns / P ** 2, av["nu_B"]))
            comps["SB"].append((av["var_SB"] / ns / P ** 2, av["nu_SB"]))
            comps["Q"].append((float(diag.var(ddof=1)) / P ** 2, len(t) - 1.0))
            gains[p] = float(np.mean(np.diag(d)))
        u2 = {c: sum(v for v, _ in comps[c]) for c in comps}
        nu = {c: cm.welch_satterthwaite(comps[c]) for c in comps}
        u2["I"], nu["I"] = (half_width_pp / 100) ** 2 / 3, math.inf
        uc2 = sum(u2.values())
        nu_eff = cm.welch_satterthwaite([(u2[c], nu[c]) for c in u2])
        kf = cm.t975(nu_eff)
        out[str(k)] = {"gain": statistics.fmean(gains.values()), "u_c": math.sqrt(uc2), "nu_eff": nu_eff, "k": kf,
                       "U95": kf * math.sqrt(uc2), "shares": {c: u2[c] / uc2 for c in u2},
                       "positive_products": sum(g > 0 for g in gains.values()), "per_product_gain": gains}
    return out


def deployment_split(budget):
    out = {}
    for k, r in budget["per_K"].items():
        rows, all_, dep = {}, [], []
        for p, v in r["anova_per_product"].items():
            parts = [(v["var_S"], inf(v["nu_S"])), (v["var_B"], inf(v["nu_B"])), (v["var_SB"], inf(v["nu_SB"]))]
            vd = sum(x for x, _ in parts)
            va = vd + v["var_Q_single"]
            td = cm.t975(cm.welch_satterthwaite(parts))
            ta = cm.t975(cm.welch_satterthwaite(parts + [(v["var_Q_single"], v["nu_Q"])]))
            m = v["mean"]
            rows[p] = {"mean": m, "sd_deployment": math.sqrt(vd), "sd_with_test_sample": math.sqrt(va),
                       "test_sample_share": v["var_Q_single"] / va,
                       "pi95_deployment": [max(0.0, m - td * math.sqrt(vd)), min(1.0, m + td * math.sqrt(vd))],
                       "pi95_with_test_sample": [max(0.0, m - ta * math.sqrt(va)), min(1.0, m + ta * math.sqrt(va))]}
            all_.append(va)
            dep.append(vd)
        out[k] = {"pooled_sd_deployment": math.sqrt(statistics.fmean(dep)),
                  "pooled_sd_with_test_sample": math.sqrt(statistics.fmean(all_)), "per_product": rows}
    return out


def main() -> int:
    from experiments.revision import revision_common as rc
    run = rc.load_json(SRC / "u3" / "u3_run.json")
    cells = [json.loads(x) for x in (SRC / "u3" / "u3_cells.jsonl").open(encoding="utf-8")]
    budget = rc.load_json(SRC / "u3_analysis" / "u3_budget.json")
    half = budget["per_K"]["5"]["benchmark_mean"]["components"]["implementation I (Type B, rectangular)"]["half_width_pp"]
    with cm.single_shot(OUT) as staging:
        res = {"schema_version": "p47_u3_paired_v1", "provenance": rc.git_provenance(),
               "inputs": ["ckpt/p47/u3/u3_cells.jsonl", "ckpt/p47/u3/u3_run.json", "ckpt/p47/u3_analysis/u3_budget.json"],
               "paired_gain_over_simpleshot": paired_gain(cells, run, half), "deployment_split": deployment_split(budget)}
        rc.write_json(staging / "u3_paired.json", json.loads(json.dumps(res).replace("Infinity", "null")))
        lines = ["# P47 U3 follow-up: paired gain over SimpleShot and the single-deployment SD", "",
                 "| K | Gain (pp) | u_c (pp) | U95 (pp) | k | nu_eff | Shares S / B / SxB / Q / I | Products > 0 |",
                 "|---|---|---|---|---|---|---|---|"]
        for k, g in res["paired_gain_over_simpleshot"].items():
            sh = " / ".join(f"{100 * g['shares'][c]:.0f} %" for c in ("S", "B", "SB", "Q", "I"))
            lines.append(f"| {k} | {100 * g['gain']:+.2f} | {100 * g['u_c']:.2f} | {100 * g['U95']:.2f} | {g['k']:.2f} | "
                         f"{g['nu_eff']:.0f} | {sh} | {g['positive_products']}/13 |")
        lines += ["", "| K | Pooled SD, support and bank (pp) | Pooled SD, with test sample (pp) |", "|---|---|---|"]
        for k, v in res["deployment_split"].items():
            lines.append(f"| {k} | {100 * v['pooled_sd_deployment']:.2f} | {100 * v['pooled_sd_with_test_sample']:.2f} |")
        lines += ["", "K = 5, per product: mean, SD (support and bank / with test sample), 95 % prediction intervals", "",
                  "| Product | Mean | SD | SD with test sample | PI95 | PI95 with test sample |", "|---|---|---|---|---|---|"]
        for p, v in sorted(res["deployment_split"]["5"]["per_product"].items(), key=lambda x: x[1]["mean"]):
            lines.append(f"| {p} | {100 * v['mean']:.2f} | {100 * v['sd_deployment']:.2f} | {100 * v['sd_with_test_sample']:.2f} | "
                         f"[{100 * v['pi95_deployment'][0]:.1f}, {100 * v['pi95_deployment'][1]:.1f}] | "
                         f"[{100 * v['pi95_with_test_sample'][0]:.1f}, {100 * v['pi95_with_test_sample'][1]:.1f}] |")
        text = "\n".join(lines)
        (staging / "u3_paired.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
