#!/usr/bin/env python3
"""P47 U1-D: calibration uncertainty of the false-reject rate on the deployed engine's own scores (offline).

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U1-D". Input: the P46 device score files
(ckpt/p46/jetson_orin_nano/p46_<task>.json: per seed, the scores of the 100 P15 calibration normals and of all
queries). Per task and seed, the calibration normals and the normal queries form one pool of good parts; a calibration
set of size N is drawn from the pool (R = 1000, random.Random(f"p47calD|{task}|{seed}|{N}|{r}")) and the rest of the
pool is the test set. Rule (as P15 / P46): k = ceil((N + 1)(1 - alpha)), threshold t = k-th smallest calibration score
(infinite if k > N), a part is rejected when s > t. Escapes use the defect queries.

Rules: (a) marginal (P15); (b) PAC with delta in {0.05, 0.10}: smallest k with P(Beta(N - k + 1, k) > alpha) <= delta,
reported as infeasible when no k <= N qualifies.

Theory: the false-reject rate realised by one calibration set is Beta(N - k + 1, k) for continuous exchangeable scores;
on m test normals the count of false rejects is beta-binomial(m, N - k + 1, k). Device scores are FP16, so ties occur;
with the strict rule s > t ties lower the realised rate slightly. The tie share is reported.

Integrity: the original P15 calibration draw (all 100 calibration normals, tested on the normal queries) reproduces the
P46 unit false-reject rates exactly; otherwise nothing is written.

    python -m experiments.p47.u1d            # writes ckpt/p47/u1d/ (write-once)
    python -m experiments.p47.u1d --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "ckpt" / "p46" / "jetson_orin_nano"
OUT = PROJECT_ROOT / "ckpt" / "p47" / "u1d"
NS = (20, 50, 100)
ALPHAS = (0.01, 0.02, 0.05, 0.10)
DELTAS = (0.05, 0.10)
R = 1000


def k_marginal(n: int, alpha: float) -> int:
    return math.ceil((n + 1) * (1 - alpha) - 1e-12)


def k_pac(n: int, alpha: float, delta: float) -> int | None:
    for k in range(1, n + 1):
        if stats.beta(n - k + 1, k).sf(alpha) <= delta:
            return k
    return None


def threshold(cal_sorted: list[float], k: int | None) -> float:
    return math.inf if k is None or k > len(cal_sorted) else cal_sorted[k - 1]


def p46_integrity(units: list[dict], files: dict) -> int:
    checked = 0
    for u in units:
        d = files[u["task"]]
        v = d["seeds"][u["seed"]]
        q, normal = v["queries"], d["query_is_normal"]
        cal = sorted(v["calibration"])
        for a in ("0.01", "0.02", "0.05"):
            t = threshold(cal, k_marginal(len(cal), float(a)))
            frr = sum(s > t for s, n in zip(q, normal) if n) / sum(normal)
            if abs(frr - u[f"frr_{a}"]) > 1e-12:
                raise SystemExit(f"integrity: {u['task']}/{u['seed']} alpha {a}: {frr} != {u[f'frr_{a}']}")
            checked += 1
    return checked


def run_unit(task: str, seed: str, cal: list[float], q: list[float], normal: list[bool]) -> dict:
    pool = list(cal) + [s for s, n in zip(q, normal) if n]
    defects = [s for s, n in zip(q, normal) if not n]
    out = {"pool": len(pool), "defects": len(defects), "tie_share_pool": 1 - len(set(pool)) / len(pool), "cells": {}}
    for n_cal in NS:
        if len(pool) < 2 * n_cal:
            continue
        rules = {f"marginal|{a}": k_marginal(n_cal, a) for a in ALPHAS}
        rules.update({f"pac{d}|{a}": k_pac(n_cal, a, d) for a in ALPHAS for d in DELTAS})
        frr = {r: [] for r in rules}
        esc = {r: [] for r in rules}
        m = len(pool) - n_cal
        for rep in range(R):
            rng = random.Random(f"p47calD|{task}|{seed}|{n_cal}|{rep}")
            idx = set(rng.sample(range(len(pool)), n_cal))
            c = sorted(pool[i] for i in idx)
            test = np.array([pool[i] for i in range(len(pool)) if i not in idx])
            dfx = np.array(defects)
            for r, k in rules.items():
                if k is None or k > n_cal:
                    continue
                t = c[k - 1]
                frr[r].append(float((test > t).mean()))
                esc[r].append(float((dfx <= t).mean()) if len(dfx) else float("nan"))
        cells = {}
        for r, k in rules.items():
            if not frr[r]:
                cells[r] = {"k": k, "feasible": False}
                continue
            f = np.array(frr[r])
            cells[r] = {"k": k, "feasible": True, "m": m, "frr": frr[r], "escape_mean": float(np.nanmean(esc[r])),
                        "mean": float(f.mean()), "sd": float(f.std(ddof=1)),
                        "q05": float(np.quantile(f, 0.05)), "q50": float(np.quantile(f, 0.5)),
                        "q95": float(np.quantile(f, 0.95)),
                        "p_exceed": float((f > float(r.split("|")[1])).mean())}
        out["cells"][str(n_cal)] = cells
    return out


def theory(n_cal: int, k: int, m: int, alpha: float) -> dict:
    a_, b_ = n_cal - k + 1, k
    bb = stats.betabinom(m, a_, b_)
    x = np.arange(m + 1)
    pmf = bb.pmf(x)
    cdf = np.cumsum(pmf)
    q = lambda p: float(x[np.searchsorted(cdf, p)] / m)  # noqa: E731
    return {"mean": float(bb.mean() / m), "sd": float(bb.std() / m), "q05": q(0.05), "q50": q(0.5), "q95": q(0.95),
            "p_exceed": float(pmf[x / m > alpha].sum()), "beta_mean": a_ / (a_ + b_)}


def summarise(units: dict) -> dict:
    """Pooled over tasks and seeds per (N, rule): empirical vs theoretical (mixture over units of beta-binomials with
    each unit's m), randomized-PIT KS test, and the bank / calibration variance decomposition."""
    res = {}
    keys = {(n, r) for u in units.values() for n, cells in u["cells"].items() for r, c in cells.items() if c["feasible"]}
    rng = np.random.default_rng(47)
    for n, r in sorted(keys, key=lambda t: (int(t[0]), t[1])):
        alpha = float(r.split("|")[1])
        emp, pit, th, per_unit_means, per_unit_vars = [], [], [], {}, []
        for (task, seed), u in units.items():
            c = u["cells"].get(n, {}).get(r)
            if not c or not c["feasible"]:
                continue
            m, k = c["m"], c["k"]
            f = np.array(c["frr"])
            emp.extend(f.tolist())
            bb = stats.betabinom(m, int(n) - k + 1, k)
            cnt = np.round(f * m).astype(int)
            lo, hi = bb.cdf(cnt - 1), bb.cdf(cnt)
            pit.extend((lo + rng.uniform(size=len(cnt)) * (hi - lo)).tolist())
            th.append(theory(int(n), k, m, alpha))
            per_unit_means.setdefault(task, []).append(float(f.mean()))
            per_unit_vars.append(float(f.var(ddof=1)))
        e = np.array(emp)
        ks = stats.kstest(pit, "uniform")
        # variance decomposition: total = within (calibration draw, given bank) + between banks (seed means per task)
        within = float(np.mean(per_unit_vars))
        between = float(np.mean([np.var(v, ddof=1) for v in per_unit_means.values() if len(v) > 1]))
        res[f"{n}|{r}"] = {
            "N": int(n), "rule": r.split("|")[0], "alpha": alpha, "units": len(th), "draws": int(e.size),
            "empirical": {"mean": float(e.mean()), "sd": float(e.std(ddof=1)), "q05": float(np.quantile(e, 0.05)),
                          "q50": float(np.quantile(e, 0.5)), "q95": float(np.quantile(e, 0.95)),
                          "p_exceed": float((e > alpha).mean())},
            "theory_mean_over_units": {k_: float(np.mean([t[k_] for t in th])) for k_ in
                                       ("mean", "sd", "q05", "q50", "q95", "p_exceed", "beta_mean")},
            "pit_ks": {"D": float(ks.statistic), "p": float(ks.pvalue),
                       "note": "draws from one pool are not independent; p is indicative"},
            "variance": {"within_calibration": within, "between_banks": between},
        }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        rng = np.random.default_rng(0)
        cal = rng.normal(size=100).tolist()
        q = rng.normal(size=500).tolist() + (rng.normal(size=50) + 3).tolist()
        normal = [True] * 500 + [False] * 50
        u = run_unit("t", "1", cal, q, normal)
        c = u["cells"]["100"]["marginal|0.05"]
        th = theory(100, c["k"], c["m"], 0.05)
        assert abs(c["mean"] - th["mean"]) < 0.005, (c["mean"], th["mean"])
        assert u["cells"]["20"]["marginal|0.01"]["feasible"] is False
        print(f"[p47-u1d] selftest ok (mean FRR {c['mean']:.4f} vs theory {th['mean']:.4f})")
        return 0
    if (OUT / "p47_u1d_results.json").exists():
        raise SystemExit("ckpt/p47/u1d exists (write-once)")
    p46 = json.loads((SRC / "p46_results.json").read_text(encoding="utf-8"))
    files = {}
    for f in sorted(SRC.glob("p46_*__*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        files[d["task"]] = d
    checked = p46_integrity(p46["units"], files)
    units = {}
    for task, d in files.items():
        for seed, v in d["seeds"].items():
            units[(task, seed)] = run_unit(task, seed, v["calibration"], v["queries"], d["query_is_normal"])
        print(f"[p47-u1d] {task}: pool {units[(task, seed)]['pool']}", flush=True)
    summary = summarise(units)
    OUT.mkdir(parents=True, exist_ok=True)
    slim = {f"{t}|{s}": {**{k: v for k, v in u.items() if k != "cells"},
                          "cells": {n: {r: {k: v for k, v in c.items() if k != "frr"} for r, c in cells.items()}
                                    for n, cells in u["cells"].items()}} for (t, s), u in units.items()}
    res = {"schema_version": "p47_u1d_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u1-d",
           "label": "report only; protocol state at run time: draft, U1-D section unchanged since commit 36c2f30",
           "source": "ckpt/p46/jetson_orin_nano (device scores, FP16 engine)", "integrity_p46_cells_reproduced": checked,
           "R": R, "Ns": NS, "alphas": ALPHAS, "deltas": DELTAS, "summary": summary, "units": slim}
    (OUT / "p47_u1d_results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    draws = {f"{t}|{s}|{n}|{r}": c["frr"] for (t, s), u in units.items() for n, cells in u["cells"].items()
             for r, c in cells.items() if c.get("feasible") and r.startswith("marginal")}
    (OUT / "p47_u1d_draws_marginal.json").write_text(json.dumps(draws) + "\n", encoding="utf-8")
    lines = ["# P47 U1-D: false-reject rate of one calibration, deployed engine scores", "",
             f"Integrity: {checked} P46 unit false-reject rates reproduced. R = {R} draws per task × seed × N.", "",
             "| N | rule | α | units | empirical mean | theory mean | emp. 5–95 % | theory 5–95 % | emp. P(>α) | "
             "theory P(>α) | KS D (p) | var within / between banks |", "|" + "---|" * 12]
    for key, s_ in summary.items():
        e, t = s_["empirical"], s_["theory_mean_over_units"]
        lines.append(f"| {s_['N']} | {s_['rule']} | {s_['alpha']:.0%} | {s_['units']} | {e['mean']:.2%} | "
                     f"{t['mean']:.2%} | {e['q05']:.2%}–{e['q95']:.2%} | {t['q05']:.2%}–{t['q95']:.2%} | "
                     f"{e['p_exceed']:.2f} | {t['p_exceed']:.2f} | {s_['pit_ks']['D']:.3f} ({s_['pit_ks']['p']:.2g}) | "
                     f"{s_['variance']['within_calibration']:.2e} / {s_['variance']['between_banks']:.2e} |")
    (OUT / "p47_u1d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
