#!/usr/bin/env python3
"""P47 U2-A: decision risks with the calibration component from the order-statistic law (amendment to U2).

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U2-A". Reads the U2 / U5 extraction (ckpt/p47/u25_external/),
writes ckpt/p47/u2a/ (write-once). CPU only.

    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u2a
    python -m experiments.p47.u2a --selftest
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from experiments.p47 import common as cm  # noqa: E402
from experiments.p47 import u2u5 as u  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p47" / "u2a"
ALL_ALPHAS = tuple(sorted({*u.ALPHAS_2, u.ALPHA_REJECT, *u.ALPHA_PASS}))


def k_of(a: float, n_cal: int) -> int:
    return math.ceil((n_cal + 1) * (1 - a))


def observed_taus(cal: np.ndarray, reps: int) -> dict:
    s = np.sort(cal)
    return {a: np.full(reps, s[k_of(a, len(s)) - 1] if k_of(a, len(s)) <= len(s) else np.inf) for a in ALL_ALPHAS}


def beta_taus(pooled: np.ndarray, n_cal: int, reps: int, rng) -> dict:
    """tau_a = Q(U_(k_a)) with one sorted sample of n_cal uniforms per replicate (joint order statistics)."""
    uo = np.sort(rng.random((reps, n_cal)), axis=1)
    return {a: (np.quantile(pooled, uo[:, k_of(a, n_cal) - 1]) if k_of(a, n_cal) <= n_cal else np.full(reps, np.inf))
            for a in ALL_ALPHAS}


def rates_from_taus(taus: dict, qg: np.ndarray, qb: np.ndarray) -> dict:
    """As u2u5.unit_rates, with the thresholds given per replicate. qg / qb [R, n]."""
    out = {}
    for a in u.ALPHAS_2:
        t = taus[a][:, None]
        out[f"two|{a}"] = {"frr": (qg > t).mean(1), "escape": (qb <= t).mean(1)}
    t_rej = taus[u.ALPHA_REJECT][:, None]
    for a in u.ALPHA_PASS:
        t_pass = taus[a][:, None]
        out[f"three|{a}"] = {"frr": (qg > t_rej).mean(1), "escape": (qb <= t_pass).mean(1),
                             "review_good": ((qg > t_pass) & (qg <= t_rej)).mean(1),
                             "review_bad": ((qb > t_pass) & (qb <= t_rej)).mean(1)}
    return out


def mc_rates(units: dict, tasks: list, seeds: list, mode: str, rng, reps: int, cluster: bool) -> dict:
    """mode in {'combined', 'cal', 'test'}; units[(task, seed)] = (cal, qg, qb)."""
    acc = {}
    pick = (rng.integers(0, len(tasks), (reps, len(tasks))) if mode in ("combined", "test")
            else np.tile(np.arange(len(tasks)), (reps, 1))) if cluster else None
    for ti, task in enumerate(tasks):
        for seed in seeds:
            cal, qg, qb = units[(task, seed)]
            taus = beta_taus(np.concatenate([cal, qg]), len(cal), reps, rng) if mode in ("combined", "cal") \
                else observed_taus(cal, reps)
            if not cluster and mode in ("combined", "test"):
                g = qg[rng.integers(0, len(qg), (reps, len(qg)))]
                b = qb[rng.integers(0, len(qb), (reps, len(qb)))]
            else:
                g, b = np.broadcast_to(qg, (reps, len(qg))), np.broadcast_to(qb, (reps, len(qb)))
            for rule, d in rates_from_taus(taus, g, b).items():
                for key, v in d.items():
                    acc.setdefault(rule, {}).setdefault(key, np.zeros((reps, len(tasks))))[:, ti] += v / len(seeds)
    return {rule: {key: (np.take_along_axis(m, pick, axis=1).mean(1) if cluster else m.mean(1)) for key, m in d.items()}
            for rule, d in acc.items()}


def run() -> int:
    import torch
    from scipy import stats
    from experiments.p10 import external_evaluate as ee
    from experiments.revision import revision_common as rc
    src = u.OUT["u25_external"]
    if not (src / "u25_external_info.json").is_file():
        raise cm.P47Error("the U2 / U5 extraction is missing")
    u2 = rc.load_json(u.OUT["u2"] / "u2_risks.json")
    with cm.single_shot(OUT) as staging:
        units = {}
        for task in ee.TASKS:
            d = torch.load(src / f"{task}.pt", map_location="cpu", weights_only=False)
            good = np.asarray([t == d["normal"] for t in d["q_true"]])
            for seed in rc.SEEDS:
                s = np.asarray(d["seeds"][seed]["s_q"], dtype=float)
                units[(task, seed)] = (np.asarray(d["seeds"][seed]["s_cal"], dtype=float), s[good], s[~good])
        rng = cm.np_rng("p47u2a")
        res = {"schema_version": "p47_u2a_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u2-a",
               "provenance": rc.git_provenance(), "pi": list(u.PI), "R": u.R_MC, "groups": {}}
        for name, tasks, cluster in (("visa_object_macro", list(ee.VISA_TASKS), True), ("mtd", [ee.MTD_TASK], False)):
            point = u2["groups"][name]["rates"]          # the actual P15 calibration draw (checked against P15 in U2)
            mc = {mode: mc_rates(units, tasks, list(rc.SEEDS), mode, rng, u.R_MC, cluster) for mode in ("combined", "cal", "test")}
            spread = {}
            n_cal = len(units[(tasks[0], rc.SEEDS[0])][0])
            for a in u.ALPHAS_2:
                k = k_of(a, n_cal)
                sds = []
                for task in tasks:
                    for seed in rc.SEEDS:
                        cal, qg, qb = units[(task, seed)]
                        taus = beta_taus(np.concatenate([cal, qg]), n_cal, 2000, rng)
                        sds.append(float(rates_from_taus(taus, np.broadcast_to(qg, (2000, len(qg))),
                                                         np.broadcast_to(qb, (2000, len(qb))))[f"two|{a}"]["frr"].std()))
                spread[str(a)] = {"cal_only_sd_unit_mean": statistics.fmean(sds),
                                  "beta_sd": float(stats.beta(n_cal - k + 1, k).std()),
                                  "u2_bootstrap_sd": u2["groups"][name]["calibration_spread_vs_beta"][str(a)]["bootstrap_sd_unit_mean"]}
            res["groups"][name] = {"tasks": tasks, "risks": u.summarize_u2(point, mc), "rates": point,
                                   "calibration_spread": spread}
        rc.write_json(staging / "u2a_risks.json", res)
        text = table(res, u2)
        (staging / "u2a_risks.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def table(res: dict, u2: dict) -> str:
    def f(e, k="ci95_combined"):
        return f"{1000 * e['value']:.1f} [{1000 * e[k][0]:.1f}, {1000 * e[k][1]:.1f}]"
    lines = ["# P47 U2-A - decision risks, calibration component from the order-statistic law", "",
             "Per 1000 parts; 95 % interval (Monte Carlo: calibration law + test sample). U2 column: the same cell with",
             "the with-replacement calibration bootstrap (fails at the extreme quantile; kept for comparison).", ""]
    for name, g in res["groups"].items():
        lines += [f"## {name}", "", "| Rule | π | R_P (false rejects) | R_C (escapes) | Review | U2: R_P |",
                  "|---|---|---|---|---|---|"]
        for rule, by_pi in g["risks"].items():
            for pi, row in by_pi.items():
                rev = f(row["review"]) if "review" in row else "—"
                old = f(u2["groups"][name]["risks"][rule][pi]["R_P"])
                lines.append(f"| {rule} | {pi} | {f(row['R_P'])} | {f(row['R_C'])} | {rev} | {old} |")
        lines += ["", "FRR spread from calibration alone (unit mean SD): " + ", ".join(
            f"α={a}: U2-A {100 * v['cal_only_sd_unit_mean']:.2f}, Beta {100 * v['beta_sd']:.2f}, U2 {100 * v['u2_bootstrap_sd']:.2f} pp"
            for a, v in g["calibration_spread"].items()), ""]
    return "\n".join(lines)


def selftest() -> int:
    from scipy import stats
    rng = np.random.default_rng(5)
    n_cal = 100
    cal, qg, qb = rng.normal(0, 1, n_cal), rng.normal(0, 1, 5000), rng.normal(2.5, 1, 200)
    pooled = np.concatenate([cal, qg])
    taus = beta_taus(pooled, n_cal, 20000, rng)
    assert np.all(taus[0.01] >= taus[0.02]) and np.all(taus[0.02] >= taus[0.05]) and np.all(taus[0.1] >= taus[0.3])
    r = rates_from_taus(taus, np.broadcast_to(qg, (20000, len(qg))), np.broadcast_to(qb, (20000, len(qb))))
    for a in (0.01, 0.05):
        k = k_of(a, n_cal)
        sd_b = stats.beta(n_cal - k + 1, k).std()
        sd = r[f"two|{a}"]["frr"].std()
        assert abs(sd / sd_b - 1) < 0.12, (a, sd, sd_b)                    # matches the U1 law
        assert abs(r[f"two|{a}"]["frr"].mean() - stats.beta(n_cal - k + 1, k).mean()) < 0.004
    # observed thresholds give exactly u2u5.unit_rates
    obs = rates_from_taus(observed_taus(cal, 1), qg[None, :], qb[None, :])
    ref = u.unit_rates(cal, qg, qb)
    for rule in ref:
        for key in ref[rule]:
            assert obs[rule][key][0] == ref[rule][key][0], (rule, key)
    units = {(t, s): (rng.normal(0, 1, 100), rng.normal(0, 1, 200), rng.normal(2, 1, 50)) for t in "abc" for s in (1, 2)}
    for cluster, tasks in ((True, list("abc")), (False, ["a"])):
        mc = {m: mc_rates(units, tasks, [1, 2], m, np.random.default_rng(2), 400, cluster) for m in ("combined", "cal", "test")}
        point = {rule: {key: float(statistics.fmean(u.unit_rates(*units[(t, s)])[rule][key][0] for t in tasks for s in (1, 2)))
                        for key in d} for rule, d in u.unit_rates(*units[(tasks[0], 1)]).items()}
        e = u.summarize_u2(point, mc)["two|0.01"]["0.95"]["R_P"]
        assert e["ci95_combined"][0] < e["value"] < e["ci95_combined"][1], e          # two-sided now
    print("u2a selftest passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    return selftest() if a.selftest else run()


if __name__ == "__main__":
    sys.exit(main())
