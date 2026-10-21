#!/usr/bin/env python3
"""P47 U2 (decision risks, JCGM 106) and U5 (conformal type sets, unknown-type flag).

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U2", "U5" and "U2 / U5 implementation details".
Three write-once stages:

1. extract - VisA / MTD recomputed exactly as P15 external_task (GPU); integrity against P14 predictions and P15 units.
             -> ckpt/p47/u25_external/<task>.pt
2. u2      - producer / consumer risks and review rate with Monte Carlo 95 % intervals (CPU). -> ckpt/p47/u2/
3. u5      - jackknife conformal type sets on MVTec-FS (cached features) and VisA (stage 1), unknown-type flag.
             -> ckpt/p47/u5/

    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u2u5 --stage extract --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u2u5 --stage u2
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u2u5 --stage u5 --device cuda
    python -m experiments.p47.u2u5 --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from experiments.p47 import common as cm  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p47" / s for s in ("u25_external", "u2", "u5")}
PI = (0.90, 0.95, 0.99, 0.999)
ALPHAS_2 = (0.01, 0.02, 0.05)
ALPHA_REJECT = 0.01
ALPHA_PASS = (0.10, 0.30)
R_MC = 10000
U5_K = (3, 5)
U5_ALPHA = (0.10, 0.20)
SCREEN_ALPHA = 0.05


# --------------------------------------------------------------------------- shared scoring

def arr_scores(g, a, r, si, s_lab, qrows, n):
    """ARR fused scores [Q, n] exactly as experiments.p14.final_formal.episode_scores (zb + z(sa) + z(srr))."""
    from experiments.p10 import run_pilot as rp
    zb = rp._zscore(rp.simpleshot(g[si], s_lab, g[qrows], n))
    return zb + rp._zscore(rp.simpleshot(a[si], s_lab, a[qrows], n)) + rp._zscore(rp.simpleshot(r[si], s_lab, r[qrows], n))


def loo_nonconformity(g, a, r, si, s_lab, n):
    """r_i = -f_{y_i}(x_i) with every support-dependent part rebuilt without support i."""
    import torch
    out = []
    for i in range(len(s_lab)):
        keep = [j for j in range(len(s_lab)) if j != i]
        sk = si[torch.as_tensor(keep, device=si.device)]
        lab = [s_lab[j] for j in keep]
        f = arr_scores(g, a, r, sk, lab, si[i:i + 1], n)
        out.append(-float(f[0, s_lab[i]]))
    return out


def conformal_q(cal: list[float], alpha: float) -> float:
    s = sorted(cal)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else s[k - 1]


def type_sets(f, q_hat: float, cols=None) -> list[list[int]]:
    """Gamma(x) = {c : -f_c(x) <= q_hat}, restricted to columns cols (all if None)."""
    cols = list(range(f.shape[1])) if cols is None else cols
    nc = (-f[:, cols]).cpu().numpy()
    return [[cols[j] for j in np.flatnonzero(row <= q_hat)] for row in nc]


def set_metrics(sets, true_idx, classes) -> dict:
    """Coverage, class-conditional coverage, size, singleton rate / accuracy, empty rate (one episode)."""
    cov = [t in s for s, t in zip(sets, true_idx)]
    per_class = {c: statistics.fmean(cv for cv, t in zip(cov, true_idx) if t == c)
                 for c in classes if any(t == c for t in true_idx)}
    sizes = [len(s) for s in sets]
    single = [s for s in sets if len(s) == 1]
    single_true = [t for s, t in zip(sets, true_idx) if len(s) == 1]
    return {"coverage": statistics.fmean(cov), "class_coverage_mean": statistics.fmean(per_class.values()),
            "class_coverage_min": min(per_class.values()), "mean_size": statistics.fmean(sizes),
            "singleton_rate": len(single) / len(sets),
            "singleton_acc": (statistics.fmean(s[0] == t for s, t in zip(single, single_true)) if single else None),
            "empty_rate": statistics.fmean(sz == 0 for sz in sizes), "n": len(sets)}


# --------------------------------------------------------------------------- stage 1: VisA / MTD

def run_extract(device, batch_size: int) -> int:
    import torch
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10 import external_evaluate as ee
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.p15 import screening as sc
    from experiments.p15 import two_stage as ts
    from experiments.revision import revision_common as rc
    stat = ts.selected_statistic()
    p14 = {}
    for line in (ff.OUT["external"] / "p14_external_predictions.jsonl").open(encoding="utf-8"):
        u = json.loads(line)
        if u["method"] in ts.FROZEN:
            p14[(u["method"], u["task"], int(u["K"]), int(u["seed"]))] = (u["rgb_sha256"], u["pred"])
    p15 = {(u["task"], int(u["K"]), int(u["seed"])): u
           for u in rc.load_json(ts.OUT["external"] / "p15_external_results.json")["units"]}
    t0 = time.time()
    with cm.single_shot(OUT["u25_external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        info = {}
        for task in ee.TASKS:
            order = orders[task]
            dataset = task.split("__")[0]
            records = [json.loads(line) for line in (ee.BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
            draws = json.load((ee.BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
            by_sha = {r["rgb_sha256"]: r for r in records}
            queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
            normal = ee.NORMAL_CLASS[dataset]
            normal_candidates = [r["rgb_sha256"] for r in records
                                 if r["partition"] == "support_candidate" and r["class_label"] == normal]
            query_shas = {r["rgb_sha256"] for r in queries}
            banks, needed, used_normals = {}, set(), set()
            for seed in rc.SEEDS:                                   # as P15 external_task
                banks[seed] = ee.bank_draw(task, seed, normal_candidates, draws[normal][str(seed)]["K5"])
                support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
                if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
                    raise cm.P47Error(f"{task}/s{seed}: bank/support/query overlap")
                needed |= support_all | set(banks[seed])
                used_normals |= set(banks[seed]) | set(draws[normal][str(seed)]["K5"])
            shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
            index = {s: i for i, s in enumerate(shas)}
            with torch.no_grad():
                g, clip = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size)
                cal = random.Random(f"p15cal|{task}").sample(sorted(set(normal_candidates) - used_normals - query_shas),
                                                             ts.N_CAL)
                _, clip_cal = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in cal], device,
                                         batch_size)
                qrows = [index[r["rgb_sha256"]] for r in queries]
                qi = torch.as_tensor(qrows, device=device)
                q_true = [r["class_label"] for r in queries]
                q_shas = [r["rgb_sha256"] for r in queries]
                g = g.to(device)
                per_seed, supports = {}, {}
                for seed in rc.SEEDS:
                    bank_imgs = clip[torch.as_tensor([index[s] for s in banks[seed]])].to(device).float()
                    bank = bank_imgs.reshape(-1, clip.shape[-1])
                    tau = fa.tau99(bank_imgs)
                    parts = [ff.c1_descriptors(chunk.to(device).float(), bank_imgs, tau) for chunk in clip.split(ee.Q_CHUNK)]
                    pooled, soft, rr = (torch.cat(x) for x in zip(*parts))
                    per_m = {"": (pooled, soft, rr)}
                    s_all = ts.scores_of(clip, bank, stat, device, ee.Q_CHUNK)
                    s_q = [s_all[i] for i in qrows]
                    s_cal = ts.scores_of(clip_cal, bank, stat, device, ee.Q_CHUNK)
                    for k_shot in rc.K_VALUES:
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for s in draws[cls][str(seed)][f"K{k_shot}"]:
                                s_idx.append(index[s])
                                s_lab.append(c)
                        supports[(seed, k_shot)] = (s_idx, s_lab)
                        si = torch.as_tensor(s_idx, device=device)
                        zeros = torch.zeros(len(q_true), len(order), device=device)
                        scores = ff.episode_scores(g, per_m, si, s_lab, qi, len(order), zeros, zeros)
                        for m in ts.FROZEN:
                            pub = p14[(m, task, k_shot, seed)]
                            if pub != (q_shas, [order[int(i)] for i in scores[m].argmax(dim=1).cpu()]):
                                raise cm.P47Error(f"{task}/K{k_shot}/s{seed}/{m}: P14 predictions not reproduced")
                        mine = arr_scores(g, soft, rr, si, s_lab, qi, len(order))
                        if not torch.equal(mine, scores[ff.ARR]):
                            raise cm.P47Error(f"{task}/K{k_shot}/s{seed}: arr_scores differs from episode_scores")
                        types = {m: sc.restricted_type(scores[m], order, normal) for m in ts.TYPE_METHODS}
                        dm = ts.decision_metrics(s_q, s_cal, q_true, types, normal)
                        ref = p15[(task, k_shot, seed)]
                        worst = max([abs(dm["auroc"] - ref["auroc"])]
                                    + [abs(dm["taus"][a] - ref["taus"][a]) for a in ref["taus"]]
                                    + [abs(dm["two_level"][ff.ARR][a][x] - ref["two_level"][ff.ARR][a][x])
                                       for a in ref["two_level"][ff.ARR] for x in ("false_reject_rate", "escape_rate")]
                                    + [abs(dm["three_way"][a][x] - ref["three_way"][a][x]) for a in ref["three_way"]
                                       for x in ("auto_false_reject_rate", "escape_rate", "review_rate_good",
                                                 "review_rate_defect")])
                        if worst > 1e-12:
                            raise cm.P47Error(f"{task}/K{k_shot}/s{seed}: P15 unit not reproduced ({worst:.2e})")
                    per_seed[seed] = {"s_q": s_q, "s_cal": s_cal, "soft": soft.float().cpu(), "rr": rr.float().cpu()}
            torch.save({"schema_version": "p47_u25_external_v1", "task": task, "order": order, "normal": normal,
                        "q_true": q_true, "q_shas": q_shas, "qrows": qrows, "g": g.float().cpu(),
                        "supports": {f"{s}|{k}": v for (s, k), v in supports.items()}, "seeds": per_seed,
                        "statistic": stat}, staging / f"{task}.pt")
            info[task] = {"queries": len(queries), "rows": len(shas), "calibration": len(cal)}
            print(f"[p47-u25-extract] {task}: {len(queries)} queries; P14 and P15 reproduced; "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        rc.write_json(staging / "u25_external_info.json", {"provenance": provenance, "statistic": stat, "tasks": info,
                                                            "integrity": "P14 predictions (A, ARR, SimpleShot; K 1/3/5) "
                                                            "and P15 units (AUROC, taus, two-level ARR, three-way) reproduced"})
    return 0


# --------------------------------------------------------------------------- stage 2: U2

def unit_rates(cal, qg, qb):
    """cal [R, Nc] (or [Nc]); qg / qb [R, n] (or [n]). Rates for every rule, vectorised over replicates."""
    cal = np.atleast_2d(cal)
    qg, qb = np.atleast_2d(qg), np.atleast_2d(qb)
    s = np.sort(cal, axis=1)
    n_cal = s.shape[1]

    def tau(a):
        k = math.ceil((n_cal + 1) * (1 - a))
        return np.full(s.shape[0], np.inf) if k > n_cal else s[:, k - 1]
    out = {}
    for a in ALPHAS_2:
        t = tau(a)[:, None]
        out[f"two|{a}"] = {"frr": (qg > t).mean(1), "escape": (qb <= t).mean(1)}
    t_rej = tau(ALPHA_REJECT)[:, None]
    for a in ALPHA_PASS:
        t_pass = tau(a)[:, None]
        out[f"three|{a}"] = {"frr": (qg > t_rej).mean(1), "escape": (qb <= t_pass).mean(1),
                             "review_good": ((qg > t_pass) & (qg <= t_rej)).mean(1),
                             "review_bad": ((qb > t_pass) & (qb <= t_rej)).mean(1)}
    return out


def risks(rates: dict, pi: float) -> dict:
    out = {"R_P": pi * rates["frr"], "R_C": (1 - pi) * rates["escape"]}
    if "review_good" in rates:
        out["review"] = pi * rates["review_good"] + (1 - pi) * rates["review_bad"]
    return out


def mc_rates(units: dict, tasks: list, seeds: list, mode: str, rng, reps: int, cluster: bool) -> dict:
    """Replicate-level macro rates. mode in {'combined', 'cal', 'test'}. units[(task, seed)] = (cal, qg, qb)."""
    acc = {}
    if cluster:
        pick = rng.integers(0, len(tasks), (reps, len(tasks))) if mode in ("combined", "test") else \
            np.tile(np.arange(len(tasks)), (reps, 1))
    for ti, task in enumerate(tasks):
        for seed in seeds:
            cal, qg, qb = units[(task, seed)]
            c = cal[rng.integers(0, len(cal), (reps, len(cal)))] if mode in ("combined", "cal") else cal[None, :].repeat(reps, 0)
            if not cluster and mode in ("combined", "test"):
                g = qg[rng.integers(0, len(qg), (reps, len(qg)))]
                b = qb[rng.integers(0, len(qb), (reps, len(qb)))]
            else:
                g, b = qg[None, :].repeat(reps, 0), qb[None, :].repeat(reps, 0)
            r = unit_rates(c, g, b)
            for rule, d in r.items():
                for k, v in d.items():
                    acc.setdefault(rule, {}).setdefault(k, np.zeros((reps, len(tasks))))[:, ti] += v / len(seeds)
    out = {}
    for rule, d in acc.items():
        out[rule] = {}
        for k, m in d.items():
            out[rule][k] = (np.take_along_axis(m, pick, axis=1).mean(1) if cluster else m.mean(1))
    return out


def summarize_u2(point: dict, mc: dict) -> dict:
    res = {}
    for rule in point:
        res[rule] = {}
        for pi in PI:
            p = risks(point[rule], pi)
            row = {}
            for k, v in p.items():
                e = {"value": float(v)}
                for mode, m in mc.items():
                    draws = risks(m[rule], pi)[k]
                    e[f"ci95_{mode}"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
                row[k] = e
            res[rule][str(pi)] = row
    return res


def run_u2() -> int:
    import torch
    from scipy import stats
    from experiments.p10 import external_evaluate as ee
    from experiments.p14 import final_formal as ff
    from experiments.p15 import two_stage as ts
    from experiments.revision import revision_common as rc
    if not (OUT["u25_external"] / "u25_external_info.json").is_file():
        raise cm.P47Error("run --stage extract first")
    p15 = rc.load_json(ts.OUT["external"] / "p15_external_results.json")
    with cm.single_shot(OUT["u2"]) as staging:
        units, normals = {}, {}
        for task in ee.TASKS:
            d = torch.load(OUT["u25_external"] / f"{task}.pt", map_location="cpu", weights_only=False)
            good = np.asarray([t == d["normal"] for t in d["q_true"]])
            for seed in rc.SEEDS:
                s = np.asarray(d["seeds"][seed]["s_q"], dtype=float)
                units[(task, seed)] = (np.asarray(d["seeds"][seed]["s_cal"], dtype=float), s[good], s[~good])
        groups = {"visa_object_macro": (list(ee.VISA_TASKS), True, p15["visa_object_macro"]),
                  "mtd": ([ee.MTD_TASK], False, p15["mtd"])}
        res = {"schema_version": "p47_u2_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u2",
               "provenance": rc.git_provenance(), "pi": list(PI), "R": R_MC, "groups": {}}
        rng = cm.np_rng("p47u2")
        for name, (tasks, cluster, agg) in groups.items():
            point = {}
            for rule in unit_rates(*units[(tasks[0], rc.SEEDS[0])]):
                point[rule] = {}
            for task in tasks:
                for seed in rc.SEEDS:
                    for rule, dct in unit_rates(*units[(task, seed)]).items():
                        for k, v in dct.items():
                            point[rule][k] = point[rule].get(k, 0.0) + float(v[0]) / len(tasks) / len(rc.SEEDS)
            # integrity: the point values equal the P15 aggregates (K = 5 row; the screening does not depend on K)
            a5 = agg["5"]
            checks = {f"two|{a}": {"frr": a5[f"{ff.ARR}|a={a}|false_reject_rate"]["mean"],
                                   "escape": a5[f"{ff.ARR}|a={a}|escape_rate"]["mean"]} for a in ALPHAS_2}
            for a in ALPHA_PASS:
                checks[f"three|{a}"] = {"frr": a5[f"3way|pass={a}|auto_false_reject_rate"]["mean"],
                                        "escape": a5[f"3way|pass={a}|escape_rate"]["mean"]}
            worst = max(abs(point[r][k] - v) for r, d in checks.items() for k, v in d.items())
            rev = max(abs(1000 * (0.95 * point[f"three|{a}"]["review_good"] + 0.05 * point[f"three|{a}"]["review_bad"])
                          - a5[f"3way|pass={a}|review_items_per_1000"]["mean"]) / 1000 for a in ALPHA_PASS)
            worst = max(worst, rev)
            if worst > 1e-12:
                raise cm.P47Error(f"{name}: point values do not reproduce the P15 aggregates ({worst:.2e})")
            mc = {mode: mc_rates(units, tasks, list(rc.SEEDS), mode, rng, R_MC, cluster) for mode in ("combined", "cal", "test")}
            res["groups"][name] = {"tasks": tasks, "point_vs_p15_max_abs": worst, "checked_keys": list(checks),
                                   "risks": summarize_u2(point, mc), "rates": point}
            # calibration spread of the FRR against Beta(N - k + 1, k)
            n_cal = len(units[(tasks[0], rc.SEEDS[0])][0])
            beta = {}
            for a in ALPHAS_2:
                k = math.ceil((n_cal + 1) * (1 - a))
                sd_theory = float(stats.beta(n_cal - k + 1, k).std()) if k <= n_cal else None
                sds = []
                for task in tasks:
                    for seed in rc.SEEDS:
                        cal, qg, qb = units[(task, seed)]
                        r = unit_rates(cal[rng.integers(0, len(cal), (2000, len(cal)))], qg[None, :].repeat(2000, 0),
                                       qb[None, :].repeat(2000, 0))
                        sds.append(float(r[f"two|{a}"]["frr"].std()))
                beta[str(a)] = {"bootstrap_sd_unit_mean": statistics.fmean(sds), "beta_sd": sd_theory}
            res["groups"][name]["calibration_spread_vs_beta"] = beta
            if not cluster:      # MTD: Clopper-Pearson of the seed-mean proportions (reference, not combined)
                cal, qg, qb = units[(tasks[0], rc.SEEDS[0])]
                cp = {}
                for rule, dct in point.items():
                    for k in ("frr", "escape"):
                        n = len(qg) if k == "frr" else len(qb)
                        x = round(dct[k] * n)
                        lo = 0.0 if x == 0 else float(stats.beta(x, n - x + 1).ppf(0.025))
                        hi = 1.0 if x == n else float(stats.beta(x + 1, n - x).ppf(0.975))
                        cp[f"{rule}|{k}"] = [lo, hi]
                res["groups"][name]["clopper_pearson_test_only"] = cp
        rc.write_json(staging / "u2_risks.json", res)
        text = u2_table(res)
        (staging / "u2_risks.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def u2_table(res: dict) -> str:
    def f(e, k="ci95_combined"):
        return f"{1000 * e['value']:.1f} [{1000 * e[k][0]:.1f}, {1000 * e[k][1]:.1f}]"
    lines = ["# P47 U2 - decision risks (per 1000 parts; 95 % interval, calibration + test sample, Monte Carlo)", ""]
    for name, g in res["groups"].items():
        lines += [f"## {name} (point vs P15 max |diff| = {g['point_vs_p15_max_abs']:.1e})", "",
                  "| Rule | π | R_P (false rejects) | R_C (escapes) | Review |", "|---|---|---|---|---|"]
        for rule, by_pi in g["risks"].items():
            for pi, row in by_pi.items():
                rev = f(row["review"]) if "review" in row else "—"
                lines.append(f"| {rule} | {pi} | {f(row['R_P'])} | {f(row['R_C'])} | {rev} |")
        lines += ["", "Calibration spread of the FRR (bootstrap SD vs Beta SD): " + ", ".join(
            f"α={a}: {100 * v['bootstrap_sd_unit_mean']:.2f} vs {100 * v['beta_sd']:.2f} pp" if v["beta_sd"] else f"α={a}: infeasible"
            for a, v in g["calibration_spread_vs_beta"].items()), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- stage 3: U5

def episode_u5(g, a, r, si, s_lab, qi, q_true_idx, n, alpha, cols=None) -> dict:
    import torch
    f = arr_scores(g, a, r, si, s_lab, qi, n)
    keep = [j for j, y in enumerate(s_lab) if cols is None or y in cols]
    cal = loo_nonconformity(g, a, r, si, s_lab, n)
    q_hat = conformal_q([cal[j] for j in keep], alpha)
    sets = type_sets(f, q_hat, cols)
    classes = cols if cols is not None else list(range(n))
    out = set_metrics(sets, q_true_idx, classes)
    out["q_hat"] = q_hat if math.isfinite(q_hat) else None
    out["argmax_acc"] = statistics.fmean(int(torch.as_tensor(row).argmax()) == t
                                         for row, t in zip(f[:, classes].cpu().tolist(), [classes.index(t) for t in q_true_idx]))
    return out


def unknown_episode(g, a, r, si, s_lab, qi, q_true_idx, n, held, alpha) -> dict:
    """Class `held` removed from the supports; scores over the remaining classes; empty set = unknown."""
    import torch
    rest = [c for c in range(n) if c != held]
    remap = {c: i for i, c in enumerate(rest)}
    keep = [j for j, y in enumerate(s_lab) if y != held]
    si2 = si[torch.as_tensor(keep, device=si.device)]
    lab2 = [remap[s_lab[j]] for j in keep]
    f = arr_scores(g, a, r, si2, lab2, qi, len(rest))
    q_hat = conformal_q(loo_nonconformity(g, a, r, si2, lab2, len(rest)), alpha)
    nc_min = (-f).min(dim=1).values.cpu().numpy()
    flag = nc_min > q_hat
    unk = np.asarray([t == held for t in q_true_idx])
    return {"detect": float(flag[unk].mean()) if unk.any() else None, "false_alarm": float(flag[~unk].mean()),
            "auroc": cm.auroc(nc_min[~unk], nc_min[unk]) if unk.any() and (~unk).any() else None,
            "n_unknown": int(unk.sum()), "n_known": int((~unk).sum())}


def run_u5(device) -> int:
    import torch
    from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract
    from experiments.p10 import external_evaluate as ee
    from experiments.p10 import run_pilot as rp
    from experiments.p10.formal_evaluate import MANIFEST_DIR
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.revision import revision_common as rc
    published = {(r["product"], int(r["K"]), int(r["seed"])): r["balanced_accuracy"]
                 for r in rc.load_json(ff.OUT["mvtec"] / "p14_mvtec_results.json")["raw_metrics"] if r["method"] == ff.ARR}
    with cm.single_shot(OUT["u5"]) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
        rows, unknown, worst = [], [], 0.0
        with torch.no_grad():
            for product in rc.PRODUCTS:
                dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
                qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
                nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
                path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
                n_pool = len(dev["rel_paths"])
                g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
                clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
                q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
                qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
                normal_clip = nf["clip18_patches"].to(device).float()
                name_index = {x: i for i, x in enumerate(nf["names"])}
                for seed in rc.SEEDS:
                    draw = nf["draws"][str(seed)][:20]
                    bank_imgs = normal_clip[torch.as_tensor([name_index[x] for x in draw], device=device)]
                    _, soft, rr = ff.c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))
                    for k in U5_K:
                        man = rc.load_json(MANIFEST_DIR / product / f"k{k}_s{seed}.json")
                        order = list(man["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in man["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        si = torch.as_tensor(s_idx, device=device)
                        n = len(order)
                        f = arr_scores(g, soft, rr, si, s_lab, qi, n)
                        ba = classification_metrics(q_true, [order[int(i)] for i in f.argmax(dim=1).cpu()], order)["balanced_accuracy"]
                        worst = max(worst, abs(ba - published[(product, k, seed)]))
                        t_idx = [order.index(t) for t in q_true]
                        for alpha in U5_ALPHA:
                            m = episode_u5(g, soft, rr, si, s_lab, qi, t_idx, n, alpha)
                            rows.append({"dataset": "mvtec", "unit": product, "seed": seed, "K": k, "alpha": alpha,
                                         "classes": n, **m})
                            if n >= 3:
                                for held in range(n):
                                    u = unknown_episode(g, soft, rr, si, s_lab, qi, t_idx, n, held, alpha)
                                    unknown.append({"product": product, "seed": seed, "K": k, "alpha": alpha,
                                                    "held": order[held], "classes": n, **u})
                del normal_clip, clip
                torch.cuda.empty_cache()
                print(f"[p47-u5] {product}: done", flush=True)
            if worst >= 1e-9:
                raise cm.P47Error(f"ARR with all supports does not reproduce P14 (max |dBA| = {worst:.2e})")
            for task in ee.VISA_TASKS:
                d = torch.load(OUT["u25_external"] / f"{task}.pt", map_location="cpu", weights_only=False)
                order, normal = d["order"], d["normal"]
                n = len(order)
                cols = [c for c in range(n) if order[c] != normal]
                g = d["g"].to(device)
                good = [t == normal for t in d["q_true"]]
                for seed in rc.SEEDS:
                    soft, rr = d["seeds"][seed]["soft"].to(device), d["seeds"][seed]["rr"].to(device)
                    t5 = conformal_q(d["seeds"][seed]["s_cal"], SCREEN_ALPHA)
                    sel = [i for i, (s, gd) in enumerate(zip(d["seeds"][seed]["s_q"], good)) if not gd and s > t5]
                    if not sel:
                        continue
                    qi = torch.as_tensor([d["qrows"][i] for i in sel], device=device)
                    t_idx = [order.index(d["q_true"][i]) for i in sel]
                    for k in U5_K:
                        s_idx, s_lab = d["supports"][f"{seed}|{k}"]
                        si = torch.as_tensor(s_idx, device=device)
                        for alpha in U5_ALPHA:
                            m = episode_u5(g, soft, rr, si, s_lab, qi, t_idx, n, alpha, cols)
                            rows.append({"dataset": "visa", "unit": task, "seed": seed, "K": k, "alpha": alpha,
                                         "classes": len(cols), **m})
                print(f"[p47-u5] {task}: done", flush=True)
        summary = summarize_u5(rows, unknown)
        res = {"schema_version": "p47_u5_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u5",
               "provenance": provenance, "integrity_mvtec_all_supports_vs_p14_max_abs_dBA": worst,
               "summary": summary, "episodes": rows, "unknown": unknown,
               "caveat": "jackknife: worst-case coverage guarantee 1 - 2 alpha (Barber et al. 2021)"}
        rc.write_json(staging / "u5_typesets.json", res)
        text = u5_table(summary, worst)
        (staging / "u5_typesets.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def summarize_u5(rows: list, unknown: list) -> dict:
    out = {}
    keys = ("coverage", "class_coverage_mean", "class_coverage_min", "mean_size", "singleton_rate", "singleton_acc",
            "empty_rate", "argmax_acc")
    for ds in ("mvtec", "visa"):
        for k in U5_K:
            for alpha in U5_ALPHA:
                sel = [r for r in rows if r["dataset"] == ds and r["K"] == k and r["alpha"] == alpha]
                if not sel:
                    continue
                units = sorted({r["unit"] for r in sel})
                e = {}
                for key in keys:
                    per_unit = [statistics.fmean(r[key] for r in sel if r["unit"] == u and r[key] is not None)
                                for u in units if any(r["unit"] == u and r[key] is not None for r in sel)]
                    e[key] = statistics.fmean(per_unit) if per_unit else None
                e["coverage_pooled"] = sum(r["coverage"] * r["n"] for r in sel) / sum(r["n"] for r in sel)
                e["units"] = len(units)
                out[f"{ds}|K{k}|a{alpha}"] = e
    for k in U5_K:
        for alpha in U5_ALPHA:
            sel = [u for u in unknown if u["K"] == k and u["alpha"] == alpha]
            if sel:
                det = [u["detect"] for u in sel if u["detect"] is not None]
                auc = [u["auroc"] for u in sel if u["auroc"] is not None]
                out[f"unknown|K{k}|a{alpha}"] = {"detect": statistics.fmean(det), "false_alarm": statistics.fmean(
                    u["false_alarm"] for u in sel), "auroc": statistics.fmean(auc), "episodes": len(sel)}
    return out


def u5_table(summary: dict, worst: float) -> str:
    def p(x):
        return "—" if x is None else f"{100 * x:.1f} %"
    lines = ["# P47 U5 - conformal type sets (jackknife; worst-case coverage 1 - 2α) and the unknown-type flag", "",
             f"Integrity: ARR with all supports reproduces P14 (MVTec-FS max |dBA| = {worst:.1e}).", "",
             "| Data | K | α | Coverage | Class cov. mean / min | Mean size | Singleton rate / acc. | Empty | Argmax acc. |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, e in summary.items():
        if key.startswith("unknown"):
            continue
        ds, k, a = key.split("|")
        lines.append(f"| {ds} | {k[1:]} | {a[1:]} | {p(e['coverage'])} | {p(e['class_coverage_mean'])} / {p(e['class_coverage_min'])} | "
                     f"{e['mean_size']:.2f} | {p(e['singleton_rate'])} / {p(e['singleton_acc'])} | {p(e['empty_rate'])} | "
                     f"{p(e['argmax_acc'])} |")
    lines += ["", "Unknown type (MVTec-FS, C ≥ 3, one class held out at a time):", "",
              "| K | α | Detection | False alarm (known) | AUROC | Episodes |", "|---|---|---|---|---|---|"]
    for key, e in summary.items():
        if key.startswith("unknown"):
            _, k, a = key.split("|")
            lines.append(f"| {k[1:]} | {a[1:]} | {p(e['detect'])} | {p(e['false_alarm'])} | {e['auroc']:.3f} | {e['episodes']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- self-test

def selftest() -> int:
    import torch
    import torch.nn.functional as F
    gen = torch.Generator().manual_seed(0)
    n, k, d = 4, 5, 16
    centers = F.normalize(torch.randn(n, d, generator=gen), dim=-1)
    lab = [c for c in range(n) for _ in range(k)]
    qlab = [c for c in range(n) for _ in range(30)]

    def feats(labels, noise):
        return torch.stack([centers[c] + noise * torch.randn(d, generator=gen) for c in labels])
    g = torch.cat([feats(lab, 0.6), feats(qlab, 0.6)])
    a = torch.cat([feats(lab, 0.6), feats(qlab, 0.6)])
    r = torch.cat([feats(lab, 0.6), feats(qlab, 0.6)])
    si = torch.arange(len(lab))
    qi = torch.arange(len(lab), len(lab) + len(qlab))
    m = episode_u5(g, a, r, si, lab, qi, qlab, n, 0.10)
    assert 0.7 <= m["coverage"] <= 1.0 and m["mean_size"] >= 1.0, m
    m2 = episode_u5(g, a, r, si, lab, qi, qlab, n, 0.40)
    assert m2["mean_size"] <= m["mean_size"] + 1e-9                     # larger alpha -> smaller sets
    u = unknown_episode(g, a, r, si, lab, qi, qlab, n, 0, 0.10)
    assert u["n_unknown"] == 30 and 0 <= u["detect"] <= 1 and 0 <= u["false_alarm"] <= 1
    # U2: the vectorised rates equal P15's definitions on a random unit
    rng = np.random.default_rng(1)
    cal, qg, qb = rng.normal(0, 1, 100), rng.normal(0, 1, 300), rng.normal(2, 1, 80)
    from experiments.p15 import screening as sc
    rates = unit_rates(cal, qg, qb)
    s, truth = list(qg) + list(qb), ["n"] * 300 + ["d"] * 80
    for a_ in ALPHAS_2:
        t = sc.conformal_threshold(list(cal), a_)
        assert abs(rates[f"two|{a_}"]["frr"][0] - np.mean(qg > t)) < 1e-15
    for a_ in ALPHA_PASS:
        tw = sc.three_way(s, truth, "n", sc.conformal_threshold(list(cal), ALPHA_REJECT), sc.conformal_threshold(list(cal), a_))
        rr_ = rates[f"three|{a_}"]
        assert abs(rr_["frr"][0] - tw["auto_false_reject_rate"]) < 1e-15 and abs(rr_["escape"][0] - tw["escape_rate"]) < 1e-15
        assert abs(rr_["review_good"][0] - tw["review_rate_good"]) < 1e-15 and abs(rr_["review_bad"][0] - tw["review_rate_defect"]) < 1e-15
    units = {(t, s_): (rng.normal(0, 1, 100), rng.normal(0, 1, 200), rng.normal(2, 1, 50)) for t in "abc" for s_ in (1, 2)}
    for cluster, tasks in ((True, list("abc")), (False, ["a"])):
        mc = {mode: mc_rates(units, tasks, [1, 2], mode, np.random.default_rng(2), 500, cluster) for mode in ("combined", "cal", "test")}
        point = {rule: {kk: float(statistics.fmean(unit_rates(*units[(t, s_)])[rule][kk][0] for t in tasks for s_ in (1, 2)))
                        for kk in dct} for rule, dct in unit_rates(*units[(tasks[0], 1)]).items()}
        res = summarize_u2(point, mc)
        e = res["two|0.05"]["0.95"]["R_P"]
        assert e["ci95_combined"][0] <= e["value"] <= e["ci95_combined"][1] + 1e-12, e
        assert (e["ci95_combined"][1] - e["ci95_combined"][0]) >= 0.5 * (e["ci95_cal"][1] - e["ci95_cal"][0])
    u2_table({"groups": {"x": {"point_vs_p15_max_abs": 0.0, "risks": res,
                               "calibration_spread_vs_beta": {"0.05": {"bootstrap_sd_unit_mean": 0.02, "beta_sd": 0.0215}}}}})
    print("u2u5 selftest passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("extract", "u2", "u5"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "u2":
        return run_u2()
    import torch
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.stage == "extract":
        return run_extract(device, a.batch_size)
    if a.stage == "u5":
        return run_u5(device)
    ap.error("--stage or --selftest is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
