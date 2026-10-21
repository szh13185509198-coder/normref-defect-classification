#!/usr/bin/env python3
"""P47 U2-B and U5-A (amendments to U2-A and U5).

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U2-B / U5-A". Two write-once stages:

1. u2b - as U2-A, but one calibration draw per task shared by its 5 bank seeds (P15 draws the calibration set once per
         task, seed "p15cal|{task}"); U2-A drew it independently per seed. Also: rank agreement of the calibration
         scores across seeds, and the calibration-only percentile of each task's P15 point. CPU. -> ckpt/p47/u2b/
2. u5a - jackknife+ type sets next to the U5 jackknife sets, on the same episodes; the jackknife numbers must reproduce
         ckpt/p47/u5/ before anything is written. -> ckpt/p47/u5a/

    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u2b_u5a --stage u2b
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u2b_u5a --stage u5a --device cuda
    python -m experiments.p47.u2b_u5a --selftest
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
from experiments.p47 import u2a  # noqa: E402
from experiments.p47 import u2u5 as u  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p47" / s for s in ("u2b", "u5a")}
TOL = 1e-12


# --------------------------------------------------------------------------- U2-B

def taus_from_u(pooled: np.ndarray, uo: np.ndarray, n_cal: int) -> dict:
    """tau_a = Q(U_(k_a)) for one sorted uniform sample per replicate, uo [R, n_cal]."""
    out = {}
    for a in u2a.ALL_ALPHAS:
        k = u2a.k_of(a, n_cal)
        out[a] = np.quantile(pooled, uo[:, k - 1]) if k <= n_cal else np.full(uo.shape[0], np.inf)
    return out


def mc_rates_shared(units: dict, tasks: list, seeds: list, mode: str, rng, reps: int, cluster: bool,
                    shared: bool = True) -> dict:
    """As u2a.mc_rates; with shared=True one sorted uniform sample per task and replicate serves all seeds."""
    acc = {}
    pick = (rng.integers(0, len(tasks), (reps, len(tasks))) if mode in ("combined", "test")
            else np.tile(np.arange(len(tasks)), (reps, 1))) if cluster else None
    for ti, task in enumerate(tasks):
        n_cal = len(units[(task, seeds[0])][0])
        uo = np.sort(rng.random((reps, n_cal)), axis=1) if shared and mode in ("combined", "cal") else None
        boot = None
        if not cluster and mode in ("combined", "test"):   # one query resample per replicate, shared by the seeds
            _, qg0, qb0 = units[(task, seeds[0])]
            boot = (rng.integers(0, len(qg0), (reps, len(qg0))), rng.integers(0, len(qb0), (reps, len(qb0))))
        for seed in seeds:
            cal, qg, qb = units[(task, seed)]
            if mode in ("combined", "cal"):
                uu = uo if shared else np.sort(rng.random((reps, n_cal)), axis=1)
                taus = taus_from_u(np.concatenate([cal, qg]), uu, n_cal)
            else:
                taus = u2a.observed_taus(cal, reps)
            if boot is not None:
                g, b = qg[boot[0]], qb[boot[1]]
            else:
                g, b = np.broadcast_to(qg, (reps, len(qg))), np.broadcast_to(qb, (reps, len(qb)))
            for rule, d in u2a.rates_from_taus(taus, g, b).items():
                for key, v in d.items():
                    acc.setdefault(rule, {}).setdefault(key, np.zeros((reps, len(tasks))))[:, ti] += v / len(seeds)
    return {rule: {key: (np.take_along_axis(m, pick, axis=1).mean(1) if cluster else m.mean(1)) for key, m in d.items()}
            for rule, d in acc.items()}


def task_point(units: dict, task: str, seeds: list) -> dict:
    """Seed-mean rates of one task with the actual P15 calibration draw."""
    per = [u.unit_rates(*units[(task, s)]) for s in seeds]
    return {rule: {key: float(statistics.fmean(p[rule][key][0] for p in per)) for key in per[0][rule]} for rule in per[0]}


def mid_rank(draws: np.ndarray, x: float) -> float:
    return float(((draws < x - TOL).sum() + 0.5 * (np.abs(draws - x) <= TOL).sum()) / len(draws))


def seed_rank_agreement(units: dict, task: str, seeds: list) -> dict:
    from scipy import stats
    rho = [stats.spearmanr(units[(task, a)][0], units[(task, b)][0]).statistic
           for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    return {"spearman_cal_mean": float(statistics.fmean(rho)), "spearman_cal_min": float(min(rho))}


def run_u2b() -> int:
    import torch
    from scipy import stats
    from experiments.p10 import external_evaluate as ee
    from experiments.revision import revision_common as rc
    src = u.OUT["u25_external"]
    if not (src / "u25_external_info.json").is_file():
        raise cm.P47Error("the U2 / U5 extraction is missing")
    u2_res = rc.load_json(u.OUT["u2"] / "u2_risks.json")
    u2a_res = rc.load_json(u2a.OUT / "u2a_risks.json")
    seeds = list(rc.SEEDS)
    with cm.single_shot(OUT["u2b"]) as staging:
        units = {}
        for task in ee.TASKS:
            d = torch.load(src / f"{task}.pt", map_location="cpu", weights_only=False)
            good = np.asarray([t == d["normal"] for t in d["q_true"]])
            for seed in seeds:
                s = np.asarray(d["seeds"][seed]["s_q"], dtype=float)
                units[(task, seed)] = (np.asarray(d["seeds"][seed]["s_cal"], dtype=float), s[good], s[~good])
        rng = cm.np_rng("p47u2b")
        res = {"schema_version": "p47_u2b_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u2-b",
               "provenance": rc.git_provenance(), "pi": list(u.PI), "R": u.R_MC, "groups": {}, "tasks": {}}
        for name, tasks, cluster in (("visa_object_macro", list(ee.VISA_TASKS), True), ("mtd", [ee.MTD_TASK], False)):
            point = u2_res["groups"][name]["rates"]
            mine = {rule: {key: statistics.fmean(task_point(units, t, seeds)[rule][key] for t in tasks)
                           for key in point[rule]} for rule in point}
            worst = max(abs(mine[r][k] - point[r][k]) for r in point for k in point[r])
            if worst > 1e-12:
                raise cm.P47Error(f"{name}: point rates differ from U2 ({worst:.2e})")
            mc = {mode: mc_rates_shared(units, tasks, seeds, mode, rng, u.R_MC, cluster) for mode in ("combined", "cal", "test")}
            res["groups"][name] = {"tasks": tasks, "risks": u.summarize_u2(point, mc), "rates": point}
        n_cal = len(units[(ee.TASKS[0], seeds[0])][0])
        for task in ee.TASKS:                             # per task: calibration-only law of the seed-mean FRR
            point = task_point(units, task, seeds)
            row = {"rank_agreement": seed_rank_agreement(units, task, seeds), "alpha": {}}
            for shared in (True, False):
                mc = mc_rates_shared(units, [task], seeds, "cal", rng, 4000, False, shared)
                for a in u.ALPHAS_2:
                    k = u2a.k_of(a, n_cal)
                    draws = mc[f"two|{a}"]["frr"]
                    e = row["alpha"].setdefault(str(a), {"point_frr": point[f"two|{a}"]["frr"],
                                                         "beta_mean": float(stats.beta(n_cal - k + 1, k).mean()),
                                                         "beta_sd": float(stats.beta(n_cal - k + 1, k).std())})
                    tag = "shared" if shared else "independent"
                    e[f"sd_{tag}"] = float(draws.std())
                    e[f"percentile_{tag}"] = mid_rank(draws, point[f"two|{a}"]["frr"])
            res["tasks"][task] = row
        visa = [res["tasks"][t]["alpha"]["0.05"]["percentile_shared"] for t in ee.VISA_TASKS]
        res["visa_percentiles_ks_uniform_p"] = float(stats.kstest(visa, "uniform").pvalue)
        rc.write_json(staging / "u2b_risks.json", res)
        text = table(res, u2a_res)
        (staging / "u2b_risks.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def table(res: dict, u2a_res: dict) -> str:
    def f(e, k="ci95_combined"):
        return f"{1000 * e['value']:.1f} [{1000 * e[k][0]:.1f}, {1000 * e[k][1]:.1f}]"
    lines = ["# P47 U2-B - decision risks, one calibration draw per task shared by its bank seeds", "",
             "Per 1000 parts; 95 % interval (Monte Carlo: calibration law + test sample). U2-A column: the same cell",
             "with independent calibration draws per seed (kept for comparison).", ""]
    for name, g in res["groups"].items():
        lines += [f"## {name}", "", "| Rule | π | R_P (false rejects) | R_C (escapes) | Review | U2-A: R_P |",
                  "|---|---|---|---|---|---|"]
        for rule, by_pi in g["risks"].items():
            for pi, row in by_pi.items():
                rev = f(row["review"]) if "review" in row else "—"
                old = f(u2a_res["groups"][name]["risks"][rule][pi]["R_P"])
                lines.append(f"| {rule} | {pi} | {f(row['R_P'])} | {f(row['R_C'])} | {rev} | {old} |")
        lines.append("")
    lines += ["## Per task: calibration only, seed-mean FRR at α = 5 %", "",
              "| Task | Spearman across seeds (mean / min) | P15 point | SD shared / independent / Beta | "
              "Percentile of the point, shared / independent |", "|---|---|---|---|---|"]
    for task, row in res["tasks"].items():
        e, ra = row["alpha"]["0.05"], row["rank_agreement"]
        lines.append(f"| {task} | {ra['spearman_cal_mean']:.3f} / {ra['spearman_cal_min']:.3f} | {100 * e['point_frr']:.2f} % | "
                     f"{100 * e['sd_shared']:.2f} / {100 * e['sd_independent']:.2f} / {100 * e['beta_sd']:.2f} pp | "
                     f"{100 * e['percentile_shared']:.1f} % / {100 * e['percentile_independent']:.1f} % |")
    lines += ["", f"VisA, percentiles of the 11 points (shared draw) against uniform: KS p = "
              f"{res['visa_percentiles_ks_uniform_p']:.3f}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- U5-A

def loo_models(g, a, r, si, s_lab, qi, n, rows):
    """For each support i in rows: R_i = -f^(-i)_{y_i}(x_i) and the query nonconformity -f^(-i)(x) [Q, n]."""
    import torch
    res, test = [], []
    for i in rows:
        keep = [j for j in range(len(s_lab)) if j != i]
        sk = si[torch.as_tensor(keep, device=si.device)]
        lab = [s_lab[j] for j in keep]
        f = u.arr_scores(g, a, r, sk, lab, torch.cat([si[i:i + 1], qi]), n)
        res.append(-float(f[0, s_lab[i]]))
        test.append((-f[1:]).cpu().numpy())
    return np.asarray(res), np.stack(test)


def jkplus_sets(test_nc: np.ndarray, res: np.ndarray, alpha: float, cols=None) -> list[list[int]]:
    """Jackknife+ (Barber et al. 2021): c in Gamma(x) iff #{i : -f^(-i)_c(x) > R_i} < (1 - alpha)(n + 1)."""
    cols = list(range(test_nc.shape[2])) if cols is None else cols
    count = (test_nc[:, :, cols] > res[:, None, None]).sum(0)          # [Q, len(cols)]
    inc = count < (1 - alpha) * (len(res) + 1)
    return [[cols[j] for j in np.flatnonzero(row)] for row in inc]


def episode_u5a(g, a, r, si, s_lab, qi, q_true_idx, n, alpha, cols=None) -> dict:
    rows = [j for j, y in enumerate(s_lab) if cols is None or y in cols]
    res, test_nc = loo_models(g, a, r, si, s_lab, qi, n, rows)
    classes = cols if cols is not None else list(range(n))
    return u.set_metrics(jkplus_sets(test_nc, res, alpha, cols), q_true_idx, classes)


def unknown_u5a(g, a, r, si, s_lab, qi, q_true_idx, n, held, alpha) -> dict:
    import torch
    rest = [c for c in range(n) if c != held]
    remap = {c: i for i, c in enumerate(rest)}
    keep = [j for j, y in enumerate(s_lab) if y != held]
    si2 = si[torch.as_tensor(keep, device=si.device)]
    lab2 = [remap[s_lab[j]] for j in keep]
    res, test_nc = loo_models(g, a, r, si2, lab2, qi, len(rest), list(range(len(lab2))))
    flag = np.asarray([len(s) == 0 for s in jkplus_sets(test_nc, res, alpha)])
    unk = np.asarray([t == held for t in q_true_idx])
    return {"detect": float(flag[unk].mean()) if unk.any() else None, "false_alarm": float(flag[~unk].mean())}


def same(x, y) -> bool:
    return (x is None and y is None) or (x is not None and y is not None and abs(x - y) <= 1e-9)


def run_u5a(device) -> int:
    import torch
    from experiments.p6.p6b_stage3_common import load_query_contract
    from experiments.p10 import external_evaluate as ee
    from experiments.p10 import run_pilot as rp
    from experiments.p10.formal_evaluate import MANIFEST_DIR
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.revision import revision_common as rc
    u5 = rc.load_json(u.OUT["u5"] / "u5_typesets.json")
    ref = {(e["dataset"], e["unit"], e["seed"], e["K"], e["alpha"]): e for e in u5["episodes"]}
    ref_unk = {(e["product"], e["seed"], e["K"], e["alpha"], e["held"]): e for e in u5["unknown"]}
    with cm.single_shot(OUT["u5a"]) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
        rows, unknown, checked = [], [], 0

        def check(key, m_jk, store):
            nonlocal checked
            e = store[key]
            for k_ in ("coverage", "mean_size", "empty_rate", "class_coverage_mean", "detect", "false_alarm"):
                if k_ in m_jk and not same(m_jk[k_], e[k_]):
                    raise cm.P47Error(f"{key}: jackknife {k_} does not reproduce U5 ({m_jk[k_]} vs {e[k_]})")
            checked += 1

        with torch.no_grad():
            for product in rc.PRODUCTS:                       # data loading exactly as u2u5.run_u5
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
                    for k in u.U5_K:
                        man = rc.load_json(MANIFEST_DIR / product / f"k{k}_s{seed}.json")
                        order = list(man["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in man["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        si = torch.as_tensor(s_idx, device=device)
                        n = len(order)
                        t_idx = [order.index(t) for t in q_true]
                        for alpha in u.U5_ALPHA:
                            check(("mvtec", product, seed, k, alpha),
                                  u.episode_u5(g, soft, rr, si, s_lab, qi, t_idx, n, alpha), ref)
                            m = episode_u5a(g, soft, rr, si, s_lab, qi, t_idx, n, alpha)
                            rows.append({"dataset": "mvtec", "unit": product, "seed": seed, "K": k, "alpha": alpha,
                                         "classes": n, **m})
                            if n >= 3:
                                for held in range(n):
                                    check((product, seed, k, alpha, order[held]),
                                          u.unknown_episode(g, soft, rr, si, s_lab, qi, t_idx, n, held, alpha), ref_unk)
                                    m_u = unknown_u5a(g, soft, rr, si, s_lab, qi, t_idx, n, held, alpha)
                                    unknown.append({"product": product, "seed": seed, "K": k, "alpha": alpha,
                                                    "held": order[held], "classes": n, **m_u})
                del normal_clip, clip
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[p47-u5a] {product}: done", flush=True)
            for task in ee.VISA_TASKS:                       # as u2u5.run_u5
                d = torch.load(u.OUT["u25_external"] / f"{task}.pt", map_location="cpu", weights_only=False)
                order, normal = d["order"], d["normal"]
                n = len(order)
                cols = [c for c in range(n) if order[c] != normal]
                g = d["g"].to(device)
                good = [t == normal for t in d["q_true"]]
                for seed in rc.SEEDS:
                    soft, rr = d["seeds"][seed]["soft"].to(device), d["seeds"][seed]["rr"].to(device)
                    t5 = u.conformal_q(d["seeds"][seed]["s_cal"], u.SCREEN_ALPHA)
                    sel = [i for i, (s, gd) in enumerate(zip(d["seeds"][seed]["s_q"], good)) if not gd and s > t5]
                    if not sel:
                        continue
                    qi = torch.as_tensor([d["qrows"][i] for i in sel], device=device)
                    t_idx = [order.index(d["q_true"][i]) for i in sel]
                    for k in u.U5_K:
                        s_idx, s_lab = d["supports"][f"{seed}|{k}"]
                        si = torch.as_tensor(s_idx, device=device)
                        for alpha in u.U5_ALPHA:
                            check(("visa", task, seed, k, alpha),
                                  u.episode_u5(g, soft, rr, si, s_lab, qi, t_idx, n, alpha, cols), ref)
                            m = episode_u5a(g, soft, rr, si, s_lab, qi, t_idx, n, alpha, cols)
                            rows.append({"dataset": "visa", "unit": task, "seed": seed, "K": k, "alpha": alpha,
                                         "classes": len(cols), **m})
                print(f"[p47-u5a] {task}: done", flush=True)
        if checked != len(ref) + len(ref_unk):
            raise cm.P47Error(f"checked {checked} jackknife episodes, U5 has {len(ref) + len(ref_unk)}")
        for e in rows:                                       # argmax accuracy does not depend on the set rule
            e["argmax_acc"] = ref[(e["dataset"], e["unit"], e["seed"], e["K"], e["alpha"])]["argmax_acc"]
        for e in unknown:
            src = ref_unk[(e["product"], e["seed"], e["K"], e["alpha"], e["held"])]
            e["auroc"], e["n_unknown"], e["n_known"] = src["auroc"], src["n_unknown"], src["n_known"]
        summary = u.summarize_u5(rows, unknown)
        res = {"schema_version": "p47_u5a_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u5-a",
               "provenance": provenance, "jackknife_episodes_reproduced": checked, "summary": summary,
               "episodes": rows, "unknown": unknown,
               "guarantee": "jackknife+: coverage >= 1 - 2 alpha for a test part exchangeable with the supports "
                            "(Barber et al. 2021); with K supports per class this is the class-mean coverage"}
        rc.write_json(staging / "u5a_typesets.json", res)
        text = u5a_table(summary, u5["summary"], checked)
        (staging / "u5a_typesets.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def u5a_table(s: dict, s_jk: dict, checked: int) -> str:
    def p(x):
        return "—" if x is None else f"{100 * x:.1f} %"
    lines = ["# P47 U5-A - jackknife+ type sets (coverage >= 1 - 2α) next to the U5 jackknife sets", "",
             f"Integrity: the jackknife sets of all {checked} U5 episodes reproduced.", "",
             "| Data | K | α | Class cov. mean (J+ / J) | Class cov. min (J+ / J) | Coverage (J+ / J) | "
             "Mean size (J+ / J) | Singleton rate / acc. (J+) | Empty (J+ / J) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, e in s.items():
        if key.startswith("unknown"):
            continue
        ds, k, a = key.split("|")
        j = s_jk[key]
        lines.append(f"| {ds} | {k[1:]} | {a[1:]} | {p(e['class_coverage_mean'])} / {p(j['class_coverage_mean'])} | "
                     f"{p(e['class_coverage_min'])} / {p(j['class_coverage_min'])} | {p(e['coverage'])} / {p(j['coverage'])} | "
                     f"{e['mean_size']:.2f} / {j['mean_size']:.2f} | {p(e['singleton_rate'])} / {p(e['singleton_acc'])} | "
                     f"{p(e['empty_rate'])} / {p(j['empty_rate'])} |")
    lines += ["", "Unknown type (MVTec-FS, C ≥ 3, one class held out; flag = empty set):", "",
              "| K | α | Detection (J+ / J) | False alarm (J+ / J) | Episodes |", "|---|---|---|---|---|"]
    for key, e in s.items():
        if key.startswith("unknown"):
            _, k, a = key.split("|")
            j = s_jk[key]
            lines.append(f"| {k[1:]} | {a[1:]} | {p(e['detect'])} / {p(j['detect'])} | "
                         f"{p(e['false_alarm'])} / {p(j['false_alarm'])} | {e['episodes']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- self-test

def selftest() -> int:
    import torch
    import torch.nn.functional as F
    from scipy import stats
    rng = np.random.default_rng(7)
    # U2-B: comonotone seeds -> the seed-mean FRR spread equals the single-draw (Beta) spread; independent -> smaller
    base_cal, base_g = rng.normal(0, 1, 100), rng.normal(0, 1, 4000)
    units = {("t", s): (base_cal + 0.01 * s, base_g + 0.01 * s, rng.normal(3, 1, 200)) for s in range(5)}
    k = u2a.k_of(0.05, 100)
    sd_b = stats.beta(100 - k + 1, k).std()
    sh = mc_rates_shared(units, ["t"], list(range(5)), "cal", np.random.default_rng(1), 6000, False, True)
    ind = mc_rates_shared(units, ["t"], list(range(5)), "cal", np.random.default_rng(1), 6000, False, False)
    assert abs(sh["two|0.05"]["frr"].std() / sd_b - 1) < 0.1, sh["two|0.05"]["frr"].std()
    assert abs(ind["two|0.05"]["frr"].std() / (sd_b / math.sqrt(5)) - 1) < 0.15, ind["two|0.05"]["frr"].std()
    # with one seed, shared and independent are the same law as U2-A
    one = {("t", 0): units[("t", 0)]}
    a1 = u2a.mc_rates(one, ["t"], [0], "cal", np.random.default_rng(3), 6000, False)["two|0.05"]["frr"].std()
    b1 = mc_rates_shared(one, ["t"], [0], "cal", np.random.default_rng(3), 6000, False)["two|0.05"]["frr"].std()
    assert abs(a1 / b1 - 1) < 0.08, (a1, b1)
    # point and percentile helpers
    tp = task_point(units, "t", list(range(5)))
    assert abs(tp["two|0.05"]["frr"] - statistics.fmean(u.unit_rates(*units[("t", s)])["two|0.05"]["frr"][0]
                                                         for s in range(5))) < 1e-15
    assert mid_rank(np.array([0.0, 1.0, 1.0, 2.0]), 1.0) == 0.5
    assert seed_rank_agreement(units, "t", list(range(5)))["spearman_cal_min"] > 0.999
    # cluster mode runs and keeps the point inside the combined interval
    cl = {(t, s): (rng.normal(0, 1, 100), rng.normal(0, 1, 300), rng.normal(2, 1, 60)) for t in "abc" for s in (1, 2)}
    mc = {m: mc_rates_shared(cl, list("abc"), [1, 2], m, np.random.default_rng(4), 800, True) for m in ("combined", "cal", "test")}
    pt = {rule: {kk: statistics.fmean(task_point(cl, t, [1, 2])[rule][kk] for t in "abc") for kk in d}
          for rule, d in task_point(cl, "a", [1, 2]).items()}
    e = u.summarize_u2(pt, mc)["two|0.05"]["0.95"]["R_P"]
    assert e["ci95_combined"][0] < e["value"] < e["ci95_combined"][1], e
    # U5-A: identical LOO models -> jackknife+ equals split conformal with the same residuals
    f = rng.normal(0, 1, (40, 4))
    res = rng.normal(0, 1, 20)
    test_nc = np.repeat(-f[None], 20, axis=0)
    for a in (0.1, 0.2, 0.01):
        jp = jkplus_sets(test_nc, res, a)
        sp = u.type_sets(torch.as_tensor(f), u.conformal_q(list(res), a))
        assert jp == sp, a
    assert jkplus_sets(test_nc[:, :, :], res, 0.1, [1, 3]) == u.type_sets(torch.as_tensor(f), u.conformal_q(list(res), 0.1), [1, 3])
    # coverage on synthetic episodes: jackknife+ class-mean coverage >= 1 - 2 alpha (on average over episodes)
    gen = torch.Generator().manual_seed(0)
    n, kk, d = 4, 5, 16
    covs = []
    for _ in range(30):
        centers = F.normalize(torch.randn(n, d, generator=gen), dim=-1)
        lab = [c for c in range(n) for _ in range(kk)]
        qlab = [c for c in range(n) for _ in range(25)]

        def feats(labels):
            return torch.stack([centers[c] + 0.7 * torch.randn(d, generator=gen) for c in labels])
        g, a_, r_ = (torch.cat([feats(lab), feats(qlab)]) for _ in range(3))
        si, qi = torch.arange(len(lab)), torch.arange(len(lab), len(lab) + len(qlab))
        m = episode_u5a(g, a_, r_, si, lab, qi, qlab, n, 0.2)
        mj = u.episode_u5(g, a_, r_, si, lab, qi, qlab, n, 0.2)
        assert m["mean_size"] >= 0 and mj["mean_size"] >= 0
        covs.append(m["class_coverage_mean"])
    assert statistics.fmean(covs) >= 0.6, statistics.fmean(covs)
    um = unknown_u5a(g, a_, r_, si, lab, qi, qlab, n, 0, 0.2)
    assert 0 <= um["detect"] <= 1 and 0 <= um["false_alarm"] <= 1
    print("u2b_u5a selftest passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("u2b", "u5a"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "u2b":
        return run_u2b()
    if a.stage == "u5a":
        import torch
        return run_u5a(torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.error("--stage or --selftest is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
