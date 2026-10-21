#!/usr/bin/env python3
"""P15: two-stage inspection (normal-calibrated screening + ARR type classification).

Protocol: docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md. The frozen classifiers are not changed;
test stages first reproduce the published per-query predictions exactly.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p15.two_stage --stage dev --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p15.two_stage --stage external --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p15.two_stage --stage cable-test --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import cable_oof as co  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.industrial_metrics import normal_aware  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p13 import pilot as p13  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p15 import screening as sc  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p15" / s for s in ("dev", "external", "cable_test")}
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13", "experiments/p14",
           "experiments/p15", "experiments/revision", "docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md")
CABLE_TEST = PROJECT_ROOT / "dataset" / "metadata" / "MVTecAD-Cable" / "test-shot.jsonl"
N_CAL = 100
STATS = ("max", "top1pct")
FROZEN = (ff.SS, ff.A, ff.ARR)          # argmax predictions that must reproduce P14 exactly
TYPE_METHODS = (ff.ARR, ff.SS)           # stage-2 classifiers (ARR primary, SimpleShot reference)


@contextlib.contextmanager
def single_shot(out: Path):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; the P15 stage is single-shot")
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


def scores_of(clip_cpu: torch.Tensor, bank: torch.Tensor, stat: str, device, chunk: int = 16) -> list[float]:
    out = []
    for c in clip_cpu.split(chunk):
        out += sc.image_score(ra.patch_anomaly(c.to(device).float(), bank), stat).tolist()
    return out


def decision_metrics(s_q, s_cal, truth, types_by_method, normal) -> dict:
    """All P15 metrics for one unit (task/product, K, seed)."""
    good = [x for x, t in zip(s_q, truth) if t == normal]
    bad = [x for x, t in zip(s_q, truth) if t != normal]
    taus = {a: sc.conformal_threshold(s_cal, a) for a in (*sc.ALPHAS, *sc.ALPHA_PASS)}
    out = {"auroc": sc.auroc(good, bad), "taus": {str(a): t for a, t in taus.items()}, "two_level": {}, "three_way": {}}
    for m, types in types_by_method.items():
        out["two_level"][m] = {str(a): sc.two_level(s_q, truth, types, normal, taus[a]) for a in sc.ALPHAS}
    for a_lo in sc.ALPHA_PASS:
        out["three_way"][str(a_lo)] = sc.three_way(s_q, truth, normal, taus[sc.ALPHA_REJECT], taus[a_lo])
    out["oracle_min_review"] = sc.oracle_min_review(s_q, truth, normal, taus[sc.ALPHA_REJECT])
    return out


def flatten(unit: dict) -> dict:
    """Scalar metrics of one unit, keyed by readable names (for averaging)."""
    f = {"auroc": unit["auroc"]}
    for m, per_a in unit["two_level"].items():
        for a, v in per_a.items():
            for k in ("false_reject_rate", "escape_rate", "type_acc_rejected_defects"):
                f[f"{m}|a={a}|{k}"] = v[k]
            for k, x in v["per_1000"].items():
                f[f"{m}|a={a}|{k}_per_1000"] = x
    for a_lo, v in unit["three_way"].items():
        for k in ("auto_false_reject_rate", "escape_rate"):
            f[f"3way|pass={a_lo}|{k}"] = v[k]
        for k, x in v["per_1000"].items():
            f[f"3way|pass={a_lo}|{k}_per_1000"] = x
    for k, x in unit["oracle_min_review"]["per_1000"].items():
        f[f"oracle|{k}_per_1000"] = x
    f["oracle|escape_rate"] = unit["oracle_min_review"]["escape_rate"]
    return f


def mean_sd(values):
    values = list(values)
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def report_lines(title: str, agg: dict, argmax_rows: dict | None = None) -> list[str]:
    """agg: K -> key -> {mean, sd}."""
    lines = [f"## {title}", "", "| K | AUROC | α | FRR | Escape rate | Type acc. (ARR) | FR / 1000 | Escapes / 1000 | Wrong type / 1000 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, v in agg.items():
        for a in map(str, sc.ALPHAS):
            p = f"{ff.ARR}|a={a}|"
            lines.append(f"| {k} | {v['auroc']['mean']:.3f} | {a} | {100 * v[p + 'false_reject_rate']['mean']:.1f}% | "
                         f"{100 * v[p + 'escape_rate']['mean']:.1f}% | {100 * v[p + 'type_acc_rejected_defects']['mean']:.1f}% | "
                         f"{v[p + 'false_rejects_per_1000']['mean']:.0f} | {v[p + 'escapes_per_1000']['mean']:.1f} | "
                         f"{v[p + 'wrong_type_per_1000']['mean']:.1f} |")
    lines += ["", "Stage-2 reference (SimpleShot restricted), type acc. at α=0.02: " + ", ".join(
        f"K={k} {100 * v[f'{ff.SS}|a=0.02|type_acc_rejected_defects']['mean']:.1f}%" for k, v in agg.items()), "",
        "| K | Review band (auto-pass α) | Auto FRR | Escape rate | FR / 1000 | Escapes / 1000 | Review / 1000 |", "|---|---|---|---|---|---|---|"]
    for k, v in agg.items():
        for a_lo in map(str, sc.ALPHA_PASS):
            p = f"3way|pass={a_lo}|"
            lines.append(f"| {k} | {a_lo} | {100 * v[p + 'auto_false_reject_rate']['mean']:.1f}% | {100 * v[p + 'escape_rate']['mean']:.1f}% | "
                         f"{v[p + 'false_rejects_per_1000']['mean']:.0f} | {v[p + 'escapes_per_1000']['mean']:.1f} | {v[p + 'review_items_per_1000']['mean']:.0f} |")
        lines.append(f"| {k} | oracle (escape ≤ 1%, test labels) | — | {100 * v['oracle|escape_rate']['mean']:.1f}% | "
                     f"{v['oracle|false_rejects_per_1000']['mean']:.0f} | {v['oracle|escapes_per_1000']['mean']:.1f} | "
                     f"{v['oracle|review_items_per_1000']['mean']:.0f} |")
    if argmax_rows:
        lines += ["", "Argmax ARR (P14, no screening), per 1000 at 95 % good: " + ", ".join(
            f"K={k} FR {r['false_rejects_per_1000']:.0f} / escapes {r['escapes_per_1000']:.1f}" for k, r in argmax_rows.items())]
    return lines + [""]


# --------------------------------------------------------------------------- dev

def run_dev(mvtec_root: Path, device) -> int:
    with single_shot(OUT["dev"]) as staging:
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        per = {stat: {} for stat in STATS}
        info = {}
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            good_dir = mvtec_root / product / "train" / "good"
            excluded = set(normal["names"])  # dev-bank images; formal-bank images are reference normals, not test data
            names = sorted(p.name for p in good_dir.glob("*.png") if p.name not in excluded)
            pick = random.Random(f"p15dev|{product}").sample(names, min(2 * N_CAL, len(names)))
            if len(pick) <= N_CAL:
                raise rc.P6BFailClosedError(f"{product}: only {len(pick)} held-out normals")
            cal_names, ev_names = pick[:N_CAL], pick[N_CAL:]
            with torch.no_grad():
                _, extra = ee.extract(factory, [good_dir / n for n in pick], device, 4)
                name_index = {x: i for i, x in enumerate(normal["names"])}
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank = normal["clip18_patches"][idx].to(device).float().reshape(-1, extra.shape[-1])
                    for stat in STATS:
                        s_def = scores_of(dev["clip18_patches"], bank, stat, device)
                        s_all = scores_of(extra, bank, stat, device)
                        s_cal, s_ev = s_all[:N_CAL], s_all[N_CAL:]
                        per[stat][(product, seed)] = {
                            "auroc": sc.auroc(s_ev, s_def),
                            **{f"frr@{a}": statistics.fmean(x > sc.conformal_threshold(s_cal, a) for x in s_ev) for a in sc.ALPHAS},
                            **{f"recall@{a}": statistics.fmean(x > sc.conformal_threshold(s_cal, a) for x in s_def) for a in sc.ALPHAS}}
            info[product] = {"held_out_normals": len(names), "calibration": len(cal_names), "evaluation": len(ev_names),
                             "defects": len(dev["labels"])}
            print(f"[p15-dev] {product} done", flush=True)
        summary = {stat: {k: statistics.fmean(v[k] for v in per[stat].values()) for k in next(iter(per[stat].values()))}
                   for stat in STATS}
        selected = "max" if summary["max"]["auroc"] > summary["top1pct"]["auroc"] else "top1pct"
        result = {"schema_version": "p15_dev_v1", "protocol": "docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md#2",
                  "provenance": rc.git_provenance(), "selected_statistic": selected, "summary": summary, "info": info,
                  "per_product_seed": {stat: {f"{p}/{s}": v for (p, s), v in d.items()} for stat, d in per.items()}}
        lines = ["# P15 dev — screening statistic and conformal calibration (MVTec AD held-out normals vs MVTec-FS support pool)", "",
                 "| Statistic | Image AUROC | FRR @1% / 2% / 5% | Defect recall @1% / 2% / 5% |", "|---|---|---|---|"]
        for stat, v in summary.items():
            lines.append(f"| {stat} | {v['auroc']:.3f} | " + " / ".join(f"{100 * v[f'frr@{a}']:.1f}%" for a in sc.ALPHAS)
                         + " | " + " / ".join(f"{100 * v[f'recall@{a}']:.1f}%" for a in sc.ALPHAS) + " |")
        lines += ["", f"Selected statistic (higher mean AUROC; ties → top1pct): **{selected}**"]
        text = "\n".join(lines)
        rc.write_json(staging / "p15_dev_results.json", result)
        (staging / "p15_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def selected_statistic() -> str:
    path = OUT["dev"] / "p15_dev_results.json"
    if not path.is_file():
        raise rc.P6BFailClosedError("run the P15 dev stage first (the statistic is selected there)")
    return rc.load_json(path)["selected_statistic"]


# --------------------------------------------------------------------------- VisA / MTD

def external_task(task, order, factory, device, batch_size, stat, sink):
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
    for seed in rc.SEEDS:
        banks[seed] = ee.bank_draw(task, seed, normal_candidates, draws[normal][str(seed)]["K5"])
        support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
        if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
            raise rc.P6BFailClosedError(f"{task}/s{seed}: bank/support/query overlap")
        needed |= support_all | set(banks[seed])
        used_normals |= set(banks[seed]) | set(draws[normal][str(seed)]["K5"])
    shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
    index = {s: i for i, s in enumerate(shas)}
    g, clip = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size)  # P14 call
    cal = random.Random(f"p15cal|{task}").sample(sorted(set(normal_candidates) - used_normals - query_shas), N_CAL)
    _, clip_cal = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in cal], device, batch_size)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_shas = [r["rgb_sha256"] for r in queries]
    g = g.to(device)
    units = {}
    for seed in rc.SEEDS:
        bank_imgs = clip[torch.as_tensor([index[s] for s in banks[seed]])].to(device).float()
        bank = bank_imgs.reshape(-1, clip.shape[-1])
        tau = fa.tau99(bank_imgs)
        parts = [ff.c1_descriptors(chunk.to(device).float(), bank_imgs, tau) for chunk in clip.split(ee.Q_CHUNK)]
        per_m = {"": tuple(torch.cat(x) for x in zip(*parts))}
        s_all = scores_of(clip, bank, stat, device, ee.Q_CHUNK)
        s_q = [s_all[i] for i in qi.tolist()]
        s_cal = scores_of(clip_cal, bank, stat, device, ee.Q_CHUNK)
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            si = torch.as_tensor(s_idx, device=device)
            zeros = torch.zeros(len(q_true), len(order), device=device)  # cross-class / consensus not needed here
            scores = ff.episode_scores(g, per_m, si, s_lab, qi, len(order), zeros, zeros)
            for m in FROZEN:
                sink.append({"method": m, "task": task, "K": k_shot, "seed": seed, "rgb_sha256": q_shas,
                             "pred": [order[int(i)] for i in scores[m].argmax(dim=1).cpu()]})
            types = {m: sc.restricted_type(scores[m], order, normal) for m in TYPE_METHODS}
            units[(k_shot, seed)] = decision_metrics(s_q, s_cal, q_true, types, normal)
    return units, {"queries": len(queries), "calibration_normals": len(cal)}


def run_external(device, batch_size) -> int:
    stat = selected_statistic()
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        units, info, sink = {}, {}, []
        for task in ee.TASKS:
            with torch.no_grad():
                u, i = external_task(task, orders[task], factory, device, batch_size, stat, sink)
            units.update({(task, *k): v for k, v in u.items()})
            info[task] = i
            print(f"[p15-ext] {task}: {i['queries']} queries, {i['calibration_normals']} calibration normals", flush=True)
        p14 = {}
        for line in (ff.OUT["external"] / "p14_external_predictions.jsonl").open(encoding="utf-8"):
            u = json.loads(line)
            if u["method"] in FROZEN:
                p14[(u["method"], u["task"], u["K"], u["seed"])] = (u["rgb_sha256"], u["pred"])
        mismatches = [(u["method"], u["task"], u["K"], u["seed"]) for u in sink
                      if p14[(u["method"], u["task"], u["K"], u["seed"])] != (u["rgb_sha256"], u["pred"])]
        if mismatches or len(sink) != len(p14):
            raise rc.P6BFailClosedError(f"frozen predictions not reproduced: {len(mismatches)} units, e.g. {mismatches[:3]}")
        flat = {key: flatten(v) for key, v in units.items()}
        keys = next(iter(flat.values())).keys()
        visa = {str(k): {x: mean_sd(statistics.fmean(flat[(t, k, s)][x] for t in ee.VISA_TASKS) for s in rc.SEEDS) for x in keys}
                for k in rc.K_VALUES}
        mtd = {str(k): {x: mean_sd(flat[(ee.MTD_TASK, k, s)][x] for s in rc.SEEDS) for x in keys} for k in rc.K_VALUES}
        argmax = rc.load_json(ff.OUT["external"] / "p14_external_results.json")["normal_aware"][ff.ARR]
        result = {"schema_version": "p15_external_v1", "protocol": "docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md#3",
                  "label": "user-approved post-P14 industrial analysis; single run", "provenance": provenance,
                  "statistic": stat, "integrity": {"frozen_units_reproduced": len(sink), "mismatches": 0},
                  "task_info": info, "visa_object_macro": visa, "mtd": mtd,
                  "units": [{"task": t, "K": k, "seed": s, **v} for (t, k, s), v in units.items()]}
        lines = [f"# P15 — two-stage inspection on VisA / MTD (single run; statistic = {stat})", "",
                 f"Commit `{provenance['head']}`. Frozen argmax predictions of {', '.join(FROZEN)} reproduced exactly "
                 f"({len(sink)} units). Rates are seed mean; per-1000 at 95 % good.", ""]
        lines += report_lines("VisA (object macro)", visa,
                              {k: argmax[k]["visa_per_1000_at_95pct_good"] for k in map(str, rc.K_VALUES)})
        lines += report_lines("MTD", mtd, {k: argmax[k]["mtd_per_1000_at_95pct_good"] for k in map(str, rc.K_VALUES)})
        text = "\n".join(lines)
        rc.write_json(staging / "p15_external_results.json", result)
        (staging / "p15_external_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- Cable sealed test

def run_cable_test(mvtec_root: Path, device, batch_size) -> int:
    stat = selected_statistic()
    methods = (ff.SS, ff.XC, ff.CONS, ff.NR, ff.A, ff.ARR)
    with single_shot(OUT["cable_test"]) as staging:
        provenance = rc.git_provenance()
        folds = {f: (co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_train.jsonl"),
                     co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_val.jsonl")) for f in co.FOLDS}
        truth = {r["image_path"]: r["class_name"] for f in co.FOLDS for r in folds[f][1]}
        order = co.load_v1(111)[next(iter(truth))]["class_names"]
        pool = sorted(truth)
        test, seen, dropped = [], set(), []
        for r in co.read_jsonl(CABLE_TEST):  # Amendment P15-A1: drop byte-identical duplicate entries
            h = co.sha(co.IMAGE_ROOT / r["image_path"])
            if h in seen:
                dropped.append(r["image_path"])
                continue
            seen.add(h)
            test.append(r)
        test_paths = [r["image_path"] for r in test]
        test_true = [r["class_name"] for r in test]
        normal_dir = mvtec_root / "cable" / "train" / "good"
        normal_names = sorted(p.name for p in normal_dir.glob("*.png"))
        draws = {s: random.Random(f"p10cable|{s}").sample(normal_names, co.BANK) for s in rc.SEEDS}
        if {str(s): v for s, v in draws.items()} != rc.load_json(co.OUT / "cable_results.json")["normal_bank_draws"]:
            raise rc.P6BFailClosedError("normal-bank draws differ from P10 R1")
        bank_names = sorted({n for d in draws.values() for n in d})
        cal = random.Random("p15cal|cable").sample(sorted(set(normal_names) - set(bank_names)), N_CAL)
        test_hash = {co.sha(co.IMAGE_ROOT / p) for p in test_paths}
        other = {co.sha(co.IMAGE_ROOT / p) for p in pool} | {co.sha(normal_dir / n) for n in bank_names + cal}
        if test_hash & other or len(test_hash) != len(test_paths):
            raise rc.P6BFailClosedError("Cable test overlaps pool / bank / calibration images (or duplicates)")
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        with torch.no_grad():
            g_all, clip_all = ee.extract(factory, [co.IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names],
                                         device, batch_size)  # the P14 Cable call
            g_t, clip_t = ee.extract(factory, [co.IMAGE_ROOT / p for p in test_paths], device, batch_size)
            _, clip_cal = ee.extract(factory, [normal_dir / n for n in cal], device, batch_size)
        n_pool = len(pool)
        index = {p: i for i, p in enumerate(pool)}
        bank_index = {n: n_pool + i for i, n in enumerate(bank_names)}
        clip_pool = clip_all[:n_pool].to(device).float()
        g = torch.cat([g_all[:n_pool], g_t]).to(device)
        clip = torch.cat([clip_pool, clip_t.to(device).float()])
        n = len(order)
        si_all = torch.arange(n_pool, device=device)
        qi_test = torch.arange(n_pool, n_pool + len(test_paths), device=device)
        oof_preds = {m: {} for m in FROZEN}
        test_preds = {m: {} for m in methods}
        units = {}
        with torch.no_grad():
            for seed in rc.SEEDS:
                bank_imgs = clip_all[[bank_index[x] for x in draws[seed]]].to(device).float()
                bank = bank_imgs.reshape(-1, clip.shape[-1])
                tau = fa.tau99(bank_imgs)
                # integrity: the P14 Cable OOF path on the pool alone
                per_pool = {"": ff.c1_descriptors(clip_pool, bank_imgs, tau)}
                g_pool = g_all[:n_pool].to(device)
                oof_preds_seed = {m: {} for m in FROZEN}
                for f in co.FOLDS:
                    train, val = folds[f]
                    si = torch.as_tensor([index[r["image_path"]] for r in train], device=device)
                    sl = [order.index(r["class_name"]) for r in train]
                    qi = torch.as_tensor([index[r["image_path"]] for r in val], device=device)
                    zeros = torch.zeros(len(val), n, device=device)
                    s_oof = ff.episode_scores(g_pool, per_pool, si, sl, qi, n, zeros, zeros)
                    for m in FROZEN:
                        for r, c in zip(val, s_oof[m].argmax(dim=1).cpu().tolist()):
                            oof_preds_seed[m][r["image_path"]] = order[c]
                for m in FROZEN:
                    oof_preds[m][seed] = oof_preds_seed[m]
                # sealed test: supports = the whole 72-image pool
                per_m = {"": ff.c1_descriptors(clip, bank_imgs, tau)}
                sl_all = [order.index(truth[p]) for p in pool]
                xc = rp.local_xclass(clip[si_all], sl_all, clip[qi_test], n)
                cons = p13.consensus_scores(clip, si_all, sl_all, qi_test, n)[ff.CONS]
                scores = ff.episode_scores(g, per_m, si_all, sl_all, qi_test, n, xc, cons)
                for m in methods:
                    test_preds[m][seed] = [order[int(i)] for i in scores[m].argmax(dim=1).cpu()]
                s_all = scores_of(clip[qi_test].cpu(), bank, stat, device)
                s_cal = scores_of(clip_cal, bank, stat, device)
                types = {m: sc.restricted_type(scores[m], order, co.NORMAL) for m in TYPE_METHODS}
                units[seed] = decision_metrics(s_all, s_cal, test_true, types, co.NORMAL)
        old = {}
        for line in (ff.OUT["cable"] / "p14_cable_predictions.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["method"] in FROZEN:
                old[(r["method"], int(r["seed"]), r["image_path"])] = r["pred"]
        mismatch = [(m, s, i) for m in FROZEN for s in rc.SEEDS for i in pool if oof_preds[m][s][i] != old[(m, s, i)]]
        if mismatch:
            raise rc.P6BFailClosedError(f"Cable OOF predictions not reproduced: {len(mismatch)}, e.g. {mismatch[:3]}")
        classification = {}
        for m in methods:
            per_seed = [{**classification_metrics(test_true, test_preds[m][s], order),
                         **normal_aware(test_true, test_preds[m][s], co.NORMAL)} for s in rc.SEEDS]
            classification[m] = {k: mean_sd(v[k] for v in per_seed) for k in per_seed[0] if not k.startswith("n_")}
        flat = {s: flatten(v) for s, v in units.items()}
        agg = {"8": {x: mean_sd(flat[s][x] for s in rc.SEEDS) for x in next(iter(flat.values()))}}
        result = {"schema_version": "p15_cable_test_v1", "protocol": "docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md#3",
                  "label": "sealed Cable test, first and only read; supports = 72-image dev pool (K=8)",
                  "provenance": provenance, "statistic": stat,
                  "test_composition": {c: test_true.count(c) for c in sorted(set(test_true))},
                  "duplicates_dropped_P15_A1": dropped,
                  "integrity": {"oof_predictions_reproduced": len(FROZEN) * len(rc.SEEDS) * len(pool), "mismatches": 0},
                  "classification_argmax": classification, "two_stage": agg,
                  "units": {str(s): v for s, v in units.items()},
                  "predictions": {m: {str(s): v for s, v in d.items()} for m, d in test_preds.items()}, "test_paths": test_paths}

        def cell(m, k):
            v = classification[m][k]
            return f"{100 * v['mean']:.1f} ± {100 * v['sd']:.1f}"
        n_good = test_true.count(co.NORMAL)
        lines = [f"# P15 — sealed Cable test ({len(test_true)} unique images: {n_good} good, {len(test_true) - n_good} defects; "
                 f"first read; statistic = {stat})", "",
                 f"Amendment P15-A1: byte-identical duplicate manifest entries dropped: {dropped or 'none'}.", "",
                 f"Commit `{provenance['head']}`. Cable OOF predictions of {', '.join(FROZEN)} reproduce P14 exactly. "
                 "Supports: whole 72-image dev pool (K = 8); 5 bank seeds.", "",
                 "## Classifiers alone (argmax, 9 classes incl. good)", "",
                 "| Method | 9-class BA | Good recall | Defect recall | Binary BA | Type acc. (detected) |", "|---|---|---|---|---|---|"]
        for m in methods:
            lines.append(f"| {m} | {cell(m, 'balanced_accuracy')} | {cell(m, 'good_recall')} | {cell(m, 'defect_recall')} | "
                         f"{cell(m, 'binary_ba')} | {cell(m, 'defect_type_acc_detected')} |")
        lines += [""] + report_lines("Two-stage (screening + ARR type), K = 8", agg)
        text = "\n".join(lines)
        rc.write_json(staging / "p15_cable_test_results.json", result)
        (staging / "p15_cable_test_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "external", "cable-test"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "external":
        return run_external(device, args.batch_size)
    if not args.mvtec_root:
        raise SystemExit("--mvtec-root is required for this stage")
    root = Path(args.mvtec_root)
    return run_dev(root, device) if args.stage == "dev" else run_cable_test(root, device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
