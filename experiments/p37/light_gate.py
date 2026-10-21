#!/usr/bin/env python3
"""P37: light gate -- DINOv2-S screening ahead of the CLIP-L/14 classifier (docs/P37_LIGHT_GATE_PROTOCOL.md).

Every stage recomputes P15's CLIP layer-18 gate from the same features and must reproduce P15's saved results exactly
(dev per product and seed; VisA / MTD and Cable per unit) before anything is written. It then computes the same metrics
with the DINOv2-S screening score s_D: the main configuration's own DINOv2-S patch tokens (the tensor DINOv2-S already
receives), P15's `top1pct` statistic, the same M = 20 bank images, calibration normals, alphas and review bands, and the
same type classifier of rejected parts. `--stage report` applies the protocol's stop and recommendation rules.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p37.light_gate --stage dev --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p37.light_gate --stage external --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p37.light_gate --stage cable-test --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p37.light_gate --stage report
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p10 import cable_oof as co  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p15 import screening as sc  # noqa: E402
from experiments.p15 import two_stage as ts  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p37" / s for s in ("dev", "external", "cable_test", "report")}
GUARDED = (*ts.GUARDED, "experiments/p37", "docs/P37_LIGHT_GATE_PROTOCOL.md")
STAT = "top1pct"            # P15's selected statistic; not re-selected (protocol §1)
GATES = ("clip_l18", "dino_s")
DEV_STOP_MARGIN = 0.05      # protocol §2
RECOMMEND_MARGIN = 0.02     # protocol §5
GOOD_RATES = (0.95, 0.99, 0.999)


@contextlib.contextmanager
def single_shot(out: Path):
    """experiments.p15.two_stage.single_shot with the P37 paths guarded as well."""
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; the P37 stage is single-shot")
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


def extract_with_dino(factory, paths, device, batch_size):
    """external_evaluate.extract (the P14 / P15 call, unchanged) that also returns the DINOv2-S patch tokens of the
    same forward pass, L2-normalised in FP32 and stored in half like the CLIP tokens (as the P10 caches store them)."""
    glob, clip, dino_t = [], [], []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
        batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
        with torch.no_grad():
            pooled, tokens = factory.clip_model.visual(batch, [CLIP_LAYER])
            dino = factory.dino_extractor.forward_patches(batch)
            glob.append(factory.fusion.fuse(models.l2_rows(pooled),
                                            models.l2_rows(dino.mean(dim=1))).float().cpu())
            clip.append(F.normalize(tokens[0][:, 1:, :].float(), dim=-1).half().cpu())
            dino_t.append(F.normalize(dino.float(), dim=-1).half().cpu())
    return torch.cat(glob), torch.cat(clip), torch.cat(dino_t)


def same(a, b) -> bool:
    """Exact equality of nested results (dict keys as strings, NaN == NaN), after a JSON round trip of `a`."""
    def norm(x):
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x

    def eq(x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y))
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return True
        return x == y
    return eq(json.loads(json.dumps(norm(a))), norm(b))


def pass_rates(s_q, truth, normal, taus: dict) -> dict:
    """Share of normal / defect queries that stop at the gate (s <= tau), for every threshold of the unit."""
    good = [x for x, t in zip(s_q, truth) if t == normal]
    bad = [x for x, t in zip(s_q, truth) if t != normal]
    return {a: {"normal": statistics.fmean(x <= t for x in good) if good else float("nan"),
                "defect": statistics.fmean(x <= t for x in bad) if bad else float("nan")} for a, t in taus.items()}


def gate_units(s_q: dict, s_cal: dict, truth, types, normal) -> dict:
    out = {}
    for gate in GATES:
        u = ts.decision_metrics(s_q[gate], s_cal[gate], truth, types, normal)
        u["pass_rates"] = pass_rates(s_q[gate], truth, normal, u["taus"])
        out[gate] = u
    return out


def without_pass_rates(u: dict) -> dict:
    return {k: v for k, v in u.items() if k != "pass_rates"}


# --------------------------------------------------------------------------- dev

def run_dev(mvtec_root: Path, device) -> int:
    if ts.selected_statistic() != STAT:
        raise rc.P6BFailClosedError("P15 did not select top1pct")
    p15 = rc.load_json(ts.OUT["dev"] / "p15_dev_results.json")["per_product_seed"][STAT]
    with single_shot(OUT["dev"]) as staging:
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        per = {g: {} for g in GATES}
        info = {}
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            good_dir = mvtec_root / product / "train" / "good"
            excluded = set(normal["names"])
            names = sorted(p.name for p in good_dir.glob("*.png") if p.name not in excluded)
            pick = random.Random(f"p15dev|{product}").sample(names, min(2 * ts.N_CAL, len(names)))
            if len(pick) <= ts.N_CAL:
                raise rc.P6BFailClosedError(f"{product}: only {len(pick)} held-out normals")
            with torch.no_grad():
                _, extra, extra_d = extract_with_dino(factory, [good_dir / n for n in pick], device, 4)
                name_index = {x: i for i, x in enumerate(normal["names"])}
                sources = {"clip_l18": (dev["clip18_patches"], normal["clip18_patches"], extra),
                           "dino_s": (dev["dino_patches"], normal["dino_patches"], extra_d)}
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                    for gate, (defects, bank_src, held) in sources.items():
                        bank = bank_src[idx].to(device).float().reshape(-1, held.shape[-1])
                        s_def = ts.scores_of(defects, bank, STAT, device)
                        s_all = ts.scores_of(held, bank, STAT, device)
                        s_cal, s_ev = s_all[:ts.N_CAL], s_all[ts.N_CAL:]
                        per[gate][(product, seed)] = {
                            "auroc": sc.auroc(s_ev, s_def),
                            **{f"frr@{a}": statistics.fmean(x > sc.conformal_threshold(s_cal, a) for x in s_ev)
                               for a in sc.ALPHAS},
                            **{f"recall@{a}": statistics.fmean(x > sc.conformal_threshold(s_cal, a) for x in s_def)
                               for a in sc.ALPHAS}}
                    if not same(per["clip_l18"][(product, seed)], p15[f"{product}/{seed}"]):
                        raise rc.P6BFailClosedError(f"P15 dev not reproduced at {product}/{seed}")
            info[product] = {"calibration": ts.N_CAL, "evaluation": len(pick) - ts.N_CAL, "defects": len(dev["labels"])}
            print(f"[p37-dev] {product} done", flush=True)
        summary = {g: {k: statistics.fmean(v[k] for v in per[g].values()) for k in next(iter(per[g].values()))}
                   for g in GATES}
        stop = summary["dino_s"]["auroc"] < summary["clip_l18"]["auroc"] - DEV_STOP_MARGIN
        result = {"schema_version": "p37_dev_v1", "protocol": "docs/P37_LIGHT_GATE_PROTOCOL.md#2",
                  "provenance": rc.git_provenance(), "statistic": STAT,
                  "integrity": {"p15_dev_reproduced": len(per["clip_l18"]), "mismatches": 0},
                  "summary": summary, "stop_rule": {"margin": DEV_STOP_MARGIN, "stop": stop}, "info": info,
                  "per_product_seed": {g: {f"{p}/{s}": v for (p, s), v in d.items()} for g, d in per.items()}}
        lines = ["# P37 dev — light gate (DINOv2-S) vs P15 CLIP layer-18 gate (MVTec AD held-out normals vs MVTec-FS "
                 "support pool)", "", f"P15 dev reproduced exactly ({len(per['clip_l18'])} product × seed units).", "",
                 "| Gate | Image AUROC | FRR @1% / 2% / 5% | Defect recall @1% / 2% / 5% |", "|---|---|---|---|"]
        for g, v in summary.items():
            lines.append(f"| {g} | {v['auroc']:.3f} | " + " / ".join(f"{100 * v[f'frr@{a}']:.1f}%" for a in sc.ALPHAS)
                         + " | " + " / ".join(f"{100 * v[f'recall@{a}']:.1f}%" for a in sc.ALPHAS) + " |")
        lines += ["", f"Stop rule (DINOv2-S AUROC more than {DEV_STOP_MARGIN} below CLIP): "
                      f"**{'STOP — no test stage' if stop else 'continue to the test stages'}**"]
        text = "\n".join(lines)
        rc.write_json(staging / "p37_dev_results.json", result)
        (staging / "p37_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def dev_allows_tests() -> None:
    path = OUT["dev"] / "p37_dev_results.json"
    if not path.is_file():
        raise rc.P6BFailClosedError("run the P37 dev stage first")
    if rc.load_json(path)["stop_rule"]["stop"]:
        raise rc.P6BFailClosedError("the P37 dev stop rule fired: no test stage runs (protocol §2)")


# --------------------------------------------------------------------------- VisA / MTD

def external_task(task, order, factory, device, batch_size, sink):
    """experiments.p15.two_stage.external_task with the DINOv2-S gate computed from the same forward pass."""
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
    g, clip, dino = extract_with_dino(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device,
                                      batch_size)
    cal = random.Random(f"p15cal|{task}").sample(sorted(set(normal_candidates) - used_normals - query_shas), ts.N_CAL)
    _, clip_cal, dino_cal = extract_with_dino(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in cal],
                                              device, batch_size)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_shas = [r["rgb_sha256"] for r in queries]
    g = g.to(device)
    units = {}
    for seed in rc.SEEDS:
        bank_idx = torch.as_tensor([index[s] for s in banks[seed]])
        bank_imgs = clip[bank_idx].to(device).float()
        bank = bank_imgs.reshape(-1, clip.shape[-1])
        bank_d = dino[bank_idx].to(device).float().reshape(-1, dino.shape[-1])
        tau = fa.tau99(bank_imgs)
        parts = [ff.c1_descriptors(chunk.to(device).float(), bank_imgs, tau) for chunk in clip.split(ee.Q_CHUNK)]
        per_m = {"": tuple(torch.cat(x) for x in zip(*parts))}
        s_all = {"clip_l18": ts.scores_of(clip, bank, STAT, device, ee.Q_CHUNK),
                 "dino_s": ts.scores_of(dino, bank_d, STAT, device, ee.Q_CHUNK)}
        s_q = {k: [v[i] for i in qi.tolist()] for k, v in s_all.items()}
        s_cal = {"clip_l18": ts.scores_of(clip_cal, bank, STAT, device, ee.Q_CHUNK),
                 "dino_s": ts.scores_of(dino_cal, bank_d, STAT, device, ee.Q_CHUNK)}
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            si = torch.as_tensor(s_idx, device=device)
            zeros = torch.zeros(len(q_true), len(order), device=device)
            scores = ff.episode_scores(g, per_m, si, s_lab, qi, len(order), zeros, zeros)
            for m in ts.FROZEN:
                sink.append({"method": m, "task": task, "K": k_shot, "seed": seed, "rgb_sha256": q_shas,
                             "pred": [order[int(i)] for i in scores[m].argmax(dim=1).cpu()]})
            types = {m: sc.restricted_type(scores[m], order, normal) for m in ts.TYPE_METHODS}
            units[(k_shot, seed)] = gate_units(s_q, s_cal, q_true, types, normal)
    return units, {"queries": len(queries), "calibration_normals": len(cal)}


def aggregate_external(units: dict, gate: str) -> tuple[dict, dict]:
    flat = {key: ts.flatten(without_pass_rates(v[gate])) for key, v in units.items()}
    for key, v in units.items():
        for a, pr in v[gate]["pass_rates"].items():
            flat[key][f"pass|a={a}|normal"] = pr["normal"]
            flat[key][f"pass|a={a}|defect"] = pr["defect"]
    keys = next(iter(flat.values())).keys()
    visa = {str(k): {x: ts.mean_sd(statistics.fmean(flat[(t, k, s)][x] for t in ee.VISA_TASKS) for s in rc.SEEDS)
                     for x in keys} for k in rc.K_VALUES}
    mtd = {str(k): {x: ts.mean_sd(flat[(ee.MTD_TASK, k, s)][x] for s in rc.SEEDS) for x in keys} for k in rc.K_VALUES}
    return visa, mtd


def run_external(device, batch_size) -> int:
    dev_allows_tests()
    p15 = rc.load_json(ts.OUT["external"] / "p15_external_results.json")
    p15_units = {(u["task"], u["K"], u["seed"]): {k: v for k, v in u.items() if k not in ("task", "K", "seed")}
                 for u in p15["units"]}
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        units, info, sink = {}, {}, []
        for task in ee.TASKS:
            with torch.no_grad():
                u, i = external_task(task, orders[task], factory, device, batch_size, sink)
            for key, v in u.items():
                if not same(without_pass_rates(v["clip_l18"]), p15_units[(task, *key)]):
                    raise rc.P6BFailClosedError(f"P15 external unit not reproduced: {task} K={key[0]} seed={key[1]}")
            units.update({(task, *k): v for k, v in u.items()})
            info[task] = i
            print(f"[p37-ext] {task}: {i['queries']} queries, P15 units reproduced", flush=True)
        if len(units) != len(p15_units):
            raise rc.P6BFailClosedError(f"{len(units)} units vs {len(p15_units)} in P15")
        p14 = {}
        for line in (ff.OUT["external"] / "p14_external_predictions.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["method"] in ts.FROZEN:
                p14[(r["method"], r["task"], r["K"], r["seed"])] = (r["rgb_sha256"], r["pred"])
        mismatches = [(u["method"], u["task"], u["K"], u["seed"]) for u in sink
                      if p14[(u["method"], u["task"], u["K"], u["seed"])] != (u["rgb_sha256"], u["pred"])]
        if mismatches or len(sink) != len(p14):
            raise rc.P6BFailClosedError(f"frozen predictions not reproduced: {len(mismatches)} units")
        agg = {gate: dict(zip(("visa_object_macro", "mtd"), aggregate_external(units, gate))) for gate in GATES}
        result = {"schema_version": "p37_external_v1", "protocol": "docs/P37_LIGHT_GATE_PROTOCOL.md#3",
                  "provenance": provenance, "statistic": STAT,
                  "integrity": {"p15_units_reproduced": len(units), "frozen_predictions_reproduced": len(sink)},
                  "task_info": info, "aggregate": agg,
                  "units": [{"task": t, "K": k, "seed": s, **v} for (t, k, s), v in units.items()]}
        lines = ["# P37 — light gate on VisA / MTD (single run)", "",
                 f"Commit `{provenance['head']}`. P15 units reproduced exactly ({len(units)}); frozen predictions "
                 f"reproduced ({len(sink)} units). Rates are seed means; per-1000 at 95 % good.", ""]
        for gate in GATES:
            lines += ts.report_lines(f"VisA (object macro) — gate {gate}", agg[gate]["visa_object_macro"])
            lines += ts.report_lines(f"MTD — gate {gate}", agg[gate]["mtd"])
        text = "\n".join(lines)
        rc.write_json(staging / "p37_external_results.json", result)
        (staging / "p37_external_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- Cable held-out split

def run_cable_test(mvtec_root: Path, device, batch_size) -> int:
    """experiments.p15.two_stage.run_cable_test (screening part) with the DINOv2-S gate added."""
    dev_allows_tests()
    p15 = rc.load_json(ts.OUT["cable_test"] / "p15_cable_test_results.json")
    with single_shot(OUT["cable_test"]) as staging:
        provenance = rc.git_provenance()
        folds = {f: (co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_train.jsonl"),
                     co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_val.jsonl")) for f in co.FOLDS}
        truth = {r["image_path"]: r["class_name"] for f in co.FOLDS for r in folds[f][1]}
        order = co.load_v1(111)[next(iter(truth))]["class_names"]
        pool = sorted(truth)
        test, seen = [], set()
        for r in co.read_jsonl(ts.CABLE_TEST):  # Amendment P15-A1
            h = co.sha(co.IMAGE_ROOT / r["image_path"])
            if h in seen:
                continue
            seen.add(h)
            test.append(r)
        test_paths = [r["image_path"] for r in test]
        test_true = [r["class_name"] for r in test]
        if test_paths != p15["test_paths"]:
            raise rc.P6BFailClosedError("Cable test list differs from P15")
        normal_dir = mvtec_root / "cable" / "train" / "good"
        normal_names = sorted(p.name for p in normal_dir.glob("*.png"))
        draws = {s: random.Random(f"p10cable|{s}").sample(normal_names, co.BANK) for s in rc.SEEDS}
        if {str(s): v for s, v in draws.items()} != rc.load_json(co.OUT / "cable_results.json")["normal_bank_draws"]:
            raise rc.P6BFailClosedError("normal-bank draws differ from P10 R1")
        bank_names = sorted({n for d in draws.values() for n in d})
        cal = random.Random("p15cal|cable").sample(sorted(set(normal_names) - set(bank_names)), ts.N_CAL)
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        with torch.no_grad():
            g_all, clip_all, dino_all = extract_with_dino(
                factory, [co.IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names], device, batch_size)
            g_t, clip_t, dino_t = extract_with_dino(factory, [co.IMAGE_ROOT / p for p in test_paths], device, batch_size)
            _, clip_cal, dino_cal = extract_with_dino(factory, [normal_dir / n for n in cal], device, batch_size)
        n_pool = len(pool)
        bank_index = {n: n_pool + i for i, n in enumerate(bank_names)}
        g = torch.cat([g_all[:n_pool], g_t]).to(device)
        clip = torch.cat([clip_all[:n_pool].to(device).float(), clip_t.to(device).float()])
        n = len(order)
        si_all = torch.arange(n_pool, device=device)
        qi_test = torch.arange(n_pool, n_pool + len(test_paths), device=device)
        units = {}
        with torch.no_grad():
            for seed in rc.SEEDS:
                rows = [bank_index[x] for x in draws[seed]]
                bank_imgs = clip_all[rows].to(device).float()
                bank = bank_imgs.reshape(-1, clip.shape[-1])
                bank_d = dino_all[rows].to(device).float().reshape(-1, dino_all.shape[-1])
                tau = fa.tau99(bank_imgs)
                per_m = {"": ff.c1_descriptors(clip, bank_imgs, tau)}
                sl_all = [order.index(truth[p]) for p in pool]
                zeros = torch.zeros(len(test_paths), n, device=device)   # cross-class / consensus not needed here
                scores = ff.episode_scores(g, per_m, si_all, sl_all, qi_test, n, zeros, zeros)
                s_q = {"clip_l18": ts.scores_of(clip[qi_test].cpu(), bank, STAT, device),
                       "dino_s": ts.scores_of(dino_t, bank_d, STAT, device)}
                s_cal = {"clip_l18": ts.scores_of(clip_cal, bank, STAT, device),
                         "dino_s": ts.scores_of(dino_cal, bank_d, STAT, device)}
                types = {m: sc.restricted_type(scores[m], order, co.NORMAL) for m in ts.TYPE_METHODS}
                units[seed] = gate_units(s_q, s_cal, test_true, types, co.NORMAL)
                if not same(without_pass_rates(units[seed]["clip_l18"]), p15["units"][str(seed)]):
                    raise rc.P6BFailClosedError(f"P15 Cable unit not reproduced: seed {seed}")
        agg = {}
        for gate in GATES:
            flat = {s: ts.flatten(without_pass_rates(v[gate])) for s, v in units.items()}
            for s, v in units.items():
                for a, pr in v[gate]["pass_rates"].items():
                    flat[s][f"pass|a={a}|normal"] = pr["normal"]
                    flat[s][f"pass|a={a}|defect"] = pr["defect"]
            agg[gate] = {"8": {x: ts.mean_sd(flat[s][x] for s in rc.SEEDS) for x in next(iter(flat.values()))}}
        result = {"schema_version": "p37_cable_test_v1", "protocol": "docs/P37_LIGHT_GATE_PROTOCOL.md#3",
                  "provenance": provenance, "statistic": STAT,
                  "integrity": {"p15_units_reproduced": len(units)},
                  "test_composition": {c: test_true.count(c) for c in sorted(set(test_true))},
                  "aggregate": agg, "units": {str(s): v for s, v in units.items()}}
        lines = [f"# P37 — light gate on the Cable held-out split ({len(test_true)} images; single run)", "",
                 f"Commit `{provenance['head']}`. P15 Cable units reproduced exactly ({len(units)} seeds).", ""]
        for gate in GATES:
            lines += ts.report_lines(f"Cable, K = 8 — gate {gate}", agg[gate])
        text = "\n".join(lines)
        rc.write_json(staging / "p37_cable_test_results.json", result)
        (staging / "p37_cable_test_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- report (protocol §4–5)

def run_report() -> int:
    dev = rc.load_json(OUT["dev"] / "p37_dev_results.json")
    if dev["stop_rule"]["stop"]:
        raise SystemExit("dev stop rule fired; the dev table is the P37 result")
    ext = rc.load_json(OUT["external"] / "p37_external_results.json")["aggregate"]
    cab = rc.load_json(OUT["cable_test"] / "p37_cable_test_results.json")["aggregate"]
    sets = {"VisA": {g: ext[g]["visa_object_macro"]["5"] for g in GATES},
            "MTD": {g: ext[g]["mtd"]["5"] for g in GATES},
            "Cable": {g: cab[g]["8"] for g in GATES}}
    # the screening score does not depend on K; AUROC is identical across K (checked here)
    for name, src in (("VisA", lambda g, k: ext[g]["visa_object_macro"][k]), ("MTD", lambda g, k: ext[g]["mtd"][k])):
        for g in GATES:
            if len({round(src(g, str(k))["auroc"]["mean"], 12) for k in rc.K_VALUES}) != 1:
                raise rc.P6BFailClosedError(f"{name}/{g}: AUROC differs across K")
    auroc = {d: {g: v[g]["auroc"]["mean"] for g in GATES} for d, v in sets.items()}
    recommend = all(a["dino_s"] >= a["clip_l18"] - RECOMMEND_MARGIN for a in auroc.values())
    p36 = PROJECT_ROOT / "ckpt" / "p36" / "jetson_orin_nano" / "p36_results.json"
    latency = None
    if p36.is_file():
        summ = rc.load_json(p36)["summary"]

        def med(block, key):
            vals = [summ[block][p][key] for p in summ[block]]
            return statistics.fmean(vals)
        t = {"clip_l18": (med("gate_l18_fp16", "e2e_pass_ms"), med("gate_l18_fp16", "e2e_flagged_ms")),
             "dino_s": (med("gate_dino_fp16", "e2e_pass_ms"), med("gate_dino_fp16", "e2e_flagged_ms")),
             "none": med("gate_l18_fp16", "e2e_full_ms")}
        latency = {}
        for d, v in sets.items():
            latency[d] = {}
            for g in GATES:
                for a in [*map(str, sc.ALPHAS), *map(str, sc.ALPHA_PASS)]:
                    pn, pd = v[g][f"pass|a={a}|normal"]["mean"], v[g][f"pass|a={a}|defect"]["mean"]
                    for good in GOOD_RATES:
                        p_pass = good * pn + (1 - good) * pd
                        latency[d][f"{g}|a={a}|good={good}"] = p_pass * t[g][0] + (1 - p_pass) * t[g][1]
            latency[d]["none"] = t["none"]
    out = OUT["report"]
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    rc.write_json(out / "p37_report.json", {"auroc": auroc, "recommend_light_gate": recommend,
                                            "margin": RECOMMEND_MARGIN, "expected_latency_ms": latency,
                                            "p36_source": str(p36) if latency else None})
    lines = ["# P37 report — recommendation rule (protocol §5)", "",
             "| Set | AUROC CLIP layer 18 | AUROC DINOv2-S | DINOv2-S ≥ CLIP − 0.02 |", "|---|---|---|---|"]
    for d, a in auroc.items():
        lines.append(f"| {d} | {a['clip_l18']:.3f} | {a['dino_s']:.3f} | {a['dino_s'] >= a['clip_l18'] - RECOMMEND_MARGIN} |")
    lines += ["", f"**Light gate recommended as the deployment default: {recommend}**", ""]
    if latency:
        lines += ["Expected per-part latency on the Jetson Orin Nano (P36 FP16 medians; ms):", ""]
        for d, v in latency.items():
            lines.append(f"- {d} (99 % good): no gate {v['none']:.1f}; " + "; ".join(
                f"{k.rsplit('|', 1)[0]} {x:.1f}" for k, x in v.items() if k.endswith("|good=0.99")))
    (out / "p37_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "external", "cable-test", "report"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "external":
        return run_external(device, args.batch_size)
    if args.stage == "report":
        return run_report()
    if not args.mvtec_root:
        raise SystemExit("--mvtec-root is required for this stage")
    root = Path(args.mvtec_root)
    return run_dev(root, device) if args.stage == "dev" else run_cable_test(root, device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
