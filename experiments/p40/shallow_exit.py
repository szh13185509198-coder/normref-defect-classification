#!/usr/bin/env python3
"""P40: earlier exit of good parts -- the P15 screening score at a shallower CLIP layer (docs/P40_SHALLOW_EXIT_PROTOCOL.md).

Every stage recomputes P15's layer-18 results from the same code path and must reproduce them exactly before writing;
the layer-k score uses the same images, banks, calibration normals and decision layer, only the token layer changes.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p40.shallow_exit --stage dev --mvtec-root D:\\intergration\\MVTEC
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p40.shallow_exit --stage external
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p40.shallow_exit --stage cable-test --mvtec-root D:\\intergration\\MVTEC
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p40.shallow_exit --stage report
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
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
from experiments.p37 import light_gate as lg  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p40" / s for s in ("dev", "external", "cable_test", "report")}
GUARDED = (*ts.GUARDED, "experiments/p37", "experiments/p40", "docs/P40_SHALLOW_EXIT_PROTOCOL.md")
STAT = "top1pct"
CANDIDATES = (6, 9, 12, 15)
DEV_MARGIN = 0.01        # protocol §2
RECOMMEND_MARGIN = 0.02  # protocol §5
GOOD_RATES = (0.95, 0.99, 0.999)


def single_shot(out: Path):
    lg.GUARDED = GUARDED   # same lock / staging / clean-tree logic as P37, with the P40 paths guarded
    return lg.single_shot(out)


def extract_layers(factory, paths, device, batch_size, layers):
    """external_evaluate.extract (the P14 / P15 call, unchanged numerics) returning the tokens of several CLIP layers:
    (global fusion [N, 1152] float, {layer: L2-normalised patch tokens [N, 576, 1024] half})."""
    layers = sorted(set(layers))
    glob, toks = [], {L: [] for L in layers}
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
        batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
        with torch.no_grad():
            pooled, tokens = factory.clip_model.visual(batch, layers)
            dino = factory.dino_extractor.forward_patches(batch)
            glob.append(factory.fusion.fuse(models.l2_rows(pooled), models.l2_rows(dino.mean(dim=1))).float().cpu())
            for L, t in zip(layers, tokens):
                toks[L].append(F.normalize(t[:, 1:, :].float(), dim=-1).half().cpu())
    return torch.cat(glob), {L: torch.cat(v) for L, v in toks.items()}


def dev_metrics(s_ev, s_def, s_cal) -> dict:
    return {"auroc": sc.auroc(s_ev, s_def),
            **{f"frr@{a}": statistics.fmean(x > sc.conformal_threshold(s_cal, a) for x in s_ev) for a in sc.ALPHAS},
            **{f"recall@{a}": statistics.fmean(x > sc.conformal_threshold(s_cal, a) for x in s_def) for a in sc.ALPHAS}}


# --------------------------------------------------------------------------- dev

def run_dev(mvtec_root: Path, device) -> int:
    if ts.selected_statistic() != STAT:
        raise rc.P6BFailClosedError("P15 did not select top1pct")
    p15 = rc.load_json(ts.OUT["dev"] / "p15_dev_results.json")["per_product_seed"][STAT]
    layers_all = (*CANDIDATES, CLIP_LAYER)
    with single_shot(OUT["dev"]) as staging:
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        per = {L: {} for L in layers_all}
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            good_dir = mvtec_root / product / "train" / "good"
            names = sorted(p.name for p in good_dir.glob("*.png") if p.name not in set(normal["names"]))
            pick = random.Random(f"p15dev|{product}").sample(names, min(2 * ts.N_CAL, len(names)))
            with torch.no_grad():
                _, held = extract_layers(factory, [good_dir / n for n in pick], device, 4, layers_all)  # P15 call
                _, defects = extract_layers(factory, [rc.DATASET_ROOT / r for r in dev["rel_paths"]], device, 4, CANDIDATES)
                _, banks = extract_layers(factory, [good_dir / n for n in normal["names"]], device, 4, CANDIDATES)
                name_index = {x: i for i, x in enumerate(normal["names"])}
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                    for L in layers_all:
                        if L == CLIP_LAYER:   # exactly P15's sources: cached defects and bank, fresh held-out normals
                            d_src, b_src = dev["clip18_patches"], normal["clip18_patches"]
                        else:
                            d_src, b_src = defects[L], banks[L]
                        bank = b_src[idx].to(device).float().reshape(-1, 1024)
                        s_all = ts.scores_of(held[L], bank, STAT, device)
                        per[L][(product, seed)] = dev_metrics(s_all[ts.N_CAL:], ts.scores_of(d_src, bank, STAT, device),
                                                              s_all[:ts.N_CAL])
                    if not lg.same(per[CLIP_LAYER][(product, seed)], p15[f"{product}/{seed}"]):
                        raise rc.P6BFailClosedError(f"P15 dev not reproduced at {product}/{seed}")
            print(f"[p40-dev] {product} done", flush=True)
        summary = {L: {k: statistics.fmean(v[k] for v in per[L].values()) for k in next(iter(per[L].values()))}
                   for L in layers_all}
        ok = [L for L in CANDIDATES if summary[L]["auroc"] >= summary[CLIP_LAYER]["auroc"] - DEV_MARGIN]
        selected = min(ok) if ok else None
        result = {"schema_version": "p40_dev_v1", "protocol": "docs/P40_SHALLOW_EXIT_PROTOCOL.md#2",
                  "provenance": rc.git_provenance(), "integrity": {"p15_dev_reproduced": len(per[CLIP_LAYER])},
                  "summary": {str(L): v for L, v in summary.items()}, "selected_layer": selected,
                  "per_product_seed": {str(L): {f"{p}/{s}": v for (p, s), v in d.items()} for L, d in per.items()}}
        lines = ["# P40 dev — screening AUROC by CLIP layer (MVTec AD held-out normals vs MVTec-FS support pool)", "",
                 f"P15 dev (layer 18) reproduced exactly ({len(per[CLIP_LAYER])} units).", "",
                 "| Layer | Image AUROC | FRR @1% / 2% / 5% | Defect recall @1% / 2% / 5% |", "|---|---|---|---|"]
        for L in layers_all:
            v = summary[L]
            lines.append(f"| {L} | {v['auroc']:.3f} | " + " / ".join(f"{100 * v[f'frr@{a}']:.1f}%" for a in sc.ALPHAS)
                         + " | " + " / ".join(f"{100 * v[f'recall@{a}']:.1f}%" for a in sc.ALPHAS) + " |")
        lines += ["", f"Selection (shallowest layer with AUROC ≥ layer 18 − {DEV_MARGIN}): **"
                  + (f"layer {selected}" if selected else "none — P40 stops after dev") + "**"]
        text = "\n".join(lines)
        rc.write_json(staging / "p40_dev_results.json", result)
        (staging / "p40_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def selected_layer() -> int:
    path = OUT["dev"] / "p40_dev_results.json"
    if not path.is_file():
        raise rc.P6BFailClosedError("run the P40 dev stage first")
    k = rc.load_json(path)["selected_layer"]
    if k is None:
        raise rc.P6BFailClosedError("no layer qualified on dev: P40 stops (protocol §2)")
    return int(k)


def units_for(s_q: dict, s_cal: dict, truth, types, normal) -> dict:
    out = {}
    for gate in s_q:
        u = ts.decision_metrics(s_q[gate], s_cal[gate], truth, types, normal)
        u["pass_rates"] = lg.pass_rates(s_q[gate], truth, normal, u["taus"])
        out[gate] = u
    return out


# --------------------------------------------------------------------------- VisA / MTD

def external_task(task, order, factory, device, batch_size, sink, k):
    """experiments.p15.two_stage.external_task, with the layer-k score from the same forward pass."""
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
    g, tk = extract_layers(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size,
                           (k, CLIP_LAYER))
    cal = random.Random(f"p15cal|{task}").sample(sorted(set(normal_candidates) - used_normals - query_shas), ts.N_CAL)
    _, tcal = extract_layers(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in cal], device, batch_size,
                             (k, CLIP_LAYER))
    clip = tk[CLIP_LAYER]
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_shas = [r["rgb_sha256"] for r in queries]
    g = g.to(device)
    units = {}
    for seed in rc.SEEDS:
        bank_idx = torch.as_tensor([index[s] for s in banks[seed]])
        bank_imgs = clip[bank_idx].to(device).float()
        tau = fa.tau99(bank_imgs)
        parts = [ff.c1_descriptors(chunk.to(device).float(), bank_imgs, tau) for chunk in clip.split(ee.Q_CHUNK)]
        per_m = {"": tuple(torch.cat(x) for x in zip(*parts))}
        s_q, s_cal = {}, {}
        for name, L in (("layer18", CLIP_LAYER), (f"layer{k}", k)):
            bank = tk[L][bank_idx].to(device).float().reshape(-1, 1024)
            s_all = ts.scores_of(tk[L], bank, STAT, device, ee.Q_CHUNK)
            s_q[name] = [s_all[i] for i in qi.tolist()]
            s_cal[name] = ts.scores_of(tcal[L], bank, STAT, device, ee.Q_CHUNK)
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
            units[(k_shot, seed)] = units_for(s_q, s_cal, q_true, types, normal)
    return units, {"queries": len(queries), "calibration_normals": len(cal)}


def flat_with_pass(u: dict) -> dict:
    f = ts.flatten(lg.without_pass_rates(u))
    for a, pr in u["pass_rates"].items():
        f[f"pass|a={a}|normal"], f[f"pass|a={a}|defect"] = pr["normal"], pr["defect"]
    return f


def run_external(device, batch_size) -> int:
    k = selected_layer()
    p15 = rc.load_json(ts.OUT["external"] / "p15_external_results.json")
    p15_units = {(u["task"], u["K"], u["seed"]): {x: v for x, v in u.items() if x not in ("task", "K", "seed")}
                 for u in p15["units"]}
    gates = ("layer18", f"layer{k}")
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        units, info, sink = {}, {}, []
        for task in ee.TASKS:
            with torch.no_grad():
                u, i = external_task(task, orders[task], factory, device, batch_size, sink, k)
            for key, v in u.items():
                if not lg.same(lg.without_pass_rates(v["layer18"]), p15_units[(task, *key)]):
                    raise rc.P6BFailClosedError(f"P15 external unit not reproduced: {task} {key}")
            units.update({(task, *key): v for key, v in u.items()})
            info[task] = i
            print(f"[p40-ext] {task}: P15 units reproduced", flush=True)
        if len(units) != len(p15_units):
            raise rc.P6BFailClosedError("unit count differs from P15")
        agg = {}
        for gate in gates:
            flat = {key: flat_with_pass(v[gate]) for key, v in units.items()}
            keys = next(iter(flat.values())).keys()
            agg[gate] = {"visa_object_macro": {str(K): {x: ts.mean_sd(statistics.fmean(flat[(t, K, s)][x] for t in ee.VISA_TASKS)
                                                                      for s in rc.SEEDS) for x in keys} for K in rc.K_VALUES},
                         "mtd": {str(K): {x: ts.mean_sd(flat[(ee.MTD_TASK, K, s)][x] for s in rc.SEEDS) for x in keys}
                                 for K in rc.K_VALUES}}
        result = {"schema_version": "p40_external_v1", "protocol": "docs/P40_SHALLOW_EXIT_PROTOCOL.md#3",
                  "provenance": provenance, "layer": k, "integrity": {"p15_units_reproduced": len(units)},
                  "task_info": info, "aggregate": agg,
                  "units": [{"task": t, "K": K, "seed": s, **v} for (t, K, s), v in units.items()]}
        lines = [f"# P40 — screening at layer {k} vs 18 on VisA / MTD (single run)", "",
                 f"Commit `{provenance['head']}`. P15 units reproduced exactly ({len(units)}).", ""]
        for gate in gates:
            lines += ts.report_lines(f"VisA (object macro) — {gate}", agg[gate]["visa_object_macro"])
            lines += ts.report_lines(f"MTD — {gate}", agg[gate]["mtd"])
        text = "\n".join(lines)
        rc.write_json(staging / "p40_external_results.json", result)
        (staging / "p40_external_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- Cable held-out split

def run_cable_test(mvtec_root: Path, device, batch_size) -> int:
    k = selected_layer()
    p15 = rc.load_json(ts.OUT["cable_test"] / "p15_cable_test_results.json")
    gates = ("layer18", f"layer{k}")
    with single_shot(OUT["cable_test"]) as staging:
        provenance = rc.git_provenance()
        folds = {f: (co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_train.jsonl"),
                     co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_val.jsonl")) for f in co.FOLDS}
        truth = {r["image_path"]: r["class_name"] for f in co.FOLDS for r in folds[f][1]}
        order = co.load_v1(111)[next(iter(truth))]["class_names"]
        pool = sorted(truth)
        test, seen = [], set()
        for r in co.read_jsonl(ts.CABLE_TEST):
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
        L2 = (k, CLIP_LAYER)
        with torch.no_grad():
            g_all, t_all = extract_layers(factory, [co.IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names],
                                          device, batch_size, L2)
            g_t, t_t = extract_layers(factory, [co.IMAGE_ROOT / p for p in test_paths], device, batch_size, L2)
            _, t_cal = extract_layers(factory, [normal_dir / n for n in cal], device, batch_size, L2)
        n_pool = len(pool)
        bank_index = {n: n_pool + i for i, n in enumerate(bank_names)}
        g = torch.cat([g_all[:n_pool], g_t]).to(device)
        clip = torch.cat([t_all[CLIP_LAYER][:n_pool].to(device).float(), t_t[CLIP_LAYER].to(device).float()])
        n = len(order)
        si_all = torch.arange(n_pool, device=device)
        qi_test = torch.arange(n_pool, n_pool + len(test_paths), device=device)
        units = {}
        with torch.no_grad():
            for seed in rc.SEEDS:
                rows = [bank_index[x] for x in draws[seed]]
                bank_imgs = t_all[CLIP_LAYER][rows].to(device).float()
                tau = fa.tau99(bank_imgs)
                per_m = {"": ff.c1_descriptors(clip, bank_imgs, tau)}
                sl_all = [order.index(truth[p]) for p in pool]
                zeros = torch.zeros(len(test_paths), n, device=device)
                scores = ff.episode_scores(g, per_m, si_all, sl_all, qi_test, n, zeros, zeros)
                s_q, s_cal = {}, {}
                for name, L in (("layer18", CLIP_LAYER), (f"layer{k}", k)):
                    bank = t_all[L][rows].to(device).float().reshape(-1, 1024)
                    q_src = clip[qi_test].cpu() if L == CLIP_LAYER else t_t[L]
                    s_q[name] = ts.scores_of(q_src, bank, STAT, device)
                    s_cal[name] = ts.scores_of(t_cal[L], bank, STAT, device)
                types = {m: sc.restricted_type(scores[m], order, co.NORMAL) for m in ts.TYPE_METHODS}
                units[seed] = units_for(s_q, s_cal, test_true, types, co.NORMAL)
                if not lg.same(lg.without_pass_rates(units[seed]["layer18"]), p15["units"][str(seed)]):
                    raise rc.P6BFailClosedError(f"P15 Cable unit not reproduced: seed {seed}")
        agg = {}
        for gate in gates:
            flat = {s: flat_with_pass(v[gate]) for s, v in units.items()}
            agg[gate] = {"8": {x: ts.mean_sd(flat[s][x] for s in rc.SEEDS) for x in next(iter(flat.values()))}}
        result = {"schema_version": "p40_cable_test_v1", "protocol": "docs/P40_SHALLOW_EXIT_PROTOCOL.md#3",
                  "provenance": provenance, "layer": k, "integrity": {"p15_units_reproduced": len(units)},
                  "aggregate": agg, "units": {str(s): v for s, v in units.items()}}
        lines = [f"# P40 — screening at layer {k} vs 18 on the Cable held-out split (single run)", "",
                 f"Commit `{provenance['head']}`. P15 Cable units reproduced exactly ({len(units)} seeds).", ""]
        for gate in gates:
            lines += ts.report_lines(f"Cable, K = 8 — {gate}", agg[gate])
        text = "\n".join(lines)
        rc.write_json(staging / "p40_cable_test_results.json", result)
        (staging / "p40_cable_test_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- report (protocol §4-5)

def run_report(pass_ms: float | None, full_ms: float | None) -> int:
    k = selected_layer()
    gk = f"layer{k}"
    ext = rc.load_json(OUT["external"] / "p40_external_results.json")["aggregate"]
    cab = rc.load_json(OUT["cable_test"] / "p40_cable_test_results.json")["aggregate"]
    sets = {"VisA": {g: ext[g]["visa_object_macro"]["5"] for g in ("layer18", gk)},
            "MTD": {g: ext[g]["mtd"]["5"] for g in ("layer18", gk)},
            "Cable": {g: cab[g]["8"] for g in ("layer18", gk)}}
    auroc = {d: {g: v[g]["auroc"]["mean"] for g in v} for d, v in sets.items()}
    recommend = all(a[gk] >= a["layer18"] - RECOMMEND_MARGIN for a in auroc.values())
    latency = None
    if pass_ms and full_ms:
        latency = {}
        for d, v in sets.items():
            for a in [*map(str, sc.ALPHAS), *map(str, sc.ALPHA_PASS)]:
                pn, pd = v[gk][f"pass|a={a}|normal"]["mean"], v[gk][f"pass|a={a}|defect"]["mean"]
                for good in GOOD_RATES:
                    p = good * pn + (1 - good) * pd
                    latency[f"{d}|a={a}|good={good}"] = p * pass_ms + (1 - p) * (pass_ms + full_ms)
    out = OUT["report"]
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    rc.write_json(out / "p40_report.json", {"layer": k, "auroc": auroc, "recommend": recommend, "margin": RECOMMEND_MARGIN,
                                            "pass_ms": pass_ms, "full_ms": full_ms, "expected_latency_ms": latency})
    lines = [f"# P40 report — layer {k}", "", "| Set | AUROC layer 18 | AUROC layer k | ≥ layer 18 − 0.02 |", "|---|---|---|---|"]
    lines += [f"| {d} | {a['layer18']:.3f} | {a[gk]:.3f} | {a[gk] >= a['layer18'] - RECOMMEND_MARGIN} |" for d, a in auroc.items()]
    lines += ["", f"**Exit at layer {k} recommended as the default: {recommend}**"]
    (out / "p40_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=("dev", "external", "cable-test", "report"))
    ap.add_argument("--mvtec-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pass-ms", type=float, default=None, help="report: Jetson TensorRT pass path at layer k*")
    ap.add_argument("--full-ms", type=float, default=None, help="report: Jetson TensorRT main-configuration path")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    if a.stage == "external":
        return run_external(device, a.batch_size)
    if a.stage == "report":
        return run_report(a.pass_ms, a.full_ms)
    if not a.mvtec_root:
        raise SystemExit("--mvtec-root is required for this stage")
    root = Path(a.mvtec_root)
    return run_dev(root, device) if a.stage == "dev" else run_cable_test(root, device, a.batch_size)


if __name__ == "__main__":
    sys.exit(main())
