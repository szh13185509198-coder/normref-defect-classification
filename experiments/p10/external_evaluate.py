#!/usr/bin/env python3
"""P10 external single run on VisA (confirmatory) and MTD (descriptive).

Implements docs/P10_EXTERNAL_VISA_MTD_PROTOCOL.md: frozen external manifests, P10 methods
with no external tuning, C1 normal banks disjoint from supports and queries, object-cluster
bootstrap for VisA and seed-level differences for MTD. Features are extracted per task and
never written to disk.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.external_evaluate --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
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
from experiments.p6.mvrec_rgb_formal.decision_evaluator import percentile_bootstrap  # noqa: E402
from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER  # noqa: E402
from experiments.p10.run_controls_a import centered_proto  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

BUNDLE = PROJECT_ROOT / "data" / "external_transfer_manifest_bundle"
FROZEN_QUERY_BUNDLE = PROJECT_ROOT / "ckpt" / "external_query_grid_bundle.tar.gz"
ROOTS = {"visa": PROJECT_ROOT / "data" / "VisA_raw",
         "mtd": PROJECT_ROOT / "data" / "MTD_raw" / "Magnetic-tile-defect-datasets.-master"}
NORMAL_CLASS = {"visa": "normal", "mtd": "MT_Free"}
VISA_TASKS = tuple(f"visa__{o}" for o in (
    "candle", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"))
MTD_TASK = "mtd__mtd6way"
TASKS = VISA_TASKS + (MTD_TASK,)
BOOTSTRAP_SEED = 20260814
BANK = 20
RESULTS_DIR = PROJECT_ROOT / "ckpt" / "p10_external" / "results"
GUARDED_PATHS = ("experiments/p10", "experiments/revision", "docs/P10_EXTERNAL_VISA_MTD_PROTOCOL.md")
NEW_METHODS = ("fusion_proto_local", "simpleshot_cl2n", "local_xclass_clip18", "fused_xclass_clip18",
               "normref_pool_clip18", "fused_normref_clip18", "normcenter_mean_clip18",
               "fused_normcenter_mean_clip18", "fused_normref_clip18_m5")
FROZEN_METHODS = ("ours", "frozen_fusion_proto", "mvrec_rgb_adaptation")
HYPOTHESES = {"E1_primary_C1": ("fused_normref_clip18", "simpleshot_cl2n"),
              "E2_attribution_C1": ("fused_normref_clip18", "fused_normcenter_mean_clip18"),
              "E3_strict_C0": ("fused_xclass_clip18", "simpleshot_cl2n")}
Q_CHUNK = 16


def guard() -> dict:
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED_PATHS],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if RESULTS_DIR.exists():
        raise rc.P6BFailClosedError(f"{RESULTS_DIR} exists; the external run is single-shot")
    return rc.git_provenance()


def frozen_artifacts():
    with tarfile.open(FROZEN_QUERY_BUNDLE) as archive:
        orders = {t: json.load(archive.extractfile(f"./ours/{t}/k1/seed_111/query_unit.json"))["class_order"]
                  for t in TASKS}
        cells = json.load(archive.extractfile("./per_cell_metrics.json"))["per_cell_metrics"]
    return orders, [c for c in cells if c["method"] in FROZEN_METHODS]


def bank_draw(task, seed, normal_candidates, k5_normals):
    pool = sorted(set(normal_candidates) - set(k5_normals))
    return random.Random(f"p10ext|{seed}|{task}").sample(pool, BANK)


def extract(factory, paths, device, batch_size):
    glob, clip = [], []
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
    return torch.cat(glob), torch.cat(clip)


def xclass_chunked(sp, s_lab, q_patches_cpu, n, device):
    return torch.cat([rp.local_xclass(sp, s_lab, chunk.to(device).float(), n)
                      for chunk in q_patches_cpu.split(Q_CHUNK)])


def run_task(task, order, factory, device, batch_size, pred_sink=None):
    dataset = task.split("__")[0]
    records = [json.loads(line) for line in (BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
    draws = json.load((BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
    by_sha = {r["rgb_sha256"]: r for r in records}
    queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
    normal = NORMAL_CLASS[dataset]
    normal_candidates = [r["rgb_sha256"] for r in records
                         if r["partition"] == "support_candidate" and r["class_label"] == normal]
    query_shas = {r["rgb_sha256"] for r in queries}
    banks = {}
    needed = set()
    for seed in rc.SEEDS:
        k5_normals = draws[normal][str(seed)]["K5"]
        banks[seed] = bank_draw(task, seed, normal_candidates, k5_normals)
        support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
        if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
            raise rc.P6BFailClosedError(f"{task}/s{seed}: bank/support/query overlap")
        needed |= support_all | set(banks[seed])
    shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
    index = {s: i for i, s in enumerate(shas)}
    g, clip = extract(factory, [ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_patches = clip[qi.cpu()]
    g = g.to(device)
    rows = []
    for seed in rc.SEEDS:
        bank_idx = torch.as_tensor([index[s] for s in banks[seed]])
        pooled20, pooled5, mean_desc = [], [], []
        banks_gpu = {m: clip[bank_idx[:m]].to(device).float().reshape(-1, clip.shape[-1])
                     for m in (BANK, ra.M_SENS)}
        for chunk in clip.split(Q_CHUNK):
            c = chunk.to(device).float()
            pooled20.append(ra.anomaly_pool(c, ra.patch_anomaly(c, banks_gpu[BANK])))
            pooled5.append(ra.anomaly_pool(c, ra.patch_anomaly(c, banks_gpu[ra.M_SENS])))
            mean_desc.append(c.mean(dim=1))
        pooled20, pooled5, mean_desc = torch.cat(pooled20), torch.cat(pooled5), torch.cat(mean_desc)
        normal_mu = clip[bank_idx].to(device).float().mean(dim=1).mean(dim=0, keepdim=True)
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            n = len(order)
            si = torch.as_tensor(s_idx, device=device)
            base = rp.simpleshot(g[si], s_lab, g[qi], n)
            zb = rp._zscore(base)
            xc = xclass_chunked(clip[si.cpu()].to(device).float(), s_lab, q_patches, n, device)
            nr = rp.simpleshot(pooled20[si], s_lab, pooled20[qi], n)
            nr5 = rp.simpleshot(pooled5[si], s_lab, pooled5[qi], n)
            nc = centered_proto(mean_desc[si], s_lab, mean_desc[qi], n, normal_mu)
            scores = {
                "fusion_proto_local": rp.fusion_proto(g[si], s_lab, g[qi], n),
                "simpleshot_cl2n": base,
                "local_xclass_clip18": xc,
                "fused_xclass_clip18": zb + rp._zscore(xc),
                "normref_pool_clip18": nr,
                "fused_normref_clip18": zb + rp._zscore(nr),
                "normcenter_mean_clip18": nc,
                "fused_normcenter_mean_clip18": zb + rp._zscore(nc),
                "fused_normref_clip18_m5": zb + rp._zscore(nr5),
            }
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                rows.append({"method": method, "task": task, "K": k_shot, "seed": seed,
                             **classification_metrics(q_true, pred, order)})
                if pred_sink is not None:
                    pred_sink.append({"method": method, "task": task, "K": k_shot, "seed": seed,
                                      "rgb_sha256": [r["rgb_sha256"] for r in queries],
                                      "true": q_true, "pred": pred})
    return rows, {"queries": len(queries), "extracted_images": len(shas),
                  "bank_draws": {str(s): banks[s] for s in rc.SEEDS}}


def grid_of(rows):
    grid = {}
    for r in rows:
        grid.setdefault(r["method"], {})[(r["task"], int(r["K"]), int(r["seed"]))] = r
    return grid


def visa_contrast(grid, target, ref, metric):
    d = {k: {(t, s): grid[target][(t, k, s)][metric] - grid[ref][(t, k, s)][metric]
             for t in VISA_TASKS for s in rc.SEEDS} for k in rc.K_VALUES}
    per_k = {}
    for k in rc.K_VALUES:
        boot = percentile_bootstrap({t: [d[k][(t, s)] for s in rc.SEEDS] for t in VISA_TASKS},
                                    seed=BOOTSTRAP_SEED)
        obj = {t: statistics.fmean(d[k][(t, s)] for s in rc.SEEDS) for t in VISA_TASKS}
        per_k[str(k)] = {"D": statistics.fmean(obj.values()), "ci95": [boot["ci95_lower"], boot["ci95_upper"]],
                         "positive_objects": sum(v > 0 for v in obj.values()), "per_object": obj}
    boot = percentile_bootstrap({t: [d[k][(t, s)] for k in rc.K_VALUES for s in rc.SEEDS]
                                 for t in VISA_TASKS}, seed=BOOTSTRAP_SEED)
    obj_all = {t: statistics.fmean(d[k][(t, s)] for k in rc.K_VALUES for s in rc.SEEDS) for t in VISA_TASKS}
    return {"per_K": per_k, "Dbar": statistics.fmean(per_k[str(k)]["D"] for k in rc.K_VALUES),
            "Dbar_ci95": [boot["ci95_lower"], boot["ci95_upper"]],
            "Dbar_positive_objects": sum(v > 0 for v in obj_all.values())}


def mtd_contrast(grid, target, ref, metric):
    out = {}
    for k in rc.K_VALUES:
        diffs = [grid[target][(MTD_TASK, k, s)][metric] - grid[ref][(MTD_TASK, k, s)][metric] for s in rc.SEEDS]
        out[str(k)] = {"mean": statistics.fmean(diffs), "sd": statistics.stdev(diffs),
                       "positive_seeds": sum(v > 0 for v in diffs)}
    return out


def visa_macro(grid, method):
    out = {}
    for k in rc.K_VALUES:
        per_seed = [statistics.fmean(grid[method][(t, k, s)]["balanced_accuracy"] for t in VISA_TASKS)
                    for s in rc.SEEDS]
        out[str(k)] = {"mean": statistics.fmean(per_seed), "sd": statistics.stdev(per_seed)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    provenance = guard()
    device = torch.device(args.device)
    orders, frozen_cells = frozen_artifacts()
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))

    rows, task_info = [], {}
    for task in TASKS:
        with torch.no_grad():
            r, info = run_task(task, orders[task], factory, device, args.batch_size)
        rows += r
        task_info[task] = info
        print(f"[p10ext] {task}: {info['queries']} queries, {info['extracted_images']} images", flush=True)

    grid = grid_of(rows + frozen_cells)
    for method in NEW_METHODS + FROZEN_METHODS:
        missing = [(t, k, s) for t in TASKS for k in rc.K_VALUES for s in rc.SEEDS
                   if (t, k, s) not in grid.get(method, {})]
        if missing:
            raise rc.P6BFailClosedError(f"{method}: missing {len(missing)} cells, e.g. {missing[:2]}")

    hypotheses = {}
    for name, (target, ref) in HYPOTHESES.items():
        res = {m: visa_contrast(grid, target, ref, m) for m in rc.METRICS}
        ok = res["balanced_accuracy"]["Dbar_ci95"][0] > 0
        verdict = ("SUPPORTED" if ok else "NOT SUPPORTED") if name.startswith("E2") else \
                  ("CONFIRMED" if ok else "NOT CONFIRMED")
        hypotheses[name] = {"target": target, "reference": ref, "verdict_BA": verdict, "visa": res,
                            "mtd_descriptive": {m: mtd_contrast(grid, target, ref, m) for m in rc.METRICS}}
    descriptive = {}
    for target in ("fused_normref_clip18", "fused_xclass_clip18", "simpleshot_cl2n"):
        for ref in FROZEN_METHODS:
            descriptive[f"{target}_minus_{ref}"] = {
                "visa_BA": visa_contrast(grid, target, ref, "balanced_accuracy"),
                "mtd_BA": mtd_contrast(grid, target, ref, "balanced_accuracy")}
    summary = {m: {"visa_object_macro_BA": visa_macro(grid, m),
                   "mtd_BA": {str(k): {"mean": statistics.fmean(grid[m][(MTD_TASK, k, s)]["balanced_accuracy"]
                                                                 for s in rc.SEEDS),
                                       "sd": statistics.stdev(grid[m][(MTD_TASK, k, s)]["balanced_accuracy"]
                                                              for s in rc.SEEDS)} for k in rc.K_VALUES}}
               for m in NEW_METHODS + FROZEN_METHODS}
    integrity = {str(k): 100 * (summary["fusion_proto_local"]["visa_object_macro_BA"][str(k)]["mean"]
                                - summary["frozen_fusion_proto"]["visa_object_macro_BA"][str(k)]["mean"])
                 for k in rc.K_VALUES}
    result = {"schema_version": "p10_external_results_v1",
              "protocol": "docs/P10_EXTERNAL_VISA_MTD_PROTOCOL.md", "provenance": provenance,
              "device": str(device), "task_info": task_info, "hypotheses": hypotheses,
              "descriptive_contrasts": descriptive, "summary": summary,
              "integrity_fusion_proto_local_minus_frozen_pp": integrity, "raw_metrics": rows}

    def ci(c, k=None):
        if k is None:
            return f"{100 * c['Dbar']:+.2f} [{100 * c['Dbar_ci95'][0]:+.2f}, {100 * c['Dbar_ci95'][1]:+.2f}] ({c['Dbar_positive_objects']}/11)"
        r = c["per_K"][str(k)]
        return f"{100 * r['D']:+.2f} [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}] ({r['positive_objects']}/11)"

    def mtd(c):
        return " / ".join(f"{100 * c[str(k)]['mean']:+.2f}±{100 * c[str(k)]['sd']:.2f}" for k in rc.K_VALUES)

    lines = ["# P10 external evaluation (single run): VisA confirmatory, MTD descriptive", "",
             f"Commit `{provenance['head']}`. BA; Δ in pp; VisA 95% object-cluster bootstrap CI (positive objects/11).",
             "", "## VisA hypotheses", "",
             "| Hypothesis | Contrast | K=1 | K=3 | K=5 | Mean over K | Verdict |", "|---|---|---|---|---|---|---|"]
    for name, h in hypotheses.items():
        c = h["visa"]["balanced_accuracy"]
        lines.append(f"| {name} | {h['target']} − {h['reference']} | " +
                     " | ".join(ci(c, k) for k in rc.K_VALUES) + f" | {ci(c)} | {h['verdict_BA']} |")
    lines += ["", "## MTD (descriptive, seed-level mean ± SD, K=1 / 3 / 5)", "", "| Contrast | Δ BA |", "|---|---|"]
    for name, h in hypotheses.items():
        lines.append(f"| {h['target']} − {h['reference']} | {mtd(h['mtd_descriptive']['balanced_accuracy'])} |")
    lines += ["", "## Levels (BA %, mean ± SD over seeds)", "",
              "| Method | VisA K=1 | VisA K=3 | VisA K=5 | MTD K=1 | MTD K=3 | MTD K=5 |", "|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        lines.append(f"| {m} | " + " | ".join(
            f"{100 * s['visa_object_macro_BA'][str(k)]['mean']:.2f} ± {100 * s['visa_object_macro_BA'][str(k)]['sd']:.2f}"
            for k in rc.K_VALUES) + " | " + " | ".join(
            f"{100 * s['mtd_BA'][str(k)]['mean']:.2f} ± {100 * s['mtd_BA'][str(k)]['sd']:.2f}"
            for k in rc.K_VALUES) + " |")
    lines += ["", "## Versus frozen external methods (BA)", "",
              "| Contrast | VisA mean over K | MTD K=1 / 3 / 5 |", "|---|---|---|"]
    for key, v in descriptive.items():
        lines.append(f"| {key} | {ci(v['visa_BA'])} | {mtd(v['mtd_BA'])} |")
    lines += ["", "Integrity: fusion_proto_local − frozen_fusion_proto, VisA object-macro BA (pp): " +
              ", ".join(f"K={k} {v:+.2f}" for k, v in integrity.items())]
    table = "\n".join(lines)

    staging = RESULTS_DIR.with_name(RESULTS_DIR.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rc.write_json(staging / "external_results.json", result)
    (staging / "external_table.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    staging.rename(RESULTS_DIR)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
