#!/usr/bin/env python3
"""P28 (external part): the same error decomposition on VisA and MTD, report-only.

The mask-oracle per-query predictions were not saved by P27, so this recomputes only the oracle with P27's exact code
path (same tasks, supports, banks, extraction batch and order), checks that its per-cell metrics equal P27's saved
oracle metrics exactly, then aligns it with the saved dual-branch predictions of P14 external and decomposes the errors.
No method, constant or saved result is changed.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p28.error_decomposition_external --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
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
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p27 import external_fill as p27  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

ARR, ORACLE = "fused_adareg_regres_soft_q99", "oracle_maskpool_clip18"
P14_PRED = PROJECT_ROOT / "ckpt" / "p14_formal" / "external" / "p14_external_predictions.jsonl"
P27_RES = PROJECT_ROOT / "ckpt" / "p27" / "external" / "p27_results.json"
OUT = PROJECT_ROOT / "ckpt" / "p28"
CELLS = ("both_right", "arr_only_wrong", "oracle_only_wrong", "both_wrong")


def oracle_predictions(task, order, factory, device):
    """P27's run_task prelude verbatim (same image list and order, extraction batch p27.BATCH), oracle only."""
    dataset = task.split("__")[0]
    records = [json.loads(line) for line in (ee.BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
    draws = json.load((ee.BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
    by_sha = {r["rgb_sha256"]: r for r in records}
    queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
    normal = ee.NORMAL_CLASS[dataset]
    normal_candidates = [r["rgb_sha256"] for r in records
                         if r["partition"] == "support_candidate" and r["class_label"] == normal]
    needed = set()
    for seed in rc.SEEDS:
        k5_normals = draws[normal][str(seed)]["K5"]
        bank = ee.bank_draw(task, seed, normal_candidates, k5_normals)
        needed |= {s for cls in order for s in draws[cls][str(seed)]["K5"]} | set(bank)
    shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
    index = {s: i for i, s in enumerate(shas)}
    _, clip = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, p27.BATCH)
    masks = torch.zeros(len(shas), p27.GRID * p27.GRID)
    for s in shas:
        if by_sha[s]["class_label"] != normal:
            masks[index[s]] = p27.mask_cells(p27.mask_path(dataset, ee.ROOTS[dataset], by_sha[s]["image_rel"]), dataset)
    oracle_desc = p27.pool_by_mask(clip, masks, device)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_shas = [r["rgb_sha256"] for r in queries]
    out = {}
    for seed in rc.SEEDS:
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            si = torch.as_tensor(s_idx, device=device)
            sc = rp.simpleshot(oracle_desc[si], s_lab, oracle_desc[qi], len(order))
            pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
            out[(task, k_shot, seed)] = {"shas": q_shas, "true": q_true, "pred": pred,
                                         "metrics": classification_metrics(q_true, pred, order)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "p28_external_error_decomposition.json"
    if target.exists():
        raise rc.P6BFailClosedError(f"{target} exists; this analysis is single-shot")
    orders, _ = ee.frozen_artifacts()
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=a.device))
    oracle = {}
    for task in ee.TASKS:
        with torch.no_grad():
            oracle.update(oracle_predictions(task, orders[task], factory, a.device))
        print(f"[p28-ext] {task} done", flush=True)
    saved = {(r["task"], int(r["K"]), int(r["seed"])): r for r in rc.load_json(P27_RES)["raw_metrics"] if r["method"] == ORACLE}
    worst = max(abs(v["metrics"][m] - saved[key][m]) for key, v in oracle.items() for m in rc.METRICS)
    if len(saved) != len(oracle) or worst >= 1e-12:
        raise rc.P6BFailClosedError(f"oracle does not reproduce P27 (cells {len(saved)} vs {len(oracle)}, max |d| {worst})")
    arr = {}
    with P14_PRED.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["method"] == ARR:
                arr[(r["task"], int(r["K"]), int(r["seed"]))] = dict(zip(r["rgb_sha256"], zip(r["true"], r["pred"])))
    per = {}
    for key, o in oracle.items():
        q = arr[key]
        if set(q) != set(o["shas"]):
            raise rc.P6BFailClosedError(f"{key}: query sets differ")
        cells = dict.fromkeys(CELLS, 0)
        same = 0
        for s, t, po in zip(o["shas"], o["true"], o["pred"]):
            ta, pa = q[s]
            if ta != t:
                raise rc.P6BFailClosedError(f"{key}: true labels differ")
            a_ok, o_ok = pa == t, po == t
            cells["both_right" if a_ok and o_ok else "arr_only_wrong" if o_ok else
                  "oracle_only_wrong" if a_ok else "both_wrong"] += 1
            same += (not a_ok and not o_ok and pa == po)
        per[key] = {"n": len(q), "counts": cells, "both_wrong_same_label": same}
    summary = {}
    for name, tasks in (("visa", ee.VISA_TASKS), ("mtd", (ee.MTD_TASK,))):
        summary[name] = {}
        for k in rc.K_VALUES:
            keys = [key for key in per if key[0] in tasks and key[1] == k]
            share = {c: statistics.fmean(statistics.fmean(per[(t, k, s)]["counts"][c] / per[(t, k, s)]["n"] for s in rc.SEEDS)
                                         for t in tasks) for c in CELLS}
            pooled = {c: sum(per[key]["counts"][c] for key in keys) for c in CELLS}
            err = pooled["arr_only_wrong"] + pooled["both_wrong"]
            summary[name][str(k)] = {"object_macro_share": share, "pooled_counts": pooled,
                                     "arr_errors_fixed_by_oracle": pooled["arr_only_wrong"] / err if err else None,
                                     "both_wrong_same_label": (sum(per[key]["both_wrong_same_label"] for key in keys)
                                                               / pooled["both_wrong"]) if pooled["both_wrong"] else None}
    result = {"schema_version": "p28_external_error_decomposition_v1", "provenance": rc.git_provenance(),
              "oracle_reproduces_p27_max_abs_diff": worst, "summary": summary,
              "per_cell": {f"{t}|K{k}|s{s}": v for (t, k, s), v in per.items()},
              "oracle_predictions": {f"{t}|K{k}|s{s}": {"rgb_sha256": v["shas"], "pred": v["pred"]}
                                     for (t, k, s), v in oracle.items()}}
    rc.write_json(target, result)
    lines = ["# P28 (external) — dual branch vs mask oracle, per query", "",
             f"Oracle reproduces P27 exactly (max |d| = {worst}).", "",
             "| Data | K | both right | dual only wrong | oracle only wrong | both wrong | dual errors fixed by oracle | both wrong, same label |",
             "|---|---|---|---|---|---|---|---|"]
    for name, by_k in summary.items():
        for k, v in by_k.items():
            m = v["object_macro_share"]
            lines.append(f"| {name} | {k} | {100 * m['both_right']:.1f} | {100 * m['arr_only_wrong']:.1f} | "
                         f"{100 * m['oracle_only_wrong']:.1f} | {100 * m['both_wrong']:.1f} | "
                         f"{100 * v['arr_errors_fixed_by_oracle']:.0f}% | {100 * v['both_wrong_same_label']:.0f}% |")
    text = "\n".join(lines)
    (OUT / "p28_external_error_decomposition.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
