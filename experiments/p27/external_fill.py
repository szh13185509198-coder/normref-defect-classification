#!/usr/bin/env python3
"""P27: missing VisA / MTD rows (Liu residual, logistic regression, mask oracle), report-only.

Protocol docs/P27_EXTERNAL_FILL_PROTOCOL.md. Frozen P10 external tasks, banks, supports and queries; the
recomputed SimpleShot and fixed-fraction rows must equal P10 exactly before anything is written.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p27.external_fill --device cuda [--resume]
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c2_frozen_feature_baselines import logreg_rfs  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p27" / "external"
# Attempt 1 (external.staging / external.lock, extraction batch 16) failed the integrity gate and is kept as a
# record. Attempt 2 extracts with batch 4, exactly as P10 / P12 external, in its own staging and lock.
STAGING = OUT.with_name(OUT.name + ".attempt2.staging")
LOCK = OUT.with_name(OUT.name + ".attempt2.lock")
BATCH = 4                      # P10 / P12 external extraction batch; the features depend on it bitwise
GUARDED_PATHS = ("experiments/p27", "experiments/p10", "experiments/revision", "docs/P27_EXTERNAL_FILL_PROTOCOL.md")
P11_EXTERNAL = PROJECT_ROOT / "ckpt" / "p11_formal" / "external" / "p11_formal_external_results.json"
P14_EXTERNAL = PROJECT_ROOT / "ckpt" / "p14_formal" / "external" / "p14_external_results.json"
SS, NR = "simpleshot_cl2n", "fused_normref_clip18"
RES, FRES, LR, ORACLE = "residual_mean_clip18", "fused_residual_mean_clip18", "logreg_rfs", "oracle_maskpool_clip18"
A, ARR, ADAPT = "fused_adareg_soft_q99", "fused_adareg_regres_soft_q99", "ours"
NEW = (RES, FRES, LR, ORACLE)
GRID = 24                      # 336 / 14 patches per side
MASK_ON = {"visa": 0, "mtd": 127}   # a mask pixel counts when its value is above this
CONTRASTS = {"R1_A_minus_residual": (A, RES), "R2_ARR_minus_residual": (ARR, RES),
             "R3_logreg_minus_simpleshot": (LR, SS), "R4_adaptation_minus_simpleshot": (ADAPT, SS),
             "R5_oracle_minus_simpleshot": (ORACLE, SS), "R6_ARR_minus_logreg": (ARR, LR)}


def mask_path(dataset: str, root: Path, image_rel: str) -> Path:
    rel = Path(image_rel)
    if dataset == "visa":       # <obj>/Data/Images/Anomaly/000.JPG -> <obj>/Data/Masks/Anomaly/000.png
        parts = list(rel.parts)
        parts[parts.index("Images")] = "Masks"
        return root / Path(*parts).with_suffix(".png")
    return root / rel.with_suffix(".png")


def mask_cells(path: Path, dataset: str, grid: int = GRID) -> torch.Tensor:
    """Defect-area fraction per grid cell; the evaluation view is a squash resize, so is the mask."""
    with Image.open(path) as handle:
        binary = (np.asarray(handle.convert("L")) > MASK_ON[dataset]).astype(np.uint8) * 255
    small = Image.fromarray(binary).resize((grid, grid), Image.BOX)
    return torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).flatten()


def pool_by_mask(clip: torch.Tensor, masks: torch.Tensor, device) -> torch.Tensor:
    out = []
    for chunk, m in zip(clip.split(ee.Q_CHUNK), masks.split(ee.Q_CHUNK)):
        c, w = chunk.to(device).float(), m.to(device).float().clone()
        w[w.sum(dim=1) <= 0] = 1.0
        out.append((c * (w / w.sum(dim=1, keepdim=True)).unsqueeze(-1)).sum(dim=1))
    return torch.cat(out)


def residual_mean(clip: torch.Tensor, bank: torch.Tensor, device) -> torch.Tensor:
    out = []
    for chunk in clip.split(ee.Q_CHUNK):
        c = chunk.to(device).float()
        idx = torch.einsum("npd,bd->npb", c, bank).argmax(dim=2)
        out.append((c - bank[idx]).mean(dim=1))
    return torch.cat(out)


def run_task(task, order, factory, device, batch_size):
    dataset = task.split("__")[0]
    records = [json.loads(line) for line in (ee.BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
    draws = json.load((ee.BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
    by_sha = {r["rgb_sha256"]: r for r in records}
    queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
    normal = ee.NORMAL_CLASS[dataset]
    normal_candidates = [r["rgb_sha256"] for r in records
                         if r["partition"] == "support_candidate" and r["class_label"] == normal]
    query_shas = {r["rgb_sha256"] for r in queries}
    banks, needed = {}, set()
    for seed in rc.SEEDS:
        k5_normals = draws[normal][str(seed)]["K5"]
        banks[seed] = ee.bank_draw(task, seed, normal_candidates, k5_normals)
        support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
        if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
            raise rc.P6BFailClosedError(f"{task}/s{seed}: bank/support/query overlap")
        needed |= support_all | set(banks[seed])
    shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
    index = {s: i for i, s in enumerate(shas)}
    g, clip = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size)
    masks = torch.zeros(len(shas), GRID * GRID)
    empty_defect_masks = 0
    for s in shas:
        r = by_sha[s]
        if r["class_label"] != normal:
            masks[index[s]] = mask_cells(mask_path(dataset, ee.ROOTS[dataset], r["image_rel"]), dataset)
            empty_defect_masks += int(masks[index[s]].sum() <= 0)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    g = g.to(device)
    oracle_desc = pool_by_mask(clip, masks, device)
    rows = []
    for seed in rc.SEEDS:
        bank_idx = torch.as_tensor([index[s] for s in banks[seed]])
        bank20 = clip[bank_idx[:ee.BANK]].to(device).float().reshape(-1, clip.shape[-1])
        pooled20 = []
        for chunk in clip.split(ee.Q_CHUNK):          # exactly the P10 external code path (integrity rows)
            c = chunk.to(device).float()
            pooled20.append(ra.anomaly_pool(c, ra.patch_anomaly(c, bank20)))
        pooled20 = torch.cat(pooled20)
        resid = residual_mean(clip, bank20, device)
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
            rs = rp.simpleshot(resid[si], s_lab, resid[qi], n)
            scores = {
                SS: base,
                NR: zb + rp._zscore(rp.simpleshot(pooled20[si], s_lab, pooled20[qi], n)),
                RES: rs,
                FRES: zb + rp._zscore(rs),
                LR: torch.from_numpy(logreg_rfs(g[si].cpu(), s_lab, g[qi].cpu(), n)),
                ORACLE: rp.simpleshot(oracle_desc[si], s_lab, oracle_desc[qi], n),
            }
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                rows.append({"method": method, "task": task, "K": k_shot, "seed": seed,
                             **classification_metrics(q_true, pred, order)})
    return rows, {"queries": len(queries), "extracted_images": len(shas),
                  "defect_images_with_empty_mask": empty_defect_masks}


def exact_signflip(values: list[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    obs = abs(x.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(x))))
    return float(np.mean(np.abs(signs @ x) / len(x) >= obs - 1e-15))


def integrity(rows) -> dict:
    ref = {(r["method"], r["task"], int(r["K"]), int(r["seed"])): r
           for r in rc.load_json(ee.RESULTS_DIR / "external_results.json")["raw_metrics"] if r["method"] in (SS, NR)}
    out = {}
    for m in (SS, NR):
        mine = [r for r in rows if r["method"] == m]
        if len(mine) != len(ee.TASKS) * len(rc.K_VALUES) * len(rc.SEEDS):
            raise rc.P6BFailClosedError(f"{m}: {len(mine)} cells")
        out[m] = max(abs(r[k] - ref[(m, r["task"], int(r["K"]), int(r["seed"]))][k])
                     for r in mine for k in rc.METRICS)
    return out


def score(rows, info, provenance) -> dict:
    frozen = [c for c in ee.frozen_artifacts()[1] if c["method"] == ADAPT]
    others = [r for r in rc.load_json(P11_EXTERNAL)["raw_metrics"] if r["method"] == A]
    others += [r for r in rc.load_json(P14_EXTERNAL)["raw_metrics"] if r["method"] == ARR]
    grid = ee.grid_of(rows + others + frozen)
    methods = (*NEW, SS, ADAPT, A, ARR)
    levels = {m: {"visa": ee.visa_macro(grid, m),
                  "mtd": {str(k): {"mean": statistics.fmean(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"]
                                                            for s in rc.SEEDS),
                                   "sd": statistics.stdev(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"]
                                                          for s in rc.SEEDS)} for k in rc.K_VALUES}}
              for m in methods}
    contrasts = {}
    for name, (t, r) in CONTRASTS.items():
        v = ee.visa_contrast(grid, t, r, "balanced_accuracy")
        per_obj = [statistics.fmean(grid[t][(o, k, s)]["balanced_accuracy"] - grid[r][(o, k, s)]["balanced_accuracy"]
                                    for k in rc.K_VALUES for s in rc.SEEDS) for o in ee.VISA_TASKS]
        v["signflip_p_exact"] = exact_signflip(per_obj)
        contrasts[name] = {"target": t, "reference": r, "visa": v,
                           "mtd": ee.mtd_contrast(grid, t, r, "balanced_accuracy")}
    headroom = {}
    for m in (ARR, A):
        headroom[m] = {}
        for ds in ("visa", "mtd"):
            headroom[m][ds] = {}
            for k in map(str, rc.K_VALUES):
                gap = levels[ORACLE][ds][k]["mean"] - levels[SS][ds][k]["mean"]
                headroom[m][ds][k] = (levels[m][ds][k]["mean"] - levels[SS][ds][k]["mean"]) / gap if gap else None
    return {"schema_version": "p27_external_fill_v1", "protocol": "docs/P27_EXTERNAL_FILL_PROTOCOL.md",
            "label": "report-only; no method or constant changed", "provenance": provenance, "task_info": info,
            "levels_BA": levels, "contrasts_BA": contrasts, "oracle_headroom_recovered": headroom, "raw_metrics": rows}


def table(result) -> str:
    lv, ct = result["levels_BA"], result["contrasts_BA"]
    lines = ["# P27 — VisA / MTD fill (report-only)", "", f"Integrity: {result['integrity_max_abs_diff']}", "",
             "| Method | VisA K=1 / 3 / 5 | MTD K=1 / 3 / 5 |", "|---|---|---|"]
    for m, v in lv.items():
        cell = lambda d: " / ".join(f"{100 * d[str(k)]['mean']:.2f} ± {100 * d[str(k)]['sd']:.2f}" for k in rc.K_VALUES)
        lines.append(f"| {m} | {cell(v['visa'])} | {cell(v['mtd'])} |")
    lines += ["", "| Contrast | VisA pooled [95% CI] (+objects/11), sign-flip p | MTD K=1 / 3 / 5 (mean ± SD) |",
              "|---|---|---|"]
    for name, c in ct.items():
        v, m = c["visa"], c["mtd"]
        lines.append(f"| {name} | {100 * v['Dbar']:+.2f} [{100 * v['Dbar_ci95'][0]:+.2f}, {100 * v['Dbar_ci95'][1]:+.2f}] "
                     f"({v['Dbar_positive_objects']}/11), p={v['signflip_p_exact']:.4f} | "
                     + " / ".join(f"{100 * m[str(k)]['mean']:+.2f}±{100 * m[str(k)]['sd']:.2f}" for k in rc.K_VALUES) + " |")
    lines += ["", "Oracle headroom recovered (K=1 / 3 / 5):"]
    for m, v in result["oracle_headroom_recovered"].items():
        for ds, d in v.items():
            lines.append(f"- {m} {ds}: " + " / ".join("n/a" if d[str(k)] is None else f"{d[str(k)]:.2f}"
                                                     for k in rc.K_VALUES))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *GUARDED_PATHS],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; P27 is single-shot")
    provenance = rc.git_provenance()
    if args.resume:
        if not LOCK.exists() or not STAGING.exists():
            raise rc.P6BFailClosedError("--resume needs the lock and staging directories of an interrupted run")
    else:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            LOCK.mkdir()
        except FileExistsError:
            raise rc.P6BFailClosedError(f"{LOCK} exists: a run is active or was interrupted (use --resume)")
        STAGING.mkdir()
    orders, _ = ee.frozen_artifacts()
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=args.device))
    rows, info = [], {}
    for task in ee.TASKS:
        ckpt = STAGING / f"{task}.json"
        if ckpt.exists():
            saved = json.loads(ckpt.read_text(encoding="utf-8"))
            if saved["head"] != provenance["head"] or saved.get("batch") != BATCH:
                raise rc.P6BFailClosedError(f"{ckpt} comes from commit {saved['head']}")
            r, i = saved["rows"], saved["info"]
            print(f"[p27] {task}: from checkpoint", flush=True)
        else:
            with torch.no_grad():
                r, i = run_task(task, orders[task], factory, args.device, BATCH)
            tmp = ckpt.with_suffix(".tmp")
            tmp.write_text(json.dumps({"head": provenance["head"], "batch": BATCH, "rows": r, "info": i}), encoding="utf-8")
            tmp.replace(ckpt)
            print(f"[p27] {task}: {i}", flush=True)
        rows += r
        info[task] = i
    check = integrity(rows)
    if any(v >= 1e-9 for v in check.values()):
        raise rc.P6BFailClosedError(f"integrity check failed: {check}")
    result = score(rows, info, provenance)
    result["integrity_max_abs_diff"] = check
    final = STAGING / "final"
    final.mkdir()
    rc.write_json(final / "p27_results.json", result)
    text = table(result)
    (final / "p27_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    shutil.move(str(final), str(OUT))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
