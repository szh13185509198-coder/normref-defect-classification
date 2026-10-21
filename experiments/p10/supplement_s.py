#!/usr/bin/env python3
"""P10 Supplement S: support-mask condition (S1) and normal-residual baseline (S2).

Stages (docs/P10_SUPPLEMENT_S_PROTOCOL.md):
    python -m experiments.p10.supplement_s --stage dev --device cuda
    python -m experiments.p10.supplement_s --stage formal --device cuda   (single run)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR, RESULTS_DIR as FORMAL_RESULTS  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT_DEV = PROJECT_ROOT / "ckpt" / "p10_pilot"
OUT_FORMAL = PROJECT_ROOT / "ckpt" / "p10_formal" / "supplement_s"
GUARDED_PATHS = ("experiments/p10", "experiments/revision", "docs/P10_SUPPLEMENT_S_PROTOCOL.md")
S_METHODS = ("supmask_normref_clip18", "fused_supmask_normref_clip18",
             "residual_mean_clip18", "fused_residual_mean_clip18")
REFERENCES = ("simpleshot_cl2n", "fused_normref_clip18")
CHUNK = 16


def normal_descriptors(clip: torch.Tensor, bank: torch.Tensor, device):
    """Per image: anomaly-selected pooled descriptor and mean normal-residual descriptor.

    The pooled descriptor uses exactly the P10 formal code path (whole tensor on the
    device, ``patch_anomaly`` + ``anomaly_pool``) so the integrity check can be bitwise.
    """
    clip_dev = clip.to(device).float()
    pooled = ra.anomaly_pool(clip_dev, ra.patch_anomaly(clip_dev, bank))
    residual = []
    for c in clip_dev.split(CHUNK):
        idx = torch.einsum("npd,bd->npb", c, bank).argmax(dim=2)
        residual.append((c - bank[idx]).mean(dim=1))
    return pooled, torch.cat(residual)


def mask_pool(clip: torch.Tensor, masks: torch.Tensor, device) -> torch.Tensor:
    out = []
    for chunk, m in zip(clip.split(CHUNK), masks.split(CHUNK)):
        c, w = chunk.to(device).float(), m.to(device).float().clone()
        w[w.sum(dim=1) <= 0] = 1.0
        out.append((c * (w / w.sum(dim=1, keepdim=True)).unsqueeze(-1)).sum(dim=1))
    return torch.cat(out)


def episode_scores(g, pooled, residual, supmask, si, sl, qi, n):
    base = rp.simpleshot(g[si], sl, g[qi], n)
    zb = rp._zscore(base)
    nr = rp.simpleshot(pooled[si], sl, pooled[qi], n)
    sm = rp.simpleshot(supmask[si], sl, pooled[qi], n)
    rs = rp.simpleshot(residual[si], sl, residual[qi], n)
    return {"simpleshot_cl2n": base, "fused_normref_clip18": zb + rp._zscore(nr),
            "supmask_normref_clip18": sm, "fused_supmask_normref_clip18": zb + rp._zscore(sm),
            "residual_mean_clip18": rs, "fused_residual_mean_clip18": zb + rp._zscore(rs)}


def score_rows(scores, order, truth, meta):
    rows = []
    for method, sc in scores.items():
        pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
        rows.append({"method": method, **meta, **classification_metrics(truth, pred, order)})
    return rows


def run_dev(device) -> int:
    rows = []
    for product in rc.PRODUCTS:
        data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        name_index = {n: i for i, n in enumerate(normal["names"])}
        g = data["global_fusion"].to(device)
        clip = data["clip18_patches"]
        with torch.no_grad():
            supmask = mask_pool(clip, data["mask_grid"], device)
            for seed in rp.DEV_SEEDS:
                idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                bank = normal["clip18_patches"][idx].to(device).float().reshape(-1, clip.shape[-1])
                pooled, residual = normal_descriptors(clip, bank, device)
                for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, product):
                    si = torch.as_tensor(s_idx, device=device)
                    qi = torch.as_tensor(q_idx, device=device)
                    scores = episode_scores(g, pooled, residual, supmask, si, s_lab, qi, len(order))
                    rows += score_rows(scores, order, [order[c] for c in q_lab],
                                       {"product": product, "K": k_shot, "seed": seed})
        print(f"[S-dev] {product} done", flush=True)
    result = rp.summarize(rows)
    for v in result["methods"].values():
        v.pop("gate_pass", None)
    result.update({"schema_version": "p10_supplement_s_dev_v1", "provenance": rc.git_provenance(), "rows": rows})
    rc.write_json(OUT_DEV / "supplement_s_dev_results.json", result)
    lines = ["# Supplement S — dev sanity (not gating)", "",
             "| Method | BA K=1 | BA K=3 | Δ vs SimpleShot (mean) | +products |", "|---|---|---|---|---|"]
    for m, v in sorted(result["methods"].items(), key=lambda kv: -kv[1]["mean_delta_pp"]):
        lines.append(f"| {m} | {100 * v['ba_by_K']['1']:.2f} | {100 * v['ba_by_K']['3']:.2f} | "
                     f"{v['mean_delta_pp']:+.2f} | {v['positive_products']}/13 |")
    text = "\n".join(lines)
    (OUT_DEV / "supplement_s_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def run_formal(device) -> int:
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED_PATHS],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if OUT_FORMAL.exists():
        raise rc.P6BFailClosedError(f"{OUT_FORMAL} exists; supplement S formal stage is single-shot")
    lock = OUT_FORMAL.with_name(OUT_FORMAL.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()  # atomic: a second concurrent invocation fails here
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another formal-stage run is active or was interrupted")
    provenance = rc.git_provenance()
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows = []
    for product in rc.PRODUCTS:
        dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
        n_pool = len(dev["rel_paths"])
        g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
        clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]])
        no_mask = torch.zeros(len(qf["rgb_sha256"]), dev["mask_grid"].shape[1])  # queries: never masked
        name_index = {n: i for i, n in enumerate(nf["names"])}
        q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
        qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
        with torch.no_grad():
            supmask = mask_pool(clip, torch.cat([dev["mask_grid"], no_mask]), device)
            for seed in rc.SEEDS:
                idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                bank = nf["clip18_patches"][idx].to(device).float().reshape(-1, clip.shape[-1])
                pooled, residual = normal_descriptors(clip, bank, device)
                for k_shot in rc.K_VALUES:
                    manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                    order = list(manifest["class_order"])
                    s_idx, s_lab = [], []
                    for c, cls in enumerate(order):
                        for item in manifest["classes"][cls]["items"]:
                            s_idx.append(path_index[item["rel_path"]])
                            s_lab.append(c)
                    si = torch.as_tensor(s_idx, device=device)
                    scores = episode_scores(g, pooled, residual, supmask, si, s_lab, qi, len(order))
                    rows += score_rows(scores, order, q_true, {"product": product, "K": k_shot, "seed": seed})
        print(f"[S-formal] {product} done", flush=True)

    grid = rc.metric_grid_from_rows(rows)
    formal = rc.load_json(FORMAL_RESULTS / "formal_results.json")["product_macro_summary"]
    integrity = {}
    for ref in REFERENCES:
        now = rc.product_macro_summary(grid, ref)
        integrity[ref] = max(abs(now[str(k)]["balanced_accuracy"]["mean"] -
                                 formal[ref][str(k)]["balanced_accuracy"]["mean"]) for k in rc.K_VALUES)
    if any(v >= 1e-9 for v in integrity.values()):
        raise rc.P6BFailClosedError(f"integrity check vs P10 formal failed: {integrity}")
    contrasts = {f"{m}_minus_{ref}": {metric: contrast(grid, ref, metric, target=m) for metric in rc.METRICS}
                 for m in S_METHODS for ref in REFERENCES}
    summary = {m: rc.product_macro_summary(grid, m) for m in (*REFERENCES, *S_METHODS)}
    result = {"schema_version": "p10_supplement_s_formal_v1",
              "protocol": "docs/P10_SUPPLEMENT_S_PROTOCOL.md", "provenance": provenance,
              "label": "post-formal descriptive reference rows (added during revision)",
              "integrity_max_abs_diff_vs_p10_formal": integrity,
              "product_macro_summary": summary, "contrasts": contrasts, "raw_metrics": rows}

    def fmt(c):
        return (f"{100 * c['Dbar']:+.2f} [{100 * c['Dbar_ci95'][0]:+.2f}, {100 * c['Dbar_ci95'][1]:+.2f}] "
                f"({c['Dbar_positive_products']}/13)")
    lines = ["# Supplement S — formal descriptive reference rows (single run)", "",
             f"Commit `{provenance['head']}`. Integrity vs P10 formal: " +
             ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
             "| Method | Condition | BA K=1 | BA K=3 | BA K=5 | − SimpleShot (mean over K) | − fused_normref_clip18 (mean over K) |",
             "|---|---|---|---|---|---|---|"]
    cond = {"simpleshot_cl2n": "C0", "fused_normref_clip18": "C1",
            "supmask_normref_clip18": "C2 (support masks)", "fused_supmask_normref_clip18": "C2 (support masks)",
            "residual_mean_clip18": "C1", "fused_residual_mean_clip18": "C1"}
    for m, s in summary.items():
        levels = " | ".join(f"{100 * s[str(k)]['balanced_accuracy']['mean']:.2f}" for k in rc.K_VALUES)
        c1 = fmt(contrasts[f"{m}_minus_simpleshot_cl2n"]["balanced_accuracy"]) if m in S_METHODS else "—"
        c2 = fmt(contrasts[f"{m}_minus_fused_normref_clip18"]["balanced_accuracy"]) if m in S_METHODS else "—"
        lines.append(f"| {m} | {cond[m]} | {levels} | {c1} | {c2} |")
    text = "\n".join(lines)
    staging = OUT_FORMAL.with_name(OUT_FORMAL.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rc.write_json(staging / "supplement_s_results.json", result)
    (staging / "supplement_s_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    staging.rename(OUT_FORMAL)
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "formal"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    return run_dev(device) if args.stage == "dev" else run_formal(device)


if __name__ == "__main__":
    sys.exit(main())
