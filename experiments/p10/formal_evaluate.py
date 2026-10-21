#!/usr/bin/env python3
"""P10 formal step 2: single formal evaluation (docs/P10_FORMAL_EVALUATION_PROTOCOL.md).

Guards: refuses to run with uncommitted changes in the P10/revision code or protocols,
and refuses to overwrite an existing results directory.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.formal_evaluate --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FEATURE_DIR  # noqa: E402
from experiments.p10.run_controls_a import centered_proto  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "ckpt" / "p10_formal" / "results"
MANIFEST_DIR = rc.STAGE1_BUNDLE / "p6b_support_manifests"
MVREC_ARCHIVE = PROJECT_ROOT / "ckpt" / "mvrec_rgb_formal_query_227300b.tar.gz"
GUARDED_PATHS = ("experiments/p10", "experiments/revision",
                 "docs/P10_FORMAL_EVALUATION_PROTOCOL.md",
                 "docs/P10_ANNOTATION_FREE_LOCALIZATION_PILOT_PROTOCOL.md")

C0_METHODS = ("fusion_proto_local", "simpleshot_cl2n", "local_xclass_clip18", "fused_xclass_clip18")
C1_METHODS = ("normref_pool_clip18", "fused_normref_clip18", "normcenter_mean_clip18",
              "fused_normcenter_mean_clip18", "fused_normref_clip18_m5")
ORACLE_METHODS = ("oracle_maskpool_clip18", "oracle_maskpool_dino")
NEW_METHODS = C0_METHODS + C1_METHODS + ORACLE_METHODS
HYPOTHESES = {
    "H1_primary_C1": ("fused_normref_clip18", "simpleshot_cl2n"),
    "H2_attribution_C1": ("fused_normref_clip18", "fused_normcenter_mean_clip18"),
    "H3_strict_C0": ("fused_xclass_clip18", "simpleshot_cl2n"),
}
COMPARATORS = ("simpleshot_cl2n", "ours", "fusion_proto", "mvrec_rgb")


def guard() -> dict:
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED_PATHS],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if RESULTS_DIR.exists():
        raise rc.P6BFailClosedError(f"{RESULTS_DIR} exists; the formal run is single-shot")
    return rc.git_provenance()


def mvrec_rows() -> list[dict]:
    with tarfile.open(MVREC_ARCHIVE) as archive:
        lines = archive.extractfile("mvrec_rgb_formal_query_predictions.jsonl").read().decode("utf-8")
    grouped = {}
    for line in lines.splitlines():
        row = json.loads(line)
        grouped.setdefault((row["product"], int(row["K"]), int(row["seed"])), []).append(row)
    out = []
    for (product, k_shot, seed), rows in grouped.items():
        order = rows[0]["class_order"]
        m = classification_metrics([r["true_label"] for r in rows],
                                   [r["predicted_label"] for r in rows], order)
        out.append({"method": "mvrec_rgb", "product": product, "K": k_shot, "seed": seed, **m})
    return out


def run_product(product, truth, device):
    dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
    qf = torch.load(FEATURE_DIR / f"queries_{product}.pt", map_location="cpu", weights_only=False)
    nf = torch.load(FEATURE_DIR / f"normals_{product}.pt", map_location="cpu", weights_only=False)
    path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
    n_pool = len(dev["rel_paths"])
    # stack support pool and queries: indices < n_pool are pool images, the rest queries
    g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
    desc = {name: torch.cat([dev[f"{name}_patches"], qf[f"{name}_patches"]]).to(device).float()
            for name in ("dino", "clip18")}
    masks = torch.cat([dev["mask_grid"], qf["mask_grid_oracle_only"]]).to(device)
    normal_clip = nf["clip18_patches"].to(device).float()
    name_index = {n: i for i, n in enumerate(nf["names"])}
    q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
    qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
    clip = desc["clip18"]
    rows, preds = [], []
    for seed in rc.SEEDS:
        draw = nf["draws"][str(seed)]
        pooled, normal_mu = {}, None
        for m, suffix in ((ra.M_PRIMARY, ""), (ra.M_SENS, "_m5")):
            bank_imgs = normal_clip[torch.as_tensor([name_index[n] for n in draw[:m]], device=device)]
            anomaly = ra.patch_anomaly(clip, bank_imgs.reshape(-1, clip.shape[-1]))
            pooled[suffix] = ra.anomaly_pool(clip, anomaly)
            if m == ra.M_PRIMARY:
                normal_mu = bank_imgs.mean(dim=1).mean(dim=0, keepdim=True)
        mean_desc = clip.mean(dim=1)
        for k_shot in rc.K_VALUES:
            manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
            order = list(manifest["class_order"])
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for item in manifest["classes"][cls]["items"]:
                    if item["rel_path"] not in path_index:
                        raise rc.P6BFailClosedError(f"support {item['rel_path']} not in pool features")
                    s_idx.append(path_index[item["rel_path"]])
                    s_lab.append(c)
            if any(t not in order for t in q_true):
                raise rc.P6BFailClosedError(f"query label outside class order: {product}")
            n = len(order)
            si = torch.as_tensor(s_idx, device=device)
            base = rp.simpleshot(g[si], s_lab, g[qi], n)
            zb = rp._zscore(base)
            xc = rp.local_xclass(clip[si], s_lab, clip[qi], n)
            nr = rp.simpleshot(pooled[""][si], s_lab, pooled[""][qi], n)
            nr5 = rp.simpleshot(pooled["_m5"][si], s_lab, pooled["_m5"][qi], n)
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
            for name in ("clip18", "dino"):
                d = desc[name]
                scores[f"oracle_maskpool_{name}"] = rp.oracle_maskpool(
                    d[si], masks[si], s_lab, d[qi], masks[qi], n)
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                rows.append({"method": method, "product": product, "K": k_shot, "seed": seed,
                             **classification_metrics(q_true, pred, order)})
                preds += [{"method": method, "product": product, "K": k_shot, "seed": seed,
                           "rgb_sha256": sha, "class_order": order, "predicted_label": p,
                           "true_label": t, "scores": [float(v) for v in s]}
                          for sha, p, t, s in zip(qf["rgb_sha256"], pred, q_true, sc.cpu())]
    return rows, preds


def fmt_ci(c: dict, k=None) -> str:
    r = c if k is None else c["per_K"][str(k)]
    d, ci = (c["Dbar"], c["Dbar_ci95"]) if k is None else (r["D"], r["ci95"])
    pos = c["Dbar_positive_products"] if k is None else r["positive_products"]
    return f"{100 * d:+.2f} [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] ({pos}/13)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    provenance = guard()
    device = torch.device(args.device)

    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows, preds = [], []
    for product in rc.PRODUCTS:
        with torch.no_grad():
            r, p = run_product(product, truth, device)
        rows += r
        preds += p
        print(f"[p10F] {product} done", flush=True)

    grid = rc.metric_grid_from_rows(rows)
    formal = rc.load_formal_metric_grid()
    grid["ours"], grid["fusion_proto"] = formal["ours"], formal["fusion_proto"]
    grid.update(rc.metric_grid_from_rows(mvrec_rows()))
    for method in (*NEW_METHODS, "ours", "fusion_proto", "mvrec_rgb"):
        rc.require_complete(grid, method)

    hypotheses = {}
    for name, (target, ref) in HYPOTHESES.items():
        res = {metric: contrast(grid, ref, metric, target=target) for metric in rc.METRICS}
        lower = res["balanced_accuracy"]["Dbar_ci95"][0]
        verdict = ("SUPPORTED" if name.startswith("H2") else "CONFIRMED") if lower > 0 else \
                  ("NOT SUPPORTED" if name.startswith("H2") else "NOT CONFIRMED")
        hypotheses[name] = {"target": target, "reference": ref, "verdict_BA": verdict,
                            "contrast": res}
    descriptive = {
        f"{target}_minus_{ref}": {metric: contrast(grid, ref, metric, target=target)
                                  for metric in rc.METRICS}
        for target in NEW_METHODS if target != "simpleshot_cl2n"
        for ref in COMPARATORS if ref != target
    }
    result = {
        "schema_version": "p10_formal_results_v1",
        "protocol": "docs/P10_FORMAL_EVALUATION_PROTOCOL.md",
        "provenance": provenance, "device": str(device),
        "conditions": {"C0": list(C0_METHODS), "C1": list(C1_METHODS), "oracle": list(ORACLE_METHODS)},
        "hypotheses": hypotheses,
        "product_macro_summary": {m: rc.product_macro_summary(grid, m)
                                  for m in (*NEW_METHODS, "ours", "fusion_proto", "mvrec_rgb")},
        "descriptive_contrasts": descriptive,
        "raw_metrics": rows,
    }

    lines = ["# P10 formal evaluation (single run)", "",
             f"Commit `{provenance['head']}`. BA primary; Δ in pp, 95% product-cluster bootstrap CI, "
             "(positive products/13).", "", "## Hypotheses (BA)", "",
             "| Hypothesis | Contrast | K=1 | K=3 | K=5 | Mean over K | Verdict |", "|---|---|---|---|---|---|---|"]
    for name, h in hypotheses.items():
        c = h["contrast"]["balanced_accuracy"]
        lines.append(f"| {name} | {h['target']} − {h['reference']} | " +
                     " | ".join(fmt_ci(c, k) for k in rc.K_VALUES) + f" | {fmt_ci(c)} | {h['verdict_BA']} |")
    lines += ["", "## Product-macro BA (mean ± SD over 5 seeds, %)", "",
              "| Method | Condition | K=1 | K=3 | K=5 |", "|---|---|---|---|---|"]
    cond = {**{m: "C0" for m in C0_METHODS}, **{m: "C1" for m in C1_METHODS},
            **{m: "oracle" for m in ORACLE_METHODS},
            "ours": "C0 (P6.B)", "fusion_proto": "C0 (P6.B)", "mvrec_rgb": "C0 + text prior"}
    for m, s in result["product_macro_summary"].items():
        lines.append(f"| {m} | {cond[m]} | " + " | ".join(
            f"{100 * s[str(k)]['balanced_accuracy']['mean']:.2f} ± {100 * s[str(k)]['balanced_accuracy']['std_ddof1']:.2f}"
            for k in rc.K_VALUES) + " |")
    lines += ["", "## Selected contrasts vs existing formal methods (BA, mean over K)", "",
              "| Target | vs ours | vs fusion_proto | vs mvrec_rgb |", "|---|---|---|---|"]
    for target in ("fused_normref_clip18", "fused_xclass_clip18", "simpleshot_cl2n"):
        cells = []
        for ref in ("ours", "fusion_proto", "mvrec_rgb"):
            key = f"{target}_minus_{ref}"
            c = descriptive[key]["balanced_accuracy"] if key in descriptive else \
                contrast(grid, ref, "balanced_accuracy", target=target)
            cells.append(fmt_ci(c))
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    staging = RESULTS_DIR.with_name(RESULTS_DIR.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rc.write_json(staging / "formal_results.json", result)
    rc.write_jsonl(staging / "formal_predictions.jsonl", preds)
    (staging / "formal_table.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    staging.rename(RESULTS_DIR)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
