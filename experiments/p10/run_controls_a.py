#!/usr/bin/env python3
"""P10-A2: attribution controls for cue A on the dev episodes (Addendum P10-A2).

Compares the selected cue-A variant against same-information controls that use the
normal images without localization, and against random-subset pooling.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.run_controls_a --device cuda
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

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

SELECTED = "fused_normref_clip18"
CONTROL = "fused_normcenter_mean_clip18"


def centered_proto(gs, sl, gq, n, mu):
    """SimpleShot-CL2N with an externally supplied centering vector."""
    return F.normalize(gq - mu, dim=-1) @ rp._class_means(F.normalize(gs - mu, dim=-1), sl, n).t()


def random_pool(patches: torch.Tensor, seed: int) -> torch.Tensor:
    top = math.ceil(rp.RHO * patches.shape[1])
    out = []
    for i in range(patches.shape[0]):
        gen = torch.Generator().manual_seed(seed * 1000 + i)
        idx = torch.randperm(patches.shape[1], generator=gen)[:top].to(patches.device)
        out.append(patches[i, idx].mean(dim=0))
    return torch.stack(out)


def run_product(data: dict, normal: dict, device) -> list[dict]:
    rows = []
    g = data["global_fusion"].to(device)
    desc = {"dino": data["dino_patches"].to(device).float(),
            "clip18": data["clip18_patches"].to(device).float()}
    normal_desc = {"dino": normal["dino_patches"].to(device).float(),
                   "clip18": normal["clip18_patches"].to(device).float()}
    name_index = {n: i for i, n in enumerate(normal["names"])}
    for seed in rp.DEV_SEEDS:
        idx = torch.as_tensor([name_index[n] for n in normal["draws"][str(seed)][:ra.M_PRIMARY]],
                              device=device)
        per_desc = {}
        for name, patches in desc.items():
            bank_imgs = normal_desc[name][idx]                       # [20, P, d]
            anomaly = ra.patch_anomaly(patches, bank_imgs.reshape(-1, patches.shape[-1]))
            per_desc[name] = {
                "normref": ra.anomaly_pool(patches, anomaly),
                "mean": patches.mean(dim=1),
                "normal_mu": bank_imgs.mean(dim=1).mean(dim=0, keepdim=True),
                "random": random_pool(patches, seed),
            }
        for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, data["product"]):
            n = len(order)
            si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
            base = rp.simpleshot(g[si], s_lab, g[qi], n)
            zb = rp._zscore(base)
            scores = {rp.PRIMARY: base}
            for name, d in per_desc.items():
                nr = rp.simpleshot(d["normref"][si], s_lab, d["normref"][qi], n)
                nc = centered_proto(d["mean"][si], s_lab, d["mean"][qi], n, d["normal_mu"])
                rnd = rp.simpleshot(d["random"][si], s_lab, d["random"][qi], n)
                scores[f"fused_normref_{name}"] = zb + rp._zscore(nr)
                scores[f"normcenter_mean_{name}"] = nc
                scores[f"fused_normcenter_mean_{name}"] = zb + rp._zscore(nc)
                scores[f"random_pool_{name}"] = rnd
                scores[f"fused_random_pool_{name}"] = zb + rp._zscore(rnd)
            truth = [order[c] for c in q_lab]
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                rows.append({"method": method, "product": data["product"], "K": k_shot,
                             "seed": seed, **classification_metrics(truth, pred, order)})
    return rows


def attribution(rows: list[dict]) -> dict:
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    per_product = {p: 100 * statistics.fmean(cell[(SELECTED, p, k, s)] - cell[(CONTROL, p, k, s)]
                                             for k in rp.DEV_K for s in rp.DEV_SEEDS)
                   for p in rc.PRODUCTS}
    per_k = {str(k): 100 * statistics.fmean(
        statistics.fmean(cell[(SELECTED, p, k, s)] - cell[(CONTROL, p, k, s)] for p in rc.PRODUCTS)
        for s in rp.DEV_SEEDS) for k in rp.DEV_K}
    mean_delta = statistics.fmean(per_k.values())
    positive = sum(v > 0 for v in per_product.values())
    return {"selected": SELECTED, "control": CONTROL, "delta_by_K_pp": per_k,
            "mean_delta_pp": mean_delta, "positive_products": positive,
            "per_product_diff_pp": per_product,
            "localization_attribution_supported": mean_delta >= rp.GATE_DELTA_PP
            and positive >= rp.GATE_MIN_POSITIVE}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    rows = []
    for product in rc.PRODUCTS:
        data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            rows += run_product(data, normal, device)
        print(f"[p10A2] {product} done", flush=True)
    result = rp.summarize(rows)
    for v in result["methods"].values():
        v.pop("gate_pass", None)  # controls are not selection candidates
    result.update({"schema_version": "p10A2_controls_v1",
                   "protocol": "docs/P10_ANNOTATION_FREE_LOCALIZATION_PILOT_PROTOCOL.md#addendum-p10-a2",
                   "attribution": attribution(rows),
                   "provenance": rc.git_provenance(), "rows": rows})
    rc.write_json(rp.OUT_DIR / "controls_a_results.json", result)
    a = result["attribution"]
    lines = ["# P10-A2 attribution controls (dev)", "",
             "| Method | BA K=1 | BA K=3 | Δ vs SimpleShot (mean) | +products |", "|---|---|---|---|---|"]
    for m, v in sorted(result["methods"].items(), key=lambda kv: -kv[1]["mean_delta_pp"]):
        lines.append(f"| {m} | {100 * v['ba_by_K']['1']:.2f} | {100 * v['ba_by_K']['3']:.2f} | "
                     f"{v['mean_delta_pp']:+.2f} | {v['positive_products']}/13 |")
    lines += ["", f"**{SELECTED} minus {CONTROL}:** K=1 {a['delta_by_K_pp']['1']:+.2f}, "
                  f"K=3 {a['delta_by_K_pp']['3']:+.2f}, mean {a['mean_delta_pp']:+.2f} pp, "
                  f"{a['positive_products']}/13 positive -> localization attribution supported: "
                  f"{a['localization_attribution_supported']}"]
    text = "\n".join(lines)
    (rp.OUT_DIR / "controls_a_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
