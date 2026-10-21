#!/usr/bin/env python3
"""Dev-only sensitivity over CLIP layer and top fraction (docs/P10_FIGURES_AND_SENSITIVITY_PROTOCOL.md §1).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.sensitivity_dev --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import math
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
from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

LAYERS = (12, 18, 24)
RHOS = (0.05, 0.10, 0.20, 0.30)
OUT = PROJECT_ROOT / "ckpt" / "p10_pilot" / "sensitivity"


def extract_layers(factory, paths, device, batch_size=4):
    """Same batching and normalization as the P10 extractors, for several CLIP layers."""
    out = {layer: [] for layer in LAYERS}
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
        batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
        with torch.no_grad():
            _, tokens = factory.clip_model.visual(batch, list(LAYERS))
        for layer, tok in zip(LAYERS, tokens):
            out[layer].append(F.normalize(tok[:, 1:, :].float(), dim=-1).half().cpu())
    return {layer: torch.cat(v) for layer, v in out.items()}


def xclass(sp, sl, qp, n, rho):
    lab = torch.as_tensor(sl, device=sp.device)
    s = rp._nearest_per_class(qp, [sp[lab == c].reshape(-1, sp.shape[-1]) for c in range(n)])
    top = math.ceil(rho * qp.shape[1])
    scores = []
    for c in range(n):
        others = torch.cat([s[..., :c], s[..., c + 1:]], dim=2).max(dim=2).values
        scores.append((s[..., c] - others).topk(top, dim=1).values.mean(dim=1))
    return torch.stack(scores, dim=1)


def pool(patches, anomaly, rho):
    top = math.ceil(rho * patches.shape[1])
    idx = anomaly.topk(top, dim=1).indices
    return torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])).mean(dim=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; write-once output")
    device = torch.device(args.device)
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    rows = []
    for product in rc.PRODUCTS:
        dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        dev_layers = extract_layers(factory, [rc.DATASET_ROOT / p for p in dev["rel_paths"]], device)
        normal_dir = Path(args.mvtec_root) / product / "train" / "good"
        normal_layers = extract_layers(factory, [normal_dir / n for n in normal["names"]], device)
        name_index = {n: i for i, n in enumerate(normal["names"])}
        g = dev["global_fusion"].to(device)
        with torch.no_grad():
            for seed in rp.DEV_SEEDS:
                bank_idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                per_layer = {}
                for layer in LAYERS:
                    clip = dev_layers[layer].to(device).float()
                    bank = normal_layers[layer][bank_idx].to(device).float().reshape(-1, clip.shape[-1])
                    residual = []
                    for c in clip.split(8):
                        idx = torch.einsum("npd,bd->npb", c, bank).argmax(dim=2)
                        residual.append((c - bank[idx]).mean(dim=1))
                    per_layer[layer] = {"clip": clip, "anomaly": ra.patch_anomaly(clip, bank),
                                        "residual": torch.cat(residual)}
                for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(dev["labels"], seed, product):
                    n = len(order)
                    si = torch.as_tensor(s_idx, device=device)
                    qi = torch.as_tensor(q_idx, device=device)
                    truth = [order[c] for c in q_lab]
                    base = rp.simpleshot(g[si], s_lab, g[qi], n)
                    zb = rp._zscore(base)
                    scores = {("simpleshot_cl2n", None, None): base}
                    for layer, d in per_layer.items():
                        scores[("residual_mean", layer, None)] = rp.simpleshot(d["residual"][si], s_lab, d["residual"][qi], n)
                        for rho in RHOS:
                            xc = xclass(d["clip"][si], s_lab, d["clip"][qi], n, rho)
                            pv = pool(d["clip"], d["anomaly"], rho)
                            scores[("fused_xclass", layer, rho)] = zb + rp._zscore(xc)
                            scores[("fused_normref", layer, rho)] = zb + rp._zscore(rp.simpleshot(pv[si], s_lab, pv[qi], n))
                    for (method, layer, rho), sc in scores.items():
                        pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                        rows.append({"method": method, "layer": layer, "rho": rho, "product": product,
                                     "K": k_shot, "seed": seed,
                                     "balanced_accuracy": classification_metrics(truth, pred, order)["balanced_accuracy"]})
        print(f"[sens] {product} done", flush=True)

    cell = {(r["method"], r["layer"], r["rho"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    ref = {(p, k, s): cell[("simpleshot_cl2n", None, None, p, k, s)]
           for p in rc.PRODUCTS for k in rp.DEV_K for s in rp.DEV_SEEDS}
    configs = sorted({(r["method"], r["layer"], r["rho"]) for r in rows if r["method"] != "simpleshot_cl2n"},
                     key=lambda c: (c[0], c[1], c[2] or 0))
    summary = []
    for method, layer, rho in configs:
        diffs = {p: statistics.fmean(cell[(method, layer, rho, p, k, s)] - ref[(p, k, s)]
                                     for k in rp.DEV_K for s in rp.DEV_SEEDS) for p in rc.PRODUCTS}
        summary.append({"method": method, "layer": layer, "rho": rho,
                        "mean_delta_pp": 100 * statistics.fmean(diffs.values()),
                        "positive_products": sum(v > 0 for v in diffs.values()),
                        "per_product_pp": {p: 100 * v for p, v in diffs.items()}})
    sanity = {m: next(s["mean_delta_pp"] for s in summary if s["method"] == m and s["layer"] == 18 and s["rho"] == 0.10)
              for m in ("fused_normref", "fused_xclass")}
    expected = {"fused_normref": rc.load_json(rp.OUT_DIR / "pilot_a_results.json")["methods"]["fused_normref_clip18"]["mean_delta_pp"],
                "fused_xclass": rc.load_json(rp.OUT_DIR / "pilot_results.json")["methods"]["fused_xclass_clip18"]["mean_delta_pp"]}
    sanity_ok = all(abs(sanity[m] - expected[m]) < 1e-9 for m in sanity)
    result = {"schema_version": "p10_sensitivity_dev_v1", "protocol": "docs/P10_FIGURES_AND_SENSITIVITY_PROTOCOL.md#1",
              "provenance": rc.git_provenance(), "sanity_layer18_rho010": {"recomputed": sanity, "pilot": expected,
                                                                          "reproduced": sanity_ok},
              "summary": summary, "rows": rows}
    lines = ["# Dev-only sensitivity: CLIP layer × top fraction (Δ BA vs SimpleShot, mean over K=1/3; +products/13)", "",
             f"Sanity (layer 18, rho 0.10) reproduces the P10 dev pilot: {sanity_ok}", "",
             "| Method | Layer | rho=0.05 | rho=0.10 | rho=0.20 | rho=0.30 |", "|---|---|---|---|---|---|"]
    for method in ("fused_normref", "fused_xclass"):
        for layer in LAYERS:
            cells = []
            for rho in RHOS:
                s = next(x for x in summary if x["method"] == method and x["layer"] == layer and x["rho"] == rho)
                cells.append(f"{s['mean_delta_pp']:+.2f} ({s['positive_products']}/13)")
            lines.append(f"| {method} | {layer} | " + " | ".join(cells) + " |")
    for layer in LAYERS:
        s = next(x for x in summary if x["method"] == "residual_mean" and x["layer"] == layer)
        lines.append(f"| residual_mean (rho-free) | {layer} | {s['mean_delta_pp']:+.2f} ({s['positive_products']}/13) | | | |")
    text = "\n".join(lines)
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "sensitivity_results.json", result)
    (OUT / "sensitivity_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
