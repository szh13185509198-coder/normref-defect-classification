#!/usr/bin/env python3
"""P10-A step 2: dev-only pilot of cue A (normal-reference anomaly-selected pooling).

Reuses the P10 dev episodes, SimpleShot, cue D and the mask oracle from ``run_pilot``;
adds the Addendum P10-A methods, the M=5 sensitivity, and the anomaly-map AUROC
diagnostic. Formal queries are never read (dev features come from the support pool).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.run_pilot_a --device cuda
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

from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10.extract_normal_features import OUT_DIR as NORMAL_DIR  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

M_PRIMARY = 20
M_SENS = 5


def patch_anomaly(patches: torch.Tensor, bank: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """[N, P, d] x [B, d] -> [N, P] anomaly = 1 - max cosine to the normal bank."""
    out = []
    for block in patches.split(chunk):
        out.append(1.0 - torch.einsum("npd,bd->npb", block, bank).max(dim=2).values)
    return torch.cat(out)


def anomaly_pool(patches: torch.Tensor, anomaly: torch.Tensor) -> torch.Tensor:
    """Mean of the top ceil(0.1 P) most anomalous patches per image -> [N, d]."""
    top = math.ceil(rp.RHO * patches.shape[1])
    idx = anomaly.topk(top, dim=1).indices
    return torch.gather(patches, 1, idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])).mean(dim=1)


def patch_auroc(scores: torch.Tensor, positive: torch.Tensor):
    """Rank-based AUROC for one image; None if only one class is present."""
    pos, neg = int(positive.sum()), int((~positive).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, device=scores.device,
                                           dtype=scores.dtype)
    return float((ranks[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def run_product(data: dict, normal: dict, device):
    rows, aurocs = [], {d: [] for d in rp.DESCRIPTORS}
    g = data["global_fusion"].to(device)
    desc = {"dino": data["dino_patches"].to(device).float(),
            "clip18": data["clip18_patches"].to(device).float()}
    normal_desc = {"dino": normal["dino_patches"].to(device).float(),
                   "clip18": normal["clip18_patches"].to(device).float()}
    name_index = {n: i for i, n in enumerate(normal["names"])}
    masks = data["mask_grid"].to(device)
    for seed in rp.DEV_SEEDS:
        draw = normal["draws"][str(seed)]
        pooled = {}
        for name, patches in desc.items():
            for m in (M_PRIMARY, M_SENS):
                idx = torch.as_tensor([name_index[n] for n in draw[:m]], device=device)
                bank = normal_desc[name][idx].reshape(-1, patches.shape[-1])
                anomaly = patch_anomaly(patches, bank)
                pooled[(name, m)] = anomaly_pool(patches, anomaly)
                if m == M_PRIMARY:
                    for i in range(anomaly.shape[0]):
                        value = patch_auroc(anomaly[i], masks[i] > 0)
                        if value is not None:
                            aurocs[name].append(value)
        for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, data["product"]):
            n = len(order)
            si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
            base = rp.simpleshot(g[si], s_lab, g[qi], n)
            scores = {rp.PRIMARY: base}
            for name, patches in desc.items():
                sp, qp = patches[si], patches[qi]
                xc = rp.local_xclass(sp, s_lab, qp, n)
                scores[f"local_xclass_{name}"] = xc
                scores[f"oracle_maskpool_{name}"] = rp.oracle_maskpool(sp, masks[si], s_lab, qp, masks[qi], n)
                for m, suffix in ((M_PRIMARY, ""), (M_SENS, "_m5")):
                    pv = pooled[(name, m)]
                    nr = rp.simpleshot(pv[si], s_lab, pv[qi], n)
                    scores[f"normref_pool_{name}{suffix}"] = nr
                    scores[f"fused_normref_{name}{suffix}"] = rp._zscore(base) + rp._zscore(nr)
                    scores[f"fused_normref_xclass_{name}{suffix}"] = (
                        rp._zscore(base) + rp._zscore(nr) + rp._zscore(xc))
            truth = [order[c] for c in q_lab]
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                rows.append({"method": method, "product": data["product"], "K": k_shot,
                             "seed": seed, **classification_metrics(truth, pred, order)})
    return rows, aurocs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    rows, aurocs = [], {d: [] for d in rp.DESCRIPTORS}
    for product in rc.PRODUCTS:
        data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        normal = torch.load(NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        if data.get("schema_version") != "p10_dev_features_v1" or \
                normal.get("schema_version") != "p10_normal_features_v1":
            raise rc.P6BFailClosedError(f"unexpected feature schema for {product}")
        with torch.no_grad():
            product_rows, product_auroc = run_product(data, normal, device)
        rows += product_rows
        for d in rp.DESCRIPTORS:
            aurocs[d] += product_auroc[d]
        print(f"[p10A] {product} done", flush=True)

    result = rp.summarize(rows)
    for method, v in result["methods"].items():
        if method.endswith("_m5"):
            v["gate_pass"] = None
            v["sensitivity_only"] = True
        for d in rp.DESCRIPTORS:
            if method.endswith(f"_{d}") or method.endswith(f"_{d}_m5"):
                oracle = result["methods"][f"oracle_maskpool_{d}"]["mean_delta_pp"]
                v["fraction_of_oracle_headroom"] = v["mean_delta_pp"] / oracle if oracle > 0 else None
    passing = [(m, v) for m, v in result["methods"].items() if v.get("gate_pass")]
    selected = max(passing, key=lambda mv: (mv[1]["positive_products"], mv[1]["mean_delta_pp"]),
                   default=(None, None))[0]
    result.update({
        "schema_version": "p10A_pilot_results_v1",
        "protocol": "docs/P10_ANNOTATION_FREE_LOCALIZATION_PILOT_PROTOCOL.md#addendum-p10-a",
        "bank_size_primary": M_PRIMARY, "bank_size_sensitivity": M_SENS,
        "anomaly_map_patch_auroc": {d: {"mean": statistics.fmean(v), "images": len(v)}
                                    for d, v in aurocs.items() if v},
        "selected_by_rule_among_this_run": selected,
        "provenance": rc.git_provenance(), "rows": rows,
    })
    rc.write_json(rp.OUT_DIR / "pilot_a_results.json", result)
    lines = [rp.table(result).replace("# P10 pilot", "# P10-A pilot (cue A + cue D)"), "",
             "Anomaly-map patch AUROC vs masks (M=20): " + ", ".join(
                 f"{d} {v['mean']:.3f} (n={v['images']})"
                 for d, v in result["anomaly_map_patch_auroc"].items()),
             "", "Fraction of oracle headroom recovered:"]
    for m, v in sorted(result["methods"].items()):
        if v.get("fraction_of_oracle_headroom") is not None and not v["is_oracle"]:
            lines.append(f"- {m}: {v['fraction_of_oracle_headroom']:.2f}")
    lines += ["", f"Selected by the P10-A rule among gate-passing variants of this run: {selected}"]
    text = "\n".join(lines)
    (rp.OUT_DIR / "pilot_a_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
