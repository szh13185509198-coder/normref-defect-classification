#!/usr/bin/env python3
"""P10 step 2: dev-only pilot of annotation-free localization cue D (cross-class contrast).

Reads only ``ckpt/p10_pilot/dev_features`` (support pool). Builds dev episodes
(K in {1,3}, seeds 9001-9005), runs every method of protocol Section 4, applies the
Section 5 gate, and writes ``pilot_results.json`` and ``pilot_table.md``.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.run_pilot --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
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
from experiments.revision import revision_common as rc  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "ckpt" / "p10_pilot" / "dev_features"
OUT_DIR = PROJECT_ROOT / "ckpt" / "p10_pilot"
DEV_SEEDS = (9001, 9002, 9003, 9004, 9005)
DEV_K = (1, 3)
RHO = 0.1
DESCRIPTORS = ("dino", "clip18")
GATE_DELTA_PP = 3.0
GATE_MIN_POSITIVE = 9
PRIMARY = "simpleshot_cl2n"


# --------------------------------------------------------------------------- episodes

def episodes(labels: list[str], seed: int, product: str):
    """Yield (K, class_order, support_idx, support_lab, query_idx, query_lab); K nested."""
    class_order = sorted(set(labels))
    shuffled = {}
    for cls in class_order:
        idx = [i for i, lab in enumerate(labels) if lab == cls]
        random.Random(f"p10dev|{seed}|{product}|{cls}").shuffle(idx)
        shuffled[cls] = idx
    for k_shot in DEV_K:
        s_idx, s_lab, q_idx, q_lab = [], [], [], []
        for c, cls in enumerate(class_order):
            if len(shuffled[cls]) <= k_shot:
                raise rc.P6BFailClosedError(f"class {cls} too small for K={k_shot}")
            s_idx += shuffled[cls][:k_shot]
            s_lab += [c] * k_shot
            q_idx += shuffled[cls][k_shot:]
            q_lab += [c] * (len(shuffled[cls]) - k_shot)
        yield k_shot, class_order, s_idx, s_lab, q_idx, q_lab



# --------------------------------------------------------------------------- methods

def _class_means(x: torch.Tensor, labels: list[int], n_classes: int) -> torch.Tensor:
    lab = torch.as_tensor(labels, device=x.device)
    return torch.stack([F.normalize(x[lab == c].mean(0), dim=-1) for c in range(n_classes)])


def fusion_proto(gs, sl, gq, n):
    return F.normalize(gq, dim=-1) @ _class_means(F.normalize(gs, dim=-1), sl, n).t()


def simpleshot(gs, sl, gq, n):
    mu = gs.mean(0, keepdim=True)
    return F.normalize(gq - mu, dim=-1) @ _class_means(F.normalize(gs - mu, dim=-1), sl, n).t()


def _local_scores(q_patches: torch.Tensor, class_sets: list[torch.Tensor]) -> torch.Tensor:
    """q_patches [Q, P, d]; class_sets[c] [M_c, d] -> scores [Q, C] (mean of top-58 maxima)."""
    top = math.ceil(RHO * q_patches.shape[1])
    out = []
    for patches in class_sets:
        sim = torch.einsum("qpd,md->qpm", q_patches, patches)  # [Q, P, M]
        best = sim.max(dim=2).values  # [Q, P]
        out.append(best.topk(top, dim=1).values.mean(dim=1))
    return torch.stack(out, dim=1)


def local_plain(sp, sl, qp, n):
    lab = torch.as_tensor(sl, device=sp.device)
    return _local_scores(qp, [sp[lab == c].reshape(-1, sp.shape[-1]) for c in range(n)])


def _nearest_per_class(q_patches: torch.Tensor, class_sets: list[torch.Tensor]) -> torch.Tensor:
    """[Q, P, C]: for every query patch, max cosine to each class's support patches."""
    return torch.stack([torch.einsum("qpd,md->qpm", q_patches, patches).max(dim=2).values
                        for patches in class_sets], dim=2)


def local_xclass(sp, sl, qp, n):
    """Cue D: query-side cross-class contrast.

    margin_c(q) = s_c(q) - max_{c' != c} s_c'(q). Background patches match every class
    alike (margin ~ 0); class-specific evidence yields a positive margin for its class.
    Class score = mean of the top ceil(0.1 P) margins over query patches.
    """
    if n < 2:
        raise rc.P6BFailClosedError("cross-class contrast needs at least two classes")
    lab = torch.as_tensor(sl, device=sp.device)
    s = _nearest_per_class(qp, [sp[lab == c].reshape(-1, sp.shape[-1]) for c in range(n)])
    top = math.ceil(RHO * qp.shape[1])
    scores = []
    for c in range(n):
        others = torch.cat([s[..., :c], s[..., c + 1:]], dim=2).max(dim=2).values
        scores.append((s[..., c] - others).topk(top, dim=1).values.mean(dim=1))
    return torch.stack(scores, dim=1)


def _zscore(scores: torch.Tensor) -> torch.Tensor:
    std = scores.std(dim=1, keepdim=True, unbiased=False)
    z = (scores - scores.mean(dim=1, keepdim=True)) / std.clamp_min(1e-12)
    return torch.where(std > 0, z, torch.zeros_like(z))


def oracle_maskpool(sp, sm, sl, qp, qm, n):
    def pool(p, m):
        w = m.clone()
        empty = w.sum(dim=1) <= 0
        w[empty] = 1.0
        return (p * (w / w.sum(dim=1, keepdim=True)).unsqueeze(-1)).sum(dim=1)
    return simpleshot(pool(sp, sm), sl, pool(qp, qm), n)


def oracle_masklocal(sp, sm, sl, qp, qm, n):
    lab = torch.as_tensor(sl, device=sp.device)
    sets = []
    for c in range(n):
        p, m = sp[lab == c].reshape(-1, sp.shape[-1]), sm[lab == c].reshape(-1)
        sets.append(p[m > 0] if bool((m > 0).any()) else p)
    top = math.ceil(RHO * qp.shape[1])
    scores = torch.zeros(qp.shape[0], n, device=qp.device)
    for i in range(qp.shape[0]):
        inside = qm[i] > 0
        q = qp[i][inside] if bool(inside.any()) else qp[i]
        for c in range(n):
            best = (q @ sets[c].t()).max(dim=1).values
            scores[i, c] = best.topk(min(top, best.shape[0])).values.mean()
    return scores


# --------------------------------------------------------------------------- driver

def run_product(data: dict, device) -> list[dict]:
    rows = []
    g = data["global_fusion"].to(device)
    desc = {"dino": data["dino_patches"].to(device).float(),
            "clip18": data["clip18_patches"].to(device).float()}
    masks = data["mask_grid"].to(device)
    for seed in DEV_SEEDS:
        for k_shot, order, s_idx, s_lab, q_idx, q_lab in episodes(data["labels"], seed, data["product"]):
            n = len(order)
            si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
            scores = {"fusion_proto": fusion_proto(g[si], s_lab, g[qi], n),
                      PRIMARY: simpleshot(g[si], s_lab, g[qi], n)}
            for name, patches in desc.items():
                sp, qp = patches[si], patches[qi]
                scores[f"local_plain_{name}"] = local_plain(sp, s_lab, qp, n)
                xc = local_xclass(sp, s_lab, qp, n)
                scores[f"local_xclass_{name}"] = xc
                scores[f"fused_xclass_{name}"] = _zscore(scores[PRIMARY]) + _zscore(xc)
                scores[f"oracle_maskpool_{name}"] = oracle_maskpool(sp, masks[si], s_lab, qp, masks[qi], n)
                scores[f"oracle_masklocal_{name}"] = oracle_masklocal(sp, masks[si], s_lab, qp, masks[qi], n)
            truth = [order[c] for c in q_lab]
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                m = classification_metrics(truth, pred, order)
                rows.append({"method": method, "product": data["product"], "K": k_shot,
                             "seed": seed, **m})
    return rows


def summarize(rows: list[dict]) -> dict:
    methods = sorted({r["method"] for r in rows})
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    summary = {}
    for m in methods:
        per_k = {}
        for k in DEV_K:
            per_k[str(k)] = statistics.fmean(
                statistics.fmean(cell[(m, p, k, s)] for p in rc.PRODUCTS) for s in DEV_SEEDS)
        prod_diff = {p: statistics.fmean(cell[(m, p, k, s)] - cell[(PRIMARY, p, k, s)]
                                         for k in DEV_K for s in DEV_SEEDS) for p in rc.PRODUCTS}
        delta = {k: per_k[k] - statistics.fmean(
            statistics.fmean(cell[(PRIMARY, p, int(k), s)] for p in rc.PRODUCTS) for s in DEV_SEEDS)
            for k in per_k}
        mean_delta_pp = 100 * statistics.fmean(delta.values())
        positive = sum(v > 0 for v in prod_diff.values())
        summary[m] = {"ba_by_K": per_k, "delta_vs_simpleshot_pp": {k: 100 * v for k, v in delta.items()},
                      "mean_delta_pp": mean_delta_pp, "positive_products": positive,
                      "per_product_diff_pp": {p: 100 * v for p, v in prod_diff.items()},
                      "is_oracle": m.startswith("oracle_")}
        if not m.startswith("oracle_") and m not in (PRIMARY, "fusion_proto"):
            summary[m]["gate_pass"] = (mean_delta_pp >= GATE_DELTA_PP
                                       and positive >= GATE_MIN_POSITIVE)
    oracle_best = max((v["mean_delta_pp"] for m, v in summary.items() if v["is_oracle"]), default=None)
    return {"methods": summary, "oracle_headroom_pp": oracle_best,
            "localization_direction_alive": oracle_best is not None and oracle_best >= GATE_DELTA_PP}


def table(result: dict) -> str:
    lines = ["# P10 pilot (dev support pool only)", "",
             f"Oracle headroom over SimpleShot: {result['oracle_headroom_pp']:+.2f} pp "
             f"-> localization direction alive: {result['localization_direction_alive']}", "",
             "| Method | BA K=1 | BA K=3 | Δ vs SimpleShot (mean) | +products | Gate |",
             "|---|---|---|---|---|---|"]
    order = sorted(result["methods"].items(), key=lambda kv: (kv[1]["is_oracle"], -kv[1]["mean_delta_pp"]))
    for m, v in order:
        gate = {True: "PASS", False: "fail"}.get(
            v.get("gate_pass"),
            "oracle" if v["is_oracle"] else ("sens" if v.get("sensitivity_only") else "ref"))
        lines.append(f"| {m} | {100 * v['ba_by_K']['1']:.2f} | {100 * v['ba_by_K']['3']:.2f} | "
                     f"{v['mean_delta_pp']:+.2f} | {v['positive_products']}/13 | {gate} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    rows = []
    for product in rc.PRODUCTS:
        data = torch.load(FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        if data.get("schema_version") != "p10_dev_features_v1":
            raise rc.P6BFailClosedError(f"unexpected feature schema for {product}")
        with torch.no_grad():
            rows += run_product(data, device)
        print(f"[p10] {product} done", flush=True)
    result = summarize(rows)
    result.update({"schema_version": "p10_pilot_results_v1",
                   "protocol": "docs/P10_ANNOTATION_FREE_LOCALIZATION_PILOT_PROTOCOL.md",
                   "dev_seeds": list(DEV_SEEDS), "dev_K": list(DEV_K), "rho": RHO,
                   "provenance": rc.git_provenance(), "rows": rows})
    rc.write_json(OUT_DIR / "pilot_results.json", result)
    text = table(result)
    (OUT_DIR / "pilot_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
