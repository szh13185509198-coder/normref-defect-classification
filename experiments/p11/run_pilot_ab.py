#!/usr/bin/env python3
"""P11: dev-only pilot of the adaptive region (A) and anomaly-guided matching (B) cues.

Protocol: docs/P11_ADAPTIVE_REGION_AND_GUIDED_MATCHING_PILOT_PROTOCOL.md. Reads only the
P10 dev features (support pool) and the P10-A normal banks; formal queries are never read.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p11.run_pilot_ab --device cuda
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
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10 import supplement_s as ss  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p11_pilot"
TARGET = "fused_normref_clip18"
QUANTILES = {"q95": 0.95, "q99": 0.99, "q999": 0.999}
MIN_REGION = 6
GATE_DELTA_PP = 1.0
GATE_MIN_POSITIVE = 8
FAMILIES = {
    "A": ("fused_adareg_hard_q95", "fused_adareg_hard_q99", "fused_adareg_hard_q999",
          "fused_adareg_soft_q99"),
    "B": ("fused_guided_plain", "fused_guided_xclass", "fused_guided_xclass_normref"),
}
EXPECTED_P10 = {"fused_normref_clip18": ("pilot_a_results.json", "fused_normref_clip18"),
                "fused_xclass_clip18": ("pilot_results.json", "fused_xclass_clip18")}


# --------------------------------------------------------------------------- family A

def loo_anomalies(bank_imgs: torch.Tensor) -> torch.Tensor:
    """Leave-one-image-out patch anomalies inside the normal bank [M, P, d] -> [M * P]."""
    values = []
    for j in range(bank_imgs.shape[0]):
        others = torch.cat([bank_imgs[:j], bank_imgs[j + 1:]]).reshape(-1, bank_imgs.shape[-1])
        values.append(ra.patch_anomaly(bank_imgs[j:j + 1], others).reshape(-1))
    return torch.cat(values).double()


def loo_threshold(bank_imgs: torch.Tensor, q: float) -> float:
    """tau_q: q-quantile of the leave-one-image-out bank anomalies (normal images only)."""
    return float(torch.quantile(loo_anomalies(bank_imgs), q))


def region_weights(anomaly: torch.Tensor, tau: float, soft: bool) -> torch.Tensor:
    """[N, P] pooling weights: a > tau (hard) or max(a - tau, 0) (soft); top-6 fallback."""
    w = (anomaly - tau).clamp_min(0) if soft else (anomaly > tau).to(anomaly.dtype)
    small = (w > 0).sum(dim=1) < MIN_REGION
    if bool(small.any()):
        fallback = torch.zeros_like(anomaly[small])
        fallback.scatter_(1, anomaly[small].topk(MIN_REGION, dim=1).indices, 1.0)
        w[small] = fallback
    return w


def weighted_pool(patches: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("np,npd->nd", w, patches) / w.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- family B

def guided_scores(region_patches: torch.Tensor, sl: list[int], si, qi, n: int):
    """region_patches [N, R, d] (the fixed top-58 anomalous patches of every image).

    Returns (plain [Q, C], xclass [Q, C]) computed only between anomalous regions."""
    if n < 2:
        raise rc.P6BFailClosedError("guided cross-class matching needs at least two classes")
    lab = torch.as_tensor(sl, device=region_patches.device)
    sup = region_patches[si]
    sets = [sup[lab == c].reshape(-1, sup.shape[-1]) for c in range(n)]
    s = rp._nearest_per_class(region_patches[qi], sets)  # [Q, R, C]
    plain = s.mean(dim=1)
    margins = []
    for c in range(n):
        others = torch.cat([s[..., :c], s[..., c + 1:]], dim=2).max(dim=2).values
        margins.append((s[..., c] - others).mean(dim=1))
    return plain, torch.stack(margins, dim=1)


# --------------------------------------------------------------------------- diagnostics

def region_agreement(selected: torch.Tensor, masks: torch.Tensor) -> list[dict]:
    out = []
    for sel, m in zip(selected.bool(), masks > 0):
        if not bool(m.any()):
            continue
        tp = int((sel & m).sum())
        precision = tp / max(int(sel.sum()), 1)
        recall = tp / int(m.sum())
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        out.append({"precision": precision, "recall": recall, "f1": f1})
    return out


# --------------------------------------------------------------------------- driver

def run_product(data: dict, normal: dict, device):
    rows = []
    diag = {"fixed_top58": [], **{f"hard_{k}": [] for k in QUANTILES}}
    sizes = {f"hard_{k}": [] for k in QUANTILES} | {"soft_q99": []}
    fallback = {k: [] for k in sizes}
    thresholds = {}
    g = data["global_fusion"].to(device)
    clip = data["clip18_patches"].to(device).float()
    masks = data["mask_grid"].to(device)
    normal_clip = normal["clip18_patches"].to(device).float()
    name_index = {n: i for i, n in enumerate(normal["names"])}
    top = math.ceil(rp.RHO * clip.shape[1])
    for seed in rp.DEV_SEEDS:
        draw = normal["draws"][str(seed)]
        per_m = {}
        for m, suffix in ((ra.M_PRIMARY, ""), (ra.M_SENS, "_m5")):
            idx = torch.as_tensor([name_index[x] for x in draw[:m]], device=device)
            bank_imgs = normal_clip[idx]
            bank = bank_imgs.reshape(-1, clip.shape[-1])
            anomaly = ra.patch_anomaly(clip, bank)
            pooled, residual = ss.normal_descriptors(clip, bank, device)
            fixed_idx = anomaly.topk(top, dim=1).indices
            region = torch.gather(clip, 1, fixed_idx.unsqueeze(-1).expand(-1, -1, clip.shape[-1]))
            adareg = {}
            loo = loo_anomalies(bank_imgs)
            for key, q in QUANTILES.items():
                tau = float(torch.quantile(loo, q))
                thresholds[f"seed{seed}_m{m}_{key}"] = tau
                for kind in ("hard", "soft"):
                    if kind == "soft" and key != "q99":
                        continue
                    w = region_weights(anomaly, tau, soft=(kind == "soft"))
                    adareg[f"{kind}_{key}"] = weighted_pool(clip, w)
                    if m == ra.M_PRIMARY:
                        count = (w > 0).sum(dim=1).float()
                        sizes[f"{kind}_{key}"] += count.tolist()
                        raw = ((anomaly - tau).clamp_min(0) > 0).sum(dim=1) if kind == "soft" \
                            else (anomaly > tau).sum(dim=1)
                        fallback[f"{kind}_{key}"] += (raw < MIN_REGION).tolist()
                        if kind == "hard":
                            diag[f"hard_{key}"] += region_agreement(w > 0, masks)
            if m == ra.M_PRIMARY:
                fixed = torch.zeros_like(anomaly).scatter_(1, fixed_idx, 1.0)
                diag["fixed_top58"] += region_agreement(fixed > 0, masks)
            per_m[suffix] = {"pooled": pooled, "residual": residual, "region": region,
                             "adareg": adareg}
        for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, data["product"]):
            n = len(order)
            si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
            base = rp.simpleshot(g[si], s_lab, g[qi], n)
            zb = rp._zscore(base)
            xc = rp.local_xclass(clip[si], s_lab, clip[qi], n)
            scores = {rp.PRIMARY: base, "fused_xclass_clip18": zb + rp._zscore(xc),
                      "oracle_maskpool_clip18": rp.oracle_maskpool(clip[si], masks[si], s_lab,
                                                                   clip[qi], masks[qi], n)}
            for suffix, d in per_m.items():
                nr = rp.simpleshot(d["pooled"][si], s_lab, d["pooled"][qi], n)
                rs = rp.simpleshot(d["residual"][si], s_lab, d["residual"][qi], n)
                scores[f"normref_pool_clip18{suffix}"] = nr
                scores[f"fused_normref_clip18{suffix}"] = zb + rp._zscore(nr)
                scores[f"fused_residual_mean_clip18{suffix}"] = zb + rp._zscore(rs)
                for name, h in d["adareg"].items():
                    scores[f"fused_adareg_{name}{suffix}"] = zb + rp._zscore(
                        rp.simpleshot(h[si], s_lab, h[qi], n))
                plain, gx = guided_scores(d["region"], s_lab, si, qi, n)
                scores[f"fused_guided_plain{suffix}"] = zb + rp._zscore(plain)
                scores[f"fused_guided_xclass{suffix}"] = zb + rp._zscore(gx)
                scores[f"fused_guided_xclass_normref{suffix}"] = zb + rp._zscore(nr) + rp._zscore(gx)
            truth = [order[c] for c in q_lab]
            for method, sc in scores.items():
                pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
                rows.append({"method": method, "product": data["product"], "K": k_shot,
                             "seed": seed, **classification_metrics(truth, pred, order)})
    return rows, diag, sizes, fallback, thresholds


def vs_target(rows: list[dict], methods) -> dict:
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    out = {}
    for m in methods:
        target = TARGET + ("_m5" if m.endswith("_m5") else "")
        per_k = {k: 100 * statistics.fmean(
            statistics.fmean(cell[(m, p, k, s)] - cell[(target, p, k, s)] for p in rc.PRODUCTS)
            for s in rp.DEV_SEEDS) for k in rp.DEV_K}
        per_product = {p: 100 * statistics.fmean(cell[(m, p, k, s)] - cell[(target, p, k, s)]
                                                 for k in rp.DEV_K for s in rp.DEV_SEEDS)
                       for p in rc.PRODUCTS}
        mean = statistics.fmean(per_k.values())
        positive = sum(v > 0 for v in per_product.values())
        out[m] = {"reference": target, "delta_pp_by_K": {str(k): v for k, v in per_k.items()},
                  "mean_delta_pp": mean, "positive_products": positive,
                  "per_product_delta_pp": per_product,
                  "gate_pass": None if m.endswith("_m5") else
                  (mean >= GATE_DELTA_PP and positive >= GATE_MIN_POSITIVE)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; write-once output")
    device = torch.device(args.device)

    rows, thresholds = [], {}
    diag = {"fixed_top58": [], **{f"hard_{k}": [] for k in QUANTILES}}
    sizes, fallback = {}, {}
    for product in rc.PRODUCTS:
        data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        if data.get("schema_version") != "p10_dev_features_v1" or \
                normal.get("schema_version") != "p10_normal_features_v1":
            raise rc.P6BFailClosedError(f"unexpected feature schema for {product}")
        with torch.no_grad():
            r, d, sz, fb, th = run_product(data, normal, device)
        rows += r
        for k, v in d.items():
            diag[k] += v
        for k, v in sz.items():
            sizes.setdefault(k, []).extend(v)
        for k, v in fb.items():
            fallback.setdefault(k, []).extend(v)
        thresholds[product] = th
        print(f"[p11] {product} done", flush=True)

    summary = rp.summarize(rows)
    for v in summary["methods"].values():
        v["p10_gate_vs_simpleshot"] = v.pop("gate_pass", None)
    candidates = [m for fam in FAMILIES.values() for m in fam]
    compare = vs_target(rows, candidates + [m + "_m5" for m in candidates]
                        + ["fused_residual_mean_clip18", "fused_xclass_clip18"])
    for m in ("fused_residual_mean_clip18", "fused_xclass_clip18"):
        compare[m]["gate_pass"] = None  # reference rows, not candidates
    selected = {}
    for fam, members in FAMILIES.items():
        passing = [m for m in members if compare[m]["gate_pass"]]
        selected[fam] = max(passing, key=lambda m: (compare[m]["positive_products"],
                                                    compare[m]["mean_delta_pp"]), default=None)
    sanity = {}
    for method, (fname, key) in EXPECTED_P10.items():
        expected = rc.load_json(rp.OUT_DIR / fname)["methods"][key]["mean_delta_pp"]
        got = summary["methods"][method]["mean_delta_pp"]
        sanity[method] = {"recomputed": got, "p10": expected, "reproduced": abs(got - expected) < 1e-9}
    region = {
        "agreement_vs_mask": {k: {s: statistics.fmean(x[s] for x in v) for s in ("precision", "recall", "f1")}
                              | {"images": len(v)} for k, v in diag.items()},
        "mean_region_size": {k: statistics.fmean(v) for k, v in sizes.items()},
        "fallback_fraction": {k: statistics.fmean(float(x) for x in v) for k, v in fallback.items()},
    }
    result = {"schema_version": "p11_pilot_ab_v1",
              "protocol": "docs/P11_ADAPTIVE_REGION_AND_GUIDED_MATCHING_PILOT_PROTOCOL.md",
              "target": TARGET, "gate": {"delta_pp": GATE_DELTA_PP, "min_positive": GATE_MIN_POSITIVE},
              "sanity_vs_p10": sanity, "selected": selected, "vs_target": compare,
              "vs_simpleshot": summary, "region_diagnostics": region, "thresholds": thresholds,
              "provenance": rc.git_provenance(), "rows": rows}

    ss_m = summary["methods"]
    lines = ["# P11 dev pilot: adaptive region (A) and anomaly-guided matching (B)", "",
             "Sanity vs P10 dev: " + ", ".join(f"{m} {'OK' if v['reproduced'] else 'MISMATCH'}"
                                              for m, v in sanity.items()), "",
             f"Gate vs {TARGET}: mean Δ ≥ +{GATE_DELTA_PP} pp and ≥ {GATE_MIN_POSITIVE}/13 products", "",
             "| Method | BA K=1 | BA K=3 | Δ vs SimpleShot | Δ vs normref (K=1 / K=3 / mean) | +products vs normref | Gate |",
             "|---|---|---|---|---|---|---|"]
    ref_rows = [rp.PRIMARY, TARGET, "fused_residual_mean_clip18", "fused_xclass_clip18", "oracle_maskpool_clip18"]
    for m in ref_rows + candidates + [c + "_m5" for c in candidates]:
        v, c = ss_m[m], compare.get(m)
        gate = "ref" if c is None or c["gate_pass"] is None else ("PASS" if c["gate_pass"] else "fail")
        if m.endswith("_m5"):
            gate = "sens"
        vs = (f"{c['delta_pp_by_K']['1']:+.2f} / {c['delta_pp_by_K']['3']:+.2f} / {c['mean_delta_pp']:+.2f}"
              if c else "—")
        pos = f"{c['positive_products']}/13" if c else "—"
        lines.append(f"| {m} | {100 * v['ba_by_K']['1']:.2f} | {100 * v['ba_by_K']['3']:.2f} | "
                     f"{v['mean_delta_pp']:+.2f} ({v['positive_products']}/13) | {vs} | {pos} | {gate} |")
    lines += ["", f"Selected: A = {selected['A']}, B = {selected['B']}", "",
              "Region diagnostics (M=20): precision / recall / F1 vs mask, mean size, fallback fraction"]
    for k, v in region["agreement_vs_mask"].items():
        size_key = k.replace("fixed_top58", "")
        extra = (f", size {region['mean_region_size'][size_key]:.1f}, fallback "
                 f"{region['fallback_fraction'][size_key]:.3f}") if size_key in region["mean_region_size"] else ", size 58"
        lines.append(f"- {k}: {v['precision']:.3f} / {v['recall']:.3f} / {v['f1']:.3f} (n={v['images']}){extra}")
    lines.append(f"- soft_q99: size {region['mean_region_size']['soft_q99']:.1f}, "
                 f"fallback {region['fallback_fraction']['soft_q99']:.3f}")
    text = "\n".join(lines)
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "pilot_ab_results.json", result)
    (OUT / "pilot_ab_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
