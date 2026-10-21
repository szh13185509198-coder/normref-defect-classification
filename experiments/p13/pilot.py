#!/usr/bin/env python3
"""P13 dev pilot: region residual (C1), support-consensus background (C0), Cable multi-region exploration.

Protocol: docs/P13_RESIDUAL_CONSENSUS_MULTIREGION_PILOT_PROTOCOL.md.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p13.pilot --stage dev --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p13.pilot --stage cable-explore --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import random
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
from scipy import ndimage  # noqa: E402

from experiments.p10 import cable_oof as co  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12 import cable_ab  # noqa: E402
from experiments.p12.threshold_controls import P11_DEV, exact_equal  # noqa: E402
from experiments.p12.thresholds import EIGHT_CONNECTED, tau_top10  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = {"dev": PROJECT_ROOT / "ckpt" / "p13_pilot" / "dev",
       "cable-explore": PROJECT_ROOT / "ckpt" / "p13_pilot" / "cable_explore"}
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13",
           "docs/P13_RESIDUAL_CONSENSUS_MULTIREGION_PILOT_PROTOCOL.md")
A, NR, SS, XC = fa.A, fa.NR, fa.SS, "fused_xclass_clip18"
IDEA1 = ("fused_regres_soft_q99", "fused_adareg_regres_soft_q99")
IDEA2 = ("fused_cons_soft_top10", "fused_cons_soft_supq90", "fused_cons_soft_supq95")
GATES = {"idea1": (IDEA1, A, 1.0, 8), "idea2": (IDEA2, XC, 2.0, 8)}
CHUNK = 8


@contextlib.contextmanager
def write_once(out: Path):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; write-once output")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


# --------------------------------------------------------------------------- idea 1

def region_residual(clip: torch.Tensor, bank: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Soft-weighted mean of t - nn_B(t) with A's weights w [N, P]."""
    out = []
    for c, wc in zip(clip.split(CHUNK), w.split(CHUNK)):
        idx = torch.einsum("npd,bd->npb", c, bank).argmax(dim=2)
        out.append(ab.weighted_pool(c - bank[idx], wc))
    return torch.cat(out)


# --------------------------------------------------------------------------- idea 2

def consensus_anomaly(clip: torch.Tensor, si, s_lab: list[int], qi, n: int):
    """Support-consensus anomaly (no normal images): 1 - similarity to a non-own class.

    Queries: second-largest per-class nearest similarity. Supports of class y: max over c != y."""
    if n < 2:
        raise rc.P6BFailClosedError("consensus background needs at least two classes")
    lab = torch.as_tensor(s_lab, device=clip.device)
    sup = clip[si]
    sets = [sup[lab == c].reshape(-1, sup.shape[-1]) for c in range(n)]
    sq = torch.cat([rp._nearest_per_class(q, sets) for q in clip[qi].split(CHUNK)])      # [Q, P, C]
    ss = torch.cat([rp._nearest_per_class(s, sets) for s in sup.split(CHUNK)])           # [S, P, C]
    own = torch.nn.functional.one_hot(lab, n).bool()[:, None, :].expand_as(ss)
    b_sup = ss.masked_fill(own, float("-inf")).max(dim=2).values
    b_q = sq.topk(2, dim=2).values[..., 1]
    return 1.0 - b_sup, 1.0 - b_q


def consensus_scores(clip, si, s_lab, qi, n) -> dict:
    a_s, a_q = consensus_anomaly(clip, si, s_lab, qi, n)
    taus = {"fused_cons_soft_top10": (tau_top10(a_s), tau_top10(a_q))}
    flat = a_s.reshape(-1).double()
    for name, q in (("fused_cons_soft_supq90", 0.90), ("fused_cons_soft_supq95", 0.95)):
        t = float(torch.quantile(flat, q))
        taus[name] = (t, t)
    out = {}
    for name, (ts, tq) in taus.items():
        hs = ab.weighted_pool(clip[si], ab.region_weights(a_s, ts, soft=True))
        hq = ab.weighted_pool(clip[qi], ab.region_weights(a_q, tq, soft=True))
        out[name] = rp.simpleshot(hs, s_lab, hq, n)
    return out


# --------------------------------------------------------------------------- dev

def vs_target(rows, methods, target):
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    out = {}
    for m in methods:
        per_k = {k: 100 * statistics.fmean(statistics.fmean(cell[(m, p, k, s)] - cell[(target, p, k, s)]
                                                            for p in rc.PRODUCTS) for s in rp.DEV_SEEDS) for k in rp.DEV_K}
        per_product = {p: 100 * statistics.fmean(cell[(m, p, k, s)] - cell[(target, p, k, s)]
                                                 for k in rp.DEV_K for s in rp.DEV_SEEDS) for p in rc.PRODUCTS}
        out[m] = {"reference": target, "delta_pp_by_K": {str(k): v for k, v in per_k.items()},
                  "mean_delta_pp": statistics.fmean(per_k.values()),
                  "positive_products": sum(v > 0 for v in per_product.values()), "per_product_delta_pp": per_product}
    return out


def run_dev(device) -> int:
    with write_once(OUT["dev"]) as staging:
        rows = []
        for product in rc.PRODUCTS:
            data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            g = data["global_fusion"].to(device)
            clip = data["clip18_patches"].to(device).float()
            name_index = {x: i for i, x in enumerate(normal["names"])}
            normal_clip = normal["clip18_patches"].to(device).float()
            with torch.no_grad():
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]], device=device)
                    bank_imgs = normal_clip[idx]
                    bank = bank_imgs.reshape(-1, clip.shape[-1])
                    a = ra.patch_anomaly(clip, bank)
                    tau = float(torch.quantile(ab.loo_anomalies(bank_imgs), 0.99))  # P11 pilot path
                    w = ab.region_weights(a, tau, soft=True)
                    h_a = ab.weighted_pool(clip, w)
                    h_rr = region_residual(clip, bank, w)
                    pooled = ra.anomaly_pool(clip, a)
                    for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, product):
                        n = len(order)
                        si = torch.as_tensor(s_idx, device=device)
                        qi = torch.as_tensor(q_idx, device=device)
                        base = rp.simpleshot(g[si], s_lab, g[qi], n)
                        zb = rp._zscore(base)
                        sa = rp.simpleshot(h_a[si], s_lab, h_a[qi], n)
                        srr = rp.simpleshot(h_rr[si], s_lab, h_rr[qi], n)
                        scores = {SS: base, NR: zb + rp._zscore(rp.simpleshot(pooled[si], s_lab, pooled[qi], n)),
                                  A: zb + rp._zscore(sa),
                                  XC: zb + rp._zscore(rp.local_xclass(clip[si], s_lab, clip[qi], n)),
                                  "fused_regres_soft_q99": zb + rp._zscore(srr),
                                  "fused_adareg_regres_soft_q99": zb + rp._zscore(sa) + rp._zscore(srr)}
                        for name, sc in consensus_scores(clip, si, s_lab, qi, n).items():
                            scores[name] = zb + rp._zscore(sc)
                        rows += fa.score_rows(scores, order, [order[c] for c in q_lab],
                                              {"product": product, "K": k_shot, "seed": seed})
            print(f"[p13-dev] {product} done", flush=True)

        integrity = {"A_NR_SS_vs_p11_pilot": exact_equal(rows, rc.load_json(P11_DEV)["rows"], (A, NR, SS), ("product", "K", "seed")),
                     "xclass_vs_p10_pilot": exact_equal(rows, rc.load_json(rp.OUT_DIR / "pilot_results.json")["rows"],
                                                        (XC,), ("product", "K", "seed"))}
        if any(v >= 1e-12 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"dev integrity failed: {integrity}")
        summary = rp.summarize(rows)
        for v in summary["methods"].values():
            v.pop("gate_pass", None)
        compare, selected = {}, {}
        for idea, (members, target, delta, min_pos) in GATES.items():
            c = vs_target(rows, members, target)
            for v in c.values():
                v["gate_pass"] = v["mean_delta_pp"] >= delta and v["positive_products"] >= min_pos
            compare[idea] = c
            passing = [m for m in members if c[m]["gate_pass"]]
            selected[idea] = max(passing, key=lambda m: (c[m]["positive_products"], c[m]["mean_delta_pp"]), default=None)
        cross = vs_target(rows, IDEA2, A)
        result = {"schema_version": "p13_pilot_dev_v1", "protocol": "docs/P13_RESIDUAL_CONSENSUS_MULTIREGION_PILOT_PROTOCOL.md",
                  "provenance": rc.git_provenance(), "integrity": integrity, "gates": {k: v[1:] for k, v in GATES.items()},
                  "vs_target": compare, "idea2_vs_A_cross_condition": cross, "selected": selected,
                  "vs_simpleshot": summary, "rows": rows}
        lines = ["# P13 dev pilot — region residual (C1) and support-consensus background (C0)", "",
                 "Integrity: " + ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Method | Cond. | BA K=1 | BA K=3 | Δ vs SimpleShot (+prod) | Δ vs target (K=1 / K=3 / mean) | +prod vs target | Gate |",
                 "|---|---|---|---|---|---|---|---|"]
        cond = {SS: "C0", XC: "C0", NR: "C1", A: "C1", **{m: "C1" for m in IDEA1}, **{m: "C0" for m in IDEA2}}
        for m in (SS, XC, NR, A, *IDEA1, *IDEA2):
            v = summary["methods"][m]
            c = compare["idea1"].get(m) or compare["idea2"].get(m)
            tgt = (f"{c['delta_pp_by_K']['1']:+.2f} / {c['delta_pp_by_K']['3']:+.2f} / {c['mean_delta_pp']:+.2f} ({c['reference']})"
                   if c else "—")
            lines.append(f"| {m} | {cond[m]} | {100 * v['ba_by_K']['1']:.2f} | {100 * v['ba_by_K']['3']:.2f} | "
                         f"{v['mean_delta_pp']:+.2f} ({v['positive_products']}/13) | {tgt} | "
                         + (f"{c['positive_products']}/13 | {'PASS' if c['gate_pass'] else 'fail'}" if c else "— | ref") + " |")
        lines += ["", f"Selected: idea 1 = {selected['idea1']}, idea 2 = {selected['idea2']}", "",
                  "Idea 2 vs A (cross-condition, descriptive): " + ", ".join(
                      f"{m} {v['mean_delta_pp']:+.2f} ({v['positive_products']}/13)" for m, v in cross.items())]
        text = "\n".join(lines)
        rc.write_json(staging / "p13_dev_results.json", result)
        (staging / "p13_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- Cable exploration

def components(sel: torch.Tensor) -> int:
    side = int(round(sel.numel() ** 0.5))
    return int(ndimage.label(sel.reshape(side, side).cpu().numpy(), structure=EIGHT_CONNECTED)[1])


def run_cable(mvtec_root: Path, device, batch_size: int) -> int:
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10.external_evaluate import extract
    with write_once(OUT["cable-explore"]) as staging:
        folds = {f: co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_val.jsonl") for f in co.FOLDS}
        truth = {r["image_path"]: r["class_name"] for f in co.FOLDS for r in folds[f]}
        pool = sorted(truth)
        normal_dir = mvtec_root / "cable" / "train" / "good"
        normal_names = sorted(p.name for p in normal_dir.glob("*.png"))
        draws = {s: random.Random(f"p10cable|{s}").sample(normal_names, co.BANK) for s in rc.SEEDS}
        if {str(s): v for s, v in draws.items()} != rc.load_json(co.OUT / "cable_results.json")["normal_bank_draws"]:
            raise rc.P6BFailClosedError("normal-bank draws differ from P10 R1")
        bank_names = sorted({n for d in draws.values() for n in d})
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        with torch.no_grad():
            _, clip_all = extract(factory, [co.IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names],
                                  device, batch_size)
            clip = clip_all[:len(pool)].to(device).float()
            bank_index = {n: len(pool) + i for i, n in enumerate(bank_names)}
            per_image = {p: [] for p in pool}
            for seed in rc.SEEDS:
                bank_imgs = clip_all[[bank_index[n] for n in draws[seed]]].to(device).float()
                a = ra.patch_anomaly(clip, bank_imgs.reshape(-1, clip.shape[-1]))
                w = ab.region_weights(a, fa.tau99(bank_imgs), soft=True)
                for p, wi in zip(pool, w):
                    per_image[p].append({"components": components(wi > 0), "size": int((wi > 0).sum())})
        persistent = set(rc.load_json(cable_ab.OUT / "cable_ab_results.json")["persistent_errors_all_seeds"][A])
        classes = sorted({truth[p] for p in pool})

        def stats(images):
            comp = [x["components"] for p in images for x in per_image[p]]
            size = [x["size"] for p in images for x in per_image[p]]
            return {"images": len(images), "mean_components": statistics.fmean(comp),
                    "frac_ge2_components": statistics.fmean(c >= 2 for c in comp), "mean_region_size": statistics.fmean(size)}
        by_class = {c: stats([p for p in pool if truth[p] == c]) for c in classes}
        groups = {"combined": stats([p for p in pool if truth[p] == "combined"]),
                  "single_defect": stats([p for p in pool if truth[p] not in ("combined", co.NORMAL)]),
                  "good": stats([p for p in pool if truth[p] == co.NORMAL]),
                  "A_persistent_errors": stats(sorted(persistent)) if persistent else None}
        result = {"schema_version": "p13_cable_explore_v1", "protocol": "docs/P13_RESIDUAL_CONSENSUS_MULTIREGION_PILOT_PROTOCOL.md#4",
                  "label": "descriptive exploration; no decision rule", "provenance": rc.git_provenance(),
                  "by_class": by_class, "groups": groups, "per_image": per_image}
        lines = ["# P13 Cable multi-region exploration (descriptive; A's region, 5 seeds)", "",
                 "| Class / group | Images | Mean components | Fraction ≥ 2 components | Mean region size |", "|---|---|---|---|---|"]
        for name, s in [*by_class.items(), *((f"[{k}]", v) for k, v in groups.items() if v)]:
            lines.append(f"| {name} | {s['images']} | {s['mean_components']:.2f} | {s['frac_ge2_components']:.2f} | {s['mean_region_size']:.1f} |")
        text = "\n".join(lines)
        rc.write_json(staging / "cable_explore_results.json", result)
        (staging / "cable_explore_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "cable-explore"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "dev":
        return run_dev(device)
    if not args.mvtec_root:
        raise SystemExit("--mvtec-root is required for cable-explore")
    return run_cable(Path(args.mvtec_root), device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
