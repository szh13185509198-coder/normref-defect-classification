#!/usr/bin/env python3
"""R1: Cable 8-fold OOF — training-free P10 cues vs the archived V1 (P3.F) method.

Single-shot (docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md, R1). The sealed Cable formal test
is never read.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.cable_oof --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10.external_evaluate import extract  # noqa: E402
from experiments.p10.industrial_metrics import mean_sd, normal_aware  # noqa: E402
from experiments.p10.run_controls_a import centered_proto  # noqa: E402
from experiments.p10.supplement_s import normal_descriptors  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

FOLD_DIR = PROJECT_ROOT / "dataset" / "metadata" / "MVTecAD-Cable" / "oof_8fold_v1"
IMAGE_ROOT = PROJECT_ROOT / "data" / "mvtecad-cable"
P3F_DIR = PROJECT_ROOT / "ckpt" / "p3f_clean_five_seed"
OUT = PROJECT_ROOT / "ckpt" / "p10_cable" / "results"
GUARDED = ("experiments/p10", "experiments/revision", "docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md")
FOLDS = range(8)
BANK = 20
TAU = 0.006
NORMAL = "good"
NEW_METHODS = ("fusion_proto_local", "simpleshot_cl2n", "fused_xclass_clip18",
               "fused_normref_clip18", "residual_mean_clip18", "fused_normcenter_mean_clip18")


def guard_and_lock() -> dict:
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; R1 is single-shot")
    lock = OUT.with_name(OUT.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    return rc.git_provenance()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v1(seed: int) -> dict:
    """{image_path: archived P3.F OOF row} for one seed."""
    out = {}
    for fold in FOLDS:
        for row in read_jsonl(P3F_DIR / f"s{seed}" / f"fold_{fold:02d}" / f"oof_predictions_fold_{fold:02d}.jsonl"):
            out[row["image_path"]] = row
    if len(out) != 72:
        raise rc.P6BFailClosedError(f"V1 seed {seed}: {len(out)} OOF rows, expected 72")
    return out


def summarize(method_preds: dict, truth: dict, order: list[str]) -> dict:
    """method -> seed -> {image: pred}  =>  per-seed metrics and mean/sd."""
    out = {}
    images = sorted(truth)
    for method, by_seed in method_preds.items():
        per_seed = {}
        for seed, preds in by_seed.items():
            t = [truth[i] for i in images]
            p = [preds[i] for i in images]
            per_seed[str(seed)] = {**classification_metrics(t, p, order), **normal_aware(t, p, NORMAL)}
        keys = next(iter(per_seed.values())).keys()
        out[method] = {"per_seed": per_seed,
                       "mean_sd": {k: mean_sd(v[k] for v in per_seed.values())
                                   for k in keys if not k.startswith("n_")}}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    provenance = guard_and_lock()
    device = torch.device(args.device)

    folds = {f: (read_jsonl(FOLD_DIR / f"fold_{f:02d}_train.jsonl"),
                 read_jsonl(FOLD_DIR / f"fold_{f:02d}_val.jsonl")) for f in FOLDS}
    truth = {r["image_path"]: r["class_name"] for f in FOLDS for r in folds[f][1]}
    if len(truth) != 72:
        raise rc.P6BFailClosedError(f"OOF union has {len(truth)} images, expected 72")
    order = load_v1(111)[next(iter(truth))]["class_names"]
    pool = sorted(truth)
    normal_dir = Path(args.mvtec_root) / "cable" / "train" / "good"
    normal_names = sorted(p.name for p in normal_dir.glob("*.png"))
    pool_hashes = {sha(IMAGE_ROOT / p) for p in pool}
    overlap = [n for n in normal_names if sha(normal_dir / n) in pool_hashes]
    if overlap:
        raise rc.P6BFailClosedError(f"normal bank overlaps the Cable pool: {overlap[:3]}")
    draws = {s: random.Random(f"p10cable|{s}").sample(normal_names, BANK) for s in rc.SEEDS}
    bank_names = sorted({n for d in draws.values() for n in d})

    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    paths = [IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names]
    with torch.no_grad():
        g_all, clip_all = extract(factory, paths, device, args.batch_size)
    n_pool = len(pool)
    index = {p: i for i, p in enumerate(pool)}
    bank_index = {n: n_pool + i for i, n in enumerate(bank_names)}
    g = g_all[:n_pool].to(device)
    clip = clip_all[:n_pool]

    preds = {m: {} for m in (*NEW_METHODS, "v1_ours_argmax", "v1_ours_tau")}
    with torch.no_grad():
        for seed in rc.SEEDS:
            bank_imgs = clip_all[[bank_index[n] for n in draws[seed]]].to(device).float()
            pooled, residual = normal_descriptors(clip, bank_imgs.reshape(-1, clip.shape[-1]), device)
            mean_desc = clip.to(device).float().mean(dim=1)
            normal_mu = bank_imgs.mean(dim=1).mean(dim=0, keepdim=True)
            seed_preds = {m: {} for m in NEW_METHODS}
            for f in FOLDS:
                train, val = folds[f]
                si = torch.as_tensor([index[r["image_path"]] for r in train], device=device)
                sl = [order.index(r["class_name"]) for r in train]
                qi = torch.as_tensor([index[r["image_path"]] for r in val], device=device)
                n = len(order)
                base = rp.simpleshot(g[si], sl, g[qi], n)
                zb = rp._zscore(base)
                xc = rp.local_xclass(clip[si.cpu()].to(device).float(), sl, clip[qi.cpu()].to(device).float(), n)
                scores = {
                    "fusion_proto_local": rp.fusion_proto(g[si], sl, g[qi], n),
                    "simpleshot_cl2n": base,
                    "fused_xclass_clip18": zb + rp._zscore(xc),
                    "fused_normref_clip18": zb + rp._zscore(rp.simpleshot(pooled[si], sl, pooled[qi], n)),
                    "residual_mean_clip18": rp.simpleshot(residual[si], sl, residual[qi], n),
                    "fused_normcenter_mean_clip18": zb + rp._zscore(
                        centered_proto(mean_desc[si], sl, mean_desc[qi], n, normal_mu)),
                }
                for m, sc in scores.items():
                    for r, c in zip(val, sc.argmax(dim=1).cpu().tolist()):
                        seed_preds[m][r["image_path"]] = order[c]
            for m in NEW_METHODS:
                preds[m][seed] = seed_preds[m]
            v1 = load_v1(seed)
            preds["v1_ours_argmax"][seed] = {i: v1[i]["predicted_class_name"] for i in pool}
            # archived V1 normal-aware rule: defect iff defect-minus-good margin > tau; the
            # defect type is then the best defect class; otherwise good.
            preds["v1_ours_tau"][seed] = {
                i: (v1[i]["best_defect_class_name"] if v1[i]["defect_minus_good_margin"] > TAU else NORMAL)
                for i in pool}

    summary = summarize(preds, truth, order)
    audit = rc.load_json(P3F_DIR / "p3f_final_audit.json")["seed_metrics"]
    reproduction = {}
    for seed in rc.SEEDS:
        a = audit[str(seed)]
        arg = summary["v1_ours_argmax"]["per_seed"][str(seed)]
        tau = summary["v1_ours_tau"]["per_seed"][str(seed)]
        reproduction[str(seed)] = {
            "nine_class_ba_audit": a["nine_class_balanced_accuracy"],
            "nine_class_ba_argmax": arg["balanced_accuracy"], "nine_class_ba_tau": tau["balanced_accuracy"],
            "good_recall_audit": a["good_recall"], "good_recall_tau": tau["good_recall"],
            "defect_recall_audit": a["defect_recall"], "defect_recall_tau": tau["defect_recall"]}
    reproduced = all(abs(r["nine_class_ba_audit"] - r["nine_class_ba_tau"]) < 1e-12
                     and abs(r["good_recall_audit"] - r["good_recall_tau"]) < 1e-12
                     and abs(r["defect_recall_audit"] - r["defect_recall_tau"]) < 1e-12
                     for r in reproduction.values())
    d = 100 * (summary["v1_ours_argmax"]["mean_sd"]["balanced_accuracy"]["mean"]
               - summary["simpleshot_cl2n"]["mean_sd"]["balanced_accuracy"]["mean"])
    reading = ("features, not adaptation" if d <= 2.0 else
               "adaptation helps at 8-shot" if d >= 5.0 else "inconclusive")
    correct_counts = {m: {i: sum(preds[m][s][i] == truth[i] for s in rc.SEEDS) for i in pool} for m in preds}
    persistent = {m: sorted(i for i, c in cc.items() if c == 0) for m, cc in correct_counts.items()}
    result = {"schema_version": "p10_cable_oof_v1", "protocol": "docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md#r1",
              "provenance": provenance, "class_order": order, "normal_bank_draws": {str(s): v for s, v in draws.items()},
              "v1_minus_simpleshot_ba_pp": d, "reading_rule_outcome": reading,
              "caveat": "V1 was developed on this pool (home advantage); 72 images, 1 image = 1.39 pp Top-1",
              "v1_audit_reproduction": reproduction, "v1_audit_reproduced": reproduced,
              "summary": summary, "persistent_errors_all_seeds": persistent,
              "correct_seed_counts": correct_counts}

    def cell(m, k):
        v = summary[m]["mean_sd"][k]
        return f"{100 * v['mean']:.2f} ± {100 * v['sd']:.2f}"
    lines = ["# R1 — Cable 8-fold OOF (72 images, 5 seeds)", "",
             f"Commit `{provenance['head']}`. V1 − SimpleShot (9-class BA) = {d:+.2f} pp → **{reading}**. "
             "Caveat: V1 was designed on this pool; 72 images.",
             f"V1 archive reproduces the P3.F audit (tau-calibrated 9-class BA, good/defect recall): {reproduced}. "
             "Reading rule uses the argmax V1 row (no threshold for any method); the tau row is the paper-reported V1.", "",
             "| Method | 9-class BA | Macro-F1 | Good recall | Defect recall | Binary BA | Type acc. (detected) | Persistent errors |",
             "|---|---|---|---|---|---|---|---|"]
    for m in (*NEW_METHODS, "v1_ours_argmax", "v1_ours_tau"):
        lines.append(f"| {m} | {cell(m, 'balanced_accuracy')} | {cell(m, 'macro_f1')} | {cell(m, 'good_recall')} | "
                     f"{cell(m, 'defect_recall')} | {cell(m, 'binary_ba')} | {cell(m, 'defect_type_acc_detected')} | "
                     f"{len(persistent[m])} |")
    text = "\n".join(lines)
    staging = OUT.with_name(OUT.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rc.write_json(staging / "cable_results.json", result)
    rc.write_jsonl(staging / "cable_predictions.jsonl",
                   [{"method": m, "seed": s, "image_path": i, "true": truth[i], "pred": p}
                    for m in preds for s in rc.SEEDS for i, p in preds[m][s].items()])
    (staging / "cable_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    staging.rename(OUT)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
