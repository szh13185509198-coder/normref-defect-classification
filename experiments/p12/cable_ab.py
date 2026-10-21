#!/usr/bin/env python3
"""P12 R1': Cable 8-fold OOF with A and B added (single-shot, descriptive).

Same pool, folds, seeds, normal-bank draws and extraction as P10 R1; the sealed Cable formal
test is never read. Integrity: SimpleShot and fused_normref_clip18 predictions must equal the
published P10 R1 predictions image by image.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.cable_ab --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p10 import cable_oof as co  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.external_evaluate import extract  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12.threshold_controls import single_shot  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p12" / "cable"
P10_CABLE = co.OUT
METHODS = (fa.SS, fa.NR, fa.A, fa.B)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    with single_shot(OUT) as staging:
        provenance = rc.git_provenance()
        folds = {f: (co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_train.jsonl"),
                     co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_val.jsonl")) for f in co.FOLDS}
        truth = {r["image_path"]: r["class_name"] for f in co.FOLDS for r in folds[f][1]}
        order = co.load_v1(111)[next(iter(truth))]["class_names"]
        pool = sorted(truth)
        normal_dir = Path(args.mvtec_root) / "cable" / "train" / "good"
        normal_names = sorted(p.name for p in normal_dir.glob("*.png"))
        draws = {s: random.Random(f"p10cable|{s}").sample(normal_names, co.BANK) for s in rc.SEEDS}
        published = rc.load_json(P10_CABLE / "cable_results.json")
        if {str(s): v for s, v in draws.items()} != published["normal_bank_draws"]:
            raise rc.P6BFailClosedError("normal-bank draws differ from P10 R1")
        bank_names = sorted({n for d in draws.values() for n in d})
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        with torch.no_grad():
            g_all, clip_all = extract(factory, [co.IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names],
                                      device, args.batch_size)
        n_pool = len(pool)
        index = {p: i for i, p in enumerate(pool)}
        bank_index = {n: n_pool + i for i, n in enumerate(bank_names)}
        g = g_all[:n_pool].to(device)
        clip = clip_all[:n_pool].to(device).float()
        top = math.ceil(rp.RHO * clip.shape[1])
        preds = {m: {} for m in METHODS}
        with torch.no_grad():
            for seed in rc.SEEDS:
                bank_imgs = clip_all[[bank_index[n] for n in draws[seed]]].to(device).float()
                a = ra.patch_anomaly(clip, bank_imgs.reshape(-1, clip.shape[-1]))  # the P10 R1 path
                pooled = ra.anomaly_pool(clip, a)
                soft = ab.weighted_pool(clip, ab.region_weights(a, fa.tau99(bank_imgs), soft=True))
                region = torch.gather(clip, 1, a.topk(top, dim=1).indices.unsqueeze(-1).expand(-1, -1, clip.shape[-1]))
                seed_preds = {m: {} for m in METHODS}
                for f in co.FOLDS:
                    train, val = folds[f]
                    si = torch.as_tensor([index[r["image_path"]] for r in train], device=device)
                    sl = [order.index(r["class_name"]) for r in train]
                    qi = torch.as_tensor([index[r["image_path"]] for r in val], device=device)
                    n = len(order)
                    base = rp.simpleshot(g[si], sl, g[qi], n)
                    zb = rp._zscore(base)
                    nr = rp.simpleshot(pooled[si], sl, pooled[qi], n)
                    _, gx = ab.guided_scores(region, sl, si, qi, n)
                    scores = {fa.SS: base, fa.NR: zb + rp._zscore(nr),
                              fa.A: zb + rp._zscore(rp.simpleshot(soft[si], sl, soft[qi], n)),
                              fa.B: zb + rp._zscore(nr) + rp._zscore(gx)}
                    for m, sc in scores.items():
                        for r, c in zip(val, sc.argmax(dim=1).cpu().tolist()):
                            seed_preds[m][r["image_path"]] = order[c]
                for m in METHODS:
                    preds[m][seed] = seed_preds[m]

        old = {}
        for line in (P10_CABLE / "cable_predictions.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["method"] in (fa.SS, fa.NR):
                old[(r["method"], int(r["seed"]), r["image_path"])] = r["pred"]
        mismatch = [(m, s, i) for m in (fa.SS, fa.NR) for s in rc.SEEDS for i in pool if preds[m][s][i] != old[(m, s, i)]]
        if mismatch:
            raise rc.P6BFailClosedError(f"integrity vs P10 R1 failed: {len(mismatch)} predictions, e.g. {mismatch[:3]}")

        summary = co.summarize(preds, truth, order)
        for m in ("v1_ours_tau", "v1_ours_argmax", "residual_mean_clip18"):
            summary[m] = published["summary"][m]
        correct = {m: {i: sum(preds[m][s][i] == truth[i] for s in rc.SEEDS) for i in pool} for m in METHODS}
        persistent = {m: sorted(i for i, c in cc.items() if c == 0) for m, cc in correct.items()}
        persistent.update({m: published["persistent_errors_all_seeds"][m] for m in ("v1_ours_tau", "residual_mean_clip18")})
        result = {"schema_version": "p12_cable_ab_v1", "protocol": "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md#r1",
                  "provenance": provenance, "integrity": {"predictions_rechecked": 2 * len(rc.SEEDS) * len(pool), "mismatches": 0},
                  "caveat": "descriptive; 72 images; V1 rows copied from P10 R1 (V1 was developed on this pool)",
                  "class_order": order, "summary": summary, "persistent_errors_all_seeds": persistent,
                  "correct_seed_counts": correct}

        def cell(m, k):
            v = summary[m]["mean_sd"][k]
            return f"{100 * v['mean']:.2f} ± {100 * v['sd']:.2f}"
        lines = ["# P12 R1' — Cable 8-fold OOF with A and B (72 images, 5 seeds; descriptive)", "",
                 f"Commit `{provenance['head']}`. SimpleShot and normref predictions reproduce P10 R1 exactly.", "",
                 "| Method | 9-class BA | Macro-F1 | Good recall | Defect recall | Binary BA | Type acc. (detected) | Persistent errors (combined) |",
                 "|---|---|---|---|---|---|---|---|"]
        for m in (*METHODS, "residual_mean_clip18", "v1_ours_tau", "v1_ours_argmax"):
            pe = persistent.get(m)
            pe_txt = f"{len(pe)} ({sum(p.startswith('combined/') for p in pe)})" if pe is not None else "—"
            lines.append(f"| {m} | {cell(m, 'balanced_accuracy')} | {cell(m, 'macro_f1')} | {cell(m, 'good_recall')} | "
                         f"{cell(m, 'defect_recall')} | {cell(m, 'binary_ba')} | {cell(m, 'defect_type_acc_detected')} | {pe_txt} |")
        text = "\n".join(lines)
        rc.write_json(staging / "cable_ab_results.json", result)
        rc.write_jsonl(staging / "cable_ab_predictions.jsonl",
                       [{"method": m, "seed": s, "image_path": i, "true": truth[i], "pred": p}
                        for m in METHODS for s in rc.SEEDS for i, p in preds[m][s].items()])
        (staging / "cable_ab_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
