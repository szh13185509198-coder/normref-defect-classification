#!/usr/bin/env python3
"""Addendum C2 step 1: re-extract frozen Frozen-Fusion query features + fidelity gate.

Extracts the 1152-d frozen fusion feature (CLIP native global embedding + DINOv2
full-patch mean, same factory as the formal ``fusion_proto``) for the 530 formal
queries, then checks that these features reproduce every formal ``fusion_proto``
prediction. C2 baselines may only run if the gate passes.

The feature file stores features keyed by product and rgb_sha256 only (no labels).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.revision.c2_extract_query_features --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch; cole/OpenMP safety)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import (  # noqa: E402
    classification_metrics, load_query_contract,
)
from experiments.revision import revision_common as rc  # noqa: E402

SCORE_TOLERANCE = 1e-3  # cross-hardware; P8 observed max 7.57e-6 on this codebase
BA_TOLERANCE = 0.001  # Amendment C2-A1: 0.1 pp product-macro BA


def extract(queries, device, batch_size) -> dict:
    factory = support._build_lazy_clip_and_dino(
        SimpleNamespace(project_root=PROJECT_ROOT, device=torch.device(device)))
    by_product = {}
    for product in rc.PRODUCTS:
        records = [q for q in queries if q["product"] == product]
        chunks = []
        for start in range(0, len(records), batch_size):
            images = []
            for record in records[start:start + batch_size]:
                with Image.open(record["image_path"]) as handle:
                    images.append(handle.convert("RGB").copy())
            with torch.no_grad():
                clip = factory.clip_extractor.embed_images(images)
                dino = factory.dino_extractor_interface.embed_images(images)
                chunks.append(factory.fusion.fuse(clip, dino).detach().cpu())
        by_product[product] = {"rgb_sha256": [r["rgb_sha256"] for r in records],
                               "features": torch.cat(chunks, dim=0)}
        print(f"[extract] {product}: {len(records)} queries", flush=True)
    return {"features": by_product, "provenance": factory.provenance}


def load_formal_fusion_rows() -> dict:
    rows = {}
    with rc.FORMAL_PREDICTIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["method"] != "fusion_proto":
                continue
            rows[(row["product"], int(row["K"]), int(row["seed"]), row["rgb_sha256"])] = row
    return rows


def fidelity(features: dict) -> dict:
    """Original strict gate (recorded) plus Amendment C2-A1 gate (decisive).

    C2-A1 (docs/P6B_REVISION_ADDENDUM_C.md): scores within SCORE_TOLERANCE; every
    label disagreement is a near-tie whose formal top-1/top-2 margin is at most
    2 x the observed max score difference; locally recomputed fusion_proto
    product-macro BA within BA_TOLERANCE of formal at every K.
    """
    formal = load_formal_fusion_rows()
    checked = label_agree = 0
    max_diff = 0.0
    worst = None
    disagreements = []
    labels = {}  # (product, K, seed) -> (true, local, formal, class_order)
    for product in rc.PRODUCTS:
        entry = features[product]
        query = F.normalize(entry["features"].float(), dim=-1)
        for k_shot in rc.K_VALUES:
            for seed in rc.SEEDS:
                unit = rc.SUPPORT_GRID / "fusion_proto" / product / f"k{k_shot}_s{seed}"
                data = torch.load(unit / "fusion_proto_prototypes.pt", map_location="cpu",
                                  weights_only=False)
                prototypes = F.normalize(torch.as_tensor(data["prototypes"]).float(), dim=-1)
                class_order = data["class_order"]
                scores = query @ prototypes.t()
                unit_labels = labels.setdefault((product, k_shot, seed), ([], [], [], class_order))
                for row_idx, sha in enumerate(entry["rgb_sha256"]):
                    formal_row = formal.get((product, k_shot, seed, sha))
                    if formal_row is None:
                        raise rc.P6BFailClosedError(f"no formal fusion_proto row for {product}/{sha}")
                    if formal_row["class_order"] != class_order:
                        raise rc.P6BFailClosedError(f"class_order mismatch at {product}/K{k_shot}/s{seed}")
                    local = scores[row_idx]
                    diff = float((local - torch.tensor(formal_row["scores"])).abs().max())
                    if diff > max_diff:
                        max_diff, worst = diff, (product, k_shot, seed, sha)
                    predicted = class_order[int(torch.argmax(local))]
                    if predicted == formal_row["predicted_label"]:
                        label_agree += 1
                    else:
                        top = sorted(formal_row["scores"], reverse=True)
                        disagreements.append({"product": product, "K": k_shot, "seed": seed,
                                              "rgb_sha256": sha,
                                              "formal_top2_margin": top[0] - top[1]})
                    unit_labels[0].append(formal_row["true_label"])
                    unit_labels[1].append(predicted)
                    unit_labels[2].append(formal_row["predicted_label"])
                    checked += 1
    expected = len(formal)
    strict = checked == expected and label_agree == expected and max_diff <= SCORE_TOLERANCE

    ba_by_k = {}
    for k_shot in rc.K_VALUES:
        local_ba, formal_ba = [], []
        for seed in rc.SEEDS:
            lb, fb = [], []
            for product in rc.PRODUCTS:
                true, loc, frm, order = labels[(product, k_shot, seed)]
                lb.append(classification_metrics(true, loc, order)["balanced_accuracy"])
                fb.append(classification_metrics(true, frm, order)["balanced_accuracy"])
            local_ba.append(statistics.fmean(lb))
            formal_ba.append(statistics.fmean(fb))
        ba_by_k[str(k_shot)] = {"local": statistics.fmean(local_ba),
                                "formal": statistics.fmean(formal_ba),
                                "abs_difference": abs(statistics.fmean(local_ba) - statistics.fmean(formal_ba))}
    margin_limit = 2.0 * max_diff
    near_tie_ok = all(d["formal_top2_margin"] <= margin_limit for d in disagreements)
    ba_ok = all(v["abs_difference"] <= BA_TOLERANCE for v in ba_by_k.values())
    amended = checked == expected and max_diff <= SCORE_TOLERANCE and near_tie_ok and ba_ok
    return {"checked_rows": checked, "expected_rows": expected,
            "label_agreement": label_agree, "max_abs_score_difference": max_diff,
            "worst_unit": list(worst) if worst else None,
            "score_tolerance": SCORE_TOLERANCE,
            "passed_strict_original": strict,
            "amendment": "C2-A1",
            "disagreements": disagreements,
            "near_tie_margin_limit": margin_limit,
            "all_disagreements_near_tie": near_tie_ok,
            "fusion_proto_product_macro_ba": ba_by_k,
            "ba_tolerance": BA_TOLERANCE,
            "passed": amended}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fidelity-only", action="store_true",
                        help="reuse the saved feature file and only (re)compute the fidelity gate")
    args = parser.parse_args(argv)

    if args.fidelity_only:
        saved = torch.load(rc.QUERY_FEATURES, map_location="cpu", weights_only=False)
        return _report(fidelity(saved["features"]))

    queries, query_audit = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE)
    extracted = extract(queries, args.device, args.batch_size)
    rc.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "revision_c2_fusion_query_features_v1",
        "representation": "frozen_fusion_concat_l2norm (formal fusion_proto factory)",
        "features": extracted["features"],
        "backbone_provenance": extracted["provenance"],
        "formal_query_count": query_audit["formal_query_count"],
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "torch_version": torch.__version__,
        "provenance": rc.git_provenance(),
    }, rc.QUERY_FEATURES)

    return _report(fidelity(extracted["features"]))


def _report(report: dict) -> int:
    report.update({"schema_version": "revision_c2_fidelity_v2",
                   "addendum": "docs/P6B_REVISION_ADDENDUM_C.md#c2",
                   "feature_file": str(rc.QUERY_FEATURES.relative_to(PROJECT_ROOT)),
                   "provenance": rc.git_provenance()})
    rc.write_json(rc.FIDELITY_REPORT, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("FIDELITY GATE FAILED: do not run C2 baselines (Addendum C2).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
