#!/usr/bin/env python3
"""Addendum C2 step 2: SimpleShot-CL2N and RFS-style logistic regression.

Both baselines use the formal Frozen-Fusion support embeddings (bitwise, from the
formal support grid) and the re-extracted query features from step 1. Refuses to
run unless the step-1 fidelity gate passed.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.revision.c2_frozen_feature_baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from experiments.p6.p6b_stage3_common import (  # noqa: E402
    classification_metrics, load_query_contract,
)
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

METHODS = ("simpleshot_cl2n", "logreg_rfs", "fusion_proto_local")
LOGREG_KWARGS = {"solver": "lbfgs", "C": 1.0, "max_iter": 1000}


def _first_argmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise argmax; ties resolve to the lowest index (manifest class_order)."""
    return np.argmax(scores, axis=1)


def simpleshot_cl2n(support: torch.Tensor, labels: list[int], query: torch.Tensor,
                    n_classes: int) -> np.ndarray:
    """Center by the support mean of the current unit, L2, class-mean prototypes, cosine."""
    mu = support.mean(dim=0, keepdim=True)
    s = F.normalize(support - mu, dim=-1)
    q = F.normalize(query - mu, dim=-1)
    label_t = torch.as_tensor(labels)
    prototypes = torch.stack([F.normalize(s[label_t == c].mean(dim=0), dim=-1)
                              for c in range(n_classes)])
    return (q @ prototypes.t()).double().numpy()


def logreg_rfs(support: torch.Tensor, labels: list[int], query: torch.Tensor,
               n_classes: int) -> np.ndarray:
    """RFS-style probe on the L2-normalized features; no tuning, no augmentation."""
    x = F.normalize(support, dim=-1).double().numpy()
    q = F.normalize(query, dim=-1).double().numpy()
    clf = LogisticRegression(**LOGREG_KWARGS)
    clf.fit(x, np.asarray(labels))
    if list(clf.classes_) != list(range(n_classes)):
        raise rc.P6BFailClosedError(f"logreg classes {clf.classes_} != 0..{n_classes - 1}")
    return clf.decision_function(q).reshape(len(q), n_classes)


def fusion_proto_local(support: torch.Tensor, labels: list[int], query: torch.Tensor,
                       n_classes: int, prototypes: torch.Tensor = None) -> np.ndarray:
    """Formal fusion_proto prototypes against the locally extracted query features.

    Within-hardware reference for SimpleShot / logreg (Amendment C2-A1)."""
    return (F.normalize(query, dim=-1) @ F.normalize(prototypes, dim=-1).t()).double().numpy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    fidelity = rc.load_json(rc.FIDELITY_REPORT) if rc.FIDELITY_REPORT.is_file() else {}
    if fidelity.get("passed") is not True:
        raise rc.P6BFailClosedError("C2 fidelity gate has not passed; run step 1 first")
    feature_file = torch.load(rc.QUERY_FEATURES, map_location="cpu", weights_only=False)
    features = feature_file["features"]

    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE,
                                     verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}

    raw_metrics, prediction_rows = [], []
    for product in rc.PRODUCTS:
        entry = features[product]
        query = entry["features"].float()
        shas = entry["rgb_sha256"]
        true_labels = [truth[(product, sha)] for sha in shas]
        for k_shot in rc.K_VALUES:
            for seed in rc.SEEDS:
                unit = rc.SUPPORT_GRID / "fusion_proto" / product / f"k{k_shot}_s{seed}"
                data = torch.load(unit / "fusion_proto_prototypes.pt", map_location="cpu",
                                  weights_only=False)
                class_order = list(data["class_order"])
                support = torch.as_tensor(data["support_embeddings"]).float()
                labels = [int(v) for v in data["support_labels"]]
                if len(labels) != len(class_order) * k_shot or support.shape[0] != len(labels):
                    raise rc.P6BFailClosedError(f"support size mismatch at {unit}")
                formal_prototypes = torch.as_tensor(data["prototypes"]).float()
                for method, fn in (("simpleshot_cl2n", simpleshot_cl2n),
                                   ("logreg_rfs", logreg_rfs),
                                   ("fusion_proto_local", fusion_proto_local)):
                    kwargs = {"prototypes": formal_prototypes} if method == "fusion_proto_local" else {}
                    scores = fn(support, labels, query, len(class_order), **kwargs)
                    predicted = [class_order[i] for i in _first_argmax(scores)]
                    metrics = classification_metrics(true_labels, predicted, class_order)
                    raw_metrics.append({"method": method, "product": product, "K": k_shot,
                                        "seed": seed, **metrics})
                    for sha, pred, true, row in zip(shas, predicted, true_labels, scores):
                        prediction_rows.append({
                            "method": method, "product": product, "K": k_shot, "seed": seed,
                            "rgb_sha256": sha, "class_order": class_order,
                            "predicted_label": pred, "true_label": true,
                            "scores": [float(v) for v in row],
                        })
        print(f"[c2] {product} done", flush=True)

    grid = rc.metric_grid_from_rows(raw_metrics)
    results = {
        "schema_version": "revision_c2_frozen_feature_baselines_v1",
        "addendum": "docs/P6B_REVISION_ADDENDUM_C.md#c2",
        "label": "post-hoc revision analysis",
        "fidelity_gate_passed": True,
        "fidelity_report": str(rc.FIDELITY_REPORT.relative_to(PROJECT_ROOT)),
        "methods": {
            "simpleshot_cl2n": "support-mean centering, L2, class-mean prototypes, cosine argmax",
            "logreg_rfs": f"sklearn LogisticRegression{LOGREG_KWARGS} on L2-normalized fused features",
            "fusion_proto_local": "formal fusion_proto prototypes x local query features (reference)",
        },
        "provenance": rc.git_provenance(),
        "product_macro_summary": {m: rc.product_macro_summary(grid, m) for m in METHODS},
        "within_feature_contrasts": {
            f"{target}_minus_fusion_proto_local": {
                metric: contrast(grid, "fusion_proto_local", metric, target=target)
                for metric in rc.METRICS}
            for target in ("simpleshot_cl2n", "logreg_rfs")},
        "raw_metrics": raw_metrics,
    }
    rc.write_json(rc.C2_RESULTS, results)
    rc.write_jsonl(rc.C2_PREDICTIONS, prediction_rows)
    for method in METHODS:
        summary = results["product_macro_summary"][method]
        cells = ", ".join(f"K={k}: BA {100 * summary[str(k)]['balanced_accuracy']['mean']:.2f}"
                          f"±{100 * summary[str(k)]['balanced_accuracy']['std_ddof1']:.2f}"
                          for k in rc.K_VALUES)
        print(f"{method}: {cells}")
    for name, by_metric in results["within_feature_contrasts"].items():
        ba = by_metric["balanced_accuracy"]
        cells = ", ".join(f"K={k}: {100 * ba['per_K'][str(k)]['D']:+.2f} "
                          f"[{100 * ba['per_K'][str(k)]['ci95'][0]:+.2f}, "
                          f"{100 * ba['per_K'][str(k)]['ci95'][1]:+.2f}] "
                          f"({ba['per_K'][str(k)]['positive_products']}/13)" for k in rc.K_VALUES)
        print(f"{name} BA: {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
