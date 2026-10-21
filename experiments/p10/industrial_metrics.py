"""Normal-aware industrial decision metrics (docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md).

Decisions are the argmax class; a prediction counts as "defect" whenever it is any
non-normal class. No threshold is involved.
"""

from __future__ import annotations

import statistics


def normal_aware(truth: list[str], pred: list[str], normal: str) -> dict:
    if len(truth) != len(pred) or not truth:
        raise ValueError("truth and pred must be aligned and non-empty")
    good = [(t, p) for t, p in zip(truth, pred) if t == normal]
    bad = [(t, p) for t, p in zip(truth, pred) if t != normal]
    if not good or not bad:
        raise ValueError("normal-aware metrics need both normal and defect queries")
    good_recall = sum(p == normal for _, p in good) / len(good)
    detected = [(t, p) for t, p in bad if p != normal]
    defect_recall = len(detected) / len(bad)
    type_acc = (sum(t == p for t, p in detected) / len(detected)) if detected else 0.0
    return {"good_recall": good_recall, "false_reject_rate": 1.0 - good_recall,
            "defect_recall": defect_recall, "escape_rate": 1.0 - defect_recall,
            "binary_ba": 0.5 * (good_recall + defect_recall),
            "defect_type_acc_detected": type_acc,
            "n_good": len(good), "n_defect": len(bad)}


def per_thousand(frr: float, escape: float, good_share: float = 0.95) -> dict:
    return {"good_share": good_share,
            "false_rejects_per_1000": 1000.0 * good_share * frr,
            "escapes_per_1000": 1000.0 * (1.0 - good_share) * escape}


def mean_sd(values) -> dict:
    values = list(values)
    return {"mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}
