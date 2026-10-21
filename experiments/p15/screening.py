"""P15 two-stage decision layer (docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md). Pure functions."""

from __future__ import annotations

import math

import torch

ALPHAS = (0.01, 0.02, 0.05)
ALPHA_REJECT = 0.01
ALPHA_PASS = (0.10, 0.30)
GOOD_SHARE = 0.95


def image_score(a: torch.Tensor, stat: str) -> torch.Tensor:
    """Image anomaly score from patch anomalies a [N, P]."""
    if stat == "max":
        return a.max(dim=1).values
    if stat == "top1pct":
        return a.topk(math.ceil(0.01 * a.shape[1]), dim=1).values.mean(dim=1)
    raise ValueError(stat)


def conformal_threshold(cal: list[float], alpha: float) -> float:
    """Split-conformal: k-th smallest calibration score, k = ceil((n+1)(1-alpha)); inf if k > n."""
    s = sorted(cal)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else s[k - 1]


def auroc(normal: list[float], defect: list[float]) -> float:
    """P(defect score > normal score) + 0.5 P(tie)."""
    wins = sum((d > n) + 0.5 * (d == n) for d in defect for n in normal)
    return wins / (len(normal) * len(defect))


def restricted_type(scores: torch.Tensor, order: list[str], normal: str) -> list[str]:
    """Argmax over the defect classes only."""
    cols = [i for i, c in enumerate(order) if c != normal]
    return [order[cols[int(j)]] for j in scores[:, cols].argmax(dim=1).cpu()]


def two_level(s: list[float], truth: list[str], types: list[str], normal: str, tau: float) -> dict:
    good = [(si, t) for si, t in zip(s, truth) if t == normal]
    bad = [(si, t, p) for si, t, p in zip(s, truth, types) if t != normal]
    frr = sum(si > tau for si, _ in good) / len(good)
    detected = [(t, p) for si, t, p in bad if si > tau]
    recall = len(detected) / len(bad)
    type_acc = sum(t == p for t, p in detected) / len(detected) if detected else 0.0
    return {"false_reject_rate": frr, "defect_recall": recall, "escape_rate": 1 - recall,
            "type_acc_rejected_defects": type_acc,
            "per_1000": {"false_rejects": 1000 * GOOD_SHARE * frr, "escapes": 1000 * (1 - GOOD_SHARE) * (1 - recall),
                         "wrong_type": 1000 * (1 - GOOD_SHARE) * recall * (1 - type_acc)}}


def three_way(s: list[float], truth: list[str], normal: str, tau_reject: float, tau_pass: float) -> dict:
    good = [si for si, t in zip(s, truth) if t == normal]
    bad = [si for si, t in zip(s, truth) if t != normal]
    auto_fr = sum(x > tau_reject for x in good) / len(good)
    escape = sum(x <= tau_pass for x in bad) / len(bad)
    review_good = sum(tau_pass < x <= tau_reject for x in good) / len(good)
    review_bad = sum(tau_pass < x <= tau_reject for x in bad) / len(bad)
    return {"auto_false_reject_rate": auto_fr, "escape_rate": escape,
            "review_rate_good": review_good, "review_rate_defect": review_bad,
            "per_1000": {"false_rejects": 1000 * GOOD_SHARE * auto_fr, "escapes": 1000 * (1 - GOOD_SHARE) * escape,
                         "review_items": 1000 * (GOOD_SHARE * review_good + (1 - GOOD_SHARE) * review_bad)}}


def oracle_min_review(s: list[float], truth: list[str], normal: str, tau_reject: float,
                      max_escape: float = 0.01) -> dict:
    """Descriptive (uses test labels): lowest pass threshold keeping escape <= max_escape."""
    bad = sorted(x for x, t in zip(s, truth) if t != normal)
    allowed = math.floor(max_escape * len(bad))  # number of defects allowed to auto-pass
    tau_pass = bad[allowed] - 1e-12 if allowed < len(bad) else float("inf")
    tau_pass = min(tau_pass, tau_reject)
    return {"tau_pass": tau_pass, **three_way(s, truth, normal, tau_reject, tau_pass)}
