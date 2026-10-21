"""Protocol B K8 nested shot-scaling evaluator (locked rules only).

Implements `docs/P6B_K8_NESTED_SHOT_SCALING_PROTOCOL.md` sections 1/5/7 as a
pure function over per-(method, product, K, seed) metrics:

- d[b,p,s,K] = BA[Ours,p,s,K] - BA[b,p,s,K]
- Delta[b,K] = mean_p mean_s d[b,p,s,K]
- C[b] = Delta[b,8] - Delta[b,1]

- primary scaling contrast: C[MVREC-RGB adaptation];
- predeclared secondary contrast: C[Frozen-Fusion];
- paired product-cluster percentile bootstrap: sampling unit = the nine
  products, all five paired seeds per sampled product stay together,
  10,000 resamples, RNG seed 20260813, two-sided 95% percentile CI;
- CONFIRMED_MONOTONIC_SHOT_SCALING requires ALL of:
  1. C[b] > 0 and its 95% CI lower bound > 0;
  2. Delta[b,1] <= Delta[b,3] <= Delta[b,5] <= Delta[b,8];
  3. at least 6/9 product-level c[b,p] = mean_s(d[..,8]-d[..,1]) positive;
  4. at least 4/5 seed-level c[b,s] positive;
- if Delta[b,8] <= Delta[b,5] or the ordered trajectory breaks at K=8, the
  required decision text is:
  "Positive adaptation benefit exists, but no monotonic shot-scaling
  advantage is confirmed.";
- if C[b] > 0 but CI or paired-direction gates fail:
  DIRECTIONALLY_POSITIVE_BUT_UNCONFIRMED;
- if C[b] <= 0: NO_POSITIVE_K8_VS_K1_SCALING.

K8 does not change the already-passed V1 GO; it only decides whether the
paper may add a shot-scaling mechanism claim.  This module computes no
metric from raw predictions — it consumes already-produced per-unit metric
rows only.
"""

from __future__ import annotations

import math
import random

from experiments.p6.p6b_execution_common import P6BFailClosedError

K8_PRODUCTS = (
    "bottle", "capsule", "hazelnut", "metal_nut", "screw", "zipper",
    "carpet", "leather", "tile",
)
SEEDS = (111, 222, 333, 444, 555)
K_VALUES = (1, 3, 5, 8)
METHODS = ("ours", "mvrec_rgb", "frozen_fusion")
PRIMARY_BASELINE = "mvrec_rgb"
SECONDARY_BASELINE = "frozen_fusion"
PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRIC = "macro_f1"
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_RESAMPLES = 10000
CI_LEVEL = 0.95


def validate_metric_grid(metrics_by_method):
    """Fail-closed completeness: 9 products x 5 seeds x K in {1,3,5,8} per method."""
    for method in METHODS:
        grid = metrics_by_method.get(method)
        if not grid:
            raise P6BFailClosedError(f"missing metric grid for {method}")
        for product in K8_PRODUCTS:
            for k_shot in K_VALUES:
                for seed in SEEDS:
                    row = grid.get((product, k_shot, seed))
                    if not row:
                        raise P6BFailClosedError(
                            f"{method}/{product}/K{k_shot}/s{seed}: missing"
                        )
                    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
                        value = row.get(metric)
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise P6BFailClosedError(
                                f"{method}/{product}/K{k_shot}/s{seed}/{metric}: "
                                f"non-finite {value!r}"
                            )


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def paired_differences(metrics_by_method, baseline):
    """d[b,p,s,K] = BA[ours,p,s,K] - BA[baseline,p,s,K] (and Macro-F1)."""
    ours = metrics_by_method["ours"]
    base = metrics_by_method[baseline]
    d = {}
    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        d[metric] = {
            k: {(p, s): ours[(p, k, s)][metric] - base[(p, k, s)][metric]
                for p in K8_PRODUCTS for s in SEEDS}
            for k in K_VALUES
        }
    return d


def paired_delta(metrics_by_method, baseline):
    """Delta[b,K] = mean_p mean_s d and C[b] = Delta[b,8] - Delta[b,1]."""
    d = paired_differences(metrics_by_method, baseline)
    out = {}
    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        delta = {k: _mean(d[metric][k].values()) for k in K_VALUES}
        out[metric] = {
            "d": d[metric],
            "Delta": delta,
            "C": delta[8] - delta[1],
        }
    return out


def _percentile_bootstrap(values_by_cluster, resamples=BOOTSTRAP_RESAMPLES,
                          seed=BOOTSTRAP_SEED, level=CI_LEVEL):
    clusters = list(values_by_cluster.keys())
    if len(clusters) != len(K8_PRODUCTS):
        raise P6BFailClosedError(
            f"bootstrap cluster set has {len(clusters)} != 9 products"
        )
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _i in range(len(clusters)):
            cluster = rng.choice(clusters)
            for value in values_by_cluster[cluster]:
                total += value
                count += 1
        estimates.append(total / count)
    estimates.sort()
    lower_idx = int(round((1 - level) / 2 * (len(estimates) - 1)))
    upper_idx = int(round((1 + level) / 2 * (len(estimates) - 1)))
    return {
        "resamples": len(estimates),
        "seed": seed,
        "mean": _mean(estimates),
        "ci95_lower": estimates[lower_idx],
        "ci95_upper": estimates[upper_idx],
    }


def contrast_bootstrap(d, baseline, metric):
    """Bootstrap C[b] = mean_p mean_s (d[..,8] - d[..,1]) with product clusters
    (all five paired seeds per sampled product stay together).

    ``d`` is the per-metric difference dict: d[k][(p, s)].
    """
    clusters = {
        p: [d[8][(p, s)] - d[1][(p, s)] for s in SEEDS]
        for p in K8_PRODUCTS
    }
    return _percentile_bootstrap(clusters)


def evaluate_scaling(metrics_by_method, baseline):
    """Protocol B locked evaluation for one baseline method.

    Returns the decision object with state in
    {CONFIRMED_MONOTONIC_SHOT_SCALING,
     DIRECTIONALLY_POSITIVE_BUT_UNCONFIRMED,
     NO_POSITIVE_K8_VS_K1_SCALING}.
    """
    validate_metric_grid(metrics_by_method)
    result = {"baseline": baseline, "metrics": {}}
    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        d = paired_differences(metrics_by_method, baseline)[metric]
        delta = {k: _mean(d[k].values()) for k in K_VALUES}
        c = delta[8] - delta[1]
        boot = contrast_bootstrap(d, baseline, metric)
        # product-level scaling contrasts c[b,p] = mean_s (d8 - d1)
        c_product = {
            p: _mean(d[8][(p, s)] - d[1][(p, s)] for s in SEEDS)
            for p in K8_PRODUCTS
        }
        # seed-level scaling contrasts c[b,s] = mean_p (d8 - d1)
        c_seed = {
            s: _mean(d[8][(p, s)] - d[1][(p, s)] for p in K8_PRODUCTS)
            for s in SEEDS
        }
        result["metrics"][metric] = {
            "d": d,
            "Delta": delta,
            "C": c,
            "C_bootstrap": boot,
            "c_product": c_product,
            "c_seed": c_seed,
        }
    return result


def final_scaling_decision(metrics_by_method):
    """Evaluate both contrasts and return the locked decision text.

    Primary contrast: MVREC-RGB adaptation.  If MVREC-RGB previously
    received STOP_NOT_REPRESENTATIVE its primary contrast would be
    unavailable; here SEMANTIC_PASS was granted, so the primary contrast
    is evaluated.  Secondary contrast: Frozen-Fusion (cannot be silently
    promoted to replacement primary).
    """
    primary = evaluate_scaling(metrics_by_method, PRIMARY_BASELINE)
    secondary = evaluate_scaling(metrics_by_method, SECONDARY_BASELINE)

    def decide(evaluation):
        m = evaluation["metrics"][PRIMARY_METRIC]
        c = m["C"]
        delta = m["Delta"]
        ci_lower = m["C_bootstrap"]["ci95_lower"]
        c_product = m["c_product"]
        c_seed = m["c_seed"]
        g1 = c > 0 and ci_lower > 0
        g2 = (delta[1] <= delta[3] <= delta[5] <= delta[8])
        g3 = sum(v > 0 for v in c_product.values()) >= 6
        g4 = sum(v > 0 for v in c_seed.values()) >= 4
        if g1 and g2 and g3 and g4:
            state = "CONFIRMED_MONOTONIC_SHOT_SCALING"
            text = None
        elif c <= 0:
            # Locked priority: C <= 0 dominates the trajectory-break text
            # (C <= 0 implies Delta[8] <= Delta[1] <= Delta[5]).
            state = "NO_POSITIVE_K8_VS_K1_SCALING"
            text = None
        elif delta[8] <= delta[5]:
            state = "NO_MONOTONIC_ADVANTAGE_AT_K8"
            text = ("Positive adaptation benefit exists, but no monotonic "
                    "shot-scaling advantage is confirmed.")
        else:
            state = "DIRECTIONALLY_POSITIVE_BUT_UNCONFIRMED"
            text = None
        return {
            "state": state,
            "required_decision_text": text,
            "gates": {
                "c_positive_with_ci_lower_positive": g1,
                "delta_monotonic_1_3_5_8": g2,
                "at_least_6_of_9_products_positive": g3,
                "at_least_4_of_5_seeds_positive": g4,
            },
            "C": c,
            "C_ci95": [m["C_bootstrap"]["ci95_lower"],
                       m["C_bootstrap"]["ci95_upper"]],
            "Delta": m["Delta"],
            "c_product_positive_count":
                sum(v > 0 for v in m["c_product"].values()),
            "c_seed_positive_count":
                sum(v > 0 for v in m["c_seed"].values()),
        }

    return {
        "schema_version": "p6b_k8_scaling_decision_v1",
        "primary_metric": "product-macro balanced_accuracy",
        "secondary_metric": "product-macro macro_f1",
        "primary_contrast": decide(primary),
        "secondary_contrast": decide(secondary),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "sampling_unit": "product cluster (five paired seeds together)",
            "level": CI_LEVEL,
        },
        "note": ("K8 does not change the already-passed V1 GO; it only "
                 "decides whether the paper may add a shot-scaling "
                 "mechanism claim."),
    }
