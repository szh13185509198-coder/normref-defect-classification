"""V1 GO/NO-GO decision evaluator (P6B_V1_GO_NO_GO_DECISION_RULE.md).

Implements the locked V1 decision rule comparing Ours vs MVREC-RGB
adaptation on 13 products x K={1,3,5} x seeds {111..555}:

- primary metric: product-macro Balanced Accuracy;
- secondary metric: product-macro Macro-F1;
- d[M,p,s,K] = M[Ours,p,s,K] - M[MVREC-RGB,p,s,K];
- D[M,K] = mean_p mean_s d;  Dbar[M] = mean_K D[M,K];
- uncertainty: paired product-cluster percentile bootstrap, 10,000
  resamples of the 13 products (all five paired seeds and all three K stay
  together per product), RNG seed 20260813, two-sided 95% CI for
  superiority, one-sided 95% lower bounds for non-inferiority;
- non-inferiority margin: 0.010 (BA and Macro-F1, absolute);
- resource-advantage gate: at least one of trainable-ratio <= 0.50,
  batch-1 latency ratio <= 0.75 with batch-64 throughput not worse than
  10%, support-GPU-time ratio <= 0.67 with peak memory not worse than 10%,
  and no measured ratio exceeding 1.25.

Decision states:
- GO -- PERFORMANCE_POSITIVE
- GO -- PRACTICAL_NONINFERIOR (requires the resource-advantage gate)
- NO-GO -- V1_STOP (incl. the stable-MVREC-dominance description)

Pure function of (ours_metrics, mvrec_rgb_metrics, resource): no query data
is read here; the module consumes already-produced per-unit metrics only.
"""

from __future__ import annotations

import math
import random

from experiments.p6.p6b_execution_common import (
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    P6BFailClosedError,
)

PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRIC = "macro_f1"
NI_MARGIN = 0.010
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_RESAMPLES = 10000
CI_LEVEL = 0.95

METHOD_OURS = "ours"
METHOD_MVREC_RGB = "mvrec_rgb"


def validate_metric_grid(metrics_by_method):
    """Validate a {(method, product, K, seed): {metric: value}} grid."""
    for method in (METHOD_OURS, METHOD_MVREC_RGB):
        grid = metrics_by_method.get(method)
        if not grid:
            raise P6BFailClosedError(f"missing metric grid for {method}")
        for product in LOCKED_PRODUCTS:
            for k_shot in LOCKED_K:
                for seed in LOCKED_SEEDS:
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


def paired_differences(metrics_by_method):
    """Return d[metric][K] = mean_p mean_s d and the per-product / per-seed
    arrays used by the decision gates."""
    validate_metric_grid(metrics_by_method)
    ours = metrics_by_method[METHOD_OURS]
    mvrec = metrics_by_method[METHOD_MVREC_RGB]
    d = {}
    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        d[metric] = {}
        for k_shot in LOCKED_K:
            d[metric][k_shot] = {
                (product, seed): ours[(product, k_shot, seed)][metric]
                - mvrec[(product, k_shot, seed)][metric]
                for product in LOCKED_PRODUCTS
                for seed in LOCKED_SEEDS
            }
    D = {metric: {k: _mean(d[metric][k].values()) for k in LOCKED_K}
         for metric in (PRIMARY_METRIC, SECONDARY_METRIC)}
    Dbar = {metric: _mean(D[metric].values())
            for metric in (PRIMARY_METRIC, SECONDARY_METRIC)}
    d_product = {metric: {k: {p: _mean(d[metric][k][(p, s)] for s in LOCKED_SEEDS)
                              for p in LOCKED_PRODUCTS} for k in LOCKED_K}
                 for metric in (PRIMARY_METRIC, SECONDARY_METRIC)}
    d_seed = {metric: {k: {s: _mean(d[metric][k][(p, s)] for p in LOCKED_PRODUCTS)
                           for s in LOCKED_SEEDS} for k in LOCKED_K}
              for metric in (PRIMARY_METRIC, SECONDARY_METRIC)}
    return {"d": d, "D": D, "Dbar": Dbar,
            "d_product": d_product, "d_seed": d_seed}


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def percentile_bootstrap(values, resamples=BOOTSTRAP_RESAMPLES,
                         seed=BOOTSTRAP_SEED, level=CI_LEVEL):
    """Two-sided percentile CI of the mean over paired samples.

    ``values`` is a dict {cluster_key: list_of_paired_observations}; all
    observations of one sampled cluster stay together.
    """
    clusters = list(values.keys())
    n = len(clusters)
    if n == 0:
        raise P6BFailClosedError("bootstrap over empty cluster set")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _i in range(n):
            cluster = rng.choice(clusters)
            for value in values[cluster]:
                total += value
                count += 1
        estimates.append(total / count)
    estimates.sort()
    lower_idx = int(round((1 - level) / 2 * (len(estimates) - 1)))
    upper_idx = int(round((1 + level) / 2 * (len(estimates) - 1)))
    one_sided_lower_idx = int(round((1 - level) * (len(estimates) - 1)))
    return {
        "resamples": len(estimates),
        "seed": seed,
        "mean": _mean(estimates),
        "ci95_lower": estimates[lower_idx],
        "ci95_upper": estimates[upper_idx],
        "ci95_one_sided_lower": estimates[one_sided_lower_idx],
        "ci95_one_sided_upper": estimates[-(one_sided_lower_idx + 1)],
    }


def bootstrap_superiority(paired):
    """Bootstrap CI for Dbar[M] and D[M,K] with product clusters (all five
    seeds x three K of a sampled product stay together)."""
    out = {}
    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        d = paired["d"][metric]
        # Dbar: cluster = product, values = all (K, seed) diffs of the product.
        dbar_clusters = {
            p: [d[k][(p, s)] for k in LOCKED_K for s in LOCKED_SEEDS]
            for p in LOCKED_PRODUCTS
        }
        out[metric] = {
            "Dbar": percentile_bootstrap(dbar_clusters),
            "D": {
                k: percentile_bootstrap({
                    p: [d[k][(p, s)] for s in LOCKED_SEEDS]
                    for p in LOCKED_PRODUCTS
                })
                for k in LOCKED_K
            },
        }
    return out


def resource_advantage_gate(resource):
    """Quantitative resource-advantage gate (decision rule section 5.2).

    All ratios are Ours / MVREC-RGB:
    - trainable ratio: lower-is-better, advantage at <= 0.50;
    - batch-1 latency ratio: lower-is-better, advantage at <= 0.75;
    - batch-64 throughput ratio: HIGHER-is-better; "not worse by more than
      10%" means throughput_ratio >= 0.90, and it must NOT participate in
      the runtime/memory upper-bound gate (throughput has no upper bound);
    - support-GPU-time ratio: lower-is-better, advantage at <= 0.67;
    - peak-memory ratio: lower-is-better, must stay <= 1.10 inside the
      GPU-time advantage condition;
    - every other measured runtime/memory ratio must stay <= 1.25.

    FAIL-CLOSED completeness rule (decision rule section 5.1): ALL five
    ratios must be present, finite and strictly positive.  A missing, NaN,
    negative or zero ratio makes ``clear_resource_advantage=False`` and is
    reported — partial resource data can never open the gate.  Timing
    ratios must come from the dedicated serial alternating-order resource
    measurement on identical hardware; parallel-grid wall-clock timing is
    diagnostic only and must not be fed into this gate.
    """
    required = (
        "trainable_ratio", "latency_ratio", "throughput_ratio",
        "gpu_time_ratio", "memory_ratio",
    )
    values = {}
    problems = []
    for key in required:
        raw = resource.get(key)
        if raw is None:
            problems.append(f"{key}: missing")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{key}: not numeric ({raw!r})")
            continue
        if not math.isfinite(value):
            problems.append(f"{key}: non-finite ({value!r})")
            continue
        if value <= 0:
            problems.append(f"{key}: non-positive ({value!r})")
            continue
        values[key] = value
    if problems:
        return {
            "clear_resource_advantage": False,
            "resource_measurement_problems": problems,
            "condition1_trainable": False,
            "condition2_latency": False,
            "condition3_gpu_time": False,
            "no_measured_ratio_exceeds_1_25": False,
        }

    trainable = values["trainable_ratio"]
    latency = values["latency_ratio"]
    throughput = values["throughput_ratio"]
    gpu_time = values["gpu_time_ratio"]
    memory = values["memory_ratio"]
    cond1 = trainable <= 0.50
    cond2 = latency <= 0.75 and throughput >= 0.90
    cond3 = gpu_time <= 0.67 and memory <= 1.10
    # Runtime/memory upper-bound gate: only lower-is-better runtime and
    # memory ratios participate.  throughput (higher-is-better) and the
    # trainable-parameter ratio (not a runtime/memory dimension) are
    # excluded.
    no_other_exceeds = all(r <= 1.25 for r in (latency, gpu_time, memory))
    return {
        "clear_resource_advantage": bool((cond1 or cond2 or cond3)
                                         and no_other_exceeds),
        "resource_measurement_problems": [],
        "condition1_trainable": cond1,
        "condition2_latency": cond2,
        "condition3_gpu_time": cond3,
        "no_measured_ratio_exceeds_1_25": no_other_exceeds,
    }


def evaluate_v1_decision(metrics_by_method, resource=None):
    """Evaluate the locked V1 GO/NO-GO decision.

    Returns the decision object with state in
    {GO-PERFORMANCE_POSITIVE, GO-PRACTICAL_NONINFERIOR, NO-GO-V1_STOP}.
    """
    paired = paired_differences(metrics_by_method)
    boot = bootstrap_superiority(paired)
    resource = resource or {}

    ba = paired["D"][PRIMARY_METRIC]
    ba_bar = paired["Dbar"][PRIMARY_METRIC]
    f1 = paired["D"][SECONDARY_METRIC]
    f1_bar = paired["Dbar"][SECONDARY_METRIC]

    # -- PERFORMANCE_POSITIVE gates
    g1 = boot[PRIMARY_METRIC]["Dbar"]["ci95_lower"] > 0
    g2 = (sum(ba[k] > 0 for k in LOCKED_K) >= 2
          and all(ba[k] >= -NI_MARGIN for k in LOCKED_K))
    d_product_omnibus = {
        p: _mean(paired["d"][PRIMARY_METRIC][k][(p, s)]
                 for k in LOCKED_K for s in LOCKED_SEEDS)
        for p in LOCKED_PRODUCTS
    }
    g3 = sum(v > 0 for v in d_product_omnibus.values()) >= 7
    d_seed_omnibus = {
        s: _mean(paired["d"][PRIMARY_METRIC][k][(p, s)]
                 for p in LOCKED_PRODUCTS for k in LOCKED_K)
        for s in LOCKED_SEEDS
    }
    g4 = sum(v > 0 for v in d_seed_omnibus.values()) >= 4
    g5 = (f1_bar >= 0
          and sum(f1[k] >= 0 for k in LOCKED_K) >= 2
          and all(f1[k] >= -NI_MARGIN for k in LOCKED_K))
    performance_positive = bool(g1 and g2 and g3 and g4 and g5)

    # -- PRACTICAL_NONINFERIOR gates (only if not performance-positive)
    ni1 = all(boot[PRIMARY_METRIC]["D"][k]["ci95_one_sided_lower"] > -NI_MARGIN
              for k in LOCKED_K)
    ni2 = boot[PRIMARY_METRIC]["Dbar"]["ci95_one_sided_lower"] > -NI_MARGIN
    ni3 = (boot[SECONDARY_METRIC]["Dbar"]["ci95_one_sided_lower"] > -NI_MARGIN
           and all(f1[k] >= -NI_MARGIN for k in LOCKED_K))
    resource_gate = resource_advantage_gate(resource)
    practical_noninferior = bool(ni1 and ni2 and ni3
                                 and resource_gate["clear_resource_advantage"])

    # -- stable MVREC-RGB dominance description (decision rule 6.3)
    negative_ba = sum(ba[k] < 0 for k in LOCKED_K)
    negative_f1 = sum(f1[k] < 0 for k in LOCKED_K)
    dominance_upper = boot[PRIMARY_METRIC]["Dbar"]["ci95_upper"] < 0
    dominance_products = sum(v < 0 for v in d_product_omnibus.values()) >= 7
    dominance_seeds = sum(v < 0 for v in d_seed_omnibus.values()) >= 4
    stable_dominance = bool(
        negative_ba >= 2 and negative_f1 >= 2 and dominance_upper
        and dominance_products and dominance_seeds
    )

    if performance_positive:
        state = "GO-PERFORMANCE_POSITIVE"
    elif practical_noninferior:
        state = "GO-PRACTICAL_NONINFERIOR"
    else:
        state = "NO-GO-V1_STOP"

    return {
        "schema_version": "p6b_v1_decision_v1",
        "state": state,
        "stable_mvrec_dominance": stable_dominance,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "non_inferiority_margin": NI_MARGIN,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "sampling_unit": "product cluster (five paired seeds together)",
            "level": CI_LEVEL,
        },
        "D": paired["D"],
        "Dbar": paired["Dbar"],
        "bootstrap_results": boot,
        "d_product_omnibus": d_product_omnibus,
        "d_seed_omnibus": d_seed_omnibus,
        "gates": {
            "g1_dbar_ba_ci_lower_positive": g1,
            "g2_ba_two_k_positive_no_below_margin": g2,
            "g3_at_least_7_products_positive": g3,
            "g4_at_least_4_seeds_positive": g4,
            "g5_macro_f1_nonnegative": g5,
            "ni1_all_k_ba_one_sided": ni1,
            "ni2_dbar_ba_one_sided": ni2,
            "ni3_macro_f1_one_sided": ni3,
            "resource_advantage_gate": resource_gate,
            "stable_dominance_conditions": {
                "negative_ba_k_count": negative_ba,
                "negative_f1_k_count": negative_f1,
                "dbar_ba_ci_upper_negative": dominance_upper,
                "at_least_7_products_negative": dominance_products,
                "at_least_4_seeds_negative": dominance_seeds,
            },
        },
    }
