# P6.B V1 GO/NO-GO Preregistered Decision Rule

> Status: **PREREGISTERED — LOCKED BEFORE ANY MVREC-RGB/K8 PERFORMANCE**
>
> Date locked: 2026-08-13

## 0. Purpose and firewall

This rule determines whether the current V1 method is strong enough to support the
strict RGB-only SCI-Q2 paper route or must stop in favour of a genuinely new V2
mechanism. Existing P6.B results are known; no MVREC-RGB or K=8 performance is known.

Protocol A, Protocol B, and this rule must be archived together before any new
performance run. The decision evaluator must verify their canonical-LF SHA256 values
and the clean execution commit before reading predictions.

Engineering integrity failure yields `INCONCLUSIVE_ENGINEERING` and an exact rerun.
It never permits threshold, method, metric, product, or exclusion changes.

## 1. Locked comparison population

The V1 performance decision uses Ours versus **MVREC-RGB adaptation** on:

- all original 13 P6.B products;
- K = 1, 3, 5;
- paired seeds 111, 222, 333, 444, 555;
- exact P6.B support and query identities;
- the same image-level product-specific class universes.

Protocol A must first return `SEMANTIC_PASS`. If it returns
`STOP_NOT_REPRESENTATIVE`, no substitute weakened MVREC may be created and V1 cannot
receive a positive GO from this rule. Under the current no-baseline-expansion route,
the Q2 evidence gate remains unmet and the project proceeds to the V2 decision.

## 2. Metrics and paired estimands

Primary metric: **product-macro Balanced Accuracy (BA)**.

Secondary metric: **product-macro Macro-F1**.

For metric `M`, product `p`, seed `s`, and shot `K`, define:

```text
d[M,p,s,K] = M[Ours,p,s,K] - M[MVREC-RGB,p,s,K]
D[M,K]     = mean_p mean_s d[M,p,s,K]
Dbar[M]    = mean over K in {1,3,5} of D[M,K]
```

Products and K values are equally weighted. Sample-count weighting is forbidden.

Paired product evidence uses
`d_product[M,p,K] = mean_s d[M,p,s,K]`. Paired seed evidence uses
`d_seed[M,s,K] = mean_p d[M,p,s,K]`.

## 3. Locked uncertainty calculation

Use a paired product-cluster percentile bootstrap:

- resample the 13 products with replacement;
- retain all five paired seeds and all three K values for each sampled product;
- use 10,000 resamples and RNG seed `20260813`;
- use identical resample indices for BA and Macro-F1;
- report two-sided 95% CIs for superiority estimates;
- report one-sided 95% lower bounds for non-inferiority.

Products are the generalization unit; seeds are repeated paired measurements, not 65
independent products.

## 4. Practical non-inferiority margin

The absolute non-inferiority margin is locked to:

```text
margin_BA       = 0.010  # one percentage point
margin_MacroF1  = 0.010  # one percentage point
```

The margin applies to product-macro metrics, not relative percentage change. It may
not be enlarged after results.

## 5. Resource measurement and advantage

All resource dimensions must be measured and reported; a favourable dimension
cannot be shown while suppressing an unfavourable one.

### 5.1 Measurements

- **Annotation requirement:** exact human-provided support/query fields. Strict
  RGB-only plus image-level class labels is the maximum allowed for both methods.
- **Trainable parameters:** unique scalar parameters with `requires_grad=True` and
  present in the optimizer; report total parameters separately.
- **Support training:** median GPU-seconds and peak allocated GPU memory across all
  formal `(product,K,seed)` units on the same device/software stack.
- **Query latency:** use the same original-image sequence, device, precision, and
  batch; each method uses its own pre-locked deterministic evaluation transform.
  Exclude disk I/O but include preprocessing after decode, host-to-device transfer,
  feature extraction, and classification. Use 50 warm-up iterations followed by at
  least 500 timed images with CUDA synchronization. Batch-1 latency is primary;
  batch-64 throughput is secondary.
- **Implementation complexity:** number of trainable components, backbone passes,
  and external model dependencies. Lines of code are descriptive only.

The hardware model, driver, CUDA, framework versions, precision, power limit, batch,
and timing script hash must be recorded. Methods must be timed serially in alternating
order, with five timing repeats; compare medians.

### 5.2 Quantitative resource-advantage gate

Ours has a **clear resource advantage** only if it uses no additional annotation and
at least one of the following preregistered conditions holds:

1. trainable-parameter ratio `Ours / MVREC-RGB <= 0.50`; or
2. primary batch-1 latency ratio `<= 0.75`, while batch-64 throughput is not worse by
   more than 10%; or
3. support-training GPU-time ratio `<= 0.67`, while peak GPU memory is not worse by
   more than 10%.

In addition, no other measured runtime/memory ratio may exceed `1.25`. Implementation
complexity or comparison with privileged official MVREC cannot alone pass this gate.

## 6. V1 decision states

### 6.1 GO — PERFORMANCE_POSITIVE

All conditions are required:

1. the 95% CI lower bound for `Dbar[BA]` is greater than zero;
2. `D[BA,K] > 0` for at least two of K=1/3/5, and no `D[BA,K] < -0.010`;
3. at least 7/13 product-level BA differences averaged over K/seeds are positive;
4. at least 4/5 seed-level BA differences averaged over products/K are positive;
5. `Dbar[MacroF1] >= 0`, at least two K-specific Macro-F1 differences are
   non-negative, and none is below `-0.010`.

Resource measurements are still reported but are not required to rescue performance.

### 6.2 GO — PRACTICAL_NONINFERIOR

This state is considered only if `PERFORMANCE_POSITIVE` is not met. All conditions
are required:

1. for every K, the one-sided 95% lower bound of `D[BA,K]` is greater than `-0.010`;
2. the one-sided 95% lower bound of `Dbar[BA]` is greater than `-0.010`;
3. the one-sided 95% lower bound of `Dbar[MacroF1]` is greater than `-0.010`, and no
   K-specific Macro-F1 point estimate is below `-0.010`;
4. the quantitative resource-advantage gate in §5.2 passes.

Performance within the margin without a qualifying resource advantage is not a GO.

### 6.3 NO-GO — V1_STOP

After a valid formal evaluation, V1 stops if neither GO state is satisfied. In
particular, stable MVREC-RGB dominance is established when:

- Ours has negative BA differences at at least two K values and negative Macro-F1
  differences at at least two K values;
- `Dbar[BA]` has a 95% CI upper bound below zero;
- at least 7/13 product-level and 4/5 seed-level omnibus BA differences are negative.

If this occurs and Protocol B also fails to confirm monotonic shot scaling, the
mandatory interpretation is:

> **V1 is stopped for the SCI-Q2 method route. Existing positive adaptation evidence
> remains valid, but the method is not sufficiently competitive or distinctive;
> proceed to V2 support-conditioned spatial evidence selection.**

K=8 scaling cannot rescue material K=1/3/5 inferiority. Conversely, failure to
confirm monotonic scaling does not force V1 STOP when either GO state above passes;
it only removes the scaling-mechanism claim.

## 7. V2 boundary after NO-GO

Only after `V1_STOP` may V2 implementation begin. The permitted research question is:

> Can class-consistent support evidence induce spatial/patch weighting for query
> classification without GT masks, boxes, or query localization annotations?

V2 must not be a larger terminal adapter, new loss chosen from V1 query errors, or a
restart of the rejected P5 multi-prototype route. It requires a separately
preregistered development/validation firewall and new mechanism-specific evidence.

## 8. Actions forbidden before the decision

- P7 implementation;
- MVTec-AD-RGB normal-inclusive execution;
- P6.A execution;
- expansion to another baseline;
- new adapter, loss, prompt search, threshold, or prototype count;
- product/seed/class removal;
- query-guided tuning or reinterpretation of the fixed margins.
