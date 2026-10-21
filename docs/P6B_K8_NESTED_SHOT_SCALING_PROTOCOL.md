# Protocol B — P6.B 9-Product K=8 Nested Shot-Scaling Protocol

> Status: **PREREGISTERED — LOCKED BEFORE ANY K=8 PERFORMANCE**
>
> Date locked: 2026-08-13
>
> Scope: post-P6.B, availability-only extension.

## 0. Pre-performance declaration

The original 13-product K=1/3/5 P6.B aggregate results are known. Before locking
this protocol:

- the K=8 decision used support availability only;
- no K=8 model prediction or metric has been generated or read;
- existing per-product K=1/3/5 results may already be present in the closed P6.B
  artifacts, but neither those results nor any derived nine-product aggregate was
  used to choose the subset or define the statistical rule;
- Cable 8-shot results are not used as comparable evidence or as a threshold source.

## 1. Scientific hypothesis

The observed original P6.B differences versus Frozen-Fusion motivate, but do not
confirm, a shot-dependent adaptation hypothesis:

> Support-side adaptation may exploit additional labelled examples more effectively
> than a frozen RGB-only baseline.

For baseline `b`, shot `K`, product `p`, and paired seed `s`, define

```text
d[b,p,s,K] = BA[Ours,p,s,K] - BA[b,p,s,K]
Delta[b,K] = mean_p mean_s d[b,p,s,K]
C[b]       = Delta[b,8] - Delta[b,1]
```

The **primary scaling contrast** is `C[MVREC-RGB adaptation]`. The predeclared
secondary contrast is `C[Frozen-Fusion]`. If MVREC-RGB receives
`STOP_NOT_REPRESENTATIVE`, its primary contrast is unavailable and must be reported
as such; Frozen-Fusion cannot be silently promoted to a replacement primary.

## 2. Availability-only product set

The candidate rule is exactly the existing Stage 1 rule: support-pool only, no query
pool contact, unique original RGB SHA256, single defect type, and
`multi_type_removed=false`. Every class of a retained product must have at least
eight candidates.

The locked nine products are:

- objects: `bottle`, `capsule`, `hazelnut`, `metal_nut`, `screw`, `zipper`;
- textures: `carpet`, `leather`, `tile`.

The excluded products are `grid`, `pill`, `transistor`, and `wood`. They must not be
restored, pooled, oversampled, synthesized, or supplemented because of performance.
`cable` and `toothbrush` retain their original P6.B exclusion.

Availability minima are respectively: bottle 10, capsule 10, carpet 9, hazelnut 9,
leather 9, metal_nut 9, screw 11, tile 8, and zipper 8 candidates per class.

Evidence source:
`data/p6b_probe_output_v2/p6b_mvtec_fs_rgb_image_records.jsonl` plus the frozen
`support_candidates_by_class` rule in
`experiments/p6/p6b_execution_common.py`.

## 3. Nested support identity

For every `(product, class, seed)`, retain the original P6.B deterministic ordering:

```text
sub_seed = first 8 bytes, big-endian, of
           SHA256("p6b_draw_v1|{seed}|{product}|{class}")
order     = random.Random(sub_seed).shuffle(SHA256-sorted candidates)
support_K = first K records of order
```

This must satisfy, by RGB SHA256 identity:

```text
K1 subset K3 subset K5 subset K8
```

The existing K=1/3/5 manifests and their identities are immutable. K=8 extends the
same ordered list; it does not redraw earlier shots. Every manifest must retain zero
support/query SHA overlap.

## 4. Methods and execution scope

Core methods only:

1. Frozen-Fusion ProtoNet-style;
2. MVREC-RGB adaptation, conditional on Protocol A `SEMANTIC_PASS`;
3. Ours.

R1 may be reported only as an explicitly secondary diagnostic if its inclusion is
implemented without changing any primary run or statistic. No other baseline,
adapter, loss, prompt, augmentation, or prototype extension may be added.

For Frozen-Fusion and Ours, the already closed K=1/3/5 predictions are reused and
filtered to the locked nine products. They are not retrained or modified. MVREC-RGB
uses K=1/3/5/8 under Protocol A. New K=8 support training uses the frozen method
semantics and final-checkpoint rule.

All shots use the same query identities, fixed class universe, paired seeds
`111/222/333/444/555`, image-level evaluation unit, and P6.B query firewall.

## 5. Metrics and aggregation

Primary metric: **product-macro Balanced Accuracy**.

Secondary metric: **product-macro Macro-F1**.
Top-1 Accuracy remains descriptive.

For each `(method, product, K, seed)`, first calculate the metric within that
product. Aggregate products with equal weight and then seeds with equal weight.
Sample-weighted pooling across products is forbidden.

The K=1/3/5 values used for a scaling trajectory must be recomputed as aggregates of
the same nine locked products; the published 13-product macro values must not be
mixed into the trajectory.

## 6. Locked paired inference

For each baseline `b`, calculate:

- `Delta[b,1]`, `Delta[b,3]`, `Delta[b,5]`, `Delta[b,8]`;
- product-level paired differences `mean_s d[b,p,s,K]`;
- seed-level paired differences `mean_p d[b,p,s,K]`;
- product-level scaling contrasts
  `c[b,p] = mean_s(d[b,p,s,8] - d[b,p,s,1])`;
- seed-level scaling contrasts
  `c[b,s] = mean_p(d[b,p,s,8] - d[b,p,s,1])`.

For `C[b]`, form a paired product-cluster percentile bootstrap:

- sampling unit: the nine products;
- all five paired seeds for a sampled product remain together;
- resamples: 10,000;
- RNG seed: `20260813`;
- interval: two-sided 95% percentile CI.

The identical bootstrap indices must be used for all compared methods and metrics.
No alternative test may replace this analysis after results are seen.

## 7. Scaling decisions

**CONFIRMED_MONOTONIC_SHOT_SCALING** against a baseline requires all of:

1. `C[b] > 0` and its 95% CI lower bound is greater than zero;
2. point estimates satisfy
   `Delta[b,1] <= Delta[b,3] <= Delta[b,5] <= Delta[b,8]`;
3. at least 6/9 product-level `c[b,p]` values are positive;
4. at least 4/5 seed-level `c[b,s]` values are positive.

The primary mechanism claim requires these gates for MVREC-RGB. A Fusion-only pass
is supporting evidence but cannot establish superiority to the strongest strict
RGB comparator.

If `Delta[b,8] <= Delta[b,5]`, or the ordered trajectory breaks at K=8, the required
decision text is:

> **Positive adaptation benefit exists, but no monotonic shot-scaling advantage is
> confirmed.**

If `C[b] > 0` but its CI or paired-direction gates fail, report
`DIRECTIONALLY_POSITIVE_BUT_UNCONFIRMED`. If `C[b] <= 0`, report
`NO_POSITIVE_K8_VS_K1_SCALING`.

Macro-F1 undergoes the same locked calculation as secondary evidence but cannot
override a failed Balanced Accuracy decision.

## 8. Interpretation boundary

A confirmed result supports a mechanism-level inference that the locked adaptation
benefit grows with additional same-class support under this nine-product protocol.
It is not a new algorithm, does not prove universal monotonicity, and cannot be
combined numerically with Cable 8-shot results.

A negative or inconclusive scaling result does not erase the completed P6.B positive
adaptation benefit. It only forbids the monotonic shot-scaling claim.

## 9. Fail-closed rules

- The canonical-LF SHA256 values of Protocol A, this protocol, and the V1 decision
  rule, plus the clean execution commit, must be pinned and verified at run start and
  completion.
- Missing product/class/seed/shot, identity non-nesting, hash mismatch, query change,
  overlap, non-finite metric, or provenance mismatch makes the scaling audit
  `INCONCLUSIVE_ENGINEERING`.
- Integrity failures may be repaired without changing scientific definitions.
- Products, metrics, baseline priority, bootstrap seed, margin, or aggregation cannot
  be changed after any K=8 performance is visible.
- No product exclusions based on difficulty or variance.
- No new adapter, loss, multi-prototype, query tuning, P7, P6.A, or MVTec-AD-RGB run
  before the V1 decision.
