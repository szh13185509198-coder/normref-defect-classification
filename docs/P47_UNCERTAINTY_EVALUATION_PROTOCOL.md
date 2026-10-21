# P47 — Uncertainty evaluation (report only)

Status: **DRAFT — NOT LOCKED.** The author locks it by committing the final version before any P47 output. P47 is
sequentially added after all earlier rounds and is disclosed in S2. No method, constant or earlier result changes: ARR
(`fused_adareg_regres_soft_q99`), CONS, SimpleShot and the P15 screening rule stay frozen. U1-PAC (a threshold rule)
and U5 (type sets) are new decision layers on top of the frozen scores, reported like P15. Real-IAD is not used; the
blind set stays closed.

## 0. Why, and how the system is framed

Measurement Science and Technology asks that every article consider "the uncertainty, precision and/or accuracy of
the measurements presented". The inspection system produces two results per part:
- **A conformity decision** (conforming / non-conforming) from a quantitative indication, the screening score s(x).
  The decision risks are the global producer's risk R_P and the global consumer's risk R_C of JCGM 106:2012
  (§3.3.15–3.3.16). Both are **joint** probabilities (§9.4.1, §9.5.1): R_P = P(conforming ∧ rejected) = π·FRR and
  R_C = P(non-conforming ∧ accepted) = (1 − π)·escape, where π is the share of good parts.
- **The examination of a nominal property** (the defect type). Uncertainty is expressed through misclassification
  probabilities and set- or PMF-valued results: Eurachem/CITAC AQA 2021; Bilson et al., Metrologia 62 (2025) 064001.

The reported performance figures (BA, FRR, escape, latency, energy) get GUM-style treatment (JCGM 100:2008):
- Type A components from designed replication;
- Type B components from existing device recomputations;
- combined standard uncertainty, effective degrees of freedom (Welch–Satterthwaite) and expanded uncertainty;
- bootstrap or Monte Carlo propagation where the model is not linear (JCGM 101:2008).

General rules (as P15–P17):
- report only; every cell is reported and none is selected;
- each stage runs once, with a lock directory and a staging directory;
- every stage first reproduces the published values it builds on, bitwise (P14 predictions, P15 unit values);
- all random draws use the string seeds given below.

## U1. Calibration uncertainty of the false-reject rate (producer's risk)

**Question.** P15 used a single calibration draw (N = 100) per task, and its guarantee is marginal (it holds in
expectation over calibration draws). How much does the realized FRR of one deployment vary, and how many calibration
normals does a stated precision need?

**Theory (stated before any data).** For continuous scores and exchangeable normals, the realized FRR given the
calibration set is FRR ~ Beta(N − k + 1, k), with k = ⌈(N + 1)(1 − α)⌉ (Vovk, Mach. Learn. 92 (2013) 349–376;
Angelopoulos & Bates). Measured on m test normals, the realized FRR follows the matching beta-binomial distribution.

PAC (tolerance-limit) rule: choose the smallest k with P(Beta(N − k + 1, k) > α) ≤ δ. With k = N, this is Wilks's
one-sided distribution-free tolerance limit (Ann. Math. Statist. 12 (1941) 91–96), which needs N ≥ ln δ / ln(1 − α).
For α = δ = 5 %, that gives N = 59.

Pre-computed predictions (scipy, this protocol):

| N | α | k | mean FRR | SD | 5 %–95 % | P(FRR > α) | PAC k (δ = 0.05) / mean FRR |
|---|---|---|---|---|---|---|---|
| 50 | 5 % | 49 | 3.92 % | 2.69 % | 0.72–9.14 % | 0.28 | infeasible |
| 100 | 1 % | 100 | 0.99 % | 0.98 % | 0.05–2.95 % | 0.37 | infeasible (needs N ≥ 299) |
| 100 | 5 % | 96 | 4.95 % | 2.15 % | 1.99–8.92 % | 0.44 | 99 / 1.98 % |
| 200 | 5 % | 191 | 4.98 % | 1.53 % | 2.74–7.72 % | 0.46 | 196 / 2.49 % |
| 500 | 5 % | 476 | 4.99 % | 0.97 % | 3.50–6.68 % | 0.47 | 484 / 3.39 % |
| 1000 | 1 % | 991 | 1.00 % | 0.31 % | 0.54–1.57 % | 0.46 | 996 / 0.50 % |

**Data.** VisA (11 tasks) and MTD.
- Banks: the P15 bank draws (5 seeds, M = 20).
- Calibration pool: the normal support candidates that are in none of that task's bank draws and none of its K = 5
  normal supports. The P15 calibration draw lies inside this pool.
- Test set: all normal and defect queries, unchanged from P15.
- Cable and MVTec dev: reported only if the pool holds at least 2N images; pool sizes are reported either way.

**Procedure.** For each task × bank seed, compute s(x) (the P15 top-1 % statistic, CLIP layer 18) for all pool and
query images. Then for:
- N ∈ {20, 50, 100, 200, 300}, where the pool holds at least N;
- R = 1000 calibration draws, `random.Random(f"p47cal|{task}|{seed}|{N}|{r}")`;
- α ∈ {1, 2, 5, 10} %;

compute the realized FRR (normal queries) and the escape rate (defect queries) under two rules:
- (a) the marginal rule (P15);
- (b) the PAC rule with δ ∈ {0.05, 0.10}; infeasible cells are reported as infeasible.

**Integrity.** The P15 calibration draw reproduces the P15 unit FRR and AUROC exactly.

**Reported.**
- Per task and pooled: mean, SD and 5/50/95 % quantiles of the realized FRR.
- Empirical against theoretical P(FRR > α); Q–Q plots and KS tests against the beta-binomial prediction.
- Nested random-effects decomposition of the FRR variance into bank and calibration components.
- The escape cost of the PAC rule relative to the marginal rule.
- A design table: N needed for a stated 90 % interval width and for a PAC guarantee.

**Use.** The §7.7 sentence "the realized FRR never exceeded α" is replaced by the distribution statement and the
design rule. A clear excess spread over the beta-binomial (pooled KS p < 0.01) is reported as evidence of
non-exchangeability.

## U2. Decision risks in JCGM 106 terms (re-analysis of P15 and U1)

- **Risks.** R_P = π·FRR and R_C = (1 − π)·escape, for π ∈ {0.90, 0.95, 0.99, 0.999}. These are the existing
  "per 1000 parts" numbers divided by 1000.
- **Three-way rule.** The P15 three-way rule is read as guard banding:
  - acceptance zone: s ≤ τ_pass;
  - rejection zone: s > τ_reject;
  - review zone in between, resolved by a second, more accurate measurement (human review).
- **Intervals (95 %).** Each risk and the review rate get a 95 % interval combining:
  - the calibration component, from the U1 draws;
  - the test-sample component: object-cluster bootstrap for VisA, Clopper–Pearson for MTD;
  - the two components are combined by Monte Carlo (JCGM 101).

## U3. Uncertainty budget of the reported BA

**Measurands.**
- (a) The benchmark mean BA reported in Table 6: product macro over 13 products, mean over draws.
- (b) The BA of a single deployment: one product, one support set, one normal bank. This is a dispersion, reported as
  a prediction interval.

**Factors.**
- Support draw S: the 5 formal manifests.
- Bank draw B.
- Query sampling Q: bootstrap within product, stratified by class, 1000 replicates.
- Bank size M.
- Implementation I: Type B.

**Design (MVTec-FS formal, cached features plus MVTec AD `train/good` features).**
- Full cross S × B (5 × 5) at M = 20 and K ∈ {1, 3, 5}, for ARR. SimpleShot has no bank, so it gets S and Q only.
  The diagonal cells are the published ones (integrity check).
- Bank-size sweep, M ∈ {5, 10, 20, 50}: 5 supports × 10 bank draws, `random.Random(f"p47bank|{product}|{M}|{j}")`,
  from `train/good`.

**Analysis.**
- Two-way random-effects ANOVA per product × K (method of moments, negative variance estimates set to 0) gives σ_S²,
  σ_B² and σ_SB²; σ_Q² comes from the bootstrap.
- These are combined by the law of propagation for the macro mean, with Welch–Satterthwaite ν_eff and coverage
  factor k = t_{0.975, ν_eff}.
- Also reported: the per-deployment SD per product and pooled, and the curve σ_B(M).
- Type B components:
  - implementation: device recomputations P32/P38/P41, |Δ| ≤ 0.15 pp mean with CI half-width ≤ 0.35 pp; rectangular
    distribution with half-width equal to the largest |CI bound|;
  - reference labels: from U7, if it is done.
- Sensitivity coefficients from the P17 grid (∂BA/∂q, ∂BA/∂fallback size) are reported but not added to the budget,
  because the constants are fixed.

**Output.** An uncertainty-budget table for K = 5 (component, u, ν, share of u_c²), and a comparison with the existing
cluster-bootstrap CIs (which also contain between-product heterogeneity).

## U4. Influence quantities: acquisition variation, repeatability, drift and recalibration

**Perturbations.** Applied to the decoded RGB image at native resolution, before the fixed 336 px evaluation view.
All are deterministic, with per-image seeds `random.Random(f"p47pert|{sha}|{cond}")`.

| Code | Influence quantity | Levels |
|---|---|---|
| P1 | illumination gain | ×0.8, ×0.9, ×1.1, ×1.2 (clipped) |
| P2 | gamma | 0.8, 1.25 |
| P3 | sensor noise (Gaussian, DN) | σ = 2, 5, 10 |
| P4 | defocus (Gaussian blur, px) | σ = 0.5, 1, 2 |
| P5 | misalignment (translation, % of width, reflect padding) | 1, 2.5, 5 |
| P6 | rotation (reflect padding) | 2°, 5° |
| P7 | compression (JPEG quality) | 90, 70, 50 |
| J | re-acquisition jitter, R = 5 replicates | gain U[0.97, 1.03], noise σ U[0, 2], shift U[−0.5, 0.5] %, rotation U[−0.5°, 0.5°] |

**Scenarios.**
- **D (drift, no recalibration).** Queries are perturbed; supports, bank and calibration stay unperturbed.
- **RC (recalibrated).** Queries, bank and calibration normals are perturbed alike; supports stay unperturbed. This is
  the realistic case: the lighting changed, the normal reference was recalibrated in seconds, and the defect library
  was kept.

**Data.**
- MVTec-FS formal: classification, K = 5, 5 seeds.
- VisA, 11 tasks: classification and screening, at α = 5 %, under the P15 calibration draws. If needed, the normal
  queries are limited to 250 per task; the selection is fixed before the run.
- MTD: classification and screening.

**Metrics.**
- BA and its paired Δ against the unperturbed run, with cluster bootstrap.
- Decision flip rate.
- Screening: realized FRR at α = 5 % and AUROC.
- For J (attribute agreement analysis, AIAG MSA-4):
  - within-system repeatability: the share of parts whose R replicates all agree;
  - Fleiss' κ over the replicates;
  - agreement with the reference labels;
  - all three for both the type decision and the pass / reject decision.

**Derived quantities.**
- Sensitivity coefficients per unit of each influence quantity.
- Drift tolerance: the largest level with FRR ≤ 2α without recalibration.
- The FRR after recalibration.

## U3 / U4 implementation details (fixed before any U3 / U4 output, 2026-10-03)

Code: `experiments/p47/u3_budget.py`, `experiments/p47/u4_influence.py`, `experiments/p47/common.py`. Each stage is
write-once (`<out>.lock` left in place); every stage refuses to start with uncommitted changes in the guarded code
paths. Choices the sections above leave open:

**U3.**
- Cross S × B: support seed i (formal manifest `k{K}_s{i}`) × bank seed j (formal draw j, first 20 names). Cell
  (i, i) is the published P14 cell; ARR and SimpleShot must reproduce the P14 BA rows to < 1e-9, else nothing is
  written.
- One deterministic value per cell, so σ_SB² = MS_SB (interaction and residual are not separable).
  Satterthwaite degrees of freedom for σ_S² and σ_B²; ν_SB = (a − 1)(b − 1).
- Query bootstrap: 1000 replicates, stratified by true class, the same resamples for all cells of a product × K
  (`p47q|{product}|{K}`). σ_Q² for the benchmark mean is the bootstrap variance of the mean of the 5 diagonal cells;
  for a single deployment it is the cell-wise bootstrap variance averaged over the 25 cells; ν_Q = n_queries − 1.
- Benchmark mean (13-product macro, 5 diagonal draws): u² = Σ_p (σ_S² + σ_B² + σ_SB²) / (5 · 13²) + Σ_p σ_Q,bench² / 13²
  (+ the Type B implementation term). Welch–Satterthwaite first over products per component, then over components.
- Bank-size sweep: features of all train/good images in the union of the sweep draws are extracted once
  (`ckpt/p47/u3_features/`, ≈ 4 GB); all K are scored (K = 5 is the reported one).
- The Type B implementation half-width is passed to the analysis stage by the operator, with the files it was read
  from; without it the budget is reported without that row and says so.

**U4.**
- Per-image key `sha`: the manifest `rgb_sha256` (MVTec-FS, VisA, MTD); for MVTec AD train/good bank images, the
  SHA-256 of the file.
- Operations on the float RGB array at native resolution, then rounded and clipped to 0–255: P1 multiply; P2
  255 (x/255)^γ; P3 additive Gaussian; P4 Gaussian filter (truncate 4σ, reflect); P5 shift by the level × width in a
  direction drawn uniformly from [0, 2π) (bilinear, reflect); P6 rotation by ± the level, sign drawn (bilinear,
  reflect, same size); P7 JPEG encode/decode with 4:2:0 subsampling. J: rotation U[−0.5°, 0.5°], then shift
  U[−0.5, 0.5] % of the width on each axis, then gain U[0.97, 1.03], then noise σ U[0, 2]; R = 5 replicates J1–J5.
- J is applied to the queries only (scenario D): it measures the repeatability of the measurement of one part.
  RC covers "none" and P1–P7; under RC the bank and calibration normals are re-extracted through the U4 path for
  "none" too, so that every delta compares like with like.
- Integrity: the unperturbed reference computed as published must reproduce P14 (MVTec-FS BA rows; VisA/MTD ARR and
  SimpleShot predictions) and P15 (AUROC and FRR at α = 5 %) at K = 5, else the stage stops. The U4-path "none"
  (process-pool decoding, query-only batches) is compared with that reference and reported; deltas use U4 "none".
- Normal queries above 250 per task: `random.Random(f"p47u4cap|{task}").sample(sorted normal sha256, 250)`; all defect
  queries kept. This affects 8 VisA tasks (candle, chewinggum, macaroni1, macaroni2, pcb1–pcb4) and MTD.
- Classification K = 5 only, 5 seeds. Screening at α = 5 % with the P15 statistic (`ckpt/p15/dev`).
- Sensitivity coefficient: least-squares slope through the reference (x = 0, Δ = 0) over the levels of a factor, in
  the units of `common.UNITS` (P1 per % gain, P2 per unit ln γ, P7 per quality step 100 − q).
- Drift tolerance, per factor and direction: the largest level such that it and every smaller level keep the
  realized FRR (scenario D, VisA object macro or MTD, seed mean) ≤ 2α.
- Attribute agreement (AIAG MSA-4) per task/product and seed, then cluster macro and seed mean: within-system
  repeatability (all R trials agree), Fleiss' κ, agreement with the reference labels (all R agree and equal the
  label; for pass/reject the label is "reject" for every defect), and agreement with the unperturbed decision.

## U4-A. Geometric influence quantities without padding (amendment, 2026-10-04, fixed before any U4-A output)

**Why.** U4 moved the part by translating or rotating the whole image with reflect padding (P5, P6, and the
geometric part of J), so mirrored image content entered at the border. A diagnostic on MVTec AD good parts
(`experiments/p47/diag_geometry_padding.py`; Hugging Face CLIP weights, not the frozen caches; 20 test parts per
product; results in `ckpt/p47/diag_geometry_padding/`) compared this with a padding-free move of the field of view.
With padding, a 5 % shift raised the false-reject rate from 5 % to 90 % (grid), from 0 % to 25 % (zipper) and from
0 % to 60 % (bottle); without padding it stayed at 0–5 %, and the screening score barely moved. U4's geometric
results therefore mostly measure the simulated border, not the displacement of the part.

**Design.** As U4 (data, K = 5, seeds, metrics, normal-query cap, cluster bootstrap, attribute agreement), except:
- every image (supports, bank, calibration normals, queries) is evaluated through a window of FR × the width and
  FR × the height of the decoded image, then the locked 336 px view. FR = min(0.88, the largest window that a 5°
  rotation about the centre keeps inside every image of the product or task, less 3 px); it is reported per product
  and task;
- P5 (1, 2.5, 5 %): the window moves by v % of its width and v % of its height along the U4 direction θ (same
  per-image generator `random.Random(f"p47pert|{sha}|{cond}")` and draw order);
- P6 (2°, 5°): the image is rotated about its centre by ± v (U4 sign) and the centred window is taken; no padded
  pixel reaches the window (checked by the self-test);
- J1–J5: as U4 J (gain, noise, rotation), with the window moved by U[−0.5, 0.5] % of its width and height;
- scenario D for all conditions; RC (bank and calibration normals in the same condition) for none, P5 = 5 % and
  P6 = 5°;
- the reference is the windowed view. It is not compared with the published full-view numbers for integrity; the
  analysis reports it next to U4's reference.

**Reporting.** U4-A replaces U4's P5, P6 and J results in the paper. U4's photometric results (P1–P4, P7) are not
affected and stand. U4's geometric results go to S16, labelled "whole image moved, reflect padding".

Code: `experiments/p47/u4a_geometry.py`. Outputs: `ckpt/p47/u4a_{mvtec,external,analysis}/`.

## U5. Uncertainty of the nominal output: conformal type sets and the unknown-type flag

**Score.** The ARR fused score f_c(x) (P14); the nonconformity is r(x, c) = −f_c(x).

**Calibration.** Leave-one-out (jackknife) over the supports. For each support i:
- rebuild every support-dependent part of the state without i: the centering mean and the prototypes for g, h_A
  and h_R; the bank and τ do not change;
- compute r_i = r(x_i, y_i).

The threshold q̂ is the k-th smallest of the CK values, k = ⌈(CK + 1)(1 − α)⌉; if k > CK, the cell is infeasible and
the set is "all classes". The type set is Γ(x) = {c : r(x, c) ≤ q̂}.

**Grid.** K ∈ {3, 5} (K = 1 has no leave-one-out) and α ∈ {0.10, 0.20}:
- MVTec-FS formal, 5 seeds;
- VisA defect classes, for the screened-defect queries only (descriptive).

**Reported.**
- Marginal and class-conditional coverage.
- Mean set size, singleton rate and singleton accuracy, and empty-set rate.
- The jackknife caveat: the worst-case guarantee is 1 − 2α (Barber et al., Ann. Statist. 49(1) (2021), doi:10.1214/20-AOS1965).

**Unknown types.** Leave one class out, on products with C ≥ 3 so that at least 2 known classes remain:
- the supports of class c are removed, and the queries of c become "unknown";
- an empty set Γ(x) raises the "unknown type" flag.

Reported: the flag rate on unknown queries (detection), the flag rate on known queries (false alarm, expected to be
≤ α), and the AUROC of min_c r(x, c).

This turns the nominal result into a per-part uncertainty statement: "type ∈ Γ(x) with coverage ≥ 1 − α". It also
closes the loop with the zero-training addition of a new type.

## U6. Timing and energy uncertainty (re-analysis of existing logs; optional new 4090 sessions)

**Measurands.** The end-to-end latency of the fused engine (Jetson), the forward latency (RTX 4090, Table 11) and the
energy per part.

**Components.**
- Repeatability: the SD of repeated runs divided by √N (Type A).
- Between-session reproducibility: forward against reverse passes; r3 against the P36 sessions.
- Drift: the trend over the 1 h run.
- Timer resolution (Type B).
- Power:
  - sampling over the 100 ms windows (Type A);
  - sensor accuracy (Type B; the datasheet value must be verified before use);
  - subtraction of the idle baseline.

Combined standard uncertainty u and expanded uncertainty U (k = 2). The device measurements are designed in U6-D.
Optional: three additional 4090 sessions on different days, with process restarts.

## U6-D. Device part (Jetson Orin Nano; must run before the rental ends)

The only P47 measurements that need the device. Everything else (U1–U5, U7) runs offline or on the RTX 4090. The
device-side U1 variant also runs offline (U1-D below). Code: `experiments/p47/run_jetson_u6.sh`, which calls the
existing P43 modes and `experiments/p47/jetson_u6.py`. One run, write-once outputs; the script stops at the first error.

**Preconditions (author's execution agent).**
- MAXN_SUPER with `jetson_clocks` applied, as in P43 and P45.
- `drm_holder.py` running; the script stops if it is not.
- Headless, with no other load.

**Task.** The P43 task: bottle, K = 5, seed 111, M = 20, deployed through the P41 fused engine with GPU preprocessing.

| Block | What | Design | Output |
|---|---|---|---|
| D1 Session reproducibility | Deployed end-to-end latency across fresh processes | 10 processes, each running P43 `--repeat` phase A: 50 warm-up calls, then 500 end-to-end and 500 split timings | `ckpt/p43/p47_jetson/p43_repeat_s01…s10.json` |
| D2 Energy repeatability | Energy per part over repeated windows | 5 windows, each 30 s idle + 60 s deployed end to end (P43 `--energy`); `tegrastats` VDD_IN every 100 ms with host timestamps | `ckpt/p43/p47_jetson_e1…e5/`, `ckpt/p47/p47_jetson/tegrastats.log` |
| D3 Scaling | (a) end-to-end latency against the number of classes C ∈ {2, 3, 5, 8, 16} (the engine's dynamic dimension; random prototypes, decisions not used); (b) the head alone (`fused_engine.head`, PyTorch FP32 / FP16) against M ∈ {5, 10, 20, 50, 100} (the engine's bank shape is fixed at M = 20; timing does not depend on the values, so the real bank rows are repeated); (c) per-part latency with 1, 4 and 9 tiles, each tile through the deployed path | 200 repetitions per cell (100 for tiles) | `ckpt/p47/p47_jetson/p47_u6_scaling.json` |

Duration: about 30–35 min.

**Local analysis.**
- D1: a one-way random-effects model with session as the factor gives the repeatability SD (within session) and the
  between-session SD. Combined with the P43 B1, P43-A2 and P45 sessions as descriptive anchors, this gives a GUM
  budget for the 67 ms figure.
- D2: P43 `--report-energy` per window gives the energy per part; the mean, SD and Type A u come from the 5 windows,
  with the dynamic energy computed against each window's idle period. The power-sensor accuracy enters as Type B,
  once its datasheet value is verified.
- D3: reported as measured. It is used for the cost side of the bank-size trade-off in U3 and for the cost of finer
  sampling (tiles).

**U1-D (offline, from P46 outputs).** Per task and seed, pool the P46 device scores of the P15 calibration normals
and the normal queries. Then resample calibration sets of size N ∈ {20, 50, 100} (R = 1000,
`random.Random(f"p47calD|{task}|{seed}|{N}|{r}")`) and test on the remaining normals. This gives the realized FRR
distribution on the deployed engine's own scores, compared with the U1 PyTorch result and the Beta prediction. It needs
no device time.

## U7. Reference-label check (optional)

Hash-match the MVTec-FS query images to MVTec AD test images, and compare the MVTec-FS type label with the original
MVTec AD folder label. The disagreement rate is reported as information on the uncertainty of the reference values.
The evaluation does not change.

## Run order and cost

0. **Jetson, before the rental ends:** U6-D (`experiments/p47/run_jetson_u6.sh`), about 30–35 min. U1-D then runs
   offline from the P46 outputs.

On the RTX 4090:

1. **No new features:** U2 (P15 outputs) and U6 (logs). Minutes, on CPU.
2. **Normal-pool features** (VisA ≈ 8k, MTD ≈ 1k, MVTec AD `train/good` ≈ 3.5k images): U1 and U3. About 15–30 min
   of GPU, then CPU analysis.
3. **U5** on cached features. Minutes.
4. **U4** perturbed features. About 1–2 h of GPU.

Outputs: `ckpt/p47/{u1,u2,u3,u4,u5,u6,u7}/`. Code: `experiments/p47/`.

## References to verify at citation time
- JCGM 100:2008; JCGM 101:2008; JCGM 106:2012 (BIPM).
- Eurachem/CITAC Guide AQA, 1st edition, 2021, R. Bettencourt da Silva and S. L. R. Ellison (eds.),
  ISBN 978-0-948926-39-6.
- S. Bilson, M. Cox, A. Pustogvar, A. Thompson, Metrologia 62 (2025) 064001, doi:10.1088/1681-7575/ae1bae.
- S. S. Wilks, Ann. Math. Statist. 12 (1941) 91–96, doi:10.1214/aoms/1177731788.
- V. Vovk, Machine Learning 92 (2013) 349–376, doi:10.1007/s10994-013-5355-6.
- R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani, Ann. Statist. 49 (2021), doi:10.1214/20-AOS1965.
- G. Nordin et al., Clin. Chem. Lab. Med. 48 (2010) 1553–1566, doi:10.1515/CCLM.2010.325 (nominal properties).
- AIAG, Measurement Systems Analysis Reference Manual, 4th edition (2010).

## U2 / U5 implementation details (fixed before any U2 / U5 output, 2026-10-04, local session)

Code: `experiments/p47/u2u5.py`, three write-once stages.

**Stage `extract` (GPU; VisA 11 tasks + MTD).** Recomputes P15 `external_task` exactly: the same banks (5 seeds, M = 20),
the same P15 calibration draw (N = 100), the P15 top-1 % statistic, all queries (no cap), the P14 extraction call.
Integrity, before anything is written: the ARR / A / SimpleShot argmax predictions reproduce the published P14
predictions for K ∈ {1, 3, 5} and every seed, and the recomputed P15 unit metrics (AUROC, thresholds, two-level FRR
and escape of ARR at α ∈ {1, 2, 5} %, three-way metrics at α_pass ∈ {10, 30} %) equal the published P15 units to
1e-12. Stored per task: query and calibration scores per seed, and the per-seed descriptors (g, h_A, h_R) of all
support and query rows (needed by U5). Output `ckpt/p47/u25_external/`.

**Stage `u2` (CPU).** Units = task × bank seed (the screening score does not depend on K). Rules as P15:
two-level at α ∈ {1, 2, 5} % (reject if s > τ_α); three-way with τ_reject = τ_1 % and τ_pass ∈ {τ_10 %, τ_30 %}
(accept s ≤ τ_pass, review τ_pass < s ≤ τ_reject, reject s > τ_reject). R_P = π·(automatic false-reject rate),
R_C = (1 − π)·(escape rate), review rate = π·(good share in review) + (1 − π)·(defect share in review), for
π ∈ {0.90, 0.95, 0.99, 0.999}. Point values: VisA object macro (mean over tasks per seed, then over seeds), MTD mean
over seeds; they must equal the P15 aggregates.
95 % intervals by Monte Carlo (JCGM 101), R = 10 000, numpy generator seeded from "p47u2":
- calibration component (deviation from the text above: U1 itself was not run, so the U1 draws do not exist): in every
  replicate each unit's calibration set is a with-replacement resample of its 100 P15 calibration scores; the spread
  of the realized FRR is compared with Beta(N − k + 1, k) and reported;
- test-sample component: VisA, object-cluster bootstrap (11 tasks resampled with replacement per replicate, the same
  multiset for all seeds); MTD, stratified bootstrap of its normal and defect queries (deviation: Clopper–Pearson
  intervals of the seed-mean proportions are reported next to it as a reference, not combined);
- combined: both in the same replicate. Component-only intervals are reported too.
Output `ckpt/p47/u2/`.

**Stage `u5` (GPU for MVTec-FS from the cached P10 / P14 features; VisA from stage `extract`).** ARR scores exactly as
P14 (`episode_scores`); with all supports they must reproduce the published P14 ARR balanced accuracy (MVTec-FS,
to 1e-9) and predictions (VisA). Leave-one-out per support i: centering means and class prototypes of g, h_A, h_R
rebuilt from the other supports (bank, τ and descriptors unchanged), r_i = −f_{y_i}(x_i). Type sets Γ(x) as above.
- MVTec-FS: 13 products × K ∈ {3, 5} × 5 seeds (bank draw = episode seed, as P14) × α ∈ {0.10, 0.20}; all queries.
  Reported: marginal coverage, class-conditional coverage (mean and minimum over classes), mean set size, singleton
  rate and accuracy, empty-set rate; product macro over seeds and pooled.
- Unknown types: MVTec-FS products with C ≥ 3, each class c in turn: supports of c removed, scores and LOO
  recomputed over the remaining classes, queries of c are unknown, all other queries known. Reported: flag (empty set)
  rate on unknown and on known queries, and the AUROC of min_c r(x, c) (unknown = positive).
- VisA (descriptive): defect classes only. The episode keeps all classes (as P15's restricted type); LOO over the
  defect supports; queries = defect queries screened at α = 5 % (s > τ_5 %); Γ over defect classes.
Output `ckpt/p47/u5/`.

## U2-A. Calibration component from the order-statistic law (amendment, 2026-10-04, before any U2-A output)

**Why.** The U2 calibration component resampled each unit's 100 calibration scores with replacement. At α = 1 % the
threshold is the largest calibration score, and a with-replacement resample can never exceed the original maximum, so
the bootstrap only moves that threshold down: the intervals are one-sided (VisA R_P at α = 1 %, π = 0.95:
7.0 [5.6, 22.5] per 1000) and the bootstrap SD of the FRR disagrees with Beta(N − k + 1, k) (MTD at α = 1 %: 0.37 vs
0.98 pp). This is the known failure of the nonparametric bootstrap for extreme order statistics. U2 stays as run and
is reported as such; U2-A replaces its calibration component.

**Method.** Per unit (task × bank seed) and replicate: draw N = 100 independent uniforms, sort them (U_(1) ≤ … ≤ U_(N)),
and set each threshold τ_α = Q̂(U_(k_α)), k_α = ⌈(N + 1)(1 − α)⌉, with Q̂ the empirical quantile function (numpy
"linear") of the unit's pooled normal scores (its 100 calibration scores and its normal queries). Using one sorted
uniform sample per replicate keeps all thresholds of a replicate jointly consistent; marginally U_(k) ~
Beta(k, N − k + 1), so the realized FRR follows the U1 law. Rules, π, point values (the actual P15 calibration draw),
test-sample component, R = 10 000 and the reported modes (combined, calibration only, test only) are as U2; the
generator is seeded from "p47u2a". Reported next to U2: the calibration-only SD of the FRR against the Beta SD.
Code: `experiments/p47/u2a.py`. Output `ckpt/p47/u2a/` (write-once).

## U2-B / U5-A. Shared calibration draw; jackknife+ type sets (amendment, 2026-10-04, before any U2-B / U5-A output)

Code: `experiments/p47/u2b_u5a.py`, two write-once stages. Inputs: the U2 / U5 extraction (`ckpt/p47/u25_external/`),
the cached MVTec-FS features (as U5), and the U2, U2-A and U5 outputs.

**U2-B. Why.** P15 draws the calibration set once per task (`random.Random("p15cal|{task}")`) and scores the same 100
images under each of the 5 bank seeds. U2-A drew an independent sorted uniform sample per (task, seed), i.e. treated
the 5 seeds as 5 independent calibration draws. This shrinks the calibration component of a task's seed-mean FRR by
up to √5 and makes a single unlucky draw look like a systematic deviation. U2 and U2-A also resampled MTD's queries
independently per seed, although the queries are the same images.

**U2-B. Method.** As U2-A, except:
- per task and replicate, one sorted uniform sample U_(1) ≤ … ≤ U_(N) serves all seeds; each seed's thresholds are
  τ_α = Q̂_seed(U_(k_α)) with Q̂_seed the empirical quantile function of that seed's pooled normal scores (comonotone
  across seeds, justified by the rank agreement reported below);
- MTD test component: one stratified query resample per replicate, shared by the seeds;
- generator seeded from "p47u2b"; point values must equal U2's to 1e-12.
Reported in addition, per task: the Spearman correlation of the calibration scores between seeds (mean and minimum
over pairs); the calibration-only SD of the seed-mean FRR, shared and independent, against the Beta SD; the
calibration-only percentile (mid-rank) of the task's P15 point, shared and independent; for VisA, a KS test of the
11 percentiles at α = 5 % against uniform. The percentiles replace the per-seed KS test as the check of
exchangeability between calibration and query normals (the partition is by SHA-256 order, so exchangeability holds
by construction unless the percentiles say otherwise).

**U5-A. Why.** U5 is the plain jackknife: test scores from the model fitted on all supports, threshold from the
leave-one-out residuals. Barber et al. (2021) prove the 1 − 2α worst-case guarantee for jackknife+, not for the plain
jackknife, which has no distribution-free guarantee. The U5 caveat ("worst-case guarantee 1 − 2α") is withdrawn; U5's
coverages stay as reported, as empirical results.

**U5-A. Method.** Same episodes as U5 (MVTec-FS 13 products × K ∈ {3, 5} × 5 seeds × α ∈ {0.10, 0.20}; unknown-type
episodes; VisA screened defects). For each support i: the leave-one-out model gives R_i = −f^(−i)_{y_i}(x_i) and the
query nonconformity −f^(−i)_c(x). Jackknife+ set: c ∈ Γ⁺(x) iff #{i : −f^(−i)_c(x) > R_i} < (1 − α)(n + 1). With
identical leave-one-out models this reduces to the U5 set (checked in the self-test). Unknown-type flag: Γ⁺(x) empty.
Integrity, before anything is written: the U5 jackknife sets are recomputed for every episode and must reproduce
U5's coverage, class-mean coverage, mean size, empty rate, detection and false-alarm rate (1e-9).
Reported: as U5, jackknife+ next to jackknife. The guarantee applies to a test part exchangeable with the supports;
supports are drawn K per class, so the guaranteed quantity is the class-mean coverage (each class equally weighted),
not the coverage over the query mix.
Outputs `ckpt/p47/u2b/`, `ckpt/p47/u5a/` (write-once).
