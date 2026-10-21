# P10 Pilot — Annotation-free localization cues for few-shot fine-grained defect classification

Status: **DEVELOPMENT PROTOCOL, LOCKED BEFORE ANY P10 OUTPUT** (2026-09-26)

## 1. Motivation (from REVISION-C)

Training-free SimpleShot-CL2N on the frozen global CLIP+DINOv2 fusion feature matches the
V1 support-side adaptation (RESEARCH_LEDGER, REVISION-C). Hypothesis: the bottleneck is
not the classifier but global pooling that dilutes small defect evidence with product
background. P10 asks whether annotation-free cues about *where to look* recover part of
what defect masks provide. Three cue sources are planned: (D) cross-class contrast inside
the support set (this pilot, no extra data), (A) unlabeled normal images, (F) zero-shot
vision-language anomaly maps. A and F get their own addenda before they run.

## 2. Firewall

- Development uses **only the MVTec-FS support pool** (`source_pools == ["support"]`,
  single-type, 565 images, 13 products). The 530 formal query images are never read:
  the code refuses any record whose pool is not exactly `["support"]`.
- Formal support manifests draw from the same pool; that is allowed (support images are
  not test data). Formal evaluation of any P10 method happens only after this pilot's
  gate and a separate formal addendum.
- Masks (labelme polygons) are used **only** for the oracle upper bound (Section 4.5),
  never inside a non-oracle method.

## 3. Development episodes

- Products: the 13 locked products. Shots: **K in {1, 3}** (K=5 is infeasible on the dev
  pool: several classes hold only 5 images). Dev seeds: 9001–9005.
- Per product, class, seed: shuffle the class's pool with
  `random.Random(f"p10dev|{seed}|{product}|{class}")`; the first K images are pseudo-support,
  the rest pseudo-queries. K=1 and K=3 are nested (same shuffle).
- Metric: balanced accuracy per episode; product-macro per (K, seed); mean over seeds.

## 4. Methods (all training-free; constants fixed here)

Features are extracted once with the formal frozen backbones on the deterministic 336 view:
global fusion (identical to formal `fusion_proto`), DINOv2 ViT-S/14 patch tokens
(24x24x384), CLIP ViT-L/14@336 layer-18 patch tokens (24x24x1024, CLS removed). Patch
descriptors are L2-normalized.

1. `fusion_proto` — class-mean cosine on the global fusion feature (reference).
2. `simpleshot_cl2n` — REVISION-C definition on the global fusion feature (**primary
   baseline**).
3. `local_plain_<desc>` — image-to-class local matching without cue: class patch set =
   all patches of its supports; a_c(q) = max cosine to that set; class score = mean of the
   top ceil(0.1 x 576) = 58 values of a_c over query patches.
4. `local_xclass_<desc>` — **cue D (query-side cross-class contrast)**. For each query
   patch q and class c, s_c(q) = max cosine to class-c support patches;
   margin_c(q) = s_c(q) - max over c' != c of s_c'(q). Class score = mean of the top 58
   margins over query patches. Shared background matches all classes alike (margin ~ 0),
   so only class-specific evidence contributes.
   *Revision before any output (2026-09-26):* the first draft selected the top 10% of
   support patches by distance to other classes; a synthetic unit test showed background
   leaks into that set whenever defects cover < 10% of an image and saturates all class
   scores. The query-side margin replaces it. No real-data output existed.
5. `fused_xclass_<desc>` — per query, z-score (over classes) the `simpleshot_cl2n` scores
   and the `local_xclass_<desc>` scores, sum with equal weight, argmax. If the std over
   classes is 0, that term contributes 0.
6. Oracles (upper bounds, use GT masks; never ranked as methods):
   `oracle_maskpool_<desc>`: each image's descriptor = mask-weighted mean of its patch
   tokens (weight = mask area fraction per patch; fallback to plain mean if the mask is
   empty), then SimpleShot-CL2N. `oracle_masklocal_<desc>`: method 3 with support and
   query patch sets restricted to patches with mask fraction > 0 (fallback all patches).

`<desc>` in {`dino`, `clip18`}. Ties resolve to the lowest class index.

## 5. Pilot gate (decided now)

For each non-oracle P10 method M (3, 4, 5 x both descriptors):

- Delta_K = dev product-macro BA(M) - BA(simpleshot_cl2n), averaged over seeds.
- **PASS** iff mean(Delta_1, Delta_3) >= **+3.0 pp** and at least **9/13** products have a
  positive seed-and-K-averaged difference.
- Oracle headroom H = mean over K of BA(best oracle) - BA(simpleshot_cl2n). If H < 3.0 pp,
  localization is not the bottleneck on this data and the whole P10 direction (D, A, F)
  is stopped.
- Reporting: every method and descriptor is reported regardless of outcome. Choosing the
  best of six non-oracle variants is itself a selection; any method that passes goes to a
  formal addendum that fixes that single variant before touching the formal queries.

## 6. Outputs

`ckpt/p10_pilot/` (gitignored): `dev_features/<product>.pt`, `pilot_results.json`,
`pilot_table.md`.

---

# Addendum P10-A — cue A (normal-reference) dev pilot

Status: **LOCKED BEFORE ANY CUE-A OUTPUT** (2026-09-26). Cue-D dev results are already
known (RESEARCH_LEDGER "P10 PILOT") and are disclosed as prior information.

## A.1 Normal reference data

- Source: official MVTec AD release at `D:\intergration\MVTEC`, archive SHA256
  `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`; **only
  `<product>/train/good`** is used (3,345 images for the 13 products). `test/good` and all
  defect folders are never read.
- Byte-identity check: 0 of the 3,345 normal files equals any of the 1,136 MVTec-FS
  records (support or query).
- Information condition: "image-level labels + unlabeled normal images of the same
  product". Reported separately from cue D, which uses image-level labels only.

## A.2 Normal bank

- Per product and dev seed: `random.Random(f"p10A|{seed}|{product}").sample(sorted(train/good), 20)`.
  M=20 is the primary bank; M=5 = the first 5 of the same draw (nested), sensitivity only.
- Descriptors: the same L2-normalized DINOv2 / CLIP layer-18 patch tokens as the pilot,
  deterministic 336 view.

## A.3 Cue A and methods

Per image and descriptor X: patch anomaly a(p) = 1 - max cosine of p to the bank patches.
Anomaly-selected pooling: descriptor = mean of the top ceil(0.1 x 576) = 58 patches by a
(no temperature, no threshold). Classification = SimpleShot-CL2N on these descriptors.

Gate-eligible variants (M=20), X in {dino, clip18}:
1. `normref_pool_X` — anomaly-selected pooling + SimpleShot-CL2N.
2. `fused_normref_X` — z(simpleshot_cl2n) + z(normref_pool_X).
3. `fused_normref_xclass_X` — z(simpleshot_cl2n) + z(normref_pool_X) + z(local_xclass_X).

Sensitivity (not gate-eligible): the same three with M=5 (suffix `_m5`).
Diagnostic: per-image patch AUROC of a(p) against the mask grid (mask fraction > 0 as
positive), averaged over dev images that contain both positive and negative patches.

## A.4 Gate and selection

- Same gate as Section 5: mean(Delta_1, Delta_3) >= +3.0 pp vs `simpleshot_cl2n` and
  >= 9/13 positive products. Report the fraction of oracle headroom recovered
  (Delta / oracle Delta of the same descriptor, maskpool oracle).
- **Formal selection rule (fixed now, applied jointly to all gate-passing D and A
  variants):** highest number of positive products; ties broken by higher mean Delta.
  Exactly one variant goes to the formal addendum. Under this rule cue D alone would
  select `fused_xclass_clip18` (12/13).

---

# Addendum P10-A2 — attribution controls for cue A (dev)

Status: **LOCKED BEFORE ANY CONTROL OUTPUT** (2026-09-26). Selected variant from P10-A:
`fused_normref_clip18` (dev +7.70 pp vs SimpleShot, 12/13).

Question: is the cue-A gain due to *localization*, or merely to *having normal images*
(information condition C1) or to *pooling a patch subset*?

Controls (not selection candidates), same dev episodes, same M=20 banks, X in
{dino, clip18}:

1. `normcenter_mean_X` (C1, same information and descriptor as cue A, no localization):
   descriptor = mean of all 576 patches; SimpleShot-CL2N whose centering vector is the
   mean descriptor of the 20 normal-bank images instead of the support mean.
2. `fused_normcenter_mean_X` = z(simpleshot_cl2n) + z(normcenter_mean_X): the direct
   same-information analogue of `fused_normref_X`.
3. `random_pool_X` (C0): descriptor = mean of 58 uniformly random patches per image
   (`torch.Generator().manual_seed(seed * 1000 + image_index)`), SimpleShot-CL2N.
   Tests whether any patch-subset pooling helps.
4. `fused_random_pool_X` = z(simpleshot_cl2n) + z(random_pool_X).

Attribution rule (fixed now): the manuscript may attribute the cue-A gain to
localization on dev only if `fused_normref_clip18` minus `fused_normcenter_mean_clip18`
has mean over K >= +3.0 pp and >= 9/13 positive products. Otherwise the claim must be
restated as "unlabeled normal images help", not "localization helps". Controls 3–4 are
descriptive.
