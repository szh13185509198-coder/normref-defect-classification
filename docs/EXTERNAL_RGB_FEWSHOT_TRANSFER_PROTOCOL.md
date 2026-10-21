# External Few-Shot Transfer Protocol — VisA & MTD

> **Status: PREREGISTERED — LOCKED BEFORE ANY EXTERNAL PERFORMANCE**
>
> This protocol is locked before any external-dataset model run. It
> modifies no existing P6.B / MVREC-RGB / K8 protocol, artifact or
> decision. It authorizes no performance experiment by itself; formal
> support-side execution requires separate later authorization. This
> file intentionally does NOT contain its own hash; the machine-readable
> pin lives in `docs/EXTERNAL_RGB_FEWSHOT_TRANSFER_PROTOCOL_LOCK.json`.

## 1. Scientific Question

The external experiment answers exactly one question:

> **Does the frozen method transfer beyond the MVTec-derived evaluation
> setting under the same no-privileged-localization few-shot
> supervision?**

It is NOT used for: hyperparameter selection; method redesign; V2
triggering; shot-scaling mechanism claims; post-hoc dataset selection.

This document defines a **preregistered external few-shot transfer
protocol** built on the public VisA and MTD datasets. It is not an
official VisA or MTD few-shot protocol; no such official few-shot
protocol is claimed or used (see §13).

## 2. Regime Naming (corrected)

The overall regime is named **no-privileged-localization image-only
few-shot supervision** — NOT "RGB-only external datasets". VisA (like
MVTec-FS-RGB before it) provides true RGB images; **MTD is a grayscale
transfer case** (its native images are single-channel). A
cross-color-condition observation may be reported descriptively, but no
"cross-color robustness" claim is made unless the results actually
support one, and any such wording remains descriptive, never a
mechanism claim.

## 3. Data Sources and Immutability

| Dataset | Archive | SHA256 (pinned from feasibility audit) |
|---|---|---|
| VisA | `data/VisA_20220922.tar` | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| MTD | `data/Magnetic-tile-defect-datasets.-master.zip` | `39bd97027c027116dc97f858db6e5028f98fdcf94b70e695c063f74e40976cf5` |

Feasibility commit: `f8db649ffc47eb5c8ad06ac7a5c452217f6ed5cc`
(`docs(external): add VisA and MTD feasibility audit`). Raw archives and
the extracted trees `data/VisA_raw/`, `data/MTD_raw/` are immutable and
Git-ignored. No image, label or archive may be modified, overwritten,
cropped into originals, or deleted; no class/object/sample may be
selected or dropped based on model performance.

## 4. Dataset 1 — VisA (primary external multi-object benchmark)

### 4.1 Task unit

Each VisA object independently forms an N-way classification task:
`Good` (= annotation label `normal`) + eligible defect subtypes. Labels
of different objects are never merged into a global class space.

### 4.2 Annotation boundary

Defect subtypes come only from the official `image_anno.csv`
image-level text label column. Forbidden: masks, bbox, ROI, polygon,
pixel locations, annotation-derived crops, annotation-derived views.

### 4.3 Multi-label exclusion (structural, frozen)

Any anomaly image with `len(defect_types) != 1` is excluded from the
strict single-label protocol. Sole reason: incompatible with strict
single-label classification. Multi-label counts per object (frozen from
the feasibility records): candle 13, capsules 70, cashew 10,
chewinggum 43, fryum 13, macaroni1 20, macaroni2 22, pcb1 9, pcb2 9,
pcb3 7, pcb4 56, pipe_fryum 12 (total 284).

### 4.4 Availability-only class rule (frozen before any performance)

A class (defect subtype or Good) is retained iff it has **≥ 10 unique
single-label original images** (decoded-RGB-SHA256 identity). Any
defect subtype below 10 is excluded and must never be restored.

### 4.5 Object eligibility gate (frozen)

An object enters the primary benchmark iff it keeps Good + **at least 2
retained defect subtypes** (N ≥ 3 classes). An object degenerating to
Good + 1 defect does not enter the primary VisA object-macro. This rule
is based on availability and the fine-grained task definition only.

### 4.6 Frozen retained class map

Counts are `eligible_unique/support_candidates/query` (partition per
§6). Class order is frozen: `normal` first, then ascending label text.

| Object | N classes | Classes (eligible/support/query) | Images |
|---|---:|---|---:|
| candle | 6 | normal 1000/500/500; chunk of wax missing 12/6/6; damaged corner of packaging 18/9/9; different colour spot 15/8/7; foreign particals on candle 12/6/6; wax melded out of the candle 12/6/6 | 1069 |
| cashew | 6 | normal 500/250/250; burnt 10/5/5; corner or edge breakage 14/7/7; same colour spot 10/5/5; small holes 20/10/10; small scratches 15/8/7 | 569 |
| chewinggum | 4 | normal 503/252/251; chunk of gum missing 16/8/8; corner missing 13/7/6; similar colour spot 19/10/9 | 551 |
| fryum | 5 | normal 500/250/250; different colour spot 14/7/7; fryum stuck together 20/10/10; middle breakage 18/9/9; similar colour spot 12/6/6 | 564 |
| macaroni1 | 6 | normal 1000/500/500; chip around edge and corner 17/9/8; different colour spot 15/8/7; similar colour spot 21/11/10; small cracks 11/6/5; small scratches 16/8/8 | 1080 |
| macaroni2 | 5 | normal 1000/500/500; color spot similar to the object 23/12/11; different color spot 12/6/6; small chip around edge 19/10/9; small scratches 16/8/8 | 1070 |
| pcb1 | 5 | normal 1004/502/502; bent 14/7/7; melt 46/23/23; missing 17/9/8; scratch 14/7/7 | 1095 |
| pcb2 | 5 | normal 1001/501/500; bent 12/6/6; melt 48/24/24; missing 17/9/8; scratch 14/7/7 | 1092 |
| pcb3 | 5 | normal 1006/503/503; bent 19/10/9; melt 38/19/19; missing 16/8/8; scratch 20/10/10 | 1099 |
| pcb4 | 3 | normal 1005/503/502; extra 12/6/6; missing 19/10/9 | 1036 |
| pipe_fryum | 7 | normal 500/250/250; burnt 15/8/7; corner and edge breakage 15/8/7; different colour spot 10/5/5; similar colour spot 20/10/10; small scratches 18/9/9; stuck together 10/5/5 | 588 |

**Retained: 11 objects, 57 class-units (11 Good + 46 defect), 9,813
eligible images.**

### 4.7 Frozen exclusions (never restorable)

Object-level: **capsules** — fewer than 2 retained defect subtypes
(only `bubble`, 30 unique, survives §4.4; N = 2 < 3).

Class-level (sole reason: availability < 10 unique single-label images):

| Object | Excluded subtype | Unique count |
|---|---|---:|
| candle | extra wax in candle | 9 |
| candle | weird candle wick | 9 |
| cashew | different colour spot | 9 |
| cashew | middle breakage | 6 |
| cashew | stuck together | 6 |
| chewinggum | scratches | 9 |
| fryum | burnt | 9 |
| fryum | corner or edge breakage | 8 |
| fryum | small scratches | 6 |
| macaroni2 | breakage down the middle | 1 |
| macaroni2 | small cracks | 7 |
| pcb4 | burnt | 2 |
| pcb4 | damage | 7 |
| pcb4 | scratch | 2 |
| pcb4 | wrong place | 2 |

## 5. Dataset 2 — MTD (secondary external benchmark)

Fixed **normal-inclusive 6-way** task: MT_Free (normal) + MT_Blowhole,
MT_Break, MT_Crack, MT_Fray, MT_Uneven. Class order frozen: MT_Free
first, then ascending label. Eligible/support/query: MT_Blowhole
115/58/57; MT_Break 85/43/42; MT_Crack 57/29/28; MT_Fray 32/16/16;
MT_Free 952/476/476; MT_Uneven 103/52/51 (total 1,344).

The defect-only 5-way variant is NOT a primary result and must not be
switched to because of performance.

**Grayscale contract:** MTD native images are single-channel grayscale.
The only allowed model input mapping is the deterministic channel
replication `RGB(x) = stack(L, L, L)` (three identical channels).
Forbidden: learned colorization, pseudo-color, channel-specific
transformations, any mask use. The paper must record MTD's native input
as grayscale.

## 6. Fixed Support-Candidate/Query Partition (seed-independent)

For every (dataset, task, class):

1. Collect the class's unique identities (decoded RGB SHA256, one per
   eligible original image).
2. Sort identities ascending by the SHA256 hex string (byte-order
   equivalent, platform-independent; no Python built-in `hash()`).
3. First `ceil(n/2)` identities → support candidate pool; remaining
   `floor(n/2)` → fixed query pool. **Predefined odd rule:** when n is
   odd, the extra image goes to the support candidate pool.
4. Fail-closed guarantee: support candidate pool ≥ 5 and query pool ≥
   5 (satisfied by all retained classes; minimum class n = 10 → 5/5).

The partition is model-independent and seed-independent; **query
identity is completely fixed across all K, seeds and methods**.

## 7. Nested Support Draw

Fixed: K = {1, 3, 5}; seeds = {111, 222, 333, 444, 555}.

For each (dataset, task, class, seed): rank every support candidate by

```
rank = SHA256("external_rgb_fewshot_transfer_v1|sha256_ranked_nested_v1|<dataset>|<task>|<class>|<seed>|<rgb_sha256>")
```

sort ascending by (rank, identity), and take the first 1 / 3 / 5
identities as K1 / K3 / K5. K1 ⊂ K3 ⊂ K5 holds by construction; the
query set does not change with K or seed. Mandatory check before any
use: support RGB SHA ∩ query RGB SHA = ∅.

### 7.1 Five-seed availability claim boundary

The five seeds control the deterministic nested support draw and the
training RNG only. For any class whose support candidate pool size
equals K, the K support identities necessarily coincide across seeds:
with pool = 5 and K = 5, every seed's K5 support set is the entire
pool. In this protocol the classes with support candidate pool exactly
5 are: cashew `burnt`, cashew `same colour spot`, pipe_fryum
`different colour spot`, pipe_fryum `stuck together` (eligible = 10 →
5 support / 5 query). This protocol must therefore NOT be described as
providing "five independent support sets for every K/class". The
correct description is: **five preregistered paired seeds**; any
identity coincidence across seeds is due solely to finite
availability.

## 8. Methods

Exactly three methods, all with the frozen semantics and
hyperparameters of the completed P6.B / MVREC-RGB formal experiments:

1. **Frozen-Fusion Proto**
2. **MVREC-RGB adaptation** (always named exactly this; whole images
   only; no VisA/MTD mask, ROI or bbox may be used)
3. **Ours**

Forbidden: external-dataset-specific tuning, learning-rate changes,
epoch changes, prompt search, prototype-count changes, adapter changes,
threshold tuning, query-assisted checkpoint selection.

## 9. Metrics

- Primary: **Balanced Accuracy**. Secondary: **Macro-F1**. Descriptive:
  **Accuracy**.
- VisA: metrics are computed within each object task first, then
  aggregated over retained objects with **equal-weight object macro**.
  Sample-weighted global pooling is forbidden as a primary result.
- MTD: six-way metrics computed directly on the single task.
- Across the five seeds: report mean ± SD and keep paired per-seed
  differences.

## 10. Statistical and Reporting Policy

- **No performance-driven GO/NO-GO branch.** Results are reported
  regardless of sign.
- Forbidden: result → tune method → rerun the same external query;
  starting V2 from these results; adding/removing classes or objects
  because of VisA/MTD results.
- Predefined transfer-evidence statistics: paired `Ours − baseline`
  differences with bootstrap CI as descriptive/confirmatory evidence.
- VisA bootstrap: paired **object-cluster** percentile bootstrap over
  the 11 retained objects (all 5 paired seeds of a sampled object
  together), 10,000 resamples, RNG seed `20260814`, two-sided 95% CI.
- MTD: a single task — **no fabricated object clusters**. Uncertainty
  is reported as seed-level paired differences (mean ± SD over the 5
  paired seeds) only.

## 11. Machine-Readable Protocol Lock

`docs/EXTERNAL_RGB_FEWSHOT_TRANSFER_PROTOCOL_LOCK.json` pins: this
protocol's canonical-LF SHA256; the feasibility commit; both archive
SHA256; retained VisA object list and class map; the exclusion map;
partition and support-draw algorithm versions; class-order version; K;
seeds; method registry; metric registry; statistics policy; grayscale
contract; manifest bundle completion SHA256. Lock verification is
fail-closed: any field loss, version drift or protocol-text change
invalidates the lock.

## 12. Formal Manifests (identity-only)

Pixel-read semantics, stated explicitly — two different things must
never be conflated:

- **Identity/integrity pixel decoding** — the only pixel reading
  allowed in the preregistration phase, exactly for: decoded RGB
  identity hashing, archive verification, manifest construction. The
  decoded RGB SHA256 identities used here were produced this way.
- **Model/performance pixel access** — zero: no model image access, no
  feature extraction, no prediction, no performance metric.

Artifacts record this honestly with the explicit fields
`image_pixels_identity_hash_read=true`,
`image_pixels_model_read=false`,
`performance_metrics_generated=false`.

After this protocol is locked, identity-only manifests may be generated
into `data/external_transfer_manifest_bundle/` (Git-ignored local
artifact; no raw data is committed): relative paths, labels,
raw/native SHA, decoded RGB SHA, fixed support-candidate/query
membership, K1/K3/K5 nested support identities, class order,
provenance. The global completion record pins: the exact task list;
all class maps; every manifest relative path with SHA256 and size;
protocol/lock identity; archive SHA256; the feasibility commit;
partition/draw versions; exact counts; and an aggregate SHA. A
lightweight Git-tracked index
`audit/external_transfer_manifest_index.json` mirrors every bundle
file's relative path + SHA256 + size plus the bundle completion SHA.
Formal cloud execution must use a transferred copy of this frozen
local bundle; regenerating the formal split on cloud is forbidden.
Manifest generation is NOT performance execution; models must not read
support/query pixels. Mandatory post-generation audits: exact count
audit; nesting audit; support/query overlap audit; duplicate identity
audit; archive provenance audit.

## 13. Claim Boundary

- The comparator is always "MVREC-RGB adaptation"; no claim against
  official MVREC.
- External results are transfer evidence under the frozen method; no
  SOTA, no universal-superiority, no shot-scaling mechanism claim.
- MTD must be described as a grayscale transfer case; no true-color
  information claim; no strong cross-color-condition robustness claim
  unless supported by the actual results, and then only descriptively.
- Negative or null external results are reported as-is; they do not
  retroactively modify P6.B, V1 or K8 records.
- This is a **preregistered external few-shot transfer protocol**. It
  must never be called an "official VisA few-shot protocol" or an
  "official MTD few-shot protocol"; no official few-shot protocol of
  either dataset is claimed or used.

## 14. Change Control

Any change to rules, thresholds, class/object sets, partitions, draws,
methods, metrics or statistics requires a new protocol version and a
new lock; the exclusions of §4.7 are permanent and cannot be revoked
within this protocol line.

---

*End of protocol. PREREGISTERED — LOCKED BEFORE ANY EXTERNAL
PERFORMANCE. No model, training, inference or metric was involved in
producing this document.*
