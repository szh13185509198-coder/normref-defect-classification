# P13 Pilot — region residual (idea 1, C1), support-consensus background (idea 2, C0), multi-region exploration (idea 5)

Status: **DEVELOPMENT PROTOCOL, LOCKED BEFORE ANY P13 OUTPUT** (2026-09-27).

A = `fused_adareg_soft_q99` stays the C1 main method. P13 is dev-only. Candidates passing
the gates may enter **one final formal addendum** (written after this pilot, declared
post-hoc, single run). No further formal rounds after that.

## 1. Firewall

Dev features (MVTec-FS support pool), P10-A normal banks, P10 dev episodes (seeds 9001–9005,
K = 1/3). Formal queries are never read. Masks: region diagnostics only. The Cable
exploration (§4) uses the Cable dev pool only; the sealed Cable test is never read.

## 2. Idea 1 — region residual (C1, target A)

With A's bank, threshold τ_0.99 and soft weights w(t) = max(a(t) − τ, 0) (top-6 fallback):
residual r(t) = t − nn_B(t) (nearest bank patch, cosine); h_RR(x) = Σ w r / Σ w.

- `fused_regres_soft_q99`: z(S) + z(SimpleShot on h_RR).
- `fused_adareg_regres_soft_q99` (parallel branches): z(S) + z(A score) + z(h_RR score).

**Gate (vs A):** mean over K of product-macro Δ ≥ **+1.0 pp** and Δ > 0 on **≥ 8/13** products.
Selection: most positive products, ties by mean Δ; at most one goes to formal.

## 3. Idea 2 — support-consensus background (C0, target cross-class)

No normal images. Per episode, s_c(t) = max cosine of patch t to the patches of class-c
supports (CLIP layer 18).

- **Background similarity** b(t): for a query patch, the second-largest s_c(t) over classes
  (it matches a class other than its best one); for a support image of class y,
  max over c ≠ y of s_c(t) (its own class is never used, so no leave-one-out is needed).
- Consensus anomaly a0(t) = 1 − b(t): shared background matches several classes (low a0);
  class-specific defect patches match only their own class (high a0).
- Soft weights w0(t) = max(a0(t) − τ0, 0) with the top-6 fallback; h0(x) = Σ w0 t / Σ w0;
  SimpleShot on h0; fused z(S) + z(·).
- τ0 variants: `fused_cons_soft_top10` (per image: 59th largest a0 → 58 patches),
  `fused_cons_soft_supq90` / `fused_cons_soft_supq95` (per episode: 0.90 / 0.95 quantile of
  a0 over all support patches — support images only).

**Gate (vs `fused_xclass_clip18`):** mean Δ ≥ **+2.0 pp** and ≥ **8/13** products.
Selection as in §2. Also reported: Δ vs SimpleShot, and vs A (cross-condition, descriptive).

## 4. Idea 5 — multi-region exploration on Cable (descriptive only, no gate, no method)

Cable 72-image dev pool, the P10 R1 normal banks (5 seeds, M = 20), A's region (w > 0).
Per image: number of 8-connected components of the region on the 24×24 grid and the region
size. Report per class the mean component count and the fraction of images with ≥ 2
components, averaged over seeds, plus the same for A's persistent errors (P12 R1'). This only
tests whether co-occurring defects appear as separate regions; no decision rule is changed.

## 5. References and integrity

Same run recomputes `simpleshot_cl2n`, `fused_xclass_clip18`, `fused_normref_clip18`, A;
A must equal the P11 dev pilot rows and `fused_xclass_clip18` the P10 dev pilot value exactly.

## 6. Outputs

`ckpt/p13_pilot/dev/` and `ckpt/p13_pilot/cable_explore/` (write-once).
