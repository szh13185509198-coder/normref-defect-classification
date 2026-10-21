# P16 — Figures v4 and evidence archive v3 (no evaluation)

Status: fixed before rendering (2026-09-27). Reads only frozen result files (P10–P15) and cached
features; no new evaluation, no selection. Output `docs/paper/figures_v4/` (PNG, PDF, CSV each).

1. MVTec-FS line chart: mask oracle, ARR (ours, C1), A, P10 fixed-fraction normal reference,
   CONS (ours, C0), cross-class, SimpleShot, V1 adaptation, MVREC-RGB.
2. Per-product Δ vs SimpleShot: ARR, P10 normal reference, CONS (sorted by ARR).
3. Ablation chain (MVTec-FS formal, pooled BA, 95 % CI): P10 H1; P12 soft top-10 % − P10;
   P12 A − soft top-10 %; P14 H6. C0: P10 H3; P14 H7.
4. Threshold-source controls: A − Otsu / MEBin-style / soft top-10 % on MVTec-FS and VisA (P12).
5. VisA forest: E6, ARR − SimpleShot, E7, CONS − SimpleShot, E1.
6. Industrial argmax false rejects per 1,000 (VisA, MTD): SimpleShot, cross-class, CONS,
   P10 normal reference, ARR (P14), V1, MVREC-RGB (P10 R2).
7. Two-stage operating points (P15): VisA K=5, MTD K=5, Cable held-out split K=8 — α = 1/2/5 %,
   three-way bands, argmax ARR; review items annotated.
8. Qualitative (deterministic, as in v3 but with ARR predictions from P14): 3 largest and 3
   smallest ARR − SimpleShot products; K=5, seed 111; first success / first failure by `rgb_sha256`.
9. Region comparison: re-rendered v3 figure (the region is shared by A and ARR).
10. Cable: first two `combined` persistent errors of ARR on the OOF pool (P14).

The Cable test is labelled a **same-source held-out split** of MVTec AD `cable/test`, never
"independent". Evidence: `experiments/p16/archive_evidence_v3.py` →
`docs/paper/P16_EVIDENCE_MANIFEST.json` + `p16_evidence_*.tar.gz`.
