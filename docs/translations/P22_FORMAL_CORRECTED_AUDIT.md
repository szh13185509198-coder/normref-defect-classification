<!-- translation-header -->
> English translation of [`P22_FORMAL_CORRECTED_AUDIT.md`](../P22_FORMAL_CORRECTED_AUDIT.md), which is kept unchanged as the committed record (SHA-256 `4bb044ce4970a84f10689ca5ee554b0fd23fc8c00c3c3bfacff4e8e7aa27728b`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P2.2 formal test (normal-aware): mechanical audit of the correction

This audit is based only on `ckpt/p22_fixed_threshold_tau0006/p22_formal_acceptance.json` and the three
`normal_aware_predictions.jsonl` files. It contains no new threshold search, causal interpretation or change of
method.

## A. Files and hash integrity

- s111, s222 and s333 each contain 79 distinct formal prediction records.
- All three use the locked threshold `τ=0.006`.
- The SHA-256 of the calibration file is `f242afff133310550edde66a795b3006a6a4d48f66fdfa7042a04679ec491929` for all three.
- SHA-256 of the formal checkpoints:
  - s111: `13c2006888793b77c0910859679a2ab6b1f00899f049ed719afd2cbde7eb5567`
  - s222: `0f79f24e14f743d56f3d6dab42b7002f188f667386945d395921ab7a1c749a31`
  - s333: `1cd7ac4380a16a45216bc4f8290111f62eafdc0e47d52ce3e5b0715acaa7efb1`

## B. Raw and calibrated metrics

| Seed | Overall | Nine-class BA | Macro-F1 | Good recall | Defect recall | Binary BA | FP | Changed / corrected / harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 111 | 87.34% → 89.87% | 90.81% → 91.26% | 83.03% → 84.22% | 84% → 88% | 100% → 100% | 92% → 94% | 8 → 6 | 2 / 2 / 0 |
| 222 | 87.34% → 87.34% | 90.81% → 90.81% | 83.03% → 83.03% | 84% → 84% | 100% → 100% | 92% → 92% | 8 → 8 | 0 / 0 / 0 |
| 333 | 83.54% → 84.81% | 90.15% → 90.37% | 81.66% → 82.08% | 78% → 80% | 100% → 100% | 89% → 90% | 11 → 10 | 1 / 1 / 0 |

Mean over the three seeds:

- Overall: `86.08% → 87.34%`
- Nine-class BA: `90.59% → 90.81%`
- Macro-F1: `82.57% → 83.11%`
- Good recall: `82% → 84%`
- Defect recall: `100% → 100%`
- Binary BA: `91% → 92%`

## C. The three changed predictions

1. s111: `good/train/images/019.png`, `poke_insulation → good`, margin=`0.0019763112`
2. s111: `good/train/images/034.png`, `poke_insulation → good`, margin=`0.0027542710`
3. s333: `good/train/images/019.png`, `poke_insulation → good`, margin=`0.0007056594`

All three changes are corrections; no defect was newly missed and no other prediction was made worse.

## D. Locked conclusion

- The pre-specified acceptance result of P2.2 is `accepted`.
- `τ=0.006` is locked permanently and is not adjusted on the formal test.
- The result shows a conservative improvement in a consistent direction at no cost in defect recall; the number of
  corrections is small, so it is not described as statistically significant or as a large gain.
- This document does not cover the causal interpretation or the architecture choice of the next stage.
