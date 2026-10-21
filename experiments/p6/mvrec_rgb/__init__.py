"""P6.B MVREC-RGB support-only semantic-preservation audit (Protocol A).

This package implements A1-A5 of
`docs/P6B_MVREC_RGB_SEMANTIC_PRESERVATION_FAIR_COMPARISON_PROTOCOL.md`:
upstream component map, minimal RGB-only interface adaptation, dynamic
poison audit, support-only semantic smoke, and the machine-readable audit
artifact.

Strict rules implemented here:
- support-only: no query dataset may be instantiated during the audit;
- performance-blind: no accuracy / balanced accuracy / macro-F1 / confusion
  matrix is computed anywhere in this package;
- RGB-only firewall: no GT mask / bbox / ROI / points / localization field is
  read by the production adaptation path.

The adaptation reuses the official MVREC model components verbatim
(AlphaClip ViT-L/14 backbone, EchoClassfierF Zip-Adapter-F head, text prior,
loss, optimizer, scheduler, step budget) and only replaces the
annotation-driven data interface with a whole-image RGB loader plus a
runtime-generated constant mask placeholder.
"""
