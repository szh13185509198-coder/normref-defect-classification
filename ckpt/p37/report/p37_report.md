# P37 report — recommendation rule (protocol §5)

| Set | AUROC CLIP layer 18 | AUROC DINOv2-S | DINOv2-S ≥ CLIP − 0.02 |
|---|---|---|---|
| VisA | 0.929 | 0.904 | False |
| MTD | 0.798 | 0.851 | True |
| Cable | 0.969 | 0.948 | False |

**Light gate recommended as the deployment default: False**

Expected per-part latency on the Jetson Orin Nano (P36 FP16 medians; ms):

- VisA (99 % good): no gate 170.1; clip_l18|a=0.01 119.8; clip_l18|a=0.02 120.4; clip_l18|a=0.05 122.1; clip_l18|a=0.1 125.0; clip_l18|a=0.3 135.3; dino_s|a=0.01 39.4; dino_s|a=0.02 40.9; dino_s|a=0.05 45.1; dino_s|a=0.1 52.0; dino_s|a=0.3 80.9
- MTD (99 % good): no gate 170.1; clip_l18|a=0.01 119.4; clip_l18|a=0.02 119.5; clip_l18|a=0.05 120.2; clip_l18|a=0.1 122.6; clip_l18|a=0.3 132.1; dino_s|a=0.01 40.1; dino_s|a=0.02 41.2; dino_s|a=0.05 45.2; dino_s|a=0.1 51.0; dino_s|a=0.3 78.4
- Cable (99 % good): no gate 170.1; clip_l18|a=0.01 119.6; clip_l18|a=0.02 119.9; clip_l18|a=0.05 120.9; clip_l18|a=0.1 122.4; clip_l18|a=0.3 129.5; dino_s|a=0.01 38.7; dino_s|a=0.02 38.8; dino_s|a=0.05 41.8; dino_s|a=0.1 45.3; dino_s|a=0.3 60.2
