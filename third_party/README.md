# Third-party code (download yourself)

This folder is empty on purpose. Place each checkout at the path below.

| Path | Upstream | Version | License | Used by |
|---|---|---|---|---|
| `third_party/dinov2` | https://github.com/facebookresearch/dinov2 | main branch (only `hubconf.py` and the model code are used) | Apache-2.0 | DINOv2 features in all rounds, loaded offline with `torch.hub.load(..., source="local")`; the checkpoint SHA256 is checked |
| `third_party/CLIP-LoRA` | https://github.com/MaxZanella/CLIP-LoRA | commit `ce3701ba370576f082e1295fbd90760961ead724` | no license file upstream | CLIP-LoRA baseline (`p21`, `p25`, `p29`), timing (`p26`, `p30`), GDA data helpers (`p44`) |
| `alpha_clip` Python package | https://github.com/SunzeY/AlphaCLIP | as released | Apache-2.0 | MVREC-RGB baseline only (`p6`, `p23`), installed with `pip install -e .` from the checkout |

```bash
git clone https://github.com/facebookresearch/dinov2 third_party/dinov2
git clone https://github.com/MaxZanella/CLIP-LoRA third_party/CLIP-LoRA
git -C third_party/CLIP-LoRA checkout ce3701ba370576f082e1295fbd90760961ead724
```

The CLIP-LoRA runners refuse a checkout with local modifications.
