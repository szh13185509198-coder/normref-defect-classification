#!/usr/bin/env python3
"""P35 (report only): ONNX export of the backbone forwards timed in P26/P31, for TensorRT latency on the Jetson.

Each configuration is exported exactly as ``experiments.p26.cost_profile.build_ours`` runs its backbone (same modules,
same layers, batch 1, fixed input size), with the timing input of P26 (bottle, first support image, preprocessed).
For every model the script writes ``<cfg>.onnx``, the input tensor ``<cfg>_input.bin`` (float32, raw) and the PyTorch
FP32 outputs ``<cfg>_ref.npz``, and checks onnxruntime (CPU, FP32) against PyTorch before anything goes to the device.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p35.export_onnx --mvtec-root D:\\intergration\\MVTEC --out D:\\intergration\\p35_trt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p26 import cost_profile as cp  # noqa: E402

CONFIGS = ("main", "clip_l14_336", "clip_l14_l18", "clip_l14_l12", "dinov2_b_336", "dinov2_s_336", "clip_b16_224", "clip_b16_336")


class Wrap(torch.nn.Module):
    def __init__(self, fn, modules):
        super().__init__()
        self.mods = torch.nn.ModuleList(modules)   # registers the weights
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def build(cfg: str, device, bake: bool = False, explicit: bool = False):
    """(module, prep, output names) running exactly the backbone call of build_ours."""
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p10.extract_dev_features import CLIP_LAYER
    from experiments.p18 import backbones as bb
    from experiments.p22 import lightweight as lw
    if cfg == "main":
        from experiments.p6 import run_p6b_stage2_support as support
        f = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))

        def fn(x):
            pooled, toks = f.clip_model.visual(x, [CLIP_LAYER])
            return pooled, toks[0], f.dino_extractor.forward_patches(x)
        return (Wrap(fn, [f.clip_model.visual, f.dino_extractor.model]), lambda im: models.deterministic_eval_tensor(im),
                ["clip_pooled", "clip_l18_tokens", "dino_patches"])
    if cfg in ("clip_l14_l18", "clip_l14_l12"):
        # CLIP-L/14@336 truncated after block k (P38-A1: k = 18, the CLIP-only configuration and the layer-18 gate;
        # P40: k = 12, the shallower screening exit); experiments.p36 ClipSplit, bitwise equal to the full forward
        from experiments.p36.early_exit_profile import ClipSplit
        k = int(cfg.rsplit("_l", 1)[1])
        clip_model, _ = models.build_clip_backbone(PROJECT_ROOT, device)
        split = ClipSplit(clip_model.visual, k)

        def fn(x):
            return (split.to_layer(x).permute(1, 0, 2),)
        return Wrap(fn, [clip_model.visual]), lambda im: models.deterministic_eval_tensor(im), [f"clip_l{k}_tokens"]
    if cfg == "clip_l14_336":
        clip_model, _ = models.build_clip_backbone(PROJECT_ROOT, device)

        def fn(x):
            pooled, toks = clip_model.visual(x, [CLIP_LAYER])
            return pooled, toks[0]
        return Wrap(fn, [clip_model.visual]), lambda im: models.deterministic_eval_tensor(im), ["clip_pooled", "clip_l18_tokens"]
    if cfg.startswith("dinov2_"):
        lw_cfg = {"dinov2_b_336": "dinov2_b_r336_fp32", "dinov2_s_336": "dinov2_s_r336_fp32"}[cfg]
        layer = lw.rc.load_json(lw.OUT["dev"] / "p22_dev_results.json")["selection"][lw_cfg]
        backbone = lw.build(lw_cfg, device)
        layers = sorted({layer, 12})

        if bake:
            # TensorRT recomputes the bicubic Resize of DINOv2's position-embedding interpolation differently (P35: engine
            # outputs disagreed, cos ~ 0). The input size is fixed, so the interpolated table is computed once here and
            # used as a constant; the forward is otherwise unchanged (checked bitwise against the original below).
            m = backbone.model
            table = m.interpolate_pos_encoding(torch.zeros(1, 1 + (backbone.res // m.patch_size) ** 2, m.embed_dim),
                                               backbone.res, backbone.res).detach()
            m.interpolate_pos_encoding = lambda x, w, h: table.to(x.dtype)
        if explicit:
            # The exported scaled_dot_product_attention derives its scale from the tensor shape (Shape -> Cast -> Sqrt
            # -> Div); the TensorRT engine disagreed even in FP32. Here every attention block uses the explicit form
            # softmax(q k^T * scale) v with scale a Python constant (numerically equivalent, checked below).
            import types

            def explicit_forward(self, x, is_causal=False):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
                return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
            for blk in backbone.model.blocks:
                blk.attn.forward = types.MethodType(explicit_forward, blk.attn)

        def fn(x):
            outs = backbone.model.get_intermediate_layers(x, n=[L - 1 for L in layers], norm=True, return_class_token=True)
            flat = []
            for patches, cls in outs:
                flat += [patches, cls]
            return tuple(flat)
        names = [n for L in layers for n in (f"L{L}_patches", f"L{L}_cls")]
        return Wrap(fn, [backbone.model]), lambda im: bb.eval_tensor(im, backbone.res), names
    from experiments.p18.run import dev_selection, parse_config
    from experiments.p24.clipb224 import layer_224
    res = int(cfg.rsplit("_", 1)[1])
    layer = layer_224() if res == 224 else parse_config(dev_selection()["clip_b16"])[2]
    backbone = bb.ClipB(device, res)

    def fn(x):
        pooled, toks = backbone.model.visual(x, [layer])
        return pooled, toks[0]
    return Wrap(fn, [backbone.model.visual]), lambda im: bb.eval_tensor(im, backbone.res), ["pooled", f"L{layer}_tokens"]


def main(argv=None) -> int:
    import onnx
    import onnxruntime as ort
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--bake-pos-embed", action="store_true", help="DINOv2: constant interpolated position table (_pb)")
    ap.add_argument("--explicit-attn", action="store_true", help="DINOv2: also explicit attention with a constant scale (_ex)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")   # export on the CPU: deterministic FP32 graph
    image = cp.task_inputs(Path(a.mvtec_root))[0][0]
    summary = {}
    for base in [c for c in CONFIGS if not a.only or c in a.only.split(",")]:
        if a.bake_pos_embed and not base.startswith("dinov2_"):
            raise SystemExit("--bake-pos-embed applies to the DINOv2 configurations only")
        if a.explicit_attn and not a.bake_pos_embed:
            raise SystemExit("--explicit-attn is used together with --bake-pos-embed")
        cfg = base + ("_ex" if a.explicit_attn else "_pb" if a.bake_pos_embed else "")
        onnx_path = a.out / f"{cfg}.onnx"
        if onnx_path.exists():
            raise SystemExit(f"{onnx_path} exists")
        module, prep, names = build(base, device)
        module.eval()
        x = prep(image).float()
        with torch.no_grad():
            ref = module(x)
            if a.bake_pos_embed:
                module, _, _ = build(base, device, bake=True, explicit=a.explicit_attn)
                module.eval()
                baked = module(x)
                if a.explicit_attn:   # explicit attention: equal up to FP32 rounding (the reference stays PyTorch's own)
                    worst = max(float((r - b).abs().max()) for r, b in zip(ref, baked))
                    print(f"[p35] {cfg}: explicit attention vs scaled_dot_product_attention max |d| = {worst:.2e}")
                    if worst > 1e-3:
                        raise SystemExit(f"{cfg}: explicit attention changes the outputs")
                elif not all(torch.equal(r, b) for r, b in zip(ref, baked)):
                    raise SystemExit(f"{cfg}: baked position table changes the outputs")
            torch.onnx.export(module, (x,), str(onnx_path), input_names=["input"], output_names=names, opset_version=17,
                              do_constant_folding=True, dynamo=False)
        onnx.checker.check_model(str(onnx_path))
        x.numpy().astype(np.float32).tofile(a.out / f"{cfg}_input.bin")
        np.savez(a.out / f"{cfg}_ref.npz", **{n: r.float().numpy() for n, r in zip(names, ref)})
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"input": x.numpy()})
        diff = {n: float(np.abs(o - r.float().numpy()).max()) for n, o, r in zip(names, outs, ref)}
        summary[cfg] = {"input_shape": list(x.shape), "outputs": {n: list(r.shape) for n, r in zip(names, ref)},
                        "ort_vs_torch_max_abs": diff, "onnx_mb": onnx_path.stat().st_size / 2 ** 20}
        print(f"[p35] {cfg}: {summary[cfg]}", flush=True)
        del module
    (a.out / f"p35_export_summary{'_ex' if a.explicit_attn else '_pb' if a.bake_pos_embed else ''}{'_' + a.only.replace(',', '_') if a.only else ''}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
