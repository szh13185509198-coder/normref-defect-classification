#!/usr/bin/env python3
"""P36 (report only): device timing of the deployment variants that follow from the method's structure.

The patch evidence, the patch anomaly a(t) and the two-level screening score (P15) all use CLIP ViT-L/14@336 layer 18
of 24; layers 19-24 serve only the global score. With the P26 harness conventions (bottle, K = 5, seed 111, M = 20
bank, batch 1, every block in its own process, a forward and a reverse pass), this measures:

  A1  l14_l18_fp32 / _fp16   P30's CLIP-only `a_r` (appearance + deviation, no global score) from blocks 1-18 only;
  A2  gate_l18_fp16          two-level inspection with the screening decision taken right after block 18: the "pass"
                             path stops there, the "flagged" path continues from block 18 (blocks 19-24, DINOv2-S,
                             the dual-branch head) -- every decision identical to P15;
  B   gate_dino_fp16         light gate (P37): screening with the main configuration's own DINOv2-S patches against a
                             DINOv2-S normal bank; the "flagged" path then runs CLIP-L/14 and the head (DINOv2-S reused);
  D   *_precast              FP16 with the Linear / Conv / attention in-projection weights cast once ahead of time,
                             under the same autocast (no per-forward weight cast); features must be bitwise equal;
  anchors                    ours_main_fp16 and ours_l14_336_fp16 exactly as in r3 (run-to-run comparison).

Before timing, every block checks itself and fails closed: truncated / continued forwards and pre-cast features bitwise
equal to the original forward, screening scores equal to the P15 formula, timed heads within 1e-4 of the evaluation
code. No accuracy is measured (A and D change no decision; the accuracy of B is P37).

Usage:
    python -m experiments.p36.early_exit_profile --tag <tag> --mvtec-root <MVTec AD root> [--only a,b] [--passes 2]
    (internally: python -m experiments.p36.early_exit_profile --worker <block> ...)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p26 import cost_profile as cp  # noqa: E402  (no torch at import)

BLOCKS = ("anchor_main_fp16", "anchor_l14_336_fp16", "l14_l18_fp32", "l14_l18_fp16", "gate_l18_fp16",
          "gate_dino_fp16", "main_fp16_precast", "clipb16_224_fp16_precast", "clipb16_336_fp16_precast",
          "l14_l18_fp16_precast")
ANCHORS = {"anchor_main_fp16": "ours_main_fp16", "anchor_l14_336_fp16": "ours_l14_336_fp16"}
PRECAST_BASE = {"main_fp16_precast": "ours_main_fp16", "clipb16_224_fp16_precast": "ours_clipb16_224_fp16",
                "clipb16_336_fp16_precast": "ours_clipb16_336_fp16"}
OUT_ROOT = PROJECT_ROOT / "ckpt" / "p36"
SCREEN_STAT = "top1pct"   # P15's screening statistic (mean of the top 1 % patch anomalies)


# --------------------------------------------------------------------------- the CLIP tower split at a block

class ClipSplit:
    """VisionTransformer.forward (model/transformer.py) split after block `layer`: the same operations in the same
    order (OpenAI CLIP layout: no input patchnorm, no attention pooling, no grad checkpointing in eval).
    to_layer(x) -> hidden state after block `layer` (LND; its NLD view is entry `layer` of visual(x, [layer]));
    rest(h) -> the projected global embedding that visual(x, out_layers) returns."""

    def __init__(self, visual, layer: int):
        if visual.input_patchnorm or visual.attn_pool is not None or visual.transformer.grad_checkpointing:
            raise RuntimeError("ClipSplit mirrors the OpenAI CLIP forward only")
        blocks = list(visual.transformer.resblocks)
        if not 0 < layer <= len(blocks):
            raise ValueError(layer)
        self.visual, self.layer, self.first, self.after = visual, layer, blocks[:layer], blocks[layer:]

    def to_layer(self, x):
        import torch
        v = self.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device), x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for r in self.first:
            x, _ = r(x, attn_mask=None)
        return x

    def rest(self, h):
        v = self.visual
        x = h
        for r in self.after:
            x, _ = r(x, attn_mask=None)
        x = x.permute(1, 0, 2)
        pooled, _ = v._global_pool(x)
        pooled = v.ln_post(pooled)
        if v.proj is not None:
            pooled = pooled @ v.proj
        return pooled

    def params_to_layer(self) -> int:
        v = self.visual
        own = [v.class_embedding, v.positional_embedding]
        return (sum(p.numel() for p in own) + sum(p.numel() for m in (v.conv1, v.ln_pre, *self.first)
                                                  for p in m.parameters()))


def precast_fp16(*modules) -> int:
    """Cast once the weights autocast would otherwise cast at every forward: weight and bias of every Linear / Conv2d
    and the MultiheadAttention input projection. LayerNorm, embeddings, LayerScale and the CLIP projection stay as they
    are (autocast treats them per op exactly as before). Returns the number of parameters cast."""
    import torch
    n = 0
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, (torch.nn.Linear, torch.nn.Conv2d)):
                names = ("weight", "bias")
            elif isinstance(sub, torch.nn.MultiheadAttention):
                names = ("in_proj_weight", "in_proj_bias")
            else:
                continue
            for name in names:
                p = getattr(sub, name, None)
                if p is not None and p.dtype == torch.float32:
                    p.data = p.data.half()
                    n += p.numel()
    return n


class ClipL18Pipeline:
    """cost_profile.ClipOnlyPipeline (P30 `a_r` features: layer-18 tokens, CLS dropped, L2-normalised) computed from
    blocks 1-18 only; precision "fp16" = CUDA autocast of the forward, as ClipOnlyPipeline."""

    def __init__(self, split: ClipSplit, precision: str, device):
        if precision not in ("fp32", "fp16"):
            raise ValueError(precision)
        self.split, self.precision, self.device = split, precision, device

    def forward(self, batch):
        import torch
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.precision == "fp16" and self.device.type == "cuda"
               else contextlib.nullcontext())
        with ctx:
            return self.split.to_layer(batch)

    def features(self, image):
        import torch.nn.functional as F
        from experiments.p6 import p6b_stage2_models as models
        h = self.forward(models.deterministic_eval_tensor(image).to(self.device))
        return None, F.normalize(h.permute(1, 0, 2)[:, 1:, :].float(), dim=-1)


# --------------------------------------------------------------------------- workers

def _equal_features(a, b) -> bool:
    import torch
    return all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a, b))


def _max_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b) if x is not None)


def _finish(res: dict, forward, prep, image, device) -> dict:
    import torch
    x = prep(image)
    res["model_forward"] = cp.timed(lambda: forward(x), device)
    res["gflops_per_image_backbone"] = cp.gflops(lambda: forward(x))
    res["precision"] = "fp16"
    res["per_task_preparation_s"] = sum(res["offline_ms"].values()) / 1000.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    return res


def worker_l14_l18(block: str, args, device) -> dict:
    """A1 (and A1 + D): P30's `a_r` from blocks 1-18; timed with cost_profile.profile_ar like the r3 block."""
    import torch
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p10.extract_dev_features import CLIP_LAYER
    supports, s_lab, n, bank, _, _ = cp.task_inputs(Path(args.mvtec_root))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    precision = "fp32" if block == "l14_l18_fp32" else "fp16"
    with torch.no_grad():
        clip_model, _ = models.build_clip_backbone(PROJECT_ROOT, device)
        split = ClipSplit(clip_model.visual, CLIP_LAYER)
        pipe = ClipL18Pipeline(split, precision, device)
        full = cp.ClipOnlyPipeline(clip_model, CLIP_LAYER, precision, device)   # the r3 / P30 pipeline
        probe = [supports[0], supports[-1], bank[0]]
        before = [pipe.features(im) for im in probe]
        checks = {"layer18_tokens_bitwise_equal_full_forward": all(_equal_features(b, full.features(im))
                                                                    for b, im in zip(before, probe))}
        if block.endswith("_precast"):
            checks["precast_params"] = precast_fp16(clip_model.visual)
            after = [pipe.features(im) for im in probe]
            checks["precast_features_bitwise_equal"] = all(_equal_features(a, b) for a, b in zip(after, before))
            if not checks["precast_features_bitwise_equal"]:
                checks["precast_max_abs_diff"] = max(_max_diff(a, b) for a, b in zip(after, before))
        if not all(v for k, v in checks.items() if k.endswith("equal") or k.endswith("full_forward")):
            raise SystemExit(f"{block}: bitwise check failed: {checks}")
        res = cp.profile_ar(block, pipe, supports, bank, s_lab, n, device, sync)
        prep = lambda im: models.deterministic_eval_tensor(im).to(device)  # noqa: E731
        res = _finish(res, pipe.forward, prep, supports[0], device)
    res["precision"] = precision
    res["checks"] = checks
    res["params_loaded"] = sum(p.numel() for p in clip_model.visual.parameters())
    res["params_used"] = split.params_to_layer()
    return res


def _backbone_modules(pipe) -> list:
    inner = pipe.pipe   # cost_profile.Autocast16
    if hasattr(inner, "factory"):
        return [inner.factory.clip_model.visual, inner.factory.dino_extractor]
    return [inner.backbone.model.visual]


def worker_precast(block: str, args, device) -> dict:
    """D: an r3 FP16 block with its backbone weights pre-cast; timed exactly like the r3 block."""
    import torch
    from experiments.p18.resource_profile import profile
    supports, s_lab, n, bank, _, _ = cp.task_inputs(Path(args.mvtec_root))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        pipe, params, prep, forward = cp.build_ours(PRECAST_BASE[block], device)
        probe = [supports[0], supports[-1], bank[0]]
        before = [pipe.features(im) for im in probe]
        checks = {"precast_params": precast_fp16(*_backbone_modules(pipe))}
        after = [pipe.features(im) for im in probe]
        checks["precast_features_bitwise_equal"] = all(_equal_features(a, b) for a, b in zip(after, before))
        if not checks["precast_features_bitwise_equal"]:
            checks["precast_max_abs_diff"] = max(_max_diff(a, b) for a, b in zip(after, before))
            raise SystemExit(f"{block}: bitwise check failed: {checks}")
        res = profile(block, pipe, supports, bank, s_lab, n, device, sync)
        res = _finish(res, forward, prep, supports[0], device)
    res["params"] = params
    res["checks"] = checks
    return res


def worker_gate(block: str, args, device) -> dict:
    """A2 (gate_l18_fp16) / B (gate_dino_fp16) on the main configuration, FP16 exactly as r3's ours_main_fp16
    (autocast of the feature extraction, heads in FP32)."""
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10 import run_pilot as rp
    from experiments.p10 import run_pilot_a as ra
    from experiments.p10.extract_dev_features import CLIP_LAYER
    from experiments.p11 import formal_ab as fa
    from experiments.p11 import run_pilot_ab as ab
    from experiments.p13 import pilot as p13
    from experiments.p14.resource_profile import proto_space, score
    from experiments.p15 import screening as sc
    from experiments.revision.c3_resource_profile import _time
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    ac = lambda: torch.autocast("cuda", dtype=torch.float16)  # noqa: E731
    prep = lambda im: models.deterministic_eval_tensor(im).to(device)  # noqa: E731
    offline = {}

    def timed_once(fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        return out, (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        visual, dino, fusion = factory.clip_model.visual, factory.dino_extractor, factory.fusion
        split = ClipSplit(visual, CLIP_LAYER)

        def fuse(pooled, d):
            return fusion.fuse(models.l2_rows(pooled), models.l2_rows(d.mean(dim=1))).float()

        def full_features(image):
            """== cost_profile.Autocast16(p14 Pipeline).features (r3 ours_main_fp16), plus the DINOv2-S patches."""
            with ac():
                x = prep(image)
                pooled, tokens = visual(x, [CLIP_LAYER])
                d = dino.forward_patches(x)
                return fuse(pooled, d), F.normalize(tokens[0][:, 1:, :].float(), dim=-1), d

        query = supports[0]
        for _ in range(5):
            full_features(query)
        feats, offline["support_feature_extraction_ms"] = timed_once(lambda: [full_features(im) for im in supports])
        bank_feats, offline["bank_feature_extraction_ms"] = timed_once(
            lambda: [full_features(im) for im in bank_images])
        g_s, p_s = torch.cat([f[0] for f in feats]), torch.cat([f[1] for f in feats])
        bank_imgs = torch.cat([b[1] for b in bank_feats])
        bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
        bank_d = F.normalize(torch.cat([b[2] for b in bank_feats]).float(), dim=-1).reshape(-1, 384)
        tau, offline["tau_calibration_loo_ms"] = timed_once(lambda: fa.tau99(bank_imgs))

        def prepare():
            w = ab.region_weights(ra.patch_anomaly(p_s, bank), tau, soft=True)
            return {"g": proto_space(g_s, s_lab, n), "a": proto_space(ab.weighted_pool(p_s, w), s_lab, n),
                    "rr": proto_space(p13.region_residual(p_s, bank, w), s_lab, n)}
        state, offline["support_state_ms"] = timed_once(prepare)

        def head(g, c, best, idx):
            """cost_profile / p18 profile's dual_branch_ARR, with the nearest-neighbour search passed in."""
            w = ab.region_weights(1.0 - best, tau, soft=True)
            sa = score(ab.weighted_pool(c, w), state["a"])
            srr = score(ab.weighted_pool(c - bank[idx], w), state["rr"])
            return rp._zscore(score(g, state["g"])) + rp._zscore(sa) + rp._zscore(srr)

        def full_path(image):
            g, c, _ = full_features(image)
            best, idx = torch.einsum("npd,bd->npb", c, bank).max(dim=2)
            return head(g, c, best, idx)

        def l18_gate(image):
            x = prep(image)
            with ac():
                h = split.to_layer(x)
                c = F.normalize(h.permute(1, 0, 2)[:, 1:, :].float(), dim=-1)
            best, idx = torch.einsum("npd,bd->npb", c, bank).max(dim=2)
            return x, h, c, best, idx, float(sc.image_score(1.0 - best, SCREEN_STAT)[0])

        def l18_flagged(image):
            x, h, c, best, idx, _ = l18_gate(image)
            with ac():
                pooled = split.rest(h)
                g = fuse(pooled, dino.forward_patches(x))
            return head(g, c, best, idx)

        def dino_gate(image):
            x = prep(image)
            with ac():
                d = dino.forward_patches(x)
                dn = F.normalize(d.float(), dim=-1)
            best_d = torch.einsum("npd,bd->npb", dn, bank_d).max(dim=2).values
            return x, d, float(sc.image_score(1.0 - best_d, SCREEN_STAT)[0])

        def dino_flagged(image):
            x, d, _ = dino_gate(image)
            with ac():
                pooled, tokens = visual(x, [CLIP_LAYER])
                g = fuse(pooled, d)
                c = F.normalize(tokens[0][:, 1:, :].float(), dim=-1)
            best, idx = torch.einsum("npd,bd->npb", c, bank).max(dim=2)
            return head(g, c, best, idx)

        # ---- checks (fail closed) ----
        g_q, c_q, d_q = full_features(query)
        both_g, both_c = torch.cat([g_s, g_q]), torch.cat([p_s, c_q])
        si = torch.arange(len(s_lab), device=device)
        qi = torch.tensor([len(s_lab)], device=device)
        w_all = ab.region_weights(ra.patch_anomaly(both_c, bank), tau, soft=True)
        soft_all, rr_all = ab.weighted_pool(both_c, w_all), p13.region_residual(both_c, bank, w_all)
        reference = (rp._zscore(rp.simpleshot(both_g[si], s_lab, both_g[qi], n))
                     + rp._zscore(rp.simpleshot(soft_all[si], s_lab, soft_all[qi], n))
                     + rp._zscore(rp.simpleshot(rr_all[si], s_lab, rr_all[qi], n)))
        checks = {"full_head_vs_evaluation_code": float((full_path(query) - reference).abs().max())}
        probe = [supports[0], supports[-1], bank_images[0]]
        if block == "gate_l18_fp16":
            ok = True
            for im in probe:
                with ac():
                    x = prep(im)
                    pooled_full, tokens_full = visual(x, [CLIP_LAYER])
                    h = split.to_layer(x)
                    ok &= torch.equal(h.permute(1, 0, 2), tokens_full[0]) and torch.equal(split.rest(h), pooled_full)
            checks["split_bitwise_equal_full_forward"] = bool(ok)
            checks["flagged_head_vs_evaluation_code"] = float((l18_flagged(query) - reference).abs().max())
            s_eval = float(sc.image_score(ra.patch_anomaly(c_q, bank), SCREEN_STAT)[0])
            checks["screen_score_equal_p15_formula"] = l18_gate(query)[-1] == s_eval
        else:
            checks["flagged_head_vs_evaluation_code"] = float((dino_flagged(query) - reference).abs().max())
            s_eval = float(sc.image_score(ra.patch_anomaly(F.normalize(d_q.float(), dim=-1), bank_d), SCREEN_STAT)[0])
            checks["screen_score_equal_p15_formula"] = dino_gate(query)[-1] == s_eval
        bad = [k for k, v in checks.items() if (v is False) or (isinstance(v, float) and v > 1e-4)]
        if bad:
            raise SystemExit(f"{block}: checks failed {bad}: {checks}")

        # ---- timing: end to end from the decoded image (as r3), then the components ----
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        e2e = {"full": _time(lambda: full_path(query), device)}
        if block == "gate_l18_fp16":
            e2e["pass"] = _time(lambda: l18_gate(query), device)
            e2e["flagged"] = _time(lambda: l18_flagged(query), device)
        else:
            e2e["pass"] = _time(lambda: dino_gate(query), device)
            e2e["flagged"] = _time(lambda: dino_flagged(query), device)
        x = prep(query)
        with ac():
            h = split.to_layer(x)
            d = dino.forward_patches(x)
        dn = F.normalize(d.float(), dim=-1)
        components = {
            "full_forward": cp.timed(lambda: cp.autocast16(lambda t: (visual(t, [CLIP_LAYER]),
                                                                       dino.forward_patches(t)))(x), device),
            "clip_to_layer18": cp.timed(lambda: cp.autocast16(split.to_layer)(x), device),
            "clip_layers19_24": cp.timed(lambda: cp.autocast16(split.rest)(h), device),
            "dino_forward": cp.timed(lambda: cp.autocast16(dino.forward_patches)(x), device),
            "knn_clip_bank": cp.timed(lambda: torch.einsum("npd,bd->npb", c_q, bank).max(dim=2), device),
            "knn_dino_bank": cp.timed(lambda: torch.einsum("npd,bd->npb", dn, bank_d).max(dim=2), device),
            "preprocess_to_device": cp.timed(lambda: prep(query), device),
        }
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    return {"gates": block, "precision": "fp16", "checks": checks, "offline_ms": offline, "e2e": e2e,
            "components": components, "peak_allocated_bytes": peak,
            "storage": {"clip_bank_fp32_mb": bank.numel() * 4 / 2 ** 20, "dino_bank_fp32_mb": bank_d.numel() * 4 / 2 ** 20}}


def run_worker(args) -> int:
    import numpy  # noqa: F401  (before torch: Windows OpenMP clash otherwise)
    import torch
    device = torch.device(args.device)
    t0 = time.perf_counter()
    b = args.worker
    if b in ANCHORS:
        res = cp.worker_ours(ANCHORS[b], args, device)
    elif b.startswith("l14_l18"):
        res = worker_l14_l18(b, args, device)
    elif b in PRECAST_BASE:
        res = worker_precast(b, args, device)
    else:
        res = worker_gate(b, args, device)
    res["block_wall_s"] = time.perf_counter() - t0
    res["torch"] = torch.__version__
    (Path(args.out) / f"pass{args.pass_index}_{b}.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                                                     encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- orchestrator

def headline(block: str, r: dict) -> dict:
    if "gates" in r:
        return {**{f"e2e_{k}_ms": v["median_ms"] for k, v in r["e2e"].items()},
                **{f"{k}_ms": v["median_ms"] for k, v in r["components"].items()}}
    head = r["online"][cp.AR_HEAD if "l14" in block else "dual_branch_ARR"]
    return {"prep_s": r["per_task_preparation_s"], "model_forward_ms": r["model_forward"]["median_ms"],
            "end_to_end_ms": head["end_to_end_from_pil"]["median_ms"],
            "head_ms": head["head_only_given_features"]["median_ms"], "gflops": r["gflops_per_image_backbone"],
            "peak_mb": head["peak_allocated_bytes"] / 2 ** 20 if head["peak_allocated_bytes"] else None}


def run_orchestrator(args) -> int:
    out = OUT_ROOT / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --tag")
    only = [b for b in args.only.split(",") if b]
    unknown = sorted(set(only) - set(BLOCKS))
    if unknown:
        raise SystemExit(f"unknown blocks {unknown}")
    blocks = [b for b in BLOCKS if not only or b in only]
    out.mkdir(parents=True)
    env = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
           "telemetry_start": cp.telemetry(), **cp.machine_info()}
    log = []
    for pass_index, order in ((1, blocks), (2, list(reversed(blocks))))[:args.passes]:
        for block in order:
            before = cp.telemetry()
            cmd = [sys.executable, "-u", "-m", "experiments.p36.early_exit_profile", "--worker", block,
                   "--pass-index", str(pass_index), "--out", str(out), "--device", args.device,
                   "--mvtec-root", args.mvtec_root]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=args.block_timeout)
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(cmd, "timeout", "", f"timed out after {exc.timeout} s")
            entry = {"pass": pass_index, "block": block, "returncode": proc.returncode,
                     "wall_s": time.perf_counter() - t0, "telemetry_before": before, "telemetry_after": cp.telemetry()}
            if proc.returncode != 0:
                entry["stderr_tail"] = proc.stderr[-3000:]
            log.append(entry)
            print(f"[p36] pass {pass_index} {block}: rc={proc.returncode} {entry['wall_s']:.0f}s", flush=True)
            (out / "run_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    summary = {}
    for block in blocks:
        summary[block] = {p: headline(block, json.loads((out / f"pass{p}_{block}.json").read_text(encoding="utf-8")))
                          for p in (1, 2) if (out / f"pass{p}_{block}.json").exists()}
    (out / "p36_results.json").write_text(json.dumps({"schema_version": "p36_v1", "tag": args.tag,
                                                      "environment": env, "setting": {"product": cp.PRODUCT,
                                                      "K": cp.K_SHOT, "seed": cp.SEED, "bank": 20, "batch": 1},
                                                      "summary": summary, "run_log": log}, indent=1, default=str)
                                          + "\n", encoding="utf-8")
    lines = [f"# P36 early exit / gates / pre-cast — {args.tag}", "", "Values: pass 1 / pass 2 (median ms).", ""]
    for block, pp in summary.items():
        keys = sorted({k for h in pp.values() for k in h})
        cells = "; ".join(f"{k} " + " / ".join("—" if pp.get(p, {}).get(k) is None else f"{pp[p][k]:.2f}"
                                               for p in (1, 2)) for k in keys)
        lines.append(f"- **{block}**: {cells}")
    (out / "p36_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag")
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="comma-separated subset of blocks")
    ap.add_argument("--block-timeout", type=int, default=3600)
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--worker", choices=BLOCKS)
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.worker:
        return run_worker(args)
    if not args.tag:
        raise SystemExit("--tag is required")
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
