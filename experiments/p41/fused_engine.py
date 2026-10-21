#!/usr/bin/env python3
"""P41 (report only): the whole method as one TensorRT engine, the normal reference as runtime input
(docs/P41_FUSED_ENGINE_PROTOCOL.md).

Local (export + checks):
    python -m experiments.p41.fused_engine --export --mvtec-root <root> --out D:\\intergration\\p35_trt
    python -m experiments.p41.fused_engine --write-draws                      # bank draws of the formal set (for the device)
On the device (PYTHONPATH = the tensorrt link directory):
    python -m experiments.p41.fused_engine --time --tag <tag> --mvtec-root <root> --engine <file> --pass-index 1
    python -m experiments.p41.fused_engine --extract --tag <tag> --mvtec-root <root> --engine <file>
Local scoring:
    python -m experiments.p41.fused_engine --score --tag <tag>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.p26 import cost_profile as cp  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p41"
DRAWS = OUT / "bank_draws.json"
IN_NAMES = ["input", "bank", "tau", "mu_g", "proto_g", "mu_a", "proto_a", "mu_r", "proto_r"]
OUT_NAMES = ["scores", "screen", "g", "h_a", "h_r", "tokens"]
MIN_REGION = 6
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- the fused module

def explicit_dino(model, res: int = 336):
    """P35 round-2 fix applied to the main configuration's DINOv2-S: constant position table for the fixed input size
    and explicit attention with a constant scale (TensorRT 10.3 mis-evaluates the shape-derived SDPA scale)."""
    table = model.interpolate_pos_encoding(torch.zeros(1, 1 + (res // model.patch_size) ** 2, model.embed_dim,
                                                       device=model.pos_embed.device), res, res).detach()
    model.interpolate_pos_encoding = lambda x, w, h: table.to(x.dtype)

    def forward(self, x, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
    for blk in model.blocks:
        blk.attn.forward = types.MethodType(forward, blk.attn)
    return model


def zscore(s):
    m = s.mean(dim=1, keepdim=True)
    std = ((s - m) ** 2).mean(dim=1, keepdim=True).sqrt()
    return torch.where(std > 0, (s - m) / std.clamp_min(1e-12), torch.zeros_like(s))


def head(c, bank, tau, mu, protos):
    """The dual-branch head on one image: c [576, d] normalised layer-18 tokens; bank [B, d]; tau [1];
    mu / protos: dicts of the three spaces. Returns (h_a, h_r, screen) and, given g, the scores (see Fused)."""
    sims = c @ bank.t()
    best, idx = sims.max(dim=1)
    a = 1.0 - best
    w = (a - tau).clamp_min(0)
    top = a.topk(MIN_REGION).indices
    fb = torch.zeros_like(a).scatter(0, top, 1.0)
    w = torch.where((w > 0).sum() < MIN_REGION, fb, w)
    h_a = (w @ c) / w.sum()
    h_r = (w @ (c - bank[idx])) / w.sum()
    screen = a.topk(MIN_REGION).values.mean().reshape(1)
    return h_a[None], h_r[None], screen


class Fused(torch.nn.Module):
    def __init__(self, factory):
        super().__init__()
        from experiments.p10.extract_dev_features import CLIP_LAYER
        self.visual, self.dino = factory.clip_model.visual, factory.dino_extractor
        self.layer = CLIP_LAYER

    def forward(self, x, bank, tau, mu_g, proto_g, mu_a, proto_a, mu_r, proto_r):
        pooled, toks = self.visual(x, [self.layer])
        d = self.dino.forward_patches(x)
        g = F.normalize(torch.cat([F.normalize(pooled.float(), dim=-1), F.normalize(d.mean(dim=1).float(), dim=-1)], -1), dim=-1)
        c = F.normalize(toks[0][:, 1:, :].float(), dim=-1)
        h_a, h_r, screen = head(c[0], bank, tau, None, None)
        s = (zscore(F.normalize(g - mu_g, dim=-1) @ proto_g.t()) + zscore(F.normalize(h_a - mu_a, dim=-1) @ proto_a.t())
             + zscore(F.normalize(h_r - mu_r, dim=-1) @ proto_r.t()))
        return s, screen, g, h_a, h_r, c


def proto_space(h, labels, n):
    """SimpleShot state (== experiments.p14.resource_profile.proto_space)."""
    from experiments.p10 import run_pilot as rp
    mu = h.mean(0, keepdim=True)
    return mu, rp._class_means(F.normalize(h - mu, dim=-1), labels, n)


def gpu_prep(image, device):
    """Decoded PIL image -> CLIP-normalised 336 tensor on the GPU: 8-bit upload, bicubic anti-aliased resize, 8-bit
    rounding (as PIL), normalisation."""
    x = torch.from_numpy(np.asarray(image.convert("RGB"))).to(device, non_blocking=True)
    x = x.permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(336, 336), mode="bicubic", antialias=True, align_corners=False).round().clamp(0, 255) / 255.0
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def build_factory(device):
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    f = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    explicit_dino(f.dino_extractor.model)
    return f


# --------------------------------------------------------------------------- local: export with checks

def task_state(run, supports, s_lab, n, bank_images, prep):
    """New task with any callable run(x, bank, tau, protos) -> (scores, screen, g, h_a, h_r, tokens)."""
    from experiments.p11 import formal_ab as fa
    dummy = dummy_protos(next(iter([None])), 2)
    zero_bank = torch.zeros(11520, 1024, device=DEVICE[0])
    toks = torch.cat([run(prep(im), zero_bank, torch.zeros(1, device=DEVICE[0]), dummy)[5] for im in bank_images])
    bank = toks.reshape(-1, toks.shape[-1]).contiguous()
    tau = torch.tensor([fa.tau99(toks)], device=DEVICE[0], dtype=torch.float32)
    feats = [run(prep(im), bank, tau, dummy) for im in supports]
    gs, ha, hr = (torch.cat([f[i] for f in feats]) for i in (2, 3, 4))
    spaces = {k: proto_space(h, s_lab, n) for k, h in (("g", gs), ("a", ha), ("r", hr))}
    protos = {"mu_g": spaces["g"][0], "proto_g": spaces["g"][1], "mu_a": spaces["a"][0], "proto_a": spaces["a"][1],
              "mu_r": spaces["r"][0], "proto_r": spaces["r"][1]}
    return bank, tau, protos, (gs, ha, hr, toks)


DEVICE = [None]


def dummy_protos(_, n):
    d = DEVICE[0]
    return {"mu_g": torch.zeros(1, 1152, device=d), "proto_g": F.normalize(torch.ones(n, 1152, device=d), dim=-1),
            "mu_a": torch.zeros(1, 1024, device=d), "proto_a": F.normalize(torch.ones(n, 1024, device=d), dim=-1),
            "mu_r": torch.zeros(1, 1024, device=d), "proto_r": F.normalize(torch.ones(n, 1024, device=d), dim=-1)}


def export(args) -> int:
    import onnxruntime as ort
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p10 import run_pilot as rp
    from experiments.p10 import run_pilot_a as ra
    from experiments.p11 import run_pilot_ab as ab
    from experiments.p13 import pilot as p13
    from experiments.p15 import screening as sc
    out = Path(args.out)
    onnx_path = out / "main_fused.onnx"
    if onnx_path.exists():
        raise SystemExit(f"{onnx_path} exists")
    device = torch.device("cpu")
    DEVICE[0] = device
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    with torch.no_grad():
        model = Fused(build_factory(device)).eval()
        run = lambda x, b, t, p: model(x, b, t, **p)  # noqa: E731
        prep = lambda im: models.deterministic_eval_tensor(im)  # noqa: E731
        bank, tau, protos, (gs, ha, hr, toks) = task_state(run, supports, s_lab, n, bank_images, prep)
        q = prep(supports[0])
        s, screen, g, h_a, h_r, c = model(q, bank, tau, **protos)
        # check 1: the head against the evaluation code on the module's own features
        pall = torch.cat([torch.cat([run(prep(im), bank, tau, dummy_protos(None, 2))[5] for im in supports]), c])
        si, qi = torch.arange(len(s_lab)), torch.tensor([len(s_lab)])
        gall = torch.cat([gs, g])
        w = ab.region_weights(ra.patch_anomaly(pall, bank), float(tau), soft=True)
        ref = (rp._zscore(rp.simpleshot(gall[si], s_lab, gall[qi], n))
               + rp._zscore(rp.simpleshot(ab.weighted_pool(pall, w)[si], s_lab, ab.weighted_pool(pall, w)[qi], n))
               + rp._zscore(rp.simpleshot(p13.region_residual(pall, bank, w)[si], s_lab, p13.region_residual(pall, bank, w)[qi], n)))
        head_diff = float((s - ref).abs().max())
        screen_ref = float(sc.image_score(ra.patch_anomaly(c, bank), "top1pct")[0])
        if head_diff > 1e-4 or abs(float(screen) - screen_ref) > 1e-6:
            raise SystemExit(f"fused module differs from the evaluation code: scores {head_diff}, screen {float(screen)} vs {screen_ref}")
        inputs = (q, bank, tau, protos["mu_g"], protos["proto_g"], protos["mu_a"], protos["proto_a"], protos["mu_r"], protos["proto_r"])
        torch.onnx.export(model, inputs, str(onnx_path), input_names=IN_NAMES, output_names=OUT_NAMES, opset_version=17,
                          do_constant_folding=True, dynamo=False,
                          dynamic_axes={"proto_g": {0: "C"}, "proto_a": {0: "C"}, "proto_r": {0: "C"}, "scores": {1: "C"}})
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in zip(IN_NAMES, inputs)}
    o = dict(zip(OUT_NAMES, sess.run(None, feed)))
    ort_diff = {"scores": float(np.abs(o["scores"] - s.numpy()).max()), "screen": float(abs(o["screen"][0] - float(screen))),
                "h_a": float(np.abs(o["h_a"] - h_a.numpy()).max()), "h_r": float(np.abs(o["h_r"] - h_r.numpy()).max())}
    for k, v in zip(IN_NAMES, inputs):
        v.numpy().astype(np.float32).tofile(out / f"main_fused_in_{k}.bin")
    np.savez(out / "main_fused_ref.npz", **{"scores": s.numpy(), "screen": screen.numpy(), "g": g.numpy(), "h_a": h_a.numpy(),
                                            "h_r": h_r.numpy()})
    shapes = {k: list(v.shape) for k, v in zip(IN_NAMES, inputs)}
    summary = {"head_vs_evaluation_code": head_diff, "screen_vs_p15": abs(float(screen) - screen_ref), "ort_vs_torch": ort_diff,
               "input_shapes": shapes, "onnx_mb": onnx_path.stat().st_size / 2 ** 20}
    (out / "p41_export_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def write_draws() -> int:
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p10 import run_pilot_a as ra
    from experiments.revision import revision_common as rc
    draws = {}
    for product in rc.PRODUCTS:
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        draws[product] = {str(s): list(nf["draws"][str(s)][:ra.M_PRIMARY]) for s in rc.SEEDS}
    OUT.mkdir(parents=True, exist_ok=True)
    if DRAWS.exists():
        raise SystemExit(f"{DRAWS} exists")
    DRAWS.write_text(json.dumps(draws, indent=1), encoding="utf-8")
    print(f"{DRAWS}: {len(draws)} products")
    return 0


# --------------------------------------------------------------------------- device: engine wrapper

class FusedEngine:
    """The fused TensorRT engine; protos have a dynamic class dimension."""

    def __init__(self, path, device):
        import tensorrt as trt
        self.trt, self.device = trt, device
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.outs, self.n = {}, None

    def __call__(self, x, bank, tau, protos):
        feed = {"input": x, "bank": bank, "tau": tau, **protos}
        n = protos["proto_g"].shape[0]
        for k in IN_NAMES:
            t = feed[k].float().contiguous()
            feed[k] = t
            self.context.set_input_shape(k, tuple(t.shape))
            self.context.set_tensor_address(k, t.data_ptr())
        if n != self.n:
            self.outs = {}
            for k in OUT_NAMES:
                self.outs[k] = torch.empty(tuple(self.context.get_tensor_shape(k)), dtype=torch.float32, device=self.device)
            self.n = n
        for k in OUT_NAMES:
            self.context.set_tensor_address(k, self.outs[k].data_ptr())
        self._keep = feed
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("engine execution failed")
        o = self.outs
        return o["scores"], o["screen"], o["g"], o["h_a"], o["h_r"], o["tokens"]


def device_time(args) -> int:
    """Timing on the device (protocol §3): new task through the engine, then end to end from the decoded image."""
    from experiments.revision.c3_resource_profile import _time
    device = torch.device("cuda")
    DEVICE[0] = device
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"pass{args.pass_index}_fused.json"
    if f.exists():
        raise SystemExit(f"{f} exists")
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    eng = FusedEngine(args.engine, device)
    sync = torch.cuda.synchronize
    run = lambda x, b, t, p: tuple(v.clone() for v in eng(x, b, t, p))  # noqa: E731
    prep = lambda im: gpu_prep(im, device)  # noqa: E731
    with torch.no_grad():
        for _ in range(5):
            run(prep(supports[0]), torch.zeros(11520, 1024, device=device), torch.zeros(1, device=device), dummy_protos(None, n))
        sync()
        t0 = time.perf_counter()
        bank, tau, protos, _ = task_state(run, supports, s_lab, n, bank_images, prep)
        sync()
        new_task_s = time.perf_counter() - t0
        q = supports[0]

        def deployed(image):
            s, screen = eng(prep(image), bank, tau, protos)[:2]
            return int(s.argmax()), float(screen)
        res = {"new_task_s": new_task_s, "e2e_from_pil": _time(lambda: deployed(q), device),
               "engine_only": cp.timed(lambda: eng(prep(q), bank, tau, protos), device),
               "gpu_preprocess": cp.timed(lambda: prep(q), device), "classes": n, "pass": args.pass_index,
               "engine": str(args.engine), "torch": torch.__version__}
    f.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: (v["median_ms"] if isinstance(v, dict) and "median_ms" in v else v) for k, v in res.items()}))
    return 0


def device_extract(args) -> int:
    """Accuracy extraction (protocol §4): per product and bank seed, engine bank, LOO tau, then g / h_a / h_r of every
    support-pool and query image (GPU preprocessing + engine)."""
    from experiments.p11 import formal_ab as fa
    from experiments.p32 import device_extract as dx
    device = torch.device("cuda")
    DEVICE[0] = device
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    draws = json.loads(DRAWS.read_text(encoding="utf-8"))
    groups = dx.inputs(Path(args.mvtec_root))
    eng = FusedEngine(args.engine, device)
    dummy = dummy_protos(None, 2)
    zero_bank, zero_tau = torch.zeros(11520, 1024, device=device), torch.zeros(1, device=device)
    with torch.no_grad():
        for product in [x for x in args.products.split(",") if x] or list(groups):
            path = out / f"fused_{product}.pt"
            if path.exists():
                raise SystemExit(f"{path} exists")
            g = groups[product]
            normal_paths = dict(g["normals"])
            t0 = time.perf_counter()
            res = {"support_rel_paths": [i for i, _ in g["support"]], "query_rgb_sha256": [i for i, _ in g["queries"]],
                   "seeds": {}}
            tok_cache = {}
            for seed, names in draws[product].items():
                for nm in names:
                    if nm not in tok_cache:
                        tok_cache[nm] = eng(gpu_prep(dx.open_rgb(normal_paths[nm]), device), zero_bank, zero_tau, dummy)[5].clone()
                toks = torch.cat([tok_cache[nm] for nm in names])
                bank = toks.reshape(-1, 1024).contiguous()
                tau = torch.tensor([fa.tau99(toks)], device=device)
                per = {}
                for part in ("support", "queries"):
                    gl, al, rl = [], [], []
                    for _, p in g[part]:
                        _, _, gg, ha, hr, _ = eng(gpu_prep(dx.open_rgb(p), device), bank, tau, dummy)
                        gl.append(gg.clone().cpu()); al.append(ha.clone().cpu()); rl.append(hr.clone().cpu())
                    per[part] = {"g": torch.cat(gl), "h_a": torch.cat(al), "h_r": torch.cat(rl)}
                res["seeds"][seed] = {"tau": float(tau), **per}
            torch.save(res, path)
            print(f"[p41] {product}: {time.perf_counter() - t0:.0f} s", flush=True)
    print("[p41] extraction done", flush=True)
    return 0


# --------------------------------------------------------------------------- local scoring (protocol §4)

def score(args) -> int:
    from experiments.p6.p6b_stage3_common import load_query_contract
    from experiments.p10 import run_pilot as rp
    from experiments.p10.formal_evaluate import MANIFEST_DIR
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.p12.threshold_controls import exact_equal
    from experiments.p14.final_formal import ARR, c1_descriptors
    from experiments.p10 import run_pilot_a as ra
    from experiments.revision import revision_common as rc
    from experiments.revision.c1_paired_bootstrap import contrast
    src = OUT / args.tag
    out = src / "p41_scores.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    device = torch.device("cuda")
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows, sink, diffs = [], [], {}
    with torch.no_grad():
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            fu = torch.load(src / f"fused_{product}.pt", map_location="cpu", weights_only=False)
            dev_paths, shas = list(dev["rel_paths"]), list(qf["rgb_sha256"])
            if fu["support_rel_paths"] != dev_paths or fu["query_rgb_sha256"] != shas:
                raise SystemExit(f"{product}: image order differs")
            q_true = [truth[(product, s)] for s in shas]
            path_index = {p: i for i, p in enumerate(dev_paths)}
            n_pool = len(dev_paths)
            qi = torch.arange(n_pool, n_pool + len(shas), device=device)
            g_c = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
            clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
            normals = nf["clip18_patches"].to(device).float()
            name_index = {x: i for i, x in enumerate(nf["names"])}
            for seed in rc.SEEDS:
                draw = nf["draws"][str(seed)][:ra.M_PRIMARY]
                bank_imgs = normals[torch.as_tensor([name_index[x] for x in draw], device=device)]
                _, soft, rr = c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))
                fs = fu["seeds"][str(seed)]
                g_f = torch.cat([fs["support"]["g"], fs["queries"]["g"]]).to(device)
                a_f = torch.cat([fs["support"]["h_a"], fs["queries"]["h_a"]]).to(device)
                r_f = torch.cat([fs["support"]["h_r"], fs["queries"]["h_r"]]).to(device)
                diffs.setdefault(product, {})[str(seed)] = {k: float(F.cosine_similarity(x, y).mean()) for k, x, y in
                                                            (("g", g_f, g_c), ("h_a", a_f, soft), ("h_r", r_f, rr))}
                for k_shot in rc.K_VALUES:
                    manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                    order = list(manifest["class_order"])
                    s_idx, s_lab = [], []
                    for c, cls in enumerate(order):
                        for item in manifest["classes"][cls]["items"]:
                            s_idx.append(path_index[item["rel_path"]])
                            s_lab.append(c)
                    si = torch.as_tensor(s_idx, device=device)
                    n = len(order)
                    sc = {}
                    for src_name, (gg, aa, rr2) in (("cache", (g_c, soft, rr)), ("fused", (g_f, a_f, r_f))):
                        sc[f"{ARR}|{src_name}"] = (rp._zscore(rp.simpleshot(gg[si], s_lab, gg[qi], n))
                                                   + rp._zscore(rp.simpleshot(aa[si], s_lab, aa[qi], n))
                                                   + rp._zscore(rp.simpleshot(rr2[si], s_lab, rr2[qi], n)))
                    rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed}, sink, shas)
            print(f"[p41] {product} scored", flush=True)
    gate = exact_equal([{**r, "method": ARR} for r in rows if r["method"] == f"{ARR}|cache"],
                       rc.load_json(PROJECT_ROOT / "ckpt" / "p14_formal" / "mvtec" / "p14_mvtec_results.json")["raw_metrics"],
                       (ARR,), ("product", "K", "seed"))
    if gate >= 1e-9:
        raise SystemExit(f"cache path does not reproduce P14: {gate}")
    grid = dict(rc.metric_grid_from_rows(rows))
    c = contrast(grid, f"{ARR}|cache", "balanced_accuracy", target=f"{ARR}|fused")
    by = {(u["method"], u["product"], u["K"], u["seed"]): u["pred"] for u in sink}
    same = total = 0
    for (m, p, k, s), pred in by.items():
        if m == f"{ARR}|fused":
            ref = by[(f"{ARR}|cache", p, k, s)]
            same += sum(x == y for x, y in zip(pred, ref))
            total += len(pred)
    d, ci = 100 * c["Dbar"], [100 * x for x in c["Dbar_ci95"]]
    levels = {m: {str(k): 100 * rc.product_macro_summary(grid, m)[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES}
              for m in (f"{ARR}|cache", f"{ARR}|fused")}
    result = {"schema_version": "p41_scores_v1", "protocol": "docs/P41_FUSED_ENGINE_PROTOCOL.md#4", "gate_cache_vs_P14": gate,
              "levels_ba_pct": levels, "fused_minus_cache": {"per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()},
                                                            "Dbar_pp": d, "ci95_pp": ci, "agreement": same / total,
                                                            "decision": "keeps accuracy" if abs(d) < 0.5 and ci[0] <= 0 <= ci[1]
                                                            else "report engine accuracy"},
              "descriptor_cosine": diffs, "provenance": rc.git_provenance(), "raw_metrics": rows}
    rc.write_json(out, result)
    print(json.dumps({k: result[k] for k in ("gate_cache_vs_P14", "levels_ba_pct", "fused_minus_cache")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--write-draws", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--mvtec-root")
    ap.add_argument("--out")
    ap.add_argument("--tag")
    ap.add_argument("--engine")
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--products", default="")
    a = ap.parse_args(argv)
    if a.export:
        return export(a)
    if a.write_draws:
        return write_draws()
    if a.time:
        return device_time(a)
    if a.extract:
        return device_extract(a)
    if a.score:
        return score(a)
    raise SystemExit("choose --export, --write-draws, --time, --extract or --score")


if __name__ == "__main__":
    sys.exit(main())
