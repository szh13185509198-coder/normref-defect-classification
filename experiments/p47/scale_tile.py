#!/usr/bin/env python3
"""P47 (report only): scaling and tiling of the deployed method (docs/P47_SCALING_AND_TILING_PROTOCOL.md).

On the device (PYTHONPATH = the tensorrt link directory):
    python -m experiments.p47.scale_tile --export dyn|g2|g3 --mvtec-root <root> --out <dir>
    python -m experiments.p47.scale_tile --check dyn|g2|g3 --engine <file> --out <dir> [--r8 <file>]
    python -m experiments.p47.scale_tile --time --tag <tag> --mvtec-root <root> --engines <dir> --pass-index 1|2
Local:
    python -m experiments.p47.scale_tile --tile-extract 1|2|3 --mvtec-root <root> --tag <tag>
    python -m experiments.p47.scale_tile --tile-score --tag <tag> --grid 1|2|3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402,F401  (deliberately before torch, as P41)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.p26 import cost_profile as cp  # noqa: E402
from experiments.p41 import fused_engine as fe  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p47"
P = 576                          # patches per 336 view
D = 1024
M_SWEEP = (20, 50, 100, 200)
C_SWEEP = (2, 4, 8, 16)
S_SWEEP = (1, 10, 40)
B_MIN, B_OPT, B_MAX = P * 5, P * 20, P * 200
ROW_CHUNKS = {1: 1, 2: 1, 3: 4}   # amendment A3: single allocations of ~1 GiB fail on the device
TILE_M = {2: 20, 3: 10}   # device bank size per grid (amendment A2: g = 3 does not fit with M = 20)
TILED_IN = ["full", "tiles", "bank", "tau", "mu_g", "proto_g", "mu_a", "proto_a", "mu_r", "proto_r"]


# --------------------------------------------------------------------------- tiles

def boxes(w: int, h: int, grid: int):
    """g x g non-overlapping regions, integer boundaries round(i*W/g) (protocol §2), row-major."""
    xs = [round(i * w / grid) for i in range(grid + 1)]
    ys = [round(j * h / grid) for j in range(grid + 1)]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(grid) for i in range(grid)]


def gpu_tiles(image, grid: int, device):
    """Decoded PIL image -> [g*g, 3, 336, 336]: each region preprocessed exactly as fe.gpu_prep does a whole image."""
    x = torch.from_numpy(np.asarray(image.convert("RGB"))).to(device, non_blocking=True).permute(2, 0, 1)[None].float()
    h, w = x.shape[-2:]
    mean = torch.tensor(fe.CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(fe.CLIP_STD, device=device).view(1, 3, 1, 1)
    out = []
    for x0, y0, x1, y1 in boxes(w, h, grid):
        t = F.interpolate(x[..., y0:y1, x0:x1], size=(336, 336), mode="bicubic", antialias=True, align_corners=False)
        out.append((t.round().clamp(0, 255) / 255.0 - mean) / std)
    return torch.cat(out)


def pil_tiles(image, grid: int):
    """Local accuracy path: each region through the locked deterministic 336 view."""
    from experiments.p6 import p6b_stage2_models as models
    w, h = image.size
    return torch.cat([models.deterministic_eval_tensor(image.crop(b)) for b in boxes(w, h, grid)])


# --------------------------------------------------------------------------- modules

def tiled_head(c, bank, tau, k_screen: int, row_chunks: int = 1):
    """fe.head on the union of tile patches; the screening statistic is the top ceil(1 %) mean (protocol §2).
    row_chunks > 1 computes the nearest normal patch for blocks of rows (amendment A3: memory only, same result)."""
    if row_chunks == 1:
        best, idx = (c @ bank.t()).max(dim=1)
    else:
        parts = [(blk @ bank.t()).max(dim=1) for blk in c.chunk(row_chunks)]
        best, idx = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    a = 1.0 - best
    w = (a - tau).clamp_min(0)
    top = a.topk(fe.MIN_REGION).indices
    fb = torch.zeros_like(a).scatter(0, top, 1.0)
    w = torch.where((w > 0).sum() < fe.MIN_REGION, fb, w)
    h_a = (w @ c) / w.sum()
    h_r = (w @ (c - bank[idx])) / w.sum()
    screen = a.topk(k_screen).values.mean().reshape(1)
    return h_a[None], h_r[None], screen


class FusedTiled(torch.nn.Module):
    """P41's fused module with tiled evidence: g from the whole image, layer-18 tokens of every tile (blocks 1-18)."""

    def __init__(self, factory, grid: int):
        super().__init__()
        from experiments.p10.extract_dev_features import CLIP_LAYER
        from experiments.p36.early_exit_profile import ClipSplit
        self.visual, self.dino = factory.clip_model.visual, factory.dino_extractor
        self.layer, self.grid = CLIP_LAYER, grid
        self.split = ClipSplit(self.visual, CLIP_LAYER)
        self.k_screen = math.ceil(0.01 * grid * grid * P)

    def tile_tokens(self, tiles):
        h = self.split.to_layer(tiles).permute(1, 0, 2)
        return F.normalize(h[:, 1:, :].float(), dim=-1).reshape(-1, h.shape[-1])

    def forward(self, full, tiles, bank, tau, mu_g, proto_g, mu_a, proto_a, mu_r, proto_r):
        pooled, _ = self.visual(full, [self.layer])
        d = self.dino.forward_patches(full)
        g = F.normalize(torch.cat([F.normalize(pooled.float(), dim=-1), F.normalize(d.mean(dim=1).float(), dim=-1)], -1), dim=-1)
        c = self.tile_tokens(tiles)
        h_a, h_r, screen = tiled_head(c, bank, tau, self.k_screen, ROW_CHUNKS[self.grid])
        s = (fe.zscore(F.normalize(g - mu_g, dim=-1) @ proto_g.t()) + fe.zscore(F.normalize(h_a - mu_a, dim=-1) @ proto_a.t())
             + fe.zscore(F.normalize(h_r - mu_r, dim=-1) @ proto_r.t()))
        return s, screen, g, h_a, h_r, c


def loo_tau(toks, rows: int = 1024) -> float:
    """fa.tau99 with the anomaly of each left-out image computed in blocks of patch rows (memory only)."""
    m = toks.shape[0]
    vals = []
    for j in range(m):
        others = torch.cat([toks[:j], toks[j + 1:]]).reshape(-1, toks.shape[-1])
        q = toks[j]
        vals.append(torch.cat([1.0 - (blk @ others.t()).max(dim=1).values for blk in q.split(rows)]))
    return quantile_linear(torch.cat(vals).double(), 0.99)


def quantile_linear(v, q: float) -> float:
    """torch.quantile's default (linear) definition via a sort; torch.quantile refuses very large inputs."""
    s = v.flatten().sort().values
    pos = q * (s.numel() - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, s.numel() - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


# --------------------------------------------------------------------------- generic TensorRT wrapper

class Engine:
    def __init__(self, path, device):
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.device = device
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.key, self.outs = None, {}

    def __call__(self, feed: dict):
        key = []
        keep = {}
        for k in self.inputs:
            t = feed[k].float().contiguous()
            keep[k] = t
            self.context.set_input_shape(k, tuple(t.shape))
            self.context.set_tensor_address(k, t.data_ptr())
            key.append(tuple(t.shape))
        if tuple(key) != self.key:
            self.outs = {k: torch.empty(tuple(self.context.get_tensor_shape(k)), dtype=torch.float32, device=self.device)
                         for k in self.outputs}
            self.key = tuple(key)
        for k in self.outputs:
            self.context.set_tensor_address(k, self.outs[k].data_ptr())
        self._keep = keep
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("engine execution failed")
        return self.outs


def feed_plain(x, bank, tau, protos):
    return {"input": x, "bank": bank, "tau": tau, **protos}


def feed_tiled(full, tiles, bank, tau, protos):
    return {"full": full, "tiles": tiles, "bank": bank, "tau": tau, **protos}


# --------------------------------------------------------------------------- device: export

def _protos_from(gs, ha, hr, s_lab, n):
    spaces = {k: fe.proto_space(h, s_lab, n) for k, h in (("g", gs), ("a", ha), ("r", hr))}
    return {"mu_g": spaces["g"][0], "proto_g": spaces["g"][1], "mu_a": spaces["a"][0], "proto_a": spaces["a"][1],
            "mu_r": spaces["r"][0], "proto_r": spaces["r"][1]}


def export(args) -> int:
    kind = args.export
    out = Path(args.out)
    onnx_path = out / f"p47_{kind}.onnx"
    if onnx_path.exists():
        raise SystemExit(f"{onnx_path} exists")
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    factory = fe.build_factory(device)
    checks = {}
    with torch.no_grad():
        if kind == "dyn":
            model = fe.Fused(factory).eval()
            run = lambda x, b, t, p: model(x, b, t, **p)  # noqa: E731
            bank, tau, protos, _ = fe.task_state(run, supports, s_lab, n, bank_images, lambda im: fe.gpu_prep(im, device))
            x = fe.gpu_prep(supports[0], device)
            inputs = (x, bank, tau, *protos.values())
            names = fe.IN_NAMES
            dyn = {"bank": {0: "B"}, "proto_g": {0: "C"}, "proto_a": {0: "C"}, "proto_r": {0: "C"}, "scores": {1: "C"}}
            ref = {"M20": [v.cpu() for v in model(*inputs)]}
            big = torch.cat([bank] * 3)   # M = 60, for the dynamic-shape check
            ref["M60"] = [v.cpu() for v in model(x, big, tau, *protos.values())]
            torch.save({"inputs": [v.cpu() for v in inputs], "ref": ref}, out / f"p47_{kind}_ref.pt")
        else:
            grid = int(kind[1])
            model = FusedTiled(factory, grid).eval()
            # check: tile tokens from blocks 1-18 equal the full forward's layer-18 tokens
            t = gpu_tiles(supports[0], grid, device)
            _, toks = model.visual(t, [model.layer])
            full_way = F.normalize(toks[0][:, 1:, :].float(), dim=-1).reshape(-1, D)
            checks["tile_tokens_vs_full_forward_max_abs"] = float((model.tile_tokens(t) - full_way).abs().max())
            if checks["tile_tokens_vs_full_forward_max_abs"] > 1e-6:
                raise SystemExit(f"split forward differs: {checks}")
            bank_images = bank_images[:TILE_M[grid]]
            zero = torch.zeros(grid * grid * P * TILE_M[grid], D, device=device)
            dummy = fe.dummy_protos(None, 2)
            toks = torch.stack([model(fe.gpu_prep(im, device), gpu_tiles(im, grid, device), zero,
                                      torch.zeros(1, device=device), *dummy.values())[5] for im in bank_images])
            bank = toks.reshape(-1, D).contiguous()
            tau = torch.tensor([loo_tau(toks)], device=device)
            feats = [model(fe.gpu_prep(im, device), gpu_tiles(im, grid, device), bank, tau, *dummy.values()) for im in supports]
            protos = _protos_from(*(torch.cat([f[i] for f in feats]) for i in (2, 3, 4)), s_lab, n)
            inputs = (fe.gpu_prep(supports[0], device), gpu_tiles(supports[0], grid, device), bank, tau, *protos.values())
            names = TILED_IN
            dyn = {"proto_g": {0: "C"}, "proto_a": {0: "C"}, "proto_r": {0: "C"}, "scores": {1: "C"}}
            torch.save({"inputs": [v.cpu() for v in inputs], "ref": {"M20": [v.cpu() for v in model(*inputs)]}},
                       out / f"p47_{kind}_ref.pt")
    checks["input_shapes"] = {k: list(v.shape) for k, v in zip(names, inputs)}
    (out / f"p47_{kind}_prep.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(json.dumps(checks))
    # the ONNX export follows in a separate process (--onnx-only, on the CPU as P41): one model in memory at a time
    return 0


def onnx_only(args) -> int:
    kind = args.onnx_only
    out = Path(args.out)
    onnx_path = out / f"p47_{kind}.onnx"
    if onnx_path.exists():
        raise SystemExit(f"{onnx_path} exists")
    cpu = torch.device("cpu")
    fe.DEVICE[0] = cpu
    inputs = tuple(torch.load(out / f"p47_{kind}_ref.pt", map_location="cpu", weights_only=False)["inputs"])
    if kind == "dyn":
        names = fe.IN_NAMES
        dyn = {"bank": {0: "B"}, "proto_g": {0: "C"}, "proto_a": {0: "C"}, "proto_r": {0: "C"}, "scores": {1: "C"}}
    else:
        names = TILED_IN
        dyn = {"proto_g": {0: "C"}, "proto_a": {0: "C"}, "proto_r": {0: "C"}, "scores": {1: "C"}}
    with torch.no_grad():
        factory = fe.build_factory(cpu)
        model = (fe.Fused(factory) if kind == "dyn" else FusedTiled(factory, int(kind[1]))).eval()
        torch.onnx.export(model, inputs, str(onnx_path), input_names=names, output_names=fe.OUT_NAMES, opset_version=17,
                          do_constant_folding=True, dynamo=False, dynamic_axes=dyn)
    info = {"onnx_mb": onnx_path.stat().st_size / 2 ** 20}
    (out / f"p47_{kind}_export.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))
    return 0


# --------------------------------------------------------------------------- device: parity

def check(args) -> int:
    kind = args.check
    out = Path(args.out)
    target = out / f"p47_{kind}_check.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    device = torch.device("cuda")
    saved = torch.load(out / f"p47_{kind}_ref.pt", map_location="cpu", weights_only=False)
    inputs = [v.to(device) for v in saved["inputs"]]
    eng = Engine(args.engine, device)
    names = fe.IN_NAMES if kind == "dyn" else TILED_IN
    res = {}

    def diff(o, ref):
        return {k: float((o[k].cpu() - r).abs().max()) for k, r in zip(fe.OUT_NAMES, ref) if k != "tokens"} | {
            "argmax_equal": int(o["scores"].argmax()) == int(ref[0].argmax())}
    with torch.no_grad():
        res["M20"] = diff(eng(dict(zip(names, inputs))), saved["ref"]["M20"])
        if kind == "dyn":
            big = dict(zip(names, inputs))
            big["bank"] = torch.cat([inputs[1]] * 3)
            res["M60"] = diff(eng(big), saved["ref"]["M60"])
            if args.r8:
                r8 = Engine(args.r8, device)
                a, b = eng(dict(zip(names, inputs))), {k: v.clone() for k, v in eng(dict(zip(names, inputs))).items()}
                o8 = r8(dict(zip(names, inputs)))
                res["dyn_vs_r8_scores_max_abs"] = float((b["scores"] - o8["scores"]).abs().max())
                res["dyn_ms"] = cp.timed(lambda: eng(dict(zip(names, inputs))), device)["median_ms"]
                res["r8_ms"] = cp.timed(lambda: r8(dict(zip(names, inputs))), device)["median_ms"]
                del a
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


# --------------------------------------------------------------------------- device: timing

def bottle_normals(mvtec_root: Path):
    from experiments.p14.resource_profile import load
    files = sorted((mvtec_root / cp.PRODUCT / "train" / "good").iterdir())
    return [load(p) for p in files]


def gpu_mem():
    return {"allocated_mb": torch.cuda.memory_allocated() / 2 ** 20, "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20}


def part_m(eng_path, supports, s_lab, n, normals, device):
    eng = Engine(eng_path, device)
    prep = lambda im: fe.gpu_prep(im, device)  # noqa: E731
    small = torch.zeros(B_MIN, D, device=device)
    zero_tau, dummy = torch.zeros(1, device=device), fe.dummy_protos(None, 2)
    out = {}
    with torch.no_grad():
        for _ in range(5):
            eng(feed_plain(prep(supports[0]), small, zero_tau, dummy))
        for m in M_SWEEP:
            imgs = [normals[i % len(normals)] for i in range(m)]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            toks = torch.stack([eng(feed_plain(prep(im), small, zero_tau, dummy))["tokens"].clone() for im in imgs])
            torch.cuda.synchronize()
            t_bank = time.perf_counter() - t0
            bank = toks.reshape(-1, D).contiguous()
            tau = torch.tensor([loo_tau(toks)], device=device)
            torch.cuda.synchronize()
            t_tau = time.perf_counter() - t0 - t_bank
            feats = [{k: v.clone() for k, v in eng(feed_plain(prep(im), bank, tau, dummy)).items()} for im in supports]
            protos = _protos_from(*(torch.cat([f[k] for f in feats]) for k in ("g", "h_a", "h_r")), s_lab, n)
            torch.cuda.synchronize()
            new_task = time.perf_counter() - t0
            mem = gpu_mem()
            x = prep(supports[0])

            def deployed(image):
                o = eng(feed_plain(prep(image), bank, tau, protos))
                return int(o["scores"].argmax()), float(o["screen"])
            out[str(m)] = {"new_task_s": new_task, "bank_tokens_s": t_bank, "loo_tau_s": t_tau,
                           "engine_only": cp.timed(lambda: eng(feed_plain(x, bank, tau, protos)), device),
                           "e2e_from_pil": cp.timed(lambda: deployed(supports[0]), device),
                           "memory": mem, "bank_mb_fp32": bank.numel() * 4 / 2 ** 20, "unique_images": min(m, len(normals))}
            print(f"[p47] M={m}: new task {new_task:.2f} s, engine {out[str(m)]['engine_only']['median_ms']:.1f} ms", flush=True)
            del toks, bank, feats
            torch.cuda.empty_cache()
    return out


def part_c(r8_path, supports, s_lab, n, bank_images, device):
    eng = Engine(r8_path, device)
    prep = lambda im: fe.gpu_prep(im, device)  # noqa: E731
    run = lambda x, b, t, p: tuple(eng(feed_plain(x, b, t, p))[k].clone() for k in fe.OUT_NAMES)  # noqa: E731
    out = {}
    with torch.no_grad():
        bank, tau, _, _ = fe.task_state(run, supports, s_lab, n, bank_images, prep)
        x = prep(supports[0])
        for c in C_SWEEP:
            protos = fe.dummy_protos(None, c)
            out[str(c)] = {"engine_only": cp.timed(lambda: eng(feed_plain(x, bank, tau, protos)), device)}
            print(f"[p47] C={c}: {out[str(c)]['engine_only']['median_ms']:.1f} ms", flush=True)
        # SKUs: S resident banks; consecutive parts use different SKUs (round robin)
        sku = {}
        protos = fe.dummy_protos(None, n)
        for s in S_SWEEP:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            banks = [bank.clone() for _ in range(s)]
            torch.cuda.synchronize()
            added = (torch.cuda.memory_allocated() - before) / 2 ** 20
            i = [0]

            def step():
                b = banks[i[0] % s]
                i[0] += 1
                eng(feed_plain(x, b, tau, protos))
            sku[str(s)] = {"banks_mb": added, "engine_only_round_robin": cp.timed(step, device)}
            print(f"[p47] SKUs={s}: +{added:.0f} MB, {sku[str(s)]['engine_only_round_robin']['median_ms']:.1f} ms", flush=True)
            del banks
            torch.cuda.empty_cache()
        out["skus"] = sku
    return out


def part_tiles(eng_path, grid, supports, s_lab, n, bank_images, device):
    eng = Engine(eng_path, device)
    prep = lambda im: fe.gpu_prep(im, device)  # noqa: E731
    tiles = lambda im: gpu_tiles(im, grid, device)  # noqa: E731
    bank_images = bank_images[:TILE_M[grid]]
    zero = torch.zeros(grid * grid * P * TILE_M[grid], D, device=device)
    zero_tau, dummy = torch.zeros(1, device=device), fe.dummy_protos(None, 2)
    with torch.no_grad():
        for _ in range(3):
            eng(feed_tiled(prep(supports[0]), tiles(supports[0]), zero, zero_tau, dummy))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        toks = torch.stack([eng(feed_tiled(prep(im), tiles(im), zero, zero_tau, dummy))["tokens"].clone() for im in bank_images])
        bank = toks.reshape(-1, D).contiguous()
        tau = torch.tensor([loo_tau(toks)], device=device)
        torch.cuda.synchronize()
        t_bank_tau = time.perf_counter() - t0
        feats = [{k: v.clone() for k, v in eng(feed_tiled(prep(im), tiles(im), bank, tau, dummy)).items()} for im in supports]
        protos = _protos_from(*(torch.cat([f[k] for f in feats]) for k in ("g", "h_a", "h_r")), s_lab, n)
        torch.cuda.synchronize()
        new_task = time.perf_counter() - t0
        full, tl = prep(supports[0]), tiles(supports[0])

        def deployed(image):
            o = eng(feed_tiled(prep(image), tiles(image), bank, tau, protos))
            return int(o["scores"].argmax()), float(o["screen"])
        res = {"grid": grid, "M": TILE_M[grid], "new_task_s": new_task, "bank_and_tau_s": t_bank_tau, "memory": gpu_mem(),
               "bank_mb_fp32": bank.numel() * 4 / 2 ** 20,
               "engine_only": cp.timed(lambda: eng(feed_tiled(full, tl, bank, tau, protos)), device),
               "gpu_tile_preprocess": cp.timed(lambda: (prep(supports[0]), tiles(supports[0])), device),
               "e2e_from_pil": cp.timed(lambda: deployed(supports[0]), device)}
    print(f"[p47] tiles g={grid}: new task {new_task:.2f} s, engine {res['engine_only']['median_ms']:.1f} ms, "
          f"e2e {res['e2e_from_pil']['median_ms']:.1f} ms", flush=True)
    return res


def device_time(args) -> int:
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"pass{args.pass_index}.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    root = Path(args.mvtec_root)
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(root)
    engines = Path(args.engines)
    parts = [p for p in args.parts.split(",") if p]
    if args.pass_index == 2:
        parts.reverse()
    res = {"pass": args.pass_index, "order": parts, "torch": torch.__version__, "classes": n}
    for part in parts:
        if part == "m":
            res["m_sweep"] = part_m(engines / "p47_dyn.engine", supports, s_lab, n, bottle_normals(root), device)
        elif part == "c":
            res["c_and_skus"] = part_c(engines / "main_fused_r8.engine", supports, s_lab, n, bank_images, device)
        else:
            res[f"tiles_{part}"] = part_tiles(engines / f"p47_{part}.engine", int(part[1]), supports, s_lab, n, bank_images, device)
        torch.cuda.empty_cache()
    target.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("[p47] timing done", flush=True)
    return 0


# --------------------------------------------------------------------------- local: tiled accuracy (protocol §3)

def tile_extract(args) -> int:
    """Per product and bank seed: g (whole image), tiled h_A / h_R with the unchanged evaluation code
    (memory-only chunking), written in P41's per-product format so P41's scoring applies unchanged."""
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p10 import run_pilot_a as ra
    from experiments.p11 import run_pilot_ab as ab
    from experiments.p13 import pilot as p13
    from experiments.p32 import device_extract as dx
    grid = int(args.tile_extract)
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    out = OUT / args.tag / f"g{grid}"
    out.mkdir(parents=True, exist_ok=True)
    draws = json.loads(fe.DRAWS.read_text(encoding="utf-8"))
    groups = dx.inputs(Path(args.mvtec_root))
    model = FusedTiled(fe.build_factory(device), grid).eval()
    p13.CHUNK = 1
    with torch.no_grad():
        for product in [x for x in args.products.split(",") if x] or list(groups):
            path = out / f"fused_{product}.pt"
            if path.exists():
                raise SystemExit(f"{path} exists")
            t0 = time.perf_counter()
            g = groups[product]

            def feats(p):
                im = dx.open_rgb(p)
                full = models.deterministic_eval_tensor(im).to(device)
                pooled, _ = model.visual(full, [model.layer])
                d = model.dino.forward_patches(full)
                gg = F.normalize(torch.cat([F.normalize(pooled.float(), dim=-1), F.normalize(d.mean(dim=1).float(), dim=-1)], -1), dim=-1)
                return gg.cpu(), model.tile_tokens(pil_tiles(im, grid).to(device)).cpu()
            normal_paths = dict(g["normals"])
            names = sorted({nm for s in draws[product].values() for nm in s})
            ntok = {nm: feats(normal_paths[nm])[1] for nm in names}
            part_feats = {part: [feats(p) for _, p in g[part]] for part in ("support", "queries")}
            res = {"support_rel_paths": [i for i, _ in g["support"]], "query_rgb_sha256": [i for i, _ in g["queries"]],
                   "grid": grid, "seeds": {}}
            for seed, nms in draws[product].items():
                bank_imgs = torch.stack([ntok[nm] for nm in nms]).to(device)
                bank = bank_imgs.reshape(-1, D)
                tau = loo_tau(bank_imgs)
                per = {}
                for part, fl in part_feats.items():
                    gl, al, rl = [], [], []
                    for gg, c in fl:
                        c = c.to(device)[None]
                        w = ab.region_weights(ra.patch_anomaly(c, bank, chunk=1), tau, soft=True)
                        gl.append(gg)
                        al.append(ab.weighted_pool(c, w).cpu())
                        rl.append(p13.region_residual(c, bank, w).cpu())
                    per[part] = {"g": torch.cat(gl), "h_a": torch.cat(al), "h_r": torch.cat(rl)}
                res["seeds"][seed] = {"tau": tau, **per}
                del bank_imgs, bank
            torch.save(res, path)
            del ntok, part_feats
            torch.cuda.empty_cache()
            print(f"[p47] g={grid} {product}: {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"[p47] g={grid} extraction done", flush=True)
    return 0


def tile_score(args) -> int:
    """P41's scoring, unchanged, pointed at the tiled descriptors (cache = P14, target = this grid)."""
    fe.OUT = OUT / args.tag
    return fe.score(argparse.Namespace(tag=f"g{args.grid}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", choices=("dyn", "g2", "g3"))
    ap.add_argument("--check", choices=("dyn", "g2", "g3"))
    ap.add_argument("--onnx-only", choices=("dyn", "g2", "g3"))
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--tile-extract", choices=("1", "2", "3"))
    ap.add_argument("--tile-score", action="store_true")
    ap.add_argument("--grid", type=int)
    ap.add_argument("--mvtec-root")
    ap.add_argument("--out")
    ap.add_argument("--engine")
    ap.add_argument("--engines")
    ap.add_argument("--r8")
    ap.add_argument("--tag")
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--products", default="")
    ap.add_argument("--parts", default="m,c,g2,g3")
    a = ap.parse_args(argv)
    if a.export:
        return export(a)
    if a.check:
        return check(a)
    if a.onnx_only:
        return onnx_only(a)
    if a.time:
        return device_time(a)
    if a.tile_extract:
        return tile_extract(a)
    if a.tile_score:
        return tile_score(a)
    raise SystemExit("choose a mode")


if __name__ == "__main__":
    sys.exit(main())
