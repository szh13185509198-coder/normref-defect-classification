#!/usr/bin/env python3
"""P38 (report only): TensorRT deployment on the Jetson -- end-to-end latency and accuracy (docs/P38_TRT_DEPLOY_PROTOCOL.md).

The validated P35 TensorRT FP16 engines replace the PyTorch backbones; everything else (preprocessing, heads, timing)
is the P26 / P18 code. Needs the TensorRT Python bindings (system package on JetPack): run with
PYTHONPATH=<dir holding only a `tensorrt` link>.

Usage (on the device):
    python -m experiments.p38.trt_deploy --tag jetson_orin_nano --mvtec-root <root> --engines <dir>      # latency
    python -m experiments.p38.trt_deploy --extract --tag jetson_orin_nano_trt --mvtec-root <root> --engines <dir>
Scoring (local):
    python -m experiments.p32.score_device --tag jetson_orin_nano_trt --precisions trt
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p26 import cost_profile as cp  # noqa: E402  (no torch at import)

ENGINES = {"clip_l14": "clip_l14_336_r2_fp16.engine", "dino_s": "dinov2_s_336_ex_r5.engine",
           "dino_b": "dinov2_b_336_ex_r5.engine", "clip_b16_224": "clip_b16_224_fp16.engine",
           "clip_b16_336": "clip_b16_336_r2_fp16.engine", "clip_l14_l18": "clip_l14_l18_r6.engine",
           "clip_l14_l12": "clip_l14_l12_r7.engine"}
BLOCKS = ("trt_main", "trt_b336", "trt_s336", "trt_clipb16_224", "trt_clipb16_336", "trt_gate_dino",
          "trt_l14_l18", "trt_gate_l18",   # amendment P38-A1
          "trt_gate_l12")                  # P40 section 4: pass path of the layer-12 screening exit
PYTORCH_BASE = {"trt_main": "ours_main", "trt_b336": "ours_b336_fp32", "trt_s336": "ours_s336_fp32",
                "trt_clipb16_224": "ours_clipb16_224", "trt_clipb16_336": "ours_clipb16_336", "trt_gate_dino": "ours_main",
                "trt_l14_l18": "ours_l14_336_fp32", "trt_gate_l18": "ours_l14_336_fp32", "trt_gate_l12": None}
OUT_ROOT = PROJECT_ROOT / "ckpt" / "p38"


class TrtEngine:
    """One TensorRT engine (static shapes, input 'input'); outputs live in preallocated torch tensors that the next
    call overwrites."""

    def __init__(self, path: Path, device):
        import tensorrt as trt
        import torch
        self.path = Path(path)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        self.context = self.engine.create_execution_context()
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = tuple(self.engine.get_tensor_shape(name))
                continue
            t = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=dtypes[self.engine.get_tensor_dtype(name)],
                            device=device)
            self.outputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

    def __call__(self, x):
        import torch
        x = x.float().contiguous()
        if tuple(x.shape) != self.input_shape:
            raise ValueError(f"{self.path.name}: input {tuple(x.shape)} != {self.input_shape}")
        self.context.set_tensor_address("input", x.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"{self.path.name}: execution failed")
        return self.outputs


class TrtMain:
    """p14 Pipeline (main configuration) with both backbones as TensorRT engines on the same 336 tensor."""

    precision = "trt_fp16"

    def __init__(self, engines, device):
        from experiments.p6 import p6b_stage2_models as models
        self.models, self.device = models, device
        self.clip, self.dino = engines["clip_l14"], engines["dino_s"]
        self.fusion = models.FrozenFusion()

    def forward(self, x):
        oc = self.clip(x)
        od = self.dino(x)
        return oc, od

    def features(self, image, with_dino: bool = False):
        import torch.nn.functional as F
        x = self.models.deterministic_eval_tensor(image).to(self.device)
        oc, od = self.forward(x)
        d = od["L12_patches"]
        g = self.fusion.fuse(self.models.l2_rows(oc["clip_pooled"]), self.models.l2_rows(d.mean(dim=1))).float()
        c = F.normalize(oc["clip_l18_tokens"][:, 1:, :].float(), dim=-1)
        return (g, c, F.normalize(d.float(), dim=-1)) if with_dino else (g, c)


class TrtClipL18:
    """cost_profile.ClipOnlyPipeline (P30 `a_r` features) from the truncated CLIP-L/14 engine (blocks 1-18)."""

    precision = "trt_fp16"

    def __init__(self, engine, device, layer: int = 18):
        self.engine, self.device, self.key = engine, device, f"clip_l{layer}_tokens"

    def forward(self, x):
        return self.engine(x)

    def features(self, image):
        import torch.nn.functional as F
        from experiments.p6 import p6b_stage2_models as models
        o = self.engine(models.deterministic_eval_tensor(image).to(self.device))
        return None, F.normalize(o[self.key][:, 1:, :].float(), dim=-1)


class TrtDino:
    """p22 LightPipeline (single DINOv2 backbone) with the TensorRT engine."""

    precision = "trt_fp16"

    def __init__(self, engine, layer: int, device):
        self.engine, self.layer, self.device = engine, layer, device

    def forward(self, x):
        return self.engine(x)

    def features(self, image):
        import torch.nn.functional as F
        from experiments.p18 import backbones as bb
        o = self.engine(bb.eval_tensor(image, 336).to(self.device))
        return F.normalize(o["L12_cls"].float(), dim=-1), F.normalize(o[f"L{self.layer}_patches"].float(), dim=-1)


class TrtClipB:
    """p18 AltPipeline (CLIP-B/16) with the TensorRT engine."""

    precision = "trt_fp16"

    def __init__(self, engine, res: int, layer: int, device):
        self.engine, self.res, self.layer, self.device = engine, res, layer, device

    def forward(self, x):
        return self.engine(x)

    def features(self, image):
        import torch.nn.functional as F
        from experiments.p18 import backbones as bb
        o = self.engine(bb.eval_tensor(image, self.res).to(self.device))
        return F.normalize(o["pooled"].float(), dim=-1), F.normalize(o[f"L{self.layer}_tokens"][:, 1:, :].float(), dim=-1)


def build_trt(block: str, engine_dir: Path, device):
    """(pipe, prep) for a block."""
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p18 import backbones as bb
    from experiments.p18.run import dev_selection, parse_config
    from experiments.p22 import lightweight as lw
    from experiments.p24.clipb224 import layer_224
    if block in ("trt_main", "trt_gate_dino"):
        engines = {k: TrtEngine(engine_dir / ENGINES[k], device) for k in ("clip_l14", "dino_s")}
        return TrtMain(engines, device), (lambda im: models.deterministic_eval_tensor(im).to(device))
    if block in ("trt_l14_l18", "trt_gate_l18", "trt_gate_l12"):
        layer = 12 if block == "trt_gate_l12" else 18
        eng = TrtEngine(engine_dir / ENGINES[f"clip_l14_l{layer}"], device)
        return TrtClipL18(eng, device, layer), (lambda im: models.deterministic_eval_tensor(im).to(device))
    if block in ("trt_b336", "trt_s336"):
        cfg = {"trt_b336": "dinov2_b_r336_fp32", "trt_s336": "dinov2_s_r336_fp32"}[block]
        layer = lw.rc.load_json(lw.OUT["dev"] / "p22_dev_results.json")["selection"][cfg]
        eng = TrtEngine(engine_dir / ENGINES["dino_b" if block == "trt_b336" else "dino_s"], device)
        return TrtDino(eng, layer, device), (lambda im: bb.eval_tensor(im, 336).to(device))
    res = int(block.rsplit("_", 1)[1])
    layer = layer_224() if res == 224 else parse_config(dev_selection()["clip_b16"])[2]
    eng = TrtEngine(engine_dir / ENGINES[f"clip_b16_{res}"], device)
    return TrtClipB(eng, res, layer, device), (lambda im: bb.eval_tensor(im, res).to(device))


def _cos(a, b) -> dict:
    import torch.nn.functional as F
    a, b = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, b.shape[-1])
    c = F.cosine_similarity(a, b, dim=-1)
    return {"mean_cos": float(c.mean()), "min_cos": float(c.min())}


def worker(block: str, args, device) -> dict:
    import torch
    from experiments.p18.resource_profile import profile
    supports, s_lab, n, bank, _, _ = cp.task_inputs(Path(args.mvtec_root))
    sync = torch.cuda.synchronize
    probe = [supports[0], supports[-1], bank[0]]
    with torch.no_grad():
        if PYTORCH_BASE[block] is None:   # layer-12 tokens: reference = PyTorch FP32 blocks 1-12 (P36 ClipSplit)
            from experiments.p6 import p6b_stage2_models as models
            from experiments.p36.early_exit_profile import ClipL18Pipeline, ClipSplit
            clip_model, _ = models.build_clip_backbone(PROJECT_ROOT, device)
            ref_pipe = ClipL18Pipeline(ClipSplit(clip_model.visual, 12), "fp32", device)
        else:
            ref_pipe = cp.build_ours(PYTORCH_BASE[block], device)[0]          # PyTorch FP32 reference features
        ref = [ref_pipe.features(im) for im in probe]
        del ref_pipe
        torch.cuda.empty_cache()
        pipe, prep = build_trt(block, Path(args.engines), device)
        got = [pipe.features(im) for im in probe]
        feature_check = {"patches": [_cos(a[1], b[1]) for a, b in zip(got, ref)]}
        if got[0][0] is not None:
            feature_check["global"] = [_cos(a[0], b[0]) for a, b in zip(got, ref)]
        if block == "trt_l14_l18":
            res = cp.profile_ar(block, pipe, supports, bank, s_lab, n, device, sync)
            x = prep(supports[0])
            res["model_forward"] = cp.timed(lambda: pipe.forward(x), device)
            res["per_task_preparation_s"] = sum(res["offline_ms"].values()) / 1000.0
        elif block in ("trt_gate_l18", "trt_gate_l12"):
            res = gate_l18(pipe, prep, supports, bank, device)
        elif block != "trt_gate_dino":
            res = profile(block, pipe, supports, bank, s_lab, n, device, sync)
            x = prep(supports[0])
            res["model_forward"] = cp.timed(lambda: pipe.forward(x), device)
            res["per_task_preparation_s"] = sum(res["offline_ms"].values()) / 1000.0
        else:
            res = gate_dino(pipe, prep, supports, bank, device)
    res.update({"block": block, "precision": "trt_fp16", "feature_check_vs_pytorch_fp32": feature_check,
                "engines": {k: v for k, v in ENGINES.items()}})
    return res


def gate_dino(pipe, prep, supports, bank_images, device) -> dict:
    """Light-gate pass path with TensorRT: preprocessing, DINOv2-S engine, DINOv2-S bank nearest neighbour, screening
    score (P15 top1pct), host decision. The score is checked against the P15 formula on the same features."""
    import torch
    import torch.nn.functional as F
    from experiments.p10 import run_pilot_a as ra
    from experiments.p15 import screening as sc
    from experiments.revision.c3_resource_profile import _time
    bank_d = torch.cat([pipe.features(im, with_dino=True)[2] for im in bank_images]).reshape(-1, 384)

    def pass_path(image):
        od = pipe.dino(prep(image))
        dn = F.normalize(od["L12_patches"].float(), dim=-1)
        best = torch.einsum("npd,bd->npb", dn, bank_d).max(dim=2).values
        return float(sc.image_score(1.0 - best, "top1pct")[0])
    q = supports[0]
    dq = pipe.features(q, with_dino=True)[2]
    check = pass_path(q) == float(sc.image_score(ra.patch_anomaly(dq, bank_d), "top1pct")[0])
    if not check:
        raise SystemExit("gate score differs from the P15 formula")
    x = prep(q)
    return {"screen_score_equal_p15_formula": check, "e2e": {"pass": _time(lambda: pass_path(q), device)},
            "components": {"dino_engine": cp.timed(lambda: pipe.dino(x), device),
                           "clip_engine": cp.timed(lambda: pipe.clip(x), device),
                           "preprocess_to_device": cp.timed(lambda: prep(q), device)}}


def gate_l18(pipe, prep, supports, bank_images, device) -> dict:
    """Two-level inspection pass path with the truncated engine: preprocessing, CLIP blocks 1-18, CLIP bank nearest
    neighbour, screening score (P15 top1pct), host decision; checked against the P15 formula on the same features."""
    import torch
    import torch.nn.functional as F
    from experiments.p10 import run_pilot_a as ra
    from experiments.p15 import screening as sc
    from experiments.revision.c3_resource_profile import _time
    bank = torch.cat([pipe.features(im)[1] for im in bank_images]).reshape(-1, 1024)

    def pass_path(image):
        o = pipe.engine(prep(image))
        c = F.normalize(o[pipe.key][:, 1:, :].float(), dim=-1)
        best = torch.einsum("npd,bd->npb", c, bank).max(dim=2).values
        return float(sc.image_score(1.0 - best, "top1pct")[0])
    q = supports[0]
    check = pass_path(q) == float(sc.image_score(ra.patch_anomaly(pipe.features(q)[1], bank), "top1pct")[0])
    if not check:
        raise SystemExit("gate score differs from the P15 formula")
    x = prep(q)
    return {"screen_score_equal_p15_formula": check, "e2e": {"pass": _time(lambda: pass_path(q), device)},
            "components": {"clip_truncated_engine": cp.timed(lambda: pipe.engine(x), device),
                           "preprocess_to_device": cp.timed(lambda: prep(q), device)}}


# --------------------------------------------------------------------------- accuracy: P32-format extraction

def extract(args, device) -> int:
    import torch
    from experiments.p32 import device_extract as dx
    out_dir = PROJECT_ROOT / "ckpt" / "p32" / args.tag
    groups = dx.inputs(Path(args.mvtec_root))
    wanted = [x for x in args.products.split(",") if x] or list(groups)
    existing = [x for x in wanted if (out_dir / f"trt_{x}.pt").exists()]
    if existing:
        raise SystemExit(f"{out_dir}: trt files for {existing} exist (write-once)")
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe, _ = build_trt("trt_main", Path(args.engines), device)
    log = {"engines": ENGINES, "batch": 1, "products": {}}
    with torch.no_grad():
        for product in wanted:
            g = groups[product]
            t0 = time.perf_counter()

            def run(items, with_global):
                gl, cl = [], []
                for _, path in items:
                    gg, cc = pipe.features(dx.open_rgb(path))
                    if with_global:
                        gl.append(gg.cpu())
                    cl.append(cc.half().cpu())
                return (torch.cat(gl) if gl else None), torch.cat(cl)
            sg, scl = run(g["support"], True)
            qg, qcl = run(g["queries"], True)
            _, ncl = run(g["normals"], False)
            torch.save({"schema_version": "p32_device_features_v1", "product": product, "precision": "trt",
                        "support_rel_paths": [i for i, _ in g["support"]], "support_global": sg, "support_clip18": scl,
                        "query_rgb_sha256": [i for i, _ in g["queries"]], "query_global": qg, "query_clip18": qcl,
                        "normal_names": [i for i, _ in g["normals"]], "normal_clip18": ncl}, out_dir / f"trt_{product}.pt")
            nimg = len(g["support"]) + len(g["queries"]) + len(g["normals"])
            log["products"][product] = {"images": nimg, "seconds": time.perf_counter() - t0}
            print(f"[p38] {product}: {nimg} images, {time.perf_counter() - t0:.0f} s", flush=True)
    (out_dir / f"p38_extract_log_{'_'.join(wanted)}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("[p38] extraction done", flush=True)
    return 0


# --------------------------------------------------------------------------- orchestrator (as P36)

def headline(block: str, r: dict) -> dict:
    fc = {f"cos_{k}": min(x["mean_cos"] for x in v) for k, v in r["feature_check_vs_pytorch_fp32"].items()}
    if "e2e" in r:
        return {"e2e_pass_ms": r["e2e"]["pass"]["median_ms"], **{f"{k}_ms": v["median_ms"] for k, v in r["components"].items()}, **fc}
    head = r["online"][cp.AR_HEAD if block == "trt_l14_l18" else "dual_branch_ARR"]
    return {"prep_s": r["per_task_preparation_s"], "engine_ms": r["model_forward"]["median_ms"],
            "end_to_end_ms": head["end_to_end_from_pil"]["median_ms"],
            "head_ms": head["head_only_given_features"]["median_ms"], **fc}


def run_worker(args) -> int:
    import numpy  # noqa: F401
    import torch
    t0 = time.perf_counter()
    res = worker(args.worker, args, torch.device("cuda"))
    res["block_wall_s"] = time.perf_counter() - t0
    res["torch"] = torch.__version__
    (Path(args.out) / f"pass{args.pass_index}_{args.worker}.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                                                           encoding="utf-8")
    return 0


def run_orchestrator(args) -> int:
    out = OUT_ROOT / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --tag")
    only = [b for b in args.only.split(",") if b]
    if set(only) - set(BLOCKS):
        raise SystemExit(f"unknown blocks {sorted(set(only) - set(BLOCKS))}")
    blocks = [b for b in BLOCKS if not only or b in only]
    out.mkdir(parents=True)
    env = {"platform": platform.platform(), "python": sys.version.split()[0], "telemetry_start": cp.telemetry(),
           **cp.machine_info()}
    log = []
    for pass_index, order in ((1, blocks), (2, list(reversed(blocks))))[:args.passes]:
        for block in order:
            before = cp.telemetry()
            cmd = [sys.executable, "-u", "-m", "experiments.p38.trt_deploy", "--worker", block, "--pass-index",
                   str(pass_index), "--out", str(out), "--mvtec-root", args.mvtec_root, "--engines", args.engines]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=args.block_timeout)
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(cmd, "timeout", "", f"timed out after {exc.timeout} s")
            entry = {"pass": pass_index, "block": block, "returncode": proc.returncode, "wall_s": time.perf_counter() - t0,
                     "telemetry_before": before, "telemetry_after": cp.telemetry()}
            if proc.returncode != 0:
                entry["stderr_tail"] = proc.stderr[-3000:]
            log.append(entry)
            print(f"[p38] pass {pass_index} {block}: rc={proc.returncode} {entry['wall_s']:.0f}s", flush=True)
            (out / "run_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    summary = {b: {p: headline(b, json.loads((out / f"pass{p}_{b}.json").read_text(encoding="utf-8")))
                   for p in (1, 2) if (out / f"pass{p}_{b}.json").exists()} for b in blocks}
    (out / "p38_results.json").write_text(json.dumps({"schema_version": "p38_v1", "tag": args.tag, "environment": env,
                                                      "summary": summary, "run_log": log}, indent=1, default=str) + "\n",
                                          encoding="utf-8")
    lines = [f"# P38 TensorRT deployment — {args.tag}", "", "Values: pass 1 / pass 2.", ""]
    for b, pp in summary.items():
        keys = sorted({k for h in pp.values() for k in h})
        lines.append(f"- **{b}**: " + "; ".join(f"{k} " + " / ".join("—" if pp.get(p, {}).get(k) is None else f"{pp[p][k]:.4g}"
                                                                     for p in (1, 2)) for k in keys))
    (out / "p38_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag")
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--block-timeout", type=int, default=3600)
    ap.add_argument("--worker", choices=BLOCKS)
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--products", default="")
    args = ap.parse_args(argv)
    if args.worker:
        return run_worker(args)
    if not args.tag:
        raise SystemExit("--tag is required")
    if args.extract:
        import numpy  # noqa: F401
        import torch
        return extract(args, torch.device("cuda"))
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
