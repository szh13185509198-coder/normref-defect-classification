#!/usr/bin/env python3
"""Addendum C3: parameter counts, batch-1 latency/memory, support-adaptation time.

Subcommands (run each latency pipeline in its own fresh process, GPU otherwise idle):

    python -m experiments.revision.c3_resource_profile params
    python -m experiments.revision.c3_resource_profile latency --pipeline ours
    python -m experiments.revision.c3_resource_profile latency --pipeline fusion_proto
    python -m experiments.revision.c3_resource_profile latency --pipeline clip_proto
    python -m experiments.revision.c3_resource_profile latency --pipeline dino_proto
    python -m experiments.revision.c3_resource_profile adapt-times

The timing input is a support-pool image (never a query image).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.revision import revision_common as rc  # noqa: E402

WARMUP = 50
TIMED = 200
N_CLASSES_FOR_TIMING = 5
TIMING_MANIFEST = rc.STAGE1_BUNDLE / "p6b_support_manifests" / "bottle" / "k5_s111.json"
PIPELINES = ("ours", "fusion_proto", "clip_proto", "dino_proto")


def _environment(device) -> dict:
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "platform": platform.platform(), "processor": platform.processor(),
            "device": str(device)}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def _count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _build_ours(device):
    from experiments.p6.evaluate_p6b_formal_grid import _build_adapted_clip
    return _build_adapted_clip(PROJECT_ROOT, device)


def cmd_params(args) -> dict:
    device = torch.device("cpu")
    model = _build_ours(device)
    top_level = {name: _count(child) for name, child in model.named_children()}
    image_adapter = {name: _count(child) for name, child in model.image_adapter.items()}
    det_proj = image_adapter.get("det_proj", 0)
    fusion_dim = 1152
    per_product = {}
    for product in rc.PRODUCTS:
        manifest = rc.load_json(rc.STAGE1_BUNDLE / "p6b_support_manifests" / product / "k1_s111.json")
        n_classes = len(manifest["class_order"])
        per_product[product] = {
            "classes": n_classes,
            "ours_trainable": det_proj + n_classes * fusion_dim + 1,
            "linear_probe_trainable": n_classes * fusion_dim + n_classes,
            "logreg_rfs_parameters": n_classes * fusion_dim + n_classes,
        }
    return {
        "schema_version": "revision_c3_params_v1",
        "note": ("Counts from the instantiated Ours query model. Dormant AA-CLIP scaffold "
                 "modules (layer_adapters, seg_proj, text adapters) are constructed but are "
                 "not on the terminal_only inference path; P8-E0 slices them from the "
                 "deployment artifact without changing any decision."),
        "top_level_modules": top_level,
        "image_adapter_submodules": image_adapter,
        "trainable_terminal_projection": det_proj,
        "prototype_and_scale_formula": "C x 1152 + 1",
        "baselines_trainable": {"clip_proto": 0, "dino_proto": 0, "fusion_proto": 0,
                                "p30_frozen_proto": 0, "simpleshot_cl2n": 0,
                                "linear_probe": "C x 1152 + C", "logreg_rfs": "C x 1152 + C"},
        "per_product": per_product,
        "total_parameters_instantiated": _count(model),
    }


def _timing_image() -> Image.Image:
    manifest = rc.load_json(TIMING_MANIFEST)
    first_class = manifest["class_order"][0]
    rel = manifest["classes"][first_class]["items"][0]["rel_path"]
    with Image.open(rc.DATASET_ROOT / rel) as handle:
        return handle.convert("RGB").copy()


def _time(fn, device) -> dict:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        sync()
        samples = []
        for _ in range(TIMED):
            start = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.fmean(samples),
            "p90_ms": samples[int(0.9 * (len(samples) - 1))], "n": len(samples)}


def cmd_latency(args) -> dict:
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p6 import run_p6b_stage2_support as support

    device = torch.device(args.device)
    image = _timing_image()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prototypes = None
    ctx = SimpleNamespace(project_root=PROJECT_ROOT, device=device)

    if args.pipeline == "ours":
        model = _build_ours(device)
        prototypes = F.normalize(torch.randn(N_CLASSES_FOR_TIMING, 1152, device=device), dim=-1)
        scale = 10.0

        def forward(batch):
            return scale * (F.normalize(model.forward_fusion(batch), dim=-1) @ prototypes.t())
    else:
        if args.pipeline == "clip_proto":
            factory = support._build_lazy_clip_only(ctx)
            dim = models.EXPECTED_CLIP_DIM
        elif args.pipeline == "dino_proto":
            factory = support._build_lazy_dino_only(ctx)
            dim = models.EXPECTED_DINO_DIM
        else:
            factory = support._build_lazy_clip_and_dino(ctx)
            dim = models.EXPECTED_FUSION_DIM
        prototypes = F.normalize(torch.randn(N_CLASSES_FOR_TIMING, dim, device=device), dim=-1)

        def forward(batch):
            if args.pipeline == "clip_proto":
                feat = factory.clip_extractor.embed_images([batch])
            elif args.pipeline == "dino_proto":
                feat = factory.dino_extractor_interface.embed_images([batch])
            else:
                feat = factory.fusion.fuse(factory.clip_extractor.embed_images([batch]),
                                           factory.dino_extractor_interface.embed_images([batch]))
            return feat.to(device) @ prototypes.t()

    resident = torch.cuda.memory_allocated() if device.type == "cuda" else None
    tensor = models.deterministic_eval_tensor(image).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model_only = _time(lambda: forward(tensor), device)
    end_to_end = _time(lambda: forward(models.deterministic_eval_tensor(image).to(device)), device)
    return {
        "schema_version": "revision_c3_latency_v1",
        "pipeline": args.pipeline,
        "boundary_model_forward": model_only,
        "boundary_end_to_end_from_pil": end_to_end,
        "batch_size": 1, "precision": "fp32", "warmup": WARMUP, "timed": TIMED,
        "input_view": "deterministic 336 evaluation view",
        "timing_image": "support-pool image from " + str(TIMING_MANIFEST.relative_to(PROJECT_ROOT)),
        "resident_allocated_bytes": resident,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "environment": _environment(device),
    }


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _parse_ts(line: str):
    match = _TS.match(line)
    return dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f") if match else None


def cmd_adapt_times(args) -> dict:
    units = []
    for product in rc.PRODUCTS:
        for k_shot in rc.K_VALUES:
            for seed in rc.SEEDS:
                log = rc.SUPPORT_GRID / "ours" / product / f"k{k_shot}_s{seed}" / "ours_train_multiclass.log"
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                start = next(t for t in map(_parse_ts, lines) if t)
                end = None
                for line in lines:
                    if "Checkpoint saved" in line:
                        end = _parse_ts(line)
                if end is None:
                    raise rc.P6BFailClosedError(f"no 'Checkpoint saved' in {log}")
                units.append({"product": product, "K": k_shot, "seed": seed,
                              "start": start, "end": end,
                              "seconds": (end - start).total_seconds()})
    events = sorted([(u["start"], 1) for u in units] + [(u["end"], -1) for u in units],
                    key=lambda e: (e[0], e[1]))
    concurrent = max_concurrent = 0
    for _, delta in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    overlapping = sum(
        any(o is not u and o["start"] < u["end"] and u["start"] < o["end"] for o in units)
        for u in units)
    per_k = {}
    for k_shot in rc.K_VALUES:
        secs = [u["seconds"] for u in units if u["K"] == k_shot]
        per_k[str(k_shot)] = {"median_s": statistics.median(secs), "mean_s": statistics.fmean(secs),
                              "min_s": min(secs), "max_s": max(secs), "units": len(secs)}
    return {
        "schema_version": "revision_c3_adapt_times_v1",
        "source": "formal cloud logs, first log line to 'Checkpoint saved'",
        "units": len(units),
        "max_concurrent_units": max_concurrent,
        "units_overlapping_another": overlapping,
        "caveat": ("wall time on the original cloud host; if max_concurrent_units > 1 the "
                   "figures include contention from concurrent units"),
        "per_K": per_k,
        "per_unit": [{**{k: v for k, v in u.items() if k not in ("start", "end")},
                      "start": u["start"].isoformat(), "end": u["end"].isoformat()}
                     for u in units],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("params")
    lat = sub.add_parser("latency")
    lat.add_argument("--pipeline", required=True, choices=PIPELINES)
    lat.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub.add_parser("adapt-times")
    args = parser.parse_args(argv)

    if args.cmd == "params":
        result, name = cmd_params(args), "c3_params.json"
    elif args.cmd == "latency":
        result, name = cmd_latency(args), f"c3_latency_{args.pipeline}_{args.device.split(':')[0]}.json"
    else:
        result, name = cmd_adapt_times(args), "c3_adapt_times.json"
    result["provenance"] = rc.git_provenance()
    rc.write_json(rc.C3_DIR / name, result)
    printable = {k: v for k, v in result.items() if k not in ("per_unit", "per_product")}
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
