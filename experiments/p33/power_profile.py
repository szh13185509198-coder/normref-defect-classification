#!/usr/bin/env python3
"""P33 (report only): energy per image of the backbone forward on the device.

For each configuration, the same model and input as ``experiments.p26.cost_profile.worker_ours`` (bottle, the first
support image, batch 1) run back to back for a fixed time; the run records the wall-clock window and the number of
forwards. Board power comes from ``tegrastats`` logging at 100 ms in parallel (started by the dispatch); energy per
image = mean board power over the window x mean time per forward. A 30 s idle window comes first.

Usage (on the device, tegrastats already logging):
    python -m experiments.p33.power_profile --mvtec-root <root> --tag <tag> [--seconds 60]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p26 import cost_profile as cp  # noqa: E402

CONFIGS = ("ours_main", "ours_main_fp16", "ours_l14_336_fp32", "ours_l14_336_fp16", "ours_b336_fp32", "ours_b336_fp16",
           "ours_s336_fp32", "ours_s336_fp16", "ours_clipb16_224", "ours_clipb16_224_fp16")
OUT_ROOT = PROJECT_ROOT / "ckpt" / "p33"


def build(block: str, device):
    """(prep, forward) exactly as cost_profile.worker_ours builds them for this block (shared builder)."""
    _, _, prep, forward = cp.build_ours(block, device)
    return prep, forward


def stamp() -> dict:
    return {"epoch": time.time(), "local": time.strftime("%Y-%m-%d %H:%M:%S")}


def main(argv=None) -> int:
    """One process per configuration on small-memory devices: ``--only <block>``; ``--idle-seconds`` > 0 records the
    idle window (once, in the first call). Each call writes ``p33_power_<block>.json`` (write-once)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--idle-seconds", type=float, default=0.0)
    ap.add_argument("--only", default="", help="comma-separated subset of CONFIGS (default: all, one process)")
    ap.add_argument("--window", type=float, default=0.0, help="> 0: record the forward count every WINDOW seconds "
                    "(sustained-load drift check)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    blocks = [b for b in CONFIGS if not a.only or b in a.only.split(",")]
    out = OUT_ROOT / a.tag
    targets = [out / f"p33_power_{b}.json" for b in blocks] + ([out / "p33_idle.json"] if a.idle_seconds > 0 else [])
    if any(t.exists() for t in targets):
        raise SystemExit(f"{[t.name for t in targets if t.exists()]} exist in {out}; write-once")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    supports = cp.task_inputs(Path(a.mvtec_root))[0]
    if a.idle_seconds > 0:
        idle = {"start": stamp()}
        time.sleep(a.idle_seconds)
        idle["end"] = stamp()
        (out / "p33_idle.json").write_text(json.dumps({"idle": idle, "machine": cp.machine_info()}, indent=1,
                                                      default=str) + "\n", encoding="utf-8")
    for block in blocks:
        prep, forward = build(block, device)
        x = prep(supports[0])
        with torch.no_grad():
            for _ in range(20):
                forward(x)
            torch.cuda.synchronize()
            n, start, windows = 0, stamp(), []
            t0 = time.perf_counter()
            next_mark = a.window if a.window > 0 else float("inf")
            while time.perf_counter() - t0 < a.seconds:
                forward(x)
                torch.cuda.synchronize()
                n += 1
                if time.perf_counter() - t0 >= next_mark:
                    windows.append({"t_s": time.perf_counter() - t0, "forwards": n, "local": stamp()["local"]})
                    next_mark += a.window
            elapsed = time.perf_counter() - t0
        res = {"schema_version": "p33_power_v2", "block": block, "seconds": a.seconds, "torch": torch.__version__,
               "start": start, "end": stamp(), "forwards": n, "mean_ms": 1000 * elapsed / n,
               "windows": windows}
        (out / f"p33_power_{block}.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"[p33] {block}: {n} forwards, {1000 * elapsed / n:.2f} ms each", flush=True)
        del forward, prep, x
        torch.cuda.empty_cache()
        time.sleep(10)   # let the power settle between configurations
    print("[p33] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
