#!/usr/bin/env python3
"""P47 U6-D (report only): device-side scaling measurements for the uncertainty evaluation
(docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, section U6-D).

On the device (PYTHONPATH = the tensorrt link directory); run by experiments/p47/run_jetson_u6.sh:
    python -m experiments.p47.jetson_u6 --scaling --tag <tag> --mvtec-root <root> --fused <engine>
Local check of the code path with a stub engine (CPU; no TensorRT, no data):
    python -m experiments.p47.jetson_u6 --selftest

Measured on the deployed path (P41 fused engine, GPU preprocessing), bottle K = 5 seed 111, M = 20 (the P43 task):
(a) end-to-end latency against the number of classes C (the engine's dynamic class dimension; random prototypes,
    decisions not used);
(b) the dual-branch head alone (experiments.p41.fused_engine.head, PyTorch FP32 / FP16) against the bank size M, because
    the engine's bank input has a fixed shape (M = 20); timing is value-independent, so the real M = 20 bank rows are
    repeated to the M × 576 rows;
(c) per-part latency when the decoded image is split into g × g tiles, each tile going through the deployed path
    (batch 1, sequential): the cost of a g-times finer sampling of the part.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / "ckpt" / "p47"
CLASSES = (2, 3, 5, 8, 16)          # fused-engine class dimension (dynamic, 2-16 in P41)
BANK_IMAGES = (5, 10, 20, 50, 100)  # M for the head-only timing
TILE_GRID = (1, 2, 3)               # tiles per side: 1, 4 and 9 deployed calls per part
PATCHES, DIM, INPUT_PX = 576, 1024, 336


def stamp() -> dict:
    return {"epoch": time.time(), "local": time.strftime("%Y-%m-%d %H:%M:%S")}


def max_temp_c():
    vals = []
    for f in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        try:
            vals.append(int(Path(f).read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(vals) if vals else None


def timer(fn, sync, warmup: int, n: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def random_protos(state_protos: dict, n: int, gen, device) -> dict:
    """Prototypes with n classes, for timing the class dimension only (the decisions are not used)."""
    import torch
    import torch.nn.functional as F
    out = {}
    for k in ("g", "a", "r"):
        mu = state_protos[f"mu_{k}"]
        out[f"mu_{k}"] = mu
        out[f"proto_{k}"] = F.normalize(torch.randn(n, mu.shape[-1], generator=gen), dim=-1).to(device)
    return out


def tiles(image, g: int) -> list:
    """g x g crops covering the decoded image (row-major)."""
    w, h = image.size
    xs = [round(i * w / g) for i in range(g + 1)]
    ys = [round(j * h / g) for j in range(g + 1)]
    return [image.crop((xs[i], ys[j], xs[i + 1], ys[j + 1])) for j in range(g) for i in range(g)]


def measure(eng, prep, q, state, device, n_rep: int, quantiles, gpu_freq, bank_images=BANK_IMAGES) -> dict:
    import torch
    from experiments.p41.fused_engine import head
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    bank, tau, protos = state

    def deployed(image, st):
        s, screen = eng(prep(image), *st)[:2]
        return int(s.argmax()), float(screen)

    res = {"start": stamp(), "gpu_freq_start": gpu_freq(), "max_temp_c_start": max_temp_c()}
    with torch.no_grad():
        gen = torch.Generator().manual_seed(47)
        res["classes_e2e"] = {}
        for n in CLASSES:
            st = (bank, tau, random_protos(protos, n, gen, device))
            res["classes_e2e"][str(n)] = quantiles(timer(lambda: deployed(q, st), sync, 20, n_rep))
            print(f"[p47-u6] C={n}: {res['classes_e2e'][str(n)]['p50_ms']:.2f} ms", flush=True)

        c = eng(prep(q), *state)[5][0].float().clone()      # [576, 1024] normalised layer-18 tokens of the query
        res["head_vs_M"] = {}
        for m in bank_images:
            rows = PATCHES * m
            bm = bank.float().repeat(math.ceil(rows / bank.shape[0]), 1)[:rows].contiguous()
            row = {"bank_rows": rows}
            for prec, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
                cc, bb, tt = c.to(dt), bm.to(dt), tau.float().to(dt)
                row[prec] = quantiles(timer(lambda: head(cc, bb, tt, None, None), sync, 10, n_rep))
                del cc, bb, tt
            res["head_vs_M"][str(m)] = row
            del bm
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[p47-u6] head M={m}: fp32 {row['fp32']['p50_ms']:.2f} ms, fp16 {row['fp16']['p50_ms']:.2f} ms", flush=True)

        res["tiles_per_part"] = {}
        for g in TILE_GRID:
            parts = tiles(q, g)
            r = quantiles(timer(lambda: [deployed(t, state) for t in parts], sync, 5, max(2, n_rep // 2)))
            res["tiles_per_part"][str(g * g)] = {"grid": g, "tile_px": list(parts[0].size), "native_px": list(q.size),
                                                 "sampling_px_per_side": INPUT_PX * g, **r}
            print(f"[p47-u6] {g}x{g} tiles: {r['p50_ms']:.2f} ms per part", flush=True)
    res.update({"end": stamp(), "gpu_freq_end": gpu_freq(), "max_temp_c_end": max_temp_c()})
    return res


def scaling(args) -> int:
    import numpy  # noqa: F401  (before torch, as the other runners)
    import torch
    from experiments.p26 import cost_profile as cp
    from experiments.p43.edge_measure import fused_setup, gpu_freq, quantiles, write_once
    target = OUT_ROOT / args.tag / "p47_u6_scaling.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    device = torch.device("cuda")
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    eng, prep, state, new_s = fused_setup(args.fused, supports, s_lab, n, bank_images, device)
    res = measure(eng, prep, supports[0], state, device, args.n_rep, quantiles, gpu_freq)
    res.update({"protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u6-d", "engine": str(args.fused),
                "torch": torch.__version__, "task": {"product": cp.PRODUCT, "K": cp.K_SHOT, "seed": cp.SEED,
                                                     "classes": n, "bank_rows": int(state[0].shape[0])},
                "new_task_s": new_s, "n_rep": args.n_rep})
    write_once(target, res)
    print(json.dumps({"classes_p50_ms": {k: v["p50_ms"] for k, v in res["classes_e2e"].items()},
                      "head_p50_ms": {k: (v["fp32"]["p50_ms"], v["fp16"]["p50_ms"]) for k, v in res["head_vs_M"].items()},
                      "tiles_p50_ms": {k: v["p50_ms"] for k, v in res["tiles_per_part"].items()}}, indent=1))
    return 0


def selftest() -> int:
    """The measurement code path with a stub engine on the CPU (shapes as the P41 engine; tiny repetitions)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from experiments.p43.edge_measure import quantiles

    class Stub:
        def __call__(self, x, bank, tau, protos):
            assert x.shape == (1, 3, INPUT_PX, INPUT_PX) and bank.shape[1] == DIM and tau.shape == (1,)
            n = protos["proto_g"].shape[0]
            c = F.normalize(torch.randn(1, PATCHES, DIM), dim=-1)
            return torch.randn(1, n), torch.randn(1), torch.randn(1, 1152), torch.randn(1, DIM), torch.randn(1, DIM), c

    def prep(image):
        assert image.size[0] > 0 and image.size[1] > 0
        return torch.zeros(1, 3, INPUT_PX, INPUT_PX)

    protos = {"mu_g": torch.zeros(1, 1152), "proto_g": F.normalize(torch.ones(3, 1152), dim=-1),
              "mu_a": torch.zeros(1, DIM), "proto_a": F.normalize(torch.ones(3, DIM), dim=-1),
              "mu_r": torch.zeros(1, DIM), "proto_r": F.normalize(torch.ones(3, DIM), dim=-1)}
    state = (F.normalize(torch.randn(PATCHES * 20, DIM), dim=-1), torch.tensor([0.2]), protos)
    res = measure(Stub(), prep, Image.new("RGB", (900, 900)), state, torch.device("cpu"), 2, quantiles,
                  lambda: None, bank_images=(1, 2))
    assert sorted(res["classes_e2e"], key=int) == [str(c) for c in CLASSES]
    assert res["head_vs_M"]["2"]["bank_rows"] == 2 * PATCHES
    assert [res["tiles_per_part"][k]["grid"] for k in ("1", "4", "9")] == [1, 2, 3]
    assert res["tiles_per_part"]["4"]["tile_px"] == [450, 450] and res["tiles_per_part"]["9"]["tile_px"] == [300, 300]
    json.dumps(res)
    print("[p47-u6] selftest ok", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--mvtec-root")
    ap.add_argument("--fused")
    ap.add_argument("--n-rep", type=int, default=200)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.scaling:
        if not (a.tag and a.mvtec_root and a.fused):
            raise SystemExit("--scaling needs --tag, --mvtec-root and --fused")
        return scaling(a)
    raise SystemExit("choose --scaling or --selftest")


if __name__ == "__main__":
    sys.exit(main())
