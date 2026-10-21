#!/usr/bin/env python3
"""P47 amendment A4: P46 screening with the 2 x 2 tiled engine (docs/P47_SCALING_AND_TILING_PROTOCOL.md#A4).

Device:  python -m experiments.p47.screen_tiled --ext <dir holding EXT/> --engine <p47_g2.engine> --tag <tag>
Local:   python -m experiments.p46.screen_device --score --tag <tag>        (unchanged P46 scoring)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402,F401
import torch  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p46"   # P46's layout, so its scoring applies unchanged
GRID = 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ext", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    from experiments.p14.resource_profile import load
    from experiments.p41 import fused_engine as fe
    from experiments.p47 import scale_tile as st
    root = Path(a.ext)
    units = json.loads((root / "EXT" / "P46_UNITS.json").read_text(encoding="utf-8"))["tasks"]
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    eng = st.Engine(a.engine, device)
    dummy = fe.dummy_protos(None, 2)
    zero_bank = torch.zeros(GRID * GRID * st.P * st.TILE_M[GRID], st.D, device=device)
    zero_tau = torch.zeros(1, device=device)

    def inputs(p):
        im = load(root / p)
        return fe.gpu_prep(im, device), st.gpu_tiles(im, GRID, device)
    with torch.no_grad():
        for task, u in units.items():
            f = out / f"p46_{task}.json"
            if f.exists():
                raise SystemExit(f"{f} exists (write-once)")
            t0 = time.perf_counter()
            banks = {}
            for seed, bank_paths in u["banks"].items():
                if len(bank_paths) != st.TILE_M[GRID]:
                    raise SystemExit(f"{task} {seed}: bank of {len(bank_paths)} images")
                toks = [eng(st.feed_tiled(*inputs(p), zero_bank, zero_tau, dummy))["tokens"].clone() for p in bank_paths]
                banks[seed] = torch.cat(toks).contiguous()
            paths = list(dict.fromkeys(list(u["calibration"]) + [q["path"] for q in u["queries"]]))
            score = {s: {} for s in banks}
            for p in paths:
                full, tiles = inputs(p)
                for s, b in banks.items():
                    score[s][p] = float(eng(st.feed_tiled(full, tiles, b, zero_tau, dummy))["screen"])
            res = {"task": task, "grid": GRID, "query_sha256": [q["sha256"] for q in u["queries"]],
                   "query_is_normal": [q["is_normal"] for q in u["queries"]],
                   "seeds": {s: {"calibration": [score[s][p] for p in u["calibration"]],
                                 "queries": [score[s][q["path"]] for q in u["queries"]]} for s in banks}}
            f.write_text(json.dumps(res) + "\n", encoding="utf-8")
            del banks
            torch.cuda.empty_cache()
            print(f"[p47-A4] {task}: {len(paths)} images, {time.perf_counter() - t0:.0f} s", flush=True)
    print("[p47-A4] extraction done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
