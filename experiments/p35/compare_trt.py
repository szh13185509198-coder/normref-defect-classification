#!/usr/bin/env python3
"""P35 (report only): compare the TensorRT FP16 engine outputs (trtexec --exportOutput) with the PyTorch FP32
reference of the same input: per output, max |d| and the cosine similarity per token / vector (the features the method
uses are L2-normalised tokens, so cosine is the relevant measure).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p35.compare_trt --dir D:\\intergration\\p35_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, type=Path)
    a = ap.parse_args(argv)
    report = {}
    for out_json in sorted(a.dir.glob("*_trt_out.json")):
        cfg = out_json.name[:-len("_trt_out.json")]
        base = cfg   # run suffixes (_r2, _r3, _fp32b) and the baked-position export (_pb) share the reference
        while not (a.dir / f"{base}_ref.npz").exists() and "_" in base and (base.rsplit("_", 1)[1] in ("fp32b", "pb", "ex", "dbg") or base.rsplit("_", 1)[1][1:].isdigit() and base.rsplit("_", 1)[1][0] == "r"):
            base = base.rsplit("_", 1)[0]
        ref = np.load(a.dir / f"{base}_ref.npz")
        # trtexec --exportOutput: [{"name": ..., "dimensions": "1x576x384", "values": [...]}, ...]
        trt = {o["name"]: np.asarray(o["values"], dtype=np.float32).reshape([int(d) for d in str(o["dimensions"]).split("x")])
               for o in json.loads(out_json.read_text(encoding="utf-8"))}
        rows = {}
        for name in ref.files:
            r, t = ref[name].astype(np.float32), trt[name].reshape(ref[name].shape)
            r2, t2 = r.reshape(-1, r.shape[-1]), t.reshape(-1, t.shape[-1])
            cos = (r2 * t2).sum(-1) / (np.linalg.norm(r2, axis=-1) * np.linalg.norm(t2, axis=-1) + 1e-12)
            rows[name] = {"max_abs": float(np.abs(r - t).max()), "mean_cos": float(cos.mean()), "min_cos": float(cos.min())}
        report[cfg] = rows
        print(cfg, json.dumps(rows))
    (a.dir / "p35_trt_vs_torch.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
