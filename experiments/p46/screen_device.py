#!/usr/bin/env python3
"""P46 (report only): the screening guarantee with the deployed fused TensorRT engine
(docs/P46_DEVICE_SCREENING_PROTOCOL.md).

On the device:  python -m experiments.p46.screen_device --extract --ext <dir holding EXT/> --fused <engine> --tag <tag>
Local:          python -m experiments.p46.screen_device --score --tag <tag>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "ckpt" / "p46"
ALPHAS = ("0.01", "0.02", "0.05")
P15 = PROJECT_ROOT / "ckpt" / "p15" / "external" / "p15_external_results.json"
METHOD = "fused_adareg_regres_soft_q99"


def extract(args) -> int:
    import numpy  # noqa: F401
    import torch
    from experiments.p14.resource_profile import load
    from experiments.p41 import fused_engine as fe
    root = Path(args.ext)
    units = json.loads((root / "EXT" / "P46_UNITS.json").read_text(encoding="utf-8"))["tasks"]
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    eng = fe.FusedEngine(args.fused, device)
    dummy = fe.dummy_protos(None, 2)
    zero_bank, zero_tau = torch.zeros(11520, 1024, device=device), torch.zeros(1, device=device)
    with torch.no_grad():
        for task, u in units.items():
            f = out / f"p46_{task}.json"
            if f.exists():
                raise SystemExit(f"{f} exists (write-once)")
            t0 = time.perf_counter()
            paths = sorted({p for v in u["banks"].values() for p in v} | set(u["calibration"]) | {q["path"] for q in u["queries"]})
            x = {p: fe.gpu_prep(load(root / p), device) for p in paths}
            res = {"task": task, "query_sha256": [q["sha256"] for q in u["queries"]],
                   "query_is_normal": [q["is_normal"] for q in u["queries"]], "seeds": {}}
            for seed, bank_paths in u["banks"].items():
                toks = torch.cat([eng(x[p], zero_bank, zero_tau, dummy)[5].clone() for p in bank_paths])
                bank = toks.reshape(-1, 1024).contiguous()

                def sc(p):
                    return float(eng(x[p], bank, zero_tau, dummy)[1])
                res["seeds"][seed] = {"calibration": [sc(p) for p in u["calibration"]],
                                      "queries": [sc(q["path"]) for q in u["queries"]]}
            f.write_text(json.dumps(res) + "\n", encoding="utf-8")
            del x
            torch.cuda.empty_cache()
            print(f"[p46] {task}: {len(paths)} images, {time.perf_counter() - t0:.0f} s", flush=True)
    print("[p46] extraction done", flush=True)
    return 0


def auroc(normal, defect) -> float:
    """P(defect score > normal score) + 0.5 P(tie), by ranks."""
    s = sorted([(v, 0) for v in normal] + [(v, 1) for v in defect])
    ranks, i = [0.0] * len(s), 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1][0] == s[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r_def = sum(r for r, (_, lab) in zip(ranks, s) if lab == 1)
    n1, n0 = len(defect), len(normal)
    return (r_def - n1 * (n1 + 1) / 2) / (n1 * n0)


def tau(cal, alpha) -> float:
    s = sorted(cal)
    k = math.ceil((len(s) + 1) * (1 - float(alpha)))
    return float("inf") if k > len(s) else s[k - 1]


def score(args) -> int:
    src = OUT / args.tag
    target = src / "p46_results.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    p15 = json.loads(P15.read_text(encoding="utf-8"))
    ref = {(u["task"], str(u["seed"])): u for u in p15["units"] if u["K"] == 1}
    rows = []
    for f in sorted(src.glob("p46_*__*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        normal = d["query_is_normal"]
        for seed, v in d["seeds"].items():
            q = v["queries"]
            r = ref[(d["task"], seed)]
            row = {"task": d["task"], "seed": seed,
                   "auroc": auroc([s for s, n in zip(q, normal) if n], [s for s, n in zip(q, normal) if not n]),
                   "auroc_p15": r["auroc"]}
            for a in ALPHAS:
                t = tau(v["calibration"], a)
                row[f"tau_{a}"], row[f"tau_{a}_p15"] = t, r["taus"][a]
                row[f"frr_{a}"] = sum(s > t for s, n in zip(q, normal) if n) / sum(normal)
                row[f"frr_{a}_p15"] = r["two_level"][METHOD][a]["false_reject_rate"]
            rows.append(row)
    agg = {}
    for ds in ("visa", "mtd"):
        sel = [r for r in rows if r["task"].startswith(ds)]
        seeds = sorted({r["seed"] for r in sel})

        def macro(key):
            return statistics.fmean(statistics.fmean(r[key] for r in sel if r["seed"] == s) for s in seeds)
        keys = [k for k in rows[0] if k not in ("task", "seed")]
        agg[ds] = {"tasks": len({r["task"] for r in sel}), "units": len(sel), **{k: macro(k) for k in keys},
                   "max_abs_auroc_diff": max(abs(r["auroc"] - r["auroc_p15"]) for r in sel),
                   **{f"guarantee_{a}": macro(f"frr_{a}") <= float(a) for a in ALPHAS}}
    res = {"schema_version": "p46_v1", "protocol": "docs/P46_DEVICE_SCREENING_PROTOCOL.md", "aggregate": agg, "units": rows}
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--ext")
    ap.add_argument("--fused")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    return extract(a) if a.extract else score(a) if a.score else 1


if __name__ == "__main__":
    sys.exit(main())
