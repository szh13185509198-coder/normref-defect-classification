#!/usr/bin/env python3
"""P48 (report only): patch-level false selection on the deployed engine (docs/P48_DEPLOYED_PATCH_CALIBRATION_PROTOCOL.md).

Local:   python -m experiments.p48.patch_calib --reference                # R + tau_ref -> ckpt/p48/p48_reference.json
Device:  python -m experiments.p48.patch_calib --device --mvtec-root <root> --engine <main_fused_r8.engine> --tag <tag>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402,F401  (before torch, as P41)
import torch  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p48"
REF = OUT / "p48_reference.json"
M = 20


def rates(bank_imgs, held, tau, rows: int = 1024) -> float:
    """Fraction of held-out patches with a = 1 - max cos to the bank above tau."""
    bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
    over = total = 0
    for img in held:
        for blk in img.split(rows):
            a = 1.0 - (blk @ bank.t()).max(dim=1).values
            over += int((a > tau).sum())
            total += a.numel()
    return over / total


def reference(args) -> int:
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.revision import revision_common as rc
    if REF.exists():
        raise SystemExit(f"{REF} exists")
    device = torch.device("cpu")   # the laptop GPU is busy with P47; same arithmetic
    torch.set_num_threads(8)
    draws = json.loads((PROJECT_ROOT / "ckpt" / "p41" / "bank_draws.json").read_text(encoding="utf-8"))
    res = {"schema_version": "p48_reference_v1", "protocol": "docs/P48_DEPLOYED_PATCH_CALIBRATION_PROTOCOL.md",
           "products": {}}
    with torch.no_grad():
        for product in rc.PRODUCTS:
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            names = list(nf["names"])
            toks = nf["clip18_patches"].float()
            idx = {n: i for i, n in enumerate(names)}
            per = {}
            for seed, bank_names in draws[product].items():
                if list(nf["draws"][seed][:M]) != list(bank_names):
                    raise SystemExit(f"{product} {seed}: P41 draw differs from the P10 draw")
                bank = toks[[idx[n] for n in bank_names]].to(device)
                held = [toks[idx[n]].to(device) for n in names if n not in set(bank_names)]
                tau = fa.tau99(bank)
                per[seed] = {"bank": list(bank_names), "tau_ref": tau, "rate_R": rates(bank, held, tau), "held": len(held)}
            res["products"][product] = {"names": names, "seeds": per}
            print(f"[p48] {product}: R " + " ".join(f"{v['rate_R']:.4f}" for v in per.values()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REF.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def device(args) -> int:
    from experiments.p41 import fused_engine as fe
    from experiments.p47.scale_tile import loo_tau
    from experiments.p32 import device_extract as dx
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    target = out / "p48_device.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    ref = json.loads(REF.read_text(encoding="utf-8"))
    dev = torch.device("cuda")
    fe.DEVICE[0] = dev
    eng = fe.FusedEngine(args.engine, dev)
    zero_bank, zero_tau, dummy = torch.zeros(11520, 1024, device=dev), torch.zeros(1, device=dev), fe.dummy_protos(None, 2)
    root = Path(args.mvtec_root)
    res = {"schema_version": "p48_device_v1", "engine": str(args.engine), "products": {}}
    with torch.no_grad():
        for product, p in ref["products"].items():
            t0 = time.perf_counter()
            toks = {n: eng(fe.gpu_prep(dx.open_rgb(root / product / "train" / "good" / n), dev), zero_bank, zero_tau,
                           dummy)[5].clone() for n in p["names"]}
            per = {}
            for seed, r in p["seeds"].items():
                bank = torch.stack([toks[n] for n in r["bank"]])
                held = [toks[n] for n in p["names"] if n not in set(r["bank"])]
                tau_eng = loo_tau(bank)
                per[seed] = {"tau_eng": tau_eng, "tau_ref": r["tau_ref"], "rate_E": rates(bank, held, tau_eng),
                             "rate_X": rates(bank, held, r["tau_ref"]), "rate_R": r["rate_R"], "held": len(held)}
            res["products"][product] = per
            del toks
            torch.cuda.empty_cache()
            print(f"[p48] {product}: {time.perf_counter() - t0:.0f} s, E " + " ".join(f"{v['rate_E']:.4f}" for v in per.values())
                  + " | X " + " ".join(f"{v['rate_X']:.4f}" for v in per.values()), flush=True)
    agg = {}
    for key in ("rate_R", "rate_E", "rate_X"):
        prod = {pr: statistics.fmean(v[key] for v in s.values()) for pr, s in res["products"].items()}
        agg[key] = {"macro": statistics.fmean(prod.values()), "min_product": min(prod.values()), "max_product": max(prod.values()),
                    "per_product": prod}
    agg["tau_eng_minus_ref_mean"] = statistics.fmean(v["tau_eng"] - v["tau_ref"] for s in res["products"].values() for v in s.values())
    res["aggregate"] = agg
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ("macro", "min_product", "max_product")})
                      for k, v in agg.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--mvtec-root")
    ap.add_argument("--engine")
    ap.add_argument("--tag", default="jetson_orin_nano")
    a = ap.parse_args(argv)
    return reference(a) if a.reference else device(a) if a.device else 1


if __name__ == "__main__":
    sys.exit(main())
