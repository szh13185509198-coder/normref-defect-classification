#!/usr/bin/env python3
"""P43 (report only): edge measurements of the TensorRT deployment (docs/P43_EDGE_MEASUREMENT_PROTOCOL.md).

On the device (PYTHONPATH = the tensorrt link directory):
    python -m experiments.p43.edge_measure --latency --tag <tag> --mvtec-root <root> --fused <engine>          # B1 + B2
    python -m experiments.p43.edge_measure --cold --run-index <i> --launch-epoch <t> --tag <tag> --mvtec-root <root> --fused <engine>
    python -m experiments.p43.edge_measure --energy --idle-seconds 30 --tag <tag> --mvtec-root <root>          # idle window
    python -m experiments.p43.edge_measure --energy --only <cfg> --tag <tag> --mvtec-root <root> --engines <dir> --fused <engine>
    python -m experiments.p43.edge_measure --extract-single --tag <tag> --mvtec-root <root> --engines <dir>    # C
Local:
    python -m experiments.p43.edge_measure --score-single --tag <tag> --mvtec-root <root>                      # C
    python -m experiments.p43.edge_measure --report-energy --results-dir <dir> --tegrastats <log>              # A (and P33)
"""

from __future__ import annotations

import time

T_PY = time.time()   # cold-start milestone: the interpreter reached this module (before numpy / torch)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / "ckpt" / "p43"
ENERGY = ("fused_engine_only", "fused_e2e", "gate_dino_pass", "gate_l18_pass", "trt_s336_engine", "trt_b336_engine")
SWITCH_PRODUCT = "capsule"   # protocol B2: class count 5 vs bottle's 3
SINGLE = ("trt_s336", "trt_b336")
REFERENCE = {"trt_s336": ("dinov2_s_r336_fp32", "arr_dinov2_s_r336_fp32"), "trt_b336": ("dinov2_b_r336_fp32", "arr_dinov2_b")}


def stamp() -> dict:
    return {"epoch": time.time(), "local": time.strftime("%Y-%m-%d %H:%M:%S")}


def write_once(path: Path, obj) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")


def quantiles(samples) -> dict:
    s = sorted(samples)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    med = statistics.median(s)
    return {"n": len(s), "mean_ms": statistics.fmean(s), "sd_ms": statistics.pstdev(s), "min_ms": s[0], "p50_ms": med,
            "p90_ms": q(0.90), "p95_ms": q(0.95), "p99_ms": q(0.99), "p999_ms": q(0.999), "max_ms": s[-1],
            "over_2x_median": sum(x > 2 * med for x in s)}


# --------------------------------------------------------------------------- fused engine helpers (P41)

def fused_setup(engine_path, supports, s_lab, n, bank_images, device):
    """Engine + task state exactly as experiments.p41.fused_engine.device_time builds them."""
    import torch
    from experiments.p41 import fused_engine as fe
    fe.DEVICE[0] = device
    eng = fe.FusedEngine(engine_path, device)
    run = lambda x, b, t, p: tuple(v.clone() for v in eng(x, b, t, p))  # noqa: E731
    prep = lambda im: fe.gpu_prep(im, device)  # noqa: E731
    with torch.no_grad():
        for _ in range(5):
            run(prep(supports[0]), torch.zeros(11520, 1024, device=device), torch.zeros(1, device=device),
                fe.dummy_protos(None, n))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bank, tau, protos, _ = fe.task_state(run, supports, s_lab, n, bank_images, prep)
        torch.cuda.synchronize()
    return eng, prep, (bank, tau, protos), time.perf_counter() - t0


def deployed(eng, prep, image, state):
    """= P41 `deployed`: decoded image -> (class index, screening score) on the host."""
    s, screen = eng(prep(image), *state)[:2]
    return int(s.argmax()), float(screen)


def task_inputs_for(product: str, mvtec_root: Path):
    """K = 5 seed-111 supports (stage-1 manifest; fallback: first five support-pool images per class) and the P41
    seed-111 bank draw of `product`."""
    from experiments.p14.resource_profile import load
    from experiments.p41.fused_engine import DRAWS
    from experiments.revision import revision_common as rc
    path = rc.STAGE1_BUNDLE / "p6b_support_manifests" / product / "k5_s111.json"
    if path.exists():
        manifest = rc.load_json(path)
        order = list(manifest["class_order"])
        items = [(c, it["rel_path"]) for c, cls in enumerate(order) for it in manifest["classes"][cls]["items"]]
        source = "stage-1 manifest k5_s111"
    else:
        from experiments.p10.extract_dev_features import support_pool_records
        pool = [r["canonical_relative_path"] for r in support_pool_records()[product]]
        order = sorted({p.split("/")[3] for p in pool})
        items = [(c, p) for c, cls in enumerate(order) for p in [x for x in pool if x.split("/")[3] == cls][:5]]
        source = "fallback: first five support-pool images per class"
    names = json.loads(DRAWS.read_text(encoding="utf-8"))[product]["111"]
    supports = [load(rc.DATASET_ROOT / rel) for _, rel in items]
    bank = [load(mvtec_root / product / "train" / "good" / x) for x in names]
    return supports, [c for c, _ in items], len(order), bank, source


# --------------------------------------------------------------------------- B1 + B2

def latency(args) -> int:
    import numpy as np
    import torch
    from experiments.p26 import cost_profile as cp
    device = torch.device("cuda")
    out = OUT_ROOT / args.tag
    target = out / "p43_latency.json"
    if target.exists() or (out / "p43_latency_samples.npy").exists():
        raise SystemExit(f"{target} or its samples exist (write-once)")
    out.mkdir(parents=True, exist_ok=True)
    sync = torch.cuda.synchronize
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    eng, prep, state_b, new_b = fused_setup(args.fused, supports, s_lab, n, bank_images, device)
    q = supports[0]
    with torch.no_grad():
        # B1: distribution
        for _ in range(50):
            deployed(eng, prep, q, state_b)
        sync()
        lat, start = [], stamp()
        for _ in range(args.n_latency):
            t0 = time.perf_counter()
            deployed(eng, prep, q, state_b)
            sync()
            lat.append((time.perf_counter() - t0) * 1000.0)
        dist = {**quantiles(lat), "start": start, "end": stamp()}
        np.save(out / "p43_latency_samples.npy", np.asarray(lat, dtype=np.float32))
        # B2: a second product through the same engine
        sup_c, lab_c, n_c, bank_c, source = task_inputs_for(SWITCH_PRODUCT, Path(args.mvtec_root))
        run = lambda x, b, t, p: tuple(v.clone() for v in eng(x, b, t, p))  # noqa: E731
        from experiments.p41 import fused_engine as fe
        sync()
        t0 = time.perf_counter()
        bank, tau, protos, _ = fe.task_state(run, sup_c, lab_c, n_c, bank_c, prep)
        sync()
        new_c = time.perf_counter() - t0
        state_c = (bank, tau, protos)
        files, load_ms = {}, {}
        for name, st in (("bottle", state_b), (SWITCH_PRODUCT, state_c)):
            f = out / f"p43_state_{name}.pt"
            if f.exists():
                raise SystemExit(f"{f} exists (write-once)")
            torch.save({"bank": st[0].cpu(), "tau": st[1].cpu(), "protos": {k: v.cpu() for k, v in st[2].items()}}, f)
            files[name] = f.stat().st_size

            def load_state(path=f):
                d = torch.load(path, map_location="cpu", weights_only=False)
                st2 = (d["bank"].to(device), d["tau"].to(device), {k: v.to(device) for k, v in d["protos"].items()})
                sync()
                return st2
            reps = []
            for _ in range(10):
                t0 = time.perf_counter()
                load_state()
                reps.append((time.perf_counter() - t0) * 1000.0)
            load_ms[name] = quantiles(reps)
        qc = sup_c[0]
        ref = {"bottle": deployed(eng, prep, q, state_b), SWITCH_PRODUCT: deployed(eng, prep, qc, state_c)}
        same, alt, alt_out = [], [], {"bottle": set(), SWITCH_PRODUCT: set()}
        for _ in range(200):
            t0 = time.perf_counter()
            deployed(eng, prep, q, state_b)
            sync()
            same.append((time.perf_counter() - t0) * 1000.0)
        for i in range(200):
            name, img, st = (("bottle", q, state_b) if i % 2 == 0 else (SWITCH_PRODUCT, qc, state_c))
            t0 = time.perf_counter()
            r = deployed(eng, prep, img, st)
            sync()
            alt.append((time.perf_counter() - t0) * 1000.0)
            alt_out[name].add(r)
        consistent = all(len({d for d, _ in alt_out[k]}) == 1 and next(iter(alt_out[k]))[0] == ref[k][0]
                         and max(abs(s - ref[k][1]) for _, s in alt_out[k]) <= 1e-5 for k in ref)
        if not consistent:
            raise SystemExit(f"decisions differ between same-product and alternating order: {ref} vs {alt_out}")
        res = {"protocol": "docs/P43_EDGE_MEASUREMENT_PROTOCOL.md#B", "engine": str(args.fused), "torch": torch.__version__,
               "B1_distribution_e2e": dist,
               "B2_switch": {"second_product": SWITCH_PRODUCT, "support_source": source, "classes": {"bottle": n, SWITCH_PRODUCT: n_c},
                             "new_task_s": {"bottle": new_b, SWITCH_PRODUCT: new_c}, "state_file_bytes": files,
                             "load_state_disk_to_gpu": load_ms, "same_product_e2e": quantiles(same),
                             "alternating_e2e": quantiles(alt), "decisions_identical": consistent,
                             "decisions": {k: list(v) for k, v in ref.items()}}}
    write_once(target, res)
    print(json.dumps({"p50": dist["p50_ms"], "p99": dist["p99_ms"], "max": dist["max_ms"],
                      "switch_same_p50": res["B2_switch"]["same_product_e2e"]["p50_ms"],
                      "switch_alt_p50": res["B2_switch"]["alternating_e2e"]["p50_ms"], "load_ms": {k: v["p50_ms"] for k, v in load_ms.items()},
                      "new_task_s": res["B2_switch"]["new_task_s"]}, indent=1))
    return 0


# --------------------------------------------------------------------------- B3

def cold(args) -> int:
    t = {"launch": args.launch_epoch, "module": T_PY}
    import numpy  # noqa: F401
    import torch
    import tensorrt  # noqa: F401
    t["imports"] = time.time()
    from experiments.p26 import cost_profile as cp
    from experiments.p41 import fused_engine as fe
    from experiments.p14.resource_profile import load
    from experiments.revision import revision_common as rc
    t["project_imports"] = time.time()
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    torch.cuda.synchronize()
    t["cuda_context"] = time.time()
    fe.DEVICE[0] = device
    eng = fe.FusedEngine(args.fused, device)
    t["engine_ready"] = time.time()
    d = torch.load(OUT_ROOT / args.tag / "p43_state_bottle.pt", map_location="cpu", weights_only=False)
    state = (d["bank"].to(device), d["tau"].to(device), {k: v.to(device) for k, v in d["protos"].items()})
    torch.cuda.synchronize()
    t["state_on_gpu"] = time.time()
    manifest = rc.load_json(rc.STAGE1_BUNDLE / "p6b_support_manifests" / cp.PRODUCT / f"k{cp.K_SHOT}_s{cp.SEED}.json")
    first = manifest["classes"][manifest["class_order"][0]]["items"][0]["rel_path"]
    image = load(rc.DATASET_ROOT / first)
    t["image_decoded"] = time.time()
    prep = lambda im: fe.gpu_prep(im, device)  # noqa: E731
    with torch.no_grad():
        decision = deployed(eng, prep, image, state)
        t["first_decision"] = time.time()
        steady = []
        for _ in range(50):
            t0 = time.perf_counter()
            deployed(eng, prep, image, state)
            torch.cuda.synchronize()
            steady.append((time.perf_counter() - t0) * 1000.0)
    keys = list(t)
    phases = {f"{a}->{b}_s": t[b] - t[a] for a, b in zip(keys, keys[1:]) if t[a] is not None}
    res = {"run_index": args.run_index, "epochs": t, "phases_s": phases,
           "launch_to_first_decision_s": (t["first_decision"] - t["launch"]) if t["launch"] else None,
           "module_to_first_decision_s": t["first_decision"] - t["module"], "decision": decision,
           "steady_e2e": quantiles(steady), "note": "page cache not dropped (no sudo): file reads are warm"}
    write_once(OUT_ROOT / args.tag / f"p43_cold_run{args.run_index}.json", res)
    print(json.dumps({"phases_s": phases, "launch_to_first_decision_s": res["launch_to_first_decision_s"]}, indent=1))
    return 0


# --------------------------------------------------------------------------- A

def energy_fn(cfg, args, device):
    """(callable, check) for one energy configuration; every callable is synchronised by the caller."""
    import torch
    import torch.nn.functional as F
    from experiments.p10 import run_pilot_a as ra
    from experiments.p15 import screening as sc
    from experiments.p26 import cost_profile as cp
    from experiments.p38 import trt_deploy as td
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    q = supports[0]
    if cfg.startswith("fused"):
        eng, prep, state, _ = fused_setup(args.fused, supports, s_lab, n, bank_images, device)
        if cfg == "fused_engine_only":
            x = prep(q)
            return (lambda: eng(x, *state)), {}
        return (lambda: deployed(eng, prep, q, state)), {}
    if cfg == "gate_dino_pass":   # = experiments.p38.trt_deploy.gate_dino, pass path
        pipe, prep = td.build_trt("trt_gate_dino", Path(args.engines), device)
        bank_d = torch.cat([pipe.features(im, with_dino=True)[2] for im in bank_images]).reshape(-1, 384)

        def pass_path(image):
            od = pipe.dino(prep(image))
            dn = F.normalize(od["L12_patches"].float(), dim=-1)
            best = torch.einsum("npd,bd->npb", dn, bank_d).max(dim=2).values
            return float(sc.image_score(1.0 - best, "top1pct")[0])
        ok = pass_path(q) == float(sc.image_score(ra.patch_anomaly(pipe.features(q, with_dino=True)[2], bank_d), "top1pct")[0])
    elif cfg == "gate_l18_pass":   # = experiments.p38.trt_deploy.gate_l18, pass path
        pipe, prep = td.build_trt("trt_gate_l18", Path(args.engines), device)
        bank = torch.cat([pipe.features(im)[1] for im in bank_images]).reshape(-1, 1024)

        def pass_path(image):
            o = pipe.engine(prep(image))
            c = F.normalize(o[pipe.key][:, 1:, :].float(), dim=-1)
            best = torch.einsum("npd,bd->npb", c, bank).max(dim=2).values
            return float(sc.image_score(1.0 - best, "top1pct")[0])
        ok = pass_path(q) == float(sc.image_score(ra.patch_anomaly(pipe.features(q)[1], bank), "top1pct")[0])
    else:   # trt_s336_engine / trt_b336_engine: engine forward only
        pipe, prep = td.build_trt(cfg.rsplit("_", 1)[0], Path(args.engines), device)
        x = prep(q)
        return (lambda: pipe.forward(x)), {}
    if not ok:
        raise SystemExit(f"{cfg}: screening score differs from the P15 formula")
    return (lambda: pass_path(q)), {"screen_score_equal_p15_formula": True}


def energy(args) -> int:
    out = OUT_ROOT / args.tag
    if args.idle_seconds > 0:
        from experiments.p26 import cost_profile as cp
        idle = {"start": stamp()}
        time.sleep(args.idle_seconds)
        idle["end"] = stamp()
        write_once(out / "p43_idle.json", {"idle": idle, "machine": cp.machine_info()})
        print(f"[p43] idle {args.idle_seconds:.0f} s recorded", flush=True)
        if not args.only:
            return 0
    import numpy  # noqa: F401
    import torch
    for cfg in [c for c in args.only.split(",") if c]:
        if cfg not in ENERGY:
            raise SystemExit(f"unknown configuration {cfg}")
        target = out / f"p43_energy_{cfg}.json"
        if target.exists():
            raise SystemExit(f"{target} exists (write-once)")
        device = torch.device("cuda")
        with torch.no_grad():
            fn, check = energy_fn(cfg, args, device)
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            n, start = 0, stamp()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < args.seconds:
                fn()
                torch.cuda.synchronize()
                n += 1
            elapsed = time.perf_counter() - t0
        res = {"schema_version": "p43_energy_v1", "block": cfg, "seconds": args.seconds, "torch": torch.__version__,
               "start": start, "end": stamp(), "forwards": n, "mean_ms": 1000 * elapsed / n, "check": check}
        write_once(target, res)
        print(f"[p43] {cfg}: {n} calls, {1000 * elapsed / n:.2f} ms each", flush=True)
    return 0


def report_energy(args) -> int:
    """Mean VDD_IN over each window of a P33 / P43 results directory, energy per call and dynamic energy."""
    samples = []
    for line in Path(args.tegrastats).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
        p = re.search(r"VDD_IN (\d+)mW", line)
        if m and p:
            samples.append((datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S"), int(p.group(1)) / 1000))
    d = Path(args.results_dir)

    def window(st):
        v = [w for t, w in samples if st["start"]["local"] <= t <= st["end"]["local"]]
        return (statistics.fmean(v), len(v)) if v else (None, 0)
    idle_file = next(iter(sorted(d.glob("p*_idle.json"))), None)
    idle_w = window(json.loads(idle_file.read_text(encoding="utf-8"))["idle"])[0] if idle_file else None
    rows = {}
    for f in sorted(d.glob("p3*_power_*.json")) + sorted(d.glob("p43_energy_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        w, k = window(r)
        rows[r["block"]] = {"mean_ms": r["mean_ms"], "calls": r["forwards"], "power_w": w, "samples": k,
                            "energy_j": w * r["mean_ms"] / 1000 if w else None,
                            "dynamic_energy_j": (w - idle_w) * r["mean_ms"] / 1000 if w and idle_w else None}
    res = {"results_dir": str(d), "tegrastats": str(args.tegrastats), "idle_power_w": idle_w, "rows": rows}
    print(json.dumps(res, indent=1))
    if args.out:
        write_once(Path(args.out), res)
    return 0


# --------------------------------------------------------------------------- C

def extract_single(args) -> int:
    import numpy  # noqa: F401
    import torch
    from experiments.p11 import formal_ab as fa
    from experiments.p18.run import c1_parts
    from experiments.p32 import device_extract as dx
    from experiments.p38 import trt_deploy as td
    from experiments.p41.fused_engine import DRAWS
    device = torch.device("cuda")
    out = OUT_ROOT / args.tag
    draws = json.loads(DRAWS.read_text(encoding="utf-8"))
    groups = dx.inputs(Path(args.mvtec_root))
    products = [x for x in args.products.split(",") if x] or list(groups)
    for block in SINGLE:
        existing = [p for p in products if (out / f"single_{block}_{p}.pt").exists()]
        if existing:
            raise SystemExit(f"{block}: files for {existing} exist (write-once)")
    out.mkdir(parents=True, exist_ok=True)
    for block in SINGLE:
        pipe, _ = td.build_trt(block, Path(args.engines), device)
        with torch.no_grad():
            for product in products:
                t0 = time.perf_counter()
                g = groups[product]

                def feats(items):
                    gl, pl = [], []
                    for _, p in items:
                        gg, pp = pipe.features(dx.open_rgb(p))
                        gl.append(gg.clone())
                        pl.append(pp.half().float())   # P22 stores patches in half precision (backbones.extract)
                    return torch.cat(gl), torch.cat(pl)
                g_sq, p_sq = feats(g["support"] + g["queries"])
                _, p_n = feats(g["normals"])
                name_index = {name: i for i, (name, _) in enumerate(g["normals"])}
                res = {"block": block, "layer": pipe.layer, "support_rel_paths": [i for i, _ in g["support"]],
                       "query_rgb_sha256": [i for i, _ in g["queries"]], "g": g_sq.cpu(), "seeds": {}}
                for seed, names in draws[product].items():
                    bank = p_n[torch.as_tensor([name_index[x] for x in names], device=device)]
                    tau = fa.tau99(bank)
                    _, soft, rr = c1_parts(p_sq, bank, tau)[0]
                    res["seeds"][seed] = {"tau": float(tau), "soft": soft.cpu(), "rr": rr.cpu()}
                torch.save(res, out / f"single_{block}_{product}.pt")
                del g_sq, p_sq, p_n
                torch.cuda.empty_cache()
                print(f"[p43] {block} {product}: {time.perf_counter() - t0:.0f} s", flush=True)
        del pipe
        torch.cuda.empty_cache()
    print("[p43] single-backbone extraction done", flush=True)
    return 0


def score_single(args) -> int:
    import numpy  # noqa: F401
    import torch
    import torch.nn.functional as F
    from experiments.p6.p6b_stage3_common import load_query_contract
    from experiments.p10 import run_pilot as rp
    from experiments.p10 import run_pilot_a as ra
    from experiments.p10.formal_evaluate import MANIFEST_DIR
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.p12.threshold_controls import exact_equal
    from experiments.p14 import final_formal as ff
    from experiments.p18 import backbones as bb
    from experiments.p18.run import c1_parts, open_rgb
    from experiments.p22 import lightweight as lw
    from experiments.revision import revision_common as rc
    from experiments.revision.c1_paired_bootstrap import contrast
    src = OUT_ROOT / args.tag
    target = src / "p43_single_scores.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    device = torch.device("cuda")
    sel = lw.rc.load_json(lw.OUT["dev"] / "p22_dev_results.json")["selection"]
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    by_key = {(q["product"], q["rgb_sha256"]): q for q in queries}
    rows, sink, cos = [], [], {}
    for block, (cfg, ref_method) in REFERENCE.items():
        backbone, layer = lw.build(cfg, device), sel[cfg]
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            trt = torch.load(src / f"single_{block}_{product}.pt", map_location="cpu", weights_only=False)
            shas = list(qf["rgb_sha256"])
            if trt["support_rel_paths"] != list(dev["rel_paths"]) or trt["query_rgb_sha256"] != shas or trt["layer"] != layer:
                raise SystemExit(f"{block} {product}: image order or layer differs")
            images = ([open_rgb(rc.DATASET_ROOT / rel) for rel in dev["rel_paths"]] +
                      [open_rgb(by_key[(product, sha)]["image_path"]) for sha in shas])
            normals = [open_rgb(Path(args.mvtec_root) / product / "train" / "good" / x) for x in nf["names"]]
            gb, pb = bb.extract(backbone, images, device, 4, layers=(layer,))      # = P22 (batch 4)
            _, pn = bb.extract(backbone, normals, device, 4, layers=(layer,))
            g_ref, p_ref = gb.to(device), pb[layer].to(device).float()
            g_trt = trt["g"].to(device)
            name_index = {x: i for i, x in enumerate(nf["names"])}
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            qi = torch.arange(n_pool, n_pool + len(shas), device=device)
            q_true = [by_key[(product, s)]["true_label"] for s in shas]
            cos.setdefault(block, {})[product] = {"g": float(F.cosine_similarity(g_trt, g_ref).mean())}
            with torch.no_grad():
                for seed in rc.SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank = pn[layer][idx].to(device).float()
                    parts_ref = c1_parts(p_ref, bank, fa.tau99(bank))[0]
                    ts = trt["seeds"][str(seed)]
                    parts_trt = (None, ts["soft"].to(device), ts["rr"].to(device))
                    cos[block][product][str(seed)] = {k: float(F.cosine_similarity(a, b).mean()) for k, a, b in
                                                      (("soft", parts_trt[1], parts_ref[1]), ("rr", parts_trt[2], parts_ref[2]))}
                    for k_shot in rc.K_VALUES:
                        manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                        order = list(manifest["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in manifest["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        si, n = torch.as_tensor(s_idx, device=device), len(order)
                        dummy = torch.zeros(len(qi), n, device=device)
                        sc = {ref_method: ff.episode_scores(g_ref, {"": parts_ref}, si, s_lab, qi, n, dummy, dummy)[ff.ARR],
                              f"{ref_method}|trt": ff.episode_scores(g_trt, {"": (parts_ref[0],) + parts_trt[1:]}, si, s_lab,
                                                                     qi, n, dummy, dummy)[ff.ARR]}
                        rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed}, sink, shas)
            print(f"[p43] {block} {product} scored", flush=True)
        del backbone
        torch.cuda.empty_cache()
    p22 = rc.load_json(PROJECT_ROOT / "ckpt" / "p22" / "formal" / "mvtec" / "p22_formal_mvtec_results.json")["raw_metrics"]
    gates = {m: exact_equal(rows, p22, (m,), ("product", "K", "seed")) for _, m in REFERENCE.values()}
    if any(v >= 1e-9 for v in gates.values()):
        raise SystemExit(f"reference does not reproduce P22: {gates}")
    grid = dict(rc.metric_grid_from_rows(rows))
    preds = {(u["method"], u["product"], u["K"], u["seed"]): u["pred"] for u in sink}
    result = {"schema_version": "p43_single_v1", "protocol": "docs/P43_EDGE_MEASUREMENT_PROTOCOL.md#C",
              "gates_reference_vs_p22": gates, "comparisons": {}, "levels_ba_pct": {}, "descriptor_cosine": cos}
    for block, (_, m) in REFERENCE.items():
        t = f"{m}|trt"
        c = contrast(grid, m, "balanced_accuracy", target=t)
        same = total = 0
        for (meth, p, k, s), pred in preds.items():
            if meth == t:
                ref = preds[(m, p, k, s)]
                same += sum(x == y for x, y in zip(pred, ref))
                total += len(pred)
        d, ci = 100 * c["Dbar"], [100 * x for x in c["Dbar_ci95"]]
        result["comparisons"][block] = {"per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()}, "Dbar_pp": d,
                                        "ci95_pp": ci, "positive_products": c["Dbar_positive_products"],
                                        "agreement": same / total,
                                        "decision": "keeps accuracy" if abs(d) < 0.5 and ci[0] <= 0 <= ci[1]
                                        else "report TensorRT accuracy"}
        for meth in (m, t):
            s = rc.product_macro_summary(grid, meth)
            result["levels_ba_pct"][meth] = {str(k): 100 * s[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES}
    result.update({"provenance": rc.git_provenance(), "raw_metrics": rows})
    rc.write_json(target, result)
    print(json.dumps({k: result[k] for k in ("gates_reference_vs_p22", "levels_ba_pct", "comparisons")}, indent=1))
    return 0


# --------------------------------------------------------------------------- A2: repeat phases, decode, other power modes

def gpu_freq():
    for p in ("/sys/class/devfreq/17000000.gpu/cur_freq", "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"):
        try:
            return int(Path(p).read_text().strip())
        except OSError:
            continue
    return None


def repeat(args) -> int:
    """A2-B4: e2e (no inner syncs) and split timings (preprocess / engine / host read-back, synchronised between) in
    phases of one fresh process: A after setup, B again, C build a second task state, D again, E with the bottle state
    loaded from disk (the B2 file)."""
    import torch
    from experiments.p26 import cost_profile as cp
    from experiments.p41 import fused_engine as fe
    device = torch.device("cuda")
    target = OUT_ROOT / args.tag / f"p43_repeat_{args.label}.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    sync = torch.cuda.synchronize
    supports, s_lab, n, bank_images, _, _ = cp.task_inputs(Path(args.mvtec_root))
    eng, prep, state, new_s = fused_setup(args.fused, supports, s_lab, n, bank_images, device)
    q = supports[0]

    def measure(st):
        e2e, pre, en, rb = [], [], [], []
        for _ in range(50):
            deployed(eng, prep, q, st)
        sync()
        f0 = gpu_freq()
        for _ in range(args.n_phase):
            t0 = time.perf_counter()
            deployed(eng, prep, q, st)
            sync()
            e2e.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(args.n_phase):
            t0 = time.perf_counter()
            x = prep(q)
            sync()
            t1 = time.perf_counter()
            s, screen = eng(x, *st)[:2]
            sync()
            t2 = time.perf_counter()
            int(s.argmax()), float(screen)
            t3 = time.perf_counter()
            pre.append((t1 - t0) * 1000.0)
            en.append((t2 - t1) * 1000.0)
            rb.append((t3 - t2) * 1000.0)
        return {"e2e": quantiles(e2e), "split_median_ms": {"preprocess": statistics.median(pre), "engine": statistics.median(en),
                                                           "readback": statistics.median(rb)},
                "gpu_freq_start": f0, "gpu_freq_end": gpu_freq(), "local": stamp()["local"]}
    res = {"label": args.label, "phases": {}, "new_task_s": new_s, "engine": str(args.fused)}
    with torch.no_grad():
        for ph in args.phases:
            if ph in "ABD":
                res["phases"][ph] = measure(state)
            elif ph == "C":
                sup_c, lab_c, n_c, bank_c, _ = task_inputs_for(SWITCH_PRODUCT, Path(args.mvtec_root))
                run = lambda x, b, t, p: tuple(v.clone() for v in eng(x, b, t, p))  # noqa: E731
                t0 = time.perf_counter()
                fe.task_state(run, sup_c, lab_c, n_c, bank_c, prep)
                sync()
                res["phases"]["C"] = {"new_task_s": time.perf_counter() - t0}
            elif ph == "E":
                d = torch.load(OUT_ROOT / args.state_tag / "p43_state_bottle.pt", map_location="cpu", weights_only=False)
                loaded = (d["bank"].to(device), d["tau"].to(device), {k: v.to(device) for k, v in d["protos"].items()})
                res["phases"]["E"] = measure(loaded)
            print(f"[p43] repeat {args.label} phase {ph}: "
                  + (f"{res['phases'][ph]['e2e']['p50_ms']:.2f} ms" if "e2e" in res["phases"][ph] else "state built"), flush=True)
    write_once(target, res)
    return 0


def decode(args) -> int:
    """A2-B5: CPU time to read and decode the formal query images (PIL, as the pipeline loads them); three reads each,
    the first one after whatever the page cache holds (no sudo)."""
    from experiments.p14.resource_profile import load
    from experiments.p32 import device_extract as dx
    target = OUT_ROOT / args.tag / "p43_decode.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    groups = dx.inputs(Path(args.mvtec_root))
    per, first, rest = {}, [], []
    for product, g in groups.items():
        times, size = [], None
        for _, path in g["queries"]:
            r = []
            for _ in range(3):
                t0 = time.perf_counter()
                im = load(path)
                r.append((time.perf_counter() - t0) * 1000.0)
            size = im.size
            first.append(r[0])
            rest += r[1:]
            times.append(statistics.median(r[1:]))
        per[product] = {"images": len(times), "size": size, "median_ms": statistics.median(times)}
    res = {"per_product": per, "first_read": quantiles(first), "repeat_reads": quantiles(rest)}
    write_once(target, res)
    print(json.dumps({k: (v["median_ms"], v["size"]) for k, v in per.items()}), flush=True)
    return 0


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--latency", "--cold", "--energy", "--extract-single", "--score-single", "--report-energy", "--repeat",
                 "--decode"):
        ap.add_argument(flag, action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--mvtec-root")
    ap.add_argument("--fused")
    ap.add_argument("--engines")
    ap.add_argument("--only", default="")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--idle-seconds", type=float, default=0.0)
    ap.add_argument("--n-latency", type=int, default=10000)
    ap.add_argument("--run-index", type=int, default=1)
    ap.add_argument("--launch-epoch", type=float)
    ap.add_argument("--products", default="")
    ap.add_argument("--results-dir")
    ap.add_argument("--tegrastats")
    ap.add_argument("--out")
    ap.add_argument("--label", default="r1")
    ap.add_argument("--phases", default="ABCDE")
    ap.add_argument("--n-phase", type=int, default=2000)
    ap.add_argument("--state-tag", default="jetson_orin_nano")
    a = ap.parse_args(argv)
    for flag, fn in (("latency", latency), ("cold", cold), ("energy", energy), ("extract_single", extract_single),
                     ("score_single", score_single), ("report_energy", report_energy),
                     ("repeat", repeat), ("decode", decode)):
        if getattr(a, flag):
            return fn(a)
    raise SystemExit("choose a mode")


if __name__ == "__main__":
    sys.exit(main())
