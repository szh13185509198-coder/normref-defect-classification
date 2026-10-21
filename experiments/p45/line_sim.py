#!/usr/bin/env python3
"""P45 (report only): production-line simulation with the single fused TensorRT engine
(docs/P45_LINE_SIMULATION_PROTOCOL.md).

On the device (PYTHONPATH = the tensorrt link directory; tegrastats logging started separately):
    python -m experiments.p45.line_sim --idle 30 --tag <tag>
    python -m experiments.p45.line_sim --run frame_seq|file_seq|file_pipe --tag <tag> --mvtec-root <root> --fused <engine>
Local:
    python -m experiments.p45.line_sim --report --tag <tag> --tegrastats <log>
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import re
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "ckpt" / "p45"
RUNS = ("frame_seq", "file_seq", "file_pipe")
ROUNDS = 3
A_REJECT, A_ACCEPT = 0.05, 0.30
QUEUE_DEPTH = 4


def stamp() -> dict:
    return {"epoch": time.time(), "local": time.strftime("%Y-%m-%d %H:%M:%S")}


def write_once(path: Path, obj) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")


def conformal(scores, alpha) -> float:
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else s[k - 1]


def quant(xs) -> dict:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "mean_ms": statistics.fmean(s), "p50_ms": statistics.median(s), "p90_ms": q(0.9),
            "p99_ms": q(0.99), "max_ms": s[-1]}


def prep_array(arr, device):
    """= experiments.p41.fused_engine.gpu_prep applied to np.asarray(image.convert("RGB")) (checked at start)."""
    import torch
    import torch.nn.functional as F
    from experiments.p41.fused_engine import CLIP_MEAN, CLIP_STD
    x = torch.from_numpy(arr).to(device, non_blocking=True)
    x = x.permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(336, 336), mode="bicubic", antialias=True, align_corners=False).round().clamp(0, 255) / 255.0
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def idle(args) -> int:
    rec = {"start": stamp()}
    time.sleep(args.idle)
    rec["end"] = stamp()
    write_once(OUT / args.tag / "p45_idle.json", rec)
    print(f"[p45] idle {args.idle:.0f} s", flush=True)
    return 0


def run(args) -> int:
    import resource
    import numpy as np
    import torch
    from experiments.p14.resource_profile import load
    from experiments.p32 import device_extract as dx
    from experiments.p41 import fused_engine as fe
    from experiments.p43.edge_measure import task_inputs_for
    from experiments.revision import revision_common as rc
    mode = args.run
    target = OUT / args.tag / f"p45_{mode}.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    sync = torch.cuda.synchronize
    groups = dx.inputs(Path(args.mvtec_root))
    draws = json.loads(fe.DRAWS.read_text(encoding="utf-8"))
    eng = fe.FusedEngine(args.fused, device)
    clone_run = lambda x, b, t, p: tuple(v.clone() for v in eng(x, b, t, p))  # noqa: E731
    prep_pil = lambda im: prep_array(np.asarray(im.convert("RGB")), device)  # noqa: E731

    # check: the array preprocessing equals P41 gpu_prep exactly
    probe = load(groups[rc.PRODUCTS[0]]["queries"][0][1])
    if not torch.equal(prep_pil(probe), fe.gpu_prep(probe, device)):
        raise SystemExit("prep_array differs from fused_engine.gpu_prep")
    with torch.no_grad():
        for _ in range(5):
            clone_run(prep_pil(probe), torch.zeros(11520, 1024, device=device), torch.zeros(1, device=device),
                      fe.dummy_protos(None, 3))
        sync()

    def gpu_part(arr, state, thr):
        s, screen = eng(prep_array(arr, device), *state)[:2]
        cls, sc = int(s.argmax()), float(screen)
        return cls, ("reject" if sc > thr[0] else "accept" if sc <= thr[1] else "review")

    lines, all_lat, windows = {}, [], []
    total_parts, total_stream_s = 0, 0.0
    with torch.no_grad():
        for product in rc.PRODUCTS:
            g = groups[product]
            # changeover: decode the supports and bank images, build the task state through the engine
            t0 = time.perf_counter()
            supports, s_lab, n, bank_imgs, source = task_inputs_for(product, Path(args.mvtec_root))
            t1 = time.perf_counter()
            bank, tau, protos, _ = fe.task_state(clone_run, supports, s_lab, n, bank_imgs, prep_pil)
            sync()
            t2 = time.perf_counter()
            state = (bank, tau, protos)
            in_bank = set(draws[product]["111"])
            held = sorted(name for name, _ in g["normals"] if name not in in_bank)
            paths = dict(g["normals"])
            calib, stream_normals = held[0::2], held[1::2]
            cal = [float(eng(prep_pil(load(paths[x])), *state)[1]) for x in calib]
            thr = (conformal(cal, A_REJECT), conformal(cal, A_ACCEPT))
            base = [paths[x] for x in stream_normals] + [p for _, p in g["queries"]]
            items = []
            for r in range(ROUNDS):
                order = list(base)
                random.Random(f"p45|{product}|{r}").shuffle(order)
                items += order
            frames = {p: np.asarray(load(p).convert("RGB")) for p in base} if mode == "frame_seq" else None
            sync()
            lat, decisions = [], {"reject": 0, "accept": 0, "review": 0}
            win = {"product": product, "start": stamp()}
            ts = time.perf_counter()
            if mode in ("frame_seq", "file_seq"):
                for p in items:
                    t_part = time.perf_counter()
                    arr = frames[p] if frames is not None else np.asarray(load(p).convert("RGB"))
                    _, d = gpu_part(arr, state, thr)
                    lat.append((time.perf_counter() - t_part) * 1000.0)
                    decisions[d] += 1
            else:
                q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)

                def producer():
                    for p in items:
                        t_part = time.perf_counter()
                        q.put((t_part, np.asarray(load(p).convert("RGB"))))
                    q.put(None)
                th = threading.Thread(target=producer, daemon=True)
                th.start()
                while True:
                    it = q.get()
                    if it is None:
                        break
                    t_part, arr = it
                    _, d = gpu_part(arr, state, thr)
                    lat.append((time.perf_counter() - t_part) * 1000.0)
                    decisions[d] += 1
                th.join()
            stream_s = time.perf_counter() - ts
            win["end"] = stamp()
            win["parts"], win["seconds"] = len(items), stream_s
            windows.append(win)
            total_parts += len(items)
            total_stream_s += stream_s
            all_lat += lat
            lines[product] = {"classes": n, "support_source": source, "changeover_decode_s": t1 - t0,
                              "changeover_state_s": t2 - t1, "calibration_normals": len(calib),
                              "stream_normals": len(stream_normals), "defect_queries": len(g["queries"]),
                              "parts": len(items), "throughput_pps": len(items) / stream_s, "latency": quant(lat),
                              "thresholds": {"reject": thr[0], "accept": thr[1]}, "decision_counts_not_evaluated": decisions}
            print(f"[p45] {mode} {product}: {len(items)} parts, {len(items) / stream_s:.2f} parts/s, "
                  f"p50 {statistics.median(lat):.1f} ms, changeover {t2 - t0:.1f} s", flush=True)
            del frames, state, bank, protos
    res = {"schema_version": "p45_v1", "protocol": "docs/P45_LINE_SIMULATION_PROTOCOL.md", "mode": mode,
           "engine": str(args.fused), "rounds": ROUNDS, "queue_depth": QUEUE_DEPTH if mode == "file_pipe" else None,
           "overall": {"parts": total_parts, "stream_seconds": total_stream_s, "throughput_pps": total_parts / total_stream_s,
                       "latency": quant(all_lat)},
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "lines": lines, "windows": windows,
           "torch": torch.__version__}
    write_once(target, res)
    print(json.dumps({"mode": mode, "throughput_pps": res["overall"]["throughput_pps"],
                      "latency": res["overall"]["latency"], "peak_rss_mb": res["peak_rss_mb"]}, indent=1), flush=True)
    return 0


# --------------------------------------------------------------------------- amendment A1: the other configurations

TRT_CONFIGS = {"clip_l18": "trt_l14_l18", "dino_b": "trt_b336", "dino_s": "trt_s336",
               "clipb16_224": "trt_clipb16_224", "clipb16_336": "trt_clipb16_336"}


def run_trt(args) -> int:
    """A1: the same line simulation for a single-backbone / CLIP-only configuration with its P38 TensorRT pipeline
    (configuration's own preprocessing on the CPU, TensorRT engine, PyTorch head = the evaluation head)."""
    import resource
    import torch
    from experiments.p10 import run_pilot as rp
    from experiments.p10 import run_pilot_a as ra
    from experiments.p11 import formal_ab as fa
    from experiments.p11 import run_pilot_ab as ab
    from experiments.p13 import pilot as p13
    from experiments.p14.resource_profile import load, proto_space, score
    from experiments.p15 import screening as sc
    from experiments.p32 import device_extract as dx
    from experiments.p38 import trt_deploy as td
    from experiments.p41 import fused_engine as fe
    from experiments.p43.edge_measure import task_inputs_for
    from experiments.revision import revision_common as rc
    mode, cfg = args.run, args.config
    target = OUT / args.tag / f"p45_{cfg}_{mode}.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    device = torch.device("cuda")
    sync = torch.cuda.synchronize
    groups = dx.inputs(Path(args.mvtec_root))
    draws = json.loads(fe.DRAWS.read_text(encoding="utf-8"))
    pipe, _ = td.build_trt(TRT_CONFIGS[cfg], Path(args.engines), device)
    use_g = cfg != "clip_l18"

    def build_state(supports, s_lab, n, bank_imgs):
        feats = [pipe.features(im) for im in supports]
        bank_imgs_t = torch.cat([pipe.features(im)[1] for im in bank_imgs])
        bank = bank_imgs_t.reshape(-1, bank_imgs_t.shape[-1])
        tau = fa.tau99(bank_imgs_t)
        p_s = torch.cat([f[1] for f in feats])
        w = ab.region_weights(ra.patch_anomaly(p_s, bank), tau, soft=True)
        st = {"bank": bank, "tau": tau, "a": proto_space(ab.weighted_pool(p_s, w), s_lab, n),
              "rr": proto_space(p13.region_residual(p_s, bank, w), s_lab, n)}
        if use_g:
            st["g"] = proto_space(torch.cat([f[0] for f in feats]), s_lab, n)
        return st, feats

    def head(g_q, c_q, st):   # = experiments.p18.resource_profile.profile head "dual_branch_ARR" (a_r without g)
        best, idx = torch.einsum("npd,bd->npb", c_q, st["bank"]).max(dim=2)
        w = ab.region_weights(1.0 - best, st["tau"], soft=True)
        s = rp._zscore(score(ab.weighted_pool(c_q, w), st["a"])) + rp._zscore(score(ab.weighted_pool(c_q - st["bank"][idx], w), st["rr"]))
        if use_g:
            s = s + rp._zscore(score(g_q, st["g"]))
        return s, sc.image_score(1.0 - best, "top1pct")

    def part(image, st, thr):
        g_q, c_q = pipe.features(image)
        s, screen = head(g_q, c_q, st)
        cls, scr = int(s.argmax()), float(screen[0])
        return cls, ("reject" if scr > thr[0] else "accept" if scr <= thr[1] else "review")

    lines, all_lat, windows = {}, [], []
    total_parts, total_stream_s, head_check = 0, 0.0, None
    with torch.no_grad():
        for product in rc.PRODUCTS:
            g = groups[product]
            t0 = time.perf_counter()
            supports, s_lab, n, bank_imgs, source = task_inputs_for(product, Path(args.mvtec_root))
            t1 = time.perf_counter()
            st, feats = build_state(supports, s_lab, n, bank_imgs)
            sync()
            t2 = time.perf_counter()
            if head_check is None:   # the per-part head against the evaluation code (whole tensors), first product
                q_img = load(g["queries"][0][1])
                g_q, c_q = pipe.features(q_img)
                s_head, _ = head(g_q, c_q, st)
                p_all = torch.cat([f[1] for f in feats] + [c_q])
                si, qi = torch.arange(len(s_lab), device=device), torch.tensor([len(s_lab)], device=device)
                w_all = ab.region_weights(ra.patch_anomaly(p_all, st["bank"]), st["tau"], soft=True)
                soft_all, rr_all = ab.weighted_pool(p_all, w_all), p13.region_residual(p_all, st["bank"], w_all)
                ref = rp._zscore(rp.simpleshot(soft_all[si], s_lab, soft_all[qi], n)) + rp._zscore(rp.simpleshot(rr_all[si], s_lab, rr_all[qi], n))
                if use_g:
                    g_all = torch.cat([f[0] for f in feats] + [g_q])
                    ref = ref + rp._zscore(rp.simpleshot(g_all[si], s_lab, g_all[qi], n))
                head_check = float((s_head - ref).abs().max())
                if head_check > 1e-4:
                    raise SystemExit(f"head differs from the evaluation code: {head_check}")
            in_bank = set(draws[product]["111"])
            held = sorted(name for name, _ in g["normals"] if name not in in_bank)
            paths = dict(g["normals"])
            calib, stream_normals = held[0::2], held[1::2]
            cal = []
            for x in calib:
                g_c, c_c = pipe.features(load(paths[x]))
                cal.append(float(head(g_c, c_c, st)[1][0]))
            thr = (conformal(cal, A_REJECT), conformal(cal, A_ACCEPT))
            base = [paths[x] for x in stream_normals] + [p for _, p in g["queries"]]
            items = []
            for r in range(ROUNDS):
                order = list(base)
                random.Random(f"p45|{product}|{r}").shuffle(order)
                items += order
            frames = {p: load(p) for p in base} if mode == "frame_seq" else None
            sync()
            lat, decisions = [], {"reject": 0, "accept": 0, "review": 0}
            win = {"product": product, "start": stamp()}
            ts = time.perf_counter()
            if mode in ("frame_seq", "file_seq"):
                for p in items:
                    t_part = time.perf_counter()
                    _, d = part(frames[p] if frames is not None else load(p), st, thr)
                    lat.append((time.perf_counter() - t_part) * 1000.0)
                    decisions[d] += 1
            else:
                q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)

                def producer():
                    for p in items:
                        t_part = time.perf_counter()
                        q.put((t_part, load(p)))
                    q.put(None)
                th = threading.Thread(target=producer, daemon=True)
                th.start()
                while True:
                    it = q.get()
                    if it is None:
                        break
                    t_part, img = it
                    _, d = part(img, st, thr)
                    lat.append((time.perf_counter() - t_part) * 1000.0)
                    decisions[d] += 1
                th.join()
            stream_s = time.perf_counter() - ts
            win["end"] = stamp()
            win["parts"], win["seconds"] = len(items), stream_s
            windows.append(win)
            total_parts += len(items)
            total_stream_s += stream_s
            all_lat += lat
            lines[product] = {"classes": n, "support_source": source, "changeover_decode_s": t1 - t0,
                              "changeover_state_s": t2 - t1, "calibration_normals": len(calib),
                              "stream_normals": len(stream_normals), "defect_queries": len(g["queries"]),
                              "parts": len(items), "throughput_pps": len(items) / stream_s, "latency": quant(lat),
                              "thresholds": {"reject": thr[0], "accept": thr[1]}, "decision_counts_not_evaluated": decisions}
            print(f"[p45] {cfg} {mode} {product}: {len(items)} parts, {len(items) / stream_s:.2f} parts/s, "
                  f"p50 {statistics.median(lat):.1f} ms, changeover {t2 - t0:.1f} s", flush=True)
            del frames, st, feats
    res = {"schema_version": "p45_v1", "protocol": "docs/P45_LINE_SIMULATION_PROTOCOL.md#A1", "config": cfg, "mode": mode,
           "block": TRT_CONFIGS[cfg], "engines": td.ENGINES, "head_check_max_abs": head_check, "rounds": ROUNDS,
           "queue_depth": QUEUE_DEPTH if mode == "file_pipe" else None,
           "overall": {"parts": total_parts, "stream_seconds": total_stream_s, "throughput_pps": total_parts / total_stream_s,
                       "latency": quant(all_lat)},
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "lines": lines, "windows": windows,
           "torch": torch.__version__}
    write_once(target, res)
    print(json.dumps({"config": cfg, "mode": mode, "throughput_pps": res["overall"]["throughput_pps"],
                      "latency": res["overall"]["latency"], "head_check": head_check}, indent=1), flush=True)
    return 0


# --------------------------------------------------------------------------- amendment A2: one hour of continuous production

def sustain(args) -> int:
    """A2: the fused engine (frame mode) serving the first 12 product lines for SEGMENT s each, back to back; per-minute
    windows of throughput and latency; GPU clock read from sysfs at every window."""
    import numpy as np
    import torch
    from experiments.p14.resource_profile import load
    from experiments.p32 import device_extract as dx
    from experiments.p41 import fused_engine as fe
    from experiments.p43.edge_measure import gpu_freq, task_inputs_for
    from experiments.revision import revision_common as rc
    target = OUT / args.tag / "p45_sustain.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    device = torch.device("cuda")
    fe.DEVICE[0] = device
    groups = dx.inputs(Path(args.mvtec_root))
    draws = json.loads(fe.DRAWS.read_text(encoding="utf-8"))
    eng = fe.FusedEngine(args.fused, device)
    clone_run = lambda x, b, t, p: tuple(v.clone() for v in eng(x, b, t, p))  # noqa: E731
    prep_pil = lambda im: prep_array(np.asarray(im.convert("RGB")), device)  # noqa: E731
    segment = args.sustain / 12
    minutes, cur, changeovers = [], {"lat": [], "start": stamp(), "t0": time.perf_counter()}, []
    t_begin = time.perf_counter()
    with torch.no_grad():
        for product in rc.PRODUCTS[:12]:
            g = groups[product]
            t0 = time.perf_counter()
            supports, s_lab, n, bank_imgs, _ = task_inputs_for(product, Path(args.mvtec_root))
            st = fe.task_state(clone_run, supports, s_lab, n, bank_imgs, prep_pil)[:3]
            in_bank = set(draws[product]["111"])
            base = [p for name, p in g["normals"] if name not in in_bank] + [p for _, p in g["queries"]]
            frames = [np.asarray(load(p).convert("RGB")) for p in base]
            changeovers.append({"product": product, "seconds": time.perf_counter() - t0})
            order, r, ts = list(range(len(frames))), 0, time.perf_counter()
            while time.perf_counter() - ts < segment:
                random.Random(f"p45s|{product}|{r}").shuffle(order)
                for i in order:
                    t_part = time.perf_counter()
                    s, screen = eng(prep_array(frames[i], device), *st)[:2]
                    int(s.argmax()), float(screen)
                    cur["lat"].append((time.perf_counter() - t_part) * 1000.0)
                    if time.perf_counter() - cur["t0"] >= 60:
                        minutes.append({"minute": len(minutes) + 1, "local_end": stamp()["local"], "product": product,
                                        "parts": len(cur["lat"]), "seconds": time.perf_counter() - cur["t0"],
                                        "latency": quant(cur["lat"]), "gpu_freq": gpu_freq()})
                        print(f"[p45] minute {len(minutes)}: {len(cur['lat'])} parts, p50 {statistics.median(cur['lat']):.1f} ms", flush=True)
                        cur = {"lat": [], "start": stamp(), "t0": time.perf_counter()}
                r += 1
            del frames, st
    res = {"schema_version": "p45_v1", "protocol": "docs/P45_LINE_SIMULATION_PROTOCOL.md#A2", "seconds": time.perf_counter() - t_begin,
           "segment_s": segment, "changeovers": changeovers, "minutes": minutes, "torch": torch.__version__}
    write_once(target, res)
    print("[p45] sustain done", flush=True)
    return 0


def report(args) -> int:
    samples = []
    for line in Path(args.tegrastats).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
        p = re.search(r"VDD_IN (\d+)mW", line)
        if m and p:
            samples.append((datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S"), int(p.group(1)) / 1000))

    def mean_w(a, b):
        v = [w for t, w in samples if a <= t <= b]
        return (statistics.fmean(v), len(v)) if v else (None, 0)
    d = OUT / args.tag
    idle_rec = json.loads((d / "p45_idle.json").read_text(encoding="utf-8"))
    idle_w = mean_w(idle_rec["start"]["local"], idle_rec["end"]["local"])[0]
    out = {"idle_power_w": idle_w, "runs": {}}
    for f in sorted(d.glob("p45_*.json")):
        if f.name == "p45_idle.json":
            continue
        mode = f.stem[len("p45_"):]          # frame_seq / file_seq / file_pipe (fused) or <config>_<mode> (A1)
        r = json.loads(f.read_text(encoding="utf-8"))
        energy = dyn = 0.0
        n_samples = 0
        for w in r["windows"]:
            p, k = mean_w(w["start"]["local"], w["end"]["local"])
            n_samples += k
            energy += p * w["seconds"]
            dyn += (p - idle_w) * w["seconds"]
        parts = r["overall"]["parts"]
        out["runs"][mode] = {"throughput_pps": r["overall"]["throughput_pps"], "latency": r["overall"]["latency"],
                             "mean_power_w": energy / r["overall"]["stream_seconds"], "energy_per_part_j": energy / parts,
                             "dynamic_energy_per_part_j": dyn / parts, "power_samples": n_samples,
                             "peak_rss_mb": r["peak_rss_mb"],
                             "changeover_s": {p: v["changeover_decode_s"] + v["changeover_state_s"] for p, v in r["lines"].items()}}
    print(json.dumps(out, indent=1))
    if args.out:
        write_once(Path(args.out), out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--idle", type=float, default=0.0)
    ap.add_argument("--sustain", type=float, default=0.0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mvtec-root")
    ap.add_argument("--fused")
    ap.add_argument("--config", default="main_fused", choices=("main_fused", *TRT_CONFIGS))
    ap.add_argument("--engines")
    ap.add_argument("--tegrastats")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.report:
        return report(a)
    if a.idle > 0:
        return idle(a)
    if a.sustain > 0:
        return sustain(a)
    if a.run:
        return run(a) if a.config == "main_fused" else run_trt(a)
    raise SystemExit("choose --run, --idle or --report")


if __name__ == "__main__":
    sys.exit(main())
