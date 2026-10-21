#!/usr/bin/env python3
"""P18 A8: batch-1 latency / peak memory of the main pipeline vs the single-backbone swaps (no test data).

Same setting as the P14 resource profile (bottle, formal support manifest K=5 seed 111, M=20 bank
from MVTec AD train/good), measured in ONE session for all three feature pipelines:
main (CLIP ViT-L/14@336 layer 18 + DINOv2-S), DINOv2-B (P18 dev selection) and CLIP-B/16 (P18 dev
selection). Before timing, every head is checked against the evaluation code.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p18.resource_profile --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
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

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p13 import pilot as p13  # noqa: E402
from experiments.p14.resource_profile import K_SHOT, PRODUCT, SEED, Pipeline, load, proto_space, score  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18.run import dev_selection, parse_config  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c3_resource_profile import TIMED, WARMUP, _environment, _time  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p18_resource"
HEADS = ("simpleshot", "dual_branch_ARR", "consensus_C0")


class AltPipeline:
    """Single alternative backbone: global and patch features from one forward pass."""

    def __init__(self, backbone, layer, device):
        self.backbone, self.layer, self.device = backbone, layer, device

    def features(self, image):
        batch = bb.eval_tensor(image, self.backbone.res).to(self.device)
        g, per = self.backbone.features(batch, (self.layer,))
        return g, per[self.layer]


def count_params(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def profile(name, pipe, support_images, bank_images, s_lab, n, device, sync):
    offline = {}

    def timed_once(fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        return out, (time.perf_counter() - t0) * 1000.0

    query = support_images[0]
    for _ in range(5):
        pipe.features(query)
    feats, offline["support_feature_extraction_ms"] = timed_once(lambda: [pipe.features(im) for im in support_images])
    bank_feats, offline["bank_feature_extraction_ms"] = timed_once(lambda: [pipe.features(im)[1] for im in bank_images])
    g_s = torch.cat([f[0] for f in feats])
    p_s = torch.cat([f[1] for f in feats])
    bank_imgs = torch.cat(bank_feats)
    bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
    tau, offline["tau_calibration_loo_ms"] = timed_once(lambda: fa.tau99(bank_imgs))

    def prepare():
        a = ra.patch_anomaly(p_s, bank)
        w = ab.region_weights(a, tau, soft=True)
        lab = torch.as_tensor(s_lab, device=device)
        sets = [p_s[lab == c].reshape(-1, p_s.shape[-1]) for c in range(n)]
        ss_sim = rp._nearest_per_class(p_s, sets)
        own = F.one_hot(lab, n).bool()[:, None, :].expand_as(ss_sim)
        a0_s = 1.0 - ss_sim.masked_fill(own, float("-inf")).max(dim=2).values
        tau0 = float(torch.quantile(a0_s.reshape(-1).double(), 0.90))
        return {"g": proto_space(g_s, s_lab, n), "a": proto_space(ab.weighted_pool(p_s, w), s_lab, n),
                "rr": proto_space(p13.region_residual(p_s, bank, w), s_lab, n),
                "cons": proto_space(ab.weighted_pool(p_s, ab.region_weights(a0_s, tau0, soft=True)), s_lab, n),
                "sets": sets, "tau0": tau0}
    state, offline["support_state_ms"] = timed_once(prepare)

    def head(kind, g_q, c_q):
        base = score(g_q, state["g"])
        if kind == "simpleshot":
            return base
        zb = rp._zscore(base)
        if kind == "dual_branch_ARR":
            best, idx = torch.einsum("npd,bd->npb", c_q, bank).max(dim=2)
            w = ab.region_weights(1.0 - best, tau, soft=True)
            sa = score(ab.weighted_pool(c_q, w), state["a"])
            srr = score(ab.weighted_pool(c_q - bank[idx], w), state["rr"])
            return zb + rp._zscore(sa) + rp._zscore(srr)
        s = rp._nearest_per_class(c_q, state["sets"])
        a0 = 1.0 - s.topk(2, dim=2).values[..., 1]
        return zb + rp._zscore(score(ab.weighted_pool(c_q, ab.region_weights(a0, state["tau0"], soft=True)), state["cons"]))

    g_q, c_q = pipe.features(query)
    both_g, both_c = torch.cat([g_s, g_q]), torch.cat([p_s, c_q])
    si = torch.arange(len(s_lab), device=device)
    qi = torch.tensor([len(s_lab)], device=device)
    zb = rp._zscore(rp.simpleshot(both_g[si], s_lab, both_g[qi], n))
    w_all = ab.region_weights(ra.patch_anomaly(both_c, bank), tau, soft=True)
    soft_all, rr_all = ab.weighted_pool(both_c, w_all), p13.region_residual(both_c, bank, w_all)
    reference = {
        "simpleshot": rp.simpleshot(both_g[si], s_lab, both_g[qi], n),
        "dual_branch_ARR": zb + rp._zscore(rp.simpleshot(soft_all[si], s_lab, soft_all[qi], n))
        + rp._zscore(rp.simpleshot(rr_all[si], s_lab, rr_all[qi], n)),
        "consensus_C0": zb + rp._zscore(p13.consensus_scores(both_c, si, s_lab, qi, n)["fused_cons_soft_supq90"]),
    }
    check = {k: float((head(k, g_q, c_q) - v).abs().max()) for k, v in reference.items()}
    if any(v > 1e-4 for v in check.values()):
        raise rc.P6BFailClosedError(f"{name}: timed heads differ from the evaluation code: {check}")
    online = {}
    for kind in HEADS:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        online[kind] = {"end_to_end_from_pil": _time(lambda: head(kind, *pipe.features(query)), device),
                        "head_only_given_features": _time(lambda: head(kind, g_q, c_q), device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    storage = {"patch_grid": int(p_s.shape[1]), "patch_dim": int(p_s.shape[2]), "bank_patches": int(bank.shape[0]),
               "bank_fp32_mb": bank.numel() * 4 / 2 ** 20}
    return {"head_check_max_abs_diff": check, "offline_ms": offline, "online": online, "storage": storage}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; write-once output")
    sel = dev_selection()
    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    manifest = rc.load_json(rc.STAGE1_BUNDLE / "p6b_support_manifests" / PRODUCT / f"k{K_SHOT}_s{SEED}.json")
    order = list(manifest["class_order"])
    items = [(c, it["rel_path"]) for c, cls in enumerate(order) for it in manifest["classes"][cls]["items"]]
    s_lab, n = [c for c, _ in items], len(order)
    nf = torch.load(FORMAL_FEATURES / f"normals_{PRODUCT}.pt", map_location="cpu", weights_only=False)
    support_images = [load(rc.DATASET_ROOT / rel) for _, rel in items]
    bank_images = [load(Path(args.mvtec_root) / PRODUCT / "train" / "good" / x) for x in nf["draws"][str(SEED)][:ra.M_PRIMARY]]

    results, params = {}, {}
    with torch.no_grad():
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        params["main"] = count_params(factory.clip_model.visual, factory.dino_extractor.model)
        results["main"] = profile("main", Pipeline(factory, device), support_images, bank_images, s_lab, n, device, sync)
        del factory
        torch.cuda.empty_cache() if device.type == "cuda" else None
        for b in ("dinov2_b", "clip_b16"):
            name, res, layer = parse_config(sel[b])
            backbone = bb.DinoB(device, sel["dinov2_b_sha256"]) if name == "dinov2_b" else bb.ClipB(device, res)
            params[b] = count_params(backbone.model.visual if name == "clip_b16" else backbone.model)
            results[b] = profile(b, AltPipeline(backbone, layer, device), support_images, bank_images, s_lab, n, device, sync)
            del backbone
            torch.cuda.empty_cache() if device.type == "cuda" else None

    result = {"schema_version": "p18_resource_v1", "protocol": "docs/P18_BASELINES_BACKBONES_PURITY_PROTOCOL.md (A8)",
              "product": PRODUCT, "K": K_SHOT, "seed": SEED, "classes": n, "supports": len(items), "bank_images": len(bank_images),
              "batch_size": 1, "precision": "fp32", "warmup": WARMUP, "timed": TIMED, "selection": sel,
              "vision_params": params, "results": results, "environment": _environment(device), "provenance": rc.git_provenance()}
    lines = [f"# P18 resource profile — {PRODUCT}, K={K_SHOT}, {n} classes, M=20 bank (batch 1, FP32, one session)", "",
             "| Pipeline | Vision params (M) | Head | End-to-end median ms (p90) | Head only median ms | Peak GPU MB |",
             "|---|---|---|---|---|---|"]
    for p, r in results.items():
        for kind, o in r["online"].items():
            e, h = o["end_to_end_from_pil"], o["head_only_given_features"]
            peak = f"{o['peak_allocated_bytes'] / 2 ** 20:.0f}" if o["peak_allocated_bytes"] else "—"
            lines.append(f"| {p} | {params[p] / 1e6:.1f} | {kind} | {e['median_ms']:.1f} ({e['p90_ms']:.1f}) | {h['median_ms']:.2f} | {peak} |")
    lines.append("")
    for p, r in results.items():
        lines.append(f"- {p}: offline " + ", ".join(f"{k} {v:.0f} ms" for k, v in r["offline_ms"].items())
                     + f"; bank {r['storage']['bank_patches']} patches × {r['storage']['patch_dim']} ({r['storage']['bank_fp32_mb']:.1f} MB fp32)"
                     + "; head check max|Δ| " + ", ".join(f"{k} {v:.1e}" for k, v in r["head_check_max_abs_diff"].items()))
    text = "\n".join(lines)
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "resource_results.json", result)
    (OUT / "resource_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
