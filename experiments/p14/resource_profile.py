#!/usr/bin/env python3
"""P14 §4: batch-1 latency / peak memory of the training-free pipelines (no test data).

Product bottle, formal support manifest K=5 seed 111 (support-pool images), M=20 bank from
MVTec AD train/good (the formal seed-111 draw). Online = per query given precomputed support
and bank states; offline = one-time preparation per product. Before timing, every online
head is checked against the evaluation code (rp.simpleshot, local_xclass, p13 functions).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p14.resource_profile --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import math
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

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p13 import pilot as p13  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c3_resource_profile import TIMED, WARMUP, _environment, _time  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p14_resource"
PRODUCT, K_SHOT, SEED = "bottle", 5, 111


def load(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


class Pipeline:
    def __init__(self, factory, device):
        self.factory, self.device = factory, device

    def features(self, image: Image.Image):
        batch = models.deterministic_eval_tensor(image).to(self.device)
        pooled, tokens = self.factory.clip_model.visual(batch, [CLIP_LAYER])
        dino = self.factory.dino_extractor.forward_patches(batch)
        g = self.factory.fusion.fuse(models.l2_rows(pooled), models.l2_rows(dino.mean(dim=1))).float()
        return g, F.normalize(tokens[0][:, 1:, :].float(), dim=-1)


def proto_space(h_s: torch.Tensor, labels, n: int):
    """SimpleShot state: support mean and centered class prototypes (== rp.simpleshot)."""
    mu = h_s.mean(0, keepdim=True)
    return mu, rp._class_means(F.normalize(h_s - mu, dim=-1), labels, n)


def score(h_q, state):
    mu, protos = state
    return F.normalize(h_q - mu, dim=-1) @ protos.t()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; write-once output")
    device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    manifest = rc.load_json(rc.STAGE1_BUNDLE / "p6b_support_manifests" / PRODUCT / f"k{K_SHOT}_s{SEED}.json")
    order = list(manifest["class_order"])
    items = [(c, it["rel_path"]) for c, cls in enumerate(order) for it in manifest["classes"][cls]["items"]]
    s_lab = [c for c, _ in items]
    n = len(order)
    nf = torch.load(FORMAL_FEATURES / f"normals_{PRODUCT}.pt", map_location="cpu", weights_only=False)
    bank_paths = [Path(args.mvtec_root) / PRODUCT / "train" / "good" / x for x in nf["draws"][str(SEED)][:ra.M_PRIMARY]]
    support_images = [load(rc.DATASET_ROOT / rel) for _, rel in items]
    bank_images = [load(p) for p in bank_paths]
    query_image = support_images[0]

    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    pipe = Pipeline(factory, device)
    offline = {}

    def timed_once(fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        return out, (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        for _ in range(5):  # warm-up of the backbones
            pipe.features(query_image)
        feats, offline["support_feature_extraction_ms"] = timed_once(lambda: [pipe.features(im) for im in support_images])
        bank_feats, offline["bank_feature_extraction_ms"] = timed_once(lambda: [pipe.features(im)[1] for im in bank_images])
        g_s = torch.cat([f[0] for f in feats])
        clip_s = torch.cat([f[1] for f in feats])
        bank_imgs = torch.cat(bank_feats)
        bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
        tau, offline["tau_calibration_loo_ms"] = timed_once(lambda: fa.tau99(bank_imgs))

        def prepare():
            a = ra.patch_anomaly(clip_s, bank)
            w = ab.region_weights(a, tau, soft=True)
            pooled = ra.anomaly_pool(clip_s, a)
            soft = ab.weighted_pool(clip_s, w)
            rr = p13.region_residual(clip_s, bank, w)
            lab = torch.as_tensor(s_lab, device=device)
            sets = [clip_s[lab == c].reshape(-1, clip_s.shape[-1]) for c in range(n)]
            ss_sim = rp._nearest_per_class(clip_s, sets)
            own = F.one_hot(lab, n).bool()[:, None, :].expand_as(ss_sim)
            a0_s = 1.0 - ss_sim.masked_fill(own, float("-inf")).max(dim=2).values
            tau0 = float(torch.quantile(a0_s.reshape(-1).double(), 0.90))
            cons_s = ab.weighted_pool(clip_s, ab.region_weights(a0_s, tau0, soft=True))
            return {"g": proto_space(g_s, s_lab, n), "nr": proto_space(pooled, s_lab, n), "a": proto_space(soft, s_lab, n),
                    "rr": proto_space(rr, s_lab, n), "cons": proto_space(cons_s, s_lab, n), "sets": sets, "tau0": tau0}
        state, offline["support_state_ms"] = timed_once(prepare)

        top = math.ceil(rp.RHO * clip_s.shape[1])

        def head(kind, g_q, c_q):
            base = score(g_q, state["g"])
            if kind == "simpleshot":
                return base
            zb = rp._zscore(base)
            if kind in ("normref_fixed", "adaptive_A", "dual_branch_ARR"):
                sim = torch.einsum("npd,bd->npb", c_q, bank)
                best, idx = sim.max(dim=2)
                a = 1.0 - best
                if kind == "normref_fixed":
                    pooled = torch.gather(c_q, 1, a.topk(top, dim=1).indices.unsqueeze(-1).expand(-1, -1, c_q.shape[-1])).mean(1)
                    return zb + rp._zscore(score(pooled, state["nr"]))
                w = ab.region_weights(a, tau, soft=True)
                sa = score(ab.weighted_pool(c_q, w), state["a"])
                if kind == "adaptive_A":
                    return zb + rp._zscore(sa)
                srr = score(ab.weighted_pool(c_q - bank[idx], w), state["rr"])
                return zb + rp._zscore(sa) + rp._zscore(srr)
            s = rp._nearest_per_class(c_q, state["sets"])
            if kind == "cross_class":
                margins = [(s[..., c] - torch.cat([s[..., :c], s[..., c + 1:]], dim=2).max(dim=2).values).topk(top, dim=1).values.mean(1)
                           for c in range(n)]
                return zb + rp._zscore(torch.stack(margins, dim=1))
            a0 = 1.0 - s.topk(2, dim=2).values[..., 1]
            h0 = ab.weighted_pool(c_q, ab.region_weights(a0, state["tau0"], soft=True))
            return zb + rp._zscore(score(h0, state["cons"]))

        # correctness of the timed heads against the evaluation code (query = first support image)
        g_q, c_q = pipe.features(query_image)
        both_g, both_c = torch.cat([g_s, g_q]), torch.cat([clip_s, c_q])
        si = torch.arange(len(items), device=device)
        qi = torch.tensor([len(items)], device=device)
        zb = rp._zscore(rp.simpleshot(both_g[si], s_lab, both_g[qi], n))
        a_all = ra.patch_anomaly(both_c, bank)
        w_all = ab.region_weights(a_all, tau, soft=True)
        soft_all = ab.weighted_pool(both_c, w_all)
        rr_all = p13.region_residual(both_c, bank, w_all)
        pooled_all = ra.anomaly_pool(both_c, a_all)
        sa = rp.simpleshot(soft_all[si], s_lab, soft_all[qi], n)
        reference = {
            "simpleshot": rp.simpleshot(both_g[si], s_lab, both_g[qi], n),
            "normref_fixed": zb + rp._zscore(rp.simpleshot(pooled_all[si], s_lab, pooled_all[qi], n)),
            "adaptive_A": zb + rp._zscore(sa),
            "dual_branch_ARR": zb + rp._zscore(sa) + rp._zscore(rp.simpleshot(rr_all[si], s_lab, rr_all[qi], n)),
            "cross_class": zb + rp._zscore(rp.local_xclass(both_c[si], s_lab, both_c[qi], n)),
            "consensus_C0": zb + rp._zscore(p13.consensus_scores(both_c, si, s_lab, qi, n)["fused_cons_soft_supq90"]),
        }
        check = {k: float((head(k, g_q, c_q) - v).abs().max()) for k, v in reference.items()}
        if any(v > 1e-4 for v in check.values()):
            raise rc.P6BFailClosedError(f"timed heads differ from the evaluation code: {check}")

        results = {}
        for kind in reference:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            end_to_end = _time(lambda: head(kind, *pipe.features(query_image)), device)
            head_only = _time(lambda: head(kind, g_q, c_q), device)
            results[kind] = {"end_to_end_from_pil": end_to_end, "head_only_given_features": head_only,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    storage = {"bank_patches": int(bank.shape[0]), "bank_fp32_mb": bank.numel() * 4 / 2 ** 20,
               "support_patches": int(clip_s.shape[0] * clip_s.shape[1]),
               "support_patch_fp32_mb": clip_s.numel() * 4 / 2 ** 20}
    result = {"schema_version": "p14_resource_v1", "protocol": "docs/P14_FINAL_FORMAL_ADDENDUM.md#4",
              "product": PRODUCT, "K": K_SHOT, "seed": SEED, "classes": n, "supports": len(items), "bank_images": len(bank_images),
              "batch_size": 1, "precision": "fp32", "warmup": WARMUP, "timed": TIMED,
              "head_check_max_abs_diff": check, "offline_ms": offline, "online": results, "storage": storage,
              "environment": _environment(device), "provenance": rc.git_provenance()}
    lines = [f"# P14 resource profile — {PRODUCT}, K={K_SHOT}, {n} classes, M=20 bank (batch 1, FP32)", "",
             "| Pipeline | End-to-end median ms (p90) | Head only median ms | Peak GPU MB |", "|---|---|---|---|"]
    for kind, r in results.items():
        e, h = r["end_to_end_from_pil"], r["head_only_given_features"]
        peak = f"{r['peak_allocated_bytes'] / 2 ** 20:.0f}" if r["peak_allocated_bytes"] else "—"
        lines.append(f"| {kind} | {e['median_ms']:.1f} ({e['p90_ms']:.1f}) | {h['median_ms']:.2f} | {peak} |")
    lines += ["", "Offline (once per product): " + ", ".join(f"{k} {v:.0f} ms" for k, v in offline.items()),
              f"Storage: bank {storage['bank_patches']} patches ({storage['bank_fp32_mb']:.1f} MB fp32), "
              f"support patches {storage['support_patches']} ({storage['support_patch_fp32_mb']:.1f} MB fp32)",
              "Head check vs evaluation code (max |Δ|): " + ", ".join(f"{k} {v:.1e}" for k, v in check.items())]
    text = "\n".join(lines)
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "resource_results.json", result)
    (OUT / "resource_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
