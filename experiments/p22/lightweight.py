#!/usr/bin/env python3
"""P22: lightweight single-backbone configurations of the dual-branch method (report-only).

Protocol: docs/P22_LIGHTWEIGHT_PROTOCOL.md. Grid: DINOv2-S / DINOv2-B × input 224 / 336 / 448 ×
FP32 / FP16 (autocast), one forward pass giving the global CLS feature and the patch tokens of one
layer. Stage ``dev`` reads only the MVTec-FS support pool and MVTec AD train/good normals (dev
episodes, seeds 9001–9005) and selects the layer of every configuration; stage ``profile`` measures
batch-1 latency / memory of every selected configuration next to the main pipeline. No formal query
is read here. The main methods and their constants are untouched.

Usage (each stage once; a stopped dev stage continues only with --resume, after authorization):
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p22.lightweight --stage dev --mvtec-root D:\\intergration\\MVTEC
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p22.lightweight --stage profile --mvtec-root D:\\intergration\\MVTEC
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
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
from experiments.p10.extract_dev_features import mask_grid, support_pool_records  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p14.resource_profile import K_SHOT, PRODUCT, SEED, Pipeline, load  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18.resource_profile import count_params, profile  # noqa: E402
from experiments.p18.run import c1_parts, open_rgb  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c3_resource_profile import TIMED, WARMUP, _environment  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p22" / s for s in ("dev", "profile")}
PROTOCOL = "docs/P22_LIGHTWEIGHT_PROTOCOL.md"
GUARDED = ("experiments/p22", "experiments/p18", "experiments/p14", "experiments/p11", "experiments/p10", PROTOCOL)
P18_DEV = PROJECT_ROOT / "ckpt" / "p18" / "dev" / "p18_dev_results.json"
BACKBONES = {"dinov2_s": ("dinov2_vits14", PROJECT_ROOT / "pretrained" / "dinov2_vits14_pretrain.pth"),
             "dinov2_b": ("dinov2_vitb14", bb.DINO_B_CKPT)}
RESOLUTIONS = (224, 336, 448)
PRECISIONS = ("fp32", "fp16")
LAYERS = bb.LAYERS
METHODS = {"ss": ff.SS, "nr": ff.NR, "a": ff.A, "arr": ff.ARR}
ANCHOR = "dinov2_b_r336_fp32"   # the P18 single-backbone option (formal ≈ main on three datasets)
MARGIN_PP = 1.0


def config_names() -> list[str]:
    return [f"{b}_r{r}_{p}" for b in BACKBONES for r in RESOLUTIONS for p in PRECISIONS]


class Dino:
    """DINOv2 ViT-S/14 or ViT-B/14 at r x r (squash resize, CLIP normalisation as in the main view);
    global = final normalised CLS token (as bb.DinoB); FP16 = CUDA autocast of the forward pass."""

    def __init__(self, name: str, res: int, precision: str, device):
        hub, ckpt = BACKBONES[name]
        self.name, self.res, self.precision, self.device = name, res, precision, device
        self.sha256 = bb.file_sha256(ckpt)
        with torch.random.fork_rng(devices=[]):
            model = torch.hub.load(str(bb.DINO_HUB_REPO), hub, source="local", pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, batch: torch.Tensor, layers) -> tuple[torch.Tensor, dict]:
        layers = sorted(set(layers) | {12})
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.precision == "fp16"
               else contextlib.nullcontext())
        with ctx:
            outs = self.model.get_intermediate_layers(batch, n=[L - 1 for L in layers], norm=True,
                                                      return_class_token=True)
        by_layer = dict(zip(layers, outs))
        glob = F.normalize(by_layer[12][1].float(), dim=-1)
        return glob, {L: F.normalize(by_layer[L][0].float(), dim=-1) for L in layers}


def build(cfg: str, device) -> Dino:
    name, rest = cfg.rsplit("_r", 1)
    res, precision = rest.split("_")
    return Dino(name, int(res), precision, device)


def light_scores(g, patches, parts, si, s_lab, qi, n) -> dict:
    """SS / fixed-fraction / A / ARR exactly as ff.episode_scores (the C0 heads are not part of P22)."""
    dummy = torch.zeros(len(qi), n, device=g.device)
    out = ff.episode_scores(g, {"": parts}, si, s_lab, qi, n, dummy, dummy)
    return {k: out[m] for k, m in METHODS.items()}


# --------------------------------------------------------------------------- single-shot with checkpoints

@contextlib.contextmanager
def single_shot(out: Path, resume: bool):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; this P22 stage is single-shot")
    lock, staging = out.with_name(out.name + ".lock"), out.with_name(out.name + ".staging")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and not resume:
        raise rc.P6BFailClosedError(f"{lock} exists: a run was started before; continue only with --resume "
                                    "after the user authorizes it (nothing is deleted)")
    lock.mkdir(exist_ok=True)
    staging.mkdir(exist_ok=True)
    log = staging / "launches.json"
    launches = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    launches.append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": resume, "provenance": rc.git_provenance()})
    rc.write_json(log, launches)
    yield staging
    staging.rename(out)


# --------------------------------------------------------------------------- dev

def dev_product(product: str, mvtec_root: Path, device, batch_size: int) -> list[dict]:
    data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
    normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
    records = support_pool_records()[product]
    if [r["canonical_relative_path"] for r in records] != list(data["rel_paths"]):
        raise rc.P6BFailClosedError(f"{product}: support-pool records differ from the cached dev features")
    images = [open_rgb(rc.DATASET_ROOT / rel) for rel in data["rel_paths"]]
    normals = [open_rgb(mvtec_root / product / "train" / "good" / x) for x in normal["names"]]
    name_index = {x: i for i, x in enumerate(normal["names"])}
    episodes = {seed: list(rp.episodes(data["labels"], seed, product)) for seed in rp.DEV_SEEDS}
    rows = []
    for cfg in config_names():
        backbone = build(cfg, device)
        gb, pb = bb.extract(backbone, images, device, batch_size, LAYERS)
        _, pn = bb.extract(backbone, normals, device, batch_size, LAYERS)
        del backbone
        torch.cuda.empty_cache() if device.type == "cuda" else None
        side = int(round(pb[LAYERS[0]].shape[1] ** 0.5))
        masks = torch.stack([mask_grid(r, im.size, side) for r, im in zip(records, images)]).to(device)
        g = gb.to(device)
        with torch.no_grad():
            for L in LAYERS:
                patches = pb[L].to(device).float()
                for seed, eps in episodes.items():
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank = pn[L][idx].to(device).float()
                    parts = c1_parts(patches, bank, fa.tau99(bank))[0]
                    for k_shot, order, s_idx, s_lab, q_idx, q_lab in eps:
                        n = len(order)
                        si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
                        sc = {f"{m}_{cfg}_L{L}": v for m, v in light_scores(g, patches, parts, si, s_lab, qi, n).items()}
                        sc[f"oracle_{cfg}_L{L}"] = rp.oracle_maskpool(patches[si], masks[si], s_lab, patches[qi], masks[qi], n)
                        rows += fa.score_rows(sc, order, [order[c] for c in q_lab], {"product": product, "K": k_shot, "seed": seed})
                del patches
    return rows


def p18_integrity(rows) -> float:
    """dinov2_b_r336_fp32 must reproduce the P18 dev rows of the same backbone exactly."""
    ref = rc.load_json(P18_DEV)["rows"]
    mapped = []
    for r in rows:
        m = r["method"]
        for short in METHODS:
            prefix = f"{short}_{ANCHOR}_L"
            if m.startswith(prefix):
                mapped.append({**r, "method": f"{short}_dinov2_b_L{m[len(prefix):]}"})
    methods = {r["method"] for r in mapped}
    return exact_equal(mapped, ref, methods, ("product", "K", "seed"))


def mean_ba(rows) -> dict:
    """Dev mean BA in %, the ba_by_K definition of rp.summarize (product mean -> seed mean -> K mean);
    rp.summarize itself needs 'simpleshot_cl2n' rows, which P22 does not produce."""
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    return {m: 100 * statistics.fmean(
        statistics.fmean(statistics.fmean(cell[(m, p, k, s)] for p in rc.PRODUCTS) for s in rp.DEV_SEEDS)
        for k in rp.DEV_K) for m in sorted({r["method"] for r in rows})}


def select(ba: dict) -> dict:
    """Per configuration: the layer with the highest dev ARR BA (ties -> shallower layer, as P18)."""
    return {cfg: max(LAYERS, key=lambda L: (round(ba[f"arr_{cfg}_L{L}"], 10), -L)) for cfg in config_names()}


def run_dev(mvtec_root: Path, device, batch_size: int, resume: bool) -> int:
    with single_shot(OUT["dev"], resume) as staging:
        ckpt_dir = staging / "products"
        ckpt_dir.mkdir(exist_ok=True)
        rows = []
        for product in rc.PRODUCTS:
            path = ckpt_dir / f"{product}.json"
            if path.exists():
                rows += json.loads(path.read_text(encoding="utf-8"))
                print(f"[p22-dev] {product} (checkpoint)", flush=True)
                continue
            t0 = time.time()
            prod_rows = dev_product(product, mvtec_root, device, batch_size)
            rc.write_json(path, prod_rows)
            rows += prod_rows
            print(f"[p22-dev] {product} done ({time.time() - t0:.0f} s)", flush=True)
        integrity = p18_integrity(rows)
        if integrity >= 1e-12:
            raise rc.P6BFailClosedError(f"dev integrity vs P18 (DINOv2-B, 336, FP32) failed: {integrity}")
        ba = mean_ba(rows)
        layer = select(ba)
        main_dev = 100 * rc.load_json(P18_DEV)["dev_mean_ba"][ff.ARR]
        table = {cfg: {"layer": L, "arr": ba[f"arr_{cfg}_L{L}"], "a": ba[f"a_{cfg}_L{L}"],
                       "ss": ba[f"ss_{cfg}_L{L}"], "nr": ba[f"nr_{cfg}_L{L}"], "oracle": ba[f"oracle_{cfg}_L{L}"]}
                 for cfg, L in layer.items()}
        result = {"schema_version": "p22_dev_v1", "protocol": PROTOCOL, "provenance": rc.git_provenance(),
                  "integrity_max_abs_diff_vs_p18": integrity, "selection": layer, "main_arr_dev_ba": main_dev,
                  "anchor": ANCHOR, "anchor_arr_dev_ba": table[ANCHOR]["arr"], "margin_pp": MARGIN_PP,
                  "eligible_for_recommendation": sorted(c for c, v in table.items()
                                                        if v["arr"] >= table[ANCHOR]["arr"] - MARGIN_PP),
                  "selected_table": table, "dev_mean_ba": ba,
                  "weights_sha256": {b: bb.file_sha256(p) for b, (_, p) in BACKBONES.items()}, "rows": rows}
        lines = ["# P22 dev — lightweight single-backbone configurations (dev episodes only)", "",
                 f"Main ARR (CLIP-L@336 + DINOv2-S) dev BA {main_dev:.2f}; anchor {ANCHOR} {table[ANCHOR]['arr']:.2f}; "
                 f"integrity vs P18 max|Δ| {integrity:.1e}", "",
                 "| Configuration | Layer | ARR | A | fixed fraction | SimpleShot | oracle |", "|---|---|---|---|---|---|---|"]
        for cfg, v in table.items():
            lines.append(f"| {cfg} | {v['layer']} | {v['arr']:.2f} | {v['a']:.2f} | {v['nr']:.2f} | {v['ss']:.2f} | {v['oracle']:.2f} |")
        lines += ["", f"Eligible (dev ARR ≥ anchor − {MARGIN_PP} pp): " + ", ".join(result["eligible_for_recommendation"])]
        text = "\n".join(lines)
        rc.write_json(staging / "p22_dev_results.json", result)
        (staging / "p22_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- profile

class LightPipeline:
    def __init__(self, backbone: Dino, layer: int, device):
        self.backbone, self.layer, self.device = backbone, layer, device

    def features(self, image):
        batch = bb.eval_tensor(image, self.backbone.res).to(self.device)
        g, per = self.backbone.features(batch, (self.layer,))
        return g, per[self.layer]


def run_profile(mvtec_root: Path, device, resume: bool) -> int:
    dev_path = OUT["dev"] / "p22_dev_results.json"
    if not dev_path.is_file():
        raise rc.P6BFailClosedError("run the P22 dev stage first")
    dev = rc.load_json(dev_path)
    with single_shot(OUT["profile"], resume) as staging:
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        manifest = rc.load_json(rc.STAGE1_BUNDLE / "p6b_support_manifests" / PRODUCT / f"k{K_SHOT}_s{SEED}.json")
        order = list(manifest["class_order"])
        items = [(c, it["rel_path"]) for c, cls in enumerate(order) for it in manifest["classes"][cls]["items"]]
        s_lab, n = [c for c, _ in items], len(order)
        nf = torch.load(FORMAL_FEATURES / f"normals_{PRODUCT}.pt", map_location="cpu", weights_only=False)
        support_images = [load(rc.DATASET_ROOT / rel) for _, rel in items]
        bank_images = [load(mvtec_root / PRODUCT / "train" / "good" / x) for x in nf["draws"][str(SEED)][:ra.M_PRIMARY]]
        results, params = {}, {}
        with torch.no_grad():
            factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
            params["main"] = count_params(factory.clip_model.visual, factory.dino_extractor.model)
            results["main"] = profile("main", Pipeline(factory, device), support_images, bank_images, s_lab, n, device, sync)
            del factory
            torch.cuda.empty_cache() if device.type == "cuda" else None
            for cfg, layer in dev["selection"].items():
                backbone = build(cfg, device)
                params[cfg] = count_params(backbone.model)
                results[cfg] = profile(cfg, LightPipeline(backbone, layer, device), support_images, bank_images,
                                       s_lab, n, device, sync)
                del backbone
                torch.cuda.empty_cache() if device.type == "cuda" else None
        arr_ms = {c: r["online"]["dual_branch_ARR"]["end_to_end_from_pil"]["median_ms"] for c, r in results.items()}
        eligible = [c for c in dev["eligible_for_recommendation"] if c in arr_ms]
        recommended = min(eligible, key=lambda c: (arr_ms[c], c)) if eligible else None
        result = {"schema_version": "p22_profile_v1", "protocol": PROTOCOL, "product": PRODUCT, "K": K_SHOT,
                  "seed": SEED, "classes": n, "batch_size": 1, "warmup": WARMUP, "timed": TIMED,
                  "selection": dev["selection"], "vision_params": params, "arr_end_to_end_median_ms": arr_ms,
                  "recommended": recommended, "results": results, "environment": _environment(device),
                  "provenance": rc.git_provenance()}
        lines = [f"# P22 profile — {PRODUCT}, K={K_SHOT}, {n} classes, M=20 bank, batch 1 (one session)", "",
                 "| Pipeline | Layer | Params (M) | Dev ARR BA | ARR end-to-end median ms (p90) | Peak GPU MB |", "|---|---|---|---|---|---|"]
        for c, r in results.items():
            o = r["online"]["dual_branch_ARR"]
            e = o["end_to_end_from_pil"]
            dev_ba = dev["main_arr_dev_ba"] if c == "main" else dev["selected_table"][c]["arr"]
            layer = "18 + fusion" if c == "main" else dev["selection"][c]
            peak = f"{o['peak_allocated_bytes'] / 2 ** 20:.0f}" if o["peak_allocated_bytes"] else "—"
            lines.append(f"| {c} | {layer} | {params[c] / 1e6:.1f} | {dev_ba:.2f} | {e['median_ms']:.1f} ({e['p90_ms']:.1f}) | {peak} |")
        lines += ["", f"Recommended (fastest with dev ARR ≥ {ANCHOR} − {MARGIN_PP} pp): {recommended}"]
        text = "\n".join(lines)
        rc.write_json(staging / "p22_profile_results.json", result)
        (staging / "p22_profile_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "profile"))
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)   # as P18 (the integrity check depends on it)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "dev":
        return run_dev(Path(args.mvtec_root), device, args.batch_size, args.resume)
    return run_profile(Path(args.mvtec_root), device, args.resume)


if __name__ == "__main__":
    sys.exit(main())
