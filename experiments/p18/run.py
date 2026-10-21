#!/usr/bin/env python3
"""P18: adapted published baselines, backbone swap and evidence purity (report-only).

Protocol: docs/P18_BASELINES_BACKBONES_PURITY_PROTOCOL.md. No result here changes the main
methods (ARR, CONS) or their constants.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p18.run --stage dev --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p18.run --stage mvtec --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p18.run --stage external --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER, mask_grid, support_pool_records  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES, query_annotation_records  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p13 import pilot as p13  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p17.robustness import signflip_p  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18 import baselines as bl  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p18" / s for s in ("dev", "mvtec", "external")}
PROTOCOL = "docs/P18_BASELINES_BACKBONES_PURITY_PROTOCOL.md"
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13", "experiments/p14",
           "experiments/p17", "experiments/p18", "experiments/revision", PROTOCOL)
SS, NR, XC, A, ARR, CONS = ff.SS, ff.NR, ff.XC, ff.A, ff.ARR, ff.CONS
MAIN = (SS, NR, XC, A, ARR, CONS)
SHORT = {SS: "ss", NR: "nr", XC: "xc", A: "a", ARR: "arr", CONS: "cons"}
ALT = ("dinov2_b", "clip_b16")
CROP_BATCH = 64


def tag(variant) -> str:
    return f"{variant[0]}_s{variant[1]}"


def dev_configs() -> list[str]:
    return ([f"dinov2_b_L{L}" for L in bb.LAYERS] +
            [f"clip_b16_r{r}_L{L}" for r in bb.CLIP_B_RES for L in bb.LAYERS])


# --------------------------------------------------------------------------- shared helpers

@contextlib.contextmanager
def single_shot(out: Path):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; this P18 stage is single-shot")
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


def open_rgb(path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def c1_parts(patches, bank_imgs, tau):
    """(fixed-fraction, soft appearance, soft residual) descriptors + anomalies + ARR weights (== ff.c1_descriptors)."""
    bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
    a = ra.patch_anomaly(patches, bank)
    w = ab.region_weights(a, tau, soft=True)
    return (ra.anomaly_pool(patches, a), ab.weighted_pool(patches, w), p13.region_residual(patches, bank, w)), a, w


def c1_parts_chunked(patches_cpu, bank_imgs, tau, device):
    outs = [c1_parts(ch.to(device).float(), bank_imgs, tau) for ch in patches_cpu.split(ee.Q_CHUNK)]
    return tuple(torch.cat([o[0][i] for o in outs]) for i in range(3)), torch.cat([o[2] for o in outs])


def whole_scores(g, patches, parts, si, s_lab, qi, n) -> dict:
    """The P14 MVTec-FS path (whole tensors) for any feature set."""
    xc = rp.local_xclass(patches[si], s_lab, patches[qi], n)
    cons = p13.consensus_scores(patches, si, s_lab, qi, n)[CONS]
    return ff.episode_scores(g, {"": parts}, si, s_lab, qi, n, xc, cons)


def chunked_scores(g, patches_cpu, parts, si, s_lab, qi, q_patches_cpu, n, device) -> dict:
    """The P14 external path (query chunks) for any feature set."""
    sup = patches_cpu[si.cpu()].to(device).float()
    xc = ee.xclass_chunked(sup, s_lab, q_patches_cpu, n, device)
    s_range = torch.arange(len(s_lab), device=device)
    cons = []
    for chunk in q_patches_cpu.split(ee.Q_CHUNK):
        both = torch.cat([sup, chunk.to(device).float()])
        cons.append(p13.consensus_scores(both, s_range, s_lab, torch.arange(len(s_lab), both.shape[0], device=device), n)[CONS])
    return ff.episode_scores(g, {"": parts}, si, s_lab, qi, n, xc, torch.cat(cons))


def renamed(scores: dict, suffix: str) -> dict:
    return {f"{SHORT[m]}_{suffix}": v for m, v in scores.items()}


def encode_global(factory, images, device, batch_size) -> torch.Tensor:
    """Main global feature (CLIP-L projected + DINOv2-S patch mean, fused) of PIL images."""
    out = []
    for start in range(0, len(images), batch_size):
        batch = torch.cat([models.deterministic_eval_tensor(im) for im in images[start:start + batch_size]]).to(device)
        with torch.no_grad():
            pooled, _ = factory.clip_model.visual(batch, [CLIP_LAYER])
            dino = factory.dino_extractor.forward_patches(batch)
            out.append(factory.fusion.fuse(models.l2_rows(pooled), models.l2_rows(dino.mean(dim=1))).float().cpu())
    return torch.cat(out)


def crop_features(factory, load, n_images: int, wanted, boxes: dict, device, batch_size) -> dict:
    """boxes {key: [box per image]} -> {key: [n_images, d]} re-encoded crops (zero rows where not wanted)."""
    out, crops, where = {}, [], []

    def flush():
        if crops:
            feats = encode_global(factory, crops, device, batch_size)
            for (key, i), f in zip(where, feats):
                if key not in out:
                    out[key] = torch.zeros(n_images, f.shape[0])
                out[key][i] = f
            crops.clear()
            where.clear()

    for i in wanted:
        image = load(i)
        for key, per_image in boxes.items():
            crops.append(bl.apply_box(image, per_image[i]))
            where.append((key, i))
        if len(crops) >= CROP_BATCH:
            flush()
    flush()
    return out


def add_local_baselines(sc: dict, dn4: torch.Tensor, crop_globals: dict, si, s_lab, qi, n, device):
    zb = rp._zscore(sc[SS])
    sc["dn4"], sc["dn4_fused"] = dn4, zb + rp._zscore(dn4)
    for name, gc in crop_globals.items():
        gc = gc.to(device)
        c = rp.simpleshot(gc[si], s_lab, gc[qi], n)
        sc[name], sc[f"{name}_fused"] = c, zb + rp._zscore(c)


def mean_dev_ba(rows) -> dict:
    summary = rp.summarize(rows)["methods"]
    return {m: statistics.fmean(v["ba_by_K"].values()) for m, v in summary.items()}


def select_crop(ba: dict):
    return max(bl.CROP_VARIANTS, key=lambda v: (round(ba[f"crop_{tag(v)}_fused"], 12), -v[1], v[0] == "largest"))


def select_config(ba: dict, backbone: str) -> str:
    cands = [c for c in dev_configs() if c.startswith(backbone)]

    def key(cfg):
        layer = int(cfg.rsplit("_L", 1)[1])
        res = int(cfg.split("_r")[1].split("_")[0]) if backbone == "clip_b16" else 0
        return round(ba[f"arr_{cfg}"], 12), -layer, -res
    return max(cands, key=key)


def dev_selection() -> dict:
    path = OUT["dev"] / "p18_dev_results.json"
    if not path.is_file():
        raise rc.P6BFailClosedError("run the P18 dev stage first (selections are read from it)")
    return rc.load_json(path)["selection"]


def parse_config(cfg: str) -> tuple[str, int, int]:
    """'dinov2_b_L10' -> ('dinov2_b', 336, 10); 'clip_b16_r224_L9' -> ('clip_b16', 224, 9)."""
    layer = int(cfg.rsplit("_L", 1)[1])
    if cfg.startswith("clip_b16"):
        return "clip_b16", int(cfg.split("_r")[1].split("_")[0]), layer
    return "dinov2_b", bb.DinoB.res, layer


def build_selected(sel: dict, device) -> dict:
    out = {}
    for b in ALT:
        name, res, layer = parse_config(sel[b])
        backbone = bb.DinoB(device, sel["dinov2_b_sha256"]) if name == "dinov2_b" else bb.ClipB(device, res)
        out[b] = (backbone, layer)
    return out


# --------------------------------------------------------------------------- dev

def run_dev(mvtec_root: Path, device, batch_size) -> int:
    with single_shot(OUT["dev"]) as staging:
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        backs = bb.build_all(device)
        pools = support_pool_records()
        rows, entries, shares = [], [], []
        dist = {}
        for product in rc.PRODUCTS:
            data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            records = pools[product]
            if [r["canonical_relative_path"] for r in records] != list(data["rel_paths"]):
                raise rc.P6BFailClosedError(f"{product}: support-pool records differ from the cached dev features")
            images = [open_rgb(rc.DATASET_ROOT / rel) for rel in data["rel_paths"]]
            normals = [open_rgb(mvtec_root / product / "train" / "good" / x) for x in normal["names"]]
            feats = {}
            for key, backbone in backs.items():
                gb, pb = bb.extract(backbone, images, device, batch_size)
                _, pn = bb.extract(backbone, normals, device, batch_size)
                side = int(round(pb[bb.LAYERS[0]].shape[1] ** 0.5))
                mb = torch.stack([mask_grid(r, im.size, side) for r, im in zip(records, images)])
                for L in bb.LAYERS:
                    feats[f"{key}_L{L}"] = (gb, pb[L], pn[L], mb)
            g = data["global_fusion"].to(device)
            clip = data["clip18_patches"].to(device).float()
            masks = data["mask_grid"].to(device)
            valid = (masks > 0).any(dim=1)
            name_index = {x: i for i, x in enumerate(normal["names"])}
            with torch.no_grad():
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank_imgs = normal["clip18_patches"][idx].to(device).float()
                    tau = fa.tau99(bank_imgs)
                    parts, a, w = c1_parts(clip, bank_imgs, tau)
                    # §P purity and background exceedance
                    lam = {s: bl.purity(v, masks) for s, v in bl.scheme_weights(a, tau).items()}
                    rho = (masks > 0).float().mean(dim=1)
                    d = bl.mask_distance(masks)
                    for i in valid.nonzero().flatten().tolist():
                        entries.append({"product": product, "seed": seed, "rho": float(rho[i]),
                                        **{f"{s}|any": float(v[0][i]) for s, v in lam.items()},
                                        **{f"{s}|area": float(v[1][i]) for s, v in lam.items()}})
                        wi, total = w[i], float(w[i].sum())
                        shares.append({"mask": float(wi[masks[i] > 0].sum()) / total,
                                       "near": float(wi[(d[i] >= 1) & (d[i] <= 2)].sum()) / total,
                                       "far": float(wi[d[i] >= 3].sum()) / total})
                    exceed = a > tau
                    for lo, hi in bl.DIST_BINS:
                        cell = (d >= lo) & (d <= hi) & valid[:, None]
                        dist.setdefault(f"{lo}-{hi}", []).append((int((exceed & cell).sum()), int(cell.sum())))
                    # §B crops (all variants, for dev selection)
                    boxes = {f"crop_{tag(v)}": bl.crop_boxes(w, *v) for v in bl.CROP_VARIANTS}
                    crops = crop_features(factory, lambda i: images[i], len(images), range(len(images)), boxes, device, batch_size)
                    episodes = list(rp.episodes(data["labels"], seed, product))
                    for k_shot, order, s_idx, s_lab, q_idx, q_lab in episodes:
                        n = len(order)
                        si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
                        sc = whole_scores(g, clip, parts, si, s_lab, qi, n)
                        add_local_baselines(sc, bl.dn4_scores(clip[si], s_lab, clip[qi], n), crops, si, s_lab, qi, n, device)
                        sc["oracle_clipl"] = rp.oracle_maskpool(clip[si], masks[si], s_lab, clip[qi], masks[qi], n)
                        rows += fa.score_rows(sc, order, [order[c] for c in q_lab], {"product": product, "K": k_shot, "seed": seed})
                    # §K backbone configurations
                    for cfg, (gb, pb, pn, mb) in feats.items():
                        gbd, pbd, mbd = gb.to(device), pb.to(device).float(), mb.to(device)
                        bank_b = pn[idx].to(device).float()
                        parts_b = c1_parts(pbd, bank_b, fa.tau99(bank_b))[0]
                        for k_shot, order, s_idx, s_lab, q_idx, q_lab in episodes:
                            n = len(order)
                            si, qi = torch.as_tensor(s_idx, device=device), torch.as_tensor(q_idx, device=device)
                            sc = renamed(whole_scores(gbd, pbd, parts_b, si, s_lab, qi, n), cfg)
                            sc[f"oracle_{cfg}"] = rp.oracle_maskpool(pbd[si], mbd[si], s_lab, pbd[qi], mbd[qi], n)
                            rows += fa.score_rows(sc, order, [order[c] for c in q_lab], {"product": product, "K": k_shot, "seed": seed})
            print(f"[p18-dev] {product} done", flush=True)

        integrity = exact_equal(rows, rc.load_json(p13.OUT["dev"] / "p13_dev_results.json")["rows"], MAIN, ("product", "K", "seed"))
        if integrity >= 1e-12:
            raise rc.P6BFailClosedError(f"dev integrity vs P13 failed: {integrity}")
        ba = mean_dev_ba(rows)
        crop_v = select_crop(ba)
        selection = {"crop": list(crop_v), "dinov2_b": select_config(ba, "dinov2_b"),
                     "clip_b16": select_config(ba, "clip_b16"), "dinov2_b_sha256": backs["dinov2_b"].sha256}
        purity_summary = summarize_purity(entries, rows)
        dist_summary = {k: statistics.fmean(e / t for e, t in v if t > 0) for k, v in dist.items()}
        share_summary = {k: statistics.fmean(s[k] for s in shares) for k in ("mask", "near", "far")}
        result = {"schema_version": "p18_dev_v1", "protocol": PROTOCOL, "provenance": rc.git_provenance(),
                  "integrity_max_abs_diff_vs_p13": integrity, "selection": selection, "dev_mean_ba": ba,
                  "purity": purity_summary, "background_exceedance_by_distance": dist_summary,
                  "soft_weight_shares": share_summary, "rows": rows}
        text = dev_table(result)
        rc.write_json(staging / "p18_dev_results.json", result)
        (staging / "p18_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def summarize_purity(entries, rows) -> dict:
    out = {"n_image_seed_units": len(entries), "mean": {}, "by_rho_bin": {}, "per_product": {}}
    for s in bl.SCHEMES:
        out["mean"][s] = {k: statistics.fmean(e[f"{s}|{k}"] for e in entries) for k in ("any", "area")}
    for i, (lo, hi) in enumerate(bl.RHO_BINS):
        sub = [e for e in entries if bl.rho_bin(e["rho"]) == i]
        out["by_rho_bin"][f"{lo}-{hi}"] = {"n": len(sub), **({s: statistics.fmean(e[f"{s}|any"] for e in sub)
                                                              for s in bl.SCHEMES} if sub else {})}
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
    gains, dlam = [], []
    for p in rc.PRODUCTS:
        sub = [e for e in entries if e["product"] == p]
        out["per_product"][p] = {s: statistics.fmean(e[f"{s}|any"] for e in sub) for s in bl.SCHEMES}
        dlam.append(out["per_product"][p]["ours_soft"] - out["per_product"][p]["global"])
        gains.append(100 * statistics.fmean(cell[(ARR, p, k, s)] - cell[(SS, p, k, s)] for k in rp.DEV_K for s in rp.DEV_SEEDS))
    rho_s, p_s = stats.spearmanr(dlam, gains)
    out["spearman_purity_gain_vs_ba_gain"] = {"rho": float(rho_s), "p": float(p_s), "n": len(gains)}
    return out


def dev_table(result) -> str:
    ba, sel = result["dev_mean_ba"], result["selection"]
    lines = ["# P18 dev — adapted baselines, backbone configurations, evidence purity (dev only)", "",
             f"Integrity vs P13 dev (SimpleShot, fixed fraction, cross-class, A, ARR, CONS): max|Δ| = "
             f"{result['integrity_max_abs_diff_vs_p13']:.1e}. Selection: {json.dumps(sel)}", "",
             "## Baselines (dev mean BA over K = 1/3, %)", "",
             f"- main: SimpleShot {100 * ba[SS]:.2f}, CONS {100 * ba[CONS]:.2f}, A {100 * ba[A]:.2f}, "
             f"ARR {100 * ba[ARR]:.2f}, oracle (CLIP-L) {100 * ba['oracle_clipl']:.2f}",
             f"- DN4: alone {100 * ba['dn4']:.2f}, fused {100 * ba['dn4_fused']:.2f}"]
    lines += [f"- crop {tag(v)}: alone {100 * ba[f'crop_{tag(v)}']:.2f}, fused {100 * ba[f'crop_{tag(v)}_fused']:.2f}"
              for v in bl.CROP_VARIANTS]
    lines += ["", "## Backbone configurations (dev mean BA, %)", "",
              "| Config | SS | fixed fraction | cross-class | A | ARR | CONS | oracle | ARR headroom recovered |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cfg in dev_configs():
        v = {s: ba[f"{s}_{cfg}"] for s in ("ss", "nr", "xc", "a", "arr", "cons", "oracle")}
        frac = (v["arr"] - v["ss"]) / (v["oracle"] - v["ss"]) if v["oracle"] != v["ss"] else float("nan")
        mark = " **(selected)**" if cfg in (sel["dinov2_b"], sel["clip_b16"]) else ""
        lines.append(f"| {cfg}{mark} | " + " | ".join(f"{100 * v[s]:.2f}" for s in ("ss", "nr", "xc", "a", "arr", "cons", "oracle"))
                     + f" | {frac:.2f} |")
    pur = result["purity"]
    lines += ["", f"## Evidence purity (dev pool, {pur['n_image_seed_units']} image × seed units; masks for evaluation only)", "",
              "| Scheme | λ_any | λ_area | " + " | ".join(f"λ_any, ρ∈{k} (n={v['n']})" for k, v in pur["by_rho_bin"].items()) + " |",
              "|---|---|---|" + "---|" * len(pur["by_rho_bin"])]
    for s in bl.SCHEMES:
        lines.append(f"| {s} | {pur['mean'][s]['any']:.3f} | {pur['mean'][s]['area']:.3f} | " +
                     " | ".join(f"{v[s]:.3f}" if s in v else "—" for v in pur["by_rho_bin"].values()) + " |")
    sp = pur["spearman_purity_gain_vs_ba_gain"]
    lines += ["", f"Spearman over products, λ_any(ours_soft) − λ_any(global) vs dev BA gain ARR − SimpleShot: "
                  f"ρ = {sp['rho']:+.2f} (p = {sp['p']:.3f}, n = {sp['n']}).", "",
              "## Background exceedance (a > τ_0.99) of non-mask patches by Chebyshev distance to the mask", "",
              "| distance (cells) | " + " | ".join(result["background_exceedance_by_distance"]) + " |",
              "|---|" + "---|" * len(result["background_exceedance_by_distance"]),
              "| exceedance | " + " | ".join(f"{100 * v:.2f}%" for v in result["background_exceedance_by_distance"].values()) + " |",
              "", "Share of ARR's soft weight: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in result["soft_weight_shares"].items())
              + " (near = background within 2 cells, far = background ≥ 3 cells)."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- report (formal)

def contrast_pairs() -> list[tuple[str, str]]:
    pairs = [("crop_fused", SS), ("dn4_fused", SS), (ARR, "crop_fused"), (CONS, "dn4_fused"), (ARR, "dn4_fused"),
             ("crop", SS), ("dn4", SS)]
    for b in ALT:
        pairs += [(f"arr_{b}", f"ss_{b}"), (f"nr_{b}", f"ss_{b}"), (f"a_{b}", f"nr_{b}"), (f"arr_{b}", f"a_{b}"),
                  (f"cons_{b}", f"xc_{b}"), (f"cons_{b}", f"ss_{b}"), (f"arr_{b}", ARR)]
    return pairs


def write_report(staging, stage, provenance, integrity, sel, report, levels, rows, n_clusters, extra):
    result = {"schema_version": f"p18_{stage}_v1", "protocol": PROTOCOL, "label": "report-only; no main-method change",
              "provenance": provenance, "integrity_max_abs_diff_vs_p14": integrity, "selection": sel,
              "contrasts": report, "levels": levels, **extra, "raw_metrics": rows}
    unit = "products" if stage == "mvtec" else "VisA objects"
    lines = [f"# P18 {stage} — adapted baselines and backbone swap (report-only, single run)", "",
             f"Commit `{provenance['head']}`. Integrity vs P14 (SimpleShot, fixed fraction, cross-class, A, ARR, CONS): "
             f"max|Δ| = {integrity:.1e}. Selection (dev): {json.dumps(sel)}", "",
             "| Contrast | Δ BA pooled (95% CI) | +" + unit + f"/{n_clusters} | sign-flip p |", "|---|---|---|---|"]
    for name, v in report.items():
        lines.append(f"| {name} | {v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | "
                     f"{v['positive']} | {v['signflip_p']:.4f} |")
    lines += ["", "| Method | BA K=1 / 3 / 5 |", "|---|---|"]
    lines += [f"| {m} | " + " / ".join(f"{x:.2f}" for x in v) + " |" for m, v in levels.items()]
    if "oracle_headroom_recovered" in extra:
        lines += ["", "Oracle headroom recovered (arr − ss) / (oracle − ss), K = 1 / 3 / 5: " + "; ".join(
            f"{k} " + " / ".join(f"{x:.2f}" for x in v) for k, v in extra["oracle_headroom_recovered"].items())]
    if extra.get("mtd_descriptive"):
        lines += ["", "MTD (descriptive, Δ BA, seed mean at K = 1 / 3 / 5):"]
        lines += [f"- {k}: " + " / ".join(f"{x:+.2f}" for x in v) for k, v in extra["mtd_descriptive"].items()]
    text = "\n".join(lines)
    rc.write_json(staging / f"p18_{stage}_results.json", result)
    (staging / f"p18_{stage}_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


# --------------------------------------------------------------------------- MVTec-FS formal

def run_mvtec(mvtec_root: Path, device, batch_size) -> int:
    sel = dev_selection()
    crop_v = tuple(sel["crop"])
    with single_shot(OUT["mvtec"]) as staging:
        provenance = rc.git_provenance()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        alt = build_selected(sel, device)
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        by_key = {(q["product"], q["rgb_sha256"]): q for q in queries}
        raw_q, pools = query_annotation_records(), support_pool_records()
        rows = []
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            if [r["canonical_relative_path"] for r in pools[product]] != list(dev["rel_paths"]):
                raise rc.P6BFailClosedError(f"{product}: support-pool records differ from the cached features")
            images = ([open_rgb(rc.DATASET_ROOT / rel) for rel in dev["rel_paths"]] +
                      [open_rgb(by_key[(product, sha)]["image_path"]) for sha in qf["rgb_sha256"]])
            records = pools[product] + [raw_q[(product, sha)] for sha in qf["rgb_sha256"]]
            normals = [open_rgb(mvtec_root / product / "train" / "good" / x) for x in nf["names"]]
            g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
            clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
            masks = torch.cat([dev["mask_grid"], qf["mask_grid_oracle_only"]]).to(device)
            feats = {}
            for b, (backbone, layer) in alt.items():
                gb, pb = bb.extract(backbone, images, device, batch_size, layers=(layer,))
                _, pn = bb.extract(backbone, normals, device, batch_size, layers=(layer,))
                side = int(round(pb[layer].shape[1] ** 0.5))
                mb = torch.stack([mask_grid(r, im.size, side) for r, im in zip(records, images)])
                feats[b] = (gb.to(device), pb[layer].to(device).float(), pn[layer], mb.to(device))
            name_index = {x: i for i, x in enumerate(nf["names"])}
            q_true = [by_key[(product, sha)]["true_label"] for sha in qf["rgb_sha256"]]
            qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
            with torch.no_grad():
                for seed in rc.SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank_imgs = nf["clip18_patches"][idx].to(device).float()
                    parts, _, w = c1_parts(clip, bank_imgs, fa.tau99(bank_imgs))
                    crops = crop_features(factory, lambda i: images[i], len(images), range(len(images)),
                                          {"crop": bl.crop_boxes(w, *crop_v)}, device, batch_size)
                    alt_parts = {}
                    for b, (gb, pb, pn, mb) in feats.items():
                        bank_b = pn[idx].to(device).float()
                        alt_parts[b] = c1_parts(pb, bank_b, fa.tau99(bank_b))[0]
                    for k_shot in rc.K_VALUES:
                        manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                        order = list(manifest["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in manifest["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        si = torch.as_tensor(s_idx, device=device)
                        n = len(order)
                        sc = whole_scores(g, clip, parts, si, s_lab, qi, n)
                        add_local_baselines(sc, bl.dn4_scores(clip[si], s_lab, clip[qi], n), crops, si, s_lab, qi, n, device)
                        sc["oracle_clipl"] = rp.oracle_maskpool(clip[si], masks[si], s_lab, clip[qi], masks[qi], n)
                        for b, (gb, pb, pn, mb) in feats.items():
                            sc.update(renamed(whole_scores(gb, pb, alt_parts[b], si, s_lab, qi, n), b))
                            sc[f"oracle_{b}"] = rp.oracle_maskpool(pb[si], mb[si], s_lab, pb[qi], mb[qi], n)
                        rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed})
            print(f"[p18-mvtec] {product} done", flush=True)
        integrity = exact_equal(rows, rc.load_json(ff.OUT["mvtec"] / "p14_mvtec_results.json")["raw_metrics"],
                                MAIN, ("product", "K", "seed"))
        if integrity >= 1e-9:
            raise rc.P6BFailClosedError(f"integrity vs P14 formal failed: {integrity}")
        grid = rc.metric_grid_from_rows(rows)
        report = {}
        for t, r in contrast_pairs():
            c = contrast(grid, r, "balanced_accuracy", target=t)
            report[f"{t} − {r}"] = {"delta_pp": 100 * c["Dbar"], "ci95_pp": [100 * x for x in c["Dbar_ci95"]],
                                    "positive": c["Dbar_positive_products"],
                                    "signflip_p": signflip_p(list(c["Dbar_per_product_mean_difference"].values()))}
        levels = {m: [100 * rc.product_macro_summary(grid, m)[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES]
                  for m in sorted(grid)}
        headroom = {}
        for name, (arr, ss, orc) in {"main (CLIP-L)": (ARR, SS, "oracle_clipl"),
                                     **{b: (f"arr_{b}", f"ss_{b}", f"oracle_{b}") for b in ALT}}.items():
            headroom[name] = [(levels[arr][i] - levels[ss][i]) / (levels[orc][i] - levels[ss][i]) for i in range(len(rc.K_VALUES))]
        write_report(staging, "mvtec", provenance, integrity, sel, report, levels, rows, 13,
                     {"oracle_headroom_recovered": headroom})
    return 0


# --------------------------------------------------------------------------- VisA / MTD

def external_task(task, order, factory, alt, crop_v, device, batch_size):
    dataset = task.split("__")[0]
    records = [json.loads(line) for line in (ee.BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
    draws = json.load((ee.BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
    by_sha = {r["rgb_sha256"]: r for r in records}
    queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
    normal = ee.NORMAL_CLASS[dataset]
    normal_candidates = [r["rgb_sha256"] for r in records
                         if r["partition"] == "support_candidate" and r["class_label"] == normal]
    query_shas = {r["rgb_sha256"] for r in queries}
    banks, needed, supports = {}, set(), set()
    for seed in rc.SEEDS:
        banks[seed] = ee.bank_draw(task, seed, normal_candidates, draws[normal][str(seed)]["K5"])
        support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
        if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
            raise rc.P6BFailClosedError(f"{task}/s{seed}: bank/support/query overlap")
        needed |= support_all | set(banks[seed])
        supports |= support_all
    shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
    index = {s: i for i, s in enumerate(shas)}
    paths = [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas]
    g, clip = ee.extract(factory, paths, device, batch_size)  # the P14 path
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_patches = clip[qi.cpu()]
    g = g.to(device)
    feats = {}
    for b, (backbone, layer) in alt.items():
        gb, pb = bb.extract(backbone, paths, device, batch_size, layers=(layer,))
        feats[b] = (gb.to(device), pb[layer])
    per_seed, boxes = {}, {}
    for seed in rc.SEEDS:
        bank_imgs = clip[torch.as_tensor([index[s] for s in banks[seed]])].to(device).float()
        per_seed[seed], w = c1_parts_chunked(clip, bank_imgs, fa.tau99(bank_imgs), device)
        boxes[seed] = bl.crop_boxes(w, *crop_v)
    wanted = sorted({index[s] for s in supports} | set(qi.tolist()))
    crops = crop_features(factory, lambda i: open_rgb(paths[i]), len(shas), wanted, boxes, device, batch_size)
    rows = []
    for seed in rc.SEEDS:
        bidx = torch.as_tensor([index[s] for s in banks[seed]])
        alt_parts = {}
        for b, (gb, pb) in feats.items():
            bank_b = pb[bidx].to(device).float()
            alt_parts[b] = c1_parts_chunked(pb, bank_b, fa.tau99(bank_b), device)[0]
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            n = len(order)
            si = torch.as_tensor(s_idx, device=device)
            sc = chunked_scores(g, clip, per_seed[seed], si, s_lab, qi, q_patches, n, device)
            sup = clip[si.cpu()].to(device).float()
            dn4 = torch.cat([bl.dn4_scores(sup, s_lab, ch.to(device).float(), n) for ch in q_patches.split(ee.Q_CHUNK)])
            add_local_baselines(sc, dn4, {"crop": crops[seed]}, si, s_lab, qi, n, device)
            for b, (gb, pb) in feats.items():
                sc.update(renamed(chunked_scores(gb, pb, alt_parts[b], si, s_lab, qi, pb[qi.cpu()], n, device), b))
            rows += fa.score_rows(sc, order, q_true, {"task": task, "K": k_shot, "seed": seed})
    return rows


def run_external(device, batch_size) -> int:
    sel = dev_selection()
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        alt = build_selected(sel, device)
        rows = []
        for task in ee.TASKS:
            with torch.no_grad():
                rows += external_task(task, orders[task], factory, alt, tuple(sel["crop"]), device, batch_size)
            print(f"[p18-ext] {task} done", flush=True)
        integrity = exact_equal(rows, rc.load_json(ff.OUT["external"] / "p14_external_results.json")["raw_metrics"],
                                MAIN, ("task", "K", "seed"))
        if integrity >= 1e-9:
            raise rc.P6BFailClosedError(f"integrity vs P14 external failed: {integrity}")
        grid = ee.grid_of(rows)
        report, mtd = {}, {}
        for t, r in contrast_pairs():
            c = ee.visa_contrast(grid, t, r, "balanced_accuracy")
            objs = c["per_K"]["1"]["per_object"].keys()
            per_obj = [statistics.fmean(c["per_K"][str(k)]["per_object"][o] for k in rc.K_VALUES) for o in objs]
            report[f"{t} − {r}"] = {"delta_pp": 100 * c["Dbar"], "ci95_pp": [100 * x for x in c["Dbar_ci95"]],
                                    "positive": c["Dbar_positive_objects"], "signflip_p": signflip_p(per_obj)}
            mc = ee.mtd_contrast(grid, t, r, "balanced_accuracy")
            mtd[f"{t} − {r}"] = [100 * mc[str(k)]["mean"] for k in rc.K_VALUES]
        levels = {m: [100 * ee.visa_macro(grid, m)[str(k)]["mean"] for k in rc.K_VALUES] for m in sorted(grid)}
        write_report(staging, "external", provenance, integrity, sel, report, levels, rows, 11, {"mtd_descriptive": mtd})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "mvtec", "external"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage in ("dev", "mvtec") and not args.mvtec_root:
        raise SystemExit("--mvtec-root is required for the dev and mvtec stages")
    if args.stage == "dev":
        return run_dev(Path(args.mvtec_root), device, args.batch_size)
    if args.stage == "mvtec":
        return run_mvtec(Path(args.mvtec_root), device, args.batch_size)
    return run_external(device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
