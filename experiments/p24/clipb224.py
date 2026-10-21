#!/usr/bin/env python3
"""P24 (paper to-do A13): our methods on CLIP ViT-B/16 at 224 px on the MVTec-FS formal set (report-only).

Purpose: a comparison with the official CLIP-LoRA (ViT-B/16, 224 px, P21) that matches both backbone
and input resolution. The layer follows the P18 dev rule restricted to the 224-px candidates (read from
the P18 dev results, no new selection). Integrity: the same code at 336 px / the P18-selected layer must
reproduce the P18 formal CLIP-B/16 rows exactly. Disclosed: this matched comparison was added after a
partial look at P21 (32/195 episodes).

Usage (once):
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p24.clipb224 --mvtec-root D:\\intergration\\MVTEC
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import mask_grid, support_pool_records  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES, query_annotation_records  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18.run import SHORT, c1_parts, dev_selection, open_rgb, parse_config, renamed, whole_scores  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p24" / "mvtec"
GUARDED = ("experiments/p24", "experiments/p18", "experiments/p14", "experiments/p11", "experiments/p10")
P18_DEV = PROJECT_ROOT / "ckpt" / "p18" / "dev" / "p18_dev_results.json"
P18_MVTEC = PROJECT_ROOT / "ckpt" / "p18" / "mvtec" / "p18_mvtec_results.json"
TAG_224, TAG_336 = "clip_b16_r224", "clip_b16"   # the P18 formal rows are named *_clip_b16


def layer_224() -> int:
    """P18 dev rule (highest dev ARR, ties -> shallower layer), restricted to the 224-px candidates."""
    ba = rc.load_json(P18_DEV)["dev_mean_ba"]
    return max(bb.LAYERS, key=lambda L: (round(ba[f"arr_clip_b16_r224_L{L}"], 12), -L))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)   # as P18
    args = parser.parse_args(argv)
    device, mvtec_root = torch.device(args.device), Path(args.mvtec_root)
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    lock = OUT.with_name(OUT.name + ".lock")
    if OUT.exists() or lock.exists():
        raise rc.P6BFailClosedError(f"{OUT} or its lock exists; P24 runs once (report, do not delete)")
    lock.mkdir(parents=True)

    _, res336, layer336 = parse_config(dev_selection()["clip_b16"])
    configs = {TAG_224: (bb.ClipB(device, 224), layer_224()), TAG_336: (bb.ClipB(device, res336), layer336)}
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    by_key = {(q["product"], q["rgb_sha256"]): q for q in queries}
    raw_q, pools = query_annotation_records(), support_pool_records()
    rows = []
    for product in rc.PRODUCTS:
        dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        if [r["canonical_relative_path"] for r in pools[product]] != list(dev["rel_paths"]):
            raise rc.P6BFailClosedError(f"{product}: support-pool records differ from the cached features")
        path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
        n_pool = len(dev["rel_paths"])
        images = ([open_rgb(rc.DATASET_ROOT / rel) for rel in dev["rel_paths"]] +
                  [open_rgb(by_key[(product, sha)]["image_path"]) for sha in qf["rgb_sha256"]])
        records = pools[product] + [raw_q[(product, sha)] for sha in qf["rgb_sha256"]]
        normals = [open_rgb(mvtec_root / product / "train" / "good" / x) for x in nf["names"]]
        name_index = {x: i for i, x in enumerate(nf["names"])}
        q_true = [by_key[(product, sha)]["true_label"] for sha in qf["rgb_sha256"]]
        qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
        feats = {}
        for tag, (backbone, layer) in configs.items():
            gb, pb = bb.extract(backbone, images, device, args.batch_size, layers=(layer,))
            _, pn = bb.extract(backbone, normals, device, args.batch_size, layers=(layer,))
            side = int(round(pb[layer].shape[1] ** 0.5))
            mb = torch.stack([mask_grid(r, im.size, side) for r, im in zip(records, images)])
            feats[tag] = (gb.to(device), pb[layer].to(device).float(), pn[layer], mb.to(device))
        with torch.no_grad():
            for seed in rc.SEEDS:
                idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                parts = {}
                for tag, (gb, pb, pn, mb) in feats.items():
                    bank = pn[idx].to(device).float()
                    parts[tag] = c1_parts(pb, bank, fa.tau99(bank))[0]
                for k_shot in rc.K_VALUES:
                    manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                    order = list(manifest["class_order"])
                    s_idx, s_lab = [], []
                    for c, cls in enumerate(order):
                        for item in manifest["classes"][cls]["items"]:
                            s_idx.append(path_index[item["rel_path"]])
                            s_lab.append(c)
                    si, n = torch.as_tensor(s_idx, device=device), len(order)
                    sc = {}
                    for tag, (gb, pb, pn, mb) in feats.items():
                        sc.update(renamed(whole_scores(gb, pb, parts[tag], si, s_lab, qi, n), tag))
                        sc[f"oracle_{tag}"] = rp.oracle_maskpool(pb[si], mb[si], s_lab, pb[qi], mb[qi], n)
                    rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed})
        print(f"[p24] {product} done", flush=True)

    methods_336 = {f"{s}_{TAG_336}" for s in SHORT.values()} | {f"oracle_{TAG_336}"}
    integrity = exact_equal(rows, rc.load_json(P18_MVTEC)["raw_metrics"], methods_336, ("product", "K", "seed"))
    if integrity >= 1e-9:
        raise rc.P6BFailClosedError(f"integrity vs P18 formal CLIP-B/16 failed: {integrity}")
    grid = rc.metric_grid_from_rows(rows)
    levels = {m: {str(k): {"mean": 100 * s[str(k)]["balanced_accuracy"]["mean"],
                           "sd": 100 * s[str(k)]["balanced_accuracy"]["std_ddof1"]} for k in rc.K_VALUES}
              for m in sorted(grid) if m.endswith(TAG_224) for s in [rc.product_macro_summary(grid, m)]}
    within = {}
    for t, r in ((f"arr_{TAG_224}", f"ss_{TAG_224}"), (f"a_{TAG_224}", f"nr_{TAG_224}"),
                 (f"arr_{TAG_224}", f"arr_{TAG_336}")):
        c = contrast(grid, r, "balanced_accuracy", target=t)
        within[f"{t} - {r}"] = {"Dbar_pp": 100 * c["Dbar"], "ci95_pp": [100 * x for x in c["Dbar_ci95"]],
                                "positive_products": c["Dbar_positive_products"]}
    result = {"schema_version": "p24_clipb224_v1", "layer_224": configs[TAG_224][1], "layer_336": layer336,
              "integrity_max_abs_diff_vs_p18": integrity, "provenance": rc.git_provenance(),
              "levels_ba_pct": levels, "contrasts": within,
              "raw_metrics": [r for r in rows if r["method"].endswith(TAG_224)]}
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "p24_results.json", result)
    print(f"layer {result['layer_224']}; integrity vs P18 {integrity:.1e}")
    for m, v in levels.items():
        print(f"  {m:24s} " + " / ".join(f"{v[str(k)]['mean']:.2f}" for k in rc.K_VALUES))
    for name, c in within.items():
        print(f"  {name}: {c['Dbar_pp']:+.2f} [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}] ({c['positive_products']}/13)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
