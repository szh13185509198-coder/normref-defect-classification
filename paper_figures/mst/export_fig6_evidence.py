#!/usr/bin/env python3
"""Export the data of MST Figure 6 (evidence regions on three MVTec-FS queries); run once on the local machine.

Same computation and the same three queries as the original Figure 5 (`experiments/p12/make_figures_v3.py`,
`fig_region_comparison`): for the three products with the largest gain of the normal-calibrated region over the fixed
fraction, the first query (by SHA-256) that the proposed method classifies correctly and the fixed fraction does not
(K = 5, seed 111; bank: seed-111 draw, M = 20). The selected SHA-256s must equal those of the original Figure 5 CSV.

Output (write-once): paper_figures/data/ckpt/fig6_evidence/
  fig6_evidence.npz   per query q = 0, 1, 2: view_q (336 x 336 x 3 uint8, the bicubic 336-px view), a_q (24 x 24 patch
                      anomaly a(t)), w_q (24 x 24 normal-calibrated weights), fixed_q / otsu_q / mebin_q (24 x 24 bool),
                      mask_q (24 x 24 ground-truth patch grid, evaluation only)
  fig6_evidence.json  product, SHA-256, labels and predictions, tau, patch counts, purity lambda, GT polygons in
                      336-px coordinates

    D:\\Anaconda\\envs\\cole\\python.exe paper_figures/mst/export_fig6_evidence.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (before torch, as the P12 scripts)
import torch  # noqa: E402

from experiments.p10 import make_figures as mf  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12 import make_figures_v3 as v3  # noqa: E402
from experiments.p12.thresholds import tau_mebin, tau_otsu  # noqa: E402
from experiments.p18 import baselines as bl  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "paper_figures" / "data" / "ckpt" / "fig6_evidence"
ORIGINAL_CSV = PROJECT_ROOT / "paper_figures" / "legacy_main" / "fig05_region_comparison" / "fig05_region_comparison.csv"


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the export is write-once")
    expected = [(r["product"], r["rgb_sha256"]) for r in csv.DictReader(ORIGINAL_CSV.open(encoding="utf-8"))]
    cell = v3.formal_cells()
    gains = {p: statistics.fmean(cell[(v3.A, p, k, s)] - cell[(v3.NR, p, k, s)] for k in v3.K for s in rc.SEEDS)
             for p in rc.PRODUCTS}
    products = sorted(rc.PRODUCTS, key=lambda p: -gains[p])[:3]
    preds = v3.k5s111_predictions((v3.A, v3.NR))
    records = v3.query_records()
    arrays, meta = {}, {"source": "experiments/p12/make_figures_v3.py fig_region_comparison (K = 5, seed 111, M = 20)",
                        "grid": 24, "view_px": 336, "queries": []}
    for q, product in enumerate(products):
        qf, bank_imgs, tau = v3.query_context(product)
        shas = qf["rgb_sha256"]
        pick = next((i for i in sorted(range(len(shas)), key=lambda j: shas[j])
                     if preds[(v3.A, product, shas[i])][0] == preds[(v3.A, product, shas[i])][1]
                     and preds[(v3.NR, product, shas[i])][0] != preds[(v3.NR, product, shas[i])][1]), None)
        if pick is None or (product, shas[pick]) != expected[q]:
            raise SystemExit(f"query {q}: selection {product}/{None if pick is None else shas[pick]} differs from the "
                             f"original Figure 5 ({expected[q]})")
        sha = shas[pick]
        rec = records[(product, sha)]
        image = v3.image_of(rec)
        w0, h0 = image.size
        view = np.asarray(image.convert("RGB").resize((336, 336), mf.Image.BICUBIC))
        with torch.no_grad():
            a = ra.patch_anomaly(qf["clip18_patches"][pick:pick + 1].float(), bank_imgs.reshape(-1, bank_imgs.shape[-1]))
            fixed = torch.zeros_like(a).scatter_(1, a.topk(math.ceil(rp.RHO * a.shape[1]), dim=1).indices, 1.0)[0] > 0
            otsu, mebin = (a > tau_otsu(a))[0], (a > tau_mebin(a))[0]
            w = ab.region_weights(a, tau, soft=True)[0]
            mask = qf["mask_grid_oracle_only"][pick:pick + 1].float()
            lam = {name: float(bl.purity(r[None].float(), mask)[0])
                   for name, r in (("fixed", fixed), ("otsu", otsu), ("mebin", mebin), ("ours", w))}
        side = int(round(a.shape[1] ** 0.5))
        arrays.update({f"view_{q}": view, f"a_{q}": a[0].reshape(side, side).numpy().astype(np.float32),
                       f"w_{q}": w.reshape(side, side).numpy().astype(np.float32),
                       f"fixed_{q}": fixed.reshape(side, side).numpy(), f"otsu_{q}": otsu.reshape(side, side).numpy(),
                       f"mebin_{q}": mebin.reshape(side, side).numpy(),
                       f"mask_{q}": mask[0].reshape(side, side).numpy().astype(np.float32)})
        meta["queries"].append({
            "q": q, "product": product, "rgb_sha256": sha, "true": preds[(v3.A, product, sha)][1],
            "pred_ours": preds[(v3.A, product, sha)][0], "pred_fixed": preds[(v3.NR, product, sha)][0],
            "gain_over_fixed_pp": 100 * gains[product], "tau": float(tau),
            "patches": {"fixed": int(fixed.sum()), "otsu": int(otsu.sum()), "mebin": int(mebin.sum()),
                        "ours_nonzero": int((w > 0).sum())},
            "purity_lambda": lam,
            "gt_polygons_336": [[[x * 336 / w0, y * 336 / h0] for x, y in poly] for poly in mf.polygons_of(rec)]})
        print(f"[fig6] {product}: {sha[:12]} true {meta['queries'][-1]['true']}; ours {int((w > 0).sum())} patches, "
              f"lambda {lam['ours']:.2f}; fixed lambda {lam['fixed']:.2f}", flush=True)
    staging = OUT.with_name(OUT.name + ".staging")
    staging.mkdir(parents=True)
    np.savez_compressed(staging / "fig6_evidence.npz", **arrays)
    (staging / "fig6_evidence.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                                newline="\n")
    staging.rename(OUT)
    print(f"[fig6] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
