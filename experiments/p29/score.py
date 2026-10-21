#!/usr/bin/env python3
"""P29: score CLIP-LoRA with ViT-L/14@336px locally and pair it (protocol docs/P29_CLIPLORA_L14_336_PROTOCOL.md).

Reads ckpt/p29/mvtec (copied back from the cloud) and existing result files only; query labels come from the local
frozen query contract.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p29.score
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p20.stats_holm import holm  # noqa: E402
from experiments.p21 import cliplora_runner as runner  # noqa: E402
from experiments.p21 import score as p21  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

RUN_DIR = PROJECT_ROOT / "ckpt" / "p29" / "mvtec"
P25_DIR = PROJECT_ROOT / "ckpt" / "p25" / "mvtec"
P17_MVTEC = PROJECT_ROOT / "ckpt" / "p17" / "mvtec" / "p17_mvtec_results.json"
OUT = RUN_DIR / "p29_results.json"
LORA_L14, ZS_L14 = "cliplora_l14_336", "clip_zeroshot_l14_336"
LORA_B16_336 = "cliplora_b16_r336"
A_R = "a_r"   # P17: the dual-branch method without the global score = CLIP-L/14@336 layer-18 patches only
DECLARED = {"N1 ARR(main) - CLIP-LoRA L/14@336": (p21.ARR, LORA_L14),
            "N2 a_r (CLIP-L/14@336 only) - CLIP-LoRA L/14@336 [backbone-matched]": (A_R, LORA_L14),
            "N3 CLIP-LoRA L/14@336 - CLIP-LoRA B/16@336": (LORA_L14, LORA_B16_336),
            "N4 CLIP-LoRA L/14@336 - SimpleShot(main)": (LORA_L14, p21.SS)}
DESCRIPTIVE = {"CLIP-LoRA L/14@336 - zero-shot CLIP L/14@336": (LORA_L14, ZS_L14),
               "(P24) ARR(CLIP-B/16@224) - CLIP-LoRA B/16@224": (p21.ARR_B16_224, p21.LORA),
               "(P25) ARR(CLIP-B/16@336) - CLIP-LoRA B/16@336": (p21.ARR_B16, LORA_B16_336)}
N_CELLS = len(rc.PRODUCTS) * len(rc.K_VALUES) * len(rc.SEEDS)


def check_records(run_dir: Path, doc: dict) -> None:
    """Every episode record must come from the ViT-L/14@336px path at 336 px."""
    for e in doc["episodes"]:
        rec = json.loads((run_dir / "episodes" / f"{runner.episode_key(e)}.json").read_text(encoding="utf-8"))
        if rec.get("backbone") != runner.L14_336 or rec.get("resolution") != 336 \
                or rec.get("loralib_layer_table") != "ViT-L/14":
            raise rc.P6BFailClosedError(f"{rec.get('key')} is not a ViT-L/14@336px episode")


def main() -> int:
    doc = runner.load_episodes(p21.EPISODES)
    for d, want in ((RUN_DIR, runner.L14_336), (P25_DIR, None)):
        marker = d / "backbone.txt"
        if want and (not marker.exists() or marker.read_text().strip() != want):
            raise rc.P6BFailClosedError(f"{d} is not bound to {want}")
    check_records(RUN_DIR, doc)
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows = [{**r, "method": {p21.LORA: LORA_L14, p21.ZS: ZS_L14}[r["method"]]}
            for r in p21.lora_rows(RUN_DIR, doc, truth)]
    rows_b16_336 = [{**r, "method": LORA_B16_336} for r in p21.lora_rows(P25_DIR, doc, truth) if r["method"] == p21.LORA]
    grid = p21.comparator_grid()
    grid.update(rc.metric_grid_from_rows(rows))
    grid.update(rc.metric_grid_from_rows(rows_b16_336))
    grid.update(rc.metric_grid_from_rows(p21.lora_rows(PROJECT_ROOT / "ckpt" / "p21" / "mvtec", doc, truth)))
    grid.update(rc.metric_grid_from_rows([r for r in rc.load_json(P17_MVTEC)["raw_metrics"] if r["method"] == A_R]))
    for m in {m for pair in (*DECLARED.values(), *DESCRIPTIVE.values()) for m in pair}:
        if len(grid.get(m, {})) != N_CELLS:
            raise rc.P6BFailClosedError(f"incomplete grid for {m}")
    declared = {n: p21.paired(grid, t, r) for n, (t, r) in DECLARED.items()}
    adj = holm({n: v["signflip_p"] for n, v in declared.items()})
    for n, v in declared.items():
        v["holm_p"] = adj[n]
    descriptive = {n: p21.paired(grid, t, r) for n, (t, r) in DESCRIPTIVE.items()}
    levels = {m: p21.summary(grid, m) for m in (LORA_L14, ZS_L14, LORA_B16_336, p21.LORA, p21.ARR, A_R,
                                                p21.ARR_B16, p21.ARR_B16_224, p21.SS)}
    out = {"schema_version": "p29_results_v1", "protocol": "docs/P29_CLIPLORA_L14_336_PROTOCOL.md",
           "levels_pct": levels, "declared_contrasts": declared, "descriptive_contrasts": descriptive,
           "train_seconds_total": sum(json.loads(p.read_text(encoding="utf-8"))["train_seconds"]
                                      for p in (RUN_DIR / "episodes").glob("*.json")),
           "launches": [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUN_DIR / "launches").glob("*.json"))],
           "raw_metrics": rows}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("BA (%) K=1/3/5, mean ± seed SD")
    for m, v in levels.items():
        print(f"  {m:30s} " + " | ".join(f"{v[str(k)]['balanced_accuracy']['mean']:.2f} ± "
                                         f"{v[str(k)]['balanced_accuracy']['sd']:.2f}" for k in rc.K_VALUES))
    for n, v in {**declared, **descriptive}.items():
        print(f"  {n:70s} {v['Dbar_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] "
              f"{v['positive_products']}/13 p={v['signflip_p']:.4f}" + (f" Holm {v['holm_p']:.4f}" if "holm_p" in v else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
