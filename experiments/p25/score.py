#!/usr/bin/env python3
"""P25: score CLIP-LoRA (ViT-B/16) at 336 px locally and pair it (protocol docs/P25_CLIPLORA_336_PROTOCOL.md).

Reads ckpt/p25/mvtec (copied back from the cloud) and existing result files only; query labels come from
the local frozen query contract.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p25.score
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

RUN_DIR = PROJECT_ROOT / "ckpt" / "p25" / "mvtec"
OUT = RUN_DIR / "p25_results.json"
LORA336, ZS336 = "cliplora_b16_r336", "clip_zeroshot_b16_r336"
DECLARED = {"M1 ARR(CLIP-B/16@336) - CLIP-LoRA@336 [matched]": (p21.ARR_B16, LORA336),
            "M2 CLIP-LoRA@336 - CLIP-LoRA@224": (LORA336, p21.LORA),
            "M3 ARR(main) - CLIP-LoRA@336": (p21.ARR, LORA336),
            "M4 CLIP-LoRA@336 - SimpleShot(main)": (LORA336, p21.SS)}


def main() -> int:
    doc = runner.load_episodes(p21.EPISODES)
    for p in (RUN_DIR / "episodes").glob("*.json"):
        if json.loads(p.read_text(encoding="utf-8")).get("resolution") != 336:
            raise rc.P6BFailClosedError(f"{p.name} is not a 336-px episode")
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows = [{**r, "method": {p21.LORA: LORA336, p21.ZS: ZS336}[r["method"]]}
            for r in p21.lora_rows(RUN_DIR, doc, truth)]
    grid = p21.comparator_grid()
    grid.update(rc.metric_grid_from_rows(rows))
    grid.update(rc.metric_grid_from_rows(p21.lora_rows(PROJECT_ROOT / "ckpt" / "p21" / "mvtec", doc, truth)))
    declared = {n: p21.paired(grid, t, r) for n, (t, r) in DECLARED.items()}
    adj = holm({n: v["signflip_p"] for n, v in declared.items()})
    for n, v in declared.items():
        v["holm_p"] = adj[n]
    if p21.ARR_B16_224 in grid:   # the P24 pair at 224, repeated for the two-resolution summary
        declared_224 = p21.paired(grid, p21.ARR_B16_224, p21.LORA)
    else:
        declared_224 = None
    levels = {m: p21.summary(grid, m) for m in (LORA336, p21.LORA, ZS336, p21.ARR_B16, p21.ARR_B16_224, p21.ARR, p21.SS)
              if m in grid}
    out = {"schema_version": "p25_results_v1", "protocol": "docs/P25_CLIPLORA_336_PROTOCOL.md",
           "levels_pct": levels, "declared_contrasts": declared, "matched_at_224_for_reference": declared_224,
           "launches": [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUN_DIR / "launches").glob("*.json"))],
           "raw_metrics": rows}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("BA (%) K=1/3/5")
    for m, v in levels.items():
        print(f"  {m:28s} " + " | ".join(f"{v[str(k)]['balanced_accuracy']['mean']:.2f} ± "
                                         f"{v[str(k)]['balanced_accuracy']['sd']:.2f}" for k in rc.K_VALUES))
    for n, v in declared.items():
        print(f"  {n:48s} {v['Dbar_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] "
              f"{v['positive_products']}/13 Holm {v['holm_p']:.4f}")
    if declared_224:
        print(f"  (224 matched, P24) {declared_224['Dbar_pp']:+.2f} [{declared_224['ci95_pp'][0]:+.2f}, "
              f"{declared_224['ci95_pp'][1]:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
