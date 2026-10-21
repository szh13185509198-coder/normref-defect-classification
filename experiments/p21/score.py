#!/usr/bin/env python3
"""P21: score the cloud CLIP-LoRA predictions locally and pair them with the frozen results.

Reads ckpt/p21/mvtec/ (copied back from the cloud) and existing result files only; query labels
come from the local frozen query contract. Protocol: docs/P21_CLIPLORA_MVTEC_FS_PROTOCOL.md.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p21.score
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract  # noqa: E402
from experiments.p10.formal_evaluate import mvrec_rows  # noqa: E402
from experiments.p20.stats_holm import exact_signflip, holm  # noqa: E402
from experiments.p21 import cliplora_runner as runner  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

RUN_DIR = PROJECT_ROOT / "ckpt" / "p21" / "mvtec"
EPISODES = PROJECT_ROOT / "docs" / "p21" / "P21_MVTEC_FS_EPISODES.json"
OUT = RUN_DIR / "p21_results.json"
LORA, ZS = "cliplora_b16", "clip_zeroshot_b16"
ARR, SS, CONS, XC = ("fused_adareg_regres_soft_q99", "simpleshot_cl2n", "fused_cons_soft_supq90",
                     "fused_xclass_clip18")
ARR_B16, SS_B16 = "arr_clip_b16", "ss_clip_b16"
# pre-declared contrasts (target, reference); L1-L5 form the Holm family, the rest are descriptive
DECLARED = {"L1 ARR(main) - CLIP-LoRA": (ARR, LORA),
            "L2 ARR(CLIP-B/16) - CLIP-LoRA": (ARR_B16, LORA),
            "L3 CLIP-LoRA - SimpleShot(main)": (LORA, SS),
            "L4 CLIP-LoRA - Tip-Adapter-F*": (LORA, "tip_adapter_f"),
            "L5 CLIP-LoRA - CLIP zero-shot": (LORA, ZS)}
DESCRIPTIVE = {"CLIP-LoRA - SimpleShot(CLIP-B/16)": (LORA, SS_B16),
               "CLIP-LoRA - MVREC-RGB": (LORA, "mvrec_rgb"),
               "CLIP-LoRA - trainable adaptation": (LORA, "ours"),
               "CLIP-LoRA - cross-class (C0)": (LORA, XC),
               "CLIP-LoRA - consensus (C0)": (LORA, CONS)}
# backbone AND resolution matched (P24, paper to-do A13); added after a partial look at 32/195 episodes
ARR_B16_224 = "arr_clip_b16_r224"
P24 = PROJECT_ROOT / "ckpt" / "p24" / "mvtec" / "p24_results.json"
MATCHED = {"ARR(CLIP-B/16@224) - CLIP-LoRA [matched; added after a partial look]": (ARR_B16_224, LORA)}
EXTENSION_MARGIN_PP = 2.0


def lora_rows(run_dir: Path, doc: dict, truth: dict) -> list[dict]:
    rows = []
    for e in doc["episodes"]:
        path = run_dir / "episodes" / f"{runner.episode_key(e)}.json"
        if not path.exists():
            raise rc.P6BFailClosedError(f"missing episode {path.name}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        want = [q["sha256"] for q in doc["queries"][e["product"]]]
        if (rec["schema_version"] != runner.SCHEMA or rec["class_order"] != e["class_order"]
                or [q["sha256"] for q in rec["queries"]] != want):
            raise rc.P6BFailClosedError(f"episode record does not match the episode file: {path.name}")
        if not rec["fidelity_share_class0"]["match"]:
            raise rc.P6BFailClosedError(f"fidelity check failed: {path.name}")
        y = [truth[(e["product"], q["sha256"])] for q in rec["queries"]]
        for method, field in ((LORA, "cliplora_pred"), (ZS, "zeroshot_pred")):
            rows.append({"method": method, "product": e["product"], "K": e["K"], "seed": e["seed"],
                         **classification_metrics(y, [q[field] for q in rec["queries"]], e["class_order"])})
    return rows


def comparator_grid() -> dict:
    grid = {}
    for rel in ("p14_formal/mvtec/p14_mvtec_results.json", "p18/mvtec/p18_mvtec_results.json"):
        grid.update(rc.metric_grid_from_rows(rc.load_json(PROJECT_ROOT / "ckpt" / rel)["raw_metrics"]))
    frozen = dict(rc.load_formal_metric_grid())
    frozen.update(rc.metric_grid_from_rows(mvrec_rows()))
    for m in ("ours", "tip_adapter_f", "mvrec_rgb"):
        grid[m] = frozen[m]
    if P24.is_file():
        grid.update(rc.metric_grid_from_rows(rc.load_json(P24)["raw_metrics"]))
    return grid


def summary(grid: dict, method: str) -> dict:
    s = rc.product_macro_summary(grid, method)
    return {str(k): {m: {"mean": 100 * s[str(k)][m]["mean"], "sd": 100 * s[str(k)][m]["std_ddof1"]}
                     for m in rc.METRICS} for k in rc.K_VALUES}


def paired(grid: dict, target: str, reference: str) -> dict:
    c = contrast(grid, reference, "balanced_accuracy", target=target)
    per = [100 * v for v in c["Dbar_per_product_mean_difference"].values()]
    return {"target": target, "reference": reference,
            "per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()},
            "Dbar_pp": 100 * c["Dbar"], "ci95_pp": [100 * x for x in c["Dbar_ci95"]],
            "positive_products": sum(v > 0 for v in per), "signflip_p": exact_signflip(per)}


def main() -> int:
    doc = runner.load_episodes(EPISODES)
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows = lora_rows(RUN_DIR, doc, truth)
    grid = comparator_grid()
    grid.update(rc.metric_grid_from_rows(rows))
    for m in (LORA, ZS, ARR, SS, ARR_B16, SS_B16, "tip_adapter_f"):
        if len(grid[m]) != len(rc.PRODUCTS) * len(rc.K_VALUES) * len(rc.SEEDS):
            raise rc.P6BFailClosedError(f"incomplete grid for {m}")
    declared = {name: paired(grid, t, r) for name, (t, r) in DECLARED.items()}
    adj = holm({name: v["signflip_p"] for name, v in declared.items()})
    for name, v in declared.items():
        v["holm_p"] = adj[name]
    levels = {m: summary(grid, m) for m in (LORA, ZS, ARR, ARR_B16, SS, SS_B16, "tip_adapter_f",
                                            "mvrec_rgb", "ours", XC, CONS)}
    gap = {k: levels[ARR][k]["balanced_accuracy"]["mean"] - levels[LORA][k]["balanced_accuracy"]["mean"]
           for k in levels[LORA]}
    launches = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUN_DIR / "launches").glob("*.json"))]
    out = {"schema_version": "p21_results_v1", "protocol": "docs/P21_CLIPLORA_MVTEC_FS_PROTOCOL.md",
           "levels_pct": levels, "declared_contrasts": declared,
           "descriptive_contrasts": {n: paired(grid, t, r) for n, (t, r) in
                                     {**DESCRIPTIVE, **(MATCHED if ARR_B16_224 in grid else {})}.items()},
           "extension_rule": {"margin_pp": EXTENSION_MARGIN_PP, "ARR_minus_CLIPLoRA_BA_pp": gap,
                              "triggered": any(v < EXTENSION_MARGIN_PP for v in gap.values())},
           "train_seconds_total": sum(json.loads(p.read_text(encoding="utf-8"))["train_seconds"]
                                      for p in (RUN_DIR / "episodes").glob("*.json")),
           "launches": launches, "raw_metrics": rows}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("BA (%) K=1/3/5, mean ± seed SD")
    for m, v in levels.items():
        print(f"  {m:32s} " + " | ".join(f"{v[str(k)]['balanced_accuracy']['mean']:.2f} ± "
                                         f"{v[str(k)]['balanced_accuracy']['sd']:.2f}" for k in rc.K_VALUES))
    for name, v in {**declared, **out["descriptive_contrasts"]}.items():
        print(f"  {name:40s} {v['Dbar_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] "
              f"{v['positive_products']}/13 p={v['signflip_p']:.4f}" + (f" Holm {v['holm_p']:.4f}" if "holm_p" in v else ""))
    print(f"extension rule triggered: {out['extension_rule']['triggered']} (ARR - CLIP-LoRA per K: {gap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
