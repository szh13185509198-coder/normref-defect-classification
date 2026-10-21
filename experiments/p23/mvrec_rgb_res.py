#!/usr/bin/env python3
"""P23: MVREC-RGB adaptation re-run at 336 px (AlphaCLIP ViT-L/14@336px), report-only.

Protocol: docs/P23_MVREC_RGB_336_PROTOCOL.md. The P6 MVREC-RGB code is reused unchanged: the official
support pipeline (``support_runner.run_support_unit``: official ClipModel, Zip-Adapter-F, ft_epo 500,
all P6 gates) and the official query path (``query_evaluator.rebuild_model_for_unit`` +
``_query_rows_for_unit``). The only change is the input resolution / backbone, applied at run time
to the four module constants that carry it; the official AlphaClip backbone already supports
"ViT-L/14@336px" through ``MVREC_ALPHA_L14_336_CKPT``.

Stages (each once; a stopped formal run continues only with --resume, after authorization):
    --stage repro   224 px, one unit (bottle K=1 seed 111): the wrapper must reproduce the archived
                    P6 predictions of that unit exactly (proves the wrapper changes nothing else)
    --stage formal  336 px, all 195 units, per-unit checkpoints, then metrics and paired contrasts

    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p23.mvrec_rgb_res --stage repro --alpha-ckpt model\\clip_l14_grit20m_fultune_2xe.pth
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p23.mvrec_rgb_res --stage formal --alpha-ckpt model\\clip_l14_336_grit_20m_4xe.pth
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6 import p6b_stage3_common as stage3  # noqa: E402
from experiments.p6.mvrec_rgb import rgb_dataset, rgb_model, rgb_param  # noqa: E402
from experiments.p6.mvrec_rgb_formal import query_evaluator as qe  # noqa: E402
from experiments.p6.mvrec_rgb_formal import support_runner as sr  # noqa: E402
from experiments.p6.mvrec_rgb_formal.execution_manifest import MvrecRgbUnit, expected_units, load_frozen_manifest  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p23"
PROTOCOL = "docs/P23_MVREC_RGB_336_PROTOCOL.md"
GUARDED = ("experiments/p23", "experiments/p6/mvrec_rgb", "experiments/p6/mvrec_rgb_formal", "MVREC-main", PROTOCOL)
ARCHIVE_224 = PROJECT_ROOT / "ckpt" / "mvrec_rgb_formal_query_227300b.tar.gz"
REPRO_UNIT = MvrecRgbUnit("bottle", 1, 111)
SETTINGS = {224: "ViT-L/14", 336: "ViT-L/14@336px"}


def configure(resolution: int, alpha_ckpt: Path) -> None:
    """Point the four modules that carry the input resolution / backbone at ``resolution``."""
    backbone = SETTINGS[resolution]
    for module in (rgb_param, rgb_model):
        module.OFFICIAL_INPUT_SHAPE = resolution
        module.OFFICIAL_BACKBONE = backbone
    rgb_dataset.OFFICIAL_INPUT_SHAPE = resolution
    qe.OFFICIAL_INPUT_SHAPE = resolution
    if resolution == 336:  # the official AlphaClip reads this variable for ViT-L/14@336px
        os.environ["MVREC_ALPHA_L14_336_CKPT"] = str(Path(alpha_ckpt).resolve())
    # the official classifier dumps debug features after training (no effect on results); keep them here
    os.environ["MVREC_FEATURE_SAVE_DIR"] = str(OUT / "official_feature_dumps")


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    return {"commit": commit, "dirty": bool(status.strip()), "allow_dirty_diagnostic": False}


def queries_by_product() -> dict:
    queries, _ = stage3.load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=True)
    out = {}
    for q in queries:
        out.setdefault(q["product"], []).append(q)
    return out


def run_unit(unit, grid: Path, alpha_ckpt: Path, device: str, execution: dict, queries: list, batch_size: int) -> list:
    """Official support training (skipped if the unit was already published), then official query inference."""
    unit_dir = unit.directory(grid)
    if not unit_dir.exists():
        manifest = load_frozen_manifest(rc.STAGE1_BUNDLE, unit)
        sr.run_support_unit(unit=unit, manifest=manifest, manifest_path=rc.STAGE1_BUNDLE / unit.manifest_relative_path,
                            image_root=rc.DATASET_ROOT, alpha_ckpt=alpha_ckpt, device=device, output_dir=unit_dir,
                            execution=execution)
    config = rc.load_json(unit_dir / sr.RUN_CONFIG)
    if config["input_shape"] != rgb_dataset.OFFICIAL_INPUT_SHAPE:
        raise rc.P6BFailClosedError(f"{unit.key}: unit trained at {config['input_shape']} px")
    model = qe.rebuild_model_for_unit(unit_dir, alpha_ckpt, device)
    rows = qe._query_rows_for_unit(model, unit, queries, device, batch_size=batch_size)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


def archived_224_rows() -> list:
    with tarfile.open(ARCHIVE_224) as archive:
        text = archive.extractfile(qe.PREDICTIONS_NAME).read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@contextlib.contextmanager
def single_shot(out: Path, resume: bool):
    state = git_state()
    if state["dirty"]:
        raise rc.P6BFailClosedError("uncommitted changes in guarded paths")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; this P23 stage is single-shot")
    lock, staging = out.with_name(out.name + ".lock"), out.with_name(out.name + ".staging")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and not resume:
        raise rc.P6BFailClosedError(f"{lock} exists: continue only with --resume after the user authorizes it "
                                    "(nothing is deleted)")
    lock.mkdir(exist_ok=True)
    staging.mkdir(exist_ok=True)
    log = staging / "launches.json"
    launches = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    launches.append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": resume, **state})
    rc.write_json(log, launches)
    yield staging, state
    staging.rename(out)


def run_repro(alpha_ckpt: Path, device: str, batch_size: int) -> int:
    configure(224, alpha_ckpt)
    with single_shot(OUT / "repro", False) as (staging, state):
        rows = run_unit(REPRO_UNIT, staging / "grid", alpha_ckpt, device, state,
                        queries_by_product()[REPRO_UNIT.product], batch_size)
        ref = {r["rgb_sha256"]: r["predicted_label"] for r in archived_224_rows()
               if (r["product"], r["K"], r["seed"]) == (REPRO_UNIT.product, REPRO_UNIT.k_shot, REPRO_UNIT.seed)}
        new = {r["rgb_sha256"]: r["predicted_label"] for r in rows}
        same = sum(new[s] == ref.get(s) for s in new)
        result = {"unit": REPRO_UNIT.key, "queries": len(new), "archived": len(ref), "identical_predictions": same,
                  "exact": same == len(ref) == len(new), "alpha_ckpt_sha256": stage3.file_sha256(alpha_ckpt)}
        rc.write_json(staging / "repro_result.json", result)
    print(json.dumps(result, indent=1))
    if not result["exact"]:
        raise rc.P6BFailClosedError("the 224 px wrapper does not reproduce the archived P6 predictions")
    return 0


def contrasts(new_rows: list) -> dict:
    from experiments.p10.formal_evaluate import mvrec_rows
    from experiments.revision.c1_paired_bootstrap import contrast
    grid = rc.metric_grid_from_rows(rc.load_json(PROJECT_ROOT / "ckpt/p14_formal/mvtec/p14_mvtec_results.json")["raw_metrics"])
    grid.update(rc.metric_grid_from_rows(mvrec_rows()))
    grid.update(rc.metric_grid_from_rows([{**r, "method": "mvrec_rgb_336"} for r in new_rows]))
    pairs = {"MVREC-RGB 336 - 224": ("mvrec_rgb_336", "mvrec_rgb"),
             "ARR - MVREC-RGB 336": ("fused_adareg_regres_soft_q99", "mvrec_rgb_336"),
             "SimpleShot - MVREC-RGB 336": ("simpleshot_cl2n", "mvrec_rgb_336"),
             "consensus (C0) - MVREC-RGB 336": ("fused_cons_soft_supq90", "mvrec_rgb_336")}
    out = {}
    for name, (t, r) in pairs.items():
        c = contrast(grid, r, "balanced_accuracy", target=t)
        out[name] = {"per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()}, "Dbar_pp": 100 * c["Dbar"],
                     "ci95_pp": [100 * x for x in c["Dbar_ci95"]], "positive_products": c["Dbar_positive_products"]}
    levels = {}
    for m in ("mvrec_rgb_336", "mvrec_rgb", "fused_adareg_regres_soft_q99", "simpleshot_cl2n"):
        s = rc.product_macro_summary(grid, m)
        levels[m] = {str(k): {"mean": 100 * s[str(k)]["balanced_accuracy"]["mean"],
                              "sd": 100 * s[str(k)]["balanced_accuracy"]["std_ddof1"]} for k in rc.K_VALUES}
    return {"levels_ba_pct": levels, "contrasts": out}


def run_formal(alpha_ckpt: Path, device: str, batch_size: int, resume: bool) -> int:
    repro = OUT / "repro" / "repro_result.json"
    if not repro.is_file() or not rc.load_json(repro)["exact"]:
        raise rc.P6BFailClosedError("run --stage repro first (it must pass)")
    configure(336, alpha_ckpt)
    with single_shot(OUT / "formal", resume) as (staging, state):
        queries = queries_by_product()
        pred_dir = staging / "unit_predictions"
        pred_dir.mkdir(exist_ok=True)
        all_rows = []
        for i, unit in enumerate(expected_units(), 1):
            path = pred_dir / f"{unit.product}_k{unit.k_shot}_s{unit.seed}.json"
            if path.exists():
                all_rows += json.loads(path.read_text(encoding="utf-8"))
                continue
            t0 = time.time()
            rows = run_unit(unit, staging / "grid", alpha_ckpt, device, state, queries[unit.product], batch_size)
            rc.write_json(path, rows)
            all_rows += rows
            print(f"[p23] {i}/195 {unit.key} {time.time() - t0:.0f}s", flush=True)
        metric_rows, macro = qe._aggregate_metrics(all_rows)
        result = {"schema_version": "p23_mvrec_rgb_336_v1", "protocol": PROTOCOL, "resolution": 336,
                  "backbone": SETTINGS[336], "alpha_ckpt_sha256": stage3.file_sha256(alpha_ckpt),
                  "launch": state, "raw_metrics": [{**r, "method": "mvrec_rgb_336"} for r in metric_rows],
                  "product_macro_per_seed": macro, **contrasts(metric_rows)}
        rc.write_json(staging / "p23_results.json", result)
        rc.write_jsonl(staging / "p23_predictions.jsonl", all_rows)
    lv = result["levels_ba_pct"]
    print("BA % K=1/3/5: " + "; ".join(f"{m} " + " / ".join(f"{lv[m][str(k)]['mean']:.2f}" for k in rc.K_VALUES) for m in lv))
    for name, c in result["contrasts"].items():
        print(f"  {name}: {c['Dbar_pp']:+.2f} [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}] ({c['positive_products']}/13)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("repro", "formal"))
    parser.add_argument("--alpha-ckpt", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)   # as the P6 query evaluator
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.alpha_ckpt.is_file():
        raise SystemExit(f"missing {args.alpha_ckpt}")
    if args.stage == "repro":
        return run_repro(args.alpha_ckpt, args.device, args.batch_size)
    return run_formal(args.alpha_ckpt, args.device, args.batch_size, args.resume)


if __name__ == "__main__":
    sys.exit(main())
