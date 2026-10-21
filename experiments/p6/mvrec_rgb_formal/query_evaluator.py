"""MVREC-RGB formal query evaluator — fully implemented, gate CLOSED.

Implements the complete 13-product K=1/3/5 five-seed query evaluation for
the MVREC-RGB adaptation (Protocol A section 8):

- validate the 195-unit support grid (fail-closed);
- open the frozen query contract through the P6.B Stage 3 loader
  (RGB SHA verified against disk);
- per unit, rebuild the official model from its provenance (K recovered
  from the unit run_config, never passed as None) and load the trained
  head checkpoint through the official ``_init_weight`` path;
- inference strictly reuses the official ``ClipModel.forward`` locked
  ``infer_style="assemble_uncertainty"`` path with ``mulit_view=True`` and
  the constant all-ones mask placeholder — never a re-implemented
  mean-embedding approximation;
- produce per-query predictions plus per-(product,K,seed) metrics with the
  official ``classification_metrics`` definitions.

**GATE:** every entry point fail-closes unless ``--open-query-gate`` is
given (main-agent authorization).  Tests exercise the implementation with
fake models and synthetic contracts only; no real query pixel or label is
read in this branch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import (  # noqa: E402
    P6BFailClosedError,
    atomic_write_text,
    sha256_bytes,
)
from experiments.p6.mvrec_rgb.rgb_param import (  # noqa: E402
    OFFICIAL_INPUT_SHAPE,
)
from experiments.p6.mvrec_rgb_formal.execution_manifest import (  # noqa: E402
    GRID_COMPLETION,
    expected_units,
)
from experiments.p6.mvrec_rgb_formal.grid_audit import (  # noqa: E402
    load_json,
    validate_formal_grid,
)
from experiments.p6.mvrec_rgb_formal.support_runner import (  # noqa: E402
    HEAD_CHECKPOINT,
    RUN_CONFIG,
    SUPPORT_PROTO,
)
from experiments.p6 import p6b_stage3_common as stage3  # noqa: E402

PREDICTIONS_NAME = "mvrec_rgb_formal_query_predictions.jsonl"
PREDICTION_MANIFEST_NAME = "mvrec_rgb_formal_query_predictions_manifest.json"
AUDIT_NAME = "mvrec_rgb_formal_query_audit.json"
PREDICTION_SCHEMA = "mvrec_rgb_formal_query_prediction_v1"
PREDICTION_MANIFEST_SCHEMA = "mvrec_rgb_formal_query_predictions_manifest_v1"
AUDIT_SCHEMA = "mvrec_rgb_formal_query_audit_v1"

# Official mask normalization constants (modules/clip_preprocess.py
# mask_transform: Normalize(0.5, 0.26)); the placeholder equals the
# support-side constant so support and query features share semantics.
_MASK_MEAN = 0.5
_MASK_STD = 0.26

# Official CLIP normalization constants (modules/clip_preprocess.py).
_RGB_MEAN = (0.48145466, 0.4578275, 0.40821073)
_RGB_STD = (0.26862954, 0.26130258, 0.27577711)

QUERY_GATE_OPEN = False


# ---------------------------------------------------------------------------
# Checkpoint namespace compatibility (incident: authorized query attempt #1)
# ---------------------------------------------------------------------------
# The support runner serializes the head checkpoint from ``model.state_dict()``
# filtered by ``k.startswith("head.")``, so every saved key carries the exact
# ``head.`` prefix.  Loading that dict directly into ``model.head`` fails
# ``strict=True`` with Missing/Unexpected keys.  This is a deterministic
# serialization namespace mismatch, NOT a model/scientific problem:
# query contract and RGB verification had already happened, zero model
# inference completed, zero prediction rows and zero metrics were produced.
# The fix below is a checkpoint-namespace compatibility adapter only; it is
# independent of any query outcome and changes no support artifact, no
# protocol, no query contract, no class order, no batch size, no inference
# path and no metric definition.


def _strip_head_namespace(state_dict):
    """Strip exactly one ``head.`` prefix from checkpoint keys, fail-closed.

    Validation rules (any violation raises P6BFailClosedError):
    - the checkpoint must be a non-empty dict;
    - every key must be a string;
    - every key must start with the exact prefix ``head.``;
    - the remainder after stripping must be non-empty;
    - no duplicate key may appear after stripping (defensive: the strip
      map is injective, so this cannot trigger for well-formed dicts).

    Returns a new dict keyed by the stripped names, ready for
    ``model.head.load_state_dict(..., strict=True)``.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        raise P6BFailClosedError(
            "head checkpoint must be a non-empty dict, got "
            f"{type(state_dict).__name__}"
        )
    stripped = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise P6BFailClosedError(
                f"head checkpoint key is not a string: {key!r}"
            )
        if not key.startswith("head."):
            raise P6BFailClosedError(
                f"head checkpoint key {key!r} lacks the required 'head.' "
                "namespace prefix"
            )
        new_key = key[len("head."):]
        if not new_key:
            raise P6BFailClosedError(
                "head checkpoint contains the degenerate key 'head.'"
            )
        if new_key in stripped:
            raise P6BFailClosedError(
                f"duplicate head checkpoint key after namespace strip: "
                f"{new_key!r}"
            )
        stripped[new_key] = value
    return stripped


def rebuild_model_for_unit(unit_dir, alpha_ckpt, device, model_factory=None):
    """Rebuild the official model for one unit and load its trained head.

    K is recovered from the unit ``support_run_config.json`` (never passed
    as None).  ``model_factory(class_names, k_shot)`` defaults to the real
    official model constructor; tests inject a fake.  The head is
    re-constructed from the stored support proto through the official
    ``_init_weight`` path and then overwritten with the trained checkpoint
    (strict).
    """
    import torch

    from experiments.p6.mvrec_rgb.rgb_model import build_mvrec_rgb_model

    run_config = load_json(Path(unit_dir) / RUN_CONFIG)
    class_order = run_config["class_order"]
    k_shot = int(run_config["K"])

    if model_factory is None:
        def model_factory(class_names, k_shot_value):
            model, _ = build_mvrec_rgb_model(
                class_names=class_names,
                alpha_ckpt_path=str(alpha_ckpt),
                k_shot=k_shot_value,
                ft_epo=500,
                zip_config_index=6,
                seed=None,
            )
            return model

    model = model_factory(class_order, k_shot)
    weight_x, weight_y = torch.load(
        str(unit_dir / SUPPORT_PROTO), map_location="cpu", weights_only=True
    )
    head_state = torch.load(
        str(unit_dir / HEAD_CHECKPOINT), map_location="cpu", weights_only=True
    )
    head_state = _strip_head_namespace(head_state)
    model.set_mode("eval")
    model.head._init_weight(weight_x.to(device), weight_y.to(device), "onehot")
    model.head.load_state_dict({k: v.to(device) for k, v in head_state.items()},
                               strict=True)
    model.set_mode("eval")
    return model


def _load_query_views(record, device):
    """Whole-image RGB views for one query record: deterministic resize to
    the official input resolution plus the official CLIP normalization.
    Produces [B=1, V=1, C, H, W] with the fixed global view list."""
    import torch
    from PIL import Image
    from torchvision import transforms

    img = Image.open(record["image_path"]).convert("RGB")
    img = img.resize((OFFICIAL_INPUT_SHAPE, OFFICIAL_INPUT_SHAPE))
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_RGB_MEAN, _RGB_STD),
    ])(img)
    views = x.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, C, H, W]
    return views


def _predict_batch(model, image_views, device):
    """Official locked inference path — no re-implemented aggregation.

    Calls the official ``ClipModel.forward(input_dict, mulit_view=True)``
    which uses the locked ``infer_style="assemble_uncertainty"``
    (per-token head application + ``weighted_logits_with_uncertainty`` +
    softmax) and ``fvnq=1``.  The mask input is the constant all-ones
    placeholder; the RGB input is the whole-image view.  Eval mode, no
    gradient, dummy labels unused by the official eval path.

    NOTE the official shape contract: the official forward applies
    ``.squeeze()`` to the aggregated predicts, so the return is ``[C]`` for
    batch=1 and ``[B, C]`` for batch>1.  Callers MUST go through
    ``_predicted_indices``, which strictly validates both shapes.
    """
    import torch

    model.set_mode("eval")
    b = image_views.shape[0]
    constant_mask_value = (1.0 - _MASK_MEAN) / _MASK_STD
    input_dict = {
        "x": image_views.to(device),
        "masks": torch.full(
            (b, image_views.shape[1], 1,
             image_views.shape[3], image_views.shape[4]),
            constant_mask_value, device=device,
        ),
        "y": torch.zeros(b, dtype=torch.long, device=device),
    }
    with torch.no_grad():
        results = model.forward(input_dict, True)  # mulit_view=True (locked)
    return results["predicts"]


def _predicted_indices(predicts, n_classes):
    """Strictly interpret the official forward output shape.

    The official ``assemble_uncertainty`` path squeezes its output:
    batch=1 yields ``[C]`` and batch>1 yields ``[B, C]``.  Any other shape
    (e.g. a scalar or a wrong class count) is fail-closed — a prediction
    must never silently degrade to a constant class index.
    """
    import torch

    if predicts.dim() == 1:
        if predicts.shape[0] != n_classes:
            raise P6BFailClosedError(
                f"official forward returned [{predicts.shape[0]}] but "
                f"class universe has {n_classes} classes"
            )
        return [int(predicts.argmax())]
    if predicts.dim() == 2:
        if predicts.shape[1] != n_classes:
            raise P6BFailClosedError(
                f"official forward returned [B,{predicts.shape[1]}] but "
                f"class universe has {n_classes} classes"
            )
        return [int(row.argmax()) for row in predicts]
    raise P6BFailClosedError(
        f"official forward returned unsupported predicts shape "
        f"{tuple(predicts.shape)}"
    )


def _query_rows_for_unit(model, unit, product_queries, device, batch_size=16):
    """Batch inference over one unit's product queries.

    ``batch_size`` is honored: query views are stacked into real batches
    and the official forward is called once per chunk.  Predictions are
    interpreted through the strict ``_predicted_indices`` shape contract.
    """
    import torch

    rows = []
    for start in range(0, len(product_queries), batch_size):
        chunk = product_queries[start:start + batch_size]
        views = torch.cat([_load_query_views(record, device)
                           for record in chunk], dim=0)  # [B, V=1, C, H, W]
        predicts = _predict_batch(model, views, device)
        indices = _predicted_indices(predicts, len(chunk[0]["class_order"]))
        if len(indices) != len(chunk):
            raise P6BFailClosedError(
                f"batch inference returned {len(indices)} predictions "
                f"for {len(chunk)} queries"
            )
        for record, pred_index in zip(chunk, indices):
            rows.append({
                "schema_version": PREDICTION_SCHEMA,
                "method": "mvrec_rgb",
                "product": unit.product,
                "K": unit.k_shot,
                "seed": unit.seed,
                "rgb_sha256": record["rgb_sha256"],
                "true_label": record["true_label"],
                "predicted_index": int(pred_index),
                "predicted_label": record["class_order"][int(pred_index)],
                "class_order": record["class_order"],
            })
    return rows


def _aggregate_metrics(rows):
    """Per-(product,K,seed) official metrics + product-macro per (K, seed)."""
    import statistics

    metric_rows = []
    for unit in expected_units():
        unit_rows = [r for r in rows if r["product"] == unit.product
                     and r["K"] == unit.k_shot and r["seed"] == unit.seed]
        if not unit_rows:
            raise P6BFailClosedError(f"no prediction rows for {unit.key}")
        metrics = stage3.classification_metrics(
            [r["true_label"] for r in unit_rows],
            [r["predicted_label"] for r in unit_rows],
            unit_rows[0]["class_order"],
        )
        metric_rows.append({
            "method": "mvrec_rgb", "product": unit.product,
            "K": unit.k_shot, "seed": unit.seed, **metrics,
        })
    macro = []
    for k_shot in (1, 3, 5):
        for seed in (111, 222, 333, 444, 555):
            rows_k_s = [r for r in metric_rows if r["K"] == k_shot
                        and r["seed"] == seed]
            macro.append({
                "K": k_shot, "seed": seed,
                "top1_accuracy": statistics.fmean(
                    r["top1_accuracy"] for r in rows_k_s),
                "balanced_accuracy": statistics.fmean(
                    r["balanced_accuracy"] for r in rows_k_s),
                "macro_f1": statistics.fmean(r["macro_f1"] for r in rows_k_s),
            })
    return metric_rows, macro


def _git_head_and_dirty():
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    ).stdout.strip())
    return commit, dirty


def evaluate_query_grid(grid_root, records_path, dataset_root, stage1_bundle,
                        output_dir, alpha_ckpt, device=None, batch_size=16,
                        open_query_gate=False, model_factory=None):
    """Run the full formal query evaluation for the MVREC-RGB grid.

    GATED: fail-closed unless ``open_query_gate=True`` (main-agent
    authorization).  When opened:

    - requires a clean git tree (evaluator commit recorded);
    - validates the completed 195-unit support grid and records the grid
      completion SHA, the frozen records SHA and the protocol hashes;
    - evaluates every query with the official forward path in real batches
      (``batch_size`` honored);
    - publishes the predictions, manifest and audit artifacts through a
      sibling staging directory with a single atomic directory rename.
    """
    if not open_query_gate and not QUERY_GATE_OPEN:
        raise P6BFailClosedError(
            "query evaluation is gated pending main-agent authorization; "
            "re-open only after the 195-unit support grid passes its audit"
        )
    import torch

    from experiments.p6.mvrec_rgb.protocol_lock import compute_protocol_hashes

    if batch_size < 1:
        raise P6BFailClosedError("batch_size must be >= 1")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    grid_root = Path(grid_root)
    output_dir = Path(output_dir)

    evaluator_commit, evaluator_dirty = _git_head_and_dirty()
    if evaluator_dirty:
        raise P6BFailClosedError(
            "formal query evaluation requires a clean git tree"
        )

    validated = validate_formal_grid(grid_root,
                                     require_global_completion=True)
    # Locked AlphaCLIP weight identity gate: the CLI-provided checkpoint
    # MUST hash to the exact weight SHA256 pinned in the support-grid
    # backbone identities; any mismatch fail-closes BEFORE the query
    # contract or any model is touched.
    grid_alpha_sha = validated["backbone_identities"]["alpha_clip"][
        "weight_sha256"]
    cli_alpha_sha = stage3.file_sha256(Path(alpha_ckpt))
    if cli_alpha_sha != grid_alpha_sha:
        raise P6BFailClosedError(
            f"AlphaCLIP checkpoint SHA mismatch: CLI "
            f"{cli_alpha_sha} != support-grid locked {grid_alpha_sha}"
        )

    queries, contract = stage3.load_query_contract(
        Path(records_path), Path(dataset_root), Path(stage1_bundle),
        verify_rgb=True,
    )
    grid_completion_sha = stage3.file_sha256(grid_root / GRID_COMPLETION)
    records_sha = stage3.file_sha256(Path(records_path))
    protocol_hashes = compute_protocol_hashes()

    if output_dir.exists():
        raise P6BFailClosedError(f"query output directory exists: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".mvrec_rgb_query_staging")
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        all_rows = []
        for unit in expected_units():
            unit_dir = unit.directory(grid_root)
            model = rebuild_model_for_unit(unit_dir, alpha_ckpt, device,
                                           model_factory=model_factory)
            product_queries = [q for q in queries
                               if q["product"] == unit.product]
            all_rows.extend(_query_rows_for_unit(
                model, unit, product_queries, device, batch_size=batch_size
            ))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metric_rows, macro = _aggregate_metrics(all_rows)

        lines = [json.dumps(row, sort_keys=True, separators=(",", ":"))
                 for row in all_rows]
        predictions_path = staging / PREDICTIONS_NAME
        atomic_write_text(predictions_path, "\n".join(lines) + "\n")

        manifest = {
            "schema_version": PREDICTION_MANIFEST_SCHEMA,
            "status": "completed",
            "unit_count": 195,
            "prediction_count": len(all_rows),
            "predictions_relative_path": PREDICTIONS_NAME,
            "predictions_sha256": stage3.file_sha256(predictions_path),
            "evaluator_commit": evaluator_commit,
            "support_grid_completion_sha256": grid_completion_sha,
            "records_jsonl_sha256": records_sha,
            "alpha_clip_weight_sha256": grid_alpha_sha,
            "protocol_bundle": {"hash_mode": "canonical-lf",
                                 **protocol_hashes},
            "metrics": {"raw": metric_rows,
                        "product_macro_per_seed": macro},
        }
        with open(staging / PREDICTION_MANIFEST_NAME, "w", encoding="utf-8",
                  newline="\n") as handle:
            json.dump(manifest, handle, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")

        audit = {
            "schema_version": AUDIT_SCHEMA,
            "status": "completed",
            "formal_eligible": True,
            "query_pixels_read": True,   # honest record: the evaluator phase reads query pixels
            "query_labels_read": True,   # honest record: true labels used for metrics only
            "unit_count": 195,
            "prediction_count": len(all_rows),
            "raw_query_count": contract["raw_query_count"],
            "multi_type_excluded_count": contract["multi_type_excluded_count"],
            "formal_query_count": contract["formal_query_count"],
            "predictions_sha256": manifest["predictions_sha256"],
            "evaluator_commit": evaluator_commit,
            "evaluator_dirty": False,
            "support_grid_completion_sha256": grid_completion_sha,
            "records_jsonl_sha256": records_sha,
            "alpha_clip_weight_sha256": grid_alpha_sha,
            "protocol_bundle": {"hash_mode": "canonical-lf",
                                 **protocol_hashes},
        }
        with open(staging / AUDIT_NAME, "w", encoding="utf-8",
                  newline="\n") as handle:
            json.dump(audit, handle, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")

        os.replace(staging, output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "completed", "prediction_count": len(all_rows),
            "manifest": manifest, "audit": audit}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--records-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage1-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-ckpt", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--open-query-gate", action="store_true",
                        help="main-agent authorization flag")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_query_grid(
            grid_root=Path(args.grid_root),
            records_path=Path(args.records_path),
            dataset_root=Path(args.dataset_root),
            stage1_bundle=Path(args.stage1_bundle),
            output_dir=Path(args.output_dir),
            alpha_ckpt=Path(args.alpha_ckpt),
            device=args.device,
            batch_size=args.batch_size,
            open_query_gate=args.open_query_gate,
        )
        print(f"[mvrec-rgb query] {result['status']} "
              f"predictions={result['prediction_count']}")
        return 0
    except P6BFailClosedError as exc:
        print(f"[mvrec-rgb query] GATED/FAIL-CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
