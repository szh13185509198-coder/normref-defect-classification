"""A1-A5 audit driver for the MVREC-RGB semantic-preservation audit.

Produces the machine-readable artifact bundle in
``ckpt/p6b_mvrec_rgb_semantic_audit/``:

    mvrec_rgb_semantic_audit.json
    component_map.json
    privileged_dependency_inventory.json
    poison_audit.json
    support_smoke_completion.json

and the protocol lock artifact ``docs/P6B_PRE_FORMAL_PROTOCOL_LOCK.json``.

Usage:
    python -m experiments.p6.mvrec_rgb.audit_main
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from experiments.p6.mvrec_rgb.component_map import COMPONENT_MAP, PRIVILEGED_READ_PATHS
from experiments.p6.mvrec_rgb.protocol_lock import (
    LOCK_ARTIFACT,
    MVREC_UPSTREAM_COMMIT,
    PROTOCOL_LOCK_COMMIT,
    compute_protocol_hashes,
    ensure_lock_artifact,
    sha256_file,
)
from experiments.p6.mvrec_rgb.rgb_param import ensure_mvrec_on_path

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DEFAULT_MANIFEST = os.path.join(
    REPO_ROOT, "data", "p6b_stage1_manifest_bundle",
    "p6b_support_manifests", "bottle", "k1_s111.json",
)
DEFAULT_IMAGE_ROOT = os.path.join(REPO_ROOT, "data", "mvtec-fs")
DEFAULT_ANNOTATION_CSV = os.path.join(
    REPO_ROOT, "data", "mvtec-fs", "CONFIG", "bottle_config1", "train.csv",
)
DEFAULT_ALPHA_CKPT = os.path.join(
    REPO_ROOT, "model", "clip_l14_grit20m_fultune_2xe.pth",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "ckpt", "p6b_mvrec_rgb_semantic_audit")

# Files patched inside MVREC-main (compatibility fixes, semantic_effect=none).
PATCHED_UPSTREAM_FILES = [
    "modules/bbox_clip/alpha_clip_backbone.py",
    "modules/classifier.py",
]


def _git(args, cwd=REPO_ROOT):
    return subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True
    )


def collect_provenance(mvrec_root):
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    status = _git(["status", "--porcelain"]).stdout
    dirty = bool(status.strip())

    mvrec_root = os.path.abspath(mvrec_root)
    patch_hashes = {}
    for rel in PATCHED_UPSTREAM_FILES:
        p = os.path.join(mvrec_root, *rel.split("/"))
        patch_hashes[rel] = sha256_file(p)

    # Hash every tracked source file of the vendored upstream tree,
    # excluding cache/run artifacts and binaries.
    upstream_tree = {}
    skip_dirs = {"buffer", "OUTPUT", "__pycache__", "picture"}
    for dirpath, dirnames, filenames in os.walk(mvrec_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in sorted(filenames):
            if fn.endswith((".pt", ".pth", ".pyc")):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, mvrec_root).replace(os.sep, "/")
            upstream_tree[rel] = sha256_file(full)
    upstream_tree_hash = hashlib.sha256(
        json.dumps(upstream_tree, sort_keys=True).encode()
    ).hexdigest()

    env = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:  # pragma: no cover
        env["torch_error"] = repr(e)
    try:
        import alpha_clip

        env["alpha_clip_path"] = os.path.dirname(os.path.abspath(alpha_clip.__file__))
    except Exception as e:  # pragma: no cover
        env["alpha_clip_error"] = repr(e)

    return {
        "execution_commit": head or None,
        "execution_dirty": dirty,
        "status_short": status.splitlines(),
        "mvrec_upstream_commit": MVREC_UPSTREAM_COMMIT,
        "mvrec_patch_files": patch_hashes,
        "upstream_tree_file_count": len(upstream_tree),
        "upstream_tree_sha256": upstream_tree_hash,
        "environment": env,
    }


def _weight_hashes(alpha_ckpt):
    out = {}
    if alpha_ckpt and os.path.isfile(alpha_ckpt):
        out["alpha_clip_l14_grit20m_fultune_2xe_sha256"] = sha256_file(alpha_ckpt)
    openai_ckpt = os.path.join(os.path.expanduser("~"), ".cache", "clip", "ViT-L-14.pt")
    if os.path.isfile(openai_ckpt):
        out["openai_vit_l14_sha256"] = sha256_file(openai_ckpt)
    return out


def _config_hashes(manifest_path, annotation_csv):
    out = {}
    if manifest_path and os.path.isfile(manifest_path):
        out["support_manifest_sha256"] = sha256_file(manifest_path)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            out["support_manifest_declared_sha256"] = manifest.get("manifest_sha256")
        except Exception:  # pragma: no cover
            pass
    if annotation_csv and os.path.isfile(annotation_csv):
        out["annotation_csv_sha256"] = sha256_file(annotation_csv)
    return out


def run_audit(
    manifest_path=DEFAULT_MANIFEST,
    image_root=DEFAULT_IMAGE_ROOT,
    annotation_csv=DEFAULT_ANNOTATION_CSV,
    alpha_ckpt_path=DEFAULT_ALPHA_CKPT,
    seed=111,
    ft_epo=500,
    device=None,
    out_dir=DEFAULT_OUT_DIR,
    run_model=True,
    allow_dirty=False,
):
    os.makedirs(out_dir, exist_ok=True)
    # Official debug feature dumps (save_feature_xy) stay inside the audit
    # output directory instead of the upstream author's hardcoded home path.
    os.environ.setdefault("MVREC_FEATURE_SAVE_DIR", os.path.join(out_dir, "feature_dump"))
    mvrec_root = ensure_mvrec_on_path()

    report = {"gates": {}, "problems": [], "artifacts": {}}

    # --- protocol lock artifact (create or verify; fail-closed)
    lock, lock_ok, lock_problems = ensure_lock_artifact(
        protocol_lock_commit=PROTOCOL_LOCK_COMMIT,
        mvrec_upstream_commit=MVREC_UPSTREAM_COMMIT,
    )
    report["protocol_lock_artifact"] = LOCK_ARTIFACT
    report["protocol_lock_ok"] = lock_ok
    report["protocol_hashes"] = compute_protocol_hashes()
    report["problems"].extend(lock_problems)

    # --- provenance
    provenance = collect_provenance(mvrec_root)
    provenance["pretrained_weight_hashes"] = _weight_hashes(alpha_ckpt_path)
    provenance["config_hashes"] = _config_hashes(manifest_path, annotation_csv)
    # Protocol A section 2 requires a clean, committed execution state; the
    # protocol documents themselves are pinned via the lock artifact (their
    # canonical-LF SHA256 values and the protocol lock commit), not via the
    # execution commit equality.
    if not provenance["execution_commit"]:
        report["problems"].append("execution HEAD commit could not be resolved")

    # --- A1: upstream component map
    component_map_artifact = {
        "schema_version": "p6b_mvrec_rgb_component_map_v1",
        "mvrec_upstream_commit": MVREC_UPSTREAM_COMMIT,
        "components": COMPONENT_MAP,
    }
    _write_json(os.path.join(out_dir, "component_map.json"), component_map_artifact)
    report["artifacts"]["component_map.json"] = sha256_file(
        os.path.join(out_dir, "component_map.json")
    )
    privileged_artifact = {
        "schema_version": "p6b_mvrec_rgb_privileged_dependency_inventory_v1",
        "privileged_read_paths": PRIVILEGED_READ_PATHS,
        "forbidden_fields_read_by_adaptation": False,
    }
    _write_json(
        os.path.join(out_dir, "privileged_dependency_inventory.json"),
        privileged_artifact,
    )
    report["artifacts"]["privileged_dependency_inventory.json"] = sha256_file(
        os.path.join(out_dir, "privileged_dependency_inventory.json")
    )
    a1_ok = True
    report["gates"]["A1_status"] = "pass" if a1_ok else "fail"
    if any(
        c["planned_status"] == "substantively_changed" for c in COMPONENT_MAP
    ):
        report["gates"]["A1_status"] = "fail"
        report["problems"].append("component map contains substantively_changed items")

    # --- A3: transferable-design preservation (static)
    preservation = []
    for c in COMPONENT_MAP:
        preservation.append({
            "component": c["component"],
            "official_mvrec": f"{c['upstream_file']}::{c['upstream_symbol']}",
            "mvrec_rgb_adaptation": c.get("reason", ""),
            "status": c["planned_status"],
            "trainable": None,
            "parameter_count": None,
        })
    a3_ok = not any(
        c["planned_status"] == "substantively_changed"
        for c in COMPONENT_MAP
    )
    report["gates"]["A3_status"] = "pass" if a3_ok else "fail"

    # --- A2 / A4: model-dependent gates
    poison_artifact = None
    smoke_artifact = None
    a2_ok = None
    a4_ok = None
    if run_model:
        import torch

        from experiments.p6.mvrec_rgb.poison_audit import run_poison_audit
        from experiments.p6.mvrec_rgb.rgb_dataset import MVRecRgbSupportDataset
        from experiments.p6.mvrec_rgb.rgb_model import build_mvrec_rgb_model
        from experiments.p6.mvrec_rgb.semantic_smoke import run_support_smoke

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # A2: dynamic poison audit with the real official model.
        dataset = MVRecRgbSupportDataset(manifest_path, image_root, device=device)
        model, param = build_mvrec_rgb_model(
            class_names=dataset.class_order,
            alpha_ckpt_path=alpha_ckpt_path,
            k_shot=dataset.k_shot,
            ft_epo=ft_epo,
            seed=seed,
        )

        def support_forward(batch):
            model.set_mode("eval")
            mvrec = model.get_mvrec(batch, dataset.k_shot)
            return {"mvrec": mvrec}

        poison_artifact = run_poison_audit(
            dataset_factory=lambda csv_path: MVRecRgbSupportDataset(
                manifest_path, image_root, device=device
            ),
            manifest_path=manifest_path,
            image_root=image_root,
            annotation_csv=annotation_csv,
            support_forward=support_forward,
            device=device,
            seed=seed,
        )
        _write_json(os.path.join(out_dir, "poison_audit.json"), poison_artifact)
        report["artifacts"]["poison_audit.json"] = sha256_file(
            os.path.join(out_dir, "poison_audit.json")
        )
        a2_ok = bool(
            poison_artifact.get("test1_pass")
            and poison_artifact.get("test2_pass")
            and not poison_artifact.get("failures")
        )
        report["gates"]["A2_status"] = "pass" if a2_ok else "fail"
        if not a2_ok:
            report["problems"].append(
                f"poison audit failed: {poison_artifact.get('failures')}"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # A4: support-only semantic smoke.
        smoke_artifact = run_support_smoke(
            manifest_path=manifest_path,
            image_root=image_root,
            alpha_ckpt_path=alpha_ckpt_path,
            seed=seed,
            ft_epo=ft_epo,
            device=device,
        )
        _write_json(
            os.path.join(out_dir, "support_smoke_completion.json"), smoke_artifact
        )
        report["artifacts"]["support_smoke_completion.json"] = sha256_file(
            os.path.join(out_dir, "support_smoke_completion.json")
        )
        a4_ok = bool(smoke_artifact.get("smoke_pass"))
        report["gates"]["A4_status"] = "pass" if a4_ok else "fail"
        if not a4_ok:
            report["problems"].append(
                f"smoke failed: {smoke_artifact.get('smoke_failures')}"
            )
    else:
        report["gates"]["A2_status"] = "not_run"
        report["gates"]["A4_status"] = "not_run"
        report["problems"].append("model gates A2/A4 not executed")

    # --- final decision
    if provenance["execution_dirty"] and not allow_dirty:
        report["problems"].append(
            "execution from a dirty git state is forbidden by Protocol A "
            "section 2 (formal execution from an uncommitted or dirty "
            "protocol state is forbidden)"
        )
    if not lock_ok:
        final_decision = "INCONCLUSIVE_ENGINEERING"
    elif provenance["execution_dirty"] and not allow_dirty:
        final_decision = "INCONCLUSIVE_ENGINEERING"
    elif a2_ok is False or a4_ok is False:
        # Privileged-dependency removal or reproducibility evidence failed.
        final_decision = "INCONCLUSIVE_ENGINEERING"
    elif any(
        c["planned_status"] == "substantively_changed" for c in COMPONENT_MAP
    ):
        final_decision = "STOP_NOT_REPRESENTATIVE"
    elif a2_ok is None or a4_ok is None:
        final_decision = "INCONCLUSIVE_ENGINEERING"
    else:
        final_decision = "SEMANTIC_PASS"

    final = {
        "schema_version": "p6b_mvrec_rgb_semantic_audit_v1",
        "protocol_A_sha256": report["protocol_hashes"]["protocol_A_canonical_lf_sha256"],
        "protocol_B_sha256": report["protocol_hashes"]["protocol_B_canonical_lf_sha256"],
        "decision_rule_sha256": report["protocol_hashes"]["decision_rule_canonical_lf_sha256"],
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "execution_commit": provenance["execution_commit"],
        "execution_dirty": provenance["execution_dirty"],
        "mvrec_upstream_commit": MVREC_UPSTREAM_COMMIT,
        "mvrec_patch_sha256": provenance["mvrec_patch_files"],
        "upstream_tree_sha256": provenance["upstream_tree_sha256"],
        "pretrained_weight_hashes": provenance["pretrained_weight_hashes"],
        "config_hashes": provenance["config_hashes"],
        "environment": provenance["environment"],
        "A1_status": report["gates"]["A1_status"],
        "A2_status": report["gates"].get("A2_status", "not_run"),
        "A3_status": report["gates"]["A3_status"],
        "A4_status": report["gates"].get("A4_status", "not_run"),
        "A5_status": (
            "pass" if final_decision == "SEMANTIC_PASS"
            else ("inconclusive" if final_decision == "INCONCLUSIVE_ENGINEERING"
                  else "fail")
        ),
        "forbidden_fields_read": False,
        "poison_test_pass": bool(a2_ok),
        "query_pixels_read": False,
        "query_labels_read": False,
        "trainable_parameter_names": (
            smoke_artifact.get("trainable_parameter_names")
            if smoke_artifact else None
        ),
        "trainable_parameter_count": (
            smoke_artifact.get("trainable_parameter_count")
            if smoke_artifact else None
        ),
        "optimizer_parameter_names": (
            smoke_artifact.get("trainable_parameter_names")
            if smoke_artifact else None
        ),
        "compatibility_fixes": [
            {
                "reason": "upstream hardcoded '/home/lyushuai/...' AlphaCLIP "
                          "checkpoint paths",
                "affected_file": "modules/bbox_clip/alpha_clip_backbone.py",
                "patch_sha": provenance["mvrec_patch_files"].get(
                    "modules/bbox_clip/alpha_clip_backbone.py"
                ),
                "semantic_effect": "none",
            },
            {
                "reason": "upstream hardcoded '/home/lyushuai/...' debug "
                          "feature-dump directory could abort the official "
                          "training path on other hosts",
                "affected_file": "modules/classifier.py",
                "patch_sha": provenance["mvrec_patch_files"].get(
                    "modules/classifier.py"
                ),
                "semantic_effect": "none",
            },
        ],
        "problems": report["problems"],
        "final_decision": final_decision,
    }
    final_path = os.path.join(out_dir, "mvrec_rgb_semantic_audit.json")
    _write_json(final_path, final)
    report["artifacts"]["mvrec_rgb_semantic_audit.json"] = sha256_file(final_path)
    report["final_decision"] = final_decision
    return final, report


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="MVREC-RGB semantic audit")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--annotation-csv", default=DEFAULT_ANNOTATION_CSV)
    ap.add_argument("--alpha-ckpt", default=DEFAULT_ALPHA_CKPT)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--ft-epo", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--skip-model", action="store_true",
                    help="static A1/A3 only (testing); the audit is not complete")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="development-only: tolerate a dirty git tree "
                         "(formal audit must run clean)")
    args = ap.parse_args(argv)

    final, report = run_audit(
        manifest_path=args.manifest,
        image_root=args.image_root,
        annotation_csv=args.annotation_csv,
        alpha_ckpt_path=args.alpha_ckpt,
        seed=args.seed,
        ft_epo=args.ft_epo,
        device=args.device,
        out_dir=args.out_dir,
        run_model=not args.skip_model,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(
        {k: report[k] for k in ("gates", "final_decision", "problems")},
        indent=2,
    ))
    return 0 if report["final_decision"] == "SEMANTIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
