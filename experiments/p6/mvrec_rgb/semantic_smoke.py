"""A4 support-only semantic smoke for the MVREC-RGB adaptation.

Runs the official support-side pipeline on one locked P6.B support manifest
(product=bottle, K=1, seed=111) and reports ONLY:

- model construct success and tensor shapes;
- finite-value checks;
- deterministic repeatability (two identical seeded runs compared at
  atol=1e-7 / rtol=1e-6);
- gradient ownership / trainable parameter whitelist;
- optimizer parameter ownership (which parameters the official
  AdamW actually updates);
- expected vs actual step count of the official ft_epo budget;
- completion schema;
- support identity (manifest RGB SHA256 recomputed from disk);
- provenance hooks consumed by A5.

Strictly forbidden here: accuracy, balanced accuracy, macro-F1, query
predictions, confusion matrices, query manifests, query pixels, query labels.
The support-side loss is captured only to verify numeric sanity of the
training loop and is never used for any method decision.
"""

import time

import numpy as np
import torch

from experiments.p6.mvrec_rgb.poison_audit import _compare
from experiments.p6.mvrec_rgb.rgb_dataset import MVRecRgbSupportDataset
from experiments.p6.mvrec_rgb.rgb_model import build_mvrec_rgb_model

# Official trainable groups of EchoClassfierF with zip_config_index=6
# (modules/classifier.py config list): support_key + zifa.
EXPECTED_TRAINABLE = {"head.support_key", "head.zifa.adapter.0.weight",
                      "head.zifa.adapter.0.bias"}


class _TrainingCapture:
    """Captures the official AdamW optimizer / CosineAnnealingLR instances
    and the per-step CE loss values without touching official code paths."""

    def __init__(self):
        self.optimizer = None
        self.scheduler = None
        self.ce_losses = []
        self._orig_adamw = torch.optim.AdamW
        self._orig_cosann = torch.optim.lr_scheduler.CosineAnnealingLR
        self._orig_ce = torch.nn.functional.cross_entropy
        capture = self

        class _AdamW(self._orig_adamw):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                if capture.optimizer is None:
                    capture.optimizer = self

        class _CosAnn(self._orig_cosann):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                if capture.scheduler is None:
                    # PyTorch >= 2.0 LRScheduler calls _initial_step() at
                    # construction, so the absolute last_epoch starts at 0;
                    # record the construction-time value to count actual
                    # training steps as the increment.
                    self._capture_initial_last_epoch = int(self.last_epoch)
                    capture.scheduler = self

        def _ce(*a, **k):
            out = capture._orig_ce(*a, **k)
            capture.ce_losses.append(float(out.detach().cpu()))
            return out

        torch.optim.AdamW = _AdamW
        torch.optim.lr_scheduler.CosineAnnealingLR = _CosAnn
        torch.nn.functional.cross_entropy = _ce

    def restore(self):
        torch.optim.AdamW = self._orig_adamw
        torch.optim.lr_scheduler.CosineAnnealingLR = self._orig_cosann
        torch.nn.functional.cross_entropy = self._orig_ce


def _fix_rng(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _finite(t):
    if isinstance(t, torch.Tensor):
        return bool(torch.isfinite(t.detach().cpu().float()).all())
    return all(_finite(v) for v in t.values())


def _trainable_names(model):
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def check_trainable_whitelist(names):
    """Return True iff the trainable set is exactly the official Zip-Adapter-F
    whitelist (support_key + zifa).  Exposed for regression tests."""
    return set(names) == EXPECTED_TRAINABLE


def _optimizer_ownership(optimizer, ft_epo):
    """Verify that exactly the whitelisted trainable parameters were updated
    for the full budget and that no frozen parameter received any update."""
    entries = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            step = state.get("step")
            entries.append({
                "shape": list(p.shape),
                "requires_grad": bool(p.requires_grad),
                "update_steps": int(step) if step is not None else None,
            })
    ok = True
    reason = None
    trainable_entries = [e for e in entries if e["requires_grad"]]
    frozen_entries = [e for e in entries if not e["requires_grad"]]
    if len(trainable_entries) == 0:
        ok, reason = False, "no trainable parameter present in optimizer"
    elif any(e["update_steps"] != int(ft_epo) for e in trainable_entries):
        ok, reason = False, "trainable param not updated for the full budget"
    elif any(e["update_steps"] is not None for e in frozen_entries):
        ok, reason = False, "frozen parameter updated by optimizer"
    return ok, reason, entries


def _run_once(manifest_path, image_root, alpha_ckpt_path, seed, ft_epo, device):
    """One deterministic pass of the official support-side pipeline."""
    _fix_rng(seed)
    dataset = MVRecRgbSupportDataset(manifest_path, image_root, device=device)
    identity_ok, identity_mismatches = dataset.verify_support_identity()

    capture = _TrainingCapture()
    try:
        model, param = build_mvrec_rgb_model(
            class_names=dataset.class_order,
            alpha_ckpt_path=alpha_ckpt_path,
            k_shot=dataset.k_shot,
            ft_epo=ft_epo,
            seed=seed,
        )
        device_t = param.device

        # --- official support-side pipeline (modules/model.py set_img_prototype)
        t0 = time.perf_counter()
        model.set_img_prototype(dataset.k_shot, dataset)
        train_seconds = time.perf_counter() - t0

        # NOTE: ownership must be inspected BEFORE set_mode("eval"), which
        # freezes every parameter (the official eval path sets
        # requires_grad=False everywhere).
        actual_steps = None
        if capture.scheduler is not None:
            actual_steps = int(capture.scheduler.last_epoch) - int(
                getattr(capture.scheduler, "_capture_initial_last_epoch", 0)
            )
        ownership_ok, ownership_reason, ownership_entries = (
            _optimizer_ownership(capture.optimizer, ft_epo)
            if capture.optimizer is not None
            else (False, "official optimizer instance not captured", [])
        )
        # Trainable whitelist must also be captured before the eval freeze.
        trainable_names = _trainable_names(model)
        # Optimizer-state step counter as an independent second witness of
        # the executed budget.
        optimizer_step_witness = None
        if capture.optimizer is not None:
            for group in capture.optimizer.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        st = capture.optimizer.state.get(p, {}).get("step")
                        if st is not None:
                            optimizer_step_witness = int(st)
                            break
                if optimizer_step_witness is not None:
                    break

        # --- support mvrec snapshot for determinism comparison
        collate = dataset.get_collate_fn()
        batch = collate([dataset[i] for i in range(len(dataset))])
        batch_gpu = {
            "x": batch["x"].to(device_t),
            "y": batch["y"].to(device_t),
            "masks": batch["masks"].to(device_t),
        }
        model.set_mode("eval")
        with torch.no_grad():
            mvrec = model.get_mvrec(batch_gpu, dataset.k_shot)

        losses = capture.ce_losses
        finite_loss = bool(len(losses) > 0 and all(np.isfinite(x) for x in losses))

        record = {
            "support_mvrec": mvrec.detach().cpu(),
            "head_state": {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()
                           if k.startswith("head.")},
            "trainable_names": trainable_names,
            "ce_losses": list(losses),
            "train_seconds": train_seconds,
            "identity_ok": identity_ok,
            "identity_mismatches": identity_mismatches,
            "text_features_shape": tuple(model.text_features.shape),
            "input_shape": dataset.input_shape,
            "num_support": len(dataset),
            "class_order": list(dataset.class_order),
            "actual_steps": actual_steps,
            "optimizer_step_witness": optimizer_step_witness,
            "loss_finite_ok": finite_loss,
            "optimizer_ownership_ok": ownership_ok,
            "optimizer_ownership_reason": ownership_reason,
            "optimizer_ownership_entries": ownership_entries,
            "optimizer_parameter_count": (
                sum(len(g["params"]) for g in capture.optimizer.param_groups)
                if capture.optimizer is not None else None
            ),
        }
        return record, dataset
    finally:
        capture.restore()


def run_support_smoke(
    manifest_path,
    image_root,
    alpha_ckpt_path,
    seed=111,
    ft_epo=500,
    device=None,
):
    """Execute the A4 support-only smoke twice (determinism) and report."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    rec1, ds1 = _run_once(manifest_path, image_root, alpha_ckpt_path,
                          seed, ft_epo, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rec2, ds2 = _run_once(manifest_path, image_root, alpha_ckpt_path,
                          seed, ft_epo, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determinism comparison (protocol tolerance atol=1e-7, rtol=1e-6).
    det_ok, det_diffs = _compare(
        {"support_mvrec": rec1["support_mvrec"],
         "head_state": rec1["head_state"]},
        {"support_mvrec": rec2["support_mvrec"],
         "head_state": rec2["head_state"]},
    )

    finite_ok = (
        _finite(rec1["support_mvrec"])
        and _finite(rec1["head_state"])
        and rec1["loss_finite_ok"]
        and rec2["loss_finite_ok"]
    )

    failures = []
    if not det_ok:
        failures.append("deterministic repeatability failed")
    if not rec1["optimizer_ownership_ok"]:
        failures.append(f"optimizer ownership (run1): {rec1['optimizer_ownership_reason']}")
    if not rec2["optimizer_ownership_ok"]:
        failures.append(f"optimizer ownership (run2): {rec2['optimizer_ownership_reason']}")
    if not check_trainable_whitelist(rec1["trainable_names"]):
        failures.append(
            f"trainable whitelist mismatch: "
            f"{set(rec1['trainable_names']) ^ EXPECTED_TRAINABLE}"
        )
    if rec1["actual_steps"] != int(ft_epo):
        failures.append(
            f"step count mismatch: expected {ft_epo}, actual {rec1['actual_steps']}"
        )
    if rec1["optimizer_step_witness"] != int(ft_epo):
        failures.append(
            f"optimizer step witness mismatch: expected {ft_epo}, "
            f"witness {rec1['optimizer_step_witness']}"
        )
    if not rec1["identity_ok"] or not rec2["identity_ok"]:
        failures.append("support identity mismatch")
    if not finite_ok:
        failures.append("non-finite tensor observed")

    completion = {
        "status": "diagnostic",
        "formal_eligible": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "formal_test_used": False,
        "performance_metrics_generated": False,
        "product": ds1.product,
        "seed": ds1.seed,
        "K": ds1.k_shot,
        "support_identities": [
            {"rel_path": s["rel_path"], "rgb_sha256": s["rgb_sha256"],
             "class_name": s["class_name"]}
            for s in ds1.samples
        ],
        "support_identity_ok": bool(rec1["identity_ok"] and rec2["identity_ok"]),
        "identity_mismatches": rec1["identity_mismatches"] + rec2["identity_mismatches"],
        "input_shape": ds1.input_shape,
        "num_support": len(ds1),
        "class_order": list(ds1.class_order),
        "text_features_shape": list(rec1["text_features_shape"]),
        "support_mvrec_shape": list(rec1["support_mvrec"].shape),
        "trainable_parameter_names": rec1["trainable_names"],
        "trainable_parameter_count": sum(
            1 for _ in rec1["trainable_names"]
        ),
        "optimizer_parameter_count": rec1["optimizer_parameter_count"],
        "optimizer_ownership_ok": bool(
            rec1["optimizer_ownership_ok"] and rec2["optimizer_ownership_ok"]
        ),
        "optimizer_ownership_reason": rec1["optimizer_ownership_reason"],
        "optimizer_ownership_entries": rec1["optimizer_ownership_entries"],
        "expected_steps": int(ft_epo),
        "actual_steps": rec1["actual_steps"],
        "optimizer_step_witness": rec1["optimizer_step_witness"],
        "ce_loss_first": rec1["ce_losses"][0] if rec1["ce_losses"] else None,
        "ce_loss_last": rec1["ce_losses"][-1] if rec1["ce_losses"] else None,
        "loss_finite_ok": bool(rec1["loss_finite_ok"] and rec2["loss_finite_ok"]),
        "deterministic_repeatability_ok": bool(det_ok),
        "determinism_diffs": det_diffs,
        "finite_ok": bool(finite_ok),
        "train_seconds_run1": rec1["train_seconds"],
        "train_seconds_run2": rec2["train_seconds"],
        "device": device,
        "smoke_failures": failures,
        "smoke_pass": len(failures) == 0,
    }
    return completion
