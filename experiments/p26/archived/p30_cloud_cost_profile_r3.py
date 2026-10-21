#!/usr/bin/env python3
"""P26 (paper to-do A15): deployment-cost profile, device-agnostic (laptop / server GPU / Jetson).

No accuracy is measured. Every method is timed in its OWN fresh process, one at a time; the whole list
runs twice, in forward and then in reverse order, and GPU telemetry (temperature, SM clock) is recorded
before and after every block so that throttling is visible. Setting (as A8 / P22): bottle, K = 5,
seed 111 (3 classes, 15 support images), M = 20 normal bank, batch 1 for inference.

Reported per method:
  * per-task preparation / training time (ours: features + calibration; restricted adaptation: its
    official 80-step training through the formal runner; CLIP-LoRA: the official 500 x K = 2,500
    steps, extrapolated from the official loop and, separately, compute only);
  * model forward time (input already on the device) and end-to-end latency from the PIL image;
  * GFLOPs per image (and per CLIP-LoRA training step), peak memory, parameters.
Our method on CLIP ViT-L/14@336 alone (P30: blocks ours_l14_336_fp32 / _fp16) is P17's `a_r` (appearance +
deviation on the layer-18 patches, no global score, no DINOv2), the P29 backbone-matched comparator of
CLIP-LoRA L/14@336; only CLIP is loaded, and FP16 is CUDA autocast of the CLIP forward (as P22).
CLIP-LoRA at the official batch is tried first; if it does not fit, that fact is recorded and the
compute-only step is timed with gradient accumulation (same effective batch; never checkpointing).

Usage:
    python -m experiments.p26.cost_profile --tag laptop4050 --mvtec-root <MVTec AD root> --cliplora <CLIP-LoRA dir>
    (internally: python -m experiments.p26.cost_profile --worker <block> ...)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BLOCKS = ("ours_main", "ours_b336_fp32", "ours_s336_fp32", "ours_s336_fp16", "ours_clipb16_224", "ours_clipb16_336",
          "ours_l14_336_fp32", "ours_l14_336_fp16", "restricted_adaptation",
          "cliplora_b16_224", "cliplora_b16_336", "cliplora_l14_336")
AR_BLOCKS = ("ours_l14_336_fp32", "ours_l14_336_fp16")   # P30: CLIP-L/14@336 alone, head = appearance + deviation
LORA = {"cliplora_b16_224": ("ViT-B/16", 224), "cliplora_b16_336": ("ViT-B/16", 336),
        "cliplora_l14_336": ("ViT-L/14@336px", 336)}
PRODUCT, K_SHOT, SEED = "bottle", 5, 111
OFFICIAL_STEPS = 500 * K_SHOT          # CLIP-LoRA: n_iters (500) x shots
STEPS_SHORT, STEPS_LONG = 25, 125      # two official runs; their difference removes fixed overhead
COMPUTE_WARMUP, COMPUTE_TIMED = 5, 30
# official OpenAI ViT-L/14@336px weights (the SHA256 is part of OpenAI's download URL)
L14_336_WEIGHTS = Path(os.path.expanduser("~")) / ".cache" / "clip" / "ViT-L-14-336px.pt"
L14_336_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- helpers (no torch at import)

def telemetry() -> dict:
    """GPU temperature / clock / power if available (nvidia-smi; tegrastats on Jetson); never fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,"
                              "memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            keys = ("name", "temp_c", "sm_mhz", "sm_max_mhz", "power_w", "mem_used_mib")
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.split()
            return {"source": "nvidia-smi", **dict(zip(keys, (v.strip() for v in out.stdout.splitlines()[0].split(",")))),
                    "other_gpu_processes": len(apps)}
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline().strip()
        p.kill()
        return {"source": "tegrastats", "line": line}
    except (OSError, subprocess.SubprocessError):
        return {"source": None}


def summarize(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    return {"median_ms": statistics.median(s), "p90_ms": s[int(0.9 * (len(s) - 1))], "n": len(s)}


def timed(fn, device, warmup: int = 20, n: int = 100) -> dict:
    import torch
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            sync()
            out.append((time.perf_counter() - t0) * 1000.0)
    return summarize(out)


def gflops(fn) -> float | None:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            fn()
        return counter.get_total_flops() / 1e9
    except Exception:  # noqa: BLE001 - FLOP counting is best-effort on unusual kernels
        return None


def over_physical(device) -> bool:
    """True if the peak allocation exceeded ~physical memory (Windows drivers may silently spill to system
    RAM instead of raising out-of-memory; that is 'does not fit' too)."""
    import torch
    if device.type != "cuda":
        return False
    return torch.cuda.max_memory_allocated() > 0.98 * torch.cuda.get_device_properties(device).total_memory


def peak_mb(device) -> float | None:
    import torch
    return torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None


def task_inputs(mvtec_root: Path):
    """Support images/labels of bottle K=5 s111 and the M=20 normal bank of seed 111 (as A8 / P22)."""
    from experiments.p10 import run_pilot_a as ra
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p14.resource_profile import load
    from experiments.revision import revision_common as rc
    import torch
    manifest = rc.load_json(rc.STAGE1_BUNDLE / "p6b_support_manifests" / PRODUCT / f"k{K_SHOT}_s{SEED}.json")
    order = list(manifest["class_order"])
    items = [(c, it["rel_path"]) for c, cls in enumerate(order) for it in manifest["classes"][cls]["items"]]
    nf = torch.load(FORMAL_FEATURES / f"normals_{PRODUCT}.pt", map_location="cpu", weights_only=False)
    supports = [load(rc.DATASET_ROOT / rel) for _, rel in items]
    bank = [load(mvtec_root / PRODUCT / "train" / "good" / x) for x in nf["draws"][str(SEED)][:ra.M_PRIMARY]]
    return supports, [c for c, _ in items], len(order), bank, order, items


# --------------------------------------------------------------------------- P30: CLIP-L/14@336 alone

AR_HEAD = "appearance_deviation_AR"


class ClipOnlyPipeline:
    """The main configuration's CLIP view alone: CLIP ViT-L/14@336, deterministic 336 view, layer-18 patch tokens
    (L2-normalised, CLS dropped) exactly as experiments.p14.resource_profile.Pipeline, without DINOv2 and without the
    global feature. precision "fp16" = CUDA autocast of the CLIP forward (as experiments.p22.lightweight.Dino)."""

    def __init__(self, clip_model, layer: int, precision: str, device):
        if precision not in ("fp32", "fp16"):
            raise ValueError(precision)
        self.clip_model, self.layer, self.precision, self.device = clip_model, layer, precision, device

    def forward(self, batch):
        import torch
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.precision == "fp16" and self.device.type == "cuda"
               else contextlib.nullcontext())
        with ctx:
            return self.clip_model.visual(batch, [self.layer])

    def features(self, image):
        import torch.nn.functional as F
        from experiments.p6 import p6b_stage2_models as models
        _, tokens = self.forward(models.deterministic_eval_tensor(image).to(self.device))
        return None, F.normalize(tokens[0][:, 1:, :].float(), dim=-1)


def profile_ar(name, pipe, support_images, bank_images, s_lab, n, device, sync):
    """experiments.p18.resource_profile.profile for P17's `a_r` = z(SS[h_A]) + z(SS[h_R]) (no global score): same
    warm-up, offline steps and online timing; the timed head is first checked against the evaluation code."""
    import torch
    from experiments.p10 import run_pilot as rp
    from experiments.p10 import run_pilot_a as ra
    from experiments.p11 import formal_ab as fa
    from experiments.p11 import run_pilot_ab as ab
    from experiments.p13 import pilot as p13
    from experiments.p14.resource_profile import proto_space, score
    from experiments.revision import revision_common as rc
    from experiments.revision.c3_resource_profile import _time
    offline = {}

    def timed_once(fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        return out, (time.perf_counter() - t0) * 1000.0

    query = support_images[0]
    for _ in range(5):
        pipe.features(query)
    feats, offline["support_feature_extraction_ms"] = timed_once(lambda: [pipe.features(im) for im in support_images])
    bank_feats, offline["bank_feature_extraction_ms"] = timed_once(lambda: [pipe.features(im)[1] for im in bank_images])
    p_s = torch.cat([f[1] for f in feats])
    bank_imgs = torch.cat(bank_feats)
    bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
    tau, offline["tau_calibration_loo_ms"] = timed_once(lambda: fa.tau99(bank_imgs))

    def prepare():
        w = ab.region_weights(ra.patch_anomaly(p_s, bank), tau, soft=True)
        return {"a": proto_space(ab.weighted_pool(p_s, w), s_lab, n),
                "rr": proto_space(p13.region_residual(p_s, bank, w), s_lab, n)}
    state, offline["support_state_ms"] = timed_once(prepare)

    def head(c_q):
        best, idx = torch.einsum("npd,bd->npb", c_q, bank).max(dim=2)
        w = ab.region_weights(1.0 - best, tau, soft=True)
        sa = score(ab.weighted_pool(c_q, w), state["a"])
        srr = score(ab.weighted_pool(c_q - bank[idx], w), state["rr"])
        return rp._zscore(sa) + rp._zscore(srr)

    _, c_q = pipe.features(query)
    both_c = torch.cat([p_s, c_q])
    si = torch.arange(len(s_lab), device=device)
    qi = torch.tensor([len(s_lab)], device=device)
    w_all = ab.region_weights(ra.patch_anomaly(both_c, bank), tau, soft=True)
    soft_all, rr_all = ab.weighted_pool(both_c, w_all), p13.region_residual(both_c, bank, w_all)
    reference = (rp._zscore(rp.simpleshot(soft_all[si], s_lab, soft_all[qi], n))
                 + rp._zscore(rp.simpleshot(rr_all[si], s_lab, rr_all[qi], n)))
    check = {AR_HEAD: float((head(c_q) - reference).abs().max())}
    if check[AR_HEAD] > 1e-4:
        raise rc.P6BFailClosedError(f"{name}: timed head differs from the evaluation code: {check}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    online = {AR_HEAD: {"end_to_end_from_pil": _time(lambda: head(pipe.features(query)[1]), device),
                        "head_only_given_features": _time(lambda: head(c_q), device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}}
    storage = {"patch_grid": int(p_s.shape[1]), "patch_dim": int(p_s.shape[2]), "bank_patches": int(bank.shape[0]),
               "bank_fp32_mb": bank.numel() * 4 / 2 ** 20}
    return {"head_check_max_abs_diff": check, "offline_ms": offline, "online": online, "storage": storage,
            "precision": pipe.precision}


# --------------------------------------------------------------------------- workers

def worker_ours(block: str, args, device) -> dict:
    import torch
    from types import SimpleNamespace
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p14.resource_profile import Pipeline
    from experiments.p18 import backbones as bb
    from experiments.p18.resource_profile import count_params, profile
    from experiments.p22 import lightweight as lw
    supports, s_lab, n, bank, _, _ = task_inputs(Path(args.mvtec_root))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        if block == "ours_main":
            factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
            pipe = Pipeline(factory, device)
            params = count_params(factory.clip_model.visual, factory.dino_extractor.model)
            from experiments.p10.extract_dev_features import CLIP_LAYER
            prep = lambda im: models.deterministic_eval_tensor(im).to(device)  # noqa: E731
            forward = lambda x: (factory.clip_model.visual(x, [CLIP_LAYER]),  # noqa: E731
                                 factory.dino_extractor.forward_patches(x))
        elif block in AR_BLOCKS:
            # our method on CLIP ViT-L/14@336 alone = P17 `a_r` (the P29 comparator); CLIP only, no DINOv2 loaded
            from experiments.p10.extract_dev_features import CLIP_LAYER
            clip_model, _ = models.build_clip_backbone(PROJECT_ROOT, device)
            pipe = ClipOnlyPipeline(clip_model, CLIP_LAYER, block.rsplit("_", 1)[1], device)
            params = count_params(clip_model.visual)
            prep = lambda im: models.deterministic_eval_tensor(im).to(device)  # noqa: E731
            forward = pipe.forward
        elif block.startswith("ours_clipb16_"):
            # our method on CLIP ViT-B/16, matched to CLIP-LoRA; layers are the existing dev selections
            # (224: the P18 dev rule restricted to 224 px, as P24; 336: the P18 selection)
            from experiments.p18.resource_profile import AltPipeline
            from experiments.p18.run import dev_selection, parse_config
            from experiments.p24.clipb224 import layer_224
            res = int(block.rsplit("_", 1)[1])
            layer = layer_224() if res == 224 else parse_config(dev_selection()["clip_b16"])[2]
            backbone = bb.ClipB(device, res)
            pipe = AltPipeline(backbone, layer, device)
            params = count_params(backbone.model.visual)
            prep = lambda im: bb.eval_tensor(im, backbone.res).to(device)  # noqa: E731
            forward = lambda x: backbone.features(x, (layer,))  # noqa: E731
        else:
            cfg = {"ours_s336_fp32": "dinov2_s_r336_fp32", "ours_s336_fp16": "dinov2_s_r336_fp16",
                   "ours_b336_fp32": "dinov2_b_r336_fp32"}[block]   # DINOv2-B: layer 11, = P18
            layer = lw.rc.load_json(lw.OUT["dev"] / "p22_dev_results.json")["selection"][cfg]
            backbone = lw.build(cfg, device)
            from experiments.p22.lightweight import LightPipeline
            pipe = LightPipeline(backbone, layer, device)
            params = count_params(backbone.model)
            prep = lambda im: bb.eval_tensor(im, backbone.res).to(device)  # noqa: E731
            forward = lambda x: backbone.features(x, (layer,))  # noqa: E731
        res = (profile_ar if block in AR_BLOCKS else profile)(block, pipe, supports, bank, s_lab, n, device, sync)
        x = prep(supports[0])
        res["model_forward"] = timed(lambda: forward(x), device)
        res["gflops_per_image_backbone"] = gflops(lambda: forward(x))
    res["params"] = params
    res["per_task_preparation_s"] = sum(res["offline_ms"].values()) / 1000.0
    return res


def worker_restricted(args, device, out_dir: Path) -> dict:
    """The formal runner of the restricted support-side adaptation (terminal_only, 80 steps), unchanged."""
    import datetime as dt
    import re
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.revision import revision_common as rc
    from experiments.revision.c3_resource_profile import cmd_latency
    unit_dir = out_dir / f"restricted_unit_{args.pass_index}"
    if unit_dir.exists():
        raise SystemExit(f"{unit_dir} exists")
    provenance_note = None
    if not (PROJECT_ROOT / ".git").exists():
        # Offline bundle (git archive, no .git; no git init on the target machine): the runner's provenance preflight
        # gets the bundle's recorded commit, flagged dirty, so the run is labelled diagnostic. Training is unchanged.
        head = json.loads((PROJECT_ROOT.parent / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))["git_head"]
        support.builder.git_commit_and_dirty = lambda _root: (head, True)
        provenance_note = f"no .git in the bundle; runner provenance = bundle git_head {head[:8]} (diagnostic)"
    t0 = time.perf_counter()
    code = support.main(["--bundle-dir", str(rc.STAGE1_BUNDLE), "--image-root", str(rc.DATASET_ROOT),
                         "--product", PRODUCT, "--k", str(K_SHOT), "--seed", str(SEED), "--method", "ours",
                         "--output-dir", str(unit_dir), "--device", str(device), "--allow-dirty-diagnostic"])
    wall = time.perf_counter() - t0
    if code != 0:
        raise SystemExit(f"restricted adaptation runner failed with code {code}")
    stamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    lines = (unit_dir / "ours_train_multiclass.log").read_text(encoding="utf-8", errors="replace").splitlines()
    ts = [dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") for m in map(stamp.match, lines) if m]
    end = [dt.datetime.strptime(stamp.match(l).group(1), "%Y-%m-%d %H:%M:%S,%f") for l in lines
           if "Checkpoint saved" in l and stamp.match(l)]
    audit = json.loads((unit_dir / "ours_step_audit.json").read_text(encoding="utf-8"))
    latency = cmd_latency(SimpleNamespace(device=str(device), pipeline="ours"))
    return {"training_s": (end[-1] - ts[0]).total_seconds() if end else None, "runner_wall_s": wall,
            "optimizer_steps": audit["actual_optimizer_steps"], "trainable_whitelist": audit["trainable_parameter_whitelist"],
            "model_forward": latency["boundary_model_forward"], "end_to_end": latency["boundary_end_to_end_from_pil"],
            "peak_allocated_bytes": latency["peak_allocated_bytes"], "provenance_note": provenance_note}


def worker_lora(block: str, args, device) -> dict:
    """Official CLIP-LoRA loop (timed at two lengths) + compute-only step (with accumulation if needed)."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(Path(args.cliplora).resolve()))   # before anything imports a module named utils
    os.chdir(args.cliplora)
    import clip
    import torchvision.transforms as T
    from datasets.utils import Datum, build_data_loader
    from loralib.utils import apply_lora, get_lora_parameters, mark_only_lora_as_trainable
    from lora import run_lora
    from run_utils import set_random_seed
    from experiments.p21 import cliplora_runner as runner
    backbone, res = LORA[block]
    doc = runner.load_episodes(PROJECT_ROOT / "docs" / "p21" / "P21_MVTEC_FS_EPISODES.json")
    ep = next(e for e in doc["episodes"] if (e["product"], e["K"], e["seed"]) == (PRODUCT, K_SHOT, SEED))
    root = PROJECT_ROOT / "data" / "mvtec-fs"
    order = ep["class_order"]
    dataset = type("D", (), {"classnames": list(order), "template": list(runner.TEMPLATE)})()
    train_x = [Datum(impath=str(root / s["rel_path"]), label=s["label"], classname=order[s["label"]]) for s in ep["supports"]]
    workers = min(8, os.cpu_count() or 1)   # official 8

    def fresh(n_iters):
        a = runner.official_args(K_SHOT, SEED)
        a.backbone, a.n_iters = backbone, n_iters
        set_random_seed(a.seed)
        name = backbone
        if backbone not in clip.available_models():   # the CLIP copy vendored by CLIP-LoRA lacks L/14@336;
            name = str(L14_336_WEIGHTS)               # clip.load accepts the official OpenAI file directly
            if sha256_file(L14_336_WEIGHTS) != L14_336_SHA256:
                raise SystemExit(f"{L14_336_WEIGHTS} is not the official OpenAI ViT-L/14@336px file")
        model, preprocess = clip.load(name, device=device)
        if res != model.visual.input_resolution:
            runner.resize_positional_embedding(model, res)
            preprocess = clip.clip._transform(res)
        if backbone == "ViT-L/14@336px":   # loralib has no @336 entry; same 24-layer ViT-L/14 layout
            a.backbone = "ViT-L/14"
        return a, model.eval(), preprocess

    def train_tf():
        return T.Compose([T.RandomResizedCrop(size=res, scale=(0.08, 1), interpolation=T.InterpolationMode.BICUBIC),
                          T.RandomHorizontalFlip(p=0.5), T.ToTensor(), T.Normalize(runner.MEAN, runner.STD)])

    out = {"backbone": backbone, "resolution": res, "official_steps": OFFICIAL_STEPS, "loader_workers": workers,
           "loralib_layer_table": "ViT-L/14" if backbone == "ViT-L/14@336px" else backbone}
    # (1) official loop at two lengths
    official = {}
    try:
        for steps in (STEPS_SHORT, STEPS_LONG):
            a, model, preprocess = fresh(steps // K_SHOT)
            test = [train_x[0]]
            vl = build_data_loader(data_source=test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False, num_workers=workers)
            tl = build_data_loader(data_source=train_x, batch_size=a.batch_size, tfm=train_tf(), is_train=True, shuffle=True, num_workers=workers)
            torch.cuda.synchronize() if device.type == "cuda" else None
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            t0 = time.perf_counter()
            run_lora(a, model, runner.LOGIT_SCALE, dataset, tl, vl, vl)
            torch.cuda.synchronize() if device.type == "cuda" else None
            official[steps] = time.perf_counter() - t0
            if over_physical(device):
                raise torch.cuda.OutOfMemoryError("peak allocation exceeded physical GPU memory")
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        per_step = (official[STEPS_LONG] - official[STEPS_SHORT]) / (STEPS_LONG - STEPS_SHORT)
        out["official"] = {"fits_official_batch": True, "run_s": official, "s_per_step": per_step,
                           "task_training_s_extrapolated": per_step * OFFICIAL_STEPS}
    except torch.cuda.OutOfMemoryError as exc:
        out["official"] = {"fits_official_batch": False, "error": str(exc).splitlines()[0]}
        torch.cuda.empty_cache()
    # (2) compute only: one fixed augmented batch on the device, the official step, micro-batched if needed
    a, model, preprocess = fresh(1)
    apply_lora(a, model)
    model = model.to(device)
    mark_only_lora_as_trainable(model)
    opt = torch.optim.AdamW(get_lora_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    tf = train_tf()
    images = torch.stack([tf(Image_open(root / s["rel_path"])) for s in ep["supports"]]).to(device)
    target = torch.tensor([s["label"] for s in ep["supports"]], device=device)
    texts = clip.tokenize([dataset.template[0].format(c.replace("_", " ")) for c in order]).to(device)

    def step(micro):
        model.train()
        opt.zero_grad()
        for i in range(0, len(images), micro):
            x, y = images[i:i + micro], target[i:i + micro]
            # exactly the official run_lora step: normalisation happens outside autocast
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                emb = model.encode_text(texts)
            tfeat = emb / emb.norm(dim=-1, keepdim=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                ifeat = model.encode_image(x)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            loss = F.cross_entropy(runner.LOGIT_SCALE * ifeat @ tfeat.t(), y) * (len(x) / len(images))
            scaler.scale(loss).backward() if scaler else loss.backward()
        (scaler.step(opt), scaler.update()) if scaler else opt.step()

    micro_used = None
    for micro in (len(images), 8, 4, 2, 1):
        try:
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            for _ in range(COMPUTE_WARMUP):
                step(micro)
            torch.cuda.synchronize() if device.type == "cuda" else None
            if over_physical(device):
                raise torch.cuda.OutOfMemoryError("peak allocation exceeded physical GPU memory")
            samples = []
            for _ in range(COMPUTE_TIMED):
                t0 = time.perf_counter()
                step(micro)
                torch.cuda.synchronize() if device.type == "cuda" else None
                samples.append((time.perf_counter() - t0) * 1000.0)
            micro_used = micro
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if micro_used is None:
        out["compute_only"] = {"fits": False}
    else:
        st = summarize(samples)
        out["compute_only"] = {"batch": len(images), "micro_batch": micro_used,
                               "gradient_accumulation": micro_used < len(images), "step": st,
                               # with accumulation the (small) text encoder runs once per micro-batch, so the
                               # figure is, if anything, slightly pessimistic for CLIP-LoRA
                               "text_encoder_calls_per_step": -(-len(images) // micro_used),
                               "task_training_s_extrapolated": st["median_ms"] / 1000.0 * OFFICIAL_STEPS,
                               "gflops_per_step": gflops(lambda: step(micro_used)),
                               "peak_train_mb": peak_mb(device)}
    out["trainable_params"] = sum(p.numel() for p in get_lora_parameters(model))
    # (3) inference: text features cached, one image
    model.eval()
    with torch.no_grad():   # as the official evaluate_lora: encode under autocast, normalise outside
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            emb = model.encode_text(texts)
        tfeat = emb / emb.norm(dim=-1, keepdim=True)
    img = Image_open(root / ep["supports"][0]["rel_path"])
    xq = preprocess(img).unsqueeze(0).to(device)

    def infer(x):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            f = model.encode_image(x)
        return (f / f.norm(dim=-1, keepdim=True)) @ tfeat.t()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    out["model_forward"] = timed(lambda: infer(xq), device)
    out["end_to_end"] = timed(lambda: infer(preprocess(img).unsqueeze(0).to(device)), device)
    out["peak_infer_mb"] = peak_mb(device)
    out["gflops_per_image"] = gflops(lambda: infer(xq))
    return out


def Image_open(path):
    from PIL import Image
    with Image.open(path) as h:
        return h.convert("RGB").copy()


def trace_file_opens(record: Path) -> None:
    """Record every file this process opens (for building an exact offline bundle); no effect on timing
    beyond one set lookup per open."""
    seen, busy = set(), [False]

    def hook(event, hook_args):
        if event != "open" or busy[0] or not hook_args or not isinstance(hook_args[0], (str, bytes, os.PathLike)):
            return
        p = os.path.abspath(os.fsdecode(hook_args[0]))
        if p in seen:
            return
        seen.add(p)
        busy[0] = True            # our own write below must not re-enter the hook
        try:
            with open(str(record), "a", encoding="utf-8") as h:
                h.write(p + "\n")
        finally:
            busy[0] = False
    sys.addaudithook(hook)


def run_worker(args) -> int:
    if os.environ.get("P26_TRACE_OPENS"):
        trace_file_opens(Path(os.environ["P26_TRACE_OPENS"]))
    import torch
    device = torch.device(args.device)
    out_dir = Path(args.out)
    t0 = time.perf_counter()
    if args.worker.startswith("ours_"):
        res = worker_ours(args.worker, args, device)
    elif args.worker == "restricted_adaptation":
        res = worker_restricted(args, device, out_dir)
    else:
        res = worker_lora(args.worker, args, device)
    res["block_wall_s"] = time.perf_counter() - t0
    res["torch"] = torch.__version__
    (out_dir / f"pass{args.pass_index}_{args.worker}.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                                                        encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- orchestrator

def headline(block: str, r: dict) -> dict:
    """The comparable numbers of one block."""
    if block.startswith("ours_"):
        head = r["online"][AR_HEAD if block in AR_BLOCKS else "dual_branch_ARR"]
        return {"prep_or_train_s": r["per_task_preparation_s"], "model_forward_ms": r["model_forward"]["median_ms"],
                "end_to_end_ms": head["end_to_end_from_pil"]["median_ms"],
                "gflops_per_image": r["gflops_per_image_backbone"], "params": r["params"], "trainable": 0,
                "peak_mb": head["peak_allocated_bytes"] / 2 ** 20 if head["peak_allocated_bytes"] else None}
    if block == "restricted_adaptation":
        return {"prep_or_train_s": r["training_s"], "model_forward_ms": r["model_forward"]["median_ms"],
                "end_to_end_ms": r["end_to_end"]["median_ms"], "optimizer_steps": r["optimizer_steps"]}
    off = r.get("official", {})
    co = r.get("compute_only", {})
    return {"prep_or_train_s": off.get("task_training_s_extrapolated"),
            "train_compute_only_s": co.get("task_training_s_extrapolated"),
            "fits_official_batch": off.get("fits_official_batch"), "gradient_accumulation": co.get("gradient_accumulation"),
            "model_forward_ms": r["model_forward"]["median_ms"], "end_to_end_ms": r["end_to_end"]["median_ms"],
            "gflops_per_image": r.get("gflops_per_image"), "gflops_per_train_step": co.get("gflops_per_step"),
            "trainable": r.get("trainable_params"), "peak_train_mb": co.get("peak_train_mb")}


def run_orchestrator(args) -> int:
    out = PROJECT_ROOT / "ckpt" / "p26" / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --tag")
    out.mkdir(parents=True)
    blocks = [b for b in BLOCKS if not args.only or b in args.only.split(",")]
    env = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
           "telemetry_start": telemetry()}
    log = []
    passes = ((1, blocks), (2, list(reversed(blocks))))[:args.passes]
    for pass_index, order in passes:
        for block in order:
            before = telemetry()
            cmd = [sys.executable, "-u", "-m", "experiments.p26.cost_profile", "--worker", block, "--pass-index",
                   str(pass_index), "--out", str(out), "--device", args.device, "--mvtec-root", args.mvtec_root,
                   "--cliplora", args.cliplora]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=args.block_timeout)
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(cmd, "timeout", "", f"timed out after {exc.timeout} s")
            entry = {"pass": pass_index, "block": block, "returncode": proc.returncode,
                     "wall_s": time.perf_counter() - t0, "telemetry_before": before, "telemetry_after": telemetry()}
            if proc.returncode != 0:
                entry["stderr_tail"] = proc.stderr[-3000:]
            log.append(entry)
            print(f"[p26] pass {pass_index} {block}: rc={proc.returncode} {entry['wall_s']:.0f}s", flush=True)
            (out / "run_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    summary = {}
    for block in blocks:
        per_pass = {}
        for p in (1, 2):
            f = out / f"pass{p}_{block}.json"
            if f.exists():
                per_pass[p] = headline(block, json.loads(f.read_text(encoding="utf-8")))
        summary[block] = per_pass
    result = {"schema_version": "p26_cost_v1", "tag": args.tag, "environment": env, "setting":
              {"product": PRODUCT, "K": K_SHOT, "seed": SEED, "bank": 20, "inference_batch": 1},
              "summary": summary, "run_log": log}
    (out / "cost_results.json").write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    lines = [f"# P26 deployment cost — {args.tag}", "", "Two passes (forward / reverse order); values are pass 1 / pass 2.", "",
             "| Block | prep or training (s) | train compute-only (s) | model forward (ms) | end-to-end (ms) | GFLOPs/image | notes |",
             "|---|---|---|---|---|---|---|"]

    def fmt(v, spec=".1f"):
        return "—" if v is None else format(v, spec)
    for block, pp in summary.items():
        h1, h2 = pp.get(1, {}), pp.get(2, {})
        notes = []
        if h1.get("fits_official_batch") is False:
            notes.append("official batch does not fit")
        if h1.get("gradient_accumulation"):
            notes.append("compute-only with gradient accumulation")
        lines.append(f"| {block} | {fmt(h1.get('prep_or_train_s'))} / {fmt(h2.get('prep_or_train_s'))} | "
                     f"{fmt(h1.get('train_compute_only_s'))} / {fmt(h2.get('train_compute_only_s'))} | "
                     f"{fmt(h1.get('model_forward_ms'), '.2f')} / {fmt(h2.get('model_forward_ms'), '.2f')} | "
                     f"{fmt(h1.get('end_to_end_ms'))} / {fmt(h2.get('end_to_end_ms'))} | "
                     f"{fmt(h1.get('gflops_per_image'))} | {'; '.join(notes)} |")
    (out / "cost_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", help="device tag for the output folder, e.g. laptop4050, rtx4090, jetson_orin")
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--cliplora", required=True, help="the official CLIP-LoRA checkout (commit ce3701ba)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="", help="comma-separated subset of blocks (for smoke tests)")
    ap.add_argument("--block-timeout", type=int, default=3600, help="seconds per method; exceeded = recorded")
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2), help="2 = forward + reverse (for measurements)")
    ap.add_argument("--worker", choices=BLOCKS)
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.worker:
        return run_worker(args)
    if not args.tag:
        raise SystemExit("--tag is required")
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
