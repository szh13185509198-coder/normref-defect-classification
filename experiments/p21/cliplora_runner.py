#!/usr/bin/env python3
"""P21 cloud runner: official CLIP-LoRA (Zanella & Ben Ayed, CVPRW 2024) on the frozen MVTec-FS
formal episodes. Protocols: docs/P21_CLIPLORA_MVTEC_FS_PROTOCOL.md (ViT-B/16, 224),
docs/P25_CLIPLORA_336_PROTOCOL.md (ViT-B/16, 336) and docs/P29_CLIPLORA_L14_336_PROTOCOL.md (ViT-L/14@336px).

Standalone on purpose: it imports nothing from this repository, only the unmodified official
checkout (--cliplora). Training is the official ``lora.run_lora`` with the official default
arguments (``run_utils.get_arguments``), loaders, and the train transform and logit scale set as in the
official ``main.py``; the only inputs we supply are the episode images and class names.

The cloud never holds query labels: queries enter the official loaders with placeholder label 0,
so every accuracy printed by the official code is meaningless and must not be reported.
Predictions are written per episode and scored locally (experiments/p21/score.py).

Usage (on the cloud, from any directory):
    python cliplora_runner.py --verify-only ...          # hash every image, no GPU work
    python cliplora_runner.py --smoke ...                # time the heaviest episode (discarded)
    python cliplora_runner.py --shard 0/3 ...            # formal run, one of 3 parallel shards
    python cliplora_runner.py --status ...               # count finished episodes
Common arguments: --episodes P21_MVTEC_FS_EPISODES.json --data-root <mvtec-fs root>
                  --cliplora <CLIP-LoRA checkout> --out <output dir>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "p21_cliplora_episode_v1"
EPISODES_SHA256 = "11de94bf7979fbb555ec52f1c26f1a09f1a6ee2f9e65b7ee97edd575d1811339"
BACKBONE = "ViT-B/16"            # official default
L14_336 = "ViT-L/14@336px"       # P29: the backbone of our main method, native 336 px
BACKBONES = (BACKBONE, L14_336)
# The CLIP copy vendored by CLIP-LoRA has no ViT-L/14@336px entry; clip.load accepts the official OpenAI file,
# whose SHA256 is part of OpenAI's download URL.
L14_336_FILE = Path(os.path.expanduser("~")) / ".cache" / "clip" / "ViT-L-14-336px.pt"
L14_336_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
_VERIFIED = set()
TEMPLATE = ["a photo of a {}."]  # same template as Tip-Adapter-F* in this study
LOGIT_SCALE = 100                # official main.py
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episodes(path: Path) -> dict:
    if sha256_file(path) != EPISODES_SHA256:
        raise SystemExit(f"episode file SHA mismatch: {path}")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("query_labels_included") is not False:
        raise SystemExit("episode file must not contain query labels")
    return doc


def episode_key(e: dict) -> str:
    return f"{e['product']}_k{e['K']}_s{e['seed']}"


def parse_shard(text: str) -> tuple[int, int]:
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad shard {text}")
    return i, n


def shard_of(episodes: list[dict], i: int, n: int) -> list[dict]:
    return [e for j, e in enumerate(episodes) if j % n == i]


def heaviest(episodes: list[dict]) -> dict:
    return max(episodes, key=lambda e: len(e["supports"]) * e["K"])  # first of the ties


def verify_images(doc: dict, root: Path) -> int:
    items = {s["rel_path"]: s["sha256"] for e in doc["episodes"] for s in e["supports"]}
    items.update({q["rel_path"]: q["sha256"] for qs in doc["queries"].values() for q in qs})
    bad = [p for p, h in items.items() if not (root / p).is_file() or sha256_file(root / p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} image(s) missing or with a different SHA256, first: {bad[:5]}")
    return len(items)


def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def official_args(shots: int, seed: int, backbone: str = BACKBONE):
    """The official argument parser with its defaults; only shots, seed and backbone are set."""
    from run_utils import get_arguments
    saved = sys.argv
    sys.argv = ["main.py", "--shots", str(shots), "--seed", str(seed), "--backbone", backbone]
    try:
        return get_arguments()
    finally:
        sys.argv = saved


def lora_cosines(clip_model, loader, dataset):
    """Per-query cosine scores, computed exactly as the official ``evaluate_lora``."""
    import clip
    import torch
    clip_model.eval()
    with torch.no_grad():
        texts = [dataset.template[0].format(c.replace("_", " ")) for c in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            emb = clip_model.encode_text(clip.tokenize(texts).cuda())
        text_features = emb / emb.norm(dim=-1, keepdim=True)
        out = []
        for images, _ in loader:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                f = clip_model.encode_image(images.cuda())
            f = f / f.norm(dim=-1, keepdim=True)
            out.append((f @ text_features.t()).float().cpu())
    return torch.cat(out)


def resize_positional_embedding(clip_model, resolution: int) -> None:
    """P25: run the ViT at ``resolution`` px by bicubic interpolation of the patch-grid positional
    embeddings (the class-token embedding is kept). The only change to the official model."""
    import torch
    import torch.nn.functional as F
    visual = clip_model.visual
    patch = visual.conv1.kernel_size[0]
    pe = visual.positional_embedding.data
    old, new = int(round((pe.shape[0] - 1) ** 0.5)), resolution // patch
    if new != old:
        grid = pe[1:].float().reshape(1, old, old, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(new, new), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(new * new, -1).to(pe.dtype)
        visual.positional_embedding = torch.nn.Parameter(torch.cat([pe[:1], grid]))
    visual.input_resolution = resolution


def load_clip(clip, backbone: str):
    """-> (model, preprocess, loralib layer-table key). ViT-B/16 is the official path, unchanged."""
    if backbone != L14_336:
        model, preprocess = clip.load(backbone)
        return model, preprocess, backbone
    if L14_336 not in _VERIFIED:
        if not L14_336_FILE.is_file() or sha256_file(L14_336_FILE) != L14_336_SHA256:
            raise SystemExit(f"{L14_336_FILE} is missing or is not the official OpenAI ViT-L/14@336px file")
        _VERIFIED.add(L14_336)
    model, preprocess = clip.load(str(L14_336_FILE))
    return model, preprocess, "ViT-L/14"   # loralib has no @336px entry; the same 24-layer ViT-L/14 layout


def run_episode(e: dict, queries: list[dict], root: Path, workers: int, resolution: int = 224,
                backbone: str = BACKBONE, smoke_iters: int | None = None) -> dict:
    import clip
    import torch
    import torchvision.transforms as transforms
    from datasets.utils import Datum, build_data_loader
    from lora import evaluate_lora, run_lora
    from run_utils import set_random_seed
    from utils import clip_classifier, pre_load_features

    args = official_args(e["K"], e["seed"], backbone)
    set_random_seed(args.seed)
    clip_model, preprocess, layer_table = load_clip(clip, backbone)
    if resolution != clip_model.visual.input_resolution:  # P25 (B/16 at 336); B/16@224 and L/14@336 are native
        resize_positional_embedding(clip_model, resolution)
        preprocess = clip.clip._transform(resolution)   # the official CLIP test transform at n_px
    args.backbone = layer_table
    if smoke_iters:   # smoke only (predictions discarded): shorter training, same model, batch and memory peak
        args.n_iters = max(1, smoke_iters // args.shots)
    clip_model.eval()
    order = e["class_order"]
    dataset = SimpleNamespace(classnames=list(order), template=list(TEMPLATE))
    train_x = [Datum(impath=str(root / s["rel_path"]), label=s["label"], classname=order[s["label"]])
               for s in e["supports"]]
    test = [Datum(impath=str(root / q["rel_path"]), label=0, classname="") for q in queries]  # placeholder
    train_tranform = transforms.Compose([  # official main.py
        transforms.RandomResizedCrop(size=resolution, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_loader = build_data_loader(data_source=train_x, batch_size=256, is_train=False, tfm=preprocess,
                                   shuffle=False, num_workers=workers)
    test_loader = build_data_loader(data_source=test, batch_size=256, is_train=False, tfm=preprocess,
                                    shuffle=False, num_workers=workers)
    train_loader = build_data_loader(data_source=train_x, batch_size=args.batch_size, tfm=train_tranform,
                                     is_train=True, shuffle=True, num_workers=workers)

    # zero-shot CLIP with the same template (the official run_lora prints this before training)
    text_w = clip_classifier(dataset.classnames, dataset.template, clip_model)
    q_feat, _ = pre_load_features(clip_model, test_loader)
    zeroshot = (LOGIT_SCALE * q_feat.cuda() @ text_w).float().cpu()

    torch.cuda.synchronize()
    t0 = time.time()
    run_lora(args, clip_model, LOGIT_SCALE, dataset, train_loader, val_loader, test_loader)
    torch.cuda.synchronize()
    seconds = time.time() - t0

    cos = lora_cosines(clip_model, test_loader, dataset)
    pred = cos.argmax(dim=1).tolist()
    # fidelity check: with placeholder labels 0 the official evaluate_lora returns the share of
    # queries predicted as class 0; it must equal ours (same model, same computation)
    official_share0 = float(evaluate_lora(args, clip_model, test_loader, dataset))
    our_share0 = 100.0 * sum(p == 0 for p in pred) / len(pred)
    return {
        "schema_version": SCHEMA, "key": episode_key(e), "product": e["product"], "K": e["K"],
        "seed": e["seed"], "class_order": order, "template": TEMPLATE[0], "resolution": resolution,
        "backbone": backbone, "loralib_layer_table": layer_table,
        "official_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else list(v))
                          for k, v in vars(args).items()},
        "total_iters": args.n_iters * args.shots,
        "train_seconds": seconds,
        "fidelity_share_class0": {"official_evaluate_lora": official_share0, "ours": our_share0,
                                  "match": abs(official_share0 - our_share0) < 1e-6},
        "queries": [{"rel_path": q["rel_path"], "sha256": q["sha256"],
                     "cliplora_cosine": [float(v) for v in c], "cliplora_pred": order[p],
                     "zeroshot_logit": [float(v) for v in z], "zeroshot_pred": order[int(z.argmax())]}
                    for q, c, p, z in zip(queries, cos, pred, zeroshot)],
    }


def checkout_dirty(path: Path) -> bool:
    """Tracked files only: Python's own __pycache__ folders are untracked and do not alter the code."""
    out = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def environment(cliplora: Path) -> dict:
    import clip
    import torch
    commit = subprocess.run(["git", "-C", str(cliplora), "log", "-1", "--format=%H %cd"],
                            capture_output=True, text=True).stdout.strip()
    return {"host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cliplora_commit": commit, "cliplora_dirty": checkout_dirty(cliplora), "clip_module": clip.__file__,
            "runner_sha256": sha256_file(__file__), "episodes_sha256": EPISODES_SHA256}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cliplora", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=8)  # official num_workers
    ap.add_argument("--resolution", type=int, default=224, choices=(224, 336),
                    help="224 = official; 336 = P25 (positional-embedding interpolation only)")
    ap.add_argument("--backbone", default=BACKBONE, choices=BACKBONES,
                    help="ViT-B/16 = official default; ViT-L/14@336px = P29 (native 336 px only)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue a shard whose lock exists (needs user authorization)")
    ap.add_argument("--smoke-iters", type=int, default=None,
                    help="--smoke only: train this many iterations instead of 500 x K (predictions are discarded)")
    a = ap.parse_args(argv)
    for name in ("episodes", "data_root", "cliplora", "out"):  # the run later changes directory
        setattr(a, name, getattr(a, name).resolve())

    if a.backbone == L14_336 and a.resolution != 336:
        raise SystemExit("ViT-L/14@336px runs at its native 336 px only (--resolution 336)")
    doc = load_episodes(a.episodes)
    episodes = doc["episodes"]
    ep_dir = a.out / "episodes"
    if a.status:
        done = {p.stem for p in ep_dir.glob("*.json")} if ep_dir.exists() else set()
        missing = [episode_key(e) for e in episodes if episode_key(e) not in done]
        print(f"finished {len(episodes) - len(missing)}/{len(episodes)}; missing: {missing[:20]}"
              + (" ..." if len(missing) > 20 else ""))
        return 0
    print(f"verified {verify_images(doc, a.data_root)} images (SHA256 match)")
    marker = a.out / "resolution.txt"   # one output directory = one resolution and one backbone, never mixed
    if marker.exists() and marker.read_text().strip() != str(a.resolution):
        raise SystemExit(f"{a.out} holds a {marker.read_text().strip()} px run; use another --out")
    bmarker = a.out / "backbone.txt"    # absent in the P21 / P25 directories, which are ViT-B/16
    held = bmarker.read_text().strip() if bmarker.exists() else (BACKBONE if marker.exists() else None)
    if held is not None and held != a.backbone:
        raise SystemExit(f"{a.out} holds a {held} run; use another --out")
    if not (a.verify_only or a.status):
        a.out.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(a.resolution))
        bmarker.write_text(a.backbone)
    if a.verify_only:
        return 0

    sys.path.insert(0, str(a.cliplora))
    os.chdir(a.cliplora)  # the official code uses repository-relative imports
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    env = environment(a.cliplora)
    if env["cliplora_dirty"]:
        raise SystemExit("the CLIP-LoRA checkout has local modifications")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if a.smoke:
        e = heaviest(episodes)
        smoke_dir = a.out / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        rec = run_episode(e, doc["queries"][e["product"]], a.data_root, a.workers, a.resolution, a.backbone,
                          a.smoke_iters)
        info = {"episode": rec["key"], "backbone": a.backbone, "resolution": a.resolution,
                "total_iters": rec["total_iters"], "train_seconds": rec["train_seconds"],
                "seconds_per_iter": rec["train_seconds"] / rec["total_iters"],
                "fidelity_share_class0": rec["fidelity_share_class0"],
                "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20, "environment": env}
        total_iters = sum(x["K"] for x in episodes) * 500   # the official n_iters, whatever the smoke length
        info["extrapolated_hours_single_process"] = total_iters * info["seconds_per_iter"] / 3600
        write_atomic(smoke_dir / f"smoke_{stamp}.json", info)  # predictions are discarded
        print(json.dumps(info, indent=1))
        return 0

    if a.smoke_iters:
        raise SystemExit("--smoke-iters is for --smoke only")
    i, n = parse_shard(a.shard)
    if n > 1 and n % len(doc["K_values"]) == 0:  # K cycles within each product-seed: shard j mod n
        raise SystemExit(f"n = {n} puts every K=5 episode on the same shard(s); use n not divisible by "
                         f"{len(doc['K_values'])} (e.g. 5, 8 or 10)")
    lock = a.out / "locks" / f"shard_{i}_of_{n}"
    if lock.exists() and not a.resume:
        raise SystemExit(f"{lock} exists: this shard was launched before; relaunch only with --resume "
                         "after the user authorizes it")
    lock.mkdir(parents=True, exist_ok=True)
    ep_dir.mkdir(parents=True, exist_ok=True)
    launch_dir = a.out / "launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    mine = shard_of(episodes, i, n)
    todo = [e for e in mine if not (ep_dir / f"{episode_key(e)}.json").exists()]
    write_atomic(launch_dir / f"shard_{i}_of_{n}_{stamp}.json",
                 {"shard": a.shard, "backbone": a.backbone, "resolution": a.resolution, "resume": a.resume,
                  "started": stamp, "assigned": len(mine),
                  "already_done": len(mine) - len(todo), "environment": env})
    for j, e in enumerate(todo, 1):
        rec = run_episode(e, doc["queries"][e["product"]], a.data_root, a.workers, a.resolution, a.backbone)
        if not rec["fidelity_share_class0"]["match"]:
            raise SystemExit(f"fidelity check failed on {rec['key']}: {rec['fidelity_share_class0']}")
        write_atomic(ep_dir / f"{rec['key']}.json", rec)
        print(f"[shard {a.shard}] {j}/{len(todo)} {rec['key']} {rec['train_seconds']:.0f}s", flush=True)
    print(f"[shard {a.shard}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
