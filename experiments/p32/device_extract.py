#!/usr/bin/env python3
"""P32 (report only): re-extract the P10 MVTec-FS features on the deployment device, FP32 and FP16 autocast.

Same images, order, batch size (4) and code path as the P10 caches (``experiments.p10.formal_extract._extract``):
support pool, formal queries and the formal normal-bank images, for every product. Only the global fusion feature and
the CLIP layer-18 patch tokens are kept (all that the main configuration and the CLIP-L/14-only head use). Scoring is
done locally (``experiments.p32.score_device``) with the unchanged evaluation code, against the cached features.

The normal-image names come from ``ckpt/p32/normal_names.json`` (written locally from the P10 caches), because the
bank draws depend on the full ``train/good`` listing, which the device copy does not need.

Usage:
    python -m experiments.p32.device_extract --write-normal-names                 # local, once
    python -m experiments.p32.device_extract --list-inputs <file> --mvtec-root <root>   # local: files to ship
    python -m experiments.p32.device_extract --mvtec-root <root> --tag jetson_orin_nano   # on the device
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10.extract_dev_features import support_pool_records  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "ckpt" / "p32"
NAMES = OUT_ROOT / "normal_names.json"
BATCH = 4          # = P10; the features depend bitwise on the batch composition
PRECISIONS = ("fp32", "fp16")


def open_rgb(path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def inputs(mvtec_root: Path) -> dict:
    """{product: {"support": [(id, path)], "queries": [(sha, path)], "normals": [(name, path)]}} in P10 order."""
    pools = support_pool_records()
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    names = json.loads(NAMES.read_text(encoding="utf-8"))
    out = {}
    for product in rc.PRODUCTS:
        out[product] = {
            "support": [(r["canonical_relative_path"], rc.DATASET_ROOT / r["canonical_relative_path"]) for r in pools[product]],
            "queries": [(q["rgb_sha256"], Path(q["image_path"])) for q in queries if q["product"] == product],
            "normals": [(n, mvtec_root / product / "train" / "good" / n) for n in names[product]],
        }
    return out


def extract(factory, paths, device, with_global: bool, precision: str):
    from experiments.p10.formal_extract import _extract
    glob, clip = [], []
    for start in range(0, len(paths), BATCH):
        images = [open_rgb(p) for p in paths[start:start + BATCH]]
        ctx = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else contextlib.nullcontext()
        with ctx:
            g, _, c = _extract(factory, images, device, with_global)
        if g is not None:
            glob.append(g)
        clip.append(c)
    return (torch.cat(glob) if glob else None), torch.cat(clip)


def write_normal_names() -> int:
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    names = {}
    for product in rc.PRODUCTS:
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        names[product] = list(nf["names"])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    NAMES.write_text(json.dumps(names, indent=1), encoding="utf-8")
    print(f"{NAMES}: {sum(map(len, names.values()))} normal images")
    return 0


def list_inputs(dest: Path, mvtec_root: Path) -> int:
    """Every file the device run reads besides code and model weights: the images, plus the metadata files that
    load_query_contract / support_pool_records open (captured with an audit hook)."""
    opened = set()

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            opened.add(str(Path(args[0]).resolve()))
    sys.addaudithook(hook)
    groups = inputs(mvtec_root)
    files = {str(Path(p).resolve()) for g in groups.values() for part in g.values() for _, p in part}
    files |= {p for p in opened if Path(p).is_file() and not p.endswith((".py", ".pyc"))}
    files.add(str(NAMES.resolve()))
    dest.write_text("\n".join(sorted(files)), encoding="utf-8")
    print(f"{dest}: {len(files)} files")
    return 0


def run(mvtec_root: Path, tag: str, device_name: str, products=None) -> int:
    """Extract the given products (default: all) into ckpt/p32/<tag>/. Write-once per output file: a product whose
    file already exists stops the run. Run one product per process on small-memory devices (the Jetson run that did
    all thirteen in one process was OOM-killed at the eighth)."""
    from experiments.p6 import run_p6b_stage2_support as support
    out_dir = OUT_ROOT / tag
    groups = inputs(mvtec_root)
    wanted = list(groups) if not products else products
    unknown = [x for x in wanted if x not in groups]
    if unknown:
        raise SystemExit(f"unknown products: {unknown}")
    existing = [f"{pr}_{x}.pt" for x in wanted for pr in PRECISIONS if (out_dir / f"{pr}_{x}.pt").exists()]
    if existing:
        raise rc.P6BFailClosedError(f"{out_dir}: {existing} exist; P32 output files are write-once")
    device = torch.device(device_name)
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"schema_version": "p32_device_extract_v2", "provenance": rc.git_provenance(), "batch": BATCH,
           "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "platform": platform.platform(), "products": {}}
    for product in wanted:
        g = groups[product]
        for precision in PRECISIONS:
            t0 = time.perf_counter()
            sg, sc = extract(factory, [p for _, p in g["support"]], device, True, precision)
            qg, qc = extract(factory, [p for _, p in g["queries"]], device, True, precision)
            _, nc = extract(factory, [p for _, p in g["normals"]], device, False, precision)
            torch.save({"schema_version": "p32_device_features_v1", "product": product, "precision": precision,
                        "support_rel_paths": [i for i, _ in g["support"]], "support_global": sg, "support_clip18": sc,
                        "query_rgb_sha256": [i for i, _ in g["queries"]], "query_global": qg, "query_clip18": qc,
                        "normal_names": [i for i, _ in g["normals"]], "normal_clip18": nc},
                       out_dir / f"{precision}_{product}.pt")
            n = len(g["support"]) + len(g["queries"]) + len(g["normals"])
            log["products"].setdefault(product, {})[precision] = {"images": n, "seconds": time.perf_counter() - t0}
            print(f"[p32] {product} {precision}: {n} images, {time.perf_counter() - t0:.0f} s", flush=True)
            del sg, sc, qg, qc, nc
            torch.cuda.empty_cache()
    (out_dir / f"p32_extract_log_{'_'.join(wanted) if products else 'all'}.json").write_text(
        json.dumps(log, indent=1), encoding="utf-8")
    print("[p32] done", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-normal-names", action="store_true")
    ap.add_argument("--list-inputs", type=Path)
    ap.add_argument("--mvtec-root", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--products", default="", help="comma-separated subset (default: all thirteen)")
    a = ap.parse_args(argv)
    if a.write_normal_names:
        return write_normal_names()
    if a.list_inputs:
        return list_inputs(a.list_inputs, a.mvtec_root)
    if not (a.mvtec_root and a.tag):
        raise SystemExit("--mvtec-root and --tag are required")
    return run(a.mvtec_root, a.tag, a.device, [x for x in a.products.split(",") if x])


if __name__ == "__main__":
    sys.exit(main())
