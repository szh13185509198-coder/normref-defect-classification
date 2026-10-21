#!/usr/bin/env python3
"""P44 (report only): GDA (Wang et al., ICLR 2024; official code github.com/mrflogs/ICLR24), a training-free CLIP
classifier, on MVTec-FS (docs/P44_GDA_TRAINING_FREE_PROTOCOL.md).

Local:
    python -m experiments.p44.gda --dev --backbone b16_224|l14_336        # alpha on the development tasks
    python -m experiments.p44.gda --formal --backbone b16_224|l14_336     # 195 formal episodes, once
    python -m experiments.p44.gda --score
On the device:
    python -m experiments.p44.gda --time --backbone ... --mvtec-root <root> --pass-index 1|2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CLIP_LORA = PROJECT_ROOT / "third_party" / "CLIP-LoRA"
if str(CLIP_LORA) not in sys.path:
    sys.path.insert(0, str(CLIP_LORA))

OUT = PROJECT_ROOT / "ckpt" / "p44"
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)   # official grid
AUG_EPOCHS = 10                                          # official augment_epoch
PINV_ON_CPU = False   # amendment A2: True only for the Jetson timing (CUDA pinv does not load there)
TEMPLATE = ["a photo of a {}."]                          # = Tip-Adapter-F* / CLIP-LoRA in this study
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
LORA_DIRS = {"b16_224": PROJECT_ROOT / "ckpt" / "p21" / "mvtec", "l14_336": PROJECT_ROOT / "ckpt" / "p29" / "mvtec"}
OURS = {"b16_224": "arr_clip_b16_r224", "l14_336": "a_r"}


def load_clip(backbone: str, device):
    import clip
    from experiments.p21 import cliplora_runner as runner
    if backbone == "b16_224":
        model, preprocess = clip.load("ViT-B/16", device=device)
    else:
        if runner.sha256_file(runner.L14_336_FILE) != runner.L14_336_SHA256:
            raise SystemExit("not the official ViT-L/14@336px file")
        model, preprocess = clip.load(str(runner.L14_336_FILE), device=device)
    model.eval()
    return model, preprocess, model.visual.input_resolution


def train_transform(n_px: int):
    """Official GDA few-shot train transform (size = the encoder's input resolution)."""
    import torchvision.transforms as T
    return T.Compose([T.RandomResizedCrop(size=n_px, scale=(0.5, 1), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(p=0.5), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def text_weights(model, classnames):
    from utils import clip_classifier            # CLIP-LoRA's copy == the official GDA / Tip-Adapter function
    return clip_classifier(list(classnames), TEMPLATE, model).float()


def encode(model, batch):
    import torch
    with torch.no_grad():
        f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.float()


def support_features(model, images, labels, n_px, seed_text, device):
    """10 augmented epochs over the supports (official cache construction), seeded per episode."""
    import torch
    torch.manual_seed(int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16))
    tf = train_transform(n_px)
    vecs, labs = [], []
    for _ in range(AUG_EPOCHS):
        vecs.append(encode(model, torch.stack([tf(im) for im in images]).to(device)))
        labs += list(labels)
    return torch.cat(vecs), torch.as_tensor(labs, device=device)


def gda_fit(vecs, labels, n):
    """GDA from the paper's equations: class means, shared covariance with trace shrinkage, uniform prior.
    No code from the official repository (AGPL-3.0) is included; the outputs equal the run reported in the
    paper bitwise (checked on CPU and CUDA)."""
    import torch
    classes = range(n)
    means = torch.stack([vecs[labels == c].mean(dim=0) for c in classes])
    resid = torch.cat([vecs[labels == c] - means[c] for c in classes])
    dim = resid.shape[1]
    cov = resid.T.cov()
    shrunk = (resid.shape[0] - 1) * cov + cov.trace() * torch.eye(dim, device=vecs.device)
    precision = dim * (torch.linalg.pinv(shrunk.cpu()).to(vecs.device) if PINV_ON_CPU else torch.linalg.pinv(shrunk))
    weight = torch.einsum("kd,de->ek", means, precision)
    quad = torch.einsum("kd,de,ke->k", means, precision, means)
    bias = (torch.ones(n, device=vecs.device) * 1. / n).log() - quad / 2
    return weight, bias


def logits(f, clip_w, W, b, alpha):
    return 100. * f.float() @ clip_w.float() + alpha * (f.float() @ W + b)


def open_rgb(path):
    from PIL import Image
    with Image.open(path) as h:
        return h.convert("RGB").copy()


def query_features(model, preprocess, paths, device, batch=16):
    import torch
    out = []
    for i in range(0, len(paths), batch):
        out.append(encode(model, torch.stack([preprocess(open_rgb(p)) for p in paths[i:i + batch]]).to(device)))
    return torch.cat(out)


# --------------------------------------------------------------------------- development tasks: alpha

def dev(args) -> int:
    import torch
    from experiments.p6.p6b_stage3_common import classification_metrics
    from experiments.p10 import run_pilot as rp
    from experiments.revision import revision_common as rc
    target = OUT / args.backbone / "p44_dev.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    device = torch.device("cuda")
    model, preprocess, n_px = load_clip(args.backbone, device)
    rows = []
    for product in rc.PRODUCTS:
        data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        paths = [rc.DATASET_ROOT / rel for rel in data["rel_paths"]]
        images = [open_rgb(p) for p in paths]
        qf = query_features(model, preprocess, paths, device)
        tw = {}
        for seed in rp.DEV_SEEDS:
            for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, product):
                key = tuple(order)
                if key not in tw:
                    tw[key] = text_weights(model, order)
                vecs, labs = support_features(model, [images[i] for i in s_idx], s_lab, n_px,
                                              f"p44dev|{args.backbone}|{product}|K{k_shot}|{seed}", device)
                W, b = gda_fit(vecs, labs, len(order))
                f = qf[torch.as_tensor(q_idx, device=device)]
                truth = [order[c] for c in q_lab]
                for alpha in ALPHAS:
                    pred = [order[int(i)] for i in logits(f, tw[key], W, b, alpha).argmax(dim=1).cpu()]
                    rows.append({"method": f"gda_a{alpha:g}", "product": product, "K": k_shot, "seed": seed,
                                 **classification_metrics(truth, pred, order)})
        print(f"[p44 dev {args.backbone}] {product}", flush=True)
    mean_ba = {a: statistics.fmean(r["balanced_accuracy"] for r in rows if r["method"] == f"gda_a{a:g}") for a in ALPHAS}
    best = max(ALPHAS, key=lambda a: (round(mean_ba[a], 12), -a))
    OUT.joinpath(args.backbone).mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"protocol": "docs/P44_GDA_TRAINING_FREE_PROTOCOL.md#2", "backbone": args.backbone,
                                  "mean_dev_ba": {f"{a:g}": v for a, v in mean_ba.items()}, "selected_alpha": best,
                                  "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({"mean_dev_ba": {f"{a:g}": round(100 * v, 2) for a, v in mean_ba.items()}, "selected": best}))
    return 0


# --------------------------------------------------------------------------- formal episodes

def formal(args) -> int:
    import torch
    from experiments.p21 import cliplora_runner as runner
    from experiments.p21 import score as p21
    from experiments.revision import revision_common as rc
    out = OUT / args.backbone / "episodes"
    if out.exists():
        raise SystemExit(f"{out} exists (single shot)")
    alpha = json.loads((OUT / args.backbone / "p44_dev.json").read_text(encoding="utf-8"))["selected_alpha"]
    doc = runner.load_episodes(p21.EPISODES)
    device = torch.device("cuda")
    model, preprocess, n_px = load_clip(args.backbone, device)
    out.mkdir(parents=True)
    qcache = {}
    for e in doc["episodes"]:
        product = e["product"]
        if product not in qcache:
            qcache[product] = query_features(model, preprocess, [rc.DATASET_ROOT / q["rel_path"] for q in doc["queries"][product]], device)
        images = [open_rgb(rc.DATASET_ROOT / s["rel_path"]) for s in e["supports"]]
        labels = [s["label"] for s in e["supports"]]
        order = e["class_order"]
        tw = text_weights(model, order)
        vecs, labs = support_features(model, images, labels, n_px, f"p44|{args.backbone}|{product}|K{e['K']}|{e['seed']}", device)
        W, b = gda_fit(vecs, labs, len(order))
        f = qcache[product]
        rec = {"key": runner.episode_key(e), "backbone": args.backbone, "alpha": alpha, "class_order": order,
               "queries": [{"sha256": q["sha256"]} for q in doc["queries"][product]],
               "gda_pred": [order[int(i)] for i in logits(f, tw, W, b, alpha).argmax(dim=1).cpu()],
               "zeroshot_pred": [order[int(i)] for i in (100. * f @ tw).argmax(dim=1).cpu()],
               "all_alpha_pred": {f"{a:g}": [order[int(i)] for i in logits(f, tw, W, b, a).argmax(dim=1).cpu()] for a in ALPHAS}}
        (out / f"{rec['key']}.json").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    print(f"[p44 formal {args.backbone}] {len(doc['episodes'])} episodes, alpha {alpha:g}", flush=True)
    return 0


def gda_rows(backbone, doc, truth):
    from experiments.p6.p6b_stage3_common import classification_metrics
    from experiments.p21 import cliplora_runner as runner
    rows = []
    for e in doc["episodes"]:
        rec = json.loads((OUT / backbone / "episodes" / f"{runner.episode_key(e)}.json").read_text(encoding="utf-8"))
        if rec["class_order"] != e["class_order"] or [q["sha256"] for q in rec["queries"]] != [q["sha256"] for q in doc["queries"][e["product"]]]:
            raise SystemExit(f"record does not match the episode file: {rec['key']}")
        y = [truth[(e["product"], q["sha256"])] for q in rec["queries"]]
        preds = {f"gda_{backbone}": rec["gda_pred"], f"zeroshot_{backbone}": rec["zeroshot_pred"],
                 **{f"gda_{backbone}_a{a}": v for a, v in rec["all_alpha_pred"].items()}}
        for m, pred in preds.items():
            rows.append({"method": m, "product": e["product"], "K": e["K"], "seed": e["seed"],
                         **classification_metrics(y, pred, e["class_order"])})
    return rows


def score(args) -> int:
    from experiments.p6.p6b_stage3_common import load_query_contract
    from experiments.p20.stats_holm import holm
    from experiments.p21 import cliplora_runner as runner
    from experiments.p21 import score as p21
    from experiments.revision import revision_common as rc
    target = OUT / "p44_results.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    doc = runner.load_episodes(p21.EPISODES)
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    grid = p21.comparator_grid()
    for bb in ("b16_224", "l14_336"):
        grid.update(rc.metric_grid_from_rows(gda_rows(bb, doc, truth)))
        grid.update(rc.metric_grid_from_rows([{**r, "method": {p21.LORA: f"cliplora_{bb}", p21.ZS: f"p2x_zeroshot_{bb}"}[r["method"]]}
                                              for r in p21.lora_rows(LORA_DIRS[bb], doc, truth)]))
    grid.update(rc.metric_grid_from_rows([r for r in rc.load_json(PROJECT_ROOT / "ckpt/p17/mvtec/p17_mvtec_results.json")["raw_metrics"]
                                          if r["method"] == "a_r"]))
    declared = {"G1 a_r (ours, CLIP-L/14@336) - GDA L/14@336": ("a_r", "gda_l14_336"),
                "G2 ours CLIP-B/16@224 - GDA B/16@224": ("arr_clip_b16_r224", "gda_b16_224"),
                "G3 GDA B/16@224 - CLIP-LoRA B/16@224": ("gda_b16_224", "cliplora_b16_224"),
                "G4 GDA L/14@336 - CLIP-LoRA L/14@336": ("gda_l14_336", "cliplora_l14_336")}
    descriptive = {"GDA B/16@224 - zero-shot B/16@224": ("gda_b16_224", "zeroshot_b16_224"),
                   "GDA L/14@336 - zero-shot L/14@336": ("gda_l14_336", "zeroshot_l14_336"),
                   "GDA B/16@224 - SimpleShot B/16@224": ("gda_b16_224", "ss_clip_b16_r224"),
                   "zero-shot check B/16@224 (P44 - P21)": ("zeroshot_b16_224", "p2x_zeroshot_b16_224"),
                   "zero-shot check L/14@336 (P44 - P29)": ("zeroshot_l14_336", "p2x_zeroshot_l14_336")}
    n_cells = len(rc.PRODUCTS) * len(rc.K_VALUES) * len(rc.SEEDS)
    for m in {m for pair in (*declared.values(), *descriptive.values()) for m in pair}:
        if len(grid.get(m, {})) != n_cells:
            raise SystemExit(f"incomplete grid for {m}")
    res = {n: p21.paired(grid, t, r) for n, (t, r) in declared.items()}
    adj = holm({n: v["signflip_p"] for n, v in res.items()})
    for n in res:
        res[n]["holm_p"] = adj[n]
    oracle = {}
    for bb in ("b16_224", "l14_336"):   # amendment A1: descriptive upper bound with test labels
        per_alpha = {f"{a:g}": statistics.fmean(100 * rc.product_macro_summary(grid, f"gda_{bb}_a{a:g}")[str(k)]["balanced_accuracy"]["mean"]
                                                for k in rc.K_VALUES) for a in ALPHAS}
        best = max(per_alpha, key=per_alpha.get)
        oracle[bb] = {"test_ba_by_alpha_mean_over_K": per_alpha, "test_oracle_alpha": best}
        descriptive[f"ours - GDA {bb} (test-oracle alpha, uses test labels)"] = (OURS[bb], f"gda_{bb}_a{best}")
    desc = {n: p21.paired(grid, t, r) for n, (t, r) in descriptive.items()}
    levels = {m: {str(k): 100 * rc.product_macro_summary(grid, m)[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES}
              for m in ("gda_b16_224", "gda_l14_336", "zeroshot_b16_224", "zeroshot_l14_336", "a_r", "arr_clip_b16_r224",
                        "cliplora_b16_224", "cliplora_l14_336", "ss_clip_b16_r224")}
    alphas = {bb: json.loads((OUT / bb / "p44_dev.json").read_text(encoding="utf-8"))["selected_alpha"] for bb in ("b16_224", "l14_336")}
    result = {"schema_version": "p44_v1", "protocol": "docs/P44_GDA_TRAINING_FREE_PROTOCOL.md", "alpha": alphas,
              "declared": res, "descriptive": desc, "test_oracle_alpha_A1": oracle, "levels_ba_pct": levels, "provenance": rc.git_provenance()}
    rc.write_json(target, result)
    print(json.dumps({"alpha": alphas, "levels": levels,
                      "declared": {n: {k: v[k] for k in v if k in ("Dbar_pp", "ci95_pp", "positive", "holm_p", "signflip_p")} for n, v in res.items()}},
                     indent=1, default=str))
    return 0


# --------------------------------------------------------------------------- device timing

def device_time(args) -> int:
    import torch
    from experiments.p26 import cost_profile as cp
    out = OUT / "jetson_orin_nano"
    f = out / f"pass{args.pass_index}_{args.backbone}.json"
    if f.exists():
        raise SystemExit(f"{f} exists (write-once)")
    out.mkdir(parents=True, exist_ok=True)
    global PINV_ON_CPU
    PINV_ON_CPU = True
    device = torch.device("cuda")
    supports, s_lab, n, _, order, _ = cp.task_inputs(Path(args.mvtec_root))
    model, preprocess, n_px = load_clip(args.backbone, device)
    alpha = 1.0   # timing only: alpha does not change the cost
    sync = torch.cuda.synchronize
    for _ in range(5):
        encode(model, preprocess(supports[0])[None].to(device))
    sync()
    t0 = time.perf_counter()
    tw = text_weights(model, order)
    vecs, labs = support_features(model, supports, s_lab, n_px, "p44time", device)
    W, b = gda_fit(vecs, labs, n)
    sync()
    new_task_s = time.perf_counter() - t0
    q = supports[0]
    x = preprocess(q)[None].to(device)

    def e2e():
        f = encode(model, preprocess(q)[None].to(device))
        return int(logits(f, tw, W, b, alpha).argmax())
    res = {"backbone": args.backbone, "pass": args.pass_index, "new_task_s": new_task_s,
           "e2e_from_pil": cp.timed(e2e, device), "model_forward": cp.timed(lambda: model.encode_image(x), device),
           "precision": str(next(model.visual.parameters()).dtype), "pinv_on_cpu": PINV_ON_CPU, "torch": torch.__version__,
           "machine": cp.machine_info(), "telemetry": cp.telemetry()}
    f.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"new_task_s": new_task_s, "e2e_ms": res["e2e_from_pil"]["median_ms"],
                      "forward_ms": res["model_forward"]["median_ms"]}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--dev", "--formal", "--score", "--time"):
        ap.add_argument(flag, action="store_true")
    ap.add_argument("--backbone", choices=("b16_224", "l14_336"))
    ap.add_argument("--mvtec-root")
    ap.add_argument("--pass-index", type=int, default=1)
    a = ap.parse_args(argv)
    for flag, fn in (("dev", dev), ("formal", formal), ("score", score), ("time", device_time)):
        if getattr(a, flag):
            return fn(a)
    raise SystemExit("choose --dev, --formal, --score or --time")


if __name__ == "__main__":
    sys.exit(main())
