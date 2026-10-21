#!/usr/bin/env python3
"""P19: one-shot blind test on Real-IAD (docs/P19_REALIAD_BLIND_TEST_PROTOCOL.md).

Every method and constant is the one frozen after P18; the scoring helpers are the P18 ones,
which reproduce P14 exactly. Run once:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p19.run --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.mvrec_rgb_formal.decision_evaluator import percentile_bootstrap  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18 import baselines as bl  # noqa: E402
from experiments.p18 import run as r18  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

PROTOCOL = "docs/P19_REALIAD_BLIND_TEST_PROTOCOL.md"
MANIFEST = PROJECT_ROOT / "docs" / "p19" / "P19_REALIAD_MANIFEST.json"
DATA = PROJECT_ROOT / "data" / "Real-IAD" / "realiad_512"
OUT = PROJECT_ROOT / "ckpt" / "p19" / "blind"
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13", "experiments/p14",
           "experiments/p17", "experiments/p18", "experiments/p19", "experiments/revision", PROTOCOL, "docs/p19")
SS, NR, XC, A, ARR, CONS = r18.SS, r18.NR, r18.XC, r18.A, r18.ARR, r18.CONS
K_VALUES = (1, 3, 5)
DINO_B = ("dinov2_b", 11)
CROP = ("largest", 1.0)
SETTINGS = ("defect", "normal")
GRID = 24
BOOT_SEED = 20260928
SIGNFLIP_DRAWS = 200_000
HYPOTHESES = {"B1": (ARR, SS), "B2": (CONS, SS), "B3": (A, NR), "B4": (ARR, A), "B5": (ARR, "crop_fused"),
              "B6": (CONS, "dn4_fused")}
DESCRIPTIVE = [("arr_dinov2_b", ARR), ("cons_dinov2_b", CONS), ("arr_dinov2_b", "ss_dinov2_b"),
               ("a_dinov2_b", "nr_dinov2_b"), ("crop_fused", SS), ("dn4_fused", SS)]


@contextlib.contextmanager
def single_shot(out: Path, resume: bool = False):
    """Lock + staging. resume=True (Amendment P19-A1, user-authorised only) continues inside the lock and
    staging left by an interrupted launch instead of deleting them; per-object checkpoints are reused."""
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; P19 is single-shot")
    lock = out.with_name(out.name + ".lock")
    staging = out.with_name(out.name + ".staging")
    if resume:
        if not (lock.is_dir() and staging.is_dir()):
            raise rc.P6BFailClosedError("--resume needs the lock and staging of an interrupted launch")
    else:
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted - report, do not delete")
        if staging.exists():
            raise rc.P6BFailClosedError(f"{staging} exists - report, do not delete")
        staging.mkdir(parents=True)
    launches = staging / "launches.json"
    history = json.loads(launches.read_text(encoding="utf-8")) if launches.is_file() else []
    history.append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": resume})
    launches.write_text(json.dumps(history, indent=1), encoding="utf-8")
    yield staging
    staging.rename(out)


def checkpoint_path(staging: Path, obj: str) -> Path:
    return staging / "objects" / f"{obj}.json"


def load_checkpoint(staging: Path, obj: str, head: str, manifest_sha: str):
    path = checkpoint_path(staging, obj)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["head"] != head or data["manifest_sha256"] != manifest_sha:
        raise rc.P6BFailClosedError(f"checkpoint {path} comes from another commit or manifest")
    return data["rows"]


def save_checkpoint(staging: Path, obj: str, head: str, manifest_sha: str, rows) -> None:
    path = checkpoint_path(staging, obj)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"head": head, "manifest_sha256": manifest_sha, "rows": rows}), encoding="utf-8")
    tmp.replace(path)


def mask_cells(path: Path, grid: int = GRID) -> torch.Tensor:
    """Area fraction of the official defect mask per grid cell (the 336 view is a squash resize)."""
    with Image.open(path) as handle:
        small = handle.convert("L").resize((grid, grid), Image.BOX)
    return torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).flatten()


def signflip_mc(d: list[float], draws: int = SIGNFLIP_DRAWS, seed: int = BOOT_SEED) -> float:
    x = np.asarray(d, dtype=np.float64)
    obs = abs(x.mean())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(draws, len(x)))
    return float((1 + np.sum(np.abs(signs @ x) / len(x) >= obs - 1e-15)) / (1 + draws))


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def contrast(rows, setting, target, ref) -> dict:
    cell = {(r["method"], r["object"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows if r["setting"] == setting}
    objects = sorted({r["object"] for r in rows if r["setting"] == setting})
    seeds = sorted({r["seed"] for r in rows})
    per = {o: [cell[(target, o, k, s)] - cell[(ref, o, k, s)] for k in K_VALUES for s in seeds] for o in objects}
    obj_mean = {o: statistics.fmean(v) for o, v in per.items()}
    boot = percentile_bootstrap(per, seed=BOOT_SEED)
    per_k = {str(k): 100 * statistics.fmean(cell[(target, o, k, s)] - cell[(ref, o, k, s)] for o in objects for s in seeds)
             for k in K_VALUES}
    return {"delta_pp": 100 * statistics.fmean(obj_mean.values()), "ci95_pp": [100 * boot["ci95_lower"], 100 * boot["ci95_upper"]],
            "per_K_pp": per_k, "positive_objects": sum(v > 0 for v in obj_mean.values()), "n_objects": len(objects),
            "signflip_p": signflip_mc(list(obj_mean.values())), "per_object_pp": {o: 100 * v for o, v in obj_mean.items()}}


def levels(rows, setting) -> dict:
    out = {}
    for m in sorted({r["method"] for r in rows if r["setting"] == setting}):
        out[m] = [100 * statistics.fmean(r["balanced_accuracy"] for r in rows
                                         if r["setting"] == setting and r["method"] == m and r["K"] == k) for k in K_VALUES]
    return out


def object_rows(obj, task, factory, dino_b, device, batch_size) -> list[dict]:
    root = DATA / obj
    classes = list(task["classes"])
    seeds = sorted(task["banks"])
    images = sorted({p for c in task["classes"].values() for p in c["pool"] + c["queries"]}
                    | set(task["normal"]["pool"] + task["normal"]["queries"])
                    | {p for b in task["banks"].values() for p in b})
    index = {p: i for i, p in enumerate(images)}
    paths = [root / p for p in images]
    g, clip = ee.extract(factory, paths, device, batch_size)
    gb, pb = bb.extract(dino_b, paths, device, batch_size, layers=(DINO_B[1],))
    gb, pb = gb.to(device), pb[DINO_B[1]]
    g = g.to(device)
    masks = torch.zeros(len(images), GRID * GRID)
    for c in task["classes"].values():
        for p, mp in c["masks"].items():
            masks[index[p]] = mask_cells(root / mp)
    masks = masks.to(device)
    wanted = sorted({index[p] for c in task["classes"].values() for p in c["pool"] + c["queries"]}
                    | {index[p] for p in task["normal"]["pool"] + task["normal"]["queries"]})
    per_seed, boxes = {}, {}
    for s in seeds:
        bank_imgs = clip[torch.as_tensor([index[p] for p in task["banks"][s]])].to(device).float()
        per_seed[s], w = r18.c1_parts_chunked(clip, bank_imgs, fa.tau99(bank_imgs), device)
        boxes[s] = bl.crop_boxes(w, *CROP)
    crops = r18.crop_features(factory, lambda i: r18.open_rgb(paths[i]), len(images), wanted, boxes, device, batch_size)
    rows = []
    for s in seeds:
        bidx = torch.as_tensor([index[p] for p in task["banks"][s]])
        bank_b = pb[bidx].to(device).float()
        parts_b = r18.c1_parts_chunked(pb, bank_b, fa.tau99(bank_b), device)[0]
        for setting in SETTINGS:
            order = sorted(classes + (["OK"] if setting == "normal" else []))
            q_paths, q_true = [], []
            for c in order:
                qs = task["normal"]["queries"] if c == "OK" else task["classes"][c]["queries"]
                q_paths += qs
                q_true += [c] * len(qs)
            qi = torch.as_tensor([index[p] for p in q_paths], device=device)
            for k in K_VALUES:
                s_idx, s_lab = [], []
                for ci, c in enumerate(order):
                    for p in task["supports"][s][c][:k]:
                        s_idx.append(index[p])
                        s_lab.append(ci)
                si = torch.as_tensor(s_idx, device=device)
                n = len(order)
                sc = r18.chunked_scores(g, clip, per_seed[s], si, s_lab, qi, clip[qi.cpu()], n, device)
                sup = clip[si.cpu()].to(device).float()
                dn4 = torch.cat([bl.dn4_scores(sup, s_lab, ch.to(device).float(), n) for ch in clip[qi.cpu()].split(ee.Q_CHUNK)])
                r18.add_local_baselines(sc, dn4, {"crop": crops[s]}, si, s_lab, qi, n, device)
                if setting == "defect":
                    qp = clip[qi.cpu()].to(device).float()
                    sc["oracle_clipl"] = rp.oracle_maskpool(sup, masks[si], s_lab, qp, masks[qi], n)
                sc.update(r18.renamed(r18.chunked_scores(gb, pb, parts_b, si, s_lab, qi, pb[qi.cpu()], n, device), "dinov2_b"))
                new = fa.score_rows(sc, order, q_true, {"object": obj, "setting": setting, "K": k, "seed": int(s)})
                if setting == "normal":
                    ok = torch.as_tensor([t == "OK" for t in q_true], device=device)
                    for row in new:
                        row["good_recall"] = float((sc[row["method"]].argmax(dim=1)[ok] == order.index("OK")).float().mean())
                rows += new
    return rows


def industrial(rows) -> dict:
    """Normal setting, argmax: good recall and false rejects / 1000 at 95 % good (per-method mean over cells)."""
    out = {}
    for m in (SS, A, ARR, CONS, "arr_dinov2_b"):
        rec = [r["good_recall"] for r in rows if r["setting"] == "normal" and r["method"] == m and r["K"] == 5]
        good = statistics.fmean(rec)
        out[m] = {"good_recall_K5": 100 * good, "false_rejects_per_1000_K5": 1000 * 0.95 * (1 - good)}
    return out


def run(device, batch_size, resume: bool = False) -> int:
    manifest = rc.load_json(MANIFEST)
    manifest_sha = bb.file_sha256(MANIFEST)
    with single_shot(OUT, resume) as staging:
        provenance = rc.git_provenance()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        sel = rc.load_json(r18.OUT["dev"] / "p18_dev_results.json")["selection"]
        if sel["dinov2_b"] != f"dinov2_b_L{DINO_B[1]}" or tuple(sel["crop"]) != CROP:
            raise rc.P6BFailClosedError(f"P18 selections differ from the frozen P19 constants: {sel}")
        dino_b = bb.DinoB(device, sel["dinov2_b_sha256"])
        rows, reused = [], []
        for obj, task in manifest["objects"].items():
            done = load_checkpoint(staging, obj, provenance["head"], manifest_sha)
            if done is not None:
                rows += done
                reused.append(obj)
                print(f"[p19] {obj} reused from checkpoint", flush=True)
                continue
            t0 = time.time()
            with torch.no_grad():
                new = object_rows(obj, task, factory, dino_b, device, batch_size)
            save_checkpoint(staging, obj, provenance["head"], manifest_sha, new)
            rows += new
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            print(f"[p19] {time.strftime('%H:%M:%S')} {obj} done in {time.time() - t0:.0f} s (peak GPU {peak:.2f} GB)", flush=True)
        hyp = {h: contrast(rows, "defect", t, r) for h, (t, r) in HYPOTHESES.items()}
        adj = holm({h: v["signflip_p"] for h, v in hyp.items()})
        for h, v in hyp.items():
            v.update({"target": HYPOTHESES[h][0], "reference": HYPOTHESES[h][1], "holm_p": adj[h],
                      "verdict": "CONFIRMED" if v["ci95_pp"][0] > 0 and adj[h] < 0.05 else "NOT CONFIRMED"})
        desc = {setting: {f"{t} − {r}": contrast(rows, setting, t, r)
                          for t, r in DESCRIPTIVE + ([] if setting == "defect" else list(HYPOTHESES.values()))}
                for setting in SETTINGS}
        lv = {s: levels(rows, s) for s in SETTINGS}
        d = lv["defect"]
        headroom = [(d[ARR][i] - d[SS][i]) / (d["oracle_clipl"][i] - d[SS][i]) if d["oracle_clipl"][i] != d[SS][i] else float("nan")
                    for i in range(len(K_VALUES))]
        result = {"schema_version": "p19_blind_v1", "protocol": PROTOCOL, "label": "one-shot blind test; frozen methods",
                  "provenance": provenance, "manifest_sha256": manifest_sha, "hypotheses": hyp,
                  "launches": json.loads((staging / "launches.json").read_text(encoding="utf-8")),
                  "objects_reused_from_checkpoint": reused,
                  "descriptive": desc, "levels": lv, "oracle_headroom_recovered_defect": headroom,
                  "industrial_normal_setting": industrial(rows), "raw_metrics": rows}
        text = table(result)
        rc.write_json(staging / "p19_results.json", result)
        (staging / "p19_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def fmt(v) -> str:
    return (f"{v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] "
            f"({v['positive_objects']}/{v['n_objects']}, p {v['signflip_p']:.4f})")


def table(result) -> str:
    lines = ["# P19 — Real-IAD one-shot blind test (30 objects, frozen methods)", "",
             f"Commit `{result['provenance']['head']}`; manifest SHA256 `{result['manifest_sha256']}`.", "",
             "## Hypotheses (defect setting, BA pooled over K; object-cluster bootstrap; sign-flip; Holm over B1–B6)", "",
             "| ID | Contrast | Δ pooled [95% CI] (+objects, p) | K=1 / 3 / 5 | Holm p | Verdict |", "|---|---|---|---|---|---|"]
    for h, v in result["hypotheses"].items():
        lines.append(f"| {h} | {v['target']} − {v['reference']} | {fmt(v)} | "
                     + " / ".join(f"{x:+.2f}" for x in v["per_K_pp"].values()) + f" | {v['holm_p']:.4f} | {v['verdict']} |")
    for setting, cs in result["descriptive"].items():
        lines += ["", f"## Descriptive — {setting} setting", "", "| Contrast | Δ pooled [95% CI] (+objects, p) |", "|---|---|"]
        lines += [f"| {k} | {fmt(v)} |" for k, v in cs.items()]
    for setting, lv in result["levels"].items():
        lines += ["", f"## BA levels — {setting} setting (object macro, %)", "", "| Method | K=1 / 3 / 5 |", "|---|---|"]
        lines += [f"| {m} | " + " / ".join(f"{x:.2f}" for x in v) + " |" for m, v in lv.items()]
    lines += ["", "Oracle headroom recovered (defect, K=1/3/5): " + " / ".join(f"{x:.2f}" for x in result["oracle_headroom_recovered_defect"]),
              "", "Normal setting, K=5 argmax: " + "; ".join(
                  f"{m} good recall {v['good_recall_K5']:.1f}%, false rejects {v['false_rejects_per_1000_K5']:.0f}/1000"
                  for m, v in result["industrial_normal_setting"].items())]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Amendment P19-A1: continue an interrupted launch (user authorisation required)")
    args = parser.parse_args(argv)
    return run(torch.device(args.device), args.batch_size, args.resume)


if __name__ == "__main__":
    sys.exit(main())
