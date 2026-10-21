#!/usr/bin/env python3
"""P28: error decomposition of the dual-branch method against the mask oracle (MVTec-FS formal), read-only.

Answers an external review (M5): how much of the remaining gap to the mask oracle is evidence selection
(the oracle, with the true region, gets it right) and how much is type confusion that even perfect
localization does not fix. Uses only saved per-query predictions; no model is run, no result is changed.

- dual branch  `fused_adareg_regres_soft_q99`: ckpt/p14_formal/mvtec/p14_mvtec_predictions.jsonl
- mask oracle  `oracle_maskpool_clip18`:        ckpt/p10_formal/results/formal_predictions.jsonl

Gate: each method's per-episode metrics recomputed from these predictions must equal its raw metrics
(P14 / P10 formal results) exactly, and both must cover the same 195 episodes and the same queries.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p28.error_decomposition
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_stage3_common import classification_metrics  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

CKPT = PROJECT_ROOT / "ckpt"
ARR, ORACLE = "fused_adareg_regres_soft_q99", "oracle_maskpool_clip18"
P14_PRED = CKPT / "p14_formal" / "mvtec" / "p14_mvtec_predictions.jsonl"
P14_RES = CKPT / "p14_formal" / "mvtec" / "p14_mvtec_results.json"
P10_PRED = CKPT / "p10_formal" / "results" / "formal_predictions.jsonl"
P10_RES = CKPT / "p10_formal" / "results" / "formal_results.json"
OUT = CKPT / "p28"
CELLS = ("both_right", "arr_only_wrong", "oracle_only_wrong", "both_wrong")


def load():
    arr, oracle, order = {}, defaultdict(dict), {}
    with P14_PRED.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["method"] == ARR:
                arr[(r["product"], int(r["K"]), int(r["seed"]))] = dict(zip(r["rgb_sha256"], zip(r["true"], r["pred"])))
    with P10_PRED.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["method"] == ORACLE:
                key = (r["product"], int(r["K"]), int(r["seed"]))
                oracle[key][r["rgb_sha256"]] = (r["true_label"], r["predicted_label"])
                order[r["product"]] = r["class_order"]
    return arr, dict(oracle), order


def gate(arr, oracle, order) -> dict:
    if set(arr) != set(oracle) or len(arr) != 195:
        raise rc.P6BFailClosedError(f"episode sets differ: {len(arr)} vs {len(oracle)}")
    out = {}
    for method, preds, res in ((ARR, arr, P14_RES), (ORACLE, oracle, P10_RES)):
        raw = {(r["product"], int(r["K"]), int(r["seed"])): r
               for r in rc.load_json(res)["raw_metrics"] if r["method"] == method}
        worst = 0.0
        for key, q in preds.items():
            shas = sorted(q)
            m = classification_metrics([q[s][0] for s in shas], [q[s][1] for s in shas], order[key[0]])
            worst = max(worst, *(abs(m[k] - raw[key][k]) for k in rc.METRICS))
        out[method] = worst
    for key in arr:
        if set(arr[key]) != set(oracle[key]):
            raise rc.P6BFailClosedError(f"{key}: query sets differ")
        if any(arr[key][s][0] != oracle[key][s][0] for s in arr[key]):
            raise rc.P6BFailClosedError(f"{key}: true labels differ")
    if any(v >= 1e-12 for v in out.values()):
        raise rc.P6BFailClosedError(f"metric reproduction failed: {out}")
    return out


def decompose(arr, oracle):
    """Per episode: share of queries in each cell; for both-wrong, share with the same wrong label."""
    per_episode = {}
    for key, q in arr.items():
        n = len(q)
        cells = dict.fromkeys(CELLS, 0)
        same_wrong = 0
        for s, (true, pa) in q.items():
            po = oracle[key][s][1]
            a_ok, o_ok = pa == true, po == true
            cells["both_right" if a_ok and o_ok else "arr_only_wrong" if o_ok else
                  "oracle_only_wrong" if a_ok else "both_wrong"] += 1
            same_wrong += (not a_ok and not o_ok and pa == po)
        per_episode[key] = {"n": n, **{c: cells[c] / n for c in CELLS}, "counts": cells,
                            "both_wrong_same_label": same_wrong}
    return per_episode


def summarize(per_episode) -> dict:
    """Equal-product macro per (K, seed), then mean over seeds (the paper's convention); plus pooled counts."""
    out = {}
    for k in rc.K_VALUES:
        keys = [key for key in per_episode if key[1] == k]
        by_seed = defaultdict(list)
        for key in keys:
            by_seed[key[2]].append(per_episode[key])
        macro = {c: statistics.fmean(statistics.fmean(e[c] for e in eps) for eps in by_seed.values()) for c in CELLS}
        pooled = {c: sum(per_episode[key]["counts"][c] for key in keys) for c in CELLS}
        both = pooled["both_wrong"]
        arr_err = pooled["arr_only_wrong"] + both
        out[str(k)] = {
            "product_macro_share": macro,
            "pooled_counts": pooled,
            "queries": sum(pooled.values()),
            "arr_errors_fixed_by_oracle": pooled["arr_only_wrong"] / arr_err if arr_err else None,
            "arr_errors_also_oracle_errors": both / arr_err if arr_err else None,
            "both_wrong_same_label": sum(per_episode[key]["both_wrong_same_label"] for key in keys) / both if both else None,
        }
    return out


def per_product(per_episode) -> dict:
    out = {}
    for product in sorted({key[0] for key in per_episode}):
        eps = [v for key, v in per_episode.items() if key[0] == product]
        err = sum(e["counts"]["arr_only_wrong"] + e["counts"]["both_wrong"] for e in eps)
        out[product] = {"arr_errors": err,
                        "fixed_by_oracle": sum(e["counts"]["arr_only_wrong"] for e in eps) / err if err else None,
                        "oracle_only_wrong": sum(e["counts"]["oracle_only_wrong"] for e in eps)}
    return out


def main() -> int:
    arr, oracle, order = load()
    check = gate(arr, oracle, order)
    per_episode = decompose(arr, oracle)
    result = {"schema_version": "p28_error_decomposition_v1", "label": "read-only analysis of saved predictions",
              "provenance": rc.git_provenance(), "methods": {"dual_branch": ARR, "mask_oracle": ORACLE},
              "metric_reproduction_max_abs_diff": check, "by_K": summarize(per_episode),
              "by_product_all_K": per_product(per_episode)}
    OUT.mkdir(parents=True, exist_ok=True)
    rc.write_json(OUT / "p28_mvtec_error_decomposition.json", result)
    lines = ["# P28 — dual branch vs mask oracle, MVTec-FS formal (per-query, saved predictions)", "",
             f"Metric reproduction (max |Δ|): {check}", "",
             "| K | both right | dual only wrong | oracle only wrong | both wrong | dual errors fixed by oracle | "
             "both wrong, same label |", "|---|---|---|---|---|---|---|"]
    for k, v in result["by_K"].items():
        m = v["product_macro_share"]
        lines.append(f"| {k} | {100 * m['both_right']:.1f} | {100 * m['arr_only_wrong']:.1f} | "
                     f"{100 * m['oracle_only_wrong']:.1f} | {100 * m['both_wrong']:.1f} | "
                     f"{100 * v['arr_errors_fixed_by_oracle']:.0f}% | {100 * v['both_wrong_same_label']:.0f}% |")
    lines += ["", "Shares are % of queries (equal-product macro, mean over seeds). Pooled counts in the JSON.", "",
              "| Product | dual errors (all K, seeds) | fixed by oracle | oracle-only errors |", "|---|---|---|---|"]
    for p, v in result["by_product_all_K"].items():
        fixed = "n/a" if v["fixed_by_oracle"] is None else f"{100 * v['fixed_by_oracle']:.0f}%"
        lines.append(f"| {p} | {v['arr_errors']} | {fixed} | {v['oracle_only_wrong']} |")
    text = "\n".join(lines)
    (OUT / "p28_mvtec_error_decomposition.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
