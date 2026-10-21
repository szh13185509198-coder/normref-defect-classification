#!/usr/bin/env python3
"""P47 U6-D local analysis (docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, section U6-D). Torch-free.

After the device run, with its outputs copied back into ckpt/:
    python -m experiments.p47.analyze_u6d --tag p47_jetson
Self-check on synthetic files:
    python -m experiments.p47.analyze_u6d --selftest

D1: one-way random-effects model (session as factor) from the per-session P43 phase-A summaries (mean, SD, n):
    within-session variance s_w^2 = pooled SD^2; between-session variance s_b^2 = max(0, var(session means) - s_w^2/n).
    The standard uncertainty of a single deployed measurement of the latency, u = sqrt(s_w^2 + s_b^2), and the
    standard uncertainty of the grand mean, sqrt(var(session means)/k), are reported (Type A, JCGM 100 4.2).
D2: per window, mean VDD_IN over the idle and the run windows (tegrastats lines inside the window's start/end), energy
    per part = power x mean latency, dynamic energy = (power - idle power) x mean latency; mean, SD and u = SD/sqrt(5).
D3: tabulated as measured.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT = PROJECT_ROOT / "ckpt"


def tegrastats_samples(path: Path) -> list[tuple[str, float]]:
    """(local 'YYYY-MM-DD HH:MM:SS', VDD_IN watts) per line, as experiments.p43.edge_measure.report_energy parses."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
        p = re.search(r"VDD_IN (\d+)mW", line)
        if m and p:
            out.append((datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S"),
                        int(p.group(1)) / 1000))
    return out


def window_power(samples, start: str, end: str):
    v = [w for t, w in samples if start <= t <= end]
    return (statistics.fmean(v), len(v)) if v else (None, 0)


def d1(p43_dir: Path) -> dict:
    files = sorted(p43_dir.glob("p43_repeat_s*.json"))
    if not files:
        raise SystemExit(f"no p43_repeat_s*.json in {p43_dir}")
    sess = []
    for f in files:
        e = json.loads(f.read_text(encoding="utf-8"))["phases"]["A"]["e2e"]
        sess.append({"session": f.stem.split("_")[-1], "n": e["n"], "mean_ms": e["mean_ms"], "sd_ms": e["sd_ms"],
                     "p50_ms": e["p50_ms"], "p99_ms": e["p99_ms"], "max_ms": e["max_ms"]})
    k = len(sess)
    n = statistics.fmean(s["n"] for s in sess)
    means = [s["mean_ms"] for s in sess]
    s_w2 = statistics.fmean(s["sd_ms"] ** 2 for s in sess)
    var_means = statistics.variance(means) if k > 1 else 0.0
    s_b2 = max(0.0, var_means - s_w2 / n)
    return {"sessions": sess, "k": k, "grand_mean_ms": statistics.fmean(means),
            "median_of_session_medians_ms": statistics.median(s["p50_ms"] for s in sess),
            "range_of_session_means_ms": [min(means), max(means)],
            "within_session_sd_ms": math.sqrt(s_w2), "between_session_sd_ms": math.sqrt(s_b2),
            "u_single_measurement_ms": math.sqrt(s_w2 + s_b2),
            "u_grand_mean_ms": math.sqrt(var_means / k) if k > 1 else None, "dof_grand_mean": k - 1}


def d2(p43_root: Path, tag: str, tegra: Path) -> dict:
    samples = tegrastats_samples(tegra)
    if not samples:
        raise SystemExit(f"no VDD_IN lines in {tegra}")
    rows = []
    for d in sorted(p43_root.glob(f"{tag}_e*")):
        idle = json.loads((d / "p43_idle.json").read_text(encoding="utf-8"))["idle"]
        run = json.loads((d / "p43_energy_fused_e2e.json").read_text(encoding="utf-8"))
        p_idle, k_idle = window_power(samples, idle["start"]["local"], idle["end"]["local"])
        p_run, k_run = window_power(samples, run["start"]["local"], run["end"]["local"])
        e = p_run * run["mean_ms"] / 1000 if p_run else None
        rows.append({"window": d.name, "mean_ms": run["mean_ms"], "calls": run["forwards"], "power_w": p_run,
                     "samples": k_run, "idle_power_w": p_idle, "idle_samples": k_idle, "energy_j": e,
                     "dynamic_energy_j": (p_run - p_idle) * run["mean_ms"] / 1000 if p_run and p_idle else None})
    if not rows:
        raise SystemExit(f"no {tag}_e* windows in {p43_root}")

    def summ(key):
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return None
        sd = statistics.stdev(v) if len(v) > 1 else None
        return {"mean": statistics.fmean(v), "sd": sd, "u_mean": sd / math.sqrt(len(v)) if sd is not None else None,
                "n": len(v)}
    return {"windows": rows, **{k: summ(k) for k in ("energy_j", "dynamic_energy_j", "power_w", "idle_power_w", "mean_ms")},
            "note": "Type B power-sensor accuracy not included (datasheet value to be verified)"}


def analyze(tag: str, root: Path = CKPT) -> dict:
    out_dir = root / "p47" / tag
    res = {"protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u6-d", "tag": tag,
           "D1_sessions": d1(root / "p43" / tag),
           "D2_energy": d2(root / "p43", tag, out_dir / "tegrastats.log")}
    sc = json.loads((out_dir / "p47_u6_scaling.json").read_text(encoding="utf-8"))
    res["D3_scaling"] = {"classes_p50_ms": {k: v["p50_ms"] for k, v in sc["classes_e2e"].items()},
                         "head_p50_ms": {k: {"fp32": v["fp32"]["p50_ms"], "fp16": v["fp16"]["p50_ms"],
                                             "bank_rows": v["bank_rows"]} for k, v in sc["head_vs_M"].items()},
                         "tiles_p50_ms_per_part": {k: v["p50_ms"] for k, v in sc["tiles_per_part"].items()},
                         "gpu_freq": [sc.get("gpu_freq_start"), sc.get("gpu_freq_end")],
                         "max_temp_c": [sc.get("max_temp_c_start"), sc.get("max_temp_c_end")]}
    a, e = res["D1_sessions"], res["D2_energy"]
    lines = ["# P47 U6-D — device uncertainty summary", "",
             f"D1 ({a['k']} sessions): grand mean {a['grand_mean_ms']:.2f} ms; within-session SD "
             f"{a['within_session_sd_ms']:.3f} ms; between-session SD {a['between_session_sd_ms']:.3f} ms; "
             f"u(single measurement) {a['u_single_measurement_ms']:.3f} ms; u(grand mean) "
             f"{a['u_grand_mean_ms']:.3f} ms (dof {a['dof_grand_mean']}).", "",
             "| Session | n | mean (ms) | SD (ms) | p50 | p99 | max |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {s['session']} | {s['n']} | {s['mean_ms']:.2f} | {s['sd_ms']:.3f} | {s['p50_ms']:.2f} | "
              f"{s['p99_ms']:.2f} | {s['max_ms']:.2f} |" for s in a["sessions"]]
    lines += ["", "| Window | mean (ms) | power (W) | idle (W) | energy (J) | dynamic (J) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['window']} | {r['mean_ms']:.2f} | {r['power_w'] or float('nan'):.2f} | "
              f"{r['idle_power_w'] or float('nan'):.2f} | {r['energy_j'] or float('nan'):.3f} | "
              f"{r['dynamic_energy_j'] or float('nan'):.3f} |" for r in e["windows"]]
    if e["energy_j"]:
        lines.append(f"\nEnergy per part: {e['energy_j']['mean']:.3f} J, SD {e['energy_j']['sd'] or 0:.3f} J, "
                     f"u(mean) {e['energy_j']['u_mean'] or 0:.3f} J (n = {e['energy_j']['n']}); {e['note']}.")
    lines += ["", f"D3: {json.dumps(res['D3_scaling'])}"]
    (out_dir / "p47_u6d_summary.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (out_dir / "p47_u6d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


def selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tag = "p47_jetson"
        (root / "p43" / tag).mkdir(parents=True)
        (root / "p47" / tag).mkdir(parents=True)
        for i, m in enumerate((67.0, 67.2, 66.9), 1):
            (root / "p43" / tag / f"p43_repeat_s0{i}.json").write_text(json.dumps({"phases": {"A": {"e2e": {
                "n": 500, "mean_ms": m, "sd_ms": 0.1, "p50_ms": m, "p99_ms": m + 0.2, "max_ms": m + 0.5}}}}))
        lines = []
        for w in (1, 2):
            d = root / "p43" / f"{tag}_e{w}"
            d.mkdir()
            base = f"2026-10-03 18:0{w}:"
            d.joinpath("p43_idle.json").write_text(json.dumps({"idle": {"start": {"local": base + "00"},
                                                                      "end": {"local": base + "09"}}}))
            d.joinpath("p43_energy_fused_e2e.json").write_text(json.dumps({"start": {"local": base + "10"},
                                                                           "end": {"local": base + "19"},
                                                                           "mean_ms": 67.0, "forwards": 900}))
            for s in range(20):
                lines.append(f"10-03-2026 18:0{w}:{s:02d} RAM 1/2MB VDD_IN {6500 if s < 10 else 20500}mW/1mW")
        (root / "p47" / tag / "tegrastats.log").write_text("\n".join(lines) + "\n")
        (root / "p47" / tag / "p47_u6_scaling.json").write_text(json.dumps({
            "classes_e2e": {"2": {"p50_ms": 66.0}}, "head_vs_M": {"20": {"bank_rows": 11520, "fp32": {"p50_ms": 9.0},
                                                                          "fp16": {"p50_ms": 5.0}}},
            "tiles_per_part": {"1": {"p50_ms": 67.0}}}))
        res = analyze(tag, root)
        assert res["D1_sessions"]["k"] == 3 and abs(res["D1_sessions"]["grand_mean_ms"] - 67.0333) < 1e-3
        assert abs(res["D2_energy"]["energy_j"]["mean"] - 20.5 * 0.067) < 1e-9
        assert abs(res["D2_energy"]["dynamic_energy_j"]["mean"] - 14.0 * 0.067) < 1e-9
    print("[p47-u6d] selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="p47_jetson")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    return selftest() if a.selftest else (analyze(a.tag) and 0)


if __name__ == "__main__":
    sys.exit(main())
