#!/usr/bin/env bash
# P47 U6-D on the Jetson Orin Nano (docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, section U6-D). Report only.
#
# Preconditions, checked by the author's execution agent before starting (not changed by this script):
#   - power mode MAXN_SUPER with jetson_clocks applied (as in P43 / P45);
#   - drm_holder.py running (the script stops if it is not);
#   - no other GPU or heavy CPU load; headless (multi-user.target).
# Usage (one run; every output is write-once and the script stops at the first error):
#   R=<repo root on the device> MVTEC=<MVTec AD root> FUSED=<fused engine file> \
#   TRTPY=<tensorrt link directory> PY=<venv python> bash experiments/p47/run_jetson_u6.sh
# Duration: about 30-35 min (sessions ~14 min, energy windows ~9 min, scaling ~5 min).
set -euo pipefail
: "${R:?set R}" "${MVTEC:?set MVTEC}" "${FUSED:?set FUSED}" "${TRTPY:?set TRTPY}" "${PY:?set PY}"
cd "$R"
export PYTHONPATH="$TRTPY${PYTHONPATH:+:$PYTHONPATH}"
unset HF_ENDPOINT || true

TAG=p47_jetson
OUT="$R/ckpt/p47/$TAG"
if [ -e "$OUT/DONE" ] || [ -e "$R/ckpt/p43/$TAG" ]; then
  echo "P47 U6-D outputs already exist; this stage runs once" >&2
  exit 1
fi
mkdir -p "$OUT"
LOG="$OUT/run_log.txt"
exec > >(tee -a "$LOG") 2>&1

echo "[p47] start $(date '+%F %T')  commit $(git -C "$R" rev-parse HEAD 2>/dev/null || echo unknown)"
[ -f "$FUSED" ] || { echo "engine not found: $FUSED"; exit 1; }
pgrep -f drm_holder.py > /dev/null || { echo "drm_holder.py is not running: start it first (handoff section 7)"; exit 1; }
nvpmodel -q 2>&1 | sed 's/^/[nvpmodel] /' || true
for f in /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/{cur_freq,min_freq,max_freq}; do
  if [ -r "$f" ]; then echo "[gpu] $(basename "$f") $(cat "$f")"; fi
done

# 1. Between-session reproducibility of the deployed end-to-end latency: 10 fresh processes, P43 phase A
#    (50 warm-up calls, 500 end-to-end and 500 split timings each). Outputs: ckpt/p43/$TAG/p43_repeat_sNN.json
for i in 01 02 03 04 05 06 07 08 09 10; do
  "$PY" -m experiments.p43.edge_measure --repeat --tag "$TAG" --label "s$i" --phases A --n-phase 500 \
    --mvtec-root "$MVTEC" --fused "$FUSED"
done

# 2. Energy: five repeated windows (30 s idle + 60 s deployed end-to-end each, as P43-A), with VDD_IN logged every
#    100 ms. Outputs: ckpt/p43/${TAG}_eN/{p43_idle.json,p43_energy_fused_e2e.json} and $OUT/tegrastats.log
TS_LOG="$OUT/tegrastats.log"
tegrastats --interval 100 | "$PY" -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write(time.strftime("%m-%d-%Y %H:%M:%S ") + line); sys.stdout.flush()' > "$TS_LOG" &
TS_PID=$!
trap 'pkill -P $$ tegrastats 2>/dev/null || true; kill $TS_PID 2>/dev/null || true' EXIT
sleep 5
grep -q "VDD_IN" "$TS_LOG" || { echo "tegrastats gives no VDD_IN lines: stop"; exit 1; }
for w in 1 2 3 4 5; do
  "$PY" -m experiments.p43.edge_measure --energy --idle-seconds 30 --only fused_e2e --seconds 60 \
    --tag "${TAG}_e$w" --mvtec-root "$MVTEC" --fused "$FUSED"
done
pkill -P $$ tegrastats 2>/dev/null || true
kill $TS_PID 2>/dev/null || true

# 3. Scaling: classes 2-16 (engine), head against the bank size M (PyTorch), 1 / 4 / 9 tiles per part.
#    Output: $OUT/p47_u6_scaling.json
"$PY" -m experiments.p47.jetson_u6 --scaling --tag "$TAG" --mvtec-root "$MVTEC" --fused "$FUSED"

f=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq
if [ -r "$f" ]; then echo "[gpu] end cur_freq $(cat "$f")"; fi
date '+%F %T' > "$OUT/DONE"
echo "[p47] done $(date '+%F %T'). Copy back: ckpt/p43/$TAG ckpt/p43/${TAG}_e1..e5 ckpt/p47/$TAG"
