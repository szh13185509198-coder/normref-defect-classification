#!/usr/bin/env bash
set -uo pipefail
unset HF_ENDPOINT
cd $HOME/p31/repo
F=ckpt/revision_c/c3_resource/c3_latency_ours_cuda.json
for p in 1 2; do
  [ -e $F ] && { echo "$F exists before pass $p: stop"; exit 1; }
  $HOME/p31/venv/bin/python -u -m experiments.revision.c3_resource_profile latency --pipeline ours --device cuda > ckpt/p49/pass${p}_stdout.log 2>&1 || { echo "pass $p failed"; exit 1; }
  mv $F ckpt/p49/pass${p}_c3_latency_ours_cuda.json
  echo "pass $p done $(date +%T)"
done
echo ALL_DONE
