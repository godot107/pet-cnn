#!/usr/bin/env bash
# The from-scratch CNN, reserved for the GPU: refuses to run on CPU (train --device cuda).
# All 60 epochs of the cosine schedule, keeping the best-val epoch (--patience 0). Patience 5 stopped
# unseen-breeds at epoch 14 on 2026-10-21, mid-schedule, while val loss still swung. ~11 s/epoch on a GTX 1660.
# Usage: scripts/train_scratch_gpu.sh [scheme ...]   (default: official unseen-breeds leaky-2023)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "nvidia-smi failed: reboot to fix a driver/library mismatch, then rerun." >&2; exit 1; }
for scheme in "${@:-official unseen-breeds leaky-2023}"; do
  for s in $scheme; do
    $PY -m petcnn train --arch smallcnn --scheme "$s" --device cuda --epochs 60 --patience 0 --name "smallcnn-gpu-$s"
    $PY -m petcnn eval --name "smallcnn-gpu-$s"
  done
done
$PY -m petcnn report
