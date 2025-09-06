#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

OUTDIR="test_results"
MODELDIR="src/models"
DATAROOT="datasets/vudenc/prepared"

# write per-task threshold JSONs the evaluator reads
declare -A THR=(
  [sql]=0.5
  [command_injection]=0.5
  [remote_code_execution]=0.2
  [path_disclosure]=0.4
  [xss]=0.3
  [xsrf]=0.5
  [open_redirect]=0.4
)
mkdir -p "$MODELDIR"
for v in "${!THR[@]}"; do
  printf '{"threshold": %.2f}\n' "${THR[$v]}" > "$MODELDIR/val_best_threshold_${v}.json"
done

echo "[*] Running batch eval with per-task thresholds…"
poetry run python -m models.eval_all \
  --models-dir "$MODELDIR" \
  --datasets-root "$DATAROOT" \
  --outdir "$OUTDIR" \
  --hf_model_name "microsoft/graphcodebert-base" \
  --threshold-json "$MODELDIR/val_best_threshold_{vuln}.json"
