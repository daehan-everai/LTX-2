#!/usr/bin/env bash
# Run one LoRA column with the official HQ + natten eval runner.
set -euo pipefail
column=${CHECKPOINT_STEP:?CHECKPOINT_STEP e.g. 09135}
root=${LTX_EVAL_ROOT:-/workspace/ltx-official}
assets=${LTX_EVAL_ASSETS:-/workspace/assets/ltx25-eval}
venv=${LTX_EVAL_VENV:-/opt/ltx-venv}
# shellcheck disable=SC1091
source /workspace/env.sh
mkdir -p "$root/outputs/$column" "$root/status" "$root/logs"
exec > >(tee -a "$root/logs/run-$column.log") 2>&1
until [[ -s "$root/status/bootstrap.complete" && -s "$root/input/manifest.jsonl" ]]; do
  if [[ -s "$root/status/bootstrap.failed" ]]; then
    echo "bootstrap failed" >&2
    exit 1
  fi
  sleep 10
done
if [[ -f "$root/status/weights.complete" ]]; then
  :
elif [[ ! -s "$root/models/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" ]]; then
  echo "waiting for weights at $root/models" >&2
  until [[ -s "$root/status/weights.complete" ]]; do
    sleep 10
  done
fi
ckpt="$root/models/checkpoints/lora_weights_step_${column}.safetensors"
if [[ ! -s "$ckpt" ]]; then
  echo "missing checkpoint $ckpt" >&2
  date -u +%FT%TZ >"$root/status/run.$column.failed"
  exit 1
fi
date -u +%FT%TZ >"$root/status/run.$column.running"
rm -f "$root/status/run.$column.complete" "$root/status/run.$column.failed"
trap 'status=$?; if [[ $status -ne 0 ]]; then date -u +%FT%TZ >"$root/status/run.'"$column"'.failed"; fi' EXIT
export PATH="$venv/bin:$PATH"
PYTHONPATH="$assets${PYTHONPATH:+:$PYTHONPATH}" \
  "$venv/bin/python" "$assets/run_official.py" \
  --model-root "$root/models" \
  --checkpoint "$ckpt" \
  --column "$column" \
  --manifest "$root/input/manifest.jsonl" \
  --output "$root/outputs/$column"
date -u +%FT%TZ >"$root/status/run.$column.complete"
trap - EXIT
