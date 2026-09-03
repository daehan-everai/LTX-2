#!/usr/bin/env bash
# Official LTX-2.5 HQ + natten eval bootstrap. Overlay must already be on the pod.
set -euo pipefail
: "${HF_TOKEN_FILE:?HF_TOKEN_FILE must point to a mode-600 token file}"
HF_TOKEN=$(<"$HF_TOKEN_FILE")
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
rm -f "$HF_TOKEN_FILE"
unset HF_TOKEN_FILE

root=${LTX_EVAL_ROOT:-/workspace/ltx-official}
assets=${LTX_EVAL_ASSETS:-/workspace/assets/ltx25-eval}
venv=${LTX_EVAL_VENV:-/opt/ltx-venv}
mkdir -p "$root"/{logs,status,models,outputs,input} /workspace/hf_home /opt/uv-cache
exec > >(tee -a "$root/logs/bootstrap.log") 2>&1
trap 'status=$?; if [[ $status -ne 0 ]]; then date -u +%FT%TZ >"$root/status/bootstrap.failed"; fi' EXIT

if [[ ! -d "$root/overlay/ltx-core" || ! -d "$root/overlay/ltx-pipelines" ]]; then
  echo "missing overlay at $root/overlay/{ltx-core,ltx-pipelines}" >&2
  exit 1
fi

cat > /workspace/env.sh <<'ENV'
export HF_HOME=/workspace/hf_home
export LTX_OVERLAY_ROOT=/workspace/ltx-official/overlay
export PYTHONUNBUFFERED=1
ENV
# shellcheck disable=SC1091
source /workspace/env.sh

python3 -m pip install --break-system-packages --disable-pip-version-check uv
export UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=copy
if [[ ! -d "$venv" ]]; then
  uv venv "$venv" --python python3 --system-site-packages
fi
# Image torch is cu130; H200s with driver 12.8 need cu128 before any CUDA call.
uv pip install --python "$venv/bin/python" \
  torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$venv/bin/python" - <<'PY'
import torch
print("VENV_TORCH", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.cuda.is_available()
PY
uv pip install --python "$venv/bin/python" \
  einops numpy av 'transformers==5.10.1' accelerate safetensors scipy pillow imageio imageio-ffmpeg \
  soundfile huggingface_hub hf_transfer ninja tqdm cloudpickle
uv pip install --python "$venv/bin/python" --no-deps \
  -e "$root/overlay/ltx-core" \
  -e "$root/overlay/ltx-pipelines"
if ! uv pip install --python "$venv/bin/python" \
  'natten==0.21.5+torch290cu128' \
  -f https://whl.natten.org; then
  uv pip install --python "$venv/bin/python" \
    'natten==0.21.5+torch290cu130' \
    -f https://whl.natten.org
fi

"$venv/bin/python" - <<'PY'
import natten, torch
from ltx_core.model.video_vae.transformer.attention import natten_available
assert natten_available(), "natten_available() is False"
print("NATTEN_OK", natten.__version__, torch.__version__, torch.version.cuda)
PY

"$venv/bin/python" - <<'PY'
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
assert "gemma4_unified" in CONFIG_MAPPING, sorted(k for k in CONFIG_MAPPING if "gemma" in k)
print("TRANSFORMERS_GEMMA4_OK", transformers.__version__)
PY

if [[ -n "${LTX_EVAL_HF_PREFIX:-}" ]]; then
  : "${CHECKPOINT_STEPS:=${CHECKPOINT_STEP:-}}"
  if [[ -z "$CHECKPOINT_STEPS" ]]; then
    echo "LTX_EVAL_HF_PREFIX set but CHECKPOINT_STEPS is empty" >&2
    exit 1
  fi
  export CHECKPOINT_STEPS LTX_EVAL_HF_PREFIX
  mkdir -p "$root/models/checkpoints"
  "$venv/bin/python" - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

prefix = os.environ["LTX_EVAL_HF_PREFIX"].strip("/")
for column in os.environ["CHECKPOINT_STEPS"].split():
    dest = Path(f"/workspace/ltx-official/models/checkpoints/lora_weights_step_{column}.safetensors")
    if dest.is_file() and dest.stat().st_size > 1_000_000_000:
        print("LORA_OK_EXISTING", dest, dest.stat().st_size)
        continue
    last_err = None
    path = None
    for name in (
        f"{prefix}/checkpoints/lora_weights_step_{column}.safetensors",
        f"{prefix}/checkpoints/lora_weights_step_{int(column)}.safetensors",
    ):
        try:
            path = hf_hub_download(
                repo_id="EverAI-AI/LTX-2-LORA",
                filename=name,
                local_dir=str(dest.parent / "_hf"),
                token=os.environ["HF_TOKEN"],
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if path is None:
        raise RuntimeError(f"missing LoRA for {column}: {last_err}")
    shutil.copy2(path, dest)
    print("LORA_OK", dest, dest.stat().st_size)
PY
fi

if [[ -d "$assets/input" ]]; then
  rsync -rlt --no-owner --no-group --no-perms "$assets/input/" "$root/input/"
fi
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
date -u +%FT%TZ > "$root/status/bootstrap.complete"
trap - EXIT
