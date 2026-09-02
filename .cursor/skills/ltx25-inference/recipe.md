# LTX-2.5 official HQ recipe

Copy `assets/run_official.py` from a canonical tree. Do not re-derive SGLang.

## Weights (`--model-root`)

| Relpath |
| --- |
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| `vae/ltx-2.5-video-vae-bf16.safetensors` |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` |

SFT LoRA (optional): `checkpoints/lora_weights_step_<column>.safetensors`. Baseline = no SFT LoRA (`loras=()`).

Prefer rsync from a live official-natten `/workspace/ltx-official/models` with `--bwlimit`. HF fallback: official LTX-2.5 pack + `EverAI-AI/LTX-2-LORA` for this run’s prefix.

## Knobs (do not change unless asked)

| Knob | Value |
| --- | --- |
| Pipeline | `TI2VidTwoStagesHQPipeline` |
| Size | 832×1216, 241 frames, 24 fps, ~10s |
| Seed | 42 |
| Image cond | first frame, strength 1.0 |
| Distilled LoRA | 1.0 on the module; **strength 0.5** both stages |
| SFT LoRA | 1.0 (omit on baseline) |
| Params | `LTX_2_3_HQ_PARAMS` + `STAGE_2_DISTILLED_SIGMAS` |
| Tiling | `AUTO_TILING` |
| DiffVAE | `DiffVAEMode.CHUNKED_EAGER` **with natten installed** |
| Audio | encode audio from the pipeline (probe must have an audio stream) |

## Required patches in the runner

1. `assert natten_available()` at start. Raise if false.
2. Unwrap `@inference_mode` on `TI2VidTwoStagesHQPipeline.__call__` and replace with `torch.no_grad()`. natten **0.21.5** autograd Function cannot consume inference-mode tensors (DiffVAE RMSNorm). Official HQ `__call__` is `@inference_mode` and the lazy decode iterator runs after that context exits.

```python
_hq_call = TI2VidTwoStagesHQPipeline.__call__
while hasattr(_hq_call, "__wrapped__"):
    _hq_call = _hq_call.__wrapped__
TI2VidTwoStagesHQPipeline.__call__ = torch.no_grad()(_hq_call)
```

3. Env: `LTX_OVERLAY_ROOT=/workspace/ltx-official/overlay`. Never `LTX_FORCE_EAGER_NA`.

## Probe

Valid mp4: video 832×1216, duration 9.6–10.6s, audio stream present. `job_id` in the manifest should match the review filename (`002.mp4`).

## Why not SGLang / eager

LTX-2.5 uses DiffVAE (`CausalDiffusionVAE`), not the 2.3 conv VAE. Without natten the host falls back Triton → eager SDPA. Triton NA hits CUDA grid-Y > 65535 on 241f / 1216-tall decode ([LTX-2#277](https://github.com/Lightricks/LTX-2/issues/277)). `LTX_FORCE_EAGER_NA=1` avoids the crash but paints regular horizontal blinds on LoRA outputs. Official natten does not.

## Reference bootstrap

Pin torch 2.9.1+cu130, `transformers==5.10.1`, editable overlay, `natten==0.21.5+torch290cu130`.
