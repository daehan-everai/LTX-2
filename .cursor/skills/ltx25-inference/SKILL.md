---
name: ltx25-inference
description: >-
  Run official LTX-2.5 I2V inference with TI2VidTwoStagesHQ + natten (no SGLang).
  Use when the user asks to infer, generate, or review LTX 2.5 / LTX-2.5 clips,
  publish AIVI-1358 columns, debug DiffVAE banding, Triton NA, eager NA, or
  LTX_FORCE_EAGER_NA, or stand up a 2.5 infer pod.
---

# LTX-2.5 inference (official + natten)

**Only** `ltx_pipelines.ti2vid_two_stages_hq.TI2VidTwoStagesHQPipeline` + **natten**.
Do **not** use SGLang, ComfyParity, Triton NA, or eager SDPA for LTX-2.5.

Weights, knobs, probe, banding: [recipe.md](recipe.md).

Canonical EverAI trees (clone; do not invent a new stack):

- `training/ltx-official-aivi1358-of-sft-ltx25-natten30-20260827`
- `training/ltx-comfy-aivi1358-of-sft-ltx25-v10-20260827` (same runner; multi-column)

## Hard rules

- **Refuse to run** if `natten_available()` is false. Abort; do not fall back.
- **Never set** `LTX_FORCE_EAGER_NA`. Eager/Triton DiffVAE decode produces horizontal venetian-blind banding on LoRA columns. Baseline can look fine; LoRA will not.
- **Do not use** `run_prod.py` / `bootstrap_prod.sh` / `sglang-port` for 2.5.
- **Same caption** on baseline and every LoRA column. Do not silently swap the eval manifest.
- **Do not overwrite** another agent's review clips or outputs.
- **Do not stop/delete** pods you did not create.
- **Spend cap** is $80/hr on this RunPod account. Count livecall + other agents before creating H200s.
- **Secrets:** GCP Secret Manager `runpod_key` / `hf_token` via the usual wrapper. Never print tokens.

## Host

| Knob | Value |
| --- | --- |
| Image | `runpod/pytorch:1.0.6-cu1300-torch291-ubuntu2404` |
| Torch | image `2.9.1+cu130` (do not let pip pull 2.13) |
| natten | `natten==0.21.5+torch290cu130` from `https://whl.natten.org` |
| transformers | `5.10.1` (need `gemma4_unified`) |
| Overlay | LTX-2.5 `packages/{ltx-core,ltx-pipelines}` |
| GPU | H200, else H100 80GB HBM3 / H100 SXM. Volume 360GB, disk 80GB, port `22/tcp` |
| Driver | Need CUDA 13. Working: H100 **580.126.09**, H200 **570.195.03**. **570.124.06** (CUDA 12.8) cannot load this torch |

`uv venv --system-site-packages`. Overlay extra `natten` is empty / pins cu132 — do not use it.

After bootstrap, require log lines `NATTEN_OK` and `TRANSFORMERS_GEMMA4_OK`.

## Workflow

```
- [ ] Confirm eval manifest, columns (baseline / LoRA steps), review split, row count
- [ ] Clone official-natten tree; do not copy SGLang bootstrap
- [ ] Create GPU only if spend cap allows; distinct pod name
- [ ] Rsync LTX-2.5 weights from an existing official-natten volume (bwlimit). Do not re-download if a live pod already has them
- [ ] Bootstrap; assert natten_available(); never export LTX_FORCE_EAGER_NA
- [ ] Generate; probe 832×1216 / ~10s / audio
- [ ] Banding check on first generated frame of LoRA + baseline before publish
- [ ] Publish review; keep SFW vs SFW+NSFW apps separate if that is the eval
- [ ] Monitor until columns finish unless told to stop
- [ ] Stop/delete only this run’s infer pods
```

### Generate

See [recipe.md](recipe.md). ~2–2.5 min/clip on H100/H200 for 241f HQ. First 30 rewrite-v4 rows unless the user asks for more.

### Review

Reuse the job’s `build_review_site.py` / `publish_review.sh`. Caption only while a row is playing (no hover caption).

### Banding check

Before calling a column good, extract `ffmpeg -ss 0.04` and `-ss 3` from at least two LoRA clips and one baseline clip. Venetian blinds = wrong NA backend. Do not publish that column.
