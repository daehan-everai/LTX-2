# LTX-2.5 official HQ + natten eval

Canonical EverAI eval stack for LTX-2.5 LoRA columns.

- Pipeline: `ltx_pipelines.ti2vid_two_stages_hq.TI2VidTwoStagesHQPipeline`
- Decode: natten 0.21.5 (refuse if `natten_available()` is false)
- Do **not** use SGLang, ComfyParity, Triton NA, or `LTX_FORCE_EAGER_NA`

Agent recipe: [../../.cursor/skills/ltx25-inference/SKILL.md](../../.cursor/skills/ltx25-inference/SKILL.md).
Upstream pin: [SOURCE.md](SOURCE.md).

## Layout

```
eval/ltx25-hq-natten/
  overlay/ltx-core/          # Lightricks 2.5, pinned
  overlay/ltx-pipelines/     # includes TI2VidTwoStagesHQPipeline
  run_official.py            # 832×1216, 241f, distilled 0.5, SFT LoRA 1.0
  run_common.py
  bootstrap.sh
  run_step.sh
```

Repo-root `packages/` stays LTX-2.3 for the trainer fork. Eval rsyncs **this**
overlay onto the pod, not `packages/`.

## Pod overlay

```bash
rsync -a --delete --exclude '.git' --exclude '__pycache__' \
  eval/ltx25-hq-natten/overlay/ltx-core/     /workspace/ltx-official/overlay/ltx-core/
rsync -a --delete --exclude '.git' --exclude '__pycache__' \
  eval/ltx25-hq-natten/overlay/ltx-pipelines/ /workspace/ltx-official/overlay/ltx-pipelines/
```

`LTX_OVERLAY_ROOT=/workspace/ltx-official/overlay`. Never export `LTX_FORCE_EAGER_NA`.
