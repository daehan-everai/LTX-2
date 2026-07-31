# Small-dataset I2V LoRA recipe

Use `configs/ltx2_i2v_lora_small_dataset.yaml` as the default profile for a
small, video-only dataset where every inference request supplies a first frame.

The profile uses rank/alpha 32, video attention projections only, learning rate
`1e-4`, AdamW, linear decay to `1e-5`, BF16, shifted-logit-normal timestep
sampling, and an effective batch of one. These settings preserve the official
LTX I2V starting point while avoiding the extra capacity of FFN adapters.

`first_frame_conditioning_p` is `1.0` because this profile is I2V-only. Set it
to `0.5` when the same adapter must retain mixed T2V/I2V behavior.

For a small dataset, treat 2,000 steps as a maximum rather than a guaranteed
optimum. Compare deterministic holdout loss and fixed-seed generations at step
0, step 1,000, and step 2,000, then select the checkpoint with the best
holdout/visual tradeoff.

## Why this is not the Sulphur rank

Sulphur 2's published adapter metadata reports rank/alpha 768, 118,036 training
items, and 2,200 completed steps. That adapter performs broad LTX-2.3 base
adaptation. It is not a suitable capacity default for a dataset with only a few
hundred clips. The overlapping, transferable choices are LR `1e-4`, AdamW,
BF16, no adapter dropout, gradient clipping at `1.0`, and shifted-logit-normal
timestep sampling.

References:

- LTX-2 paper: https://arxiv.org/abs/2601.03233
- Official I2V config: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/configs/i2v_lora.yaml
- Official configuration reference: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/configuration-reference.md
- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- Sulphur 2 model card: https://huggingface.co/SulphurAI/Sulphur-2-base
