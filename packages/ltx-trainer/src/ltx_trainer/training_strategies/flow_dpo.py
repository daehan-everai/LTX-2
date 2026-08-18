"""Flow-DPO training strategy for rectified-flow video models.

Implements the Kling VideoReward / Flow-DPO objective (Gao et al. 2025,
arXiv:2501.13918) on top of LTX's existing velocity parameterization:

    noisy = (1 - t) * x + t * noise
    v*    = noise - x
    loss  = -log sigmoid(-0.5 * beta * (w_diff - l_diff))

Chosen and rejected latents share the same timestep and the same noise.
``beta`` is a constant (not ``beta * (1-t)^2``); the paper found the
timestep-dependent KL term caused reward hacking on flow models.

Per-sample velocity errors use the same masked mean as SFT, not an
unnormalized Frobenius norm. Video tokens would otherwise saturate the
sigmoid for any practical ``beta``.
"""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

KLING_FLOW_DPO_BETA = 2.69897000434


class FlowDPOConfig(TrainingStrategyConfigBase):
    """Configuration for Flow-DPO (chosen / rejected pair) training."""

    name: Literal["flow_dpo"] = "flow_dpo"

    first_frame_conditioning_p: float = Field(
        default=1.0,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Audio DPO is not implemented. Must stay false.",
    )

    beta: float = Field(
        default=KLING_FLOW_DPO_BETA,
        description="Constant KL coefficient from Kling Flow-DPO. Do not schedule by timestep.",
        gt=0.0,
    )

    chosen_latents_dir: str = Field(
        default="chosen_latents",
        description="Directory name under the preprocessed root for preferred-video latents",
    )

    rejected_latents_dir: str = Field(
        default="rejected_latents",
        description="Directory name under the preprocessed root for non-preferred-video latents",
    )


class FlowDPOStrategy(TrainingStrategy):
    """Pairwise Flow-DPO over I2V (or T2V) precomputed latents."""

    config: FlowDPOConfig

    def __init__(self, config: FlowDPOConfig):
        super().__init__(config)
        if config.with_audio:
            raise ValueError("flow_dpo does not support audio. Set with_audio=false.")

    def get_data_sources(self) -> dict[str, str]:
        return {
            self.config.chosen_latents_dir: "chosen_latents",
            self.config.rejected_latents_dir: "rejected_latents",
            "conditions": "conditions",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        chosen_latents, chosen_meta = self._load_video_latents(batch["chosen_latents"])
        rejected_latents, rejected_meta = self._load_video_latents(batch["rejected_latents"])
        if chosen_latents.shape != rejected_latents.shape:
            raise ValueError(
                "Chosen and rejected latents must share shape for Flow-DPO "
                f"(got {tuple(chosen_latents.shape)} vs {tuple(rejected_latents.shape)})"
            )

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size, seq_len, _ = chosen_latents.shape
        device = chosen_latents.device
        height = chosen_meta["height"]
        width = chosen_meta["width"]
        num_frames = chosen_meta["num_frames"]
        fps = chosen_meta["fps"]

        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        sigmas = timestep_sampler.sample_for(chosen_latents)
        noise = torch.randn_like(chosen_latents)

        noisy_w, targets_w, timesteps_w = self._add_flow_noise(
            chosen_latents, noise, sigmas, conditioning_mask
        )
        noisy_l, targets_l, timesteps_l = self._add_flow_noise(
            rejected_latents, noise, sigmas, conditioning_mask
        )

        positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=torch.float32,
        )

        chosen_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_w,
            timesteps=timesteps_w,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )
        rejected_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_l,
            timesteps=timesteps_l,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        video_loss_mask = (~conditioning_mask).float()
        return ModelInputs(
            video=chosen_modality,
            audio=None,
            video_targets=targets_w,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            rejected_video=rejected_modality,
            rejected_video_targets=targets_l,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        raise RuntimeError("flow_dpo uses compute_dpo_loss; the trainer must call that path")

    def compute_dpo_loss(
        self,
        policy_w: Tensor,
        policy_l: Tensor,
        ref_w: Tensor,
        ref_l: Tensor,
        inputs: ModelInputs,
    ) -> tuple[Tensor, dict[str, float]]:
        """Return per-sample Flow-DPO loss [B,] and scalar logs."""
        if inputs.rejected_video_targets is None:
            raise ValueError("rejected_video_targets is required for Flow-DPO")

        model_w_err = masked_mse(policy_w, inputs.video_targets, inputs.video_loss_mask)
        model_l_err = masked_mse(policy_l, inputs.rejected_video_targets, inputs.video_loss_mask)
        ref_w_err = masked_mse(ref_w, inputs.video_targets, inputs.video_loss_mask)
        ref_l_err = masked_mse(ref_l, inputs.rejected_video_targets, inputs.video_loss_mask)

        w_diff = model_w_err - ref_w_err
        l_diff = model_l_err - ref_l_err
        inside = -0.5 * self.config.beta * (w_diff - l_diff)
        loss = F.softplus(-inside)  # -log sigmoid(inside)

        if inputs.sigma_loss_weights is not None:
            loss = loss * inputs.sigma_loss_weights

        with torch.no_grad():
            metrics = {
                "train/dpo_accuracy": (w_diff < l_diff).float().mean().item(),
                "train/dpo_margin": (l_diff - w_diff).mean().item(),
                "train/chosen_err": model_w_err.mean().item(),
                "train/rejected_err": model_l_err.mean().item(),
                "train/ref_chosen_err": ref_w_err.mean().item(),
                "train/ref_rejected_err": ref_l_err.mean().item(),
                "train/dpo_beta": float(self.config.beta),
            }
        return loss, metrics

    def _load_video_latents(self, latent_batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        video_latents = latent_batch["latents"]
        _, _, num_frames, height, width = video_latents.shape
        video_latents = self._video_patchifier.patchify(video_latents)

        fps = latent_batch.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                "Different FPS values found in the batch. Found: %s, using the first one: %s",
                fps.tolist(),
                fps[0].item(),
            )
        fps_value = fps[0].item() if fps is not None else DEFAULT_FPS
        return video_latents, {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": fps_value,
        }

    def _add_flow_noise(
        self,
        latents: Tensor,
        noise: Tensor,
        sigmas: Tensor,
        conditioning_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy = (1 - sigmas_expanded) * latents + sigmas_expanded * noise
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)
        noisy = torch.where(conditioning_mask_expanded, latents, noisy)
        targets = noise - latents
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())
        return noisy, targets, timesteps


def masked_mse(pred: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
    """Per-sample masked MSE over sequence and channels. Returns [B]."""
    video_loss = (pred - target).pow(2)
    mask = loss_mask.unsqueeze(-1).float()
    video_loss = video_loss.mul(mask).mean(dim=[-2, -1])
    return video_loss.div(mask.mean(dim=[-2, -1]).clamp(min=1e-8))


def flow_dpo_loss_from_errors(
    model_w_err: Tensor,
    model_l_err: Tensor,
    ref_w_err: Tensor,
    ref_l_err: Tensor,
    beta: float = KLING_FLOW_DPO_BETA,
) -> Tensor:
    """Kling Flow-DPO scalar (or per-sample) loss from already-reduced errors."""
    w_diff = model_w_err - ref_w_err
    l_diff = model_l_err - ref_l_err
    inside = -0.5 * beta * (w_diff - l_diff)
    return F.softplus(-inside)
