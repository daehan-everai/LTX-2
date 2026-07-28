"""Text-to-video training strategy.
This strategy implements standard text-to-video generation training where:
- Only target latents are used (no reference videos)
- Standard noise application and loss computation
- Supports first frame conditioning
- Optionally supports joint audio-video training
"""

from typing import Any, Literal

import torch
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


class TextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-video training strategy."""

    name: Literal["text_to_video"] = "text_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    h_flip: bool = Field(
        default=False,
        description="Whether to apply random horizontal flip augmentation during training "
        "(50% chance per sample). Requires the dataset to be preprocessed with --with-h-flip.",
    )

    temporal_boundary_loss_weight: float = Field(
        default=1.0,
        description=(
            "Loss weight multiplier applied to the first N non-conditioning frames immediately "
            "following the conditioned first frame. Values > 1.0 increase gradient signal at the "
            "temporal boundary, strengthening the model's use of the conditioning anchor. "
            "Only active when first_frame_conditioning_p > 0. "
            "Recommended range: 1.5–3.0. Default 1.0 (disabled)."
        ),
        ge=1.0,
    )

    temporal_boundary_frames: int = Field(
        default=1,
        description=(
            "Number of non-conditioning latent frames to apply temporal_boundary_loss_weight to. "
            "Only active when temporal_boundary_loss_weight > 1.0."
        ),
        ge=1,
    )

    caption_dropout_p: float = Field(
        default=0.0,
        description=(
            "Probability of replacing a sample's caption with the empty prompt during training "
            "(classifier-free guidance dropout). Keeps the unconditional branch meaningful so "
            "CFG at inference (guidance_scale > 1) stays effective as the LoRA strengthens. "
            "Typical range 0.05-0.1. Default 0.0 (disabled)."
        ),
        ge=0.0,
        le=1.0,
    )


class TextToVideoStrategy(TrainingStrategy):
    """Text-to-video training strategy.
    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Supports first frame conditioning
    - Optionally supports joint audio-video training when with_audio=True
    """

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Text-to-video configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        Text-to-video training requires latents and text conditions.
        When with_audio is True, also requires audio latents.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        return sources

    def get_optional_data_sources(self) -> set[str]:
        return {"audio_latents"} if self.config.with_audio else set()

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]

        # Audio-only batch: no video latents present (has_video=False for all samples).
        # The video branch is fed a neutral single-token dummy (no loss); only audio is trained.
        has_video_flag = latents.get("has_video")
        if has_video_flag is not None and not bool(has_video_flag.any()):
            return self._prepare_audio_only_inputs(batch, timestep_sampler)

        video_latents = latents["latents"]

        # Get video (latent) dimensions directly from the tensor shape [B, C, F, H, W].
        _, _, num_frames, height, width = video_latents.shape

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS with backward compatibility
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps (conditioning tokens get timestep 0)
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Generate video positions using ltx_core's native implementation
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=torch.float32,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Video loss mask: float weights for loss computation (0 = excluded, 1 = normal, >1 = boosted).
        video_loss_mask = (~video_conditioning_mask).float()
        if self.config.temporal_boundary_loss_weight > 1.0:
            frame_size = height * width
            boundary_end = frame_size + self.config.temporal_boundary_frames * frame_size
            # Per-sample: whether the first (conditioning) frame is active. Shape [B, 1].
            cond_active = video_conditioning_mask[:, :frame_size].any(dim=1, keepdim=True).float()
            # Boundary region indicator over sequence positions. Shape [1, seq_len].
            positions = torch.arange(video_seq_len, device=device)
            boundary_region = ((positions >= frame_size) & (positions < boundary_end)).unsqueeze(0).float()
            # weight = 1 everywhere, boosted to temporal_boundary_loss_weight inside the
            # boundary region for samples whose first frame is a conditioning frame.
            extra = (self.config.temporal_boundary_loss_weight - 1.0) * cond_active * boundary_region
            video_loss_mask *= 1.0 + extra

        # Guard mixed batches (batch_size > 1) that contain audio-only samples: their
        # zero-padded video latents must not contribute to the video loss.
        if has_video_flag is not None:
            has_video = has_video_flag.to(device=device)
            if not bool(has_video.all()):
                video_loss_mask = video_loss_mask * has_video.view(-1, 1).float()

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_audio_only_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for an audio-only batch (no video latents).

        The video branch receives a neutral single-token dummy latent so that the
        audio branch can still run video-to-audio cross-attention against clean
        context. The video loss is fully masked out; only the audio branch is trained.
        """
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        audio_latents_raw = batch["audio_latents"]["latents"]
        device = audio_latents_raw.device
        dtype = audio_latents_raw.dtype
        batch_size = audio_latents_raw.shape[0]

        # Sample sigmas based on the audio sequence length (drives the timestep distribution).
        audio_latents_patched = self._audio_patchifier.patchify(audio_latents_raw.to(device=device, dtype=dtype))
        sigmas = timestep_sampler.sample_for(audio_latents_patched)

        # Neutral single-token dummy video as clean-ish cross-attention context (excluded from loss).
        video_seq_len = 1
        dummy_video_latents = torch.zeros(batch_size, video_seq_len, 128, device=device, dtype=dtype)
        video_noise = torch.randn_like(dummy_video_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * dummy_video_latents + sigmas_expanded * video_noise
        video_targets = video_noise - dummy_video_latents
        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_positions = self._get_video_positions(
            num_frames=1,
            height=1,
            width=1,
            batch_size=batch_size,
            fps=DEFAULT_FPS,
            device=device,
            dtype=torch.float32,
        )

        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Fully masked video loss: audio-only samples do not train the video branch.
        video_loss_mask = torch.zeros(batch_size, video_seq_len, device=device)

        audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
            batch=batch,
            sigmas=sigmas,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint audio-video training.
        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        audio_data = batch["audio_latents"]
        has_audio = audio_data.get("has_audio", torch.ones(batch_size, dtype=torch.bool))
        has_audio = has_audio.to(device)

        if has_audio.any():
            audio_latents = audio_data["latents"].to(device=device, dtype=dtype)
            audio_latents = self._audio_patchifier.patchify(audio_latents)
            audio_seq_len = audio_latents.shape[1]
        else:
            audio_seq_len = 1
            audio_latents = torch.zeros(batch_size, audio_seq_len, 128, device=device, dtype=dtype)

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio (same sigma as video)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Compute audio targets
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma (no conditioning mask)
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: True = compute loss. False for video-only samples.
        audio_loss_mask = has_audio.unsqueeze(1).expand(-1, audio_seq_len)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and optionally audio."""
        # Video loss: normalize per-sample over (seq, channels) → [B,]
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).mean(dim=[-2, -1])
        video_loss = video_loss.div(video_loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8))

        # Apply per-sample sigma loss weights (e.g. bell weighting) if configured
        if inputs.sigma_loss_weights is not None:
            video_loss = video_loss * inputs.sigma_loss_weights

        # If no audio, return per-sample video loss [B,]
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        # Audio loss per-sample [B,], zeroed for video-only samples.
        audio_loss_mask = inputs.audio_loss_mask.unsqueeze(-1).float()
        audio_loss = (audio_pred - inputs.audio_targets).pow(2)
        audio_loss = audio_loss.mul(audio_loss_mask).mean(dim=[-2, -1])

        mask_mean = audio_loss_mask.mean(dim=[-2, -1])
        audio_loss = torch.where(
            mask_mean > 0,
            audio_loss / mask_mean.clamp(min=1e-8),
            audio_loss * 0.0,
        )

        # When the batch carries no video training signal (audio-only), audio is the
        # primary objective and should not be down-weighted. Otherwise keep the joint
        # AV balance where audio is a secondary objective.
        audio_coeff = 0.1 if bool(inputs.video_loss_mask.any()) else 1.0

        return video_loss + audio_coeff * audio_loss
