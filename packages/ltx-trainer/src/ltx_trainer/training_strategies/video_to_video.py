"""Video-to-video training strategy for IC-LoRA.
This strategy implements training with reference video conditioning where:
- Reference latents (clean) are concatenated with target latents (noised)
- Video coordinates handle both reference and target sequences
- Loss is computed only on the target portion
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


class VideoToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for video-to-video (IC-LoRA) training strategy."""

    name: Literal["video_to_video"] = "video_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for latents of reference videos",
    )

    temporal_boundary_loss_weight: float = Field(
        default=1.0,
        description=(
            "Loss weight multiplier applied to the first N non-conditioning frames immediately "
            "following the conditioned first frame of the target video. Values > 1.0 increase "
            "gradient signal at the temporal boundary, strengthening the model's use of the "
            "conditioning anchor. Only active when first_frame_conditioning_p > 0. "
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


class VideoToVideoStrategy(TrainingStrategy):
    """Video-to-video training strategy for IC-LoRA.
    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates handle both reference and target sequences
    - Loss is computed only on the target portion
    Attributes:
        reference_downscale_factor: The inferred downscale factor of reference videos.
            This is computed from the first batch and cached for metadata export.
    """

    config: VideoToVideoConfig
    reference_downscale_factor: int | None

    def __init__(self, config: VideoToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Video-to-video configuration
        """
        super().__init__(config)
        self.reference_downscale_factor = None  # Will be inferred from first batch

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents."""
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for IC-LoRA training with reference videos."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Get video (latent) dimensions directly from the tensor shapes [B, C, F, H, W].
        _, _, num_frames, height, width = target_latents.shape
        _, _, ref_frames, ref_height, ref_width = ref_latents.shape

        # Infer reference downscale factor from dimension ratios
        # This allows training with downscaled reference videos for efficiency
        reference_downscale_factor = self._infer_reference_downscale_factor(
            target_height=height,
            target_width=width,
            ref_height=ref_height,
            ref_width=ref_width,
        )

        # Cache the scale factor for metadata export (only on first batch)
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = reference_downscale_factor
        elif self.reference_downscale_factor != reference_downscale_factor:
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={reference_downscale_factor}. "
                f"All training samples must use the same reference/target resolution ratio."
            )

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        target_latents = self._video_patchifier.patchify(target_latents)
        ref_latents = self._video_patchifier.patchify(ref_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        # Video-to-video uses only video embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]
        device = target_latents.device

        # Create conditioning mask
        # Reference tokens are always conditioning (timestep=0)
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Combined conditioning mask
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Sample noise and sigmas for target
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Apply noise to target
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise

        # For first frame conditioning in target, use clean latents
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        # Targets for loss computation
        targets = noise - target_latents

        # Concatenate reference (clean) and target (noisy)
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        # Create per-token timesteps
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        # Generate positions for reference and target separately, then concatenate
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=torch.float32,
        )

        # Scale reference positions to match target coordinate space
        # This maps ref positions from (0, ref_H, ref_W) to (0, target_H, target_W)
        # Position tensor shape: [B, 3, seq_len, 2] where dim 1 is (time, height, width)
        if reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= reference_downscale_factor  # height axis
            ref_positions[:, 2, ...] *= reference_downscale_factor  # width axis
            # Time axis (index 0) remains unchanged

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=torch.float32,
        )

        # Concatenate positions along sequence dimension
        positions = torch.cat([ref_positions, target_positions], dim=2)

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss mask: float weights (0 = excluded, 1 = normal, >1 = boosted).
        # Reference tokens: 0 (no loss). Target tokens: 1 where not conditioning.
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.float32, device=device)
        target_loss_mask = (~target_conditioning_mask).float()
        if self.config.temporal_boundary_loss_weight > 1.0:
            frame_size = height * width
            boundary_end = frame_size + self.config.temporal_boundary_frames * frame_size
            # Per-sample: whether the first (conditioning) frame of the target is active. Shape [B, 1].
            cond_active = target_conditioning_mask[:, :frame_size].any(dim=1, keepdim=True).float()
            # Boundary region indicator over target-local sequence positions. Shape [1, target_seq_len].
            positions_idx = torch.arange(target_seq_len, device=device)
            boundary_region = ((positions_idx >= frame_size) & (positions_idx < boundary_end)).unsqueeze(0).float()
            # weight = 1 everywhere, boosted to temporal_boundary_loss_weight inside the
            # boundary region for samples whose first target frame is a conditioning frame.
            extra = (self.config.temporal_boundary_loss_weight - 1.0) * cond_active * boundary_region
            target_loss_mask *= 1.0 + extra
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked loss on target portion only. Returns per-sample loss [B,]."""
        # Extract target portion of prediction (skip prepended reference tokens)
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]

        # Get target portion of loss mask
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        # Masked MSE, normalized per-sample over (seq, channels) → [B,]
        loss = (target_pred - inputs.video_targets).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        loss = loss.mul(loss_mask).mean(dim=[-2, -1])
        loss = loss.div(loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8))

        # Apply per-sample sigma loss weights (e.g. bell weighting) if configured
        if inputs.sigma_loss_weights is not None:
            loss = loss * inputs.sigma_loss_weights

        return loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata for checkpoint files."""
        metadata: dict[str, Any] = {}
        # Always include reference_downscale_factor for IC-LoRAs so inference
        # pipelines know the expected scale factor for reference videos.
        if self.reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self.reference_downscale_factor
        return metadata

    @staticmethod
    def _infer_reference_downscale_factor(
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> int:
        """Infer the reference downscale factor from target and reference dimensions."""
        # If dimensions match, no scaling needed
        if target_height == ref_height and target_width == ref_width:
            return 1

        # Calculate scale factors for each dimension
        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
                f"of reference dimensions ({ref_height}x{ref_width})"
            )

        scale_h = target_height // ref_height
        scale_w = target_width // ref_width

        if scale_h != scale_w:
            raise ValueError(
                f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}. "
                f"Target: {target_height}x{target_width}, Reference: {ref_height}x{ref_width}"
            )

        if scale_h < 1:
            raise ValueError(
                f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
                f"target dimensions ({target_height}x{target_width})"
            )

        return scale_h
