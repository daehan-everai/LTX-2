"""Stitch the audio and video halves of two AV LoRA checkpoints into one.

LTX-2's transformer keeps separate module stacks for the two modalities:

    video : ``attn1`` (self), ``attn2`` (cross-text), ``ff``
    audio : ``audio_attn1`` (self), ``audio_attn2`` (cross-text), ``audio_ff``

joined only by two small cross-modal bridges, ``audio_to_video_attn`` (feeds the
*video* prediction) and ``video_to_audio_attn`` (feeds the *audio* prediction).
Because the LoRA adapts each of these as its own tensor, the audio half and the
video half of a checkpoint are effectively independent and can be taken from
different training steps.

Typical use: audio overfits faster than video on a small/near-duplicate dataset,
so an early step has the best audio while a later step has the cleaner video.
This script keeps the early audio together with the late video - no retraining.

Example:
    python scripts/merge_av_lora.py \\
        --audio-checkpoint  outputs/run/checkpoints/lora_weights_step_05076.safetensors \\
        --video-checkpoint  outputs/run/checkpoints/lora_weights_step_11280.safetensors \\
        --output            outputs/run/checkpoints/lora_merged_a05076_v11280.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Audio self-attention, text cross-attention and feed-forward. These are the
# parameters that actually overfit the audio, so they follow the audio checkpoint.
AUDIO_CORE_MARKERS = ("audio_attn1", "audio_attn2", "audio_ff")
# Cross-modal bridges. Conceptually each belongs to the prediction it feeds.
AUDIO_TO_VIDEO_BRIDGE = "audio_to_video_attn"  # part of the VIDEO prediction path
VIDEO_TO_AUDIO_BRIDGE = "video_to_audio_attn"  # part of the AUDIO prediction path


def is_audio_key(key: str, bridges: str) -> bool:
    """Return True if ``key`` should be taken from the audio checkpoint.

    Args:
        key: A LoRA state-dict key (e.g. ``diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.lora_A.weight``).
        bridges: How to route the two cross-modal bridges - one of:
            ``"split"`` (default): video->audio bridge from audio ckpt, audio->video bridge from video ckpt
            ``"audio"``: both bridges from the audio ckpt
            ``"video"``: both bridges from the video ckpt
    """
    if any(marker in key for marker in AUDIO_CORE_MARKERS):
        return True
    if VIDEO_TO_AUDIO_BRIDGE in key:
        return bridges in ("split", "audio")
    if AUDIO_TO_VIDEO_BRIDGE in key:
        return bridges == "audio"
    return False


def merge(
    audio_checkpoint: Path,
    video_checkpoint: Path,
    output: Path,
    bridges: str,
) -> None:
    audio_sd = load_file(str(audio_checkpoint))
    video_sd = load_file(str(video_checkpoint))

    audio_keys, video_keys = set(audio_sd), set(video_sd)
    if audio_keys != video_keys:
        only_audio = sorted(audio_keys - video_keys)
        only_video = sorted(video_keys - audio_keys)
        details = ""
        if only_audio:
            details += f"\n  {len(only_audio)} key(s) only in audio ckpt, e.g. {only_audio[:3]}"
        if only_video:
            details += f"\n  {len(only_video)} key(s) only in video ckpt, e.g. {only_video[:3]}"
        raise SystemExit(
            "Checkpoints are not compatible (different LoRA architecture/rank?)." + details
        )

    merged: dict = {}
    from_audio = 0
    from_video = 0
    for key in video_sd:  # deterministic iteration order
        if is_audio_key(key, bridges):
            if audio_sd[key].shape != video_sd[key].shape:
                raise SystemExit(
                    f"Shape mismatch for {key}: "
                    f"audio {tuple(audio_sd[key].shape)} vs video {tuple(video_sd[key].shape)}"
                )
            merged[key] = audio_sd[key]
            from_audio += 1
        else:
            merged[key] = video_sd[key]
            from_video += 1

    # Preserve safetensors metadata (rank/alpha/etc.) from the video checkpoint.
    with safe_open(str(video_checkpoint), framework="pt") as f:
        metadata = f.metadata() or {}

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output), metadata=metadata)

    print("Merged LoRA written to", output)
    print(f"  tensors from audio ckpt ({audio_checkpoint.name}): {from_audio}")
    print(f"  tensors from video ckpt ({video_checkpoint.name}): {from_video}")
    print(f"  bridge routing: {bridges}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio-checkpoint",
        type=Path,
        required=True,
        help="LoRA .safetensors to take the audio branch from (e.g. the early, non-overfit step).",
    )
    parser.add_argument(
        "--video-checkpoint",
        type=Path,
        required=True,
        help="LoRA .safetensors to take the video branch from (e.g. the later, cleaner step).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the merged LoRA .safetensors.",
    )
    parser.add_argument(
        "--bridges",
        choices=("split", "audio", "video"),
        default="split",
        help=(
            "Which checkpoint the cross-modal bridges come from. "
            "'split' (default): audio->video bridge follows video, video->audio bridge follows audio. "
            "'audio'/'video': both bridges follow that checkpoint."
        ),
    )
    args = parser.parse_args()

    merge(
        audio_checkpoint=args.audio_checkpoint,
        video_checkpoint=args.video_checkpoint,
        output=args.output,
        bridges=args.bridges,
    )


if __name__ == "__main__":
    main()
