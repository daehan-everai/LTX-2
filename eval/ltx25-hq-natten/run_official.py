#!/usr/bin/env python3
"""Official LTX 2.5 two-stage HQ I2V. No SGLang. DiffVAE decode via natten."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

OVERLAY = Path(os.environ.get("LTX_OVERLAY_ROOT", "/workspace/ltx-official/overlay"))
for package in ("ltx-core/src", "ltx-pipelines/src"):
    sys.path.insert(0, str(OVERLAY / package))

import torch
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.model.video_vae.transformer.attention import natten_available
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths

from run_common import atomic_json, job_id_of, load_manifest, probe_media, sha256, utcnow

# natten 0.21.5 autograd Function cannot consume inference-mode tensors (RMSNorm in
# DiffVAE NABlock). Official HQ __call__ is @inference_mode and yields a lazy decode
# iterator that runs *after* that context exits.
_hq_call = TI2VidTwoStagesHQPipeline.__call__
while hasattr(_hq_call, "__wrapped__"):
    _hq_call = _hq_call.__wrapped__
TI2VidTwoStagesHQPipeline.__call__ = torch.no_grad()(_hq_call)

WIDTH, HEIGHT = 832, 1216
NUM_FRAMES = 241
FPS = 24.0


def lora(path: Path, strength: float) -> LoraPathStrengthAndSDOps:
    return LoraPathStrengthAndSDOps(str(path), strength, LTXV_LORA_COMFY_RENAMING_MAP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--column", default="02115")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not natten_available():
        raise RuntimeError("natten is not installed; refusing Triton NA fallback")
    print("NATTEN_OK", flush=True)

    rows = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "receipt.json"
    completed = {}
    if receipt_path.is_file():
        import json
        completed = {
            item["job_id"]: item
            for item in (json.loads(receipt_path.read_text()).get("outputs") or [])
            if item.get("job_id")
        }

    transformer = args.model_root / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    text_encoder = args.model_root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    video_vae = args.model_root / "vae/ltx-2.5-video-vae-bf16.safetensors"
    audio_vae = args.model_root / "vae/ltx-2.5-audio-vae-bf16.safetensors"
    spatial = args.model_root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    distilled = args.model_root / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    for path in (transformer, text_encoder, video_vae, audio_vae, spatial, distilled, args.checkpoint):
        if not path.is_file():
            raise RuntimeError(f"missing {path}")

    receipt = {
        "status": "running",
        "started_at": utcnow(),
        "role": "official_ltx25_hq_natten",
        "column": args.column,
        "serving": {
            "backend": "ltx_pipelines.ti2vid_two_stages_hq",
            "sglang": False,
            "natten": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "diffvae_optimization": DiffVAEMode.CHUNKED_EAGER.value,
        },
        "generation": {
            "width": WIDTH,
            "height": HEIGHT,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "seed": 42,
            "distilled_lora_strength_stage_1": 0.5,
            "distilled_lora_strength_stage_2": 0.5,
            "sft_lora_strength": 1.0,
            "image_strength": 1.0,
        },
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "expected": len(rows),
        "completed_count": len(completed),
        "outputs": list(completed.values()),
    }
    atomic_json(receipt_path, receipt)

    load_started = time.monotonic()
    pipeline = TI2VidTwoStagesHQPipeline(
        model_paths=ModelPaths.from_split(
            transformer_path=str(transformer),
            text_encoder_path=str(text_encoder),
            video_vae_path=str(video_vae),
            audio_vae_path=str(audio_vae),
        ),
        distilled_lora=[lora(distilled, 1.0)],
        distilled_lora_strength_stage_1=0.5,
        distilled_lora_strength_stage_2=0.5,
        spatial_upsampler_path=str(spatial),
        loras=(lora(args.checkpoint, 1.0),),
        diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
    )
    receipt["model_load_seconds"] = time.monotonic() - load_started
    atomic_json(receipt_path, receipt)
    print(f"MODEL_LOADED {receipt['model_load_seconds']:.1f}s", flush=True)

    params = LTX_2_3_HQ_PARAMS
    started = time.monotonic()
    try:
        for row in rows:
            job_id = job_id_of(row)
            output = args.output / f"{job_id}.mp4"
            sample_started = time.monotonic()
            media = probe_media(output, WIDTH, HEIGHT)
            if media is None:
                image = args.manifest.parent / row["first_frame_path"]
                if not image.is_file():
                    raise RuntimeError(f"missing first frame {image}")
                with torch.no_grad():
                    video, audio, num_frames, tiling_config = pipeline(
                        prompt=row["prompt"],
                        negative_prompt=row["negative_prompt"],
                        seed=42,
                        height=HEIGHT,
                        width=WIDTH,
                        frame_rate=FPS,
                        num_inference_steps=params.num_inference_steps,
                        video_guider_params=params.video_guider_params,
                        audio_guider_params=params.audio_guider_params,
                        images=[ImageConditioningInput(str(image), 0, 1.0)],
                        num_frames=NUM_FRAMES,
                        tiling_config=AUTO_TILING,
                        stage_2_sigmas=STAGE_2_DISTILLED_SIGMAS,
                    )
                    encode_video(
                        video=video,
                        fps=FPS,
                        audio=audio,
                        output_path=str(output),
                        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
                    )
                media = probe_media(output, WIDTH, HEIGHT)
            if media is None:
                raise RuntimeError(f"Generated media failed validation: {output}")
            completed[job_id] = {
                "job_id": job_id,
                "sample_id": row["sample_id"],
                "prompt": row["prompt"],
                "file": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "elapsed_seconds": time.monotonic() - sample_started,
                "ffprobe": media,
            }
            receipt.update(
                completed_count=len(completed),
                outputs=[completed[job_id_of(item)] for item in rows if job_id_of(item) in completed],
                updated_at=utcnow(),
                elapsed_seconds=time.monotonic() - started,
            )
            atomic_json(receipt_path, receipt)
            print(f"SAMPLE_OK {job_id} {completed[job_id]['elapsed_seconds']:.1f}s {len(completed)}/{len(rows)}", flush=True)
        receipt.update(status="complete", completed_at=utcnow(), elapsed_seconds=time.monotonic() - started)
        atomic_json(receipt_path, receipt)
    except Exception as error:
        receipt.update(status="failed", failed_at=utcnow(), error=repr(error), elapsed_seconds=time.monotonic() - started)
        atomic_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    main()
