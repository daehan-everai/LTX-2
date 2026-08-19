# OpenRouter contact-sheet captioning

Guide: https://github.com/daehan-everai/LTX-2/blob/daehan/train/packages/ltx-trainer/docs/contact-sheet-captioning/README.md

This is the captioning path used for the doggy SFT / DPO recaptions. It does **not** send the MP4 to the model. It tiles chronological stills, then captions those pages with a hosted vision model on OpenRouter.

Use this instead of [`scripts/caption_videos.py`](../../scripts/caption_videos.py) (local Qwen2.5-Omni / Gemini Flash) when:

- the clip is NSFW and the local/Gemini captioner refuses or sanitizes it
- you do not want a local VLM on GPU
- the hosted model will not ingest a full MP4 (Grok on OpenRouter often will not)

Scripts live in `packages/ltx-trainer/scripts/`. Run them from that directory after `uv sync` at the repo root.

## Layout

```
videos_dir/
  clip_a.mp4
  clip_b.mp4

sheets/                         # written by build_contact_sheets.py
  clip_a-p1.jpg … clip_a-p4.jpg
  clip_b-p1.jpg … clip_b-p4.jpg
  manifest.jsonl

captions.jsonl                  # per-clip records + cost/model metadata
dataset.json                    # {caption, media_path} for process_dataset.py
```

Clip id is the video stem (`clip_a`). Sheet files must be `{id}-p{page}.jpg`.

Supported video suffixes: `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`.

## Prerequisites

- Repo env: from the LTX-2 root, `uv sync --frozen` then `source .venv/bin/activate` (or prefix commands with `uv run`).
- `ffmpeg` and `ffprobe` on `PATH`.
- `OPENROUTER_API_KEY` in the environment. Do not pass the key on the CLI.
- Working directory: `packages/ltx-trainer`.

## 1. Build contact sheets

Default layout matches the doggy recaptions: **4 pages × 12 frames**, 4 columns, 400 px cells. Frames are sampled across the clip (skipping a small head/tail margin), scaled to a square cell, and stamped with `{index} {seconds}s`.

```bash
cd packages/ltx-trainer

uv run python scripts/build_contact_sheets.py /path/to/videos_dir \
  --output-dir /path/to/sheets \
  --manifest /path/to/sheets/manifest.jsonl \
  --frames-per-page 12 \
  --pages 4 \
  --columns 4 \
  --cell 400 \
  --workers 4
```

With no `--ids`, every video in `videos_dir` is processed. To restrict:

```bash
uv run python scripts/build_contact_sheets.py /path/to/videos_dir \
  --output-dir /path/to/sheets \
  --manifest /path/to/sheets/manifest.jsonl \
  --ids clip_a clip_b

# or one id per line
uv run python scripts/build_contact_sheets.py /path/to/videos_dir \
  --output-dir /path/to/sheets \
  --manifest /path/to/sheets/manifest.jsonl \
  --ids-file ids.txt
```

`manifest.jsonl` rows look like:

```json
{
  "id": "clip_a",
  "video": "/path/to/videos_dir/clip_a.mp4",
  "duration_seconds": 5.02,
  "sample_times_seconds": [0.08, 0.18],
  "boards": ["clip_a-p1.jpg", "clip_a-p2.jpg", "clip_a-p3.jpg", "clip_a-p4.jpg"]
}
```

### `build_contact_sheets.py` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `video_dir` (positional) | required | Directory of source clips |
| `--output-dir` | required | Where `{id}-pN.jpg` is written |
| `--manifest` | required | JSONL index of boards |
| `--ids` | all videos | Clip stems to process |
| `--ids-file` | none | Extra stems, one per line |
| `--frames-per-page` | `12` | Tiles on each page |
| `--pages` | `4` | Number of `{id}-pN.jpg` files |
| `--columns` | `4` | Grid columns (rows = ceil(frames/columns)) |
| `--cell` | `400` | Square cell size in pixels |
| `--workers` | `4` | Parallel ffmpeg jobs |

## 2. Caption the sheets

The captioner sends every page as a high-detail image, with **no old caption**, and asks the model for JSON that includes `"caption"`. Default model is Grok 4.6; Gemini 3.7 Flash is the fallback if Grok fails.

```bash
export OPENROUTER_API_KEY=...   # do not echo this

cd packages/ltx-trainer

uv run python scripts/caption_from_contact_sheets.py \
  --video-dir /path/to/videos_dir \
  --board-dir /path/to/sheets \
  --prompt docs/contact-sheet-captioning/prompts/doggy_contact_sheet_caption.md \
  --output /path/to/captions.jsonl \
  --dataset-json /path/to/dataset.json \
  --pages 4 \
  --model x-ai/grok-4.6 \
  --fallback-model google/gemini-3.7-flash \
  --workers 6
```

That `--prompt` is the doggy recaption prompt. For a generic cinematic paragraph, omit `--prompt` (defaults to [`prompts/contact_sheet_caption.md`](prompts/contact_sheet_caption.md)).

Ids are discovered from `*-p1.jpg` in `--board-dir`, else from videos in `--video-dir`. Restrict with `--ids` / `--ids-file`. `--pages` must match the sheet build.

Already-ok rows in `--output` are skipped. Pass `--override` to recaption everything.

### `caption_from_contact_sheets.py` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--video-dir` | required | Clips; used to resolve `media_path` |
| `--board-dir` | required | Directory of `{id}-pN.jpg` |
| `--output` | required | JSONL of per-clip records |
| `--dataset-json` | optional | `{caption, media_path}` list for `process_dataset.py` |
| `--prompt` | generic prompt in this folder | System prompt file |
| `--ids` / `--ids-file` | all discovered ids | Subset of stems |
| `--pages` | `4` | Pages sent per clip |
| `--model` | `x-ai/grok-4.6` | OpenRouter model id |
| `--fallback-model` | `google/gemini-3.7-flash` | Tried if the primary fails |
| `--workers` | `6` | Parallel OpenRouter calls |
| `--override` | off | Ignore existing `--output` rows |

`--fallback-model` is skipped when it equals `--model`.

## Prompts

| File | When to use |
|------|-------------|
| [`prompts/doggy_contact_sheet_caption.md`](prompts/doggy_contact_sheet_caption.md) | Doggy SFT/DPO recaptions. Caption must start with `Doggystyle.`, include pace/roughness/hands/spanking from the sheets, 75–125 words, JSON schema with `caption`, `pose`, `pace`, `roughness`, `spanking_present`, `spank_count`. |
| [`prompts/contact_sheet_caption.md`](prompts/contact_sheet_caption.md) | Generic LTX-style paragraph. JSON `{"caption":"..."}` only. |

Both prompts tell the model **not** to invent audio, mention the contact sheet, or copy a prior caption. The script only requires a non-empty `caption` field; extra schema keys are stored under `fields` in `captions.jsonl`.

## Outputs

`captions.jsonl` one object per clip:

```json
{
  "id": "clip_a",
  "caption": "Doggystyle. ...",
  "model_requested": "x-ai/grok-4.6",
  "model_returned": "x-ai/grok-4.6",
  "provider": "...",
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "cost_usd": 0.0,
  "fields": {"pose": "doggystyle", "pace": "fast"}
}
```

`dataset.json` (only if `--dataset-json` is set) is what the trainer preprocess step consumes. `media_path` is relative to the JSON file’s directory:

```json
[
  {
    "caption": "Doggystyle. ...",
    "media_path": "videos_dir/clip_a.mp4"
  }
]
```

Then:

```bash
uv run python scripts/process_dataset.py /path/to/dataset.json \
  --resolution-buckets "448x768x49" \
  --model-path /path/to/ltx-2.safetensors \
  --text-encoder-path /path/to/gemma \
  --caption-column caption \
  --video-column media_path
```

Use the resolution bucket and `--frame-sampling` your training recipe actually uses.

## Doggy recaption recipe (copy-paste)

This is the combination used for the reviewed doggy clips: 4×12 sheets, doggy prompt, Grok 4.6 with Gemini Flash fallback, no previous caption in the request.

```bash
cd packages/ltx-trainer
export OPENROUTER_API_KEY=...

uv run python scripts/build_contact_sheets.py /path/to/videos_dir \
  --output-dir /path/to/sheets \
  --manifest /path/to/sheets/manifest.jsonl \
  --frames-per-page 12 --pages 4 --columns 4 --cell 400

uv run python scripts/caption_from_contact_sheets.py \
  --video-dir /path/to/videos_dir \
  --board-dir /path/to/sheets \
  --prompt docs/contact-sheet-captioning/prompts/doggy_contact_sheet_caption.md \
  --output /path/to/captions.jsonl \
  --dataset-json /path/to/dataset.json
```

## What this script does not do

- It does not run the older three-call motion/spanking **audit** then caption pipeline. This is the one-shot sheet → JSON caption path (same sheet layout, same OpenRouter models).
- It does not upload the MP4. If you need true audio transcription, use [`caption_videos.py`](../../scripts/caption_videos.py) instead, or add audio in a separate pass.
- It does not rewrite `process_dataset.py` inputs other than `caption` / `media_path`.

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `missing binaries on PATH: ffmpeg, ffprobe` | Install ffmpeg. |
| `OPENROUTER_API_KEY is required` | Export the key in the same shell. |
| `missing contact sheets` | `--pages` must match the build; files must be `{id}-p1.jpg` … `{id}-pN.jpg` in `--board-dir`. |
| `no video for {id}` | Stem of the mp4 must match the sheet id; file must be in `--video-dir`. |
| `HTTP 429` | The script retries; lower `--workers`. |
| empty / invalid JSON caption | Prompt must require `{"caption":"..."}`. Failures are logged as `caption_error`; the run exits non-zero if any id is missing. |
| Gemini `PROHIBITED_CONTENT` | Fallback to Grok should trigger; if both fail, recaption that id alone after checking the sheets. |
| Resume looks stuck | Delete the bad line from `captions.jsonl` or pass `--override` for a full redo. |

Progress lines are JSON on stdout: `caption_resume`, `caption_ok`, `caption_error`, `dataset_json`, `caption_done`.
