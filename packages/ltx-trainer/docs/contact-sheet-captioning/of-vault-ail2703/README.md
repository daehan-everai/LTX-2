# OF-vault captioning (AIL-2703)

This is the packer that produced the first LTX-2.5 SFT set on comment [00f56aa0](https://linear.app/everai/issue/AIL-2703/first-sft-phase-with-of-vault-to-see-performance-improvement#comment-00f56aa0):

`gs://candy-team-ai-sandbox-store/daehan/ail-2703-of-vault-captioned/2026-08-25/work/clips`

It is **not** the generic doggy recaptioner. For already-cropped local clips, use [`caption_from_contact_sheets.py`](../../../scripts/caption_from_contact_sheets.py) with [`../prompts/of_selfplay_v4_audio_v1.md`](../prompts/of_selfplay_v4_audio_v1.md).

## What ran

| Piece | Path |
| --- | --- |
| Script | [`scripts/run_full.py`](scripts/run_full.py) |
| Prompt | [`prompts/of_selfplay_v4_audio_v1.md`](prompts/of_selfplay_v4_audio_v1.md) (AIVI-1358 rewrite-v4 / `of_selfplay_v4`) |
| Model | `x-ai/grok-4.6`, fallback `google/gemini-3.7-flash` |
| Layout | 10s crop from `t=0`, 4 pages × 12 frames, 4 columns, 360 px cells |

The script reads a pinned `inventory.json`, downloads raw OF objects from `creators/` (read-only), crops 10s, builds contact sheets, captions via OpenRouter, writes `dataset_sft_only.jsonl` (`sfw_ok` = no visible sex act), and uploads under `gs://…/daehan/`.

## Layout

```
of-vault-ail2703/
  config.json          # you supply; see example below
  inventory.json       # { "clips": [ {id, gcs, object, creator, size, start_sec}, ... ] }
  prompts/of_selfplay_v4_audio_v1.md
  scripts/run_full.py
  work/                # clips, sheets, captions.jsonl, datasets
```

## Config example

```json
{
  "ticket": "AIL-2703",
  "crop_seconds": 10,
  "model": "x-ai/grok-4.6",
  "fallback_model": "google/gemini-3.7-flash",
  "prompt": "prompts/of_selfplay_v4_audio_v1.md",
  "gcs_project": "candy-factory-79825583",
  "gcs_bucket": "candy-team-ai-sandbox-store",
  "gcs_dest": "gs://candy-team-ai-sandbox-store/daehan/ail-2703-of-vault-captioned/YYYY-MM-DD/",
  "raw_prefixes_readonly": [
    "gs://candy-team-ai-sandbox-store/creators/Renea/",
    "gs://candy-team-ai-sandbox-store/creators/Sophie Dee/"
  ]
}
```

Writes are refused unless `gcs_dest` is under `gs://<bucket>/daehan/` and not under `creators/`.

## Run

```bash
export OPENROUTER_API_KEY=...
export GCS_TOKEN_FILE=/path/to/gcs_access_token   # default /workspace/.gcs_token

python scripts/run_full.py \
  --inventory inventory.json \
  --work work \
  --workers 96
```

`--skip-upload` keeps the pack local. `--upload-only` uploads an existing `work/` tree.
