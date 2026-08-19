#!/usr/bin/env python3

"""Caption videos from contact-sheet pages via OpenRouter.

Used when the local Qwen/Gemini `caption_videos.py` path is a poor fit:
send dense chronological stills to a hosted vision model instead of the MP4.

Requires `OPENROUTER_API_KEY`. Output JSON/JSONL uses `caption` + `media_path`
so it can be passed to `process_dataset.py`.

Basic usage:
    uv run python scripts/caption_from_contact_sheets.py \
        --video-dir videos_dir/ --board-dir sheets/ \
        --output captions.jsonl --dataset-json dataset.json
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console

console = Console()

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROMPT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "contact-sheet-captioning"
    / "prompts"
    / "contact_sheet_caption.md"
)
CERT_CANDIDATES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _ssl_context() -> ssl.SSLContext:
    cert_file = next((candidate for candidate in CERT_CANDIDATES if Path(candidate).is_file()), None)
    return ssl.create_default_context(cafile=cert_file)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_json_content(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def _image_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"}}


def _page_paths(board_dir: Path, sample_id: str, pages: int) -> list[Path]:
    paths = [board_dir / f"{sample_id}-p{page}.jpg" for page in range(1, pages + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing contact sheets: {missing}")
    return paths


def _discover_ids(video_dir: Path, board_dir: Path, ids: list[str]) -> list[str]:
    if ids:
        return ids
    found = sorted(
        {path.stem.rsplit("-p", 1)[0] for path in board_dir.glob("*-p1.jpg")},
    )
    if found:
        return found
    return sorted(path.stem for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def _media_path(video_dir: Path, sample_id: str) -> Path:
    for suffix in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no video for {sample_id} in {video_dir}")


def _openrouter_chat(api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=240) as response:
        return json.load(response)


def caption_one(
    sample_id: str,
    board_dir: Path,
    pages: int,
    system: str,
    model: str,
    fallback_model: str | None,
    api_key: str,
) -> dict:
    content: list[dict] = [
        {
            "type": "text",
            "text": "Write the caption from these chronological contact-sheet pages. Do not use any prior caption.",
        },
    ]
    for page, path in enumerate(_page_paths(board_dir, sample_id, pages), start=1):
        content.extend([{"type": "text", "text": f"Page {page} of {pages}:"}, _image_part(path)])

    models = [model]
    if fallback_model and fallback_model != model:
        models.append(fallback_model)
    last_error = "unknown"
    for current_model in models:
        payload = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "usage": {"include": True},
        }
        for attempt in range(4):
            try:
                body = _openrouter_chat(api_key, payload)
                parsed = _parse_json_content(body["choices"][0]["message"]["content"])
                caption = " ".join(str(parsed.get("caption") or "").split())
                if not caption:
                    raise ValueError("empty caption")
                usage = body.get("usage") or {}
                return {
                    "id": sample_id,
                    "caption": caption,
                    "model_requested": current_model,
                    "model_returned": body.get("model"),
                    "provider": body.get("provider"),
                    "prompt_tokens": usage.get("prompt_tokens") or 0,
                    "completion_tokens": usage.get("completion_tokens") or 0,
                    "cost_usd": float((usage.get("cost") or 0) or 0),
                    "fields": {key: value for key, value in parsed.items() if key != "caption"},
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                last_error = f"HTTP {exc.code}: {detail[:400]}"
                if exc.code == 429 or exc.code in {408, 409, 500, 502, 503, 504}:
                    time.sleep(min(20, (2**attempt) + random.random()))
                    continue
                break
            except (OSError, KeyError, IndexError, json.JSONDecodeError, ValueError, FileNotFoundError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(12, (2**attempt) + random.random()))
    raise RuntimeError(f"{sample_id}: {last_error}")


def _write_dataset_json(path: Path, rows: list[dict], video_dir: Path) -> None:
    dataset = []
    for row in rows:
        media = _media_path(video_dir, row["id"])
        dataset.append(
            {
                "caption": row["caption"],
                "media_path": os.path.relpath(media, start=path.parent),
            },
        )
    path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption contact sheets with an OpenRouter vision model.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--board-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL of per-clip caption records")
    parser.add_argument("--dataset-json", type=Path, help="process_dataset.py JSON with caption + media_path")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--model", default="x-ai/grok-4.6")
    parser.add_argument("--fallback-model", default="google/gemini-3.7-flash")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    system = args.prompt.read_text(encoding="utf-8")
    ids = list(args.ids)
    if args.ids_file:
        ids.extend(line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip())
    ids = _discover_ids(args.video_dir, args.board_dir, ids)
    if not ids:
        raise SystemExit("no clip ids found")

    completed = {} if args.override else {row["id"]: row for row in _read_jsonl(args.output) if row.get("caption")}
    pending = [sample_id for sample_id in ids if sample_id not in completed]
    lock = threading.Lock()
    console.print(
        json.dumps(
            {
                "event": "caption_resume",
                "total": len(ids),
                "done": len(completed),
                "pending": len(pending),
                "model": args.model,
                "prompt": str(args.prompt),
            },
        ),
    )

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for sample_id in ids:
                row = completed.get(sample_id)
                if row:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    caption_one,
                    sample_id,
                    args.board_dir,
                    args.pages,
                    system,
                    args.model,
                    args.fallback_model,
                    api_key,
                ): sample_id
                for sample_id in pending
            }
            for future in concurrent.futures.as_completed(futures):
                sample_id = futures[future]
                try:
                    row = future.result()
                    with lock:
                        completed[row["id"]] = row
                        persist()
                    console.print(json.dumps({"event": "caption_ok", "id": row["id"], "complete": len(completed)}))
                except Exception as exc:  # noqa: BLE001
                    console.print(json.dumps({"event": "caption_error", "id": sample_id, "error": str(exc)}))
        missing = [sample_id for sample_id in ids if sample_id not in completed]
        if missing:
            raise SystemExit(f"caption incomplete missing={len(missing)} example={missing[:8]}")

    if args.dataset_json:
        ordered = [completed[sample_id] for sample_id in ids]
        _write_dataset_json(args.dataset_json, ordered, args.video_dir)
        console.print(json.dumps({"event": "dataset_json", "path": str(args.dataset_json), "rows": len(ordered)}))
    console.print(json.dumps({"event": "caption_done", "ok": len(ids)}))


if __name__ == "__main__":
    main()
