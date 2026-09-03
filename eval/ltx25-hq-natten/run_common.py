#!/usr/bin/env python3
"""Shared helpers for the official LTX 2.5 30-sample runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_ROWS = 30


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def probe_media(path: Path, width: int, height: int) -> dict | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        data = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
            "-of", "json", str(path),
        ], text=True, timeout=30))
    except Exception:
        return None
    video = next((item for item in data.get("streams", []) if item.get("codec_type") == "video"), None)
    audio = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), None)
    duration = float((data.get("format") or {}).get("duration") or 0)
    if not video or video.get("width") != width or video.get("height") != height:
        return None
    if not 9.6 <= duration <= 10.6:
        return None
    if not audio:
        return None
    return data


def job_id_of(row: dict) -> str:
    return row.get("job_id") or f"{row['sample_id']}__{row.get('variant', 'default')}"


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rewrite-v4 eval rows, got {len(rows)}")
    return rows
