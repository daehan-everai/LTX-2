#!/usr/bin/env python3

"""Build chronological contact-sheet pages from short video clips.

This is the first half of the OpenRouter captioning path used when a vision
model cannot (or should not) ingest a full MP4. Each clip becomes numbered,
timestamped frame tiles: `{stem}-p1.jpg`, `{stem}-p2.jpg`, ...

Requires ffmpeg and ffprobe on PATH.

Basic usage:
    uv run python scripts/build_contact_sheets.py videos_dir/ \
        --output-dir sheets/ --manifest sheets/manifest.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _require_binaries() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing binaries on PATH: {', '.join(missing)}")


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return float(result.stdout.strip())


def _extract_frame(video: Path, timestamp: float, target: Path, cell: int) -> float:
    decoded_at = timestamp
    for _ in range(5):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{decoded_at:.5f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={cell}:{cell}:force_original_aspect_ratio=decrease,"
                f"pad={cell}:{cell}:(ow-iw)/2:(oh-ih)/2:black",
                str(target),
            ],
            check=True,
        )
        if target.is_file() and target.stat().st_size:
            return decoded_at
        decoded_at *= 0.92
    raise RuntimeError(f"Could not decode {video} near {timestamp:.3f}s")


def _annotate(path: Path, label: str) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"failed to read extracted frame {path}")
    cv2.rectangle(image, (0, 0), (120, 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        label,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), image)


def build_sheets(
    video: Path,
    output_dir: Path,
    frames_per_page: int,
    pages: int,
    columns: int,
    cell: int,
) -> dict[str, object]:
    total = _duration(video)
    count = frames_per_page * pages
    rows = (frames_per_page + columns - 1) // columns
    planned = [total * (0.015 + 0.92 * index / max(count - 1, 1)) for index in range(count)]
    frames: list[np.ndarray] = []
    actual: list[float] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index, timestamp in enumerate(planned):
            target = root / f"{index:02d}.png"
            decoded_at = _extract_frame(video, timestamp, target, cell)
            _annotate(target, f"{index + 1:02d} {decoded_at:.2f}s")
            actual.append(decoded_at)
            frame = cv2.imread(str(target))
            if frame is None:
                raise RuntimeError(f"failed to read {target}")
            frames.append(frame)

    outputs: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for page in range(pages):
        subset = frames[page * frames_per_page : (page + 1) * frames_per_page]
        sheet = np.zeros((rows * cell, columns * cell, 3), dtype=np.uint8)
        for index, frame in enumerate(subset):
            y = (index // columns) * cell
            x = (index % columns) * cell
            sheet[y : y + cell, x : x + cell] = frame
        output = output_dir / f"{video.stem}-p{page + 1}.jpg"
        cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 91])
        outputs.append(output.name)
    return {
        "id": video.stem,
        "video": str(video),
        "duration_seconds": total,
        "sample_times_seconds": actual,
        "boards": outputs,
    }


def _video_for_id(video_dir: Path, sample_id: str) -> Path:
    for suffix in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"missing video for id {sample_id} in {video_dir}")


def _discover_videos(video_dir: Path, ids: list[str]) -> list[Path]:
    if ids:
        return [_video_for_id(video_dir, sample_id) for sample_id in ids]
    videos = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"no videos in {video_dir}")
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chronological contact sheets from videos.")
    parser.add_argument("video_dir", type=Path, help="Directory of source clips")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--frames-per-page", type=int, default=12)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    _require_binaries()
    ids = list(args.ids)
    if args.ids_file:
        ids.extend(line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip())
    videos = _discover_videos(args.video_dir, ids)

    def task(video: Path) -> dict[str, object]:
        return build_sheets(video, args.output_dir, args.frames_per_page, args.pages, args.columns, args.cell)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(task, videos))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    console.print(
        json.dumps({"videos": len(records), "boards": len(records) * args.pages, "output_dir": str(args.output_dir)}),
    )


if __name__ == "__main__":
    main()
