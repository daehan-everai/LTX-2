#!/usr/bin/env python3
"""
Compute resolution buckets for LTX-2 dataset training.

Extracts the bucketing logic from ai-toolkit to show how input files
(.mp4, .jpeg, .jpg, .png) are assigned to resolution buckets.

Stdout: a single JSON object  { "resolutions": { "<N>": [{ "key": "WxH", "fileCount": N }, ...], ... } }
Stderr: warnings / skipped-file messages
Exit 1: dataset_path is not a directory

Usage:
    python bucket_sizes.py /path/to/dataset
    python bucket_sizes.py /path/to/dataset --resolution 768 --divisibility 32
    python bucket_sizes.py /path/to/dataset --resolution 512 768
"""

import json
import os
import cv2
import sys
import math
import argparse

from PIL import Image
from PIL.ImageOps import exif_transpose
from typing import Dict, List, TypedDict
from collections import OrderedDict


class BucketResolution(TypedDict):
    width: int
    height: int


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
ALL_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS


def get_bucket_for_image_size(
    width: int,
    height: int,
    resolution: int = 512,
    divisibility: int = 32,
) -> BucketResolution:
    """Compute a bucket that preserves aspect ratio with ~resolution² total pixels.

    Dimensions are rounded to the nearest multiple of `divisibility`.
    """
    aspect = width / height
    target_pixels = min(resolution * resolution, width * height)

    ideal_h = math.sqrt(target_pixels / aspect)
    ideal_w = math.sqrt(target_pixels * aspect)

    h = max(round(ideal_h / divisibility) * divisibility, divisibility)
    w = max(round(ideal_w / divisibility) * divisibility, divisibility)

    return {"width": w, "height": h}


def get_video_dimensions(path: str) -> tuple:
    """Return (width, height) of a video file."""
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return width, height


def get_image_dimensions(path: str) -> tuple:
    """Return (width, height) of an image file."""
    img = exif_transpose(Image.open(path))
    return img.size


def get_file_dimensions(path: str) -> tuple:
    """Return (width, height) for a supported image or video file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return get_video_dimensions(path)
    if ext in IMAGE_EXTENSIONS:
        return get_image_dimensions(path)
    raise ValueError(f"Unsupported file type: {ext}")


def compute_buckets(
    dataset_path: str,
    resolution: int = 512,
    divisibility: int = 32,
) -> Dict[str, List[str]]:
    """Walk `dataset_path`, assign each supported file to a resolution bucket.

    Returns an ordered dict mapping "WxH" bucket keys to lists of file paths.
    """
    file_list = sorted(
        os.path.join(root, fname)
        for root, _dirs, files in os.walk(dataset_path)
        for fname in files
        if fname.lower().endswith(ALL_EXTENSIONS)
    )

    if not file_list:
        print(f"No supported files found in {dataset_path}", file=sys.stderr)
        return {}

    buckets: Dict[str, List[str]] = OrderedDict()
    errors = 0

    for path in file_list:
        try:
            width, height = get_file_dimensions(path)
        except Exception as exc:
            print(f"  Skipping {path}: {exc}", file=sys.stderr)
            errors += 1
            continue

        bucket = get_bucket_for_image_size(width, height, resolution, divisibility)
        key = f'{bucket["width"]}x{bucket["height"]}'
        buckets.setdefault(key, []).append(path)

    if errors:
        print(f"  ({errors} file(s) skipped due to errors)", file=sys.stderr)

    return buckets


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Compute resolution buckets for LTX-2 dataset training."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to directory containing media files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=[512],
        help="Target base resolution(s) (default: 512).",
    )
    parser.add_argument(
        "--divisibility",
        type=int,
        default=32,
        help="Bucket dimensions are divisible by this value (default: 32).",
    )
    return parser.parse_args(input_args)


def main(args):
    dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.isdir(dataset_path):
        print(f"Error: {dataset_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    resolutions: Dict[str, List[Dict]] = {}
    for resolution in args.resolution:
        buckets = compute_buckets(dataset_path, resolution, args.divisibility)
        if buckets:
            resolutions[str(resolution)] = [
                {"key": key, "fileCount": len(files)}
                for key, files in buckets.items()
            ]

    print(json.dumps({"resolutions": resolutions}))


if __name__ == "__main__":
    main(parse_args())
