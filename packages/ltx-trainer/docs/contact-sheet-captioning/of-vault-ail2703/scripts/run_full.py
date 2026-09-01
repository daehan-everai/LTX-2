#!/usr/bin/env python3
"""Inventory-pinned OF vault labeling: 10s crop, contact sheets, Grok caption, GCS upload.

Reads raw objects from creators/ but never writes there.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import html
import json
import mimetypes
import os
import random
import re
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
PROMPT_PATH = ROOT / CONFIG["prompt"]
WORK = ROOT / "work"
RAW = WORK / "raw"
CLIPS = WORK / "clips"
SHEETS = WORK / "sheets"
FRAMES = WORK / "first_frames"
CAPTIONS = WORK / "captions.jsonl"
PREPARED = WORK / "prepared.json"
REVIEW = ROOT / "review.html"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CERT_CANDIDATES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt")
CROP_SECONDS = int(CONFIG.get("crop_seconds") or 10)
PAGES = 4
FRAMES_PER_PAGE = 12
COLUMNS = 4
CELL = 360
GCS_PROJECT = CONFIG["gcs_project"]
GCS_BUCKET = CONFIG["gcs_bucket"]
GCS_DEST = CONFIG["gcs_dest"].rstrip("/") + "/"
WRITE_LOCK = threading.Lock()
PREP_LOCK = threading.Lock()
_client = None
_client_mtime = None
_client_lock = threading.Lock()


def log(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def token_path() -> Path:
    return Path(os.environ.get("GCS_TOKEN_FILE") or "/workspace/.gcs_token")


def gcs_token() -> str:
    path = token_path()
    if not path.is_file():
        raise RuntimeError(f"missing GCS token file: {path}")
    return path.read_text().strip()


def gcs_client():
    global _client, _client_mtime
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    path = token_path()
    mtime = path.stat().st_mtime
    with _client_lock:
        if _client is None or mtime != _client_mtime:
            creds = Credentials(token=gcs_token())
            _client = storage.Client(project=GCS_PROJECT, credentials=creds)
            _client_mtime = mtime
        return _client


def reset_gcs_client() -> None:
    global _client, _client_mtime
    with _client_lock:
        _client = None
        _client_mtime = None


def assert_dest_safe(prefix: str) -> None:
    if not prefix.startswith(f"gs://{GCS_BUCKET}/daehan/"):
        raise SystemExit(f"refusing to write outside daehan prefix: {prefix}")
    if "/creators/" in prefix:
        raise SystemExit(f"refusing to write into creators/: {prefix}")


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return float(out)


def has_audio_stream(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return bool(out)


def audio_probe(path: Path) -> dict:
    probe = {"stream": has_audio_stream(path), "mean_volume_db": None, "silence_ratio": None, "effectively_silent": True}
    if not probe["stream"]:
        return probe
    logged = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stderr
    match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", logged)
    if match:
        probe["mean_volume_db"] = float(match.group(1))
    silenced = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "silencedetect=n=-35dB:d=0.4", "-f", "null", "-"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start:\s*([-\d.]+)", silenced)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([-\d.]+)", silenced)]
    total = duration(path)
    silent = 0.0
    for start, end in zip(starts, ends + [total] * max(0, len(starts) - len(ends))):
        silent += max(0.0, min(end, total) - max(0.0, start))
    probe["silence_ratio"] = round(min(1.0, silent / total), 3) if total else 1.0
    mean = probe["mean_volume_db"]
    probe["effectively_silent"] = (mean is not None and mean < -45) or (
        probe["silence_ratio"] is not None and probe["silence_ratio"] >= 0.92
    )
    return probe


def suffix_for(clip: dict) -> str:
    ext = Path(clip["gcs"]).suffix.lower() or ".mov"
    if ext == ".jpeg":
        ext = ".jpg"
    return ext


def set_work(work: Path) -> None:
    global WORK, RAW, CLIPS, SHEETS, FRAMES, CAPTIONS, PREPARED, REVIEW
    WORK = work
    RAW = WORK / "raw"
    CLIPS = WORK / "clips"
    SHEETS = WORK / "sheets"
    FRAMES = WORK / "first_frames"
    CAPTIONS = WORK / "captions.jsonl"
    PREPARED = WORK / "prepared.json"
    REVIEW = work.parent / "review.html" if work.name != "work" else ROOT / "review.html"


def download_one(clip: dict) -> Path:
    local = clip.get("local_path")
    if local:
        path = Path(local)
        if path.is_file() and path.stat().st_size > 1000:
            return path
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f'{clip["id"]}{suffix_for(clip)}'
    expected = int(clip.get("size") or 0)
    if dest.is_file() and dest.stat().st_size > 10_000 and (not expected or dest.stat().st_size == expected):
        return dest
    last_error = "unknown"
    for attempt in range(6):
        try:
            client = gcs_client()
            blob = client.bucket(GCS_BUCKET).blob(clip["object"])
            tmp = dest.with_suffix(dest.suffix + ".partial")
            blob.download_to_filename(str(tmp))
            if tmp.stat().st_size < 10_000:
                raise RuntimeError(f"tiny download {tmp.stat().st_size}")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            reset_gcs_client()
            time.sleep(min(20, (2**attempt) + random.random()))
    raise RuntimeError(f'{clip["id"]}: download failed: {last_error}')


def crop_one(clip: dict, source: Path) -> Path:
    CLIPS.mkdir(parents=True, exist_ok=True)
    dest = CLIPS / f'{clip["id"]}.mp4'
    if dest.is_file() and dest.stat().st_size > 10_000:
        return dest
    total = duration(source)
    start = float(clip.get("start_sec") or 0)
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
    if total > CROP_SECONDS + 0.15 or start > 0.05:
        args = [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{max(0.0, start):.3f}", "-t", str(CROP_SECONDS), "-i", str(source),
        ]
    args += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest),
    ]
    run(args)
    return dest


def extract_frame(video: Path, timestamp: float, target: Path, cell: int) -> None:
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{timestamp:.5f}", "-i", str(video),
        "-frames:v", "1",
        "-vf", f"scale={cell}:{cell}:force_original_aspect_ratio=decrease,pad={cell}:{cell}:(ow-iw)/2:(oh-ih)/2:black",
        str(target),
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        if timestamp > 0.001:
            extract_frame(video, 0.0, target, cell)
            return
        raise


def annotate(path: Path, label: str) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 150), 22), fill="black")
    draw.text((5, 4), label, fill="white")
    image.save(path, quality=90)


def build_sheets(clip_id: str, video: Path) -> list[str]:
    SHEETS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    first = FRAMES / f"{clip_id}.jpg"
    if not first.is_file():
        extract_frame(video, 0.04, first, 480)
    boards = [SHEETS / f"{clip_id}-p{page}.jpg" for page in range(1, PAGES + 1)]
    if all(path.is_file() and path.stat().st_size for path in boards):
        return [path.name for path in boards]
    total = max(duration(video), 0.04)
    count = FRAMES_PER_PAGE * PAGES
    last = max(0.0, total - 0.08)
    planned = [min(last, total * (0.015 + 0.92 * index / max(count - 1, 1))) for index in range(count)]
    rows = (FRAMES_PER_PAGE + COLUMNS - 1) // COLUMNS
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def grab(item: tuple[int, float]) -> tuple[int, Path]:
            index, timestamp = item
            target = root / f"{index:02d}.jpg"
            try:
                extract_frame(video, timestamp, target, CELL)
            except subprocess.CalledProcessError:
                extract_frame(video, 0.0, target, CELL)
            if not target.is_file() or target.stat().st_size < 200:
                Image.new("RGB", (CELL, CELL), "black").save(target, quality=90)
            annotate(target, f"{index + 1:02d} {timestamp:.2f}s")
            return index, target

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            grabbed = dict(pool.map(grab, enumerate(planned)))
        cells = [Image.open(grabbed[index]).convert("RGB") for index in range(count)]
        for page in range(PAGES):
            subset = cells[page * FRAMES_PER_PAGE : (page + 1) * FRAMES_PER_PAGE]
            sheet = Image.new("RGB", (COLUMNS * CELL, rows * CELL), "black")
            for index, cell in enumerate(subset):
                sheet.paste(cell, ((index % COLUMNS) * CELL, (index // COLUMNS) * CELL))
            sheet.save(boards[page], quality=90)
    return [path.name for path in boards]


def prepare_one(clip: dict) -> dict:
    last_error = "unknown"
    for attempt in range(3):
        try:
            return _prepare_one_attempt(clip)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            extra = list(SHEETS.glob(f'{clip["id"]}-p*.jpg')) if SHEETS.exists() else []
            for path in [CLIPS / f'{clip["id"]}.mp4', FRAMES / f'{clip["id"]}.jpg', *extra]:
                try:
                    path.unlink()
                except OSError:
                    pass
            time.sleep(min(8, 1.5 * (attempt + 1)))
    raise RuntimeError(f'{clip["id"]}: {last_error}')


def _prepare_one_attempt(clip: dict) -> dict:
    cropped_path = CLIPS / f'{clip["id"]}.mp4'
    source_duration = None
    source_bytes = int(clip.get("size") or 0)
    if cropped_path.is_file() and cropped_path.stat().st_size > 10_000:
        cropped = cropped_path
    else:
        source = download_one(clip)
        source_bytes = source.stat().st_size
        source_duration = round(duration(source), 3)
        cropped = crop_one(clip, source)
        local = clip.get("local_path")
        if not local or Path(local).resolve() != source.resolve():
            try:
                source.unlink()
            except OSError:
                pass
    boards = build_sheets(clip["id"], cropped)
    probe = audio_probe(cropped)
    row = {
        **clip,
        "source_bytes": source_bytes,
        "source_duration": source_duration if source_duration is not None else round(duration(cropped), 3),
        "crop_duration": round(duration(cropped), 3),
        "boards": boards,
        "audio_probe": probe,
    }
    log(
        {
            "event": "prepared",
            "id": clip["id"],
            "crop": row["crop_duration"],
            "source": row["source_duration"],
            "silent": probe["effectively_silent"],
        }
    )
    return row


def ssl_context() -> ssl.SSLContext:
    cert = next((path for path in CERT_CANDIDATES if Path(path).is_file()), None)
    return ssl.create_default_context(cafile=cert)


def image_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"}}


def parse_json_content(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def openrouter_chat(api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, context=ssl_context(), timeout=240) as response:
        return json.load(response)


def caption_one(clip: dict, probe: dict, api_key: str, model: str, fallback: str) -> dict:
    system = PROMPT_PATH.read_text(encoding="utf-8")
    content: list[dict] = [
        {
            "type": "text",
            "text": "Write the caption from these chronological contact-sheet pages. Do not use any prior caption.\n"
            f"AUDIO_PROBE={json.dumps(probe)}",
        }
    ]
    for page in range(1, PAGES + 1):
        path = SHEETS / f'{clip["id"]}-p{page}.jpg'
        content.extend([{"type": "text", "text": f"Page {page} of {PAGES}:"}, image_part(path)])
    last_error = "unknown"
    for current in [model] + ([fallback] if fallback and fallback != model else []):
        payload = {
            "model": current,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "usage": {"include": True},
        }
        for attempt in range(6):
            try:
                body = openrouter_chat(api_key, payload)
                parsed = parse_json_content(body["choices"][0]["message"]["content"])
                caption = " ".join(str(parsed.get("caption") or "").split())
                if not caption:
                    raise ValueError("empty caption")
                usage = body.get("usage") or {}
                return {
                    "id": clip["id"],
                    "caption": caption,
                    "model_requested": current,
                    "model_returned": body.get("model"),
                    "provider": body.get("provider"),
                    "prompt_tokens": usage.get("prompt_tokens") or 0,
                    "completion_tokens": usage.get("completion_tokens") or 0,
                    "cost_usd": float((usage.get("cost") or 0) or 0),
                    "audio_probe": probe,
                    "fields": {key: value for key, value in parsed.items() if key != "caption"},
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                last_error = f"HTTP {exc.code}: {detail[:400]}"
                if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                    time.sleep(min(30, (2**attempt) + random.random()))
                    continue
                break
            except (OSError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(16, (2**attempt) + random.random()))
    raise RuntimeError(f'{clip["id"]}: {last_error}')


def load_captions() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in (ROOT / "seed" / "captions.jsonl", CAPTIONS):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") and row.get("caption"):
                rows[row["id"]] = row
    return rows


def persist_captions(rows: dict[str, dict], order: list[str]) -> None:
    CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with CAPTIONS.open("w", encoding="utf-8") as handle:
            for sample_id in order:
                row = rows.get(sample_id)
                if row:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sex_act_of(fields: dict) -> str:
    sex = str(fields.get("sex_act") or "none").strip().lower()
    if sex in {"", "null", "none"}:
        return "none"
    return sex


def is_sfw(fields: dict) -> bool:
    return sex_act_of(fields) == "none"


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def dataset_row(clip: dict, caption: dict) -> dict:
    fields = caption.get("fields") or {}
    sfw_ok = is_sfw(fields)
    return {
        "id": clip["id"],
        "caption": caption["caption"],
        "media_path": f'clips/{clip["id"]}.mp4',
        "gcs_clip": f'{GCS_DEST}clips/{clip["id"]}.mp4',
        "source_gcs": clip["gcs"],
        "creator": clip["creator"],
        "sfw_ok": sfw_ok,
        "sft_ok": sfw_ok,
        "sex_act": sex_act_of(fields),
        "people_count": fields.get("people_count"),
        "undressed": fields.get("undressed"),
        "crop_seconds": CROP_SECONDS,
        "model": caption.get("model_returned") or caption.get("model_requested"),
    }


def pack_datasets(clips: list[dict], captions: dict[str, dict]) -> dict:
    all_rows = [dataset_row(clip, captions[clip["id"]]) for clip in clips]
    sft_rows = [row for row in all_rows if row["sfw_ok"]]
    (WORK / "dataset_all.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8"
    )
    (WORK / "dataset_sft_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sft_rows), encoding="utf-8"
    )
    summary = {
        "ticket": CONFIG["ticket"],
        "clips": len(all_rows),
        "sfw": sum(1 for row in all_rows if row["sfw_ok"]),
        "nsfw": sum(1 for row in all_rows if not row["sfw_ok"]),
        "cost_usd": round(sum(float(captions[row["id"]].get("cost_usd") or 0) for row in all_rows), 4),
        "gcs_dest": GCS_DEST,
        "raw_untouched": CONFIG["raw_prefixes_readonly"],
        "crop_seconds": CROP_SECONDS,
        "model": CONFIG["model"],
        "prompt": CONFIG["prompt"],
        "sex_act_counts": {},
    }
    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["sex_act"]] = counts.get(row["sex_act"], 0) + 1
    summary["sex_act_counts"] = counts
    (WORK / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_review(clips: list[dict], captions: dict[str, dict]) -> None:
    cards = []
    sfw_true = 0
    for index, clip in enumerate(clips, 1):
        row = captions.get(clip["id"]) or {}
        fields = row.get("fields") or {}
        sfw_ok = is_sfw(fields)
        if sfw_ok:
            sfw_true += 1
        sfw_class = "sfw" if sfw_ok else "nsfw"
        sfw_text = "SFW" if sfw_ok else "NSFW"
        sex = sex_act_of(fields)
        sex_badge = "" if sfw_ok else f'<span class="badge nsfw">{esc(sex)}</span>'
        boards = "".join(
            f'<a href="work/sheets/{esc(clip["id"])}-p{page}.jpg" target="_blank">'
            f'<img src="work/sheets/{esc(clip["id"])}-p{page}.jpg" alt="page {page}"></a>'
            for page in range(1, PAGES + 1)
        )
        cards.append(
            f"""
<article data-sfw="{sfw_class}" data-creator="{esc(clip['creator'])}">
  <div class="title"><span>{index:03d}</span><code>{esc(clip['id'])}</code>
    <span class="badge {sfw_class}">{esc(sfw_text)}</span>
    {sex_badge}
    <span class="badge">{esc(clip['creator'])}</span>
  </div>
  <div class="media">
    <div>
      <video controls preload="none" playsinline src="work/clips/{esc(clip['id'])}.mp4"></video>
    </div>
    <div class="boards">{boards}</div>
  </div>
  <section class="caption"><p>{esc(row.get('caption'))}</p></section>
</article>"""
        )
    REVIEW.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIL-2703 OF vault captions · {len(clips)} clips</title>
<style>
:root{{color-scheme:dark;--bg:#0b0c10;--panel:#15171d;--line:#2a2e38;--muted:#a8adba;--accent:#d7ff67;--text:#f5f6f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
header,main{{max-width:1500px;margin:auto;padding:20px 24px}} h1{{margin:4px 0 8px;font-size:24px}} .sub{{color:var(--muted)}}
.summary,.filters{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}} .summary span,.badge,button{{border:1px solid var(--line);padding:4px 9px;border-radius:999px;font-size:12px;color:#c9ced8;background:transparent}}
.badge.sfw,button[data-f="sfw"]{{border-color:#7dffa0;color:#7dffa0}} .badge.nsfw,button[data-f="nsfw"]{{border-color:#ff8a8a;color:#ff8a8a;font-weight:700}}
button.active{{background:#22262f;color:var(--accent);border-color:var(--accent)}}
article{{border-top:1px solid var(--line);padding:22px 0}} .title{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}
.title span:first-child{{color:var(--accent);font-weight:700}} code{{font:12px ui-monospace,Menlo,monospace;color:var(--accent)}}
.media{{display:grid;grid-template-columns:minmax(260px,.8fr) minmax(520px,1.8fr);gap:14px}}
.boards{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
video,img{{width:100%;max-height:42vh;object-fit:contain;background:#050506;border-radius:8px;display:block}}
.caption{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:12px}}
p{{margin:0;white-space:pre-wrap}} a{{color:var(--accent)}}
@media(max-width:900px){{.media,.boards{{grid-template-columns:1fr}}}}
article.hide{{display:none}}
</style></head><body>
<header>
  <div class="sub">AIL-2703 · OF vault · Grok 4.6 · one 10s crop from t=0</div>
  <h1>{len(clips)}-clip caption review</h1>
  <p class="sub"><b>SFW</b> = no sex act (nude/tease OK). <b>NSFW</b> = penetration, oral, or dildo play.</p>
  <div class="summary">
    <span>clips {len(clips)}</span>
    <span>SFW {sfw_true}/{len(captions)}</span>
    <span>NSFW {len(captions) - sfw_true}/{len(captions)}</span>
  </div>
  <div class="filters">
    <button class="active" data-f="all">all</button>
    <button data-f="sfw">SFW</button>
    <button data-f="nsfw">NSFW</button>
  </div>
</header>
<main>
{''.join(cards)}
</main>
<script>
const buttons=[...document.querySelectorAll('button[data-f]')];
const articles=[...document.querySelectorAll('article')];
buttons.forEach(btn=>btn.onclick=()=>{{
  buttons.forEach(b=>b.classList.toggle('active', b===btn));
  const f=btn.dataset.f;
  articles.forEach(a=>a.classList.toggle('hide', f!=='all' && a.dataset.sfw!==f));
}});
</script>
</body></html>
""",
        encoding="utf-8",
    )


def iter_upload_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for rel in (
        "work/captions.jsonl",
        "work/prepared.json",
        "work/dataset_all.jsonl",
        "work/dataset_sft_only.jsonl",
        "work/manifest.json",
        "review.html",
        "config.json",
        "inventory.json",
        "prompts/of_selfplay_v4_audio_v1.md",
    ):
        path = ROOT / rel
        if path.is_file():
            files.append((path, rel))
    for folder, prefix in ((CLIPS, "work/clips"), (SHEETS, "work/sheets"), (FRAMES, "work/first_frames")):
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    files.append((path, f"{prefix}/{path.name}"))
    return files


def upload_one(item: tuple[Path, str]) -> str:
    path, rel = item
    object_name = GCS_DEST.replace(f"gs://{GCS_BUCKET}/", "") + rel
    last_error = "unknown"
    for attempt in range(5):
        try:
            blob = gcs_client().bucket(GCS_BUCKET).blob(object_name)
            blob.upload_from_filename(str(path))
            return f"gs://{GCS_BUCKET}/{object_name}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            reset_gcs_client()
            time.sleep(min(12, (2**attempt) + random.random()))
    raise RuntimeError(f"upload failed {rel}: {last_error}")


def upload_all(workers: int) -> int:
    assert_dest_safe(GCS_DEST)
    files = iter_upload_files()
    log({"event": "upload_start", "n": len(files), "dest": GCS_DEST})
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload_one, item) for item in files]
        for future in concurrent.futures.as_completed(futures):
            uri = future.result()
            done += 1
            if done % 25 == 0 or done == len(files):
                log({"event": "upload_progress", "done": done, "n": len(files), "last": uri})
    return done


def seed_captions_if_needed() -> None:
    seed = ROOT / "seed" / "captions.jsonl"
    if seed.is_file() and not CAPTIONS.is_file():
        CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
        CAPTIONS.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--prepare-workers", type=int, default=12)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--upload-workers", type=int, default=24)
    parser.add_argument("--model", default=CONFIG["model"])
    parser.add_argument("--fallback-model", default=CONFIG["fallback_model"])
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--inventory", type=Path, default=ROOT / "inventory.json")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    args = parser.parse_args()
    set_work(args.work if args.work.is_absolute() else ROOT / args.work)
    if not args.skip_upload:
        assert_dest_safe(GCS_DEST)
    clips = json.loads(args.inventory.read_text())["clips"]
    WORK.mkdir(parents=True, exist_ok=True)
    if args.upload_only:
        n = upload_all(args.upload_workers)
        log({"event": "done", "uploaded": n, "dest": GCS_DEST})
        return
    if not args.no_seed:
        seed_captions_if_needed()
    log({"event": "start", "clips": len(clips), "caption_workers": args.workers, "dest": GCS_DEST})

    meta_by_id: dict[str, dict] = {}
    if PREPARED.is_file():
        for row in json.loads(PREPARED.read_text()):
            meta_by_id[row["id"]] = row
    pending_prepare = [clip for clip in clips if clip["id"] not in meta_by_id or not (CLIPS / f'{clip["id"]}.mp4').is_file()]
    log({"event": "prepare_resume", "done": len(clips) - len(pending_prepare), "pending": len(pending_prepare)})
    if pending_prepare:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.prepare_workers) as pool:
            futures = {pool.submit(prepare_one, clip): clip["id"] for clip in pending_prepare}
            for future in concurrent.futures.as_completed(futures):
                sample_id = futures[future]
                try:
                    row = future.result()
                    with PREP_LOCK:
                        meta_by_id[row["id"]] = row
                        PREPARED.write_text(
                            json.dumps([meta_by_id[clip["id"]] for clip in clips if clip["id"] in meta_by_id], indent=2) + "\n"
                        )
                except Exception as exc:  # noqa: BLE001
                    log({"event": "prepare_error", "id": sample_id, "error": str(exc)})
        missing_prep = [clip["id"] for clip in clips if clip["id"] not in meta_by_id]
        if missing_prep:
            raise SystemExit(f"prepare incomplete missing={len(missing_prep)} sample={missing_prep[:8]}")
    meta = [meta_by_id[clip["id"]] for clip in clips]
    PREPARED.write_text(json.dumps(meta, indent=2) + "\n")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    completed = {} if args.override else load_captions()
    pending = [clip for clip in clips if clip["id"] not in completed]
    log({"event": "caption_resume", "done": len(completed), "pending": len(pending)})
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    caption_one,
                    clip,
                    next(item["audio_probe"] for item in meta if item["id"] == clip["id"]),
                    api_key,
                    args.model,
                    args.fallback_model,
                ): clip["id"]
                for clip in pending
            }
            for future in concurrent.futures.as_completed(futures):
                sample_id = futures[future]
                try:
                    row = future.result()
                    completed[row["id"]] = row
                    persist_captions(completed, [clip["id"] for clip in clips])
                    log(
                        {
                            "event": "caption_ok",
                            "id": row["id"],
                            "complete": len(completed),
                            "pending": len(clips) - len(completed),
                            "cost_usd": row.get("cost_usd"),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log({"event": "caption_error", "id": sample_id, "error": str(exc)})
        missing = [clip["id"] for clip in clips if clip["id"] not in completed]
        if missing:
            persist_captions(completed, [clip["id"] for clip in clips])
            raise SystemExit(f"caption incomplete missing={len(missing)} sample={missing[:8]}")
    persist_captions(completed, [clip["id"] for clip in clips])
    summary = pack_datasets(clips, completed)
    build_review(clips, completed)
    log({"event": "packed", **summary})
    if not args.skip_upload:
        n = upload_all(args.upload_workers)
        summary["uploaded"] = n
    (WORK / "DONE.json").write_text(json.dumps(summary, indent=2) + "\n")
    log({"event": "done", **summary})


if __name__ == "__main__":
    main()
