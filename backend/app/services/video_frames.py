"""
services/video_frames.py — free, local video-to-frames extraction via
ffmpeg, used to make video-table-extraction cheaper and more reliable
than uploading the whole clip to Gemini's video File API.

Why this exists: Gemini's video-understanding endpoint is the slowest,
priciest, and (per the documented SDK timeout issues in
ai/gemini_client.py) least reliable way to read a table shown in a
video. A table on screen doesn't need Gemini to *watch* the video —
it needs Gemini to *read* whichever few frames actually show the
table clearly. ffmpeg can find those frames locally, for free, and
each one then reuses the already-hardened image extraction path
(services/table_extractor.py::extract_table_from_image).

This is the same scene-change-detection technique used by tools like
bradautomates/claude-video — a standard, publicly documented ffmpeg
filter graph, reimplemented here directly (not vendored) so it runs as
a plain backend function instead of an interactive agent skill.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

MAX_FRAMES = 8            # keep Gemini image-extraction calls bounded
SCENE_THRESHOLD = 0.25     # ffmpeg scene-change score, 0-1 (lower = more sensitive)
# Table/spreadsheet frames are often mostly blank with a small content
# region (a few changed rows in an otherwise-static screen recording, or
# a short text label against a big blank background). A single mean-diff
# over the WHOLE thumbnail dilutes that into near-zero — confirmed with a
# synthetic test where two frames with completely different visible text
# scored a whole-thumbnail diff of ~0.2-0.7 out of 255, because the
# changed region was a small fraction of total pixels. Comparing in grid
# BLOCKS and taking the single most-changed block, instead of averaging
# across all of them, catches a localized change even when 95% of the
# frame is identical — the right side to err on, given what this data
# feeds into (a few extra harmless Gemini calls on true near-duplicates
# beats silently collapsing a distinct table view).
DEDUP_THUMB_SIZE = 64
DEDUP_GRID = 8              # 8x8 blocks of 8x8 px each at THUMB_SIZE=64
DEDUP_THRESHOLD = 10.0      # max single-block abs brightness delta (0-255)


class FrameExtractionError(Exception):
    pass


def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise FrameExtractionError(f"ffmpeg step timed out after {timeout}s")
    except FileNotFoundError:
        raise FrameExtractionError(
            "ffmpeg is not installed on this server — video frame "
            "extraction is unavailable")


def _probe_duration(path: str) -> Optional[float]:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", path], timeout=30)
    try:
        return float(r.stdout.decode().strip())
    except (ValueError, AttributeError):
        return None


def _extract_scene_frames(path: str, out_dir: str, threshold: float) -> list[str]:
    pattern = os.path.join(out_dir, "scene_%04d.jpg")
    _run([
        "ffmpeg", "-y", "-i", path,
        "-vf", f"select='gt(scene,{threshold})',scale=1024:-2",
        "-vsync", "vfr", "-q:v", "3", "-frames:v", str(MAX_FRAMES * 4),
        pattern,
    ], timeout=180)
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith("scene_"))


def _extract_uniform_frames(path: str, out_dir: str, duration: float, n: int) -> list[str]:
    """Fallback for videos with too little scene change to detect (a
    static dashboard, a slow scroll) — even time-based sampling instead."""
    pattern = os.path.join(out_dir, "uniform_%04d.jpg")
    fps = max(n / max(duration, 1), 0.05)
    _run([
        "ffmpeg", "-y", "-i", path,
        "-vf", f"fps={fps:.4f},scale=1024:-2",
        "-frames:v", str(n), "-q:v", "3",
        pattern,
    ], timeout=180)
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith("uniform_"))


def _thumbnail_signature(path: str) -> Optional[list[int]]:
    """Grayscale thumbnail as a flat list of 0-255 ints, for cheap
    near-duplicate comparison. Pure PIL, no extra dependency. See the
    DEDUP_* comment above for why comparison uses blocks, not a flat mean."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((DEDUP_THUMB_SIZE, DEDUP_THUMB_SIZE))
        return list(img.getdata())
    except Exception:
        return None


def _max_block_diff(sig_a: list[int], sig_b: list[int]) -> float:
    """Splits both thumbnails into a DEDUP_GRID x DEDUP_GRID set of blocks
    and returns the single largest per-block mean difference — a localized
    change (a few different table rows) shows up clearly here even when it
    would be averaged away by comparing the whole image at once."""
    size = DEDUP_THUMB_SIZE
    cell = size // DEDUP_GRID
    worst = 0.0
    for by in range(DEDUP_GRID):
        for bx in range(DEDUP_GRID):
            total = 0
            count = 0
            for y in range(by * cell, (by + 1) * cell):
                row = y * size
                for x in range(bx * cell, (bx + 1) * cell):
                    total += abs(sig_a[row + x] - sig_b[row + x])
                    count += 1
            if count:
                worst = max(worst, total / count)
    return worst


def _dedup(frame_paths: list[str], threshold: float) -> list[str]:
    """Drops frames that are visually near-identical to the last frame
    that was kept (not just the previous one) — catches a table that
    sits on screen across many sampled frames without keeping every copy."""
    kept: list[str] = []
    last_sig: Optional[list[int]] = None
    for p in frame_paths:
        sig = _thumbnail_signature(p)
        if sig is None:
            kept.append(p)  # can't compare — keep it rather than risk dropping real content
            continue
        if last_sig is not None:
            if _max_block_diff(sig, last_sig) <= threshold:
                continue  # near-duplicate of the last kept frame
        kept.append(p)
        last_sig = sig
    return kept


def extract_table_frames_with_budget(video_path: str) -> tuple:
    """Frames, plus whether the frame budget cut the video short.

    A three-minute screen recording scrolling through 500 rows produces
    far more distinct views than MAX_FRAMES. Taking the first 8 and
    saying nothing hands back a dataset that is a fraction of what the
    user filmed, with no sign that anything is missing.
    """
    frames, truncated = _extract_frames(video_path)
    return frames, truncated


def extract_table_frames(video_path: str) -> list[bytes]:
    """Returns a small set of distinct JPEG frames (as bytes) most likely
    to each show a clear, distinguishable view of whatever table/
    spreadsheet/dashboard is in the video — for feeding one at a time
    into extract_table_from_image(). Raises FrameExtractionError with a
    clear message on any hard failure (missing ffmpeg, unreadable file)."""
    return _extract_frames(video_path)[0]


def _extract_frames(video_path: str) -> tuple:
    if not os.path.exists(video_path):
        raise FrameExtractionError("Video file not found")

    duration = _probe_duration(video_path)
    if duration is None:
        raise FrameExtractionError(
            "Could not read this video — the file may be corrupt or in an "
            "unsupported format")

    with tempfile.TemporaryDirectory() as out_dir:
        frames = _extract_scene_frames(video_path, out_dir, SCENE_THRESHOLD)
        if len(frames) < 2:
            # static content (a dashboard that never visibly "cuts") —
            # fall back to sampling evenly across the timeline instead
            n = min(MAX_FRAMES, max(3, int(duration // 5) + 1))
            frames = _extract_uniform_frames(video_path, out_dir, duration, n)

        if not frames:
            raise FrameExtractionError(
                "No readable frames could be extracted from this video")

        distinct = _dedup(frames, DEDUP_THRESHOLD)
        # More distinct views than the budget means the tail of the video
        # is never looked at. The caller has to be able to say so.
        truncated = max(0, len(distinct) - MAX_FRAMES)
        frames = distinct[:MAX_FRAMES]

        out: list[bytes] = []
        for p in frames:
            with open(p, "rb") as f:
                out.append(f.read())
        return out, truncated
