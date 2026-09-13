"""
Frame selection for video-to-dataset extraction (services/video_frames).

Two failure modes, opposite to each other, and both were real:

  1. Distinct screens collapsing into one. The original score was a
     mean difference across the whole thumbnail, which dilutes a small
     changed region into roughly zero — and a small changed region
     against a mostly blank background is exactly what a table on a
     slide looks like. Three different screens came out as one frame.
  2. Static content not collapsing at all. The overcorrection: turn
     dedup off and a six-second still becomes one frame per sampled
     instant, every one of them an identical paid extraction call.

The videos are built locally with ffmpeg — no network, no API key.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from app.services.video_frames import FrameExtractionError, extract_table_frames

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="needs ffmpeg and ffprobe on PATH",
)


def _clip(path: str, text: str, seconds: int) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "color=c=white:s=640x360:d={}".format(seconds),
        "-vf", ("drawtext=text='{}':fontcolor=black:fontsize=28:x=50:y=100"
                .format(text)),
        path,
    ], check=True, capture_output=True)


@pytest.fixture(scope="module")
def cut_between_screens(tmp_path_factory):
    """A screen recording that cuts between three different views."""
    out = str(tmp_path_factory.mktemp("video") / "multi.mp4")
    labels = ["Screen One Region A 100", "Screen Two Region B 200",
              "Screen Three Region C 300"]

    with tempfile.TemporaryDirectory() as d:
        parts = []
        for i, text in enumerate(labels):
            part = os.path.join(d, "part{}.mp4".format(i))
            _clip(part, text, seconds=2)
            parts.append(part)

        listing = os.path.join(d, "list.txt")
        with open(listing, "w") as fh:
            for part in parts:
                fh.write("file '{}'\n".format(part))

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out,
        ], check=True, capture_output=True)

    return extract_table_frames(out)


@pytest.fixture(scope="module")
def one_still_screen(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("video") / "static.mp4")
    _clip(out, "Static Table", seconds=6)
    return extract_table_frames(out)


def test_three_different_screens_are_not_collapsed_into_one(
        cut_between_screens):
    assert len(cut_between_screens) >= 3, (
        "distinct screens were deduplicated — a small changed region "
        "against a blank background scores near zero on a whole-frame "
        "mean difference")


def test_a_still_video_collapses_to_almost_nothing(one_still_screen):
    """Every extra frame here is an identical extraction call paid for
    twice."""
    assert len(one_still_screen) <= 2


def test_what_comes_back_is_actually_jpeg(cut_between_screens):
    for frame in cut_between_screens:
        assert len(frame) > 100
        assert frame[:2] == b"\xff\xd8", "not a JPEG"


def test_a_missing_file_is_an_error_not_an_empty_list():
    """An empty list would read as 'no tables in this video' and send
    the user looking at their recording instead of their path."""
    with pytest.raises(FrameExtractionError):
        extract_table_frames("/nonexistent/path.mp4")
