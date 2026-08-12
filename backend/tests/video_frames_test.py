"""
Regression test for video-to-dataset extraction's frame-selection logic
(services/video_frames.py) — the free, local ffmpeg-based replacement
for uploading whole videos to Gemini's video File API.

Builds small synthetic videos with ffmpeg (no network, no API key
needed) to verify two failure modes that were caught and fixed during
development:
  1. Distinct content must NOT be collapsed as "duplicate" frames —
     the original whole-thumbnail-mean-diff approach diluted a small
     changed region (exactly what a table/text region against a mostly
     blank background looks like) into a near-zero score.
  2. Genuinely static content SHOULD collapse to very few frames, so
     dedup isn't disabled outright as an overcorrection.
Run:  python -m tests.video_frames_test   (from backend/, needs ffmpeg)
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.video_frames import extract_table_frames, FrameExtractionError  # noqa: E402

FAILURES = []


def check(name, condition):
    print(f"{'✅' if condition else '❌'} {name}")
    if not condition:
        FAILURES.append(name)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _build_multi_screen_video(out_path: str, texts: list[str], seg_seconds: int = 2) -> None:
    """Concatenates one short clip per text label — simulates a screen
    recording that cuts between a few different table/dashboard views."""
    with tempfile.TemporaryDirectory() as d:
        parts = []
        for i, txt in enumerate(texts):
            p = os.path.join(d, f"part{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                f"color=c=white:s=640x360:d={seg_seconds}",
                "-vf", f"drawtext=text='{txt}':fontcolor=black:fontsize=28:x=50:y=100",
                p,
            ], check=True, capture_output=True)
            parts.append(p)
        list_path = os.path.join(d, "list.txt")
        with open(list_path, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
        ], check=True, capture_output=True)


def _build_static_video(out_path: str, text: str, seconds: int = 6) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=white:s=640x360:d={seconds}",
        "-vf", f"drawtext=text='{text}':fontcolor=black:fontsize=28:x=50:y=100",
        out_path,
    ], check=True, capture_output=True)


def main():
    if not _ffmpeg_available():
        print("⚠️  ffmpeg/ffprobe not found on PATH — skipping (this suite "
              "needs them; the app itself degrades to a clear error in "
              "the same situation, see README's ffmpeg note).")
        return

    with tempfile.TemporaryDirectory() as d:
        multi_path = os.path.join(d, "multi.mp4")
        _build_multi_screen_video(
            multi_path, ["Screen One Region A 100", "Screen Two Region B 200",
                         "Screen Three Region C 300"])
        frames = extract_table_frames(multi_path)
        check("distinct-content video: 3 legitimately different screens "
              "are NOT collapsed into 1 (regression: whole-thumbnail mean "
              "diff used to dilute a small text region into ~0)",
              len(frames) >= 3)

        static_path = os.path.join(d, "static.mp4")
        _build_static_video(static_path, "Static Table")
        frames2 = extract_table_frames(static_path)
        check("static video: genuinely unchanging content collapses to "
              "a small frame count, not one-per-sampled-instant",
              len(frames2) <= 2)

        check("frames are non-empty JPEG bytes",
              all(len(f) > 100 and f[:2] == b"\xff\xd8" for f in frames))

    try:
        extract_table_frames("/nonexistent/path.mp4")
        check("missing file raises FrameExtractionError", False)
    except FrameExtractionError:
        check("missing file raises FrameExtractionError", True)

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("🎉 VIDEO FRAME EXTRACTION REGRESSION SUITE PASSES")


if __name__ == "__main__":
    main()
