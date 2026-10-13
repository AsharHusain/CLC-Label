# /// script
# requires-python = "<3.13"
# dependencies = ["opencv-python-headless"]
# ///
"""W1: index the already-extracted video archive into a flat manifest.

Videos_Cropped1..6 are upload partitions, not duplicates -- flattens them
into one row per (word, take), failing loudly on any collision.

Two filename layouts are understood (any directory depth under a partition):
  - original:   <partition>/<Word>/<Word>_<take>.mp4            -> id Word_<take>
  - new_video:  <partition>/New_videos/<Word>_<session>_<take>.mp4
                                                                -> id Word_<session>_<take>
Files matching neither (e.g. "Thaamara.mp4", "Svargam_2_3(1).mp4") are
skipped with a warning. IDs never collide between the two layouts.

Usage:
    uv run pipeline/build_manifest.py <path to Videos_Cropped dir> data/manifest.csv
    uv run pipeline/build_manifest.py <path to new_video dir>      data/manifest_new.csv
"""
import csv
import re
import sys
from pathlib import Path

import cv2

NEW_RE = re.compile(r"^(?P<word>.+)_(?P<session>\d+)_(?P<take>\d+)\.mp4$", re.IGNORECASE)
OLD_RE = re.compile(r"^(?P<word>.+)_(?P<take>\d+)\.mp4$", re.IGNORECASE)


def parse_name(name: str):
    """-> (word, session or "", take) or None. Two-number form is tried first
    so 'Aakasham_2_1' is not misread as word 'Aakasham_2', take 1."""
    m = NEW_RE.match(name)
    if m:
        return m.group("word"), m.group("session"), m.group("take")
    m = OLD_RE.match(name)
    if m:
        return m.group("word"), "", m.group("take")
    return None


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return dict(fps=fps, frame_count=frame_count, duration_sec=frame_count / fps, width=width, height=height)


def main(videos_root: str, out_csv: str):
    videos_root = Path(videos_root)
    partitions = sorted(p for p in videos_root.iterdir() if p.is_dir())
    if not partitions:
        raise SystemExit(f"no partition dirs found under {videos_root}")

    seen = {}  # video_id -> path (to detect collisions across partitions)
    rows = []
    for part in partitions:
        for mp4 in sorted(part.rglob("*.mp4")):
            parsed = parse_name(mp4.name)
            if not parsed:
                print(f"  ! skipping unrecognized filename: {mp4}")
                continue
            word, session, take = parsed
            video_id = f"{word}_{session}_{take}" if session else f"{word}_{take}"

            if video_id in seen:
                raise SystemExit(
                    f"DUPLICATE video_id '{video_id}': {seen[video_id]}  vs  {mp4}\n"
                    f"Partitions are supposed to be non-overlapping uploads -- investigate before continuing."
                )
            seen[video_id] = mp4

            base = dict(video_id=video_id, word=word, session=session, take=take,
                        path=str(mp4), partition=part.name)
            info = probe(mp4)
            if info is None:
                rows.append(dict(base, fps="", frame_count="", duration_sec="",
                                 width="", height="", status="unreadable"))
                print(f"  ! unreadable: {mp4}")
                continue

            status = "too_short" if info["duration_sec"] < 0.3 else "ok"
            rows.append(dict(base, fps=round(info["fps"], 3),
                             frame_count=info["frame_count"], duration_sec=round(info["duration_sec"], 3),
                             width=info["width"], height=info["height"], status=status))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "word", "session", "take", "path", "partition",
                                                "fps", "frame_count", "duration_sec", "width", "height", "status"])
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_words = len(set(r["word"] for r in rows))
    print(f"\nWrote {len(rows)} rows ({n_ok} ok) across {n_words} words to {out_path}")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print(f"Non-ok rows ({len(bad)}):")
        for r in bad:
            print(f"  {r['video_id']}: {r['status']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <Videos_Cropped dir> <out manifest.csv>")
    main(sys.argv[1], sys.argv[2])
