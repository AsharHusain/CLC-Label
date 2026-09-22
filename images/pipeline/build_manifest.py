# /// script
# requires-python = "<3.13"
# dependencies = ["opencv-python-headless"]
# ///
"""W1: index the already-extracted video archive into a flat manifest.

Videos_Cropped1..6 are upload partitions, not duplicates -- flattens them
into one row per (word, take), failing loudly on any collision.

Usage:
    uv run pipeline/build_manifest.py <path to Videos_Cropped dir> data/manifest.csv
"""
import csv
import re
import sys
from pathlib import Path

import cv2

FNAME_RE = re.compile(r"^(?P<word>.+)_(?P<take>\d+)\.mp4$", re.IGNORECASE)


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
        for word_dir in sorted(p for p in part.iterdir() if p.is_dir()):
            word = word_dir.name
            for mp4 in sorted(word_dir.glob("*.mp4")):
                m = FNAME_RE.match(mp4.name)
                if not m:
                    print(f"  ! skipping unrecognized filename: {mp4}")
                    continue
                take = m.group("take")
                video_id = f"{word}_{take}"

                if video_id in seen:
                    raise SystemExit(
                        f"DUPLICATE video_id '{video_id}': {seen[video_id]}  vs  {mp4}\n"
                        f"Partitions are supposed to be non-overlapping uploads -- investigate before continuing."
                    )
                seen[video_id] = mp4

                info = probe(mp4)
                if info is None:
                    rows.append(dict(video_id=video_id, word=word, take=take, path=str(mp4),
                                      partition=part.name, fps="", frame_count="", duration_sec="",
                                      width="", height="", status="unreadable"))
                    print(f"  ! unreadable: {mp4}")
                    continue

                status = "too_short" if info["duration_sec"] < 0.3 else "ok"
                rows.append(dict(video_id=video_id, word=word, take=take, path=str(mp4),
                                  partition=part.name, fps=round(info["fps"], 3),
                                  frame_count=info["frame_count"], duration_sec=round(info["duration_sec"], 3),
                                  width=info["width"], height=info["height"], status=status))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "word", "take", "path", "partition",
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
