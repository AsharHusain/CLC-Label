# /// script
# requires-python = "<3.13"
# dependencies = [
#     "mediapipe==0.10.14",
#     "opencv-python-headless",
#     "numpy",
# ]
# ///
"""Batch driver for extract_keypoints.py: run W3 over every ok row of a
manifest, optionally one shard of it (for SLURM array jobs / parallel runs).

Already-extracted videos are skipped (extract_keypoints.main checks the
.npz), so an interrupted job can simply be resubmitted.

Usage (run from images/):
    python pipeline/extract_all.py --manifest data/manifest_new.csv \
        --out-dir keypoints --shard 0 --num-shards 8
"""
import argparse
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_keypoints import main as extract_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_new.csv")
    ap.add_argument("--out-dir", default="keypoints")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    rows.sort(key=lambda r: r["video_id"])
    mine = rows[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(rows)} videos")

    failed = []
    for r in mine:
        try:
            extract_one(r["path"], str(Path(args.out_dir) / f"{r['video_id']}.npz"))
        except Exception:
            traceback.print_exc()
            failed.append(r["video_id"])
    if failed:
        print(f"FAILED ({len(failed)}): {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
