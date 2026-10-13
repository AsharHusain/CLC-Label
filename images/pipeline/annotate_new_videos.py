# /// script
# requires-python = "<3.13"
# dependencies = [
#     "mediapipe==0.10.14",
#     "opencv-python-headless",
#     "numpy",
#     "pympi-ling",
# ]
# ///
"""Annotate a folder of new videos, one by one, end to end.

For each Word_<session>_<take>.mp4 found under VIDEO_DIR (any depth) it runs:
    keypoints  -> keypoints/<id>.npz      (extract_keypoints.py)
    segments   -> segments/<id>.json      (detect_boundaries.py, uses labels/ledger.csv)
    ELAN draft -> elan/<id>.eaf           (export_elan.py)
and records the video in data/manifest_new.csv.

Safe to stop and rerun: finished videos (an .eaf already exists) are skipped.
Words with no entry in labels/ledger.csv are skipped (nothing to annotate
against) and listed at the end -- add them to the ledger and rerun.

Usage (run from images/):
    python pipeline/annotate_new_videos.py /path/to/new_video
    python pipeline/annotate_new_videos.py /path/to/new_video --only Aakasham Aana
    python pipeline/annotate_new_videos.py /path/to/new_video --limit 3     # quick test

Afterwards, to build the combined CSV + QA report and review queue:
    python pipeline/export_elan.py --all --manifest data/manifest_new.csv --queue-out review/queue_new.csv
    python pipeline/export_csv.py --manifest data/manifest_new.csv --out mudra_segments_new.csv --qa-out qa_report_new.txt
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_manifest import parse_name
from classify import PrototypeDB
from detect_boundaries import load_ledger, process_video
from export_elan import build_eaf
from extract_keypoints import extract

import numpy as np

MANIFEST_FIELDS = ["video_id", "word", "session", "take", "path"]


def find_videos(root: Path, only):
    vids, skipped = [], []
    for p in sorted(root.rglob("*.mp4")):
        parsed = parse_name(p.name)
        if not parsed:
            skipped.append(p)
            continue
        word, session, take = parsed
        if only and word not in only:
            continue
        vid = f"{word}_{session}_{take}" if session else f"{word}_{take}"
        vids.append(dict(video_id=vid, word=word, session=session, take=take, path=str(p.resolve())))
    return vids, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--ledger", default="labels/ledger.csv")
    ap.add_argument("--prototypes", default="pipeline/prototypes.npz")
    ap.add_argument("--manifest-out", default="data/manifest_new.csv")
    ap.add_argument("--only", nargs="*", help="only these words")
    ap.add_argument("--limit", type=int, help="stop after N newly annotated videos")
    ap.add_argument("--redo", action="store_true", help="re-annotate even if an .eaf exists")
    args = ap.parse_args()

    ledger = load_ledger(args.ledger)
    db = PrototypeDB(args.prototypes)
    vids, skipped = find_videos(Path(args.video_dir), set(args.only or []))
    for p in skipped:
        print(f"! ignoring unrecognised filename: {p.name}")

    seen = {}
    for v in vids:
        if v["video_id"] in seen:
            raise SystemExit(f"DUPLICATE video_id {v['video_id']}:\n  {seen[v['video_id']]}\n  {v['path']}")
        seen[v["video_id"]] = v["path"]

    for d in ("keypoints", "segments", "elan"):
        Path(d).mkdir(exist_ok=True)

    no_ledger, failed, done = set(), [], []
    manifest = []
    todo = 0
    t_start = time.time()
    for i, v in enumerate(vids, 1):
        vid, word = v["video_id"], v["word"]
        if word not in ledger:
            no_ledger.add(word)
            continue
        manifest.append(v)
        eaf_path = Path("elan") / f"{vid}.eaf"
        if eaf_path.exists() and not args.redo:
            continue
        if args.limit is not None and todo >= args.limit:
            continue
        todo += 1
        print(f"[{i}/{len(vids)}] {vid}", flush=True)
        t0 = time.time()
        try:
            kp_path = Path("keypoints") / f"{vid}.npz"
            if not kp_path.exists():
                np.savez(kp_path, **extract(v["path"]))
            seg = process_video(vid, word, kp_path, ledger, db)
            Path("segments", f"{vid}.json").write_text(json.dumps(seg, indent=2))
            build_eaf(seg, v["path"]).to_file(str(eaf_path))
        except Exception:
            traceback.print_exc()
            failed.append(vid)
            continue
        confs = [h["confidence"] for h in seg["hands"].values() if h["confidence"] is not None]
        print(f"    done in {time.time() - t0:.0f}s, min confidence {min(confs) if confs else None}"
              f"{'  FLAGS: ' + str(seg['flags']) if seg['flags'] else ''}", flush=True)
        done.append(vid)

    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)

    print(f"\nannotated {len(done)} video(s) this run in {(time.time() - t_start) / 60:.1f} min; manifest -> {out}")
    if no_ledger:
        print(f"skipped (no ledger entry): {', '.join(sorted(no_ledger))}")
    if failed:
        print(f"FAILED ({len(failed)}): {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
