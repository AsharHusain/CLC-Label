# /// script
# requires-python = "<3.13"
# dependencies = ["pympi-ling"]
# ///
"""W6: consolidate segments/*.json (+ any human-corrected elan/*.eaf) into
the final mudra_segments.csv, then run the QA gate.

A human-corrected .eaf (tracked via review/reviewed.csv -- one video_id per
line, written by whoever finishes reviewing it in ELAN) always overrides the
auto segments for that video.

Usage:
    uv run pipeline/export_csv.py --out mudra_segments.csv
"""
import argparse
import csv
import json
from pathlib import Path

import pympi


def load_reviewed(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def rows_from_auto(vid, word, take, segments_json):
    rows = []
    for hand, info in segments_json.get("hands", {}).items():
        for i, seg in enumerate(info["segments"]):
            rows.append(dict(
                video_id=vid, word=word, take=take, hand=hand, segment_index=i,
                start_sec=seg["start_sec"], end_sec=seg["end_sec"],
                duration_sec=round(seg["end_sec"] - seg["start_sec"], 3),
                label=seg["label"], kind=seg["kind"], confidence=seg.get("confidence"),
                review_status="auto_failed" if "auto-detect failed" in seg["label"] else "auto_accepted",
                annotator="",
            ))
    return rows


def rows_from_eaf(vid, word, take, eaf_path, annotator=""):
    eaf = pympi.Elan.Eaf(str(eaf_path))
    rows = []
    for hand in ("LEFT", "RIGHT"):
        if hand not in eaf.tiers:
            continue
        anns = sorted(eaf.get_annotation_data_for_tier(hand))
        for i, (start_ms, end_ms, value) in enumerate(anns):
            rows.append(dict(
                video_id=vid, word=word, take=take, hand=hand, segment_index=i,
                start_sec=round(start_ms / 1000, 3), end_sec=round(end_ms / 1000, 3),
                duration_sec=round((end_ms - start_ms) / 1000, 3),
                label=value, kind="transition" if value == "transition" else "hold",
                confidence="", review_status="human_verified", annotator=annotator,
            ))
    return rows


def qa_check(rows):
    errors = []
    by_hand = {}
    for r in rows:
        by_hand.setdefault((r["video_id"], r["hand"]), []).append(r)

    for (vid, hand), segs in by_hand.items():
        segs = sorted(segs, key=lambda r: r["start_sec"])
        for i, r in enumerate(segs):
            if r["duration_sec"] < 0.1:
                errors.append(f"{vid}/{hand}: segment {i} duration {r['duration_sec']}s < 0.1s")
            if r["label"] == "":
                errors.append(f"{vid}/{hand}: segment {i} has empty label")
        for i in range(len(segs) - 1):
            gap = segs[i + 1]["start_sec"] - segs[i]["end_sec"]
            if gap > 0.05:
                errors.append(f"{vid}/{hand}: gap of {gap:.3f}s between segment {i} and {i+1}")
            if gap < -0.01:
                errors.append(f"{vid}/{hand}: overlap of {-gap:.3f}s between segment {i} and {i+1}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--segments-dir", default="segments")
    ap.add_argument("--eaf-dir", default="elan")
    ap.add_argument("--reviewed", default="review/reviewed.csv")
    ap.add_argument("--out", default="mudra_segments.csv")
    ap.add_argument("--qa-out", default="qa_report.txt")
    args = ap.parse_args()

    manifest_rows = {}
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            manifest_rows[r["video_id"]] = r

    reviewed = load_reviewed(Path(args.reviewed))

    all_rows = []
    n_auto = n_human = n_missing = 0
    for vid, m in sorted(manifest_rows.items()):
        word, take = m["word"], m["take"]
        eaf_path = Path(args.eaf_dir) / f"{vid}.eaf"
        seg_path = Path(args.segments_dir) / f"{vid}.json"

        if vid in reviewed and eaf_path.exists():
            all_rows.extend(rows_from_eaf(vid, word, take, eaf_path))
            n_human += 1
        elif seg_path.exists():
            all_rows.extend(rows_from_auto(vid, word, take, json.loads(seg_path.read_text())))
            n_auto += 1
        else:
            n_missing += 1

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        fieldnames = ["video_id", "word", "take", "hand", "segment_index", "start_sec", "end_sec",
                      "duration_sec", "label", "kind", "confidence", "review_status", "annotator"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    errors = qa_check(all_rows)
    qa_path = Path(args.qa_out)
    with open(qa_path, "w") as f:
        f.write(f"videos: {n_human} human-verified, {n_auto} auto, {n_missing} missing segments\n")
        f.write(f"total segment rows: {len(all_rows)}\n")
        f.write(f"QA errors: {len(errors)}\n\n")
        for e in errors:
            f.write(f"  {e}\n")

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    print(f"videos: {n_human} human-verified, {n_auto} auto-accepted, {n_missing} missing")
    print(f"QA: {len(errors)} error(s) -- see {qa_path}")


if __name__ == "__main__":
    main()
