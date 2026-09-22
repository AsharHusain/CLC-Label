# /// script
# requires-python = "<3.13"
# dependencies = ["pympi-ling", "numpy"]
# ///
"""W5: turn segments/<video_id>.json into a draft .eaf (LEFT/RIGHT tiers,
controlled vocabulary of the 24 canonical mudras + transition + no_mudra),
and build review/queue.csv ranked worst-confidence-first.

Usage:
    uv run pipeline/export_elan.py --all
    uv run pipeline/export_elan.py <video_id> --video <path to mp4>
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pympi

sys.path.insert(0, str(Path(__file__).parent))
from mudra_common import MUDRAS


CV_ID = "MudraVocab"
EXTRA_CV_ENTRIES = ["transition", "no_mudra", "UNCERTAIN"]


def build_eaf(segments_json: dict, video_path: str) -> "pympi.Elan.Eaf":
    eaf = pympi.Elan.Eaf()
    # pympi-ling has a known bug in its DATE formatting: it does `ctz % 3600`
    # (seconds) where it means minutes, producing an XSD-invalid timezone
    # offset like "+05:1800" for any non-whole-hour zone (e.g. IST +5:30).
    # pympi's own reader doesn't validate this, but ELAN's strict XML
    # validator rejects it and NPEs on load. Overwrite with a correct value.
    eaf.adocument["DATE"] = datetime.now().astimezone().isoformat()
    eaf.remove_tier("default")

    eaf.add_language("und")
    eaf.add_controlled_vocabulary(CV_ID)
    # ELAN's EAF parser requires a DESCRIPTION on the CV before entries --
    # without it, ELAN's internal per-language entry index is never
    # initialized and it NPEs (BasicControlledVocabulary.addEntry) the
    # moment it tries to load the first CV_ENTRY_ML.
    eaf.add_cv_description(CV_ID, "und", "Canonical Kathakali mudra names")
    cv_values = [disp for _, disp in MUDRAS] + EXTRA_CV_ENTRIES
    for i, value in enumerate(cv_values):
        eaf.add_cv_entry(CV_ID, f"cve_{i}", [(value, "und", None)])
    eaf.add_linguistic_type("MudraType")
    eaf.linguistic_types["MudraType"]["CONTROLLED_VOCABULARY_REF"] = CV_ID

    for tier in ("LEFT", "RIGHT"):
        eaf.add_tier(tier, ling="MudraType")
    if video_path:
        mimetype = "video/mp4" if video_path.lower().endswith(".mp4") else None
        # ELAN's MEDIA_URL must be a proper file:// URI (Java's URI parser
        # requires a scheme) -- a bare filesystem path produces "URI is not
        # absolute" and the media silently fails to load.
        eaf.add_linked_file(Path(video_path).resolve().as_uri(), mimetype=mimetype)

    valid_labels = set(cv_values)
    for tier, info in segments_json.get("hands", {}).items():
        for seg in info["segments"]:
            start_ms = int(round(seg["start_sec"] * 1000))
            end_ms = int(round(seg["end_sec"] * 1000))
            if end_ms <= start_ms:
                continue
            # Detailed failure strings (e.g. "X -> Y (auto-detect failed)") are kept in
            # segments/*.json and the CSV for humans/scripts reading logs, but the .eaf
            # controlled vocabulary must stay strict so ELAN's dropdown/autocomplete
            # works uniformly -- anything not a real CV entry becomes UNCERTAIN here.
            label = seg["label"] if seg["label"] in valid_labels else "UNCERTAIN"
            eaf.add_annotation(tier, start_ms, end_ms, label)
    return eaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", nargs="?")
    ap.add_argument("--video")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--segments-dir", default="segments")
    ap.add_argument("--eaf-dir", default="elan")
    ap.add_argument("--queue-out", default="review/queue.csv")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    manifest_rows = {}
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            manifest_rows[r["video_id"]] = r

    if args.all:
        seg_files = sorted(Path(args.segments_dir).glob("*.json"))
        targets = [p.stem for p in seg_files]
    else:
        if not args.video_id:
            raise SystemExit("provide a video_id or --all")
        targets = [args.video_id]

    Path(args.eaf_dir).mkdir(parents=True, exist_ok=True)
    queue_rows = []
    n_done = 0
    for vid in targets:
        seg_path = Path(args.segments_dir) / f"{vid}.json"
        if not seg_path.exists():
            print(f"  ! no segments for {vid}, skipping")
            continue
        segments_json = json.loads(seg_path.read_text())
        video_path = args.video if (args.video and not args.all) else manifest_rows.get(vid, {}).get("path")

        eaf = build_eaf(segments_json, video_path)
        out_path = Path(args.eaf_dir) / f"{vid}.eaf"
        eaf.to_file(str(out_path))
        n_done += 1

        confs = [h["confidence"] for h in segments_json.get("hands", {}).values() if h["confidence"] is not None]
        min_conf = min(confs) if confs else 0.0
        queue_rows.append({
            "video_id": vid, "word": segments_json.get("word", ""),
            "min_confidence": round(min_conf, 3),
            "flags": "; ".join(segments_json.get("flags", [])),
            "eaf_path": str(out_path),
        })

    queue_rows.sort(key=lambda r: r["min_confidence"])
    queue_path = Path(args.queue_out)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "word", "min_confidence", "flags", "eaf_path"])
        writer.writeheader()
        writer.writerows(queue_rows)

    print(f"\nWrote {n_done} .eaf files to {args.eaf_dir}/")
    print(f"Wrote review queue ({len(queue_rows)} rows, sorted worst-confidence-first) to {queue_path}")
    print(f"\nControlled vocabulary for ELAN (24 canonical + transition + no_mudra):")
    print(", ".join(disp for _, disp in MUDRAS) + ", transition, no_mudra")


if __name__ == "__main__":
    main()
