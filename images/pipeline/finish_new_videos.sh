#!/bin/bash
# Steps 4-6 for the new_video batch, run from anywhere after extraction finishes.
# Writes *_new outputs so the original mudra_segments.csv / qa_report.txt /
# review/queue.csv are not overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."            # -> images/
python pipeline/detect_boundaries.py --all --manifest data/manifest_new.csv
python pipeline/export_elan.py --all --manifest data/manifest_new.csv --queue-out review/queue_new.csv
python pipeline/export_csv.py --manifest data/manifest_new.csv --out mudra_segments_new.csv --qa-out qa_report_new.txt
