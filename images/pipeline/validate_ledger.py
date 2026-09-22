# /// script
# requires-python = "<3.13"
# dependencies = ["numpy"]
# ///
"""W2 validator: checks labels/ledger.csv against the canonical 24 mudras,
checks seq_index contiguity per (word,hand), and cross-checks against
mudras.csv to surface conflicts (never silently resolves them).

Usage:
    uv run pipeline/validate_ledger.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mudra_common import DISPLAY_TO_DATASET

ROOT = Path(__file__).parent.parent
LEDGER = ROOT / "labels" / "ledger.csv"
MUDRAS_CSV = ROOT.parent / "mudras.csv"  # /home/ash/Projects/CLC/mudras.csv

VALID_LABELS = set(DISPLAY_TO_DATASET.keys()) | {"no_mudra"}


def load_ledger():
    with open(LEDGER, newline="") as f:
        return list(csv.DictReader(f))


def check_labels(rows):
    errors = []
    for r in rows:
        if r["mudra"].strip().lower() not in VALID_LABELS:
            errors.append(f"{r['word']}/{r['hand']}/seq{r['seq_index']}: unknown mudra '{r['mudra']}'")
    return errors


def check_sequences(rows):
    errors = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["word"], r["hand"])].append(int(r["seq_index"]))
    for (word, hand), seqs in groups.items():
        seqs_sorted = sorted(seqs)
        expected = list(range(len(seqs)))
        if seqs_sorted != expected:
            errors.append(f"{word}/{hand}: seq_index {seqs_sorted} is not contiguous from 0 (expected {expected})")
        if len(seqs) != len(set(seqs)):
            errors.append(f"{word}/{hand}: duplicate seq_index values {seqs}")
    return errors


def check_low_confidence(rows):
    return [f"{r['word']}/{r['hand']}/seq{r['seq_index']}: confidence={r['confidence']} -- {r['notes']}"
            for r in rows if r["confidence"] != "high"]


def cross_check_csv(rows):
    """Compare each word's set of mudras against mudras.csv's columns 2-4.
    mudras.csv is a looser source (no hand/order structure confirmed), so
    this is a set-overlap comparison, not a strict equality check."""
    if not MUDRAS_CSV.exists():
        return [f"mudras.csv not found at {MUDRAS_CSV} -- skipping cross-check"]

    csv_mudras = {}
    with open(MUDRAS_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            word = row[0].strip().rstrip(" (?)").strip()
            vals = {v.strip() for v in row[1:4] if v.strip()}
            if vals:
                csv_mudras[word] = vals

    ledger_words = defaultdict(set)
    for r in rows:
        if r["mudra"] != "no_mudra":
            ledger_words[r["word"]].add(r["mudra"])

    conflicts = []
    for word, ledger_set in ledger_words.items():
        csv_set = csv_mudras.get(word)
        if csv_set is None:
            continue
        if ledger_set != csv_set:
            conflicts.append(f"{word}: ledger={sorted(ledger_set)} vs mudras.csv={sorted(csv_set)}")
    return conflicts


def main():
    rows = load_ledger()
    print(f"Loaded {len(rows)} ledger rows across {len(set(r['word'] for r in rows))} words\n")

    label_errors = check_labels(rows)
    seq_errors = check_sequences(rows)
    low_conf = check_low_confidence(rows)
    conflicts = cross_check_csv(rows)

    def report(title, items, fatal):
        print(f"--- {title} ({len(items)}) ---")
        for it in items:
            print(f"  {'FAIL' if fatal else 'warn'}: {it}")
        print()

    report("Unknown mudra names", label_errors, fatal=True)
    report("seq_index contiguity errors", seq_errors, fatal=True)
    report("Low-confidence / flagged rows (human review recommended)", low_conf, fatal=False)
    report("Conflicts with mudras.csv", conflicts, fatal=False)

    n_fatal = len(label_errors) + len(seq_errors)
    if n_fatal:
        print(f"VALIDATION FAILED: {n_fatal} fatal error(s)")
        sys.exit(1)
    print("VALIDATION PASSED (fatal checks clean; review warnings above)")


if __name__ == "__main__":
    main()
