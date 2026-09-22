# /// script
# requires-python = "<3.13"
# dependencies = ["numpy"]
# ///
"""W4: turn per-frame keypoints + the known mudra sequence (ledger) into
timestamped segments per hand.

Generalizes label_video.py's 2-mudra crossover logic to N mudras (walks
consecutive pairs), adds a motion gate using the wrist-position body
keypoints from extract_keypoints.py (a similarity crossover with no
corresponding wrist-speed peak is distrusted, not trusted blindly), and
emits an honest confidence score that pipeline/detect_boundaries drives
the W5 review queue with.

Usage:
    uv run pipeline/detect_boundaries.py <video_id> \
        --keypoints keypoints/<video_id>.npz \
        --ledger labels/ledger.csv \
        --manifest data/manifest.csv \
        --prototypes pipeline/prototypes.npz \
        --out segments/<video_id>.json

    uv run pipeline/detect_boundaries.py --all   # process every manifest row
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from classify import PrototypeDB

SMOOTH_WINDOW = 15             # wide smoothing -- pilot data showed frame-to-frame similarity noise
                                # (both candidate mudras often score 0.88-0.99) that a narrow window
                                # doesn't suppress
MARGIN = 0.015                 # smaller than the original 0.03: real separations are often <0.07 total
HOLD_FRAMES = 5
LOCAL_RADIUS_SEC = 2.0         # bound the t1/t2 refinement search to this far from the changepoint,
                                # so a noise spike elsewhere in the clip can't hijack the boundary
MOTION_PAD_SEC = 0.15          # widen the search window around [t1,t2] when looking for a motion peak
MOTION_PEAK_RATIO = 1.4        # required peak speed vs. median hold-speed to call it "real motion"
MIN_SEPARATION = 0.02          # minimum (mean-after - mean-before) at the changepoint to trust it at all
L_WRIST_IDX, R_WRIST_IDX = 4, 5  # index into body (n,6,2), order = BODY_ORDER in extract_keypoints.py


def load_ledger(path):
    """word -> hand -> ordered list of mudra names (seq_index order)."""
    seqs = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            seqs.setdefault(r["word"], {}).setdefault(r["hand"], {})[int(r["seq_index"])] = r["mudra"]
    return {w: {h: [seq[i] for i in sorted(seq)] for h, seq in hands.items()} for w, hands in seqs.items()}


def smooth_nan_aware(x, window=SMOOTH_WINDOW):
    """Moving average that treats NaN as 'no data' rather than poisoning
    neighboring windows (plain np.convolve on NaN-containing arrays spreads
    the NaN outward)."""
    valid = ~np.isnan(x)
    filled = np.nan_to_num(x)
    kernel = np.ones(window) / window
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out


def wrist_speed(body, wrist_idx, fps):
    pos = body[:, wrist_idx, :]  # (n, 2), may contain NaN
    valid = ~np.isnan(pos).any(axis=1)
    speed = np.full(len(pos), np.nan)
    for i in range(1, len(pos)):
        if valid[i] and valid[i - 1]:
            speed[i] = np.linalg.norm(pos[i] - pos[i - 1]) * fps
    return speed


def find_crossover(times, sim_a, sim_b):
    """Two-stage crossover: first find the single changepoint that maximizes
    separation between the mean signal before/after it (robust to noisy
    individual-frame crossings elsewhere in the clip -- found necessary on
    real pilot data, where both candidate mudras often score 0.88-0.99 and
    a naive 'last negative frame anywhere' scan gets hijacked by late noise).
    Then refine t1/t2 locally around that changepoint only.

    Returns (t1, t2, separation) where separation = mean(diff after) -
    mean(diff before) at the chosen changepoint -- this IS the primary
    confidence signal, not a per-boundary margin.
    """
    n = len(times)
    if n < 2 * HOLD_FRAMES:
        return None, None, 0.0

    diff = smooth_nan_aware(sim_b - sim_a)
    if np.isnan(diff).all():
        return None, None, 0.0

    best_s, best_sep = None, -np.inf
    lo, hi = HOLD_FRAMES, n - HOLD_FRAMES
    if hi <= lo:
        return None, None, 0.0
    for s in range(lo, hi):
        before = diff[:s]
        after = diff[s:]
        before, after = before[~np.isnan(before)], after[~np.isnan(after)]
        if len(before) < 3 or len(after) < 3:
            continue
        sep = after.mean() - before.mean()
        if sep > best_sep:
            best_sep, best_s = sep, s

    if best_s is None or best_sep < MIN_SEPARATION:
        return None, None, max(best_sep, 0.0)

    fps_est = n / (times[-1] - times[0] + 1e-9)
    radius = max(HOLD_FRAMES, int(LOCAL_RADIUS_SEC * fps_est))
    lo_r, hi_r = max(0, best_s - radius), min(n, best_s + radius)

    t1 = None
    for k in range(lo_r, best_s + 1):
        if not np.isnan(diff[k]) and diff[k] < -MARGIN:
            t1 = times[k]
    t2 = None
    for k in range(best_s, hi_r - HOLD_FRAMES + 1):
        window = diff[k:k + HOLD_FRAMES]
        if not np.isnan(window).any() and np.all(window > MARGIN):
            t2 = times[k]
            break

    if t1 is None:
        t1 = times[best_s]
    if t2 is None:
        t2 = times[best_s]
    return t1, t2, float(best_sep)


def motion_agreement(speed, times, t1, t2):
    """Does a wrist-speed peak actually occur in [t1,t2] (padded)? Returns
    (agrees: bool, peak_ratio: float) -- ratio of in-window peak speed to
    the median speed just outside the window (the holds)."""
    if t1 is None or t2 is None or np.isnan(speed).all():
        return False, 0.0
    in_win = (times >= t1 - MOTION_PAD_SEC) & (times <= t2 + MOTION_PAD_SEC)
    out_win = ~in_win
    in_speeds = speed[in_win]
    out_speeds = speed[out_win]
    in_speeds = in_speeds[~np.isnan(in_speeds)]
    out_speeds = out_speeds[~np.isnan(out_speeds)]
    if len(in_speeds) == 0 or len(out_speeds) == 0:
        return False, 0.0
    peak = np.max(in_speeds)
    baseline = np.median(out_speeds) + 1e-6
    ratio = float(peak / baseline)
    return bool(ratio >= MOTION_PEAK_RATIO), ratio


def process_hand(vecs, present, conf, body, fps, mudra_seq, db, wrist_idx):
    """vecs: (n,42) per-frame hand vector (NaN where absent).
    Returns (segments: list of dict, hand_confidence: float, detection_rate: float)."""
    n = len(vecs)
    times = np.arange(n) / fps
    detection_rate = float(present.mean()) if n else 0.0

    if len(mudra_seq) == 1 and mudra_seq[0] == "no_mudra":
        return [], None, None  # tier not used for this word

    if len(mudra_seq) == 1:
        return ([{"start_sec": 0.0, "end_sec": n / fps, "label": mudra_seq[0],
                   "kind": "hold", "confidence": round(detection_rate, 3)}],
                round(detection_rate, 3), detection_rate)

    speed = wrist_speed(body, wrist_idx, fps)
    segments = []
    boundary_confidences = []
    cursor = 0.0
    valid_mask = present  # bool (n,)
    reached_end = False

    for i in range(len(mudra_seq) - 1):
        label_a, label_b = mudra_seq[i], mudra_seq[i + 1]
        sim_a = np.full(n, np.nan)
        sim_b = np.full(n, np.nan)
        for k in range(n):
            if valid_mask[k]:
                sim_a[k] = db.similarity_to(vecs[k], label_a)
                sim_b[k] = db.similarity_to(vecs[k], label_b)

        ok = times >= cursor
        t_ok, a_ok, b_ok = times[ok], sim_a[ok], sim_b[ok]
        t1, t2, separation = find_crossover(t_ok, a_ok, b_ok) if len(t_ok) else (None, None, 0.0)

        if t1 is None or t2 is None or t1 >= t2:
            segments.append({"start_sec": round(cursor, 3), "end_sec": round(n / fps, 3),
                              "label": f"{label_a} -> {label_b} (auto-detect failed)",
                              "kind": "hold", "confidence": 0.0})
            boundary_confidences.append(0.0)
            cursor = n / fps
            reached_end = True
            break

        agrees, peak_ratio = motion_agreement(speed, times, t1, t2)
        # separation is the primary signal (mean-after minus mean-before similarity gap
        # at the changepoint) -- pilot data showed real transitions with total separation
        # as low as ~0.07, so 0.10 is treated as "fully confident" rather than a larger number
        separation_score = min(1.0, max(0.0, separation) / 0.10)
        motion_score = min(1.0, peak_ratio / 2.5) if agrees else 0.3 * min(1.0, peak_ratio / 2.5)
        boundary_conf = round(0.6 * separation_score + 0.4 * motion_score, 3)

        segments.append({"start_sec": round(cursor, 3), "end_sec": round(t1, 3), "label": label_a,
                          "kind": "hold", "confidence": round(detection_rate, 3)})
        segments.append({"start_sec": round(t1, 3), "end_sec": round(t2, 3), "label": "transition",
                          "kind": "transition", "confidence": boundary_conf,
                          "separation": round(separation, 4),
                          "motion_agrees": agrees, "motion_peak_ratio": round(peak_ratio, 2)})
        boundary_confidences.append(boundary_conf)
        cursor = t2

    if not reached_end:
        segments.append({"start_sec": round(cursor, 3), "end_sec": round(n / fps, 3), "label": mudra_seq[-1],
                          "kind": "hold", "confidence": round(detection_rate, 3)})

    hand_confidence = round(min(boundary_confidences) * (0.5 + 0.5 * detection_rate), 3)
    return segments, hand_confidence, detection_rate


def process_video(video_id, word, kp_path, ledger, db):
    d = np.load(kp_path)
    fps = float(d["fps"])
    n = len(d["frame_indices"])
    duration = n / fps

    word_ledger = ledger.get(word, {})
    hands_out = {}
    flags = []
    for tier, slot, wrist_idx in [("LEFT", 0, L_WRIST_IDX), ("RIGHT", 1, R_WRIST_IDX)]:
        mudra_seq = word_ledger.get(tier)
        if not mudra_seq:
            flags.append(f"no ledger entry for {word}/{tier}")
            continue
        vecs = d["hand_vecs"][:, slot, :]
        present = d["hand_present"][:, slot]
        conf = d["hand_conf"][:, slot]
        segs, hand_conf, det_rate = process_hand(vecs, present, conf, d["body"], fps, mudra_seq, db, wrist_idx)
        if segs:
            hands_out[tier] = {"segments": segs, "confidence": hand_conf, "detection_rate": round(det_rate, 3)}
            if det_rate < 0.6:
                flags.append(f"{tier} hand-detection rate {det_rate:.1%} < 60%")

    return {
        "video_id": video_id, "word": word, "fps": fps, "duration_sec": round(duration, 3),
        "hands": hands_out, "flags": flags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", nargs="?")
    ap.add_argument("--keypoints")
    ap.add_argument("--ledger", default="labels/ledger.csv")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--prototypes", default="pipeline/prototypes.npz")
    ap.add_argument("--out")
    ap.add_argument("--keypoints-dir", default="keypoints")
    ap.add_argument("--out-dir", default="segments")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    ledger = load_ledger(args.ledger)
    db = PrototypeDB(args.prototypes)

    manifest_rows = {}
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            manifest_rows[r["video_id"]] = r

    if args.all:
        targets = list(manifest_rows.keys())
    else:
        if not args.video_id:
            raise SystemExit("provide a video_id or --all")
        targets = [args.video_id]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    for vid in targets:
        kp_path = Path(args.keypoints) if (args.keypoints and not args.all) else Path(args.keypoints_dir) / f"{vid}.npz"
        if not kp_path.exists():
            n_skip += 1
            continue
        word = manifest_rows[vid]["word"]
        result = process_video(vid, word, kp_path, ledger, db)
        out_path = Path(args.out) if (args.out and not args.all) else Path(args.out_dir) / f"{vid}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        n_done += 1
        confs = [h["confidence"] for h in result["hands"].values()]
        min_conf = min(confs) if confs else None
        flag_str = f" FLAGS: {result['flags']}" if result["flags"] else ""
        print(f"{vid}: min_hand_confidence={min_conf}{flag_str}")

    print(f"\n{n_done} processed, {n_skip} skipped (no keypoints file yet)")


if __name__ == "__main__":
    main()
