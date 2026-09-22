# /// script
# requires-python = "<3.13"
# dependencies = [
#     "mediapipe==0.10.14",
#     "opencv-python-headless",
#     "numpy",
#     "pympi-ling",
# ]
# ///
"""Level 2: auto-draft LEFT/RIGHT ELAN tiers for one word-video.

Given the mudra you already know from the ledger (initial + final, per
hand), this finds *when* each hand transitions between them -- it does not
guess which mudra is shown; your team already determined that by eye.

Usage:
    uv run label_video.py \
        --video Aakasham.mp4 \
        --left-initial Hamsapaksham --left-final Ardhachandram \
        --right-initial Hamsapaksham --right-final Ardhachandram \
        --prototypes full_prototypes.npz \
        --out Aakasham.eaf

Ledger conventions:
  - Ledger cell is "-" (dash)  -> pass "-" as that hand's --*-initial. The
    hand starts already in its final pose; no transition segment is drawn.
  - Ledger says "no mudra"     -> omit --*-final for that hand entirely
    (leave it blank). That tier gets no annotation for this word.

IMPORTANT: MediaPipe's Left/Right handedness assumes a normal (non-mirrored)
camera view. If your source videos are mirrored (e.g. exported from a
selfie camera), pass --mirror. Verify against ONE known video (e.g. Aakasham,
where you already know both hands go Hamsapaksham -> Ardhachandram) before
trusting a whole batch.

This script AUTO-DRAFTS boundaries; it does not replace opening the result
in ELAN and checking it by eye (Level 3). Treat every timestamp it writes as
a suggestion, especially where it prints "auto-detect failed".
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import pympi

sys.path.insert(0, str(Path(__file__).parent))
from mudra_common import normalize_landmarks
from classify import PrototypeDB

SMOOTH_WINDOW = 5      # frames, moving-average smoothing of the similarity signal
MARGIN = 0.03           # min similarity gap to call a side "dominant"
HOLD_FRAMES = 3         # consecutive dominant samples required to confirm a hold


def track_hand_similarities(video_path, side, db, initial_label, final_label, sample_every_n):
    """Sample `video_path`, and for whichever detected hand matches `side`
    ('Left' or 'Right') each sampled frame, record its similarity to the
    initial and final reference mudras. Returns (duration_sec, times, sim_i, sim_f).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    times, sim_i, sim_f = [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            t = frame_idx / fps
            if result.multi_hand_landmarks:
                for lm, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    if handedness.classification[0].label != side:
                        continue
                    xyz = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                    vec = normalize_landmarks(xyz)
                    times.append(t)
                    sim_i.append(db.similarity_to(vec, initial_label) if initial_label and initial_label != '-' else 0.0)
                    sim_f.append(db.similarity_to(vec, final_label))
                    break
        frame_idx += 1
    cap.release()
    hands.close()
    duration = frame_idx / fps
    return duration, np.array(times), np.array(sim_i), np.array(sim_f)


def smooth(x, window=SMOOTH_WINDOW):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def find_transition(times, sim_i, sim_f):
    """Return (t1, t2): last confidently-initial time, first confidently-final
    time that then holds for HOLD_FRAMES consecutive samples."""
    if len(times) == 0:
        return None, None
    diff = smooth(sim_f - sim_i)

    t1 = None
    for k in range(len(diff)):
        if diff[k] < -MARGIN:
            t1 = times[k]

    t2 = None
    for k in range(len(diff) - HOLD_FRAMES + 1):
        if diff[k] > MARGIN and bool(np.all(diff[k:k + HOLD_FRAMES] > MARGIN)):
            t2 = times[k]
            break

    return t1, t2


def build_hand_segments(duration, initial_label, final_label, t1, t2):
    """(initial_label, transition, final_label, boundaries) -> list of
    (start_sec, end_sec, label) covering [0, duration]."""
    if not initial_label or initial_label == '-':
        return [(0.0, duration, final_label)]
    if initial_label == final_label:
        return [(0.0, duration, final_label)]
    if t1 is None or t2 is None or t1 >= t2:
        # Couldn't confidently locate the transition. Don't guess a boundary
        # silently -- hand the whole clip to the human as one block so it's
        # obvious this one needs manual marking in ELAN.
        return [(0.0, duration, f'{initial_label} -> {final_label} (auto-detect failed, mark by hand)')]
    return [
        (0.0, t1, initial_label),
        (t1, t2, 'transition'),
        (t2, duration, final_label),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--left-initial', default='')
    p.add_argument('--left-final', default='')
    p.add_argument('--right-initial', default='')
    p.add_argument('--right-final', default='')
    p.add_argument('--prototypes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mirror', action='store_true',
                    help="swap Left/Right if your source video is mirrored")
    p.add_argument('--sample-every-n', type=int, default=2,
                    help='process every Nth frame (speed/accuracy tradeoff)')
    args = p.parse_args()

    db = PrototypeDB(args.prototypes)
    left_side = 'Right' if args.mirror else 'Left'
    right_side = 'Left' if args.mirror else 'Right'

    eaf = pympi.Elan.Eaf()
    eaf.remove_tier('default')
    eaf.add_tier('LEFT')
    eaf.add_tier('RIGHT')
    eaf.add_linked_file(str(Path(args.video).resolve()), mimetype='video/mp4')

    for tier, side, init_label, final_label in [
        ('LEFT', left_side, args.left_initial, args.left_final),
        ('RIGHT', right_side, args.right_initial, args.right_final),
    ]:
        if not final_label:
            continue  # "no mudra" for this hand on this word
        dur, times, sim_i, sim_f = track_hand_similarities(
            args.video, side, db, init_label, final_label, args.sample_every_n
        )
        t1, t2 = find_transition(times, sim_i, sim_f)
        for start, end, label in build_hand_segments(dur, init_label, final_label, t1, t2):
            eaf.add_annotation(tier, int(start * 1000), int(end * 1000), label)
        print(f'{tier}: t1={t1}, t2={t2}, duration={dur:.2f}s ({len(times)} tracked samples)')

    eaf.to_file(args.out)
    print(f'wrote draft {args.out} -- open in ELAN to verify/correct boundaries')


if __name__ == '__main__':
    main()
