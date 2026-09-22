# /// script
# requires-python = "<3.13"
# dependencies = [
#     "mediapipe==0.10.14",
#     "opencv-python-headless",
#     "numpy",
# ]
# ///
"""W3: dense per-frame keypoint extraction for one video.

Design note (found during the Acchan pilot, not guessed): running MediaPipe
Hands on the full body-framed video frame finds a SECOND hand almost never
(measured 0.4% of frames with both hands, vs 95%+ with only one) because the
hands are small relative to the full-body frame. The fix is two-stage:
MediaPipe Pose locates each wrist on the full frame, then we crop a tight
region around each wrist (scaled by elbow-to-wrist distance) and run Hands
on that crop alone. This raised right-hand detection from ~1% to ~95% on
the pilot video.

This also sidesteps the Left/Right mirroring risk: we assign a hand to the
LEFT or RIGHT tier by which pose wrist we cropped around, not by MediaPipe
Hands' own (separate, less reliable when hands are close) handedness
classifier. Both hands still ultimately depend on Pose's own left/right
convention, so the mirroring check in the pilot (see label_video.py
--mirror) still matters -- it now just needs checking once, via Pose.

Usage:
    uv run pipeline/extract_keypoints.py <video path> <out .npz path>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent))
from mudra_common import normalize_landmarks

CROP_MARGIN = 2.5       # multiplier on elbow-wrist distance for crop half-size
MIN_CROP_HALF = 80      # px, floor so a fully-extended-arm frame doesn't over-shrink the crop
# Note: tuned against the pilot's worst case (a static closed-fist/Mushti hand, originally
# 18.8% detected at margin=1.8/min_half=60). Widening the crop + lowering confidence below
# raised that to 88.3% without hurting the already-good hand's detection rate (verified).
BODY_INDICES = {         # mediapipe Pose landmark indices we keep for motion gating
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
}
BODY_ORDER = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]


def crop_around(frame, cx, cy, half_size):
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(cx - half_size)), min(w, int(cx + half_size))
    y0, y1 = max(0, int(cy - half_size)), min(h, int(cy + half_size))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return frame[y0:y1, x0:x1]


def extract(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.3,
                                   min_tracking_confidence=0.3)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.15)

    hand_vecs, hand_present, hand_conf, body, frame_indices = [], [], [], [], []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pres = pose.process(rgb)

        body_row = np.full((6, 2), np.nan, dtype=np.float32)
        vecs_row = np.full((2, 42), np.nan, dtype=np.float32)
        present_row = np.zeros(2, dtype=bool)
        conf_row = np.zeros(2, dtype=np.float32)

        if pres.pose_landmarks:
            lm = pres.pose_landmarks.landmark
            for i, name in enumerate(BODY_ORDER):
                li = BODY_INDICES[name]
                body_row[i] = (lm[li].x * w_frame, lm[li].y * h_frame)

            for slot, (wrist_key, elbow_key) in enumerate([("l_wrist", "l_elbow"), ("r_wrist", "r_elbow")]):
                wi, ei = BODY_INDICES[wrist_key], BODY_INDICES[elbow_key]
                wx, wy = lm[wi].x * w_frame, lm[wi].y * h_frame
                ex, ey = lm[ei].x * w_frame, lm[ei].y * h_frame
                dist = ((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5
                half = max(MIN_CROP_HALF, CROP_MARGIN * dist)
                crop = crop_around(frame, wx, wy, half)
                if crop is None:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                hres = hands.process(crop_rgb)
                if hres.multi_hand_landmarks:
                    hlm = hres.multi_hand_landmarks[0]
                    xyz = np.array([[p.x, p.y, p.z] for p in hlm.landmark])
                    vecs_row[slot] = normalize_landmarks(xyz)
                    present_row[slot] = True
                    conf_row[slot] = hres.multi_handedness[0].classification[0].score

        hand_vecs.append(vecs_row)
        hand_present.append(present_row)
        hand_conf.append(conf_row)
        body.append(body_row)
        frame_indices.append(idx)
        idx += 1

    cap.release()
    pose.close()
    hands.close()

    return dict(
        hand_vecs=np.stack(hand_vecs),        # (n, 2, 42) -- slot 0=LEFT, 1=RIGHT
        hand_present=np.stack(hand_present),  # (n, 2) bool
        hand_conf=np.stack(hand_conf),        # (n, 2) float32
        body=np.stack(body),                  # (n, 6, 2) -- order = BODY_ORDER
        fps=np.array(fps, dtype=np.float32),
        frame_indices=np.array(frame_indices, dtype=np.int32),
    )


def main(video_path: str, out_path: str):
    out_path = Path(out_path)
    if out_path.exists():
        try:
            d = np.load(out_path)
            if "hand_vecs" in d:
                print(f"skip (already exists): {out_path}")
                return
        except Exception:
            pass  # corrupt/partial -- overwrite

    data = extract(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)

    n = len(data["frame_indices"])
    left_rate = data["hand_present"][:, 0].mean()
    right_rate = data["hand_present"][:, 1].mean()
    print(f"{Path(video_path).name}: {n} frames, LEFT detected {left_rate:.1%}, RIGHT detected {right_rate:.1%} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <video path> <out .npz path>")
    main(sys.argv[1], sys.argv[2])
