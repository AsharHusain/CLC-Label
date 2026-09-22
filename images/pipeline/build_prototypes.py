# /// script
# requires-python = "<3.13"
# dependencies = [
#     "mediapipe==0.10.14",
#     "opencv-python-headless",
#     "numpy",
# ]
# ///
"""Level 1: turn the Kavitha/IIIT-Kottayam Hasta Mudra Dataset into a
reference vector database for hand-shape matching.

One-time / occasional step, run by whoever owns the ML side (Person A).

Usage:
    git clone --depth 1 https://github.com/kavitharaju/KathakaliMudraDataset.git
    uv run build_prototypes.py KathakaliMudraDataset/images prototypes.npz

Output: prototypes.npz with arrays:
    vectors: (N, 42) float64  -- normalized landmark vectors (see mudra_common)
    labels:  (N,) str         -- display mudra name for each vector
    sources: (N,) str         -- source image path, for debugging bad matches
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent))
from mudra_common import DATASET_TO_DISPLAY, normalize_landmarks


def main(images_dir: str, out_path: str) -> None:
    images_dir = Path(images_dir)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
    )

    vectors, labels, sources = [], [], []
    skipped = []

    class_dirs = sorted(images_dir.iterdir())
    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue
        # folder names look like "8.hamsapaksha" -> slug "hamsapaksha"
        slug = class_dir.name.split(".", 1)[-1].strip().lower()
        display = DATASET_TO_DISPLAY.get(slug)
        if display is None:
            print(f"  ! unrecognized class folder, skipping: {class_dir.name}")
            continue

        img_paths = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append(str(img_path))
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if not result.multi_hand_landmarks:
                skipped.append(str(img_path))
                continue
            lm = result.multi_hand_landmarks[0]
            xyz = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            vectors.append(normalize_landmarks(xyz))
            labels.append(display)
            sources.append(str(img_path))

        print(f"  {display:16s} ({slug}): {sum(1 for l in labels if l == display)} vectors")

    hands.close()

    if not vectors:
        raise SystemExit("No hand landmarks extracted at all -- check images_dir path")

    np.savez(
        out_path,
        vectors=np.array(vectors),
        labels=np.array(labels),
        sources=np.array(sources),
    )
    print(f"\nSaved {len(vectors)} reference vectors across {len(set(labels))} classes to {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} images where MediaPipe found no hand (low quality/occluded):")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <path to dataset/images dir> <out .npz path>")
    main(sys.argv[1], sys.argv[2])
