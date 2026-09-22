# /// script
# requires-python = "<3.13"
# dependencies = ["numpy"]
# ///
"""Shared constants and landmark math for the mudra-labelling pipeline.

Index alignment note: the Kavitha (IIIT-Kottayam) dataset's 24 folder names
and your canonical 24-mudra list (from the printed Hasta Lakshana Deepika
sheet) are the SAME 24 mudras in the SAME order, just different
transliteration styles (Sanskrit-ish vs. Malayalam-ish). MUDRAS below pairs
them by that shared index so the pipeline's internal labels line up with
what you actually write in ELAN/CSV.
"""
import numpy as np

# (dataset folder slug, display name for ELAN/CSV)
MUDRAS = [
    ("pataka", "Pataaka"),
    ("mudrakhya", "Mudrakhyam"),
    ("kataka", "Katakam"),
    ("mushti", "Mushti"),
    ("kartarimukha", "Kartareemukham"),
    ("sukatunda", "Shukatundam"),
    ("kapittha", "Kapitthakam"),
    ("hamsapaksha", "Hamsapaksham"),
    ("sikhara", "Shikaram"),
    ("hamsasya", "Hamsaasyam"),
    ("anjali", "Anjali"),
    ("ardhachandra", "Ardhachandram"),
    ("mukura", "Mukuram"),
    ("bhramara", "Bhramaram"),
    ("suchimukha", "Soocheemukham"),
    ("pallava", "Pallavam"),
    ("tripataka", "Thripataaka"),
    ("mrigasirsha", "Mrigasheersham"),
    ("sarpasiras", "Sarpasirassu"),
    ("vardhamanaka", "Vardhamaanakam"),
    ("arala", "Araalam"),
    ("urnanabha", "Oornanabham"),
    ("mukula", "Mukulam"),
    ("katakamukha", "Katakaamukham"),
]

DATASET_TO_DISPLAY = {slug: disp for slug, disp in MUDRAS}
DISPLAY_TO_DATASET = {disp.lower(): slug for slug, disp in MUDRAS}
DATASET_SLUGS = [slug for slug, _ in MUDRAS]


def display_to_slug(name: str) -> str:
    """Map a name typed in the ledger (any casing) to the dataset slug.

    Raises KeyError with the offending name if it doesn't match the 24
    canonical mudras -- surface typos loudly instead of silently guessing.
    """
    key = name.strip().lower()
    if key not in DISPLAY_TO_DATASET:
        raise KeyError(f"'{name}' is not one of the 24 canonical mudra names")
    return DISPLAY_TO_DATASET[key]


def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    """Translate+scale-normalize 21x3 MediaPipe hand landmarks.

    - Origin = wrist (landmark 0), so position in frame doesn't matter.
    - Scale = distance wrist -> middle-finger MCP (landmark 9), so distance
      from camera / crop size doesn't matter.
    - z is dropped (least reliable axis from a single RGB frame).
    Returns a flat 42-dim (21 points x 2) vector.
    """
    pts = np.asarray(landmarks_xyz, dtype=np.float64)[:, :2]
    origin = pts[0]
    scale = np.linalg.norm(pts[9] - origin)
    if scale < 1e-9:
        scale = 1e-9
    return ((pts - origin) / scale).reshape(-1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)
