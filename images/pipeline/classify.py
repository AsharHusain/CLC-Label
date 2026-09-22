# /// script
# requires-python = "<3.13"
# dependencies = ["numpy"]
# ///
"""k-NN classification of a normalized hand-landmark vector against the
reference database built by build_prototypes.py.

Used two ways by label_video.py:
  1. classify(vector, db)                 -- full 24-way guess, for QA/spot checks.
  2. similarity_to(vector, db, label)      -- similarity to ONE known class,
     which is what boundary-detection actually needs (we already know the
     initial/final mudra from the ledger; we just need to know *when* the
     hand matches each one).
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mudra_common import cosine_similarity


class PrototypeDB:
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.vectors = data["vectors"]  # (N, 42)
        self.labels = data["labels"]    # (N,)

    def similarity_to(self, query: np.ndarray, label: str) -> float:
        """Max cosine similarity of `query` against all references of `label`."""
        mask = self.labels == label
        if not mask.any():
            raise KeyError(f"no reference vectors for label '{label}'")
        sims = [cosine_similarity(query, ref) for ref in self.vectors[mask]]
        return max(sims)

    def classify(self, query: np.ndarray, k: int = 5) -> tuple[str, float]:
        """Full k-NN vote across all classes. Returns (best_label, confidence)."""
        sims = np.array([cosine_similarity(query, ref) for ref in self.vectors])
        top_k = np.argsort(sims)[-k:]
        top_labels = self.labels[top_k]
        top_sims = sims[top_k]
        labels, counts = np.unique(top_labels, return_counts=True)
        best_label = labels[np.argmax(counts)]
        confidence = float(top_sims[top_labels == best_label].mean())
        return str(best_label), confidence
