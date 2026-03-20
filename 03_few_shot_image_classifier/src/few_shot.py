"""
Few-shot prototype classifier using CLIP image embeddings.

How it works
------------
For each class, we encode all provided example images and compute their
mean embedding — the "prototype". At inference, the query image's embedding
is compared to every prototype via cosine similarity, and the closest one wins.

This is a form of metric learning: instead of fine-tuning any weights,
we're navigating CLIP's frozen embedding space, which already clusters
semantically similar images together.
"""

import numpy as np
from PIL import Image

from src.model import encode_images

# Temperature for softmax over similarities — higher = sharper/more confident
TEMPERATURE = 10.0


class PrototypeBank:
    """Stores per-class prototype embeddings built from user-supplied examples."""

    def __init__(self) -> None:
        # label -> (512,) L2-normalised prototype embedding
        self._prototypes: dict[str, np.ndarray] = {}
        # label -> number of examples used to build the prototype
        self._counts: dict[str, int] = {}

    # ── Building prototypes ───────────────────────────────────────────────────

    def add_class(self, label: str, images: list[Image.Image]) -> None:
        """
        Encode example images and store their mean as the class prototype.
        Calling this again for the same label overwrites the previous prototype.
        """
        embeddings = encode_images(images)          # (N, 512) — already normalised
        prototype  = embeddings.mean(axis=0)        # (512,)
        prototype  = prototype / (np.linalg.norm(prototype) + 1e-8)  # re-normalise
        self._prototypes[label] = prototype
        self._counts[label]     = len(images)

    def remove_class(self, label: str) -> None:
        self._prototypes.pop(label, None)
        self._counts.pop(label, None)

    def clear(self) -> None:
        self._prototypes.clear()
        self._counts.clear()

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(self, image: Image.Image) -> dict[str, float]:
        """
        Return {label: probability} sorted highest-first.

        Similarity scores are passed through a scaled softmax so they
        sum to 1 and can be read as approximate probabilities.
        """
        if not self._prototypes:
            return {}

        img_emb = encode_images([image])[0]          # (512,)
        labels  = list(self._prototypes.keys())
        protos  = np.stack([self._prototypes[l] for l in labels])  # (N, 512)

        similarities = protos @ img_emb               # (N,) cosine similarities

        # Scaled softmax
        scaled = similarities * TEMPERATURE
        scaled -= scaled.max()
        probs   = np.exp(scaled)
        probs  /= probs.sum()

        results = {lbl: float(p) for lbl, p in zip(labels, probs)}
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # ── Status helpers ────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return len(self._prototypes) == 0

    @property
    def class_names(self) -> list[str]:
        return list(self._prototypes.keys())

    def summary(self) -> str:
        if self.is_empty:
            return "No classes registered yet."
        lines = [
            f"**{lbl}** — {self._counts[lbl]} example{'s' if self._counts[lbl] > 1 else ''}"
            for lbl in self._prototypes
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._prototypes)
