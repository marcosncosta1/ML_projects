"""
Zero-shot classification via CLIP text-image cosine similarity.

Ensemble prompt engineering: averaging embeddings across multiple prompt
templates before computing similarity measurably improves accuracy, mirroring
the technique used in OpenAI's original CLIP evaluation.
"""

import numpy as np
from PIL import Image

from src.model import encode_images, encode_texts

# Standard ensemble used in the CLIP paper + common extensions
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "a {} in a photo",
    "this is a {}",
    "an image of a {}",
    "a photo of the {}",
    "a close-up photo of a {}",
    "a photo of one {}",
]

CLIP_LOGIT_SCALE = 100.0  # approximate scale matching CLIP's trained temperature


def classify(
    image: Image.Image,
    labels: list[str],
    ensemble: bool = True,
) -> dict[str, float]:
    """
    Classify a single image against a set of text labels.

    Parameters
    ----------
    image    : PIL Image
    labels   : list of class name strings
    ensemble : if True, average text embeddings across all prompt templates

    Returns
    -------
    dict {label: probability} sorted highest-first
    """
    if not labels:
        return {}

    img_emb = encode_images([image])  # (1, 512)

    if ensemble:
        # Average embeddings across templates, then re-normalise
        all_embs = []
        for template in PROMPT_TEMPLATES:
            texts = [template.format(lbl) for lbl in labels]
            all_embs.append(encode_texts(texts))           # (N, 512)
        text_emb = np.mean(all_embs, axis=0)               # (N, 512)
        norms = np.linalg.norm(text_emb, axis=-1, keepdims=True)
        text_emb = text_emb / (norms + 1e-8)
    else:
        texts = [f"a photo of a {lbl}" for lbl in labels]
        text_emb = encode_texts(texts)                     # (N, 512)

    logits = (img_emb @ text_emb.T)[0] * CLIP_LOGIT_SCALE  # (N,)

    # Numerically stable softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    results = {lbl: float(p) for lbl, p in zip(labels, probs)}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
