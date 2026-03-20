"""
CLIP model loading and core embedding utilities.
All functions are pure — they take PIL Images / strings and return numpy arrays.
"""

import functools
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
# ViT-B/32: 224×224 input, 32×32 patches → 7×7 grid = 49 patches + 1 CLS = 50 tokens
PATCH_GRID = 7


@functools.lru_cache(maxsize=1)
def _load() -> tuple[CLIPModel, CLIPProcessor]:
    # attn_implementation="eager" is required to get attention weights back.
    # The default "sdpa" (scaled dot-product attention) fuses the operation
    # and cannot return per-head attention matrices.
    model = CLIPModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def encode_images(images: list[Image.Image]) -> np.ndarray:
    """Return L2-normalised image embeddings, shape (N, 512)."""
    model, processor = _load()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def encode_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised text embeddings, shape (N, 512)."""
    model, processor = _load()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def get_attention_map(image: Image.Image) -> np.ndarray:
    """
    Extract a (7, 7) attention map from the last ViT layer's CLS token.

    Averages attention weights across all 12 heads in the final transformer
    block. The result is normalised to [0, 1].
    """
    model, processor = _load()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
        )

    # outputs.attentions[-1]: (batch=1, heads=12, seq=50, seq=50)
    attn = outputs.attentions[-1][0]          # (12, 50, 50)
    attn = attn.mean(dim=0)                   # (50, 50) — average over heads
    attn_cls = attn[0, 1:].cpu().numpy()      # CLS → patches: (49,)

    attn_map = attn_cls.reshape(PATCH_GRID, PATCH_GRID)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map
