"""
Attention map visualisation for CLIP's ViT vision encoder.

What this shows
---------------
CLIP's vision encoder is a Vision Transformer (ViT). Each transformer block
contains a multi-head self-attention layer. The final layer's CLS token
accumulates information from all image patches; its attention weights indicate
which patches the model considered most relevant when forming the final
representation.

We extract those weights, reshape them from a flat (49,) vector back into the
(7, 7) spatial grid, resize to the original image resolution, and blend the
resulting heatmap over the image.

Limitations
-----------
Raw last-layer attention is a rough proxy for "what the model looks at" — it
doesn't account for attention across earlier layers. For a more principled
approach, see Attention Rollout (Abnar & Zuidema 2020), which multiplies
attention matrices across all layers. That's a future extension.
"""

import numpy as np
import matplotlib
from PIL import Image

matplotlib.use("Agg")  # non-interactive backend — safe inside Gradio


def make_overlay(image: Image.Image, attn_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Blend a CLIP attention heatmap over the original image.

    Parameters
    ----------
    image    : original PIL Image (any size)
    attn_map : (7, 7) float array, values in [0, 1]
    alpha    : heatmap opacity — 0 = invisible, 1 = fully opaque

    Returns
    -------
    PIL Image — same size as input, RGB
    """
    w, h = image.size

    # 1. Upsample attention map to match image resolution
    attn_uint8 = (attn_map * 255).astype(np.uint8)
    attn_pil   = Image.fromarray(attn_uint8, mode="L")
    attn_pil   = attn_pil.resize((w, h), Image.BICUBIC)
    attn_arr   = np.array(attn_pil) / 255.0              # (H, W) float

    # 2. Apply "jet" colormap (blue=low, red=high attention)
    cmap    = matplotlib.colormaps["jet"]
    heatmap = cmap(attn_arr)[:, :, :3]                   # drop alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap, mode="RGB")

    # 3. Blend with original
    original_rgb = image.convert("RGB")
    blended = Image.blend(original_rgb, heatmap_pil, alpha=alpha)
    return blended
