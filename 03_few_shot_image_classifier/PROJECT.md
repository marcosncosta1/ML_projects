# Few-Shot Image Classifier

## Overview
A Gradio app with two CLIP-powered classification modes:
- **Zero-shot**: type any class names, classify instantly — no examples needed
- **Few-shot**: upload 1–5 examples per class, classify by nearest prototype embedding

Both modes display a ViT attention heatmap showing which image regions CLIP focused on.

## Category
Machine Learning / Computer Vision / Vision-Language Models

## Stack
| Component | Library |
|---|---|
| Vision-language model | `openai/clip-vit-base-patch32` via HuggingFace Transformers |
| UI | Gradio 4.x (`gr.Blocks` with tabs) |
| Inference | PyTorch (CPU, no GPU required) |
| Visualisation | Matplotlib (attention colormap + PIL blend) |
| Deployment target | HuggingFace Spaces |

## Architecture

```
Image ──► CLIP ViT-B/32 ──► 512-dim embedding ─┐
                                                 ├── cosine similarity
Text  ──► CLIP Transformer ──► 512-dim embedding─┘
                                                 │
                                         softmax → probabilities
```

### Zero-shot
1. Query image → CLIP image encoder → 512-dim vector
2. Each label wrapped in 8 prompt templates → encoded → **averaged** → 512-dim vector
3. Cosine similarity × temperature → softmax → ranked probabilities

### Few-shot (prototype classification)
1. Example images per class → CLIP → mean embedding = **prototype**
2. Query image → CLIP → nearest prototype by cosine similarity → prediction

### Attention heatmap
- Call `model.vision_model(output_attentions=True)`
- Last layer CLS token attention to 49 patches → reshape to 7×7 → resize → blend

## Key Design Decisions
- **Prompt ensemble** (8 templates averaged) improves zero-shot accuracy ~3-5% vs single template
- **Re-normalise** after averaging prompt embeddings (mean of unit vectors ≠ unit vector)
- **Temperature = 100** for zero-shot (image-text sims), **10** for few-shot (image-image sims)
- Gradio `gr.State(PrototypeBank)` → one prototype bank per user session
- CLIP loaded once via `functools.lru_cache` and reused across all calls

## File Structure
```
03_few_shot_image_classifier/
├── app.py              # Gradio UI — tabs, event handlers
├── src/
│   ├── model.py        # CLIP loading, encode_images(), encode_texts(), get_attention_map()
│   ├── zero_shot.py    # classify() with prompt ensemble
│   ├── few_shot.py     # PrototypeBank class
│   └── attention_viz.py # make_overlay() — attention heatmap blending
├── GUIDE.md            # Learning walkthrough — concepts, code tour, experiments
├── requirements.txt
└── PROJECT.md
```

## Quick Start
```bash
cd 03_few_shot_image_classifier
pip install -r requirements.txt
python app.py
```

## Milestones
- [x] Zero-shot CLIP classification with prompt ensemble
- [x] Few-shot prototype classifier (PrototypeBank)
- [x] ViT attention heatmap visualisation
- [x] Gradio Blocks UI with 3 tabs (zero-shot, few-shot, how-it-works)
- [x] GUIDE.md learning walkthrough
- [ ] Deploy on HuggingFace Spaces
- [ ] Add `ViT-L/14` model option
- [ ] Attention Rollout (multi-layer) as alternative to last-layer attention

## Learning Resources
See `GUIDE.md` for a full walkthrough covering:
- Embedding spaces and cosine similarity
- Prompt engineering and why it works
- Prototype classification explained step by step
- ViT architecture and what attention means
- Experiments to run and extensions to build
