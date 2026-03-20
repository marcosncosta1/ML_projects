# Learning Guide — Few-Shot Image Classifier

This document explains the key ideas behind the project so you can understand,
modify, and talk confidently about it in interviews.

---

## 1. The Big Idea: Embedding Spaces

The whole project rests on one concept: **embedding spaces**.

An embedding is a fixed-size vector (here, 512 numbers) that represents an
image or a piece of text. A good embedding space has the property that
**semantically similar things are geometrically close**.

CLIP learns this space by training on 400 million image-text pairs scraped from
the web. It trains two neural networks simultaneously:
- A **vision encoder** (ViT) that maps images → 512-dim vectors
- A **text encoder** (Transformer) that maps text → 512-dim vectors

The training objective: pull matching pairs closer, push non-matching pairs apart.
After training, `"a photo of a cat"` and an actual photo of a cat end up nearby.

```
High-dimensional embedding space (512 dims, shown as 2D):

  🐱 "cat" text ────► (0.3, 0.7, ...)
  🐱 cat photo ─────► (0.31, 0.68, ...)   ← close!

  🐶 "dog" text ────► (-0.4, 0.2, ...)
  🐶 dog photo ─────► (-0.38, 0.22, ...)  ← also close!

  Distance between cat and dog > distance within each class
```

---

## 2. Zero-Shot Classification

**Zero-shot** means classifying images into categories the model was never
explicitly trained to classify. There's no fine-tuning step.

### How it works (`src/zero_shot.py`)

```python
# 1. Encode the query image
img_emb = encode_images([image])          # shape: (1, 512)

# 2. Encode each class label as text
texts = ["a photo of a cat", "a photo of a dog", "a photo of a pizza"]
text_emb = encode_texts(texts)            # shape: (3, 512)

# 3. Compute cosine similarity
# (Both embeddings are L2-normalised, so dot product = cosine similarity)
logits = img_emb @ text_emb.T            # shape: (1, 3)

# 4. Softmax → probabilities
probs = softmax(logits * temperature)
```

### Cosine similarity

Because both vectors are L2-normalised (length = 1), their dot product equals
their cosine similarity:
```
cos(θ) = (a · b) / (|a| |b|) = a · b   when |a| = |b| = 1
```
Range: -1 (opposite) to 1 (identical direction).

### Why multiply by 100 (the temperature)?

Raw cosine similarities are small numbers (e.g., 0.25, 0.30). Feeding these
directly into softmax produces very flat distributions. Multiplying by ~100
sharpens the distribution so the winner is more confident. This is CLIP's
learned logit scale, approximately matching the temperature it was trained with.

---

## 3. Prompt Engineering

The label `"cat"` is too ambiguous. CLIP was trained on natural sentences, not
single words. Wrapping it in a sentence context dramatically improves accuracy:

```
"cat"                    → ok
"a photo of a cat"       → better
"a photo of a cat"  ┐
"a picture of a cat" ├── averaged → best
"a close-up of a cat"┘
```

Averaging the embeddings from multiple templates creates a "centroid" that
captures the label's meaning across different photographic contexts.
This is called **prompt ensemble** and it's used in the CLIP paper to achieve
their best zero-shot numbers.

**To try:** Disable the ensemble checkbox in the app and see how much accuracy
changes on ambiguous images.

### Where to look in the code

`src/zero_shot.py` — `PROMPT_TEMPLATES` list and the averaging logic:
```python
all_embs = []
for template in PROMPT_TEMPLATES:
    texts = [template.format(lbl) for lbl in labels]
    all_embs.append(encode_texts(texts))   # (N, 512)
text_emb = np.mean(all_embs, axis=0)       # (N, 512)
# Re-normalise after averaging!
text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
```
Note: you must re-normalise after averaging because the mean of unit vectors is
not a unit vector.

---

## 4. Few-Shot Prototype Classification

**Few-shot** classification uses 1–5 labelled examples per class instead of
thousands. No gradient descent, no weight updates.

### How it works (`src/few_shot.py`)

```
Registration phase:
  Example images of "golden retriever" ──► encode ──► 3 vectors (512,)
                                                          │
                                                     mean + normalise
                                                          │
                                                  prototype_retriever (512,)

Inference phase:
  Query image ──► encode ──► query_emb (512,)
                                  │
                        cosine similarity to each prototype
                                  │
                             softmax → probabilities
```

The prototype is the **centroid** of all example embeddings. Adding more
examples shifts the centroid towards the true cluster centre, improving accuracy.

### Why does this work with just 1 example?

Because CLIP's embedding space is already organised by visual semantics.
Even a single photo of a golden retriever creates an embedding that lands in
the right region of the space. New photos of golden retrievers land nearby.
We're not learning a new representation — we're **navigating an existing one**.

### Temperature in few-shot

```python
TEMPERATURE = 10.0
scaled = similarities * TEMPERATURE
probs = softmax(scaled)
```

Few-shot uses a higher temperature (10 vs ~100 in zero-shot) because raw
similarities between image embeddings are typically smaller than image-text
similarities. Adjust this if predictions are too flat or too sharp.

---

## 5. The Vision Transformer (ViT) and Attention

CLIP uses a ViT-B/32 as its vision encoder. "ViT" stands for Vision Transformer.

### How ViT processes an image

```
224×224 image
     │
     ▼ Split into 7×7 grid of 32×32 patches = 49 patches
     │
     ▼ Each patch → linear projection → 768-dim token
     │
     ▼ Add a special CLS token at position 0 (total: 50 tokens)
     │
     ▼ 12 Transformer layers (each with self-attention + MLP)
     │
     ▼ CLS token at the output → projected to 512-dim → image embedding
```

### Self-attention in one layer

Every token attends to every other token. The attention weight between two
tokens tells you "how much should token A look at token B to update itself".

After the final layer, the CLS token's attention to each of the 49 patch tokens
reflects which patches were most important for the final representation.

### What the heatmap shows (`src/attention_viz.py`)

```python
# Last layer attentions: (batch=1, heads=12, seq=50, seq=50)
attn = outputs.attentions[-1][0]     # (12, 50, 50)
attn = attn.mean(dim=0)              # (50, 50) — average over 12 heads
attn_cls = attn[0, 1:]               # CLS row, skip self-attention: (49,)
attn_map = attn_cls.reshape(7, 7)    # back to spatial grid
```

This is a simplified visualisation. A more principled approach is
**Attention Rollout** (multiply attention matrices across all layers), which
you can explore as an extension.

---

## 6. File-by-File Code Tour

```
app.py                  Gradio UI — tabs, event wiring, helper functions
src/
  model.py              CLIP loading (cached), encode_images(), encode_texts(),
                        get_attention_map()
  zero_shot.py          classify() — prompt templates, ensemble averaging, softmax
  few_shot.py           PrototypeBank class — add_class(), classify(), summary()
  attention_viz.py      make_overlay() — resize attn map, apply colormap, blend
```

**Start here when reading the code:**
1. `src/model.py` — understand how to get embeddings
2. `src/zero_shot.py` — understand prompt engineering
3. `src/few_shot.py` — understand prototype classification
4. `app.py` — understand how Gradio wires it together

---

## 7. Things to Try and Learn From

### Experiment A: Prompt engineering effect
1. Upload a photo of something specific (e.g., a Labrador)
2. Label: `labrador, golden retriever, poodle`
3. Try with ensemble ON vs OFF. Note the confidence change.

### Experiment B: Embedding space geometry
1. Register few-shot classes: `happy face`, `sad face`
2. Upload a neutral face — where does it land?
3. Register a third class: `neutral face` — does it shift?

### Experiment C: What confuses CLIP?
- Try visually similar but semantically different labels
- Try abstract concepts (`freedom`, `justice`) vs concrete ones
- Try fine-grained distinctions (`labrador`, `golden retriever`) — CLIP
  often struggles here because the text descriptions are so similar

### Experiment D: Attention vs prediction
- Find an image where CLIP is wrong
- Look at the attention map — is it looking at the wrong region?
- This shows that high attention ≠ correct understanding

---

## 8. How to Extend This

| Extension | Difficulty | What you'd learn |
|---|---|---|
| Attention Rollout (multiply across all layers) | Medium | Multi-layer attention propagation |
| GradCAM on CLIP (gradient-based saliency) | Medium | Gradient-based attribution |
| Add `ViT-L/14` option | Easy | Scaling laws — bigger model, better accuracy |
| Fine-tune the projection head on a small dataset | Hard | Transfer learning |
| Deploy to HuggingFace Spaces | Easy | Gradio deployment, `spaces.GPU` decorator |
| Add image-to-image retrieval mode | Medium | Embedding search, FAISS |

---

## 9. Key Papers to Read

1. **CLIP** — Radford et al. 2021 — "Learning Transferable Visual Models From Natural Language Supervision"
   https://arxiv.org/abs/2103.00020

2. **ViT** — Dosovitskiy et al. 2020 — "An Image is Worth 16x16 Words"
   https://arxiv.org/abs/2010.11929

3. **Attention Rollout** — Abnar & Zuidema 2020 — "Quantifying Attention Flow in Transformers"
   https://arxiv.org/abs/2005.00928

4. **Prototypical Networks** — Snell et al. 2017 — the formal foundation of prototype-based few-shot learning
   https://arxiv.org/abs/1703.05175
