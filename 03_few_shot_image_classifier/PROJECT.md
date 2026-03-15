# Few-Shot Image Classifier

## Overview
A zero/few-shot image classification app powered by CLIP or a pretrained ViT. Users upload images and define custom class labels in plain text — no retraining required.

## Category
Machine Learning / Computer Vision

## Stack
- **OpenAI CLIP** (`openai/clip-vit-base-patch32`) or **ViT** via HuggingFace Transformers
- **Gradio** — UI for image upload + label input
- **PyTorch** — inference
- **HuggingFace Spaces** — deployment target

## Key Features
- User provides custom class names (text labels)
- Upload one or more images
- CLIP computes cosine similarity between image embeddings and text embeddings
- Returns ranked predictions with confidence scores
- Few-shot mode: upload 1-5 examples per class to build a prototype embedding

## Architecture
### Zero-shot
1. User defines N class labels as text
2. CLIP encodes both image and text labels
3. Cosine similarity → softmax → top-k predictions

### Few-shot
1. User uploads example images per class
2. CLIP encodes examples → mean prototype embedding per class
3. Query image encoded → nearest prototype = predicted class

## Portfolio Value
- Shows understanding of modern vision-language models
- Goes beyond basic CNNs — demonstrates ML intuition
- Interactive and immediately understandable to non-technical viewers

## Milestones
- [ ] Zero-shot CLIP demo with Gradio
- [ ] Few-shot prototype classifier
- [ ] Deploy on HuggingFace Spaces
- [ ] README with architecture explanation + examples

## Notes
<!-- Add implementation notes, decisions, and progress here -->
