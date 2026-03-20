"""
Few-Shot Image Classifier — Gradio app
=======================================
Two classification modes, one interface:

  Zero-shot  — type class names, classify immediately. No examples needed.
               Uses CLIP's joint vision-language embedding space.

  Few-shot   — upload 1–5 examples per class to build a prototype embedding,
               then classify new images by nearest prototype.

Both modes display an attention heatmap showing which image regions
CLIP focused on when forming its representation.

Run:
    python app.py
    # or
    gradio app.py
"""

import numpy as np
from PIL import Image
import gradio as gr

from src.model import get_attention_map
from src.zero_shot import classify, PROMPT_TEMPLATES
from src.few_shot import PrototypeBank
from src.attention_viz import make_overlay


# ── Shared helpers ────────────────────────────────────────────────────────────

def _to_pil(img) -> Image.Image:
    """Gradio passes images as numpy arrays by default; convert to PIL."""
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img


def _attention_overlay(image: Image.Image) -> Image.Image:
    attn_map = get_attention_map(image)
    return make_overlay(image, attn_map)


# ── Zero-shot handlers ────────────────────────────────────────────────────────

def zero_shot_run(image, labels_text: str, use_ensemble: bool):
    """Main handler for the zero-shot tab."""
    if image is None:
        return None, None, "Upload an image first."

    image = _to_pil(image)
    labels = [l.strip() for l in labels_text.split(",") if l.strip()]
    if not labels:
        return None, None, "Enter at least one label (comma-separated)."

    probs   = classify(image, labels, ensemble=use_ensemble)
    overlay = _attention_overlay(image)

    template_note = (
        f"Using ensemble of {len(PROMPT_TEMPLATES)} prompt templates"
        if use_ensemble
        else "Using single template: 'a photo of a {label}'"
    )
    top_label, top_prob = next(iter(probs.items()))
    info = f"**Prediction: {top_label}** ({top_prob:.1%})  \n{template_note}"

    return overlay, probs, info


# ── Few-shot handlers ─────────────────────────────────────────────────────────

def register_class(class_name: str, example_images, bank: PrototypeBank):
    """Encode example images and store their mean as the class prototype."""
    if not class_name or not class_name.strip():
        return bank, bank.summary(), "Enter a class name."
    if not example_images:
        return bank, bank.summary(), "Upload at least one example image."

    name   = class_name.strip()
    images = [_to_pil(img) for img in example_images]
    bank.add_class(name, images)

    return bank, bank.summary(), f"Registered **{name}** with {len(images)} example(s)."


def remove_class(class_name: str, bank: PrototypeBank):
    bank.remove_class(class_name.strip())
    return bank, bank.summary(), f"Removed class '{class_name}'."


def clear_bank(bank: PrototypeBank):
    bank.clear()
    return bank, "No classes registered yet.", "Cleared all classes."


def few_shot_run(query_image, bank: PrototypeBank):
    """Main handler for the few-shot tab."""
    if query_image is None:
        return None, None, "Upload a query image."
    if bank.is_empty:
        return None, None, "Register at least one class first."
    if len(bank) < 2:
        return None, None, "Register at least 2 classes to compare."

    image   = _to_pil(query_image)
    probs   = bank.classify(image)
    overlay = _attention_overlay(image)

    top_label, top_prob = next(iter(probs.items()))
    info = f"**Prediction: {top_label}** ({top_prob:.1%})"

    return overlay, probs, info


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Few-Shot Image Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # Few-Shot Image Classifier
        Powered by **CLIP** (`ViT-B/32`) — no fine-tuning, no GPU required.

        | Mode | How it works |
        |---|---|
        | **Zero-shot** | Type class names → CLIP matches image to text in embedding space |
        | **Few-shot** | Upload 1–5 example images per class → classify by nearest prototype |

        The **attention heatmap** shows which image regions CLIP focused on (red = high, blue = low).
        """
    )

    # ── Zero-shot tab ─────────────────────────────────────────────────────────
    with gr.Tab("Zero-shot"):
        gr.Markdown(
            "Type any class names — CLIP will score the image against each label "
            "using its vision-language embedding space. No examples needed."
        )
        with gr.Row():
            with gr.Column(scale=1):
                zs_image   = gr.Image(label="Query image", type="numpy")
                zs_labels  = gr.Textbox(
                    label="Class labels (comma-separated)",
                    placeholder="cat, dog, pizza, sunset, bicycle",
                    lines=2,
                )
                zs_ensemble = gr.Checkbox(
                    label="Ensemble prompts (recommended)",
                    value=True,
                    info=f"Averages {len(PROMPT_TEMPLATES)} prompt variants — improves accuracy ~5%",
                )
                zs_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                zs_overlay = gr.Image(label="Attention heatmap", interactive=False)
                zs_label   = gr.Label(label="Confidence scores", num_top_classes=10)
                zs_info    = gr.Markdown()

        zs_btn.click(
            zero_shot_run,
            inputs=[zs_image, zs_labels, zs_ensemble],
            outputs=[zs_overlay, zs_label, zs_info],
        )

        gr.Examples(
            examples=[
                [None, "cat, dog, fish, bird", True],
                [None, "beach, forest, mountain, city", True],
                [None, "happy person, sad person, angry person", False],
            ],
            inputs=[zs_image, zs_labels, zs_ensemble],
            label="Example label sets to try",
        )

    # ── Few-shot tab ──────────────────────────────────────────────────────────
    with gr.Tab("Few-shot"):
        gr.Markdown(
            "Register custom classes by uploading 1–5 example images each. "
            "CLIP encodes them into **prototype embeddings** — then classifies "
            "new images by finding the nearest prototype in embedding space."
        )

        bank_state = gr.State(PrototypeBank)

        with gr.Row():
            # Left: prototype bank management
            with gr.Column(scale=1):
                gr.Markdown("### Register a class")
                fs_class_name = gr.Textbox(
                    label="Class name",
                    placeholder="golden retriever",
                )
                fs_examples = gr.Gallery(
                    label="Example images (drag to upload multiple)",
                    type="numpy",
                    columns=3,
                    height=200,
                )
                with gr.Row():
                    fs_register_btn = gr.Button("Add class", variant="primary")
                    fs_remove_btn   = gr.Button("Remove class", variant="secondary")
                    fs_clear_btn    = gr.Button("Clear all", variant="stop")

                gr.Markdown("### Registered classes")
                fs_bank_status = gr.Markdown("No classes registered yet.")
                fs_msg = gr.Markdown()

            # Right: query + results
            with gr.Column(scale=1):
                gr.Markdown("### Classify a new image")
                fs_query   = gr.Image(label="Query image", type="numpy")
                fs_btn     = gr.Button("Classify", variant="primary")
                fs_overlay = gr.Image(label="Attention heatmap", interactive=False)
                fs_label   = gr.Label(label="Confidence scores", num_top_classes=10)
                fs_info    = gr.Markdown()

        fs_register_btn.click(
            register_class,
            inputs=[fs_class_name, fs_examples, bank_state],
            outputs=[bank_state, fs_bank_status, fs_msg],
        )
        fs_remove_btn.click(
            remove_class,
            inputs=[fs_class_name, bank_state],
            outputs=[bank_state, fs_bank_status, fs_msg],
        )
        fs_clear_btn.click(
            clear_bank,
            inputs=[bank_state],
            outputs=[bank_state, fs_bank_status, fs_msg],
        )
        fs_btn.click(
            few_shot_run,
            inputs=[fs_query, bank_state],
            outputs=[fs_overlay, fs_label, fs_info],
        )

    # ── About tab ─────────────────────────────────────────────────────────────
    with gr.Tab("How it works"):
        gr.Markdown(
            """
            ## CLIP in a nutshell

            CLIP (Contrastive Language-Image Pretraining, OpenAI 2021) trains two encoders
            jointly — one for images, one for text — so that matching image-text pairs
            end up close in a shared 512-dimensional embedding space.

            ```
            Image ──► ViT-B/32 encoder ──► 512-dim vector ─┐
                                                             ├── cosine similarity → prediction
            Text  ──► Transformer encoder ──► 512-dim vector─┘
            ```

            ## Zero-shot: why prompt engineering matters

            When you type `"cat"`, we don't encode just that word. We wrap it in templates:

            ```
            "a photo of a cat"
            "a picture of a cat"
            "a close-up photo of a cat"
            ... (8 templates total)
            ```

            Each template produces a slightly different embedding. **Averaging them** creates
            a more robust representation that handles more photographic contexts.
            This trick alone improves accuracy by ~3-5% on standard benchmarks.

            ## Few-shot: prototype embeddings

            For each class you register:
            1. All example images are encoded → N embeddings of shape (512,)
            2. Their **mean** is computed and re-normalised → one prototype per class
            3. At inference, the query image is encoded and compared to every prototype
               via cosine similarity

            More examples = better prototype = better accuracy.
            Even 1 example often works well because CLIP's embedding space is so rich.

            ## Attention heatmap

            The ViT vision encoder divides the image into 7×7 = 49 patches.
            The final transformer layer's **CLS token** (the global representation)
            attends to each patch with a learned weight. We visualise these weights
            as a heatmap:

            - **Red** = high attention — the model relied heavily on this region
            - **Blue** = low attention — mostly ignored

            This is a proxy for explainability, not a precise attribution method.
            """
        )


if __name__ == "__main__":
    demo.launch(share=False)
