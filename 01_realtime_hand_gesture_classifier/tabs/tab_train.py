import io
import sys
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import config
from components.dataset import load_dataset, get_class_counts
from components.augmentation import augment_batch


def render() -> None:
    st.header("Train Model")

    counts = get_class_counts()
    total  = sum(counts.values())
    filled = sum(1 for v in counts.values() if v > 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", total)
    c2.metric("Classes with data", f"{filled} / 10")
    missing = [k for k, v in counts.items() if v == 0]
    c3.metric("Missing classes", len(missing))
    if missing:
        st.caption(f"Missing: {', '.join(missing)}")

    if total == 0:
        st.info("No data yet — collect samples in the **Data Collection** tab.")
        return

    # Dataset table
    rows = []
    for i in range(10):
        k = str(i)
        char, pinyin, desc = config.CHINESE_GESTURES[k]
        rows.append({
            "Digit": f"{i}  {char}",
            "Pinyin": pinyin,
            "Samples": counts[k],
            "Status": "✅" if counts[k] >= 20 else "⚠️" if counts[k] > 0 else "❌",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Training settings")

    col_a, col_b = st.columns(2)
    with col_a:
        aug_factor = st.slider(
            "Augmentation copies per real sample",
            min_value=1, max_value=20, value=5,
            help="Each real sample generates this many synthetic variants via noise/rotation/scale/translation",
        )
        effective = total * (aug_factor + 1)
        st.caption(f"Effective training samples: **{effective:,}**")

    with col_b:
        hidden_1 = st.select_slider("Hidden layer 1 neurons", options=[64, 128, 256, 512], value=256)
        hidden_2 = st.select_slider("Hidden layer 2 neurons", options=[32, 64, 128, 256], value=128)
        st.caption(f"Architecture: 63 → {hidden_1} → {hidden_2} → 10")

    min_samples = 5
    can_train = total >= min_samples and filled >= 2
    if not can_train:
        st.warning(f"Need at least {min_samples} total samples across ≥2 classes to train.")

    train_btn = st.button("🚀 Train Model", type="primary", disabled=not can_train)

    if train_btn:
        _run_training(aug_factor, hidden_1, hidden_2)

    if "train_report" in st.session_state:
        _show_results()


def _run_training(aug_factor: int, hidden_1: int, hidden_2: int) -> None:
    X, y = load_dataset()
    if X is None:
        st.error("Could not load dataset.")
        return

    present_classes = np.unique(y)
    if len(present_classes) < 2:
        st.error("Need samples from at least 2 different classes.")
        return

    status = st.status("Training…", expanded=True)

    with status:
        st.write("**Step 1/3** — Splitting dataset (stratified 80/20)…")
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"   Train: {len(X_tr)} raw samples  |  Test: {len(X_te)} raw samples")

        st.write(f"**Step 2/3** — Augmenting training split ×{aug_factor}…")
        X_tr_aug, y_tr_aug = augment_batch(X_tr, y_tr, n_copies=aug_factor)
        st.write(f"   Augmented training set: **{len(X_tr_aug):,}** samples")

        st.write("**Step 3/3** — Fitting MLP (this may take a few seconds)…")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(hidden_1, hidden_2),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False,
                random_state=42,
            )),
        ])

        # Capture any sklearn warnings/verbose output
        buf = io.StringIO()
        t0 = time.time()
        old_stdout = sys.stdout
        sys.stdout = buf
        pipeline.fit(X_tr_aug, y_tr_aug)
        sys.stdout = old_stdout
        elapsed = time.time() - t0

        config.MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, config.MODEL_PATH)

        y_pred  = pipeline.predict(X_te)
        report  = classification_report(
            y_te, y_pred,
            labels=sorted(present_classes),
            target_names=[str(c) for c in sorted(present_classes)],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_te, y_pred, labels=sorted(present_classes))

        mlp = pipeline.named_steps["mlp"]
        converged = mlp.n_iter_ < 500

        st.session_state.update({
            "trained_model":    pipeline,
            "train_report":     report,
            "train_cm":         cm,
            "train_cm_labels":  [str(c) for c in sorted(present_classes)],
            "train_meta": {
                "elapsed":          elapsed,
                "n_iter":           mlp.n_iter_,
                "converged":        converged,
                "best_val_score":   mlp.best_validation_score_,
                "loss_curve":       mlp.loss_curve_,
                "val_scores":       list(mlp.validation_scores_),
                "n_train":          len(X_tr_aug),
                "n_test":           len(X_te),
                "hidden":           (hidden_1, hidden_2),
                "aug_factor":       aug_factor,
            },
        })

        status.update(label="✅ Training complete!", state="complete", expanded=False)

    st.rerun()


def _show_results() -> None:
    report = st.session_state["train_report"]
    cm     = st.session_state["train_cm"]
    labels = st.session_state["train_cm_labels"]
    meta   = st.session_state.get("train_meta", {})

    st.divider()
    st.subheader("Training run summary")

    # ── Top metrics ──────────────────────────────────────────────────────────
    acc = report.get("accuracy", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy",    f"{acc:.1%}")
    c2.metric("Epochs run",       meta.get("n_iter", "—"))
    c3.metric("Best val accuracy", f"{meta.get('best_val_score', 0):.1%}")
    c4.metric("Training time",    f"{meta.get('elapsed', 0):.1f}s")

    n_iter   = meta.get("n_iter", 500)
    converged = meta.get("converged", False)
    if converged:
        st.success(f"Early stopping triggered at epoch {n_iter} — model converged cleanly.")
    else:
        st.warning(
            f"Reached max 500 epochs without convergence. "
            "Consider collecting more data or increasing the augmentation factor."
        )

    # ── Loss + validation curve ───────────────────────────────────────────────
    loss_curve = meta.get("loss_curve", [])
    val_scores = meta.get("val_scores", [])

    if loss_curve:
        st.subheader("Training curves")
        epochs = list(range(1, len(loss_curve) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=loss_curve,
            name="Training loss",
            line=dict(color="#e74c3c", width=2),
            yaxis="y1",
        ))
        if val_scores:
            fig.add_trace(go.Scatter(
                x=epochs, y=val_scores,
                name="Validation accuracy",
                line=dict(color="#2ecc71", width=2),
                yaxis="y2",
            ))
            # Mark best validation score
            best_epoch = int(np.argmax(val_scores)) + 1
            best_val   = max(val_scores)
            fig.add_vline(
                x=best_epoch,
                line_dash="dot", line_color="#2ecc71",
                annotation_text=f"Best val {best_val:.1%} @ ep {best_epoch}",
                annotation_position="top right",
            )

        fig.update_layout(
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Loss", side="left", showgrid=True),
            yaxis2=dict(title="Val accuracy", side="right", overlaying="y",
                        tickformat=".0%", range=[0, 1]),
            legend=dict(x=0.01, y=0.99),
            height=350,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Loss should decrease and level off. "
            "If it's still dropping steeply at the end, collect more data or raise max_iter. "
            "If val accuracy is much lower than train accuracy, the model may be overfitting — "
            "collect more samples or lower augmentation noise."
        )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Confusion matrix  (test split, no augmentation)")
    fig_cm = ff.create_annotated_heatmap(
        cm.tolist(), x=labels, y=labels,
        colorscale="Blues", showscale=False,
    )
    fig_cm.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=420, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(
        "Diagonal = correct predictions. Off-diagonal = confusions. "
        "If two gestures are often swapped, collect more samples for both and make "
        "sure your hand pose is clearly distinct."
    )

    # ── Per-class table ───────────────────────────────────────────────────────
    st.subheader("Per-class metrics")
    rows = []
    for cls in labels:
        if cls in report:
            r = report[cls]
            char = config.CHINESE_GESTURES[cls][0]
            rows.append({
                "Class":     f"{cls} {char}",
                "Precision": f"{r['precision']:.2f}",
                "Recall":    f"{r['recall']:.2f}",
                "F1":        f"{r['f1-score']:.2f}",
                "Support":   int(r["support"]),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "**Precision** — of all times the model predicted this class, how often was it right.  \n"
            "**Recall** — of all real samples of this class, how many did it catch.  \n"
            "**F1** — harmonic mean of both; the main number to optimise.  \n"
            "**Support** — number of real test samples for this class."
        )

    # ── Model architecture card ───────────────────────────────────────────────
    with st.expander("Model architecture & hyperparameters"):
        h1, h2 = meta.get("hidden", (256, 128))
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Input features | 63 (21 landmarks × x,y,z) |
| Hidden layers | {h1} → {h2} |
| Output classes | {len(labels)} |
| Activation | ReLU |
| Optimiser | Adam |
| Early stopping | Yes (patience = 20 epochs) |
| Validation fraction | 10% of training data |
| Augmentation factor | ×{meta.get("aug_factor", "—")} |
| Training samples used | {meta.get("n_train", "—"):,} |
| Test samples | {meta.get("n_test", "—")} |
| Total epochs | {meta.get("n_iter", "—")} |
        """)
