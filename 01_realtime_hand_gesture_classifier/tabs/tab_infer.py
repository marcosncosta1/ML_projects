import cv2
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

import config
from components.camera import get_camera, read_frame
from components.mediapipe_utils import get_hands, extract_and_annotate


def _get_model():
    """Return model from session state (set by training tab) or load from disk."""
    if "trained_model" in st.session_state and st.session_state["trained_model"] is not None:
        return st.session_state["trained_model"]
    if config.MODEL_PATH.exists():
        model = joblib.load(config.MODEL_PATH)
        st.session_state["trained_model"] = model
        return model
    return None


def render() -> None:
    st.header("Live Inference")

    model = _get_model()
    if model is None:
        st.info("No trained model found — train one in the **Train Model** tab first.")
        return

    conf_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
        help="Predictions below this are shown as 'uncertain'",
        key="infer_threshold",
    )

    left, right = st.columns([2, 1], gap="medium")
    with left:
        _inference_feed(model, conf_threshold)
    with right:
        _prediction_panel()


@st.fragment(run_every=0.1)
def _inference_feed(model, conf_threshold: float) -> None:
    cap   = get_camera()
    hands = get_hands()
    ok, frame = read_frame(cap)

    if not ok or frame is None:
        st.error("Camera unavailable.")
        return

    landmarks, annotated = extract_and_annotate(frame, hands)

    if landmarks is not None:
        probas      = _predict_all(model, landmarks)
        pred_idx    = int(np.argmax(probas))
        pred_label  = config.CLASS_NAMES[pred_idx]
        confidence  = float(probas[pred_idx])
        char        = config.CHINESE_GESTURES[pred_label][0]

        st.session_state["infer_pred"]   = pred_label
        st.session_state["infer_conf"]   = confidence
        st.session_state["infer_probas"] = probas

        if confidence >= conf_threshold:
            text  = f"{pred_label} {char}  {confidence:.0%}"
            color = (0, 230, 80)
        else:
            text  = f"? ({confidence:.0%})"
            color = (255, 140, 0)

        # Draw text on annotated frame (annotated is RGB uint8)
        display = annotated.copy()
        cv2.putText(display, text, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)
        st.image(display, use_container_width=True)
    else:
        st.session_state["infer_pred"]   = None
        st.session_state["infer_probas"] = None
        st.image(frame, caption="No hand detected", use_container_width=True)


@st.fragment(run_every=0.12)
def _prediction_panel() -> None:
    pred   = st.session_state.get("infer_pred")
    conf   = st.session_state.get("infer_conf", 0.0)
    probas = st.session_state.get("infer_probas")

    if pred is not None and probas is not None:
        char, pinyin, desc = config.CHINESE_GESTURES[pred]
        st.markdown(
            f"<div style='font-size:5rem;text-align:center'>{char}</div>"
            f"<div style='font-size:2rem;text-align:center'>{pred}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='text-align:center;color:grey'>{pinyin} — {desc}</div>",
                    unsafe_allow_html=True)
        st.metric("Confidence", f"{conf:.1%}")

        pred_idx = int(pred)
        fig = go.Figure(go.Bar(
            x=[f"{i} {config.CHINESE_GESTURES[str(i)][0]}" for i in range(10)],
            y=[float(probas[i]) for i in range(10)],
            marker_color=[
                "#2ecc71" if i == pred_idx else "#455a64"
                for i in range(10)
            ],
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 1], title="Confidence"),
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            "<div style='font-size:4rem;text-align:center;color:grey'>🤚</div>"
            "<div style='text-align:center;color:grey'>Show your hand</div>",
            unsafe_allow_html=True,
        )


def _predict_all(model, landmarks: np.ndarray) -> np.ndarray:
    """Return a (10,) probability array aligned to class indices 0–9."""
    probas   = model.predict_proba([landmarks])[0]
    classes  = model.classes_
    full     = np.zeros(10, dtype=np.float32)
    for cls, p in zip(classes, probas):
        full[int(cls)] = p
    return full
