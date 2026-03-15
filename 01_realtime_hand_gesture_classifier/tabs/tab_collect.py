import cv2
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import config
from components.camera import get_camera, read_frame
from components.mediapipe_utils import get_hands, extract_and_annotate
from components.dataset import save_sample, get_class_counts

# ── Spacebar JS ───────────────────────────────────────────────────────────────
_SPACEBAR_JS = """
<script>
(function () {
    var lastFired = 0;
    window.parent.document.addEventListener('keydown', function (e) {
        if (
            e.code === 'Space' &&
            Date.now() - lastFired > 400 &&
            document.activeElement.tagName !== 'INPUT' &&
            document.activeElement.tagName !== 'TEXTAREA'
        ) {
            lastFired = Date.now();
            e.preventDefault();
            var buttons = window.parent.document.querySelectorAll('button');
            for (var btn of buttons) {
                if (btn.innerText.trim().startsWith('Capture')) {
                    btn.click();
                    break;
                }
            }
        }
    }, false);
})();
</script>
"""


def render() -> None:
    # Inject spacebar once per page load (outside any fragment)
    components.html(_SPACEBAR_JS, height=0)

    left, right = st.columns([2, 1], gap="medium")

    with right:
        _controls()

    with left:
        _live_preview()


# ── Live preview (runs every 100 ms) ─────────────────────────────────────────
@st.fragment(run_every=0.1)
def _live_preview() -> None:
    cap = get_camera()
    hands = get_hands()
    ok, frame = read_frame(cap)

    if not ok or frame is None:
        st.error(
            "Camera not accessible. "
            "Go to **System Settings → Privacy & Security → Camera** "
            "and grant permission to your terminal app, then restart."
        )
        return

    landmarks, annotated = extract_and_annotate(frame, hands)

    # Always keep session state current so capture button can read it
    st.session_state["current_frame"] = frame
    st.session_state["current_landmarks"] = landmarks
    st.session_state["hand_detected"] = landmarks is not None

    caption = "✅ Hand detected — press **Capture** or **Space**" if landmarks is not None \
              else "🔍 No hand detected — show your hand to the camera"
    st.image(annotated, caption=caption, use_container_width=True)


# ── Controls panel ────────────────────────────────────────────────────────────
def _controls() -> None:
    st.subheader("Label & Capture")

    label = st.selectbox(
        "Gesture class",
        options=config.CLASS_NAMES,
        format_func=lambda k: f"{k}  {config.CHINESE_GESTURES[k][0]}  ({config.CHINESE_GESTURES[k][1]})",
        key="selected_label",
    )

    char, pinyin, desc = config.CHINESE_GESTURES[label]
    st.markdown(
        f"<div style='font-size:3rem;text-align:center'>{char}</div>"
        f"<div style='text-align:center;color:grey'>{pinyin} — {desc}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    capture_clicked = st.button(
        "📸 Capture  (or press Space)",
        type="primary",
        use_container_width=True,
    )

    if capture_clicked:
        frame = st.session_state.get("current_frame")
        lm    = st.session_state.get("current_landmarks")
        if lm is not None and frame is not None:
            save_sample(frame, lm, label)
            counts = get_class_counts()
            st.success(f"Saved! Total for {label} {char}: **{counts[label]}** samples")
        else:
            st.warning("No hand detected in the last frame — keep your hand in view and try again.")

    st.divider()
    _sample_chart()


def _sample_chart() -> None:
    counts = get_class_counts()
    total  = sum(counts.values())
    st.metric("Total samples", total)

    colors = [
        "#2ecc71" if counts[str(i)] >= 20
        else "#e67e22" if counts[str(i)] > 0
        else "#e74c3c"
        for i in range(10)
    ]
    labels = [f"{i} {config.CHINESE_GESTURES[str(i)][0]}" for i in range(10)]
    values = [counts[str(i)] for i in range(10)]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=0, r=0),
        yaxis_title="Samples",
        xaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 ≥20  🟠 1–19  🔴 0")
