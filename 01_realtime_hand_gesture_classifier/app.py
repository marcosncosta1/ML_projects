import streamlit as st

st.set_page_config(
    page_title="Chinese Hand Signs 0–9",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "current_frame":    None,
    "current_landmarks": None,
    "hand_detected":    False,
    "trained_model":    None,
    "infer_pred":       None,
    "infer_conf":       0.0,
    "infer_probas":     None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── App header ────────────────────────────────────────────────────────────────
st.title("🤚 Chinese Hand Sign Classifier  —  数字 0–9")
st.caption(
    "Collect training samples → train a gesture model → classify in real time. "
    "Supports Chinese number signs 零一二三四五六七八九."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_collect, tab_train, tab_infer = st.tabs([
    "📸 Data Collection",
    "🧠 Train Model",
    "🎯 Live Inference",
])

from tabs import tab_collect as tc, tab_train as tt, tab_infer as ti  # noqa: E402

with tab_collect:
    tc.render()

with tab_train:
    tt.render()

with tab_infer:
    ti.render()
