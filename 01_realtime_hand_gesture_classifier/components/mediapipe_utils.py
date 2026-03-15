import numpy as np
import cv2
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from pathlib import Path

# Landmark connections for drawing (same topology as the old solutions API)
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                  # palm
]

_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


@st.cache_resource
def get_hands() -> HandLandmarker:
    """Shared HandLandmarker instance — never recreate inside a fragment."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {_MODEL_PATH}. "
            "Run: python3 -c \"import urllib.request; "
            "urllib.request.urlretrieve("
            "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task', "
            "'models/hand_landmarker.task')\""
        )
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_and_annotate(
    frame_rgb: np.ndarray,
    detector: HandLandmarker,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Run hand landmark detection on an RGB frame.

    Returns
    -------
    landmarks : np.ndarray (63,) — normalised, or None if no hand detected
    annotated : np.ndarray — RGB frame with skeleton overlay
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    annotated = frame_rgb.copy()

    if not result.hand_landmarks:
        return None, annotated

    hand = result.hand_landmarks[0]
    h, w = frame_rgb.shape[:2]

    # Draw skeleton using OpenCV
    pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in _CONNECTIONS:
        cv2.line(annotated, pts_px[a], pts_px[b], (0, 200, 100), 2, cv2.LINE_AA)
    for px, py in pts_px:
        cv2.circle(annotated, (px, py), 4, (255, 255, 255), -1)
        cv2.circle(annotated, (px, py), 4, (0, 150, 80), 1)

    # Extract + normalise
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
    landmarks = _normalise(pts)
    return landmarks, annotated


def _normalise(pts: np.ndarray) -> np.ndarray:
    """
    Translate so wrist (lm 0) is origin; scale by wrist→middle-MCP (lm 9) distance.
    All three axes share the same scale so z depth cues are preserved.
    """
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()
