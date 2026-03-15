import cv2
import numpy as np
import streamlit as st


@st.cache_resource
def get_camera() -> cv2.VideoCapture:
    """Single VideoCapture instance shared across all fragments and tabs."""
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def read_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """
    Read one frame from the camera, flip horizontally (mirror mode),
    and convert to RGB for display + MediaPipe.
    Returns (success, rgb_frame).
    """
    ret, frame = cap.read()
    if not ret:
        return False, None
    frame = cv2.flip(frame, 1)
    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def is_camera_working(cap: cv2.VideoCapture) -> bool:
    """Check that the camera is open and returning non-black frames."""
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    if not ret or frame is None:
        return False
    # Silent macOS permission failure = solid black frame
    return frame.mean() > 1.0
