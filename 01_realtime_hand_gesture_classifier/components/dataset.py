import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import config


def ensure_dirs() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    for i in range(10):
        (config.RAW_DIR / str(i)).mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(exist_ok=True)


def save_sample(frame_rgb: np.ndarray, landmarks: np.ndarray, label: str) -> Path:
    """
    Persist one training sample: JPEG image + landmark row in CSV.

    Parameters
    ----------
    frame_rgb : H×W×3 uint8 RGB frame (will be saved as BGR JPEG)
    landmarks : (63,) float32 normalised landmark vector
    label     : digit string "0"–"9"
    """
    ensure_dirs()
    label_dir = config.RAW_DIR / label
    idx = len(list(label_dir.glob("*.jpg")))
    img_path = label_dir / f"{idx:05d}.jpg"

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    row = dict(zip(config.LANDMARK_COLS, landmarks.tolist()))
    row["label"] = int(label)
    row["image_path"] = str(img_path)

    df = pd.DataFrame([row])
    header = not config.LANDMARKS_CSV.exists()
    df.to_csv(config.LANDMARKS_CSV, mode="a", index=False, header=header)

    return img_path


def load_dataset() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (X, y) arrays or (None, None) if no data exists."""
    if not config.LANDMARKS_CSV.exists():
        return None, None
    df = pd.read_csv(config.LANDMARKS_CSV)
    X = df[config.LANDMARK_COLS].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


def get_class_counts() -> dict[str, int]:
    """Count images on disk per class (source of truth, not CSV)."""
    ensure_dirs()
    return {
        str(i): len(list((config.RAW_DIR / str(i)).glob("*.jpg")))
        for i in range(10)
    }


def total_samples() -> int:
    return sum(get_class_counts().values())
