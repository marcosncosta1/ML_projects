from pathlib import Path

CLASS_NAMES = [str(i) for i in range(10)]

CHINESE_GESTURES = {
    "0": ("零", "líng",  "Closed fist or circle with thumb + index"),
    "1": ("一", "yī",    "Index finger extended upward"),
    "2": ("二", "èr",    "Index + middle fingers up (V)"),
    "3": ("三", "sān",   "Index + middle + ring fingers up"),
    "4": ("四", "sì",    "Four fingers up, thumb tucked in"),
    "5": ("五", "wǔ",    "Open palm, all five fingers spread"),
    "6": ("六", "liù",   "Thumb + pinky extended, others curled"),
    "7": ("七", "qī",    "Thumb + index + middle extended"),
    "8": ("八", "bā",    "Thumb + index in L-shape (gun)"),
    "9": ("九", "jiǔ",   "Index finger hooked / curled"),
}

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
LANDMARKS_CSV = DATA_DIR / "landmarks.csv"
MODEL_PATH    = BASE_DIR / "models" / "mlp_pipeline.pkl"

# 21 landmarks × 3 axes = 63 features
LANDMARK_COLS = [f"lm_{i}_{ax}" for i in range(21) for ax in ["x", "y", "z"]]
