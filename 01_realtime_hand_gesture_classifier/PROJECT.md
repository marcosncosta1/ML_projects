# Real-Time Chinese Hand Sign Classifier (数字 0–9)

## Overview
A Streamlit app that lets you build your own Chinese number hand-sign classifier
from scratch — collect training data via webcam, train an MLP, and run real-time
inference, all in the same browser UI.

## Category
Machine Learning / Computer Vision

## Stack
| Component | Library |
|---|---|
| UI | Streamlit ≥ 1.34 (`@st.fragment` for live video) |
| Hand detection | MediaPipe Hands (21 landmarks) |
| Webcam | OpenCV (`cv2.CAP_AVFOUNDATION` on macOS) |
| Model | scikit-learn `MLPClassifier` in a `Pipeline` |
| Charts | Plotly |
| Model persistence | joblib |

## App Tabs

### 📸 Data Collection
- Live webcam preview with MediaPipe skeleton overlay (10 Hz fragment)
- Spacebar or button capture → saves JPEG + landmark CSV row
- Label selector with Chinese character, pinyin, and gesture description
- Per-class sample progress bar (green ≥ 20, orange < 20, red = 0)

### 🧠 Train Model
- Dataset table with per-class counts and ✅/⚠️/❌ status
- Augmentation factor slider (1–20× copies per raw sample)
- Stratified train/test split — augmentation applied to training split only
- Trains `StandardScaler → MLP(256, 128)` with early stopping
- Results: test accuracy, confusion matrix, per-class precision/recall/F1

### 🎯 Live Inference
- Real-time camera feed with predicted digit + confidence overlaid
- Confidence threshold slider
- Horizontal bar chart showing probability for all 10 classes
- Loads model from `models/mlp_pipeline.pkl` or from session state

## Gestures

| Digit | Char | Pinyin | Description |
|---|---|---|---|
| 0 | 零 | líng | Closed fist or circle |
| 1 | 一 | yī | Index finger up |
| 2 | 二 | èr | Index + middle up (V) |
| 3 | 三 | sān | Index + middle + ring up |
| 4 | 四 | sì | Four fingers, thumb tucked |
| 5 | 五 | wǔ | Open palm, all five fingers |
| 6 | 六 | liù | Thumb + pinky extended |
| 7 | 七 | qī | Thumb + index + middle |
| 8 | 八 | bā | Thumb + index L-shape |
| 9 | 九 | jiǔ | Index finger hooked |

## Architecture

```
Webcam frame (640×480)
  └─► MediaPipe Hands
        └─► 21 landmarks (x, y, z)
              └─► Normalize (wrist → origin, scale by lm9 distance)
                    └─► 63-dim feature vector
                          ├─► [Training] Augment → StandardScaler → MLP → pkl
                          └─► [Inference] StandardScaler → MLP → class probs
```

## Augmentation Pipeline (landmark-space)
1. Gaussian noise (σ = 0.02)
2. Random scale (0.85–1.15×)
3. Random xy-rotation (±25°)
4. Random xy-translation (±0.12)

## File Structure
```
01_realtime_hand_gesture_classifier/
├── app.py                        # Entry point
├── config.py                     # Paths, class names, Chinese descriptions
├── components/
│   ├── camera.py                 # @st.cache_resource VideoCapture
│   ├── mediapipe_utils.py        # Landmark extraction + normalization
│   ├── augmentation.py           # Landmark-space augmentation
│   └── dataset.py                # Save/load samples
├── tabs/
│   ├── tab_collect.py            # Data collection tab + fragment
│   ├── tab_train.py              # Training tab
│   └── tab_infer.py              # Inference tab + fragment
├── data/
│   ├── raw/{0..9}/               # JPEG images per class
│   └── landmarks.csv             # Feature rows
├── models/
│   └── mlp_pipeline.pkl          # Trained model
└── requirements.txt
```

## Quick Start
```bash
cd 01_realtime_hand_gesture_classifier
pip install -r requirements.txt
streamlit run app.py
```

**macOS camera permission**: on first run, macOS will show a permission dialog.
If the camera appears black, grant Terminal/iTerm camera access in
System Settings → Privacy & Security → Camera, then restart.

## Milestones
- [x] MediaPipe landmark extraction + normalization
- [x] Data collection tab with live preview, spacebar capture, label selector
- [x] Dataset storage (JPEG + CSV)
- [x] Augmentation pipeline (noise, scale, rotation, translation)
- [x] Training tab with stratified split, confusion matrix, per-class metrics
- [x] Live inference tab with confidence overlay and probability chart
- [ ] Collect 20+ samples per class and evaluate
- [ ] README demo GIF / video
- [ ] HuggingFace Spaces deployment (static image fallback)
- [ ] Optionally: CNN on raw ROI crops for comparison

## Notes
<!-- Add observations, accuracy results, tricky gestures here -->
