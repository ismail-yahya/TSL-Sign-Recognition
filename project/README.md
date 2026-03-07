# TID Sign Language - Phase 3

Real-time Turkish Sign Language recognition: camera, MediaPipe, Transformer model inference.

## Pipeline

```
Camera → MediaPipe → Landmark Extraction → Frame Buffer (80) → Model Input (1, 80, 255) → Predictor → Display
```

## Features

- **Phase 1:** Webcam, MediaPipe Holistic, (255,) feature extraction
- **Phase 2:** Sliding-window frame buffer, preprocessing
- **Phase 3:** Transformer model inference, label mapping, confidence threshold, debounce

## Setup

```bash
pip install -r requirements.txt
```

Place `best_model_transformer.keras` in `model/`.

**Note:** Use `opencv-python` (not headless) for display. MediaPipe 0.10.14. NumPy &lt; 2.

## Run

```bash
python main.py
```

- Press **q** to exit
- Predictions appear in console when confidence &gt; 0.85 (debounced)

## Structure

```
project/
├── main.py
├── model/
│   └── best_model_transformer.keras
├── inference/
│   ├── camera.py
│   ├── mediapipe_detector.py
│   ├── landmark_extractor.py
│   ├── frame_buffer.py
│   ├── preprocessing.py
│   └── predictor.py
├── utils/
│   └── labels_loader.py
└── data/
    └── labels.csv
```
