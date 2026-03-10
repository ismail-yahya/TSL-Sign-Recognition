# TID Sign Language to Text & Speech Conversion 🤘🗣️

## 📝 Project Overview

This project converts **Turkish Sign Language (TİD)** into text and audible speech in real-time. It uses a **Transformer-based Deep Learning model** for sign recognition and the **Piper TTS engine** for high-performance Turkish speech synthesis.

## 🏗️ Project Structure

The repository has been organized for production and GUI development:

```text
ASL-Sign-Recognition/
├── assets/             # Voice models, label mappings, and icons
├── models/             # Trained Keras/TensorFlow models
├── notebooks/          # Research, data extraction, and training notebooks
├── src/                # Core Python source code
│   ├── data_pipeline.py    # Landmark extraction and preprocessing
│   ├── inference_engine.py # Main AI inference logic
│   └── speech_engine.py    # Non-blocking TTS integration
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── .gitignore          # Git exclusion rules
```

## 🚀 Key Features

- **MediaPipe Integration**: Real-time extraction of 85 key landmarks.
- **Transformer Model**: High-accuracy (90%) temporal pattern recognition.
- **Robust Preprocessing**: EMA smoothing, linear interpolation for missing frames, and spatial normalization.
- **Non-Blocking TTS**: Piper-based Turkish speech synthesis running in a separate thread.
- **Stability System**: Activity filtering gates and voting buffers to ensure reliable detection.

## 🛠️ Installation & Setup

1.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Inference Engine (Webcam)**:
    ```bash
    python src/inference_engine.py
    ```

## 📅 Future Roadmap

- **Phase 2: GUI Application**: Building a professional Desktop App using PyQt6.
- **Phase 3: Documentation**: Finalizing the Thesis (Tez) and Internship Report (Staj Defteri).

---

**Advisor's Note**: The project is now standardized. Use the `src/` folder for all logic and `assets/` for external resources.
