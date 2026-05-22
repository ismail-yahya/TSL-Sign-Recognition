# TİD Sign Language to Text & Speech Conversion 🤟🗣️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![MediaPipe 0.10](https://img.shields.io/badge/MediaPipe-0.10.9-green.svg)](https://mediapipe.dev)
[![PyQt5](https://img.shields.io/badge/PyQt-5.15.10-blue.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A professional, real-time **Turkish Sign Language (TİD - Türk İşaret Dili)** translation system that converts sign language captured via webcam into written text and natural-sounding spoken Turkish speech.

Developed as a university graduation thesis, this project achieves a state-of-the-art **90.96% accuracy** on the **AUTSL dataset** (226 sign classes) using an optimized **Encoder-Only Transformer** deep learning model.

---

## 📸 System Previews

### 1. Graphical User Interface (GUI Mode)

The application features a sleek, dark-themed, and hardware-accelerated interface built with **PyQt5**. It offloads all video capture and deep learning inference to a dedicated background worker thread (`QThread`), guaranteeing a fluid, freeze-free 30+ FPS visual experience.

- **Key features of the GUI**:
  - Live camera stream with visual MediaPipe landmark overlays.
  - Real-time rolling buffer fill indicator.
  - Dynamic, large-font detection card with confidence level.
  - Animated TTS speaking indicator ("🔊 Speaking: [Word]").
  - Recent history log tracking the last 10 detected words.
  - Easy-to-use toggle switches for system configurations (Speech ON/OFF).

---

## 🏗️ Project Architecture & Directory Structure

The project has been fully restructured for modularity, speed, and academic presentation:

```text
TSL-Sign-Recognition/
├── main.py                     # Entry point: Terminal / Headless mode
├── main_gui.py                 # Entry point: PyQt5 Graphical Desktop App
├── requirements.txt            # Python environment dependencies
├── README.md                   # System documentation (This file)
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core Python Modules
│   ├── __init__.py             # Package marker
│   ├── model.py                # Keras Transformer model architecture
│   ├── data_pipeline.py        # MediaPipe landmark extraction, normalization & SequenceBuffer
│   ├── inference_engine.py     # Main prediction logic & state machine
│   └── speech_engine.py        # Non-blocking TTS (Piper) speech loop
│
├── ui/                         # PyQt5 Graphical Interface
│   ├── __init__.py             # Package marker
│   └── app_window.py           # GUI MainWindow, custom controls & worker threads
│
├── models/                     # Deep Learning Models
│   └── best_model_transformer.keras  # Trained Transformer model file (90.96% AUTSL accuracy)
│
├── assets/                     # Shared System Assets
│   ├── SignList_ClassId_TR_EN.csv    # Single source of truth for Class ID mapping
│   ├── tr_TR-dfki-medium.onnx        # Piper TTS Turkish voice model
│   ├── tr_TR-dfki-medium.onnx.json   # Piper TTS voice configuration
│   └── figures/                      # Saved plots and illustrations for thesis documentation
│       ├── sekil_3_optimization.png
│       └── sekil_4_normalization.png
│
└── docs/                       # Academic Documents & Scripts
    ├── completion_plan.md      # Graduation roadmap and advisor notes
    └── scripts/                # Independent utility scripts for thesis visuals
        ├── normalization_visualization.py
        └── optimize_visualization.py
```

---

## 🧠 Deep Learning Model Architecture

The system uses an **Encoder-Only Transformer** model optimized for sequential sign gesture classification. Rather than using raw video frames, the model is trained on preprocessed temporal sequences of 3D skeleton coordinates extracted by MediaPipe.

### 📐 Input Feature Representation
* **Temporal Window (Sequence Length)**: $80$ frames (approximately 2.6 seconds at 30 FPS).
* **Input Landmarks**: $85$ hand, lip, and body keypoints extracted per frame:
  * **Left Hand**: $21$ keypoints
  * **Right Hand**: $21$ keypoints
  * **Lips**: $40$ keypoints
  * **Pose / Shoulder**: $3$ keypoints
* **Features per Keypoint**: $3$ spatial coordinates ($X, Y, Z$).
* **Input Tensor Shape**: `(Batch, 80, 255)` (where $85 \text{ landmarks} \times 3 \text{ coordinates} = 255$ features per frame).

---

### 🧱 Layer-by-Layer Architecture

| Layer / Component | Type | Output Shape | Parameters | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Input** | `InputLayer` | `(None, 80, 255)` | 0 | Batched sequence of temporal landmarks. |
| **Feature Projection** | `Dense` | `(None, 80, 256)` | 65,536 | Projects 255 input features into a 256-dimensional embedding space. |
| **Positional Embedding** | `PositionalEmbedding` | `(None, 80, 256)` | 20,480 | Learnable spatial-temporal position embeddings added to sequence vectors. |
| **Transformer Block 0** | `TransformerBlock` | `(None, 80, 256)` | 2,498,816 | First Encoder Layer: 8-head self-attention + MLP (768 units) + LayerNorm + Dropout. |
| **Transformer Block 1** | `TransformerBlock` | `(None, 80, 256)` | 2,498,816 | Second Encoder Layer: 8-head self-attention + MLP (768 units) + LayerNorm + Dropout. |
| **Transformer Block 2** | `TransformerBlock` | `(None, 80, 256)` | 2,498,816 | Third Encoder Layer: 8-head self-attention + MLP (768 units) + LayerNorm + Dropout. |
| **Temporal Pooling** | `GlobalAveragePooling1D` | `(None, 256)` | 0 | Collapses the time dimension ($80$ frames) into a fixed-length representation. |
| **Dropout Gate** | `Dropout` (10%) | `(None, 256)` | 0 | Regularization to prevent model overfitting. |
| **Prep Classification** | `Dense` | `(None, 256)` | 65,792 | Fully-connected dense layer mapping features prior to final classification. |
| **Dropout Gate** | `Dropout` (10%) | `(None, 256)` | 0 | Secondary regularizer. |
| **Output / Classification**| `Dense` (Softmax) | `(None, 226)` | 58,082 | Maps output to the 226 Turkish Sign Language classes in the AUTSL dataset. |

---

### 📊 Model Complexity & Parameters
* **Total Parameters**: **7,706,338** (approx. **29.40 MB**)
* **Trainable Parameters**: **7,706,338** (100% trainable)
* **Non-trainable Parameters**: **0**

---

## 🛠️ Installation & Virtual Environment Setup

Follow these steps to set up and run the system on your local machine:

### 1. Prerequisites

- **OS**: Windows (Recommended), Linux, or macOS.
- **Python**: `3.10` or `3.11` is highly recommended.

### 2. Set Up a Virtual Environment (Recommended)

Open a terminal in the project root directory and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows Powershell)
.\venv\Scripts\activate

# Or Activate virtual environment (Windows Command Prompt)
.\venv\Scripts\activate.bat

# Or Activate virtual environment (Linux/macOS)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!TIP]
> **Windows PyAudio Troubleshooting**:  
> If the installation of `pyaudio` fails with a C++ compiler error during the `pip install` step, you can install the precompiled wheel directly:
>
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
>
> Alternatively, download the matching `.whl` file for your Python version from [Christoph Gohlke's Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install it using:
> `pip install PyAudio-XXXX.whl`

---

## 🚀 Execution Guide

The system supports two execution modes: **Graphical App (GUI)** and **Terminal Command Line**.

### 1. Graphical Desktop Application (Recommended)

This runs the full PyQt5 desktop interface.

```bash
python main_gui.py
```

#### Available Command Line Arguments for GUI Mode:

Customize the application's runtime settings directly via CLI flags:

| Flag           | Type    | Default                               | Description                                                              |
| -------------- | ------- | ------------------------------------- | ------------------------------------------------------------------------ |
| `--camera`     | `int`   | `0`                                   | Camera device index (e.g. `0` for built-in, `1` for external usb webcam) |
| `--model`      | `str`   | `models/best_model_transformer.keras` | Custom path to a trained Keras model                                     |
| `--labels`     | `str`   | `assets/SignList_ClassId_TR_EN.csv`   | Path to the labels translation CSV file                                  |
| `--confidence` | `float` | `0.6`                                 | Minimum prediction confidence threshold (0.0 to 1.0)                     |
| `--debounce`   | `int`   | `20`                                  | Frames to wait after a detection before triggering another prediction    |
| `--no-speech`  | `flag`  | `False`                               | Disables Piper Text-to-Speech engine                                     |

_Example of a strict, silent run on an external camera_:

```bash
python main_gui.py --camera 1 --confidence 0.80 --no-speech
```

---

### 2. Terminal / Headless Mode

This runs the system inside a lightweight command-line script. An OpenCV window will display the camera stream with FPS and latency markers, and recognized signs will print directly to your terminal.

```bash
python main.py
```

#### Available Command Line Arguments for Terminal Mode:

Terminal mode supports the exact same parameters as the GUI mode:

```bash
python main.py --camera 0 --confidence 0.65 --debounce 25
```

Press **`q`** or **`Ctrl + C`** in the terminal to exit cleanly.

---

## 📊 Evaluation & Visualizations

We have included scripts to generate visualizations showing our preprocessing pipelines and system performance:

- **Feature Normalization Visualizer**: `docs/scripts/normalization_visualization.py`
- **Performance Optimization Visualizer**: `docs/scripts/optimize_visualization.py`

You can run these scripts independently to generate high-resolution figures for academic papers and slides:

```bash
python docs/scripts/normalization_visualization.py
python docs/scripts/optimize_visualization.py
```

Generated figures are saved automatically in `assets/figures/`.

---

## 🎓 Academic Citations & Thesis Context

This system was designed for academic evaluation. Below is the comparative analysis of our system against the literature:

| Study                  | Dataset   | Class Count |  Accuracy  | Real-Time Support |    TTS Support     |
| :--------------------- | :-------- | :---------: | :--------: | :---------------: | :----------------: |
| Sincan & Keles (2020)  | AUTSL     |     226     |   ~60.0%   |        ❌         |         ❌         |
| GesSpy - Ansari (2024) | Custom    |     15      |   ~88.0%   |        ✅         |         ❌         |
| Özdemir et al. (2023)  | AUTSL     |     226     |   ~85.3%   |        ❌         |         ❌         |
| **This Work (2026)**   | **AUTSL** |   **226**   | **90.96%** | **✅ (30+ FPS)**  | **✅ (Piper TTS)** |

---

## 🤝 Contributors & Credits

- **İsmail YAHYA** — Developer / Deep Learning Researcher
- **Hasan ELRECEB** — Developer / Deep Learning Researcher
- **Model & Preprocessing Development**: Deep learning research on AUTSL dataset.
- **TTS Engine**: Powered by [Piper Speech Synthesis](https://github.com/rhasspy/piper).

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
