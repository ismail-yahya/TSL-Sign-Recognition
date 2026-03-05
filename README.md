# TID Sign Language to Text & Speech Conversion 🤘🗣️

## 📝 Project Overview

This project converts **Turkish Sign Language (TİD)** into text and audible speech in real-time. It uses a **Transformer-based Deep Learning model** for sign recognition and the **Piper TTS engine** for high-performance Turkish speech synthesis.

## 🚀 Current Status

- **Landmark Extraction**: DONE (MediaPipe)
- **Model Training**: DONE (90% Accuracy with Transformer)
- **TTS Engine**: CONFIGURED (Piper tr_TR model)
- **Integration**: IN PROGRESS 🛠️

## 📂 Project Structure

- `01_TID_Landmark_Extraction.ipynb`: Script for extracting XYZ coordinates from video frames.
- `02_TID_Pre_Processing.ipynb`: Data normalization and sequence padding.
- `03_TID_Model_Training.ipynb`: Transformer model definition and training history.
- `Piper/`: Contains the TTS engine and Turkish voice models.
- `best_model_transformer.keras`: The production-ready recognition model.

## 🛠️ Work-in-Progress (Collaboration Guide)

### For Ismail & Hasan:

To work in parallel, we need to split the tasks:

1.  **Task A: Inference Engine (`inference_engine.py`)**
    - Load `best_model_transformer.keras`.
    - Implement the `predict_sign(sequence)` function.
    - Handle the sliding window (e.g., collect 30 frames, predict, shift).

2.  **Task B: Voice Controller (`voice_controller.py`)**
    - Create a class to interface with the `Piper/` folder.
    - Function `speak(text)` that calls Piper without blocking the main thread.

3.  **Task C: Desktop GUI (`main_app.py`)**
    - Build a UI using PyQt6 or Tkinter.
    - Integrate the camera feed and display the `predict_sign` results.
    - Add a "History" sidebar to show previous signs.

## 🏗️ How to Run (Future)

```bash
python main_app.py
```

## 📚 Academic Requirements

- [ ] Complete "Staj Defteri" (Technical Logs).
- [ ] Finalize Chapter 5 (Results) and Chapter 6 (Discussion) of the Thesis.
- [ ] Prepare the Presentation PPT.

---

**Advisor's Note**: Keep the code modular. Ensure that `inference_engine.py` can work independently of the GUI for testing!
