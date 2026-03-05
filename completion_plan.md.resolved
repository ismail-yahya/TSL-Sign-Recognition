# Graduation Project Completion Plan: Sign Language to Text & Speech Conversion System

Greetings, Team! As your Academic Advisor, I have reviewed your current progress and the thesis draft. You have done excellent work by achieving 90% accuracy with a Transformer model—this is a very strong foundation.

Below is the roadmap to take this from a set of Jupyter Notebooks to a fully functional, deliverable academic project.

---

## 🏗️ Phase 1: Technical Integration (The "Connecting the Dots" Phase)

### 1. Unified Inference Script
Currently, your logic is spread across three notebooks. You need a single Python script (e.g., `inference.py`) that:
- **Init**: Loads the Keras model ([best_model_transformer.keras](file:///c:/Users/ISMAIL%20YAHYA/Desktop/%D9%85%D8%B4%D8%A7%D8%B1%D9%8A%D8%B9%20ASL%20Sign%20Recognition/best_model_transformer.keras)) and the Piper TTS engine.
- **Process**: Takes a video stream (OpenCV), extracts landmarks (MediaPipe), and feeds them into the model.
- **Queue**: Implements a sliding window or a trigger mechanism to detect when a sign is completed.

### 2. Live Recognition Loop
You need a "Buffer" system. 
- Sign language is temporal. You should collect ~30-60 frames (depending on your model input shape), run the prediction, and then reset or slide the buffer.

### 3. Piper TTS Integration
- Instead of just printing "Merhaba", call the Piper subprocess or library to speak the text.
- **Optimization**: Use a "Debounce" logic so the system doesn't try to speak the same word multiple times while the person is still holding the sign.

---

## 🖥️ Phase 2: Application Development (GUI)

To "WOW" the jury, you need a standalone application, not just a console window.
- **Framework**: Use **PyQt6** or **Tkinter**.
- **Features**:
  - Live Camera Feed with Landmark Overlays (helps the user see if they are in frame).
  - Large Text Display for the recognized word/sentence.
  - "Speak" button (manual) or "Auto-Speak" toggle.
  - History log of converted text.

---

## 📝 Phase 3: Academic & Documentation Requirements

### 1. Practicum/Internship Diary (Staj Defteri)
Since you mentioned this, ensure you document the **daily technical challenges**:
- Day 1-3: Data collection and landmark extraction issues.
- Day 4-7: Model architecture selection (Why Transformer vs LSTM?).
- Day 8-12: Hyperparameter tuning and accuracy optimization.
- Day 14-20: Integration of TTS and GUI development.

### 2. Completing the Thesis (Tez)
- **Results Section**: Add confusion matrices, accuracy graphs from Notebook 03.
- **Discussion**: Compare your Transformer results with literature (usually CNNs).
- **Conclusion**: Summarize the impact of using TID instead of ASL.
- **References**: Ensure all citations in Chapter 6 are correctly formatted in the text.

---

## 📅 Weekly Timeline (4-Week Sprint)

| Week | Focus | Tasks |
| :--- | :--- | :--- |
| **Week 1** | **Integration** | Create `inference.py`. Link Model + Piper. Test real-time lag. |
| **Week 2** | **GUI Development** | Build the desktop app interface. Add camera view and history list. |
| **Week 3** | **Refinement** | Add "Continuous Sign" logic (making sentences). Fine-tune the "Speak" trigger. |
| **Week 4** | **Documentation** | Finalize Thesis text. Write the Practicum report. Prepare the presentation slides. |

---

## 💡 Academic Advisor Tips for Excellence

1. **Non-Manual Markers**: Even if your model doesn't use them heavily, mention in the thesis that you extracted 8 facial landmarks for "potential future use in emotional context." This shows foresight.
2. **Failure Analysis**: In your report, don't just show the 90% success. Show a few signs the model confuses (e.g., "A" vs "E" if hand shapes are similar) and explain *why*. Juries love critical thinking.
3. **User Experience**: If the sign is "Hard", can the system suggest "Did you mean 'Heavy'?" based on context? (Simple NLP fix).

---

> [!IMPORTANT]
> **Next Immediate Step**: I will generate a `README.md` for your repo so you and Hasan can start working on the `inference.py` and GUI in parallel.
