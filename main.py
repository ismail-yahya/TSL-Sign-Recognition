"""
main.py — TSL Sign Language Recognition System
================================================
Entry point for the Turkish Sign Language (TİD) Real-Time Recognition System.

Usage:
    python main.py                          # Default: camera 0, all defaults
    python main.py --camera 1               # Use a different camera
    python main.py --confidence 0.75        # Stricter confidence threshold
    python main.py --no-speech              # Disable TTS output
    python main.py --help                   # Show all options

Project structure:
    src/model.py            — Transformer architecture (TransformerBlock, PositionalEmbedding)
    src/data_pipeline.py    — LandmarkExtractor, FeatureBuilder, SequenceBuffer
    src/inference_engine.py — SignLanguageInferenceEngine (main AI logic)
    src/speech_engine.py    — SpeechEngine (Piper TTS, non-blocking thread)
    assets/                 — TTS model (.onnx) + label map (.csv)
    models/                 — Trained Keras model (.keras)
"""

import os
import sys
import time
import argparse
import cv2

# ---------------------------------------------------------------------------
# Path setup — allow running from project root OR from src/ directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.inference_engine import SignLanguageInferenceEngine, load_label_map


# ---------------------------------------------------------------------------
# Default paths (resolved relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_transformer.keras")
DEFAULT_CSV_PATH   = os.path.join(PROJECT_ROOT, "assets", "SignList_ClassId_TR_EN.csv")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TSL Sign Language to Text & Speech — Real-Time Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (0 = built-in webcam, 1 = external)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to the trained Keras model file (.keras)"
    )
    parser.add_argument(
        "--labels", type=str, default=DEFAULT_CSV_PATH,
        help="Path to the SignList CSV file (ClassId, TR, EN columns)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.6,
        help="Minimum confidence threshold to accept a prediction (0.0–1.0)"
    )
    parser.add_argument(
        "--debounce", type=int, default=20,
        help="Frames to wait between consecutive sign predictions"
    )
    parser.add_argument(
        "--no-speech", action="store_true",
        help="Disable Text-to-Speech output (useful for silent environments)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FPS helper
# ---------------------------------------------------------------------------
class FPSCounter:
    """Rolling average FPS counter."""
    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Main camera loop
# ---------------------------------------------------------------------------
def run(args):
    # --- 1. Validate files exist before loading heavy models ---
    for label, path in [("Model", args.model), ("Labels CSV", args.labels)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} file not found: {path}")
            sys.exit(1)

    # --- 2. Load label map from CSV (single source of truth) ---
    print(f"[main] Loading label map from: {args.labels}")
    label_map = load_label_map(args.labels)
    if not label_map:
        print(f"[ERROR] Failed to load labels from: {args.labels}")
        sys.exit(1)
    print(f"[main] Loaded {len(label_map)} sign labels.")

    # --- 3. Initialize the inference engine ---
    print(f"[main] Initializing inference engine...")
    engine = SignLanguageInferenceEngine(
        model_path=args.model,
        label_map=label_map,
        buffer_size=80,
        confidence_threshold=args.confidence,
        debounce_frames=args.debounce,
    )

    # --- 4. Open camera ---
    print(f"[main] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}.")
        engine.stop()
        sys.exit(1)

    # Optional: request 30 FPS from camera driver
    cap.set(cv2.CAP_PROP_FPS, 30)

    auto_speak = not args.no_speech
    fps_counter = FPSCounter(window=30)

    print("\n[main] System ready. Press 'q' to quit.\n")
    print("=" * 55)
    print("  TSL Sign Language Recognition — Real-Time Mode")
    print(f"  Model     : {os.path.basename(args.model)}")
    print(f"  Signs     : {len(label_map)} classes")
    print(f"  Confidence: {args.confidence:.0%}")
    print(f"  Speech    : {'OFF' if args.no_speech else 'ON (Piper TTS)'}")
    print("=" * 55)

    # --- 5. Main loop ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from camera. Retrying...")
                time.sleep(0.05)
                continue

            # Process frame through the full AI pipeline
            out_frame, detected_label = engine.process_image(frame, auto_speak=auto_speak)

            # Log detections to terminal
            if detected_label:
                print(f"  >> Detected: {detected_label}")

            # Overlay FPS counter on frame
            fps = fps_counter.tick()
            cv2.putText(
                out_frame,
                f"FPS: {fps:.1f}",
                (out_frame.shape[1] - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA
            )

            cv2.imshow("TSL Sign Recognition — Press Q to quit", out_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[main] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user (Ctrl+C).")

    finally:
        # --- 6. Clean shutdown ---
        print("[main] Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()
        print("[main] Done. Goodbye!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run(args)
