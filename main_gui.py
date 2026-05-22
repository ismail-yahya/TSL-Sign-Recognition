"""
main_gui.py — TSL Sign Language Recognition System (GUI Mode)
=============================================================
Launches the PyQt5 graphical interface.

Usage:
    python main_gui.py                      # Default camera, all options enabled
    python main_gui.py --camera 1           # External camera
    python main_gui.py --confidence 0.75    # Stricter confidence
    python main_gui.py --no-speech          # Disable TTS
    python main_gui.py --help               # All options

For the terminal/headless mode use main.py instead.
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.inference_engine import SignLanguageInferenceEngine, load_label_map
from ui.app_window import launch_app


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_transformer.keras")
DEFAULT_CSV_PATH   = os.path.join(PROJECT_ROOT, "assets", "SignList_ClassId_TR_EN.csv")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TSL Sign Language Recognition System — GUI Mode",
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
        help="Path to the SignList CSV label file"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.6,
        help="Minimum confidence threshold (0.0–1.0)"
    )
    parser.add_argument(
        "--debounce", type=int, default=20,
        help="Frames to wait between consecutive predictions"
    )
    parser.add_argument(
        "--no-speech", action="store_true",
        help="Disable Text-to-Speech output"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Validate required files
    for label, path in [("Model", args.model), ("Labels CSV", args.labels)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} file not found: {path}")
            sys.exit(1)

    # Load labels
    print(f"[main_gui] Loading label map from: {args.labels}")
    label_map = load_label_map(args.labels)
    if not label_map:
        print(f"[ERROR] Failed to load labels from: {args.labels}")
        sys.exit(1)
    print(f"[main_gui] Loaded {len(label_map)} sign labels.")

    # Build engine
    print("[main_gui] Initialising inference engine…")
    engine = SignLanguageInferenceEngine(
        model_path=args.model,
        label_map=label_map,
        buffer_size=80,
        confidence_threshold=args.confidence,
        debounce_frames=args.debounce,
    )
    print("[main_gui] Engine ready — launching GUI…")

    # Launch Qt app
    exit_code = launch_app(
        engine=engine,
        camera_idx=args.camera,
        auto_speak=not args.no_speech,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
