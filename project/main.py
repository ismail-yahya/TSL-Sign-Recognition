"""
TID Sign Language - Phase 3: Camera pipeline + model inference + prediction display.

Pipeline:
  Camera → MediaPipe → Landmark Extraction → Frame Buffer → Model Input → Predictor → Display
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from inference.camera import Camera
from inference.mediapipe_detector import MediaPipeDetector
from inference.landmark_extractor import extract_landmarks
from inference.frame_buffer import FrameBuffer
from inference.preprocessing import prepare_model_input, has_sign_activity
from inference.predictor import SignPredictor

# Match training pipeline: 01_TID_Landmark_Extraction used 512x512 for MediaPipe
MP_FRAME_SIZE = 512
# Predict every 5 frames (~0.17 sec at 30fps) - faster response
PREDICT_EVERY_N_FRAMES = 5


def main() -> None:
    camera = Camera()
    detector = MediaPipeDetector()
    buffer = FrameBuffer()

    try:
        predictor = SignPredictor()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not camera.start():
        print("Error: Could not open webcam.")
        return

    try:
        cv2.namedWindow("TID - Landmark Detection")
        cv2.destroyWindow("TID - Landmark Detection")
        use_display = True
    except cv2.error:
        use_display = False
        print("Note: Display not available (headless env).")

    print("Camera started. Press 'q' to exit.")
    print("Pipeline: Camera -> MediaPipe -> Landmark Extraction -> Frame Buffer -> Predictor -> Display")
    print("-" * 50)

    buffer_ready_shown = False
    prediction_index = 0
    frames_since_predict = 0

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("Failed to read frame.")
                break

            # Resize to 512x512 (same as training) for consistent MediaPipe coordinates
            frame_mp = cv2.resize(frame, (MP_FRAME_SIZE, MP_FRAME_SIZE))
            # Camera -> MediaPipe -> Landmark Extraction
            frame_with_drawings, results = detector.process(frame_mp)
            landmarks_vector = extract_landmarks(results)

            # Add to frame buffer (sliding window)
            buffer.add_frame(landmarks_vector)

            # Debug output during filling
            if not buffer.is_full():
                print(f"\rFrame added to buffer - Current: {buffer.size} / 80", end="", flush=True)

            # When buffer full: predict continuously (sliding window)
            if buffer.is_full():
                sequence = buffer.get_sequence()

                if not buffer_ready_shown:
                    print()
                    print("Buffer ready - predicting when hands visible. Press 'q' to exit.")
                    print("-" * 50)
                    buffer_ready_shown = True

                # Only predict when hands are visible (reduces wrong predictions when idle)
                if not has_sign_activity(sequence):
                    if use_display:
                        display_frame = cv2.resize(frame_with_drawings, (640, 640))
                        cv2.imshow("TID - Landmark Detection", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                # Predict every N frames to reduce noise
                frames_since_predict += 1
                if frames_since_predict >= PREDICT_EVERY_N_FRAMES:
                    frames_since_predict = 0
                    model_input = prepare_model_input(sequence)
                    class_id, confidence, turkish_word, top_margin = predictor.predict(model_input)
                    prediction_index += 1
                    predictor.add_vote(class_id)
                    if (
                        predictor.vote_passes(class_id)
                        and predictor.consecutive_passes(class_id)
                        and predictor.should_display(
                            class_id, turkish_word, confidence, top_margin, prediction_index
                        )
                    ):
                        display_text = predictor.get_display_text(class_id, turkish_word)
                        predictor.record_displayed(class_id, turkish_word, prediction_index)
                        print(f"[{display_text}] confidence={confidence:.2f} | class={class_id}")

            if use_display:
                # Scale up for display if 512x512 is small
                display_frame = cv2.resize(frame_with_drawings, (640, 640))
                cv2.imshow("TID - Landmark Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        detector.close()
        if use_display:
            cv2.destroyAllWindows()

    print("Exited.")


if __name__ == "__main__":
    main()
