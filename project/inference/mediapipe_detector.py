"""
MediaPipe Holistic detector for pose, hands, and face landmarks.
"""

import cv2
import mediapipe as mp


class MediaPipeDetector:
    """
    Detects pose, left hand, right hand, and face landmarks using MediaPipe Holistic.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        refine_face_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Holistic with configurable parameters.

        Args:
            static_image_mode: If True, treats input as static images.
            model_complexity: 0, 1, or 2. Higher = more accurate but slower.
            smooth_landmarks: Whether to smooth landmarks across frames.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Drawing styles
        self._pose_style = self._mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self._hand_style = self._mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
        self._face_style = self._mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)

    def process(self, frame):
        """
        Process a BGR frame and return both drawn frame and raw landmarks.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Tuple of (frame_with_drawings, results). results is the MediaPipe
            Holistic results object containing pose_landmarks, face_landmarks,
            left_hand_landmarks, right_hand_landmarks.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        frame_with_drawings = frame.copy()
        self._draw_landmarks(frame_with_drawings, results)

        return frame_with_drawings, results

    def _draw_landmarks(self, frame, results) -> None:
        """Draw all detected landmarks on the frame."""
        if results.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
                self._pose_style,
                self._pose_style,
            )
        if results.left_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                self._hand_style,
                self._hand_style,
            )
        if results.right_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                self._hand_style,
                self._hand_style,
            )
        if results.face_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self._mp_holistic.FACEMESH_CONTOURS,
                self._face_style,
                self._face_style,
            )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._holistic.close()
