"""
Camera module for webcam access using OpenCV.
"""

import cv2


class Camera:
    """Webcam capture using OpenCV."""

    def __init__(self, device_id: int = 1, width: int = 640, height: int = 480):
        """
        Initialize camera parameters.

        Args:
            device_id: Camera device index (default 0 for default webcam).
            width: Frame width.
            height: Frame height.
        """
        self._device_id = device_id
        self._width = width
        self._height = height
        self._cap = None

    def start(self) -> bool:
        """
        Initialize and start the webcam.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return True

    def read_frame(self) -> tuple[bool, object]:
        """
        Read the next frame from the camera.

        Returns:
            Tuple of (success, frame). success is False if no frame was read.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
