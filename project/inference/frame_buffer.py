"""
Sliding window frame buffer for temporal sequence collection.

Maintains up to 80 frames of (255,) feature vectors for model input.
"""

import numpy as np

BUFFER_SIZE = 80
FEATURE_DIM = 255


class FrameBuffer:
    """
    Sliding window buffer for frame feature vectors.
    Maintains a fixed-size queue of the most recent frames.
    """

    def __init__(self, max_size: int = BUFFER_SIZE):
        """
        Initialize the frame buffer.

        Args:
            max_size: Maximum number of frames to store (default 80).
        """
        self._max_size = max_size
        self._buffer: list[np.ndarray] = []

    def add_frame(self, features: np.ndarray) -> None:
        """
        Add a frame's feature vector to the buffer.
        Applies temporal carry-forward: zeros (missing landmarks) are filled from
        the previous frame when available, matching training's NaN interpolation.

        Args:
            features: NumPy array of shape (255,).
        """
        feat = np.asarray(features, dtype=np.float32)
        if len(self._buffer) > 0:
            prev = self._buffer[-1]
            mask_zero = np.abs(feat) < 1e-8
            feat = np.where(mask_zero, prev, feat)
        if len(self._buffer) == self._max_size:
            self._buffer.pop(0)
        self._buffer.append(feat)

    def is_full(self) -> bool:
        """Return True when buffer has 80 frames."""
        return len(self._buffer) == self._max_size

    def get_sequence(self) -> np.ndarray:
        """
        Return the current buffer contents as a NumPy array.

        Returns:
            Array of shape (80, 255). Raises if buffer is not full.
        """
        if not self.is_full():
            raise ValueError(
                f"Buffer not full (has {len(self._buffer)} frames). "
                f"Cannot get sequence until {self._max_size} frames."
            )
        return np.stack(self._buffer, axis=0).astype(np.float32)

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Current number of frames in the buffer."""
        return len(self._buffer)
