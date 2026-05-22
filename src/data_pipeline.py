import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque


class LandmarkExtractor:
    """
    Extracts MediaPipe Holistic landmarks from a single frame.
    Produces a (85, 3) shape array containing ONLY the 85 landmarks
    used during training, in the exact same order:
      [0:21]  -> Left Hand  (21 points)
      [21:42] -> Right Hand  (21 points)
      [42:82] -> Lips        (40 points)
      [82:85] -> Pose        (3 points: left_shoulder, right_shoulder, nose)
    Missing landmarks are represented as np.nan.
    """
    # Lip landmark indices within the MediaPipe Face Mesh (468 points)
    LIPS_IDXS = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
    # Pose landmark indices within MediaPipe Pose (33 points)
    # 11 = left_shoulder, 12 = right_shoulder, 0 = nose
    POSE_IDXS = np.array([11, 12, 0])

    def __init__(self, resize_height=512):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.resize_height = resize_height

    def extract_landmarks(self, frame):
        """
        Processes a BGR frame and returns a (85, 3) numpy array.
        Missing landmarks are represented as np.nan.
        """
        h, w = frame.shape[:2]
        if h != self.resize_height:
            frame = cv2.resize(frame, (self.resize_height, self.resize_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)

        # Output: exactly 85 landmarks in strict order
        data = np.full((85, 3), np.nan, dtype=np.float32)

        # --- Left Hand: indices [0:21] ---
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                data[i] = [lm.x, lm.y, lm.z]

        # --- Right Hand: indices [21:42] ---
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                data[21 + i] = [lm.x, lm.y, lm.z]

        # --- Lips: indices [42:82] ---
        if results.face_landmarks:
            for j, lip_idx in enumerate(self.LIPS_IDXS):
                lm = results.face_landmarks.landmark[lip_idx]
                data[42 + j] = [lm.x, lm.y, lm.z]

        # --- Pose (shoulder_L, shoulder_R, nose): indices [82:85] ---
        if results.pose_landmarks:
            for j, pose_idx in enumerate(self.POSE_IDXS):
                lm = results.pose_landmarks.landmark[pose_idx]
                data[82 + j] = [lm.x, lm.y, lm.z]

        return data


class FeatureBuilder:
    """
    Applies the exact same preprocessing logic used during training.
    Pipeline order:
      1. EMA Smoothing        (reduce webcam jitter)
      2. NaN Interpolation     (fill missing hand frames via linear interpolation)
      3. Spatial Normalization  (nose-centered, shoulder-distance scaled)
      4. Flattening             (85, 3) -> (255,) per frame
    
    NOTE: No temporal resizing (tf.image.resize) is used.  The SequenceBuffer
    is responsible for providing exactly 80 frames.
    """
    def __init__(self, ema_alpha=0.5):
        """
        Args:
            ema_alpha: Smoothing factor for Exponential Moving Average.
                       Higher = more weight on current frame (less smoothing).
                       Range: (0, 1]. 1.0 = no smoothing.
        """
        self.ema_alpha = ema_alpha

    def _apply_ema_smoothing(self, x):
        """
        Applies Exponential Moving Average over the time axis (axis=0).
        Handles NaN values by skipping them during the smoothing process.
        Input/Output shape: (N, 85, 3)
        """
        smoothed = np.copy(x)
        for t in range(1, len(smoothed)):
            # Create mask for valid (non-NaN) values in both current and previous frames
            valid_current = ~np.isnan(smoothed[t])
            valid_prev = ~np.isnan(smoothed[t - 1])
            both_valid = valid_current & valid_prev

            # Apply EMA only where both frames have valid data
            smoothed[t] = np.where(
                both_valid,
                self.ema_alpha * smoothed[t] + (1 - self.ema_alpha) * smoothed[t - 1],
                smoothed[t]  # Keep original if either is NaN
            )
        return smoothed

    def _interpolate_nans(self, x):
        """
        Applies linear interpolation along the time axis to fill NaN gaps.
        This handles frames where hands are temporarily not detected.
        Input/Output shape: (N, 85, 3)
        """
        n_frames = x.shape[0]
        # Flatten spatial dims for pandas interpolation: (N, 255)
        x_flat = x.reshape(n_frames, -1)
        df = pd.DataFrame(x_flat)
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        df = df.fillna(0)  # Fill any remaining NaNs (e.g., all-NaN columns) with 0
        return df.values.reshape(n_frames, 85, 3)

    def _spatial_normalize(self, x):
        """
        Normalizes spatial coordinates:
          1. Translates all points so that the nose is at the origin.
          2. Scales by the Euclidean distance between the two shoulders.
        
        Landmark layout (per frame):
          [82] = left_shoulder, [83] = right_shoulder, [84] = nose
        
        Input/Output shape: (N, 85, 3)
        """
        # Extract reference points: shape (N, 3) each
        nose = x[:, 84, :]           # index 84 = nose
        left_sh = x[:, 82, :]        # index 82 = left_shoulder
        right_sh = x[:, 83, :]       # index 83 = right_shoulder

        # 1. Center on nose
        x = x - nose[:, None, :]

        # 2. Scale by shoulder distance
        shoulder_dist = np.linalg.norm(left_sh - right_sh, axis=1)  # (N,)
        shoulder_dist[shoulder_dist == 0] = 1.0  # Prevent division by zero
        x = x / shoulder_dist[:, None, None]

        return x

    def preprocess_sequence(self, frames_data):
        """
        Full preprocessing pipeline matching the training procedure.
        
        Args:
            frames_data: numpy array of shape (80, 85, 3) — raw landmark data
                         from the SequenceBuffer.
        
        Returns:
            numpy array of shape (1, 80, 255) — ready for model prediction.
        """
        x = np.array(frames_data, dtype=np.float32)

        # 1. EMA Smoothing (reduce webcam jitter)
        x = self._apply_ema_smoothing(x)

        # 2. NaN Interpolation (fill missing hand detections)
        x = self._interpolate_nans(x)

        # 3. Spatial Normalization (nose-centered, shoulder-scaled)
        x = self._spatial_normalize(x)

        # 4. Flatten spatial dimension: (80, 85, 3) -> (80, 255)
        x_final = x.reshape(x.shape[0], -1)

        # 5. Add batch dimension: (1, 80, 255)
        return np.expand_dims(x_final, axis=0)


class SequenceBuffer:
    """
    Manages a sliding window of 80 frames for real-time inference.
    
    Key features:
      - Sliding window with configurable stride for responsive predictions.
      - Allows early prediction after `min_frames` (e.g. 15) instead of
        waiting for a full 80-frame window.
      - Pads short sequences by replicating the first (oldest) frame at the
        beginning. This tells the Transformer the user was "still" before
        the gesture started, avoiding attention jumps from zero-padding.
      - Supports full buffer clear after a confirmed prediction to prevent
        old gesture frames from bleeding into new gesture detection.
    """
    def __init__(self, buffer_size=80, stride=5, min_frames_to_predict=15):
        """
        Args:
            buffer_size:          Target number of frames (must match model = 80).
            stride:               New frames between each prediction trigger.
            min_frames_to_predict: Minimum frames needed before allowing prediction.
                                   Enables fast recovery after buffer clear.
        """
        self.buffer_size = buffer_size
        self.stride = stride
        self.min_frames = min_frames_to_predict
        # ✅ Task 5: deque with maxlen auto-evicts oldest frame in O(1).
        # list.pop(0) was O(n) — it shifted every element in memory each frame.
        self.buffer = deque(maxlen=buffer_size)
        self._frames_since_last_ready = 0

    def add_frame(self, frame_landmarks):
        """
        Appends a newly extracted (85, 3) landmark frame to the buffer.
        Maintains the sliding window automatically — deque(maxlen) evicts
        the oldest frame when full, no manual pop(0) needed.
        """
        self.buffer.append(frame_landmarks)   # O(1) append + O(1) auto-eviction
        self._frames_since_last_ready += 1

    def is_ready(self):
        """
        Returns True when:
          1. The buffer has at least `min_frames` frames (e.g. 15), AND
          2. At least `stride` new frames have been added since the last trigger.
        
        This allows fast prediction after a buffer clear without waiting
        for the full 80-frame window to refill.
        """
        if len(self.buffer) < self.min_frames:
            return False
        if self._frames_since_last_ready >= self.stride:
            return True
        return False

    def get_sequence(self):
        """
        Returns a numpy array of shape (buffer_size, 85, 3).

        If the buffer has fewer than buffer_size frames, the sequence is
        padded at the BEGINNING by repeating the oldest frame (first-frame
        replication). This simulates a "still" pose before the gesture,
        avoiding the attention-distribution instability caused by zero-padding.

        Resets the stride counter after each call.

        Note: list(self.buffer) is required before np.array() because numpy
        cannot directly build a 3-D array from a deque of 2-D numpy arrays
        without the intermediate list conversion.
        """
        self._frames_since_last_ready = 0
        seq = np.array(list(self.buffer))   # deque → list → numpy (buffer_size, 85, 3)

        # --- First-Frame Replication Padding ---
        if len(seq) < self.buffer_size:
            pad_length = self.buffer_size - len(seq)
            # Repeat the oldest frame (index 0) to fill the temporal gap
            pad_array = np.repeat(seq[0:1], pad_length, axis=0)
            # Prepend padding so real frames are at the END (most recent)
            seq = np.concatenate((pad_array, seq), axis=0)

        return seq

    def clear(self):
        """Empties the buffer and resets the stride counter."""
        self.buffer = deque(maxlen=self.buffer_size)  # fresh deque preserves maxlen
        self._frames_since_last_ready = 0
