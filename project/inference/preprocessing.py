"""
Preprocessing module for model input preparation.

Applies the same spatial normalization as training (02_TID_Pre_Processing):
- Center by nose
- Scale by shoulder distance (left_sh - right_sh)
"""

import numpy as np

# Indices within the 85 points: last 3 = left_shoulder, right_shoulder, nose (from POSE_REF)
IDX_LEFT_SHOULDER = -3
IDX_RIGHT_SHOULDER = -2
IDX_NOSE = -1


def _spatial_normalize(x: np.ndarray) -> np.ndarray:
    """
    Apply spatial normalization per frame: center by nose, scale by shoulder distance.
    x shape: (T, 85, 3)
    """
    nose = x[:, IDX_NOSE, :]  # (T, 3)
    left_sh = x[:, IDX_LEFT_SHOULDER, :]
    right_sh = x[:, IDX_RIGHT_SHOULDER, :]

    x_centered = x - nose[:, None, :]
    scale = np.linalg.norm(left_sh - right_sh, axis=1)
    scale = np.where(scale == 0, 1.0, scale)
    x_scaled = x_centered / scale[:, None, None]
    return x_scaled.astype(np.float32)


# Hand landmark indices in flat 255: LH 0-62, RH 63-125
HAND_START, HAND_END = 0, 126
# Body reference (shoulders, nose): last 9 values in 255
BODY_REF_START = 246


def is_person_in_frame(sequence: np.ndarray, min_body_frames_ratio: float = 0.5) -> bool:
    """
    Reject when person is out of frame (no pose/body detected).
    Body ref points are zeros when MediaPipe doesn't detect pose.
    """
    body = sequence[:, BODY_REF_START:]  # (80, 9)
    frames_with_body = np.any(np.abs(body) > 1e-5, axis=1)
    return np.mean(frames_with_body) >= min_body_frames_ratio


def has_sign_activity(
    sequence: np.ndarray,
    min_hand_frames_ratio: float = 0.45,
    min_temporal_variance: float = 2e-5,
    min_frame_to_frame_motion: float = 0.03,
) -> bool:
    """
    Check if the sequence has sufficient hand activity (hands visible + clear motion).
    Rejects idle/static poses and out-of-frame. Call is_person_in_frame first.
    """
    if not is_person_in_frame(sequence):
        return False
    hand_data = sequence[:, HAND_START:HAND_END]  # (80, 126)
    hand_visible = np.any(np.abs(hand_data) > 1e-6, axis=1)
    if np.mean(hand_visible) < min_hand_frames_ratio:
        return False
    var_t = np.var(hand_data, axis=0)
    if np.mean(var_t) < min_temporal_variance:
        return False
    # Frame-to-frame motion: reject when standing still (hands visible but not moving)
    diff = np.abs(np.diff(hand_data, axis=0))
    total_motion = np.sum(diff)
    return total_motion >= min_frame_to_frame_motion


def prepare_model_input(sequence: np.ndarray) -> np.ndarray:
    """
    Prepare the sequence for model inference:
    1. Spatial normalization (match training pipeline)
    2. Add batch dimension

    Args:
        sequence: NumPy array of shape (80, 255).

    Returns:
        NumPy array of shape (1, 80, 255) ready for the Transformer model.
    """
    x = sequence.reshape(80, 85, 3).astype(np.float32)
    x = _spatial_normalize(x)
    x_flat = x.reshape(80, 255)
    return np.expand_dims(x_flat, axis=0).astype(np.float32)
