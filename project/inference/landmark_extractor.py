"""
Extract 85 landmarks as a (255,) feature vector from MediaPipe Holistic results.
Matches the training pipeline indices exactly.
"""

import numpy as np

# Same indices as training pipeline (02_TID_Pre_Processing)
LIPS_IDXS0 = np.array(
    [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
)
POSE_IDXS0 = np.array([11, 12, 0])  # Left shoulder, Right shoulder, Nose

LH_GLOBAL = np.arange(501, 522)       # 21 left hand points
RH_GLOBAL = np.arange(522, 543)       # 21 right hand points
LIPS_GLOBAL = 33 + LIPS_IDXS0         # 40 lips points (face offset = 33)
POSE_REF_GLOBAL = POSE_IDXS0          # 3 body reference points

KEEP_INDICES = np.concatenate([LH_GLOBAL, RH_GLOBAL, LIPS_GLOBAL, POSE_REF_GLOBAL])
NUM_SELECTED_POINTS = 85
FEATURE_DIM = 85 * 3  # 255


def _landmarks_to_array(landmarks) -> np.ndarray:
    """
    Convert MediaPipe NormalizedLandmarkList to (N, 3) array.
    Returns zeros if landmarks is None.
    """
    if landmarks is None:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _build_full_543(results) -> np.ndarray:
    """
    Build the full 543-point array from MediaPipe Holistic results.
    Layout: pose (33) + face (468) + left_hand (21) + right_hand (21) = 543.
    """
    full = np.zeros((543, 3), dtype=np.float32)

    if results.pose_landmarks:
        arr = _landmarks_to_array(results.pose_landmarks)
        full[0:33] = arr

    if results.face_landmarks:
        arr = _landmarks_to_array(results.face_landmarks)
        # Training used 468; refine_face_landmarks=True returns 478 — take first 468
        full[33:501] = arr[:468]

    if results.left_hand_landmarks:
        arr = _landmarks_to_array(results.left_hand_landmarks)
        full[501:522] = arr

    if results.right_hand_landmarks:
        arr = _landmarks_to_array(results.right_hand_landmarks)
        full[522:543] = arr

    return full


def extract_landmarks(results) -> np.ndarray:
    """
    Extract the 85 selected landmarks as a flat (255,) feature vector.

    Uses the same landmark selection as the training pipeline:
    - 21 left hand, 21 right hand, 40 lips, 3 body reference points.

    Missing landmarks are filled with zeros.

    Args:
        results: MediaPipe Holistic results object.

    Returns:
        NumPy array of shape (255,) with dtype float32.
    """
    full = _build_full_543(results)
    selected = full[KEEP_INDICES]  # (85, 3)
    return selected.flatten().astype(np.float32)  # (255,)
