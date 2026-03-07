import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf

class LandmarkExtractor:
    """
    Extracts MediaPipe Holistic landmarks from a single frame.
    Produces a (543, 3) shape array (or NaN for missing parts) 
    to match the long format Parquet structure from training.
    """
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
        Processes a BGR frame and returns a (543, 3) numpy array.
        Missing landmarks are represented as np.nan.
        """
        h, w = frame.shape[:2]
        if h != self.resize_height:
            frame = cv2.resize(frame, (self.resize_height, self.resize_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)

        data = np.full((543, 3), np.nan, dtype=np.float32)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                data[i] = [lm.x, lm.y, lm.z]
        
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                data[33 + i] = [lm.x, lm.y, lm.z]
                
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                data[501 + i] = [lm.x, lm.y, lm.z]
                
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                data[522 + i] = [lm.x, lm.y, lm.z]

        return data

class FeatureBuilder:
    """
    Applies the exact same preprocessing logic used during data generation.
    Handles landmark selection, NaN interpolation, spatial normalization,
    temporal resizing, and flattening.
    """
    def __init__(self, target_frames=80):
        self.target_frames = target_frames
        
        lips_idxs = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])
        pose_idxs = np.array([11, 12, 0])  # left_shoulder, right_shoulder, nose
        
        lh_global = np.arange(501, 522)
        rh_global = np.arange(522, 543)
        lips_global = 33 + lips_idxs
        pose_global = pose_idxs
        
        # Ensure correct ordering: Left Hand, Right Hand, Lips, Pose
        self.keep_indices = np.concatenate([lh_global, rh_global, lips_global, pose_global])

    def preprocess_sequence(self, frames_data):
        """
        Receives raw landmarks shaped (N, 543, 3).
        Outputs ready-to-predict tensor of shape (1, 80, 255).
        """
        x = np.array(frames_data)
        
        # 1. Filter landmarks -> (N, 85, 3)
        x_filtered = x[:, self.keep_indices, :]
        
        # 2. NaN Interpolation
        x_flat = x_filtered.reshape(x_filtered.shape[0], -1)
        df_tmp = pd.DataFrame(x_flat)
        df_tmp = df_tmp.interpolate(method='linear', limit_direction='both', axis=0)
        df_tmp = df_tmp.fillna(0)
        x_filtered = df_tmp.values.reshape(x_filtered.shape[0], -1, 3)
        
        # 3. Spatial Normalization
        # [-3]=left_shoulder, [-2]=right_shoulder, [-1]=nose (based on POSE_IDXS_0 ordering)
        nose = x_filtered[:, -1, :]
        left_sh = x_filtered[:, -3, :]
        right_sh = x_filtered[:, -2, :]
        
        # Relative to nose
        x_filtered = x_filtered - nose[:, None, :]
        
        # Scale by shoulder distance
        scale = np.linalg.norm(left_sh - right_sh, axis=1)
        scale[scale == 0] = 1.0
        x_filtered = x_filtered / scale[:, None, None]
        
        # 4. Temporal Resizing (to exactly 80 frames)
        if len(x_filtered) == self.target_frames:
            x_resized = x_filtered
        else:
            x_resized_tf = tf.image.resize(
                x_filtered[None, ...],
                [self.target_frames, x_filtered.shape[1]],
                method='bilinear'
            )
            x_resized = x_resized_tf.numpy()[0]
            
        # 5. Flattening the Spatial dimension: (80, 85, 3) -> (80, 255)
        x_final = x_resized.reshape(self.target_frames, -1)
        
        # 6. Add Batch dimension: (1, 80, 255)
        return np.expand_dims(x_final, axis=0)

class SequenceBuffer:
    """
    Manages a sliding window of frames for real-time inference.
    """
    def __init__(self, buffer_size=30):
        # buffer_size specifies how many frames to collect per sequence.
        # Since tf.image.resize can handle temporal scaling, this can be 
        # different from the target_frames (80), e.g., 30 for low-latency triggers.
        self.buffer_size = buffer_size
        self.buffer = []

    def add_frame(self, frame_landmarks):
        """
        Appends a newly extracted (543, 3) landmark frame to the buffer.
        Maintains the sliding window size.
        """
        self.buffer.append(frame_landmarks)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def is_ready(self):
        """
        Checks if the buffer has accumulated enough frames to form a sequence.
        """
        return len(self.buffer) == self.buffer_size

    def get_sequence(self):
        """
        Returns the current buffer of shape (buffer_size, 543, 3).
        """
        return np.array(self.buffer)
    
    def clear(self):
        """
        Empties the buffer.
        """
        self.buffer = []
