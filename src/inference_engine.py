import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque, Counter
from data_pipeline import LandmarkExtractor, FeatureBuilder, SequenceBuffer
from speech_engine import SpeechEngine

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """
    Tek bir Transformer Encoder blogu.
    Multi-Head Attention + Feed-Forward Network + Layer Normalization + Dropout
    """
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=768, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    """
    Ogrenilebilir konumsal gomme katmani.
    """
    def __init__(self, maxlen=80, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config

class SignLanguageInferenceEngine:
    """
    Real-time inference engine for Sign Language Recognition.
    Connects OpenCV, MediaPipe, the preprocessing pipeline, and the trained Transformer model.
    """
    def __init__(self, model_path, label_map, buffer_size=80, confidence_threshold=0.6, 
                 debounce_frames=20, voting_window=3, consensus_threshold=2, 
                 consecutive_threshold=2, motion_threshold=0.0001, presence_threshold=0.2):
        self.model_path = model_path
        self.label_map = label_map
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.debounce_frames = debounce_frames
        
        # Stability parameters
        self.voting_window = voting_window
        self.consensus_threshold = consensus_threshold
        self.consecutive_threshold = consecutive_threshold
        self.motion_threshold = motion_threshold
        self.presence_threshold = presence_threshold
        
        # Initialize pipeline components
        self.extractor = LandmarkExtractor(resize_height=512)
        self.builder = FeatureBuilder(ema_alpha=0.5)
        self.buffer = SequenceBuffer(buffer_size=self.buffer_size)
        
        # State variables
        self.model = None
        self.last_prediction = None
        self.frames_since_last_pred = 0
        self.voting_buffer = deque(maxlen=voting_window)
        
        # --- Voice Speech Engine & UI Callbacks ---
        self.currently_speaking = ""
        # The speech engine handles its own default path if None is passed
        self.speech_engine = SpeechEngine(cooldown=2.0)
        
        def on_speech_start(word):
            self.currently_speaking = word
            
        def on_speech_end(word):
            if self.currently_speaking == word:
                self.currently_speaking = ""
                
        self.speech_engine.add_start_callback(on_speech_start)
        self.speech_engine.add_end_callback(on_speech_end)
        
        # Load the model during initialization
        self.load_model()

    def load_model(self):
        """Loads the trained Keras model safely by providing custom layer definitions."""
        print(f"Loading model from {self.model_path}...")
        try:
            custom_objects = {
                "TransformerBlock": TransformerBlock,
                "PositionalEmbedding": PositionalEmbedding
            }
            self.model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def process_frame(self, frame):
        """Extracts landmarks from a frame and adds them to the buffer."""
        landmarks = self.extractor.extract_landmarks(frame)
        self.buffer.add_frame(landmarks)

    def has_sign_activity(self, sequence):
        """
        Activity Filtering Gate:
        1. Presence: Are hand landmarks actually detected (not just NaNs/Zeros)?
        2. Motion: Is there significant frame-to-frame movement (temporal variance) in the hand landmarks?
        
        NOTE: With the updated 85-landmark format:
          [0:21]  = Left Hand
          [21:42] = Right Hand
          [42:82] = Lips
          [82:85] = Pose
        """
        # Hand indices in the 85-landmark format: Left [0:21], Right [21:42]
        hands = sequence[:, 0:42, :]
        
        # 1. Presence Check: Check if any hand landmarks are detected (not all NaN)
        present_mask = ~np.isnan(hands).all(axis=(1, 2))
        presence_ratio = np.mean(present_mask)
        if presence_ratio < self.presence_threshold:
            return False
            
        # 2. Motion Check: Temporal variance of hand landmarks (X and Y coordinates)
        # Using nanvar to ignore NaNs in the variance calculation
        hand_vars = np.nanvar(hands[:, :, :2], axis=0)
        max_variance = np.nanmax(hand_vars) if not np.all(np.isnan(hand_vars)) else 0
        
        return max_variance > self.motion_threshold

    def predict(self, sequence_tensor):
        """Runs the model prediction on a preprocessed sequence tensor."""
        if self.model is None:
            return None, 0.0
            
        predictions = self.model.predict(sequence_tensor, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        label = self.label_map.get(predicted_class_idx, f"Class {predicted_class_idx}")
        
        return label, confidence

    def process_image(self, frame, auto_speak=True):
        """
        يعالج إطاراً واحداً (صورة) من الكاميرا ويرجع الإطار بعد رسم النصوص عليه، بالإضافة للكلمة المكتشفة.
        هذه الدالة مهيأة للعمل مع واجهة المستخدم (GUI).
        """
        if self.model is None:
            return frame, "Model not loaded"
            
        current_display_text = ""
        prediction_result = None
        
        # --- Square Crop & Resize to 512x512 ---
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        # Resize to 512x512
        frame_resized = cv2.resize(cropped_frame, (512, 512))
        # ----------------------------------------
            
        self.process_frame(frame_resized)
        
        if self.frames_since_last_pred < self.debounce_frames:
            self.frames_since_last_pred += 1
        
        if self.buffer.is_ready():
            if self.frames_since_last_pred >= self.debounce_frames:
                raw_sequence = self.buffer.get_sequence()
                
                # --- Stability Feature 1: Activity Filtering Gate ---
                if self.has_sign_activity(raw_sequence):
                    model_input_tensor = self.builder.preprocess_sequence(raw_sequence)
                    predicted_label, confidence = self.predict(model_input_tensor)
                    
                    if confidence >= self.confidence_threshold:
                        self.voting_buffer.append(predicted_label)
                    else:
                        self.voting_buffer.append(None)
                else:
                    self.voting_buffer.append(None)
                
                # --- Stability Feature 2: Prediction Stability System (Voting) ---
                if len(self.voting_buffer) >= self.consensus_threshold:
                    counts = Counter(self.voting_buffer)
                    if None in counts: del counts[None]
                    
                    if counts:
                        most_common_label, count = counts.most_common(1)[0]
                        
                        if count >= self.consensus_threshold:
                            last_few = list(self.voting_buffer)[-self.consecutive_threshold:]
                            is_consecutive = all(label == most_common_label for label in last_few)
                            
                            if is_consecutive:
                                if most_common_label != self.last_prediction:
                                    prediction_result = most_common_label
                                    self.last_prediction = most_common_label
                                    self.frames_since_last_pred = 0
                                    
                                    if auto_speak:
                                        self.speech_engine.speak(most_common_label)
                                        
                                    self.buffer.clear()
                                    self.voting_buffer.clear()
        
        # UI Overlay (Draw on the resized frame)
        if prediction_result:
            current_display_text = f"Detected Sign: {prediction_result}"
        elif self.last_prediction:
             current_display_text = f"Last Sign: {self.last_prediction}"
        else:
             current_display_text = "Waiting for sign..."
             
        cv2.putText(frame_resized, current_display_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if self.currently_speaking:
            cv2.putText(frame_resized, f"Nontaq: {self.currently_speaking}", (20, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)

        buf_len = len(self.buffer.buffer)
        buf_max = self.buffer.buffer_size
        cv2.putText(frame_resized, f"Buffer: {buf_len}/{buf_max}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        return frame_resized, prediction_result
        
    def stop(self):
        """Stops the speech engine thread cleanly."""
        if hasattr(self, 'speech_engine') and self.speech_engine:
            self.speech_engine.stop()


def load_label_map(csv_path):
    """Loads class labels from the CSV mapping file."""
    label_map = {}
    try:
        import csv
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_map[int(row['ClassId'])] = row['TR']
    except Exception as e:
        print(f"Error loading label map from {csv_path}: {e}")
    return label_map

if __name__ == "__main__":
    # Test block safely moved out and directory fixed
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file_path = os.path.join(base_dir, "models", "best_model_transformer.keras")
    
    # Shortened mapping for module test
    LABEL_MAP = { 0: "abla", 1: "acele", 2: "acikmak", 3: "afiyet_olsun", 4: "agabey" }
    
    # Initialize robust Inference Engine
    engine = SignLanguageInferenceEngine(
        model_path=model_file_path,
        label_map=LABEL_MAP,
        buffer_size=80,             # Sliding window of 80 frames to match training
        confidence_threshold=0.6,   # Ignore predictions below 60% confidence
        debounce_frames=20          # Wait 20 frames before predicting another distinct sign
    )
    
    # To run test without GUI, loop manually here:
    print("Test mode enabled. Starting camera...")
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out_frame, label = engine.process_image(frame)
        cv2.imshow("Test Mode", out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()