"""
Sign language prediction using the trained Transformer model.
"""

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from model.model_layers import PositionalEmbedding, SparseCategoricalCrossentropyWithLS, TransformerBlock
from utils.labels_loader import load_labels

CUSTOM_OBJECTS = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerBlock": TransformerBlock,
    "SparseCategoricalCrossentropyWithLS": SparseCategoricalCrossentropyWithLS,
}

CONFIDENCE_THRESHOLD = 0.6   # مسار 5: رفع قليلاً لتقليل الهلوسات
COOLDOWN_PREDICTIONS = 30   # Block repeats briefly
MIN_TOP_MARGIN = 0.08       # Lower margin - correct signs pass more often
VOTE_WINDOW = 6             # الحل 2: تصويت أصر
VOTE_MAJORITY = 4           # 4/6 same predictions to display
CONSECUTIVE_REQUIRED = 3    # الحل 1: نفس الفئة 3 مرات متتالية


class SignPredictor:
    """
    Loads the Transformer model and performs sign language prediction.
    """

    def __init__(self, model_path: str | Path = None, labels_path: str | Path = None):
        """
        Initialize predictor: load model and labels.

        Args:
            model_path: Path to best_model_transformer.keras. If None, uses model/ in project root.
            labels_path: Path to labels.csv. If None, uses default from labels_loader.
        """
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / "model" / "best_model_transformer.keras"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        self._id_to_turkish, _ = load_labels(labels_path)
        self._last_display_key: str | None = None
        self._last_display_at: int = -999
        self._vote_buffer: list[int] = []
        self._last_class_id: int | None = None
        self._consecutive_count: int = 0

    def predict(self, sequence: np.ndarray) -> tuple[int, float, str | None, float]:
        """
        Run inference on a prepared sequence.

        Args:
            sequence: Model input of shape (1, 80, 255).

        Returns:
            Tuple of (class_id, confidence, turkish_word, top_margin).
            top_margin = top1_prob - top2_prob (higher = more confident).
        """
        probabilities = self._model.predict(sequence, verbose=0)
        probs = probabilities[0]  # (num_classes,)

        top2_indices = np.argsort(probs)[-2:][::-1]
        predicted_class = int(top2_indices[0])
        confidence = float(probs[predicted_class])
        top2_prob = float(probs[top2_indices[1]]) if len(top2_indices) > 1 else 0.0
        top_margin = confidence - top2_prob
        turkish_word = self._id_to_turkish.get(predicted_class)

        return predicted_class, confidence, turkish_word, top_margin

    def get_display_text(self, class_id: int, turkish_word: str | None) -> str:
        """Return text to show: Turkish word if in labels, else class id."""
        if turkish_word is not None:
            return turkish_word
        return f"Class_{class_id}"

    def should_display(
        self,
        class_id: int,
        turkish_word: str | None,
        confidence: float,
        top_margin: float,
        prediction_index: int = 0,
    ) -> bool:
        """
        Check if prediction should be displayed (threshold + margin + debounce).
        """
        if confidence <= CONFIDENCE_THRESHOLD:
            return False
        if top_margin < MIN_TOP_MARGIN:
            return False  # Model uncertain (top-1 and top-2 too close)
        key = self.get_display_text(class_id, turkish_word)
        if key == self._last_display_key:
            if prediction_index - self._last_display_at < COOLDOWN_PREDICTIONS:
                return False
        return True

    def add_vote(self, class_id: int) -> None:
        """Add prediction to vote buffer; update consecutive count."""
        self._vote_buffer.append(class_id)
        if len(self._vote_buffer) > VOTE_WINDOW:
            self._vote_buffer.pop(0)
        if class_id == self._last_class_id:
            self._consecutive_count += 1
        else:
            self._last_class_id = class_id
            self._consecutive_count = 1

    def vote_passes(self, class_id: int) -> bool:
        """True if class_id is majority in recent predictions."""
        if len(self._vote_buffer) < VOTE_MAJORITY:
            return False
        count = sum(1 for c in self._vote_buffer if c == class_id)
        return count >= VOTE_MAJORITY

    def consecutive_passes(self, class_id: int) -> bool:
        """True if class_id appeared CONSECUTIVE_REQUIRED times in a row (filters flicker)."""
        return (
            self._last_class_id == class_id
            and self._consecutive_count >= CONSECUTIVE_REQUIRED
        )

    def record_displayed(
        self, class_id: int, turkish_word: str | None, prediction_index: int = 0
    ) -> None:
        """Record display for debounce."""
        self._last_display_key = self.get_display_text(class_id, turkish_word)
        self._last_display_at = prediction_index
