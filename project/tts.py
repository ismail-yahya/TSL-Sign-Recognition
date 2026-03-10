"""
Simple Turkish TTS integration using Piper.

Loads the Turkish Piper model once and provides a helper function
to speak a given Turkish word or sentence on Windows.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

try:
    import winsound  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None  # type: ignore[assignment]

from piper import PiperVoice


_VOICE: PiperVoice | None = None


def _get_model_path() -> Path:
    """
    Resolve the Piper ONNX model path relative to the repo root.
    Expects: Piper/tr_TR-dfki-medium.onnx
    """
    # this file is in: <repo_root>/project/tts.py
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "Piper" / "tr_TR-dfki-medium.onnx"
    return model_path


def _get_voice() -> PiperVoice:
    """Lazily load and cache the Piper voice model."""
    global _VOICE
    if _VOICE is None:
        model_path = _get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found at: {model_path}")
        _VOICE = PiperVoice.load(str(model_path))
    return _VOICE


def speak_turkish(text: str) -> None:
    """
    Synthesize and play Turkish speech for the given text.

    - Uses Piper to synthesize into a temporary WAV file.
    - On Windows, plays the WAV using winsound.
    - If audio playback is not available, silently returns after synthesis.
    """
    if not text:
        return

    voice = _get_voice()

    # Save into a temporary WAV file next to this module
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "last_tts.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    # Play sound on Windows if possible
    if winsound is not None and os.name == "nt":
        try:
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        except Exception:
            # If playback fails for any reason, do not crash the main loop
            return

