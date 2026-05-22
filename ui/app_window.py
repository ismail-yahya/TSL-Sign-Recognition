"""
ui/app_window.py — TSL Sign Recognition GUI
============================================
PyQt5-based graphical interface for the Turkish Sign Language Recognition System.

Architecture:
  CameraWorker (QThread) — Runs the full AI pipeline off the main thread.
    Signals: frame_ready, sign_detected, stats_updated
  MainWindow (QMainWindow) — Renders the UI; updated only via Qt Signals.

Usage (standalone):
    python ui/app_window.py

Usage (from main_gui.py):
    from ui.app_window import launch_app
    launch_app(engine, args)
"""

import os
import sys
import time

import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QSize
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen,
    QLinearGradient, QBrush, QFontDatabase, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QPushButton, QStatusBar, QSizePolicy,
    QGraphicsOpacityEffect
)

# ---------------------------------------------------------------------------
# Path bootstrap — allow running directly from ui/ OR from project root
# ---------------------------------------------------------------------------
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UI_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.inference_engine import SignLanguageInferenceEngine, load_label_map


# ===========================================================================
# Rolling FPS Counter (same as main.py, reproduced here for independence)
# ===========================================================================
class _FPSCounter:
    """Rolling-window FPS counter."""
    def __init__(self, window: int = 30):
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ===========================================================================
# CameraWorker — Runs AI pipeline on a dedicated QThread
# ===========================================================================
class CameraWorker(QThread):
    """
    Background thread that:
      1. Reads frames from the webcam.
      2. Passes them through the SignLanguageInferenceEngine.
      3. Emits Qt signals with results — never touches UI widgets directly.

    Signals:
      frame_ready(QImage)         — annotated camera frame (for display)
      sign_detected(str)          — new sign label (when engine confirms one)
      stats_updated(float, float) — (fps, latency_ms) every frame
      speaking_changed(str)       — current TTS word, "" when silent
      buffer_updated(int, int)    — (filled, total) buffer fill status
    """

    frame_ready      = pyqtSignal(QImage)
    sign_detected    = pyqtSignal(str)
    stats_updated    = pyqtSignal(float, float)   # fps, latency_ms
    speaking_changed = pyqtSignal(str)
    buffer_updated   = pyqtSignal(int, int)

    def __init__(self, engine: SignLanguageInferenceEngine,
                 camera_idx: int = 0,
                 auto_speak: bool = True):
        super().__init__()
        self.engine       = engine
        self.camera_idx   = camera_idx
        self.auto_speak   = auto_speak
        self._running     = False
        self._fps         = _FPSCounter(window=30)
        self._prev_speaking = ""

    # ------------------------------------------------------------------
    def run(self):
        """Main capture-inference loop. Runs on the worker thread."""
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_FPS, 30)
        self._running = True

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            # AI pipeline
            out_frame, detected_label = self.engine.process_image(
                frame, auto_speak=self.auto_speak
            )

            # FPS
            fps = self._fps.tick()
            latency = self.engine.last_latency_ms

            # Emit stats
            self.stats_updated.emit(fps, latency)

            # Emit sign detection
            if detected_label:
                self.sign_detected.emit(str(detected_label))

            # Emit speaking state
            speaking = self.engine.currently_speaking or ""
            if speaking != self._prev_speaking:
                self.speaking_changed.emit(speaking)
                self._prev_speaking = speaking

            # Emit buffer status
            buf_len = len(self.engine.buffer.buffer)
            buf_max = self.engine.buffer.buffer_size
            self.buffer_updated.emit(buf_len, buf_max)

            # Convert BGR frame → QImage for display
            rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data.tobytes(), w, h, ch * w,
                          QImage.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

        cap.release()

    def stop(self):
        """Signal the loop to exit and wait for the thread to finish."""
        self._running = False
        self.wait(3000)


# ===========================================================================
# Helper: rounded card widget
# ===========================================================================
class _Card(QFrame):
    """Styled dark card panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")


# ===========================================================================
# SpeakingIndicator — animated pulsing dot
# ===========================================================================
class _SpeakingIndicator(QLabel):
    """A small animated label that pulses when TTS is active."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._active = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._frame_idx = 0
        self._colors = ["#ff6b6b", "#ff9f43", "#ffd32a", "#a29bfe"]
        self._set_color("#444")

    def _set_color(self, color: str):
        self.setStyleSheet(
            f"background: {color}; border-radius: 9px;"
        )

    def set_active(self, active: bool):
        if active == self._active:
            return
        self._active = active
        if active:
            self._timer.start(200)
        else:
            self._timer.stop()
            self._set_color("#444")

    def _pulse(self):
        self._frame_idx = (self._frame_idx + 1) % len(self._colors)
        self._set_color(self._colors[self._frame_idx])


# ===========================================================================
# BufferBar — thin progress-bar showing buffer fill
# ===========================================================================
class _BufferBar(QWidget):
    """Thin horizontal bar visualising buffer fill level."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self._ratio = 0.0

    def update_ratio(self, filled: int, total: int):
        self._ratio = filled / max(total, 1)
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.setBrush(QColor("#2a2a3a"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 3, 3)

        # Fill gradient
        grad = QLinearGradient(0, 0, w * self._ratio, 0)
        grad.setColorAt(0, QColor("#7c6be8"))
        grad.setColorAt(1, QColor("#00cec9"))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(0, 0, int(w * self._ratio), h, 3, 3)

        painter.end()


# ===========================================================================
# MainWindow
# ===========================================================================
class MainWindow(QMainWindow):
    """
    Main application window.

    Layout (horizontal split):
      LEFT  (60%) — Camera feed + status bar below
      RIGHT (40%) — Detected word (large) + word history + speaking indicator
    """

    MAX_HISTORY = 10   # maximum words kept in history list

    def __init__(self, engine: SignLanguageInferenceEngine,
                 camera_idx: int = 0,
                 auto_speak: bool = True):
        super().__init__()
        self.engine      = engine
        self.camera_idx  = camera_idx
        self.auto_speak  = auto_speak
        self._history    = []   # list[str]
        self._current_word = ""

        self._build_ui()
        self._apply_stylesheet()
        self._start_worker()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("TSL — Turkish Sign Language Recognition")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 760)

        # Central widget + root layout
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # ----- LEFT PANEL -----
        left = _Card()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Header bar (title + stats)
        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title_lbl = QLabel("🤟  TSL Sign Recognition")
        title_lbl.setObjectName("headerTitle")
        header_layout.addWidget(title_lbl)

        header_layout.addStretch()

        self.fps_lbl = QLabel("FPS: —")
        self.fps_lbl.setObjectName("statLabel")
        header_layout.addWidget(self.fps_lbl)

        sep = QLabel("|")
        sep.setObjectName("statSep")
        header_layout.addWidget(sep)

        self.lat_lbl = QLabel("Latency: —")
        self.lat_lbl.setObjectName("statLabel")
        header_layout.addWidget(self.lat_lbl)

        left_layout.addWidget(header)

        # Camera feed label
        self.camera_label = QLabel()
        self.camera_label.setObjectName("cameraFeed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.camera_label.setMinimumSize(480, 360)

        # Placeholder while camera loads
        self.camera_label.setText("⏳  Initialising camera…")
        left_layout.addWidget(self.camera_label, stretch=1)

        # Buffer bar
        self.buffer_bar = _BufferBar()
        left_layout.addWidget(self.buffer_bar)

        # Buffer label
        self.buffer_lbl = QLabel("Buffer: 0 / 80")
        self.buffer_lbl.setObjectName("bufferLabel")
        self.buffer_lbl.setAlignment(Qt.AlignRight)
        left_layout.addWidget(self.buffer_lbl)

        root_layout.addWidget(left, stretch=6)

        # ----- RIGHT PANEL -----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # --- Detected Word Card ---
        word_card = _Card()
        word_card.setObjectName("wordCard")
        word_card_layout = QVBoxLayout(word_card)
        word_card_layout.setContentsMargins(16, 16, 16, 16)
        word_card_layout.setSpacing(6)

        detected_hdr = QLabel("DETECTED SIGN")
        detected_hdr.setObjectName("sectionHeader")
        word_card_layout.addWidget(detected_hdr)

        self.word_label = QLabel("—")
        self.word_label.setObjectName("wordLabel")
        self.word_label.setAlignment(Qt.AlignCenter)
        self.word_label.setWordWrap(True)
        word_card_layout.addWidget(self.word_label, stretch=1)

        # Confidence bar row
        conf_row = QHBoxLayout()
        conf_lbl_static = QLabel("Confidence")
        conf_lbl_static.setObjectName("microLabel")
        conf_row.addWidget(conf_lbl_static)
        conf_row.addStretch()
        self.conf_lbl = QLabel("—")
        self.conf_lbl.setObjectName("microLabel")
        conf_row.addWidget(self.conf_lbl)
        word_card_layout.addLayout(conf_row)

        right_layout.addWidget(word_card, stretch=2)

        # --- Speaking Indicator Card ---
        speak_card = _Card()
        speak_card.setObjectName("speakCard")
        speak_layout = QHBoxLayout(speak_card)
        speak_layout.setContentsMargins(14, 10, 14, 10)

        self.speak_dot = _SpeakingIndicator()
        speak_layout.addWidget(self.speak_dot)

        self.speak_lbl = QLabel("Not speaking")
        self.speak_lbl.setObjectName("speakLabel")
        speak_layout.addWidget(self.speak_lbl, stretch=1)

        right_layout.addWidget(speak_card)

        # --- Word History Card ---
        hist_card = _Card()
        hist_layout = QVBoxLayout(hist_card)
        hist_layout.setContentsMargins(12, 12, 12, 12)
        hist_layout.setSpacing(6)

        hist_hdr_row = QHBoxLayout()
        hist_hdr = QLabel("WORD HISTORY")
        hist_hdr.setObjectName("sectionHeader")
        hist_hdr_row.addWidget(hist_hdr)
        hist_hdr_row.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("clearBtn")
        clear_btn.setFixedSize(60, 24)
        clear_btn.clicked.connect(self._clear_history)
        hist_hdr_row.addWidget(clear_btn)
        hist_layout.addLayout(hist_hdr_row)

        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.setFocusPolicy(Qt.NoFocus)
        hist_layout.addWidget(self.history_list, stretch=1)

        right_layout.addWidget(hist_card, stretch=3)

        # --- Bottom controls row ---
        ctrl_row = QHBoxLayout()

        self.speech_btn = QPushButton("🔇  Speech OFF" if not self.auto_speak
                                      else "🔊  Speech ON")
        self.speech_btn.setObjectName("ctrlBtn")
        self.speech_btn.setCheckable(True)
        self.speech_btn.setChecked(self.auto_speak)
        self.speech_btn.clicked.connect(self._toggle_speech)
        ctrl_row.addWidget(self.speech_btn)

        quit_btn = QPushButton("✖  Quit")
        quit_btn.setObjectName("quitBtn")
        quit_btn.clicked.connect(self.close)
        ctrl_row.addWidget(quit_btn)

        right_layout.addLayout(ctrl_row)

        root_layout.addWidget(right, stretch=4)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("appStatus")
        self.status_bar.showMessage(
            "System ready — show a sign to the camera"
        )

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------
    def _apply_stylesheet(self):
        self.setStyleSheet("""
        /* ── Root ─────────────────────────────────────────────── */
        QMainWindow, QWidget {
            background: #0f0f1a;
            color: #e0e0f0;
        }

        /* ── Cards ─────────────────────────────────────────────── */
        QFrame#card {
            background: #16162a;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
        }
        QFrame#wordCard {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 #1a1a35, stop:1 #12122a);
            border: 1px solid #3d3d70;
            border-radius: 14px;
        }
        QFrame#speakCard {
            background: #141422;
            border: 1px solid #2a2a4a;
            border-radius: 10px;
        }

        /* ── Header ─────────────────────────────────────────────── */
        QWidget#header {
            background: #12122a;
            border-radius: 10px 10px 0 0;
        }
        QLabel#headerTitle {
            font-size: 15px;
            font-weight: 700;
            color: #a29bfe;
            letter-spacing: 0.5px;
        }
        QLabel#statLabel {
            font-size: 12px;
            color: #6c6c9a;
            font-family: 'Consolas', monospace;
        }
        QLabel#statSep {
            color: #2a2a4a;
            font-size: 12px;
        }

        /* ── Camera feed ─────────────────────────────────────────── */
        QLabel#cameraFeed {
            background: #0a0a14;
            border-radius: 0;
            color: #4a4a6a;
            font-size: 14px;
        }

        /* ── Buffer label ─────────────────────────────────────────── */
        QLabel#bufferLabel {
            font-size: 10px;
            color: #44446a;
            font-family: 'Consolas', monospace;
            padding-right: 4px;
        }

        /* ── Section headers ─────────────────────────────────────── */
        QLabel#sectionHeader {
            font-size: 10px;
            font-weight: 700;
            color: #5a5a8a;
            letter-spacing: 1.5px;
        }

        /* ── Detected word ─────────────────────────────────────────── */
        QLabel#wordLabel {
            font-size: 42px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: 1px;
        }

        /* ── Micro labels ─────────────────────────────────────────── */
        QLabel#microLabel {
            font-size: 11px;
            color: #5a5a8a;
        }

        /* ── Speaking indicator ─────────────────────────────────── */
        QLabel#speakLabel {
            font-size: 13px;
            color: #9a9ab0;
        }

        /* ── History list ─────────────────────────────────────────── */
        QListWidget#historyList {
            background: #0f0f1e;
            border: 1px solid #1e1e38;
            border-radius: 8px;
            font-size: 13px;
            color: #c0c0e0;
            outline: none;
        }
        QListWidget#historyList::item {
            padding: 5px 10px;
            border-bottom: 1px solid #1a1a2e;
        }
        QListWidget#historyList::item:selected {
            background: #2a2a50;
            color: #ffffff;
        }
        QListWidget#historyList::item:hover {
            background: #1e1e38;
        }

        /* ── Buttons ─────────────────────────────────────────────── */
        QPushButton#clearBtn {
            background: #1e1e38;
            border: 1px solid #3a3a60;
            border-radius: 6px;
            color: #7a7aa0;
            font-size: 11px;
        }
        QPushButton#clearBtn:hover {
            background: #2a2a50;
            color: #a0a0c0;
        }

        QPushButton#ctrlBtn {
            background: #1e2a4a;
            border: 1px solid #2e4a7a;
            border-radius: 8px;
            color: #6aaaf0;
            font-size: 13px;
            padding: 8px 14px;
        }
        QPushButton#ctrlBtn:hover {
            background: #253560;
        }
        QPushButton#ctrlBtn:checked {
            background: #1a3a1a;
            border-color: #2a6a2a;
            color: #60d060;
        }

        QPushButton#quitBtn {
            background: #2a1a1a;
            border: 1px solid #6a2a2a;
            border-radius: 8px;
            color: #d06060;
            font-size: 13px;
            padding: 8px 14px;
        }
        QPushButton#quitBtn:hover {
            background: #3a1a1a;
            color: #f08080;
        }

        /* ── Status bar ─────────────────────────────────────────── */
        QStatusBar#appStatus {
            background: #0a0a14;
            color: #44446a;
            font-size: 11px;
            border-top: 1px solid #1a1a2e;
        }
        """)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _start_worker(self):
        self.worker = CameraWorker(
            engine=self.engine,
            camera_idx=self.camera_idx,
            auto_speak=self.auto_speak,
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.sign_detected.connect(self._on_sign)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.speaking_changed.connect(self._on_speaking)
        self.worker.buffer_updated.connect(self._on_buffer)
        self.worker.start()

    # ------------------------------------------------------------------
    # Slots — called on the main thread from worker signals
    # ------------------------------------------------------------------
    def _on_frame(self, qimg: QImage):
        """Render the latest annotated frame into the camera label."""
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pix)

    def _on_sign(self, label: str):
        """A new sign was confidently detected — update word display & history."""
        if label == self._current_word:
            return
        self._current_word = label

        # Update big word display
        self.word_label.setText(label)

        # Update confidence display
        conf = self.engine.last_confidence
        self.conf_lbl.setText(f"{conf:.0%}")

        # Flash word label
        self._flash_word()

        # Add to history list
        self._add_to_history(label)

        # Status bar
        self.status_bar.showMessage(
            f"Detected: {label}  ({conf:.0%} confidence)"
        )

    def _on_stats(self, fps: float, latency: float):
        """Update the FPS / latency stats in the header."""
        self.fps_lbl.setText(f"FPS: {fps:.1f}")
        if latency > 0:
            self.lat_lbl.setText(f"Latency: {latency:.1f} ms")

    def _on_speaking(self, word: str):
        """Update the speaking indicator."""
        if word:
            self.speak_dot.set_active(True)
            self.speak_lbl.setText(f"🔊  Speaking: {word}")
        else:
            self.speak_dot.set_active(False)
            self.speak_lbl.setText("Not speaking")

    def _on_buffer(self, filled: int, total: int):
        """Update the buffer bar and label."""
        self.buffer_bar.update_ratio(filled, total)
        self.buffer_lbl.setText(f"Buffer: {filled} / {total}")

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _add_to_history(self, word: str):
        self._history.insert(0, word)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]

        self.history_list.clear()
        for idx, w in enumerate(self._history):
            item = QListWidgetItem()
            item.setText(f"  {idx + 1:02d}.  {w}")
            if idx == 0:
                item.setForeground(QColor("#a29bfe"))
            else:
                item.setForeground(QColor("#8888aa"))
            self.history_list.addItem(item)

    def _clear_history(self):
        self._history.clear()
        self._current_word = ""
        self.history_list.clear()
        self.word_label.setText("—")
        self.conf_lbl.setText("—")
        self.status_bar.showMessage("History cleared — show a sign to the camera")

    # ------------------------------------------------------------------
    # Word flash animation
    # ------------------------------------------------------------------
    def _flash_word(self):
        """Briefly brighten the word label to signal a new detection."""
        effect = QGraphicsOpacityEffect(self.word_label)
        self.word_label.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(400)
        anim.setStartValue(0.3)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    # ------------------------------------------------------------------
    # Speech toggle
    # ------------------------------------------------------------------
    def _toggle_speech(self, checked: bool):
        self.auto_speak = checked
        self.worker.auto_speak = checked
        if checked:
            self.speech_btn.setText("🔊  Speech ON")
        else:
            self.speech_btn.setText("🔇  Speech OFF")

    # ------------------------------------------------------------------
    # Cleanup on close
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """Stop the worker thread and release resources cleanly."""
        self.status_bar.showMessage("Shutting down…")
        if hasattr(self, "worker"):
            self.worker.stop()
        self.engine.stop()
        event.accept()


# ===========================================================================
# Public entry point
# ===========================================================================
def launch_app(engine: SignLanguageInferenceEngine,
               camera_idx: int = 0,
               auto_speak: bool = True) -> int:
    """
    Launch the PyQt5 application.

    Args:
        engine:      Fully initialised SignLanguageInferenceEngine.
        camera_idx:  Camera device index (default 0).
        auto_speak:  Whether TTS is enabled at startup.

    Returns:
        Exit code (int) — pass to sys.exit().
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("TSL Sign Recognition")
    app.setApplicationVersion("1.0")
    window = MainWindow(engine=engine,
                        camera_idx=camera_idx,
                        auto_speak=auto_speak)
    window.show()
    return app.exec_()


# ===========================================================================
# Standalone launch (python ui/app_window.py)
# ===========================================================================
if __name__ == "__main__":
    import argparse

    _DEFAULT_MODEL  = os.path.join(_PROJECT_ROOT, "models", "best_model_transformer.keras")
    _DEFAULT_LABELS = os.path.join(_PROJECT_ROOT, "assets", "SignList_ClassId_TR_EN.csv")

    parser = argparse.ArgumentParser(description="TSL Sign Recognition GUI")
    parser.add_argument("--camera",     type=int,   default=0,            help="Camera index")
    parser.add_argument("--model",      type=str,   default=_DEFAULT_MODEL)
    parser.add_argument("--labels",     type=str,   default=_DEFAULT_LABELS)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--no-speech",  action="store_true")
    _args = parser.parse_args()

    print("[GUI] Loading label map…")
    _label_map = load_label_map(_args.labels)
    if not _label_map:
        print(f"[ERROR] Cannot load labels from {_args.labels}")
        sys.exit(1)

    print(f"[GUI] Loaded {len(_label_map)} labels.")
    print("[GUI] Initialising inference engine…")
    _engine = SignLanguageInferenceEngine(
        model_path=_args.model,
        label_map=_label_map,
        confidence_threshold=_args.confidence,
    )

    sys.exit(launch_app(_engine, _args.camera, not _args.no_speech))
