import threading
import queue
import time
import sys

class SpeechEngine:
    """
    محرك نطق مستقل يستخدم Piper TTS مع تقنية التدفق الصوتي (Audio Streaming).
    يعمل في Thread منفصل لعدم عرقلة المحرك الرئيسي، ويدعم Callbacks لتحديث الواجهة.
    """
    def __init__(self, model_path=None, cooldown=3.0):
        if model_path is None:
            # Reached through the new standardized directory structure
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "assets", "tr_TR-dfki-medium.onnx")
        else:
            self.model_path = model_path
        self.queue = queue.Queue()
        self.running = True
        self.voice = None
        self.cooldown = cooldown
        
        # Idempotency Tracking
        self.last_spoken_word = None
        self.last_spoken_time = 0.0
        
        # Callbacks List
        self.on_speech_start_callbacks = []
        self.on_speech_end_callbacks = []

        # Start background worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def add_start_callback(self, callback):
        """إضافة دالة تُستدعى حين بداية النطق."""
        self.on_speech_start_callbacks.append(callback)

    def add_end_callback(self, callback):
        """إضافة دالة تُستدعى حين نهاية النطق."""
        self.on_speech_end_callbacks.append(callback)

    def speak(self, text):
        """
        طلب نطق كلمة معينة. يستخدم إدارة المحادثة (Idempotency) لمنع تكرار النطق.
        """
        # Idempotency check: don't repeat the exact same word within cooldown time
        current_time = time.time()
        if text == self.last_spoken_word and (current_time - self.last_spoken_time) < self.cooldown:
            return  # تجاهل النطق لعدم التكرار (Idempotency)
        
        self.last_spoken_word = text
        self.last_spoken_time = current_time
        
        # Add to processing queue
        self.queue.put(text)

    def _worker(self):
        """عملية بالخلفية مسؤولة عن النطق لعدم تعطيل الكاميرا (Non-Blocking)."""
        # Import dynamically to fail gracefully if uninstalled
        try:
            from piper.voice import PiperVoice
            import pyaudio
        except ImportError:
            print("[SpeechEngine] Warning: 'piper-tts' or 'pyaudio' is not installed.")
            print("[SpeechEngine] Please install them using: pip install piper-tts pyaudio")
            self.running = False
            return

        # Load Piper Model
        print(f"[SpeechEngine] Loading Piper model from {self.model_path}...")
        try:
            self.voice = PiperVoice.load(self.model_path)
        except Exception as e:
            print(f"[SpeechEngine] Error loading Piper model: {e}")
            self.running = False
            return

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=self.voice.config.sample_rate,
                            output=True)
            print("[SpeechEngine] Ready for Audio Streaming.")
        except Exception as e:
            print(f"[SpeechEngine] PyAudio error opening stream: {e}")
            self.running = False
            return

        # Processing Loop
        while self.running:
            try:
                # Wait for text to speak
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if text is None:
                break

            # Trigger Start Callbacks
            for cb in self.on_speech_start_callbacks:
                try:
                    cb(text)
                except:
                    pass

            # Audio Generation (Synthesis)
            try:
                # `self.voice.synthesize` yields `AudioChunk` objects
                for audio_chunk in self.voice.synthesize(text):
                    if not self.running:
                        break
                    # Write the signed 16-bit PCM bytes array to PyAudio stream
                    stream.write(audio_chunk.audio_int16_bytes)
            except Exception as e:
                print(f"[SpeechEngine] Synthesis streaming error: {e}")

            # Trigger End Callbacks
            for cb in self.on_speech_end_callbacks:
                try:
                    cb(text)
                except:
                    pass
                
            self.queue.task_done()

        # Shutdown Stream
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass

    def stop(self):
        """إيقاف المحرك وتحرير الموارد."""
        self.running = False
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join()
