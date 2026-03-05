import wave
from piper import PiperVoice

# Use the provided Turkish voice model
model_path = "c:\\Users\\ISMAIL YAHYA\\Desktop\\Piper\\tr_TR-dfki-medium.onnx"
voice = PiperVoice.load(model_path)

# Turkish text to synthesize
text = "Merhaba, bu Piper TTS ile oluşturulmuş bir test sesidir."

with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav(text, wav_file)

print("Audio saved to test.wav")
