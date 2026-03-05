import wave
from piper import PiperVoice, SynthesisConfig

model_path = "c:\\Users\\ISMAIL YAHYA\\Desktop\\Piper\\tr_TR-dfki-medium.onnx"
voice = PiperVoice.load(model_path)

text = "Bu, ayarlanmış parametrelerle yapılan bir testtir. Daha yavaş konuşuyorum."

# Configure synthesis: slower, louder, etc.
# length_scale > 1.0 makes it slower
# volume default is 1.0
syn_config = SynthesisConfig(
    volume=1.0,
    length_scale=1.5,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=False
)

with wave.open("test_config.wav", "wb") as wav_file:
    # Note: synthesize_wav with config requires keyword argument 'synthesis_config' 
    # OR dependent on library version, let's try passing it.
    # The user example used: voice.synthesize_wav(..., syn_config=syn_config)
    voice.synthesize_wav(text, wav_file, syn_config=syn_config)

print("Audio saved to test_config.wav")
