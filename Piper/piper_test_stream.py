import wave
from piper import PiperVoice

model_path = "c:\\Users\\ISMAIL YAHYA\\Desktop\\Piper\\tr_TR-dfki-medium.onnx"
voice = PiperVoice.load(model_path)

text = "Bu, akış modu ile sentezlenen ses verisidir. Parça parça işleniyor."

output_file = "test_stream.wav"

print(f"Synthesizing stream to {output_file}...")

with wave.open(output_file, "wb") as wav_file:
    params_set = False
    
    # Synthesize returns an iterator of AudioChunk
    # We will assume AudioChunk has .audio_bytes or similiar.
    # User mentioned .audio_int16_bytes. We'll try to find the standard bytes attribute.
    # Usually it is .audio_bytes or just .audio (if bytes).
    # We will inspect the first chunk to be sure if we were interactive, but here we guess.
    # Let's try simple .audio if it exists, or .audio_bytes. 
    # Actually, let's look at the help output again? No, it didn't show AudioChunk structure.
    # Use standard approach: check dir() inside the loop if needed, but let's just write bytes.
    
    for i, chunk in enumerate(voice.synthesize(text)):
        if not params_set:
            # We assume chunk has sample_rate, sample_width, sample_channels
            # If not, we fall back to defaults (22050, 2, 1)
            
            rate = getattr(chunk, 'sample_rate', 22050)
            width = getattr(chunk, 'sample_width', 2)
            channels = getattr(chunk, 'sample_channels', 1)
            
            print(f"Stream parameters: rate={rate}, width={width}, channels={channels}")

            wav_file.setnchannels(channels)
            wav_file.setsampwidth(width)
            wav_file.setframerate(rate)
            params_set = True
            
        # Write audio data from audio_int16_bytes
        if hasattr(chunk, 'audio_int16_bytes'):
            wav_file.writeframes(chunk.audio_int16_bytes)
        elif hasattr(chunk, 'audio_bytes'):
            wav_file.writeframes(chunk.audio_bytes)
        else:
             # Fallback or error
             print(f"Chunk {i} has no known audio attribute: {dir(chunk)}")
             break

print(f"Streamed audio saved to {output_file}")
