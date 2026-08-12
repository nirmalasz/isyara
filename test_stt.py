import time
import wave
import tempfile
import os
import pyaudio
from faster_whisper import WhisperModel
import torch

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Configuration for Microphone & Model
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4  # Records in 4-second chunks to translate live segments

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Faster-Whisper model on {device.upper()}...")

# Using 'base' model for high speed on your RTX 4050
model = WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")

p = pyaudio.PyAudio()

def record_audio_chunk():
    """Records a short audio snippet from the microphone"""
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        
    stream.stop_stream()
    stream.close()
    
    # Save to a temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_filename = temp_file.name
    temp_file.close()
    
    wf = wave.open(temp_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return temp_filename

def main():
    print("Microphone is live! Start speaking in Indonesian. Press Ctrl+C to stop.\n")
    try:
        while True:
            audio_path = record_audio_chunk()
            
            # Transcribe audio using Faster-Whisper, forced to Indonesian ('id')
            segments, info = model.transcribe(audio_path, language="id", beam_size=1)
            
            text = " ".join([segment.text for segment in segments]).strip()
            if text:
                print(f"> Transcribed: {text}")
                
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
    except KeyboardInterrupt:
        print("Stopping live transcription test.")
        p.terminate()

if __name__ == "__main__":
    main()