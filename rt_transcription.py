import asyncio
import numpy as np
import sounddevice as sd
import whisper

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Load Whisper model (small/medium/large)
model = whisper.load_model("turbo")

# Audio buffer queue
audio_buffer = asyncio.Queue()

# -----------------------------
# Capture audio from microphone
# -----------------------------
async def capture_audio():
    def callback(indata, frames, time, status):
        audio_buffer.put_nowait(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=1024
    ):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(0.01)

# -----------------------------
# Process audio with Whisper
# -----------------------------
async def transcribe_audio():
    buffer = np.zeros((0,), dtype=np.float32)
    while True:
        chunk = await audio_buffer.get()
        chunk = chunk.flatten()
        buffer = np.concatenate((buffer, chunk))

        if len(buffer) >= CHUNK_SAMPLES:
            result = model.transcribe(buffer, language='en', fp16=False)
            print("TRANSCRIPTION:", result["text"].strip())
            buffer = np.zeros((0,), dtype=np.float32)

# -----------------------------
# Main async function
# -----------------------------
async def main():
    await asyncio.gather(
        capture_audio(),
        transcribe_audio()
    )

if __name__ == "__main__":
    asyncio.run(main())
