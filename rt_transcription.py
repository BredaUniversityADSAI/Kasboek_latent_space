import asyncio
import numpy as np
import sounddevice as sd
import whisper

# Configuration
def configure_transcription(model="turbo"):
    SAMPLE_RATE = 8000
    CHUNK_DURATION = 2.0  # seconds
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

    # Load Whisper model (small/medium/large)
    model = whisper.load_model(model)

    # Audio buffer queue
    audio_buffer = asyncio.Queue()
    transcription_queue = asyncio.Queue()

    return SAMPLE_RATE, CHUNK_DURATION, CHUNK_SAMPLES, model, audio_buffer, transcription_queue

# -----------------------------
# Capture audio from microphone
# -----------------------------
async def capture_audio(audio_buffer, SAMPLE_RATE):
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
async def transcribe_audio(audio_buffer, CHUNK_SAMPLES, model, transcription_queue):

    buffer = np.zeros((0,), dtype=np.float32)
    while True:
        chunk = await audio_buffer.get()
        chunk = chunk.flatten()
        buffer = np.concatenate((buffer, chunk))

        if len(buffer) >= CHUNK_SAMPLES:
            result = model.transcribe(buffer, language='en', fp16=False)
            transcription = result["text"].strip()
            print("TRANSCRIPTION:", transcription)
            await transcription_queue.put(transcription)
            buffer = np.zeros((0,), dtype=np.float32)

# -----------------------------
# Main async function
# -----------------------------
async def main():
    SAMPLE_RATE, CHUNK_DURATION, CHUNK_SAMPLES, model, audio_buffer = configure_transcription()
    await asyncio.gather(
        capture_audio(audio_buffer, SAMPLE_RATE),
        transcribe_audio(audio_buffer, CHUNK_SAMPLES, model)
    )

if __name__ == "__main__":
    asyncio.run(main())
