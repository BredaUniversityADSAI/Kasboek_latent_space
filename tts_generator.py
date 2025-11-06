"""
Text-to-Speech using Google TTS (fast and reliable)
"""

import os
import time

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TTSGenerator:
    """Generate speech using Google TTS"""

    def __init__(self):
        self.available = TTS_AVAILABLE
        if not self.available:
            print("   TTS not available (install: uv add gtts)")

    def generate(self, text: str, pattern: str) -> str:
        """
        Generate audio from text
        Returns: path to audio file or None
        """
        if not self.available:
            return None

        try:
            clean_text = ' '.join(text.split())

            tts = gTTS(text=clean_text, lang='en', slow=False)

            os.makedirs('audio_output', exist_ok=True)
            audio_file = f"audio_output/poem_{int(time.time())}.mp3"

            tts.save(audio_file)

            return audio_file

        except Exception as e:
            print(f"   TTS error: {e}")
            return None

    def play(self, audio_file: str):
        """Play audio file"""
        if not audio_file or not os.path.exists(audio_file):
            return

        try:
            if os.system("which ffplay > /dev/null 2>&1") == 0:
                os.system(f"ffplay -nodisp -autoexit {audio_file} > /dev/null 2>&1 &")
            elif os.system("which mpg123 > /dev/null 2>&1") == 0:
                os.system(f"mpg123 -q {audio_file} &")
            elif os.name == 'nt':  # Windows
                os.system(f'start {audio_file}')
        except Exception:
            pass
