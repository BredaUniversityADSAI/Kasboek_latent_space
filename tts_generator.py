"""
Text-to-Speech generation module
"""

import os
import time

try:
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TTSGenerator:
    """Generate speech from text with voice variations"""

    def __init__(self):
        if not TTS_AVAILABLE:
            self.available = False
            return

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                "parler-tts/parler-tts-mini-v1"
            ).to(self.device)
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def get_voice_for_pattern(self, pattern: str) -> str:
        """Map gaze pattern to voice characteristics"""
        voices = {
            "focused_center": "A calm, deep, contemplative voice with slow pace",
            "scattered_fast": "An energetic, quick-paced voice with excitement",
            "methodical_scan": "A measured, deliberate voice with even rhythm",
            "erratic_jumps": "An intense, dynamic voice with varying pace",
            "slow_drift": "A dreamy, soft voice with gentle flow",
            "peripheral_avoidance": "A cautious, reserved voice with hesitation",
            "rapid_return_center": "A steady, anchoring voice with purpose",
            "diagonal_exploration": "A curious, playful voice with lightness"
        }
        return voices.get(pattern, "A neutral, clear voice")

    def generate(self, text: str, pattern: str) -> str:
        """
        Generate audio from text
        Returns: path to audio file or None
        """
        if not self.available:
            return None

        try:
            voice_desc = self.get_voice_for_pattern(pattern)

            input_ids = self.tokenizer(voice_desc, return_tensors="pt").input_ids.to(self.device)
            prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )
            audio_arr = generation.cpu().numpy().squeeze()

            os.makedirs('audio_output', exist_ok=True)
            audio_file = f"audio_output/poem_{int(time.time())}.wav"
            sf.write(audio_file, audio_arr, self.model.config.sampling_rate)

            return audio_file

        except Exception:
            return None

    def play(self, audio_file: str):
        """Play audio file"""
        if not audio_file or not os.path.exists(audio_file):
            return

        try:
            if os.system("which aplay > /dev/null 2>&1") == 0:
                os.system(f"aplay {audio_file} > /dev/null 2>&1")
            elif os.system("which ffplay > /dev/null 2>&1") == 0:
                os.system(f"ffplay -nodisp -autoexit {audio_file} > /dev/null 2>&1")
        except Exception:
            pass