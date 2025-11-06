"""
Main poem generation orchestrator
"""

import time
from rorschach_interpreter import RorschachInterpreter, GazePatternClassifier
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from print_formatter import PrintFormatter
from data_exporter import DataExporter
from config import PATHS

try:
    from tts_generator import TTSGenerator
    TTS_ENABLED = True
except Exception:
    TTS_ENABLED = False


class PoemOrchestrator:
    """Orchestrate the poem generation process"""

    def __init__(self):
        self.interpreter = RorschachInterpreter(PATHS["rorschach_excel"])
        self.classifier = GazePatternClassifier()
        self.llm = LLMClient()
        self.prompt_builder = PromptBuilder()
        self.printer = PrintFormatter()
        self.exporter = DataExporter()

        if TTS_ENABLED:
            print("   TTS: Loading (this may take a minute)...")
            self.tts = TTSGenerator()
            print("   TTS: Ready")
        else:
            self.tts = None

    def process(self, gaze_history: list) -> dict:
        """
        Process gaze data and generate all outputs
        Returns result dictionary
        """
        pattern, metrics = self.classifier.classify_gaze_pattern(gaze_history)
        print(f"Pattern detected: {pattern}")

        rorschach_data = self.interpreter.generate_interpretation_set(pattern)
        prompt = self.prompt_builder.build_poem_prompt(
            pattern,
            metrics,
            rorschach_data['all_interpretations'],
            rorschach_data['poem_tone']
        )

        print("Generating poem via LLM...")
        poem = self.llm.generate_poem(prompt)

        timestamp = time.time()

        result = {
            'poem': poem,
            'pattern': pattern,
            'metrics': metrics,
            'timestamp': timestamp,
            'rorschach_data': rorschach_data
        }

        if self.tts and self.tts.available:
            print("Generating audio...")
            audio_file = self.tts.generate(poem, pattern)
            if audio_file:
                print(f"Audio saved: {audio_file}")
                self.tts.play(audio_file)

        self.printer.save(poem, pattern, metrics, timestamp)

        self.exporter.export_poem_data(
            poem, pattern, metrics,
            rorschach_data['primary_categories'],
            timestamp, gaze_history
        )

        return result