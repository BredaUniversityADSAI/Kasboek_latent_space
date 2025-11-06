# poem_generation_pipeline.py
"""
COMPLETE Poem Generation Pipeline - End-to-End System
Handles: Eye tracking → Analysis → Poem → TTS → Print → Display
ALL COMPONENTS INTEGRATED
"""

import asyncio
import time
import json
import requests
from typing import Dict, List
from collections import deque
import cv2
import os
from datetime import datetime

# Import configuration
try:
    from config import (
        API_CONFIG,
        COLLECTION_SETTINGS,
        CLASSIFICATION_THRESHOLDS,
        PATHS,
        POEM_SETTINGS,
        DEBUG,
        validate_config
    )
except ImportError:
    print("ERROR: config.py not found!")
    exit(1)

# Import modules
from eye_tracking import EyeTracker
from rorschach_interpreter import (
    RorschachInterpreter,
    GazePatternClassifier,
    PATTERN_DESCRIPTIONS
)

# TTS imports
try:
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf

    TTS_AVAILABLE = True
except ImportError:
    print("⚠ Warning: TTS modules not available. Install: pip install parler-tts transformers soundfile")
    TTS_AVAILABLE = False


class TTSGenerator:
    """
    Text-to-Speech generation with voice variations
    """

    def __init__(self):
        if not TTS_AVAILABLE:
            print("⚠ TTS not available - poems will be text only")
            self.model = None
            return

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  Loading TTS model on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                "parler-tts/parler-tts-mini-v1"
            ).to(self.device)
            print("  ✓ TTS model loaded")
        except Exception as e:
            print(f"  ⚠ TTS loading failed: {e}")
            self.model = None

    def get_voice_description(self, gaze_pattern: str) -> str:
        """Map gaze pattern to voice characteristics"""
        voice_map = {
            "focused_center": "A calm, deep, contemplative voice with slow pace",
            "scattered_fast": "An energetic, quick-paced voice with excitement",
            "methodical_scan": "A measured, deliberate voice with even rhythm",
            "erratic_jumps": "An intense, dynamic voice with varying pace",
            "slow_drift": "A dreamy, soft voice with gentle flow",
            "peripheral_avoidance": "A cautious, reserved voice with hesitation",
            "rapid_return_center": "A steady, anchoring voice with purpose",
            "diagonal_exploration": "A curious, playful voice with lightness"
        }
        return voice_map.get(gaze_pattern, "A neutral, clear voice")

    def generate_audio(self, poem_text: str, gaze_pattern: str) -> str:
        """
        Generate audio from poem text
        Returns path to audio file
        """
        if self.model is None:
            print("  ⚠ TTS not available, skipping audio generation")
            return None

        try:
            print("  Generating audio...")

            # Get voice description based on pattern
            voice_desc = self.get_voice_description(gaze_pattern)

            # Tokenize
            input_ids = self.tokenizer(voice_desc, return_tensors="pt").input_ids.to(self.device)
            prompt_input_ids = self.tokenizer(poem_text, return_tensors="pt").input_ids.to(self.device)

            # Generate
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )
            audio_arr = generation.cpu().numpy().squeeze()

            # Save
            os.makedirs('audio_output', exist_ok=True)
            audio_file = f"audio_output/poem_{int(time.time())}.wav"
            sf.write(audio_file, audio_arr, self.model.config.sampling_rate)

            print(f"  ✓ Audio saved: {audio_file}")
            return audio_file

        except Exception as e:
            print(f"  ❌ Audio generation failed: {e}")
            return None

    def play_audio(self, audio_file: str):
        """Play audio file through speakers"""
        if audio_file is None or not os.path.exists(audio_file):
            return

        try:
            # Try different playback methods
            if os.system("which aplay > /dev/null 2>&1") == 0:
                os.system(f"aplay {audio_file} > /dev/null 2>&1")
            elif os.system("which ffplay > /dev/null 2>&1") == 0:
                os.system(f"ffplay -nodisp -autoexit {audio_file} > /dev/null 2>&1")
            else:
                print("  ⚠ No audio player found (aplay or ffplay)")
        except Exception as e:
            print(f"  ⚠ Audio playback failed: {e}")


class PrintFormatter:
    """
    Format and save poems for printing
    """

    def __init__(self):
        os.makedirs('print_queue', exist_ok=True)

    def format_for_print(self, result: Dict) -> str:
        """
        Create A4-formatted text for printing
        """
        poem = result['poem']
        pattern = result['pattern']
        timestamp = result['timestamp']
        metrics = result['metrics']

        # Create formatted output
        output = []
        output.append("=" * 60)
        output.append("INNER STATE - A Poetic Interpretation")
        output.append("=" * 60)
        output.append("")
        output.append(f"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Gaze Pattern: {pattern.replace('_', ' ').title()}")
        output.append("")
        output.append("-" * 60)
        output.append("POEM")
        output.append("-" * 60)
        output.append("")

        # Add poem with proper spacing
        for line in poem.split('\n'):
            output.append(line)

        output.append("")
        output.append("-" * 60)
        output.append("ANALYSIS")
        output.append("-" * 60)
        output.append("")
        output.append(f"Movement Speed: {metrics.get('avg_saccade_speed', 0):.3f}")
        output.append(f"Center Focus: {metrics.get('center_time_ratio', 0):.1%}")
        output.append(f"Pattern Entropy: {metrics.get('direction_entropy', 0):.2f}")
        output.append(f"Fixations: {metrics.get('fixation_count', 0)}")
        output.append("")
        output.append("=" * 60)
        output.append("Through the Eye of the Algorithm")
        output.append("=" * 60)

        return '\n'.join(output)

    def save_for_printing(self, result: Dict) -> str:
        """
        Save formatted poem to print queue
        """
        try:
            formatted = self.format_for_print(result)
            filename = f"print_queue/poem_{int(result['timestamp'])}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted)

            print(f"  ✓ Print file saved: {filename}")
            return filename

        except Exception as e:
            print(f"  ❌ Print formatting failed: {e}")
            return None

    def send_to_printer(self, filename: str):
        """
        Send file to printer (if available)
        """
        if filename is None or not os.path.exists(filename):
            return

        try:
            # Try to print using lp command (Linux)
            if os.system("which lp > /dev/null 2>&1") == 0:
                os.system(f"lp {filename}")
                print(f"  ✓ Sent to printer")
            else:
                print(f"  ⚠ Printer not configured. File saved in print_queue/")
        except Exception as e:
            print(f"  ⚠ Printing failed: {e}")


class VisualDisplayManager:
    """
    Manage visual outputs: eye tracking visualization and displays
    """

    def __init__(self):
        os.makedirs('shared', exist_ok=True)

    def save_display_data(self, result: Dict, gaze_history: List[Dict]):
        """
        Save data for visual display systems
        """
        try:
            display_data = {
                'poem': result['poem'],
                'pattern': result['pattern'],
                'timestamp': result['timestamp'],
                'metrics': result['metrics'],
                'categories': result['rorschach_data']['primary_categories'],
                'gaze_summary': self._summarize_gaze(gaze_history)
            }

            with open('shared/current_poem.json', 'w') as f:
                json.dump(display_data, f, indent=2)

            print(f"  ✓ Display data saved: shared/current_poem.json")

        except Exception as e:
            print(f"  ❌ Display data save failed: {e}")

    def _summarize_gaze(self, gaze_history: List[Dict]) -> Dict:
        """Create summary of gaze data for visualization"""
        valid_data = [g for g in gaze_history if g is not None]

        if not valid_data:
            return {}

        directions = [g.get('direction', 'center') for g in valid_data]
        speeds = [g.get('saccade_speed', 0) for g in valid_data]

        from collections import Counter
        direction_counts = Counter(directions)

        return {
            'total_frames': len(valid_data),
            'dominant_direction': direction_counts.most_common(1)[0][0] if direction_counts else 'center',
            'avg_speed': sum(speeds) / len(speeds) if speeds else 0,
            'direction_distribution': dict(direction_counts)
        }


class PoemGenerator:
    """
    Generates English poems using LLM based on gaze interpretations
    """

    def __init__(self):
        self.api_url = API_CONFIG["api_url"]
        self.assistant_key = API_CONFIG["assistant_key"]
        self.user_key = API_CONFIG["user_key"]
        self.chat_key = None
        self.poem_count = 0

        # Validate keys
        if self.user_key == "PASTE_YOUR_USER_KEY_HERE":
            raise ValueError("USER_KEY not configured in config.py")
        if self.assistant_key == "PASTE_YOUR_ASSISTANT_KEY_HERE":
            raise ValueError("ASSISTANT_KEY not configured in config.py")

    def init_conversation(self):
        """Initialize conversation with AI assistant"""
        init_endpoint = f"{self.api_url}/actions/init_assistant"
        headers = {"Content-Type": "application/json"}

        data = {
            "user_key": self.user_key,
            "assistant_key": self.assistant_key
        }

        try:
            if DEBUG["verbose"]:
                print("  Initializing LLM conversation...")

            response = requests.post(init_endpoint,
                                     data=json.dumps(data),
                                     headers=headers,
                                     timeout=10)
            assistant = response.json()

            if assistant.get('status') == "Failed":
                print(f"  ❌ ERROR: {assistant.get('error')}")
                return False

            self.chat_key = assistant.get("chat_key")

            if DEBUG["verbose"]:
                print(f"  ✓ Conversation initialized")

            return True

        except Exception as e:
            print(f"  ❌ ERROR initializing conversation: {e}")
            return False

    def create_poem_prompt(self,
                           gaze_pattern: str,
                           metrics: Dict,
                           rorschach_interpretations: List[str],
                           tone: str) -> str:
        """Build comprehensive prompt for English poem generation"""

        pattern_desc = PATTERN_DESCRIPTIONS.get(gaze_pattern, {})
        selected_interps = rorschach_interpretations[:4]

        # Translate Dutch pattern descriptions to English
        essence_map = {
            "snel, rusteloze zoektocht": "rapid, restless seeking",
            "diepe contemplatie, naar binnen keren": "deep contemplation, inward turning",
            "systematische verkenning, orde zoeken": "systematic exploration, seeking order",
            "gefragmenteerde aandacht, elektrische beweging": "fragmented attention, electric movement",
            "zwevend bewustzijn, zacht dwalen": "floating awareness, gentle wandering",
            "voorzichtige observatie, terughoudend": "cautious observation, holding back",
            "anker zoeken, grond vinden": "seeking anchor, finding ground",
            "speelse ontdekking, onconventionele paden": "playful discovery, unconventional paths"
        }

        essence = essence_map.get(pattern_desc.get('essence', ''), pattern_desc.get('essence', 'a unique pattern'))

        prompt = f"""Write a short, evocative poem in English (6-10 lines).

CONTEXT:
A person's gaze was observed for 15 seconds, revealing a pattern of attention.

GAZE PATTERN: "{gaze_pattern}"
- Essence: {essence}
- Movement speed: {metrics.get('avg_saccade_speed', 0):.3f}
- Center focus: {metrics.get('center_time_ratio', 0):.1%}
- Pattern complexity: {metrics.get('direction_entropy', 0):.2f}

SYMBOLIC IMAGES (from their gaze):
{chr(10).join(f'- {interp}' for interp in selected_interps)}

DESIRED TONE: {tone}

Create a poem that:
1. Captures this person's inner state as revealed by their gaze
2. Weaves the symbolic images into meaningful poetry
3. Uses metaphorical language
4. Feels personal yet universal
5. Reflects the tone: {tone}

Write ONLY the poem, no title, no explanation."""

        return prompt

    def generate_poem(self,
                      gaze_pattern: str,
                      metrics: Dict,
                      rorschach_data: Dict) -> str:
        """Generate poem via LLM API"""

        # Reinitialize periodically
        reinit_every = POEM_SETTINGS["reinit_conversation_every"]
        if self.poem_count % reinit_every == 0 or not self.chat_key:
            if not self.init_conversation():
                if POEM_SETTINGS["use_fallback_on_error"]:
                    return self._get_fallback_poem(gaze_pattern)
                return "ERROR: Could not initialize conversation"

        self.poem_count += 1

        # Build prompt
        interpretations = rorschach_data.get('all_interpretations', [])
        tone = rorschach_data.get('poem_tone', 'mysterious and philosophical')
        prompt = self.create_poem_prompt(gaze_pattern, metrics, interpretations, tone)

        # Call API
        ask_endpoint = f"{self.api_url}/actions/ask_assistant"
        headers = {"Content-Type": "application/json"}

        data = {
            'assistant_key': self.assistant_key,
            'chat_key': self.chat_key,
            'message': prompt,
            'user_key': self.user_key
        }

        try:
            if DEBUG["verbose"]:
                print("  Generating poem via LLM...")

            response = requests.post(ask_endpoint,
                                     data=json.dumps(data),
                                     headers=headers,
                                     timeout=POEM_SETTINGS["api_timeout"])

            result = response.json()

            # Handle response
            if isinstance(result, dict):
                poem = result.get('response', result.get('message', result.get('text', '')))
            else:
                poem = str(result)

            poem = poem.strip()

            if len(poem) < POEM_SETTINGS["min_poem_length"]:
                if DEBUG["verbose"]:
                    print(f"  ⚠ Response too short, using fallback")
                if POEM_SETTINGS["use_fallback_on_error"]:
                    return self._get_fallback_poem(gaze_pattern)

            if DEBUG["verbose"]:
                print(f"  ✓ Poem generated ({len(poem)} characters)")

            return poem

        except requests.Timeout:
            print(f"  ❌ LLM timeout")
            if POEM_SETTINGS["use_fallback_on_error"]:
                return self._get_fallback_poem(gaze_pattern)
            return "ERROR: Timeout"

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            if POEM_SETTINGS["use_fallback_on_error"]:
                return self._get_fallback_poem(gaze_pattern)
            return f"ERROR: {str(e)}"

    def _get_fallback_poem(self, gaze_pattern: str) -> str:
        """Fallback poems when LLM fails"""
        fallbacks = {
            "focused_center": """In the center of stillness
Where the gaze finds rest
A cathedral forms from mist
A mystery that persists""",

            "scattered_fast": """Eyes dance, never resting
Through a world of fragments
Seeking what doesn't exist
In the chaos of the moment""",

            "slow_drift": """Slowly attention drifts
Like clouds through evening
Each moment a new form
In the space between dreams""",
        }

        return fallbacks.get(gaze_pattern, """In the gaze lies a story
Of what cannot be spoken
A pattern that unfolds
In the silence of looking""")


class CompletePoemPipeline:
    """
    COMPLETE END-TO-END PIPELINE
    Integrates all components
    """

    def __init__(self):
        self.collection_duration = COLLECTION_SETTINGS["duration_seconds"]
        self.fps_estimate = COLLECTION_SETTINGS["fps_estimate"]

        print("\n" + "=" * 60)
        print("INITIALIZING COMPLETE POEM GENERATION SYSTEM")
        print("=" * 60)

        # Initialize all components
        print("\n1. Loading Eye Tracker...")
        self.eye_tracker = EyeTracker(flip=True)
        print("   ✓ Eye tracker ready")

        print("\n2. Loading Rorschach Interpreter...")
        self.interpreter = RorschachInterpreter(PATHS["rorschach_excel"])
        print("   ✓ Rorschach interpreter ready")

        print("\n3. Loading Gaze Classifier...")
        self.classifier = GazePatternClassifier()
        self.classifier.thresholds = CLASSIFICATION_THRESHOLDS.copy()
        print("   ✓ Classifier ready")

        print("\n4. Loading Poem Generator...")
        self.poem_generator = PoemGenerator()
        print("   ✓ Generator ready")

        print("\n5. Loading TTS System...")
        self.tts = TTSGenerator()
        print("   ✓ TTS ready")

        print("\n6. Loading Print Formatter...")
        self.printer = PrintFormatter()
        print("   ✓ Print system ready")

        print("\n7. Loading Display Manager...")
        self.display = VisualDisplayManager()
        print("   ✓ Display manager ready")

        # Data buffer
        max_frames = int(self.collection_duration * self.fps_estimate * 1.5)
        self.gaze_buffer = deque(maxlen=max_frames)

        print(f"\n✓ COMPLETE SYSTEM INITIALIZED")
        print(f"  Collection duration: {self.collection_duration}s")
        print("=" * 60 + "\n")

    def collect_gaze_data(self, cap) -> List[Dict]:
        """Collect gaze data for specified duration"""
        print(f"\n{'=' * 60}")
        print(f"COLLECTING GAZE DATA ({self.collection_duration}s)")
        print(f"{'=' * 60}")

        self.gaze_buffer.clear()
        start_time = time.time()
        frame_count = 0
        valid_count = 0

        while time.time() - start_time < self.collection_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            if self.eye_tracker.flip:
                frame = cv2.flip(frame, 1)

            try:
                _, gaze_info = self.eye_tracker.process_frame(frame, only_compute=True)
                self.gaze_buffer.append(gaze_info)

                if gaze_info is not None:
                    valid_count += 1

            except Exception as e:
                self.gaze_buffer.append(None)

            elapsed = time.time() - start_time
            remaining = self.collection_duration - elapsed
            progress_pct = (elapsed / self.collection_duration) * 100

            print(f"\rProgress: {progress_pct:.0f}% | "
                  f"Frames: {frame_count} | Valid: {valid_count} | "
                  f"Remaining: {remaining:.1f}s", end='', flush=True)

            frame_count += 1
            time.sleep(0.01)

        print(f"\n✓ Collection complete: {valid_count}/{frame_count} valid frames")
        return list(self.gaze_buffer)

    def process_and_generate(self, gaze_history: List[Dict]) -> Dict:
        """Process gaze data and generate all outputs"""
        print("\n" + "=" * 60)
        print("ANALYZING & GENERATING")
        print("=" * 60)

        # 1. Classify
        print("\n1. Classifying gaze pattern...")
        pattern, metrics = self.classifier.classify_gaze_pattern(gaze_history)
        print(f"   ✓ Pattern: {pattern}")

        # 2. Get Rorschach
        print("\n2. Gathering Rorschach interpretations...")
        rorschach_data = self.interpreter.generate_interpretation_set(pattern)
        print(f"   ✓ Categories: {rorschach_data['primary_categories']}")

        # 3. Generate poem
        print("\n3. Generating poem...")
        poem = self.poem_generator.generate_poem(pattern, metrics, rorschach_data)

        print("\n" + "=" * 60)
        print("POEM:")
        print("=" * 60)
        print(poem)
        print("=" * 60)

        result = {
            'pattern': pattern,
            'metrics': metrics,
            'rorschach_data': rorschach_data,
            'poem': poem,
            'timestamp': time.time()
        }

        # 4. Generate TTS
        print("\n4. Generating audio...")
        audio_file = self.tts.generate_audio(poem, pattern)

        # 5. Format for print
        print("\n5. Formatting for print...")
        print_file = self.printer.save_for_printing(result)

        # 6. Save display data
        print("\n6. Saving display data...")
        self.display.save_display_data(result, gaze_history)

        # 7. Play audio
        if audio_file:
            print("\n7. Playing audio...")
            self.tts.play_audio(audio_file)

        return result

    async def run_continuous(self):
        """Main continuous loop"""
        wait_between = COLLECTION_SETTINGS["wait_between_poems"]

        # Open webcam
        print("\nOpening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            for idx in range(1, 5):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    break

        if not cap.isOpened():
            print("❌ ERROR: Cannot open webcam")
            return

        print("✓ Webcam opened\n")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 60)
        print("COMPLETE POEM GENERATION SYSTEM - RUNNING")
        print("=" * 60)
        print(f"Collection: {self.collection_duration}s")
        print(f"Wait between: {wait_between}s")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        poem_count = 0

        try:
            while True:
                poem_count += 1
                print(f"\n{'#' * 60}")
                print(f"POEM #{poem_count}")
                print(f"{'#' * 60}")

                # Collect
                gaze_history = self.collect_gaze_data(cap)

                # Check validity
                valid_count = sum(1 for g in gaze_history if g is not None)
                if valid_count < 30:
                    print(f"\n⚠ Insufficient data, skipping...")
                    await asyncio.sleep(2)
                    continue

                # Process & generate all outputs
                result = self.process_and_generate(gaze_history)

                print(f"\n✓ Cycle complete!")
                print(f"  - Poem generated")
                print(f"  - Audio created")
                print(f"  - Print formatted")
                print(f"  - Display data saved")

                print(f"\nWaiting {wait_between}s before next...")
                await asyncio.sleep(wait_between)

        except KeyboardInterrupt:
            print("\n\n✓ Stopping system...")
        finally:
            cap.release()
            self.eye_tracker.release()
            print("✓ System stopped\n")

    def run_single(self, cap):
        """Single test mode"""
        print("\nSINGLE-SHOT MODE\n")

        gaze_history = self.collect_gaze_data(cap)

        valid_count = sum(1 for g in gaze_history if g is not None)
        if valid_count < 30:
            print(f"\n❌ Insufficient data")
            return None

        result = self.process_and_generate(gaze_history)
        return result


async def main():
    """Main execution"""
    if not validate_config():
        print("\n❌ Configuration invalid")
        return

    pipeline = CompletePoemPipeline()
    await pipeline.run_continuous()


if __name__ == "__main__":
    import sys

    if not validate_config():
        print("\nPlease update config.py with your API keys")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("TESTING MODE\n")
        pipeline = CompletePoemPipeline()
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            result = pipeline.run_single(cap)
            cap.release()

            if result:
                print("\n✓ TEST SUCCESSFUL!")
        else:
            print("❌ Cannot open webcam")
    else:
        print("CONTINUOUS MODE\n")
        asyncio.run(main())