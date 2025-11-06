"""
Setup and testing script for the modular poem generation system
Tests all 12 modules individually
"""

import sys
import os


print("\n1. Checking configuration...")
if not os.path.exists('config.py'):
    print("  config.py not found")
    sys.exit(1)

try:
    from config import API_CONFIG, PATHS, COLLECTION_SETTINGS, validate_config

    print("  config.py loaded")
except Exception as e:
    print(f"  Error loading config: {e}")
    sys.exit(1)

# Validate configuration
if not validate_config():
    print("  Configuration invalid")
    sys.exit(1)
print("   ✓ Configuration valid")

v = sys.version_info
print(f"   Python {v.major}.{v.minor}.{v.micro}")

if v.major < 3 or (v.major == 3 and v.minor < 8):
    print(" Python 3.8+ required")
    sys.exit(1)
elif v.major == 3 and v.minor >= 12:
    print("  Python 3.12+ may have MediaPipe issues")
print(" Python version compatible")

print("\n3. Checking required modules...")
required = {
    'cv2': 'opencv-python',
    'mediapipe': 'mediapipe',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'requests': 'requests',
    'openpyxl': 'openpyxl'
}

missing = []
for module, package in required.items():
    try:
        __import__(module)
        print(f"  {module}")
    except ImportError:
        print(f"  {module}")
        missing.append(package)

if missing:
    print(f"\n   Install missing: uv add {' '.join(missing)}")
    sys.exit(1)

print("\n4. Checking optional modules (TTS)...")
optional = ['torch', 'transformers', 'parler_tts', 'soundfile']
for module in optional:
    try:
        __import__(module)
        print(f"   {module}")
    except ImportError:
        print(f"   {module} (TTS disabled)")

print("\n5. Checking project files...")
project_files = [
    'config.py',
    'gaze_collector.py',
    'llm_client.py',
    'prompt_builder.py',
    'tts_generator.py',
    'print_formatter.py',
    'data_exporter.py',
    'poem_orchestrator.py',
    'eye_visualizer.py',
    'main_pipeline.py',
    'visual_display.py',
    'run_system.py',
    'rorschach_interpreter.py',
    'eye_tracking.py',
    PATHS["rorschach_excel"]
]

missing_files = []
for filename in project_files:
    if os.path.exists(filename):
        print(f"   {filename}")
    else:
        print(f"   {filename}")
        missing_files.append(filename)

if missing_files:
    print(f"\n   Missing files: {missing_files}")
    sys.exit(1)

print("\n6. Testing Rorschach data loading...")
try:
    from rorschach_interpreter import RorschachInterpreter

    interpreter = RorschachInterpreter(PATHS["rorschach_excel"])

    if interpreter.data is None or len(interpreter.data) == 0:
        print(" No data loaded")
        sys.exit(1)

    print(f" Loaded {len(interpreter.data)} interpretations")

    sample = interpreter.get_interpretation_by_category('psychological', n=1)
    if sample:
        print(f"Sampling works")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("\n7. Testing webcam...")
try:
    import cv2

    cap = None
    for idx in range(5):
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                print(f"   ✓ Camera found at index {idx}")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap = test_cap
                break
            test_cap.release()

    if cap:
        cap.release()
    else:
        print("No camera found (may work on Jetson)")

except Exception as e:
    print(f"Camera test error: {e}")

try:
    import requests
    import json

    init_endpoint = f"{API_CONFIG['api_url']}/actions/init_assistant"
    headers = {"Content-Type": "application/json"}
    data = {
        "user_key": API_CONFIG['user_key'],
        "assistant_key": API_CONFIG['assistant_key']
    }

    response = requests.post(init_endpoint,
                             data=json.dumps(data),
                             headers=headers,
                             timeout=10)

    result = response.json()

    if result.get('status') == 'Failed':
        print(f" API failed: {result.get('error')}")
    elif 'chat_key' in result:
        print(f"API connection successful")
    else:
        print(f"Unexpected response")

except Exception as e:
    print(f"API error: {e}")

print("\n9. Testing eye tracker...")
try:
    from eye_tracking import EyeTracker

    tracker = EyeTracker(flip=True)
    print("   ✓ Eye tracker initialized")

    import numpy as np

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        _, gaze_info = tracker.process_frame(test_frame, only_compute=True)
        print(" Frame processing works")
    except Exception:
        print(" Processing warning (expected with blank frame)")

    tracker.release()

except Exception as e:
    print(f" Error: {e}")

print("\n10. Testing gaze collector...")
try:
    from gaze_collector import GazeCollector

    collector = GazeCollector()
    print("   ✓ Gaze collector initialized")

except Exception as e:
    print(f" Error: {e}")

print("\n11. Testing gaze classifier...")
try:
    from rorschach_interpreter import GazePatternClassifier

    classifier = GazePatternClassifier()

    mock_data = [
                    {'direction': 'center', 'saccade_speed': 0.02, 'smoothed_norm': [0, 0]}
                ] * 100

    pattern, metrics = classifier.classify_gaze_pattern(mock_data)
    print(f"   ✓ Classifier works")
    print(f"   Test pattern: {pattern}")

except Exception as e:
    print(f"Error: {e}")
print("\n12. Testing LLM client...")
try:
    from llm_client import LLMClient

    client = LLMClient()
    print("LLM client initialized")

except Exception as e:
    print(f"Error: {e}")

print("\n13. Testing prompt builder...")
try:
    from prompt_builder import PromptBuilder

    builder = PromptBuilder()
    test_prompt = builder.build_poem_prompt(
        "focused_center",
        {'avg_saccade_speed': 0.02},
        ["A cathedral in mist"],
        "mysterious"
    )
    print("   ✓ Prompt builder works")

except Exception as e:
    print(f"Error: {e}")

print("\n14. Testing TTS generator...")
try:
    from tts_generator import TTSGenerator

    tts = TTSGenerator()
    if tts.available:
        print("TTS available")
    else:
        print("TTS not available (missing dependencies)")

except Exception as e:
    print(f"TTS error: {e}")

# Test print formatter
print("\n15. Testing print formatter...")
try:
    from print_formatter import PrintFormatter

    formatter = PrintFormatter()
    print("Print formatter initialized")

except Exception as e:
    print(f"Error: {e}")

print("\n16. Testing data exporter...")
try:
    from data_exporter import DataExporter

    exporter = DataExporter()
    print("Data exporter initialized")

except Exception as e:
    print(f"Error: {e}")

print("\n17. Testing poem orchestrator...")
try:
    from poem_orchestrator import PoemOrchestrator

    orchestrator = PoemOrchestrator()
    print("   ✓ Poem orchestrator initialized")

except Exception as e:
    print(f"Error: {e}")

# Test eye visualizer
print("\n18. Testing eye visualizer...")
try:
    from eye_visualizer import EyeVisualizer

    visualizer = EyeVisualizer()
    print("   ✓ Eye visualizer initialized")
    visualizer.release()

except Exception as e:
    print(f"Error: {e}")

output_dirs = ['audio_output', 'print_queue', 'shared', 'poem_logs']

for dir_name in output_dirs:
    try:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✓ {dir_name}/")
    except Exception as e:
        print(f"Could not create {dir_name}/")


print(f"\n✓ All {len(project_files)} files present")
print(f"✓ Rorschach data: {len(interpreter.data)} interpretations")
print(f"✓ Collection: {COLLECTION_SETTINGS['duration_seconds']}s")

print("\nNext steps:")
print("  python visual_display.py        # Test visual displays")
print("  python main_pipeline.py test    # Test single poem")
print("  python main_pipeline.py         # Run continuous mode")
print("  python run_system.py            # Run complete system")
