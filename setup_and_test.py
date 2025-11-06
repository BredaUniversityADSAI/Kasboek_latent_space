# setup_and_test.py
"""
Complete setup and testing script for the poem generation system
Verifies all components before running the main system
"""

import sys
import os

print("\n" + "=" * 70)
print(" POEM GENERATION SYSTEM - SETUP & VERIFICATION")
print("=" * 70)

# 0. Check for config.py
print("\n0. Checking for config.py...")
if not os.path.exists('config.py'):
    print("    config.py NOT FOUND")
    print("\n   Please create config.py first!")
    print("   The configuration file should be in the same directory as this script.")
    sys.exit(1)
print("   ✓ config.py found")

# Import config
try:
    from config import API_CONFIG, PATHS, COLLECTION_SETTINGS, validate_config

    print("    config.py loaded successfully")
except Exception as e:
    print(f"    Error loading config.py: {e}")
    sys.exit(1)

# 1. Check Python version
print("\n1. Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major < 3:
    print("    ERROR: Python 3.x required")
    sys.exit(1)
elif python_version.major == 3 and python_version.minor < 8:
    print("    ERROR: Python 3.8+ required")
    sys.exit(1)
elif python_version.major == 3 and python_version.minor >= 12:
    print("    WARNING: Python 3.12+ may have compatibility issues with MediaPipe")
    print("   Recommended: Python 3.10 or 3.11")
    print("   Continuing anyway...")
else:
    print("   ✓ Python version OK")

# 2. Check required modules
print("\n2. Checking required modules...")
required_modules = {
    'cv2': 'opencv-python',
    'mediapipe': 'mediapipe',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'requests': 'requests',
    'openpyxl': 'openpyxl'
}

missing_modules = []
for module, package in required_modules.items():
    try:
        __import__(module)
        print(f"   ✓ {module}")
    except ImportError:
        print(f"    {module} (install with: uv add {package} or pip install {package})")
        missing_modules.append(package)

# Check optional TTS modules
optional_modules = {
    'torch': 'torch',
    'transformers': 'transformers',
    'parler_tts': 'parler-tts',
    'soundfile': 'soundfile'
}

print("\n   Optional modules (for TTS):")
for module, package in optional_modules.items():
    try:
        __import__(module)
        print(f"   ✓ {module}")
    except ImportError:
        print(f"   {module} (TTS won't work without this)")

if missing_modules:
    print(f"\n  Missing required modules. Install with:")
    print(f"   uv add {' '.join(missing_modules)}")
    print("\n   OR with pip:")
    print(f"   pip install {' '.join(missing_modules)}")
    sys.exit(1)

print("\n   ✓ All required modules installed")

# 3. Validate configuration
print("\n3. Validating configuration...")
if not validate_config():
    print("\n    Configuration invalid")
    sys.exit(1)

print("   ✓ Configuration valid")
print(f"   - User Key: {API_CONFIG['user_key'][:15]}...✓")
print(f"   - Assistant Key: {API_CONFIG['assistant_key'][:15]}...✓")

# 4. Check for required files
print("\n4. Checking required files...")
required_files = [
    PATHS["rorschach_excel"],
    'eye_tracking.py',
    'rorschach_interpreter.py',
    'poem_generation_pipeline.py',
    'config.py'
]

missing_files = []
for filename in required_files:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"   ✓ {filename} ({size:,} bytes)")
    else:
        print(f"   {filename} NOT FOUND")
        missing_files.append(filename)

if missing_files:
    print(f"\n Missing files:")
    for f in missing_files:
        print(f"   - {f}")
    print(f"\nCurrent directory: {os.getcwd()}")
    print("Please ensure all files are in the same directory")
    sys.exit(1)

print("\n   ✓ All required files present")

# 5. Test Rorschach data loading
print("\n5. Testing Rorschach data loading...")
try:
    from rorschach_interpreter import RorschachInterpreter

    interpreter = RorschachInterpreter(PATHS["rorschach_excel"])

    if interpreter.data is None or len(interpreter.data) == 0:
        print("   Failed to load Rorschach data")
        print("   Check Excel file format and contents")
        print("   The file should have paired columns: Category | Interpretation")
        sys.exit(1)

    print(f"   ✓ Loaded {len(interpreter.data)} interpretations")

    # Show category distribution
    if len(interpreter.data) > 0:
        categories = interpreter.data['category'].value_counts()
        print(f"   Top categories:")
        for cat, count in categories.head(3).items():
            print(f"     - {cat}: {count} entries")

    # Test category retrieval
    test_cat = interpreter.get_interpretation_by_category('psychological', n=1)
    print(f"   ✓ Category retrieval works")
    if test_cat and len(test_cat[0]) > 0:
        print(f"   Sample: {test_cat[0][:60]}...")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 6. Test webcam
print("\n6. Testing webcam access...")
try:
    import cv2

    cap = None
    working_index = None

    for camera_idx in range(5):
        test_cap = cv2.VideoCapture(camera_idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                print(f"   ✓ Found working camera at index {camera_idx}")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                working_index = camera_idx
                cap = test_cap
                break
            test_cap.release()
        else:
            test_cap.release()

    if cap is None:
        print("   No webcam found")
        print("   The system needs a webcam to function")
        print("   This might still work on Jetson - try running anyway")
        print("\n   Troubleshooting:")
        print("     - Check camera is connected")
        print("     - Check camera permissions")
        print("     - Ensure no other app is using camera")
    else:
        cap.release()
        print("   ✓ Webcam accessible")

except Exception as e:
    print(f"   Webcam test failed: {e}")
    print("   This might work on Jetson - continuing anyway")

# 7. Test LLM API connection
print("\n7. Testing LLM API connection...")
try:
    import requests
    import json

    init_endpoint = f"{API_CONFIG['api_url']}/actions/init_assistant"
    headers = {"Content-Type": "application/json"}
    data = {
        "user_key": API_CONFIG['user_key'],
        "assistant_key": API_CONFIG['assistant_key']
    }

    print("   Sending test request to API...")
    response = requests.post(init_endpoint,
                             data=json.dumps(data),
                             headers=headers,
                             timeout=10)

    result = response.json()

    if result.get('status') == 'Failed':
        print(f"   API initialization failed: {result.get('error')}")
        print("   Check your API keys in config.py")
        print("   Make sure you're using keys from https://ai-assistants.buas.nl/")
    elif 'chat_key' in result:
        print(f"   ✓ LLM API connection successful")
        print(f"   Chat key received: {result['chat_key'][:20]}...")
    else:
        print(f"   Unexpected response from API")
        print(f"   Response: {result}")

except requests.Timeout:
    print("   API connection timeout")
    print("   Check internet connection")
except Exception as e:
    print(f"   API test failed: {e}")
    print("   System will use fallback poems if API fails")

# 8. Test eye tracking
print("\n8. Testing eye tracking module...")
try:
    from eye_tracking import EyeTracker

    tracker = EyeTracker(flip=True)
    print("   ✓ Eye tracker initialized")

    # Try processing a test frame
    import numpy as np

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        _, gaze_info = tracker.process_frame(test_frame, only_compute=True)
        print("   ✓ Frame processing works")
    except Exception as e:
        print(f"   Frame processing warning: {e}")
        print("   (This is normal with a blank test frame)")

    tracker.release()

except Exception as e:
    print(f"   Eye tracking test failed: {e}")
    import traceback

    traceback.print_exc()

# 9. Test gaze classification
print("\n9. Testing gaze classification...")
try:
    from rorschach_interpreter import GazePatternClassifier
    from config import CLASSIFICATION_THRESHOLDS

    classifier = GazePatternClassifier()
    classifier.thresholds = CLASSIFICATION_THRESHOLDS.copy()

    # Create mock data
    mock_data = [
                    {'direction': 'center', 'saccade_speed': 0.02, 'smoothed_norm': [0, 0]}
                ] * 100

    pattern, metrics = classifier.classify_gaze_pattern(mock_data)
    print(f"   ✓ Classification works")
    print(f"   Test pattern: {pattern}")
    print(
        f"   Test metrics: speed={metrics.get('avg_saccade_speed', 0):.3f}, center={metrics.get('center_time_ratio', 0):.1%}")

except Exception as e:
    print(f"   Classification test failed: {e}")
    import traceback

    traceback.print_exc()

# 10. Create output directories
print("\n10. Creating output directories...")
output_dirs = [
    'audio_output',
    'print_queue',
    'shared',
    'poem_logs'
]

for dir_name in output_dirs:
    try:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✓ {dir_name}/")
    except Exception as e:
        print(f"   ⚠ Could not create {dir_name}/: {e}")

# Summary
print("\n" + "=" * 70)
print(" SETUP VERIFICATION COMPLETE")
print("=" * 70)

print("\n✓ System is ready for testing!")
print("\nConfiguration Summary:")
print(f"  - Collection duration: {COLLECTION_SETTINGS['duration_seconds']}s")
print(f"  - Wait between poems: {COLLECTION_SETTINGS['wait_between_poems']}s")
print(f"  - Rorschach data: {len(interpreter.data)} interpretations")
print(f"  - Python version: {sys.version.split()[0]}")

print("\nNext steps:")
print("  1. Test single poem generation:")
print("     python poem_generation_pipeline.py test")
print("\n  2. Test visual displays:")
print("     python visual_display_system.py")
print("\n  3. Run complete system:")
print("     python run_complete_system.py")

print("\nTips:")
print("  - Ensure good lighting for eye tracking")
print("  - Position yourself 50-80cm from camera")
print("  - Look naturally at the camera during collection")
print("  - Press 'q' in any window to quit")

print("=" * 70 + "\n")