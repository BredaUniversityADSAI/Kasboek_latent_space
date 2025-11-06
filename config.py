"""
Configuration file for the Complete Poem Generation System
UPDATE YOUR KEYS IN THE API_CONFIG SECTION BELOW
"""


API_CONFIG = {
    "api_url": "https://ai-assistants.buas.nl/aioda-api",
    "user_key": "0494011c-6e14-4b34-ae66-6fe25b65f0ba",
    "assistant_key": "f6277ecd-02ab-423c-b7d1-9bcdf8e56930",
}


COLLECTION_SETTINGS = {
    "duration_seconds": 15.0,  # How long to observe viewer (15 seconds)
    "wait_between_poems": 3.0,  # Pause between poem generations
    "fps_estimate": 15,  # Expected FPS on Jetson
}

EYE_TRACKING = {
    "flip_camera": True,  # Mirror the camera feed
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

CLASSIFICATION_THRESHOLDS = {
    "saccade_speed_high": 0.25,  # Higher = needs faster movement to trigger
    "saccade_speed_low": 0.08,  # Lower = counts as slow
    "center_bias_high": 0.65,  # % of time in center for "focused"
    "center_bias_low": 0.30,  # Below this = peripheral
    "direction_entropy_high": 2.5,  # Higher = more random
    "fixation_duration_long": 0.8,  # Seconds for long fixation
}

PATHS = {
    "rorschach_excel": "Rorschach_Interpretations_English.xlsx",
    "poem_logs": "poem_logs",
    "print_queue": "print_queue",
    "shared_data": "shared",
    "audio_output": "audio_output",
}

OUTPUT = {
    "save_logs": True,  # Save poem generation logs
    "print_to_console": True,  # Display poems in terminal
    "save_for_printing": True,  # Save poems to print queue
    "include_metadata": True,  # Include pattern info in print
    "generate_audio": True,  # Generate TTS audio
    "save_display_data": True,  # Save data for visual displays
}

POEM_SETTINGS = {
    "reinit_conversation_every": 5,
    "api_timeout": 90,  # ← Change from 45 to 90 seconds
    "use_fallback_on_error": True,
    "min_poem_length": 20,
}


DEBUG = {
    "verbose": True,  # Print detailed logs
    "save_gaze_data": False,  # Save raw gaze data (uses lots of space)
    "test_mode": False,  # Use shorter collection time for testing
}



def validate_config():
    """
    Check if configuration is valid before running
    """
    errors = []


    import os
    if not os.path.exists(PATHS["rorschach_excel"]):
        errors.append(f" Rorschach Excel file not found: {PATHS['rorschach_excel']}")
        errors.append(f"   Current directory: {os.getcwd()}")

    if errors:
        for error in errors:
            print(error)
        print("\n update config.py ")
        print("=" * 70 + "\n")
        return False

    return True


if __name__ == "__main__":
    print("\nCollection Settings:")
    print(f"  Duration: {COLLECTION_SETTINGS['duration_seconds']}s")
    print(f"  Wait between: {COLLECTION_SETTINGS['wait_between_poems']}s")

    print("\nFile Paths:")
    import os

    for name, path in PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")

    if validate_config():
        print("CONFIGURATION VALID")
        print("\nready to run:")
    else:
        print("✗ CONFIGURATION INVALID")
    print("=" * 70 + "\n")