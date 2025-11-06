"""
Configuration file for the Complete Poem Generation System
UPDATE YOUR KEYS IN THE API_CONFIG SECTION BELOW
"""

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIG = {
    "api_url": "https://ai-assistants.buas.nl/aioda-api",
    "user_key": "0494011c-6e14-4b34-ae66-6fe25b65f0ba",
    "assistant_key": "8c1cdd40-14e7-460b-acaf-0f874f50703e",
}

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

# Gaze collection settings
COLLECTION_SETTINGS = {
    "duration_seconds": 15.0,  # How long to observe viewer (15 seconds)
    "wait_between_poems": 3.0,  # Pause between poem generations
    "fps_estimate": 15,  # Expected FPS on Jetson
}

# Eye tracking settings
EYE_TRACKING = {
    "flip_camera": True,  # Mirror the camera feed
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# Gaze pattern classification thresholds
# Adjust these if patterns always classify the same way
CLASSIFICATION_THRESHOLDS = {
    "saccade_speed_high": 0.25,  # Higher = needs faster movement to trigger
    "saccade_speed_low": 0.08,  # Lower = counts as slow
    "center_bias_high": 0.65,  # % of time in center for "focused"
    "center_bias_low": 0.30,  # Below this = peripheral
    "direction_entropy_high": 2.5,  # Higher = more random
    "fixation_duration_long": 0.8,  # Seconds for long fixation
}

# ============================================================================
# FILE PATHS
# ============================================================================

PATHS = {
    "rorschach_excel": "Rorschach_Interpretations_English.xlsx",
    "poem_logs": "poem_logs",
    "print_queue": "print_queue",
    "shared_data": "shared",
    "audio_output": "audio_output",
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

OUTPUT = {
    "save_logs": True,  # Save poem generation logs
    "print_to_console": True,  # Display poems in terminal
    "save_for_printing": True,  # Save poems to print queue
    "include_metadata": True,  # Include pattern info in print
    "generate_audio": True,  # Generate TTS audio
    "save_display_data": True,  # Save data for visual displays
}

# ============================================================================
# POEM GENERATION SETTINGS
# ============================================================================

POEM_SETTINGS = {
    "reinit_conversation_every": 5,  # Reinitialize LLM every N poems for variety
    "api_timeout": 45,  # Seconds to wait for API response
    "use_fallback_on_error": True,  # Use backup poems if API fails
    "min_poem_length": 20,  # Minimum characters for valid poem
}

# ============================================================================
# DEBUGGING
# ============================================================================

DEBUG = {
    "verbose": True,  # Print detailed logs
    "save_gaze_data": False,  # Save raw gaze data (uses lots of space)
    "test_mode": False,  # Use shorter collection time for testing
}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Check if configuration is valid before running
    """
    errors = []


    # Check file paths
    import os
    if not os.path.exists(PATHS["rorschach_excel"]):
        errors.append(f" Rorschach Excel file not found: {PATHS['rorschach_excel']}")
        errors.append(f"   Current directory: {os.getcwd()}")

    if errors:
        print("\n" + "=" * 70)
        print("CONFIGURATION ERRORS")
        print("=" * 70)
        for error in errors:
            print(error)
        print("\nPlease update config.py with your API keys and ensure files are present")
        print("=" * 70 + "\n")
        return False

    return True


if __name__ == "__main__":
    # Test configuration
    print("\n" + "=" * 70)
    print("TESTING CONFIGURATION")
    print("=" * 70)

    print("\nAPI Configuration:")
    print(f"  URL: {API_CONFIG['api_url']}")

    user_key = API_CONFIG['user_key']
    if user_key != "PASTE_YOUR_USER_KEY_HERE":
        print(f"  User Key: {user_key[:20]}...✓")
    else:
        print(f"  User Key: NOT SET ")

    assistant_key = API_CONFIG['assistant_key']
    if assistant_key != "PASTE_YOUR_ASSISTANT_KEY_HERE":
        print(f"  Assistant Key: {assistant_key[:20]}...✓")
    else:
        print(f"  Assistant Key: NOT SET ")

    print("\nCollection Settings:")
    print(f"  Duration: {COLLECTION_SETTINGS['duration_seconds']}s")
    print(f"  Wait between: {COLLECTION_SETTINGS['wait_between_poems']}s")

    print("\nFile Paths:")
    import os

    for name, path in PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")

    print("\n" + "=" * 70)
    if validate_config():
        print("✓ CONFIGURATION VALID")
        print("\nYou're ready to run:")
        print("  python run_complete_system.py")
    else:
        print("✗ CONFIGURATION INVALID")
        print("\nPlease fix the errors above")

    print("=" * 70 + "\n")
