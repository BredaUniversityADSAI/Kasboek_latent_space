import cv2
import filters
from random_filter_chooser import FilterChooser
import sys

print("\n" + "="*60)
print("JETSON WEBCAM ARTISTIC FILTERS")
print("="*60)

# Initialize MediaPipe
print("\nInitializing MediaPipe...")
print("✓ MediaPipe loaded successfully!\n")

# Try to find and open webcam
print("Searching for webcam...")
cap = None
for camera_idx in range(10):
    print(f"Trying camera index {camera_idx}...")
    test_cap = cv2.VideoCapture(camera_idx)
    if test_cap.isOpened():
        ret, test_frame = test_cap.read()
        if ret:
            print(f"✓ Found working camera at index {camera_idx}!")
            cap = test_cap
            break
        else:
            test_cap.release()
    else:
        test_cap.release()

if cap is None:
    print("\nERROR: No working webcam found!")
    print("Please check:")
    print("  - Camera is connected")
    print("  - Camera permissions are granted")
    print("  - No other application is using the camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Print available filters
print("\n" + "="*60)
print("FILTERS:")
filter_names = filters.get_filter_names()
for num, name in filter_names.items():
    print(f"  {num} - {name}")

print("\n  Press 'r' - Reset all filters")
print("  Press 'a' - Toggle AI mode (auto filter selection)")
print("  Press ESC - Quit")
print("="*60)
print("\nWebcam active. Toggle filters with number keys!\n")

# Initialize filter state
active_filters = {i: False for i in range(1, 20)}

# Initialize AI filter chooser
ai_mode = False
filter_chooser = FilterChooser()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        # Process with MediaPipe
        mediapipe_results = filters.process_frame_with_mediapipe(frame)
        
        # AI mode: let the LLM choose filters
        if ai_mode:
            recommended_filters = filter_chooser.get_filters_for_frame(frame, filter_names)
            # Apply recommended filters
            for i in range(1, 20):
                active_filters[i] = i in recommended_filters
        
        # Apply all active filters
        processed = filters.apply_filters(frame, active_filters, mediapipe_results)
        
        # Display active filters on screen
        y_pos = 30
        if ai_mode:
            cv2.putText(processed, "AI MODE: ON", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
        
        for filter_num, active in active_filters.items():
            if active:
                filter_name = f"Filter {filter_num}: ON"
                cv2.putText(processed, filter_name, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos += 20
        
        cv2.imshow('Artistic Filters', processed)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle filters with number keys 1-9
        if ord('1') <= key <= ord('9'):
            filter_num = key - ord('0')
            if filter_num in active_filters:
                active_filters[filter_num] = not active_filters[filter_num]
                status = "ON" if active_filters[filter_num] else "OFF"
                print(f"Filter {filter_num}: {status}")
        
        # Handle keys for filters 10-19 (need special handling)
        # For now, using 'q' + number (example: q+1 for filter 10)
        # You can customize this key mapping as needed
        
        # Toggle AI mode
        elif key == ord('a'):
            ai_mode = not ai_mode
            status = "ON" if ai_mode else "OFF"
            print(f"AI Mode: {status}")
            if not ai_mode:
                # Reset filters when leaving AI mode
                for i in range(1, 20):
                    active_filters[i] = False
        
        # Reset all filters
        elif key == ord('r'):
            for i in range(1, 20):
                active_filters[i] = False
            print("All filters reset")
        
        # Quit (ESC key)
        elif key == 27:
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    filters.cleanup_mediapipe()
    filter_chooser.cleanup()
    print("\n✓ Program closed successfully")