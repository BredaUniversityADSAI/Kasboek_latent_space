import cv2
import time
import numpy as np
import os
from tracking import EyeTracker
from calibration import run_automatic_calibration
from drawing import GazePainter
from utils import load_calibration, ensure_directories

# State constants
STATE_TRACKING = 1
STATE_CALIBRATING = 2
STATE_DRAWING = 3

# Configuration flags
DRAWING_IN_SEPARATE_WINDOW = True
CALIBRATION_IS_OVERLAY = True

def print_controls():
    """Print control instructions"""
    print("\n" + "="*60)
    print("EYE TRACKER CONTROLS:")
    print("="*60)
    print("c - Run calibration (look at center)")
    print("d - Toggle drawing mode")
    print("v - Toggle visual overlays (tracking visualization)")
    print("r - Reset (clear canvas in drawing mode, reset calibration otherwise)")
    print("s - Save drawing (in drawing mode)")
    print("f - Toggle frame flip (mirror mode)")
    print("q - Quit")
    print("="*60 + "\n")

def main():
    # Ensure directories exist
    ensure_directories()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Check if camera is connected.")
        return
    
    # Set camera resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize components
    tracker = EyeTracker(flip=True)
    painter = GazePainter(width=1280, height=720)
    
    # Load existing calibration if available
    calibration = load_calibration()
    if calibration:
        print("Loaded existing calibration.")
        # Apply loaded calibration to tracker
        tracker.horiz_thr = calibration.get('horiz_threshold', 0.035)
        tracker.vert_thr = calibration.get('vert_threshold', 0.028)
        tracker.deadzone = calibration.get('deadzone', 0.018)
    else:
        print("No calibration found. Press 'c' to calibrate.")
    
    # State variables
    current_state = STATE_TRACKING
    last_time = time.time()
    frame_count = 0
    fps = 0
    
    print_controls()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Calculate time delta
            now = time.time()
            dt = now - last_time
            last_time = now
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / dt / 30
                frame_count = 0
            
            # Process frame with tracker
            annotated, gaze_norm = tracker.process_frame(frame, only_compute=(current_state == STATE_CALIBRATING))
            
            # Handle different states
            if current_state == STATE_TRACKING:
                # Tracking mode - show annotated frame with tracking info
                display = annotated.copy()
                
                # Add mode and status info
                cv2.putText(display, 'MODE: TRACKING', (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f'FPS: {fps:.1f}', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if calibration:
                    cv2.putText(display, 'Calibrated', (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(display, 'Not calibrated (press c)', (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                
                cv2.imshow('Eye Tracker', display)
            
            elif current_state == STATE_CALIBRATING:
                # Calibration is handled in the function
                pass
            
            elif current_state == STATE_DRAWING:
                # Drawing mode - update painter and show canvas
                if gaze_norm is not None:
                    painter.update(gaze_norm, dt)
                
                if DRAWING_IN_SEPARATE_WINDOW:
                    # Show tracking in main window
                    display = annotated.copy()
                    cv2.putText(display, 'MODE: DRAWING', (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.putText(display, 'Press s to save, r to clear', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow('Eye Tracker', display)
                    
                    # Show canvas in separate window with cursor
                    canvas_display = painter.get_canvas_with_cursor(gaze_norm)
                    cv2.imshow('Gaze Canvas', canvas_display)
                else:
                    # Overlay mode
                    blended = painter.get_display(annotated, overlay=True)
                    cv2.putText(blended, 'MODE: DRAWING (Overlay)', (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.imshow('Eye Tracker', blended)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            
            elif key == ord('c'):
                print("\nStarting calibration...")
                current_state = STATE_CALIBRATING
                
                # Run calibration (this blocks until complete)
                new_calib = run_automatic_calibration(tracker, cap, samples=30, overlay=CALIBRATION_IS_OVERLAY)
                
                if new_calib:
                    calibration = new_calib
                    print("Calibration successful!")
                else:
                    print("Calibration failed. Please try again.")
                
                current_state = STATE_TRACKING
            
            elif key == ord('d'):
                if current_state == STATE_DRAWING:
                    # Exit drawing mode
                    current_state = STATE_TRACKING
                    if DRAWING_IN_SEPARATE_WINDOW:
                        cv2.destroyWindow('Gaze Canvas')
                    print("Exited drawing mode")
                else:
                    # Enter drawing mode
                    current_state = STATE_DRAWING
                    if DRAWING_IN_SEPARATE_WINDOW:
                        cv2.namedWindow('Gaze Canvas', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Gaze Canvas', 640, 360)
                    print("Entered drawing mode")
            
            elif key == ord('v'):
                # Toggle visualization
                show = tracker.toggle_visuals()
                print(f"Tracking visuals: {'ON' if show else 'OFF'}")
            
            elif key == ord('f'):
                # Toggle frame flip
                flip = tracker.toggle_flip()
                print(f"Frame flip (mirror): {'ON' if flip else 'OFF'}")
            
            elif key == ord('r'):
                if current_state == STATE_DRAWING:
                    # Clear canvas
                    painter.clear()
                    print("Canvas cleared")
                else:
                    # Reset calibration
                    calib_path = './calibrations/calibration.json'
                    if os.path.exists(calib_path):
                        os.remove(calib_path)
                        calibration = None
                        # Reset tracker thresholds to defaults
                        tracker.horiz_thr = 0.035
                        tracker.vert_thr = 0.028
                        tracker.deadzone = 0.018
                        print("Calibration reset to defaults")
                    else:
                        print("No calibration to reset")
            
            elif key == ord('s'):
                if current_state == STATE_DRAWING:
                    # Save drawing with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'gaze_art_{timestamp}.png'
                    filepath = painter.save(filename)
                    print(f"Drawing saved: {filepath}")
                else:
                    print("Press 'd' to enter drawing mode first")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == '__main__':
    main()