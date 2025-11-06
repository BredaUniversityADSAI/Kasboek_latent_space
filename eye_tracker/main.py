import cv2
import time
import numpy as np
import os
from tracking import EyeTracker
from calibration import run_automatic_calibration
from drawing import GazePainter
from utils import load_calibration, ensure_directories

# (Constants are unchanged)
STATE_TRACKING = 1
STATE_CALIBRATING = 2
STATE_DRAWING = 3
DRAWING_IN_SEPARATE_WINDOW = True
CALIBRATION_IS_OVERLAY = True
NO_GAZE_EXIT_SEC = 1.0
INACTIVITY_EXIT_SEC = 10.0

# --- NEW: Display Toggle States ---
display_toggles = {
    "tracking_rects": True, # 'v' - All visual trackers
    "debug_text": True,     # 't' - Saccades, direction, EAR/MAR
    "misc_text": True,      # 'm' - FPS, Mode, Pen status
    "hint_text": True,      # 'h' - All "(press 'x' to toggle)" hints
}

def print_controls():
    """Print control instructions"""
    print("\n" + "="*60)
    print("EYE TRACKER CONTROLS:")
    print("="*60)
    print("--- Gestures ---")
    print(f"Long Blink (>0.5s) : Toggle Drawing Mode (like 'd')")
    print(f"Open Mouth (Quickly): Toggle Pen (Up/Down)")
    print(f"Look Away (>1s)     : Auto-save and exit drawing mode")
    print(f"Pen Up (>10s)       : Auto-save and exit drawing mode")
    print("\n--- Keyboard (Click 'Eye Tracker' window to use) ---")
    print("c - Run calibration (look at center/camera)")
    print("d - Toggle drawing mode")
    print("r - Reset (clear canvas in drawing mode, reset calibration otherwise)")
    print("s - Save drawing (in drawing mode)")
    print("f - Toggle frame flip (mirror mode)")
    print("q - Quit")
    print("\n--- Display Toggles ---")
    print("v - Toggle Tracking Visuals (rects, arrows, etc.)")
    print("t - Toggle Debug Text (direction, saccades, EAR/MAR)")
    print("m - Toggle Misc Text (FPS, mode, pen status)")
    print("h - Toggle Usage Hints")
    print("="*60 + "\n")

# (save_drawing and exit_drawing_mode are unchanged)
def save_drawing(painter):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'gaze_art_{timestamp}.png'
    filepath = painter.save(filename)
    if filepath:
        print(f"Drawing saved: {filepath}")
    return filepath

def exit_drawing_mode(painter, reason=""):
    print(f"Exiting drawing mode. {reason}")
    save_drawing(painter)
    painter.clear()
    if DRAWING_IN_SEPARATE_WINDOW:
        try:
            cv2.destroyWindow('Gaze Canvas')
        except cv2.error:
            pass
    return STATE_TRACKING


def main():
    # (Initialization is unchanged)
    ensure_directories()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Check if camera is connected.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = EyeTracker(flip=True)
    painter = GazePainter(width=1280, height=720)
    calibration = load_calibration()
    if calibration:
        print("Loaded existing calibration.")
        tracker.horiz_thr = calibration.get('horiz_threshold', 0.035)
        tracker.vert_thr = calibration.get('vert_threshold', 0.028)
        tracker.deadzone = calibration.get('deadzone', 0.018)
    else:
        print("No calibration found. Press 'c' to calibrate.")
    
    current_state = STATE_TRACKING
    last_time = time.time()
    frame_count = 0
    fps = 0
    no_gaze_timer = 0.0
    inactivity_timer = 0.0
    
    print_controls()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            now = time.time()
            dt = now - last_time
            last_time = now
            
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / dt / 30
                frame_count = 0
            
            # --- MODIFIED: Pass toggles to tracker ---
            annotated, gaze_norm, gesture_event, _ = tracker.process_frame(
                frame, display_toggles, only_compute=(current_state == STATE_CALIBRATING)
            )
            
            if current_state == STATE_TRACKING:
                display = annotated.copy()
                
                # --- MODIFIED: Conditional Text ---
                if display_toggles["misc_text"]:
                    cv2.putText(display, 'MODE: TRACKING', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (229, 24, 24), 2)
                    cv2.putText(display, f'FPS: {fps:.1f}', (10, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    if calibration:
                        cv2.putText(display, 'Calibrated', (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (229, 24, 24), 1)
                    else:
                        cv2.putText(display, 'Not calibrated (press c)', (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                
                cv2.imshow('Eye Tracker', display)
            
            elif current_state == STATE_CALIBRATING:
                pass 
            
            elif current_state == STATE_DRAWING:
                # (Auto-stop logic is unchanged)
                if gaze_norm is None:
                    no_gaze_timer += dt
                    if no_gaze_timer > NO_GAZE_EXIT_SEC:
                        current_state = exit_drawing_mode(painter, reason="No face detected.")
                        continue 
                else:
                    no_gaze_timer = 0.0
                    painter.update(gaze_norm, dt)
                
                if not painter.pen_down:
                    inactivity_timer += dt
                    if inactivity_timer > INACTIVITY_EXIT_SEC:
                        current_state = exit_drawing_mode(painter, reason=f"Inactivity limit ({INACTIVITY_EXIT_SEC}s) reached.")
                        continue
                else:
                    inactivity_timer = 0.0
                
                # --- MODIFIED: Conditional Text ---
                if DRAWING_IN_SEPARATE_WINDOW:
                    display = annotated.copy()
                    
                    if display_toggles["misc_text"]:
                        cv2.putText(display, 'MODE: DRAWING', (10, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        
                        if not painter.pen_down:
                            time_left = INACTIVITY_EXIT_SEC - inactivity_timer
                            cv2.putText(display, f'Stop in: {time_left:.1f}s', (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                        else:
                            cv2.putText(display, 'Drawing...', (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        
                        pen_status = "DOWN (Drawing)" if painter.pen_down else "UP (Open mouth to toggle)"
                        pen_color = (0, 0, 255) if painter.pen_down else (0, 255, 0)
                        cv2.putText(display, f'Pen: {pen_status}', (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pen_color, 1)
                                   
                    cv2.imshow('Eye Tracker', display)
                    
                    canvas_display = painter.get_canvas_with_cursor(gaze_norm)
                    cv2.imshow('Gaze Canvas', canvas_display)
                else:
                    # (Overlay mode)
                    blended = painter.get_display(annotated, overlay=True)
                    if display_toggles["misc_text"]:
                        cv2.putText(blended, 'MODE: DRAWING (Overlay)', (10, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.imshow('Eye Tracker', blended)
            
            # (Gesture handling is unchanged)
            if gesture_event == 'long_blink':
                print("Gesture: Long Blink detected!")
                if current_state == STATE_DRAWING:
                    current_state = exit_drawing_mode(painter, reason="Gesture toggle.")
                else:
                    current_state = STATE_DRAWING
                    no_gaze_timer = 0.0
                    inactivity_timer = 0.0 
                    painter.clear() 
                    painter.set_pen(False)
                    if DRAWING_IN_SEPARATE_WINDOW:
                        cv2.namedWindow('Gaze Canvas', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Gaze Canvas', 640, 360)
                    print("Entered drawing mode")

            elif gesture_event == 'mouth_toggle':
                if current_state == STATE_DRAWING:
                    current_pen_state = painter.pen_down
                    painter.set_pen(not current_pen_state)
            
            # --- Keyboard input ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            
            elif key == ord('c'):
                # (Unchanged)
                print("\nStarting calibration...")
                current_state = STATE_CALIBRATING
                new_calib = run_automatic_calibration(tracker, cap, samples=30, overlay=CALIBRATION_IS_OVERLAY)
                if new_calib:
                    calibration = new_calib
                    print("Calibration successful!")
                else:
                    print("Calibration failed. Please try again.")
                current_state = STATE_TRACKING
            
            elif key == ord('d'):
                # (Unchanged)
                if current_state == STATE_DRAWING:
                    current_state = exit_drawing_mode(painter, reason="Key 'd' pressed.")
                else:
                    current_state = STATE_DRAWING
                    no_gaze_timer = 0.0
                    inactivity_timer = 0.0
                    painter.clear()
                    painter.set_pen(False)
                    if DRAWING_IN_SEPARATE_WINDOW:
                        cv2.namedWindow('Gaze Canvas', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Gaze Canvas', 640, 360)
                    print("Entered drawing mode")
            
            # --- MODIFIED: New Toggle Keys ---
            elif key == ord('v'):
                display_toggles["tracking_rects"] = not display_toggles["tracking_rects"]
                print(f"Tracking Visuals: {'ON' if display_toggles['tracking_rects'] else 'OFF'}")
            
            elif key == ord('t'):
                display_toggles["debug_text"] = not display_toggles["debug_text"]
                print(f"Debug Text: {'ON' if display_toggles['debug_text'] else 'OFF'}")

            elif key == ord('m'):
                display_toggles["misc_text"] = not display_toggles["misc_text"]
                print(f"Misc Text: {'ON' if display_toggles['misc_text'] else 'OFF'}")

            elif key == ord('h'):
                display_toggles["hint_text"] = not display_toggles["hint_text"]
                print(f"Usage Hints: {'ON' if display_toggles['hint_text'] else 'OFF'}")
            # --- END MODIFIED ---
            
            elif key == ord('f'):
                # (Unchanged)
                flip = tracker.toggle_flip()
                print(f"Frame flip (mirror): {'ON' if flip else 'OFF'}")
            
            elif key == ord('r'):
                # (Unchanged)
                if current_state == STATE_DRAWING:
                    painter.clear()
                    print("Canvas cleared")
                else:
                    calib_path = './calibrations/calibration.json'
                    if os.path.exists(calib_path):
                        os.remove(calib_path)
                        calibration = None
                        tracker.horiz_thr = 0.035
                        tracker.vert_thr = 0.028
                        tracker.deadzone = 0.018
                        print("Calibration reset to defaults")
                    else:
                        print("No calibration to reset")
            
            elif key == ord('s'):
                # (Unchanged)
                if current_state == STATE_DRAWING:
                    save_drawing(painter)
                else:
                    print("Press 'd' to enter drawing mode first")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # (Cleanup is unchanged)
        print("Cleaning up...")
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == '__main__':
    main()