import time
import cv2
import numpy as np
from utils import save_calibration

def run_automatic_calibration(tracker, cap, samples=30, overlay=True):
    """
    Simplified calibration - only captures center gaze, no directional looks.
    Matches the style from the provided reference code.
    """
    collected = []
    print("Calibration: Please look at the center of the screen for ~2 seconds...")
    
    start_time = time.time()
    
    for i in range(samples):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            continue
            
        if tracker.flip:
            frame = cv2.flip(frame, 1)
            
        # Process frame to get gaze data
        annotated, gaze_norm = tracker.process_frame(frame, only_compute=True)
        
        if gaze_norm is not None:
            collected.append(gaze_norm)
        
        # Show calibration progress if overlay is enabled
        if overlay:
            display = annotated.copy() if annotated is not None else frame.copy()
            progress = int((i / samples) * 100)
            cv2.putText(display, f'CALIBRATING: {progress}%', (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, 'Look at center of screen', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Eye Tracker', display)
            
        cv2.waitKey(30)
    
    if collected:
        # Calculate mean and standard deviation for thresholds
        arr = np.array(collected)
        mean = arr.mean(axis=0)
        stds = arr.std(axis=0)
        
        # Set thresholds based on observed variance (similar to reference code)
        horiz_thr = max(0.035, stds[0] * 3.0)  # Default or 3x std
        vert_thr = max(0.028, stds[1] * 3.0)
        deadzone = max(0.018, np.mean(stds) * 2.0)
        
        calibration_data = {
            'center': mean.tolist(),
            'horiz_threshold': float(horiz_thr),
            'vert_threshold': float(vert_thr),
            'deadzone': float(deadzone),
            'timestamp': time.time()
        }
        
        print(f"Calibration complete: Center at {mean}")
        print(f"Thresholds - Horiz: {horiz_thr:.4f}, Vert: {vert_thr:.4f}, Deadzone: {deadzone:.4f}")
        
        # Save calibration to file
        save_calibration(calibration_data)
        
        # Update tracker thresholds
        tracker.horiz_thr = horiz_thr
        tracker.vert_thr = vert_thr
        tracker.deadzone = deadzone
        
        return calibration_data
    
    print("Calibration failed — no face detected. Try again with better lighting.")
    return None