import time
import cv2
import numpy as np
from utils import save_calibration

def run_automatic_calibration(tracker, cap, samples=30, overlay=True):
    """
    Calibrates the *eye gaze offset* (self.smoothed).
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
            
        # --- MODIFIED: Accept 4 return values ---
        # We only care about annotated (for display) and raw_eye_offset (for data)
        annotated, _, _, raw_eye_offset = tracker.process_frame(frame, only_compute=True)
        
        # --- MODIFIED: Use raw_eye_offset ---
        if raw_eye_offset is not None:
            # Convert [0, 0] centered offset to [0.5, 0.5] centered
            # This maintains consistency with the original calibration logic
            collected.append(raw_eye_offset + 0.5)
        
        # (Rest of display logic is unchanged)
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
        # (Rest of calculation logic is unchanged)
        arr = np.array(collected)
        mean = arr.mean(axis=0)
        stds = arr.std(axis=0)
        
        # This center value isn't used, but the stds are
        center_offset_from_0_5 = mean - 0.5
        print(f"Mean eye offset (from center): {center_offset_from_0_5}")

        horiz_thr = max(0.035, stds[0] * 3.0)
        vert_thr = max(0.028, stds[1] * 3.0)
        deadzone = max(0.018, np.mean(stds) * 2.0)
        
        calibration_data = {
            'center': mean.tolist(), # This is the [0,1] range mean
            'horiz_threshold': float(horiz_thr),
            'vert_threshold': float(vert_thr),
            'deadzone': float(deadzone),
            'timestamp': time.time()
        }
        
        print(f"Calibration complete: Center at {mean}")
        print(f"Thresholds - Horiz: {horiz_thr:.4f}, Vert: {vert_thr:.4f}, Deadzone: {deadzone:.4f}")
        
        save_calibration(calibration_data)
        
        tracker.horiz_thr = horiz_thr
        tracker.vert_thr = vert_thr
        tracker.deadzone = deadzone
        
        return calibration_data
    
    print("Calibration failed — no face detected. Try again with better lighting.")
    return None