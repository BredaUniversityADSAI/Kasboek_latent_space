import cv2
import numpy as np
import mediapipe as mp
import sys

print("\n" + "="*60)
print("JETSON ADVANCED EYE TRACKING SYSTEM")
print("="*60)

# Initialize MediaPipe Face Mesh for eye region detection
print("\nInitializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✓ MediaPipe loaded successfully!\n")

# Eye landmark indices - more complete eye region
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Simpler indices for eye corners
LEFT_EYE_CORNERS = [33, 133]  # outer, inner corner
RIGHT_EYE_CORNERS = [362, 263]  # outer, inner corner

def detect_pupil(eye_frame):
    """
    Detect pupil using multiple methods
    Returns pupil center relative to eye frame and debug info
    """
    if eye_frame is None or eye_frame.size == 0:
        return None, None, "Empty frame"
    
    h, w = eye_frame.shape[:2]
    if h < 10 or w < 10:
        return None, None, "Frame too small"
    
    # Convert to grayscale
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Adaptive thresholding (works better in varied lighting)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Method 2: Try simple thresholding with lower threshold
        _, thresh2 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thresh = thresh2
    
    if contours:
        # Filter contours by area and circularity
        valid_contours = []
        eye_area = h * w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Check if area is reasonable (1-50% of eye region)
            if not (0.01 * eye_area < area < 0.5 * eye_area):
                continue
            
            # Check circularity (4*pi*area/perimeter^2, circle = 1.0)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Reasonably circular
                    valid_contours.append((contour, area))
        
        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            
            # Get center using moments
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), thresh, "Success"
    
    # Method 3: Fallback - find darkest region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
    if min_val < 50:  # If there's a dark enough spot
        return min_loc, thresh, "Darkest point"
    
    return None, thresh, "No pupil found"

def extract_eye_region(frame, landmarks, eye_indices, padding=5):
    """
    Extract eye region from frame with padding
    Returns eye region image and bounding box coordinates
    """
    h, w = frame.shape[:2]
    
    # Get eye points
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    
    # Get bounding box
    x_min = int(np.min(points[:, 0])) - padding
    x_max = int(np.max(points[:, 0])) + padding
    y_min = int(np.min(points[:, 1])) - padding
    y_max = int(np.max(points[:, 1])) + padding
    
    # Ensure within frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    # Check if valid region
    if x_max <= x_min or y_max <= y_min:
        return None, None
    
    # Extract region
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    if eye_region.size == 0:
        return None, None
    
    return eye_region, (x_min, y_min, x_max, y_max)

def calculate_gaze_ratio(pupil_center, eye_region_shape):
    """
    Calculate horizontal and vertical gaze ratios
    0.5 = center, <0.5 = left/up, >0.5 = right/down
    """
    if pupil_center is None:
        return None, None
    
    eye_width = eye_region_shape[1]
    eye_height = eye_region_shape[0]
    
    if eye_width == 0 or eye_height == 0:
        return None, None
    
    horizontal_ratio = pupil_center[0] / eye_width
    vertical_ratio = pupil_center[1] / eye_height
    
    return horizontal_ratio, vertical_ratio

def get_gaze_direction(h_ratio, v_ratio, h_threshold=0.15, v_threshold=0.15):
    """
    Convert gaze ratios to readable direction
    """
    if h_ratio is None or v_ratio is None:
        return "UNKNOWN"
    
    direction = ""
    
    # Vertical direction
    if v_ratio < 0.5 - v_threshold:
        direction += "UP "
    elif v_ratio > 0.5 + v_threshold:
        direction += "DOWN "
    
    # Horizontal direction
    if h_ratio < 0.5 - h_threshold:
        direction += "LEFT"
    elif h_ratio > 0.5 + h_threshold:
        direction += "RIGHT"
    
    return direction.strip() if direction else "CENTER"

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
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("CONTROLS:")
print("  Press 'd' - Toggle debug view (shows eye processing)")
print("  Press 'c' - Toggle calibration info")
print("  Press ESC - Quit")
print("="*60)
print("\nEye tracking active! Look around to test it.\n")
print("TIP: Make sure you have good lighting on your face!")

# Configuration
show_debug = False
show_calibration = False
debug_scale = 4  # Scale factor for debug windows

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        left_pupil = None
        right_pupil = None
        left_bbox = None
        right_bbox = None
        left_threshold = None
        right_threshold = None
        left_status = ""
        right_status = ""
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            # Extract eye regions
            left_eye_region, left_bbox = extract_eye_region(frame, landmarks, LEFT_EYE)
            right_eye_region, right_bbox = extract_eye_region(frame, landmarks, RIGHT_EYE)
            
            # Detect pupils
            if left_eye_region is not None:
                left_pupil, left_threshold, left_status = detect_pupil(left_eye_region)
            
            if right_eye_region is not None:
                right_pupil, right_threshold, right_status = detect_pupil(right_eye_region)
            
            # Calculate gaze ratios
            left_h_ratio, left_v_ratio = None, None
            right_h_ratio, right_v_ratio = None, None
            
            if left_pupil and left_eye_region is not None:
                left_h_ratio, left_v_ratio = calculate_gaze_ratio(left_pupil, left_eye_region.shape)
            
            if right_pupil and right_eye_region is not None:
                right_h_ratio, right_v_ratio = calculate_gaze_ratio(right_pupil, right_eye_region.shape)
            
            # Average both eyes
            if left_h_ratio is not None and right_h_ratio is not None:
                avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
                avg_v_ratio = (left_v_ratio + right_v_ratio) / 2
                
                gaze_direction = get_gaze_direction(avg_h_ratio, avg_v_ratio)
                
                # Main display
                cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                if show_calibration:
                    cv2.putText(frame, f"H: {avg_h_ratio:.3f} V: {avg_v_ratio:.3f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"L: {left_status}", (10, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, f"R: {right_status}", (10, 115),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw gaze indicator
                indicator_x = int(w/2 + (avg_h_ratio - 0.5) * w * 0.8)
                indicator_y = int(h/2 + (avg_v_ratio - 0.5) * h * 0.8)
                cv2.circle(frame, (indicator_x, indicator_y), 15, (0, 255, 255), -1)
                cv2.circle(frame, (w//2, h//2), 8, (255, 255, 255), -1)
                
            elif left_h_ratio is not None or right_h_ratio is not None:
                # At least one eye detected
                cv2.putText(frame, "Partial detection - trying...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # No pupils detected
                cv2.putText(frame, "Pupil detection failed - adjust lighting", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if show_calibration:
                    cv2.putText(frame, f"L: {left_status}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, f"R: {right_status}", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw eye bounding boxes
            if left_bbox:
                cv2.rectangle(frame, (left_bbox[0], left_bbox[1]), 
                             (left_bbox[2], left_bbox[3]), (0, 255, 0), 2)
            if right_bbox:
                cv2.rectangle(frame, (right_bbox[0], right_bbox[1]), 
                             (right_bbox[2], right_bbox[3]), (0, 255, 0), 2)
            
            # Draw pupil positions on main frame
            if left_pupil and left_bbox:
                pupil_x = left_bbox[0] + left_pupil[0]
                pupil_y = left_bbox[1] + left_pupil[1]
                cv2.circle(frame, (pupil_x, pupil_y), 4, (255, 0, 0), -1)
            
            if right_pupil and right_bbox:
                pupil_x = right_bbox[0] + right_pupil[0]
                pupil_y = right_bbox[1] + right_pupil[1]
                cv2.circle(frame, (pupil_x, pupil_y), 4, (255, 0, 0), -1)
            
            # Show debug windows
            if show_debug:
                if left_eye_region is not None and left_eye_region.size > 0:
                    left_h, left_w = left_eye_region.shape[:2]
                    if left_h > 0 and left_w > 0:
                        left_debug = cv2.resize(left_eye_region, 
                                               (left_w * debug_scale, left_h * debug_scale))
                        
                        # Draw pupil on debug view
                        if left_pupil:
                            cv2.circle(left_debug, 
                                     (left_pupil[0]*debug_scale, left_pupil[1]*debug_scale), 
                                     6, (0, 255, 0), -1)
                        
                        cv2.imshow('Left Eye', left_debug)
                
                if right_eye_region is not None and right_eye_region.size > 0:
                    right_h, right_w = right_eye_region.shape[:2]
                    if right_h > 0 and right_w > 0:
                        right_debug = cv2.resize(right_eye_region, 
                                                (right_w * debug_scale, right_h * debug_scale))
                        
                        # Draw pupil on debug view
                        if right_pupil:
                            cv2.circle(right_debug, 
                                     (right_pupil[0]*debug_scale, right_pupil[1]*debug_scale), 
                                     6, (0, 255, 0), -1)
                        
                        cv2.imshow('Right Eye', right_debug)
                
                # Show threshold images
                if left_threshold is not None and left_threshold.size > 0:
                    left_t_h, left_t_w = left_threshold.shape[:2]
                    if left_t_h > 0 and left_t_w > 0:
                        left_thresh_resized = cv2.resize(left_threshold, 
                                                        (left_t_w * debug_scale, left_t_h * debug_scale))
                        cv2.imshow('Left Threshold', left_thresh_resized)
                
                if right_threshold is not None and right_threshold.size > 0:
                    right_t_h, right_t_w = right_threshold.shape[:2]
                    if right_t_h > 0 and right_t_w > 0:
                        right_thresh_resized = cv2.resize(right_threshold,
                                                         (right_t_w * debug_scale, right_t_h * debug_scale))
                        cv2.imshow('Right Threshold', right_thresh_resized)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Eye Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                # Safely close debug windows
                try:
                    cv2.destroyWindow('Left Eye')
                    cv2.destroyWindow('Right Eye')
                    cv2.destroyWindow('Left Threshold')
                    cv2.destroyWindow('Right Threshold')
                except:
                    pass
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
        
        elif key == ord('c'):
            show_calibration = not show_calibration
            print(f"Calibration info: {'ON' if show_calibration else 'OFF'}")
        
        elif key == 27:
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\n✓ Program closed successfully")