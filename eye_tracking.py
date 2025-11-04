import cv2
import numpy as np
import mediapipe as mp
import sys

print("\n" + "="*60)
print("JETSON ADVANCED EYE TRACKING SYSTEM")
print("="*60)

# Initialize MediaPipe Face Mesh for eye region detection
print("\nInitializing MediaPipe Face Mesh...")
REFINE_LANDMARKS = True
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=REFINE_LANDMARKS,
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

# Iris landmark indices (MediaPipe Face Mesh, refine_landmarks=True)
# These give a more stable, direct estimate of the pupil/iris center when available
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Configuration constants (replace magic numbers)
WINDOW_NAME = 'Eye Tracking'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
REFINE_LANDMARKS = True  # Set to False to reduce model cost (no iris landmarks)
PROCESS_EVERY_N_FRAMES = 2  # skip processing some frames to improve FPS (1 = every frame)

# Detect/pupil parameters
BLUR_KSIZE = (7, 7)
ADAPTIVE_BLOCKSIZE = 11
ADAPTIVE_C = 2
THRESH_SIMPLE = 50
MORPH_KERNEL_SIZE = (3, 3)
CIRCULARITY_THRESHOLD = 0.3
AREA_MIN_FRAC = 0.01
AREA_MAX_FRAC = 0.5
MIN_EYE_DIM = 10
EYE_PADDING = 5

# Indicator and smoothing
INDICATOR_MULTIPLIER = 0.6
# Axis inversion (set if visual directions are flipped for your camera/mirror)
INVERT_HORIZONTAL = False
INVERT_VERTICAL = False

# Calibration / mapping
CAL_SAMPLES_PER_POINT = 25
CAL_POINTS_GRID = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
]
CALIBRATION_FILENAME = 'eye_calibration.json'

calibrating = False
calib_index = 0
calib_samples = []  # collected (h,v) samples for current point
calib_measured = []  # list of mean measured (h,v) per point
calib_targets = []  # target (x_norm, y_norm) per point (same order)
mapping_coeffs = None  # (coeffs_x, coeffs_y) each length 3 for affine [h,v,1]
 

def compute_affine_mapping(measured, targets):
    """Compute affine mapping from measured (h,v) to target (x_norm,y_norm).
    measured: list of (h,v)
    targets: list of (x_norm,y_norm)
    Returns (coeffs_x, coeffs_y) where x = [h,v,1] @ coeffs_x
    """
    if len(measured) < 3:
        return None
    A = np.array([[h, v, 1.0] for (h, v) in measured])
    tx = np.array([t[0] for t in targets])
    ty = np.array([t[1] for t in targets])
    # least squares
    try:
        coeffs_x, *_ = np.linalg.lstsq(A, tx, rcond=None)
        coeffs_y, *_ = np.linalg.lstsq(A, ty, rcond=None)
        return (coeffs_x, coeffs_y)
    except Exception:
        return None

def save_calibration(coeffs, filename=CALIBRATION_FILENAME):
    try:
        import json
        data = {
            'coeffs_x': coeffs[0].tolist(),
            'coeffs_y': coeffs[1].tolist(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        return True
    except Exception:
        return False

def load_calibration(filename=CALIBRATION_FILENAME):
    try:
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        cx = np.array(data['coeffs_x'], dtype=float)
        cy = np.array(data['coeffs_y'], dtype=float)
        return (cx, cy)
    except Exception:
        return None

# Try loading saved calibration now that load_calibration is defined
loaded = load_calibration()
if loaded is not None:
    mapping_coeffs = loaded
    print("Loaded existing calibration.")

def compute_ratio_from_iris(iris_center, landmarks, eye_corner_indices, eye_indices, image_w, image_h):
    """
    Compute horizontal and vertical gaze ratios using iris center (global coords) and
    landmark-based normalization (inner/outer corners and eyelid extrema).
    Returns (h_ratio, v_ratio) in [0,1] or (None, None) on failure.
    """
    if iris_center is None:
        return None, None

    try:
        # corners: expect [outer_idx, inner_idx]
        outer = landmarks[eye_corner_indices[0]]
        inner = landmarks[eye_corner_indices[1]]
    except Exception:
        return None, None

    # Convert to pixel coords
    outer_x = outer.x * image_w
    inner_x = inner.x * image_w

    # Horizontal: normalize iris x between inner and outer corner
    min_x = min(outer_x, inner_x)
    max_x = max(outer_x, inner_x)
    if max_x - min_x <= 1e-3:
        return None, None

    iris_x = iris_center[0]
    h_ratio = (iris_x - min_x) / (max_x - min_x)

    # Vertical: use eye landmark y extrema (top = min y, bottom = max y)
    try:
        eye_pts = [landmarks[i] for i in eye_indices]
    except Exception:
        return None, None

    ys = [p.y * image_h for p in eye_pts]
    top_y = min(ys)
    bottom_y = max(ys)
    if bottom_y - top_y <= 1e-3:
        return None, None

    iris_y = iris_center[1]
    v_ratio = (iris_y - top_y) / (bottom_y - top_y)

    # Clip
    h_ratio = float(np.clip(h_ratio, 0.0, 1.0))
    v_ratio = float(np.clip(v_ratio, 0.0, 1.0))

    if INVERT_HORIZONTAL:
        h_ratio = 1.0 - h_ratio
    if INVERT_VERTICAL:
        v_ratio = 1.0 - v_ratio

    return h_ratio, v_ratio

def detect_pupil(eye_frame):
    """
    Detect pupil using multiple methods
    Returns pupil center relative to eye frame and debug info
    """
    if eye_frame is None or eye_frame.size == 0:
        return None, None, "Empty frame"
    
    h, w = eye_frame.shape[:2]
    if h < MIN_EYE_DIM or w < MIN_EYE_DIM:
        return None, None, "Frame too small"
    
    # Convert to grayscale
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Adaptive thresholding (works better in varied lighting)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCKSIZE, ADAPTIVE_C)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Method 2: Try simple thresholding with lower threshold
        _, thresh2 = cv2.threshold(blur, THRESH_SIMPLE, 255, cv2.THRESH_BINARY_INV)
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
            
            # Check if area is reasonable
            if not (AREA_MIN_FRAC * eye_area < area < AREA_MAX_FRAC * eye_area):
                continue
            
            # Check circularity (4*pi*area/perimeter^2, circle = 1.0)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > CIRCULARITY_THRESHOLD:  # Reasonably circular
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
    if min_val < THRESH_SIMPLE:  # If there's a dark enough spot
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

def get_iris_center(landmarks, iris_indices, image_width, image_height):
    """
    Compute the average (x,y) of iris landmarks and return image coordinates.
    Returns None if any index is out of range.
    """
    try:
        pts = [landmarks[i] for i in iris_indices]
    except Exception:
        return None

    xs = [p.x * image_width for p in pts]
    ys = [p.y * image_height for p in pts]
    if not xs or not ys:
        return None
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    return (cx, cy)

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

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
show_hud = True  # on-screen diagnostics HUD
debug_scale = 4  # Scale factor for debug windows
# Smoothing for gaze to reduce jitter
smooth_avg_h_ratio = None
smooth_avg_v_ratio = None
smoothing_alpha = 0.22  # EMA alpha: 0-1, larger = more responsive, smaller = smoother

frame_count = 0
prev_results = None
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe and optionally skip frames to improve FPS
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = face_mesh.process(rgb_frame)
            prev_results = results
        else:
            results = prev_results
        
        left_pupil = None
        right_pupil = None
        left_bbox = None
        right_bbox = None
        left_threshold = None
        right_threshold = None
        left_status = ""
        right_status = ""
        method_used = "-"
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            # Extract eye regions (use named padding)
            left_eye_region, left_bbox = extract_eye_region(frame, landmarks, LEFT_EYE, padding=EYE_PADDING)
            right_eye_region, right_bbox = extract_eye_region(frame, landmarks, RIGHT_EYE, padding=EYE_PADDING)
            
            # Prefer iris-based center (more stable) and fall back to image-based pupil detection
            left_h_ratio, left_v_ratio = None, None
            right_h_ratio, right_v_ratio = None, None

            # Try iris landmarks first (global image coords) and compute landmark-based ratios
            if results and results.multi_face_landmarks:
                if left_eye_region is not None and left_bbox is not None:
                    iris_left = get_iris_center(landmarks, LEFT_IRIS, w, h)
                    if iris_left is not None:
                        # convert global iris center to eye-region relative coords (for drawing)
                        left_pupil = (iris_left[0] - left_bbox[0], iris_left[1] - left_bbox[1])
                        left_status = 'Iris'
                        left_threshold = None
                        # compute gaze ratios using landmarks (more robust than bbox-based)
                        lh, lv = compute_ratio_from_iris(iris_left, landmarks, LEFT_EYE_CORNERS, LEFT_EYE, w, h)
                        if lh is not None:
                            left_h_ratio, left_v_ratio = lh, lv
                        method_used = 'Iris'
                    else:
                        if left_eye_region is not None:
                            left_pupil, left_threshold, left_status = detect_pupil(left_eye_region)

                if right_eye_region is not None and right_bbox is not None:
                    iris_right = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                    if iris_right is not None:
                        right_pupil = (iris_right[0] - right_bbox[0], iris_right[1] - right_bbox[1])
                        right_status = 'Iris'
                        right_threshold = None
                        rh, rv = compute_ratio_from_iris(iris_right, landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE, w, h)
                        if rh is not None:
                            right_h_ratio, right_v_ratio = rh, rv
                        method_used = 'Iris'
                    else:
                        if right_eye_region is not None:
                            right_pupil, right_threshold, right_status = detect_pupil(right_eye_region)

            # If landmark-based ratios are not available, fall back to image-region-based calculation
            if (left_h_ratio is None or left_v_ratio is None) and left_pupil is not None and left_eye_region is not None:
                lh, lv = calculate_gaze_ratio(left_pupil, left_eye_region.shape)
                if lh is not None:
                    left_h_ratio = float(np.clip(lh, 0.0, 1.0))
                    left_v_ratio = float(np.clip(lv, 0.0, 1.0))
                    method_used = method_used if method_used != 'Iris' else method_used
            if (right_h_ratio is None or right_v_ratio is None) and right_pupil is not None and right_eye_region is not None:
                rh, rv = calculate_gaze_ratio(right_pupil, right_eye_region.shape)
                if rh is not None:
                    right_h_ratio = float(np.clip(rh, 0.0, 1.0))
                    right_v_ratio = float(np.clip(rv, 0.0, 1.0))
                    method_used = method_used if method_used != 'Iris' else method_used

            # Average both eyes when available
            avg_h_ratio = None
            avg_v_ratio = None
            if left_h_ratio is not None and right_h_ratio is not None:
                avg_h_ratio = (left_h_ratio + right_h_ratio) / 2.0
                avg_v_ratio = (left_v_ratio + right_v_ratio) / 2.0
            elif left_h_ratio is not None:
                avg_h_ratio = left_h_ratio
                avg_v_ratio = left_v_ratio
            elif right_h_ratio is not None:
                avg_h_ratio = right_h_ratio
                avg_v_ratio = right_v_ratio

            # Smooth ratios to remove jumpiness
            if avg_h_ratio is not None and avg_v_ratio is not None:
                if smooth_avg_h_ratio is None:
                    smooth_avg_h_ratio = avg_h_ratio
                    smooth_avg_v_ratio = avg_v_ratio
                else:
                    smooth_avg_h_ratio = smoothing_alpha * avg_h_ratio + (1 - smoothing_alpha) * smooth_avg_h_ratio
                    smooth_avg_v_ratio = smoothing_alpha * avg_v_ratio + (1 - smoothing_alpha) * smooth_avg_v_ratio

                gaze_direction = get_gaze_direction(smooth_avg_h_ratio, smooth_avg_v_ratio)

                # Main display
                cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # HUD diagnostics
                if show_hud:
                    hud_y = 35
                    cv2.putText(frame, f"method:{method_used} map:{'Y' if mapping_coeffs is not None else 'N'} invH:{INVERT_HORIZONTAL} invV:{INVERT_VERTICAL}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    hud_y += 18
                    if smooth_avg_h_ratio is not None:
                        cv2.putText(frame, f"smoothed H:{smooth_avg_h_ratio:.3f} V:{smooth_avg_v_ratio:.3f}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                        hud_y += 18
                    if left_h_ratio is not None:
                        cv2.putText(frame, f"Lraw H:{left_h_ratio:.3f} V:{left_v_ratio:.3f}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,200,250), 1)
                        hud_y += 16
                    if right_h_ratio is not None:
                        cv2.putText(frame, f"Rraw H:{right_h_ratio:.3f} V:{right_v_ratio:.3f}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,200,250), 1)
                        hud_y += 16

                if show_calibration:
                    cv2.putText(frame, f"H: {smooth_avg_h_ratio:.3f} V: {smooth_avg_v_ratio:.3f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"L: {left_status}", (10, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, f"R: {right_status}", (10, 115),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # If we're calibrating, collect samples for current target
                if calibrating:
                    tx_norm, ty_norm = CAL_POINTS_GRID[calib_index]
                    tx_px = int(tx_norm * w)
                    ty_px = int(ty_norm * h)
                    cv2.circle(frame, (tx_px, ty_px), 14, (0, 0, 255), -1)
                    cv2.putText(frame, f"Calibrating {calib_index+1}/{len(CAL_POINTS_GRID)}",
                               (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # collect sample if available
                    if smooth_avg_h_ratio is not None:
                        calib_samples.append((smooth_avg_h_ratio, smooth_avg_v_ratio))
                    # when enough samples collected, store mean and advance
                    if len(calib_samples) >= CAL_SAMPLES_PER_POINT:
                        mean = np.mean(np.array(calib_samples), axis=0)
                        calib_measured.append((float(mean[0]), float(mean[1])))
                        calib_targets.append(CAL_POINTS_GRID[calib_index])
                        calib_samples = []
                        calib_index += 1
                        # small visual feedback - continue to next point next frame
                    # if finished, compute mapping
                    if calib_index >= len(CAL_POINTS_GRID):
                        coeffs = compute_affine_mapping(calib_measured, calib_targets)
                        if coeffs is not None:
                            mapping_coeffs = coeffs
                            saved = save_calibration(coeffs)
                            print("Calibration completed.", "Saved" if saved else "Not saved")
                        else:
                            print("Calibration failed: could not compute mapping")
                        # reset calibration state
                        calibrating = False
                        calib_index = 0
                        calib_samples = []
                        calib_measured = []
                        calib_targets = []
                # Draw gaze indicator using mapping if available
                elif mapping_coeffs is not None and smooth_avg_h_ratio is not None:
                    try:
                        vec = np.array([smooth_avg_h_ratio, smooth_avg_v_ratio, 1.0])
                        cx = float(vec @ mapping_coeffs[0])
                        cy = float(vec @ mapping_coeffs[1])
                        cx = float(np.clip(cx, 0.0, 1.0))
                        cy = float(np.clip(cy, 0.0, 1.0))
                        indicator_x = int(cx * w)
                        indicator_y = int(cy * h)
                    except Exception:
                        indicator_x = int(w/2 + (smooth_avg_h_ratio - 0.5) * w * INDICATOR_MULTIPLIER)
                        indicator_y = int(h/2 + (smooth_avg_v_ratio - 0.5) * h * INDICATOR_MULTIPLIER)
                    cv2.circle(frame, (indicator_x, indicator_y), 15, (0, 255, 255), -1)
                    if show_calibration:
                        cv2.circle(frame, (w//2, h//2), 8, (255, 255, 255), -1)
                else:
                    indicator_x = int(w/2 + (smooth_avg_h_ratio - 0.5) * w * INDICATOR_MULTIPLIER)
                    indicator_y = int(h/2 + (smooth_avg_v_ratio - 0.5) * h * INDICATOR_MULTIPLIER)
                    cv2.circle(frame, (indicator_x, indicator_y), 15, (0, 255, 255), -1)
                    if show_calibration:
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
        
        cv2.imshow(WINDOW_NAME, frame)

        # If the user closed the window with the X button, exit cleanly
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

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
        
        elif key == ord('h'):
            INVERT_HORIZONTAL = not INVERT_HORIZONTAL
            print(f"Invert horizontal: {INVERT_HORIZONTAL}")

        elif key == ord('v'):
            INVERT_VERTICAL = not INVERT_VERTICAL
            print(f"Invert vertical: {INVERT_VERTICAL}")

        elif key == ord('r'):
            # Reset mapping
            mapping_coeffs = None
            try:
                import os
                if os.path.exists(CALIBRATION_FILENAME):
                    os.remove(CALIBRATION_FILENAME)
                    print("Calibration file removed.")
            except Exception:
                pass
            print("Calibration reset.")
        
        elif key == ord('p'):
            # Start interactive calibration
            calibrating = True
            calib_index = 0
            calib_samples = []
            calib_measured = []
            calib_targets = []
            print("Calibration started: look at each red dot until it advances")

        elif key == ord('l'):
            loaded = load_calibration()
            if loaded is not None:
                mapping_coeffs = loaded
                print("Calibration loaded from disk.")
            else:
                print("No calibration file found.")

        elif key == ord('s'):
            if mapping_coeffs is not None:
                ok = save_calibration(mapping_coeffs)
                print("Calibration saved." if ok else "Failed to save calibration.")
            else:
                print("No calibration to save.")
        
        elif key == 27:
            break

        frame_count += 1

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\n✓ Program closed successfully")