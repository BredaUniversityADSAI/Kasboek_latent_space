import cv2
import mediapipe as mp
import numpy as np
import time

print("\n" + "="*60)
print("JETSON WEBCAM ARTISTIC FILTERS")
print("="*60)

# Initialize MediaPipe
print("\nInitializing MediaPipe...")
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✓ MediaPipe loaded successfully!\n")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("="*60)
print("FILTERS:")
print("  0 - Blue Background")
print("  1 - Face Outline (Neon Green)")
print("  2 - Gaussian Blur")
print("  3 - Rainbow Tint")
print("  4 - Edge Detection (Colorful)")
print("  5 - Psychedelic Colors (HSV shift)")
print("  6 - Inverted Person (Normal Background)")
print("  7 - Pixelation Effect")
print("  8 - Body Skeleton (Neon)")
print("  9 - Warm/Cool Split")
print("\n  Press 'r' - Reset all filters")
print("  Press 'q' - Quit")
print("="*60)
print("\nWebcam active. Toggle filters with number keys!\n")

# Active filters dictionary
active_filters = {i: False for i in range(10)}

def apply_filter_0(frame, mask):
    """Blue background"""
    bg_color = np.full(frame.shape, [180, 100, 50], dtype=np.uint8)  # Blue in BGR
    result = np.where(mask, frame, bg_color)
    return result

def apply_filter_1(frame, face_results):
    """Face outline - neon green"""
    if face_results.multi_face_landmarks:
        h, w = frame.shape[:2]
        for face_landmarks in face_results.multi_face_landmarks:
            points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
            
            # Draw contour around face
            face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            for i in range(len(face_oval)):
                start = points[face_oval[i]]
                end = points[face_oval[(i + 1) % len(face_oval)]]
                cv2.line(frame, start, end, (0, 255, 0), 2)
    return frame

def apply_filter_2(frame):
    """Gaussian blur"""
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_filter_3(frame):
    """Rainbow tint"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    for y in range(h):
        hue = int((y / h) * 180)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        overlay[y, :] = cv2.addWeighted(overlay[y, :], 0.7, 
                                        np.full((w, 3), color, dtype=np.uint8), 0.3, 0)
    return overlay

def apply_filter_4(frame):
    """Colorful edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Create colorful edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = edges  # Blue channel
    edges_colored[:, :, 1] = 255 - edges  # Green channel
    edges_colored[:, :, 2] = edges  # Red channel
    
    return cv2.addWeighted(frame, 0.6, edges_colored, 0.4, 0)

def apply_filter_5(frame):
    """Psychedelic colors - HSV shift"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(time.time() * 50) % 180) % 180  # Shift hue over time
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Boost saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_filter_6(frame, mask):
    """Inverted person, normal background"""
    person = np.where(mask, frame, 0)
    person_inverted = cv2.bitwise_not(person)
    person_inverted = np.where(mask, person_inverted, 0)
    
    background = np.where(~mask, frame, 0)
    result = cv2.add(person_inverted, background)
    return result

def apply_filter_7(frame):
    """Pixelation effect"""
    h, w = frame.shape[:2]
    pixel_size = 12
    
    # Shrink
    small = cv2.resize(frame, (w // pixel_size, h // pixel_size), 
                       interpolation=cv2.INTER_LINEAR)
    # Enlarge back
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def apply_filter_8(frame, pose_results):
    """Body skeleton - neon colors"""
    if pose_results.pose_landmarks:
        h, w = frame.shape[:2]
        
        # Define connections
        connections = mp_pose.POSE_CONNECTIONS
        
        for connection in connections:
            start_idx, end_idx = connection
            start = pose_results.pose_landmarks.landmark[start_idx]
            end = pose_results.pose_landmarks.landmark[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                # Alternate colors for different body parts
                color = [(0, 255, 255), (255, 0, 255), (255, 255, 0)][start_idx % 3]
                cv2.line(frame, start_point, end_point, color, 3)
        
        # Draw joints
        for landmark in pose_results.pose_landmarks.landmark:
            if landmark.visibility > 0.5:
                point = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    return frame

def apply_filter_9(frame):
    """Warm/Cool split"""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    # Left half - warm (increase red/yellow)
    result[:, :w//2, 2] = np.clip(result[:, :w//2, 2] * 1.3, 0, 255)  # Red
    result[:, :w//2, 1] = np.clip(result[:, :w//2, 1] * 1.1, 0, 255)  # Green
    
    # Right half - cool (increase blue)
    result[:, w//2:, 0] = np.clip(result[:, w//2:, 0] * 1.3, 0, 255)  # Blue
    result[:, w//2:, 1] = np.clip(result[:, w//2:, 1] * 0.9, 0, 255)  # Green
    
    # Draw dividing line
    cv2.line(result, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    
    return result

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame")
        break
    
    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_results = segmentation.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    # Create mask for segmentation-based filters
    mask = seg_results.segmentation_mask > 0.5
    mask_3d = np.stack([mask] * 3, axis=-1)
    
    # Start with original frame
    processed = frame.copy()
    
    # Apply active filters in order
    if active_filters[0]:
        processed = apply_filter_0(processed, mask_3d)
    
    if active_filters[1]:
        processed = apply_filter_1(processed, face_results)
    
    if active_filters[2]:
        processed = apply_filter_2(processed)
    
    if active_filters[3]:
        processed = apply_filter_3(processed)
    
    if active_filters[4]:
        processed = apply_filter_4(processed)
    
    if active_filters[5]:
        processed = apply_filter_5(processed)
    
    if active_filters[6]:
        processed = apply_filter_6(processed, mask_3d)
    
    if active_filters[7]:
        processed = apply_filter_7(processed)
    
    if active_filters[8]:
        processed = apply_filter_8(processed, pose_results)
    
    if active_filters[9]:
        processed = apply_filter_9(processed)
    
    # Display active filters on screen
    y_pos = 30
    for i, active in active_filters.items():
        if active:
            cv2.putText(processed, f"Filter {i}: ON", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
    
    cv2.imshow('Artistic Filters - Press 0-9 to toggle', processed)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Toggle filters with number keys
    if ord('0') <= key <= ord('9'):
        filter_num = key - ord('0')
        active_filters[filter_num] = not active_filters[filter_num]
        status = "ON" if active_filters[filter_num] else "OFF"
        print(f"Filter {filter_num}: {status}")
    
    # Reset all filters
    elif key == ord('r'):
        active_filters = {i: False for i in range(10)}
        print("All filters reset")
    
    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
segmentation.close()
face_mesh.close()
pose.close()

print("\n✓ Program closed successfully")