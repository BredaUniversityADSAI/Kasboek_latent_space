import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize MediaPipe
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

def process_frame_with_mediapipe(frame):
    """Process frame with MediaPipe and return results"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_results = segmentation.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    mask = seg_results.segmentation_mask > 0.5
    mask_3d = np.stack([mask] * 3, axis=-1)
    
    return {
        'seg_results': seg_results,
        'face_results': face_results,
        'pose_results': pose_results,
        'mask_3d': mask_3d
    }

def cleanup_mediapipe():
    """Cleanup MediaPipe resources"""
    segmentation.close()
    face_mesh.close()
    pose.close()

# Filter functions
def apply_filter_1(frame, **kwargs):
    """Gaussian blur"""
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_filter_2(frame, **kwargs):
    """Rainbow tint"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    for y in range(h):
        hue = int((y / h) * 180)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        overlay[y, :] = cv2.addWeighted(overlay[y, :], 0.7, 
                                        np.full((w, 3), color, dtype=np.uint8), 0.3, 0)
    return overlay

def apply_filter_3(frame, **kwargs):
    """Colorful edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = edges
    edges_colored[:, :, 1] = 255 - edges
    edges_colored[:, :, 2] = edges
    
    return cv2.addWeighted(frame, 0.6, edges_colored, 0.4, 0)

def apply_filter_4(frame, **kwargs):
    """Psychedelic colors - HSV shift"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(time.time() * 50) % 180) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_filter_5(frame, mask_3d=None, **kwargs):
    """Inverted person, normal background"""
    if mask_3d is None:
        return frame
    
    person = np.where(mask_3d, frame, 0)
    person_inverted = cv2.bitwise_not(person)
    person_inverted = np.where(mask_3d, person_inverted, 0)
    
    background = np.where(~mask_3d, frame, 0)
    result = cv2.add(person_inverted, background)
    return result

def apply_filter_6(frame, pose_results=None, **kwargs):
    """Body skeleton - neon colors"""
    if pose_results is None or not pose_results.pose_landmarks:
        return frame
    
    h, w = frame.shape[:2]
    connections = mp_pose.POSE_CONNECTIONS
    
    for connection in connections:
        start_idx, end_idx = connection
        start = pose_results.pose_landmarks.landmark[start_idx]
        end = pose_results.pose_landmarks.landmark[end_idx]
        
        if start.visibility > 0.5 and end.visibility > 0.5:
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            color = [(0, 255, 255), (255, 0, 255), (255, 255, 0)][start_idx % 3]
            cv2.line(frame, start_point, end_point, color, 3)
    
    for landmark in pose_results.pose_landmarks.landmark:
        if landmark.visibility > 0.5:
            point = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    return frame

def apply_filter_7(frame, **kwargs):
    """Purple Haze"""
    overlay = frame.copy()
    purple = np.full(frame.shape, [180, 50, 100], dtype=np.uint8)
    return cv2.addWeighted(overlay, 0.7, purple, 0.3, 0)

def apply_filter_8(frame, **kwargs):
    """Mirror Effect"""
    h, w = frame.shape[:2]
    result = frame.copy()
    result[:, w//2:] = cv2.flip(frame[:, :w//2], 1)
    return result

def apply_filter_9(frame, **kwargs):
    """Kaleidoscope"""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    quad = frame[:h//2, :w//2]
    
    result[:h//2, :w//2] = quad
    result[:h//2, w//2:] = cv2.flip(quad, 1)
    result[h//2:, :w//2] = cv2.flip(quad, 0)
    result[h//2:, w//2:] = cv2.flip(quad, -1)
    
    return result

def apply_filter_10(frame, **kwargs):
    """Thermal Vision"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def apply_filter_11(frame, **kwargs):
    """X-Ray Effect"""
    inverted = cv2.bitwise_not(frame)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    xray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    xray[:, :, 1] = xray[:, :, 1] // 2
    return xray

def apply_filter_12(frame, **kwargs):
    """Vintage Film"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    
    noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
    sepia = cv2.add(sepia, noise)
    
    return sepia

def apply_filter_13(frame, **kwargs):
    """Neon Glow"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = edges
    edges_colored[:, :, 1] = edges
    edges_colored[:, :, 2] = 0
    
    glow = cv2.GaussianBlur(edges_colored, (9, 9), 0)
    return cv2.addWeighted(frame, 0.6, glow, 0.8, 0)

def apply_filter_14(frame, mask_3d=None, **kwargs):
    """Starfield Background"""
    if mask_3d is None:
        return frame
    
    h, w = frame.shape[:2]
    stars = np.zeros(frame.shape, dtype=np.uint8)
    
    np.random.seed(int(time.time() * 10) % 1000)
    num_stars = 200
    for _ in range(num_stars):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        brightness = np.random.randint(100, 255)
        cv2.circle(stars, (x, y), 1, (brightness, brightness, brightness), -1)
    
    result = np.where(mask_3d, frame, stars)
    return result

def apply_filter_15(frame, **kwargs):
    """Disco Lights"""
    h, w = frame.shape[:2]
    
    t = time.time() * 2
    overlay = frame.copy().astype(np.float32)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, color in enumerate(colors):
        angle = t + i * (2 * np.pi / len(colors))
        cx = int(w / 2 + np.cos(angle) * w / 4)
        cy = int(h / 2 + np.sin(angle) * h / 4)
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        spotlight = np.exp(-dist / 80)
        
        for c in range(3):
            overlay[:, :, c] += spotlight * color[c] * 0.3
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def apply_filter_16(frame, **kwargs):
    """Glitch Art"""
    result = frame.copy()
    h, w = result.shape[:2]
    
    for _ in range(5):
        y = np.random.randint(0, h - 20)
        shift = np.random.randint(-30, 30)
        if shift > 0:
            result[y:y+20, shift:] = frame[y:y+20, :-shift]
        elif shift < 0:
            result[y:y+20, :shift] = frame[y:y+20, -shift:]
    
    if np.random.random() > 0.7:
        corrupt_channel = np.random.randint(0, 3)
        y = np.random.randint(0, h - 50)
        result[y:y+50, :, corrupt_channel] = 255
    
    return result

def apply_filter_17(frame, **kwargs):
    """Solarize"""
    result = frame.copy()
    threshold = 128
    result[result < threshold] = 255 - result[result < threshold]
    return result

def apply_filter_18(frame, **kwargs):
    """Color Channel Shift"""
    result = frame.copy()
    result[:, :, 0], result[:, :, 2] = frame[:, :, 2].copy(), frame[:, :, 0].copy()
    return result

def apply_filter_19(frame, **kwargs):
    """Emboss Effect"""
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    embossed = cv2.filter2D(frame, -1, kernel)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_BGR2GRAY)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 0.5, embossed, 0.5, 128)

# Filter registry
FILTERS = {
    1: ("Gaussian Blur", apply_filter_1),
    2: ("Rainbow Tint", apply_filter_2),
    3: ("Edge Detection", apply_filter_3),
    4: ("Psychedelic Colors", apply_filter_4),
    5: ("Inverted Person", apply_filter_5),
    6: ("Body Skeleton", apply_filter_6),
    7: ("Purple Haze", apply_filter_7),
    8: ("Mirror Effect", apply_filter_8),
    9: ("Kaleidoscope", apply_filter_9),
    10: ("Thermal Vision", apply_filter_10),
    11: ("X-Ray Effect", apply_filter_11),
    12: ("Vintage Film", apply_filter_12),
    13: ("Neon Glow", apply_filter_13),
    14: ("Starfield Background", apply_filter_14),
    15: ("Disco Lights", apply_filter_15),
    16: ("Glitch Art", apply_filter_16),
    17: ("Solarize", apply_filter_17),
    18: ("Color Channel Shift", apply_filter_18),
    19: ("Emboss Effect", apply_filter_19),
}

def apply_filters(frame, active_filters, mediapipe_results):
    """Apply all active filters to the frame"""
    processed = frame.copy()
    
    for filter_num in sorted(active_filters.keys()):
        if active_filters[filter_num]:
            _, filter_func = FILTERS[filter_num]
            processed = filter_func(processed, **mediapipe_results)
    
    return processed

def get_filter_names():
    """Return a dictionary of filter numbers to names"""
    return {num: name for num, (name, _) in FILTERS.items()}