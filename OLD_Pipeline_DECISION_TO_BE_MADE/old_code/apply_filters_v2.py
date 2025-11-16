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
mp_hands = mp.solutions.hands

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
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("FILTERS (30 TOTAL):")
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
print("\n  a - Purple Haze")
print("  b - Mirror Effect")
print("  c - Kaleidoscope")
print("  d - Thermal Vision")
print("  e - X-Ray Effect")
print("  f - Vintage Film")
print("  g - Neon Glow")
print("  h - Starfield Background")
print("  i - Chromatic Aberration")
print("  j - Oil Painting")
print("  k - Vaporwave Aesthetic")
print("  l - Glitch Art")
print("  m - Scanlines (CRT)")
print("  n - Hand Trails")
print("  o - Solarize")
print("  p - Color Channel Shift")
print("  q - Cartoon Effect")
print("  s - Matrix Rain Background")
print("  t - Emboss Effect")
print("  u - Disco Lights")
print("\n  Press 'r' - Reset all filters")
print("  Press ESC - Quit")
print("="*60)
print("\nWebcam active. Toggle filters with keys!\n")

# Active filters dictionary - FIXED to include all keys
active_filters = {}
for i in range(10):
    active_filters[i] = False
# Add all letter keys a-u (skipping 'r' which is for reset)
for letter in 'abcdefghijklmnopqstu':
    active_filters[letter] = False

# Global variables for animated filters
frame_count = 0

def apply_filter_0(frame, mask):
    """Blue background"""
    bg_color = np.full(frame.shape, [180, 100, 50], dtype=np.uint8)
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
    
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = edges
    edges_colored[:, :, 1] = 255 - edges
    edges_colored[:, :, 2] = edges
    
    return cv2.addWeighted(frame, 0.6, edges_colored, 0.4, 0)

def apply_filter_5(frame):
    """Psychedelic colors - HSV shift"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(time.time() * 50) % 180) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
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
    
    small = cv2.resize(frame, (w // pixel_size, h // pixel_size), 
                       interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def apply_filter_8(frame, pose_results):
    """Body skeleton - neon colors"""
    if pose_results.pose_landmarks:
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

def apply_filter_9(frame):
    """Warm/Cool split"""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    result[:, :w//2, 2] = np.clip(result[:, :w//2, 2] * 1.3, 0, 255)
    result[:, :w//2, 1] = np.clip(result[:, :w//2, 1] * 1.1, 0, 255)
    
    result[:, w//2:, 0] = np.clip(result[:, w//2:, 0] * 1.3, 0, 255)
    result[:, w//2:, 1] = np.clip(result[:, w//2:, 1] * 0.9, 0, 255)
    
    cv2.line(result, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    
    return result

def apply_filter_a(frame):
    """Purple Haze"""
    overlay = frame.copy()
    purple = np.full(frame.shape, [180, 50, 100], dtype=np.uint8)
    return cv2.addWeighted(overlay, 0.7, purple, 0.3, 0)

def apply_filter_b(frame):
    """Mirror Effect"""
    h, w = frame.shape[:2]
    result = frame.copy()
    result[:, w//2:] = cv2.flip(frame[:, :w//2], 1)
    return result

def apply_filter_c(frame):
    """Kaleidoscope"""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    # Top-left quadrant
    quad = frame[:h//2, :w//2]
    
    # Mirror it 4 ways
    result[:h//2, :w//2] = quad
    result[:h//2, w//2:] = cv2.flip(quad, 1)
    result[h//2:, :w//2] = cv2.flip(quad, 0)
    result[h//2:, w//2:] = cv2.flip(quad, -1)
    
    return result

def apply_filter_d(frame):
    """Thermal Vision"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def apply_filter_e(frame):
    """X-Ray Effect"""
    inverted = cv2.bitwise_not(frame)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    xray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    xray[:, :, 1] = xray[:, :, 1] // 2
    return xray

def apply_filter_f(frame):
    """Vintage Film"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    
    noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
    sepia = cv2.add(sepia, noise)
    
    return sepia

def apply_filter_g(frame):
    """Neon Glow"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = edges
    edges_colored[:, :, 1] = edges
    edges_colored[:, :, 2] = 0
    
    glow = cv2.GaussianBlur(edges_colored, (9, 9), 0)
    return cv2.addWeighted(frame, 0.6, glow, 0.8, 0)

def apply_filter_h(frame, mask):
    """Starfield Background"""
    h, w = frame.shape[:2]
    stars = np.zeros(frame.shape, dtype=np.uint8)
    
    np.random.seed(int(time.time() * 10) % 1000)
    num_stars = 200
    for _ in range(num_stars):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        brightness = np.random.randint(100, 255)
        cv2.circle(stars, (x, y), 1, (brightness, brightness, brightness), -1)
    
    result = np.where(mask, frame, stars)
    return result

def apply_filter_i(frame):
    """Chromatic Aberration"""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    result[:-3, :-3, 2] = frame[3:, 3:, 2]
    result[3:, 3:, 0] = frame[:-3, :-3, 0]
    
    return result

def apply_filter_j(frame):
    """Oil Painting"""
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)

def apply_filter_k(frame):
    """Vaporwave Aesthetic"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv[:, :, 0] = (hsv[:, :, 0] + 150) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    
    vaporwave = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    h, w = vaporwave.shape[:2]
    for y in range(0, h, 20):
        cv2.line(vaporwave, (0, y), (w, y), (255, 0, 255), 1)
    
    return vaporwave

def apply_filter_l(frame):
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

def apply_filter_m(frame):
    """Scanlines (CRT)"""
    result = frame.copy()
    h = result.shape[0]
    
    for y in range(0, h, 3):
        result[y:y+1, :] = (result[y:y+1, :] * 0.5).astype(np.uint8)
    
    return result

def apply_filter_n(frame, hands_results):
    """Hand Trails"""
    if not hasattr(apply_filter_n, 'trail_buffer'):
        apply_filter_n.trail_buffer = []
    
    if hands_results.multi_hand_landmarks:
        h, w = frame.shape[:2]
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                apply_filter_n.trail_buffer.append((x, y, time.time()))
    
    current_time = time.time()
    apply_filter_n.trail_buffer = [(x, y, t) for x, y, t in apply_filter_n.trail_buffer 
                                    if current_time - t < 1.0]
    
    for i, (x, y, t) in enumerate(apply_filter_n.trail_buffer):
        age = current_time - t
        alpha = 1.0 - age
        color = (int(255 * alpha), int(100 * alpha), int(200 * alpha))
        cv2.circle(frame, (x, y), 8, color, -1)
    
    return frame

def apply_filter_o(frame):
    """Solarize"""
    result = frame.copy()
    threshold = 128
    result[result < threshold] = 255 - result[result < threshold]
    return result

def apply_filter_p(frame):
    """Color Channel Shift"""
    result = frame.copy()
    result[:, :, 0], result[:, :, 2] = frame[:, :, 2].copy(), frame[:, :, 0].copy()
    return result

def apply_filter_q(frame):
    """Cartoon Effect"""
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    cartoon = cv2.bitwise_and(color, edges)
    return cartoon

def apply_filter_s(frame, mask):
    """Matrix Rain Background"""
    h, w = frame.shape[:2]
    matrix_bg = np.zeros(frame.shape, dtype=np.uint8)
    
    np.random.seed(int(time.time() * 20) % 1000)
    for x in range(0, w, 15):
        y = (int(time.time() * 100) + x * 37) % (h + 50) - 50
        for i in range(10):
            char_y = y + i * 20
            if 0 <= char_y < h:
                brightness = max(0, 255 - i * 25)
                cv2.putText(matrix_bg, chr(np.random.randint(33, 126)), 
                           (x, char_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, brightness, 0), 1)
    
    result = np.where(mask, frame, matrix_bg)
    return result

def apply_filter_t(frame):
    """Emboss Effect"""
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    embossed = cv2.filter2D(frame, -1, kernel)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_BGR2GRAY)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 0.5, embossed, 0.5, 128)

def apply_filter_u(frame):
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame")
        break
    
    frame_count += 1
    
    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_results = segmentation.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
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
    if active_filters['a']:
        processed = apply_filter_a(processed)
    if active_filters['b']:
        processed = apply_filter_b(processed)
    if active_filters['c']:
        processed = apply_filter_c(processed)
    if active_filters['d']:
        processed = apply_filter_d(processed)
    if active_filters['e']:
        processed = apply_filter_e(processed)
    if active_filters['f']:
        processed = apply_filter_f(processed)
    if active_filters['g']:
        processed = apply_filter_g(processed)
    if active_filters['h']:
        processed = apply_filter_h(processed, mask_3d)
    if active_filters['i']:
        processed = apply_filter_i(processed)
    if active_filters['j']:
        processed = apply_filter_j(processed)
    if active_filters['k']:
        processed = apply_filter_k(processed)
    if active_filters['l']:
        processed = apply_filter_l(processed)
    if active_filters['m']:
        processed = apply_filter_m(processed)
    if active_filters['n']:
        processed = apply_filter_n(processed, hands_results)
    if active_filters['o']:
        processed = apply_filter_o(processed)
    if active_filters['p']:
        processed = apply_filter_p(processed)
    if active_filters['q']:
        processed = apply_filter_q(processed)
    if active_filters['s']:
        processed = apply_filter_s(processed, mask_3d)
    if active_filters['t']:
        processed = apply_filter_t(processed)
    if active_filters['u']:
        processed = apply_filter_u(processed)
    
    # Display active filters on screen
    y_pos = 30
    for key, active in active_filters.items():
        if active:
            filter_name = f"Filter {key}: ON"
            cv2.putText(processed, filter_name, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
    
    cv2.imshow('Artistic Filters - Press 0-9, a-u to toggle', processed)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Toggle filters with number keys
    if ord('0') <= key <= ord('9'):
        filter_num = key - ord('0')
        active_filters[filter_num] = not active_filters[filter_num]
        status = "ON" if active_filters[filter_num] else "OFF"
        print(f"Filter {filter_num}: {status}")
    
    # Toggle filters with letter keys (skip 'r' as it's for reset)
    elif ord('a') <= key <= ord('u') and chr(key) != 'r':
        filter_char = chr(key)
        if filter_char in active_filters:
            active_filters[filter_char] = not active_filters[filter_char]
            status = "ON" if active_filters[filter_char] else "OFF"
            print(f"Filter {filter_char}: {status}")
    
    # Reset all filters
    elif key == ord('r'):
        for k in active_filters:
            active_filters[k] = False
        print("All filters reset")
    
    # Quit (ESC key)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
segmentation.close()
face_mesh.close()
pose.close()
hands.close()

print("\n✓ Program closed successfully")