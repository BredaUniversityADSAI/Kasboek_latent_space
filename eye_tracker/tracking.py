import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

class EyeTracker:
    def __init__(self, flip=True):
        self.flip = flip
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # (Landmark indices remain the same)
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [473, 474, 475, 476]
        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER = 362
        self.RIGHT_EYE_INNER = 263
        self.LEFT_EYE_V = [386, 374]
        self.LEFT_EYE_H = [362, 263]
        self.RIGHT_EYE_V = [159, 145]
        self.RIGHT_EYE_H = [33, 133]
        self.MOUTH_V = [13, 14]
        self.MOUTH_H = [61, 291]
        
        # (Thresholds remain the same)
        self.horiz_thr = 0.035
        self.vert_thr = 0.028
        self.deadzone = 0.018
        
        # (Smoothing params remain the same)
        self.smooth_alpha = 0.6
        self.median_window = 5
        self.buffer = deque(maxlen=7)
        self.smoothed = None
        
        self.show_visuals = True
        
        # (Saccade params remain the same)
        self.prev = None
        self.prev_time = None
        self.saccade_count = 0
        self.saccade_speed_thr = 0.2
        
        self.gaze_amplification = 2.0
        
        # --- MODIFIED: Gesture detection ---
        self.blink_thresh = 0.25
        self.mouth_thresh = 0.4
        self.is_blinking = False
        self.blink_start_time = None
        self.is_mouth_open = False # This now tracks the state for toggling
        self.gesture_cooldown = 0.0
        self.long_blink_duration = 0.5 
        # --- END MODIFIED ---
    
    def toggle_visuals(self):
        self.show_visuals = not self.show_visuals
        return self.show_visuals
    
    def toggle_flip(self):
        self.flip = not self.flip
        return self.flip
    
    # (_nplm_to_xy, _center_of, _get_aspect_ratio, _eye_bbox, _normalize_by_eye, classify_gaze)
    # ... (These helper functions are unchanged) ...
    
    def _nplm_to_xy(self, lm, idx, w, h):
        """Convert normalized landmark to pixel coordinates"""
        p = lm[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)
    
    def _center_of(self, points):
        """Calculate center of points"""
        arr = np.array(points, dtype=np.float32)
        return arr.mean(axis=0)

    def _get_aspect_ratio(self, lm, v_indices, h_indices, w, h):
        """Helper to calculate aspect ratio (EAR or MAR)"""
        try:
            p_v1 = self._nplm_to_xy(lm, v_indices[0], w, h)
            p_v2 = self._nplm_to_xy(lm, v_indices[1], w, h)
            p_h1 = self._nplm_to_xy(lm, h_indices[0], w, h)
            p_h2 = self._nplm_to_xy(lm, h_indices[1], w, h)
            
            v_dist = np.linalg.norm(p_v1 - p_v2)
            h_dist = np.linalg.norm(p_h1 - p_h2)
            
            return v_dist / max(h_dist, 1e-6)
        except Exception:
            return 0.0

    def _eye_bbox(self, landmarks_pts):
        """Get bounding box of eye landmarks"""
        arr = np.array(landmarks_pts, dtype=np.float32)
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        return (x_min, y_min, x_max, y_max)
    
    def _normalize_by_eye(self, iris_center, eye_bbox):
        """Normalize iris position by eye bounding box"""
        x_min, y_min, x_max, y_max = eye_bbox
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        cx = (iris_center[0] - (x_min + w/2.0)) / w
        cy = (iris_center[1] - (y_min + h/2.0)) / h
        return np.array([cx, cy], dtype=np.float32)
    
    def classify_gaze(self, norm_offset):
        """Classify gaze direction with diagonal support"""
        x, y = norm_offset
        if abs(x) < self.deadzone and abs(y) < self.deadzone: return 'center'
        horiz = ''
        vert = ''
        if x < -self.horiz_thr: horiz = 'left'
        elif x > self.horiz_thr: horiz = 'right'
        if y < -self.vert_thr: vert = 'up'
        elif y > self.vert_thr: vert = 'down'
        if horiz and vert: return f"{horiz}-{vert}"
        return horiz or vert or 'center'


    def process_frame(self, frame, only_compute=False):
        """
        Process a frame and return annotated frame, normalized gaze, and gesture event.
        Returns: (annotated_frame, gaze_norm, gesture_event)
        """
        if self.flip:
            frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        gaze_norm = None
        gesture_event = None
        annotated = frame.copy()
        
        now = time.time()
        if self.prev_time is None:
            dt = 1/30.0
        else:
            dt = max(1e-6, now - self.prev_time)
        self.prev_time = now
        
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= dt
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # (Gaze Tracking logic remains the same)
            left_iris_pts = [self._nplm_to_xy(lm, i, w, h) for i in self.LEFT_IRIS]
            right_iris_pts = [self._nplm_to_xy(lm, i, w, h) for i in self.RIGHT_IRIS]
            left_center = self._center_of(left_iris_pts)
            right_center = self._center_of(right_iris_pts)
            left_corners = [self._nplm_to_xy(lm, self.LEFT_EYE_OUTER, w, h), self._nplm_to_xy(lm, self.LEFT_EYE_INNER, w, h)]
            right_corners = [self._nplm_to_xy(lm, self.RIGHT_EYE_OUTER, w, h), self._nplm_to_xy(lm, self.RIGHT_EYE_INNER, w, h)]
            left_bbox = self._eye_bbox(left_iris_pts + left_corners)
            right_bbox = self._eye_bbox(right_iris_pts + right_corners)
            left_norm = self._normalize_by_eye(left_center, left_bbox)
            right_norm = self._normalize_by_eye(right_center, right_bbox)
            avg_norm = (left_norm + right_norm) / 2.0
            
            self.buffer.append(avg_norm)
            if len(self.buffer) >= self.median_window:
                med = np.median(np.array(self.buffer), axis=0)
            else:
                med = avg_norm
            
            if self.smoothed is None: self.smoothed = med
            else: self.smoothed = self.smooth_alpha * self.smoothed + (1 - self.smooth_alpha) * med
            
            x_norm = (self.smoothed[0] * self.gaze_amplification) + 0.5
            y_norm = (self.smoothed[1] * self.gaze_amplification) + 0.5
            gaze_norm = (float(x_norm), float(y_norm))
            
            # (Saccade Detection logic remains the same)
            is_saccade = False
            saccade_speed = 0.0
            if self.prev is not None:
                delta = self.smoothed - self.prev
                speed = np.linalg.norm(delta) / dt
                is_saccade = speed > self.saccade_speed_thr
                if is_saccade: self.saccade_count += 1
                saccade_speed = speed
            self.prev = self.smoothed.copy()
            direction = self.classify_gaze(self.smoothed)
            
            # --- MODIFIED: Gesture Detection (EAR/MAR) ---
            left_ear = self._get_aspect_ratio(lm, self.LEFT_EYE_V, self.LEFT_EYE_H, w, h)
            right_ear = self._get_aspect_ratio(lm, self.RIGHT_EYE_V, self.RIGHT_EYE_H, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = self._get_aspect_ratio(lm, self.MOUTH_V, self.MOUTH_H, w, h)
            
            # Long Blink (Mode Switch)
            if avg_ear < self.blink_thresh:
                if not self.is_blinking:
                    self.is_blinking = True
                    self.blink_start_time = now
                elif (now - self.blink_start_time) > self.long_blink_duration and self.gesture_cooldown <= 0:
                    gesture_event = 'long_blink'
                    self.gesture_cooldown = 2.0  # 2-sec cooldown
                    self.is_blinking = False # Reset
            else:
                if self.is_blinking:
                    self.is_blinking = False
                    self.blink_start_time = None
            
            # --- NEW: Mouth Toggle (Pen Up/Down) ---
            if mar > self.mouth_thresh:
                # If mouth opens, we're not in cooldown, and we haven't already registered this "open"
                if self.gesture_cooldown <= 0 and not self.is_mouth_open:
                    gesture_event = 'mouth_toggle'
                    self.is_mouth_open = True  # Mark as "open" to prevent re-firing
                    self.gesture_cooldown = 1.5 # Cooldown to prevent rapid toggles
            else:
                # If mouth is closed and cooldown is over, reset the "open" state
                if self.is_mouth_open and self.gesture_cooldown <= 0:
                    self.is_mouth_open = False
            # --- END NEW ---

            # --- Visualization ---
            if not only_compute and self.show_visuals:
                # (Existing visualizations: iris, bbox, etc.)
                # ... (drawing logic is unchanged) ...
                for pt in left_iris_pts + right_iris_pts:
                    cv2.circle(annotated, tuple(pt.astype(int)), 1, (0, 255, 0), -1)
                cv2.circle(annotated, tuple(left_center.astype(int)), 3, (0, 0, 255), -1)
                cv2.circle(annotated, tuple(right_center.astype(int)), 3, (0, 0, 255), -1)
                lx1, ly1, lx2, ly2 = map(int, left_bbox); rx1, ry1, rx2, ry2 = map(int, right_bbox)
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 0, 0), 1)
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
                
                left_corner_center = self._center_of(left_corners)
                right_corner_center = self._center_of(right_corners)
                inter_ocular = np.linalg.norm(left_corner_center - right_corner_center)
                arrow_start = tuple(map(int, (left_corner_center + right_corner_center) / 2))
                vec = self.smoothed * inter_ocular * 0.8
                arrow_end = (int(arrow_start[0] + vec[0]), int(arrow_start[1] + vec[1]))
                cv2.arrowedLine(annotated, arrow_start, arrow_end, (0, 255, 255), 2, tipLength=0.3)
                
                # MODIFIED: Gesture info text
                gesture_txt = f"EAR: {avg_ear:.2f}  MAR: {mar:.2f}  Event: {gesture_event}"
                cv2.putText(annotated, gesture_txt, (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                txt = f"Dir: {direction}  Saccades: {self.saccade_count}  Speed: {saccade_speed:.3f}"
                cv2.putText(annotated, txt, (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
                
                if is_saccade:
                    cv2.putText(annotated, "SACCADE", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                cv2.putText(annotated, "Visuals: ON (press 'v' to toggle)", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (229, 24, 24), 1)
            elif not only_compute and not self.show_visuals:
                # (Minimal display logic is unchanged)
                cv2.putText(annotated, "Visuals: OFF (press 'v' to toggle)", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                txt = f"Dir: {direction}  Saccades: {self.saccade_count}  Speed: {saccade_speed:.3f}"
                cv2.putText(annotated, txt, (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
                if is_saccade:
                    cv2.putText(annotated, "SACCADE", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        return annotated, gaze_norm, gesture_event
    
    def release(self):
        """Release resources"""
        self.face_mesh.close()