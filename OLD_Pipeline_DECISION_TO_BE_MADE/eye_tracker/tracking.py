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
        
        # Landmark indices
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
        self.NOSE_TIP = 1
        
        # Thresholds
        self.horiz_thr = 0.035
        self.vert_thr = 0.028
        self.deadzone = 0.018
        
        # Smoothing
        self.smooth_alpha = 0.6
        self.median_window = 5
        self.buffer = deque(maxlen=7)
        self.smoothed = None
        self.head_smooth_alpha = 0.4
        self.smoothed_head_pos = None
        
        # Saccade detection
        self.prev = None
        self.prev_time = None
        self.saccade_count = 0
        self.saccade_speed_thr = 0.2
        
        self.eye_movement_amplification = 2.0
        
        # Gesture detection
        self.blink_thresh = 0.25
        self.mouth_thresh = 0.4
        self.is_blinking = False
        self.blink_start_time = None
        self.is_mouth_open = False
        self.gesture_cooldown = 0.0
        self.long_blink_duration = 0.5
    
    def toggle_flip(self):
        self.flip = not self.flip
        return self.flip
    def _nplm_to_xy(self, lm, idx, w, h):
        p = lm[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)
    def _center_of(self, points):
        arr = np.array(points, dtype=np.float32)
        return arr.mean(axis=0)
    def _get_aspect_ratio(self, lm, v_indices, h_indices, w, h):
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
        arr = np.array(landmarks_pts, dtype=np.float32)
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        return (x_min, y_min, x_max, y_max)
    def _normalize_by_eye(self, iris_center, eye_bbox):
        x_min, y_min, x_max, y_max = eye_bbox
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        cx = (iris_center[0] - (x_min + w/2.0)) / w
        cy = (iris_center[1] - (y_min + h/2.0)) / h
        return np.array([cx, cy], dtype=np.float32)
    def classify_gaze(self, norm_offset):
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


    def process_frame(self, frame, display_toggles, only_compute=False):
        
        if self.flip:
            frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        final_gaze_norm = None
        raw_eye_offset = None
        gesture_event = None
        annotated = frame.copy()
        
        now = time.time()
        if self.prev_time is None: dt = 1/30.0
        else: dt = max(1e-6, now - self.prev_time)
        self.prev_time = now
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= dt
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # (1. Eye Gaze)
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
            if len(self.buffer) >= self.median_window: med = np.median(np.array(self.buffer), axis=0)
            else: med = avg_norm
            if self.smoothed is None: self.smoothed = med
            else: self.smoothed = self.smooth_alpha * self.smoothed + (1 - self.smooth_alpha) * med
            raw_eye_offset = self.smoothed.copy()
            
            # (2. Head Position)
            nose_tip_pt = self._nplm_to_xy(lm, self.NOSE_TIP, w, h)
            current_head_pos = np.array([nose_tip_pt[0] / w, nose_tip_pt[1] / h])
            if self.smoothed_head_pos is None: self.smoothed_head_pos = current_head_pos
            else: self.smoothed_head_pos = self.head_smooth_alpha * self.smoothed_head_pos + \
                                         (1- self.head_smooth_alpha) * current_head_pos

            # (3. Combine Head and Gaze)
            eye_offset = self.smoothed * self.eye_movement_amplification
            final_x = self.smoothed_head_pos[0] + eye_offset[0]
            final_y = self.smoothed_head_pos[1] + eye_offset[1]
            final_gaze_norm = (float(final_x), float(final_y))

            # (4. Saccade Detection)
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
            
            # (5. Gesture Detection)
            left_ear = self._get_aspect_ratio(lm, self.LEFT_EYE_V, self.LEFT_EYE_H, w, h)
            right_ear = self._get_aspect_ratio(lm, self.RIGHT_EYE_V, self.RIGHT_EYE_H, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = self._get_aspect_ratio(lm, self.MOUTH_V, self.MOUTH_H, w, h)
            if avg_ear < self.blink_thresh:
                if not self.is_blinking:
                    self.is_blinking = True
                    self.blink_start_time = now
                elif (now - self.blink_start_time) > self.long_blink_duration and self.gesture_cooldown <= 0:
                    gesture_event = 'long_blink'
                    self.gesture_cooldown = 2.0
                    self.is_blinking = False
            else:
                if self.is_blinking:
                    self.is_blinking = False
                    self.blink_start_time = None
            if mar > self.mouth_thresh:
                if self.gesture_cooldown <= 0 and not self.is_mouth_open:
                    gesture_event = 'mouth_toggle'
                    self.is_mouth_open = True
                    self.gesture_cooldown = 1.5
            else:
                if self.is_mouth_open and self.gesture_cooldown <= 0:
                    self.is_mouth_open = False

            # (6. Visualization (Tracking Rects))
            if not only_compute and display_toggles.get("tracking_rects", False):
                # Eye trackers
                for pt in left_iris_pts + right_iris_pts:
                    cv2.circle(annotated, tuple(pt.astype(int)), 1, (0, 255, 0), -1)
                cv2.circle(annotated, tuple(left_center.astype(int)), 3, (0, 0, 255), -1)
                cv2.circle(annotated, tuple(right_center.astype(int)), 3, (0, 0, 255), -1)
                lx1, ly1, lx2, ly2 = map(int, left_bbox); rx1, ry1, rx2, ry2 = map(int, right_bbox)
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 0, 0), 1)
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
                
                # Gaze arrow
                left_corner_center = self._center_of(left_corners)
                right_corner_center = self._center_of(right_corners)
                inter_ocular = np.linalg.norm(left_corner_center - right_corner_center)
                arrow_start = tuple(map(int, (left_corner_center + right_corner_center) / 2))
                vec = self.smoothed * inter_ocular * 0.8
                arrow_end = (int(arrow_start[0] + vec[0]), int(arrow_start[1] + vec[1]))
                cv2.arrowedLine(annotated, arrow_start, arrow_end, (0, 255, 255), 2, tipLength=0.3)
                
                # Head/Gesture trackers
                all_pts = np.array([self._nplm_to_xy(lm, i, w, h) for i in range(476)], dtype=np.int32)
                fx_min, fy_min = all_pts.min(axis=0)
                fx_max, fy_max = all_pts.max(axis=0)
                cv2.rectangle(annotated, (fx_min, fy_min), (fx_max, fy_max), (0, 255, 0), 1)
                cv2.circle(annotated, tuple(nose_tip_pt.astype(int)), 4, (0, 255, 0), -1)
                mouth_pts = np.array([self._nplm_to_xy(lm, i, w, h) for i in self.MOUTH_V + self.MOUTH_H], dtype=np.int32)
                mx_min, my_min = mouth_pts.min(axis=0)
                mx_max, my_max = mouth_pts.max(axis=0)
                cv2.rectangle(annotated, (mx_min, my_min), (mx_max, my_max), (0, 255, 255), 1)

            # (7. Visualization (Debug Text))
            if not only_compute and display_toggles.get("debug_text", False):
                txt = f"Dir: {direction}  Saccades: {self.saccade_count}  Speed: {saccade_speed:.3f}"
                cv2.putText(annotated, txt, (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
                
                gesture_txt = f"EAR: {avg_ear:.2f}  MAR: {mar:.2f}  Event: {gesture_event}"
                cv2.putText(annotated, gesture_txt, (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                if is_saccade:
                    cv2.putText(annotated, "SACCADE", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                mouth_status = "OPEN" if mar > self.mouth_thresh else "CLOSED"
                cv2.putText(annotated, f"Mouth: {mouth_status}", (10, h-100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Visualization (Usage Hints)
            if not only_compute and display_toggles.get("hint_text", False):
                # Get status
                v_stat = "ON" if display_toggles.get("tracking_rects") else "OFF"
                t_stat = "ON" if display_toggles.get("debug_text") else "OFF"
                m_stat = "ON" if display_toggles.get("misc_text") else "OFF"
                
                y_pos = 130
                font_size = 0.5
                color = (229, 24, 24)
                line_height = 20

                # Draw toggle statuses
                cv2.putText(annotated, f"Visuals (v): {v_stat}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, f"Debug Text (t): {t_stat}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, f"Misc Text (m): {m_stat}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                # Hint for itself
                cv2.putText(annotated, f"Hints (h): ON", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height + 10 # Add a gap

                # Add Controls Help Text
                cv2.putText(annotated, "--- Gestures ---", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "Long Blink: Toggle Mode", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "Open Mouth: Toggle Pen", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height + 10 # Add a gap
                
                cv2.putText(annotated, "--- Keyboard ---", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "c: Calibrate", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "d: Toggle Mode", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "r: Reset", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "s: Save", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "f: Flip", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
                y_pos += line_height
                cv2.putText(annotated, "q: Quit", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)

        return annotated, final_gaze_norm, gesture_event, raw_eye_offset
    
    def release(self):
        self.face_mesh.close()