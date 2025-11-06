"""
Visual display for eye tracking
"""

import cv2
import numpy as np
import time
from collections import deque
from eye_tracking import EyeTracker


class EyeVisualizer:
    """Create visual displays of eye tracking"""

    def __init__(self):
        self.eye_tracker = EyeTracker(flip=True)
        self.gaze_trail = deque(maxlen=100)

    def extract_eyes(self, frame, face_results):
        """Extract and crop individual eye regions"""
        if not face_results.multi_face_landmarks:
            return None, None

        h, w = frame.shape[:2]
        lm = face_results.multi_face_landmarks[0].landmark

        left_indices = [33, 133, 160, 159, 158, 157, 173, 144]
        right_indices = [362, 263, 387, 386, 385, 384, 398, 373]

        def crop_eye(indices, padding=30):
            points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
            if not points:
                return None

            xs, ys = zip(*points)
            x_min = max(0, min(xs) - padding)
            x_max = min(w, max(xs) + padding)
            y_min = max(0, min(ys) - padding)
            y_max = min(h, max(ys) + padding)

            return frame[y_min:y_max, x_min:x_max].copy()

        return crop_eye(left_indices), crop_eye(right_indices)

    def apply_filter(self, eye_img, filter_type="neon"):
        """Apply artistic filter to eye image"""
        if eye_img is None or eye_img.size == 0:
            return None

        if filter_type == "neon":
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[:, :, 1] = edges
            glow = cv2.GaussianBlur(edges_colored, (9, 9), 0)
            return cv2.addWeighted(eye_img, 0.6, glow, 0.8, 0)

        elif filter_type == "thermal":
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        return eye_img

    def create_eye_display(self, frame, filter_type="neon"):
        """Create display showing both eyes with filters"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.eye_tracker.face_mesh.process(rgb_frame)

        left_eye, right_eye = self.extract_eyes(frame, face_results)

        if left_eye is None or right_eye is None:
            return np.zeros((200, 400, 3), dtype=np.uint8)

        eye_size = (200, 150)
        left_resized = cv2.resize(left_eye, eye_size)
        right_resized = cv2.resize(right_eye, eye_size)

        left_filtered = self.apply_filter(left_resized, filter_type)
        right_filtered = self.apply_filter(right_resized, filter_type)

        return np.hstack([left_filtered, right_filtered])

    def draw_gaze_trail(self, canvas, gaze_info):
        """Draw live gaze trajectory"""
        if gaze_info is None:
            return canvas

        h, w = canvas.shape[:2]

        smoothed_norm = gaze_info.get('smoothed_norm')
        if smoothed_norm is not None:
            x = int((smoothed_norm[0] + 0.5) * w)
            y = int((smoothed_norm[1] + 0.5) * h)
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            self.gaze_trail.append((x, y, time.time()))

        current_time = time.time()
        points = [(x, y) for x, y, t in self.gaze_trail if current_time - t < 2.0]

        if len(points) > 1:
            for i in range(len(points) - 1):
                age = i / len(points)
                color = (int(255 * age), int(128 * age), int(255 * (1 - age)))
                thickness = max(1, int(3 * age))
                cv2.line(canvas, points[i], points[i + 1], color, thickness)

        if points:
            cv2.circle(canvas, points[-1], 8, (0, 255, 255), -1)
            cv2.circle(canvas, points[-1], 12, (255, 255, 255), 2)

        return canvas

    def create_gaze_display(self, gaze_info):
        """Create display showing gaze trajectory"""
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas = self.draw_gaze_trail(canvas, gaze_info)

        if gaze_info:
            direction = gaze_info.get('direction', 'N/A')
            speed = gaze_info.get('saccade_speed', 0)

            cv2.putText(canvas, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"Speed: {speed:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return canvas

    def release(self):
        """Release resources"""
        self.eye_tracker.release()