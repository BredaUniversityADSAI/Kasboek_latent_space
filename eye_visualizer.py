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

    def extract_eye_region(self, frame, face_results):
        """Extract BOTH eyes in one region"""
        if not face_results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        lm = face_results.multi_face_landmarks[0].landmark

        left_outer = 33
        right_outer = 263

        eye_landmarks = [
            # Left eye
            33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155,
            # Right eye
            362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382,
            # Bridge area
            168, 6, 197, 195
        ]

        points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_landmarks]

        if not points:
            return None

        xs, ys = zip(*points)
        padding_x = 40
        padding_y = 30

        x_min = max(0, min(xs) - padding_x)
        x_max = min(w, max(xs) + padding_x)
        y_min = max(0, min(ys) - padding_y)
        y_max = min(h, max(ys) + padding_y)

        eye_region = frame[y_min:y_max, x_min:x_max].copy()

        return eye_region

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

        elif filter_type == "ascii":
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            h, w = result.shape[:2]
            for y in range(0, h, 8):
                for x in range(0, w, 6):
                    brightness = gray[min(y, h-1), min(x, w-1)]
                    char = self._brightness_to_char(brightness)
                    cv2.putText(result, char, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
            return result

        return eye_img

    def _brightness_to_char(self, brightness):
        """Convert brightness to ASCII character"""
        chars = " .:!*oe%&#@"
        idx = int((brightness / 255) * (len(chars) - 1))
        return chars[idx]

    def create_eye_display(self, frame, filter_type="neon"):
        """Create display showing BOTH eyes in one frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.eye_tracker.face_mesh.process(rgb_frame)

        eye_region = self.extract_eye_region(frame, face_results)

        if eye_region is None:
            return np.zeros((200, 400, 3), dtype=np.uint8)
        display_size = (400, 200)
        eye_resized = cv2.resize(eye_region, display_size)
        filtered = self.apply_filter(eye_resized, filter_type)

        return filtered

    def draw_gaze_trail(self, canvas, gaze_info):
        """Draw live gaze trajectory"""
        if gaze_info is None:
            return canvas

        h, w = canvas.shape[:2]

        smoothed_norm = gaze_info.get('smoothed_norm')
        if smoothed_norm is not None:
            x = int((smoothed_norm[0] + 0.5) * w)
            y = int((smoothed_norm[1] + 0.5) * h)
            x = np.clip(x, 0, w-1)
            y = np.clip(y, 0, h-1)

            self.gaze_trail.append((x, y, time.time()))

        current_time = time.time()
        points = [(x, y) for x, y, t in self.gaze_trail if current_time - t < 2.0]

        if len(points) > 1:
            for i in range(len(points) - 1):
                age = i / len(points)
                color = (int(255 * age), int(128 * age), int(255 * (1-age)))
                thickness = max(1, int(3 * age))
                cv2.line(canvas, points[i], points[i+1], color, thickness)

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
