"""
Visual display for eye tracking
(MODIFIED to accept face_results, handle NoneType, and draw status overlays)
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
        # We don't need the gaze_trail here anymore
        # self.gaze_trail = deque(maxlen=100)
        self.display_size = (400, 200) # Size for the eye region

    def extract_eye_region(self, frame, face_results):
        """Extract BOTH eyes in one region"""
        
        if not face_results or not face_results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        lm = face_results.multi_face_landmarks[0].landmark

        eye_landmarks = [
            33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155,
            362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382,
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
        # ... other filters ...
        return eye_img

    def create_eye_display(self, frame, face_results, filter_type="neon"):
        """
        Creates the 'Eyes' window content.
        If a face is found, it shows the filtered eye region.
        If not, it just shows the (flipped) camera frame.
        """
        eye_region = self.extract_eye_region(frame, face_results)

        if eye_region is None:
            # No face, just return the resized frame
            return cv2.resize(frame, self.display_size)
            
        eye_resized = cv2.resize(eye_region, self.display_size)
        filtered = self.apply_filter(eye_resized, filter_type)

        return filtered

    def add_status_overlay(self, eye_frame, text: str):
        """
        Draws a text overlay (e.g., "Searching...") on the 'Eyes' window.
        """
        h, w = eye_frame.shape[:2]
        
        # Semi-transparent black rectangle
        overlay = eye_frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        
        # Blend the overlay
        alpha = 0.6  # Transparency
        cv2.addWeighted(overlay, alpha, eye_frame, 1 - alpha, 0, eye_frame)
        
        # Put the text
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = (w - text_w) // 2
        text_y = h - 15
        
        cv2.putText(eye_frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return eye_frame

    def release(self):
        """Release resources"""
        self.eye_tracker.release()