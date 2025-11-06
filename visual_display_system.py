"""
Visual Display System for Eye Tracking
Shows cropped eyes with filters and live gaze drawing
Integrates with your existing filter system
"""

import cv2
import numpy as np
from eye_tracking import EyeTracker
import time
from collections import deque


class EyeVisualizer:
    """
    Creates visual displays of eye tracking
    - Cropped eyes with artistic filters
    - Live gaze trajectory drawing
    """

    def __init__(self):
        self.eye_tracker = EyeTracker(flip=True)
        self.gaze_trail = deque(maxlen=100)  # Store last 100 gaze points

    def extract_eye_regions(self, frame, face_results):
        """
        Extract and crop individual eye regions from face
        """
        if not face_results.multi_face_landmarks:
            return None, None

        h, w = frame.shape[:2]
        lm = face_results.multi_face_landmarks[0].landmark

        # Left eye landmarks (approximate region)
        left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 144]
        # Right eye landmarks
        right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 373]

        def get_eye_crop(indices, padding=30):
            """Get bounding box around eye with padding"""
            points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
            if not points:
                return None

            xs, ys = zip(*points)
            x_min, x_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
            y_min, y_max = max(0, min(ys) - padding), min(h, max(ys) + padding)

            return frame[y_min:y_max, x_min:x_max].copy()

        left_eye = get_eye_crop(left_eye_indices)
        right_eye = get_eye_crop(right_eye_indices)

        return left_eye, right_eye

    def apply_artistic_filter(self, eye_img, filter_type="neon"):
        """
        Apply artistic filter to eye image
        """
        if eye_img is None or eye_img.size == 0:
            return None

        if filter_type == "neon":
            # Neon glow effect
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[:, :, 1] = edges  # Green channel
            glow = cv2.GaussianBlur(edges_colored, (9, 9), 0)
            result = cv2.addWeighted(eye_img, 0.6, glow, 0.8, 0)
            return result

        elif filter_type == "thermal":
            # Thermal vision
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return thermal

        elif filter_type == "ascii":
            # ASCII art style
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Add text overlay effect
            h, w = result.shape[:2]
            for y in range(0, h, 8):
                for x in range(0, w, 6):
                    brightness = gray[min(y, h - 1), min(x, w - 1)]
                    char = self._brightness_to_char(brightness)
                    cv2.putText(result, char, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
            return result

        else:
            return eye_img

    def _brightness_to_char(self, brightness):
        """Convert brightness to ASCII character"""
        chars = " .:!*oe%&#@"
        idx = int((brightness / 255) * (len(chars) - 1))
        return chars[idx]

    def draw_gaze_trajectory(self, canvas, gaze_info):
        """
        Draw live gaze trajectory on canvas
        """
        if gaze_info is None:
            return canvas

        h, w = canvas.shape[:2]

        # Get current gaze position (normalized to screen space)
        smoothed_norm = gaze_info.get('smoothed_norm')
        if smoothed_norm is not None:
            # Convert normalized coords to screen coords
            x = int((smoothed_norm[0] + 0.5) * w)
            y = int((smoothed_norm[1] + 0.5) * h)
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            # Add to trail
            self.gaze_trail.append((x, y, time.time()))

        # Draw trail
        current_time = time.time()
        points = [(x, y) for x, y, t in self.gaze_trail if current_time - t < 2.0]

        if len(points) > 1:
            # Draw lines between points
            for i in range(len(points) - 1):
                # Color gradient: newer = brighter
                age = i / len(points)
                color = (
                    int(255 * age),  # Blue
                    int(128 * age),  # Green
                    int(255 * (1 - age))  # Red
                )
                thickness = max(1, int(3 * age))
                cv2.line(canvas, points[i], points[i + 1], color, thickness)

        # Draw current position
        if points:
            cv2.circle(canvas, points[-1], 8, (0, 255, 255), -1)
            cv2.circle(canvas, points[-1], 12, (255, 255, 255), 2)

        return canvas

    def create_eye_display(self, frame, filter_type="neon"):
        """
        Create display showing both eyes with filters side by side
        """
        # Process frame for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.eye_tracker.face_mesh.process(rgb_frame)

        # Extract eyes
        left_eye, right_eye = self.extract_eye_regions(frame, face_results)

        if left_eye is None or right_eye is None:
            # Return black screen if no eyes found
            return np.zeros((200, 400, 3), dtype=np.uint8)

        eye_size = (200, 150)
        left_eye_resized = cv2.resize(left_eye, eye_size)
        right_eye_resized = cv2.resize(right_eye, eye_size)

        left_filtered = self.apply_artistic_filter(left_eye_resized, filter_type)
        right_filtered = self.apply_artistic_filter(right_eye_resized, filter_type)
        combined = np.hstack([left_filtered, right_filtered])

        return combined

    def create_gaze_display(self, gaze_info):
        """
        Create display showing gaze trajectory
        """
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas = self.draw_gaze_trajectory(canvas, gaze_info)
        if gaze_info:
            direction = gaze_info.get('direction', 'N/A')
            speed = gaze_info.get('saccade_speed', 0)

            cv2.putText(canvas, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f"Speed: {speed:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return canvas

    def release(self):
        """Clean up resources"""
        self.eye_tracker.release()


def run_visual_display_system():
    """
    Main function to run the visual display system
    Shows two windows: cropped eyes and gaze trajectory
    """
    visualizer = EyeVisualizer()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print("Visual Display System Running")
    print("Press 'q' to quit")
    print("Press '1', '2', '3' to change eye filter")

    filter_types = ["neon", "thermal", "ascii"]
    current_filter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if visualizer.eye_tracker.flip:
                frame = cv2.flip(frame, 1)

            # Process frame for gaze
            _, gaze_info = visualizer.eye_tracker.process_frame(frame, only_compute=True)

            # Create displays
            eye_display = visualizer.create_eye_display(
                frame,
                filter_type=filter_types[current_filter]
            )
            gaze_display = visualizer.create_gaze_display(gaze_info)

            # Show windows
            cv2.imshow('Eyes with Filter', eye_display)
            cv2.imshow('Gaze Trajectory', gaze_display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_filter = 0
                print(f"Filter: {filter_types[current_filter]}")
            elif key == ord('2'):
                current_filter = 1
                print(f"Filter: {filter_types[current_filter]}")
            elif key == ord('3'):
                current_filter = 2
                print(f"Filter: {filter_types[current_filter]}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        visualizer.release()


if __name__ == "__main__":
    run_visual_display_system()