"""
Visual display system for eye tracking
"""

import cv2
from eye_visualizer import EyeVisualizer


def run_display():
    """Run visual display system"""

    visualizer = EyeVisualizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Visual display running")
    print("Press 'q' to quit, 1/2/3 to change filter")

    filter_types = ["neon", "thermal", "ascii"]
    current_filter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if visualizer.eye_tracker.flip:
                frame = cv2.flip(frame, 1)

            _, gaze_info = visualizer.eye_tracker.process_frame(frame, only_compute=True)

            eye_display = visualizer.create_eye_display(frame, filter_types[current_filter])
            gaze_display = visualizer.create_gaze_display(gaze_info)

            cv2.imshow('Eyes', eye_display)
            cv2.imshow('Gaze', gaze_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_filter = 0
            elif key == ord('2'):
                current_filter = 1
            elif key == ord('3'):
                current_filter = 2

    finally:
        cap.release()
        cv2.destroyAllWindows()
        visualizer.release()


if __name__ == "__main__":
    run_display()