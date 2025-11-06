"""
Collect gaze data from eye tracking
"""

import cv2
import time
from collections import deque
from eye_tracking import EyeTracker
from config import COLLECTION_SETTINGS


class GazeCollector:
    """Collect gaze data over specified duration"""

    def __init__(self):
        self.eye_tracker = EyeTracker(flip=True)
        self.duration = COLLECTION_SETTINGS["duration_seconds"]
        self.fps_estimate = COLLECTION_SETTINGS["fps_estimate"]

        max_frames = int(self.duration * self.fps_estimate * 1.5)
        self.buffer = deque(maxlen=max_frames)

    def collect(self, cap) -> list:
        """
        Collect gaze data for configured duration
        Returns list of gaze info dictionaries
        """
        self.buffer.clear()
        start_time = time.time()
        frame_count = 0
        valid_count = 0

        print(f"Collecting for {self.duration}s...")

        while time.time() - start_time < self.duration:
            ret, frame = cap.read()
            if not ret:
                print("!", end='', flush=True)
                continue

            if self.eye_tracker.flip:
                frame = cv2.flip(frame, 1)

            try:
                _, gaze_info = self.eye_tracker.process_frame(frame, only_compute=True)
                self.buffer.append(gaze_info)

                if gaze_info is not None:
                    valid_count += 1
                    print(".", end='', flush=True)
                else:
                    print("x", end='', flush=True)

            except Exception as e:
                self.buffer.append(None)
                print("e", end='', flush=True)

            frame_count += 1

            elapsed = time.time() - start_time
            if frame_count % 30 == 0:
                remaining = self.duration - elapsed
                print(f" [{elapsed:.0f}s/{self.duration:.0f}s, {valid_count} valid]", end='', flush=True)

            time.sleep(0.01)

        print(f"\nCollection complete: {valid_count}/{frame_count} valid frames")
        return list(self.buffer)

    def release(self):
        """Release resources"""
        self.eye_tracker.release()