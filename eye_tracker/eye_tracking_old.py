"""
Improved Eye tracker in Python using MediaPipe Face Mesh and OpenCV

Features (improvements vs previous version):
- Uses a small named landmark map instead of scattering magic numbers.
- Flips webcam feed by default (mirror view) and offers toggle.
- Computes gaze as the iris position relative to an eye bounding box (gives better vertical sensitivity).
- Different thresholds for horizontal and vertical axes and a deadzone to reduce "up vs center" noise.
- Rolling median + exponential smoothing to remove residual jitter and reduce spurious diagonals.
- Calibration routine to capture a neutral gaze (center) and optional extremes to auto-tune thresholds.
- Improved classification including diagonal directions (left-up, left-down, right-up, right-down).
- Logs saccades and allows saving saccade timestamps to CSV (toggleable).

Requirements:
- Python 3.8+
- pip install opencv-python mediapipe numpy

Run:
$ python eye_tracker.py

Controls while running:
- q : quit
- c : run a short calibration (looks for a neutral center sample)
- f : toggle frame flip (mirror)
- s : toggle saccade logging to CSV

Notes:
- Calibration helps a lot: sit naturally looking at the camera and press 'c'.
- If vertical detection is still unreliable, run calibration and look slightly up/down while calibrating extremes (future enhancement).

"""

import cv2
import time
import numpy as np
import csv
from collections import deque

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required") from e

FACEMESH_LANDMARKS = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 362,
    "right_eye_inner": 263,
    "left_iris": [468, 469, 470, 471],
    "right_iris": [473, 474, 475, 476],
}

DEFAULT_HORIZ_THRESHOLD = 0.035  # normalized fraction of inter-ocular width
DEFAULT_VERT_THRESHOLD = 0.028   # vertical sensitivity (often smaller than horizontal)
DEADZONE = 0.018                  # no-motion deadzone - avoids center vs tiny up/down
SACCADE_SPEED_THR = 0.2          # units / second
SMOOTH_ALPHA = 0.6               # exponential smoothing
MEDIAN_WINDOW = 5                # rolling median window to reduce spikes
ROLLING_BUFFER = 7               # buffer size for median smoothing


def nplm_to_xy(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def center_of(points):
    arr = np.array(points, dtype=np.float32)
    return arr.mean(axis=0)


def eye_bbox(landmarks_pts):
    arr = np.array(landmarks_pts, dtype=np.float32)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    return (x_min, y_min, x_max, y_max)


def normalize_by_eye(iris_center, eye_bbox_wh):
    x_min, y_min, x_max, y_max = eye_bbox_wh
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    cx = (iris_center[0] - (x_min + w/2.0)) / w
    cy = (iris_center[1] - (y_min + h/2.0)) / h
    return np.array([cx, cy], dtype=np.float32)


def classify_gaze(norm_offset, horiz_thr, vert_thr, deadzone):
    x, y = norm_offset
    if abs(x) < deadzone and abs(y) < deadzone:
        return 'center'

    horiz = ''
    vert = ''
    if x < -horiz_thr:
        horiz = 'left'
    elif x > horiz_thr:
        horiz = 'right'

    if y < -vert_thr:
        vert = 'up'
    elif y > vert_thr:
        vert = 'down'

    if horiz and vert:
        return f"{horiz}-{vert}"
    return horiz or vert or 'center'


class EyeTracker:
    def __init__(self, flip=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        self.flip = flip
        self.buffer = deque(maxlen=ROLLING_BUFFER)
        self.smoothed = None
        self.prev = None
        self.prev_time = None
        self.saccade_count = 0
        self.horiz_thr = DEFAULT_HORIZ_THRESHOLD
        self.vert_thr = DEFAULT_VERT_THRESHOLD
        self.logging = False
        self.logfile = None

    def toggle_flip(self):
        self.flip = not self.flip

    def toggle_logging(self):
        self.logging = not self.logging
        if self.logging:
            self.logfile = open('saccades.csv', 'w', newline='')
            self.csvw = csv.writer(self.logfile)
            self.csvw.writerow(['timestamp', 'speed', 'direction'])
        else:
            if self.logfile:
                self.logfile.close()
                self.logfile = None

    def calibrate_center(self, samples=30, cap=None):
        """Collect samples while user looks center and set deadzone based on observed variance."""
        if cap is None:
            return
        collected = []
        print("Calibration: please look at the center of the screen for ~2 seconds...")
        for _ in range(samples):
            ret, frame = cap.read()
            if not ret:
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            _, info = self.process_frame(frame, only_compute=True)
            if info is not None:
                collected.append(info['eye_norm'])
            cv2.waitKey(30)
        if collected:
            arr = np.array(collected)
            stds = arr.std(axis=0)
            self.horiz_thr = max(self.horiz_thr, stds[0] * 3.0)
            self.vert_thr = max(self.vert_thr, stds[1] * 3.0)
            global DEADZONE
            DEADZONE = max(DEADZONE, np.mean(stds) * 2.0)
            print(f"Calibration done. horiz_thr={self.horiz_thr:.4f}, vert_thr={self.vert_thr:.4f}, deadzone={DEADZONE:.4f}")
        else:
            print("Calibration failed — no face detected. Try again and ensure good lighting.")

    def process_frame(self, frame, only_compute=False):
        h, w = frame.shape[:2]
        img = frame.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        gaze_info = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_iris_pts = [nplm_to_xy(lm, i, w, h) for i in FACEMESH_LANDMARKS['left_iris']]
            right_iris_pts = [nplm_to_xy(lm, i, w, h) for i in FACEMESH_LANDMARKS['right_iris']]
            left_center = center_of(left_iris_pts)
            right_center = center_of(right_iris_pts)

            left_corners = [nplm_to_xy(lm, FACEMESH_LANDMARKS['left_eye_outer'], w, h),
                            nplm_to_xy(lm, FACEMESH_LANDMARKS['left_eye_inner'], w, h)]
            right_corners = [nplm_to_xy(lm, FACEMESH_LANDMARKS['right_eye_outer'], w, h),
                             nplm_to_xy(lm, FACEMESH_LANDMARKS['right_eye_inner'], w, h)]

            left_eye_bbox = eye_bbox(left_iris_pts + left_corners)
            right_eye_bbox = eye_bbox(right_iris_pts + right_corners)

            left_norm = normalize_by_eye(left_center, left_eye_bbox)
            right_norm = normalize_by_eye(right_center, right_eye_bbox)

            avg_norm = (left_norm + right_norm) / 2.0
            raw_norm = avg_norm.copy()

            self.buffer.append(avg_norm)
            med = np.median(np.array(self.buffer), axis=0)
            if self.smoothed is None:
                self.smoothed = med
            else:
                self.smoothed = SMOOTH_ALPHA * self.smoothed + (1 - SMOOTH_ALPHA) * med

            left_corner_center = center_of(left_corners)
            right_corner_center = center_of(right_corners)
            inter_ocular = np.linalg.norm(left_corner_center - right_corner_center)

            now = time.time()
            if self.prev_time is None:
                dt = 1/30.0
            else:
                dt = max(1e-6, now - self.prev_time)
            self.prev_time = now

            if self.prev is not None:
                delta = self.smoothed - self.prev
                speed = np.linalg.norm(delta) / dt
                is_saccade = speed > SACCADE_SPEED_THR
                if is_saccade:
                    self.saccade_count += 1
                    if self.logging and self.logfile:
                        self.csvw.writerow([time.time(), speed, classify_gaze(self.smoothed, self.horiz_thr, self.vert_thr, DEADZONE)])
                sacc_speed = speed
            else:
                is_saccade = False
                sacc_speed = 0.0

            self.prev = self.smoothed.copy()

            direction = classify_gaze(self.smoothed, self.horiz_thr, self.vert_thr, DEADZONE)

            gaze_info = {
                'left_center': left_center,
                'right_center': right_center,
                'left_bbox': left_eye_bbox,
                'right_bbox': right_eye_bbox,
                'eye_norm': raw_norm,
                'smoothed_norm': self.smoothed.copy(),
                'direction': direction,
                'saccade': is_saccade,
                'saccade_speed': float(sacc_speed),
                'saccade_count': int(self.saccade_count),
                'inter_ocular': float(inter_ocular)
            }

            if only_compute:
                return frame, gaze_info

            draw = frame
            for pt in left_iris_pts + right_iris_pts:
                cv2.circle(draw, tuple(pt.astype(int)), 1, (0, 255, 0), -1)
            cv2.circle(draw, tuple(left_center.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(draw, tuple(right_center.astype(int)), 3, (0, 0, 255), -1)
            lx1, ly1, lx2, ly2 = map(int, left_eye_bbox)
            rx1, ry1, rx2, ry2 = map(int, right_eye_bbox)
            cv2.rectangle(draw, (lx1, ly1), (lx2, ly2), (255, 0, 0), 1)
            cv2.rectangle(draw, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)

            center_screen = (w // 2, h // 2)
            arrow_start = (int((left_corner_center[0] + right_corner_center[0]) / 2), int((left_corner_center[1] + right_corner_center[1]) / 2))
            vec = np.array([self.smoothed[0] * inter_ocular * 0.8, self.smoothed[1] * inter_ocular * 0.8])
            arrow_end = (int(arrow_start[0] + vec[0]), int(arrow_start[1] + vec[1]))
            cv2.arrowedLine(draw, arrow_start, arrow_end, (0, 255, 255), 2, tipLength=0.3)

            txt = f"Dir: {direction}  Saccades: {self.saccade_count}  Speed: {gaze_info['saccade_speed']:.3f}"
            cv2.putText(draw, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            if gaze_info['saccade']:
                cv2.putText(draw, "SACCADE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            return draw, gaze_info

        return frame, None

    def release(self):
        self.face_mesh.close()
        if self.logfile:
            self.logfile.close()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Check permissions or device.")
        return

    tracker = EyeTracker(flip=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if tracker.flip:
                frame = cv2.flip(frame, 1)

            out_frame, info = tracker.process_frame(frame)

            cv2.imshow('Eye Tracker (q=quit, c=calibrate, f=flip, s=log)', out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                tracker.toggle_flip()
            elif key == ord('s'):
                tracker.toggle_logging()
            elif key == ord('c'):
                tracker.calibrate_center(cap=cap)

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()