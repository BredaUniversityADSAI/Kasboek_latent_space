"""
Eye tracker in Python using MediaPipe Face Mesh and OpenCV

Features:
- Tracks eyes and iris positions using MediaPipe Face Mesh
- Displays gaze direction (left / right / up / down / center) on the video frame
- Detects fast eye movements (saccades) by measuring gaze speed

Requirements:
- Python 3.8+
- pip install opencv-python mediapipe numpy

Usage:
$ python eye_tracker.py

Notes:
- Works best with a webcam and good lighting.
- Calibration thresholds are heuristic — you can adjust `GAZE_THRESHOLD` and `SACCADE_SPEED_THR`.

"""

import cv2
import time
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required. Install with: pip install mediapipe") from e

# -------------------- Configuration --------------------
# Indices based on MediaPipe Face Mesh (these are common indices used for iris + eye corners)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# iris landmarks (approx indices for MediaPipe 'refined' face mesh with iris)
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# thresholds - tweak these to taste
GAZE_THRESHOLD = 0.035   # normalized offset to consider "looking left/right/up/down"
SACCADE_SPEED_THR = 0.15 # normalized units per second considered a fast movement (saccade)
SMOOTHING_ALPHA = 0.6    # smoothing factor for gaze position (exponential)

# -------------------- Helpers --------------------

def normalized_landmark_to_point(landmark, image_w, image_h):
    return np.array([landmark.x * image_w, landmark.y * image_h], dtype=np.float32)


def get_eye_landmarks(lm, indices, w, h):
    return [normalized_landmark_to_point(lm[i], w, h) for i in indices]


# compute center as mean of points
def center_of(points):
    pts = np.array(points, dtype=np.float32)
    return pts.mean(axis=0)


# convert raw eye vector into simple categorical direction
def gaze_direction(offset_norm, thr=GAZE_THRESHOLD):
    x, y = offset_norm
    if abs(x) < thr and abs(y) < thr:
        return "center"
    horiz = "left" if x < -thr else ("right" if x > thr else "")
    vert = "up" if y < -thr else ("down" if y > thr else "")
    if horiz and vert:
        return f"{horiz}-{vert}"
    return horiz or vert


# normalize offset relative to inter-ocular distance (distance between eye centers)
def normalize_offset(offset_pixels, inter_ocular_dist):
    if inter_ocular_dist <= 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return offset_pixels / inter_ocular_dist


# -------------------- Main Eye Tracker Class --------------------
class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks gives iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        self.prev_gaze = None
        self.prev_time = None
        self.smoothed_gaze = None
        self.saccade_count = 0

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        draw = frame.copy()
        gaze_info = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # collect points
            left_iris_pts = get_eye_landmarks(lm, LEFT_IRIS, w, h)
            right_iris_pts = get_eye_landmarks(lm, RIGHT_IRIS, w, h)

            left_eye_corners = get_eye_landmarks(lm, [LEFT_EYE_OUTER, LEFT_EYE_INNER], w, h)
            right_eye_corners = get_eye_landmarks(lm, [RIGHT_EYE_OUTER, RIGHT_EYE_INNER], w, h)

            left_center = center_of(left_iris_pts)
            right_center = center_of(right_iris_pts)
            left_corner_center = center_of(left_eye_corners)
            right_corner_center = center_of(right_eye_corners)

            # inter-ocular distance (pixel)
            inter_ocular = np.linalg.norm(left_corner_center - right_corner_center)

            # compute offset of iris centers relative to eye corner centers (where in the eye the iris sits)
            left_offset = left_center - left_corner_center
            right_offset = right_center - right_corner_center

            # normalize offsets by inter-ocular distance so values are scale invariant
            left_off_norm = normalize_offset(left_offset, inter_ocular)
            right_off_norm = normalize_offset(right_offset, inter_ocular)

            # average both eyes
            avg_offset = (left_off_norm + right_off_norm) / 2.0

            # smoothing
            now = time.time()
            if self.prev_time is None:
                dt = 1/30
            else:
                dt = max(1e-6, now - self.prev_time)
            self.prev_time = now

            if self.smoothed_gaze is None:
                self.smoothed_gaze = avg_offset
            else:
                self.smoothed_gaze = SMOOTHING_ALPHA * self.smoothed_gaze + (1-SMOOTHING_ALPHA) * avg_offset

            # saccade detection: gaze speed (norm of change per second)
            if self.prev_gaze is not None:
                gaze_change = self.smoothed_gaze - self.prev_gaze
                speed = np.linalg.norm(gaze_change) / dt
                is_saccade = speed > SACCADE_SPEED_THR
                if is_saccade:
                    self.saccade_count += 1
            else:
                speed = 0.0
                is_saccade = False

            self.prev_gaze = self.smoothed_gaze.copy()

            direction = gaze_direction(self.smoothed_gaze)

            gaze_info = {
                'left_center': left_center,
                'right_center': right_center,
                'left_corners': left_eye_corners,
                'right_corners': right_eye_corners,
                'avg_offset': self.smoothed_gaze,
                'direction': direction,
                'saccade': is_saccade,
                'saccade_speed': float(speed),
                'saccade_count': int(self.saccade_count),
                'inter_ocular': float(inter_ocular)
            }

            # ---- Drawing ----
            # draw iris centers and eye corner centers
            for pt in left_iris_pts + right_iris_pts:
                cv2.circle(draw, tuple(pt.astype(int)), 1, (0, 255, 0), -1)

            cv2.circle(draw, tuple(left_center.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(draw, tuple(right_center.astype(int)), 3, (0, 0, 255), -1)

            cv2.circle(draw, tuple(left_corner_center.astype(int)), 3, (255, 0, 0), -1)
            cv2.circle(draw, tuple(right_corner_center.astype(int)), 3, (255, 0, 0), -1)

            # draw arrows showing gaze direction for each eye
            # arrow from eye corner center pointing toward iris center (scaled)
            for eye_center, iris_center in [(left_corner_center, left_center), (right_corner_center, right_center)]:
                start = tuple(eye_center.astype(int))
                vec = iris_center - eye_center
                end = tuple((eye_center + vec * 3).astype(int))  # scaled a bit for visibility
                cv2.arrowedLine(draw, start, end, (0, 255, 255), 2, tipLength=0.3)

            # draw text overlay
            txt = f"Direction: {direction}  Saccades: {self.saccade_count}  Speed: {self.saccade_speed_str(gaze_info['saccade_speed'])}"
            cv2.putText(draw, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

            if gaze_info['saccade']:
                cv2.putText(draw, "SACCADE DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        return draw, gaze_info

    def saccade_speed_str(self, speed):
        return f"{speed:.3f}"

    def release(self):
        self.face_mesh.close()


# -------------------- Run from Webcam --------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. If you're on Linux, try: sudo modprobe uvcvideo or check camera permissions.")
        return

    tracker = EyeTracker()

    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Mirror horizontally

            if not ret:
                print("Failed to grab frame")
                break

            out_frame, info = tracker.process_frame(frame)

            # show helper guide in window center
            h, w = out_frame.shape[:2]
            cv2.line(out_frame, (w//2 - 40, 50), (w//2 + 40, 50), (255,255,255), 1)

            cv2.imshow('Eye Tracker (press q to quit)', out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
