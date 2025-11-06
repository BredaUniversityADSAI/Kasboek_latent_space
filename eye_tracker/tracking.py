import cv2
import numpy as np
import mediapipe as mp
import time

# tracking.py
# Exposes EyeTracker class with simple API:
#   tracker = EyeTracker(flip=True)
#   annotated_frame, gaze_norm = tracker.process_frame(frame)
#   tracker.release()

class EyeTracker:
    def __init__(self, flip=True):
        self.flip = flip
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        # indices for landmarks
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [473, 474, 475, 476]
        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER = 362
        self.RIGHT_EYE_INNER = 263
        # smoothing
        self.smoothed = None
        self.alpha = 0.6

    def _nplm_to_xy(self, lm, idx, w, h):
        p = lm[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    def _center_of(self, pts):
        return np.mean(np.array(pts, dtype=np.float32), axis=0)

    def _eye_bbox(self, pts):
        arr = np.array(pts, dtype=np.float32)
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        return x_min, y_min, x_max, y_max

    def _normalize_by_eye(self, iris_center, bbox):
        x_min, y_min, x_max, y_max = bbox
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        cx = (iris_center[0] - (x_min + w/2.0)) / w
        cy = (iris_center[1] - (y_min + h/2.0)) / h
        # convert from [-0.5..0.5] to [0..1] for easier normalization across modules
        return np.array([cx + 0.5, cy + 0.5], dtype=np.float32)

    def process_frame(self, frame):
        # returns annotated_frame, (x_norm, y_norm) or None
        if self.flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        gaze_norm = None
        annotated = frame.copy()

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_iris_pts = [self._nplm_to_xy(lm, i, w, h) for i in self.LEFT_IRIS]
            right_iris_pts = [self._nplm_to_xy(lm, i, w, h) for i in self.RIGHT_IRIS]

            left_center = self._center_of(left_iris_pts)
            right_center = self._center_of(right_iris_pts)
            left_corners = [self._nplm_to_xy(lm, self.LEFT_EYE_OUTER, w, h),
                            self._nplm_to_xy(lm, self.LEFT_EYE_INNER, w, h)]
            right_corners = [self._nplm_to_xy(lm, self.RIGHT_EYE_OUTER, w, h),
                             self._nplm_to_xy(lm, self.RIGHT_EYE_INNER, w, h)]

            left_bbox = self._eye_bbox(left_iris_pts + left_corners)
            right_bbox = self._eye_bbox(right_iris_pts + right_corners)

            left_norm = self._normalize_by_eye(left_center, left_bbox)
            right_norm = self._normalize_by_eye(right_center, right_bbox)

            avg_norm = (left_norm + right_norm) / 2.0

            # smoothing
            if self.smoothed is None:
                self.smoothed = avg_norm
            else:
                self.smoothed = self.alpha * self.smoothed + (1 - self.alpha) * avg_norm

            gaze_norm = (float(self.smoothed[0]), float(self.smoothed[1]))

            # draw small debug points
            for p in left_iris_pts + right_iris_pts:
                cv2.circle(annotated, tuple(p.astype(int)), 1, (0, 255, 0), -1)
            lc = tuple(left_center.astype(int))
            rc = tuple(right_center.astype(int))
            cv2.circle(annotated, lc, 3, (0, 0, 255), -1)
            cv2.circle(annotated, rc, 3, (0, 0, 255), -1)

        return annotated, gaze_norm

    def release(self):
        self.face_mesh.close()