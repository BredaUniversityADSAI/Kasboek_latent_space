import numpy as np
import cv2
import time
from utils import norm_to_pixel

class GazePainter:
    def __init__(self, width=1280, height=720, dwell_radius=0.03):
        # canvas normalized size uses provided width/height
        self.w = width
        self.h = height
        self.canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        self.prev_point = None
        self.prev_time = None
        self.dwell_time = 0.0
        self.dwell_radius = dwell_radius  # in normalized coords
        self.stroke_color = (0, 0, 255)

    def clear(self):
        self.canvas[:] = 255
        self.prev_point = None
        self.prev_time = None
        self.dwell_time = 0.0

    def save(self, path='gaze_art.png'):
        cv2.imwrite(path, self.canvas)

    def update(self, gaze_norm, dt):
        # gaze_norm is (x_norm, y_norm) or None
        if gaze_norm is None:
            self.prev_point = None
            self.dwell_time = 0.0
            return

        # map to pixel on canvas
        gx = int(np.clip(gaze_norm[0],0,1) * self.w)
        gy = int(np.clip(gaze_norm[1],0,1) * self.h)
        cur = (gx, gy)

        if self.prev_point is None:
            self.prev_point = cur
            self.prev_time = time.time()
            self.dwell_time = 0.0
            return

        # distance in normalized space: approximate by dividing pixel dist by canvas diagonal
        dx = (cur[0] - self.prev_point[0]) / float(self.w)
        dy = (cur[1] - self.prev_point[1]) / float(self.h)
        dist = np.hypot(dx, dy)

        if dist < self.dwell_radius:
            self.dwell_time += dt
        else:
            self.dwell_time = 0.0

        # line width grows with dwell time
        width = int(min(40, 1 + self.dwell_time * 12))
        cv2.line(self.canvas, self.prev_point, cur, self.stroke_color, width)
        self.prev_point = cur

    def get_display(self, frame=None, overlay=False):
        # returns image to show: if overlay and frame provided, blend; else return canvas
        if overlay and frame is not None:
            # resize canvas to frame
            ch, cw = frame.shape[:2]
            resized = cv2.resize(self.canvas, (cw, ch))
            blended = cv2.addWeighted(frame, 0.6, resized, 0.4, 0)
            return blended
        else:
            return self.canvas