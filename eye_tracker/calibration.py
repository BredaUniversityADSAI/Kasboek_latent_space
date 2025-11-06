import time
import cv2
import numpy as np
from utils import save_calibration

# Automatic calibration UI and sampling

def run_automatic_calibration(tracker, cap, overlay=True, prep_time=1.5, sample_time=2.0):
    # tracker: EyeTracker instance from tracking.py
    # cap: cv2.VideoCapture
    directions = ['center', 'left', 'right', 'up', 'down']
    calib = {}

    for dir_name in directions:
        # prep countdown
        start_capture = time.time() + prep_time
        samples = []
        while time.time() < start_capture + sample_time:
            ret, frame = cap.read()
            if not ret:
                continue
            if tracker.flip:
                disp = cv2.flip(frame.copy(), 1)
            else:
                disp = frame.copy()
            h, w = disp.shape[:2]

            # draw background or overlay
            if not overlay:
                disp = np.zeros_like(disp)

            # instruction and countdown
            prep_remaining = max(0, int(start_capture - time.time()) + 1)
            cv2.putText(disp, f"LOOK {dir_name.upper()} ({prep_remaining})", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

            # get gaze
            annotated, gaze = tracker.process_frame(frame)
            if gaze is not None:
                # convert normalized to pixels on display
                gx = int(gaze[0] * w)
                gy = int(gaze[1] * h)
                samples.append((gaze[0], gaze[1]))
                # draw sample point
                cv2.circle(disp, (gx, gy), 3, (255,0,0), -1)
                # draw cloud of previous samples
                for s in samples:
                    sx = int(s[0] * w)
                    sy = int(s[1] * h)
                    cv2.circle(disp, (sx, sy), 2, (200,100,255), -1)

            cv2.imshow('Calibration', disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('Calibration')
                return None

        # compute mean
        if samples:
            mean = np.mean(np.array(samples), axis=0).tolist()
            calib[dir_name] = mean
            # show final mean briefly
            for _ in range(30):
                ret, frame = cap.read()
                if not ret: continue
                if tracker.flip:
                    disp = cv2.flip(frame.copy(), 1)
                else:
                    disp = frame.copy()
                if not overlay:
                    disp = np.zeros_like(disp)
                mx = int(mean[0] * disp.shape[1])
                my = int(mean[1] * disp.shape[0])
                cv2.putText(disp, f"{dir_name.upper()} SAVED", (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
                cv2.circle(disp, (mx,my), 6, (0,255,0), -1)
                cv2.imshow('Calibration', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow('Calibration')
                    return None

    save_calibration(calib)
    cv2.destroyWindow('Calibration')
    return calib