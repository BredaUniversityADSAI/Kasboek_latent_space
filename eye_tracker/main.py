import cv2
import time
import numpy as np
from tracking import EyeTracker
from calibration import run_automatic_calibration
from drawing import GazePainter
from utils import load_calibration

# State constants
STATE_TRACKING = 1
STATE_CALIBRATING = 2
STATE_DRAWING = 3

# Config flags
DRAWING_IN_SEPARATE_WINDOW = True
CALIBRATION_IS_OVERLAY = True

# Keys: c = calibrate, d = toggle drawing, r = reset (calib or clear), s = save art, q = quit


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    tracker = EyeTracker(flip=True)
    painter = GazePainter(width=1280, height=720)
    calibration = load_calibration()

    current_state = STATE_TRACKING
    last_time = time.time()

    print("Controls: c=calibrate, d=toggle drawing, r=reset, s=save (in drawing), q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            dt = now - last_time
            last_time = now

            annotated, gaze = tracker.process_frame(frame)

            # if calibrated, you might use calibration dict in classify or mapping. For drawing we use raw normalized gaze.

            if current_state == STATE_TRACKING:
                display = annotated.copy()
                cv2.putText(display, 'MODE: TRACKING', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2)
                if DRAWING_IN_SEPARATE_WINDOW:
                    cv2.imshow('Eye Tracker', display)
                else:
                    cv2.imshow('Eye Tracker', display)

            elif current_state == STATE_CALIBRATING:
                # calibration runs in its own function; if invoked it will block until finished
                pass

            elif current_state == STATE_DRAWING:
                # update painter and display accordingly
                painter.update(gaze, dt)
                if DRAWING_IN_SEPARATE_WINDOW:
                    cv2.imshow('Eye Tracker', annotated)
                    canvas_img = painter.get_display(None, overlay=False)
                    cv2.imshow('Gaze Canvas', canvas_img)
                else:
                    blended = painter.get_display(annotated, overlay=True)
                    cv2.imshow('Eye Tracker', blended)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_state = STATE_CALIBRATING
                # run calibration (blocking). It will save to file.
                new_calib = run_automatic_calibration(tracker, cap, overlay=CALIBRATION_IS_OVERLAY)
                if new_calib:
                    calibration = new_calib
                current_state = STATE_TRACKING
            elif key == ord('d'):
                if current_state == STATE_DRAWING:
                    current_state = STATE_TRACKING
                    if DRAWING_IN_SEPARATE_WINDOW:
                        cv2.destroyWindow('Gaze Canvas')
                else:
                    current_state = STATE_DRAWING
                    if DRAWING_IN_SEPARATE_WINDOW:
                        cv2.namedWindow('Gaze Canvas', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Gaze Canvas', 640, 360)
            elif key == ord('r'):
                if current_state == STATE_DRAWING:
                    painter.clear()
                else:
                    # reset calibration file
                    import os
                    if os.path.exists('calibration.json'):
                        os.remove('calibration.json')
                        calibration = None
                        print('Calibration reset.')
            elif key == ord('s'):
                if current_state == STATE_DRAWING:
                    painter.save()
                    print('Saved gaze_art.png')

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()