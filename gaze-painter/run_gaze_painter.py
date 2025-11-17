"""
SYSTEM A: Gaze Painter (Kiosk Mode)
- Shows 'Eyes' window as the main UI.
- 'Gaze Painter' canvas window is shown/hidden based on state.
- No UI text is drawn on the 'Gaze Painter' canvas.
- Automatic calibration on gaze detection.
- Gaze loss triggers save and reset.
- Timer expiry triggers save and loop (if gaze present).
"""

import cv2
import time
import numpy as np
import os
import json
from datetime import datetime

# Imports for Gaze Painting
from eye_tracking import EyeTracker
from drawing import GazePainter
from utils import save_calibration, load_calibration
from eye_visualizer import EyeVisualizer # Now handles UI text

# --- Configuration ---
SHARED_OUTPUT_DIR = "../shared_output"
DRAW_DURATION_SEC = 15.0 # 15-second drawing timer
GAZE_LOSS_TIMEOUT_SEC = 2.0 # Wait 2s after gaze loss before reset

# --- Application States ---
STATE_SEARCHING = 0
STATE_CALIBRATING = 1
STATE_DRAWING = 2
STATE_SAVING = 3

class GazeTrackingPainter:
    def __init__(self, cap_index=0):
        self.cap = cv2.VideoCapture(cap_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {cap_index}")
            exit()
            
        self.w = 1280
        self.h = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.tracker = EyeTracker(flip=True)
        self.painter = GazePainter(width=self.w, height=self.h)
        self.visualizer = EyeVisualizer() 
        
        self.running = True
        self.state = STATE_SEARCHING
        self.status_text = "Searching for gaze..."
        
        self.last_time = time.time()
        self.dt = 0
        self.draw_start_time = None
        self.last_gaze_time = None
        
        # Window names
        self.eyes_window = "Eyes"
        self.canvas_window = "Gaze Painter"
        
        # Ensure shared directory exists
        os.makedirs(SHARED_OUTPUT_DIR, exist_ok=True)
        
        # Load existing calibration if it exists, otherwise we'll auto-calibrate
        calibration = load_calibration()
        if calibration:
            self.tracker.load_calibration(calibration)
            print("Existing calibration loaded.")
        
    def run(self):
        print("Starting Gaze Painter (System A - Kiosk Mode)...")
        print("Controls: [q] Quit")
        cv2.namedWindow(self.eyes_window)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            now = time.time()
            self.dt = now - self.last_time
            self.last_time = now
            
            # --- 1. Process Frame ---
            _, gaze_info, face_results = self.tracker.process_frame(
                frame, 
                return_results=True
            )
            
            # Update gaze presence timer
            if gaze_info is not None:
                self.last_gaze_time = now
            
            norm_gaze = self.tracker.get_calibrated_gaze(gaze_info)
            
            # --- 2. State Machine Logic ---
            
            if self.state == STATE_SEARCHING:
                self.status_text = "Searching for gaze..."
                self.close_canvas() # Ensure canvas is closed
                
                if gaze_info is not None:
                    # Gaze detected! Start calibration.
                    self.state = STATE_CALIBRATING
                    self.tracker.start_auto_calibration()
                    self.status_text = "Calibrating... Look at the camera."

            elif self.state == STATE_CALIBRATING:
                if self.check_gaze_loss(now): # Check for gaze loss
                    continue 
                    
                self.status_text = f"Calibrating... {len(self.tracker.calibration_samples)}/{self.tracker.calibration_needed}"
                
                # Feed sample to calibrator
                calibration_done = self.tracker.add_calibration_sample(gaze_info)
                
                if calibration_done:
                    # Calibration finished, start drawing
                    self.state = STATE_DRAWING
                    self.draw_start_time = now
                    self.painter.clear()
                    self.open_canvas() # Show the blank canvas

            elif self.state == STATE_DRAWING:
                if self.check_gaze_loss(now): # Check for gaze loss
                    continue
                
                elapsed = now - self.draw_start_time
                remaining = DRAW_DURATION_SEC - elapsed
                self.status_text = f"Drawing... {remaining:.1f}s"
                
                # Update the drawing canvas
                self.painter.update(norm_gaze, self.dt)
                
                if remaining <= 0:
                    # Timer is up!
                    self.state = STATE_SAVING
                    self.status_text = "Saving..."

            elif self.state == STATE_SAVING:
                self.status_text = "Saving..."
                # This state is very fast, just save and transition
                self.save_and_reset()
            
            # --- 3. Update Windows ---
            
            # Update and show the "Eyes" window
            eye_display = self.visualizer.create_eye_display(frame, face_results)
            eye_display_with_ui = self.visualizer.add_status_overlay(eye_display, self.status_text)
            cv2.imshow(self.eyes_window, eye_display_with_ui)
            
            # Update the "Gaze Painter" window (only if it's open)
            if self.state == STATE_DRAWING:
                # Get canvas *without* any UI text
                canvas_display = self.painter.get_display()
                cv2.imshow(self.canvas_window, canvas_display)

            # --- 4. Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
        
        self.release()

    def check_gaze_loss(self, current_time):
        """
        Checks if gaze has been lost for too long.
        If so, saves, resets state, and returns True.
        """
        gaze_lost = self.last_gaze_time is None or (current_time - self.last_gaze_time) > GAZE_LOSS_TIMEOUT_SEC
        
        if gaze_lost:
            if self.state == STATE_DRAWING or self.state == STATE_CALIBRATING:
                # Gaze was lost while drawing or calibrating
                print("Gaze lost. Saving and resetting.")
                self.state = STATE_SAVING
                self.status_text = "Saving..."
                return True
            elif self.state == STATE_SEARCHING:
                return False # This is the expected state
        return False
        
    def save_and_reset(self):
        """Saves the canvas and decides what to do next."""
        print("Saving drawing...")
        filepath = os.path.join(SHARED_OUTPUT_DIR, "image.png")
        # Resize the canvas to 224x224 before saving
        # resized_image = cv2.resize(self.painter.canvas, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filepath, self.painter.canvas)
        print(f"Drawing saved to {filepath}. Watcher will process it.")
        
        self.close_canvas()
        self.painter.clear()
        
        # Check if gaze is *still* present
        now = time.time()
        if self.last_gaze_time is not None and (now - self.last_gaze_time) < GAZE_LOSS_TIMEOUT_SEC:
            # Gaze is still here. Loop drawing without calibration.
            print("Gaze present. Starting new drawing session.")
            self.state = STATE_DRAWING
            self.draw_start_time = now
            self.open_canvas()
        else:
            # Gaze was lost. Go back to searching.
            print("Gaze lost. Returning to search state.")
            self.state = STATE_SEARCHING
            self.status_text = "Searching for gaze..."

    def open_canvas(self):
        """Shows the canvas window."""
        try:
            cv2.namedWindow(self.canvas_window)
            canvas_display = self.painter.get_display()
            cv2.imshow(self.canvas_window, canvas_display)
        except Exception as e:
            print(f"Error opening canvas: {e}")

    def close_canvas(self):
        """Closes the canvas window safely."""
        try:
            cv2.destroyWindow(self.canvas_window)
        except cv2.error:
            pass # Window was already closed

    def release(self):
        self.running = False
        self.cap.release()
        self.tracker.release()
        self.visualizer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = GazeTrackingPainter(cap_index=0)
    app.run()