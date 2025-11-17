import numpy as np
import cv2
import time
import os

class GazePainter:
    def __init__(self, width=1280, height=720, dwell_radius=0.03):
        # (Rest of __init__ is unchanged)
        self.w = width
        self.h = height
        self.canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        self.prev_point = None
        self.prev_time = None
        self.dwell_time = 0.0
        self.dwell_radius = dwell_radius
        self.stroke_color = (0, 0, 0)
        self.stroke_history = []
        self.pen_down = False
        
    def clear(self):
        # (clear method is unchanged)
        self.canvas[:] = 255
        self.prev_point = None
        self.prev_time = None
        self.dwell_time = 0.0
        self.stroke_history = []
        self.pen_down = False
    
    def save(self, filename='gaze_art.png'):
        """Save the canvas to a file in saved_images folder"""
        
        # Check if any pixel is NOT white (255).
        # np.any(self.canvas < 255) is a fast way to check if anything was drawn.
        if not np.any(self.canvas < 255):
            print("Canvas is empty, not saving.")
            return None
            
        save_dir = './saved_images'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, self.canvas)
        print(f'Saved drawing to {filepath}')
        return filepath
    
    def set_pen(self, is_down):
        # (set_pen method is unchanged)
        if self.pen_down == is_down:
            return
        self.pen_down = is_down
        self.prev_point = None
        self.stroke_history = [] if not is_down else self.stroke_history
        print(f"Pen set to: {'DOWN' if is_down else 'UP'}")
    
    def update(self, gaze_norm, dt):
        """
        Update the drawing based on normalized gaze position.
        gaze_norm should be (x, y) in [0, 1] range
        """
        if not self.pen_down:
            # (Rest of pen up logic is unchanged)
            self.prev_point = None
            self.dwell_time = 0.0
            self.stroke_history = []
            return
            
        if gaze_norm is None:
            # (Rest of no gaze logic is unchanged)
            self.prev_point = None
            self.dwell_time = 0.0
            return
        
        # gaze_norm is now the combined head/eye position.
        # It can be outside [0, 1], so we clip it here.
        gx = int(np.clip(gaze_norm[0], 0, 1) * self.w)
        gy = int(np.clip(gaze_norm[1], 0, 1) * self.h)
        cur_point = (gx, gy)
        
        # (Rest of update logic is unchanged)
        if self.prev_point is None:
            self.prev_point = cur_point
            self.prev_time = time.time()
            self.dwell_time = 0.0
            self.stroke_history = [cur_point]
            return
        
        dx = (cur_point[0] - self.prev_point[0]) / float(self.w)
        dy = (cur_point[1] - self.prev_point[1]) / float(self.h)
        dist = np.hypot(dx, dy)
        
        if dist < self.dwell_radius:
            self.dwell_time += dt
        else:
            self.dwell_time = 0.0
        
        base_width = 2
        dwell_bonus = int(min(20, self.dwell_time * 10))
        speed_factor = max(0.2, min(1.0, dist * 50))
        line_width = int((base_width + dwell_bonus) * (2.0 - speed_factor))
        line_width = max(1, min(40, line_width))
        
        self.stroke_history.append(cur_point)
        if len(self.stroke_history) > 5:
            self.stroke_history.pop(0)
        
        if len(self.stroke_history) >= 2:
            pts = np.array(self.stroke_history, np.int32)
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    color_var = int(255 * speed_factor)
                    color = (max(0, self.stroke_color[0] - color_var//3),
                            self.stroke_color[1],
                            min(255, self.stroke_color[2] + color_var//3))
                    cv2.line(self.canvas, tuple(pts[i]), tuple(pts[i+1]), color, line_width)
        else:
            cv2.line(self.canvas, self.prev_point, cur_point, self.stroke_color, line_width)
        
        cv2.circle(self.canvas, cur_point, max(1, line_width//2), self.stroke_color, -1)
        self.prev_point = cur_point
    
    def get_display(self, frame=None, overlay=False):
        # (get_display method is unchanged)
        if overlay and frame is not None:
            ch, cw = frame.shape[:2]
            resized = cv2.resize(self.canvas, (cw, ch))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            result = frame.copy()
            result[mask > 0] = cv2.addWeighted(
                frame[mask > 0], 0.3,
                resized[mask > 0], 0.7, 0
            ).astype(np.uint8)
            return result
        else:
            return self.canvas
    
    def set_color(self, color):
        # (set_color method is unchanged)
        self.stroke_color = color
    
    def get_canvas_with_cursor(self, gaze_norm):
        # (get_canvas_with_cursor method is unchanged, but clipping is more important)
        display = self.canvas.copy()
        
        if gaze_norm is not None:
            gx = int(np.clip(gaze_norm[0], 0, 1) * self.w)
            gy = int(np.clip(gaze_norm[1], 0, 1) * self.h)
            
            cursor_color = (0, 0, 255) if self.pen_down else (0, 255, 0)
            cursor_size = 20
            cv2.line(display, (gx - cursor_size, gy), (gx + cursor_size, gy), cursor_color, 1)
            cv2.line(display, (gx, gy - cursor_size), (gx, gy + cursor_size), cursor_color, 1)
            cv2.circle(display, (gx, gy), 8, cursor_color, 1)
        
        return display