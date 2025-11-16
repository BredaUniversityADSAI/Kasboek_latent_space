# Eye Tracker & Gaze Drawing Application - User Manual

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Application Modes](#application-modes)
- [Keyboard Controls](#keyboard-controls)
- [Calibration Guide](#calibration-guide)
- [Drawing with Your Eyes](#drawing-with-your-eyes)
- [Tips for Best Results](#tips-for-best-results)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)

## Overview

This application uses your webcam and MediaPipe face mesh technology to track your eye movements in real-time. It can detect where you're looking on the screen and allows you to create artistic drawings using only your gaze. The system features three main modes: tracking, calibration, and drawing.

## Requirements

### Hardware
- Webcam (built-in or external)
- Computer with sufficient processing power for real-time video processing

### Software Dependencies
- Python 3.8+
- OpenCV (`cv2`)
- MediaPipe
- NumPy

### Installation
```bash
pip install opencv-python mediapipe numpy
```

## Getting Started

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Position Yourself**
   - Sit approximately 50-70cm from your screen
   - Ensure your face is well-lit (avoid backlighting)
   - Keep your head relatively steady for best results

3. **Initial Calibration**
   - The system will check for existing calibration on startup
   - If none exists, press `C` to calibrate (recommended for first use)

## Application Modes

### 1. **Tracking Mode** (Default)
The standard mode that displays eye tracking visualization overlaid on your video feed.

**What you'll see:**
- Green dots: Detected iris points
- Red circles: Iris centers
- Blue rectangles: Eye bounding boxes
- Yellow arrow: Current gaze direction
- Status text: Direction, saccade count, and tracking speed
- FPS counter
- Calibration status

### 2. **Calibration Mode**
Calibrates the system to your specific eye movements and viewing angle.

**Process:**
1. Press `C` to start calibration
2. Look at the center of your screen for ~2 seconds
3. Keep your gaze steady during the progress bar
4. System automatically calculates thresholds and saves calibration

### 3. **Drawing Mode**
Create artwork by moving your eyes! Your gaze controls a virtual paintbrush.

**Features:**
- Separate canvas window for your artwork
- Line thickness varies with dwell time (looking at one spot)
- Smooth stroke interpolation
- Real-time cursor indicator showing gaze position
- Canvas automatically saves to `saved_images/` folder

## Keyboard Controls

| Key | Action | Available In |
|-----|--------|--------------|
| **C** | Start calibration process | All modes |
| **D** | Toggle drawing mode on/off | All modes |
| **V** | Toggle visual overlays (tracking visualization) | Tracking/Drawing |
| **R** | Reset/Clear | • Drawing mode: Clear canvas<br>• Other modes: Reset calibration |
| **S** | Save drawing (with timestamp) | Drawing mode only |
| **F** | Toggle frame flip (mirror mode) | All modes |
| **Q** | Quit application | All modes |

## Calibration Guide

### When to Calibrate
- First time using the application
- After adjusting your seating position
- If tracking feels inaccurate
- After changing lighting conditions

### Calibration Process
1. Press `C` to begin
2. Focus on the center of your screen (look at the camera if it's centered)
3. Keep your gaze steady for the duration (~2 seconds)
4. Wait for "Calibration successful!" message

### What Calibration Does
- Establishes your neutral gaze position
- Calculates personalized movement thresholds
- Sets deadzone for small eye movements
- Saves settings to `calibrations/calibration.json`

## Drawing with Your Eyes

### Getting Started with Drawing
1. Ensure you're calibrated (press `C` if needed)
2. Press `D` to enter drawing mode
3. A separate "Gaze Canvas" window will appear
4. Move your eyes to draw!

### Drawing Techniques
- **Thin lines**: Move your eyes quickly across the screen
- **Thick lines**: Dwell (pause) your gaze in one spot
- **Smooth curves**: Move your eyes slowly and steadily
- **Sharp corners**: Make quick saccadic movements

### Drawing Controls
- **Clear canvas**: Press `R` while in drawing mode
- **Save artwork**: Press `S` (saves with timestamp)
- **Exit drawing**: Press `D` again

### Canvas Features
- White background (1280x720 pixels)
- Black strokes by default
- Green crosshair cursor showing current gaze position
- Automatic stroke smoothing for natural lines

## Tips for Best Results

### Optimal Environment
- **Lighting**: Face a light source, avoid windows behind you
- **Distance**: Maintain 50-70cm from screen
- **Stability**: Keep your head relatively still
- **Screen position**: Camera at eye level or slightly below

### Tracking Performance
- Clean your camera lens for clearer detection
- Remove glasses if they cause reflections (if possible)
- Blink naturally - the system handles brief interruptions
- Take breaks to avoid eye strain

### Drawing Tips
- Start with simple shapes (circles, lines)
- Practice controlling dwell time for line thickness
- Use peripheral vision to plan your next stroke
- Experiment with different eye movement speeds

## Troubleshooting

### "Cannot open webcam"
- Check if another application is using the camera
- Verify camera permissions for Python/terminal
- Try different camera index (modify `cv2.VideoCapture(0)` to `(1)` or `(2)`)

### Poor Tracking Accuracy
- Recalibrate (press `C`)
- Improve lighting conditions
- Clean camera lens
- Check for reflections on glasses
- Ensure face is fully visible in frame

### Drawing Not Working
- Verify calibration is complete
- Check if drawing mode is active (should show "MODE: DRAWING")
- Ensure "Gaze Canvas" window is open
- Try toggling drawing mode off and on (`D` key twice)

### Laggy Performance
- Close other applications
- Reduce camera resolution in code
- Toggle off visual overlays (press `V`)
- Check CPU usage

## File Structure

```
project/
├── main.py           # Main application entry point
├── tracking.py       # Eye tracking implementation
├── calibration.py    # Calibration routines
├── drawing.py        # Gaze painting functionality
├── utils.py          # Utility functions
├── calibrations/     # Stored calibration files
│   └── calibration.json
└── saved_images/     # Saved artwork
    └── gaze_art_[timestamp].png
```

### Generated Files

**Calibration Data** (`calibrations/calibration.json`):
```json
{
  "center": [x, y],
  "horiz_threshold": 0.035,
  "vert_threshold": 0.028,
  "deadzone": 0.018,
  "timestamp": 1234567890.0
}
```

**Saved Drawings** (`saved_images/gaze_art_YYYYMMDD_HHMMSS.png`):
- PNG format
- 1280x720 resolution
- White background with colored strokes

## Advanced Features

### Saccade Detection
The system automatically detects rapid eye movements (saccades) and displays:
- Total saccade count
- Current movement speed
- "SACCADE" indicator during rapid movements

### Gaze Classification
Recognized directions:
- center
- left, right, up, down
- Diagonal combinations (e.g., "left-up", "right-down")

### Smoothing Algorithm
- Rolling median filter (5 samples)
- Exponential smoothing (alpha=0.6)
- Buffer size of 7 frames
- Reduces jitter while maintaining responsiveness

## Safety and Comfort

- Take regular breaks (every 20-30 minutes)
- Blink frequently to avoid dry eyes
- Stop if you experience eye strain or headaches
- Maintain proper distance from screen
- Ensure adequate room lighting

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure your webcam is functioning properly
4. Review the console output for error messages

---

**Version**: 2.5
**Last Updated**: 2025.11.06