# Eye Tracker Application - Code Documentation

## Architecture Overview

The application follows a modular design pattern with five main components:

```
main.py (Controller)
    ├── tracking.py (Eye Detection & Processing)
    ├── calibration.py (Calibration System)
    ├── drawing.py (Gaze-based Drawing)
    └── utils.py (Utilities & File I/O)
```

## Module Descriptions

### 📁 `main.py` - Application Controller
**Purpose**: Entry point and main application loop that orchestrates all components.

**Key Responsibilities**:
- Initialize webcam and all components
- Manage application states (TRACKING, CALIBRATING, DRAWING)
- Handle keyboard input and mode switching
- Display appropriate UI based on current state
- Calculate and display FPS

**State Machine**:
```python
STATE_TRACKING = 1    # Default eye tracking visualization
STATE_CALIBRATING = 2 # Running calibration routine
STATE_DRAWING = 3     # Gaze-controlled drawing mode
```

**Main Functions**:
- `main()` - Primary application loop
- `print_controls()` - Display keyboard shortcuts to console

---

### 📁 `tracking.py` - Eye Tracking Engine
**Purpose**: Core eye tracking functionality using MediaPipe Face Mesh.

**Class: `EyeTracker`**

**Key Features**:
- Detects facial landmarks and iris positions
- Normalizes gaze coordinates relative to eye bounding boxes
- Applies smoothing algorithms to reduce jitter
- Detects saccadic eye movements
- Classifies gaze direction (up/down/left/right/center)

**Important Methods**:

| Method | Description |
|--------|-------------|
| `process_frame(frame, only_compute)` | Main processing pipeline - returns annotated frame and normalized gaze |
| `_normalize_by_eye(iris_center, eye_bbox)` | Converts iris position to normalized coordinates |
| `classify_gaze(norm_offset)` | Determines gaze direction based on thresholds |
| `toggle_visuals()` | Enable/disable tracking visualization overlay |
| `toggle_flip()` | Mirror mode for more natural interaction |

**Key Algorithms**:
1. **Iris Detection**: Uses MediaPipe landmarks 468-476 for iris tracking
2. **Normalization**: Maps iris position to [-1, 1] range relative to eye corners
3. **Smoothing Pipeline**:
   - Rolling median filter (window=5)
   - Exponential smoothing (α=0.6)
   - Buffer of last 7 frames
4. **Saccade Detection**: Velocity-based threshold (>0.2 units/sec)

**Landmark Indices**:
```python
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
```

---

### 📁 `calibration.py` - Calibration System
**Purpose**: Establishes personalized gaze parameters for each user.

**Function: `run_automatic_calibration()`**

**Parameters**:
- `tracker` - EyeTracker instance to calibrate
- `cap` - OpenCV video capture object
- `samples` - Number of frames to collect (default: 30)
- `overlay` - Show progress visualization

**Calibration Process**:
1. Collects 30 frames of center gaze data
2. Calculates mean position and standard deviation
3. Sets adaptive thresholds based on user's eye movement variance
4. Saves calibration to JSON file

**Threshold Calculation**:
```python
horiz_threshold = max(0.035, std_deviation[0] * 3.0)
vert_threshold = max(0.028, std_deviation[1] * 3.0)
deadzone = max(0.018, mean(std_deviation) * 2.0)
```

**Output Format**:
```json
{
  "center": [x, y],
  "horiz_threshold": 0.035,
  "vert_threshold": 0.028,
  "deadzone": 0.018,
  "timestamp": 1234567890.0
}
```

---

### 📁 `drawing.py` - Gaze Drawing System
**Purpose**: Transforms gaze coordinates into artistic drawings.

**Class: `GazePainter`**

**Key Features**:
- Variable line thickness based on dwell time
- Smooth stroke interpolation
- Canvas management (clear, save)
- Real-time cursor visualization
- Overlay mode for mixed reality drawing

**Important Methods**:

| Method | Description |
|--------|-------------|
| `update(gaze_norm, dt)` | Updates canvas with new gaze position |
| `clear()` | Resets canvas to blank white |
| `save(filename)` | Exports drawing to PNG file |
| `get_display(frame, overlay)` | Returns canvas for display |
| `get_canvas_with_cursor(gaze_norm)` | Adds green crosshair at gaze position |

**Drawing Algorithm**:
1. **Dwell Detection**: Measures time spent in small radius (0.03 normalized units)
2. **Line Width Calculation**:
   ```python
   base_width = 2
   dwell_bonus = min(20, dwell_time * 10)
   speed_factor = max(0.2, min(1.0, distance * 50))
   line_width = (base_width + dwell_bonus) * (2.0 - speed_factor)
   ```
3. **Stroke Smoothing**: Maintains history of last 5 points for polyline rendering
4. **Color Variation**: Adjusts blue channel based on movement speed

**Canvas Properties**:
- Resolution: 1280x720 pixels
- Background: White (255, 255, 255)
- Default stroke: Blue (BGR: 0, 0, 255)

---

### 📁 `utils.py` - Utility Functions
**Purpose**: File I/O operations and helper functions.

**Directory Management**:
```python
CALIBRATION_DIR = './calibrations'
SAVED_IMAGES_DIR = './saved_images'
```

**Key Functions**:

| Function | Description |
|----------|-------------|
| `ensure_directories()` | Creates required folders if missing |
| `save_calibration(data)` | Saves calibration to JSON |
| `load_calibration()` | Loads existing calibration |
| `norm_to_pixel(norm, w, h)` | Convert [0,1] to pixel coordinates |
| `pixel_to_norm(pixel, w, h)` | Convert pixels to [0,1] range |
| `save_timestamp_log()` | Save timestamped data logs |

**File Handling**:
- Automatically converts NumPy arrays to lists for JSON serialization
- Handles missing files gracefully
- Creates timestamped filenames for saved drawings

---

## Data Flow

```
1. Camera Frame
   ↓
2. EyeTracker.process_frame()
   ├── MediaPipe Face Detection
   ├── Iris Landmark Extraction
   ├── Normalization & Smoothing
   └── Returns: (annotated_frame, gaze_norm)
   ↓
3. State-based Processing
   ├── TRACKING: Display with overlays
   ├── CALIBRATING: Collect samples → Calculate thresholds
   └── DRAWING: GazePainter.update() → Canvas rendering
   ↓
4. Display Output
   └── OpenCV imshow() windows
```

## Key Algorithms Explained

### Gaze Normalization
Each eye's iris position is normalized relative to its bounding box:
```
normalized_x = (iris_x - eye_center_x) / eye_width
normalized_y = (iris_y - eye_center_y) / eye_height
```
This creates device-independent coordinates in range [-0.5, 0.5].

### Smoothing Pipeline
Reduces jitter through two-stage filtering:
1. **Median Filter**: Removes outliers from last 5 samples
2. **Exponential Smoothing**: Blends current with previous (60% weight to history)

### Adaptive Calibration
Instead of fixed thresholds, the system adapts to each user:
- Measures variance during calibration
- Sets thresholds at 3σ for horizontal/vertical movement
- Creates personalized deadzone at 2σ

### Saccade Detection
Identifies rapid eye movements by velocity:
```
velocity = distance_moved / time_delta
is_saccade = velocity > 0.2
```

## Performance Considerations

### Optimization Strategies
- **Frame Buffering**: Limited to 7 frames to balance smoothness vs memory
- **Conditional Rendering**: Visualization can be toggled off for better FPS
- **Single Face Processing**: Limits detection to 1 face for efficiency
- **Selective Computation**: `only_compute` flag skips visualization during calibration

### Resource Management
- MediaPipe models loaded once at initialization
- Calibration data cached in memory after loading
- Canvas pre-allocated at fixed resolution
- Proper cleanup in `release()` methods

## Configuration Options

### Adjustable Parameters

**EyeTracker**:
```python
smooth_alpha = 0.6        # Exponential smoothing factor
median_window = 5         # Median filter size
buffer_maxlen = 7         # Frame history size
saccade_speed_thr = 0.2   # Saccade detection threshold
```

**GazePainter**:
```python
dwell_radius = 0.03       # Radius for dwell detection
stroke_color = (0,0,255)  # BGR color tuple
canvas_size = (1280,720)  # Drawing resolution
```

**Calibration**:
```python
samples = 30              # Frames to collect
threshold_multiplier = 3.0 # Std dev multiplier
```

## Error Handling

- **Camera Failure**: Graceful exit with error message
- **Missing Calibration**: Falls back to default thresholds
- **Face Not Detected**: Returns None for gaze, continues processing
- **File I/O Errors**: Caught and logged, non-fatal

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | 4.x | Video capture, image processing, display |
| MediaPipe | 0.8+ | Face mesh and iris detection |
| NumPy | 1.19+ | Array operations and statistics |

## Extension Points

The modular design allows easy extension:

1. **New Drawing Styles**: Extend `GazePainter` class
2. **Alternative Smoothing**: Replace algorithms in `EyeTracker`
3. **Multi-user Support**: Extend calibration to store user profiles
4. **Additional Gestures**: Add blink/wink detection to `tracking.py`
5. **Export Formats**: Add new save methods to `utils.py`

---

**Version**: 2.5
**Python Version**: 3.8+