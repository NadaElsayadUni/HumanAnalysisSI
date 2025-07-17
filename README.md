# Human Motion Analysis - Modular Structure

This is a modular version of the human motion analysis system, breaking down the original monolithic `signal_image.py` into separate, focused components.

## 📁 Project Structure

```
Nada_Elsayad_256292_SI/
├── main.py                      # Main script to run the analysis
├── src/                         # Modular source code
│   ├── __init__.py              # Main package exports
│   ├── core/                    # Core analysis components
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Main orchestrator class
│   │   ├── detection.py         # Object detection logic
│   │   ├── tracking.py          # Object tracking logic
│   │   └── kalman_filter.py     # Kalman filtering
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── visualization.py     # Drawing and visualization
├── video/      # Input data
├── output/                      # Output directory
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis
```bash
python main.py
```

### 3. Alternative: Run Original Code
```bash
python signal_image.py
```

## 🔧 Core Components

### 1. HumanMotionAnalyzer (`src/core/analyzer.py`)
**Main orchestrator class** that coordinates all analysis components.

**Key Methods:**
- `__init__(video_path, reference_path)`: Initialize analyzer
- `run_analysis()`: Run complete analysis pipeline
- `process_frame(frame)`: Process single frame
- `save_analysis_results()`: Save results and visualizations

### 2. ObjectDetector (`src/core/detection.py`)
**Handles moving object detection** using background subtraction and shadow suppression.

**Key Methods:**
- `detect_moving_objects(frame)`: Detect objects in a frame
- Uses MOG2 background subtraction with shadow suppression

### 3. ObjectTracker (`src/core/tracking.py`)
**Manages object tracking**, ID assignment, and trajectory management.

**Key Methods:**
- `track_objects(detections, frame)`: Track objects across frames
- `update_trajectories()`: Update trajectory data
- `update_trajectories_with_kalman()`: Update with Kalman filtering
- `get_object_color()`: Get consistent colors for objects

### 4. KalmanFilterManager (`src/core/kalman_filter.py`)
**Manages Kalman filters** for multiple objects.

**Key Methods:**
- `update_kalman_filters(tracked_objects)`: Update all filters
- `get_kalman_center(object_id)`: Get filtered position
- `get_kalman_velocity(object_id)`: Get velocity estimate
- `get_kalman_prediction(object_id)`: Get predicted position

### 5. Visualizer (`src/utils/visualization.py`)
**Handles all visualization** and drawing operations.

**Key Methods:**
- `draw_trajectories()`: Draw real-time trajectories
- `draw_complete_trajectories()`: Draw final trajectory visualization
- `draw_complete_trajectories_realtime()`: Draw during video playback
- `save_individual_trajectories()`: Save individual trajectory images
- `smooth_trajectory()`: Smooth trajectory points

## 📊 Output Files

The analysis generates several output files:

- **`last_frame_signal.jpg`**: Last frame from the video
- **`trajectories_raw.jpg`**: Raw trajectory visualization
- **`trajectories_kalman.jpg`**: Kalman-filtered trajectory visualization
- **`detected_trajectories/`**: Individual trajectory images for each object

## 🔄 Migration from Original Code

The modular structure maintains the same functionality as the original `signal_image.py` but with these improvements:

1. **Separation of Concerns**: Each component has a single responsibility
2. **Reusability**: Components can be used independently
3. **Testability**: Each component can be tested in isolation
4. **Maintainability**: Easier to modify individual components
5. **Extensibility**: Easy to add new features or components

### Original vs Modular Structure

| Original (`signal_image.py`) | Modular Structure |
|------------------------------|-------------------|
| Single large class (845 lines) | Multiple focused classes |
| All functionality in one file | Separated by responsibility |
| Hard to test individual parts | Each component testable |
| Difficult to reuse components | Components reusable |
| Hard to modify specific features | Easy to modify specific features |

## 🛠️ Usage Examples

### Basic Usage
```python
from src import HumanMotionAnalyzer

# Initialize analyzer
analyzer = HumanMotionAnalyzer(
    video_path="path/to/video.mp4",
    reference_path="path/to/reference.jpg"
)

# Run analysis
analyzer.run_analysis(show_visualization=True, save_results=True)
```

### Advanced Usage
```python
from src.core.analyzer import HumanMotionAnalyzer
from src.core.detection import ObjectDetector
from src.core.tracking import ObjectTracker
from src.core.kalman_filter import KalmanFilterManager

# Create components individually
detector = ObjectDetector()
tracker = ObjectTracker(frame_width=1920, frame_height=1080)
kalman_manager = KalmanFilterManager()

# Use components as needed
detections, fg_mask = detector.detect_moving_objects(frame)
tracked_objects = tracker.track_objects(detections, frame)
kalman_manager.update_kalman_filters(tracked_objects)
```

## 🧪 Testing Individual Components

You can test individual components:

```python
# Test detection only
detector = ObjectDetector()
detections, fg_mask = detector.detect_moving_objects(frame)

# Test tracking only
tracker = ObjectTracker(1920, 1080)
tracked_objects = tracker.track_objects(detections, frame)

# Test visualization only
visualizer = Visualizer()
result_frame = visualizer.draw_complete_trajectories(
    frame, trajectories, colors
)
```

## 🎯 Key Benefits

1. **Maintainability**: Easy to modify individual components
2. **Testability**: Each component can be tested independently
3. **Reusability**: Components can be used in other projects
4. **Readability**: Code is easier to understand and navigate
5. **Extensibility**: Easy to add new features or modify existing ones

This modular structure makes the code much more maintainable and easier to understand, while preserving all the original functionality.