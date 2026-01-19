# Eye Movement Detection Pipeline

A modular camera pipeline that captures video from a camera feed and performs real-time eye movement detection and classification using computer vision and deep learning.

## Features

- **10-Class Eye Movement Detection**: Classifies eye movements into:
  1. Closed
  2. Top Center
  3. Top Right
  4. Top Left
  5. Bottom Center
  6. Bottom Right
  7. Bottom Left
  8. Center Left
  9. Center
  10. Center Right

- **Face Detection Integration**: Uses YOLOv8 or Haar Cascade for face detection
- **Real-time Processing**: Optimized for live camera feed
- **Modular Architecture**: Separate modules for camera, display, detection, and configuration
- **FPS Display**: Real-time performance monitoring
- **Flexible Classification**: Supports both ML models and heuristic-based classification
- **Movement History**: Tracks and analyzes eye movement patterns
- **Context Manager Support**: Automatic resource cleanup

## Project Structure

```
eye_pipeline/
├── main.py                      # Main application entry point
├── modules/
│   ├── __init__.py             # Module exports
│   ├── pipeline.py             # Base CameraPipeline & EyeMovementPipeline
│   ├── camera_input.py         # Camera capture module
│   ├── display.py              # Display window module
│   ├── eye_detector.py         # Eye movement detection module
│   └── config.py               # Configuration settings
└── README.md                    # This file
```

## How It Works

### Architecture Overview

1. **Camera Input Module** (`camera_input.py`)
   - Captures frames from the camera
   - Manages camera lifecycle
   - Configurable resolution and FPS

2. **Eye Movement Detector** (`eye_detector.py`)
   - Detects faces using YOLOv8 or Haar Cascade
   - Extracts eye regions from detected faces
   - Classifies eye movements into 10 categories
   - Supports both ML-based and heuristic classification

3. **Display Module** (`display.py`)
   - Creates and manages display window
   - Renders processed frames with annotations
   - Handles user input

4. **Pipeline Module** (`pipeline.py`)
   - Orchestrates the entire processing flow
   - Manages frame loop and FPS calculation
   - Tracks movement history and statistics

5. **Main Application** (`main.py`)
   - Initializes and configures the pipeline
   - Runs detection loop
   - Displays session statistics

### Detection Flow

```
Camera Feed → Capture Frame → Detect Face → Extract Eyes → 
Classify Movement → Annotate → Display → Repeat
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Virtual environment (recommended)

### Step 1: Install Dependencies

```bash
# Navigate to Mustan_ML_Stuff directory
cd Mustan_ML_Stuff

# Install requirements
pip install -r requirements_yolo.txt
```

### Step 2: Run the Pipeline

```bash
# Navigate to eye_pipeline directory
cd eye_pipeline

# Run the application
python main.py
```

## Usage

### Basic Usage

```python
from modules import EyeMovementPipeline, Config

# Configure pipeline
Config.EYE_MODEL_PATH = None  # or path to your trained model
Config.EYE_CONFIDENCE_THRESHOLD = 0.4
Config.FACE_MODEL_NAME = "yolov8n.pt"

# Create and run pipeline
pipeline = EyeMovementPipeline()
pipeline.run()

# Get statistics after exit
stats = pipeline.get_movement_statistics()
print(stats)
```

### Custom Configuration

```python
from modules import Config

# Camera settings
Config.CAMERA_ID = 0
Config.CAMERA_WIDTH = 1280
Config.CAMERA_HEIGHT = 720

# Detection settings
Config.EYE_CONFIDENCE_THRESHOLD = 0.6
Config.EYE_BOX_COLOR = (255, 0, 0)  # Blue

# Face detection
Config.FACE_MODEL_NAME = "yolov8n-face.pt"  # Better face detection
```

## Classification Methods

### 1. ML-Based Classification (Recommended)

If you have a trained PyTorch model:

```python
Config.EYE_MODEL_PATH = "path/to/your/eye_movement_model.pth"
```

The model should:
- Accept 224×224 RGB images
- Output 10 classes
- Be normalized with ImageNet mean/std

### 2. Heuristic-Based Classification (Default)

Uses computer vision techniques:
- Converts eye region to grayscale
- Detects iris/pupil using thresholding
- Calculates iris position
- Classifies based on position thresholds

Works without any training data but less accurate than ML models.

## Training Your Own Model

To train a custom eye movement classification model:

1. **Collect Dataset**
   - Capture images of eyes in each of the 10 positions
   - Aim for 500+ images per class
   - Ensure diverse lighting and subjects

2. **Label Dataset**
   - Organize images into class folders
   - Use standard image classification format

3. **Train Model**
   ```python
   # Use PyTorch to train a classifier
   # See model_inspector.py for model loading examples
   ```

4. **Export Model**
   ```python
   torch.save(model, 'eye_movement_model.pth')
   ```

5. **Use in Pipeline**
   ```python
   Config.EYE_MODEL_PATH = "eye_movement_model.pth"
   ```

## Performance Optimization

### Speed Optimization

```python
# Use smaller face detection model
Config.FACE_MODEL_NAME = "yolov8n.pt"  # Fastest

# Increase confidence threshold
Config.EYE_CONFIDENCE_THRESHOLD = 0.6

# Limit FPS
Config.MAX_FPS = 30
```

### Accuracy Optimization

```python
# Use better face detection model
Config.FACE_MODEL_NAME = "yolov8n-face.pt"

# Lower confidence threshold
Config.EYE_CONFIDENCE_THRESHOLD = 0.4

# Train custom classification model
Config.EYE_MODEL_PATH = "your_trained_model.pth"
```

## Features & Capabilities

### Real-time Detection
- Detects multiple faces simultaneously
- Processes both left and right eyes
- Displays current eye movement state

### Movement Tracking
- Maintains history of last 30 detections
- Provides movement statistics
- Tracks confidence scores

### Visual Feedback
- Face bounding boxes (green)
- Eye region boxes (blue)
- Movement labels with confidence
- FPS counter
- Detection count

## Troubleshooting

### Issue: No faces detected
- **Solution**: Ensure good lighting and face is visible
- Try lowering `FACE_CONFIDENCE_THRESHOLD`
- Check camera is working

### Issue: Inaccurate eye movement classification
- **Solution**: Train a custom ML model
- Adjust heuristic thresholds in `_heuristic_classification()`
- Ensure eyes are clearly visible

### Issue: Low FPS
- **Solution**: Use smaller YOLOv8 model (yolov8n)
- Reduce camera resolution
- Increase `MAX_FPS` limit

### Issue: Model file not found
- **Solution**: Set `EYE_MODEL_PATH = None` to use heuristics
- Or provide correct path to your trained model

## Applications

1. **Gaze Tracking**: Monitor where users are looking
2. **Attention Monitoring**: Detect if person is focused
3. **Drowsiness Detection**: Alert when eyes are closed
4. **Accessibility**: Control interfaces with eye movements
5. **Research**: Study eye movement patterns
6. **Gaming**: Eye-controlled gameplay
7. **Medical**: Diagnose eye movement disorders

## Comparison with Face Detection Pipeline

| Feature | Face Pipeline (HD_ML_stuff) | Eye Pipeline (Mustan_ML_Stuff) |
|---------|---------------------------|-------------------------------|
| Primary Task | Face Detection | Eye Movement Classification |
| Classes | 1 (Face/Person) | 10 (Eye Positions) |
| Model | YOLOv8 | YOLOv8 + Custom Classifier |
| Output | Face bounding boxes | Eye positions + movements |
| Use Case | Identify faces | Analyze eye movements |

## Next Steps

1. **Collect Training Data**: Build dataset for your use case
2. **Train Custom Model**: Improve classification accuracy
3. **Add Features**: Implement gaze tracking, blink detection
4. **Optimize Performance**: Profile and optimize bottlenecks
5. **Deploy**: Package for production use

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Eye Movement Datasets: Search for "eye gaze datasets" or "eye movement datasets"

## License

This project is part of the RUBIX26_15_ICodeSpeed repository.
