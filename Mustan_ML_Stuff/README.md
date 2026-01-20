# Mustan ML Stuff

Eye gaze detection and risk analysis using YOLOv8 pose estimation.

## Overview

This project implements real-time eye gaze tracking and risk assessment. It uses a custom-trained **YOLOv8-pose** model (`best.pt`) to find eye keypoints and a geometric algorithm to calculate gaze direction.

### Key Features
- **Real-time Detection**: Uses a 2-stage pipeline (MediaPipe/Fallback + YOLOv8) for high performance.
- **Risk Analysis**: Detects if a user is looking down, up, sideways, or focusing on the screen.
- **Optimized**: Threaded camera capture and optimized inference sizes for CPU performance.

## Pipeline Structure

The core logic is contained in the `eye_pipeline/` directory:

- **`main.py`**: Entry point. initializes and runs the pipeline.
- **`modules/`**:
  - **`pipeline.py`**: Orchestrates camera, detection, and display.
  - **`eye_detector.py`**: The core detection logic (MediaPipe Stage 1 -> YOLO Stage 2).
  - **`camera_input.py`**: Threaded camera handling.
  - **`config.py`**: Configuration settings (Resolution, Sensitivity, Model paths).
  - **`display.py`**: Visualization utilities.

## Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements_yolo.txt
    ```

2.  **Verify Model**
    Ensure `best.pt` is present in `eye_pipeline/best.pt`.
    *(This model is now included in the repository)*.

## Usage

To run the eye gaze detection:

```bash
cd eye_pipeline
python main.py
```

## Risk Classification
The system classifies gaze into the following categories:

*   **CENTER (SAFE)**: User is looking at the screen.
*   **LOOKING DOWN (RISK)**: User is looking down (e.g., at notes/phone).
*   **LOOKING SIDE (RISK)**: User is looking too far left or right.
*   **LOOKING UP (THINKING)**: User is looking up (generally considered safe/thinking behavior).

## Configuration
You can adjust sensitivity and performance settings in `eye_pipeline/modules/config.py`:
*   `CAMERA_WIDTH` / `CAMERA_HEIGHT`: Adjust resolution (Default: 640x480).
*   `FRAME_SKIP`: Increase to reduce CPU load.
*   `RISK_VERTICAL_THRESHOLD`: Sensitivity for looking down.

## Model Documentation
For detailed technical specifications of the `best.pt` model, see [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md).
