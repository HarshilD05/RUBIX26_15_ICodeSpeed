"""
Configuration Module
Central configuration for eye movement detection pipeline
"""


class Config:
    """Pipeline configuration settings"""
    
    # Camera Settings
    CAMERA_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Display Settings
    WINDOW_NAME = "Eye Movement Detection Pipeline"
    FULLSCREEN = False
    SHOW_FPS = True
    
    # Pipeline Settings
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Performance Settings
    MAX_FPS = 60  # Maximum FPS to process
    FRAME_SKIP = 0  # Number of frames to skip (0 = no skip)
    
    # Eye Detection Settings
    EYE_DETECTION_ENABLED = True
    EYE_MODEL_PATH = "eye_movement_model.pth"  # Path to custom trained model
    EYE_CONFIDENCE_THRESHOLD = 0.6
    EYE_BOX_COLOR = (255, 0, 0)  # Blue (B, G, R)
    EYE_BOX_THICKNESS = 2
    EYE_SHOW_CONFIDENCE = True
    EYE_SHOW_LABEL = True
    
    # Face detection for eye region extraction
    FACE_DETECTION_ENABLED = True
    FACE_MODEL_NAME = "yolov8n-face.pt"  # YOLOv8 face model
    FACE_CONFIDENCE_THRESHOLD = 0.5
    
    # Eye Movement Classes
    EYE_MOVEMENT_CLASSES = [
        "Closed",
        "Top Center",
        "Top Right",
        "Top Left",
        "Bottom Center",
        "Bottom Right",
        "Bottom Left",
        "Center Left",
        "Center",
        "Center Right"
    ]
    
    # Eye Region Settings
    EYE_REGION_CROP = True  # Crop eye region from face
    EYE_REGION_EXPAND = 0.2  # Expand eye region by 20%
    EYE_INPUT_SIZE = (224, 224)  # Input size for model
    
    @classmethod
    def from_dict(cls, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
    
    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary"""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and key.isupper()
        }
