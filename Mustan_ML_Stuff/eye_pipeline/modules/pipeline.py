"""
Pipeline Module
Base camera pipeline and eye movement detection pipeline

Eye Movement Detection Pipeline
"""
import cv2
import logging
import time
from .camera_input import CameraCapture
from .display import DisplayWindow
from .config import Config
from .eye_detector import EyeMovementDetector


class CameraPipeline:
    """Main pipeline class that orchestrates camera capture and display"""
    
    def __init__(self, config=None):
        """
        Initialize the camera pipeline
        
        Args:
            config: Configuration object (uses default Config if None)
        """
        self.config = config or Config()
        self.camera = None
        self.display = None
        self.is_running = False
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the pipeline"""
        if self.config.ENABLE_LOGGING:
            logging.basicConfig(
                level=getattr(logging, self.config.LOG_LEVEL),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize camera and display components"""
        logging.info("Initializing camera pipeline...")
        
        # Initialize camera
        self.camera = CameraCapture(
            camera_id=self.config.CAMERA_ID,
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            fps=self.config.CAMERA_FPS
        )
        
        if not self.camera.start():
            logging.error("Failed to start camera")
            return False
        
        # Initialize display
        self.display = DisplayWindow(
            window_name=self.config.WINDOW_NAME,
            fullscreen=self.config.FULLSCREEN
        )
        
        if not self.display.create_window():
            logging.error("Failed to create display window")
            self.camera.stop()
            return False
        
        logging.info("Pipeline initialized successfully")
        return True
    
    def process_frame(self, frame):
        """
        Process a single frame (override this method for custom processing)
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame
        """
        # Default: no processing, just return the frame
        return frame
    
    def run(self):
        """Main pipeline loop"""
        if not self.initialize():
            logging.error("Pipeline initialization failed")
            return
        
        self.is_running = True
        logging.info("Starting pipeline loop...")
        
        # FPS tracking
        fps = 0
        frame_count = 0
        start_time = time.time()
        frame_time = 1.0 / self.config.MAX_FPS if self.config.MAX_FPS > 0 else 0
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Read frame from camera
                success, frame = self.camera.read_frame()
                
                if not success:
                    logging.warning("Failed to read frame, continuing...")
                    continue
                
                # Process frame (can be overridden)
                processed_frame = self.process_frame(frame)
                
                # Display frame
                if self.config.SHOW_FPS:
                    self.display.show_frame(processed_frame, fps=fps)
                else:
                    self.display.show_frame(processed_frame)
                
                # Check for exit key
                if self.display.check_exit_key(1):
                    logging.info("Exit key pressed")
                    break
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
                # Frame rate limiting
                if frame_time > 0:
                    processing_time = time.time() - loop_start
                    sleep_time = frame_time - processing_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logging.info("Pipeline interrupted by user")
        
        except Exception as e:
            logging.error(f"Pipeline error: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logging.info("Cleaning up pipeline resources...")
        self.is_running = False
        
        if self.camera:
            self.camera.stop()
        
        if self.display:
            self.display.destroy_all()
        
        logging.info("Pipeline cleanup complete")


class EyeMovementPipeline(CameraPipeline):
    """Pipeline with integrated eye movement detection"""
    
    def __init__(self, config=None):
        """Initialize pipeline with eye movement detection"""
        super().__init__(config)
        self.eye_detector = None
        self.detection_count = 0
        self.movement_history = []
        self.max_history = 30  # Keep last 30 detections
        
    def initialize(self):
        """Initialize camera, display, and eye movement detector"""
        # Initialize base components
        if not super().initialize():
            return False
        
        # Initialize eye movement detector
        logging.info("Initializing eye movement detector...")
        
        self.eye_detector = EyeMovementDetector(
            model_path=self.config.EYE_MODEL_PATH,
            confidence_threshold=self.config.EYE_CONFIDENCE_THRESHOLD,
            face_model_name=self.config.FACE_MODEL_NAME
        )
        
        if not self.eye_detector.load_models():
            logging.error("Failed to load eye movement detection models")
            return False
        
        logging.info("Eye movement detector initialized successfully")
        return True
    
    def process_frame(self, frame):
        """
        Process frame with eye movement detection
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with eye movement annotations
        """
        if self.eye_detector is None:
            return frame
        
        # Detect eye movements and draw results
        processed_frame, detections = self.eye_detector.process_frame(
            frame,
            draw=True,
            color=self.config.EYE_BOX_COLOR,
            thickness=self.config.EYE_BOX_THICKNESS
        )
        
        # Update detection count
        self.detection_count = len(detections)
        
        # Store detection history
        if detections:
            for det in detections:
                self.movement_history.append({
                    'timestamp': time.time(),
                    'eye': det['eye'],
                    'movement': det['class_name'],
                    'confidence': det['confidence']
                })
            
            # Keep only recent history
            if len(self.movement_history) > self.max_history:
                self.movement_history = self.movement_history[-self.max_history:]
        
        # Add detection count to display
        cv2.putText(
            processed_frame,
            f"Eyes Detected: {self.detection_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        # Display current eye movements
        y_offset = 110
        for det in detections:
            text = f"{det['eye'].upper()}: {det['class_name']}"
            cv2.putText(
                processed_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            y_offset += 35
        
        return processed_frame
    
    def get_movement_statistics(self):
        """Get statistics about eye movements"""
        if not self.movement_history:
            return {}
        
        # Count movement types
        movement_counts = {}
        for entry in self.movement_history:
            movement = entry['movement']
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
        
        return {
            'total_detections': len(self.movement_history),
            'movement_counts': movement_counts,
            'recent_movements': self.movement_history[-5:]  # Last 5 movements
        }
