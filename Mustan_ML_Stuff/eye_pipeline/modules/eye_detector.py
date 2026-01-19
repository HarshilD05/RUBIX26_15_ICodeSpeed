"""
Eye Movement Detection Module
Detects eyes and classifies eye movement direction
"""

import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class EyeMovementDetector:
    """Detects eyes and classifies eye movement into 10 categories"""
    
    def __init__(self, model_path=None, confidence_threshold=0.6, 
                 face_model_name='yolov8n-face.pt'):
        """
        Initialize Eye Movement Detector
        
        Args:
            model_path: Path to trained eye movement classification model
            confidence_threshold: Minimum confidence for predictions
            face_model_name: YOLOv8 face detection model
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.face_model_name = face_model_name
        
        self.classification_model = None
        self.face_detector = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Eye movement classes
        self.classes = [
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
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logging.info(f"Initializing Eye Movement Detector (Device: {self.device})")
        
    def load_models(self):
        """Load face detection and eye movement classification models"""
        success = True
        
        # Load face detector
        try:
            from ultralytics import YOLO
            self.face_detector = YOLO(self.face_model_name)
            logging.info(f"Face detection model loaded: {self.face_model_name}")
        except Exception as e:
            logging.error(f"Error loading face detector: {e}")
            logging.info("Will use OpenCV Haar Cascade as fallback")
            self._load_haar_cascade()
        
        # Load eye movement classification model
        if self.model_path:
            try:
                self.classification_model = torch.load(self.model_path, map_location=self.device)
                self.classification_model.eval()
                logging.info(f"Eye movement classification model loaded from: {self.model_path}")
            except Exception as e:
                logging.warning(f"Could not load classification model: {e}")
                logging.info("Using simple heuristic-based classification instead")
                self.classification_model = None
        else:
            logging.info("No classification model provided. Using heuristic-based classification")
            self.classification_model = None
        
        return success
    
    def _load_haar_cascade(self):
        """Load Haar Cascade for face detection as fallback"""
        try:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logging.info("Haar Cascade face detector loaded")
        except Exception as e:
            logging.error(f"Failed to load Haar Cascade: {e}")
    
    def detect_faces(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            list: List of face bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        faces = []
        
        try:
            # Try YOLO face detector
            if hasattr(self.face_detector, 'predict'):
                results = self.face_detector(frame, verbose=False, conf=0.5)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0])
                        faces.append((x1, y1, x2, y2, confidence))
            
            # Fallback to Haar Cascade
            elif isinstance(self.face_detector, cv2.CascadeClassifier):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )
                
                for (x, y, w, h) in detected:
                    faces.append((x, y, x + w, y + h, 1.0))
        
        except Exception as e:
            logging.error(f"Error during face detection: {e}")
        
        return faces
    
    def extract_eye_regions(self, frame, face_box):
        """
        Extract eye regions from detected face
        
        Args:
            frame: Input frame
            face_box: Face bounding box (x1, y1, x2, y2, confidence)
            
        Returns:
            tuple: (left_eye_region, right_eye_region) or (None, None)
        """
        x1, y1, x2, y2, _ = face_box
        
        # Calculate eye region (upper 50% of face, left and right halves)
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Upper 60% of face contains eyes
        eye_region_top = y1 + int(face_height * 0.2)
        eye_region_bottom = y1 + int(face_height * 0.6)
        
        # Left eye (left 55% of face)
        left_eye_x1 = x1
        left_eye_x2 = x1 + int(face_width * 0.55)
        
        # Right eye (right 55% of face)
        right_eye_x1 = x1 + int(face_width * 0.45)
        right_eye_x2 = x2
        
        # Extract regions
        try:
            left_eye = frame[eye_region_top:eye_region_bottom, left_eye_x1:left_eye_x2]
            right_eye = frame[eye_region_top:eye_region_bottom, right_eye_x1:right_eye_x2]
            
            # Return coordinates along with regions
            left_coords = (left_eye_x1, eye_region_top, left_eye_x2, eye_region_bottom)
            right_coords = (right_eye_x1, eye_region_top, right_eye_x2, eye_region_bottom)
            
            return (left_eye, left_coords), (right_eye, right_coords)
        
        except Exception as e:
            logging.error(f"Error extracting eye regions: {e}")
            return (None, None), (None, None)
    
    def classify_eye_movement(self, eye_region):
        """
        Classify eye movement direction
        
        Args:
            eye_region: Cropped eye region image
            
        Returns:
            tuple: (class_id, class_name, confidence)
        """
        if eye_region is None or eye_region.size == 0:
            return -1, "Unknown", 0.0
        
        # If we have a trained model, use it
        if self.classification_model is not None:
            try:
                # Preprocess image
                input_tensor = self.transform(eye_region).unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    output = self.classification_model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                
                class_id = predicted_class.item()
                class_name = self.classes[class_id]
                conf = confidence.item()
                
                return class_id, class_name, conf
            
            except Exception as e:
                logging.error(f"Error during classification: {e}")
        
        # Fallback: Use simple heuristic classification
        return self._heuristic_classification(eye_region)
    
    def _heuristic_classification(self, eye_region):
        """
        Simple heuristic-based eye movement classification
        Uses iris position and eye openness
        
        Args:
            eye_region: Cropped eye region
            
        Returns:
            tuple: (class_id, class_name, confidence)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Threshold to find dark regions (pupil/iris)
            _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0, "Closed", 0.5  # Assume closed if no contours
            
            # Get largest contour (likely the iris/pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if eye is closed (very small contour area)
            area = cv2.contourArea(largest_contour)
            if area < 50:
                return 0, "Closed", 0.6
            
            # Get centroid of iris
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return 8, "Center", 0.5
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Normalize coordinates
            h, w = gray.shape
            norm_x = cx / w  # 0 to 1
            norm_y = cy / h  # 0 to 1
            
            # Classify based on position
            # Horizontal: Left (< 0.35), Center (0.35-0.65), Right (> 0.65)
            # Vertical: Top (< 0.35), Center (0.35-0.65), Bottom (> 0.65)
            
            if norm_y < 0.35:  # Top
                if norm_x < 0.35:
                    return 3, "Top Left", 0.7
                elif norm_x > 0.65:
                    return 2, "Top Right", 0.7
                else:
                    return 1, "Top Center", 0.7
            
            elif norm_y > 0.65:  # Bottom
                if norm_x < 0.35:
                    return 6, "Bottom Left", 0.7
                elif norm_x > 0.65:
                    return 5, "Bottom Right", 0.7
                else:
                    return 4, "Bottom Center", 0.7
            
            else:  # Center vertically
                if norm_x < 0.35:
                    return 7, "Center Left", 0.7
                elif norm_x > 0.65:
                    return 9, "Center Right", 0.7
                else:
                    return 8, "Center", 0.8
        
        except Exception as e:
            logging.error(f"Error in heuristic classification: {e}")
            return 8, "Center", 0.3
    
    def process_frame(self, frame, draw=True, color=(255, 0, 0), thickness=2):
        """
        Detect faces, extract eyes, and classify eye movements
        
        Args:
            frame: Input frame
            draw: Whether to draw results
            color: Color for bounding boxes
            thickness: Line thickness
            
        Returns:
            tuple: (processed_frame, detections)
                detections: list of dicts with eye movement info
        """
        processed_frame = frame.copy()
        detections = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for face_box in faces:
            x1, y1, x2, y2, face_conf = face_box
            
            # Draw face box
            if draw:
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Face: {face_conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Extract eye regions
            (left_eye, left_coords), (right_eye, right_coords) = self.extract_eye_regions(frame, face_box)
            
            # Process left eye
            if left_eye is not None and left_eye.size > 0:
                class_id, class_name, confidence = self.classify_eye_movement(left_eye)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        'eye': 'left',
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': left_coords
                    })
                    
                    if draw:
                        lx1, ly1, lx2, ly2 = left_coords
                        cv2.rectangle(processed_frame, (lx1, ly1), (lx2, ly2), color, thickness)
                        
                        label = f"L: {class_name} ({confidence:.2f})"
                        cv2.putText(processed_frame, label, (lx1, ly1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Process right eye
            if right_eye is not None and right_eye.size > 0:
                class_id, class_name, confidence = self.classify_eye_movement(right_eye)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        'eye': 'right',
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': right_coords
                    })
                    
                    if draw:
                        rx1, ry1, rx2, ry2 = right_coords
                        cv2.rectangle(processed_frame, (rx1, ry1), (rx2, ry2), color, thickness)
                        
                        label = f"R: {class_name} ({confidence:.2f})"
                        cv2.putText(processed_frame, label, (rx1, ry1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return processed_frame, detections
