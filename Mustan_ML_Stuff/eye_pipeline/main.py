"""
Main Application - 
Real-time eye movement detection using camera feed
"""

from modules import EyeMovementPipeline, Config


def main():
    """Main entry point"""
    # Configure pipeline
    Config.WINDOW_NAME = "Eye Movement Detection Pipeline"
    Config.EYE_DETECTION_ENABLED = True
    Config.EYE_MODEL_PATH = None  # Set to your model path if you have one
    Config.EYE_CONFIDENCE_THRESHOLD = 0.4
    Config.EYE_BOX_COLOR = (255, 0, 0)  # Blue
    Config.EYE_BOX_THICKNESS = 2
    Config.EYE_SHOW_CONFIDENCE = True
    Config.EYE_SHOW_LABEL = True
    
    Config.FACE_DETECTION_ENABLED = True
    Config.FACE_MODEL_NAME = "yolov8n.pt"  # Use yolov8n-face.pt if available
    Config.FACE_CONFIDENCE_THRESHOLD = 0.5
    
    # Create and run pipeline
    pipeline = EyeMovementPipeline()
    
    print("\n" + "="*70)
    print("Eye Movement Detection Pipeline")
    print("="*70)
    print("Starting eye movement detection...")
    print("\nEye Movement Classes:")
    print("  1. Closed")
    print("  2. Top Center")
    print("  3. Top Right")
    print("  4. Top Left")
    print("  5. Bottom Center")
    print("  6. Bottom Right")
    print("  7. Bottom Left")
    print("  8. Center Left")
    print("  9. Center")
    print(" 10. Center Right")
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("\nNote: First run will download YOLOv8 model (~6MB)")
    print("      Using heuristic-based classification (no ML model)")
    print("="*70 + "\n")
    
    try:
        pipeline.run()
        
        # Print statistics after exit
        print("\n" + "="*70)
        print("Session Statistics")
        print("="*70)
        stats = pipeline.get_movement_statistics()
        
        if stats:
            print(f"Total Detections: {stats.get('total_detections', 0)}")
            print("\nMovement Distribution:")
            for movement, count in stats.get('movement_counts', {}).items():
                print(f"  {movement}: {count}")
        else:
            print("No detections recorded")
        
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
