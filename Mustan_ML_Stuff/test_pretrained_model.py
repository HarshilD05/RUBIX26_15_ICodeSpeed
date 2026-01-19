"""
Quick test script for pretrainedModel.pth
Provides easy ways to test your model with different inputs
"""

import torch
import numpy as np
from model_inspector import ModelInspector
import cv2
from PIL import Image


def test_with_image(model_path, image_path):
    """
    Test model with an actual image
    
    Args:
        model_path: Path to .pth file
        image_path: Path to image file
    """
    print("Testing model with image...")
    
    # Load model
    inspector = ModelInspector(model_path)
    if not inspector.load_model():
        print("Failed to load model")
        return
    
    # Load and preprocess image
    try:
        # Try with PIL
        image = Image.open(image_path).convert('RGB')
        
        # Try common preprocessing
        transforms_to_try = [
            (224, 224),  # ImageNet
            (299, 299),  # Inception
            (256, 256),  # Common
            (128, 128),  # Small
        ]
        
        for size in transforms_to_try:
            print(f"\nTrying with image size: {size}")
            
            # Resize
            img_resized = image.resize(size)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            
            # Add batch dimension (1, C, H, W)
            img_tensor = img_tensor.unsqueeze(0)
            
            # Try prediction
            result = inspector.predict_sample(img_tensor)
            
            if result is not None:
                print("✓ Success with this image size!")
                return result
        
    except Exception as e:
        print(f"Error processing image: {e}")


def test_with_random_inputs(model_path):
    """
    Test model with various random inputs to find what it expects
    
    Args:
        model_path: Path to .pth file
    """
    print("Testing model with random inputs...")
    inspector = ModelInspector(model_path)
    inspector.full_inspection()


def test_with_custom_shape(model_path, shape):
    """
    Test model with a specific input shape
    
    Args:
        model_path: Path to .pth file
        shape: Tuple of dimensions e.g., (1, 3, 224, 224)
    """
    print(f"Testing model with custom shape: {shape}")
    
    inspector = ModelInspector(model_path)
    if not inspector.load_model():
        print("Failed to load model")
        return
    
    # Create random input
    random_input = torch.randn(*shape)
    
    # Make prediction
    result = inspector.predict_sample(random_input)
    return result


def interactive_testing(model_path="pretrainedModel.pth"):
    """Interactive testing interface"""
    print("=" * 70)
    print(" " * 20 + "MODEL TESTING INTERFACE")
    print("=" * 70)
    
    inspector = ModelInspector(model_path)
    
    while True:
        print("\n" + "-" * 70)
        print("What would you like to do?")
        print("-" * 70)
        print("1. Full model inspection (architecture, parameters, auto-detect inputs)")
        print("2. Test with an image file")
        print("3. Test with custom input shape")
        print("4. Test with random inputs of various sizes")
        print("5. Quick prediction test")
        print("6. Exit")
        print("-" * 70)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\n" + "=" * 70)
            inspector.full_inspection()
        
        elif choice == '2':
            image_path = input("Enter path to image file: ").strip()
            test_with_image(model_path, image_path)
        
        elif choice == '3':
            print("\nCommon shapes:")
            print("  - Image: 1,3,224,224 (ImageNet)")
            print("  - Image: 1,3,299,299 (Inception)")
            print("  - Vector: 1,784 (MNIST flattened)")
            print("  - Vector: 1,512 (embedding)")
            
            shape_str = input("\nEnter shape (e.g., 1,3,224,224): ").strip()
            try:
                shape = tuple(map(int, shape_str.split(',')))
                test_with_custom_shape(model_path, shape)
            except:
                print("Invalid shape format!")
        
        elif choice == '4':
            test_with_random_inputs(model_path)
        
        elif choice == '5':
            if not inspector.load_model():
                continue
            
            print("\nQuick test with common shapes...")
            shapes = [
                (1, 3, 224, 224),
                (1, 3, 299, 299),
                (1, 784),
                (1, 512)
            ]
            
            for shape in shapes:
                print(f"\n→ Testing {shape}...")
                result = inspector.test_input_shape(shape)
                if result is not None:
                    print("✓ This shape works!")
                    
                    # Try a prediction
                    sample = torch.randn(*shape)
                    inspector.predict_sample(sample)
                    break
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    import sys
    
    # Check if model path provided as argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "pretrainedModel.pth"
    
    # Run interactive testing
    interactive_testing(model_path)
