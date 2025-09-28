"""Test script to verify the face recognition system implementation."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_dataset_loading():
    """Test dataset loading functionality."""
    print("Testing dataset loading...")
    
    try:
        from data.dataset import FacecapDataset, analyze_dataset
        
        # Test dataset analysis
        stats = analyze_dataset('../facecap')
        print(f"✓ Dataset analysis successful")
        print(f"  - Train samples: {stats['train']['num_samples']:,}")
        print(f"  - Val samples: {stats['val']['num_samples']:,}")
        print(f"  - Test samples: {stats['test']['num_samples']:,}")
        print(f"  - Total classes: {stats['train']['num_classes']}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation and forward pass."""
    print("\nTesting model creation...")
    
    try:
        from models.face_recognition import FaceRecognitionModel
        
        # Create model
        model = FaceRecognitionModel(num_classes=500, feat_dim=512)
        print(f"✓ Model created successfully")
        
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 112, 112)
        dummy_labels = torch.randint(0, 500, (batch_size,))
        
        with torch.no_grad():
            features, logits = model(dummy_input, dummy_labels)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Logits shape: {logits.shape}")
        
        # Test feature extraction
        features_only = model.extract_features(dummy_input)
        print(f"✓ Feature extraction successful")
        print(f"  - Features shape: {features_only.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_face_detector():
    """Test face detection functionality."""
    print("\nTesting face detector...")
    
    try:
        from models.face_detector import FaceDetector, BoundingBox
        
        # Create detector (without loading MTCNN to avoid dependency issues)
        detector = FaceDetector()
        print(f"✓ Face detector created successfully")
        
        # Test bounding box
        bbox = BoundingBox(10, 20, 100, 120, confidence=0.95)
        print(f"✓ BoundingBox created: {bbox.to_dict()}")
        
        return True
    except Exception as e:
        print(f"✗ Face detector test failed: {e}")
        return False


def test_image_quality():
    """Test image quality assessment."""
    print("\nTesting image quality assessment...")
    
    try:
        from utils.image_quality import ImageQualityAssessor
        
        # Create quality assessor
        assessor = ImageQualityAssessor()
        print(f"✓ Image quality assessor created")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        sharpness = assessor.assess_sharpness(dummy_image)
        brightness = assessor.assess_brightness(dummy_image)
        contrast = assessor.assess_contrast(dummy_image)
        
        print(f"✓ Quality assessment successful")
        print(f"  - Sharpness: {sharpness:.2f}")
        print(f"  - Brightness: {brightness:.2f}")
        print(f"  - Contrast: {contrast:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Image quality test failed: {e}")
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics calculation...")
    
    try:
        from utils.metrics import calculate_accuracy, calculate_metrics
        
        # Test with dummy data
        predictions = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        labels = np.array([0, 1, 2, 2, 0, 2, 1, 1])
        
        accuracy = calculate_accuracy(predictions, labels)
        print(f"✓ Accuracy calculation: {accuracy:.3f}")
        
        metrics = calculate_metrics(predictions, labels)
        print(f"✓ Comprehensive metrics calculated")
        print(f"  - Accuracy: {metrics['accuracy']:.3f}")
        print(f"  - Precision: {metrics['precision']:.3f}")
        print(f"  - Recall: {metrics['recall']:.3f}")
        print(f"  - F1 Score: {metrics['f1_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import settings, get_settings
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Model path: {settings.model_path}")
        print(f"  - Feature dimension: {settings.feature_dimension}")
        print(f"  - Recognition threshold: {settings.recognition_threshold}")
        print(f"  - Batch size: {settings.batch_size}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("FACE RECOGNITION SYSTEM - IMPLEMENTATION TEST")
    print("="*60)
    
    tests = [
        test_config,
        test_dataset_loading,
        test_model_creation,
        test_face_detector,
        test_image_quality,
        test_metrics,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())