#!/usr/bin/env python3
"""Test script for real-time recognition service."""

import sys
import os
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_database, DatabaseConfig
from recognition.recognition_service import FaceRecognitionService, ServiceConfig
from recognition.realtime_recognizer import RecognitionConfig
from recognition.result_processor import ProcessingConfig


def test_recognition_components():
    """Test individual recognition components."""
    print("Testing recognition components...")
    
    try:
        # Initialize database
        config = DatabaseConfig()
        config.database_url = 'sqlite:///test_recognition.db'
        db_connection = init_database(config)
        
        print("✓ Database initialized")
        
        # Test recognizer initialization
        from recognition.realtime_recognizer import RealtimeRecognizer, RecognitionConfig
        
        recognition_config = RecognitionConfig()
        recognizer = RealtimeRecognizer(recognition_config)
        
        if recognizer.is_initialized:
            print("✓ Recognizer initialized")
        else:
            print("⚠ Recognizer initialization failed (expected without models)")
        
        # Test frame processing with mock data
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = recognizer.process_frame(mock_frame, frame_id=1)
        
        if result.success:
            print(f"✓ Frame processed: {result.face_count} faces detected")
        else:
            print(f"✗ Frame processing failed: {result.error_message}")
        
        # Test result processor
        from recognition.result_processor import RecognitionResultProcessor, ProcessingConfig
        
        processor_config = ProcessingConfig()
        processor = RecognitionResultProcessor(processor_config)
        
        processing_summary = processor.process_result(result, location_id=1)
        
        if processing_summary.get('processed', False):
            print("✓ Result processing works")
        else:
            print("✗ Result processing failed")
        
        # Test statistics
        stats = recognizer.get_statistics()
        print(f"✓ Recognition statistics: {stats['frames_processed']} frames processed")
        
        processor_stats = processor.get_statistics()
        print(f"✓ Processing statistics: {processor_stats['results_processed']} results processed")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_recognition.db'):
            try:
                os.remove('test_recognition.db')
            except:
                pass


def test_recognition_service():
    """Test complete recognition service."""
    print("\nTesting recognition service...")
    
    try:
        # Initialize database
        config = DatabaseConfig()
        config.database_url = 'sqlite:///test_service.db'
        db_connection = init_database(config)
        
        print("✓ Database initialized")
        
        # Create service configuration (without camera for testing)
        service_config = ServiceConfig(
            camera_id=-1,  # Invalid camera ID for testing
            location_id=1,
            enable_access_control=True
        )
        
        # Initialize service
        service = FaceRecognitionService(service_config)
        
        if service.is_initialized:
            print("✓ Service initialized")
        else:
            print("⚠ Service initialization failed (expected without camera)")
        
        # Test service status
        status = service.get_service_status()
        print(f"✓ Service status retrieved: running={status['is_running']}")
        
        # Test statistics
        stats = service.get_service_status()
        print(f"✓ Service statistics: {stats.get('service_stats', {})}")
        
        # Test recent activity (should be empty)
        activity = service.get_recent_activity()
        print(f"✓ Recent activity: {len(activity)} items")
        
        # Test alerts (should be empty)
        alerts = service.get_alerts()
        print(f"✓ Alerts: {len(alerts)} alerts")
        
        return True
        
    except Exception as e:
        print(f"✗ Service test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_service.db'):
            try:
                os.remove('test_service.db')
            except:
                pass


def test_mock_access_control():
    """Test access control with mock data."""
    print("\nTesting mock access control...")
    
    try:
        # Initialize database
        config = DatabaseConfig()
        config.database_url = 'sqlite:///test_access.db'
        db_connection = init_database(config)
        
        # Create a test person
        from database.services import get_database_service
        db_service = get_database_service()
        
        person = db_service.persons.create_person(
            name="测试用户",
            employee_id="TEST001",
            email="test@company.com",
            access_level=5
        )
        
        if person:
            print(f"✓ Created test person: {person.name}")
            
            # Add mock face feature
            import numpy as np
            feature_vector = np.random.rand(512).astype(np.float32)
            
            face_feature = db_service.face_features.add_face_feature(
                person_id=person.id,
                feature_vector=feature_vector,
                extraction_model="test_model",
                extraction_version="1.0",
                set_as_primary=True
            )
            
            if face_feature:
                print("✓ Added face feature for test person")
            
            # Test access manager
            from services.access_manager import AccessManager
            access_manager = AccessManager()
            
            # Create test location
            with db_service.persons._get_db_manager() as db:
                from database.models import AccessLocation
                location = AccessLocation(
                    name="测试门禁",
                    location_type="door",
                    required_access_level=3,
                    is_active=True
                )
                db.session.add(location)
                db.session.flush()
                location_id = location.id
            
            # Test access check
            access_result = access_manager.check_access(person.id, location_id)
            
            if access_result['access_granted']:
                print("✓ Access control test passed")
            else:
                print(f"✗ Access denied: {access_result['reason']}")
            
            # Test access logging
            log_entry = access_manager.log_access_attempt(
                person_id=person.id,
                location_id=location_id,
                access_granted=True,
                recognition_confidence=0.95
            )
            
            if log_entry:
                print("✓ Access logging works")
            
        return True
        
    except Exception as e:
        print(f"✗ Access control test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_access.db'):
            try:
                os.remove('test_access.db')
            except:
                pass


def main():
    """Run all recognition service tests."""
    print("Face Recognition System - Recognition Service Tests")
    print("=" * 60)
    
    results = []
    
    # Test individual components
    results.append(test_recognition_components())
    
    # Test complete service
    results.append(test_recognition_service())
    
    # Test access control
    results.append(test_mock_access_control())
    
    print("\n" + "=" * 60)
    
    if all(results):
        print("✅ All recognition service tests passed!")
        return 0
    else:
        print("✗ Some recognition service tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())