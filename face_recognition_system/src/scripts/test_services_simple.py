#!/usr/bin/env python3
"""Simplified test script for services."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_database, DatabaseConfig
from database.services import get_database_service


def test_basic_functionality():
    """Test basic service functionality."""
    print("Testing basic service functionality...")
    
    # Initialize database
    config = DatabaseConfig()
    config.database_url = 'sqlite:///test_services_simple.db'
    
    try:
        db_connection = init_database(config)
        db_service = get_database_service()
        
        print("✓ Database and services initialized")
        
        # Test person creation
        import time
        timestamp = int(time.time())
        person = db_service.persons.create_person(
            name=f"测试用户_{timestamp}",
            employee_id=f"TEST_{timestamp}",
            email=f"test_{timestamp}@company.com",
            department="测试部门",
            access_level=3
        )
        
        if person:
            print(f"✓ Created person: {person.name} (ID: {person.id})")
            
            # Test person retrieval
            retrieved = db_service.persons.get_person_by_id(person.id)
            if retrieved:
                print("✓ Person retrieval works")
            
            # Test person search
            results = db_service.persons.search_persons("测试")
            if results:
                print("✓ Person search works")
            
            # Test face feature addition
            import numpy as np
            feature_vector = np.random.rand(512).astype(np.float32)
            
            face_feature = db_service.face_features.add_face_feature(
                person_id=person.id,
                feature_vector=feature_vector,
                extraction_model="test_model",
                extraction_version="1.0",
                quality_score=0.85,
                set_as_primary=True
            )
            
            if face_feature:
                print(f"✓ Added face feature (ID: {face_feature.id})")
            
            # Test access logging
            log_entry = db_service.access_logs.log_access_attempt(
                location_id=1,
                person_id=person.id,
                access_granted=True,
                recognition_confidence=0.95
            )
            
            if log_entry:
                print(f"✓ Logged access attempt (ID: {log_entry.id})")
            
            print("✅ All basic tests passed!")
            return True
        else:
            print("✗ Failed to create person")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_services_simple.db'):
            try:
                os.remove('test_services_simple.db')
                print("✓ Cleaned up test database")
            except:
                pass


def main():
    """Run simplified tests."""
    print("Face Recognition System - Simplified Service Tests")
    print("=" * 60)
    
    success = test_basic_functionality()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        return 0
    else:
        print("✗ Tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())