#!/usr/bin/env python3
"""Simple database functionality test script."""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import with proper path
from database.connection import init_database, DatabaseConfig
from database.services import get_database_service


def test_basic_operations():
    """Test basic database operations."""
    print("Testing basic database operations...")
    
    # Initialize database
    config = DatabaseConfig()
    config.database_url = 'sqlite:///test_face_recognition.db'
    
    try:
        # Initialize database
        db_connection = init_database(config)
        db_service = get_database_service()
        
        print("✓ Database connection established")
        
        # Test person creation
        person = db_service.persons.create_person(
            name='Test User',
            employee_id='TEST001',
            email='test@company.com',
            department='Testing',
            access_level=3
        )
        
        if person:
            print(f"✓ Created person: {person.name} (ID: {person.id})")
        else:
            print("✗ Failed to create person")
            return False
        
        # Test person retrieval
        retrieved_person = db_service.persons.get_person_by_employee_id('TEST001')
        if retrieved_person and retrieved_person.id == person.id:
            print("✓ Person retrieval successful")
        else:
            print("✗ Person retrieval failed")
            return False
        
        # Test face feature addition
        feature_vector = np.random.rand(512).astype(np.float32)
        feature = db_service.face_features.add_face_feature(
            person_id=person.id,
            feature_vector=feature_vector,
            extraction_model='test_model',
            extraction_version='1.0',
            quality_score=0.95,
            set_as_primary=True
        )
        
        if feature:
            print(f"✓ Added face feature (ID: {feature.id})")
        else:
            print("✗ Failed to add face feature")
            return False
        
        # Test feature vector retrieval
        retrieved_vector = feature.get_feature_vector()
        if np.allclose(feature_vector, retrieved_vector):
            print("✓ Feature vector storage/retrieval successful")
        else:
            print("✗ Feature vector mismatch")
            return False
        
        # Test access logging
        log_entry = db_service.access_logs.log_access_attempt(
            location_id=1,
            person_id=person.id,
            access_granted=True,
            recognition_confidence=0.95,
            processing_time_ms=120
        )
        
        if log_entry:
            print(f"✓ Access log created (ID: {log_entry.id})")
        else:
            print("✗ Failed to create access log")
            return False
        
        # Test search functionality
        search_results = db_service.persons.search_persons('Test')
        if len(search_results) > 0 and search_results[0].id == person.id:
            print("✓ Person search successful")
        else:
            print("✗ Person search failed")
            return False
        
        # Test statistics
        stats = db_service.access_logs.get_access_statistics(days=1)
        if stats and stats.get('total_attempts', 0) > 0:
            print("✓ Access statistics calculation successful")
            print(f"  - Total attempts: {stats['total_attempts']}")
            print(f"  - Success rate: {stats['success_rate']}%")
        else:
            print("✗ Access statistics calculation failed")
            return False
        
        print("\n✓ All basic database operations successful!")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False
    finally:
        # Clean up test database
        try:
            if os.path.exists('test_face_recognition.db'):
                os.remove('test_face_recognition.db')
                print("✓ Test database cleaned up")
        except PermissionError:
            print("⚠ Could not remove test database file (file in use)")


def test_performance():
    """Test database performance with multiple operations."""
    print("\nTesting database performance...")
    
    config = DatabaseConfig()
    config.database_url = 'sqlite:///perf_test.db'
    
    try:
        import time
        
        # Initialize database
        db_connection = init_database(config)
        db_service = get_database_service()
        
        # Test bulk person creation
        start_time = time.time()
        persons = []
        
        for i in range(100):
            person = db_service.persons.create_person(
                name=f'Test Person {i}',
                employee_id=f'TEST{i:03d}',
                email=f'test{i}@company.com',
                department='Testing'
            )
            if person:
                persons.append(person)
        
        person_creation_time = time.time() - start_time
        print(f"✓ Created {len(persons)} persons in {person_creation_time:.2f} seconds")
        
        # Test bulk feature addition
        start_time = time.time()
        features_added = 0
        
        for person in persons[:50]:  # Add features to first 50 persons
            feature_vector = np.random.rand(512).astype(np.float32)
            feature = db_service.face_features.add_face_feature(
                person_id=person.id,
                feature_vector=feature_vector,
                extraction_model='test_model',
                extraction_version='1.0',
                quality_score=np.random.uniform(0.7, 1.0)
            )
            if feature:
                features_added += 1
        
        feature_creation_time = time.time() - start_time
        print(f"✓ Added {features_added} features in {feature_creation_time:.2f} seconds")
        
        # Test bulk access logging
        start_time = time.time()
        logs_created = 0
        
        for i in range(200):
            person = np.random.choice(persons)
            log_entry = db_service.access_logs.log_access_attempt(
                location_id=1,
                person_id=person.id,
                access_granted=np.random.choice([True, False]),
                recognition_confidence=np.random.uniform(0.5, 1.0),
                processing_time_ms=np.random.randint(50, 300)
            )
            if log_entry:
                logs_created += 1
        
        logging_time = time.time() - start_time
        print(f"✓ Created {logs_created} access logs in {logging_time:.2f} seconds")
        
        # Test search performance
        start_time = time.time()
        search_results = db_service.persons.search_persons('Test Person')
        search_time = time.time() - start_time
        print(f"✓ Search returned {len(search_results)} results in {search_time:.3f} seconds")
        
        print(f"\nPerformance Summary:")
        print(f"  - Person creation: {len(persons)/person_creation_time:.1f} persons/sec")
        print(f"  - Feature addition: {features_added/feature_creation_time:.1f} features/sec")
        print(f"  - Access logging: {logs_created/logging_time:.1f} logs/sec")
        print(f"  - Search time: {search_time*1000:.1f} ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False
    finally:
        # Clean up test database
        try:
            if os.path.exists('perf_test.db'):
                os.remove('perf_test.db')
                print("✓ Performance test database cleaned up")
        except PermissionError:
            print("⚠ Could not remove performance test database file (file in use)")


def main():
    """Run all database tests."""
    print("Face Recognition System - Database Tests")
    print("=" * 50)
    
    # Run basic functionality tests
    basic_success = test_basic_operations()
    
    # Run performance tests
    perf_success = test_performance()
    
    print("\n" + "=" * 50)
    if basic_success and perf_success:
        print("✓ All database tests passed!")
        return 0
    else:
        print("✗ Some database tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())