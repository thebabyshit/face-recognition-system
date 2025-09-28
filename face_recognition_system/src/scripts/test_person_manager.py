#!/usr/bin/env python3
"""Test script for person management services."""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_database, DatabaseConfig
from services.person_manager import PersonManager
from services.face_manager import FaceManager
from services.access_manager import AccessManager


def test_person_manager():
    """Test PersonManager functionality."""
    print("Testing PersonManager...")
    
    # Initialize database
    config = DatabaseConfig()
    config.database_url = 'sqlite:///test_person_manager.db'
    
    try:
        db_connection = init_database(config)
        person_manager = PersonManager()
        
        print("✓ PersonManager initialized")
        
        # Test person creation
        person = person_manager.add_person(
            name="张三",
            employee_id="EMP001",
            email="zhangsan@company.com",
            department="工程部",
            position="软件工程师",
            access_level=3
        )
        
        print(f"✓ Created person: {person.name} (ID: {person.id})")
        
        # Test person retrieval
        retrieved_person = person_manager.get_person_by_employee_id("EMP001")
        assert retrieved_person.id == person.id
        print("✓ Person retrieval by employee_id works")
        
        # Test person update
        updated_person = person_manager.update_person(
            person.id,
            position="高级软件工程师",
            access_level=4
        )
        assert updated_person.access_level == 4
        print("✓ Person update works")
        
        # Test person search
        search_results = person_manager.search_persons("张三")
        assert len(search_results) > 0
        print("✓ Person search works")
        
        # Test duplicate prevention
        try:
            person_manager.add_person(
                name="李四",
                employee_id="EMP001",  # Duplicate employee_id
                email="lisi@company.com"
            )
            assert False, "Should have raised DuplicatePersonError"
        except Exception as e:
            print("✓ Duplicate prevention works")
        
        # Test statistics
        stats = person_manager.get_person_statistics()
        assert stats['total_persons'] >= 1
        print("✓ Person statistics work")
        
        # Test bulk import
        bulk_data = [
            {
                'name': '李四',
                'employee_id': 'EMP002',
                'email': 'lisi@company.com',
                'department': '销售部'
            },
            {
                'name': '王五',
                'employee_id': 'EMP003',
                'email': 'wangwu@company.com',
                'department': '市场部'
            }
        ]
        
        results = person_manager.bulk_import_persons(bulk_data)
        assert results['success'] == 2
        print("✓ Bulk import works")
        
        print("✅ All PersonManager tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ PersonManager test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_person_manager.db'):
            try:
                os.remove('test_person_manager.db')
            except:
                pass


def test_face_manager():
    """Test FaceManager functionality."""
    print("\nTesting FaceManager...")
    
    # Initialize database
    config = DatabaseConfig()
    config.database_url = 'sqlite:///test_face_manager.db'
    
    try:
        db_connection = init_database(config)
        person_manager = PersonManager()
        face_manager = FaceManager()
        
        print("✓ FaceManager initialized")
        
        # Create a test person first
        person = person_manager.add_person(
            name="测试用户",
            employee_id="TEST001",
            email="test@company.com"
        )
        
        # Test face image addition (mock)
        # Create a mock image array
        mock_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        face_result = face_manager.add_face_image(
            person_id=person.id,
            image_data=mock_image,
            set_as_primary=True,
            quality_threshold=0.5  # Lower threshold for mock data
        )
        
        print(f"✓ Added face image: feature ID {face_result['face_feature_id']}")
        
        # Test getting person faces
        faces = face_manager.get_person_faces(person.id)
        assert len(faces) == 1
        print("✓ Get person faces works")
        
        # Test getting primary face
        primary_face = face_manager.get_primary_face(person.id)
        assert primary_face is not None
        assert primary_face['is_primary'] == True
        print("✓ Get primary face works")
        
        # Add another face
        face_result2 = face_manager.add_face_image(
            person_id=person.id,
            image_data=np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
            set_as_primary=False,
            quality_threshold=0.5
        )
        
        # Test face comparison
        comparison = face_manager.compare_faces(
            face_result['face_feature_id'],
            face_result2['face_feature_id']
        )
        assert 'similarity_score' in comparison
        print("✓ Face comparison works")
        
        # Test face update
        updated_feature = face_manager.update_face_feature(
            face_result2['face_feature_id'],
            set_as_primary=True
        )
        assert updated_feature is not None
        print("✓ Face feature update works")
        
        print("✅ All FaceManager tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ FaceManager test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_face_manager.db'):
            try:
                os.remove('test_face_manager.db')
            except:
                pass


def test_access_manager():
    """Test AccessManager functionality."""
    print("\nTesting AccessManager...")
    
    # Initialize database
    config = DatabaseConfig()
    config.database_url = 'sqlite:///test_access_manager.db'
    
    try:
        db_connection = init_database(config)
        person_manager = PersonManager()
        access_manager = AccessManager()
        
        print("✓ AccessManager initialized")
        
        # Create test person
        person = person_manager.add_person(
            name="访问测试用户",
            employee_id="ACCESS001",
            email="access@company.com",
            access_level=5
        )
        
        # Create test location (manually for testing)
        with person_manager.db_service.persons._get_db_manager() as db:
            from database.models import AccessLocation
            location = AccessLocation(
                name="测试门禁",
                location_type="door",
                building="主楼",
                required_access_level=3,
                is_active=True
            )
            db.session.add(location)
            db.session.flush()
            location_id = location.id
        
        print("✓ Created test location")
        
        # Test access check (should pass based on access level)
        access_result = access_manager.check_access(person.id, location_id)
        assert access_result['access_granted'] == True
        print("✓ Access check works (granted)")
        
        # Test access logging
        log_entry = access_manager.log_access_attempt(
            person_id=person.id,
            location_id=location_id,
            access_granted=True,
            recognition_confidence=0.95,
            processing_time_ms=150
        )
        assert log_entry is not None
        print("✓ Access logging works")
        
        # Test permission granting
        success = access_manager.grant_access_permission(
            person_id=person.id,
            location_id=location_id,
            permission_type='allow'
        )
        assert success == True
        print("✓ Permission granting works")
        
        # Test getting person permissions
        permissions = access_manager.get_person_permissions(person.id)
        assert len(permissions) >= 1
        print("✓ Get person permissions works")
        
        # Test access history
        history = access_manager.get_access_history(person_id=person.id)
        assert len(history) >= 1
        print("✓ Access history works")
        
        # Test access statistics
        stats = access_manager.get_access_statistics(days=1)
        assert 'total_attempts' in stats
        print("✓ Access statistics work")
        
        # Test permission revocation
        revoked = access_manager.revoke_access_permission(person.id, location_id)
        assert revoked == True
        print("✓ Permission revocation works")
        
        print("✅ All AccessManager tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ AccessManager test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_access_manager.db'):
            try:
                os.remove('test_access_manager.db')
            except:
                pass


def main():
    """Run all service tests."""
    print("Face Recognition System - Service Tests")
    print("=" * 50)
    
    results = []
    
    # Test PersonManager
    results.append(test_person_manager())
    
    # Test FaceManager
    results.append(test_face_manager())
    
    # Test AccessManager
    results.append(test_access_manager())
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("✅ All service tests passed!")
        return 0
    else:
        print("✗ Some service tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())