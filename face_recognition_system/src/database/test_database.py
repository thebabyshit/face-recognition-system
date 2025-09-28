"""Database functionality tests."""

import os
import tempfile
import unittest
import numpy as np
from datetime import datetime, timezone

from .connection import DatabaseConnection, DatabaseConfig
from .services import DatabaseService
from .models import Base


class TestDatabase(unittest.TestCase):
    """Test database functionality."""
    
    def setUp(self):
        """Set up test database."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Configure test database
        self.config = DatabaseConfig()
        self.config.database_url = f'sqlite:///{self.temp_db.name}'
        self.config.echo = False
        
        # Initialize database connection
        self.db_connection = DatabaseConnection(self.config)
        self.db_connection.create_tables()
        
        # Initialize service
        self.db_service = DatabaseService()
    
    def tearDown(self):
        """Clean up test database."""
        self.db_connection.close()
        os.unlink(self.temp_db.name)
    
    def test_database_connection(self):
        """Test database connection."""
        self.assertTrue(self.db_connection.test_connection())
    
    def test_create_person(self):
        """Test creating a person."""
        person = self.db_service.persons.create_person(
            name='John Doe',
            employee_id='EMP001',
            email='john.doe@company.com',
            department='Engineering',
            position='Software Engineer',
            access_level=3
        )
        
        self.assertIsNotNone(person)
        self.assertEqual(person.name, 'John Doe')
        self.assertEqual(person.employee_id, 'EMP001')
        self.assertEqual(person.email, 'john.doe@company.com')
        self.assertEqual(person.access_level, 3)
        self.assertTrue(person.is_active)
    
    def test_get_person_by_employee_id(self):
        """Test getting person by employee ID."""
        # Create person
        created_person = self.db_service.persons.create_person(
            name='Jane Smith',
            employee_id='EMP002',
            email='jane.smith@company.com'
        )
        
        # Retrieve person
        retrieved_person = self.db_service.persons.get_person_by_employee_id('EMP002')
        
        self.assertIsNotNone(retrieved_person)
        self.assertEqual(retrieved_person.id, created_person.id)
        self.assertEqual(retrieved_person.name, 'Jane Smith')
    
    def test_duplicate_employee_id(self):
        """Test that duplicate employee IDs are not allowed."""
        # Create first person
        person1 = self.db_service.persons.create_person(
            name='Person One',
            employee_id='EMP003'
        )
        self.assertIsNotNone(person1)
        
        # Try to create second person with same employee ID
        person2 = self.db_service.persons.create_person(
            name='Person Two',
            employee_id='EMP003'
        )
        self.assertIsNone(person2)
    
    def test_add_face_feature(self):
        """Test adding face feature."""
        # Create person
        person = self.db_service.persons.create_person(
            name='Test Person',
            employee_id='EMP004'
        )
        
        # Create dummy feature vector
        feature_vector = np.random.rand(512).astype(np.float32)
        
        # Add face feature
        feature = self.db_service.face_features.add_face_feature(
            person_id=person.id,
            feature_vector=feature_vector,
            extraction_model='test_model',
            extraction_version='1.0',
            quality_score=0.95,
            confidence_score=0.98,
            set_as_primary=True
        )
        
        self.assertIsNotNone(feature)
        self.assertEqual(feature.person_id, person.id)
        self.assertEqual(feature.extraction_model, 'test_model')
        self.assertEqual(feature.feature_dimension, 512)
        self.assertTrue(feature.is_primary)
        self.assertTrue(feature.is_active)
        
        # Verify feature vector
        retrieved_vector = feature.get_feature_vector()
        np.testing.assert_array_almost_equal(feature_vector, retrieved_vector)
    
    def test_primary_feature_management(self):
        """Test primary feature management."""
        # Create person
        person = self.db_service.persons.create_person(
            name='Test Person',
            employee_id='EMP005'
        )
        
        # Add first feature as primary
        feature1 = self.db_service.face_features.add_face_feature(
            person_id=person.id,
            feature_vector=np.random.rand(512).astype(np.float32),
            extraction_model='test_model',
            extraction_version='1.0',
            set_as_primary=True
        )
        
        # Add second feature as primary (should unset first)
        feature2 = self.db_service.face_features.add_face_feature(
            person_id=person.id,
            feature_vector=np.random.rand(512).astype(np.float32),
            extraction_model='test_model',
            extraction_version='1.0',
            set_as_primary=True
        )
        
        # Refresh features from database
        with self.db_service.face_features._get_db_manager() as db:
            feature1_updated = db.face_features.get_by_id(feature1.id)
            feature2_updated = db.face_features.get_by_id(feature2.id)
        
        # Only feature2 should be primary
        self.assertFalse(feature1_updated.is_primary)
        self.assertTrue(feature2_updated.is_primary)
    
    def test_access_logging(self):
        """Test access logging."""
        # Create person and location (assuming location exists)
        person = self.db_service.persons.create_person(
            name='Test Person',
            employee_id='EMP006'
        )
        
        # Log successful access
        log_entry = self.db_service.access_logs.log_access_attempt(
            location_id=1,  # Assuming location exists
            person_id=person.id,
            access_granted=True,
            recognition_confidence=0.95,
            similarity_score=0.92,
            processing_time_ms=150
        )
        
        self.assertIsNotNone(log_entry)
        self.assertEqual(log_entry.person_id, person.id)
        self.assertTrue(log_entry.access_granted)
        self.assertEqual(log_entry.recognition_confidence, 0.95)
        self.assertEqual(log_entry.processing_time_ms, 150)
    
    def test_search_persons(self):
        """Test person search functionality."""
        # Create test persons
        persons = [
            ('John Smith', 'EMP007'),
            ('Jane Smith', 'EMP008'),
            ('Bob Johnson', 'EMP009'),
            ('Alice Johnson', 'EMP010')
        ]
        
        for name, emp_id in persons:
            self.db_service.persons.create_person(name=name, employee_id=emp_id)
        
        # Search for "Smith"
        results = self.db_service.persons.search_persons('Smith')
        self.assertEqual(len(results), 2)
        
        # Search for "Johnson"
        results = self.db_service.persons.search_persons('Johnson')
        self.assertEqual(len(results), 2)
        
        # Search for "John" (should match both John Smith and Bob Johnson)
        results = self.db_service.persons.search_persons('John')
        self.assertEqual(len(results), 2)
    
    def test_deactivate_person(self):
        """Test person deactivation."""
        # Create person
        person = self.db_service.persons.create_person(
            name='Test Person',
            employee_id='EMP011'
        )
        
        # Verify person is active
        self.assertTrue(person.is_active)
        
        # Deactivate person
        success = self.db_service.persons.deactivate_person(person.id)
        self.assertTrue(success)
        
        # Verify person is deactivated
        updated_person = self.db_service.persons.get_person_by_id(person.id)
        self.assertFalse(updated_person.is_active)
    
    def test_get_persons_with_features(self):
        """Test getting persons with face features."""
        # Create persons
        person1 = self.db_service.persons.create_person(
            name='Person With Feature',
            employee_id='EMP012'
        )
        person2 = self.db_service.persons.create_person(
            name='Person Without Feature',
            employee_id='EMP013'
        )
        
        # Add feature to person1
        self.db_service.face_features.add_face_feature(
            person_id=person1.id,
            feature_vector=np.random.rand(512).astype(np.float32),
            extraction_model='test_model',
            extraction_version='1.0'
        )
        
        # Get persons with features
        persons_with_features = self.db_service.persons.get_persons_with_features()
        
        # Should only include person1
        self.assertEqual(len(persons_with_features), 1)
        self.assertEqual(persons_with_features[0].id, person1.id)
    
    def test_access_statistics(self):
        """Test access statistics calculation."""
        # Create person
        person = self.db_service.persons.create_person(
            name='Test Person',
            employee_id='EMP014'
        )
        
        # Log some access attempts
        for i in range(10):
            self.db_service.access_logs.log_access_attempt(
                location_id=1,
                person_id=person.id,
                access_granted=i < 8,  # 8 successful, 2 failed
                processing_time_ms=100 + i * 10
            )
        
        # Get statistics
        stats = self.db_service.access_logs.get_access_statistics(days=1)
        
        self.assertEqual(stats['total_attempts'], 10)
        self.assertEqual(stats['successful_attempts'], 8)
        self.assertEqual(stats['failed_attempts'], 2)
        self.assertEqual(stats['success_rate'], 80.0)
        self.assertGreater(stats['average_processing_time_ms'], 0)


def run_database_tests():
    """Run all database tests."""
    unittest.main(module=__name__, verbosity=2)


if __name__ == '__main__':
    run_database_tests()