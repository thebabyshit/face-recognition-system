"""Database service layer - High-level business logic operations."""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import hashlib
import secrets

from .connection import get_session
from .repositories import DatabaseManager
from .models import Person, FaceFeature, AccessLog, AccessLocation, AccessPermission

logger = logging.getLogger(__name__)


class PersonService:
    """Service for person-related operations."""
    
    def __init__(self):
        pass
    
    def _get_db_manager(self) -> DatabaseManager:
        """Get database manager with session."""
        from .connection import get_database_connection
        db_connection = get_database_connection()
        return DatabaseManager(db_connection.session_factory)
    
    def create_person(self, name: str, employee_id: str = None, email: str = None, 
                     department: str = None, position: str = None, 
                     access_level: int = 1, **kwargs) -> Optional[Person]:
        """Create a new person."""
        try:
            with self._get_db_manager() as db:
                # Check for existing employee_id or email
                if employee_id and db.persons.get_by_employee_id(employee_id):
                    logger.warning(f"Person with employee_id {employee_id} already exists")
                    return None
                
                if email and db.persons.get_by_email(email):
                    logger.warning(f"Person with email {email} already exists")
                    return None
                
                person_data = {
                    'name': name,
                    'employee_id': employee_id,
                    'email': email,
                    'department': department,
                    'position': position,
                    'access_level': access_level,
                    **kwargs
                }
                
                person = db.persons.create(**person_data)
                if person:
                    logger.info(f"Created person: {person.name} (ID: {person.id})")
                return person
        except Exception as e:
            logger.error(f"Error creating person: {e}")
            return None
    
    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        try:
            with self._get_db_manager() as db:
                return db.persons.get_by_id(person_id)
        except Exception as e:
            logger.error(f"Error getting person by ID {person_id}: {e}")
            return None
    
    def get_person_by_employee_id(self, employee_id: str) -> Optional[Person]:
        """Get person by employee ID."""
        try:
            with self._get_db_manager() as db:
                return db.persons.get_by_employee_id(employee_id)
        except Exception as e:
            logger.error(f"Error getting person by employee_id {employee_id}: {e}")
            return None
    
    def get_person_by_email(self, email: str) -> Optional[Person]:
        """Get person by email."""
        try:
            with self._get_db_manager() as db:
                return db.persons.get_by_email(email)
        except Exception as e:
            logger.error(f"Error getting person by email {email}: {e}")
            return None
    
    def search_persons(self, query: str, limit: int = 50) -> List[Person]:
        """Search persons by name."""
        try:
            with self._get_db_manager() as db:
                return db.persons.search_by_name(query, limit)
        except Exception as e:
            logger.error(f"Error searching persons: {e}")
            return []
    
    def update_person(self, person_id: int, **kwargs) -> Optional[Person]:
        """Update person information."""
        try:
            with self._get_db_manager() as db:
                return db.persons.update(person_id, **kwargs)
        except Exception as e:
            logger.error(f"Error updating person {person_id}: {e}")
            return None
    
    def deactivate_person(self, person_id: int) -> bool:
        """Deactivate a person (soft delete)."""
        try:
            with self._get_db_manager() as db:
                person = db.persons.update(person_id, is_active=False)
                if person:
                    logger.info(f"Deactivated person: {person.name} (ID: {person_id})")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deactivating person {person_id}: {e}")
            return False
    
    def get_persons_with_features(self) -> List[Person]:
        """Get all persons who have face features."""
        try:
            with self._get_db_manager() as db:
                return db.persons.get_persons_with_features()
        except Exception as e:
            logger.error(f"Error getting persons with features: {e}")
            return []


class FaceFeatureService:
    """Service for face feature operations."""
    
    def __init__(self):
        pass
    
    def _get_db_manager(self) -> DatabaseManager:
        """Get database manager with session."""
        from .connection import get_database_connection
        db_connection = get_database_connection()
        return DatabaseManager(db_connection.session_factory)
    
    def add_face_feature(self, person_id: int, feature_vector: np.ndarray,
                        extraction_model: str, extraction_version: str,
                        image_path: str = None, quality_score: float = None,
                        confidence_score: float = None, bbox: Dict = None,
                        landmarks: Dict = None, set_as_primary: bool = False,
                        **kwargs) -> Optional[FaceFeature]:
        """Add a face feature for a person."""
        try:
            with self._get_db_manager() as db:
                # Check if person exists
                person = db.persons.get_by_id(person_id)
                if not person:
                    logger.error(f"Person with ID {person_id} not found")
                    return None
                
                # Calculate image hash if image_path provided
                image_hash = None
                if image_path:
                    try:
                        with open(image_path, 'rb') as f:
                            image_hash = hashlib.sha256(f.read()).hexdigest()
                    except Exception as e:
                        logger.warning(f"Could not calculate image hash: {e}")
                
                # Prepare feature data
                feature_data = {
                    'person_id': person_id,
                    'extraction_model': extraction_model,
                    'extraction_version': extraction_version,
                    'image_path': image_path,
                    'image_hash': image_hash,
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'landmarks': landmarks,
                    'is_primary': set_as_primary,
                    **kwargs
                }
                
                # Add bounding box coordinates if provided
                if bbox:
                    feature_data.update({
                        'face_bbox_x1': bbox.get('x1'),
                        'face_bbox_y1': bbox.get('y1'),
                        'face_bbox_x2': bbox.get('x2'),
                        'face_bbox_y2': bbox.get('y2'),
                    })
                
                # Create feature
                feature = db.face_features.create(**feature_data)
                if feature:
                    # Set feature vector
                    feature.set_feature_vector(feature_vector)
                    
                    # If setting as primary, unset other primary features
                    if set_as_primary:
                        db.face_features.set_primary_feature(feature.id)
                    
                    logger.info(f"Added face feature for person {person_id} (Feature ID: {feature.id})")
                    return feature
                
                return None
        except Exception as e:
            logger.error(f"Error adding face feature: {e}")
            return None
    
    def get_person_features(self, person_id: int, active_only: bool = True) -> List[FaceFeature]:
        """Get all face features for a person."""
        try:
            with self._get_db_manager() as db:
                return db.face_features.get_by_person_id(person_id, active_only)
        except Exception as e:
            logger.error(f"Error getting features for person {person_id}: {e}")
            return []
    
    def get_primary_feature(self, person_id: int) -> Optional[FaceFeature]:
        """Get primary face feature for a person."""
        try:
            with self._get_db_manager() as db:
                return db.face_features.get_primary_feature(person_id)
        except Exception as e:
            logger.error(f"Error getting primary feature for person {person_id}: {e}")
            return None
    
    def set_primary_feature(self, feature_id: int) -> bool:
        """Set a feature as primary."""
        try:
            with self._get_db_manager() as db:
                return db.face_features.set_primary_feature(feature_id)
        except Exception as e:
            logger.error(f"Error setting primary feature {feature_id}: {e}")
            return False
    
    def get_all_active_features(self) -> List[FaceFeature]:
        """Get all active face features for recognition."""
        try:
            with self._get_db_manager() as db:
                return db.face_features.get_all_active_features()
        except Exception as e:
            logger.error(f"Error getting all active features: {e}")
            return []
    
    def deactivate_feature(self, feature_id: int) -> bool:
        """Deactivate a face feature."""
        try:
            with self._get_db_manager() as db:
                feature = db.face_features.update(feature_id, is_active=False)
                if feature:
                    logger.info(f"Deactivated face feature {feature_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deactivating feature {feature_id}: {e}")
            return False


class AccessLogService:
    """Service for access log operations."""
    
    def __init__(self):
        pass
    
    def _get_db_manager(self) -> DatabaseManager:
        """Get database manager with session."""
        from .connection import get_database_connection
        db_connection = get_database_connection()
        return DatabaseManager(db_connection.session_factory)
    
    def log_access_attempt(self, location_id: int, access_granted: bool,
                          person_id: int = None, recognition_confidence: float = None,
                          similarity_score: float = None, matched_feature_id: int = None,
                          failure_reason: str = None, image_path: str = None,
                          processing_time_ms: int = None, device_info: Dict = None,
                          **kwargs) -> Optional[AccessLog]:
        """Log an access attempt."""
        try:
            with self._get_db_manager() as db:
                log_data = {
                    'location_id': location_id,
                    'person_id': person_id,
                    'access_granted': access_granted,
                    'recognition_confidence': recognition_confidence,
                    'similarity_score': similarity_score,
                    'matched_feature_id': matched_feature_id,
                    'failure_reason': failure_reason,
                    'image_path': image_path,
                    'processing_time_ms': processing_time_ms,
                    'device_info': device_info,
                    **kwargs
                }
                
                log_entry = db.access_logs.create(**log_data)
                if log_entry:
                    status = "GRANTED" if access_granted else "DENIED"
                    logger.info(f"Access {status} - Location: {location_id}, Person: {person_id}")
                    return log_entry
                
                return None
        except Exception as e:
            logger.error(f"Error logging access attempt: {e}")
            return None
    
    def get_recent_logs(self, days: int = 7, limit: int = 100) -> List[AccessLog]:
        """Get recent access logs."""
        try:
            with self._get_db_manager() as db:
                return db.access_logs.get_recent_logs(days, limit)
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    def get_person_access_history(self, person_id: int, limit: int = 50) -> List[AccessLog]:
        """Get access history for a person."""
        try:
            with self._get_db_manager() as db:
                return db.access_logs.get_logs_by_person(person_id, limit)
        except Exception as e:
            logger.error(f"Error getting access history for person {person_id}: {e}")
            return []
    
    def get_failed_attempts(self, hours: int = 24) -> List[AccessLog]:
        """Get recent failed access attempts."""
        try:
            with self._get_db_manager() as db:
                return db.access_logs.get_failed_attempts(hours)
        except Exception as e:
            logger.error(f"Error getting failed attempts: {e}")
            return []
    
    def get_access_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get access statistics for the specified period."""
        try:
            with self._get_db_manager() as db:
                since = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Get total attempts
                total_query = db.session.query(AccessLog).filter(
                    AccessLog.timestamp >= since
                )
                total_attempts = total_query.count()
                
                # Get successful attempts
                successful_attempts = total_query.filter(
                    AccessLog.access_granted == True
                ).count()
                
                # Get failed attempts
                failed_attempts = total_attempts - successful_attempts
                
                # Calculate success rate
                success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
                
                # Get average processing time
                from sqlalchemy import func
                avg_processing_time = db.session.query(
                    func.avg(AccessLog.processing_time_ms)
                ).filter(
                    AccessLog.timestamp >= since,
                    AccessLog.processing_time_ms.isnot(None)
                ).scalar() or 0
                
                return {
                    'period_days': days,
                    'total_attempts': total_attempts,
                    'successful_attempts': successful_attempts,
                    'failed_attempts': failed_attempts,
                    'success_rate': round(success_rate, 2),
                    'average_processing_time_ms': round(avg_processing_time, 2)
                }
        except Exception as e:
            logger.error(f"Error getting access statistics: {e}")
            return {}


class DatabaseService:
    """Main database service combining all sub-services."""
    
    def __init__(self):
        self.persons = PersonService()
        self.face_features = FaceFeatureService()
        self.access_logs = AccessLogService()
    
    def initialize_default_data(self):
        """Initialize database with default data."""
        try:
            # Create default admin user if not exists
            admin = self.persons.get_person_by_employee_id('ADMIN001')
            if not admin:
                admin = self.persons.create_person(
                    name='System Administrator',
                    employee_id='ADMIN001',
                    email='admin@company.com',
                    department='IT',
                    position='Administrator',
                    access_level=10
                )
                logger.info("Created default admin user")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing default data: {e}")
            return False


# Global service instance
_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """Get global database service instance."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service