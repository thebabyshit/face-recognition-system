"""Data access layer - Repository pattern implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import numpy as np
import logging

from .models import (
    Person, FaceFeature, AccessLocation, AccessPermission, 
    AccessLog, SystemLog, FeatureIndex, UserSession, APIKey, AuditTrail
)

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    @abstractmethod
    def get_model_class(self):
        """Return the SQLAlchemy model class."""
        pass
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """Get record by ID."""
        try:
            return self.session.query(self.get_model_class()).filter_by(id=id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.get_model_class().__name__} by ID {id}: {e}")
            return None
    
    def get_by_uuid(self, uuid: str) -> Optional[Any]:
        """Get record by UUID."""
        try:
            return self.session.query(self.get_model_class()).filter_by(uuid=uuid).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.get_model_class().__name__} by UUID {uuid}: {e}")
            return None
    
    def get_all(self, limit: int = None, offset: int = None) -> List[Any]:
        """Get all records with optional pagination."""
        try:
            query = self.session.query(self.get_model_class())
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.get_model_class().__name__}: {e}")
            return []
    
    def create(self, **kwargs) -> Optional[Any]:
        """Create new record."""
        try:
            instance = self.get_model_class()(**kwargs)
            self.session.add(instance)
            self.session.flush()  # Get ID without committing
            return instance
        except SQLAlchemyError as e:
            logger.error(f"Error creating {self.get_model_class().__name__}: {e}")
            self.session.rollback()
            return None
    
    def update(self, id: int, **kwargs) -> Optional[Any]:
        """Update record by ID."""
        try:
            instance = self.get_by_id(id)
            if instance:
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                self.session.flush()
                return instance
            return None
        except SQLAlchemyError as e:
            logger.error(f"Error updating {self.get_model_class().__name__} {id}: {e}")
            self.session.rollback()
            return None
    
    def delete(self, id: int) -> bool:
        """Delete record by ID."""
        try:
            instance = self.get_by_id(id)
            if instance:
                self.session.delete(instance)
                self.session.flush()
                return True
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error deleting {self.get_model_class().__name__} {id}: {e}")
            self.session.rollback()
            return False
    
    def count(self, **filters) -> int:
        """Count records with optional filters."""
        try:
            query = self.session.query(self.get_model_class())
            for key, value in filters.items():
                if hasattr(self.get_model_class(), key):
                    query = query.filter(getattr(self.get_model_class(), key) == value)
            return query.count()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.get_model_class().__name__}: {e}")
            return 0


class PersonRepository(BaseRepository):
    """Repository for Person model."""
    
    def get_model_class(self):
        return Person
    
    def get_by_employee_id(self, employee_id: str) -> Optional[Person]:
        """Get person by employee ID."""
        try:
            return self.session.query(Person).filter_by(employee_id=employee_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting person by employee_id {employee_id}: {e}")
            return None
    
    def get_by_email(self, email: str) -> Optional[Person]:
        """Get person by email."""
        try:
            return self.session.query(Person).filter_by(email=email).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting person by email {email}: {e}")
            return None
    
    def search_by_name(self, name: str, limit: int = 50) -> List[Person]:
        """Search persons by name (partial match)."""
        try:
            return self.session.query(Person).filter(
                Person.name.ilike(f"%{name}%")
            ).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error searching persons by name {name}: {e}")
            return []
    
    def get_by_department(self, department: str) -> List[Person]:
        """Get all persons in a department."""
        try:
            return self.session.query(Person).filter_by(
                department=department, is_active=True
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting persons by department {department}: {e}")
            return []
    
    def get_active_persons(self, limit: int = None) -> List[Person]:
        """Get all active persons."""
        try:
            query = self.session.query(Person).filter_by(is_active=True)
            if limit:
                query = query.limit(limit)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting active persons: {e}")
            return []
    
    def get_persons_with_features(self) -> List[Person]:
        """Get persons who have face features."""
        try:
            return self.session.query(Person).join(FaceFeature).filter(
                Person.is_active == True,
                FaceFeature.is_active == True
            ).distinct().all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting persons with features: {e}")
            return []


class FaceFeatureRepository(BaseRepository):
    """Repository for FaceFeature model."""
    
    def get_model_class(self):
        return FaceFeature
    
    def get_by_person_id(self, person_id: int, active_only: bool = True) -> List[FaceFeature]:
        """Get all face features for a person."""
        try:
            query = self.session.query(FaceFeature).filter_by(person_id=person_id)
            if active_only:
                query = query.filter_by(is_active=True)
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting face features for person {person_id}: {e}")
            return []
    
    def get_primary_feature(self, person_id: int) -> Optional[FaceFeature]:
        """Get primary face feature for a person."""
        try:
            return self.session.query(FaceFeature).filter_by(
                person_id=person_id, is_primary=True, is_active=True
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting primary feature for person {person_id}: {e}")
            return None
    
    def set_primary_feature(self, feature_id: int) -> bool:
        """Set a feature as primary (unset others for same person)."""
        try:
            feature = self.get_by_id(feature_id)
            if not feature:
                return False
            
            # Unset other primary features for this person
            self.session.query(FaceFeature).filter_by(
                person_id=feature.person_id, is_primary=True
            ).update({'is_primary': False})
            
            # Set this feature as primary
            feature.is_primary = True
            self.session.flush()
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error setting primary feature {feature_id}: {e}")
            self.session.rollback()
            return False
    
    def get_all_active_features(self) -> List[FaceFeature]:
        """Get all active face features."""
        try:
            return self.session.query(FaceFeature).filter_by(is_active=True).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all active features: {e}")
            return []


class AccessLogRepository(BaseRepository):
    """Repository for AccessLog model."""
    
    def get_model_class(self):
        return AccessLog
    
    def get_recent_logs(self, days: int = 7, limit: int = 100) -> List[AccessLog]:
        """Get recent access logs."""
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            return self.session.query(AccessLog).filter(
                AccessLog.timestamp >= since
            ).order_by(desc(AccessLog.timestamp)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    def get_logs_by_person(self, person_id: int, limit: int = 50) -> List[AccessLog]:
        """Get access logs for a specific person."""
        try:
            return self.session.query(AccessLog).filter_by(
                person_id=person_id
            ).order_by(desc(AccessLog.timestamp)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting logs for person {person_id}: {e}")
            return []
    
    def get_failed_attempts(self, hours: int = 24) -> List[AccessLog]:
        """Get failed access attempts in recent hours."""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            return self.session.query(AccessLog).filter(
                AccessLog.timestamp >= since,
                AccessLog.access_granted == False
            ).order_by(desc(AccessLog.timestamp)).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting failed attempts: {e}")
            return []


class DatabaseManager:
    """Main database manager class."""
    
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory
        self._session = None
    
    @property
    def session(self) -> Session:
        """Get current session."""
        if self._session is None:
            self._session = self.session_factory()
        return self._session
    
    def close_session(self):
        """Close current session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def commit(self):
        """Commit current transaction."""
        try:
            self.session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error committing transaction: {e}")
            self.session.rollback()
            raise
    
    def rollback(self):
        """Rollback current transaction."""
        self.session.rollback()
    
    # Repository properties
    @property
    def persons(self) -> PersonRepository:
        """Get person repository."""
        return PersonRepository(self.session)
    
    @property
    def face_features(self) -> FaceFeatureRepository:
        """Get face feature repository."""
        return FaceFeatureRepository(self.session)
    
    @property
    def access_logs(self) -> AccessLogRepository:
        """Get access log repository."""
        return AccessLogRepository(self.session)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close_session()