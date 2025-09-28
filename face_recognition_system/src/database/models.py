"""SQLAlchemy ORM models for face recognition system."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import uuid

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey,
    LargeBinary, JSON, CheckConstraint, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, INET, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import numpy as np

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


class UUIDMixin:
    """Mixin for UUID primary key."""
    
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)


class Person(Base, TimestampMixin, UUIDMixin):
    """Person model - stores basic person information."""
    
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(String(50), unique=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    department = Column(String(100), index=True)
    position = Column(String(100))
    access_level = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    notes = Column(Text)
    created_by = Column(Integer, ForeignKey('persons.id'))
    updated_by = Column(Integer, ForeignKey('persons.id'))
    
    # Relationships
    face_features = relationship('FaceFeature', back_populates='person', cascade='all, delete-orphan')
    access_permissions = relationship('AccessPermission', back_populates='person', cascade='all, delete-orphan', foreign_keys='AccessPermission.person_id')
    access_logs = relationship('AccessLog', back_populates='person')
    user_sessions = relationship('UserSession', back_populates='user', cascade='all, delete-orphan')
    system_logs = relationship('SystemLog', back_populates='user')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('access_level >= 0 AND access_level <= 10', name='check_access_level'),
        Index('idx_persons_department_active', 'department', 'is_active'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        if email and '@' not in email:
            raise ValueError('Invalid email format')
        return email
    
    @validates('access_level')
    def validate_access_level(self, key, access_level):
        """Validate access level range."""
        if access_level < 0 or access_level > 10:
            raise ValueError('Access level must be between 0 and 10')
        return access_level
    
    def get_primary_feature(self) -> Optional['FaceFeature']:
        """Get the primary face feature for this person."""
        return next((f for f in self.face_features if f.is_primary and f.is_active), None)
    
    def get_active_features(self) -> List['FaceFeature']:
        """Get all active face features for this person."""
        return [f for f in self.face_features if f.is_active]
    
    def to_dict(self, include_features: bool = False) -> Dict:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'uuid': str(self.uuid),
            'employee_id': self.employee_id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'department': self.department,
            'position': self.position,
            'access_level': self.access_level,
            'is_active': self.is_active,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if include_features:
            data['face_features'] = [f.to_dict() for f in self.get_active_features()]
        
        return data
    
    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}', employee_id='{self.employee_id}')>"


class FaceFeature(Base, TimestampMixin, UUIDMixin):
    """Face feature model - stores face feature vectors and metadata."""
    
    __tablename__ = 'face_features'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id', ondelete='CASCADE'), nullable=False, index=True)
    feature_vector = Column(LargeBinary, nullable=False)  # Serialized numpy array
    feature_dimension = Column(Integer, nullable=False, default=512)
    extraction_model = Column(String(100), nullable=False)
    extraction_version = Column(String(20), nullable=False)
    image_path = Column(String(500))
    image_hash = Column(String(64))  # SHA-256 hash
    quality_score = Column(Float, index=True)
    confidence_score = Column(Float)
    face_bbox_x1 = Column(Float)
    face_bbox_y1 = Column(Float)
    face_bbox_x2 = Column(Float)
    face_bbox_y2 = Column(Float)
    landmarks = Column(Text)  # JSON string for SQLite compatibility
    feature_metadata = Column(Text)  # JSON string for SQLite compatibility
    is_primary = Column(Boolean, default=False, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Relationships
    person = relationship('Person', back_populates='face_features')
    access_logs = relationship('AccessLog', back_populates='matched_feature')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='check_quality_score'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_score'),
        Index('idx_face_features_person_primary', 'person_id', 'is_primary'),
        Index('idx_face_features_model_version', 'extraction_model', 'extraction_version'),
    )
    
    def set_feature_vector(self, vector: np.ndarray):
        """Set feature vector from numpy array."""
        self.feature_vector = vector.tobytes()
        self.feature_dimension = vector.shape[0]
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector as numpy array."""
        if self.feature_vector is None:
            return None
        return np.frombuffer(self.feature_vector, dtype=np.float32).reshape(self.feature_dimension)
    
    def get_landmarks(self) -> Optional[Dict]:
        """Get landmarks as dictionary."""
        if self.landmarks:
            try:
                return json.loads(self.landmarks)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_landmarks(self, landmarks: Dict):
        """Set landmarks from dictionary."""
        if landmarks:
            self.landmarks = json.dumps(landmarks)
        else:
            self.landmarks = None
    
    def get_metadata(self) -> Optional[Dict]:
        """Get metadata as dictionary."""
        if self.feature_metadata:
            try:
                return json.loads(self.feature_metadata)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_metadata(self, metadata: Dict):
        """Set metadata from dictionary."""
        if metadata:
            self.feature_metadata = json.dumps(metadata)
        else:
            self.feature_metadata = None
    
    def get_bounding_box(self) -> Optional[Dict[str, float]]:
        """Get face bounding box coordinates."""
        if all(coord is not None for coord in [self.face_bbox_x1, self.face_bbox_y1, 
                                               self.face_bbox_x2, self.face_bbox_y2]):
            return {
                'x1': self.face_bbox_x1,
                'y1': self.face_bbox_y1,
                'x2': self.face_bbox_x2,
                'y2': self.face_bbox_y2,
                'width': self.face_bbox_x2 - self.face_bbox_x1,
                'height': self.face_bbox_y2 - self.face_bbox_y1
            }
        return None
    
    def to_dict(self, include_vector: bool = False) -> Dict:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'uuid': str(self.uuid),
            'person_id': self.person_id,
            'feature_dimension': self.feature_dimension,
            'extraction_model': self.extraction_model,
            'extraction_version': self.extraction_version,
            'image_path': self.image_path,
            'image_hash': self.image_hash,
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score,
            'bounding_box': self.get_bounding_box(),
            'landmarks': self.get_landmarks(),
            'metadata': self.get_metadata(),
            'is_primary': self.is_primary,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if include_vector:
            vector = self.get_feature_vector()
            data['feature_vector'] = vector.tolist() if vector is not None else None
        
        return data
    
    def __repr__(self):
        return f"<FaceFeature(id={self.id}, person_id={self.person_id}, is_primary={self.is_primary})>"


class AccessLocation(Base, TimestampMixin, UUIDMixin):
    """Access location model - defines access control points."""
    
    __tablename__ = 'access_locations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    location_type = Column(String(50), default='door')
    building = Column(String(100))
    floor = Column(String(20))
    room = Column(String(50))
    required_access_level = Column(Integer, default=1)
    is_active = Column(Boolean, default=True, nullable=False)
    hardware_config = Column(Text)  # JSON string for SQLite compatibility
    
    # Relationships
    access_permissions = relationship('AccessPermission', back_populates='location', cascade='all, delete-orphan')
    access_logs = relationship('AccessLog', back_populates='location')
    
    def get_hardware_config(self) -> Optional[Dict]:
        """Get hardware config as dictionary."""
        if self.hardware_config:
            try:
                return json.loads(self.hardware_config)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_hardware_config(self, config: Dict):
        """Set hardware config from dictionary."""
        if config:
            self.hardware_config = json.dumps(config)
        else:
            self.hardware_config = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'name': self.name,
            'description': self.description,
            'location_type': self.location_type,
            'building': self.building,
            'floor': self.floor,
            'room': self.room,
            'required_access_level': self.required_access_level,
            'is_active': self.is_active,
            'hardware_config': self.get_hardware_config(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<AccessLocation(id={self.id}, name='{self.name}', type='{self.location_type}')>"


class AccessPermission(Base, TimestampMixin):
    """Access permission model - defines who can access what."""
    
    __tablename__ = 'access_permissions'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id', ondelete='CASCADE'), nullable=False, index=True)
    location_id = Column(Integer, ForeignKey('access_locations.id', ondelete='CASCADE'), nullable=False, index=True)
    permission_type = Column(String(20), default='allow')
    valid_from = Column(DateTime(timezone=True), default=func.now())
    valid_until = Column(DateTime(timezone=True))
    time_restrictions = Column(Text)  # JSON string for SQLite compatibility
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    granted_by = Column(Integer, ForeignKey('persons.id'))
    
    # Relationships
    person = relationship('Person', back_populates='access_permissions', foreign_keys=[person_id])
    location = relationship('AccessLocation', back_populates='access_permissions')
    granted_by_person = relationship('Person', foreign_keys=[granted_by])
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('person_id', 'location_id', name='uq_person_location'),
        Index('idx_access_permissions_validity', 'valid_from', 'valid_until'),
    )
    
    def get_time_restrictions(self) -> Optional[Dict]:
        """Get time restrictions as dictionary."""
        if self.time_restrictions:
            try:
                return json.loads(self.time_restrictions)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_time_restrictions(self, restrictions: Dict):
        """Set time restrictions from dictionary."""
        if restrictions:
            self.time_restrictions = json.dumps(restrictions)
        else:
            self.time_restrictions = None
    
    def is_valid_at(self, timestamp: datetime = None) -> bool:
        """Check if permission is valid at given timestamp."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if not self.is_active:
            return False
        
        if self.valid_from > timestamp:
            return False
        
        if self.valid_until and self.valid_until < timestamp:
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'person_id': self.person_id,
            'location_id': self.location_id,
            'permission_type': self.permission_type,
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'time_restrictions': self.get_time_restrictions(),
            'is_active': self.is_active,
            'granted_by': self.granted_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<AccessPermission(person_id={self.person_id}, location_id={self.location_id}, type='{self.permission_type}')>"


class AccessLog(Base, UUIDMixin):
    """Access log model - records all access attempts."""
    
    __tablename__ = 'access_logs'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), index=True)  # NULL if not recognized
    location_id = Column(Integer, ForeignKey('access_locations.id', ondelete='CASCADE'), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    access_granted = Column(Boolean, nullable=False, index=True)
    recognition_confidence = Column(Float)
    similarity_score = Column(Float)
    matched_feature_id = Column(Integer, ForeignKey('face_features.id'))
    failure_reason = Column(String(100))
    image_path = Column(String(500))
    processing_time_ms = Column(Integer)
    device_info = Column(Text)  # JSON string for SQLite compatibility
    log_metadata = Column(Text)  # JSON string for SQLite compatibility
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    person = relationship('Person', back_populates='access_logs')
    location = relationship('AccessLocation', back_populates='access_logs')
    matched_feature = relationship('FaceFeature', back_populates='access_logs')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('recognition_confidence >= 0 AND recognition_confidence <= 1', name='check_recognition_confidence'),
        CheckConstraint('similarity_score >= 0 AND similarity_score <= 1', name='check_similarity_score'),
    )
    
    def get_device_info(self) -> Optional[Dict]:
        """Get device info as dictionary."""
        if self.device_info:
            try:
                return json.loads(self.device_info)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_device_info(self, info: Dict):
        """Set device info from dictionary."""
        if info:
            self.device_info = json.dumps(info)
        else:
            self.device_info = None
    
    def get_metadata(self) -> Optional[Dict]:
        """Get metadata as dictionary."""
        if self.log_metadata:
            try:
                return json.loads(self.log_metadata)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_metadata(self, metadata: Dict):
        """Set metadata from dictionary."""
        if metadata:
            self.log_metadata = json.dumps(metadata)
        else:
            self.log_metadata = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'person_id': self.person_id,
            'location_id': self.location_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'access_granted': self.access_granted,
            'recognition_confidence': self.recognition_confidence,
            'similarity_score': self.similarity_score,
            'matched_feature_id': self.matched_feature_id,
            'failure_reason': self.failure_reason,
            'image_path': self.image_path,
            'processing_time_ms': self.processing_time_ms,
            'device_info': self.get_device_info(),
            'metadata': self.get_metadata(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<AccessLog(id={self.id}, person_id={self.person_id}, granted={self.access_granted})>"


class SystemLog(Base, UUIDMixin):
    """System log model - general system events and errors."""
    
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    component = Column(String(50), index=True)
    function_name = Column(String(100))
    line_number = Column(Integer)
    user_id = Column(Integer, ForeignKey('persons.id'), index=True)
    session_id = Column(String(100))
    request_id = Column(String(100))
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    stack_trace = Column(Text)
    system_metadata = Column(Text)  # JSON string for SQLite compatibility
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship('Person', back_populates='system_logs')
    
    def get_metadata(self) -> Optional[Dict]:
        """Get metadata as dictionary."""
        if self.system_metadata:
            try:
                return json.loads(self.system_metadata)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_metadata(self, metadata: Dict):
        """Set metadata from dictionary."""
        if metadata:
            self.system_metadata = json.dumps(metadata)
        else:
            self.system_metadata = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'level': self.level,
            'message': self.message,
            'component': self.component,
            'function_name': self.function_name,
            'line_number': self.line_number,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'stack_trace': self.stack_trace,
            'metadata': self.get_metadata(),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', component='{self.component}')>"


class FeatureIndex(Base, TimestampMixin, UUIDMixin):
    """Feature index model - for managing vector indices."""
    
    __tablename__ = 'feature_indices'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    index_type = Column(String(50), nullable=False)
    dimension = Column(Integer, nullable=False)
    metric = Column(String(20), nullable=False)
    model_version = Column(String(50))
    file_path = Column(String(500))
    total_vectors = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)
    build_config = Column(Text)  # JSON string for SQLite compatibility
    performance_stats = Column(Text)  # JSON string for SQLite compatibility
    
    def get_build_config(self) -> Optional[Dict]:
        """Get build config as dictionary."""
        if self.build_config:
            try:
                return json.loads(self.build_config)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_build_config(self, config: Dict):
        """Set build config from dictionary."""
        if config:
            self.build_config = json.dumps(config)
        else:
            self.build_config = None
    
    def get_performance_stats(self) -> Optional[Dict]:
        """Get performance stats as dictionary."""
        if self.performance_stats:
            try:
                return json.loads(self.performance_stats)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_performance_stats(self, stats: Dict):
        """Set performance stats from dictionary."""
        if stats:
            self.performance_stats = json.dumps(stats)
        else:
            self.performance_stats = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'name': self.name,
            'description': self.description,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'metric': self.metric,
            'model_version': self.model_version,
            'file_path': self.file_path,
            'total_vectors': self.total_vectors,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'is_active': self.is_active,
            'build_config': self.get_build_config(),
            'performance_stats': self.get_performance_stats(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<FeatureIndex(id={self.id}, name='{self.name}', type='{self.index_type}')>"


class UserSession(Base, TimestampMixin, UUIDMixin):
    """User session model - for web interface authentication."""
    
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('persons.id', ondelete='CASCADE'), nullable=False, index=True)
    session_token = Column(String(255), nullable=False, unique=True, index=True)
    refresh_token = Column(String(255), unique=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_activity = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    user = relationship('Person', back_populates='user_sessions')
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'user_id': self.user_id,
            'session_token': self.session_token,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class APIKey(Base, TimestampMixin, UUIDMixin):
    """API key model - for API access management."""
    
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    user_id = Column(Integer, ForeignKey('persons.id'))
    permissions = Column(Text)  # JSON string for SQLite compatibility
    rate_limit = Column(Integer, default=1000)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    created_by = Column(Integer, ForeignKey('persons.id'))
    
    # Relationships
    user = relationship('Person', foreign_keys=[user_id])
    creator = relationship('Person', foreign_keys=[created_by])
    
    def get_permissions(self) -> Optional[Dict]:
        """Get permissions as dictionary."""
        if self.permissions:
            try:
                return json.loads(self.permissions)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_permissions(self, permissions: Dict):
        """Set permissions from dictionary."""
        if permissions:
            self.permissions = json.dumps(permissions)
        else:
            self.permissions = None
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'name': self.name,
            'user_id': self.user_id,
            'permissions': self.get_permissions(),
            'rate_limit': self.rate_limit,
            'is_active': self.is_active,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', active={self.is_active})>"


class AuditTrail(Base, UUIDMixin):
    """Audit trail model - for tracking important changes."""
    
    __tablename__ = 'audit_trail'
    
    id = Column(Integer, primary_key=True)
    table_name = Column(String(50), nullable=False, index=True)
    record_id = Column(Integer, nullable=False, index=True)
    action = Column(String(20), nullable=False)
    old_values = Column(Text)  # JSON string for SQLite compatibility
    new_values = Column(Text)  # JSON string for SQLite compatibility
    changed_fields = Column(Text)  # JSON array as string for SQLite compatibility
    user_id = Column(Integer, ForeignKey('persons.id'), index=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship('Person')
    
    # Constraints
    __table_args__ = (
        Index('idx_audit_trail_table_record', 'table_name', 'record_id'),
    )
    
    def get_old_values(self) -> Optional[Dict]:
        """Get old values as dictionary."""
        if self.old_values:
            try:
                return json.loads(self.old_values)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_old_values(self, values: Dict):
        """Set old values from dictionary."""
        if values:
            self.old_values = json.dumps(values)
        else:
            self.old_values = None
    
    def get_new_values(self) -> Optional[Dict]:
        """Get new values as dictionary."""
        if self.new_values:
            try:
                return json.loads(self.new_values)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_new_values(self, values: Dict):
        """Set new values from dictionary."""
        if values:
            self.new_values = json.dumps(values)
        else:
            self.new_values = None
    
    def get_changed_fields(self) -> Optional[List[str]]:
        """Get changed fields as list."""
        if self.changed_fields:
            try:
                return json.loads(self.changed_fields)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_changed_fields(self, fields: List[str]):
        """Set changed fields from list."""
        if fields:
            self.changed_fields = json.dumps(fields)
        else:
            self.changed_fields = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'table_name': self.table_name,
            'record_id': self.record_id,
            'action': self.action,
            'old_values': self.get_old_values(),
            'new_values': self.get_new_values(),
            'changed_fields': self.get_changed_fields(),
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }
    
    def __repr__(self):
        return f"<AuditTrail(id={self.id}, table='{self.table_name}', action='{self.action}')>"