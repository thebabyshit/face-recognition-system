"""Pydantic models for API request/response validation."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ResponseStatus(str, Enum):
    """API response status."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class BaseResponse(BaseModel):
    """Base API response model."""
    status: ResponseStatus
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseResponse):
    """Success response model."""
    status: ResponseStatus = ResponseStatus.SUCCESS


# Person Models
class PersonCreate(BaseModel):
    """Person creation request model."""
    name: str = Field(..., min_length=2, max_length=100)
    employee_id: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=100)
    position: Optional[str] = Field(None, max_length=100)
    access_level: int = Field(1, ge=0, le=10)
    notes: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class PersonUpdate(BaseModel):
    """Person update request model."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    employee_id: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=100)
    position: Optional[str] = Field(None, max_length=100)
    access_level: Optional[int] = Field(None, ge=0, le=10)
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class PersonResponse(BaseModel):
    """Person response model."""
    id: int
    uuid: str
    name: str
    employee_id: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    department: Optional[str]
    position: Optional[str]
    access_level: int
    is_active: bool
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    face_feature_count: Optional[int] = None
    last_access: Optional[datetime] = None


class PersonSearchRequest(BaseModel):
    """Person search request model."""
    query: Optional[str] = None
    department: Optional[str] = None
    access_level_min: Optional[int] = Field(None, ge=0, le=10)
    access_level_max: Optional[int] = Field(None, ge=0, le=10)
    is_active: Optional[bool] = None
    has_face_features: Optional[bool] = None
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    include_face_count: bool = False
    include_last_access: bool = False


class PersonListResponse(BaseModel):
    """Person list response model."""
    persons: List[PersonResponse]
    total_count: int
    returned_count: int
    offset: int
    limit: int
    has_more: bool


# Face Feature Models
class FaceFeatureUpload(BaseModel):
    """Face feature upload request model."""
    person_id: int
    image_data: str  # Base64 encoded image
    image_filename: Optional[str] = None
    set_as_primary: bool = False
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0)


class FaceFeatureResponse(BaseModel):
    """Face feature response model."""
    id: int
    uuid: str
    person_id: int
    quality_score: Optional[float]
    confidence_score: Optional[float]
    is_primary: bool
    is_active: bool
    image_path: Optional[str]
    bbox: Optional[Dict[str, float]]
    landmarks: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    feature_dimension: int


class FaceFeatureUpdate(BaseModel):
    """Face feature update request model."""
    set_as_primary: Optional[bool] = None
    is_active: Optional[bool] = None


# Access Control Models
class AccessAttemptRequest(BaseModel):
    """Access attempt request model."""
    person_id: Optional[int] = None
    location_id: int
    access_method: str = "face_recognition"
    confidence_score: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None


class AccessControlResponse(BaseModel):
    """Access control response model."""
    access_granted: bool
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    access_level_required: Optional[int] = None
    person_access_level: Optional[int] = None
    confidence_score: Optional[float] = None
    access_method: Optional[str] = None
    reason: Optional[str] = None
    timestamp: Optional[datetime] = None
    log_id: Optional[int] = None


class AccessLogResponse(BaseModel):
    """Access log response model."""
    id: int
    uuid: str
    person_id: Optional[int]
    person_name: Optional[str]
    location_id: int
    location_name: Optional[str]
    access_granted: bool
    access_method: str
    confidence_score: Optional[float]
    reason: Optional[str]
    timestamp: datetime
    additional_data: Optional[Dict[str, Any]]


class AccessLogListResponse(BaseModel):
    """Access log list response model."""
    logs: List[AccessLogResponse]
    total_count: int
    returned_count: int
    offset: int
    limit: int
    has_more: bool


# Recognition Models
class RecognitionRequest(BaseModel):
    """Recognition request model."""
    image_data: str  # Base64 encoded image
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_results: int = Field(5, ge=1, le=20)


class RecognitionResult(BaseModel):
    """Recognition result model."""
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence_score: Optional[float] = None
    similarity_score: Optional[float] = None
    face_bbox: Optional[Dict[str, float]] = None
    face_landmarks: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    matched_feature_id: Optional[int] = None


class RecognitionResponse(BaseModel):
    """Recognition response model."""
    success: bool
    results: List[RecognitionResult]
    total_faces_detected: int
    processing_time: float
    image_quality_score: Optional[float] = None


# Bulk Operations Models
class BulkPersonImport(BaseModel):
    """Bulk person import model."""
    persons: List[PersonCreate]
    skip_duplicates: bool = True
    update_existing: bool = False


class BulkImportResponse(BaseModel):
    """Bulk import response model."""
    total: int
    success: int
    failed: int
    skipped: int
    errors: List[Dict[str, Any]]
    imported_persons: List[PersonResponse]