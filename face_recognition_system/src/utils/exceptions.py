"""Custom exception classes for the face recognition system."""


class FaceRecognitionError(Exception):
    """Base exception for face recognition system."""
    pass


class ValidationError(FaceRecognitionError):
    """Raised when data validation fails."""
    pass


class PersonNotFoundError(FaceRecognitionError):
    """Raised when a person is not found."""
    pass


class DuplicatePersonError(FaceRecognitionError):
    """Raised when trying to create a duplicate person."""
    pass


class PermissionError(FaceRecognitionError):
    """Raised when user lacks required permissions."""
    pass


class FaceDetectionError(FaceRecognitionError):
    """Raised when face detection fails."""
    pass


class FeatureExtractionError(FaceRecognitionError):
    """Raised when feature extraction fails."""
    pass


class DatabaseError(FaceRecognitionError):
    """Raised when database operations fail."""
    pass


class ConfigurationError(FaceRecognitionError):
    """Raised when system configuration is invalid."""
    pass


class HardwareError(FaceRecognitionError):
    """Raised when hardware operations fail."""
    pass


class AuthenticationError(FaceRecognitionError):
    """Raised when authentication fails."""
    pass


class AccessDeniedError(FaceRecognitionError):
    """Raised when access is denied."""
    pass


class ImageProcessingError(FaceRecognitionError):
    """Raised when image processing fails."""
    pass


class ModelError(FaceRecognitionError):
    """Raised when model operations fail."""
    pass


class IndexError(FaceRecognitionError):
    """Raised when vector index operations fail."""
    pass