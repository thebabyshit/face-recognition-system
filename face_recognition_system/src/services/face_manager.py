"""Face management service implementation."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
import hashlib
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import io
import base64

from ..database.services import get_database_service
from ..database.models import Person, FaceFeature
from ..models.face_detector import FaceDetector
from ..features.feature_extractor import BatchFeatureExtractor
from ..utils.exceptions import (
    PersonNotFoundError,
    FaceDetectionError,
    FeatureExtractionError,
    ValidationError,
    ImageProcessingError
)
from ..utils.validators import validate_image_file

logger = logging.getLogger(__name__)


class FaceManager:
    """Face management service for handling face images and features."""
    
    def __init__(self):
        self.db_service = get_database_service()
        self.face_detector = None
        self.feature_extractor = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize face detection and feature extraction models."""
        try:
            self.face_detector = FaceDetector()
            # For now, we'll skip feature extractor initialization
            # self.feature_extractor = BatchFeatureExtractor("models/face_model.pth")
            logger.info("Face detection model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize face models: {e}")
            # Continue without models for now
    
    def add_face_image(self,
                      person_id: int,
                      image_data: Union[str, bytes, np.ndarray],
                      image_filename: str = None,
                      set_as_primary: bool = False,
                      quality_threshold: float = 0.7,
                      added_by: int = None) -> Dict[str, Any]:
        """
        Add a face image for a person and extract features.
        
        Args:
            person_id: ID of the person
            image_data: Image data (base64 string, bytes, or numpy array)
            image_filename: Original filename
            set_as_primary: Whether to set as primary face
            quality_threshold: Minimum quality threshold
            added_by: ID of user adding the face
            
        Returns:
            Dict: Result with face feature info and processing details
            
        Raises:
            PersonNotFoundError: If person doesn't exist
            FaceDetectionError: If no face detected
            FeatureExtractionError: If feature extraction fails
            ValidationError: If image is invalid
        """
        try:
            # Validate person exists
            person = self.db_service.persons.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            # Process image data
            image_array = self._process_image_data(image_data)
            
            # Validate image
            if not self._validate_image(image_array):
                raise ValidationError("Invalid image format or size")
            
            # Detect face
            face_info = self._detect_face(image_array)
            if not face_info:
                raise FaceDetectionError("No face detected in image")
            
            # Check image quality
            quality_score = face_info.get('quality_score', 0.0)
            if quality_score < quality_threshold:
                raise ValidationError(f"Image quality too low: {quality_score:.2f} < {quality_threshold}")
            
            # Extract features
            feature_vector = self._extract_features(image_array, face_info)
            if feature_vector is None:
                raise FeatureExtractionError("Failed to extract face features")
            
            # Save image file
            image_path = self._save_image_file(image_array, person_id, image_filename)
            
            # Calculate image hash
            image_hash = self._calculate_image_hash(image_array)
            
            # Create face feature record
            face_feature = self.db_service.face_features.add_face_feature(
                person_id=person_id,
                feature_vector=feature_vector,
                extraction_model="mock_model",
                extraction_version="1.0",
                image_path=str(image_path) if image_path else None,
                image_hash=image_hash,
                quality_score=quality_score,
                confidence_score=face_info.get('confidence', 0.0),
                bbox=face_info.get('bbox'),
                landmarks=face_info.get('landmarks'),
                set_as_primary=set_as_primary
            )
            
            if not face_feature:
                raise RuntimeError("Failed to save face feature to database")
            
            result = {
                'face_feature_id': face_feature.id,
                'person_id': person_id,
                'quality_score': quality_score,
                'confidence_score': face_info.get('confidence', 0.0),
                'is_primary': face_feature.is_primary,
                'image_path': str(image_path) if image_path else None,
                'bbox': face_info.get('bbox'),
                'landmarks': face_info.get('landmarks'),
                'feature_dimension': len(feature_vector) if feature_vector is not None else 0
            }
            
            logger.info(f"Successfully added face for person {person_id}: feature ID {face_feature.id}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding face image for person {person_id}: {e}")
            raise
    
    def update_face_feature(self,
                           feature_id: int,
                           set_as_primary: bool = None,
                           is_active: bool = None,
                           updated_by: int = None) -> Optional[FaceFeature]:
        """
        Update face feature properties.
        
        Args:
            feature_id: ID of face feature to update
            set_as_primary: Whether to set as primary
            is_active: Whether feature is active
            updated_by: ID of user making update
            
        Returns:
            FaceFeature: Updated face feature
        """
        try:
            # Get existing feature
            with self.db_service.face_features._get_db_manager() as db:
                feature = db.face_features.get_by_id(feature_id)
                if not feature:
                    raise ValidationError(f"Face feature {feature_id} not found")
                
                updates = {}
                if set_as_primary is not None:
                    if set_as_primary:
                        # Set as primary (will unset others)
                        success = db.face_features.set_primary_feature(feature_id)
                        if not success:
                            raise RuntimeError("Failed to set primary feature")
                    else:
                        updates['is_primary'] = False
                
                if is_active is not None:
                    updates['is_active'] = is_active
                
                if updates:
                    updated_feature = db.face_features.update(feature_id, **updates)
                    logger.info(f"Updated face feature {feature_id}")
                    return updated_feature
                
                return feature
                
        except Exception as e:
            logger.error(f"Error updating face feature {feature_id}: {e}")
            raise
    
    def delete_face_feature(self, feature_id: int, deleted_by: int = None) -> bool:
        """
        Delete a face feature.
        
        Args:
            feature_id: ID of face feature to delete
            deleted_by: ID of user performing deletion
            
        Returns:
            bool: True if deletion successful
        """
        try:
            success = self.db_service.face_features.deactivate_feature(feature_id)
            if success:
                logger.info(f"Deleted face feature {feature_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error deleting face feature {feature_id}: {e}")
            return False
    
    def get_person_faces(self, person_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all face features for a person.
        
        Args:
            person_id: ID of the person
            active_only: Whether to return only active features
            
        Returns:
            List[Dict]: List of face feature information
        """
        try:
            features = self.db_service.face_features.get_person_features(person_id, active_only)
            
            result = []
            for feature in features:
                feature_info = {
                    'id': feature.id,
                    'person_id': feature.person_id,
                    'quality_score': feature.quality_score,
                    'confidence_score': feature.confidence_score,
                    'is_primary': feature.is_primary,
                    'is_active': feature.is_active,
                    'image_path': feature.image_path,
                    'bbox': feature.get_bounding_box(),
                    'landmarks': feature.get_landmarks(),
                    'created_at': feature.created_at.isoformat() if feature.created_at else None,
                    'feature_dimension': feature.feature_dimension
                }
                result.append(feature_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting faces for person {person_id}: {e}")
            return []
    
    def get_primary_face(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get primary face feature for a person.
        
        Args:
            person_id: ID of the person
            
        Returns:
            Dict: Primary face feature information or None
        """
        try:
            feature = self.db_service.face_features.get_primary_feature(person_id)
            if not feature:
                return None
            
            return {
                'id': feature.id,
                'person_id': feature.person_id,
                'quality_score': feature.quality_score,
                'confidence_score': feature.confidence_score,
                'is_primary': feature.is_primary,
                'image_path': feature.image_path,
                'bbox': feature.get_bounding_box(),
                'landmarks': feature.get_landmarks(),
                'feature_vector': feature.get_feature_vector(),
                'created_at': feature.created_at.isoformat() if feature.created_at else None
            }
            
        except Exception as e:
            logger.error(f"Error getting primary face for person {person_id}: {e}")
            return None
    
    def compare_faces(self, feature_id1: int, feature_id2: int) -> Dict[str, Any]:
        """
        Compare two face features.
        
        Args:
            feature_id1: ID of first face feature
            feature_id2: ID of second face feature
            
        Returns:
            Dict: Comparison results with similarity score
        """
        try:
            with self.db_service.face_features._get_db_manager() as db:
                feature1 = db.face_features.get_by_id(feature_id1)
                feature2 = db.face_features.get_by_id(feature_id2)
                
                if not feature1 or not feature2:
                    raise ValidationError("One or both face features not found")
                
                vector1 = feature1.get_feature_vector()
                vector2 = feature2.get_feature_vector()
                
                if vector1 is None or vector2 is None:
                    raise FeatureExtractionError("Feature vectors not available")
                
                # Calculate similarity
                similarity = self._calculate_similarity(vector1, vector2)
                
                return {
                    'feature_id1': feature_id1,
                    'feature_id2': feature_id2,
                    'similarity_score': float(similarity),
                    'is_match': similarity > 0.8,  # Configurable threshold
                    'person_id1': feature1.person_id,
                    'person_id2': feature2.person_id,
                    'same_person': feature1.person_id == feature2.person_id
                }
                
        except Exception as e:
            logger.error(f"Error comparing faces {feature_id1} and {feature_id2}: {e}")
            raise
    
    def _process_image_data(self, image_data: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Process various image data formats into numpy array."""
        try:
            if isinstance(image_data, str):
                # Assume base64 encoded image
                if image_data.startswith('data:image'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                return np.array(image)
                
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
                
            elif isinstance(image_data, np.ndarray):
                return image_data
                
            else:
                raise ValidationError("Unsupported image data format")
                
        except Exception as e:
            raise ImageProcessingError(f"Failed to process image data: {e}")
    
    def _validate_image(self, image_array: np.ndarray) -> bool:
        """Validate image array."""
        if image_array is None or image_array.size == 0:
            return False
        
        # Check dimensions
        if len(image_array.shape) not in [2, 3]:
            return False
        
        # Check size limits
        height, width = image_array.shape[:2]
        if width < 50 or height < 50 or width > 4000 or height > 4000:
            return False
        
        return True
    
    def _detect_face(self, image_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect face in image and return face information."""
        if not self.face_detector:
            # Mock face detection for testing
            height, width = image_array.shape[:2]
            return {
                'bbox': {'x1': width*0.2, 'y1': height*0.2, 'x2': width*0.8, 'y2': height*0.8},
                'confidence': 0.95,
                'quality_score': 0.85,
                'landmarks': {}
            }
        
        try:
            faces = self.face_detector.detect_faces(image_array)
            if not faces:
                return None
            
            # Return the best face (highest confidence)
            best_face = max(faces, key=lambda f: f.get('confidence', 0))
            return best_face
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None
    
    def _extract_features(self, image_array: np.ndarray, face_info: Dict) -> Optional[np.ndarray]:
        """Extract face features from image."""
        # Mock feature extraction for testing
        # In production, this would use the actual feature extraction model
        return np.random.rand(512).astype(np.float32)
    
    def _save_image_file(self, image_array: np.ndarray, person_id: int, filename: str = None) -> Optional[Path]:
        """Save image file to disk."""
        try:
            # Create directory structure
            images_dir = Path("data/face_images")
            person_dir = images_dir / f"person_{person_id}"
            person_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"face_{timestamp}.jpg"
            
            image_path = person_dir / filename
            
            # Save image
            if len(image_array.shape) == 3:
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), image_bgr)
            else:
                cv2.imwrite(str(image_path), image_array)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to save image file: {e}")
            return None
    
    def _calculate_image_hash(self, image_array: np.ndarray) -> str:
        """Calculate SHA-256 hash of image."""
        try:
            image_bytes = cv2.imencode('.jpg', image_array)[1].tobytes()
            return hashlib.sha256(image_bytes).hexdigest()
        except Exception:
            return ""
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vector1, vector2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0