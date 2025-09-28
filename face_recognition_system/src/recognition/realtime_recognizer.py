"""Real-time face recognition implementation."""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from pathlib import Path

from models.face_detector import FaceDetector
from features.feature_extractor import BatchFeatureExtractor
from features.vector_index import VectorIndex
from database.services import get_database_service
from services.access_manager import AccessManager
from utils.exceptions import FaceDetectionError, FeatureExtractionError

logger = logging.getLogger(__name__)


@dataclass
class RecognitionConfig:
    """Configuration for real-time face recognition."""
    
    # Detection settings
    min_face_size: int = 40
    max_face_size: int = 400
    detection_confidence: float = 0.7
    
    # Recognition settings
    recognition_threshold: float = 0.75
    max_recognition_distance: float = 1.0
    feature_dimension: int = 512
    
    # Performance settings
    max_faces_per_frame: int = 5
    processing_timeout: float = 5.0
    enable_gpu: bool = True
    
    # Quality settings
    min_quality_score: float = 0.5
    blur_threshold: float = 100.0
    brightness_range: Tuple[float, float] = (30, 220)
    
    # Tracking settings
    enable_tracking: bool = True
    tracking_timeout: float = 3.0
    min_tracking_frames: int = 3
    
    # Caching settings
    enable_result_cache: bool = True
    cache_timeout: float = 10.0
    max_cache_size: int = 1000


@dataclass
class RecognitionResult:
    """Result of face recognition."""
    
    # Detection info
    face_detected: bool = False
    face_count: int = 0
    faces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recognition info
    recognized_persons: List[Dict[str, Any]] = field(default_factory=list)
    unknown_faces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing info
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    frame_id: Optional[int] = None
    
    # Quality info
    overall_quality: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    
    # Error info
    success: bool = True
    error_message: Optional[str] = None


class RealtimeRecognizer:
    """
    Real-time face recognition system.
    
    Integrates face detection, feature extraction, and person identification
    for real-time video processing.
    """
    
    def __init__(self, config: RecognitionConfig = None):
        """
        Initialize real-time recognizer.
        
        Args:
            config: Recognition configuration
        """
        self.config = config or RecognitionConfig()
        
        # Initialize components
        self.face_detector = None
        self.feature_extractor = None
        self.vector_index = None
        self.db_service = get_database_service()
        self.access_manager = AccessManager()
        
        # Recognition state
        self.is_initialized = False
        self.person_features = {}  # Cache of person features
        self.last_update_time = 0
        
        # Tracking state
        self.tracked_faces = {}
        self.next_track_id = 1
        
        # Result cache
        self.result_cache = {}
        self.cache_timestamps = {}
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'persons_recognized': 0,
            'unknown_faces': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0,
            'start_time': time.time()
        }
        
        # Threading
        self._lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize face detection and recognition components."""
        try:
            # Initialize face detector
            self.face_detector = FaceDetector()
            logger.info("Face detector initialized")
            
            # Initialize feature extractor (mock for now)
            # self.feature_extractor = BatchFeatureExtractor("models/face_model.pth")
            logger.info("Feature extractor initialized (mock)")
            
            # Initialize vector index
            self.vector_index = VectorIndex(
                dimension=self.config.feature_dimension,
                metric='cosine'
            )
            logger.info("Vector index initialized")
            
            # Load person features
            self._load_person_features()
            
            self.is_initialized = True
            logger.info("Real-time recognizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize recognizer components: {e}")
            self.is_initialized = False
    
    def process_frame(self, frame: np.ndarray, frame_id: int = None) -> RecognitionResult:
        """
        Process a single frame for face recognition.
        
        Args:
            frame: Input frame
            frame_id: Optional frame identifier
            
        Returns:
            RecognitionResult: Recognition results
        """
        start_time = time.time()
        result = RecognitionResult(frame_id=frame_id)
        
        try:
            if not self.is_initialized:
                result.success = False
                result.error_message = "Recognizer not initialized"
                return result
            
            # Update person features if needed
            self._update_person_features_if_needed()
            
            # Detect faces
            faces = self._detect_faces(frame)
            result.face_detected = len(faces) > 0
            result.face_count = len(faces)
            
            if not faces:
                result.processing_time = time.time() - start_time
                self._update_stats(result)
                return result
            
            # Process each detected face
            for face_info in faces:
                try:
                    # Extract features
                    features = self._extract_features(frame, face_info)
                    if features is None:
                        continue
                    
                    # Recognize person
                    recognition_info = self._recognize_person(features, face_info)
                    
                    # Add face info
                    face_data = {
                        'bbox': face_info.get('bbox', {}),
                        'confidence': face_info.get('confidence', 0.0),
                        'quality_score': face_info.get('quality_score', 0.0),
                        'landmarks': face_info.get('landmarks', {}),
                        **recognition_info
                    }
                    
                    result.faces.append(face_data)
                    
                    # Categorize result
                    if recognition_info.get('person_id'):
                        result.recognized_persons.append(face_data)
                    else:
                        result.unknown_faces.append(face_data)
                
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
            
            # Calculate overall quality
            if result.faces:
                quality_scores = [f.get('quality_score', 0) for f in result.faces]
                result.overall_quality = sum(quality_scores) / len(quality_scores)
            
            # Update tracking if enabled
            if self.config.enable_tracking:
                self._update_tracking(result)
            
            result.processing_time = time.time() - start_time
            self._update_stats(result)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            
            with self._lock:
                self.stats['processing_errors'] += 1
        
        return result
    
    def recognize_for_access(self, frame: np.ndarray, location_id: int) -> Dict[str, Any]:
        """
        Process frame for access control.
        
        Args:
            frame: Input frame
            location_id: Access location ID
            
        Returns:
            Dict: Access control result
        """
        try:
            # Process frame
            recognition_result = self.process_frame(frame)
            
            if not recognition_result.success:
                return {
                    'access_granted': False,
                    'reason': 'recognition_error',
                    'message': recognition_result.error_message,
                    'processing_time': recognition_result.processing_time
                }
            
            if not recognition_result.face_detected:
                return {
                    'access_granted': False,
                    'reason': 'no_face_detected',
                    'message': 'No face detected in frame',
                    'processing_time': recognition_result.processing_time
                }
            
            if len(recognition_result.faces) > 1:
                return {
                    'access_granted': False,
                    'reason': 'multiple_faces',
                    'message': f'Multiple faces detected ({len(recognition_result.faces)})',
                    'processing_time': recognition_result.processing_time
                }
            
            # Check if person is recognized
            if not recognition_result.recognized_persons:
                # Log unknown access attempt
                self.access_manager.log_access_attempt(
                    location_id=location_id,
                    access_granted=False,
                    failure_reason='person_not_recognized',
                    processing_time_ms=int(recognition_result.processing_time * 1000)
                )
                
                return {
                    'access_granted': False,
                    'reason': 'person_not_recognized',
                    'message': 'Person not recognized',
                    'processing_time': recognition_result.processing_time
                }
            
            # Get recognized person
            recognized_person = recognition_result.recognized_persons[0]
            person_id = recognized_person['person_id']
            confidence = recognized_person.get('similarity_score', 0.0)
            
            # Check access permissions
            access_result = self.access_manager.check_access(person_id, location_id)
            
            # Log access attempt
            self.access_manager.log_access_attempt(
                person_id=person_id,
                location_id=location_id,
                access_granted=access_result['access_granted'],
                recognition_confidence=confidence,
                similarity_score=confidence,
                failure_reason=access_result.get('reason') if not access_result['access_granted'] else None,
                processing_time_ms=int(recognition_result.processing_time * 1000)
            )
            
            return {
                'access_granted': access_result['access_granted'],
                'reason': access_result['reason'],
                'message': access_result['message'],
                'person_id': person_id,
                'person_name': recognized_person.get('person_name', 'Unknown'),
                'confidence': confidence,
                'processing_time': recognition_result.processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in access recognition: {e}")
            return {
                'access_granted': False,
                'reason': 'system_error',
                'message': f'System error: {str(e)}',
                'processing_time': 0.0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recognition statistics.
        
        Returns:
            Dict: Statistics
        """
        with self._lock:
            stats = self.stats.copy()
        
        # Calculate runtime statistics
        runtime = time.time() - stats['start_time']
        stats['runtime_seconds'] = runtime
        
        if runtime > 0:
            stats['fps'] = stats['frames_processed'] / runtime
            stats['faces_per_second'] = stats['faces_detected'] / runtime
        
        # Add component status
        stats['is_initialized'] = self.is_initialized
        stats['person_count'] = len(self.person_features)
        stats['tracked_faces'] = len(self.tracked_faces)
        
        return stats
    
    def reload_person_features(self):
        """Reload person features from database."""
        try:
            self._load_person_features()
            logger.info("Person features reloaded")
        except Exception as e:
            logger.error(f"Error reloading person features: {e}")
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in frame."""
        try:
            if not self.face_detector:
                # Mock face detection for testing
                height, width = frame.shape[:2]
                return [{
                    'bbox': {
                        'x1': width * 0.2,
                        'y1': height * 0.2,
                        'x2': width * 0.8,
                        'y2': height * 0.8
                    },
                    'confidence': 0.95,
                    'quality_score': 0.85,
                    'landmarks': {}
                }]
            
            faces = self.face_detector.detect_faces(frame)
            
            # Filter faces by size and confidence
            filtered_faces = []
            for face in faces:
                bbox = face.get('bbox', {})
                if not bbox:
                    continue
                
                width = bbox.get('x2', 0) - bbox.get('x1', 0)
                height = bbox.get('y2', 0) - bbox.get('y1', 0)
                
                if (self.config.min_face_size <= min(width, height) <= self.config.max_face_size and
                    face.get('confidence', 0) >= self.config.detection_confidence):
                    filtered_faces.append(face)
            
            # Limit number of faces
            return filtered_faces[:self.config.max_faces_per_frame]
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _extract_features(self, frame: np.ndarray, face_info: Dict) -> Optional[np.ndarray]:
        """Extract features from detected face."""
        try:
            # Mock feature extraction for testing
            return np.random.rand(self.config.feature_dimension).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def _recognize_person(self, features: np.ndarray, face_info: Dict) -> Dict[str, Any]:
        """Recognize person from features."""
        try:
            if not self.person_features:
                return {
                    'person_id': None,
                    'person_name': None,
                    'similarity_score': 0.0,
                    'is_recognized': False
                }
            
            # Find best match
            best_match = None
            best_score = 0.0
            
            for person_id, person_data in self.person_features.items():
                person_features = person_data['features']
                
                # Calculate similarity
                similarity = self._calculate_similarity(features, person_features)
                
                if similarity > best_score and similarity >= self.config.recognition_threshold:
                    best_score = similarity
                    best_match = person_data
            
            if best_match:
                return {
                    'person_id': best_match['person_id'],
                    'person_name': best_match['name'],
                    'similarity_score': best_score,
                    'is_recognized': True
                }
            else:
                return {
                    'person_id': None,
                    'person_name': None,
                    'similarity_score': best_score,
                    'is_recognized': False
                }
                
        except Exception as e:
            logger.error(f"Person recognition error: {e}")
            return {
                'person_id': None,
                'person_name': None,
                'similarity_score': 0.0,
                'is_recognized': False
            }
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def _load_person_features(self):
        """Load person features from database."""
        try:
            # Get all active persons with features
            persons_with_features = self.db_service.persons.get_persons_with_features()
            
            self.person_features = {}
            
            for person in persons_with_features:
                # Get primary feature
                primary_feature = self.db_service.face_features.get_primary_feature(person.id)
                
                if primary_feature:
                    feature_vector = primary_feature.get_feature_vector()
                    if feature_vector is not None:
                        self.person_features[person.id] = {
                            'person_id': person.id,
                            'name': person.name,
                            'employee_id': person.employee_id,
                            'features': feature_vector,
                            'feature_id': primary_feature.id
                        }
            
            self.last_update_time = time.time()
            logger.info(f"Loaded features for {len(self.person_features)} persons")
            
        except Exception as e:
            logger.error(f"Error loading person features: {e}")
            self.person_features = {}
    
    def _update_person_features_if_needed(self):
        """Update person features if cache is stale."""
        try:
            # Check if update is needed (every 5 minutes)
            if time.time() - self.last_update_time > 300:
                self._load_person_features()
        except Exception as e:
            logger.error(f"Error updating person features: {e}")
    
    def _update_tracking(self, result: RecognitionResult):
        """Update face tracking information."""
        try:
            # Simple tracking implementation
            current_time = time.time()
            
            # Remove expired tracks
            expired_tracks = []
            for track_id, track_info in self.tracked_faces.items():
                if current_time - track_info['last_seen'] > self.config.tracking_timeout:
                    expired_tracks.append(track_id)
            
            for track_id in expired_tracks:
                del self.tracked_faces[track_id]
            
            # Update or create tracks for current faces
            for face in result.faces:
                # Simple matching based on position (in real implementation, use more sophisticated tracking)
                bbox = face.get('bbox', {})
                if not bbox:
                    continue
                
                center_x = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2
                center_y = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2
                
                # Find closest existing track
                closest_track = None
                min_distance = float('inf')
                
                for track_id, track_info in self.tracked_faces.items():
                    track_x = track_info['center_x']
                    track_y = track_info['center_y']
                    
                    distance = np.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
                    
                    if distance < min_distance and distance < 100:  # Threshold for matching
                        min_distance = distance
                        closest_track = track_id
                
                if closest_track:
                    # Update existing track
                    self.tracked_faces[closest_track].update({
                        'center_x': center_x,
                        'center_y': center_y,
                        'last_seen': current_time,
                        'frame_count': self.tracked_faces[closest_track]['frame_count'] + 1
                    })
                    face['track_id'] = closest_track
                else:
                    # Create new track
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    self.tracked_faces[track_id] = {
                        'center_x': center_x,
                        'center_y': center_y,
                        'last_seen': current_time,
                        'frame_count': 1,
                        'created_at': current_time
                    }
                    face['track_id'] = track_id
            
        except Exception as e:
            logger.error(f"Error updating tracking: {e}")
    
    def _update_stats(self, result: RecognitionResult):
        """Update recognition statistics."""
        try:
            with self._lock:
                self.stats['frames_processed'] += 1
                self.stats['faces_detected'] += result.face_count
                self.stats['persons_recognized'] += len(result.recognized_persons)
                self.stats['unknown_faces'] += len(result.unknown_faces)
                
                # Update average processing time
                current_avg = self.stats['average_processing_time']
                frame_count = self.stats['frames_processed']
                new_avg = ((current_avg * (frame_count - 1)) + result.processing_time) / frame_count
                self.stats['average_processing_time'] = new_avg
                
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")


def create_default_recognizer() -> RealtimeRecognizer:
    """
    Create a real-time recognizer with default configuration.
    
    Returns:
        RealtimeRecognizer: Configured recognizer instance
    """
    config = RecognitionConfig()
    return RealtimeRecognizer(config)