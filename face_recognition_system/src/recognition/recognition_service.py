"""Complete real-time face recognition service."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from camera.video_stream import VideoStream, StreamConfig, CameraConfig
from .realtime_recognizer import RealtimeRecognizer, RecognitionConfig
from .result_processor import RecognitionResultProcessor, ProcessingConfig, Alert
from services.access_manager import AccessManager

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for recognition service."""
    
    # Camera settings
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Recognition settings
    recognition_config: RecognitionConfig = None
    processing_config: ProcessingConfig = None
    
    # Service settings
    location_id: int = 1
    enable_access_control: bool = True
    enable_continuous_monitoring: bool = True
    
    # Logging
    log_all_detections: bool = True
    save_unknown_faces: bool = True
    unknown_faces_path: str = "unknown_faces"


class FaceRecognitionService:
    """
    Complete face recognition service.
    
    Integrates camera capture, face recognition, result processing,
    and access control into a single service.
    """
    
    def __init__(self, config: ServiceConfig = None):
        """
        Initialize face recognition service.
        
        Args:
            config: Service configuration
        """
        self.config = config or ServiceConfig()
        
        # Initialize configurations
        if self.config.recognition_config is None:
            self.config.recognition_config = RecognitionConfig()
        
        if self.config.processing_config is None:
            self.config.processing_config = ProcessingConfig()
        
        # Initialize components
        self.video_stream = None
        self.recognizer = None
        self.result_processor = None
        self.access_manager = AccessManager()
        
        # Service state
        self.is_running = False
        self.is_initialized = False
        
        # Statistics
        self.service_stats = {
            'start_time': None,
            'frames_processed': 0,
            'access_attempts': 0,
            'access_granted': 0,
            'access_denied': 0,
            'alerts_generated': 0,
            'errors': 0
        }
        
        # Callbacks
        self.access_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.detection_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all service components."""
        try:
            # Initialize camera stream
            camera_config = CameraConfig(
                camera_id=self.config.camera_id,
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps
            )
            
            stream_config = StreamConfig(
                camera_config=camera_config,
                process_every_n_frames=2,  # Process every 2nd frame for performance
                enable_recording=False,
                show_preview=False
            )
            
            self.video_stream = VideoStream(stream_config)
            
            # Initialize recognizer
            self.recognizer = RealtimeRecognizer(self.config.recognition_config)
            
            # Initialize result processor
            self.result_processor = RecognitionResultProcessor(self.config.processing_config)
            
            # Set up video stream processor
            self.video_stream.set_frame_processor(self._process_frame)
            
            # Set up callbacks
            self.video_stream.add_result_callback(self._on_recognition_result)
            self.result_processor.add_alert_callback(self._on_alert)
            
            self.is_initialized = True
            logger.info("Face recognition service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face recognition service: {e}")
            self.is_initialized = False
    
    def start(self) -> bool:
        """
        Start the face recognition service.
        
        Returns:
            bool: True if service started successfully
        """
        try:
            if not self.is_initialized:
                logger.error("Service not initialized")
                return False
            
            if self.is_running:
                logger.warning("Service is already running")
                return True
            
            # Start video stream
            if not self.video_stream.start():
                logger.error("Failed to start video stream")
                return False
            
            self.is_running = True
            self.service_stats['start_time'] = time.time()
            
            logger.info("Face recognition service started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start face recognition service: {e}")
            return False
    
    def stop(self):
        """Stop the face recognition service."""
        try:
            self.is_running = False
            
            if self.video_stream:
                self.video_stream.stop()
            
            logger.info("Face recognition service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping face recognition service: {e}")
    
    def process_access_request(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Process a single access request.
        
        Args:
            timeout: Timeout for processing in seconds
            
        Returns:
            Dict: Access decision result
        """
        try:
            if not self.is_running:
                return {
                    'access_granted': False,
                    'reason': 'service_not_running',
                    'message': 'Face recognition service is not running'
                }
            
            # Get current frame
            frame = self.video_stream.get_current_frame()
            if frame is None:
                return {
                    'access_granted': False,
                    'reason': 'no_camera_frame',
                    'message': 'No camera frame available'
                }
            
            # Process for access control
            result = self.recognizer.recognize_for_access(frame, self.config.location_id)
            
            # Update statistics
            with self._lock:
                self.service_stats['access_attempts'] += 1
                if result.get('access_granted', False):
                    self.service_stats['access_granted'] += 1
                else:
                    self.service_stats['access_denied'] += 1
            
            # Call access callbacks
            for callback in self.access_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in access callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing access request: {e}")
            
            with self._lock:
                self.service_stats['errors'] += 1
            
            return {
                'access_granted': False,
                'reason': 'processing_error',
                'message': f'Processing error: {str(e)}'
            }
    
    def get_live_preview(self) -> Optional[bytes]:
        """
        Get current camera frame as JPEG bytes for preview.
        
        Returns:
            bytes: JPEG encoded frame or None
        """
        try:
            frame = self.video_stream.get_current_frame()
            if frame is None:
                return None
            
            # Encode as JPEG
            import cv2
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error getting live preview: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.
        
        Returns:
            Dict: Service status information
        """
        try:
            status = {
                'is_running': self.is_running,
                'is_initialized': self.is_initialized,
                'timestamp': time.time()
            }
            
            # Add component status
            if self.video_stream:
                status['camera'] = self.video_stream.camera.get_camera_info()
                status['video_stream'] = self.video_stream.get_statistics()
            
            if self.recognizer:
                status['recognizer'] = self.recognizer.get_statistics()
            
            if self.result_processor:
                status['result_processor'] = self.result_processor.get_statistics()
            
            # Add service statistics
            with self._lock:
                status['service_stats'] = self.service_stats.copy()
            
            # Calculate runtime statistics
            if status['service_stats']['start_time']:
                runtime = time.time() - status['service_stats']['start_time']
                status['service_stats']['runtime_seconds'] = runtime
                
                if runtime > 0:
                    status['service_stats']['access_requests_per_minute'] = (
                        status['service_stats']['access_attempts'] * 60 / runtime
                    )
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                'is_running': self.is_running,
                'is_initialized': self.is_initialized,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent recognition activity.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Recent activity
        """
        try:
            if self.result_processor:
                return self.result_processor.get_recent_results(limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []
    
    def get_alerts(self, limit: int = 20) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts
            
        Returns:
            List[Alert]: Recent alerts
        """
        try:
            if self.result_processor:
                return self.result_processor.get_alerts(limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def capture_snapshot(self, filename: str = None) -> bool:
        """
        Capture current frame as snapshot.
        
        Args:
            filename: Output filename
            
        Returns:
            bool: True if snapshot saved successfully
        """
        try:
            if self.video_stream:
                return self.video_stream.capture_snapshot(filename)
            else:
                return False
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return False
    
    def reload_recognition_data(self):
        """Reload person features and recognition data."""
        try:
            if self.recognizer:
                self.recognizer.reload_person_features()
                logger.info("Recognition data reloaded")
        except Exception as e:
            logger.error(f"Error reloading recognition data: {e}")
    
    def add_access_callback(self, callback: Callable[[Dict], None]):
        """Add callback for access decisions."""
        self.access_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def add_detection_callback(self, callback: Callable[[Dict], None]):
        """Add callback for face detections."""
        self.detection_callbacks.append(callback)
    
    def _process_frame(self, frame) -> Dict[str, Any]:
        """Process frame for recognition (called by video stream)."""
        try:
            # Process frame with recognizer
            result = self.recognizer.process_frame(frame)
            
            # Update statistics
            with self._lock:
                self.service_stats['frames_processed'] += 1
            
            # Return result for video stream
            return {
                'success': result.success,
                'face_count': result.face_count,
                'recognized_count': len(result.recognized_persons),
                'unknown_count': len(result.unknown_faces),
                'processing_time': result.processing_time,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            
            with self._lock:
                self.service_stats['errors'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def _on_recognition_result(self, result_data: Dict, summary: Dict):
        """Handle recognition result from video stream."""
        try:
            result = result_data.get('result')
            if not result:
                return
            
            # Process result
            processing_summary = self.result_processor.process_result(
                result, 
                self.config.location_id
            )
            
            # Update alert statistics
            with self._lock:
                self.service_stats['alerts_generated'] += processing_summary.get('alerts_generated', 0)
            
            # Call detection callbacks
            for callback in self.detection_callbacks:
                try:
                    callback({
                        'result': result,
                        'processing_summary': processing_summary,
                        'location_id': self.config.location_id
                    })
                except Exception as e:
                    logger.error(f"Error in detection callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling recognition result: {e}")
    
    def _on_alert(self, alert: Alert):
        """Handle alert from result processor."""
        try:
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_default_service(camera_id: int = 0, location_id: int = 1) -> FaceRecognitionService:
    """
    Create a face recognition service with default configuration.
    
    Args:
        camera_id: Camera ID to use
        location_id: Location ID for access control
        
    Returns:
        FaceRecognitionService: Configured service instance
    """
    config = ServiceConfig(
        camera_id=camera_id,
        location_id=location_id
    )
    
    return FaceRecognitionService(config)