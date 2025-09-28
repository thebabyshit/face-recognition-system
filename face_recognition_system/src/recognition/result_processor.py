"""Recognition result processing and handling."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from enum import Enum

from .realtime_recognizer import RecognitionResult
from database.services import get_database_service
from services.access_manager import AccessManager

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    source: str = "recognition"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Configuration for result processing."""
    
    # Duplicate detection
    enable_duplicate_detection: bool = True
    duplicate_timeout: float = 5.0  # seconds
    duplicate_similarity_threshold: float = 0.95
    
    # Alert settings
    enable_alerts: bool = True
    max_alerts_per_minute: int = 10
    alert_cooldown: float = 30.0  # seconds between same alerts
    
    # Statistics
    enable_statistics: bool = True
    statistics_window: float = 300.0  # 5 minutes
    
    # Caching
    enable_result_cache: bool = True
    cache_size: int = 1000
    cache_timeout: float = 60.0
    
    # Performance
    max_processing_queue: int = 100
    processing_timeout: float = 10.0


class RecognitionResultProcessor:
    """
    Processes and handles recognition results.
    
    Provides duplicate detection, alerting, statistics tracking,
    and result caching for recognition systems.
    """
    
    def __init__(self, config: ProcessingConfig = None):
        """
        Initialize result processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        
        # Services
        self.db_service = get_database_service()
        self.access_manager = AccessManager()
        
        # State tracking
        self.recent_results = deque(maxlen=self.config.cache_size)
        self.person_last_seen = {}  # person_id -> timestamp
        self.alert_history = deque(maxlen=100)
        self.alert_cooldowns = {}  # alert_type -> last_time
        
        # Statistics
        self.stats = {
            'results_processed': 0,
            'duplicates_filtered': 0,
            'alerts_generated': 0,
            'persons_recognized': 0,
            'unknown_faces': 0,
            'processing_errors': 0,
            'start_time': time.time()
        }
        
        # Time-based statistics
        self.time_stats = defaultdict(lambda: {
            'count': 0,
            'recognized': 0,
            'unknown': 0,
            'timestamp': time.time()
        })
        
        # Callbacks
        self.result_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.statistics_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
    
    def process_result(self, result: RecognitionResult, location_id: int = None) -> Dict[str, Any]:
        """
        Process a recognition result.
        
        Args:
            result: Recognition result to process
            location_id: Optional location ID for context
            
        Returns:
            Dict: Processing summary
        """
        try:
            processing_summary = {
                'processed': True,
                'duplicates_filtered': 0,
                'alerts_generated': 0,
                'actions_taken': [],
                'timestamp': time.time()
            }
            
            # Update basic statistics
            self._update_statistics(result)
            
            # Check for duplicates
            if self.config.enable_duplicate_detection:
                duplicates = self._check_duplicates(result)
                processing_summary['duplicates_filtered'] = len(duplicates)
                
                # Filter out duplicates
                if duplicates:
                    result = self._filter_duplicates(result, duplicates)
            
            # Process recognized persons
            for person_info in result.recognized_persons:
                self._process_recognized_person(person_info, location_id, processing_summary)
            
            # Process unknown faces
            for face_info in result.unknown_faces:
                self._process_unknown_face(face_info, location_id, processing_summary)
            
            # Generate alerts if needed
            if self.config.enable_alerts:
                alerts = self._generate_alerts(result, location_id)
                processing_summary['alerts_generated'] = len(alerts)
                
                for alert in alerts:
                    self._handle_alert(alert)
            
            # Cache result
            if self.config.enable_result_cache:
                self._cache_result(result, processing_summary)
            
            # Call result callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result, processing_summary)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            return processing_summary
            
        except Exception as e:
            logger.error(f"Error processing recognition result: {e}")
            
            with self._lock:
                self.stats['processing_errors'] += 1
            
            return {
                'processed': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_recent_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent recognition results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List[Dict]: Recent results
        """
        with self._lock:
            return list(self.recent_results)[-limit:]
    
    def get_person_activity(self, person_id: int, hours: int = 24) -> Dict[str, Any]:
        """
        Get activity summary for a person.
        
        Args:
            person_id: Person ID
            hours: Hours to look back
            
        Returns:
            Dict: Activity summary
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            activity = {
                'person_id': person_id,
                'total_detections': 0,
                'locations': set(),
                'first_seen': None,
                'last_seen': None,
                'detections': []
            }
            
            # Search through recent results
            with self._lock:
                for result_data in self.recent_results:
                    if result_data.get('timestamp', 0) < cutoff_time:
                        continue
                    
                    result = result_data.get('result')
                    if not result:
                        continue
                    
                    # Check if person was in this result
                    for person_info in result.recognized_persons:
                        if person_info.get('person_id') == person_id:
                            activity['total_detections'] += 1
                            
                            timestamp = result_data.get('timestamp')
                            if activity['first_seen'] is None or timestamp < activity['first_seen']:
                                activity['first_seen'] = timestamp
                            
                            if activity['last_seen'] is None or timestamp > activity['last_seen']:
                                activity['last_seen'] = timestamp
                            
                            location_id = result_data.get('location_id')
                            if location_id:
                                activity['locations'].add(location_id)
                            
                            activity['detections'].append({
                                'timestamp': timestamp,
                                'location_id': location_id,
                                'confidence': person_info.get('similarity_score', 0.0)
                            })
            
            activity['locations'] = list(activity['locations'])
            activity['unique_locations'] = len(activity['locations'])
            
            return activity
            
        except Exception as e:
            logger.error(f"Error getting person activity: {e}")
            return {'person_id': person_id, 'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict: Statistics
        """
        with self._lock:
            stats = self.stats.copy()
        
        # Calculate runtime statistics
        runtime = time.time() - stats['start_time']
        stats['runtime_seconds'] = runtime
        
        if runtime > 0:
            stats['results_per_second'] = stats['results_processed'] / runtime
            stats['recognition_rate'] = (stats['persons_recognized'] / 
                                       max(1, stats['persons_recognized'] + stats['unknown_faces']))
        
        # Add recent activity
        stats['recent_alerts'] = len([a for a in self.alert_history 
                                    if time.time() - a.timestamp < 300])  # Last 5 minutes
        
        stats['cached_results'] = len(self.recent_results)
        stats['tracked_persons'] = len(self.person_last_seen)
        
        return stats
    
    def get_alerts(self, limit: int = 50, level: AlertLevel = None) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts
            level: Filter by alert level
            
        Returns:
            List[Alert]: Recent alerts
        """
        alerts = list(self.alert_history)
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]
    
    def add_result_callback(self, callback: Callable[[RecognitionResult, Dict], None]):
        """Add callback for processed results."""
        self.result_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def add_statistics_callback(self, callback: Callable[[Dict], None]):
        """Add callback for statistics updates."""
        self.statistics_callbacks.append(callback)
    
    def _update_statistics(self, result: RecognitionResult):
        """Update processing statistics."""
        with self._lock:
            self.stats['results_processed'] += 1
            self.stats['persons_recognized'] += len(result.recognized_persons)
            self.stats['unknown_faces'] += len(result.unknown_faces)
            
            # Update time-based statistics
            current_hour = int(time.time() // 3600)
            hour_stats = self.time_stats[current_hour]
            hour_stats['count'] += 1
            hour_stats['recognized'] += len(result.recognized_persons)
            hour_stats['unknown'] += len(result.unknown_faces)
            hour_stats['timestamp'] = time.time()
            
            # Clean old time statistics (keep last 24 hours)
            cutoff_hour = current_hour - 24
            old_hours = [h for h in self.time_stats.keys() if h < cutoff_hour]
            for hour in old_hours:
                del self.time_stats[hour]
    
    def _check_duplicates(self, result: RecognitionResult) -> List[int]:
        """Check for duplicate detections."""
        duplicates = []
        current_time = time.time()
        
        try:
            for i, person_info in enumerate(result.recognized_persons):
                person_id = person_info.get('person_id')
                if not person_id:
                    continue
                
                # Check if person was seen recently
                last_seen = self.person_last_seen.get(person_id, 0)
                
                if current_time - last_seen < self.config.duplicate_timeout:
                    # Check similarity of detection
                    confidence = person_info.get('similarity_score', 0.0)
                    
                    if confidence >= self.config.duplicate_similarity_threshold:
                        duplicates.append(i)
                
                # Update last seen time
                self.person_last_seen[person_id] = current_time
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
        
        return duplicates
    
    def _filter_duplicates(self, result: RecognitionResult, duplicate_indices: List[int]) -> RecognitionResult:
        """Filter out duplicate detections."""
        try:
            # Remove duplicates from recognized persons
            filtered_persons = []
            for i, person_info in enumerate(result.recognized_persons):
                if i not in duplicate_indices:
                    filtered_persons.append(person_info)
            
            result.recognized_persons = filtered_persons
            
            # Update statistics
            with self._lock:
                self.stats['duplicates_filtered'] += len(duplicate_indices)
            
        except Exception as e:
            logger.error(f"Error filtering duplicates: {e}")
        
        return result
    
    def _process_recognized_person(self, person_info: Dict, location_id: int, summary: Dict):
        """Process a recognized person."""
        try:
            person_id = person_info.get('person_id')
            confidence = person_info.get('similarity_score', 0.0)
            
            # Log successful recognition
            logger.info(f"Person recognized: ID {person_id}, confidence {confidence:.2f}")
            
            # Add to actions taken
            summary['actions_taken'].append({
                'action': 'person_recognized',
                'person_id': person_id,
                'confidence': confidence
            })
            
        except Exception as e:
            logger.error(f"Error processing recognized person: {e}")
    
    def _process_unknown_face(self, face_info: Dict, location_id: int, summary: Dict):
        """Process an unknown face detection."""
        try:
            quality_score = face_info.get('quality_score', 0.0)
            
            # Log unknown face
            logger.info(f"Unknown face detected, quality: {quality_score:.2f}")
            
            # Add to actions taken
            summary['actions_taken'].append({
                'action': 'unknown_face_detected',
                'quality_score': quality_score
            })
            
        except Exception as e:
            logger.error(f"Error processing unknown face: {e}")
    
    def _generate_alerts(self, result: RecognitionResult, location_id: int) -> List[Alert]:
        """Generate alerts based on recognition result."""
        alerts = []
        current_time = time.time()
        
        try:
            # Alert for multiple faces
            if len(result.faces) > 1:
                alert_type = "multiple_faces"
                
                if self._should_generate_alert(alert_type, current_time):
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"Multiple faces detected ({len(result.faces)} faces)",
                        metadata={
                            'face_count': len(result.faces),
                            'location_id': location_id
                        }
                    ))
            
            # Alert for unknown faces
            if result.unknown_faces:
                alert_type = "unknown_face"
                
                if self._should_generate_alert(alert_type, current_time):
                    alerts.append(Alert(
                        level=AlertLevel.INFO,
                        message=f"Unknown face detected",
                        metadata={
                            'unknown_count': len(result.unknown_faces),
                            'location_id': location_id
                        }
                    ))
            
            # Alert for low quality detections
            low_quality_faces = [f for f in result.faces 
                               if f.get('quality_score', 1.0) < 0.5]
            
            if low_quality_faces:
                alert_type = "low_quality"
                
                if self._should_generate_alert(alert_type, current_time):
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"Low quality face detection",
                        metadata={
                            'low_quality_count': len(low_quality_faces),
                            'location_id': location_id
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
        
        return alerts
    
    def _should_generate_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if alert should be generated based on cooldown."""
        last_time = self.alert_cooldowns.get(alert_type, 0)
        
        if current_time - last_time >= self.config.alert_cooldown:
            self.alert_cooldowns[alert_type] = current_time
            return True
        
        return False
    
    def _handle_alert(self, alert: Alert):
        """Handle generated alert."""
        try:
            # Add to alert history
            self.alert_history.append(alert)
            
            # Update statistics
            with self._lock:
                self.stats['alerts_generated'] += 1
            
            # Log alert
            logger.log(
                logging.WARNING if alert.level == AlertLevel.WARNING else logging.INFO,
                f"Alert: {alert.message}"
            )
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def _cache_result(self, result: RecognitionResult, summary: Dict):
        """Cache recognition result."""
        try:
            result_data = {
                'result': result,
                'summary': summary,
                'timestamp': time.time()
            }
            
            with self._lock:
                self.recent_results.append(result_data)
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")


class ResultAggregator:
    """
    Aggregates recognition results for reporting and analysis.
    """
    
    def __init__(self, window_size: float = 3600.0):  # 1 hour default
        """
        Initialize result aggregator.
        
        Args:
            window_size: Time window for aggregation in seconds
        """
        self.window_size = window_size
        self.results = deque()
        self._lock = threading.Lock()
    
    def add_result(self, result: RecognitionResult, location_id: int = None):
        """Add result to aggregation."""
        with self._lock:
            self.results.append({
                'result': result,
                'location_id': location_id,
                'timestamp': time.time()
            })
            
            # Remove old results
            cutoff_time = time.time() - self.window_size
            while self.results and self.results[0]['timestamp'] < cutoff_time:
                self.results.popleft()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary."""
        with self._lock:
            if not self.results:
                return {
                    'total_results': 0,
                    'time_window': self.window_size,
                    'window_start': None,
                    'window_end': None
                }
            
            total_faces = 0
            total_recognized = 0
            total_unknown = 0
            locations = set()
            persons = set()
            
            window_start = self.results[0]['timestamp']
            window_end = self.results[-1]['timestamp']
            
            for result_data in self.results:
                result = result_data['result']
                location_id = result_data.get('location_id')
                
                total_faces += result.face_count
                total_recognized += len(result.recognized_persons)
                total_unknown += len(result.unknown_faces)
                
                if location_id:
                    locations.add(location_id)
                
                for person_info in result.recognized_persons:
                    person_id = person_info.get('person_id')
                    if person_id:
                        persons.add(person_id)
            
            return {
                'total_results': len(self.results),
                'total_faces': total_faces,
                'total_recognized': total_recognized,
                'total_unknown': total_unknown,
                'unique_persons': len(persons),
                'unique_locations': len(locations),
                'recognition_rate': total_recognized / max(1, total_faces),
                'time_window': self.window_size,
                'window_start': window_start,
                'window_end': window_end,
                'duration': window_end - window_start if window_end > window_start else 0
            }