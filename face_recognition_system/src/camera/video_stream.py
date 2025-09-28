"""Video stream processing for real-time face recognition."""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
from pathlib import Path
import json
from datetime import datetime, timezone

from .camera_interface import CameraInterface, CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Video stream configuration."""
    # Camera settings
    camera_config: CameraConfig = None
    
    # Processing settings
    process_every_n_frames: int = 3  # Process every N frames for performance
    max_processing_queue: int = 5
    enable_recording: bool = False
    recording_path: str = "recordings"
    
    # Display settings
    show_preview: bool = False
    preview_width: int = 640
    preview_height: int = 480
    
    # Performance settings
    max_fps: float = 30.0
    processing_timeout: float = 5.0


class VideoStream:
    """
    Video stream processor for real-time face recognition.
    
    Handles video capture, frame processing, and result management
    with multi-threading for optimal performance.
    """
    
    def __init__(self, config: StreamConfig = None):
        """
        Initialize video stream processor.
        
        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        if self.config.camera_config is None:
            self.config.camera_config = CameraConfig()
        
        # Initialize camera
        self.camera = CameraInterface(self.config.camera_config)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.frame_processor = None
        
        # Frame queues
        self.processing_queue = Queue(maxsize=self.config.max_processing_queue)
        self.result_queue = Queue()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'processing_errors': 0,
            'average_processing_time': 0,
            'start_time': None
        }
        
        # Recording
        self.video_writer = None
        self.recording_active = False
        
        # Callbacks
        self.result_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
        
    def set_frame_processor(self, processor: Callable[[np.ndarray], Dict[str, Any]]):
        """
        Set frame processing function.
        
        Args:
            processor: Function that takes a frame and returns processing results
        """
        self.frame_processor = processor
    
    def start(self) -> bool:
        """
        Start video stream processing.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_processing:
                logger.warning("Video stream is already running")
                return True
            
            # Start camera
            if not self.camera.start():
                logger.error("Failed to start camera")
                return False
            
            # Initialize recording if enabled
            if self.config.enable_recording:
                self._initialize_recording()
            
            # Start processing
            self.is_processing = True
            self.stats['start_time'] = time.time()
            
            # Add camera frame callback
            self.camera.add_frame_callback(self._on_new_frame)
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("Video stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            return False
    
    def stop(self):
        """Stop video stream processing."""
        try:
            self.is_processing = False
            
            # Stop camera
            self.camera.stop()
            
            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Stop recording
            if self.recording_active:
                self._stop_recording()
            
            logger.info("Video stream stopped")
            
        except Exception as e:
            logger.error(f"Error stopping video stream: {e}")
    
    def get_latest_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get latest processing result.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Dict: Latest processing result or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get current camera frame.
        
        Returns:
            np.ndarray: Current frame or None
        """
        return self.camera.get_current_frame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get stream statistics.
        
        Returns:
            Dict: Stream statistics
        """
        with self._lock:
            stats = self.stats.copy()
        
        # Add camera info
        stats['camera_info'] = self.camera.get_camera_info()
        
        # Calculate runtime statistics
        if stats['start_time']:
            runtime = time.time() - stats['start_time']
            stats['runtime_seconds'] = runtime
            stats['fps_captured'] = stats['frames_captured'] / runtime if runtime > 0 else 0
            stats['fps_processed'] = stats['frames_processed'] / runtime if runtime > 0 else 0
        
        return stats
    
    def start_recording(self, filename: str = None) -> bool:
        """
        Start video recording.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            bool: True if recording started
        """
        try:
            if self.recording_active:
                logger.warning("Recording is already active")
                return True
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.mp4"
            
            recording_path = Path(self.config.recording_path)
            recording_path.mkdir(exist_ok=True)
            
            full_path = recording_path / filename
            
            # Get camera properties
            camera_info = self.camera.get_camera_info()
            width = camera_info.get('actual_width', self.config.camera_config.width)
            height = camera_info.get('actual_height', self.config.camera_config.height)
            fps = self.config.max_fps
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(full_path), fourcc, fps, (width, height))
            
            if not self.video_writer.isOpened():
                logger.error("Failed to initialize video writer")
                return False
            
            self.recording_active = True
            logger.info(f"Recording started: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop video recording."""
        self._stop_recording()
    
    def add_result_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add callback for processing results.
        
        Args:
            callback: Function to call with processing results
        """
        self.result_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """
        Add callback for errors.
        
        Args:
            callback: Function to call with error messages
        """
        self.error_callbacks.append(callback)
    
    def capture_snapshot(self, filename: str = None) -> bool:
        """
        Capture current frame as snapshot.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            bool: True if snapshot saved
        """
        try:
            frame = self.get_current_frame()
            if frame is None:
                logger.error("No frame available for snapshot")
                return False
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
            
            return self.camera.save_frame(filename, frame)
            
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return False
    
    def _on_new_frame(self, frame: np.ndarray):
        """Handle new frame from camera."""
        try:
            with self._lock:
                self.stats['frames_captured'] += 1
                frame_count = self.stats['frames_captured']
            
            # Record frame if recording is active
            if self.recording_active and self.video_writer:
                self.video_writer.write(frame)
            
            # Skip frames for processing based on configuration
            if frame_count % self.config.process_every_n_frames != 0:
                with self._lock:
                    self.stats['frames_skipped'] += 1
                return
            
            # Add frame to processing queue (non-blocking)
            try:
                if self.processing_queue.full():
                    # Remove oldest frame
                    self.processing_queue.get_nowait()
                
                # Add frame with timestamp
                frame_data = {
                    'frame': frame.copy(),
                    'timestamp': time.time(),
                    'frame_id': frame_count
                }
                self.processing_queue.put_nowait(frame_data)
                
            except Exception as e:
                logger.debug(f"Error adding frame to processing queue: {e}")
        
        except Exception as e:
            logger.error(f"Error handling new frame: {e}")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        logger.info("Video processing loop started")
        
        while self.is_processing:
            try:
                # Get frame from processing queue
                try:
                    frame_data = self.processing_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                if not self.frame_processor:
                    continue
                
                # Process frame
                start_time = time.time()
                
                try:
                    result = self.frame_processor(frame_data['frame'])
                    
                    # Add metadata to result
                    result['frame_id'] = frame_data['frame_id']
                    result['capture_timestamp'] = frame_data['timestamp']
                    result['processing_time'] = time.time() - start_time
                    result['processed_at'] = time.time()
                    
                    # Update statistics
                    with self._lock:
                        self.stats['frames_processed'] += 1
                        
                        # Update average processing time
                        current_avg = self.stats['average_processing_time']
                        processed_count = self.stats['frames_processed']
                        new_avg = ((current_avg * (processed_count - 1)) + result['processing_time']) / processed_count
                        self.stats['average_processing_time'] = new_avg
                    
                    # Add result to queue
                    try:
                        if self.result_queue.full():
                            # Remove oldest result
                            self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except:
                        pass
                    
                    # Call result callbacks
                    for callback in self.result_callbacks:
                        try:
                            callback(result.copy())
                        except Exception as e:
                            logger.error(f"Error in result callback: {e}")
                
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    
                    with self._lock:
                        self.stats['processing_errors'] += 1
                    
                    # Call error callbacks
                    for callback in self.error_callbacks:
                        try:
                            callback(f"Frame processing error: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error in error callback: {e}")
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Video processing loop stopped")
    
    def _initialize_recording(self):
        """Initialize recording directory."""
        try:
            recording_path = Path(self.config.recording_path)
            recording_path.mkdir(exist_ok=True)
            logger.info(f"Recording directory initialized: {recording_path}")
        except Exception as e:
            logger.error(f"Error initializing recording directory: {e}")
    
    def _stop_recording(self):
        """Stop video recording."""
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording_active = False
            logger.info("Recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class StreamManager:
    """
    Manager for multiple video streams.
    
    Handles multiple cameras and processing pipelines.
    """
    
    def __init__(self):
        """Initialize stream manager."""
        self.streams: Dict[str, VideoStream] = {}
        self.is_running = False
    
    def add_stream(self, name: str, config: StreamConfig) -> bool:
        """
        Add a video stream.
        
        Args:
            name: Stream name
            config: Stream configuration
            
        Returns:
            bool: True if stream added successfully
        """
        try:
            if name in self.streams:
                logger.warning(f"Stream '{name}' already exists")
                return False
            
            stream = VideoStream(config)
            self.streams[name] = stream
            
            logger.info(f"Stream '{name}' added")
            return True
            
        except Exception as e:
            logger.error(f"Error adding stream '{name}': {e}")
            return False
    
    def remove_stream(self, name: str) -> bool:
        """
        Remove a video stream.
        
        Args:
            name: Stream name
            
        Returns:
            bool: True if stream removed successfully
        """
        try:
            if name not in self.streams:
                logger.warning(f"Stream '{name}' not found")
                return False
            
            # Stop stream if running
            self.streams[name].stop()
            del self.streams[name]
            
            logger.info(f"Stream '{name}' removed")
            return True
            
        except Exception as e:
            logger.error(f"Error removing stream '{name}': {e}")
            return False
    
    def start_stream(self, name: str) -> bool:
        """
        Start a specific stream.
        
        Args:
            name: Stream name
            
        Returns:
            bool: True if stream started successfully
        """
        if name not in self.streams:
            logger.error(f"Stream '{name}' not found")
            return False
        
        return self.streams[name].start()
    
    def stop_stream(self, name: str) -> bool:
        """
        Stop a specific stream.
        
        Args:
            name: Stream name
            
        Returns:
            bool: True if stream stopped successfully
        """
        if name not in self.streams:
            logger.error(f"Stream '{name}' not found")
            return False
        
        self.streams[name].stop()
        return True
    
    def start_all_streams(self) -> Dict[str, bool]:
        """
        Start all streams.
        
        Returns:
            Dict: Results for each stream
        """
        results = {}
        for name, stream in self.streams.items():
            results[name] = stream.start()
        
        self.is_running = True
        return results
    
    def stop_all_streams(self):
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()
        
        self.is_running = False
    
    def get_stream(self, name: str) -> Optional[VideoStream]:
        """
        Get stream by name.
        
        Args:
            name: Stream name
            
        Returns:
            VideoStream: Stream instance or None
        """
        return self.streams.get(name)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all streams.
        
        Returns:
            Dict: Statistics for each stream
        """
        stats = {}
        for name, stream in self.streams.items():
            stats[name] = stream.get_statistics()
        
        return stats
    
    def list_streams(self) -> List[str]:
        """
        List all stream names.
        
        Returns:
            List[str]: Stream names
        """
        return list(self.streams.keys())