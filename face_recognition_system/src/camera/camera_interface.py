"""Camera interface for video capture and processing."""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any, Callable, List
from dataclasses import dataclass
from queue import Queue, Empty
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 10
    auto_exposure: bool = True
    exposure: int = -1
    brightness: int = 50
    contrast: int = 50
    saturation: int = 50
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotation: int = 0  # 0, 90, 180, 270 degrees


class CameraInterface:
    """
    Camera interface for video capture with threading support.
    
    Provides real-time video capture with frame buffering,
    error recovery, and device management.
    """
    
    def __init__(self, config: CameraConfig = None):
        """
        Initialize camera interface.
        
        Args:
            config: Camera configuration parameters
        """
        self.config = config or CameraConfig()
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=self.config.buffer_size)
        self.current_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0
        self.fps_counter = 0
        self.actual_fps = 0
        self._lock = threading.Lock()
        
        # Callbacks
        self.frame_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            bool: True if camera started successfully
        """
        try:
            if self.is_running:
                logger.warning("Camera is already running")
                return True
            
            # Initialize camera
            if not self._initialize_camera():
                return False
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"Camera {self.config.camera_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture."""
        try:
            self.is_running = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break
            
            logger.info("Camera stopped")
            
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the latest frame from camera.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            np.ndarray: Latest frame or None if not available
        """
        try:
            if not self.is_running:
                return None
            
            # Try to get frame from queue
            try:
                frame = self.frame_queue.get(timeout=timeout)
                with self._lock:
                    self.current_frame = frame
                return frame
            except Empty:
                # Return last frame if queue is empty
                with self._lock:
                    return self.current_frame
                
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame without waiting.
        
        Returns:
            np.ndarray: Current frame or None
        """
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def is_camera_available(self) -> bool:
        """
        Check if camera is available and working.
        
        Returns:
            bool: True if camera is available
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and statistics.
        
        Returns:
            Dict: Camera information
        """
        info = {
            'camera_id': self.config.camera_id,
            'is_running': self.is_running,
            'is_available': self.is_camera_available(),
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'actual_fps': self.actual_fps,
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'target_fps': self.config.fps,
                'buffer_size': self.config.buffer_size
            }
        }
        
        if self.cap:
            try:
                info['actual_width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['actual_height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['actual_fps_cap'] = self.cap.get(cv2.CAP_PROP_FPS)
            except:
                pass
        
        return info
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Add callback for new frames.
        
        Args:
            callback: Function to call with new frames
        """
        self.frame_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """
        Add callback for errors.
        
        Args:
            callback: Function to call with error messages
        """
        self.error_callbacks.append(callback)
    
    def save_frame(self, filename: str, frame: np.ndarray = None) -> bool:
        """
        Save current or specified frame to file.
        
        Args:
            filename: Output filename
            frame: Frame to save (uses current frame if None)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if frame is None:
                frame = self.get_current_frame()
            
            if frame is None:
                logger.error("No frame available to save")
                return False
            
            success = cv2.imwrite(filename, frame)
            if success:
                logger.info(f"Frame saved to {filename}")
            else:
                logger.error(f"Failed to save frame to {filename}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    def _initialize_camera(self) -> bool:
        """Initialize camera with configuration."""
        try:
            # Release existing camera
            if self.cap:
                self.cap.release()
            
            # Open camera
            self.cap = cv2.VideoCapture(self.config.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.config.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Set additional properties
            if not self.config.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
                if self.config.exposure >= 0:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
            
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast / 100.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation / 100.0)
            
            # Set buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Cannot read frame from camera")
                return False
            
            logger.info(f"Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        logger.info("Camera capture loop started")
        fps_start_time = time.time()
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.warning("Camera not available, attempting to reconnect...")
                    if not self._initialize_camera():
                        time.sleep(1.0)
                        continue
                
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.error_count += 1
                    logger.warning(f"Failed to capture frame (error count: {self.error_count})")
                    
                    # Call error callbacks
                    for callback in self.error_callbacks:
                        try:
                            callback("Frame capture failed")
                        except Exception as e:
                            logger.error(f"Error in error callback: {e}")
                    
                    # Try to recover
                    if self.error_count > 10:
                        logger.error("Too many capture errors, reinitializing camera")
                        self._initialize_camera()
                        self.error_count = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Reset error count on successful capture
                self.error_count = 0
                
                # Apply transformations
                frame = self._process_frame(frame)
                
                # Update frame count and FPS
                self.frame_count += 1
                self.fps_counter += 1
                
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    self.actual_fps = self.fps_counter / (current_time - fps_start_time)
                    self.fps_counter = 0
                    fps_start_time = current_time
                
                # Add frame to queue (non-blocking)
                try:
                    if self.frame_queue.full():
                        # Remove oldest frame
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame.copy())
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")
                
                # Control frame rate
                self.last_frame_time = current_time
                sleep_time = max(0, (1.0 / self.config.fps) - 0.001)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.error_count += 1
                time.sleep(0.1)
        
        logger.info("Camera capture loop stopped")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply frame transformations."""
        try:
            # Apply flips
            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            if self.config.flip_vertical:
                frame = cv2.flip(frame, 0)
            
            # Apply rotation
            if self.config.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.config.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.config.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def list_available_cameras() -> List[Dict[str, Any]]:
    """
    List all available cameras.
    
    Returns:
        List[Dict]: Available camera information
    """
    cameras = []
    
    # Test camera indices 0-10
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Test frame capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append({
                        'id': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'available': True
                    })
                
                cap.release()
            
        except Exception as e:
            logger.debug(f"Camera {i} not available: {e}")
    
    return cameras


def test_camera(camera_id: int = 0, duration: float = 5.0) -> Dict[str, Any]:
    """
    Test camera functionality.
    
    Args:
        camera_id: Camera ID to test
        duration: Test duration in seconds
        
    Returns:
        Dict: Test results
    """
    results = {
        'camera_id': camera_id,
        'success': False,
        'frames_captured': 0,
        'average_fps': 0,
        'errors': []
    }
    
    try:
        config = CameraConfig(camera_id=camera_id)
        camera = CameraInterface(config)
        
        if not camera.start():
            results['errors'].append("Failed to start camera")
            return results
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            frame = camera.get_frame(timeout=1.0)
            if frame is not None:
                frame_count += 1
            else:
                results['errors'].append("Failed to get frame")
            
            time.sleep(0.1)
        
        camera.stop()
        
        elapsed_time = time.time() - start_time
        results['frames_captured'] = frame_count
        results['average_fps'] = frame_count / elapsed_time if elapsed_time > 0 else 0
        results['success'] = frame_count > 0
        
    except Exception as e:
        results['errors'].append(str(e))
    
    return results