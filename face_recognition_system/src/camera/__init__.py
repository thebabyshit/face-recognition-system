"""Camera interface package for face recognition system."""

from .camera_interface import CameraInterface, CameraConfig
from .video_stream import VideoStream, StreamConfig

__all__ = [
    'CameraInterface',
    'CameraConfig',
    'VideoStream',
    'StreamConfig'
]