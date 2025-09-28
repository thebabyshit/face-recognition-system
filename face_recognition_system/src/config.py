"""Configuration management for the face recognition system."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql://localhost:5432/face_recognition_db"
    database_test_url: str = "postgresql://localhost:5432/face_recognition_test_db"
    
    # Model paths
    model_path: Path = Path("models/trained/")
    checkpoint_path: Path = Path("models/checkpoints/")
    feature_dimension: int = 512
    recognition_threshold: float = 0.85
    
    # Camera settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Security
    jwt_secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    encryption_key: str = "your-encryption-key-here"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/app.log")
    
    # Hardware
    door_controller_port: str = "/dev/ttyUSB0"
    display_port: str = "/dev/ttyUSB1"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    num_workers: int = 4
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # File upload
    max_file_size: int = 10485760  # 10MB
    allowed_extensions: str = "jpg,jpeg,png,bmp"
    upload_path: Path = Path("data/uploads/")
    
    @validator("model_path", "checkpoint_path", "log_file", "upload_path")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        if isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("allowed_extensions")
    def parse_extensions(cls, v):
        """Parse allowed extensions into a set."""
        return set(ext.strip().lower() for ext in v.split(","))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings