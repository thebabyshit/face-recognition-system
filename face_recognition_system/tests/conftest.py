"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator, Dict, Any
from fastapi.testclient import TestClient
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import create_app
from api.auth import auth_service

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def app():
    """Create FastAPI app for testing."""
    return create_app()

@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="session")
async def admin_token() -> str:
    """Get admin authentication token."""
    user = await auth_service.authenticate_user("admin", "admin123")
    if not user:
        raise ValueError("Failed to authenticate admin user")
    
    tokens = await auth_service.create_tokens(user)
    return tokens.access_token

@pytest.fixture(scope="session")
async def user_token() -> str:
    """Get regular user authentication token."""
    user = await auth_service.authenticate_user("user", "user123")
    if not user:
        raise ValueError("Failed to authenticate regular user")
    
    tokens = await auth_service.create_tokens(user)
    return tokens.access_token

@pytest.fixture
def auth_headers_admin(admin_token: str) -> Dict[str, str]:
    """Get admin authorization headers."""
    return {"Authorization": f"Bearer {admin_token}"}

@pytest.fixture
def auth_headers_user(user_token: str) -> Dict[str, str]:
    """Get user authorization headers."""
    return {"Authorization": f"Bearer {user_token}"}

@pytest.fixture
def sample_person_data() -> Dict[str, Any]:
    """Sample person data for testing."""
    return {
        "name": "Test Person",
        "employee_id": "EMP001",
        "email": "test@example.com",
        "phone": "+1234567890",
        "department": "Engineering",
        "position": "Software Engineer",
        "access_level": 3,
        "notes": "Test person for API testing"
    }

@pytest.fixture
def sample_face_data() -> Dict[str, Any]:
    """Sample face data for testing."""
    # Base64 encoded 1x1 pixel image for testing
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    return {
        "person_id": 1,
        "image_data": test_image_b64,
        "image_filename": "test_face.png",
        "set_as_primary": True,
        "quality_threshold": 0.7
    }

@pytest.fixture
def sample_recognition_data() -> Dict[str, Any]:
    """Sample recognition data for testing."""
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    return {
        "image_data": test_image_b64,
        "confidence_threshold": 0.7,
        "max_results": 5
    }

@pytest.fixture
def sample_access_attempt() -> Dict[str, Any]:
    """Sample access attempt data for testing."""
    return {
        "person_id": 1,
        "location_id": 1,
        "access_method": "face_recognition",
        "confidence_score": 0.95
    }

# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    # Add cleanup logic here if needed
    pass