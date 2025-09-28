#!/usr/bin/env python3
"""
Test script for Face Recognition API.
Tests basic API functionality and endpoints.
"""

import sys
import os
import asyncio
import json
import base64
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.app import create_app
from database.connection import init_database, DatabaseConfig
from database.services import get_database_service

def setup_test_environment():
    """Set up test environment."""
    print("Setting up test environment...")
    
    # Initialize database
    try:
        db_config = DatabaseConfig()
        init_database(db_config)
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False
    
    return True

def test_basic_endpoints():
    """Test basic API endpoints."""
    print("\nTesting basic endpoints...")
    
    # Create test client
    app = create_app()
    client = TestClient(app)
    
    # Test root endpoint
    try:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("✓ Root endpoint working")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False
    
    # Test health endpoint
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("✓ Health endpoint working")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        return False
    
    # Test OpenAPI docs
    try:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        print("✓ OpenAPI documentation available")
    except Exception as e:
        print(f"✗ OpenAPI documentation failed: {e}")
        return False
    
    return True

def test_person_endpoints():
    """Test person management endpoints."""
    print("\nTesting person endpoints...")
    
    app = create_app()
    client = TestClient(app)
    
    # Test list persons (should work without authentication for now)
    try:
        response = client.get("/api/v1/persons/")
        # May return 401 if authentication is required, which is expected
        if response.status_code in [200, 401]:
            print("✓ List persons endpoint accessible")
        else:
            print(f"✗ List persons endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ List persons endpoint failed: {e}")
        return False
    
    return True

def test_system_endpoints():
    """Test system endpoints."""
    print("\nTesting system endpoints...")
    
    app = create_app()
    client = TestClient(app)
    
    # Test system health
    try:
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("✓ System health endpoint working")
    except Exception as e:
        print(f"✗ System health endpoint failed: {e}")
        return False
    
    return True

def test_recognition_endpoints():
    """Test recognition endpoints."""
    print("\nTesting recognition endpoints...")
    
    app = create_app()
    client = TestClient(app)
    
    # Test recognition status
    try:
        response = client.get("/api/v1/recognition/status")
        # May return 401 if authentication is required
        if response.status_code in [200, 401, 503]:  # 503 if service not available
            print("✓ Recognition status endpoint accessible")
        else:
            print(f"✗ Recognition status endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Recognition status endpoint failed: {e}")
        return False
    
    return True

def test_api_documentation():
    """Test API documentation endpoints."""
    print("\nTesting API documentation...")
    
    app = create_app()
    client = TestClient(app)
    
    # Test Swagger UI
    try:
        response = client.get("/docs")
        assert response.status_code == 200
        print("✓ Swagger UI available")
    except Exception as e:
        print(f"✗ Swagger UI failed: {e}")
        return False
    
    # Test ReDoc
    try:
        response = client.get("/redoc")
        assert response.status_code == 200
        print("✓ ReDoc available")
    except Exception as e:
        print(f"✗ ReDoc failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("Face Recognition API Test Suite")
    print("=" * 40)
    
    # Setup test environment
    if not setup_test_environment():
        print("\n✗ Test environment setup failed")
        return False
    
    # Run tests
    tests = [
        test_basic_endpoints,
        test_system_endpoints,
        test_person_endpoints,
        test_recognition_endpoints,
        test_api_documentation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 40)
    print(f"Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)