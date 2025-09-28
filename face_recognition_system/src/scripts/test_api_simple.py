#!/usr/bin/env python3
"""
Simple test for Face Recognition API.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from api.app import create_app
        print("✓ API app import successful")
    except Exception as e:
        print(f"✗ API app import failed: {e}")
        return False
    
    try:
        from api.models import PersonCreate, PersonResponse
        print("✓ API models import successful")
    except Exception as e:
        print(f"✗ API models import failed: {e}")
        return False
    
    try:
        from api.routes import persons, faces, access, recognition, statistics, system
        print("✓ API routes import successful")
    except Exception as e:
        print(f"✗ API routes import failed: {e}")
        return False
    
    try:
        from api.middleware import LoggingMiddleware, ErrorHandlingMiddleware
        print("✓ API middleware import successful")
    except Exception as e:
        print(f"✗ API middleware import failed: {e}")
        return False
    
    try:
        from api.dependencies import get_db_service, require_read_permission
        print("✓ API dependencies import successful")
    except Exception as e:
        print(f"✗ API dependencies import failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test FastAPI app creation."""
    print("\nTesting app creation...")
    
    try:
        from api.app import create_app
        app = create_app()
        print("✓ FastAPI app created successfully")
        
        # Check if routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/api/v1/persons/",
            "/api/v1/faces/upload",
            "/api/v1/access/attempt",
            "/api/v1/recognition/identify",
            "/api/v1/statistics/dashboard",
            "/api/v1/system/health"
        ]
        
        found_routes = 0
        for expected in expected_routes:
            if any(expected in route for route in routes):
                found_routes += 1
        
        print(f"✓ Found {found_routes}/{len(expected_routes)} expected routes")
        return True
        
    except Exception as e:
        print(f"✗ App creation failed: {e}")
        return False

def main():
    """Main test function."""
    print("Face Recognition API Simple Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_app_creation
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
    
    print("\n" + "=" * 40)
    print(f"Test Summary: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)