#!/usr/bin/env python3
"""
Test script for authentication system.
"""

import sys
import os
import asyncio
import json
import requests
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_authentication_endpoints():
    """Test authentication endpoints."""
    print("Testing Authentication System")
    print("=" * 40)
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test data
    test_credentials = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        # Test login
        print("\n1. Testing login...")
        login_response = requests.post(
            f"{base_url}/auth/login",
            json=test_credentials,
            headers={"Content-Type": "application/json"}
        )
        
        if login_response.status_code == 200:
            print("✓ Login successful")
            tokens = login_response.json()
            access_token = tokens.get("access_token")
            print(f"  Access token received: {access_token[:20]}...")
        else:
            print(f"✗ Login failed: {login_response.status_code}")
            print(f"  Response: {login_response.text}")
            return False
        
        # Test protected endpoint
        print("\n2. Testing protected endpoint...")
        headers = {"Authorization": f"Bearer {access_token}"}
        
        me_response = requests.get(
            f"{base_url}/auth/me",
            headers=headers
        )
        
        if me_response.status_code == 200:
            print("✓ Protected endpoint access successful")
            user_info = me_response.json()
            print(f"  User: {user_info.get('username')}")
            print(f"  Permissions: {user_info.get('permissions')}")
        else:
            print(f"✗ Protected endpoint failed: {me_response.status_code}")
            return False
        
        # Test permissions endpoint
        print("\n3. Testing permissions endpoint...")
        perms_response = requests.get(
            f"{base_url}/auth/permissions",
            headers=headers
        )
        
        if perms_response.status_code == 200:
            print("✓ Permissions endpoint successful")
            perms_data = perms_response.json()
            print(f"  Available permissions: {list(perms_data.get('permissions', {}).keys())}")
        else:
            print(f"✗ Permissions endpoint failed: {perms_response.status_code}")
        
        # Test token validation
        print("\n4. Testing token validation...")
        validate_response = requests.get(
            f"{base_url}/auth/validate-token",
            headers=headers
        )
        
        if validate_response.status_code == 200:
            print("✓ Token validation successful")
            validation_data = validate_response.json()
            print(f"  Token valid: {validation_data.get('valid')}")
        else:
            print(f"✗ Token validation failed: {validate_response.status_code}")
        
        # Test logout
        print("\n5. Testing logout...")
        logout_response = requests.post(
            f"{base_url}/auth/logout",
            headers=headers
        )
        
        if logout_response.status_code == 200:
            print("✓ Logout successful")
        else:
            print(f"✗ Logout failed: {logout_response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed - make sure the API server is running")
        print("  Start the server with: python src/scripts/run_api.py")
        return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_auth_module():
    """Test authentication module directly."""
    print("\nTesting Authentication Module")
    print("=" * 40)
    
    try:
        from api.auth import AuthService, PasswordHash, JWTManager
        
        # Test password hashing
        print("\n1. Testing password hashing...")
        password_hash = PasswordHash()
        test_password = "test123"
        hashed = password_hash.hash_password(test_password)
        
        if password_hash.verify_password(test_password, hashed):
            print("✓ Password hashing and verification working")
        else:
            print("✗ Password verification failed")
            return False
        
        # Test JWT token creation
        print("\n2. Testing JWT token creation...")
        jwt_manager = JWTManager()
        test_data = {"sub": "testuser", "user_id": 1, "permissions": ["read"]}
        
        token = jwt_manager.create_access_token(test_data)
        print(f"✓ JWT token created: {token[:20]}...")
        
        # Test token verification
        payload = jwt_manager.verify_token(token)
        if payload.get("sub") == "testuser":
            print("✓ JWT token verification working")
        else:
            print("✗ JWT token verification failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication module test failed: {e}")
        return False

async def test_auth_service():
    """Test authentication service."""
    print("\nTesting Authentication Service")
    print("=" * 40)
    
    try:
        from api.auth import AuthService
        
        auth_service = AuthService()
        
        # Test user authentication
        print("\n1. Testing user authentication...")
        user = await auth_service.authenticate_user("admin", "admin123")
        
        if user:
            print("✓ User authentication successful")
            print(f"  User: {user.username}")
            print(f"  Permissions: {user.permissions}")
        else:
            print("✗ User authentication failed")
            return False
        
        # Test token creation
        print("\n2. Testing token creation...")
        tokens = await auth_service.create_tokens(user)
        
        if tokens.access_token:
            print("✓ Token creation successful")
            print(f"  Token type: {tokens.token_type}")
            print(f"  Expires in: {tokens.expires_in} seconds")
        else:
            print("✗ Token creation failed")
            return False
        
        # Test getting current user from token
        print("\n3. Testing get current user...")
        current_user = await auth_service.get_current_user(tokens.access_token)
        
        if current_user and current_user.username == user.username:
            print("✓ Get current user successful")
        else:
            print("✗ Get current user failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication service test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Face Recognition API Authentication Test Suite")
    print("=" * 50)
    
    # Test authentication module
    module_test = test_auth_module()
    
    # Test authentication service
    service_test = asyncio.run(test_auth_service())
    
    # Test authentication endpoints (requires running server)
    print("\nNote: The following test requires a running API server")
    print("Start the server with: python src/scripts/run_api.py")
    
    endpoint_test = test_authentication_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Authentication Module: {'✓ PASS' if module_test else '✗ FAIL'}")
    print(f"  Authentication Service: {'✓ PASS' if service_test else '✗ FAIL'}")
    print(f"  Authentication Endpoints: {'✓ PASS' if endpoint_test else '✗ FAIL'}")
    
    if module_test and service_test:
        print("\n✓ Core authentication system is working!")
        if not endpoint_test:
            print("  Note: Endpoint tests require a running server")
        return True
    else:
        print("\n✗ Some authentication tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)