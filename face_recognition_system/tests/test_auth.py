"""Tests for authentication system."""

import pytest
from fastapi.testclient import TestClient
from fastapi import status

class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_login_success(self, client: TestClient):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_invalid_credentials(self, client: TestClient):
        """Test login with invalid credentials."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_missing_fields(self, client: TestClient):
        """Test login with missing fields."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_current_user(self, client: TestClient, auth_headers_admin):
        """Test getting current user info."""
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "admin"
        assert "permissions" in data
        assert "admin" in data["permissions"]
    
    def test_get_current_user_unauthorized(self, client: TestClient):
        """Test getting current user without token."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_validate_token(self, client: TestClient, auth_headers_admin):
        """Test token validation."""
        response = client.get(
            "/api/v1/auth/validate-token",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["valid"] is True
        assert "user" in data
    
    def test_get_permissions(self, client: TestClient, auth_headers_admin):
        """Test getting available permissions."""
        response = client.get(
            "/api/v1/auth/permissions",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "permissions" in data
        assert "user_permissions" in data
        assert "read" in data["permissions"]
        assert "write" in data["permissions"]
        assert "admin" in data["permissions"]
    
    def test_logout(self, client: TestClient, auth_headers_admin):
        """Test user logout."""
        response = client.post(
            "/api/v1/auth/logout",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert "Logged out successfully" in data["message"]

class TestUserManagement:
    """Test user management endpoints."""
    
    def test_list_users_admin(self, client: TestClient, auth_headers_admin):
        """Test listing users as admin."""
        response = client.get(
            "/api/v1/users/",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2  # admin and user
    
    def test_list_users_forbidden(self, client: TestClient, auth_headers_user):
        """Test listing users as regular user (should be forbidden)."""
        response = client.get(
            "/api/v1/users/",
            headers=auth_headers_user
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_get_user_admin(self, client: TestClient, auth_headers_admin):
        """Test getting user by ID as admin."""
        response = client.get(
            "/api/v1/users/1",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == 1
        assert data["username"] == "admin"
    
    def test_get_user_not_found(self, client: TestClient, auth_headers_admin):
        """Test getting non-existent user."""
        response = client.get(
            "/api/v1/users/999",
            headers=auth_headers_admin
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_create_user_admin(self, client: TestClient, auth_headers_admin):
        """Test creating user as admin."""
        response = client.post(
            "/api/v1/users/",
            headers=auth_headers_admin,
            params={
                "username": "testuser",
                "password": "testpass123",
                "email": "test@example.com",
                "full_name": "Test User",
                "permissions": ["read"]
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
    
    def test_create_user_forbidden(self, client: TestClient, auth_headers_user):
        """Test creating user as regular user (should be forbidden)."""
        response = client.post(
            "/api/v1/users/",
            headers=auth_headers_user,
            params={
                "username": "testuser2",
                "password": "testpass123"
            }
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN

class TestRateLimit:
    """Test rate limiting functionality."""
    
    def test_rate_limit_anonymous(self, client: TestClient):
        """Test rate limiting for anonymous users."""
        # Make multiple requests quickly
        responses = []
        for i in range(5):
            response = client.get("/api/v1/system/health")
            responses.append(response)
        
        # All requests should succeed (health endpoint is not rate limited heavily)
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
    
    def test_rate_limit_headers(self, client: TestClient):
        """Test rate limit headers are present."""
        response = client.get("/api/v1/system/health")
        
        assert response.status_code == status.HTTP_200_OK
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

class TestSecurity:
    """Test security features."""
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are present."""
        response = client.options("/api/v1/system/health")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    def test_security_headers(self, client: TestClient):
        """Test security headers are present."""
        response = client.get("/api/v1/system/health")
        
        assert response.status_code == status.HTTP_200_OK
        # Check for security headers added by middleware
        assert "X-Process-Time" in response.headers