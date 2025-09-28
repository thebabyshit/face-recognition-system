"""Authentication API routes."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from ..auth import (
    AuthService, UserCredentials, TokenResponse, UserInfo,
    auth_service, get_current_user, get_current_active_user
)
from ..models import SuccessResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    """
    User login endpoint.
    Authenticates user and returns JWT tokens.
    """
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(
            credentials.username, 
            credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        tokens = await auth_service.create_tokens(user)
        
        logger.info(f"User {credentials.username} logged in successfully")
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.
    """
    try:
        tokens = await auth_service.refresh_access_token(refresh_token)
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: UserInfo = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return current_user

@router.post("/logout", response_model=SuccessResponse)
async def logout(current_user: UserInfo = Depends(get_current_active_user)):
    """
    User logout endpoint.
    In a production system, this would invalidate the token.
    """
    try:
        # In a real implementation, you would:
        # 1. Add the token to a blacklist
        # 2. Remove from active sessions
        # 3. Log the logout event
        
        logger.info(f"User {current_user.username} logged out")
        
        return SuccessResponse(
            message="Logged out successfully",
            data={"username": current_user.username}
        )
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/register", response_model=UserInfo)
async def register_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    full_name: Optional[str] = None,
    current_user: UserInfo = Depends(get_current_active_user)
):
    """
    Register a new user.
    Only admin users can register new users.
    """
    try:
        # Check if current user has admin permissions
        if 'admin' not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can register new users"
            )
        
        # Create new user
        new_user = await auth_service.create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name,
            permissions=['read']  # Default permissions
        )
        
        logger.info(f"New user {username} registered by {current_user.username}")
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed"
        )

@router.get("/permissions")
async def get_available_permissions(current_user: UserInfo = Depends(get_current_active_user)):
    """
    Get list of available permissions.
    """
    try:
        permissions = {
            "read": "Read access to resources",
            "write": "Write access to resources",
            "admin": "Administrative access to all resources"
        }
        
        return {
            "permissions": permissions,
            "user_permissions": current_user.permissions
        }
        
    except Exception as e:
        logger.error(f"Error getting permissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get permissions"
        )

@router.get("/validate-token")
async def validate_token(current_user: UserInfo = Depends(get_current_active_user)):
    """
    Validate current token and return user info.
    """
    return {
        "valid": True,
        "user": current_user,
        "message": "Token is valid"
    }