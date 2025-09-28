"""User management API routes."""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from ..auth import UserInfo, auth_service, require_permissions
from ..models import SuccessResponse, ErrorResponse
from ..dependencies import get_db_service, DatabaseService

logger = logging.getLogger(__name__)
router = APIRouter()

class UserCreateRequest:
    """User creation request model."""
    def __init__(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        permissions: List[str] = None
    ):
        self.username = username
        self.password = password
        self.email = email
        self.full_name = full_name
        self.permissions = permissions or ['read']

class UserUpdateRequest:
    """User update request model."""
    def __init__(
        self,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ):
        self.email = email
        self.full_name = full_name
        self.permissions = permissions
        self.is_active = is_active

@router.get("/", response_model=List[UserInfo])
async def list_users(
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    List all users.
    Only admin users can access this endpoint.
    """
    try:
        # Mock implementation - in production, get from database
        mock_users = [
            UserInfo(
                id=1,
                username="admin",
                email="admin@example.com",
                full_name="System Administrator",
                is_active=True,
                permissions=["read", "write", "admin"],
                created_at="2024-01-01T00:00:00Z",
                last_login="2024-01-15T10:30:00Z"
            ),
            UserInfo(
                id=2,
                username="user",
                email="user@example.com",
                full_name="Regular User",
                is_active=True,
                permissions=["read"],
                created_at="2024-01-02T00:00:00Z",
                last_login="2024-01-14T15:45:00Z"
            )
        ]
        
        return mock_users
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.post("/", response_model=UserInfo, status_code=status.HTTP_201_CREATED)
async def create_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    full_name: Optional[str] = None,
    permissions: List[str] = Query(default=['read']),
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Create a new user.
    Only admin users can create new users.
    """
    try:
        # Validate permissions
        valid_permissions = {'read', 'write', 'admin'}
        if not set(permissions).issubset(valid_permissions):
            invalid_perms = set(permissions) - valid_permissions
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid permissions: {', '.join(invalid_perms)}"
            )
        
        # Create user
        new_user = await auth_service.create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name,
            permissions=permissions
        )
        
        logger.info(f"User {username} created by {current_user.username}")
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/{user_id}", response_model=UserInfo)
async def get_user(
    user_id: int,
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get user by ID.
    Only admin users can access this endpoint.
    """
    try:
        # Mock implementation
        if user_id == 1:
            return UserInfo(
                id=1,
                username="admin",
                email="admin@example.com",
                full_name="System Administrator",
                is_active=True,
                permissions=["read", "write", "admin"],
                created_at="2024-01-01T00:00:00Z",
                last_login="2024-01-15T10:30:00Z"
            )
        elif user_id == 2:
            return UserInfo(
                id=2,
                username="user",
                email="user@example.com",
                full_name="Regular User",
                is_active=True,
                permissions=["read"],
                created_at="2024-01-02T00:00:00Z",
                last_login="2024-01-14T15:45:00Z"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )

@router.put("/{user_id}", response_model=UserInfo)
async def update_user(
    user_id: int,
    email: Optional[str] = None,
    full_name: Optional[str] = None,
    permissions: Optional[List[str]] = None,
    is_active: Optional[bool] = None,
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Update user information.
    Only admin users can update users.
    """
    try:
        # Validate permissions if provided
        if permissions:
            valid_permissions = {'read', 'write', 'admin'}
            if not set(permissions).issubset(valid_permissions):
                invalid_perms = set(permissions) - valid_permissions
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid permissions: {', '.join(invalid_perms)}"
                )
        
        # Mock implementation - in production, update in database
        if user_id not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Return updated user (mock)
        updated_user = UserInfo(
            id=user_id,
            username="admin" if user_id == 1 else "user",
            email=email or ("admin@example.com" if user_id == 1 else "user@example.com"),
            full_name=full_name or ("System Administrator" if user_id == 1 else "Regular User"),
            is_active=is_active if is_active is not None else True,
            permissions=permissions or (["read", "write", "admin"] if user_id == 1 else ["read"]),
            created_at="2024-01-01T00:00:00Z",
            last_login="2024-01-15T10:30:00Z"
        )
        
        logger.info(f"User {user_id} updated by {current_user.username}")
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/{user_id}", response_model=SuccessResponse)
async def delete_user(
    user_id: int,
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Delete user.
    Only admin users can delete users.
    """
    try:
        # Prevent self-deletion
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Mock implementation
        if user_id not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User {user_id} deleted by {current_user.username}")
        
        return SuccessResponse(
            message="User deleted successfully",
            data={"user_id": user_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )

@router.post("/{user_id}/change-password", response_model=SuccessResponse)
async def change_user_password(
    user_id: int,
    new_password: str,
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Change user password.
    Only admin users can change other users' passwords.
    """
    try:
        # Validate password strength
        if len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Mock implementation - in production, update password hash in database
        if user_id not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Password changed for user {user_id} by {current_user.username}")
        
        return SuccessResponse(
            message="Password changed successfully",
            data={"user_id": user_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@router.get("/{user_id}/sessions")
async def get_user_sessions(
    user_id: int,
    current_user: UserInfo = Depends(require_permissions(['admin'])),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get active sessions for a user.
    Only admin users can view user sessions.
    """
    try:
        # Mock implementation
        if user_id not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        mock_sessions = [
            {
                "session_id": "sess_123456",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "created_at": "2024-01-15T10:30:00Z",
                "last_activity": "2024-01-15T11:45:00Z",
                "is_active": True
            }
        ]
        
        return {
            "user_id": user_id,
            "sessions": mock_sessions,
            "total_sessions": len(mock_sessions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sessions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user sessions"
        )