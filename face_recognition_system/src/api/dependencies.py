"""Dependencies for FastAPI routes."""
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database.services import get_database_service, DatabaseService
from .auth import get_current_user, get_current_active_user, require_permissions, UserInfo

def get_db_service() -> DatabaseService:
    """
    Get database service dependency.
    Returns:
        DatabaseService: Database service instance
    """
    return get_database_service()

def get_recognition_service(request: Request) -> Optional[object]:
    """
    Get recognition service dependency.
    Args:
        request: FastAPI request object
    Returns:
        Recognition service instance or None
    """
    return getattr(request.app.state, 'recognition_service', None)

# Authentication dependencies
async def require_authentication(current_user: UserInfo = Depends(get_current_active_user)) -> UserInfo:
    """
    Require authentication for protected endpoints.
    Args:
        current_user: Current user from get_current_active_user
    Returns:
        UserInfo: Authenticated user information
    """
    return current_user

# Permission-based dependencies
require_read_permission = require_permissions(['read'])
require_write_permission = require_permissions(['write'])
require_admin_permission = require_permissions(['admin'])

def validate_pagination(limit: int = 50, offset: int = 0) -> dict:
    """
    Validate pagination parameters.
    Args:
        limit: Number of items to return
        offset: Number of items to skip
    Returns:
        dict: Validated pagination parameters
    Raises:
        HTTPException: If parameters are invalid
    """
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 1000"
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset must be non-negative"
        )
    return {"limit": limit, "offset": offset}

def validate_person_id(person_id: int) -> int:
    """
    Validate person ID parameter.
    Args:
        person_id: Person ID to validate
    Returns:
        int: Validated person ID
    Raises:
        HTTPException: If person ID is invalid
    """
    if person_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Person ID must be positive"
        )
    return person_id

def validate_location_id(location_id: int) -> int:
    """
    Validate location ID parameter.
    Args:
        location_id: Location ID to validate
    Returns:
        int: Validated location ID
    Raises:
        HTTPException: If location ID is invalid
    """
    if location_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Location ID must be positive"
        )
    return location_id

class CommonQueryParams:
    """Common query parameters for list endpoints."""
    def __init__(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ):
        self.limit = min(max(limit, 1), 1000)  # Clamp between 1 and 1000
        self.offset = max(offset, 0)  # Ensure non-negative
        self.sort_by = sort_by
        self.sort_order = sort_order.lower() if sort_order.lower() in ["asc", "desc"] else "asc"

def get_common_params(params: CommonQueryParams = Depends()) -> CommonQueryParams:
    """Get common query parameters dependency."""
    return params

# Backward compatibility - these will be deprecated
async def get_current_user_legacy() -> Optional[dict]:
    """Legacy function for backward compatibility."""
    return {
        "id": 1,
        "username": "admin",
        "permissions": ["read", "write", "admin"]
    }