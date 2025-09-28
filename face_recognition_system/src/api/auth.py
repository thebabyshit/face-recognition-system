"""Authentication and authorization module."""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from database.services import get_database_service

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()

class TokenData(BaseModel):
    """Token data model."""
    user_id: int
    username: str
    permissions: List[str]
    exp: datetime

class UserCredentials(BaseModel):
    """User credentials model."""
    username: str
    password: str

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class UserInfo(BaseModel):
    """User information model."""
    id: int
    username: str
    email: Optional[str]
    full_name: Optional[str]
    is_active: bool
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime]

class PasswordHash:
    """Password hashing utilities."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class JWTManager:
    """JWT token management."""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create an access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create a refresh token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

class AuthService:
    """Authentication service."""
    
    def __init__(self):
        self.db_service = get_database_service()
        self.jwt_manager = JWTManager()
        self.password_hash = PasswordHash()
    
    async def authenticate_user(self, username: str, password: str) -> Optional[UserInfo]:
        """Authenticate a user with username and password."""
        try:
            # Get user from database (mock implementation for now)
            user = await self._get_user_by_username(username)
            if not user:
                return None
            
            # Verify password
            if not self.password_hash.verify_password(password, user.get('password_hash', '')):
                return None
            
            # Update last login
            await self._update_last_login(user['id'])
            
            return UserInfo(
                id=user['id'],
                username=user['username'],
                email=user.get('email'),
                full_name=user.get('full_name'),
                is_active=user.get('is_active', True),
                permissions=user.get('permissions', []),
                created_at=user.get('created_at', datetime.now(timezone.utc)),
                last_login=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication error: {str(e)}"
            )
    
    async def create_tokens(self, user: UserInfo) -> TokenResponse:
        """Create access and refresh tokens for a user."""
        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "permissions": user.permissions
        }
        
        access_token = self.jwt_manager.create_access_token(token_data)
        refresh_token = self.jwt_manager.create_refresh_token({"sub": user.username, "user_id": user.id})
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh an access token using a refresh token."""
        payload = self.jwt_manager.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not username or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Get user permissions
        user = await self._get_user_by_username(username)
        if not user or not user.get('is_active'):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        token_data = {
            "sub": username,
            "user_id": user_id,
            "permissions": user.get('permissions', [])
        }
        
        access_token = self.jwt_manager.create_access_token(token_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,  # Keep the same refresh token
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    async def get_current_user(self, token: str) -> UserInfo:
        """Get current user from JWT token."""
        payload = self.jwt_manager.verify_token(token)
        
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not username or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user = await self._get_user_by_username(username)
        if not user or not user.get('is_active'):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return UserInfo(
            id=user['id'],
            username=user['username'],
            email=user.get('email'),
            full_name=user.get('full_name'),
            is_active=user.get('is_active', True),
            permissions=user.get('permissions', []),
            created_at=user.get('created_at', datetime.now(timezone.utc)),
            last_login=user.get('last_login')
        )
    
    async def create_user(self, username: str, password: str, email: Optional[str] = None, 
                         full_name: Optional[str] = None, permissions: List[str] = None) -> UserInfo:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self._get_user_by_username(username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Hash password
        password_hash = self.password_hash.hash_password(password)
        
        # Create user (mock implementation)
        user_data = {
            'username': username,
            'password_hash': password_hash,
            'email': email,
            'full_name': full_name,
            'permissions': permissions or ['read'],
            'is_active': True,
            'created_at': datetime.now(timezone.utc)
        }
        
        user_id = await self._create_user_in_db(user_data)
        
        return UserInfo(
            id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            is_active=True,
            permissions=permissions or ['read'],
            created_at=datetime.now(timezone.utc),
            last_login=None
        )
    
    async def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username from database (mock implementation)."""
        # Mock users for testing
        mock_users = {
            'admin': {
                'id': 1,
                'username': 'admin',
                'password_hash': self.password_hash.hash_password('admin123'),
                'email': 'admin@example.com',
                'full_name': 'System Administrator',
                'permissions': ['read', 'write', 'admin'],
                'is_active': True,
                'created_at': datetime.now(timezone.utc),
                'last_login': None
            },
            'user': {
                'id': 2,
                'username': 'user',
                'password_hash': self.password_hash.hash_password('user123'),
                'email': 'user@example.com',
                'full_name': 'Regular User',
                'permissions': ['read'],
                'is_active': True,
                'created_at': datetime.now(timezone.utc),
                'last_login': None
            }
        }
        
        return mock_users.get(username)
    
    async def _update_last_login(self, user_id: int):
        """Update user's last login time."""
        # Mock implementation
        pass
    
    async def _create_user_in_db(self, user_data: Dict[str, Any]) -> int:
        """Create user in database."""
        # Mock implementation - return a mock user ID
        return len(user_data) + 100

# Global auth service instance
auth_service = AuthService()

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Get current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await auth_service.get_current_user(credentials.credentials)

async def get_current_active_user(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_permissions(required_permissions: List[str]):
    """Create a dependency that requires specific permissions."""
    async def check_permissions(current_user: UserInfo = Depends(get_current_active_user)) -> UserInfo:
        user_permissions = set(current_user.permissions)
        required_perms = set(required_permissions)
        
        # Admin users have all permissions
        if 'admin' in user_permissions:
            return current_user
        
        # Check if user has all required permissions
        if not required_perms.issubset(user_permissions):
            missing_perms = required_perms - user_permissions
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing_perms)}"
            )
        
        return current_user
    
    return check_permissions

# Common permission dependencies
require_read = require_permissions(['read'])
require_write = require_permissions(['write'])
require_admin = require_permissions(['admin'])