"""JWT token management for authentication."""

import logging
import jwt
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import secrets
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: int
    username: str
    roles: List[str]
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    token_type: str = "access"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'roles': self.roles,
            'permissions': self.permissions,
            'iat': int(self.issued_at.timestamp()),
            'exp': int(self.expires_at.timestamp()),
            'token_type': self.token_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenPayload':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            roles=data.get('roles', []),
            permissions=data.get('permissions', []),
            issued_at=datetime.fromtimestamp(data['iat'], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(data['exp'], tz=timezone.utc),
            token_type=data.get('token_type', 'access')
        )

class JWTManager:
    """JWT token manager for authentication and authorization."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize JWT manager.
        
        Args:
            secret_key: Secret key for JWT signing. If None, generates a random key.
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = 'HS256'
        
        # Token configuration
        self.config = {
            'access_token_expire_minutes': 60,  # 1 hour
            'refresh_token_expire_days': 30,    # 30 days
            'issuer': 'face_recognition_system',
            'audience': 'face_recognition_api'
        }
        
        # Token blacklist (in production, use Redis or database)
        self.blacklisted_tokens = set()
        
        logger.info("JWT Manager initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key."""
        return secrets.token_urlsafe(32)
    
    def generate_access_token(
        self, 
        user_id: int, 
        username: str, 
        roles: List[str], 
        permissions: List[str]
    ) -> str:
        """
        Generate access token.
        
        Args:
            user_id: User ID
            username: Username
            roles: User roles
            permissions: User permissions
            
        Returns:
            JWT access token
        """
        try:
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=self.config['access_token_expire_minutes'])
            
            payload = TokenPayload(
                user_id=user_id,
                username=username,
                roles=roles,
                permissions=permissions,
                issued_at=now,
                expires_at=expires_at,
                token_type="access"
            )
            
            token_data = payload.to_dict()
            token_data.update({
                'iss': self.config['issuer'],
                'aud': self.config['audience']
            })
            
            token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
            
            logger.info(f"Access token generated for user {username} (ID: {user_id})")
            return token
            
        except Exception as e:
            logger.error(f"Error generating access token: {e}")
            raise
    
    def generate_refresh_token(
        self, 
        user_id: int, 
        username: str
    ) -> str:
        """
        Generate refresh token.
        
        Args:
            user_id: User ID
            username: Username
            
        Returns:
            JWT refresh token
        """
        try:
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=self.config['refresh_token_expire_days'])
            
            payload = TokenPayload(
                user_id=user_id,
                username=username,
                roles=[],  # Refresh tokens don't need roles/permissions
                permissions=[],
                issued_at=now,
                expires_at=expires_at,
                token_type="refresh"
            )
            
            token_data = payload.to_dict()
            token_data.update({
                'iss': self.config['issuer'],
                'aud': self.config['audience']
            })
            
            token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
            
            logger.info(f"Refresh token generated for user {username} (ID: {user_id})")
            return token
            
        except Exception as e:
            logger.error(f"Error generating refresh token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            TokenPayload if valid, None if invalid
        """
        try:
            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                logger.warning("Attempted to use blacklisted token")
                return None
            
            # Decode token
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                audience=self.config['audience'],
                issuer=self.config['issuer']
            )
            
            # Create TokenPayload object
            token_payload = TokenPayload.from_dict(payload)
            
            # Check if token is expired
            if token_payload.expires_at < datetime.now(timezone.utc):
                logger.warning(f"Expired token used by user {token_payload.username}")
                return None
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str, roles: List[str], permissions: List[str]) -> Optional[str]:
        """
        Generate new access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            roles: Updated user roles
            permissions: Updated user permissions
            
        Returns:
            New access token if refresh token is valid, None otherwise
        """
        try:
            # Verify refresh token
            token_payload = self.verify_token(refresh_token)
            
            if not token_payload:
                logger.warning("Invalid refresh token provided")
                return None
            
            if token_payload.token_type != "refresh":
                logger.warning("Non-refresh token provided to refresh endpoint")
                return None
            
            # Generate new access token
            new_access_token = self.generate_access_token(
                user_id=token_payload.user_id,
                username=token_payload.username,
                roles=roles,
                permissions=permissions
            )
            
            logger.info(f"Access token refreshed for user {token_payload.username}")
            return new_access_token
            
        except Exception as e:
            logger.error(f"Error refreshing access token: {e}")
            return None
    
    def blacklist_token(self, token: str):
        """
        Add token to blacklist.
        
        Args:
            token: Token to blacklist
        """
        try:
            # In production, store in Redis or database with expiration
            token_hash = self._hash_token(token)
            self.blacklisted_tokens.add(token_hash)
            
            logger.info("Token added to blacklist")
            
        except Exception as e:
            logger.error(f"Error blacklisting token: {e}")
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            token_hash = self._hash_token(token)
            return token_hash in self.blacklisted_tokens
        except Exception:
            return False
    
    def _hash_token(self, token: str) -> str:
        """Hash token for blacklist storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get token information without verification.
        
        Args:
            token: JWT token
            
        Returns:
            Token information if decodable, None otherwise
        """
        try:
            # Decode without verification to get info
            payload = jwt.decode(token, options={"verify_signature": False})
            
            return {
                'user_id': payload.get('user_id'),
                'username': payload.get('username'),
                'roles': payload.get('roles', []),
                'permissions': payload.get('permissions', []),
                'token_type': payload.get('token_type'),
                'issued_at': datetime.fromtimestamp(payload.get('iat', 0), tz=timezone.utc),
                'expires_at': datetime.fromtimestamp(payload.get('exp', 0), tz=timezone.utc),
                'is_expired': datetime.fromtimestamp(payload.get('exp', 0), tz=timezone.utc) < datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return None
    
    def validate_token_permissions(self, token: str, required_permissions: List[str]) -> bool:
        """
        Validate if token has required permissions.
        
        Args:
            token: JWT token
            required_permissions: List of required permissions
            
        Returns:
            True if token has all required permissions, False otherwise
        """
        try:
            token_payload = self.verify_token(token)
            
            if not token_payload:
                return False
            
            # Check if user has all required permissions
            user_permissions = set(token_payload.permissions)
            required_permissions_set = set(required_permissions)
            
            return required_permissions_set.issubset(user_permissions)
            
        except Exception as e:
            logger.error(f"Error validating token permissions: {e}")
            return False
    
    def validate_token_roles(self, token: str, required_roles: List[str]) -> bool:
        """
        Validate if token has required roles.
        
        Args:
            token: JWT token
            required_roles: List of required roles
            
        Returns:
            True if token has any of the required roles, False otherwise
        """
        try:
            token_payload = self.verify_token(token)
            
            if not token_payload:
                return False
            
            # Check if user has any of the required roles
            user_roles = set(token_payload.roles)
            required_roles_set = set(required_roles)
            
            return bool(user_roles.intersection(required_roles_set))
            
        except Exception as e:
            logger.error(f"Error validating token roles: {e}")
            return False
    
    def get_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Extract user information from token.
        
        Args:
            token: JWT token
            
        Returns:
            User information if token is valid, None otherwise
        """
        try:
            token_payload = self.verify_token(token)
            
            if not token_payload:
                return None
            
            return {
                'user_id': token_payload.user_id,
                'username': token_payload.username,
                'roles': token_payload.roles,
                'permissions': token_payload.permissions
            }
            
        except Exception as e:
            logger.error(f"Error extracting user from token: {e}")
            return None
    
    def update_config(self, config_updates: Dict[str, Any]):
        """
        Update JWT configuration.
        
        Args:
            config_updates: Configuration updates
        """
        try:
            for key, value in config_updates.items():
                if key in self.config:
                    old_value = self.config[key]
                    self.config[key] = value
                    logger.info(f"JWT config updated: {key} = {value} (was {old_value})")
                else:
                    logger.warning(f"Unknown JWT config key: {key}")
                    
        except Exception as e:
            logger.error(f"Error updating JWT config: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current JWT configuration."""
        return self.config.copy()
    
    def cleanup_blacklist(self):
        """
        Clean up expired tokens from blacklist.
        In production, this would be handled by Redis TTL or database cleanup job.
        """
        # This is a simplified implementation
        # In production, use proper storage with automatic expiration
        logger.info("Blacklist cleanup completed (simplified implementation)")
    
    def generate_api_key(self, user_id: int, username: str, description: str = "") -> str:
        """
        Generate API key for programmatic access.
        
        Args:
            user_id: User ID
            username: Username
            description: API key description
            
        Returns:
            API key token
        """
        try:
            now = datetime.now(timezone.utc)
            # API keys have longer expiration (1 year)
            expires_at = now + timedelta(days=365)
            
            payload = {
                'user_id': user_id,
                'username': username,
                'token_type': 'api_key',
                'description': description,
                'iat': int(now.timestamp()),
                'exp': int(expires_at.timestamp()),
                'iss': self.config['issuer'],
                'aud': self.config['audience']
            }
            
            api_key = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            logger.info(f"API key generated for user {username} (ID: {user_id})")
            return api_key
            
        except Exception as e:
            logger.error(f"Error generating API key: {e}")
            raise
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Verify API key.
        
        Args:
            api_key: API key to verify
            
        Returns:
            API key information if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                api_key,
                self.secret_key,
                algorithms=[self.algorithm],
                audience=self.config['audience'],
                issuer=self.config['issuer']
            )
            
            if payload.get('token_type') != 'api_key':
                return None
            
            return {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'description': payload.get('description', ''),
                'issued_at': datetime.fromtimestamp(payload['iat'], tz=timezone.utc),
                'expires_at': datetime.fromtimestamp(payload['exp'], tz=timezone.utc)
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("API key has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid API key: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None