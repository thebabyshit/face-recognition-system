"""Session management for user authentication."""

import logging
import asyncio
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User session data structure."""
    session_id: str
    user_id: int
    username: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    login_method: str = "password"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Create session from dictionary."""
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def update_last_accessed(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now(timezone.utc)

class SessionManager:
    """Manage user sessions for authentication and authorization."""
    
    def __init__(self):
        """Initialize session manager."""
        # In-memory session storage (use Redis in production)
        self.sessions: Dict[str, UserSession] = {}
        
        # Session configuration
        self.config = {
            'session_timeout_minutes': 60,  # 1 hour
            'max_sessions_per_user': 5,
            'cleanup_interval_minutes': 15,
            'remember_me_days': 30,
            'concurrent_login_allowed': True,
            'session_id_length': 32
        }
        
        # Session statistics
        self.stats = {
            'total_sessions_created': 0,
            'active_sessions': 0,
            'expired_sessions_cleaned': 0,
            'concurrent_logins_blocked': 0
        }
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        logger.info("Session manager initialized")
    
    def create_session(
        self,
        user_id: int,
        username: str,
        roles: List[str],
        permissions: List[str],
        ip_address: str,
        user_agent: str,
        remember_me: bool = False,
        login_method: str = "password"
    ) -> str:
        """
        Create a new user session.
        
        Args:
            user_id: User ID
            username: Username
            roles: User roles
            permissions: User permissions
            ip_address: Client IP address
            user_agent: Client user agent
            remember_me: Whether to extend session duration
            login_method: Login method used
            
        Returns:
            Session ID
        """
        try:
            # Check concurrent login limits
            if not self.config['concurrent_login_allowed']:
                existing_sessions = self._get_user_sessions(user_id)
                if existing_sessions:
                    # Invalidate existing sessions
                    for session in existing_sessions:
                        self._invalidate_session(session.session_id)
                    logger.info(f"Invalidated {len(existing_sessions)} existing sessions for user {user_id}")
            
            # Check max sessions per user
            user_sessions = self._get_user_sessions(user_id)
            if len(user_sessions) >= self.config['max_sessions_per_user']:
                # Remove oldest session
                oldest_session = min(user_sessions, key=lambda s: s.created_at)
                self._invalidate_session(oldest_session.session_id)
                logger.info(f"Removed oldest session for user {user_id} due to limit")
            
            # Generate session ID
            session_id = self._generate_session_id()
            
            # Calculate expiration
            now = datetime.now(timezone.utc)
            if remember_me:
                expires_at = now + timedelta(days=self.config['remember_me_days'])
            else:
                expires_at = now + timedelta(minutes=self.config['session_timeout_minutes'])
            
            # Create session
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                username=username,
                roles=roles,
                permissions=permissions,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                login_method=login_method
            )
            
            # Store session
            self.sessions[session_id] = session
            
            # Update statistics
            self.stats['total_sessions_created'] += 1
            self.stats['active_sessions'] = len([s for s in self.sessions.values() if s.is_active])
            
            logger.info(f"Session created for user {username} (ID: {user_id}), session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session for user {user_id}: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            UserSession if valid and active, None otherwise
        """
        try:
            session = self.sessions.get(session_id)
            
            if not session:
                return None
            
            # Check if session is expired
            if session.is_expired():
                self._invalidate_session(session_id)
                return None
            
            # Check if session is active
            if not session.is_active:
                return None
            
            # Update last accessed time
            session.update_last_accessed()
            
            return session
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    def validate_session(self, session_id: str, ip_address: str = None) -> bool:
        """
        Validate session.
        
        Args:
            session_id: Session ID
            ip_address: Client IP address (optional, for IP validation)
            
        Returns:
            True if session is valid, False otherwise
        """
        try:
            session = self.get_session(session_id)
            
            if not session:
                return False
            
            # Optional IP address validation
            if ip_address and session.ip_address != ip_address:
                logger.warning(f"IP address mismatch for session {session_id}: {ip_address} vs {session.ip_address}")
                # In production, you might want to invalidate the session here
                # For now, we'll just log the warning
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating session {session_id}: {e}")
            return False
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session was invalidated, False if not found
        """
        try:
            return self._invalidate_session(session_id)
            
        except Exception as e:
            logger.error(f"Error invalidating session {session_id}: {e}")
            return False
    
    def invalidate_user_sessions(self, user_id: int) -> int:
        """
        Invalidate all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of sessions invalidated
        """
        try:
            user_sessions = self._get_user_sessions(user_id)
            count = 0
            
            for session in user_sessions:
                if self._invalidate_session(session.session_id):
                    count += 1
            
            logger.info(f"Invalidated {count} sessions for user {user_id}")
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating sessions for user {user_id}: {e}")
            return 0
    
    def extend_session(self, session_id: str, minutes: int = None) -> bool:
        """
        Extend session expiration.
        
        Args:
            session_id: Session ID
            minutes: Minutes to extend (uses default timeout if None)
            
        Returns:
            True if session was extended, False otherwise
        """
        try:
            session = self.sessions.get(session_id)
            
            if not session or not session.is_active:
                return False
            
            # Calculate new expiration
            if minutes is None:
                minutes = self.config['session_timeout_minutes']
            
            new_expiration = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            session.expires_at = new_expiration
            session.update_last_accessed()
            
            logger.info(f"Session {session_id} extended by {minutes} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Error extending session {session_id}: {e}")
            return False
    
    def get_user_sessions(self, user_id: int) -> List[UserSession]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active sessions
        """
        try:
            return [s for s in self._get_user_sessions(user_id) if s.is_active and not s.is_expired()]
            
        except Exception as e:
            logger.error(f"Error getting sessions for user {user_id}: {e}")
            return []
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information dictionary
        """
        try:
            session = self.sessions.get(session_id)
            
            if not session:
                return None
            
            return {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'username': session.username,
                'created_at': session.created_at.isoformat(),
                'last_accessed': session.last_accessed.isoformat(),
                'expires_at': session.expires_at.isoformat(),
                'is_active': session.is_active,
                'is_expired': session.is_expired(),
                'login_method': session.login_method,
                'ip_address': session.ip_address,
                'user_agent': session.user_agent[:100] + '...' if len(session.user_agent) > 100 else session.user_agent
            }
            
        except Exception as e:
            logger.error(f"Error getting session info for {session_id}: {e}")
            return None
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of active session information
        """
        try:
            active_sessions = []
            
            for session in self.sessions.values():
                if session.is_active and not session.is_expired():
                    session_info = self.get_session_info(session.session_id)
                    if session_info:
                        active_sessions.append(session_info)
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Error listing active sessions: {e}")
            return []
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Session statistics dictionary
        """
        try:
            # Update active sessions count
            self.stats['active_sessions'] = len([s for s in self.sessions.values() if s.is_active and not s.is_expired()])
            
            # Calculate additional stats
            total_sessions = len(self.sessions)
            expired_sessions = len([s for s in self.sessions.values() if s.is_expired()])
            
            return {
                **self.stats,
                'total_sessions_in_memory': total_sessions,
                'expired_sessions': expired_sessions,
                'session_cleanup_needed': expired_sessions > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return self.stats.copy()
    
    def update_session_permissions(self, session_id: str, roles: List[str], permissions: List[str]) -> bool:
        """
        Update session roles and permissions.
        
        Args:
            session_id: Session ID
            roles: Updated roles
            permissions: Updated permissions
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            session = self.sessions.get(session_id)
            
            if not session or not session.is_active:
                return False
            
            session.roles = roles
            session.permissions = permissions
            session.update_last_accessed()
            
            logger.info(f"Updated permissions for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating session permissions for {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if session.is_expired() or not session.is_active:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            # Update statistics
            self.stats['expired_sessions_cleaned'] += len(expired_sessions)
            self.stats['active_sessions'] = len([s for s in self.sessions.values() if s.is_active])
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def export_sessions(self) -> str:
        """
        Export sessions to JSON string.
        
        Returns:
            JSON string containing session data
        """
        try:
            sessions_data = {}
            for session_id, session in self.sessions.items():
                sessions_data[session_id] = session.to_dict()
            
            return json.dumps(sessions_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting sessions: {e}")
            return "{}"
    
    def get_config(self) -> Dict[str, Any]:
        """Get session manager configuration."""
        return self.config.copy()
    
    def update_config(self, config_updates: Dict[str, Any]):
        """
        Update session manager configuration.
        
        Args:
            config_updates: Configuration updates
        """
        try:
            for key, value in config_updates.items():
                if key in self.config:
                    old_value = self.config[key]
                    self.config[key] = value
                    logger.info(f"Session config updated: {key} = {value} (was {old_value})")
                else:
                    logger.warning(f"Unknown session config key: {key}")
                    
        except Exception as e:
            logger.error(f"Error updating session config: {e}")
    
    # Private methods
    
    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return secrets.token_urlsafe(self.config['session_id_length'])
    
    def _get_user_sessions(self, user_id: int) -> List[UserSession]:
        """Get all sessions for a user (including expired)."""
        return [s for s in self.sessions.values() if s.user_id == user_id]
    
    def _invalidate_session(self, session_id: str) -> bool:
        """Internal method to invalidate a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            logger.info(f"Session {session_id} invalidated for user {session.username}")
            return True
        return False
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task for expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.config['cleanup_interval_minutes'] * 60)
                self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error in periodic session cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying