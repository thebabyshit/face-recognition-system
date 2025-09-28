"""Rate limiting middleware and utilities."""

import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementation using sliding window."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed for the given key.
        
        Args:
            key: Identifier for the client (IP, user ID, etc.)
            
        Returns:
            bool: True if request is allowed, False otherwise
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside the window
        while self.requests[key] and self.requests[key][0] <= window_start:
            self.requests[key].popleft()
        
        # Check if we're under the limit
        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True
        
        return False
    
    def get_reset_time(self, key: str) -> Optional[float]:
        """
        Get the time when the rate limit will reset for the key.
        
        Args:
            key: Identifier for the client
            
        Returns:
            float: Unix timestamp when limit resets, or None if no limit
        """
        if key not in self.requests or not self.requests[key]:
            return None
        
        oldest_request = self.requests[key][0]
        return oldest_request + self.window_seconds
    
    def get_remaining_requests(self, key: str) -> int:
        """
        Get remaining requests for the key.
        
        Args:
            key: Identifier for the client
            
        Returns:
            int: Number of remaining requests
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside the window
        while self.requests[key] and self.requests[key][0] <= window_start:
            self.requests[key].popleft()
        
        return max(0, self.max_requests - len(self.requests[key]))

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI."""
    
    def __init__(
        self,
        app,
        default_requests: int = 100,
        default_window: int = 60,
        per_endpoint_limits: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            default_requests: Default max requests per window
            default_window: Default window size in seconds
            per_endpoint_limits: Per-endpoint rate limits
        """
        super().__init__(app)
        self.default_limiter = RateLimiter(default_requests, default_window)
        self.endpoint_limiters: Dict[str, RateLimiter] = {}
        
        # Set up per-endpoint limiters
        if per_endpoint_limits:
            for endpoint, limits in per_endpoint_limits.items():
                self.endpoint_limiters[endpoint] = RateLimiter(
                    limits.get('requests', default_requests),
                    limits.get('window', default_window)
                )
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Get client identifier
        client_ip = request.client.host
        user_id = None
        
        # Try to get user ID from authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from .auth import JWTManager
                jwt_manager = JWTManager()
                token = auth_header.split(" ")[1]
                payload = jwt_manager.verify_token(token)
                user_id = payload.get("user_id")
            except:
                pass  # Continue with IP-based limiting
        
        # Use user ID if available, otherwise use IP
        client_key = f"user:{user_id}" if user_id else f"ip:{client_ip}"
        
        # Get appropriate rate limiter
        endpoint = request.url.path
        limiter = self.endpoint_limiters.get(endpoint, self.default_limiter)
        
        # Check rate limit
        if not limiter.is_allowed(client_key):
            reset_time = limiter.get_reset_time(client_key)
            reset_timestamp = int(reset_time) if reset_time else None
            
            logger.warning(f"Rate limit exceeded for {client_key} on {endpoint}")
            
            return Response(
                content='{"detail": "Rate limit exceeded", "retry_after": %d}' % (
                    reset_timestamp - int(time.time()) if reset_timestamp else 60
                ),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(limiter.max_requests),
                    "X-RateLimit-Remaining": str(limiter.get_remaining_requests(client_key)),
                    "X-RateLimit-Reset": str(reset_timestamp) if reset_timestamp else "",
                    "Retry-After": str(reset_timestamp - int(time.time()) if reset_timestamp else 60)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(limiter.get_remaining_requests(client_key))
        
        reset_time = limiter.get_reset_time(client_key)
        if reset_time:
            response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        
        return response

def create_rate_limit_config() -> Dict[str, Dict[str, int]]:
    """Create rate limit configuration for different endpoints."""
    return {
        # Authentication endpoints - more restrictive
        "/api/v1/auth/login": {"requests": 5, "window": 60},
        "/api/v1/auth/register": {"requests": 3, "window": 300},
        
        # Recognition endpoints - moderate limits
        "/api/v1/recognition/identify": {"requests": 30, "window": 60},
        "/api/v1/recognition/verify": {"requests": 30, "window": 60},
        
        # Upload endpoints - more restrictive
        "/api/v1/faces/upload": {"requests": 10, "window": 60},
        "/api/v1/faces/upload-file": {"requests": 10, "window": 60},
        
        # Admin endpoints - very restrictive
        "/api/v1/system/maintenance/cleanup": {"requests": 1, "window": 300},
        "/api/v1/system/backup/create": {"requests": 2, "window": 3600},
        
        # Statistics endpoints - moderate limits
        "/api/v1/statistics/generate-report": {"requests": 5, "window": 300},
    }

class IPWhitelist:
    """IP whitelist for bypassing rate limits."""
    
    def __init__(self, whitelist: Optional[List[str]] = None):
        """
        Initialize IP whitelist.
        
        Args:
            whitelist: List of IP addresses to whitelist
        """
        self.whitelist = set(whitelist or [])
        # Add localhost by default
        self.whitelist.update(["127.0.0.1", "::1", "localhost"])
    
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        return ip in self.whitelist
    
    def add_ip(self, ip: str):
        """Add IP to whitelist."""
        self.whitelist.add(ip)
    
    def remove_ip(self, ip: str):
        """Remove IP from whitelist."""
        self.whitelist.discard(ip)

class SmartRateLimitMiddleware(BaseHTTPMiddleware):
    """Smart rate limiting with user-based and IP-based limits."""
    
    def __init__(
        self,
        app,
        default_requests: int = 100,
        default_window: int = 60,
        authenticated_multiplier: float = 2.0,
        admin_multiplier: float = 5.0,
        whitelist: Optional[List[str]] = None
    ):
        """
        Initialize smart rate limit middleware.
        
        Args:
            app: FastAPI application
            default_requests: Default max requests per window
            default_window: Default window size in seconds
            authenticated_multiplier: Multiplier for authenticated users
            admin_multiplier: Multiplier for admin users
            whitelist: List of whitelisted IPs
        """
        super().__init__(app)
        self.default_requests = default_requests
        self.default_window = default_window
        self.authenticated_multiplier = authenticated_multiplier
        self.admin_multiplier = admin_multiplier
        self.ip_whitelist = IPWhitelist(whitelist)
        self.limiters: Dict[str, RateLimiter] = {}
    
    def get_limiter(self, key: str, max_requests: int, window: int) -> RateLimiter:
        """Get or create rate limiter for key."""
        limiter_key = f"{key}:{max_requests}:{window}"
        if limiter_key not in self.limiters:
            self.limiters[limiter_key] = RateLimiter(max_requests, window)
        return self.limiters[limiter_key]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with smart rate limiting."""
        client_ip = request.client.host
        
        # Check if IP is whitelisted
        if self.ip_whitelist.is_whitelisted(client_ip):
            return await call_next(request)
        
        # Determine rate limits based on user type
        max_requests = self.default_requests
        user_permissions = []
        client_key = f"ip:{client_ip}"
        
        # Try to get user info from token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from .auth import JWTManager
                jwt_manager = JWTManager()
                token = auth_header.split(" ")[1]
                payload = jwt_manager.verify_token(token)
                user_id = payload.get("user_id")
                user_permissions = payload.get("permissions", [])
                
                if user_id:
                    client_key = f"user:{user_id}"
                    
                    # Apply multipliers based on user type
                    if "admin" in user_permissions:
                        max_requests = int(self.default_requests * self.admin_multiplier)
                    else:
                        max_requests = int(self.default_requests * self.authenticated_multiplier)
                        
            except:
                pass  # Continue with default limits
        
        # Get rate limiter
        limiter = self.get_limiter(client_key, max_requests, self.default_window)
        
        # Check rate limit
        if not limiter.is_allowed(client_key):
            reset_time = limiter.get_reset_time(client_key)
            
            logger.warning(
                f"Rate limit exceeded for {client_key} "
                f"(permissions: {user_permissions}) on {request.url.path}"
            )
            
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_time)) if reset_time else "",
                    "Retry-After": str(int(reset_time - time.time())) if reset_time else "60"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(limiter.get_remaining_requests(client_key))
        
        return response