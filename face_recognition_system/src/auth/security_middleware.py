"""Security middleware for API protection."""

import logging
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitRule:
    """Rate limiting rule."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 10

@dataclass
class SecurityEvent:
    """Security event for monitoring."""
    event_type: str
    ip_address: str
    user_id: Optional[int]
    timestamp: datetime
    details: Dict[str, Any]
    severity: str = "info"  # info, warning, critical

class SecurityMiddleware:
    """Security middleware for API protection and monitoring."""
    
    def __init__(self):
        """Initialize security middleware."""
        # Rate limiting storage (use Redis in production)
        self.rate_limit_storage = defaultdict(lambda: defaultdict(deque))
        
        # Security configuration
        self.config = {
            'enable_rate_limiting': True,
            'enable_ip_blocking': True,
            'enable_request_logging': True,
            'enable_security_headers': True,
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'blocked_ips': set(),
            'allowed_ips': set(),  # If not empty, only these IPs are allowed
            'suspicious_threshold': 10,  # Failed attempts before marking as suspicious
            'auto_block_duration_minutes': 60,
            'security_headers': {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'",
                'Referrer-Policy': 'strict-origin-when-cross-origin'
            }
        }
        
        # Rate limiting rules by endpoint type
        self.rate_limit_rules = {
            'default': RateLimitRule(60, 1000, 10000, 10),
            'auth': RateLimitRule(10, 100, 500, 5),
            'upload': RateLimitRule(20, 200, 1000, 3),
            'admin': RateLimitRule(100, 2000, 20000, 20)
        }
        
        # Security events storage
        self.security_events = deque(maxlen=1000)
        
        # Suspicious activity tracking
        self.suspicious_ips = defaultdict(lambda: {'count': 0, 'first_seen': None, 'last_seen': None})
        
        # Request statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'rate_limited_requests': 0,
            'suspicious_requests': 0,
            'security_events': 0
        }
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        logger.info("Security middleware initialized")
    
    async def process_request(
        self,
        request_info: Dict[str, Any],
        endpoint_type: str = 'default'
    ) -> Dict[str, Any]:
        """
        Process incoming request through security checks.
        
        Args:
            request_info: Request information (IP, user_id, path, method, etc.)
            endpoint_type: Type of endpoint for rate limiting
            
        Returns:
            Security check result
        """
        try:
            ip_address = request_info.get('ip_address')
            user_id = request_info.get('user_id')
            path = request_info.get('path', '')
            method = request_info.get('method', 'GET')
            
            # Update statistics
            self.stats['total_requests'] += 1
            
            # Check IP allowlist/blocklist
            ip_check = self._check_ip_restrictions(ip_address)
            if not ip_check['allowed']:
                self._log_security_event('ip_blocked', ip_address, user_id, ip_check)
                return {
                    'allowed': False,
                    'reason': 'ip_blocked',
                    'message': ip_check['reason'],
                    'headers': self._get_security_headers()
                }
            
            # Check rate limiting
            if self.config['enable_rate_limiting']:
                rate_limit_check = self._check_rate_limit(ip_address, user_id, endpoint_type)
                if not rate_limit_check['allowed']:
                    self._log_security_event('rate_limited', ip_address, user_id, rate_limit_check)
                    self.stats['rate_limited_requests'] += 1
                    return {
                        'allowed': False,
                        'reason': 'rate_limited',
                        'message': rate_limit_check['message'],
                        'retry_after': rate_limit_check.get('retry_after', 60),
                        'headers': self._get_security_headers()
                    }
            
            # Check request size
            request_size = request_info.get('content_length', 0)
            if request_size > self.config['max_request_size']:
                self._log_security_event('request_too_large', ip_address, user_id, {
                    'size': request_size,
                    'max_size': self.config['max_request_size']
                })
                return {
                    'allowed': False,
                    'reason': 'request_too_large',
                    'message': f'Request size {request_size} exceeds maximum {self.config["max_request_size"]}',
                    'headers': self._get_security_headers()
                }
            
            # Check for suspicious patterns
            suspicious_check = self._check_suspicious_activity(request_info)
            if suspicious_check['is_suspicious']:
                self._log_security_event('suspicious_activity', ip_address, user_id, suspicious_check)
                self.stats['suspicious_requests'] += 1
                
                # Auto-block if threshold exceeded
                if suspicious_check.get('should_block', False):
                    self._auto_block_ip(ip_address, 'suspicious_activity')
                    return {
                        'allowed': False,
                        'reason': 'auto_blocked',
                        'message': 'IP automatically blocked due to suspicious activity',
                        'headers': self._get_security_headers()
                    }
            
            # Log successful request
            if self.config['enable_request_logging']:
                self._log_security_event('request_allowed', ip_address, user_id, {
                    'path': path,
                    'method': method,
                    'endpoint_type': endpoint_type
                }, severity='info')
            
            return {
                'allowed': True,
                'headers': self._get_security_headers(),
                'rate_limit_info': self._get_rate_limit_info(ip_address, user_id, endpoint_type)
            }
            
        except Exception as e:
            logger.error(f"Error in security middleware: {e}")
            return {
                'allowed': True,  # Fail open for availability
                'headers': self._get_security_headers(),
                'warning': 'Security check failed'
            }
    
    def _check_ip_restrictions(self, ip_address: str) -> Dict[str, Any]:
        """Check IP allowlist and blocklist."""
        try:
            # Check if IP is blocked
            if ip_address in self.config['blocked_ips']:
                return {
                    'allowed': False,
                    'reason': 'IP address is blocked'
                }
            
            # Check allowlist (if configured)
            if self.config['allowed_ips'] and ip_address not in self.config['allowed_ips']:
                return {
                    'allowed': False,
                    'reason': 'IP address not in allowlist'
                }
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Error checking IP restrictions: {e}")
            return {'allowed': True}  # Fail open
    
    def _check_rate_limit(self, ip_address: str, user_id: Optional[int], endpoint_type: str) -> Dict[str, Any]:
        """Check rate limiting rules."""
        try:
            rule = self.rate_limit_rules.get(endpoint_type, self.rate_limit_rules['default'])
            now = time.time()
            
            # Use user_id if available, otherwise use IP
            key = f"user_{user_id}" if user_id else f"ip_{ip_address}"
            
            # Clean old entries and check limits
            windows = {
                'minute': (60, rule.requests_per_minute),
                'hour': (3600, rule.requests_per_hour),
                'day': (86400, rule.requests_per_day)
            }
            
            for window_name, (window_size, limit) in windows.items():
                window_key = f"{key}_{window_name}"
                requests = self.rate_limit_storage[window_key]['requests']
                
                # Remove old requests
                while requests and requests[0] < now - window_size:
                    requests.popleft()
                
                # Check limit
                if len(requests) >= limit:
                    return {
                        'allowed': False,
                        'message': f'Rate limit exceeded: {len(requests)}/{limit} requests per {window_name}',
                        'retry_after': int(window_size - (now - requests[0])) if requests else 60,
                        'window': window_name,
                        'current_count': len(requests),
                        'limit': limit
                    }
            
            # Add current request to all windows
            for window_name in windows.keys():
                window_key = f"{key}_{window_name}"
                self.rate_limit_storage[window_key]['requests'].append(now)
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return {'allowed': True}  # Fail open
    
    def _check_suspicious_activity(self, request_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check for suspicious activity patterns."""
        try:
            ip_address = request_info.get('ip_address')
            path = request_info.get('path', '')
            method = request_info.get('method', 'GET')
            user_agent = request_info.get('user_agent', '')
            
            suspicious_indicators = []
            
            # Check for common attack patterns in path
            attack_patterns = [
                '../', '..\\\\', '/etc/passwd', '/proc/', 'cmd.exe',
                '<script', 'javascript:', 'vbscript:', 'onload=',
                'union select', 'drop table', 'insert into',
                '<?php', '<%', 'eval(', 'exec('
            ]
            
            for pattern in attack_patterns:
                if pattern.lower() in path.lower():
                    suspicious_indicators.append(f'Attack pattern in path: {pattern}')
            
            # Check for suspicious user agents
            suspicious_agents = [
                'sqlmap', 'nikto', 'nmap', 'masscan', 'zap',
                'burp', 'w3af', 'acunetix', 'nessus'
            ]
            
            for agent in suspicious_agents:
                if agent.lower() in user_agent.lower():
                    suspicious_indicators.append(f'Suspicious user agent: {agent}')
            
            # Check for excessive failed attempts from this IP
            ip_info = self.suspicious_ips[ip_address]
            if ip_info['count'] > self.config['suspicious_threshold']:
                suspicious_indicators.append(f'Excessive failed attempts: {ip_info["count"]}')
            
            # Check for rapid requests (potential DoS)
            recent_requests = self.rate_limit_storage[f"ip_{ip_address}_minute"]['requests']
            if len(recent_requests) > 30:  # More than 30 requests per minute
                suspicious_indicators.append(f'Rapid requests: {len(recent_requests)}/minute')
            
            is_suspicious = len(suspicious_indicators) > 0
            should_block = len(suspicious_indicators) >= 2 or ip_info['count'] > self.config['suspicious_threshold']
            
            return {
                'is_suspicious': is_suspicious,
                'should_block': should_block,
                'indicators': suspicious_indicators,
                'ip_failure_count': ip_info['count']
            }
            
        except Exception as e:
            logger.error(f"Error checking suspicious activity: {e}")
            return {'is_suspicious': False}
    
    def _get_security_headers(self) -> Dict[str, str]:
        """Get security headers to add to response."""
        if self.config['enable_security_headers']:
            return self.config['security_headers'].copy()
        return {}
    
    def _get_rate_limit_info(self, ip_address: str, user_id: Optional[int], endpoint_type: str) -> Dict[str, Any]:
        """Get current rate limit information."""
        try:
            rule = self.rate_limit_rules.get(endpoint_type, self.rate_limit_rules['default'])
            key = f"user_{user_id}" if user_id else f"ip_{ip_address}"
            
            info = {}
            windows = {
                'minute': (60, rule.requests_per_minute),
                'hour': (3600, rule.requests_per_hour),
                'day': (86400, rule.requests_per_day)
            }
            
            for window_name, (window_size, limit) in windows.items():
                window_key = f"{key}_{window_name}"
                requests = self.rate_limit_storage[window_key]['requests']
                current_count = len(requests)
                
                info[f'{window_name}_limit'] = limit
                info[f'{window_name}_remaining'] = max(0, limit - current_count)
                info[f'{window_name}_reset'] = int(time.time() + window_size) if requests else None
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            return {}
    
    def _log_security_event(
        self,
        event_type: str,
        ip_address: str,
        user_id: Optional[int],
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """Log security event."""
        try:
            event = SecurityEvent(
                event_type=event_type,
                ip_address=ip_address,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                details=details,
                severity=severity
            )
            
            self.security_events.append(event)
            self.stats['security_events'] += 1
            
            # Log to system logger based on severity
            log_message = f"Security event: {event_type} from {ip_address}"
            if user_id:
                log_message += f" (user {user_id})"
            
            if severity == "critical":
                logger.critical(log_message)
            elif severity == "warning":
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def _auto_block_ip(self, ip_address: str, reason: str):
        """Automatically block an IP address."""
        try:
            self.config['blocked_ips'].add(ip_address)
            
            # Schedule unblock after configured duration
            asyncio.create_task(self._schedule_unblock(ip_address))
            
            logger.warning(f"Auto-blocked IP {ip_address} for {reason}")
            
        except Exception as e:
            logger.error(f"Error auto-blocking IP {ip_address}: {e}")
    
    async def _schedule_unblock(self, ip_address: str):
        """Schedule IP unblock after configured duration."""
        try:
            await asyncio.sleep(self.config['auto_block_duration_minutes'] * 60)
            
            if ip_address in self.config['blocked_ips']:
                self.config['blocked_ips'].remove(ip_address)
                logger.info(f"Auto-unblocked IP {ip_address}")
                
        except Exception as e:
            logger.error(f"Error scheduling unblock for {ip_address}: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old data."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = time.time()
                
                # Clean up old rate limit data
                for storage_key in list(self.rate_limit_storage.keys()):
                    requests = self.rate_limit_storage[storage_key]['requests']
                    
                    # Determine window size based on key
                    if '_minute' in storage_key:
                        window_size = 60
                    elif '_hour' in storage_key:
                        window_size = 3600
                    elif '_day' in storage_key:
                        window_size = 86400
                    else:
                        continue
                    
                    # Remove old requests
                    while requests and requests[0] < now - window_size:
                        requests.popleft()
                    
                    # Remove empty storage
                    if not requests:
                        del self.rate_limit_storage[storage_key]
                
                # Clean up old suspicious IP data
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                for ip in list(self.suspicious_ips.keys()):
                    ip_info = self.suspicious_ips[ip]
                    if ip_info['last_seen'] and ip_info['last_seen'] < cutoff_time:
                        del self.suspicious_ips[ip]
                
                logger.debug("Security middleware cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in security middleware cleanup: {e}")
                await asyncio.sleep(60)
    
    # Public management methods
    
    def block_ip(self, ip_address: str, reason: str = "Manual block"):
        """Manually block an IP address."""
        try:
            self.config['blocked_ips'].add(ip_address)
            self._log_security_event('ip_manually_blocked', ip_address, None, {'reason': reason}, 'warning')
            logger.info(f"Manually blocked IP {ip_address}: {reason}")
            
        except Exception as e:
            logger.error(f"Error blocking IP {ip_address}: {e}")
    
    def unblock_ip(self, ip_address: str):
        """Manually unblock an IP address."""
        try:
            self.config['blocked_ips'].discard(ip_address)
            self._log_security_event('ip_manually_unblocked', ip_address, None, {}, 'info')
            logger.info(f"Manually unblocked IP {ip_address}")
            
        except Exception as e:
            logger.error(f"Error unblocking IP {ip_address}: {e}")
    
    def add_to_allowlist(self, ip_address: str):
        """Add IP to allowlist."""
        try:
            self.config['allowed_ips'].add(ip_address)
            logger.info(f"Added IP {ip_address} to allowlist")
            
        except Exception as e:
            logger.error(f"Error adding IP {ip_address} to allowlist: {e}")
    
    def remove_from_allowlist(self, ip_address: str):
        """Remove IP from allowlist."""
        try:
            self.config['allowed_ips'].discard(ip_address)
            logger.info(f"Removed IP {ip_address} from allowlist")
            
        except Exception as e:
            logger.error(f"Error removing IP {ip_address} from allowlist: {e}")
    
    def record_failed_attempt(self, ip_address: str, user_id: Optional[int] = None):
        """Record a failed authentication attempt."""
        try:
            now = datetime.now(timezone.utc)
            ip_info = self.suspicious_ips[ip_address]
            
            ip_info['count'] += 1
            ip_info['last_seen'] = now
            if ip_info['first_seen'] is None:
                ip_info['first_seen'] = now
            
            self._log_security_event('failed_attempt', ip_address, user_id, {
                'total_failures': ip_info['count']
            }, 'warning' if ip_info['count'] > 5 else 'info')
            
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
    
    def reset_failed_attempts(self, ip_address: str):
        """Reset failed attempts counter for an IP."""
        try:
            if ip_address in self.suspicious_ips:
                del self.suspicious_ips[ip_address]
                logger.info(f"Reset failed attempts for IP {ip_address}")
                
        except Exception as e:
            logger.error(f"Error resetting failed attempts for {ip_address}: {e}")
    
    def get_security_events(self, limit: int = 100, event_type: str = None) -> List[Dict[str, Any]]:
        """Get recent security events."""
        try:
            events = list(self.security_events)
            
            # Filter by event type if specified
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Sort by timestamp (newest first) and limit
            events.sort(key=lambda e: e.timestamp, reverse=True)
            events = events[:limit]
            
            # Convert to dictionaries
            return [
                {
                    'event_type': e.event_type,
                    'ip_address': e.ip_address,
                    'user_id': e.user_id,
                    'timestamp': e.timestamp.isoformat(),
                    'details': e.details,
                    'severity': e.severity
                }
                for e in events
            ]
            
        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security middleware statistics."""
        try:
            return {
                **self.stats,
                'blocked_ips_count': len(self.config['blocked_ips']),
                'allowed_ips_count': len(self.config['allowed_ips']),
                'suspicious_ips_count': len(self.suspicious_ips),
                'active_rate_limits': len(self.rate_limit_storage),
                'security_events_count': len(self.security_events)
            }
            
        except Exception as e:
            logger.error(f"Error getting security statistics: {e}")
            return self.stats.copy()
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update security configuration."""
        try:
            for key, value in config_updates.items():
                if key in self.config:
                    old_value = self.config[key]
                    self.config[key] = value
                    logger.info(f"Security config updated: {key} = {value} (was {old_value})")
                else:
                    logger.warning(f"Unknown security config key: {key}")
                    
        except Exception as e:
            logger.error(f"Error updating security config: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current security configuration."""
        config = self.config.copy()
        # Convert sets to lists for JSON serialization
        config['blocked_ips'] = list(config['blocked_ips'])
        config['allowed_ips'] = list(config['allowed_ips'])
        return config