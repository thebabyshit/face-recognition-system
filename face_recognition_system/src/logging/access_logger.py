"""Access logging system implementation."""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import uuid

from database.services import get_database_service
from database.models import AccessLog, SystemLog, Person, Location

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AccessResult(Enum):
    """Access result types."""
    GRANTED = "granted"
    DENIED = "denied"
    ERROR = "error"
    PENDING = "pending"

class SystemEventType(Enum):
    """System event types."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    EMERGENCY_MODE_ENABLED = "emergency_mode_enabled"
    EMERGENCY_MODE_DISABLED = "emergency_mode_disabled"
    LOCATION_LOCKED = "location_locked"
    LOCATION_UNLOCKED = "location_unlocked"
    SECURITY_ALERT = "security_alert"
    HARDWARE_ERROR = "hardware_error"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_FAILURE = "authentication_failure"
    CONFIGURATION_CHANGED = "configuration_changed"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"

class AccessLogger:
    """Comprehensive access and system logging service."""
    
    def __init__(self):
        """Initialize access logger."""
        self.db_service = get_database_service()
        self.log_queue = asyncio.Queue()
        self.batch_size = 100
        self.batch_timeout = 5.0  # seconds
        self.log_retention_days = 365
        
        # Statistics cache
        self.stats_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.last_cache_update = {}
        
        # Start background tasks
        asyncio.create_task(self._process_log_queue())
        asyncio.create_task(self._cleanup_old_logs())
        
        logger.info("Access logger initialized")
    
    async def log_access_attempt(
        self,
        person_id: Optional[int],
        location_id: int,
        access_granted: bool,
        access_method: str = "face_recognition",
        confidence_score: Optional[float] = None,
        reason: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log an access attempt.
        
        Args:
            person_id: ID of the person attempting access
            location_id: ID of the location being accessed
            access_granted: Whether access was granted
            access_method: Method used for access attempt
            confidence_score: Confidence score of identification
            reason: Reason for access result
            additional_data: Additional context data
            session_id: Session ID for tracking
            
        Returns:
            Dict containing log entry details
        """
        try:
            log_entry = {
                'id': str(uuid.uuid4()),
                'person_id': person_id,
                'location_id': location_id,
                'access_granted': access_granted,
                'access_method': access_method,
                'confidence_score': confidence_score,
                'reason': reason or ("Access granted" if access_granted else "Access denied"),
                'additional_data': additional_data or {},
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc),
                'ip_address': additional_data.get('ip_address') if additional_data else None,
                'user_agent': additional_data.get('user_agent') if additional_data else None
            }
            
            # Add to processing queue
            await self.log_queue.put(('access', log_entry))
            
            # Log to application logger
            person_name = "Unknown"
            location_name = "Unknown"
            
            if person_id:
                person = self.db_service.persons.get_person_by_id(person_id)
                if person:
                    person_name = person.name
            
            location = self.db_service.locations.get_location_by_id(location_id)
            if location:
                location_name = location.name
            
            log_message = (
                f"Access {'GRANTED' if access_granted else 'DENIED'}: "
                f"{person_name} -> {location_name} "
                f"(method: {access_method}, confidence: {confidence_score})"
            )
            
            if access_granted:
                logger.info(log_message)
            else:
                logger.warning(log_message)
            
            return {
                'log_id': log_entry['id'],
                'timestamp': log_entry['timestamp'],
                'status': 'queued'
            }
            
        except Exception as e:
            logger.error(f"Error logging access attempt: {e}")
            return {
                'log_id': None,
                'timestamp': datetime.now(timezone.utc),
                'status': 'error',
                'error': str(e)
            }
    
    async def log_system_event(
        self,
        event_type: Union[str, SystemEventType],
        description: str,
        user_id: Optional[str] = None,
        level: Union[str, LogLevel] = LogLevel.INFO,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a system event.
        
        Args:
            event_type: Type of system event
            description: Event description
            user_id: ID of user who triggered the event
            level: Log level
            additional_data: Additional context data
            
        Returns:
            Dict containing log entry details
        """
        try:
            if isinstance(event_type, SystemEventType):
                event_type = event_type.value
            if isinstance(level, LogLevel):
                level = level.value
            
            log_entry = {
                'id': str(uuid.uuid4()),
                'event_type': event_type,
                'description': description,
                'user_id': user_id,
                'level': level,
                'additional_data': additional_data or {},
                'timestamp': datetime.now(timezone.utc),
                'ip_address': additional_data.get('ip_address') if additional_data else None,
                'user_agent': additional_data.get('user_agent') if additional_data else None
            }
            
            # Add to processing queue
            await self.log_queue.put(('system', log_entry))
            
            # Log to application logger
            log_message = f"System Event [{event_type}]: {description}"
            if user_id:
                log_message += f" (user: {user_id})"
            
            if level == LogLevel.DEBUG.value:
                logger.debug(log_message)
            elif level == LogLevel.INFO.value:
                logger.info(log_message)
            elif level == LogLevel.WARNING.value:
                logger.warning(log_message)
            elif level == LogLevel.ERROR.value:
                logger.error(log_message)
            elif level == LogLevel.CRITICAL.value:
                logger.critical(log_message)
            
            return {
                'log_id': log_entry['id'],
                'timestamp': log_entry['timestamp'],
                'status': 'queued'
            }
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            return {
                'log_id': None,
                'timestamp': datetime.now(timezone.utc),
                'status': 'error',
                'error': str(e)
            }
    
    async def _process_log_queue(self):
        """Process log entries from queue in batches."""
        batch = []
        last_batch_time = datetime.now()
        
        while True:
            try:
                # Wait for log entry or timeout
                try:
                    log_type, log_entry = await asyncio.wait_for(
                        self.log_queue.get(), 
                        timeout=self.batch_timeout
                    )
                    batch.append((log_type, log_entry))
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if it's full or timeout reached
                current_time = datetime.now()
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and (current_time - last_batch_time).total_seconds() >= self.batch_timeout)
                )
                
                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Error processing log queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: List[tuple]):
        """Process a batch of log entries."""
        try:
            access_logs = []
            system_logs = []
            
            for log_type, log_entry in batch:
                if log_type == 'access':
                    access_log = AccessLog(
                        id=log_entry['id'],
                        person_id=log_entry['person_id'],
                        location_id=log_entry['location_id'],
                        access_granted=log_entry['access_granted'],
                        access_method=log_entry['access_method'],
                        confidence_score=log_entry['confidence_score'],
                        reason=log_entry['reason'],
                        additional_data=log_entry['additional_data'],
                        session_id=log_entry['session_id'],
                        timestamp=log_entry['timestamp'],
                        ip_address=log_entry['ip_address'],
                        user_agent=log_entry['user_agent']
                    )
                    access_logs.append(access_log)
                    
                elif log_type == 'system':
                    system_log = SystemLog(
                        id=log_entry['id'],
                        event_type=log_entry['event_type'],
                        description=log_entry['description'],
                        user_id=log_entry['user_id'],
                        level=log_entry['level'],
                        additional_data=log_entry['additional_data'],
                        timestamp=log_entry['timestamp'],
                        ip_address=log_entry['ip_address'],
                        user_agent=log_entry['user_agent']
                    )
                    system_logs.append(system_log)
            
            # Batch insert to database
            if access_logs:
                self.db_service.access_logs.batch_create_access_logs(access_logs)
                logger.debug(f"Inserted {len(access_logs)} access log entries")
            
            if system_logs:
                self.db_service.system_logs.batch_create_system_logs(system_logs)
                logger.debug(f"Inserted {len(system_logs)} system log entries")
            
            # Clear stats cache to force refresh
            self.stats_cache.clear()
            self.last_cache_update.clear()
            
        except Exception as e:
            logger.error(f"Error processing log batch: {e}")
            # Re-queue failed entries
            for log_type, log_entry in batch:
                await self.log_queue.put((log_type, log_entry))
    
    async def _cleanup_old_logs(self):
        """Periodically clean up old log entries."""
        while True:
            try:
                # Run cleanup daily
                await asyncio.sleep(24 * 3600)
                
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.log_retention_days)
                
                # Clean up access logs
                deleted_access = self.db_service.access_logs.delete_logs_before_date(cutoff_date)
                if deleted_access > 0:
                    logger.info(f"Cleaned up {deleted_access} old access log entries")
                
                # Clean up system logs
                deleted_system = self.db_service.system_logs.delete_logs_before_date(cutoff_date)
                if deleted_system > 0:
                    logger.info(f"Cleaned up {deleted_system} old system log entries")
                
                # Log cleanup event
                await self.log_system_event(
                    SystemEventType.SYSTEM_START,  # Using existing enum value
                    f"Log cleanup completed: {deleted_access + deleted_system} entries removed",
                    level=LogLevel.INFO
                )
                
            except Exception as e:
                logger.error(f"Error during log cleanup: {e}")
    
    # Query methods
    
    async def get_access_logs(
        self,
        person_id: Optional[int] = None,
        location_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        access_granted: Optional[bool] = None,
        access_method: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Query access logs with filters.
        
        Args:
            person_id: Filter by person ID
            location_id: Filter by location ID
            start_date: Filter by start date
            end_date: Filter by end date
            access_granted: Filter by access result
            access_method: Filter by access method
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Dict containing logs and metadata
        """
        try:
            filters = {}
            if person_id is not None:
                filters['person_id'] = person_id
            if location_id is not None:
                filters['location_id'] = location_id
            if start_date is not None:
                filters['start_date'] = start_date
            if end_date is not None:
                filters['end_date'] = end_date
            if access_granted is not None:
                filters['access_granted'] = access_granted
            if access_method is not None:
                filters['access_method'] = access_method
            
            logs = self.db_service.access_logs.get_access_logs_with_filters(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            total_count = self.db_service.access_logs.count_access_logs_with_filters(filters)
            
            # Convert to dict format
            log_data = []
            for log in logs:
                log_dict = {
                    'id': log.id,
                    'person_id': log.person_id,
                    'person_name': log.person.name if log.person else None,
                    'location_id': log.location_id,
                    'location_name': log.location.name if log.location else None,
                    'access_granted': log.access_granted,
                    'access_method': log.access_method,
                    'confidence_score': log.confidence_score,
                    'reason': log.reason,
                    'timestamp': log.timestamp,
                    'session_id': log.session_id,
                    'additional_data': log.additional_data
                }
                log_data.append(log_dict)
            
            return {
                'logs': log_data,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + len(log_data)) < total_count
            }
            
        except Exception as e:
            logger.error(f"Error querying access logs: {e}")
            return {
                'logs': [],
                'total_count': 0,
                'limit': limit,
                'offset': offset,
                'has_more': False,
                'error': str(e)
            }
    
    async def get_system_logs(
        self,
        event_type: Optional[str] = None,
        level: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Query system logs with filters.
        
        Args:
            event_type: Filter by event type
            level: Filter by log level
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Dict containing logs and metadata
        """
        try:
            filters = {}
            if event_type is not None:
                filters['event_type'] = event_type
            if level is not None:
                filters['level'] = level
            if user_id is not None:
                filters['user_id'] = user_id
            if start_date is not None:
                filters['start_date'] = start_date
            if end_date is not None:
                filters['end_date'] = end_date
            
            logs = self.db_service.system_logs.get_system_logs_with_filters(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            total_count = self.db_service.system_logs.count_system_logs_with_filters(filters)
            
            # Convert to dict format
            log_data = []
            for log in logs:
                log_dict = {
                    'id': log.id,
                    'event_type': log.event_type,
                    'description': log.description,
                    'user_id': log.user_id,
                    'level': log.level,
                    'timestamp': log.timestamp,
                    'additional_data': log.additional_data
                }
                log_data.append(log_dict)
            
            return {
                'logs': log_data,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + len(log_data)) < total_count
            }
            
        except Exception as e:
            logger.error(f"Error querying system logs: {e}")
            return {
                'logs': [],
                'total_count': 0,
                'limit': limit,
                'offset': offset,
                'has_more': False,
                'error': str(e)
            }
    
    # Statistics methods
    
    async def get_access_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """
        Get access statistics.
        
        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            group_by: Grouping period (hour, day, week, month)
            
        Returns:
            Dict containing statistics data
        """
        try:
            cache_key = f"access_stats_{start_date}_{end_date}_{group_by}"
            
            # Check cache
            if cache_key in self.stats_cache:
                cache_time = self.last_cache_update.get(cache_key, datetime.min)
                if (datetime.now() - cache_time).total_seconds() < self.cache_timeout:
                    return self.stats_cache[cache_key]
            
            # Default to last 30 days if no dates provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            stats = self.db_service.access_logs.get_access_statistics(
                start_date=start_date,
                end_date=end_date,
                group_by=group_by
            )
            
            # Cache results
            self.stats_cache[cache_key] = stats
            self.last_cache_update[cache_key] = datetime.now()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting access statistics: {e}")
            return {
                'total_attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0,
                'success_rate': 0.0,
                'timeline': [],
                'by_person': [],
                'by_location': [],
                'by_method': [],
                'error': str(e)
            }
    
    async def get_security_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get security-related statistics.
        
        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            Dict containing security statistics
        """
        try:
            cache_key = f"security_stats_{start_date}_{end_date}"
            
            # Check cache
            if cache_key in self.stats_cache:
                cache_time = self.last_cache_update.get(cache_key, datetime.min)
                if (datetime.now() - cache_time).total_seconds() < self.cache_timeout:
                    return self.stats_cache[cache_key]
            
            # Default to last 30 days if no dates provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            stats = self.db_service.system_logs.get_security_statistics(
                start_date=start_date,
                end_date=end_date
            )
            
            # Cache results
            self.stats_cache[cache_key] = stats
            self.last_cache_update[cache_key] = datetime.now()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting security statistics: {e}")
            return {
                'total_events': 0,
                'security_alerts': 0,
                'authentication_failures': 0,
                'system_errors': 0,
                'by_event_type': [],
                'by_level': [],
                'timeline': [],
                'error': str(e)
            }
    
    # Utility methods
    
    def clear_cache(self):
        """Clear statistics cache."""
        self.stats_cache.clear()
        self.last_cache_update.clear()
        logger.info("Statistics cache cleared")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queue_size': self.log_queue.qsize(),
            'batch_size': self.batch_size,
            'batch_timeout': self.batch_timeout,
            'cache_entries': len(self.stats_cache),
            'retention_days': self.log_retention_days
        }
    
    def update_configuration(self, config: Dict[str, Any]):
        """Update logger configuration."""
        if 'batch_size' in config:
            self.batch_size = config['batch_size']
        if 'batch_timeout' in config:
            self.batch_timeout = config['batch_timeout']
        if 'log_retention_days' in config:
            self.log_retention_days = config['log_retention_days']
        if 'cache_timeout' in config:
            self.cache_timeout = config['cache_timeout']
        
        logger.info(f"Logger configuration updated: {config}")