"""Access management service implementation."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

from database.services import get_database_service
from database.models import Person, AccessLocation, AccessPermission, AccessLog
from utils.exceptions import (
    PersonNotFoundError,
    ValidationError,
    AccessDeniedError,
    PermissionError
)

logger = logging.getLogger(__name__)


class AccessResult(Enum):
    """Access attempt results."""
    GRANTED = "granted"
    DENIED = "denied"
    ERROR = "error"


class AccessManager:
    """Access management service for controlling and logging access attempts."""
    
    def __init__(self):
        self.db_service = get_database_service()
    
    def check_access(self,
                    person_id: int,
                    location_id: int,
                    timestamp: datetime = None) -> Dict[str, Any]:
        """
        Check if a person has access to a location.
        
        Args:
            person_id: ID of the person
            location_id: ID of the location
            timestamp: Time of access attempt (default: now)
            
        Returns:
            Dict: Access check result with details
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        try:
            # Get person and location
            person = self.db_service.persons.get_person_by_id(person_id)
            if not person:
                return {
                    'access_granted': False,
                    'reason': 'person_not_found',
                    'message': f'Person with ID {person_id} not found'
                }
            
            with self.db_service.persons._get_db_manager() as db:
                location = db.session.query(AccessLocation).filter_by(id=location_id).first()
                if not location:
                    return {
                        'access_granted': False,
                        'reason': 'location_not_found',
                        'message': f'Location with ID {location_id} not found'
                    }
                
                # Check if person is active
                if not person.is_active:
                    return {
                        'access_granted': False,
                        'reason': 'person_inactive',
                        'message': 'Person account is inactive'
                    }
                
                # Check if location is active
                if not location.is_active:
                    return {
                        'access_granted': False,
                        'reason': 'location_inactive',
                        'message': 'Access location is inactive'
                    }
                
                # Check access level
                if person.access_level < location.required_access_level:
                    return {
                        'access_granted': False,
                        'reason': 'insufficient_access_level',
                        'message': f'Required access level: {location.required_access_level}, person has: {person.access_level}'
                    }
                
                # Check specific permissions
                permission_result = self._check_permissions(person, location, timestamp)
                if not permission_result['granted']:
                    return {
                        'access_granted': False,
                        'reason': permission_result['reason'],
                        'message': permission_result['message']
                    }
                
                # Check time restrictions
                time_result = self._check_time_restrictions(person, location, timestamp)
                if not time_result['allowed']:
                    return {
                        'access_granted': False,
                        'reason': 'time_restriction',
                        'message': time_result['message']
                    }
                
                # All checks passed
                return {
                    'access_granted': True,
                    'reason': 'authorized',
                    'message': 'Access granted',
                    'person_name': person.name,
                    'location_name': location.name,
                    'access_level': person.access_level
                }
                
        except Exception as e:
            logger.error(f"Error checking access for person {person_id} at location {location_id}: {e}")
            return {
                'access_granted': False,
                'reason': 'system_error',
                'message': f'System error: {str(e)}'
            }
    
    def log_access_attempt(self,
                          person_id: int = None,
                          location_id: int = None,
                          access_granted: bool = False,
                          recognition_confidence: float = None,
                          similarity_score: float = None,
                          matched_feature_id: int = None,
                          failure_reason: str = None,
                          image_path: str = None,
                          processing_time_ms: int = None,
                          device_info: Dict = None,
                          metadata: Dict = None) -> Optional[AccessLog]:
        """
        Log an access attempt.
        
        Args:
            person_id: ID of person (None if not recognized)
            location_id: ID of location
            access_granted: Whether access was granted
            recognition_confidence: Face recognition confidence
            similarity_score: Feature similarity score
            matched_feature_id: ID of matched face feature
            failure_reason: Reason for access denial
            image_path: Path to captured image
            processing_time_ms: Processing time in milliseconds
            device_info: Device information
            metadata: Additional metadata
            
        Returns:
            AccessLog: Created access log entry
        """
        try:
            log_entry = self.db_service.access_logs.log_access_attempt(
                location_id=location_id,
                person_id=person_id,
                access_granted=access_granted,
                recognition_confidence=recognition_confidence,
                similarity_score=similarity_score,
                matched_feature_id=matched_feature_id,
                failure_reason=failure_reason,
                image_path=image_path,
                processing_time_ms=processing_time_ms,
                device_info=device_info,
                metadata=metadata
            )
            
            if log_entry:
                logger.info(f"Logged access attempt: Person {person_id}, Location {location_id}, Granted: {access_granted}")
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Error logging access attempt: {e}")
            return None
    
    def grant_access_permission(self,
                               person_id: int,
                               location_id: int,
                               permission_type: str = 'allow',
                               valid_from: datetime = None,
                               valid_until: datetime = None,
                               time_restrictions: Dict = None,
                               granted_by: int = None) -> bool:
        """
        Grant access permission to a person for a location.
        
        Args:
            person_id: ID of the person
            location_id: ID of the location
            permission_type: Type of permission ('allow', 'deny', 'temporary')
            valid_from: Permission valid from date
            valid_until: Permission valid until date
            time_restrictions: Time-based restrictions
            granted_by: ID of user granting permission
            
        Returns:
            bool: True if permission granted successfully
        """
        try:
            # Validate person and location exist
            person = self.db_service.persons.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            with self.db_service.persons._get_db_manager() as db:
                location = db.session.query(AccessLocation).filter_by(id=location_id).first()
                if not location:
                    raise ValidationError(f"Location with ID {location_id} not found")
                
                # Check if permission already exists
                existing_permission = db.session.query(AccessPermission).filter_by(
                    person_id=person_id,
                    location_id=location_id
                ).first()
                
                if existing_permission:
                    # Update existing permission
                    existing_permission.permission_type = permission_type
                    existing_permission.valid_from = valid_from or datetime.now(timezone.utc)
                    existing_permission.valid_until = valid_until
                    existing_permission.is_active = True
                    existing_permission.granted_by = granted_by
                    
                    if time_restrictions:
                        existing_permission.set_time_restrictions(time_restrictions)
                    
                    db.session.flush()
                    logger.info(f"Updated access permission for person {person_id} at location {location_id}")
                    
                else:
                    # Create new permission
                    permission = AccessPermission(
                        person_id=person_id,
                        location_id=location_id,
                        permission_type=permission_type,
                        valid_from=valid_from or datetime.now(timezone.utc),
                        valid_until=valid_until,
                        is_active=True,
                        granted_by=granted_by
                    )
                    
                    if time_restrictions:
                        permission.set_time_restrictions(time_restrictions)
                    
                    db.session.add(permission)
                    db.session.flush()
                    logger.info(f"Granted access permission for person {person_id} at location {location_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error granting access permission: {e}")
            return False
    
    def revoke_access_permission(self, person_id: int, location_id: int, revoked_by: int = None) -> bool:
        """
        Revoke access permission for a person at a location.
        
        Args:
            person_id: ID of the person
            location_id: ID of the location
            revoked_by: ID of user revoking permission
            
        Returns:
            bool: True if permission revoked successfully
        """
        try:
            with self.db_service.persons._get_db_manager() as db:
                permission = db.session.query(AccessPermission).filter_by(
                    person_id=person_id,
                    location_id=location_id
                ).first()
                
                if permission:
                    permission.is_active = False
                    db.session.flush()
                    logger.info(f"Revoked access permission for person {person_id} at location {location_id}")
                    return True
                else:
                    logger.warning(f"No permission found to revoke for person {person_id} at location {location_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error revoking access permission: {e}")
            return False
    
    def get_person_permissions(self, person_id: int) -> List[Dict[str, Any]]:
        """
        Get all access permissions for a person.
        
        Args:
            person_id: ID of the person
            
        Returns:
            List[Dict]: List of permission information
        """
        try:
            with self.db_service.persons._get_db_manager() as db:
                permissions = db.session.query(AccessPermission).filter_by(
                    person_id=person_id,
                    is_active=True
                ).all()
                
                result = []
                for perm in permissions:
                    location = db.session.query(AccessLocation).filter_by(id=perm.location_id).first()
                    
                    perm_info = {
                        'permission_id': perm.id,
                        'location_id': perm.location_id,
                        'location_name': location.name if location else 'Unknown',
                        'permission_type': perm.permission_type,
                        'valid_from': perm.valid_from.isoformat() if perm.valid_from else None,
                        'valid_until': perm.valid_until.isoformat() if perm.valid_until else None,
                        'time_restrictions': perm.get_time_restrictions(),
                        'is_currently_valid': perm.is_valid_at()
                    }
                    result.append(perm_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting permissions for person {person_id}: {e}")
            return []
    
    def get_access_history(self,
                          person_id: int = None,
                          location_id: int = None,
                          start_date: datetime = None,
                          end_date: datetime = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get access history with optional filters.
        
        Args:
            person_id: Filter by person ID
            location_id: Filter by location ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of records
            
        Returns:
            List[Dict]: Access history records
        """
        try:
            with self.db_service.persons._get_db_manager() as db:
                query = db.session.query(AccessLog)
                
                if person_id:
                    query = query.filter(AccessLog.person_id == person_id)
                
                if location_id:
                    query = query.filter(AccessLog.location_id == location_id)
                
                if start_date:
                    query = query.filter(AccessLog.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(AccessLog.timestamp <= end_date)
                
                logs = query.order_by(AccessLog.timestamp.desc()).limit(limit).all()
                
                result = []
                for log in logs:
                    person = db.session.query(Person).filter_by(id=log.person_id).first() if log.person_id else None
                    location = db.session.query(AccessLocation).filter_by(id=log.location_id).first()
                    
                    log_info = {
                        'log_id': log.id,
                        'person_id': log.person_id,
                        'person_name': person.name if person else 'Unknown',
                        'location_id': log.location_id,
                        'location_name': location.name if location else 'Unknown',
                        'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                        'access_granted': log.access_granted,
                        'recognition_confidence': log.recognition_confidence,
                        'similarity_score': log.similarity_score,
                        'failure_reason': log.failure_reason,
                        'processing_time_ms': log.processing_time_ms
                    }
                    result.append(log_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting access history: {e}")
            return []
    
    def get_access_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get access statistics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict: Access statistics
        """
        try:
            stats = self.db_service.access_logs.get_access_statistics(days)
            
            # Add additional statistics
            with self.db_service.persons._get_db_manager() as db:
                since = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Top accessed locations
                location_stats = db.session.query(
                    AccessLog.location_id,
                    AccessLocation.name,
                    db.session.query(AccessLog).filter(
                        AccessLog.location_id == AccessLocation.id,
                        AccessLog.timestamp >= since
                    ).count().label('access_count')
                ).join(AccessLocation).filter(
                    AccessLog.timestamp >= since
                ).group_by(AccessLog.location_id, AccessLocation.name).order_by(
                    db.session.text('access_count DESC')
                ).limit(10).all()
                
                stats['top_locations'] = [
                    {'location_id': loc_id, 'name': name, 'access_count': count}
                    for loc_id, name, count in location_stats
                ]
                
                # Peak hours
                hour_stats = {}
                logs = db.session.query(AccessLog).filter(
                    AccessLog.timestamp >= since
                ).all()
                
                for log in logs:
                    hour = log.timestamp.hour
                    hour_stats[hour] = hour_stats.get(hour, 0) + 1
                
                stats['peak_hours'] = sorted(hour_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting access statistics: {e}")
            return {}
    
    def _check_permissions(self, person: Person, location: AccessLocation, timestamp: datetime) -> Dict[str, Any]:
        """Check specific access permissions."""
        try:
            with self.db_service.persons._get_db_manager() as db:
                permission = db.session.query(AccessPermission).filter_by(
                    person_id=person.id,
                    location_id=location.id,
                    is_active=True
                ).first()
                
                if not permission:
                    # No specific permission, rely on access level
                    return {'granted': True, 'reason': 'access_level', 'message': 'Access based on level'}
                
                if permission.permission_type == 'deny':
                    return {'granted': False, 'reason': 'explicit_deny', 'message': 'Access explicitly denied'}
                
                if not permission.is_valid_at(timestamp):
                    return {'granted': False, 'reason': 'permission_expired', 'message': 'Permission has expired'}
                
                return {'granted': True, 'reason': 'explicit_allow', 'message': 'Access explicitly allowed'}
                
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            return {'granted': False, 'reason': 'permission_error', 'message': str(e)}
    
    def _check_time_restrictions(self, person: Person, location: AccessLocation, timestamp: datetime) -> Dict[str, Any]:
        """Check time-based access restrictions."""
        try:
            with self.db_service.persons._get_db_manager() as db:
                permission = db.session.query(AccessPermission).filter_by(
                    person_id=person.id,
                    location_id=location.id,
                    is_active=True
                ).first()
                
                if not permission:
                    return {'allowed': True, 'message': 'No time restrictions'}
                
                time_restrictions = permission.get_time_restrictions()
                if not time_restrictions:
                    return {'allowed': True, 'message': 'No time restrictions'}
                
                # Check day of week restrictions
                if 'allowed_days' in time_restrictions:
                    allowed_days = time_restrictions['allowed_days']
                    current_day = timestamp.weekday()  # 0=Monday, 6=Sunday
                    if current_day not in allowed_days:
                        return {'allowed': False, 'message': 'Access not allowed on this day'}
                
                # Check time of day restrictions
                if 'allowed_hours' in time_restrictions:
                    allowed_hours = time_restrictions['allowed_hours']
                    current_hour = timestamp.hour
                    if 'start' in allowed_hours and 'end' in allowed_hours:
                        start_hour = allowed_hours['start']
                        end_hour = allowed_hours['end']
                        
                        if start_hour <= end_hour:
                            # Same day range
                            if not (start_hour <= current_hour <= end_hour):
                                return {'allowed': False, 'message': f'Access only allowed between {start_hour}:00 and {end_hour}:00'}
                        else:
                            # Overnight range
                            if not (current_hour >= start_hour or current_hour <= end_hour):
                                return {'allowed': False, 'message': f'Access only allowed between {start_hour}:00 and {end_hour}:00'}
                
                return {'allowed': True, 'message': 'Time restrictions satisfied'}
                
        except Exception as e:
            logger.error(f"Error checking time restrictions: {e}")
            return {'allowed': True, 'message': 'Time restriction check failed, allowing access'}