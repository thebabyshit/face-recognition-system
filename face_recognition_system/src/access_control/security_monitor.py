"""Security monitoring system for access control."""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats."""
    TAILGATING = "tailgating"
    FORCED_ENTRY = "forced_entry"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    REPEATED_FAILURES = "repeated_failures"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DEVICE_TAMPERING = "device_tampering"
    SYSTEM_INTRUSION = "system_intrusion"

class SecurityMonitor:
    """Security monitoring and threat detection system."""
    
    def __init__(self):
        """Initialize security monitor."""
        self.threat_history = []
        self.active_alerts = {}
        self.monitoring_enabled = True
        
        # Security configuration
        self.config = {
            'tailgating_detection_enabled': True,
            'tailgating_time_window': 3,  # seconds
            'max_people_per_entry': 1,
            'suspicious_behavior_threshold': 5,  # failed attempts
            'alert_cooldown_period': 300,  # seconds
            'threat_retention_days': 30,
            'auto_lockdown_enabled': True,
            'critical_threat_lockdown': True,
            'notification_enabled': True
        }
        
        # Threat detection state
        self.recent_entries = {}
        self.failed_attempts = {}
        self.suspicious_activities = {}
        
        # Alert handlers
        self.alert_handlers = []
        
        logger.info("Security monitor initialized")
    
    async def check_security_threats(
        self, 
        person_id: Optional[int], 
        location_id: int, 
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check for security threats during access attempt.
        
        Args:
            person_id: ID of person attempting access
            location_id: Location being accessed
            additional_data: Additional context data
            
        Returns:
            Dict containing threat assessment results
        """
        if not self.monitoring_enabled:
            return {'safe': True, 'threats': []}
        
        threats = []
        
        try:
            # Check for tailgating
            if self.config['tailgating_detection_enabled']:
                tailgating_threat = await self._check_tailgating(
                    person_id, location_id, additional_data
                )
                if tailgating_threat:
                    threats.append(tailgating_threat)
            
            # Check for suspicious behavior patterns
            suspicious_threat = await self._check_suspicious_behavior(
                person_id, location_id, additional_data
            )
            if suspicious_threat:
                threats.append(suspicious_threat)
            
            # Check for device tampering
            tampering_threat = await self._check_device_tampering(
                location_id, additional_data
            )
            if tampering_threat:
                threats.append(tampering_threat)
            
            # Check for forced entry attempts
            forced_entry_threat = await self._check_forced_entry(
                location_id, additional_data
            )
            if forced_entry_threat:
                threats.append(forced_entry_threat)
            
            # Determine overall threat level
            max_threat_level = ThreatLevel.LOW
            for threat in threats:
                threat_level = ThreatLevel(threat['level'])
                if threat_level.value == 'critical':
                    max_threat_level = ThreatLevel.CRITICAL
                    break
                elif threat_level.value == 'high' and max_threat_level.value != 'critical':
                    max_threat_level = ThreatLevel.HIGH
                elif threat_level.value == 'medium' and max_threat_level.value == 'low':
                    max_threat_level = ThreatLevel.MEDIUM
            
            # Record threat assessment
            if threats:
                await self._record_threat_assessment(
                    person_id, location_id, threats, max_threat_level, additional_data
                )
            
            # Determine if access should be allowed
            safe = len(threats) == 0 or max_threat_level.value in ['low', 'medium']
            
            result = {
                'safe': safe,
                'threats': threats,
                'threat_level': max_threat_level.value,
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Add reason for denial if not safe
            if not safe:
                primary_threat = max(threats, key=lambda t: ['low', 'medium', 'high', 'critical'].index(t['level']))
                result['threat_type'] = primary_threat['type']
                result['reason'] = primary_threat['description']
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking security threats: {e}")
            return {
                'safe': True,  # Fail open to avoid lockout
                'threats': [],
                'error': str(e)
            }
    
    async def _check_tailgating(
        self, 
        person_id: Optional[int], 
        location_id: int, 
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check for tailgating attempts."""
        current_time = datetime.now(timezone.utc)
        time_window = timedelta(seconds=self.config['tailgating_time_window'])
        
        # Get recent entries for this location
        location_key = f"location_{location_id}"
        if location_key not in self.recent_entries:
            self.recent_entries[location_key] = []
        
        recent_entries = self.recent_entries[location_key]
        
        # Remove old entries outside time window
        cutoff_time = current_time - time_window
        recent_entries = [
            entry for entry in recent_entries 
            if entry['timestamp'] > cutoff_time
        ]
        self.recent_entries[location_key] = recent_entries
        
        # Check if there are too many people trying to enter
        if len(recent_entries) >= self.config['max_people_per_entry']:
            return {
                'type': ThreatType.TAILGATING.value,
                'level': ThreatLevel.HIGH.value,
                'description': f'Tailgating detected: {len(recent_entries) + 1} people within {self.config["tailgating_time_window"]} seconds',
                'location_id': location_id,
                'person_id': person_id,
                'recent_entries': len(recent_entries),
                'time_window': self.config['tailgating_time_window']
            }
        
        # Add current entry to recent entries
        recent_entries.append({
            'person_id': person_id,
            'timestamp': current_time,
            'additional_data': additional_data
        })
        
        return None
    
    async def _check_suspicious_behavior(
        self, 
        person_id: Optional[int], 
        location_id: int, 
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check for suspicious behavior patterns."""
        if person_id is None:
            return None
        
        current_time = datetime.now(timezone.utc)
        person_key = f"person_{person_id}"
        
        # Initialize tracking for this person
        if person_key not in self.suspicious_activities:
            self.suspicious_activities[person_key] = {
                'failed_attempts': 0,
                'last_attempt': None,
                'locations_attempted': set(),
                'unusual_times': 0
            }
        
        activity = self.suspicious_activities[person_key]
        
        # Check for unusual access times (outside normal hours)
        hour = current_time.hour
        if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
            activity['unusual_times'] += 1
            if activity['unusual_times'] >= 3:
                return {
                    'type': ThreatType.SUSPICIOUS_BEHAVIOR.value,
                    'level': ThreatLevel.MEDIUM.value,
                    'description': f'Unusual access time pattern detected for person {person_id}',
                    'person_id': person_id,
                    'location_id': location_id,
                    'unusual_times': activity['unusual_times'],
                    'current_hour': hour
                }
        
        # Check for multiple location attempts
        activity['locations_attempted'].add(location_id)
        if len(activity['locations_attempted']) > 5:
            return {
                'type': ThreatType.SUSPICIOUS_BEHAVIOR.value,
                'level': ThreatLevel.MEDIUM.value,
                'description': f'Multiple location access attempts by person {person_id}',
                'person_id': person_id,
                'location_id': location_id,
                'locations_count': len(activity['locations_attempted'])
            }
        
        # Check for rapid successive attempts
        if activity['last_attempt']:
            time_diff = (current_time - activity['last_attempt']).total_seconds()
            if time_diff < 10:  # Less than 10 seconds between attempts
                return {
                    'type': ThreatType.SUSPICIOUS_BEHAVIOR.value,
                    'level': ThreatLevel.MEDIUM.value,
                    'description': f'Rapid successive access attempts by person {person_id}',
                    'person_id': person_id,
                    'location_id': location_id,
                    'time_between_attempts': time_diff
                }
        
        activity['last_attempt'] = current_time
        
        return None
    
    async def _check_device_tampering(
        self, 
        location_id: int, 
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check for device tampering attempts."""
        if not additional_data:
            return None
        
        # Check for hardware anomalies
        hardware_status = additional_data.get('hardware_status', {})
        
        for device_type, status in hardware_status.items():
            if status.get('status') == 'error' or status.get('tampered'):
                return {
                    'type': ThreatType.DEVICE_TAMPERING.value,
                    'level': ThreatLevel.HIGH.value,
                    'description': f'Device tampering detected: {device_type} at location {location_id}',
                    'location_id': location_id,
                    'device_type': device_type,
                    'device_status': status
                }
        
        # Check for unusual sensor readings
        sensor_data = additional_data.get('sensor_data', {})
        
        # Door sensor anomalies
        if 'door_sensor' in sensor_data:
            door_data = sensor_data['door_sensor']
            if door_data.get('forced_open') or door_data.get('unexpected_state'):
                return {
                    'type': ThreatType.FORCED_ENTRY.value,
                    'level': ThreatLevel.CRITICAL.value,
                    'description': f'Forced entry detected at location {location_id}',
                    'location_id': location_id,
                    'sensor_data': door_data
                }
        
        return None
    
    async def _check_forced_entry(
        self, 
        location_id: int, 
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check for forced entry attempts."""
        if not additional_data:
            return None
        
        # Check for physical force indicators
        force_indicators = additional_data.get('force_indicators', {})
        
        if force_indicators.get('door_forced') or force_indicators.get('lock_broken'):
            return {
                'type': ThreatType.FORCED_ENTRY.value,
                'level': ThreatLevel.CRITICAL.value,
                'description': f'Physical forced entry detected at location {location_id}',
                'location_id': location_id,
                'force_indicators': force_indicators
            }
        
        # Check for electronic bypass attempts
        bypass_attempts = additional_data.get('bypass_attempts', 0)
        if bypass_attempts > 0:
            return {
                'type': ThreatType.SYSTEM_INTRUSION.value,
                'level': ThreatLevel.HIGH.value,
                'description': f'Electronic bypass attempt detected at location {location_id}',
                'location_id': location_id,
                'bypass_attempts': bypass_attempts
            }
        
        return None
    
    async def _record_threat_assessment(
        self,
        person_id: Optional[int],
        location_id: int,
        threats: List[Dict[str, Any]],
        threat_level: ThreatLevel,
        additional_data: Optional[Dict[str, Any]]
    ):
        """Record threat assessment in history."""
        assessment = {
            'timestamp': datetime.now(timezone.utc),
            'person_id': person_id,
            'location_id': location_id,
            'threats': threats,
            'threat_level': threat_level.value,
            'additional_data': additional_data
        }
        
        self.threat_history.append(assessment)
        
        # Clean up old history
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config['threat_retention_days'])
        self.threat_history = [
            assessment for assessment in self.threat_history
            if assessment['timestamp'] > cutoff_date
        ]
        
        # Trigger alerts for high-level threats
        if threat_level.value in ['high', 'critical']:
            await self._trigger_security_alert(assessment)
    
    async def _trigger_security_alert(self, assessment: Dict[str, Any]):
        """Trigger security alert for threat assessment."""
        alert_id = f"alert_{datetime.now(timezone.utc).timestamp()}"
        
        alert = {
            'id': alert_id,
            'timestamp': assessment['timestamp'],
            'threat_level': assessment['threat_level'],
            'location_id': assessment['location_id'],
            'person_id': assessment['person_id'],
            'threats': assessment['threats'],
            'status': 'active',
            'acknowledged': False
        }
        
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        if self.config['notification_enabled']:
            await self._send_alert_notifications(alert)
        
        logger.warning(f"Security alert triggered: {alert_id} - {assessment['threat_level']} level")
    
    async def _send_alert_notifications(self, alert: Dict[str, Any]):
        """Send alert notifications to registered handlers."""
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    async def send_security_alert(
        self, 
        threat_type: str, 
        description: str, 
        additional_data: Dict[str, Any] = None
    ):
        """Send immediate security alert."""
        alert = {
            'id': f"manual_alert_{datetime.now(timezone.utc).timestamp()}",
            'timestamp': datetime.now(timezone.utc),
            'threat_type': threat_type,
            'description': description,
            'additional_data': additional_data or {},
            'status': 'active',
            'manual': True
        }
        
        self.active_alerts[alert['id']] = alert
        
        if self.config['notification_enabled']:
            await self._send_alert_notifications(alert)
        
        logger.warning(f"Manual security alert: {threat_type} - {description}")
    
    def record_failed_attempt(
        self, 
        person_id: Optional[int], 
        location_id: int, 
        reason: str,
        additional_data: Dict[str, Any] = None
    ):
        """Record failed access attempt for threat analysis."""
        key = f"{person_id}:{location_id}"
        current_time = datetime.now(timezone.utc)
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        self.failed_attempts[key].append({
            'timestamp': current_time,
            'reason': reason,
            'additional_data': additional_data
        })
        
        # Check if threshold exceeded
        recent_failures = [
            attempt for attempt in self.failed_attempts[key]
            if (current_time - attempt['timestamp']).total_seconds() < 300  # 5 minutes
        ]
        
        if len(recent_failures) >= self.config['suspicious_behavior_threshold']:
            asyncio.create_task(self.send_security_alert(
                'REPEATED_FAILED_ATTEMPTS',
                f'Multiple failed attempts detected for person {person_id} at location {location_id}',
                {
                    'person_id': person_id,
                    'location_id': location_id,
                    'failure_count': len(recent_failures),
                    'time_window': 300
                }
            ))
    
    def get_threat_history(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        threat_level: Optional[str] = None,
        location_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered threat history."""
        filtered_history = self.threat_history
        
        if start_date:
            filtered_history = [
                assessment for assessment in filtered_history
                if assessment['timestamp'] >= start_date
            ]
        
        if end_date:
            filtered_history = [
                assessment for assessment in filtered_history
                if assessment['timestamp'] <= end_date
            ]
        
        if threat_level:
            filtered_history = [
                assessment for assessment in filtered_history
                if assessment['threat_level'] == threat_level
            ]
        
        if location_id:
            filtered_history = [
                assessment for assessment in filtered_history
                if assessment['location_id'] == location_id
            ]
        
        return sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)
    
    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get all active security alerts."""
        return {
            alert_id: alert for alert_id, alert in self.active_alerts.items()
            if alert['status'] == 'active'
        }
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a security alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['acknowledged'] = True
            self.active_alerts[alert_id]['acknowledged_by'] = acknowledged_by
            self.active_alerts[alert_id]['acknowledged_at'] = datetime.now(timezone.utc)
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Resolve a security alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['status'] = 'resolved'
            self.active_alerts[alert_id]['resolved_by'] = resolved_by
            self.active_alerts[alert_id]['resolved_at'] = datetime.now(timezone.utc)
            self.active_alerts[alert_id]['resolution_notes'] = resolution_notes
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        
        return False
    
    def add_alert_handler(self, handler):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
        logger.info("Alert handler added")
    
    def remove_alert_handler(self, handler):
        """Remove alert notification handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            logger.info("Alert handler removed")
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security monitoring statistics."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate statistics for different time periods
        stats = {
            'total_threats': len(self.threat_history),
            'active_alerts': len(self.get_active_alerts()),
            'monitoring_enabled': self.monitoring_enabled,
            'configuration': self.config.copy()
        }
        
        # Threat statistics by level
        threat_levels = {}
        for assessment in self.threat_history:
            level = assessment['threat_level']
            threat_levels[level] = threat_levels.get(level, 0) + 1
        
        stats['threat_levels'] = threat_levels
        
        # Recent activity (last 24 hours)
        recent_cutoff = current_time - timedelta(hours=24)
        recent_threats = [
            assessment for assessment in self.threat_history
            if assessment['timestamp'] > recent_cutoff
        ]
        
        stats['recent_24h'] = {
            'total_threats': len(recent_threats),
            'threat_levels': {}
        }
        
        for assessment in recent_threats:
            level = assessment['threat_level']
            stats['recent_24h']['threat_levels'][level] = stats['recent_24h']['threat_levels'].get(level, 0) + 1
        
        return stats
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """Update security monitoring configuration."""
        for key, value in config_updates.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                logger.info(f"Security config updated: {key} = {value} (was {old_value})")
            else:
                logger.warning(f"Unknown security configuration key: {key}")
    
    def enable_monitoring(self):
        """Enable security monitoring."""
        self.monitoring_enabled = True
        logger.info("Security monitoring enabled")
    
    def disable_monitoring(self):
        """Disable security monitoring."""
        self.monitoring_enabled = False
        logger.warning("Security monitoring disabled")
    
    def clear_threat_history(self, older_than_days: int = None):
        """Clear threat history."""
        if older_than_days:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            original_count = len(self.threat_history)
            self.threat_history = [
                assessment for assessment in self.threat_history
                if assessment['timestamp'] > cutoff_date
            ]
            cleared_count = original_count - len(self.threat_history)
            logger.info(f"Cleared {cleared_count} threat history entries older than {older_than_days} days")
        else:
            cleared_count = len(self.threat_history)
            self.threat_history = []
            logger.info(f"Cleared all {cleared_count} threat history entries")