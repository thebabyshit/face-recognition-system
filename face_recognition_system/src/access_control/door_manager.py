"""Door management system for access control."""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DoorState(Enum):
    """Door states."""
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    OPEN = "open"
    CLOSED = "closed"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class DoorMode(Enum):
    """Door operation modes."""
    NORMAL = "normal"
    EMERGENCY_OPEN = "emergency_open"
    EMERGENCY_LOCKED = "emergency_locked"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"

class DoorManager:
    """Manager for door control and monitoring."""
    
    def __init__(self):
        """Initialize door manager."""
        self.doors = {}
        self.door_schedules = {}
        self.door_history = []
        
        # Door management configuration
        self.config = {
            'default_unlock_duration': 5,  # seconds
            'max_unlock_duration': 30,  # seconds
            'auto_lock_enabled': True,
            'door_monitoring_enabled': True,
            'emergency_override_enabled': True,
            'maintenance_mode_enabled': False,
            'door_status_check_interval': 10,  # seconds
            'history_retention_days': 90
        }
        
        # Initialize mock doors
        self._initialize_doors()
        
        # Start door monitoring
        if self.config['door_monitoring_enabled']:
            asyncio.create_task(self._monitor_doors())
        
        logger.info("Door manager initialized")
    
    def _initialize_doors(self):
        """Initialize door configurations."""
        # Mock door locations
        mock_locations = [1, 2, 3, 4, 5]
        
        for location_id in mock_locations:
            self.doors[location_id] = {
                'location_id': location_id,
                'name': f'Door {location_id}',
                'state': DoorState.LOCKED,
                'mode': DoorMode.NORMAL,
                'last_action': None,
                'last_action_time': None,
                'unlock_timer': None,
                'auto_lock_enabled': True,
                'emergency_override': False,
                'maintenance_mode': False,
                'sensor_data': {
                    'position_sensor': 'closed',
                    'lock_sensor': 'locked',
                    'tamper_sensor': 'normal',
                    'battery_level': 100
                },
                'access_count': 0,
                'last_maintenance': datetime.now(timezone.utc) - timedelta(days=30),
                'created_at': datetime.now(timezone.utc)
            }
        
        logger.info(f"Initialized {len(self.doors)} doors")
    
    async def _monitor_doors(self):
        """Monitor door status and perform maintenance tasks."""
        while True:
            try:
                await self._check_door_status()
                await self._process_scheduled_actions()
                await self._cleanup_history()
                
                await asyncio.sleep(self.config['door_status_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in door monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _check_door_status(self):
        """Check status of all doors."""
        for location_id, door in self.doors.items():
            try:
                # Simulate sensor readings
                await self._update_sensor_data(door)
                
                # Check for anomalies
                await self._check_door_anomalies(door)
                
                # Update door state based on sensors
                await self._update_door_state(door)
                
            except Exception as e:
                logger.error(f"Error checking door {location_id} status: {e}")
    
    async def _update_sensor_data(self, door: Dict[str, Any]):
        """Update door sensor data (mock implementation)."""
        # In real implementation, this would read from actual sensors
        sensor_data = door['sensor_data']
        
        # Simulate battery drain
        if sensor_data['battery_level'] > 0:
            sensor_data['battery_level'] = max(0, sensor_data['battery_level'] - 0.01)
        
        # Simulate sensor readings based on door state
        if door['state'] == DoorState.LOCKED:
            sensor_data['lock_sensor'] = 'locked'
        elif door['state'] == DoorState.UNLOCKED:
            sensor_data['lock_sensor'] = 'unlocked'
        
        # Random sensor variations (for testing)
        import random
        if random.random() < 0.001:  # 0.1% chance of anomaly
            sensor_data['tamper_sensor'] = 'alert'
        else:
            sensor_data['tamper_sensor'] = 'normal'
    
    async def _check_door_anomalies(self, door: Dict[str, Any]):
        """Check for door anomalies and alert if necessary."""
        sensor_data = door['sensor_data']
        location_id = door['location_id']
        
        # Check battery level
        if sensor_data['battery_level'] < 20:
            await self._log_door_event(
                location_id, 
                'LOW_BATTERY', 
                f"Low battery warning: {sensor_data['battery_level']}%"
            )
        
        # Check tamper sensor
        if sensor_data['tamper_sensor'] == 'alert':
            await self._log_door_event(
                location_id, 
                'TAMPER_ALERT', 
                "Tamper sensor triggered"
            )
            door['state'] = DoorState.ERROR
        
        # Check for stuck door
        if door['last_action_time']:
            time_since_action = (datetime.now(timezone.utc) - door['last_action_time']).total_seconds()
            if door['last_action'] == 'unlock' and time_since_action > self.config['max_unlock_duration'] * 2:
                await self._log_door_event(
                    location_id, 
                    'DOOR_STUCK', 
                    f"Door may be stuck - unlocked for {time_since_action} seconds"
                )
    
    async def _update_door_state(self, door: Dict[str, Any]):
        """Update door state based on sensor data."""
        sensor_data = door['sensor_data']
        
        # Update state based on lock sensor
        if sensor_data['lock_sensor'] == 'locked' and door['state'] != DoorState.LOCKED:
            if door['state'] != DoorState.ERROR:
                door['state'] = DoorState.LOCKED
        elif sensor_data['lock_sensor'] == 'unlocked' and door['state'] == DoorState.LOCKED:
            door['state'] = DoorState.UNLOCKED
    
    async def _process_scheduled_actions(self):
        """Process scheduled door actions."""
        current_time = datetime.now(timezone.utc)
        
        for location_id, schedule in self.door_schedules.items():
            if location_id not in self.doors:
                continue
            
            door = self.doors[location_id]
            
            # Check for scheduled lock/unlock
            for action in schedule.get('actions', []):
                action_time = action['time']
                if (current_time >= action_time and 
                    not action.get('executed', False) and
                    (current_time - action_time).total_seconds() < 60):  # Within 1 minute
                    
                    if action['type'] == 'lock':
                        await self.lock_door(location_id)
                    elif action['type'] == 'unlock':
                        duration = action.get('duration', self.config['default_unlock_duration'])
                        await self.open_door(location_id, duration)
                    
                    action['executed'] = True
                    await self._log_door_event(
                        location_id, 
                        'SCHEDULED_ACTION', 
                        f"Executed scheduled {action['type']}"
                    )
    
    async def _cleanup_history(self):
        """Clean up old door history entries."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config['history_retention_days'])
        
        original_count = len(self.door_history)
        self.door_history = [
            entry for entry in self.door_history
            if entry['timestamp'] > cutoff_date
        ]
        
        cleaned_count = original_count - len(self.door_history)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old door history entries")
    
    async def open_door(self, location_id: int, duration: int = None) -> Dict[str, Any]:
        """
        Open (unlock) door for specified duration.
        
        Args:
            location_id: ID of the location/door
            duration: Duration to keep door unlocked (seconds)
            
        Returns:
            Dict containing operation result
        """
        if location_id not in self.doors:
            return {'success': False, 'error': f'Door not found for location {location_id}'}
        
        door = self.doors[location_id]
        
        # Check if door is in maintenance mode
        if door['mode'] == DoorMode.MAINTENANCE:
            return {'success': False, 'error': 'Door is in maintenance mode'}
        
        # Check if door is disabled
        if door['mode'] == DoorMode.DISABLED:
            return {'success': False, 'error': 'Door is disabled'}
        
        # Use default duration if not specified
        if duration is None:
            duration = self.config['default_unlock_duration']
        
        # Validate duration
        if duration > self.config['max_unlock_duration']:
            duration = self.config['max_unlock_duration']
            logger.warning(f"Duration capped at {duration} seconds for door {location_id}")
        
        try:
            # Cancel existing unlock timer
            if door['unlock_timer']:
                door['unlock_timer'].cancel()
            
            # Unlock the door
            door['state'] = DoorState.UNLOCKED
            door['last_action'] = 'unlock'
            door['last_action_time'] = datetime.now(timezone.utc)
            door['access_count'] += 1
            
            # Schedule automatic lock
            if door['auto_lock_enabled'] and self.config['auto_lock_enabled']:
                door['unlock_timer'] = asyncio.create_task(
                    self._auto_lock_door(location_id, duration)
                )
            
            # Log the action
            await self._log_door_event(
                location_id, 
                'DOOR_UNLOCKED', 
                f"Door unlocked for {duration} seconds"
            )
            
            logger.info(f"Door {location_id} unlocked for {duration} seconds")
            
            return {
                'success': True,
                'location_id': location_id,
                'action': 'unlock',
                'duration': duration,
                'auto_lock_scheduled': door['auto_lock_enabled'],
                'timestamp': door['last_action_time']
            }
            
        except Exception as e:
            logger.error(f"Error unlocking door {location_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def lock_door(self, location_id: int) -> Dict[str, Any]:
        """
        Lock door immediately.
        
        Args:
            location_id: ID of the location/door
            
        Returns:
            Dict containing operation result
        """
        if location_id not in self.doors:
            return {'success': False, 'error': f'Door not found for location {location_id}'}
        
        door = self.doors[location_id]
        
        try:
            # Cancel unlock timer if active
            if door['unlock_timer']:
                door['unlock_timer'].cancel()
                door['unlock_timer'] = None
            
            # Lock the door
            door['state'] = DoorState.LOCKED
            door['last_action'] = 'lock'
            door['last_action_time'] = datetime.now(timezone.utc)
            
            # Log the action
            await self._log_door_event(
                location_id, 
                'DOOR_LOCKED', 
                "Door locked manually"
            )
            
            logger.info(f"Door {location_id} locked")
            
            return {
                'success': True,
                'location_id': location_id,
                'action': 'lock',
                'timestamp': door['last_action_time']
            }
            
        except Exception as e:
            logger.error(f"Error locking door {location_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _auto_lock_door(self, location_id: int, delay: int):
        """Automatically lock door after delay."""
        try:
            await asyncio.sleep(delay)
            
            door = self.doors[location_id]
            
            # Only lock if still unlocked and not in emergency mode
            if (door['state'] == DoorState.UNLOCKED and 
                door['mode'] not in [DoorMode.EMERGENCY_OPEN, DoorMode.MAINTENANCE]):
                
                door['state'] = DoorState.LOCKED
                door['last_action'] = 'auto_lock'
                door['last_action_time'] = datetime.now(timezone.utc)
                door['unlock_timer'] = None
                
                await self._log_door_event(
                    location_id, 
                    'DOOR_AUTO_LOCKED', 
                    f"Door automatically locked after {delay} seconds"
                )
                
                logger.info(f"Door {location_id} automatically locked after {delay} seconds")
                
        except asyncio.CancelledError:
            logger.debug(f"Auto-lock cancelled for door {location_id}")
        except Exception as e:
            logger.error(f"Error in auto-lock for door {location_id}: {e}")
    
    async def set_door_mode(self, location_id: int, mode: str, reason: str = "") -> Dict[str, Any]:
        """
        Set door operation mode.
        
        Args:
            location_id: ID of the location/door
            mode: New door mode
            reason: Reason for mode change
            
        Returns:
            Dict containing operation result
        """
        if location_id not in self.doors:
            return {'success': False, 'error': f'Door not found for location {location_id}'}
        
        try:
            door_mode = DoorMode(mode)
        except ValueError:
            return {'success': False, 'error': f'Invalid door mode: {mode}'}
        
        door = self.doors[location_id]
        old_mode = door['mode']
        
        try:
            door['mode'] = door_mode
            
            # Handle mode-specific actions
            if door_mode == DoorMode.EMERGENCY_OPEN:
                # Unlock door and disable auto-lock
                door['state'] = DoorState.UNLOCKED
                if door['unlock_timer']:
                    door['unlock_timer'].cancel()
                    door['unlock_timer'] = None
                
            elif door_mode == DoorMode.EMERGENCY_LOCKED:
                # Lock door immediately
                door['state'] = DoorState.LOCKED
                if door['unlock_timer']:
                    door['unlock_timer'].cancel()
                    door['unlock_timer'] = None
                
            elif door_mode == DoorMode.MAINTENANCE:
                # Lock door and disable operations
                door['state'] = DoorState.LOCKED
                door['maintenance_mode'] = True
                if door['unlock_timer']:
                    door['unlock_timer'].cancel()
                    door['unlock_timer'] = None
                
            elif door_mode == DoorMode.NORMAL:
                # Return to normal operation
                door['maintenance_mode'] = False
                door['emergency_override'] = False
            
            # Log mode change
            await self._log_door_event(
                location_id, 
                'MODE_CHANGED', 
                f"Door mode changed from {old_mode.value} to {door_mode.value}: {reason}"
            )
            
            logger.info(f"Door {location_id} mode changed to {door_mode.value}: {reason}")
            
            return {
                'success': True,
                'location_id': location_id,
                'old_mode': old_mode.value,
                'new_mode': door_mode.value,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error setting door mode for {location_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_door_status(self, location_id: int) -> Dict[str, Any]:
        """Get detailed status of a specific door."""
        if location_id not in self.doors:
            return {'error': f'Door not found for location {location_id}'}
        
        door = self.doors[location_id]
        
        status = {
            'location_id': location_id,
            'name': door['name'],
            'state': door['state'].value,
            'mode': door['mode'].value,
            'last_action': door['last_action'],
            'last_action_time': door['last_action_time'],
            'auto_lock_enabled': door['auto_lock_enabled'],
            'auto_lock_active': door['unlock_timer'] is not None,
            'emergency_override': door['emergency_override'],
            'maintenance_mode': door['maintenance_mode'],
            'access_count': door['access_count'],
            'sensor_data': door['sensor_data'].copy(),
            'last_maintenance': door['last_maintenance'],
            'created_at': door['created_at'],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Add time remaining for auto-lock
        if door['unlock_timer'] and door['last_action_time']:
            elapsed = (datetime.now(timezone.utc) - door['last_action_time']).total_seconds()
            remaining = max(0, self.config['default_unlock_duration'] - elapsed)
            status['auto_lock_remaining'] = remaining
        
        return status
    
    def get_all_doors_status(self) -> Dict[str, Any]:
        """Get status of all doors."""
        status = {
            'total_doors': len(self.doors),
            'doors': {},
            'summary': {
                'locked': 0,
                'unlocked': 0,
                'error': 0,
                'maintenance': 0
            },
            'timestamp': datetime.now(timezone.utc)
        }
        
        for location_id, door in self.doors.items():
            door_status = {
                'state': door['state'].value,
                'mode': door['mode'].value,
                'last_action': door['last_action'],
                'last_action_time': door['last_action_time'],
                'access_count': door['access_count'],
                'battery_level': door['sensor_data']['battery_level']
            }
            
            status['doors'][location_id] = door_status
            
            # Update summary counts
            if door['state'] == DoorState.LOCKED:
                status['summary']['locked'] += 1
            elif door['state'] == DoorState.UNLOCKED:
                status['summary']['unlocked'] += 1
            elif door['state'] == DoorState.ERROR:
                status['summary']['error'] += 1
            elif door['mode'] == DoorMode.MAINTENANCE:
                status['summary']['maintenance'] += 1
        
        return status
    
    async def schedule_door_action(
        self, 
        location_id: int, 
        action_type: str, 
        scheduled_time: datetime,
        duration: int = None
    ) -> Dict[str, Any]:
        """Schedule a door action for future execution."""
        if location_id not in self.doors:
            return {'success': False, 'error': f'Door not found for location {location_id}'}
        
        if action_type not in ['lock', 'unlock']:
            return {'success': False, 'error': f'Invalid action type: {action_type}'}
        
        if scheduled_time <= datetime.now(timezone.utc):
            return {'success': False, 'error': 'Scheduled time must be in the future'}
        
        # Initialize schedule for location if not exists
        if location_id not in self.door_schedules:
            self.door_schedules[location_id] = {'actions': []}
        
        action = {
            'type': action_type,
            'time': scheduled_time,
            'duration': duration,
            'created_at': datetime.now(timezone.utc),
            'executed': False
        }
        
        self.door_schedules[location_id]['actions'].append(action)
        
        await self._log_door_event(
            location_id, 
            'ACTION_SCHEDULED', 
            f"Scheduled {action_type} for {scheduled_time}"
        )
        
        logger.info(f"Scheduled {action_type} for door {location_id} at {scheduled_time}")
        
        return {
            'success': True,
            'location_id': location_id,
            'action_type': action_type,
            'scheduled_time': scheduled_time,
            'duration': duration
        }
    
    async def _log_door_event(self, location_id: int, event_type: str, description: str):
        """Log door event to history."""
        event = {
            'timestamp': datetime.now(timezone.utc),
            'location_id': location_id,
            'event_type': event_type,
            'description': description
        }
        
        self.door_history.append(event)
        
        # Log to system logger based on event type
        if event_type in ['TAMPER_ALERT', 'DOOR_STUCK']:
            logger.warning(f"Door {location_id}: {description}")
        elif event_type in ['LOW_BATTERY']:
            logger.warning(f"Door {location_id}: {description}")
        else:
            logger.info(f"Door {location_id}: {description}")
    
    def get_door_history(
        self, 
        location_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered door history."""
        filtered_history = self.door_history
        
        if location_id:
            filtered_history = [
                event for event in filtered_history
                if event['location_id'] == location_id
            ]
        
        if start_date:
            filtered_history = [
                event for event in filtered_history
                if event['timestamp'] >= start_date
            ]
        
        if end_date:
            filtered_history = [
                event for event in filtered_history
                if event['timestamp'] <= end_date
            ]
        
        if event_type:
            filtered_history = [
                event for event in filtered_history
                if event['event_type'] == event_type
            ]
        
        return sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)
    
    def get_door_statistics(self) -> Dict[str, Any]:
        """Get door usage and performance statistics."""
        stats = {
            'total_doors': len(self.doors),
            'total_access_events': sum(door['access_count'] for door in self.doors.values()),
            'doors_by_state': {},
            'doors_by_mode': {},
            'battery_status': {},
            'maintenance_due': [],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Count doors by state and mode
        for door in self.doors.values():
            state = door['state'].value
            mode = door['mode'].value
            
            stats['doors_by_state'][state] = stats['doors_by_state'].get(state, 0) + 1
            stats['doors_by_mode'][mode] = stats['doors_by_mode'].get(mode, 0) + 1
            
            # Battery status
            battery_level = door['sensor_data']['battery_level']
            if battery_level < 20:
                level_category = 'critical'
            elif battery_level < 50:
                level_category = 'low'
            else:
                level_category = 'good'
            
            stats['battery_status'][level_category] = stats['battery_status'].get(level_category, 0) + 1
            
            # Check maintenance due
            days_since_maintenance = (datetime.now(timezone.utc) - door['last_maintenance']).days
            if days_since_maintenance > 90:  # 3 months
                stats['maintenance_due'].append({
                    'location_id': door['location_id'],
                    'days_since_maintenance': days_since_maintenance
                })
        
        return stats
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """Update door manager configuration."""
        for key, value in config_updates.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                logger.info(f"Door manager config updated: {key} = {value} (was {old_value})")
            else:
                logger.warning(f"Unknown door manager configuration key: {key}")
    
    async def emergency_lock_all(self, reason: str = "Emergency lockdown") -> Dict[str, Any]:
        """Lock all doors immediately for emergency."""
        results = {}
        
        for location_id in self.doors.keys():
            result = await self.set_door_mode(location_id, DoorMode.EMERGENCY_LOCKED.value, reason)
            results[location_id] = result
        
        logger.critical(f"Emergency lockdown activated: {reason}")
        
        return {
            'success': True,
            'action': 'emergency_lock_all',
            'reason': reason,
            'results': results,
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def emergency_unlock_all(self, reason: str = "Emergency unlock") -> Dict[str, Any]:
        """Unlock all doors immediately for emergency."""
        results = {}
        
        for location_id in self.doors.keys():
            result = await self.set_door_mode(location_id, DoorMode.EMERGENCY_OPEN.value, reason)
            results[location_id] = result
        
        logger.critical(f"Emergency unlock activated: {reason}")
        
        return {
            'success': True,
            'action': 'emergency_unlock_all',
            'reason': reason,
            'results': results,
            'timestamp': datetime.now(timezone.utc)
        }