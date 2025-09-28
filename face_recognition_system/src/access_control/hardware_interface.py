"""Hardware interface for access control system."""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HardwareType(Enum):
    """Hardware device types."""
    DOOR_LOCK = "door_lock"
    DISPLAY_SCREEN = "display_screen"
    ALARM_SYSTEM = "alarm_system"
    CAMERA = "camera"
    CARD_READER = "card_reader"
    BIOMETRIC_SCANNER = "biometric_scanner"

class HardwareStatus(Enum):
    """Hardware device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class HardwareInterface:
    """Interface for controlling access control hardware."""
    
    def __init__(self):
        """Initialize hardware interface."""
        self.devices = {}
        self.device_status = {}
        self.command_queue = asyncio.Queue()
        self.response_handlers = {}
        
        # Mock hardware configuration
        self.mock_mode = True  # Set to False for real hardware
        
        # Initialize mock devices
        self._initialize_mock_devices()
        
        # Start command processor
        asyncio.create_task(self._process_commands())
        
        logger.info("Hardware interface initialized")
    
    def _initialize_mock_devices(self):
        """Initialize mock hardware devices for testing."""
        mock_locations = [1, 2, 3, 4, 5]  # Mock location IDs
        
        for location_id in mock_locations:
            # Door lock
            self.devices[f"door_lock_{location_id}"] = {
                'type': HardwareType.DOOR_LOCK,
                'location_id': location_id,
                'model': 'MockLock-2000',
                'ip_address': f'192.168.1.{100 + location_id}',
                'port': 8080,
                'status': HardwareStatus.ONLINE,
                'last_command': None,
                'last_response': None
            }
            
            # Display screen
            self.devices[f"display_{location_id}"] = {
                'type': HardwareType.DISPLAY_SCREEN,
                'location_id': location_id,
                'model': 'MockDisplay-HD',
                'ip_address': f'192.168.1.{110 + location_id}',
                'port': 8081,
                'status': HardwareStatus.ONLINE,
                'current_message': 'Ready',
                'brightness': 80
            }
            
            # Alarm system
            self.devices[f"alarm_{location_id}"] = {
                'type': HardwareType.ALARM_SYSTEM,
                'location_id': location_id,
                'model': 'MockAlarm-Pro',
                'ip_address': f'192.168.1.{120 + location_id}',
                'port': 8082,
                'status': HardwareStatus.ONLINE,
                'alarm_active': False,
                'volume': 70
            }
        
        logger.info(f"Initialized {len(self.devices)} mock devices")
    
    async def _process_commands(self):
        """Process hardware commands from queue."""
        while True:
            try:
                command = await self.command_queue.get()
                await self._execute_command(command)
                self.command_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing hardware command: {e}")
                await asyncio.sleep(1)
    
    async def _execute_command(self, command: Dict[str, Any]):
        """Execute a hardware command."""
        device_id = command['device_id']
        action = command['action']
        parameters = command.get('parameters', {})
        
        if device_id not in self.devices:
            logger.error(f"Device not found: {device_id}")
            return {'success': False, 'error': 'Device not found'}
        
        device = self.devices[device_id]
        
        try:
            if self.mock_mode:
                result = await self._execute_mock_command(device, action, parameters)
            else:
                result = await self._execute_real_command(device, action, parameters)
            
            # Update device status
            device['last_command'] = {
                'action': action,
                'parameters': parameters,
                'timestamp': datetime.now(timezone.utc),
                'result': result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing command {action} on device {device_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_mock_command(self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]):
        """Execute mock command for testing."""
        device_type = device['type']
        
        if device_type == HardwareType.DOOR_LOCK:
            return await self._mock_door_lock_command(device, action, parameters)
        elif device_type == HardwareType.DISPLAY_SCREEN:
            return await self._mock_display_command(device, action, parameters)
        elif device_type == HardwareType.ALARM_SYSTEM:
            return await self._mock_alarm_command(device, action, parameters)
        else:
            return {'success': False, 'error': f'Unsupported device type: {device_type}'}
    
    async def _mock_door_lock_command(self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]):
        """Execute mock door lock command."""
        if action == 'unlock':
            duration = parameters.get('duration', 5)
            logger.info(f"Mock: Unlocking door at location {device['location_id']} for {duration} seconds")
            device['locked'] = False
            device['unlock_time'] = datetime.now(timezone.utc)
            
            # Schedule automatic lock
            asyncio.create_task(self._auto_lock_door(device, duration))
            
            return {
                'success': True,
                'action': 'unlock',
                'duration': duration,
                'message': f'Door unlocked for {duration} seconds'
            }
            
        elif action == 'lock':
            logger.info(f"Mock: Locking door at location {device['location_id']}")
            device['locked'] = True
            device['lock_time'] = datetime.now(timezone.utc)
            
            return {
                'success': True,
                'action': 'lock',
                'message': 'Door locked'
            }
            
        elif action == 'status':
            return {
                'success': True,
                'locked': device.get('locked', True),
                'last_unlock': device.get('unlock_time'),
                'last_lock': device.get('lock_time')
            }
            
        else:
            return {'success': False, 'error': f'Unknown door lock action: {action}'}
    
    async def _mock_display_command(self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]):
        """Execute mock display command."""
        if action == 'show_message':
            message = parameters.get('message', '')
            duration = parameters.get('duration', 3)
            color = parameters.get('color', 'white')
            
            logger.info(f"Mock: Displaying message '{message}' on screen at location {device['location_id']}")
            device['current_message'] = message
            device['message_color'] = color
            device['message_time'] = datetime.now(timezone.utc)
            
            # Schedule message clear
            asyncio.create_task(self._clear_message(device, duration))
            
            return {
                'success': True,
                'action': 'show_message',
                'message': message,
                'duration': duration,
                'color': color
            }
            
        elif action == 'clear':
            logger.info(f"Mock: Clearing display at location {device['location_id']}")
            device['current_message'] = 'Ready'
            device['message_color'] = 'white'
            
            return {
                'success': True,
                'action': 'clear',
                'message': 'Display cleared'
            }
            
        elif action == 'set_brightness':
            brightness = parameters.get('brightness', 80)
            logger.info(f"Mock: Setting brightness to {brightness}% at location {device['location_id']}")
            device['brightness'] = brightness
            
            return {
                'success': True,
                'action': 'set_brightness',
                'brightness': brightness
            }
            
        else:
            return {'success': False, 'error': f'Unknown display action: {action}'}
    
    async def _mock_alarm_command(self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]):
        """Execute mock alarm command."""
        if action == 'sound':
            duration = parameters.get('duration', 5)
            volume = parameters.get('volume', 70)
            
            logger.info(f"Mock: Sounding alarm at location {device['location_id']} for {duration} seconds")
            device['alarm_active'] = True
            device['volume'] = volume
            device['alarm_start'] = datetime.now(timezone.utc)
            
            # Schedule alarm stop
            asyncio.create_task(self._stop_alarm(device, duration))
            
            return {
                'success': True,
                'action': 'sound',
                'duration': duration,
                'volume': volume
            }
            
        elif action == 'stop':
            logger.info(f"Mock: Stopping alarm at location {device['location_id']}")
            device['alarm_active'] = False
            device['alarm_stop'] = datetime.now(timezone.utc)
            
            return {
                'success': True,
                'action': 'stop',
                'message': 'Alarm stopped'
            }
            
        elif action == 'status':
            return {
                'success': True,
                'active': device.get('alarm_active', False),
                'volume': device.get('volume', 70),
                'last_start': device.get('alarm_start'),
                'last_stop': device.get('alarm_stop')
            }
            
        else:
            return {'success': False, 'error': f'Unknown alarm action: {action}'}
    
    async def _execute_real_command(self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]):
        """Execute real hardware command (placeholder for actual implementation)."""
        # This would contain actual hardware communication code
        # For now, return mock response
        logger.warning("Real hardware mode not implemented, using mock response")
        return await self._execute_mock_command(device, action, parameters)
    
    async def _auto_lock_door(self, device: Dict[str, Any], delay: int):
        """Automatically lock door after delay."""
        await asyncio.sleep(delay)
        device['locked'] = True
        device['auto_lock_time'] = datetime.now(timezone.utc)
        logger.info(f"Mock: Auto-locked door at location {device['location_id']}")
    
    async def _clear_message(self, device: Dict[str, Any], delay: int):
        """Clear display message after delay."""
        await asyncio.sleep(delay)
        device['current_message'] = 'Ready'
        device['message_color'] = 'white'
        logger.info(f"Mock: Cleared message on display at location {device['location_id']}")
    
    async def _stop_alarm(self, device: Dict[str, Any], delay: int):
        """Stop alarm after delay."""
        await asyncio.sleep(delay)
        device['alarm_active'] = False
        device['alarm_stop'] = datetime.now(timezone.utc)
        logger.info(f"Mock: Auto-stopped alarm at location {device['location_id']}")
    
    # Public interface methods
    
    async def unlock_door(self, location_id: int, duration: int = 5) -> Dict[str, Any]:
        """Unlock door at specified location."""
        device_id = f"door_lock_{location_id}"
        command = {
            'device_id': device_id,
            'action': 'unlock',
            'parameters': {'duration': duration}
        }
        
        await self.command_queue.put(command)
        return {'success': True, 'message': 'Unlock command queued'}
    
    async def lock_door(self, location_id: int) -> Dict[str, Any]:
        """Lock door at specified location."""
        device_id = f"door_lock_{location_id}"
        command = {
            'device_id': device_id,
            'action': 'lock',
            'parameters': {}
        }
        
        await self.command_queue.put(command)
        return {'success': True, 'message': 'Lock command queued'}
    
    async def display_message(self, location_id: int, message: str, duration: int = 3, color: str = 'white') -> Dict[str, Any]:
        """Display message on screen at specified location."""
        device_id = f"display_{location_id}"
        command = {
            'device_id': device_id,
            'action': 'show_message',
            'parameters': {
                'message': message,
                'duration': duration,
                'color': color
            }
        }
        
        await self.command_queue.put(command)
        return {'success': True, 'message': 'Display command queued'}
    
    async def sound_alarm(self, location_id: int, duration: int = 5, volume: int = 70) -> Dict[str, Any]:
        """Sound alarm at specified location."""
        device_id = f"alarm_{location_id}"
        command = {
            'device_id': device_id,
            'action': 'sound',
            'parameters': {
                'duration': duration,
                'volume': volume
            }
        }
        
        await self.command_queue.put(command)
        return {'success': True, 'message': 'Alarm command queued'}
    
    async def stop_alarm(self, location_id: int) -> Dict[str, Any]:
        """Stop alarm at specified location."""
        device_id = f"alarm_{location_id}"
        command = {
            'device_id': device_id,
            'action': 'stop',
            'parameters': {}
        }
        
        await self.command_queue.put(command)
        return {'success': True, 'message': 'Stop alarm command queued'}
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific device."""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all devices."""
        return self.devices.copy()
    
    def get_devices_by_location(self, location_id: int) -> List[Dict[str, Any]]:
        """Get all devices at specified location."""
        location_devices = []
        for device_id, device in self.devices.items():
            if device.get('location_id') == location_id:
                device_info = device.copy()
                device_info['device_id'] = device_id
                location_devices.append(device_info)
        
        return location_devices
    
    def get_devices_by_type(self, device_type: HardwareType) -> List[Dict[str, Any]]:
        """Get all devices of specified type."""
        type_devices = []
        for device_id, device in self.devices.items():
            if device.get('type') == device_type:
                device_info = device.copy()
                device_info['device_id'] = device_id
                type_devices.append(device_info)
        
        return type_devices
    
    async def test_device(self, device_id: str) -> Dict[str, Any]:
        """Test device connectivity and functionality."""
        if device_id not in self.devices:
            return {'success': False, 'error': 'Device not found'}
        
        device = self.devices[device_id]
        device_type = device['type']
        
        try:
            if device_type == HardwareType.DOOR_LOCK:
                # Test door lock
                result = await self._execute_mock_command(device, 'status', {})
            elif device_type == HardwareType.DISPLAY_SCREEN:
                # Test display
                result = await self._execute_mock_command(device, 'show_message', {
                    'message': 'Test Message',
                    'duration': 1
                })
            elif device_type == HardwareType.ALARM_SYSTEM:
                # Test alarm
                result = await self._execute_mock_command(device, 'status', {})
            else:
                result = {'success': False, 'error': 'Device type not testable'}
            
            if result.get('success'):
                device['last_test'] = datetime.now(timezone.utc)
                device['test_result'] = 'passed'
            else:
                device['test_result'] = 'failed'
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing device {device_id}: {e}")
            device['test_result'] = 'error'
            return {'success': False, 'error': str(e)}
    
    def set_mock_mode(self, enabled: bool):
        """Enable or disable mock mode."""
        self.mock_mode = enabled
        logger.info(f"Mock mode {'enabled' if enabled else 'disabled'}")
    
    def add_device(self, device_id: str, device_config: Dict[str, Any]) -> bool:
        """Add new device to the system."""
        try:
            if device_id in self.devices:
                logger.warning(f"Device {device_id} already exists")
                return False
            
            # Validate device configuration
            required_fields = ['type', 'location_id']
            for field in required_fields:
                if field not in device_config:
                    logger.error(f"Missing required field {field} in device config")
                    return False
            
            self.devices[device_id] = device_config
            logger.info(f"Added device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding device {device_id}: {e}")
            return False
    
    def remove_device(self, device_id: str) -> bool:
        """Remove device from the system."""
        try:
            if device_id not in self.devices:
                logger.warning(f"Device {device_id} not found")
                return False
            
            del self.devices[device_id]
            logger.info(f"Removed device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing device {device_id}: {e}")
            return False
        mock_locations = [1, 2, 3, 4, 5]  # Mock location IDs
        
        for location_id in mock_locations:
            # Door lock
            self.devices[f"door_lock_{location_id}"] = {
                'type': HardwareType.DOOR_LOCK,
                'location_id': location_id,
                'model': 'MockLock-2000',
                'ip_address': f'192.168.1.{100 + location_id}',
                'port': 8080,
                'status': HardwareStatus.ONLINE,
                'last_command': None,
                'last_response': None,
                'locked': True
            }
            
            # Display screen
            self.devices[f"display_{location_id}"] = {
                'type': HardwareType.DISPLAY_SCREEN,
                'location_id': location_id,
                'model': 'MockDisplay-HD',
                'ip_address': f'192.168.1.{110 + location_id}',
                'port': 8081,
                'status': HardwareStatus.ONLINE,
                'current_message': 'Ready',
                'brightness': 80
            }
            
            # Alarm system
            self.devices[f"alarm_{location_id}"] = {
                'type': HardwareType.ALARM_SYSTEM,
                'location_id': location_id,
                'model': 'MockAlarm-Pro',
                'ip_address': f'192.168.1.{120 + location_id}',
                'port': 8082,
                'status': HardwareStatus.ONLINE,
                'alarm_active': False,
                'volume': 70
            }
        
        logger.info(f"Initialized {len(self.devices)} mock devices")
    
    async def _process_commands(self):
        """Process hardware commands from queue."""
        while True:
            try:
                command = await self.command_queue.get()
                await self._execute_command(command)
                self.command_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing hardware command: {e}")
                await asyncio.sleep(1)
    
    async def _execute_command(self, command: Dict[str, Any]):
        """Execute a hardware command."""
        device_id = command['device_id']
        action = command['action']
        parameters = command.get('parameters', {})
        
        if device_id not in self.devices:
            logger.error(f"Device not found: {device_id}")
            return {'success': False, 'error': 'Device not found'}
        
        device = self.devices[device_id]
        
        try:
            if self.mock_mode:
                result = await self._execute_mock_command(device, action, parameters)
            else:
                result = await self._execute_real_command(device, action, parameters)
            
            # Update device status
            device['last_command'] = command
            device['last_response'] = result
            device['last_update'] = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing command for device {device_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_mock_command(
        self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mock hardware command for testing."""
        device_type = device['type']
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        if device_type == HardwareType.DOOR_LOCK:
            return await self._mock_door_lock_command(device, action, parameters)
        elif device_type == HardwareType.DISPLAY_SCREEN:
            return await self._mock_display_command(device, action, parameters)
        elif device_type == HardwareType.ALARM_SYSTEM:
            return await self._mock_alarm_command(device, action, parameters)
        else:
            return {'success': False, 'error': f'Unsupported device type: {device_type}'}
    
    async def _mock_door_lock_command(
        self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mock door lock commands."""
        if action == 'unlock':
            duration = parameters.get('duration', 5)
            device['status'] = HardwareStatus.ONLINE
            device['locked'] = False
            logger.info(f"Mock door {device['location_id']} unlocked for {duration} seconds")
            
            # Schedule automatic lock
            asyncio.create_task(self._mock_auto_lock(device, duration))
            
            return {
                'success': True,
                'action': 'unlock',
                'duration': duration,
                'timestamp': datetime.now(timezone.utc)
            }
            
        elif action == 'lock':
            device['locked'] = True
            logger.info(f"Mock door {device['location_id']} locked")
            
            return {
                'success': True,
                'action': 'lock',
                'timestamp': datetime.now(timezone.utc)
            }
            
        elif action == 'status':
            return {
                'success': True,
                'locked': device.get('locked', True),
                'status': device['status'].value,
                'timestamp': datetime.now(timezone.utc)
            }
            
        else:
            return {'success': False, 'error': f'Unknown door lock action: {action}'}
    
    async def _mock_display_command(
        self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mock display screen commands."""
        if action == 'show_message':
            message = parameters.get('message', '')
            duration = parameters.get('duration', 3)
            color = parameters.get('color', 'white')
            
            device['current_message'] = message
            device['message_color'] = color
            
            logger.info(f"Mock display {device['location_id']} showing: '{message}' ({color}) for {duration}s")
            
            # Schedule message clear
            asyncio.create_task(self._mock_clear_message(device, duration))
            
            return {
                'success': True,
                'message': message,
                'duration': duration,
                'color': color,
                'timestamp': datetime.now(timezone.utc)
            }
            
        elif action == 'clear':
            device['current_message'] = 'Ready'
            device['message_color'] = 'white'
            
            return {
                'success': True,
                'action': 'clear',
                'timestamp': datetime.now(timezone.utc)
            }
            
        elif action == 'set_brightness':
            brightness = parameters.get('brightness', 80)
            device['brightness'] = max(0, min(100, brightness))
            
            return {
                'success': True,
                'brightness': device['brightness'],
                'timestamp': datetime.now(timezone.utc)
            }
            
        else:
            return {'success': False, 'error': f'Unknown display action: {action}'}
    
    async def _mock_alarm_command(
        self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mock alarm system commands."""
        if action == 'sound':
            duration = parameters.get('duration', 5)
            volume = parameters.get('volume', 70)
            
            device['alarm_active'] = True
            device['volume'] = volume
            
            logger.warning(f"Mock alarm {device['location_id']} sounding for {duration}s at volume {volume}")
            
            # Schedule alarm stop
            asyncio.create_task(self._mock_stop_alarm(device, duration))
            
            return {
                'success': True,
                'duration': duration,
                'volume': volume,
                'timestamp': datetime.now(timezone.utc)
            }
            
        elif action == 'stop':
            device['alarm_active'] = False
            
            return {
                'success': True,
                'action': 'stop',
                'timestamp': datetime.now(timezone.utc)
            }
            
        elif action == 'status':
            return {
                'success': True,
                'active': device.get('alarm_active', False),
                'volume': device.get('volume', 70),
                'timestamp': datetime.now(timezone.utc)
            }
            
        else:
            return {'success': False, 'error': f'Unknown alarm action: {action}'}
    
    async def _mock_auto_lock(self, device: Dict[str, Any], delay: int):
        """Automatically lock door after delay."""
        await asyncio.sleep(delay)
        device['locked'] = True
        logger.info(f"Mock door {device['location_id']} automatically locked")
    
    async def _mock_clear_message(self, device: Dict[str, Any], delay: int):
        """Clear display message after delay."""
        await asyncio.sleep(delay)
        device['current_message'] = 'Ready'
        device['message_color'] = 'white'
    
    async def _mock_stop_alarm(self, device: Dict[str, Any], delay: int):
        """Stop alarm after delay."""
        await asyncio.sleep(delay)
        device['alarm_active'] = False
    
    async def _execute_real_command(
        self, device: Dict[str, Any], action: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute real hardware command (placeholder for actual implementation)."""
        # This would contain actual hardware communication code
        # For now, return mock response
        logger.warning("Real hardware mode not implemented, using mock response")
        return await self._execute_mock_command(device, action, parameters)
    
    # Public interface methods
    
    async def open_door(self, location_id: int, duration: int = 5) -> Dict[str, Any]:
        """Open door at specified location."""
        device_id = f"door_lock_{location_id}"
        
        if device_id not in self.devices:
            return {'success': False, 'error': f'Door lock not found for location {location_id}'}
        
        command = {
            'device_id': device_id,
            'action': 'unlock',
            'parameters': {'duration': duration}
        }
        
        await self.command_queue.put(command)
        
        # Wait for command execution (simplified)
        await asyncio.sleep(0.2)
        
        device = self.devices[device_id]
        if device.get('last_response', {}).get('success'):
            return device['last_response']
        else:
            return {'success': False, 'error': 'Door unlock failed'}
    
    async def lock_door(self, location_id: int) -> Dict[str, Any]:
        """Lock door at specified location."""
        device_id = f"door_lock_{location_id}"
        
        if device_id not in self.devices:
            return {'success': False, 'error': f'Door lock not found for location {location_id}'}
        
        command = {
            'device_id': device_id,
            'action': 'lock',
            'parameters': {}
        }
        
        await self.command_queue.put(command)
        
        # Wait for command execution
        await asyncio.sleep(0.2)
        
        device = self.devices[device_id]
        if device.get('last_response', {}).get('success'):
            return device['last_response']
        else:
            return {'success': False, 'error': 'Door lock failed'}
    
    async def display_message(
        self, 
        location_id: int, 
        message: str, 
        duration: int = 3,
        color: str = "white"
    ) -> Dict[str, Any]:
        """Display message on screen at specified location."""
        device_id = f"display_{location_id}"
        
        if device_id not in self.devices:
            return {'success': False, 'error': f'Display not found for location {location_id}'}
        
        command = {
            'device_id': device_id,
            'action': 'show_message',
            'parameters': {
                'message': message,
                'duration': duration,
                'color': color
            }
        }
        
        await self.command_queue.put(command)
        
        # Wait for command execution
        await asyncio.sleep(0.1)
        
        device = self.devices[device_id]
        if device.get('last_response', {}).get('success'):
            return device['last_response']
        else:
            return {'success': False, 'error': 'Display message failed'}
    
    async def sound_alarm(
        self, 
        location_id: int, 
        duration: int = 5,
        volume: int = 70
    ) -> Dict[str, Any]:
        """Sound alarm at specified location."""
        device_id = f"alarm_{location_id}"
        
        if device_id not in self.devices:
            return {'success': False, 'error': f'Alarm not found for location {location_id}'}
        
        command = {
            'device_id': device_id,
            'action': 'sound',
            'parameters': {
                'duration': duration,
                'volume': volume
            }
        }
        
        await self.command_queue.put(command)
        
        # Wait for command execution
        await asyncio.sleep(0.1)
        
        device = self.devices[device_id]
        if device.get('last_response', {}).get('success'):
            return device['last_response']
        else:
            return {'success': False, 'error': 'Alarm activation failed'}
    
    def get_device_status(self, location_id: int) -> Dict[str, Any]:
        """Get status of all devices at specified location."""
        location_devices = {
            device_id: device for device_id, device in self.devices.items()
            if device['location_id'] == location_id
        }
        
        if not location_devices:
            return {'error': f'No devices found for location {location_id}'}
        
        status = {
            'location_id': location_id,
            'devices': {},
            'timestamp': datetime.now(timezone.utc)
        }
        
        for device_id, device in location_devices.items():
            device_type = device['type'].value
            status['devices'][device_type] = {
                'status': device['status'].value,
                'model': device.get('model'),
                'ip_address': device.get('ip_address'),
                'last_update': device.get('last_update')
            }
            
            # Add device-specific status
            if device['type'] == HardwareType.DOOR_LOCK:
                status['devices'][device_type]['locked'] = device.get('locked', True)
            elif device['type'] == HardwareType.DISPLAY_SCREEN:
                status['devices'][device_type]['current_message'] = device.get('current_message', 'Ready')
                status['devices'][device_type]['brightness'] = device.get('brightness', 80)
            elif device['type'] == HardwareType.ALARM_SYSTEM:
                status['devices'][device_type]['alarm_active'] = device.get('alarm_active', False)
                status['devices'][device_type]['volume'] = device.get('volume', 70)
        
        return status  
  
    def get_all_devices_status(self) -> Dict[str, Any]:
        """Get status of all devices in the system."""
        status = {
            'total_devices': len(self.devices),
            'online_devices': 0,
            'offline_devices': 0,
            'error_devices': 0,
            'locations': {},
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Group devices by location
        for device_id, device in self.devices.items():
            location_id = device['location_id']
            device_status = device['status']
            
            # Count device status
            if device_status == HardwareStatus.ONLINE:
                status['online_devices'] += 1
            elif device_status == HardwareStatus.OFFLINE:
                status['offline_devices'] += 1
            elif device_status == HardwareStatus.ERROR:
                status['error_devices'] += 1
            
            # Group by location
            if location_id not in status['locations']:
                status['locations'][location_id] = {
                    'devices': {},
                    'total': 0,
                    'online': 0,
                    'offline': 0,
                    'error': 0
                }
            
            location_status = status['locations'][location_id]
            location_status['total'] += 1
            
            device_type = device['type'].value
            location_status['devices'][device_type] = {
                'status': device_status.value,
                'model': device.get('model'),
                'last_update': device.get('last_update')
            }
            
            if device_status == HardwareStatus.ONLINE:
                location_status['online'] += 1
            elif device_status == HardwareStatus.OFFLINE:
                location_status['offline'] += 1
            elif device_status == HardwareStatus.ERROR:
                location_status['error'] += 1
        
        return status
    
    async def test_device(self, location_id: int, device_type: str) -> Dict[str, Any]:
        """Test specific device functionality."""
        device_id = f"{device_type}_{location_id}"
        
        if device_id not in self.devices:
            return {'success': False, 'error': f'Device {device_type} not found for location {location_id}'}
        
        device = self.devices[device_id]
        
        try:
            if device_type == 'door_lock':
                # Test door lock
                unlock_result = await self.open_door(location_id, duration=1)
                if unlock_result['success']:
                    await asyncio.sleep(1.5)
                    lock_result = await self.lock_door(location_id)
                    return {
                        'success': lock_result['success'],
                        'test': 'door_lock_cycle',
                        'unlock_result': unlock_result,
                        'lock_result': lock_result
                    }
                else:
                    return {'success': False, 'error': 'Door unlock test failed'}
                    
            elif device_type == 'display':
                # Test display
                result = await self.display_message(location_id, "TEST MESSAGE", duration=2, color="blue")
                return {
                    'success': result['success'],
                    'test': 'display_message',
                    'result': result
                }
                
            elif device_type == 'alarm':
                # Test alarm
                result = await self.sound_alarm(location_id, duration=1, volume=50)
                return {
                    'success': result['success'],
                    'test': 'alarm_sound',
                    'result': result
                }
                
            else:
                return {'success': False, 'error': f'Unknown device type: {device_type}'}
                
        except Exception as e:
            logger.error(f"Error testing device {device_id}: {e}")
            return {'success': False, 'error': str(e)}