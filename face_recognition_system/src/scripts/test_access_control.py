#!/usr/bin/env python3
"""Test script for access control system."""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from access_control.access_controller import AccessController
    from access_control.hardware_interface import HardwareInterface
    from access_control.security_monitor import SecurityMonitor
    from access_control.door_manager import DoorManager
except ImportError:
    # For testing without full database setup
    print("Note: Running in mock mode without database dependencies")
    
    class MockAccessController:
        def __init__(self):
            self.emergency_mode = False
            self.config = {'default_door_open_time': 5, 'max_failed_attempts': 3}
            
        async def process_access_request(self, **kwargs):
            return {
                'result': 'granted' if kwargs.get('person_id') else 'denied',
                'person_name': f"Person {kwargs.get('person_id', 'Unknown')}",
                'door_open_duration': 5,
                'message': 'Mock access result'
            }
            
        async def enable_emergency_mode(self, user, reason):
            self.emergency_mode = True
            
        async def disable_emergency_mode(self, user):
            self.emergency_mode = False
            
        def get_system_status(self):
            return {
                'emergency_mode': self.emergency_mode,
                'configuration': self.config,
                'failed_attempts_count': 0,
                'locked_locations': []
            }
            
        def update_configuration(self, updates):
            self.config.update(updates)
    
    AccessController = MockAccessController
    
    from access_control.hardware_interface import HardwareInterface
    from access_control.security_monitor import SecurityMonitor
    from access_control.door_manager import DoorManager

async def test_hardware_interface():
    """Test hardware interface functionality."""
    print("\n=== Testing Hardware Interface ===")
    
    hardware = HardwareInterface()
    
    # Test door operations
    print("\n1. Testing door operations:")
    result = await hardware.open_door(1, duration=3)
    print(f"Open door 1: {result}")
    
    await asyncio.sleep(1)
    
    result = await hardware.lock_door(1)
    print(f"Lock door 1: {result}")
    
    # Test display operations
    print("\n2. Testing display operations:")
    result = await hardware.display_message(1, "Welcome!", duration=2, color="green")
    print(f"Display message: {result}")
    
    await asyncio.sleep(1)
    
    result = await hardware.display_message(1, "Access Denied", duration=2, color="red")
    print(f"Display denial: {result}")
    
    # Test alarm operations
    print("\n3. Testing alarm operations:")
    result = await hardware.sound_alarm(1, duration=1, volume=50)
    print(f"Sound alarm: {result}")
    
    await asyncio.sleep(2)
    
    # Test device status
    print("\n4. Testing device status:")
    status = hardware.get_device_status(1)
    print(f"Device status for location 1: {status}")
    
    all_status = hardware.get_all_devices_status()
    print(f"All devices status: {all_status}")
    
    # Test device testing
    print("\n5. Testing device functionality:")
    test_result = await hardware.test_device(1, 'door_lock')
    print(f"Door lock test: {test_result}")
    
    test_result = await hardware.test_device(1, 'display')
    print(f"Display test: {test_result}")

async def test_security_monitor():
    """Test security monitoring functionality."""
    print("\n=== Testing Security Monitor ===")
    
    monitor = SecurityMonitor()
    
    # Test normal access
    print("\n1. Testing normal access:")
    result = await monitor.check_security_threats(1, 1)
    print(f"Normal access check: {result}")
    
    # Test tailgating detection
    print("\n2. Testing tailgating detection:")
    # Simulate multiple people trying to enter quickly
    for i in range(3):
        result = await monitor.check_security_threats(i+1, 1)
        print(f"Access attempt {i+1}: {result}")
        await asyncio.sleep(0.5)
    
    # Test suspicious behavior
    print("\n3. Testing suspicious behavior detection:")
    # Simulate repeated attempts by same person
    for i in range(3):
        result = await monitor.check_security_threats(1, 2)
        print(f"Repeated attempt {i+1}: {result}")
        await asyncio.sleep(0.2)
    
    # Test manual security alert
    print("\n4. Testing manual security alert:")
    await monitor.send_security_alert(
        'TEST_ALERT', 
        'This is a test security alert',
        {'test': True}
    )
    
    # Get security statistics
    print("\n5. Security statistics:")
    stats = monitor.get_security_statistics()
    print(f"Security stats: {stats}")
    
    # Get active alerts
    alerts = monitor.get_active_alerts()
    print(f"Active alerts: {alerts}")

async def test_door_manager():
    """Test door management functionality."""
    print("\n=== Testing Door Manager ===")
    
    door_manager = DoorManager()
    
    # Test door operations
    print("\n1. Testing door operations:")
    result = await door_manager.open_door(1, duration=3)
    print(f"Open door: {result}")
    
    await asyncio.sleep(1)
    
    result = await door_manager.lock_door(1)
    print(f"Lock door: {result}")
    
    # Test door modes
    print("\n2. Testing door modes:")
    result = await door_manager.set_door_mode(1, 'maintenance', 'Testing maintenance mode')
    print(f"Set maintenance mode: {result}")
    
    result = await door_manager.set_door_mode(1, 'normal', 'Back to normal')
    print(f"Set normal mode: {result}")
    
    # Test door status
    print("\n3. Testing door status:")
    status = await door_manager.get_door_status(1)
    print(f"Door 1 status: {status}")
    
    all_status = door_manager.get_all_doors_status()
    print(f"All doors status: {all_status}")
    
    # Test emergency operations
    print("\n4. Testing emergency operations:")
    result = await door_manager.emergency_lock_all("Test emergency lockdown")
    print(f"Emergency lockdown: {result}")
    
    await asyncio.sleep(1)
    
    # Reset to normal
    for location_id in [1, 2, 3, 4, 5]:
        await door_manager.set_door_mode(location_id, 'normal', 'Reset after test')
    
    # Test statistics
    print("\n5. Door statistics:")
    stats = door_manager.get_door_statistics()
    print(f"Door stats: {stats}")

async def test_access_controller():
    """Test complete access control system."""
    print("\n=== Testing Access Controller ===")
    
    controller = AccessController()
    
    # Test successful access
    print("\n1. Testing successful access:")
    result = await controller.process_access_request(
        person_id=1,
        location_id=1,
        access_method="face_recognition",
        confidence_score=0.95
    )
    print(f"Successful access: {result}")
    
    await asyncio.sleep(2)
    
    # Test denied access - insufficient level
    print("\n2. Testing denied access (insufficient level):")
    result = await controller.process_access_request(
        person_id=1,
        location_id=2,  # Assume location 2 requires higher access level
        access_method="face_recognition",
        confidence_score=0.90
    )
    print(f"Denied access: {result}")
    
    # Test unrecognized person
    print("\n3. Testing unrecognized person:")
    result = await controller.process_access_request(
        person_id=None,
        location_id=1,
        access_method="face_recognition",
        confidence_score=0.30
    )
    print(f"Unrecognized person: {result}")
    
    # Test emergency mode
    print("\n4. Testing emergency mode:")
    await controller.enable_emergency_mode("test_user", "Testing emergency access")
    
    result = await controller.process_access_request(
        person_id=None,
        location_id=1,
        access_method="emergency_override"
    )
    print(f"Emergency access: {result}")
    
    await controller.disable_emergency_mode("test_user")
    
    # Test system status
    print("\n5. System status:")
    status = controller.get_system_status()
    print(f"System status: {status}")
    
    # Test configuration update
    print("\n6. Testing configuration update:")
    controller.update_configuration({
        'default_door_open_time': 10,
        'max_failed_attempts': 5
    })
    
    updated_status = controller.get_system_status()
    print(f"Updated configuration: {updated_status['configuration']}")

async def test_integration():
    """Test integration between all components."""
    print("\n=== Testing System Integration ===")
    
    controller = AccessController()
    
    # Simulate a complete access scenario
    print("\n1. Complete access scenario:")
    
    # Person approaches door
    print("Person approaches door...")
    
    # Face recognition occurs (simulated)
    person_id = 1
    confidence_score = 0.92
    
    # Process access request
    result = await controller.process_access_request(
        person_id=person_id,
        location_id=1,
        access_method="face_recognition",
        confidence_score=confidence_score,
        additional_data={
            'camera_id': 'cam_001',
            'image_quality': 'good',
            'lighting_conditions': 'normal'
        }
    )
    
    print(f"Access result: {result}")
    
    if result['result'] == 'granted':
        print("✓ Access granted - door should be open")
        print(f"✓ Welcome message displayed for {result.get('person_name', 'User')}")
        print(f"✓ Door will auto-lock in {result.get('door_open_duration', 5)} seconds")
    else:
        print("✗ Access denied")
        print(f"✗ Reason: {result.get('message', 'Unknown')}")
    
    # Wait for door to auto-lock
    if result['result'] == 'granted':
        door_duration = result.get('door_open_duration', 5)
        print(f"\nWaiting {door_duration} seconds for auto-lock...")
        await asyncio.sleep(door_duration + 1)
        print("✓ Door should now be locked automatically")
    
    # Test security incident
    print("\n2. Security incident simulation:")
    
    # Simulate multiple failed attempts
    for i in range(4):
        result = await controller.process_access_request(
            person_id=None,
            location_id=1,
            access_method="face_recognition",
            confidence_score=0.20
        )
        print(f"Failed attempt {i+1}: {result['result']} - {result.get('message', '')}")
        await asyncio.sleep(0.5)
    
    # Check system status after incidents
    print("\n3. System status after incidents:")
    status = controller.get_system_status()
    print(f"Failed attempts tracked: {status['failed_attempts_count']}")
    print(f"Locked locations: {status['locked_locations']}")

async def main():
    """Run all tests."""
    print("Starting Access Control System Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        await test_hardware_interface()
        await test_security_monitor()
        await test_door_manager()
        await test_access_controller()
        
        # Test integration
        await test_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)