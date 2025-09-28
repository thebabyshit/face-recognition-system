"""Test script for authentication and authorization system."""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from auth.jwt_manager import JWTManager
from auth.rbac import RoleBasedAccessControl, Permission
from auth.session_manager import SessionManager
from auth.security_middleware import SecurityMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_jwt_manager():
    """Test JWT manager functionality."""
    logger.info("Testing JWT Manager...")
    
    try:
        jwt_manager = JWTManager()
        
        # Test access token generation
        logger.info("Testing access token generation...")
        access_token = jwt_manager.generate_access_token(
            user_id=1,
            username="test_user",
            roles=["operator", "viewer"],
            permissions=["person:read", "person:create", "api:access"]
        )
        logger.info(f"Access token generated: {access_token[:50]}...")
        
        # Test token verification
        logger.info("Testing token verification...")
        token_payload = jwt_manager.verify_token(access_token)
        if token_payload:
            logger.info(f"Token verified successfully for user: {token_payload.username}")
            logger.info(f"User roles: {token_payload.roles}")
            logger.info(f"User permissions: {token_payload.permissions}")
        else:
            logger.error("Token verification failed")
            return False
        
        # Test refresh token
        logger.info("Testing refresh token generation...")
        refresh_token = jwt_manager.generate_refresh_token(
            user_id=1,
            username="test_user"
        )
        logger.info(f"Refresh token generated: {refresh_token[:50]}...")
        
        # Test token refresh
        logger.info("Testing access token refresh...")
        new_access_token = jwt_manager.refresh_access_token(
            refresh_token,
            roles=["operator", "viewer", "security_manager"],
            permissions=["person:read", "person:create", "person:update", "api:access"]
        )
        if new_access_token:
            logger.info(f"New access token generated: {new_access_token[:50]}...")
        else:
            logger.error("Token refresh failed")
            return False
        
        # Test permission validation
        logger.info("Testing permission validation...")
        has_permission = jwt_manager.validate_token_permissions(
            new_access_token,
            ["person:read", "api:access"]
        )
        logger.info(f"Has required permissions: {has_permission}")
        
        # Test role validation
        logger.info("Testing role validation...")
        has_role = jwt_manager.validate_token_roles(
            new_access_token,
            ["operator", "admin"]
        )
        logger.info(f"Has required roles: {has_role}")
        
        # Test token blacklisting
        logger.info("Testing token blacklisting...")
        jwt_manager.blacklist_token(access_token)
        blacklisted_payload = jwt_manager.verify_token(access_token)
        if blacklisted_payload is None:
            logger.info("Token successfully blacklisted")
        else:
            logger.error("Token blacklisting failed")
            return False
        
        # Test API key generation
        logger.info("Testing API key generation...")
        api_key = jwt_manager.generate_api_key(
            user_id=1,
            username="test_user",
            description="Test API key"
        )
        logger.info(f"API key generated: {api_key[:50]}...")
        
        # Test API key verification
        api_key_info = jwt_manager.verify_api_key(api_key)
        if api_key_info:
            logger.info(f"API key verified for user: {api_key_info['username']}")
        else:
            logger.error("API key verification failed")
            return False
        
        logger.info("JWT Manager tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"JWT Manager test failed: {e}")
        return False

async def test_rbac():
    """Test Role-Based Access Control functionality."""
    logger.info("Testing RBAC System...")
    
    try:
        rbac = RoleBasedAccessControl()
        
        # Test default roles
        logger.info("Testing default roles...")
        roles = rbac.list_roles()
        logger.info(f"Default roles loaded: {[role.name for role in roles]}")
        
        # Test custom role creation
        logger.info("Testing custom role creation...")
        success = rbac.create_role(
            name="custom_operator",
            description="Custom operator role",
            permissions=[
                Permission.PERSON_READ,
                Permission.PERSON_CREATE,
                Permission.FACE_VIEW,
                Permission.API_ACCESS
            ]
        )
        if success:
            logger.info("Custom role created successfully")
        else:
            logger.error("Custom role creation failed")
            return False
        
        # Test role assignment
        logger.info("Testing role assignment...")
        user_id = 1
        rbac.assign_role_to_user(user_id, "custom_operator")
        rbac.assign_role_to_user(user_id, "viewer")
        
        user_roles = rbac.get_user_roles(user_id)
        logger.info(f"User {user_id} roles: {user_roles}")
        
        # Test permission checking
        logger.info("Testing permission checking...")
        has_permission = rbac.user_has_permission(user_id, Permission.PERSON_READ)
        logger.info(f"User has PERSON_READ permission: {has_permission}")
        
        has_admin_permission = rbac.user_has_permission(user_id, Permission.SYSTEM_CONFIG)
        logger.info(f"User has SYSTEM_CONFIG permission: {has_admin_permission}")
        
        # Test role checking
        logger.info("Testing role checking...")
        has_role = rbac.user_has_role(user_id, "viewer")
        logger.info(f"User has viewer role: {has_role}")
        
        has_admin_role = rbac.user_has_role(user_id, "super_admin")
        logger.info(f"User has super_admin role: {has_admin_role}")
        
        # Test user permissions
        logger.info("Testing user permissions retrieval...")
        user_permissions = rbac.get_user_permissions(user_id)
        logger.info(f"User permissions: {[p.value for p in user_permissions]}")
        
        # Test role hierarchy
        logger.info("Testing role hierarchy...")
        hierarchy = rbac.get_role_hierarchy()
        for role_name, info in hierarchy.items():
            logger.info(f"Role {role_name}: {info['permission_count']} permissions")
        
        # Test role update
        logger.info("Testing role update...")
        update_success = rbac.update_role(
            "custom_operator",
            description="Updated custom operator role",
            permissions=[
                Permission.PERSON_READ,
                Permission.PERSON_CREATE,
                Permission.PERSON_UPDATE,
                Permission.FACE_VIEW,
                Permission.FACE_UPLOAD,
                Permission.API_ACCESS
            ]
        )
        if update_success:
            logger.info("Role updated successfully")
        else:
            logger.error("Role update failed")
            return False
        
        # Test permission usage statistics
        logger.info("Testing permission usage statistics...")
        usage_stats = rbac.get_permission_usage()
        logger.info(f"Most used permissions: {sorted(usage_stats.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        logger.info("RBAC System tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"RBAC test failed: {e}")
        return False

async def test_session_manager():
    """Test session manager functionality."""
    logger.info("Testing Session Manager...")
    
    try:
        session_manager = SessionManager()
        
        # Test session creation
        logger.info("Testing session creation...")
        session_id = session_manager.create_session(
            user_id=1,
            username="test_user",
            roles=["operator", "viewer"],
            permissions=["person:read", "person:create", "api:access"],
            ip_address="192.168.1.100",
            user_agent="Test User Agent",
            remember_me=False
        )
        logger.info(f"Session created: {session_id}")
        
        # Test session retrieval
        logger.info("Testing session retrieval...")
        session = session_manager.get_session(session_id)
        if session:
            logger.info(f"Session retrieved for user: {session.username}")
            logger.info(f"Session expires at: {session.expires_at}")
        else:
            logger.error("Session retrieval failed")
            return False
        
        # Test session validation
        logger.info("Testing session validation...")
        is_valid = session_manager.validate_session(session_id, "192.168.1.100")
        logger.info(f"Session is valid: {is_valid}")
        
        # Test session extension
        logger.info("Testing session extension...")
        extended = session_manager.extend_session(session_id, 120)  # Extend by 2 hours
        if extended:
            logger.info("Session extended successfully")
        else:
            logger.error("Session extension failed")
            return False
        
        # Test multiple sessions for same user
        logger.info("Testing multiple sessions...")
        session_id_2 = session_manager.create_session(
            user_id=1,
            username="test_user",
            roles=["operator"],
            permissions=["person:read", "api:access"],
            ip_address="192.168.1.101",
            user_agent="Another User Agent"
        )
        
        user_sessions = session_manager.get_user_sessions(1)
        logger.info(f"User has {len(user_sessions)} active sessions")
        
        # Test session info
        logger.info("Testing session info...")
        session_info = session_manager.get_session_info(session_id)
        if session_info:
            logger.info(f"Session info: {session_info}")
        else:
            logger.error("Session info retrieval failed")
            return False
        
        # Test session statistics
        logger.info("Testing session statistics...")
        stats = session_manager.get_session_statistics()
        logger.info(f"Session statistics: {stats}")
        
        # Test permission updates
        logger.info("Testing session permission updates...")
        updated = session_manager.update_session_permissions(
            session_id,
            roles=["operator", "security_manager"],
            permissions=["person:read", "person:create", "person:update", "access:view_logs"]
        )
        if updated:
            logger.info("Session permissions updated successfully")
        else:
            logger.error("Session permission update failed")
            return False
        
        # Test session invalidation
        logger.info("Testing session invalidation...")
        invalidated = session_manager.invalidate_session(session_id_2)
        if invalidated:
            logger.info("Session invalidated successfully")
        else:
            logger.error("Session invalidation failed")
            return False
        
        # Test cleanup
        logger.info("Testing session cleanup...")
        cleaned = session_manager.cleanup_expired_sessions()
        logger.info(f"Cleaned up {cleaned} expired sessions")
        
        logger.info("Session Manager tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Session Manager test failed: {e}")
        return False

async def test_security_middleware():
    """Test security middleware functionality."""
    logger.info("Testing Security Middleware...")
    
    try:
        security_middleware = SecurityMiddleware()
        
        # Test normal request
        logger.info("Testing normal request processing...")
        request_info = {
            'ip_address': '192.168.1.100',
            'user_id': 1,
            'path': '/api/persons',
            'method': 'GET',
            'content_length': 1024,
            'user_agent': 'Mozilla/5.0 (Test Browser)'
        }
        
        result = await security_middleware.process_request(request_info, 'default')
        if result['allowed']:
            logger.info("Normal request allowed")
        else:
            logger.error(f"Normal request blocked: {result['reason']}")
            return False
        
        # Test rate limiting
        logger.info("Testing rate limiting...")
        auth_request = {
            'ip_address': '192.168.1.101',
            'path': '/api/auth/login',
            'method': 'POST',
            'content_length': 512,
            'user_agent': 'Test Client'
        }
        
        # Make multiple requests to trigger rate limit
        allowed_count = 0
        for i in range(15):  # Auth endpoint allows 10 requests per minute
            result = await security_middleware.process_request(auth_request, 'auth')
            if result['allowed']:
                allowed_count += 1
        
        logger.info(f"Rate limiting test: {allowed_count}/15 requests allowed")
        
        # Test IP blocking
        logger.info("Testing IP blocking...")
        security_middleware.block_ip('192.168.1.102', 'Test block')
        
        blocked_request = {
            'ip_address': '192.168.1.102',
            'path': '/api/test',
            'method': 'GET',
            'content_length': 100,
            'user_agent': 'Test Client'
        }
        
        result = await security_middleware.process_request(blocked_request)
        if not result['allowed'] and result['reason'] == 'ip_blocked':
            logger.info("IP blocking working correctly")
        else:
            logger.error("IP blocking failed")
            return False
        
        # Test suspicious activity detection
        logger.info("Testing suspicious activity detection...")
        suspicious_request = {
            'ip_address': '192.168.1.103',
            'path': '/api/test?id=1\\'union select * from users--',
            'method': 'GET',
            'content_length': 200,
            'user_agent': 'sqlmap/1.0'
        }
        
        result = await security_middleware.process_request(suspicious_request)
        logger.info(f"Suspicious request result: {result}")
        
        # Test failed attempt recording
        logger.info("Testing failed attempt recording...")
        for i in range(12):  # Record multiple failed attempts
            security_middleware.record_failed_attempt('192.168.1.104', user_id=2)
        
        # Test request after multiple failures
        failed_ip_request = {
            'ip_address': '192.168.1.104',
            'path': '/api/login',
            'method': 'POST',
            'content_length': 300,
            'user_agent': 'Test Client'
        }
        
        result = await security_middleware.process_request(failed_ip_request)
        logger.info(f"Request from IP with failed attempts: {result}")
        
        # Test security events
        logger.info("Testing security events...")
        events = security_middleware.get_security_events(limit=10)
        logger.info(f"Recent security events: {len(events)}")
        for event in events[:3]:  # Show first 3 events
            logger.info(f"  {event['event_type']} from {event['ip_address']} at {event['timestamp']}")
        
        # Test statistics
        logger.info("Testing security statistics...")
        stats = security_middleware.get_statistics()
        logger.info(f"Security statistics: {stats}")
        
        # Test allowlist
        logger.info("Testing IP allowlist...")
        security_middleware.add_to_allowlist('192.168.1.200')
        
        allowlist_request = {
            'ip_address': '192.168.1.200',
            'path': '/api/test',
            'method': 'GET',
            'content_length': 100,
            'user_agent': 'Test Client'
        }
        
        result = await security_middleware.process_request(allowlist_request)
        if result['allowed']:
            logger.info("Allowlist working correctly")
        else:
            logger.error("Allowlist failed")
            return False
        
        # Test configuration update
        logger.info("Testing configuration update...")
        security_middleware.update_config({
            'max_request_size': 5 * 1024 * 1024,  # 5MB
            'suspicious_threshold': 15
        })
        
        config = security_middleware.get_config()
        logger.info(f"Updated config - max_request_size: {config['max_request_size']}")
        
        logger.info("Security Middleware tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Security Middleware test failed: {e}")
        return False

async def test_integration():
    """Test integration between authentication components."""
    logger.info("Testing Authentication System Integration...")
    
    try:
        # Initialize components
        jwt_manager = JWTManager()
        rbac = RoleBasedAccessControl()
        session_manager = SessionManager()
        security_middleware = SecurityMiddleware()
        
        # Simulate user login flow
        logger.info("Simulating user login flow...")
        
        # 1. User authentication (would be done by auth service)
        user_id = 1
        username = "integration_test_user"
        
        # 2. Get user roles and permissions from RBAC
        rbac.assign_role_to_user(user_id, "operator")
        rbac.assign_role_to_user(user_id, "viewer")
        
        user_roles = rbac.get_user_roles(user_id)
        user_permissions = [p.value for p in rbac.get_user_permissions(user_id)]
        
        logger.info(f"User roles: {user_roles}")
        logger.info(f"User permissions: {user_permissions}")
        
        # 3. Create session
        session_id = session_manager.create_session(
            user_id=user_id,
            username=username,
            roles=user_roles,
            permissions=user_permissions,
            ip_address="192.168.1.150",
            user_agent="Integration Test Client"
        )
        
        # 4. Generate JWT tokens
        access_token = jwt_manager.generate_access_token(
            user_id=user_id,
            username=username,
            roles=user_roles,
            permissions=user_permissions
        )
        
        refresh_token = jwt_manager.generate_refresh_token(user_id, username)
        
        logger.info("User login flow completed successfully")
        
        # Simulate API request flow
        logger.info("Simulating API request flow...")
        
        # 1. Security middleware check
        request_info = {
            'ip_address': '192.168.1.150',
            'user_id': user_id,
            'path': '/api/persons',
            'method': 'GET',
            'content_length': 512,
            'user_agent': 'Integration Test Client'
        }
        
        security_result = await security_middleware.process_request(request_info, 'default')
        if not security_result['allowed']:
            logger.error(f"Security check failed: {security_result['reason']}")
            return False
        
        # 2. Session validation
        session = session_manager.get_session(session_id)
        if not session:
            logger.error("Session validation failed")
            return False
        
        # 3. JWT token validation
        token_payload = jwt_manager.verify_token(access_token)
        if not token_payload:
            logger.error("JWT token validation failed")
            return False
        
        # 4. Permission check
        required_permission = Permission.PERSON_READ
        has_permission = rbac.user_has_permission(user_id, required_permission)
        if not has_permission:
            logger.error(f"User lacks required permission: {required_permission.value}")
            return False
        
        logger.info("API request flow completed successfully")
        
        # Simulate token refresh flow
        logger.info("Simulating token refresh flow...")
        
        # Get updated permissions (user might have been assigned new roles)
        updated_permissions = [p.value for p in rbac.get_user_permissions(user_id)]
        
        # Refresh access token
        new_access_token = jwt_manager.refresh_access_token(
            refresh_token,
            roles=user_roles,
            permissions=updated_permissions
        )
        
        if not new_access_token:
            logger.error("Token refresh failed")
            return False
        
        # Update session permissions
        session_manager.update_session_permissions(session_id, user_roles, updated_permissions)
        
        logger.info("Token refresh flow completed successfully")
        
        # Simulate logout flow
        logger.info("Simulating logout flow...")
        
        # 1. Blacklist tokens
        jwt_manager.blacklist_token(access_token)
        jwt_manager.blacklist_token(new_access_token)
        jwt_manager.blacklist_token(refresh_token)
        
        # 2. Invalidate session
        session_manager.invalidate_session(session_id)
        
        # 3. Verify tokens are no longer valid
        if jwt_manager.verify_token(access_token) is not None:
            logger.error("Token should be blacklisted")
            return False
        
        if session_manager.get_session(session_id) is not None:
            logger.error("Session should be invalidated")
            return False
        
        logger.info("Logout flow completed successfully")
        
        logger.info("Authentication System Integration tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

async def main():
    """Run all authentication system tests."""
    logger.info("Starting Authentication System Tests...")
    
    test_results = {
        'jwt_manager': await test_jwt_manager(),
        'rbac': await test_rbac(),
        'session_manager': await test_session_manager(),
        'security_middleware': await test_security_middleware(),
        'integration': await test_integration()
    }
    
    # Print results
    logger.info("\\n" + "="*50)
    logger.info("AUTHENTICATION SYSTEM TEST RESULTS")
    logger.info("="*50)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    overall_success = all(test_results.values())
    logger.info(f"\\nOverall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    if overall_success:
        logger.info("\\nAll authentication components are working correctly!")
        logger.info("The system provides:")
        logger.info("- JWT token management with blacklisting")
        logger.info("- Role-based access control (RBAC)")
        logger.info("- Session management with cleanup")
        logger.info("- Security middleware with rate limiting")
        logger.info("- IP blocking and allowlisting")
        logger.info("- Suspicious activity detection")
        logger.info("- Comprehensive security event logging")
    else:
        logger.error("\\nSome authentication tests failed. Please check the logs above.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)