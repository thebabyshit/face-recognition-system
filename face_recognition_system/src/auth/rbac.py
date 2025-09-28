"""Role-Based Access Control (RBAC) implementation."""

import logging
from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions."""
    # Person management
    PERSON_CREATE = "person:create"
    PERSON_READ = "person:read"
    PERSON_UPDATE = "person:update"
    PERSON_DELETE = "person:delete"
    PERSON_LIST = "person:list"
    
    # Face management
    FACE_UPLOAD = "face:upload"
    FACE_DELETE = "face:delete"
    FACE_VIEW = "face:view"
    
    # Access control
    ACCESS_GRANT = "access:grant"
    ACCESS_DENY = "access:deny"
    ACCESS_OVERRIDE = "access:override"
    ACCESS_VIEW_LOGS = "access:view_logs"
    
    # Location management
    LOCATION_CREATE = "location:create"
    LOCATION_UPDATE = "location:update"
    LOCATION_DELETE = "location:delete"
    LOCATION_VIEW = "location:view"
    
    # System administration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_BACKUP = "system:backup"
    SYSTEM_RESTORE = "system:restore"
    
    # User management
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_VIEW = "user:view"
    USER_LIST = "user:list"
    
    # Role management
    ROLE_CREATE = "role:create"
    ROLE_UPDATE = "role:update"
    ROLE_DELETE = "role:delete"
    ROLE_ASSIGN = "role:assign"
    
    # Reports and analytics
    REPORT_VIEW = "report:view"
    REPORT_GENERATE = "report:generate"
    REPORT_EXPORT = "report:export"
    
    # API access
    API_ACCESS = "api:access"
    API_ADMIN = "api:admin"
    
    # Emergency controls
    EMERGENCY_OVERRIDE = "emergency:override"
    EMERGENCY_LOCKDOWN = "emergency:lockdown"

@dataclass
class Role:
    """Role definition with permissions."""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    is_system_role: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        self.permissions.discard(permission)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission."""
        return permission in self.permissions
    
    def to_dict(self) -> Dict:
        """Convert role to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'permissions': [p.value for p in self.permissions],
            'is_system_role': self.is_system_role,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Role':
        """Create role from dictionary."""
        permissions = {Permission(p) for p in data.get('permissions', [])}
        return cls(
            name=data['name'],
            description=data['description'],
            permissions=permissions,
            is_system_role=data.get('is_system_role', False),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )

class RoleBasedAccessControl:
    """Role-Based Access Control system."""
    
    def __init__(self):
        """Initialize RBAC system."""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[int, Set[str]] = {}  # user_id -> role_names
        
        # Initialize default system roles
        self._initialize_default_roles()
        
        logger.info("RBAC system initialized")
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Super Admin - all permissions
        super_admin = Role(
            name="super_admin",
            description="Super Administrator with all permissions",
            permissions=set(Permission),
            is_system_role=True
        )
        
        # System Admin - system management permissions
        system_admin = Role(
            name="system_admin",
            description="System Administrator",
            permissions={
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_MONITOR,
                Permission.SYSTEM_BACKUP,
                Permission.SYSTEM_RESTORE,
                Permission.USER_CREATE,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.USER_VIEW,
                Permission.USER_LIST,
                Permission.ROLE_CREATE,
                Permission.ROLE_UPDATE,
                Permission.ROLE_DELETE,
                Permission.ROLE_ASSIGN,
                Permission.REPORT_VIEW,
                Permission.REPORT_GENERATE,
                Permission.REPORT_EXPORT,
                Permission.API_ADMIN,
                Permission.LOCATION_CREATE,
                Permission.LOCATION_UPDATE,
                Permission.LOCATION_DELETE,
                Permission.LOCATION_VIEW
            },
            is_system_role=True
        )
        
        # Security Manager - access control and monitoring
        security_manager = Role(
            name="security_manager",
            description="Security Manager",
            permissions={
                Permission.ACCESS_GRANT,
                Permission.ACCESS_DENY,
                Permission.ACCESS_OVERRIDE,
                Permission.ACCESS_VIEW_LOGS,
                Permission.PERSON_CREATE,
                Permission.PERSON_READ,
                Permission.PERSON_UPDATE,
                Permission.PERSON_DELETE,
                Permission.PERSON_LIST,
                Permission.FACE_UPLOAD,
                Permission.FACE_DELETE,
                Permission.FACE_VIEW,
                Permission.LOCATION_VIEW,
                Permission.REPORT_VIEW,
                Permission.REPORT_GENERATE,
                Permission.SYSTEM_MONITOR,
                Permission.EMERGENCY_OVERRIDE,
                Permission.EMERGENCY_LOCKDOWN,
                Permission.API_ACCESS
            },
            is_system_role=True
        )
        
        # Operator - basic operations
        operator = Role(
            name="operator",
            description="System Operator",
            permissions={
                Permission.PERSON_CREATE,
                Permission.PERSON_READ,
                Permission.PERSON_UPDATE,
                Permission.PERSON_LIST,
                Permission.FACE_UPLOAD,
                Permission.FACE_VIEW,
                Permission.ACCESS_VIEW_LOGS,
                Permission.LOCATION_VIEW,
                Permission.REPORT_VIEW,
                Permission.API_ACCESS
            },
            is_system_role=True
        )
        
        # Viewer - read-only access
        viewer = Role(
            name="viewer",
            description="Read-only Viewer",
            permissions={
                Permission.PERSON_READ,
                Permission.PERSON_LIST,
                Permission.FACE_VIEW,
                Permission.ACCESS_VIEW_LOGS,
                Permission.LOCATION_VIEW,
                Permission.REPORT_VIEW,
                Permission.SYSTEM_MONITOR,
                Permission.API_ACCESS
            },
            is_system_role=True
        )
        
        # Guest - minimal access
        guest = Role(
            name="guest",
            description="Guest User",
            permissions={
                Permission.PERSON_READ,
                Permission.LOCATION_VIEW,
                Permission.API_ACCESS
            },
            is_system_role=True
        )
        
        # Add roles to system
        for role in [super_admin, system_admin, security_manager, operator, viewer, guest]:
            self.roles[role.name] = role
        
        logger.info(f"Initialized {len(self.roles)} default roles")
    
    def create_role(self, name: str, description: str, permissions: List[Permission]) -> bool:
        """
        Create a new role.
        
        Args:
            name: Role name
            description: Role description
            permissions: List of permissions
            
        Returns:
            True if role created successfully, False otherwise
        """
        try:
            if name in self.roles:
                logger.warning(f"Role {name} already exists")
                return False
            
            role = Role(
                name=name,
                description=description,
                permissions=set(permissions),
                is_system_role=False
            )
            
            self.roles[name] = role
            logger.info(f"Role {name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating role {name}: {e}")
            return False
    
    def update_role(self, name: str, description: Optional[str] = None, 
                   permissions: Optional[List[Permission]] = None) -> bool:
        """
        Update an existing role.
        
        Args:
            name: Role name
            description: New description (optional)
            permissions: New permissions (optional)
            
        Returns:
            True if role updated successfully, False otherwise
        """
        try:
            if name not in self.roles:
                logger.warning(f"Role {name} does not exist")
                return False
            
            role = self.roles[name]
            
            # Don't allow modification of system roles
            if role.is_system_role:
                logger.warning(f"Cannot modify system role {name}")
                return False
            
            if description is not None:
                role.description = description
            
            if permissions is not None:
                role.permissions = set(permissions)
            
            logger.info(f"Role {name} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating role {name}: {e}")
            return False
    
    def delete_role(self, name: str) -> bool:
        """
        Delete a role.
        
        Args:
            name: Role name
            
        Returns:
            True if role deleted successfully, False otherwise
        """
        try:
            if name not in self.roles:
                logger.warning(f"Role {name} does not exist")
                return False
            
            role = self.roles[name]
            
            # Don't allow deletion of system roles
            if role.is_system_role:
                logger.warning(f"Cannot delete system role {name}")
                return False
            
            # Remove role from all users
            for user_id in list(self.user_roles.keys()):
                self.user_roles[user_id].discard(name)
            
            # Delete role
            del self.roles[name]
            
            logger.info(f"Role {name} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting role {name}: {e}")
            return False
    
    def assign_role_to_user(self, user_id: int, role_name: str) -> bool:
        """
        Assign role to user.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            True if role assigned successfully, False otherwise
        """
        try:
            if role_name not in self.roles:
                logger.warning(f"Role {role_name} does not exist")
                return False
            
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role_name)
            
            logger.info(f"Role {role_name} assigned to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning role {role_name} to user {user_id}: {e}")
            return False
    
    def remove_role_from_user(self, user_id: int, role_name: str) -> bool:
        """
        Remove role from user.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            True if role removed successfully, False otherwise
        """
        try:
            if user_id not in self.user_roles:
                logger.warning(f"User {user_id} has no roles assigned")
                return False
            
            self.user_roles[user_id].discard(role_name)
            
            # Clean up empty role sets
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]
            
            logger.info(f"Role {role_name} removed from user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing role {role_name} from user {user_id}: {e}")
            return False
    
    def get_user_roles(self, user_id: int) -> List[str]:
        """
        Get roles assigned to user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of role names
        """
        return list(self.user_roles.get(user_id, set()))
    
    def get_user_permissions(self, user_id: int) -> List[Permission]:
        """
        Get all permissions for user based on assigned roles.
        
        Args:
            user_id: User ID
            
        Returns:
            List of permissions
        """
        try:
            user_roles = self.user_roles.get(user_id, set())
            permissions = set()
            
            for role_name in user_roles:
                if role_name in self.roles:
                    permissions.update(self.roles[role_name].permissions)
            
            return list(permissions)
            
        except Exception as e:
            logger.error(f"Error getting user permissions for user {user_id}: {e}")
            return []
    
    def user_has_permission(self, user_id: int, permission: Permission) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            user_permissions = self.get_user_permissions(user_id)
            return permission in user_permissions
            
        except Exception as e:
            logger.error(f"Error checking permission for user {user_id}: {e}")
            return False
    
    def user_has_role(self, user_id: int, role_name: str) -> bool:
        """
        Check if user has specific role.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            True if user has role, False otherwise
        """
        try:
            user_roles = self.user_roles.get(user_id, set())
            return role_name in user_roles
            
        except Exception as e:
            logger.error(f"Error checking role for user {user_id}: {e}")
            return False
    
    def user_has_any_role(self, user_id: int, role_names: List[str]) -> bool:
        """
        Check if user has any of the specified roles.
        
        Args:
            user_id: User ID
            role_names: List of role names
            
        Returns:
            True if user has any of the roles, False otherwise
        """
        try:
            user_roles = self.user_roles.get(user_id, set())
            return bool(user_roles.intersection(set(role_names)))
            
        except Exception as e:
            logger.error(f"Error checking roles for user {user_id}: {e}")
            return False
    
    def get_role(self, name: str) -> Optional[Role]:
        """
        Get role by name.
        
        Args:
            name: Role name
            
        Returns:
            Role object if found, None otherwise
        """
        return self.roles.get(name)
    
    def list_roles(self, include_system_roles: bool = True) -> List[Role]:
        """
        List all roles.
        
        Args:
            include_system_roles: Whether to include system roles
            
        Returns:
            List of roles
        """
        if include_system_roles:
            return list(self.roles.values())
        else:
            return [role for role in self.roles.values() if not role.is_system_role]
    
    def get_role_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get role hierarchy based on permissions.
        
        Returns:
            Dictionary mapping role names to their permission counts
        """
        try:
            hierarchy = {}
            for role_name, role in self.roles.items():
                hierarchy[role_name] = {
                    'permission_count': len(role.permissions),
                    'permissions': [p.value for p in role.permissions],
                    'is_system_role': role.is_system_role
                }
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error getting role hierarchy: {e}")
            return {}
    
    def validate_permission_string(self, permission_str: str) -> bool:
        """
        Validate permission string.
        
        Args:
            permission_str: Permission string
            
        Returns:
            True if valid permission, False otherwise
        """
        try:
            Permission(permission_str)
            return True
        except ValueError:
            return False
    
    def export_roles(self) -> str:
        """
        Export roles to JSON string.
        
        Returns:
            JSON string containing all roles
        """
        try:
            roles_data = {}
            for name, role in self.roles.items():
                roles_data[name] = role.to_dict()
            
            return json.dumps(roles_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting roles: {e}")
            return "{}"
    
    def import_roles(self, roles_json: str) -> bool:
        """
        Import roles from JSON string.
        
        Args:
            roles_json: JSON string containing roles
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            roles_data = json.loads(roles_json)
            
            for name, role_dict in roles_data.items():
                # Don't overwrite system roles
                if name in self.roles and self.roles[name].is_system_role:
                    continue
                
                role = Role.from_dict(role_dict)
                self.roles[name] = role
            
            logger.info(f"Imported {len(roles_data)} roles")
            return True
            
        except Exception as e:
            logger.error(f"Error importing roles: {e}")
            return False
    
    def get_users_with_role(self, role_name: str) -> List[int]:
        """
        Get list of users with specific role.
        
        Args:
            role_name: Role name
            
        Returns:
            List of user IDs
        """
        try:
            users = []
            for user_id, roles in self.user_roles.items():
                if role_name in roles:
                    users.append(user_id)
            
            return users
            
        except Exception as e:
            logger.error(f"Error getting users with role {role_name}: {e}")
            return []
    
    def get_permission_usage(self) -> Dict[str, int]:
        """
        Get permission usage statistics.
        
        Returns:
            Dictionary mapping permissions to usage count
        """
        try:
            usage = {}
            
            for permission in Permission:
                count = 0
                for role in self.roles.values():
                    if permission in role.permissions:
                        count += 1
                usage[permission.value] = count
            
            return usage
            
        except Exception as e:
            logger.error(f"Error getting permission usage: {e}")
            return {}
    
    def cleanup_orphaned_roles(self):
        """Remove roles that are not assigned to any users."""
        try:
            assigned_roles = set()
            for roles in self.user_roles.values():
                assigned_roles.update(roles)
            
            orphaned_roles = []
            for role_name, role in self.roles.items():
                if not role.is_system_role and role_name not in assigned_roles:
                    orphaned_roles.append(role_name)
            
            for role_name in orphaned_roles:
                del self.roles[role_name]
                logger.info(f"Removed orphaned role: {role_name}")
            
            logger.info(f"Cleaned up {len(orphaned_roles)} orphaned roles")
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned roles: {e}")