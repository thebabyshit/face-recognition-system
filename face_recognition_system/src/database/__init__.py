"""Database package for face recognition system."""

from .connection import (
    DatabaseConnection, 
    DatabaseConfig, 
    init_database, 
    get_database_connection,
    get_session,
    close_database
)
from .models import (
    Base,
    Person,
    FaceFeature,
    AccessLocation,
    AccessPermission,
    AccessLog,
    SystemLog,
    FeatureIndex,
    UserSession,
    APIKey,
    AuditTrail
)
from .services import (
    DatabaseService,
    PersonService,
    FaceFeatureService,
    AccessLogService,
    get_database_service
)

__all__ = [
    # Connection
    'DatabaseConnection',
    'DatabaseConfig', 
    'init_database',
    'get_database_connection',
    'get_session',
    'close_database',
    
    # Models
    'Base',
    'Person',
    'FaceFeature',
    'AccessLocation',
    'AccessPermission',
    'AccessLog',
    'SystemLog',
    'FeatureIndex',
    'UserSession',
    'APIKey',
    'AuditTrail',
    
    # Services
    'DatabaseService',
    'PersonService',
    'FaceFeatureService',
    'AccessLogService',
    'get_database_service'
]