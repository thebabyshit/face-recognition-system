"""Database connection and configuration."""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import sqlite3

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration class."""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///face_recognition.db')
        self.echo = os.getenv('DB_ECHO', 'false').lower() == 'true'
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '5'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '10'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))
    
    def get_engine_kwargs(self) -> dict:
        """Get engine configuration kwargs."""
        kwargs = {
            'echo': self.echo,
        }
        
        # Add connection pool settings for non-SQLite databases
        if not self.database_url.startswith('sqlite'):
            kwargs.update({
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
            })
        else:
            # SQLite specific settings
            kwargs.update({
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                }
            })
        
        return kwargs


class DatabaseConnection:
    """Database connection manager."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    @property
    def engine(self) -> Engine:
        """Get database engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory
    
    def _create_engine(self) -> Engine:
        """Create database engine."""
        engine_kwargs = self.config.get_engine_kwargs()
        engine = create_engine(self.config.database_url, **engine_kwargs)
        
        # Add SQLite specific event listeners
        if self.config.database_url.startswith('sqlite'):
            self._setup_sqlite_events(engine)
        
        logger.info(f"Created database engine for: {self.config.database_url}")
        return engine
    
    def _setup_sqlite_events(self, engine: Engine):
        """Setup SQLite specific event listeners."""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance and integrity."""
            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=ON")
                # Set WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set synchronous mode for better performance
                cursor.execute("PRAGMA synchronous=NORMAL")
                # Set cache size (negative value means KB)
                cursor.execute("PRAGMA cache_size=-64000")  # 64MB
                # Set temp store to memory
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.session_factory()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            from sqlalchemy import text
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection closed")


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_database_connection(config: Optional[DatabaseConfig] = None) -> DatabaseConnection:
    """Get global database connection instance."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection(config)
    return _db_connection


def init_database(config: Optional[DatabaseConfig] = None, create_tables: bool = True) -> DatabaseConnection:
    """Initialize database connection and optionally create tables."""
    db_connection = get_database_connection(config)
    
    if create_tables:
        db_connection.create_tables()
    
    # Test connection
    if not db_connection.test_connection():
        raise RuntimeError("Failed to establish database connection")
    
    return db_connection


def get_session() -> Session:
    """Get a new database session from global connection."""
    return get_database_connection().get_session()


def close_database():
    """Close global database connection."""
    global _db_connection
    if _db_connection:
        _db_connection.close()
        _db_connection = None