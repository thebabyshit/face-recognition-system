"""Database backup and recovery utilities."""

import logging
import os
import subprocess
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import shutil
import tempfile

logger = logging.getLogger(__name__)

class DatabaseBackup:
    """Database backup and recovery manager."""
    
    def __init__(self):
        """Initialize database backup manager."""
        self.config = self._load_db_config()
        logger.info("Database backup manager initialized")
    
    def _load_db_config(self) -> Dict[str, Any]:
        """Load database configuration."""
        # In production, load from environment or config file
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'face_recognition'),
            'username': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'backup_format': 'custom',  # custom, plain, directory, tar
            'compression_level': 6,
            'include_blobs': True,
            'exclude_tables': ['temp_data', 'session_data']
        }
    
    async def backup_database(self, backup_dir: str, backup_type: str = "full") -> Dict[str, Any]:
        """
        Backup database to specified directory.
        
        Args:
            backup_dir: Directory to store backup
            backup_type: Type of backup (full, incremental, differential)
            
        Returns:
            Backup result information
        """
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"database_backup_{timestamp}.dump"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Prepare pg_dump command
            cmd = self._build_pg_dump_command(backup_path, backup_type)
            
            logger.info(f"Starting database backup: {backup_path}")
            
            # Execute backup command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_pg_env()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Get backup file size
                backup_size = os.path.getsize(backup_path) if os.path.exists(backup_path) else 0
                
                # Create backup metadata
                metadata = {
                    'backup_type': backup_type,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'database': self.config['database'],
                    'backup_file': backup_filename,
                    'size_bytes': backup_size,
                    'format': self.config['backup_format'],
                    'compression_level': self.config['compression_level'],
                    'pg_dump_version': await self._get_pg_dump_version()
                }
                
                # Save metadata
                metadata_path = os.path.join(backup_dir, f"database_backup_{timestamp}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Verify backup
                verification_result = await self._verify_backup(backup_path)
                
                result = {
                    'success': True,
                    'backup_path': backup_path,
                    'metadata_path': metadata_path,
                    'size_bytes': backup_size,
                    'file_count': 2,  # backup file + metadata
                    'verification': verification_result,
                    'duration_seconds': None  # Could be calculated if needed
                }
                
                logger.info(f"Database backup completed successfully: {backup_path}")
                return result
                
            else:
                error_msg = stderr.decode('utf-8') if stderr else 'Unknown error'
                logger.error(f"Database backup failed: {error_msg}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'size_bytes': 0,
                    'file_count': 0
                }
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'size_bytes': 0,
                'file_count': 0
            }
    
    def _build_pg_dump_command(self, backup_path: str, backup_type: str) -> List[str]:
        """Build pg_dump command based on configuration."""
        cmd = [
            'pg_dump',
            '-h', self.config['host'],
            '-p', str(self.config['port']),
            '-U', self.config['username'],
            '-d', self.config['database'],
            '-f', backup_path
        ]
        
        # Add format option
        if self.config['backup_format'] == 'custom':
            cmd.extend(['-F', 'c'])
        elif self.config['backup_format'] == 'tar':
            cmd.extend(['-F', 't'])
        elif self.config['backup_format'] == 'directory':
            cmd.extend(['-F', 'd'])
        else:
            cmd.extend(['-F', 'p'])  # plain text
        
        # Add compression
        if self.config['backup_format'] in ['custom', 'directory']:
            cmd.extend(['-Z', str(self.config['compression_level'])])
        
        # Include/exclude options
        if not self.config.get('include_blobs', True):
            cmd.append('--no-blobs')
        
        # Exclude specific tables
        for table in self.config.get('exclude_tables', []):
            cmd.extend(['--exclude-table', table])
        
        # Backup type specific options
        if backup_type == 'schema_only':
            cmd.append('--schema-only')
        elif backup_type == 'data_only':
            cmd.append('--data-only')
        
        # Additional options
        cmd.extend([
            '--verbose',
            '--no-password',
            '--create',
            '--clean'
        ])
        
        return cmd
    
    def _get_pg_env(self) -> Dict[str, str]:
        """Get environment variables for PostgreSQL commands."""
        env = os.environ.copy()
        if self.config['password']:
            env['PGPASSWORD'] = self.config['password']
        return env
    
    async def _get_pg_dump_version(self) -> str:
        """Get pg_dump version."""
        try:
            process = await asyncio.create_subprocess_exec(
                'pg_dump', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8').strip()
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    async def _verify_backup(self, backup_path: str) -> Dict[str, Any]:
        """Verify backup file integrity."""
        try:
            if not os.path.exists(backup_path):
                return {'valid': False, 'error': 'Backup file not found'}
            
            file_size = os.path.getsize(backup_path)
            if file_size == 0:
                return {'valid': False, 'error': 'Backup file is empty'}
            
            # For custom format, use pg_restore to verify
            if self.config['backup_format'] == 'custom':
                cmd = [
                    'pg_restore',
                    '--list',
                    backup_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    # Count objects in backup
                    objects_count = len([line for line in stdout.decode('utf-8').split('\n') if line.strip()])
                    return {
                        'valid': True,
                        'objects_count': objects_count,
                        'file_size': file_size
                    }
                else:
                    return {
                        'valid': False,
                        'error': stderr.decode('utf-8') if stderr else 'Verification failed'
                    }
            else:
                # For other formats, basic file checks
                return {
                    'valid': True,
                    'file_size': file_size
                }
                
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def restore_database(self, backup_path: str, target_database: str = None) -> Dict[str, Any]:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            target_database: Target database name (optional)
            
        Returns:
            Restore result information
        """
        try:
            if not os.path.exists(backup_path):
                return {'success': False, 'error': 'Backup file not found'}
            
            if target_database is None:
                target_database = self.config['database']
            
            logger.info(f"Starting database restore: {backup_path} -> {target_database}")
            
            # Determine restore command based on backup format
            if backup_path.endswith('.dump') or self.config['backup_format'] == 'custom':
                cmd = self._build_pg_restore_command(backup_path, target_database)
            else:
                cmd = self._build_psql_restore_command(backup_path, target_database)
            
            # Execute restore command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_pg_env()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Database restore completed successfully: {target_database}")
                return {
                    'success': True,
                    'target_database': target_database,
                    'restored_from': backup_path
                }
            else:
                error_msg = stderr.decode('utf-8') if stderr else 'Unknown error'
                logger.error(f"Database restore failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _build_pg_restore_command(self, backup_path: str, target_database: str) -> List[str]:
        """Build pg_restore command."""
        cmd = [
            'pg_restore',
            '-h', self.config['host'],
            '-p', str(self.config['port']),
            '-U', self.config['username'],
            '-d', target_database,
            '--verbose',
            '--no-password',
            '--clean',
            '--if-exists',
            backup_path
        ]
        
        return cmd
    
    def _build_psql_restore_command(self, backup_path: str, target_database: str) -> List[str]:
        """Build psql restore command for plain text backups."""
        cmd = [
            'psql',
            '-h', self.config['host'],
            '-p', str(self.config['port']),
            '-U', self.config['username'],
            '-d', target_database,
            '-f', backup_path
        ]
        
        return cmd
    
    async def create_incremental_backup(self, backup_dir: str, base_backup_path: str = None) -> Dict[str, Any]:
        """
        Create incremental backup using WAL files.
        
        Args:
            backup_dir: Directory to store backup
            base_backup_path: Path to base backup (for incremental)
            
        Returns:
            Backup result information
        """
        try:
            # This is a simplified implementation
            # In production, use pg_basebackup with WAL archiving
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"incremental_backup_{timestamp}")
            
            os.makedirs(backup_path, exist_ok=True)
            
            # Use pg_basebackup for base backup
            cmd = [
                'pg_basebackup',
                '-h', self.config['host'],
                '-p', str(self.config['port']),
                '-U', self.config['username'],
                '-D', backup_path,
                '-F', 'tar',
                '-z',
                '-P',
                '-v'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_pg_env()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Calculate backup size
                backup_size = sum(
                    os.path.getsize(os.path.join(backup_path, f))
                    for f in os.listdir(backup_path)
                    if os.path.isfile(os.path.join(backup_path, f))
                )
                
                return {
                    'success': True,
                    'backup_path': backup_path,
                    'size_bytes': backup_size,
                    'file_count': len(os.listdir(backup_path))
                }
            else:
                error_msg = stderr.decode('utf-8') if stderr else 'Unknown error'
                return {
                    'success': False,
                    'error': error_msg,
                    'size_bytes': 0,
                    'file_count': 0
                }
                
        except Exception as e:
            logger.error(f"Incremental backup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'size_bytes': 0,
                'file_count': 0
            }
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information for backup planning."""
        try:
            # Connect to database and get info
            # This is a simplified implementation
            
            info = {
                'database_name': self.config['database'],
                'host': self.config['host'],
                'port': self.config['port'],
                'estimated_size': 0,
                'table_count': 0,
                'last_backup': None,
                'backup_recommended': True
            }
            
            # In production, query actual database for this information
            # SELECT pg_database_size('database_name');
            # SELECT count(*) FROM information_schema.tables;
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    def list_backups(self, backup_dir: str) -> List[Dict[str, Any]]:
        """List available database backups."""
        try:
            backups = []
            
            if not os.path.exists(backup_dir):
                return backups
            
            for filename in os.listdir(backup_dir):
                if filename.startswith('database_backup_') and filename.endswith('.dump'):
                    backup_path = os.path.join(backup_dir, filename)
                    metadata_path = os.path.join(backup_dir, filename.replace('.dump', '.json'))
                    
                    backup_info = {
                        'filename': filename,
                        'path': backup_path,
                        'size_bytes': os.path.getsize(backup_path),
                        'created_at': datetime.fromtimestamp(os.path.getctime(backup_path)).isoformat()
                    }
                    
                    # Load metadata if available
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            backup_info.update(metadata)
                        except Exception as e:
                            logger.warning(f"Failed to load backup metadata: {e}")
                    
                    backups.append(backup_info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []