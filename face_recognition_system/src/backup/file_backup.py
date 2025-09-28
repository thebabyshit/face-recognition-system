"""File system backup utilities."""

import logging
import os
import shutil
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class FileBackup:
    """File system backup manager."""
    
    def __init__(self):
        """Initialize file backup manager."""
        self.config = self._load_config()
        logger.info("File backup manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load file backup configuration."""
        return {
            'backup_paths': [
                'data/face_images',
                'data/models',
                'logs',
                'config',
                'uploads'
            ],
            'exclude_patterns': [
                '*.tmp',
                '*.log',
                '__pycache__',
                '.git',
                'node_modules',
                '*.pyc'
            ],
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'follow_symlinks': False,
            'preserve_permissions': True
        }
    
    async def backup_files(self, backup_dir: str, backup_type: str = "full") -> Dict[str, Any]:
        """
        Backup files to specified directory.
        
        Args:
            backup_dir: Directory to store backup
            backup_type: Type of backup (full, incremental, differential)
            
        Returns:
            Backup result information
        """
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            total_size = 0
            total_files = 0
            backed_up_paths = []
            errors = []
            
            for backup_path in self.config['backup_paths']:
                if os.path.exists(backup_path):
                    result = await self._backup_path(backup_path, backup_dir, backup_type)
                    
                    if result['success']:
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                        backed_up_paths.append(backup_path)
                    else:
                        errors.append(f"Failed to backup {backup_path}: {result['error']}")
                else:
                    logger.warning(f"Backup path does not exist: {backup_path}")
            
            # Create backup manifest
            manifest = {
                'backup_type': backup_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'backed_up_paths': backed_up_paths,
                'total_files': total_files,
                'total_size_bytes': total_size,
                'errors': errors,
                'config': self.config
            }
            
            manifest_path = os.path.join(backup_dir, 'file_backup_manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return {
                'success': len(errors) == 0,
                'size_bytes': total_size,
                'file_count': total_files,
                'backed_up_paths': backed_up_paths,
                'errors': errors,
                'manifest_path': manifest_path
            }
            
        except Exception as e:
            logger.error(f"File backup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'size_bytes': 0,
                'file_count': 0
            }
    
    async def _backup_path(self, source_path: str, backup_dir: str, backup_type: str) -> Dict[str, Any]:
        """Backup a specific path."""
        try:
            dest_path = os.path.join(backup_dir, os.path.basename(source_path))
            
            if os.path.isfile(source_path):
                return await self._backup_file(source_path, dest_path)
            elif os.path.isdir(source_path):
                return await self._backup_directory(source_path, dest_path, backup_type)
            else:
                return {'success': False, 'error': 'Path is neither file nor directory'}
                
        except Exception as e:
            logger.error(f"Failed to backup path {source_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _backup_file(self, source_file: str, dest_file: str) -> Dict[str, Any]:
        """Backup a single file."""
        try:
            # Check file size
            file_size = os.path.getsize(source_file)
            if file_size > self.config['max_file_size']:
                return {
                    'success': False,
                    'error': f'File too large: {file_size} bytes'
                }
            
            # Check if file should be excluded
            if self._should_exclude_file(source_file):
                return {
                    'success': True,
                    'size_bytes': 0,
                    'file_count': 0,
                    'skipped': True
                }
            
            # Create destination directory
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            
            # Copy file
            shutil.copy2(source_file, dest_file)
            
            # Verify copy
            if os.path.exists(dest_file):
                dest_size = os.path.getsize(dest_file)
                if dest_size == file_size:
                    return {
                        'success': True,
                        'size_bytes': file_size,
                        'file_count': 1
                    }
                else:
                    return {
                        'success': False,
                        'error': 'File copy verification failed'
                    }
            else:
                return {
                    'success': False,
                    'error': 'Destination file not created'
                }
                
        except Exception as e:
            logger.error(f"Failed to backup file {source_file}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _backup_directory(self, source_dir: str, dest_dir: str, backup_type: str) -> Dict[str, Any]:
        """Backup a directory."""
        try:
            total_size = 0
            total_files = 0
            errors = []
            
            # Create destination directory
            os.makedirs(dest_dir, exist_ok=True)
            
            # Walk through source directory
            for root, dirs, files in os.walk(source_dir, followlinks=self.config['follow_symlinks']):
                # Filter directories
                dirs[:] = [d for d in dirs if not self._should_exclude_path(os.path.join(root, d))]
                
                for file in files:
                    source_file = os.path.join(root, file)
                    
                    if self._should_exclude_file(source_file):
                        continue
                    
                    # Calculate relative path
                    rel_path = os.path.relpath(source_file, source_dir)
                    dest_file = os.path.join(dest_dir, rel_path)
                    
                    # Backup file
                    result = await self._backup_file(source_file, dest_file)
                    
                    if result['success'] and not result.get('skipped', False):
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                    elif not result['success']:
                        errors.append(f"Failed to backup {source_file}: {result['error']}")
            
            return {
                'success': len(errors) == 0,
                'size_bytes': total_size,
                'file_count': total_files,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Failed to backup directory {source_dir}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _should_exclude_file(self, file_path: str) -> bool:
        """Check if file should be excluded from backup."""
        import fnmatch
        
        filename = os.path.basename(file_path)
        
        for pattern in self.config['exclude_patterns']:
            if fnmatch.fnmatch(filename, pattern):
                return True
            if fnmatch.fnmatch(file_path, pattern):
                return True
        
        return False
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from backup."""
        import fnmatch
        
        path_name = os.path.basename(path)
        
        for pattern in self.config['exclude_patterns']:
            if fnmatch.fnmatch(path_name, pattern):
                return True
        
        return False
    
    async def restore_files(self, backup_dir: str, restore_path: str = None) -> Dict[str, Any]:
        """
        Restore files from backup.
        
        Args:
            backup_dir: Directory containing backup
            restore_path: Path to restore to (optional)
            
        Returns:
            Restore result information
        """
        try:
            manifest_path = os.path.join(backup_dir, 'file_backup_manifest.json')
            
            if not os.path.exists(manifest_path):
                return {'success': False, 'error': 'Backup manifest not found'}
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            total_size = 0
            total_files = 0
            restored_paths = []
            errors = []
            
            # Restore each backed up path
            for backed_up_path in manifest['backed_up_paths']:
                backup_source = os.path.join(backup_dir, os.path.basename(backed_up_path))
                
                if restore_path:
                    restore_dest = os.path.join(restore_path, os.path.basename(backed_up_path))
                else:
                    restore_dest = backed_up_path
                
                if os.path.exists(backup_source):
                    result = await self._restore_path(backup_source, restore_dest)
                    
                    if result['success']:
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                        restored_paths.append(restore_dest)
                    else:
                        errors.append(f"Failed to restore {backed_up_path}: {result['error']}")
                else:
                    errors.append(f"Backup source not found: {backup_source}")
            
            return {
                'success': len(errors) == 0,
                'size_bytes': total_size,
                'file_count': total_files,
                'restored_paths': restored_paths,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"File restore failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _restore_path(self, backup_source: str, restore_dest: str) -> Dict[str, Any]:
        """Restore a specific path."""
        try:
            if os.path.isfile(backup_source):
                return await self._restore_file(backup_source, restore_dest)
            elif os.path.isdir(backup_source):
                return await self._restore_directory(backup_source, restore_dest)
            else:
                return {'success': False, 'error': 'Backup source is neither file nor directory'}
                
        except Exception as e:
            logger.error(f"Failed to restore path {backup_source}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _restore_file(self, backup_file: str, restore_file: str) -> Dict[str, Any]:
        """Restore a single file."""
        try:
            # Create destination directory
            os.makedirs(os.path.dirname(restore_file), exist_ok=True)
            
            # Copy file
            shutil.copy2(backup_file, restore_file)
            
            # Verify restore
            if os.path.exists(restore_file):
                file_size = os.path.getsize(restore_file)
                return {
                    'success': True,
                    'size_bytes': file_size,
                    'file_count': 1
                }
            else:
                return {
                    'success': False,
                    'error': 'Restored file not found'
                }
                
        except Exception as e:
            logger.error(f"Failed to restore file {backup_file}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _restore_directory(self, backup_dir: str, restore_dir: str) -> Dict[str, Any]:
        """Restore a directory."""
        try:
            total_size = 0
            total_files = 0
            errors = []
            
            # Create destination directory
            os.makedirs(restore_dir, exist_ok=True)
            
            # Walk through backup directory
            for root, dirs, files in os.walk(backup_dir):
                for file in files:
                    backup_file = os.path.join(root, file)
                    
                    # Calculate relative path
                    rel_path = os.path.relpath(backup_file, backup_dir)
                    restore_file = os.path.join(restore_dir, rel_path)
                    
                    # Restore file
                    result = await self._restore_file(backup_file, restore_file)
                    
                    if result['success']:
                        total_size += result['size_bytes']
                        total_files += result['file_count']
                    else:
                        errors.append(f"Failed to restore {backup_file}: {result['error']}")
            
            return {
                'success': len(errors) == 0,
                'size_bytes': total_size,
                'file_count': total_files,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Failed to restore directory {backup_dir}: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_backup_size(self) -> Dict[str, Any]:
        """Calculate estimated backup size."""
        try:
            total_size = 0
            total_files = 0
            path_sizes = {}
            
            for backup_path in self.config['backup_paths']:
                if os.path.exists(backup_path):
                    path_size, path_files = self._calculate_path_size(backup_path)
                    path_sizes[backup_path] = {
                        'size_bytes': path_size,
                        'file_count': path_files
                    }
                    total_size += path_size
                    total_files += path_files
            
            return {
                'total_size_bytes': total_size,
                'total_files': total_files,
                'path_breakdown': path_sizes,
                'estimated_compressed_size': int(total_size * 0.7)  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate backup size: {e}")
            return {}
    
    def _calculate_path_size(self, path: str) -> tuple[int, int]:
        """Calculate size and file count for a path."""
        total_size = 0
        total_files = 0
        
        try:
            if os.path.isfile(path):
                if not self._should_exclude_file(path):
                    return os.path.getsize(path), 1
                else:
                    return 0, 0
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path, followlinks=self.config['follow_symlinks']):
                    # Filter directories
                    dirs[:] = [d for d in dirs if not self._should_exclude_path(os.path.join(root, d))]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if not self._should_exclude_file(file_path):
                            try:
                                file_size = os.path.getsize(file_path)
                                if file_size <= self.config['max_file_size']:
                                    total_size += file_size
                                    total_files += 1
                            except (OSError, IOError):
                                # Skip files that can't be accessed
                                pass
        except Exception as e:
            logger.error(f"Error calculating size for {path}: {e}")
        
        return total_size, total_files