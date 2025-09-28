"""Comprehensive backup management system."""

import logging
import os
import shutil
import json
import asyncio
import schedule
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import zipfile
import hashlib

from .database_backup import DatabaseBackup
from .file_backup import FileBackup
from .model_backup import ModelBackup

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BackupJob:
    """Backup job definition."""
    job_id: str
    name: str
    backup_type: BackupType
    components: List[str]  # database, files, models, config
    schedule_cron: Optional[str]
    retention_days: int
    compression: bool
    encryption: bool
    destination: str
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True

@dataclass
class BackupResult:
    """Backup operation result."""
    job_id: str
    backup_id: str
    status: BackupStatus
    backup_type: BackupType
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    size_bytes: Optional[int]
    file_count: Optional[int]
    components_backed_up: List[str]
    backup_path: Optional[str]
    checksum: Optional[str]
    error_message: Optional[str] = None
    warnings: List[str] = None

class BackupManager:
    """Comprehensive backup management system."""
    
    def __init__(self, backup_root: str = "backups", config_file: str = "backup_config.json"):
        """
        Initialize backup manager.
        
        Args:
            backup_root: Root directory for backups
            config_file: Backup configuration file
        """
        self.backup_root = backup_root
        self.config_file = config_file
        
        # Create backup directory
        os.makedirs(backup_root, exist_ok=True)
        
        # Initialize backup components
        self.db_backup = DatabaseBackup()
        self.file_backup = FileBackup()
        self.model_backup = ModelBackup()
        
        # Backup jobs and results
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.backup_results: List[BackupResult] = []
        
        # Configuration
        self.config = self._load_config()
        
        # Scheduler
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Load existing jobs
        self._load_backup_jobs()
        
        logger.info(f"Backup manager initialized: {backup_root}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load backup configuration."""
        default_config = {
            "max_backup_age_days": 30,
            "max_backup_count": 10,
            "compression_level": 6,
            "encryption_enabled": True,
            "parallel_backups": 2,
            "backup_verification": True,
            "notification_enabled": True,
            "cleanup_enabled": True,
            "default_retention_days": 7
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                default_config.update(config)
            else:
                self._save_config(default_config)
            
            return default_config
            
        except Exception as e:
            logger.error(f"Failed to load backup config: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save backup configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup config: {e}")
    
    def _load_backup_jobs(self):
        """Load backup jobs from configuration."""
        jobs_file = os.path.join(self.backup_root, "backup_jobs.json")
        
        try:
            if os.path.exists(jobs_file):
                with open(jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                
                for job_data in jobs_data:
                    job = BackupJob(
                        job_id=job_data['job_id'],
                        name=job_data['name'],
                        backup_type=BackupType(job_data['backup_type']),
                        components=job_data['components'],
                        schedule_cron=job_data.get('schedule_cron'),
                        retention_days=job_data['retention_days'],
                        compression=job_data['compression'],
                        encryption=job_data['encryption'],
                        destination=job_data['destination'],
                        created_at=datetime.fromisoformat(job_data['created_at']),
                        last_run=datetime.fromisoformat(job_data['last_run']) if job_data.get('last_run') else None,
                        next_run=datetime.fromisoformat(job_data['next_run']) if job_data.get('next_run') else None,
                        enabled=job_data.get('enabled', True)
                    )
                    self.backup_jobs[job.job_id] = job
                
                logger.info(f"Loaded {len(self.backup_jobs)} backup jobs")
                
        except Exception as e:
            logger.error(f"Failed to load backup jobs: {e}")
    
    def _save_backup_jobs(self):
        """Save backup jobs to configuration."""
        jobs_file = os.path.join(self.backup_root, "backup_jobs.json")
        
        try:
            jobs_data = []
            for job in self.backup_jobs.values():
                job_dict = asdict(job)
                job_dict['backup_type'] = job.backup_type.value
                job_dict['created_at'] = job.created_at.isoformat()
                if job.last_run:
                    job_dict['last_run'] = job.last_run.isoformat()
                if job.next_run:
                    job_dict['next_run'] = job.next_run.isoformat()
                jobs_data.append(job_dict)
            
            with open(jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup jobs: {e}")
    
    def create_backup_job(
        self,
        name: str,
        backup_type: BackupType,
        components: List[str],
        schedule_cron: Optional[str] = None,
        retention_days: int = None,
        compression: bool = True,
        encryption: bool = True,
        destination: str = None
    ) -> str:
        """
        Create a new backup job.
        
        Args:
            name: Job name
            backup_type: Type of backup
            components: Components to backup
            schedule_cron: Cron schedule expression
            retention_days: Backup retention period
            compression: Enable compression
            encryption: Enable encryption
            destination: Backup destination path
            
        Returns:
            Job ID
        """
        try:
            import uuid
            
            job_id = str(uuid.uuid4())
            
            if retention_days is None:
                retention_days = self.config['default_retention_days']
            
            if destination is None:
                destination = os.path.join(self.backup_root, name.replace(' ', '_').lower())
            
            job = BackupJob(
                job_id=job_id,
                name=name,
                backup_type=backup_type,
                components=components,
                schedule_cron=schedule_cron,
                retention_days=retention_days,
                compression=compression,
                encryption=encryption,
                destination=destination,
                created_at=datetime.now(timezone.utc)
            )
            
            self.backup_jobs[job_id] = job
            self._save_backup_jobs()
            
            # Schedule job if cron expression provided
            if schedule_cron:
                self._schedule_job(job)
            
            logger.info(f"Backup job created: {name} ({job_id})")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create backup job: {e}")
            raise
    
    async def run_backup_job(self, job_id: str) -> BackupResult:
        """
        Run a backup job.
        
        Args:
            job_id: Backup job ID
            
        Returns:
            Backup result
        """
        try:
            if job_id not in self.backup_jobs:
                raise ValueError(f"Backup job not found: {job_id}")
            
            job = self.backup_jobs[job_id]
            
            if not job.enabled:
                raise ValueError(f"Backup job is disabled: {job_id}")
            
            # Generate backup ID
            import uuid
            backup_id = str(uuid.uuid4())
            
            # Initialize result
            result = BackupResult(
                job_id=job_id,
                backup_id=backup_id,
                status=BackupStatus.RUNNING,
                backup_type=job.backup_type,
                start_time=datetime.now(timezone.utc),
                end_time=None,
                duration_seconds=None,
                size_bytes=0,
                file_count=0,
                components_backed_up=[],
                backup_path=None,
                checksum=None,
                warnings=[]
            )
            
            self.backup_results.append(result)
            
            try:
                # Create backup directory
                backup_dir = os.path.join(job.destination, f"backup_{backup_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(backup_dir, exist_ok=True)
                result.backup_path = backup_dir
                
                # Backup each component
                total_size = 0
                total_files = 0
                
                for component in job.components:
                    component_result = await self._backup_component(
                        component, backup_dir, job.backup_type, job.compression
                    )
                    
                    if component_result['success']:
                        result.components_backed_up.append(component)
                        total_size += component_result.get('size_bytes', 0)
                        total_files += component_result.get('file_count', 0)
                    else:
                        result.warnings.append(f"Component backup failed: {component} - {component_result.get('error')}")
                
                result.size_bytes = total_size
                result.file_count = total_files
                
                # Create backup manifest
                manifest = {
                    'backup_id': backup_id,
                    'job_id': job_id,
                    'backup_type': job.backup_type.value,
                    'timestamp': result.start_time.isoformat(),
                    'components': result.components_backed_up,
                    'size_bytes': total_size,
                    'file_count': total_files,
                    'compression': job.compression,
                    'encryption': job.encryption
                }
                
                manifest_path = os.path.join(backup_dir, "backup_manifest.json")
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # Compress backup if enabled
                if job.compression:
                    compressed_path = await self._compress_backup(backup_dir)
                    if compressed_path:
                        # Remove uncompressed directory
                        shutil.rmtree(backup_dir)
                        result.backup_path = compressed_path
                        result.size_bytes = os.path.getsize(compressed_path)
                
                # Encrypt backup if enabled
                if job.encryption:
                    encrypted_path = await self._encrypt_backup(result.backup_path)
                    if encrypted_path:
                        os.remove(result.backup_path)
                        result.backup_path = encrypted_path
                        result.size_bytes = os.path.getsize(encrypted_path)
                
                # Calculate checksum
                if result.backup_path and os.path.exists(result.backup_path):
                    result.checksum = self._calculate_checksum(result.backup_path)
                
                # Verify backup if enabled
                if self.config.get('backup_verification', True):
                    verification_result = await self._verify_backup(result.backup_path, manifest)
                    if not verification_result['valid']:
                        result.warnings.append(f"Backup verification failed: {verification_result['error']}")
                
                result.status = BackupStatus.COMPLETED
                result.end_time = datetime.now(timezone.utc)
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                
                # Update job last run time
                job.last_run = result.start_time
                self._save_backup_jobs()
                
                logger.info(f"Backup job completed: {job.name} ({backup_id})")
                
            except Exception as e:
                result.status = BackupStatus.FAILED
                result.error_message = str(e)
                result.end_time = datetime.now(timezone.utc)
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                
                logger.error(f"Backup job failed: {job.name} - {e}")
            
            # Cleanup old backups
            if self.config.get('cleanup_enabled', True):
                await self._cleanup_old_backups(job)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run backup job: {e}")
            raise
    
    async def _backup_component(self, component: str, backup_dir: str, backup_type: BackupType, compression: bool) -> Dict[str, Any]:
        """Backup a specific component."""
        try:
            component_dir = os.path.join(backup_dir, component)
            os.makedirs(component_dir, exist_ok=True)
            
            if component == "database":
                result = await self.db_backup.backup_database(component_dir, backup_type)
            elif component == "files":
                result = await self.file_backup.backup_files(component_dir, backup_type)
            elif component == "models":
                result = await self.model_backup.backup_models(component_dir, backup_type)
            elif component == "config":
                result = await self._backup_config(component_dir)
            else:
                return {'success': False, 'error': f'Unknown component: {component}'}
            
            return result
            
        except Exception as e:
            logger.error(f"Component backup failed: {component} - {e}")
            return {'success': False, 'error': str(e)}
    
    async def _backup_config(self, backup_dir: str) -> Dict[str, Any]:
        """Backup configuration files."""
        try:
            config_files = [
                "config.py",
                "backup_config.json",
                ".env",
                "requirements.txt"
            ]
            
            backed_up_files = []
            total_size = 0
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    dest_path = os.path.join(backup_dir, os.path.basename(config_file))
                    shutil.copy2(config_file, dest_path)
                    backed_up_files.append(config_file)
                    total_size += os.path.getsize(dest_path)
            
            return {
                'success': True,
                'files_backed_up': backed_up_files,
                'file_count': len(backed_up_files),
                'size_bytes': total_size
            }
            
        except Exception as e:
            logger.error(f"Config backup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _compress_backup(self, backup_dir: str) -> Optional[str]:
        """Compress backup directory."""
        try:
            compressed_path = f"{backup_dir}.zip"
            
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=self.config['compression_level']) as zipf:
                for root, dirs, files in os.walk(backup_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, backup_dir)
                        zipf.write(file_path, arc_path)
            
            logger.info(f"Backup compressed: {compressed_path}")
            return compressed_path
            
        except Exception as e:
            logger.error(f"Backup compression failed: {e}")
            return None
    
    async def _encrypt_backup(self, backup_path: str) -> Optional[str]:
        """Encrypt backup file."""
        try:
            from security.encryption import DataEncryption
            
            # Initialize encryption
            encryptor = DataEncryption()
            
            # Read backup file
            with open(backup_path, 'rb') as f:
                backup_data = f.read()
            
            # Encrypt data
            encrypted_data = encryptor.encrypt(backup_data)
            
            # Save encrypted backup
            encrypted_path = f"{backup_path}.encrypted"
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Save encryption key securely (in production, use key management service)
            key_path = f"{backup_path}.key"
            with open(key_path, 'w') as f:
                f.write(encryptor.get_key_string())
            
            logger.info(f"Backup encrypted: {encrypted_path}")
            return encrypted_path
            
        except Exception as e:
            logger.error(f"Backup encryption failed: {e}")
            return None
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    async def _verify_backup(self, backup_path: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Verify backup integrity."""
        try:
            if not os.path.exists(backup_path):
                return {'valid': False, 'error': 'Backup file not found'}
            
            # Check file size
            actual_size = os.path.getsize(backup_path)
            if actual_size == 0:
                return {'valid': False, 'error': 'Backup file is empty'}
            
            # For compressed/encrypted backups, basic checks
            if backup_path.endswith('.zip'):
                try:
                    with zipfile.ZipFile(backup_path, 'r') as zipf:
                        # Test zip file integrity
                        zipf.testzip()
                except zipfile.BadZipFile:
                    return {'valid': False, 'error': 'Corrupted zip file'}
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def _cleanup_old_backups(self, job: BackupJob):
        """Clean up old backups based on retention policy."""
        try:
            if not os.path.exists(job.destination):
                return
            
            # Get all backup files/directories
            backup_items = []
            for item in os.listdir(job.destination):
                item_path = os.path.join(job.destination, item)
                if os.path.isfile(item_path) or os.path.isdir(item_path):
                    stat = os.stat(item_path)
                    backup_items.append({
                        'path': item_path,
                        'name': item,
                        'created': datetime.fromtimestamp(stat.st_ctime),
                        'size': stat.st_size if os.path.isfile(item_path) else 0
                    })
            
            # Sort by creation time (newest first)
            backup_items.sort(key=lambda x: x['created'], reverse=True)
            
            # Remove backups older than retention period
            cutoff_date = datetime.now() - timedelta(days=job.retention_days)
            removed_count = 0
            
            for item in backup_items:
                if item['created'] < cutoff_date:
                    try:
                        if os.path.isfile(item['path']):
                            os.remove(item['path'])
                        else:
                            shutil.rmtree(item['path'])
                        removed_count += 1
                        logger.info(f"Removed old backup: {item['name']}")
                    except Exception as e:
                        logger.error(f"Failed to remove old backup {item['name']}: {e}")
            
            # Also enforce max backup count
            max_count = self.config.get('max_backup_count', 10)
            if len(backup_items) > max_count:
                for item in backup_items[max_count:]:
                    try:
                        if os.path.isfile(item['path']):
                            os.remove(item['path'])
                        else:
                            shutil.rmtree(item['path'])
                        removed_count += 1
                        logger.info(f"Removed excess backup: {item['name']}")
                    except Exception as e:
                        logger.error(f"Failed to remove excess backup {item['name']}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old backups for job {job.name}")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def start_scheduler(self):
        """Start the backup scheduler."""
        if self.scheduler_running:
            logger.warning("Backup scheduler is already running")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def stop_scheduler(self):
        """Stop the backup scheduler."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Backup scheduler stopped")
    
    def _run_scheduler(self):
        """Run the backup scheduler."""
        while self.scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _schedule_job(self, job: BackupJob):
        """Schedule a backup job."""
        if not job.schedule_cron:
            return
        
        # Simple cron parsing (extend as needed)
        # For now, support basic patterns like "0 2 * * *" (daily at 2 AM)
        try:
            # This is a simplified implementation
            # In production, use a proper cron library like croniter
            if job.schedule_cron == "0 2 * * *":  # Daily at 2 AM
                schedule.every().day.at("02:00").do(self._scheduled_backup, job.job_id)
            elif job.schedule_cron == "0 0 * * 0":  # Weekly on Sunday
                schedule.every().sunday.at("00:00").do(self._scheduled_backup, job.job_id)
            
            logger.info(f"Scheduled backup job: {job.name} - {job.schedule_cron}")
            
        except Exception as e:
            logger.error(f"Failed to schedule job {job.name}: {e}")
    
    def _scheduled_backup(self, job_id: str):
        """Run scheduled backup."""
        try:
            asyncio.create_task(self.run_backup_job(job_id))
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
    
    def get_backup_jobs(self) -> List[Dict[str, Any]]:
        """Get all backup jobs."""
        return [asdict(job) for job in self.backup_jobs.values()]
    
    def get_backup_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get backup results."""
        results = sorted(self.backup_results, key=lambda x: x.start_time, reverse=True)
        return [asdict(result) for result in results[:limit]]
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        try:
            total_jobs = len(self.backup_jobs)
            enabled_jobs = sum(1 for job in self.backup_jobs.values() if job.enabled)
            
            recent_results = [r for r in self.backup_results if r.start_time > datetime.now(timezone.utc) - timedelta(days=7)]
            successful_backups = sum(1 for r in recent_results if r.status == BackupStatus.COMPLETED)
            failed_backups = sum(1 for r in recent_results if r.status == BackupStatus.FAILED)
            
            total_backup_size = 0
            for result in recent_results:
                if result.size_bytes:
                    total_backup_size += result.size_bytes
            
            return {
                'total_jobs': total_jobs,
                'enabled_jobs': enabled_jobs,
                'recent_backups': len(recent_results),
                'successful_backups': successful_backups,
                'failed_backups': failed_backups,
                'success_rate': (successful_backups / max(len(recent_results), 1)) * 100,
                'total_backup_size_bytes': total_backup_size,
                'scheduler_running': self.scheduler_running
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup statistics: {e}")
            return {}
    
    def delete_backup_job(self, job_id: str) -> bool:
        """Delete a backup job."""
        try:
            if job_id not in self.backup_jobs:
                return False
            
            job = self.backup_jobs[job_id]
            
            # Remove from scheduler
            schedule.clear(job.name)
            
            # Delete job
            del self.backup_jobs[job_id]
            self._save_backup_jobs()
            
            logger.info(f"Backup job deleted: {job.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup job: {e}")
            return False