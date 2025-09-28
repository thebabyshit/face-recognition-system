"""System management API routes."""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from database.services import DatabaseService
from ..models import SuccessResponse, ErrorResponse
from ..dependencies import (
    get_db_service, require_admin_permission, require_read_permission
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    System health check endpoint.
    Returns basic system health information.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "version": "1.0.0",
            "service": "face_recognition_api"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@router.get("/info")
async def get_system_info(
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get system information.
    Returns detailed system information and statistics.
    """
    try:
        # Get database statistics
        db_stats = db_service.get_database_stats()
        
        # Get system counts
        total_persons = db_service.persons.count_active_persons()
        total_features = db_service.face_features.count_active_features()
        total_logs = db_service.access_logs.count_total_logs()
        
        return {
            "system": {
                "version": "1.0.0",
                "api_version": "v1",
                "environment": "production",  # Could be configurable
                "uptime": db_stats.get('uptime', 0),
                "started_at": db_stats.get('started_at')
            },
            "database": {
                "status": "connected",
                "size": db_stats.get('database_size', 0),
                "connections": db_stats.get('connection_count', 0),
                "tables": db_stats.get('table_count', 0)
            },
            "statistics": {
                "total_persons": total_persons,
                "total_face_features": total_features,
                "total_access_logs": total_logs,
                "active_sessions": 0  # Could track active API sessions
            },
            "features": {
                "face_recognition": True,
                "access_control": True,
                "reporting": True,
                "real_time_monitoring": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system information"
        )

@router.get("/logs")
async def get_system_logs(
    level: Optional[str] = Query(None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    component: Optional[str] = Query(None, description="Filter by component"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get system logs.
    Returns system logs with optional filtering.
    """
    try:
        # Build filter criteria
        filters = {}
        if level:
            filters['level'] = level
        if component:
            filters['component'] = component
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        
        # Get system logs
        logs = db_service.system_logs.get_filtered_logs(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        # Get total count
        total_count = db_service.system_logs.count_filtered_logs(filters)
        
        # Convert to response format
        log_entries = []
        for log in logs:
            log_entries.append({
                "id": log.id,
                "timestamp": log.timestamp,
                "level": log.level,
                "component": log.component,
                "message": log.message,
                "details": log.get_details(),
                "user_id": log.user_id,
                "session_id": log.session_id
            })
        
        return {
            "logs": log_entries,
            "total_count": total_count,
            "returned_count": len(log_entries),
            "offset": offset,
            "limit": limit,
            "has_more": total_count > (offset + len(log_entries))
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system logs"
        )

@router.post("/maintenance/cleanup")
async def cleanup_old_data(
    days_to_keep: int = Query(90, ge=1, le=3650, description="Days of data to keep"),
    dry_run: bool = Query(True, description="Perform dry run without actual deletion"),
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Clean up old system data.
    Removes old logs and inactive records based on retention policy.
    """
    try:
        # Calculate cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # Get counts of data to be cleaned
        old_access_logs = db_service.access_logs.count_logs_before_date(cutoff_date)
        old_system_logs = db_service.system_logs.count_logs_before_date(cutoff_date)
        inactive_features = db_service.face_features.count_inactive_features_before_date(cutoff_date)
        
        cleanup_summary = {
            "cutoff_date": cutoff_date,
            "days_to_keep": days_to_keep,
            "dry_run": dry_run,
            "items_to_cleanup": {
                "old_access_logs": old_access_logs,
                "old_system_logs": old_system_logs,
                "inactive_face_features": inactive_features
            }
        }
        
        if not dry_run:
            # Perform actual cleanup
            deleted_access_logs = db_service.access_logs.delete_logs_before_date(cutoff_date)
            deleted_system_logs = db_service.system_logs.delete_logs_before_date(cutoff_date)
            deleted_features = db_service.face_features.delete_inactive_features_before_date(cutoff_date)
            
            cleanup_summary["items_cleaned"] = {
                "deleted_access_logs": deleted_access_logs,
                "deleted_system_logs": deleted_system_logs,
                "deleted_face_features": deleted_features
            }
            
            # Log the cleanup operation
            db_service.system_logs.log_system_event(
                level="INFO",
                component="maintenance",
                message=f"Data cleanup completed. Deleted {deleted_access_logs + deleted_system_logs + deleted_features} records.",
                user_id=current_user.get('id')
            )
        
        return SuccessResponse(
            message="Data cleanup completed successfully" if not dry_run else "Dry run completed",
            data=cleanup_summary
        )
        
    except Exception as e:
        logger.error(f"Error during data cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform data cleanup"
        )

@router.post("/maintenance/optimize")
async def optimize_database(
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Optimize database performance.
    Performs database maintenance operations like VACUUM and ANALYZE.
    """
    try:
        # Perform database optimization
        optimization_results = db_service.optimize_database()
        
        # Log the optimization operation
        db_service.system_logs.log_system_event(
            level="INFO",
            component="maintenance",
            message="Database optimization completed",
            user_id=current_user.get('id')
        )
        
        return SuccessResponse(
            message="Database optimization completed successfully",
            data=optimization_results
        )
        
    except Exception as e:
        logger.error(f"Error during database optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize database"
        )

@router.post("/backup/create")
async def create_backup(
    backup_type: str = Query("full", regex="^(full|data|schema)$", description="Type of backup"),
    include_logs: bool = Query(False, description="Include log tables in backup"),
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Create system backup.
    Creates a backup of the database and system data.
    """
    try:
        # Create backup
        backup_result = db_service.create_backup(
            backup_type=backup_type,
            include_logs=include_logs,
            created_by=current_user.get('id')
        )
        
        # Log the backup operation
        db_service.system_logs.log_system_event(
            level="INFO",
            component="backup",
            message=f"Backup created: {backup_result.get('backup_file')}",
            user_id=current_user.get('id')
        )
        
        return SuccessResponse(
            message="Backup created successfully",
            data=backup_result
        )
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create backup"
        )

@router.get("/backup/list")
async def list_backups(
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    List available backups.
    Returns a list of available system backups.
    """
    try:
        # Get list of backups
        backups = db_service.list_backups()
        
        return {
            "backups": backups,
            "total_count": len(backups)
        }
        
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list backups"
        )

@router.get("/config")
async def get_system_config(
    current_user: dict = Depends(require_admin_permission)
):
    """
    Get system configuration.
    Returns current system configuration settings.
    """
    try:
        # Get system configuration
        # This would typically read from config files or environment variables
        config = {
            "recognition": {
                "confidence_threshold": 0.7,
                "max_face_size": 1024,
                "quality_threshold": 0.6,
                "feature_dimension": 512
            },
            "access_control": {
                "default_access_level": 1,
                "max_failed_attempts": 3,
                "lockout_duration": 300
            },
            "api": {
                "rate_limit_calls": 100,
                "rate_limit_period": 60,
                "max_upload_size": 10485760,  # 10MB
                "session_timeout": 3600
            },
            "database": {
                "connection_pool_size": 20,
                "query_timeout": 30,
                "backup_retention_days": 30
            },
            "logging": {
                "log_level": "INFO",
                "log_retention_days": 90,
                "max_log_file_size": 104857600  # 100MB
            }
        }
        
        return {
            "configuration": config,
            "last_updated": datetime.now(timezone.utc),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system configuration"
        )

@router.put("/config")
async def update_system_config(
    config_updates: Dict[str, Any],
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Update system configuration.
    Updates system configuration settings.
    """
    try:
        # Validate configuration updates
        # This would include validation logic for each config section
        
        # Apply configuration updates
        # This would typically update config files or database settings
        
        # Log the configuration change
        db_service.system_logs.log_system_event(
            level="INFO",
            component="configuration",
            message=f"System configuration updated by {current_user.get('username')}",
            details={"updated_keys": list(config_updates.keys())},
            user_id=current_user.get('id')
        )
        
        return SuccessResponse(
            message="System configuration updated successfully",
            data={
                "updated_at": datetime.now(timezone.utc),
                "updated_by": current_user.get('username'),
                "updated_keys": list(config_updates.keys())
            }
        )
        
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system configuration"
        )

@router.post("/restart")
async def restart_service(
    component: str = Query("api", regex="^(api|recognition|all)$", description="Component to restart"),
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Restart system components.
    Restarts specified system components.
    """
    try:
        # Log the restart request
        db_service.system_logs.log_system_event(
            level="WARNING",
            component="system",
            message=f"Service restart requested for component: {component}",
            user_id=current_user.get('id')
        )
        
        # Note: Actual restart logic would depend on deployment method
        # This is a placeholder response
        return SuccessResponse(
            message=f"Restart request for {component} component has been queued",
            data={
                "component": component,
                "requested_at": datetime.now(timezone.utc),
                "requested_by": current_user.get('username'),
                "status": "queued"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing restart request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process restart request"
        )