"""Statistics and reporting API routes."""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from database.services import DatabaseService
from ..models import SuccessResponse, ErrorResponse
from ..dependencies import (
    get_db_service, require_read_permission, require_admin_permission
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_stats(
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get dashboard statistics.
    Returns key metrics for the main dashboard.
    """
    try:
        # Get current date ranges
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)
        month_start = today_start - timedelta(days=30)
        
        # Get basic counts
        total_persons = db_service.persons.count_active_persons()
        total_features = db_service.face_features.count_active_features()
        
        # Get today's access attempts
        today_filters = {
            'start_date': today_start,
            'end_date': now
        }
        today_attempts = db_service.access_logs.count_filtered_logs(today_filters)
        today_granted = db_service.access_logs.count_filtered_logs({
            **today_filters,
            'access_granted': True
        })
        
        # Get weekly stats
        week_filters = {
            'start_date': week_start,
            'end_date': now
        }
        week_attempts = db_service.access_logs.count_filtered_logs(week_filters)
        week_granted = db_service.access_logs.count_filtered_logs({
            **week_filters,
            'access_granted': True
        })
        
        # Get monthly stats
        month_filters = {
            'start_date': month_start,
            'end_date': now
        }
        month_attempts = db_service.access_logs.count_filtered_logs(month_filters)
        month_granted = db_service.access_logs.count_filtered_logs({
            **month_filters,
            'access_granted': True
        })
        
        # Calculate success rates
        today_success_rate = (today_granted / today_attempts * 100) if today_attempts > 0 else 0
        week_success_rate = (week_granted / week_attempts * 100) if week_attempts > 0 else 0
        month_success_rate = (month_granted / month_attempts * 100) if month_attempts > 0 else 0
        
        # Get recent activity
        recent_logs = db_service.access_logs.get_recent_logs(limit=10)
        recent_activity = []
        for log in recent_logs:
            recent_activity.append({
                "timestamp": log.timestamp,
                "person_name": log.person_name,
                "location_name": log.location_name,
                "access_granted": log.access_granted,
                "access_method": log.access_method
            })
        
        return {
            "summary": {
                "total_persons": total_persons,
                "total_face_features": total_features,
                "today_attempts": today_attempts,
                "today_success_rate": round(today_success_rate, 2)
            },
            "periods": {
                "today": {
                    "attempts": today_attempts,
                    "granted": today_granted,
                    "denied": today_attempts - today_granted,
                    "success_rate": round(today_success_rate, 2)
                },
                "week": {
                    "attempts": week_attempts,
                    "granted": week_granted,
                    "denied": week_attempts - week_granted,
                    "success_rate": round(week_success_rate, 2)
                },
                "month": {
                    "attempts": month_attempts,
                    "granted": month_granted,
                    "denied": month_attempts - month_granted,
                    "success_rate": round(month_success_rate, 2)
                }
            },
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard statistics"
        )

@router.get("/access-trends")
async def get_access_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    group_by: str = Query("day", regex="^(hour|day|week)$", description="Grouping period"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get access trends over time.
    Returns access statistics grouped by time periods.
    """
    try:
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get trend data
        trends = db_service.access_logs.get_access_trends(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": days,
                "group_by": group_by
            },
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"Error getting access trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve access trends"
        )

@router.get("/department-stats")
async def get_department_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get statistics by department.
    Returns access statistics grouped by department.
    """
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get department statistics
        dept_stats = db_service.access_logs.get_department_stats(
            start_date=start_date,
            end_date=end_date
        )
        
        # Get person counts by department
        person_counts = db_service.persons.get_department_counts()
        
        # Combine statistics
        combined_stats = []
        for dept_stat in dept_stats:
            department = dept_stat['department']
            person_count = person_counts.get(department, 0)
            
            combined_stats.append({
                "department": department,
                "person_count": person_count,
                "total_attempts": dept_stat['total_attempts'],
                "granted_attempts": dept_stat['granted_attempts'],
                "denied_attempts": dept_stat['denied_attempts'],
                "success_rate": dept_stat['success_rate'],
                "unique_persons": dept_stat['unique_persons'],
                "avg_attempts_per_person": dept_stat['avg_attempts_per_person']
            })
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "department_statistics": combined_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting department statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve department statistics"
        )

@router.get("/person-activity")
async def get_person_activity_stats(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    limit: int = Query(50, ge=1, le=500, description="Number of persons to return"),
    sort_by: str = Query("attempts", regex="^(attempts|success_rate|last_access)$"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get person activity statistics.
    Returns access statistics for individual persons.
    """
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get person activity statistics
        person_stats = db_service.access_logs.get_person_activity_stats(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            sort_by=sort_by
        )
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "sort_by": sort_by,
            "person_activity": person_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting person activity stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve person activity statistics"
        )

@router.get("/location-stats")
async def get_location_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get statistics by location.
    Returns access statistics grouped by location.
    """
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get location statistics
        location_stats = db_service.access_logs.get_location_stats(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "location_statistics": location_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting location statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve location statistics"
        )

@router.get("/recognition-performance")
async def get_recognition_performance_stats(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get face recognition performance statistics.
    Returns statistics about recognition accuracy and performance.
    """
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get recognition performance stats
        perf_stats = db_service.access_logs.get_recognition_performance_stats(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "recognition_performance": perf_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting recognition performance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recognition performance statistics"
        )

@router.get("/system-health")
async def get_system_health_stats(
    current_user: dict = Depends(require_admin_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get system health statistics.
    Returns various system health metrics and status information.
    """
    try:
        # Get database statistics
        db_stats = db_service.get_database_stats()
        
        # Get recent error logs
        error_logs = db_service.system_logs.get_recent_errors(limit=10)
        
        # Get system performance metrics
        performance_metrics = {
            "database_size": db_stats.get('database_size', 0),
            "table_sizes": db_stats.get('table_sizes', {}),
            "connection_count": db_stats.get('connection_count', 0),
            "query_performance": db_stats.get('query_performance', {}),
            "recent_errors": len(error_logs),
            "uptime": db_stats.get('uptime', 0)
        }
        
        # Calculate health score
        health_score = 100
        if len(error_logs) > 5:
            health_score -= 20
        if db_stats.get('connection_count', 0) > 50:
            health_score -= 10
        if db_stats.get('database_size', 0) > 1000000000:  # 1GB
            health_score -= 10
        
        return {
            "health_score": max(health_score, 0),
            "status": "healthy" if health_score > 80 else "warning" if health_score > 60 else "critical",
            "performance_metrics": performance_metrics,
            "recent_errors": [
                {
                    "timestamp": log.timestamp,
                    "level": log.level,
                    "message": log.message,
                    "component": log.component
                }
                for log in error_logs
            ],
            "recommendations": []  # Could add system recommendations based on metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting system health stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health statistics"
        )

@router.post("/generate-report")
async def generate_custom_report(
    report_type: str = Query(..., regex="^(access|recognition|department|person|system)$"),
    start_date: datetime = Query(..., description="Report start date"),
    end_date: datetime = Query(..., description="Report end date"),
    format: str = Query("json", regex="^(json|csv|pdf)$", description="Report format"),
    include_charts: bool = Query(False, description="Include charts in report"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Generate a custom report.
    Creates a detailed report based on specified parameters.
    """
    try:
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )
        
        # Generate report based on type
        if report_type == "access":
            report_data = await _generate_access_report(db_service, start_date, end_date)
        elif report_type == "recognition":
            report_data = await _generate_recognition_report(db_service, start_date, end_date)
        elif report_type == "department":
            report_data = await _generate_department_report(db_service, start_date, end_date)
        elif report_type == "person":
            report_data = await _generate_person_report(db_service, start_date, end_date)
        elif report_type == "system":
            report_data = await _generate_system_report(db_service, start_date, end_date)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid report type"
            )
        
        # Add metadata
        report = {
            "metadata": {
                "report_type": report_type,
                "start_date": start_date,
                "end_date": end_date,
                "generated_at": datetime.now(timezone.utc),
                "generated_by": current_user.get('username', 'unknown'),
                "format": format,
                "include_charts": include_charts
            },
            "data": report_data
        }
        
        # TODO: Implement format conversion (CSV, PDF) if needed
        if format != "json":
            # For now, return JSON with a note about format
            report["note"] = f"Report generated in JSON format. {format.upper()} export not yet implemented."
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate custom report"
        )

# Helper functions for report generation
async def _generate_access_report(db_service: DatabaseService, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Generate access report data."""
    filters = {'start_date': start_date, 'end_date': end_date}
    
    total_attempts = db_service.access_logs.count_filtered_logs(filters)
    granted_attempts = db_service.access_logs.count_filtered_logs({**filters, 'access_granted': True})
    denied_attempts = total_attempts - granted_attempts
    
    trends = db_service.access_logs.get_access_trends(start_date, end_date, 'day')
    location_stats = db_service.access_logs.get_location_stats(start_date, end_date)
    
    return {
        "summary": {
            "total_attempts": total_attempts,
            "granted_attempts": granted_attempts,
            "denied_attempts": denied_attempts,
            "success_rate": (granted_attempts / total_attempts * 100) if total_attempts > 0 else 0
        },
        "trends": trends,
        "location_breakdown": location_stats
    }

async def _generate_recognition_report(db_service: DatabaseService, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Generate recognition performance report data."""
    perf_stats = db_service.access_logs.get_recognition_performance_stats(start_date, end_date)
    
    return {
        "performance_metrics": perf_stats,
        "quality_analysis": {
            # Add quality analysis if available
        }
    }

async def _generate_department_report(db_service: DatabaseService, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Generate department report data."""
    dept_stats = db_service.access_logs.get_department_stats(start_date, end_date)
    person_counts = db_service.persons.get_department_counts()
    
    return {
        "department_statistics": dept_stats,
        "person_distribution": person_counts
    }

async def _generate_person_report(db_service: DatabaseService, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Generate person activity report data."""
    person_stats = db_service.access_logs.get_person_activity_stats(start_date, end_date, limit=100, sort_by='attempts')
    
    return {
        "person_activity": person_stats,
        "top_users": person_stats[:10] if person_stats else []
    }

async def _generate_system_report(db_service: DatabaseService, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Generate system report data."""
    db_stats = db_service.get_database_stats()
    error_logs = db_service.system_logs.get_recent_errors(limit=50)
    
    return {
        "database_statistics": db_stats,
        "error_summary": {
            "total_errors": len(error_logs),
            "recent_errors": error_logs[:10]
        },
        "system_health": {
            "status": "operational",  # Could be calculated based on metrics
            "uptime": db_stats.get('uptime', 0)
        }
    }