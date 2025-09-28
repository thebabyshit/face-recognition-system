"""Access control API routes."""
import logging
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from database.services import DatabaseService
from services.access_manager import AccessManager
from ..models import (
    AccessLogResponse, AccessLogListResponse, AccessAttemptRequest,
    AccessControlResponse, SuccessResponse, ErrorResponse
)
from ..dependencies import (
    get_db_service, require_read_permission, require_write_permission,
    validate_person_id, validate_location_id, get_common_params, CommonQueryParams
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/attempt", response_model=AccessControlResponse)
async def process_access_attempt(
    access_request: AccessAttemptRequest,
    current_user: dict = Depends(require_write_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Process an access attempt.
    Validates access permissions and logs the attempt.
    """
    try:
        access_manager = AccessManager()
        
        # Process access attempt
        result = access_manager.process_access_attempt(
            person_id=access_request.person_id,
            location_id=access_request.location_id,
            access_method=access_request.access_method,
            confidence_score=access_request.confidence_score,
            additional_data=access_request.additional_data
        )
        
        return AccessControlResponse(
            access_granted=result['access_granted'],
            person_id=result.get('person_id'),
            person_name=result.get('person_name'),
            location_id=result.get('location_id'),
            location_name=result.get('location_name'),
            access_level_required=result.get('access_level_required'),
            person_access_level=result.get('person_access_level'),
            confidence_score=result.get('confidence_score'),
            access_method=result.get('access_method'),
            reason=result.get('reason'),
            timestamp=result.get('timestamp'),
            log_id=result.get('log_id')
        )
        
    except Exception as e:
        logger.error(f"Error processing access attempt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process access attempt"
        )

@router.get("/logs", response_model=AccessLogListResponse)
async def get_access_logs(
    params: CommonQueryParams = Depends(get_common_params),
    person_id: Optional[int] = Query(None, description="Filter by person ID"),
    location_id: Optional[int] = Query(None, description="Filter by location ID"),
    access_granted: Optional[bool] = Query(None, description="Filter by access result"),
    access_method: Optional[str] = Query(None, description="Filter by access method"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get access logs with filtering and pagination.
    Returns a paginated list of access logs with optional filters.
    """
    try:
        # Build filter criteria
        filters = {}
        if person_id is not None:
            filters['person_id'] = person_id
        if location_id is not None:
            filters['location_id'] = location_id
        if access_granted is not None:
            filters['access_granted'] = access_granted
        if access_method is not None:
            filters['access_method'] = access_method
        if start_date is not None:
            filters['start_date'] = start_date
        if end_date is not None:
            filters['end_date'] = end_date
        
        # Get access logs
        access_logs = db_service.access_logs.get_filtered_logs(
            filters=filters,
            limit=params.limit,
            offset=params.offset,
            sort_by=params.sort_by or 'timestamp',
            sort_order=params.sort_order
        )
        
        # Get total count
        total_count = db_service.access_logs.count_filtered_logs(filters)
        
        # Convert to response format
        log_responses = []
        for log in access_logs:
            log_responses.append(AccessLogResponse(
                id=log.id,
                uuid=str(log.uuid),
                person_id=log.person_id,
                person_name=log.person_name,
                location_id=log.location_id,
                location_name=log.location_name,
                access_granted=log.access_granted,
                access_method=log.access_method,
                confidence_score=log.confidence_score,
                reason=log.reason,
                timestamp=log.timestamp,
                additional_data=log.get_additional_data()
            ))
        
        return AccessLogListResponse(
            logs=log_responses,
            total_count=total_count,
            returned_count=len(log_responses),
            offset=params.offset,
            limit=params.limit,
            has_more=total_count > (params.offset + len(log_responses))
        )
        
    except Exception as e:
        logger.error(f"Error getting access logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve access logs"
        )

@router.get("/logs/{log_id}", response_model=AccessLogResponse)
async def get_access_log(
    log_id: int,
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get a specific access log by ID.
    Returns detailed information about an access log entry.
    """
    try:
        access_log = db_service.access_logs.get_by_id(log_id)
        
        if not access_log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access log not found"
            )
        
        return AccessLogResponse(
            id=access_log.id,
            uuid=str(access_log.uuid),
            person_id=access_log.person_id,
            person_name=access_log.person_name,
            location_id=access_log.location_id,
            location_name=access_log.location_name,
            access_granted=access_log.access_granted,
            access_method=access_log.access_method,
            confidence_score=access_log.confidence_score,
            reason=access_log.reason,
            timestamp=access_log.timestamp,
            additional_data=access_log.get_additional_data()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting access log {log_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve access log"
        )

@router.get("/person/{person_id}/logs", response_model=AccessLogListResponse)
async def get_person_access_logs(
    person_id: int = Depends(validate_person_id),
    params: CommonQueryParams = Depends(get_common_params),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    access_granted: Optional[bool] = Query(None, description="Filter by access result"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get access logs for a specific person.
    Returns access history for the specified person.
    """
    try:
        # Validate person exists
        person = db_service.persons.get_person_by_id(person_id)
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person not found"
            )
        
        # Build filter criteria
        filters = {'person_id': person_id}
        if start_date is not None:
            filters['start_date'] = start_date
        if end_date is not None:
            filters['end_date'] = end_date
        if access_granted is not None:
            filters['access_granted'] = access_granted
        
        # Get access logs
        access_logs = db_service.access_logs.get_filtered_logs(
            filters=filters,
            limit=params.limit,
            offset=params.offset,
            sort_by=params.sort_by or 'timestamp',
            sort_order=params.sort_order
        )
        
        # Get total count
        total_count = db_service.access_logs.count_filtered_logs(filters)
        
        # Convert to response format
        log_responses = []
        for log in access_logs:
            log_responses.append(AccessLogResponse(
                id=log.id,
                uuid=str(log.uuid),
                person_id=log.person_id,
                person_name=log.person_name,
                location_id=log.location_id,
                location_name=log.location_name,
                access_granted=log.access_granted,
                access_method=log.access_method,
                confidence_score=log.confidence_score,
                reason=log.reason,
                timestamp=log.timestamp,
                additional_data=log.get_additional_data()
            ))
        
        return AccessLogListResponse(
            logs=log_responses,
            total_count=total_count,
            returned_count=len(log_responses),
            offset=params.offset,
            limit=params.limit,
            has_more=total_count > (params.offset + len(log_responses))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting person access logs {person_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve person access logs"
        )

@router.get("/location/{location_id}/logs", response_model=AccessLogListResponse)
async def get_location_access_logs(
    location_id: int = Depends(validate_location_id),
    params: CommonQueryParams = Depends(get_common_params),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    access_granted: Optional[bool] = Query(None, description="Filter by access result"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get access logs for a specific location.
    Returns access history for the specified location.
    """
    try:
        # Build filter criteria
        filters = {'location_id': location_id}
        if start_date is not None:
            filters['start_date'] = start_date
        if end_date is not None:
            filters['end_date'] = end_date
        if access_granted is not None:
            filters['access_granted'] = access_granted
        
        # Get access logs
        access_logs = db_service.access_logs.get_filtered_logs(
            filters=filters,
            limit=params.limit,
            offset=params.offset,
            sort_by=params.sort_by or 'timestamp',
            sort_order=params.sort_order
        )
        
        # Get total count
        total_count = db_service.access_logs.count_filtered_logs(filters)
        
        # Convert to response format
        log_responses = []
        for log in access_logs:
            log_responses.append(AccessLogResponse(
                id=log.id,
                uuid=str(log.uuid),
                person_id=log.person_id,
                person_name=log.person_name,
                location_id=log.location_id,
                location_name=log.location_name,
                access_granted=log.access_granted,
                access_method=log.access_method,
                confidence_score=log.confidence_score,
                reason=log.reason,
                timestamp=log.timestamp,
                additional_data=log.get_additional_data()
            ))
        
        return AccessLogListResponse(
            logs=log_responses,
            total_count=total_count,
            returned_count=len(log_responses),
            offset=params.offset,
            limit=params.limit,
            has_more=total_count > (params.offset + len(log_responses))
        )
        
    except Exception as e:
        logger.error(f"Error getting location access logs {location_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve location access logs"
        )

@router.get("/stats/summary")
async def get_access_summary(
    start_date: Optional[datetime] = Query(None, description="Start date for summary"),
    end_date: Optional[datetime] = Query(None, description="End date for summary"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get access statistics summary.
    Returns summary statistics for access attempts.
    """
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get summary statistics
        filters = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        total_attempts = db_service.access_logs.count_filtered_logs(filters)
        
        granted_filters = {**filters, 'access_granted': True}
        granted_attempts = db_service.access_logs.count_filtered_logs(granted_filters)
        
        denied_filters = {**filters, 'access_granted': False}
        denied_attempts = db_service.access_logs.count_filtered_logs(denied_filters)
        
        # Calculate success rate
        success_rate = (granted_attempts / total_attempts * 100) if total_attempts > 0 else 0
        
        # Get unique persons and locations
        unique_persons = db_service.access_logs.count_unique_persons(filters)
        unique_locations = db_service.access_logs.count_unique_locations(filters)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "summary": {
                "total_attempts": total_attempts,
                "granted_attempts": granted_attempts,
                "denied_attempts": denied_attempts,
                "success_rate": round(success_rate, 2),
                "unique_persons": unique_persons,
                "unique_locations": unique_locations
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting access summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve access summary"
        )

@router.get("/stats/hourly")
async def get_hourly_access_stats(
    date: Optional[datetime] = Query(None, description="Date for hourly stats"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get hourly access statistics.
    Returns access attempts grouped by hour for a specific date.
    """
    try:
        # Set default date if not provided
        if date is None:
            date = datetime.now(timezone.utc).date()
        
        # Get hourly statistics
        hourly_stats = db_service.access_logs.get_hourly_stats(date)
        
        return {
            "date": date,
            "hourly_stats": hourly_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting hourly access stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve hourly access statistics"
        )