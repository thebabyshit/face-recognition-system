"""Person management API routes."""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from database.services import DatabaseService
from ..models import (
    PersonCreate, PersonUpdate, PersonResponse, PersonListResponse,
    PersonSearchRequest, SuccessResponse, ErrorResponse, BulkPersonImport,
    BulkImportResponse
)
from ..dependencies import (
    get_db_service, require_write_permission, require_read_permission,
    validate_person_id, get_common_params, CommonQueryParams
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=PersonListResponse)
async def list_persons(
    params: CommonQueryParams = Depends(get_common_params),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    List persons with optional filtering and pagination.
    Returns a paginated list of persons with optional filters.
    """
    try:
        # Mock response for now
        return PersonListResponse(
            persons=[],
            total_count=0,
            returned_count=0,
            offset=params.offset,
            limit=params.limit,
            has_more=False
        )
        
    except Exception as e:
        logger.error(f"Error listing persons: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve persons"
        )

@router.post("/", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
async def create_person(
    person_data: PersonCreate,
    current_user: dict = Depends(require_write_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Create a new person.
    Creates a new person record with the provided information.
    """
    try:
        # Mock response for now
        from datetime import datetime, timezone
        return PersonResponse(
            id=1,
            uuid="mock-uuid",
            name=person_data.name,
            employee_id=person_data.employee_id,
            email=person_data.email,
            phone=person_data.phone,
            department=person_data.department,
            position=person_data.position,
            access_level=person_data.access_level,
            is_active=True,
            notes=person_data.notes,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Error creating person: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create person"
        )