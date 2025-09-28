"""Face feature management API routes."""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import numpy as np

from database.services import DatabaseService
from services.face_manager import FaceManager
from ..models import (
    FaceFeatureUpload, FaceFeatureResponse, FaceFeatureUpdate,
    SuccessResponse, ErrorResponse
)
from ..dependencies import (
    get_db_service, require_write_permission, require_read_permission,
    validate_person_id, get_common_params, CommonQueryParams
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload", response_model=FaceFeatureResponse, status_code=status.HTTP_201_CREATED)
async def upload_face_image(
    face_data: FaceFeatureUpload,
    current_user: dict = Depends(require_write_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Upload a face image for a person.
    Processes the uploaded image, detects faces, extracts features,
    and stores them in the database.
    """
    try:
        face_manager = FaceManager()
        
        # Validate person exists
        person = db_service.persons.get_person_by_id(face_data.person_id)
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person not found"
            )
        
        # Process face image
        result = face_manager.add_face_image(
            person_id=face_data.person_id,
            image_data=face_data.image_data,
            image_filename=face_data.image_filename,
            set_as_primary=face_data.set_as_primary,
            quality_threshold=face_data.quality_threshold,
            added_by=current_user.get('id')
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process face image"
            )
        
        # Get the created face feature
        feature_id = result['face_feature_id']
        with db_service.face_features._get_db_manager() as db:
            feature = db.face_features.get_by_id(feature_id)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Face feature created but not retrievable"
            )
        
        return FaceFeatureResponse(
            id=feature.id,
            uuid=str(feature.uuid),
            person_id=feature.person_id,
            quality_score=feature.quality_score,
            confidence_score=feature.confidence_score,
            is_primary=feature.is_primary,
            is_active=feature.is_active,
            image_path=feature.image_path,
            bbox=feature.get_bounding_box(),
            landmarks=feature.get_landmarks(),
            created_at=feature.created_at,
            updated_at=feature.updated_at,
            feature_dimension=feature.feature_dimension
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading face image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload face image"
        )

@router.post("/upload-file", response_model=FaceFeatureResponse, status_code=status.HTTP_201_CREATED)
async def upload_face_file(
    person_id: int,
    file: UploadFile = File(...),
    set_as_primary: bool = Query(False, description="Set as primary face"),
    quality_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Quality threshold"),
    current_user: dict = Depends(require_write_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Upload a face image file for a person.
    Processes the uploaded file, detects faces, extracts features,
    and stores them in the database.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Convert to base64 for processing
        image_data = base64.b64encode(file_content).decode('utf-8')
        
        face_manager = FaceManager()
        
        # Validate person exists
        person = db_service.persons.get_person_by_id(person_id)
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person not found"
            )
        
        # Process face image
        result = face_manager.add_face_image(
            person_id=person_id,
            image_data=image_data,
            image_filename=file.filename,
            set_as_primary=set_as_primary,
            quality_threshold=quality_threshold,
            added_by=current_user.get('id')
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process face image"
            )
        
        # Get the created face feature
        feature_id = result['face_feature_id']
        with db_service.face_features._get_db_manager() as db:
            feature = db.face_features.get_by_id(feature_id)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Face feature created but not retrievable"
            )
        
        return FaceFeatureResponse(
            id=feature.id,
            uuid=str(feature.uuid),
            person_id=feature.person_id,
            quality_score=feature.quality_score,
            confidence_score=feature.confidence_score,
            is_primary=feature.is_primary,
            is_active=feature.is_active,
            image_path=feature.image_path,
            bbox=feature.get_bounding_box(),
            landmarks=feature.get_landmarks(),
            created_at=feature.created_at,
            updated_at=feature.updated_at,
            feature_dimension=feature.feature_dimension
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading face file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload face image"
        )

@router.get("/person/{person_id}", response_model=List[FaceFeatureResponse])
async def get_person_faces(
    person_id: int = Depends(validate_person_id),
    include_inactive: bool = Query(False, description="Include inactive features"),
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get all face features for a person.
    Returns all face features associated with the specified person.
    """
    try:
        # Validate person exists
        person = db_service.persons.get_person_by_id(person_id)
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person not found"
            )
        
        # Get face features
        features = db_service.face_features.get_person_features(person_id)
        
        # Filter by active status if needed
        if not include_inactive:
            features = [f for f in features if f.is_active]
        
        # Convert to response format
        face_responses = []
        for feature in features:
            face_responses.append(FaceFeatureResponse(
                id=feature.id,
                uuid=str(feature.uuid),
                person_id=feature.person_id,
                quality_score=feature.quality_score,
                confidence_score=feature.confidence_score,
                is_primary=feature.is_primary,
                is_active=feature.is_active,
                image_path=feature.image_path,
                bbox=feature.get_bounding_box(),
                landmarks=feature.get_landmarks(),
                created_at=feature.created_at,
                updated_at=feature.updated_at,
                feature_dimension=feature.feature_dimension
            ))
        
        return face_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting person faces {person_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve face features"
        )

@router.get("/{feature_id}", response_model=FaceFeatureResponse)
async def get_face_feature(
    feature_id: int,
    current_user: dict = Depends(require_read_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Get a specific face feature by ID.
    Returns detailed information about a face feature.
    """
    try:
        with db_service.face_features._get_db_manager() as db:
            feature = db.face_features.get_by_id(feature_id)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face feature not found"
            )
        
        return FaceFeatureResponse(
            id=feature.id,
            uuid=str(feature.uuid),
            person_id=feature.person_id,
            quality_score=feature.quality_score,
            confidence_score=feature.confidence_score,
            is_primary=feature.is_primary,
            is_active=feature.is_active,
            image_path=feature.image_path,
            bbox=feature.get_bounding_box(),
            landmarks=feature.get_landmarks(),
            created_at=feature.created_at,
            updated_at=feature.updated_at,
            feature_dimension=feature.feature_dimension
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face feature {feature_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve face feature"
        )

@router.put("/{feature_id}", response_model=FaceFeatureResponse)
async def update_face_feature(
    feature_id: int,
    update_data: FaceFeatureUpdate,
    current_user: dict = Depends(require_write_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Update a face feature.
    Updates the specified face feature with the provided data.
    """
    try:
        face_manager = FaceManager()
        
        # Check if feature exists
        with db_service.face_features._get_db_manager() as db:
            existing_feature = db.face_features.get_by_id(feature_id)
        
        if not existing_feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face feature not found"
            )
        
        # Update feature
        update_dict = update_data.dict(exclude_unset=True)
        updated_feature = face_manager.update_face_feature(
            feature_id,
            updated_by=current_user.get('id'),
            **update_dict
        )
        
        if not updated_feature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update face feature"
            )
        
        return FaceFeatureResponse(
            id=updated_feature.id,
            uuid=str(updated_feature.uuid),
            person_id=updated_feature.person_id,
            quality_score=updated_feature.quality_score,
            confidence_score=updated_feature.confidence_score,
            is_primary=updated_feature.is_primary,
            is_active=updated_feature.is_active,
            image_path=updated_feature.image_path,
            bbox=updated_feature.get_bounding_box(),
            landmarks=updated_feature.get_landmarks(),
            created_at=updated_feature.created_at,
            updated_at=updated_feature.updated_at,
            feature_dimension=updated_feature.feature_dimension
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating face feature {feature_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update face feature"
        )

@router.delete("/{feature_id}", response_model=SuccessResponse)
async def delete_face_feature(
    feature_id: int,
    hard_delete: bool = Query(False, description="Permanently delete feature"),
    current_user: dict = Depends(require_write_permission),
    db_service: DatabaseService = Depends(get_db_service)
):
    """
    Delete a face feature.
    Soft deletes (deactivates) or hard deletes a face feature.
    """
    try:
        face_manager = FaceManager()
        
        # Check if feature exists
        with db_service.face_features._get_db_manager() as db:
            feature = db.face_features.get_by_id(feature_id)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face feature not found"
            )
        
        # Delete feature
        success = face_manager.delete_face_feature(
            feature_id,
            deleted_by=current_user.get('id'),
            hard_delete=hard_delete
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete face feature"
            )
        
        delete_type = "permanently deleted" if hard_delete else "deactivated"
        return SuccessResponse(
            message=f"Face feature {delete_type} successfully",
            data={"feature_id": feature_id, "hard_delete": hard_delete}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face feature {feature_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete face feature"
        )