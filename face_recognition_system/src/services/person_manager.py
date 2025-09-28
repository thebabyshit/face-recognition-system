"""Person management service implementation."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import re
from pathlib import Path

from ..database.services import get_database_service
from ..database.models import Person, FaceFeature
from ..utils.validators import validate_email, validate_phone, validate_employee_id
from ..utils.exceptions import (
    PersonNotFoundError, 
    DuplicatePersonError, 
    ValidationError,
    PermissionError
)

logger = logging.getLogger(__name__)


class PersonManager:
    """Person management service with comprehensive CRUD operations."""
    
    def __init__(self):
        self.db_service = get_database_service()
        self._operation_log = []
    
    def add_person(self, 
                   name: str,
                   employee_id: Optional[str] = None,
                   email: Optional[str] = None,
                   phone: Optional[str] = None,
                   department: Optional[str] = None,
                   position: Optional[str] = None,
                   access_level: int = 1,
                   notes: Optional[str] = None,
                   created_by: Optional[int] = None,
                   **kwargs) -> Person:
        """
        Add a new person to the system.
        
        Args:
            name: Full name of the person
            employee_id: Unique employee identifier
            email: Email address
            phone: Phone number
            department: Department name
            position: Job position
            access_level: Access level (0-10)
            notes: Additional notes
            created_by: ID of user creating this person
            **kwargs: Additional person attributes
            
        Returns:
            Person: Created person object
            
        Raises:
            ValidationError: If input data is invalid
            DuplicatePersonError: If person already exists
        """
        try:
            # Validate input data
            self._validate_person_data(
                name=name,
                employee_id=employee_id,
                email=email,
                phone=phone,
                access_level=access_level
            )
            
            # Check for duplicates
            self._check_duplicates(employee_id, email)
            
            # Create person
            person = self.db_service.persons.create_person(
                name=name.strip(),
                employee_id=employee_id,
                email=email.lower() if email else None,
                phone=phone,
                department=department,
                position=position,
                access_level=access_level,
                notes=notes,
                created_by=created_by,
                **kwargs
            )
            
            if not person:
                raise RuntimeError("Failed to create person in database")
            
            # Log operation
            self._log_operation(
                action="CREATE",
                person_id=person.id,
                details=f"Created person: {name} ({employee_id})",
                user_id=created_by
            )
            
            logger.info(f"Successfully created person: {person.name} (ID: {person.id})")
            return person
            
        except Exception as e:
            logger.error(f"Error creating person {name}: {e}")
            raise
    
    def update_person(self,
                     person_id: int,
                     updated_by: int = None,
                     **updates) -> Person:
        """
        Update an existing person's information.
        
        Args:
            person_id: ID of person to update
            updated_by: ID of user making the update
            **updates: Fields to update
            
        Returns:
            Person: Updated person object
            
        Raises:
            PersonNotFoundError: If person doesn't exist
            ValidationError: If update data is invalid
            DuplicatePersonError: If update would create duplicate
        """
        try:
            # Get existing person
            person = self.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            # Validate updates
            if updates:
                self._validate_person_updates(person, **updates)
            
            # Check for duplicates if email/employee_id changed
            if 'email' in updates or 'employee_id' in updates:
                self._check_duplicates(
                    employee_id=updates.get('employee_id', person.employee_id),
                    email=updates.get('email', person.email),
                    exclude_person_id=person_id
                )
            
            # Add metadata
            updates['updated_by'] = updated_by
            
            # Update person
            updated_person = self.db_service.persons.update_person(person_id, **updates)
            
            if not updated_person:
                raise RuntimeError("Failed to update person in database")
            
            # Log operation
            self._log_operation(
                action="UPDATE",
                person_id=person_id,
                details=f"Updated person: {person.name}",
                user_id=updated_by
            )
            
            logger.info(f"Successfully updated person: {updated_person.name} (ID: {person_id})")
            return updated_person
            
        except Exception as e:
            logger.error(f"Error updating person {person_id}: {e}")
            raise
    
    def delete_person(self, person_id: int, deleted_by: int = None, hard_delete: bool = False) -> bool:
        """
        Delete a person from the system.
        
        Args:
            person_id: ID of person to delete
            deleted_by: ID of user performing deletion
            hard_delete: If True, permanently delete; if False, soft delete
            
        Returns:
            bool: True if deletion successful
            
        Raises:
            PersonNotFoundError: If person doesn't exist
            PermissionError: If person cannot be deleted
        """
        try:
            # Get existing person
            person = self.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            # Check if person can be deleted
            self._check_deletion_permissions(person)
            
            if hard_delete:
                # Permanently delete person and all related data
                success = self._hard_delete_person(person_id)
            else:
                # Soft delete (deactivate)
                success = self.db_service.persons.deactivate_person(person_id)
            
            if success:
                # Log operation
                delete_type = "HARD_DELETE" if hard_delete else "SOFT_DELETE"
                self._log_operation(
                    action=delete_type,
                    person_id=person_id,
                    details=f"Deleted person: {person.name}",
                    user_id=deleted_by
                )
                
                logger.info(f"Successfully deleted person: {person.name} (ID: {person_id})")
                return True
            else:
                raise RuntimeError("Failed to delete person")
                
        except Exception as e:
            logger.error(f"Error deleting person {person_id}: {e}")
            raise
    
    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        return self.db_service.persons.get_person_by_id(person_id)
    
    def get_person_by_employee_id(self, employee_id: str) -> Optional[Person]:
        """Get person by employee ID."""
        return self.db_service.persons.get_person_by_employee_id(employee_id)
    
    def get_person_by_email(self, email: str) -> Optional[Person]:
        """Get person by email address."""
        return self.db_service.persons.get_person_by_email(email.lower())
    
    def search_persons(self, 
                      query: str = None,
                      department: str = None,
                      access_level: int = None,
                      is_active: bool = None,
                      limit: int = 50,
                      offset: int = 0) -> List[Person]:
        """
        Search persons with multiple criteria.
        
        Args:
            query: Search query for name/employee_id
            department: Filter by department
            access_level: Filter by access level
            is_active: Filter by active status
            limit: Maximum results to return
            offset: Number of results to skip
            
        Returns:
            List[Person]: Matching persons
        """
        try:
            # Start with all persons or search by query
            if query:
                persons = self.db_service.persons.search_persons(query, limit=limit*2)  # Get more for filtering
            else:
                persons = self.db_service.persons.get_active_persons(limit=limit*2)
            
            # Apply additional filters
            filtered_persons = []
            for person in persons:
                if department and person.department != department:
                    continue
                if access_level is not None and person.access_level != access_level:
                    continue
                if is_active is not None and person.is_active != is_active:
                    continue
                
                filtered_persons.append(person)
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            return filtered_persons[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"Error searching persons: {e}")
            return []
    
    def get_person_statistics(self) -> Dict[str, Any]:
        """
        Get person statistics.
        
        Returns:
            Dict: Statistics about persons in the system
        """
        try:
            stats = {
                'total_persons': 0,
                'active_persons': 0,
                'inactive_persons': 0,
                'persons_with_features': 0,
                'departments': {},
                'access_levels': {}
            }
            
            # Get all persons
            all_persons = self.db_service.persons.get_all()
            stats['total_persons'] = len(all_persons)
            
            # Calculate statistics
            for person in all_persons:
                if person.is_active:
                    stats['active_persons'] += 1
                else:
                    stats['inactive_persons'] += 1
                
                # Department statistics
                dept = person.department or 'Unknown'
                stats['departments'][dept] = stats['departments'].get(dept, 0) + 1
                
                # Access level statistics
                level = person.access_level
                stats['access_levels'][level] = stats['access_levels'].get(level, 0) + 1
            
            # Get persons with features
            persons_with_features = self.db_service.persons.get_persons_with_features()
            stats['persons_with_features'] = len(persons_with_features)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting person statistics: {e}")
            return {}
    
    def bulk_import_persons(self, persons_data: List[Dict], imported_by: int = None) -> Dict[str, Any]:
        """
        Bulk import persons from data list.
        
        Args:
            persons_data: List of person data dictionaries
            imported_by: ID of user performing import
            
        Returns:
            Dict: Import results with success/failure counts
        """
        results = {
            'total': len(persons_data),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            for i, person_data in enumerate(persons_data):
                try:
                    person_data['created_by'] = imported_by
                    person = self.add_person(**person_data)
                    results['success'] += 1
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'row': i + 1,
                        'data': person_data,
                        'error': str(e)
                    })
                    logger.warning(f"Failed to import person at row {i+1}: {e}")
            
            # Log bulk operation
            self._log_operation(
                action="BULK_IMPORT",
                details=f"Imported {results['success']}/{results['total']} persons",
                user_id=imported_by
            )
            
            logger.info(f"Bulk import completed: {results['success']}/{results['total']} successful")
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk import: {e}")
            results['errors'].append({'error': str(e)})
            return results
    
    def _validate_person_data(self, **data):
        """Validate person data."""
        name = data.get('name')
        if not name or not name.strip():
            raise ValidationError("Name is required")
        
        if len(name.strip()) < 2:
            raise ValidationError("Name must be at least 2 characters")
        
        employee_id = data.get('employee_id')
        if employee_id and not validate_employee_id(employee_id):
            raise ValidationError("Invalid employee ID format")
        
        email = data.get('email')
        if email and not validate_email(email):
            raise ValidationError("Invalid email format")
        
        phone = data.get('phone')
        if phone and not validate_phone(phone):
            raise ValidationError("Invalid phone number format")
        
        access_level = data.get('access_level', 1)
        if not isinstance(access_level, int) or access_level < 0 or access_level > 10:
            raise ValidationError("Access level must be between 0 and 10")
    
    def _validate_person_updates(self, person: Person, **updates):
        """Validate person update data."""
        # Only validate fields that are being updated
        validation_data = {}
        
        for field in ['name', 'employee_id', 'email', 'phone', 'access_level']:
            if field in updates:
                validation_data[field] = updates[field]
        
        if validation_data:
            self._validate_person_data(**validation_data)
    
    def _check_duplicates(self, employee_id: str = None, email: str = None, exclude_person_id: int = None):
        """Check for duplicate employee_id or email."""
        if employee_id:
            existing = self.db_service.persons.get_person_by_employee_id(employee_id)
            if existing and (not exclude_person_id or existing.id != exclude_person_id):
                raise DuplicatePersonError(f"Employee ID '{employee_id}' already exists")
        
        if email:
            existing = self.get_person_by_email(email.lower())
            if existing and (not exclude_person_id or existing.id != exclude_person_id):
                raise DuplicatePersonError(f"Email '{email}' already exists")
    
    def _check_deletion_permissions(self, person: Person):
        """Check if person can be deleted."""
        # Check if person has recent access logs
        recent_logs = self.db_service.access_logs.get_person_access_history(person.id, limit=1)
        if recent_logs:
            # Allow deletion but warn
            logger.warning(f"Deleting person {person.name} who has access history")
        
        # Add more business rules as needed
        # For example: prevent deletion of admin users, users with active sessions, etc.
    
    def _hard_delete_person(self, person_id: int) -> bool:
        """Permanently delete person and all related data."""
        try:
            # This would cascade delete related records
            # In practice, you might want to archive data instead
            with self.db_service.persons._get_db_manager() as db:
                person = db.persons.get_by_id(person_id)
                if person:
                    db.session.delete(person)
                    return True
                return False
        except Exception as e:
            logger.error(f"Error in hard delete: {e}")
            return False
    
    def _log_operation(self, action: str, person_id: int = None, details: str = None, user_id: int = None):
        """Log person management operations."""
        log_entry = {
            'timestamp': datetime.now(timezone.utc),
            'action': action,
            'person_id': person_id,
            'details': details,
            'user_id': user_id
        }
        
        self._operation_log.append(log_entry)
        
        # Also log to system logs if available
        try:
            # This would integrate with system logging
            logger.info(f"Person operation: {action} - {details}")
        except Exception:
            pass
    
    def get_operation_log(self, limit: int = 100) -> List[Dict]:
        """Get recent operation log."""
        return self._operation_log[-limit:]
    
    def activate_person(self, person_id: int, activated_by: int = None) -> bool:
        """
        Activate a deactivated person.
        
        Args:
            person_id: ID of person to activate
            activated_by: ID of user performing activation
            
        Returns:
            bool: True if activation successful
        """
        try:
            person = self.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            if person.is_active:
                logger.warning(f"Person {person_id} is already active")
                return True
            
            success = self.db_service.persons.activate_person(person_id)
            
            if success:
                self._log_operation(
                    action="ACTIVATE",
                    person_id=person_id,
                    details=f"Activated person: {person.name}",
                    user_id=activated_by
                )
                logger.info(f"Successfully activated person: {person.name} (ID: {person_id})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error activating person {person_id}: {e}")
            raise
    
    def deactivate_person(self, person_id: int, deactivated_by: int = None, reason: str = None) -> bool:
        """
        Deactivate a person (soft delete).
        
        Args:
            person_id: ID of person to deactivate
            deactivated_by: ID of user performing deactivation
            reason: Reason for deactivation
            
        Returns:
            bool: True if deactivation successful
        """
        try:
            person = self.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            if not person.is_active:
                logger.warning(f"Person {person_id} is already inactive")
                return True
            
            success = self.db_service.persons.deactivate_person(person_id)
            
            if success:
                details = f"Deactivated person: {person.name}"
                if reason:
                    details += f" (Reason: {reason})"
                
                self._log_operation(
                    action="DEACTIVATE",
                    person_id=person_id,
                    details=details,
                    user_id=deactivated_by
                )
                logger.info(f"Successfully deactivated person: {person.name} (ID: {person_id})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deactivating person {person_id}: {e}")
            raise
    
    def get_persons_by_department(self, department: str, include_inactive: bool = False) -> List[Person]:
        """
        Get all persons in a specific department.
        
        Args:
            department: Department name
            include_inactive: Whether to include inactive persons
            
        Returns:
            List[Person]: Persons in the department
        """
        try:
            all_persons = self.db_service.persons.get_all()
            
            filtered_persons = []
            for person in all_persons:
                if person.department == department:
                    if include_inactive or person.is_active:
                        filtered_persons.append(person)
            
            return filtered_persons
            
        except Exception as e:
            logger.error(f"Error getting persons by department {department}: {e}")
            return []
    
    def get_persons_by_access_level(self, access_level: int, include_inactive: bool = False) -> List[Person]:
        """
        Get all persons with a specific access level.
        
        Args:
            access_level: Access level to filter by
            include_inactive: Whether to include inactive persons
            
        Returns:
            List[Person]: Persons with the access level
        """
        try:
            all_persons = self.db_service.persons.get_all()
            
            filtered_persons = []
            for person in all_persons:
                if person.access_level == access_level:
                    if include_inactive or person.is_active:
                        filtered_persons.append(person)
            
            return filtered_persons
            
        except Exception as e:
            logger.error(f"Error getting persons by access level {access_level}: {e}")
            return []
    
    def update_access_level(self, person_id: int, new_access_level: int, updated_by: int = None, reason: str = None) -> bool:
        """
        Update a person's access level.
        
        Args:
            person_id: ID of person to update
            new_access_level: New access level (0-10)
            updated_by: ID of user making the change
            reason: Reason for access level change
            
        Returns:
            bool: True if update successful
        """
        try:
            person = self.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            if not isinstance(new_access_level, int) or new_access_level < 0 or new_access_level > 10:
                raise ValidationError("Access level must be between 0 and 10")
            
            old_access_level = person.access_level
            
            updated_person = self.update_person(
                person_id=person_id,
                access_level=new_access_level,
                updated_by=updated_by
            )
            
            if updated_person:
                details = f"Changed access level from {old_access_level} to {new_access_level}"
                if reason:
                    details += f" (Reason: {reason})"
                
                self._log_operation(
                    action="ACCESS_LEVEL_CHANGE",
                    person_id=person_id,
                    details=details,
                    user_id=updated_by
                )
                
                logger.info(f"Updated access level for {person.name}: {old_access_level} -> {new_access_level}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating access level for person {person_id}: {e}")
            raise
    
    def get_duplicate_candidates(self) -> List[Dict[str, Any]]:
        """
        Find potential duplicate persons based on name similarity.
        
        Returns:
            List[Dict]: Potential duplicate groups
        """
        try:
            all_persons = self.db_service.persons.get_all()
            duplicates = []
            
            # Simple name-based duplicate detection
            name_groups = {}
            for person in all_persons:
                # Normalize name for comparison
                normalized_name = self._normalize_name(person.name)
                
                if normalized_name not in name_groups:
                    name_groups[normalized_name] = []
                name_groups[normalized_name].append(person)
            
            # Find groups with multiple persons
            for normalized_name, persons in name_groups.items():
                if len(persons) > 1:
                    duplicates.append({
                        'normalized_name': normalized_name,
                        'persons': [
                            {
                                'id': p.id,
                                'name': p.name,
                                'employee_id': p.employee_id,
                                'email': p.email,
                                'department': p.department,
                                'is_active': p.is_active
                            }
                            for p in persons
                        ],
                        'count': len(persons)
                    })
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicate candidates: {e}")
            return []
    
    def merge_persons(self, primary_person_id: int, duplicate_person_id: int, merged_by: int = None) -> bool:
        """
        Merge two person records (move data from duplicate to primary).
        
        Args:
            primary_person_id: ID of person to keep
            duplicate_person_id: ID of person to merge and remove
            merged_by: ID of user performing merge
            
        Returns:
            bool: True if merge successful
        """
        try:
            primary_person = self.get_person_by_id(primary_person_id)
            duplicate_person = self.get_person_by_id(duplicate_person_id)
            
            if not primary_person:
                raise PersonNotFoundError(f"Primary person {primary_person_id} not found")
            if not duplicate_person:
                raise PersonNotFoundError(f"Duplicate person {duplicate_person_id} not found")
            
            # Merge face features
            duplicate_features = self.db_service.face_features.get_person_features(duplicate_person_id)
            for feature in duplicate_features:
                # Update feature to point to primary person
                self.db_service.face_features.update_feature_person(feature.id, primary_person_id)
            
            # Merge access logs (update person_id in access logs)
            # This would be implemented based on your access log structure
            
            # Merge any other related data
            # Add more merge logic as needed
            
            # Deactivate the duplicate person
            self.deactivate_person(duplicate_person_id, deactivated_by=merged_by, reason="Merged with primary record")
            
            self._log_operation(
                action="MERGE",
                person_id=primary_person_id,
                details=f"Merged person {duplicate_person.name} (ID: {duplicate_person_id}) into {primary_person.name}",
                user_id=merged_by
            )
            
            logger.info(f"Successfully merged person {duplicate_person_id} into {primary_person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging persons {duplicate_person_id} -> {primary_person_id}: {e}")
            raise
    
    def export_persons(self, format: str = 'csv', include_inactive: bool = False) -> str:
        """
        Export persons data to specified format.
        
        Args:
            format: Export format ('csv', 'json', 'xlsx')
            include_inactive: Whether to include inactive persons
            
        Returns:
            str: File path of exported data
        """
        try:
            import csv
            import json
            from datetime import datetime
            
            # Get persons data
            if include_inactive:
                persons = self.db_service.persons.get_all()
            else:
                persons = self.db_service.persons.get_active_persons()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'csv':
                filename = f"persons_export_{timestamp}.csv"
                filepath = Path("exports") / filename
                filepath.parent.mkdir(exist_ok=True)
                
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['id', 'name', 'employee_id', 'email', 'phone', 'department', 
                                'position', 'access_level', 'is_active', 'created_at', 'updated_at']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for person in persons:
                        writer.writerow({
                            'id': person.id,
                            'name': person.name,
                            'employee_id': person.employee_id,
                            'email': person.email,
                            'phone': person.phone,
                            'department': person.department,
                            'position': person.position,
                            'access_level': person.access_level,
                            'is_active': person.is_active,
                            'created_at': person.created_at.isoformat() if person.created_at else None,
                            'updated_at': person.updated_at.isoformat() if person.updated_at else None
                        })
                
                return str(filepath)
                
            elif format.lower() == 'json':
                filename = f"persons_export_{timestamp}.json"
                filepath = Path("exports") / filename
                filepath.parent.mkdir(exist_ok=True)
                
                persons_data = []
                for person in persons:
                    persons_data.append({
                        'id': person.id,
                        'name': person.name,
                        'employee_id': person.employee_id,
                        'email': person.email,
                        'phone': person.phone,
                        'department': person.department,
                        'position': person.position,
                        'access_level': person.access_level,
                        'is_active': person.is_active,
                        'created_at': person.created_at.isoformat() if person.created_at else None,
                        'updated_at': person.updated_at.isoformat() if person.updated_at else None
                    })
                
                with open(filepath, 'w', encoding='utf-8') as jsonfile:
                    json.dump(persons_data, jsonfile, indent=2, ensure_ascii=False)
                
                return str(filepath)
            
            else:
                raise ValidationError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting persons: {e}")
            raise
    
    def get_person_access_summary(self, person_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get access summary for a person over specified days.
        
        Args:
            person_id: ID of person
            days: Number of days to look back
            
        Returns:
            Dict: Access summary statistics
        """
        try:
            person = self.get_person_by_id(person_id)
            if not person:
                raise PersonNotFoundError(f"Person with ID {person_id} not found")
            
            # Get access logs for the person
            access_logs = self.db_service.access_logs.get_person_access_history(
                person_id, 
                days=days
            )
            
            summary = {
                'person_id': person_id,
                'person_name': person.name,
                'period_days': days,
                'total_attempts': len(access_logs),
                'successful_attempts': 0,
                'failed_attempts': 0,
                'unique_locations': set(),
                'access_by_day': {},
                'access_by_hour': {},
                'last_access': None,
                'most_accessed_location': None
            }
            
            location_counts = {}
            
            for log in access_logs:
                if log.access_granted:
                    summary['successful_attempts'] += 1
                else:
                    summary['failed_attempts'] += 1
                
                # Track locations
                if log.location_id:
                    summary['unique_locations'].add(log.location_id)
                    location_counts[log.location_id] = location_counts.get(log.location_id, 0) + 1
                
                # Track by day
                day_key = log.timestamp.date().isoformat()
                summary['access_by_day'][day_key] = summary['access_by_day'].get(day_key, 0) + 1
                
                # Track by hour
                hour_key = log.timestamp.hour
                summary['access_by_hour'][hour_key] = summary['access_by_hour'].get(hour_key, 0) + 1
                
                # Update last access
                if not summary['last_access'] or log.timestamp > summary['last_access']:
                    summary['last_access'] = log.timestamp
            
            # Convert set to count
            summary['unique_locations'] = len(summary['unique_locations'])
            
            # Find most accessed location
            if location_counts:
                summary['most_accessed_location'] = max(location_counts, key=location_counts.get)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting access summary for person {person_id}: {e}")
            return {}
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for duplicate detection."""
        if not name:
            return ""
        
        # Convert to lowercase, remove extra spaces, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', name.lower())
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def validate_bulk_import_data(self, persons_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate bulk import data before actual import.
        
        Args:
            persons_data: List of person data dictionaries
            
        Returns:
            Dict: Validation results
        """
        results = {
            'total': len(persons_data),
            'valid': 0,
            'invalid': 0,
            'errors': [],
            'warnings': []
        }
        
        seen_employee_ids = set()
        seen_emails = set()
        
        for i, person_data in enumerate(persons_data):
            row_errors = []
            row_warnings = []
            
            try:
                # Validate required fields
                if not person_data.get('name', '').strip():
                    row_errors.append("Name is required")
                
                # Validate formats
                employee_id = person_data.get('employee_id')
                if employee_id:
                    if employee_id in seen_employee_ids:
                        row_errors.append(f"Duplicate employee_id in import data: {employee_id}")
                    else:
                        seen_employee_ids.add(employee_id)
                        
                        # Check if exists in database
                        existing = self.get_person_by_employee_id(employee_id)
                        if existing:
                            row_errors.append(f"Employee_id already exists in database: {employee_id}")
                
                email = person_data.get('email')
                if email:
                    if not validate_email(email):
                        row_errors.append("Invalid email format")
                    elif email.lower() in seen_emails:
                        row_errors.append(f"Duplicate email in import data: {email}")
                    else:
                        seen_emails.add(email.lower())
                        
                        # Check if exists in database
                        existing = self.get_person_by_email(email)
                        if existing:
                            row_errors.append(f"Email already exists in database: {email}")
                
                phone = person_data.get('phone')
                if phone and not validate_phone(phone):
                    row_errors.append("Invalid phone format")
                
                access_level = person_data.get('access_level', 1)
                if not isinstance(access_level, int) or access_level < 0 or access_level > 10:
                    row_errors.append("Access level must be between 0 and 10")
                
                # Check for potential duplicates by name
                name = person_data.get('name', '').strip()
                if name:
                    normalized_name = self._normalize_name(name)
                    existing_persons = self.search_persons(query=name, limit=5)
                    for existing in existing_persons:
                        if self._normalize_name(existing.name) == normalized_name:
                            row_warnings.append(f"Potential duplicate name: {existing.name} (ID: {existing.id})")
                
                if row_errors:
                    results['invalid'] += 1
                    results['errors'].append({
                        'row': i + 1,
                        'data': person_data,
                        'errors': row_errors
                    })
                else:
                    results['valid'] += 1
                
                if row_warnings:
                    results['warnings'].append({
                        'row': i + 1,
                        'data': person_data,
                        'warnings': row_warnings
                    })
                    
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append({
                    'row': i + 1,
                    'data': person_data,
                    'errors': [f"Validation error: {str(e)}"]
                })
        
        return results