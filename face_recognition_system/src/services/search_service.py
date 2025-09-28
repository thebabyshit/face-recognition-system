"""Advanced search and query service for persons."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

from database.services import get_database_service
from database.models import Person, FaceFeature, AccessLog
from utils.validators import validate_email, validate_employee_id

logger = logging.getLogger(__name__)


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class SortField(Enum):
    """Available sort fields."""
    NAME = "name"
    EMPLOYEE_ID = "employee_id"
    EMAIL = "email"
    DEPARTMENT = "department"
    ACCESS_LEVEL = "access_level"
    CREATED_AT = "created_at"
    LAST_ACCESS = "last_access"


@dataclass
class SearchFilter:
    """Search filter configuration."""
    name: Optional[str] = None
    employee_id: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    access_level_min: Optional[int] = None
    access_level_max: Optional[int] = None
    is_active: Optional[bool] = None
    has_face_features: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    last_access_after: Optional[datetime] = None
    last_access_before: Optional[datetime] = None


@dataclass
class SearchOptions:
    """Search options configuration."""
    limit: int = 50
    offset: int = 0
    sort_field: SortField = SortField.NAME
    sort_order: SortOrder = SortOrder.ASC
    include_face_count: bool = False
    include_last_access: bool = False
    include_statistics: bool = False


class PersonSearchService:
    """Advanced person search and query service."""
    
    def __init__(self):
        self.db_service = get_database_service()
    
    def search_persons(self, 
                      filters: SearchFilter = None,
                      options: SearchOptions = None) -> Dict[str, Any]:
        """
        Advanced person search with filters and options.
        
        Args:
            filters: Search filters to apply
            options: Search options (pagination, sorting, etc.)
            
        Returns:
            Dict: Search results with metadata
        """
        if filters is None:
            filters = SearchFilter()
        if options is None:
            options = SearchOptions()
        
        try:
            with self.db_service.persons._get_db_manager() as db:
                # Build base query
                query = db.session.query(Person)
                
                # Apply filters
                query = self._apply_filters(query, filters, db)
                
                # Get total count before pagination
                total_count = query.count()
                
                # Apply sorting
                query = self._apply_sorting(query, options)
                
                # Apply pagination
                if options.offset > 0:
                    query = query.offset(options.offset)
                if options.limit > 0:
                    query = query.limit(options.limit)
                
                # Execute query
                persons = query.all()
                
                # Enhance results with additional data
                results = []
                for person in persons:
                    person_data = self._build_person_result(person, options, db)
                    results.append(person_data)
                
                # Build response
                response = {
                    'results': results,
                    'total_count': total_count,
                    'returned_count': len(results),
                    'offset': options.offset,
                    'limit': options.limit,
                    'has_more': (options.offset + len(results)) < total_count
                }
                
                # Add statistics if requested
                if options.include_statistics:
                    response['statistics'] = self._get_search_statistics(filters, db)
                
                return response
                
        except Exception as e:
            logger.error(f"Error in person search: {e}")
            return {
                'results': [],
                'total_count': 0,
                'returned_count': 0,
                'offset': 0,
                'limit': options.limit if options else 50,
                'has_more': False,
                'error': str(e)
            }
    
    def quick_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Quick search across multiple fields.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            
        Returns:
            List[Dict]: Quick search results
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
        try:
            with self.db_service.persons._get_db_manager() as db:
                # Search across multiple fields
                search_query = db.session.query(Person).filter(
                    db.session.query(Person).filter(
                        (Person.name.ilike(f"%{query}%")) |
                        (Person.employee_id.ilike(f"%{query}%")) |
                        (Person.email.ilike(f"%{query}%")) |
                        (Person.department.ilike(f"%{query}%")) |
                        (Person.position.ilike(f"%{query}%"))
                    ).exists()
                ).filter(Person.is_active == True).limit(limit)
                
                persons = search_query.all()
                
                results = []
                for person in persons:
                    # Calculate relevance score
                    relevance = self._calculate_relevance(person, query)
                    
                    result = {
                        'id': person.id,
                        'name': person.name,
                        'employee_id': person.employee_id,
                        'email': person.email,
                        'department': person.department,
                        'position': person.position,
                        'access_level': person.access_level,
                        'relevance_score': relevance
                    }
                    results.append(result)
                
                # Sort by relevance
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in quick search: {e}")
            return []
    
    def get_department_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics by department.
        
        Returns:
            Dict: Department statistics
        """
        try:
            with self.db_service.persons._get_db_manager() as db:
                # Get department counts
                dept_query = db.session.query(
                    Person.department,
                    db.session.query(Person).filter(
                        Person.department == Person.department,
                        Person.is_active == True
                    ).count().label('active_count'),
                    db.session.query(Person).filter(
                        Person.department == Person.department
                    ).count().label('total_count')
                ).filter(Person.department.isnot(None)).group_by(Person.department)
                
                departments = {}
                for dept, active_count, total_count in dept_query:
                    departments[dept] = {
                        'active_count': active_count,
                        'total_count': total_count,
                        'inactive_count': total_count - active_count
                    }
                
                # Get persons with face features by department
                features_query = db.session.query(
                    Person.department,
                    db.session.query(Person).join(FaceFeature).filter(
                        Person.department == Person.department,
                        Person.is_active == True,
                        FaceFeature.is_active == True
                    ).distinct().count().label('with_features')
                ).filter(Person.department.isnot(None)).group_by(Person.department)
                
                for dept, with_features in features_query:
                    if dept in departments:
                        departments[dept]['with_face_features'] = with_features
                
                return {
                    'departments': departments,
                    'total_departments': len(departments),
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting department summary: {e}")
            return {'departments': {}, 'total_departments': 0}
    
    def get_access_level_distribution(self) -> Dict[str, Any]:
        """
        Get distribution of persons by access level.
        
        Returns:
            Dict: Access level distribution
        """
        try:
            with self.db_service.persons._get_db_manager() as db:
                distribution = {}
                
                for level in range(11):  # 0-10
                    count = db.session.query(Person).filter(
                        Person.access_level == level,
                        Person.is_active == True
                    ).count()
                    
                    if count > 0:
                        distribution[str(level)] = count
                
                return {
                    'distribution': distribution,
                    'total_levels': len(distribution),
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting access level distribution: {e}")
            return {'distribution': {}, 'total_levels': 0}
    
    def find_similar_persons(self, person_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find persons similar to the given person.
        
        Args:
            person_id: ID of the reference person
            limit: Maximum similar persons to return
            
        Returns:
            List[Dict]: Similar persons
        """
        try:
            with self.db_service.persons._get_db_manager() as db:
                reference_person = db.session.query(Person).filter_by(id=person_id).first()
                if not reference_person:
                    return []
                
                # Find similar persons based on department, position, access level
                similar_query = db.session.query(Person).filter(
                    Person.id != person_id,
                    Person.is_active == True
                )
                
                # Add similarity criteria
                if reference_person.department:
                    similar_query = similar_query.filter(Person.department == reference_person.department)
                
                if reference_person.position:
                    similar_query = similar_query.filter(Person.position == reference_person.position)
                
                # Access level within ±1
                similar_query = similar_query.filter(
                    Person.access_level.between(
                        max(0, reference_person.access_level - 1),
                        min(10, reference_person.access_level + 1)
                    )
                )
                
                similar_persons = similar_query.limit(limit).all()
                
                results = []
                for person in similar_persons:
                    similarity_score = self._calculate_similarity_score(reference_person, person)
                    
                    result = {
                        'id': person.id,
                        'name': person.name,
                        'employee_id': person.employee_id,
                        'department': person.department,
                        'position': person.position,
                        'access_level': person.access_level,
                        'similarity_score': similarity_score
                    }
                    results.append(result)
                
                # Sort by similarity score
                results.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                return results
                
        except Exception as e:
            logger.error(f"Error finding similar persons: {e}")
            return []
    
    def _apply_filters(self, query, filters: SearchFilter, db):
        """Apply search filters to query."""
        if filters.name:
            query = query.filter(Person.name.ilike(f"%{filters.name}%"))
        
        if filters.employee_id:
            query = query.filter(Person.employee_id.ilike(f"%{filters.employee_id}%"))
        
        if filters.email:
            query = query.filter(Person.email.ilike(f"%{filters.email}%"))
        
        if filters.department:
            query = query.filter(Person.department.ilike(f"%{filters.department}%"))
        
        if filters.position:
            query = query.filter(Person.position.ilike(f"%{filters.position}%"))
        
        if filters.access_level_min is not None:
            query = query.filter(Person.access_level >= filters.access_level_min)
        
        if filters.access_level_max is not None:
            query = query.filter(Person.access_level <= filters.access_level_max)
        
        if filters.is_active is not None:
            query = query.filter(Person.is_active == filters.is_active)
        
        if filters.created_after:
            query = query.filter(Person.created_at >= filters.created_after)
        
        if filters.created_before:
            query = query.filter(Person.created_at <= filters.created_before)
        
        if filters.has_face_features is not None:
            if filters.has_face_features:
                query = query.join(FaceFeature).filter(FaceFeature.is_active == True)
            else:
                query = query.outerjoin(FaceFeature).filter(FaceFeature.id.is_(None))
        
        return query
    
    def _apply_sorting(self, query, options: SearchOptions):
        """Apply sorting to query."""
        sort_column = getattr(Person, options.sort_field.value, Person.name)
        
        if options.sort_order == SortOrder.DESC:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        return query
    
    def _build_person_result(self, person: Person, options: SearchOptions, db) -> Dict[str, Any]:
        """Build enhanced person result."""
        result = {
            'id': person.id,
            'name': person.name,
            'employee_id': person.employee_id,
            'email': person.email,
            'department': person.department,
            'position': person.position,
            'access_level': person.access_level,
            'is_active': person.is_active,
            'created_at': person.created_at.isoformat() if person.created_at else None
        }
        
        if options.include_face_count:
            face_count = db.session.query(FaceFeature).filter(
                FaceFeature.person_id == person.id,
                FaceFeature.is_active == True
            ).count()
            result['face_feature_count'] = face_count
        
        if options.include_last_access:
            last_access = db.session.query(AccessLog).filter(
                AccessLog.person_id == person.id
            ).order_by(AccessLog.timestamp.desc()).first()
            
            result['last_access'] = last_access.timestamp.isoformat() if last_access else None
        
        return result
    
    def _calculate_relevance(self, person: Person, query: str) -> float:
        """Calculate relevance score for search result."""
        score = 0.0
        query_lower = query.lower()
        
        # Exact matches get higher scores
        if person.name and query_lower in person.name.lower():
            score += 10.0
        if person.employee_id and query_lower in person.employee_id.lower():
            score += 8.0
        if person.email and query_lower in person.email.lower():
            score += 6.0
        if person.department and query_lower in person.department.lower():
            score += 4.0
        if person.position and query_lower in person.position.lower():
            score += 2.0
        
        # Boost for active persons
        if person.is_active:
            score += 1.0
        
        return score
    
    def _calculate_similarity_score(self, person1: Person, person2: Person) -> float:
        """Calculate similarity score between two persons."""
        score = 0.0
        
        # Department match
        if person1.department == person2.department:
            score += 3.0
        
        # Position match
        if person1.position == person2.position:
            score += 2.0
        
        # Access level similarity
        level_diff = abs(person1.access_level - person2.access_level)
        if level_diff == 0:
            score += 2.0
        elif level_diff == 1:
            score += 1.0
        
        return score
    
    def _get_search_statistics(self, filters: SearchFilter, db) -> Dict[str, Any]:
        """Get statistics for current search."""
        stats = {}
        
        try:
            # Total active persons
            stats['total_active'] = db.session.query(Person).filter(Person.is_active == True).count()
            
            # Persons with face features
            stats['with_features'] = db.session.query(Person).join(FaceFeature).filter(
                Person.is_active == True,
                FaceFeature.is_active == True
            ).distinct().count()
            
            # Department count
            stats['department_count'] = db.session.query(Person.department).filter(
                Person.department.isnot(None),
                Person.is_active == True
            ).distinct().count()
            
        except Exception as e:
            logger.error(f"Error calculating search statistics: {e}")
        
        return stats