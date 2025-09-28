"""Advanced search service for person management."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import re
from dataclasses import dataclass
from enum import Enum

from ..database.services import get_database_service
from ..database.models import Person

logger = logging.getLogger(__name__)


class SearchOperator(Enum):
    """Search operators for advanced queries."""
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class SearchFilter:
    """Search filter definition."""
    field: str
    operator: SearchOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class SortCriteria:
    """Sort criteria definition."""
    field: str
    order: SortOrder = SortOrder.ASC


@dataclass
class SearchQuery:
    """Advanced search query definition."""
    filters: List[SearchFilter]
    sort_criteria: Optional[List[SortCriteria]] = None
    limit: int = 50
    offset: int = 0
    include_inactive: bool = False


class AdvancedPersonSearch:
    """Advanced search service for persons."""
    
    def __init__(self):
        self.db_service = get_database_service()
        
        # Define searchable fields and their types
        self.searchable_fields = {
            'id': int,
            'name': str,
            'employee_id': str,
            'email': str,
            'phone': str,
            'department': str,
            'position': str,
            'access_level': int,
            'is_active': bool,
            'created_at': datetime,
            'updated_at': datetime,
            'notes': str
        }
        
        # Define