"""Data validation utilities."""

import re
from typing import Optional


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid email format
    """
    if not email:
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        bool: True if valid phone format
    """
    if not phone:
        return False
    
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    
    # Check if it's all digits and reasonable length
    return cleaned.isdigit() and 7 <= len(cleaned) <= 15


def validate_employee_id(employee_id: str) -> bool:
    """
    Validate employee ID format.
    
    Args:
        employee_id: Employee ID to validate
        
    Returns:
        bool: True if valid employee ID format
    """
    if not employee_id:
        return False
    
    # Employee ID should be alphanumeric, 3-20 characters
    pattern = r'^[A-Za-z0-9]{3,20}$'
    return bool(re.match(pattern, employee_id))


def validate_name(name: str) -> bool:
    """
    Validate person name.
    
    Args:
        name: Name to validate
        
    Returns:
        bool: True if valid name
    """
    if not name or not name.strip():
        return False
    
    # Name should be 2-100 characters, letters, spaces, and common punctuation
    cleaned_name = name.strip()
    if len(cleaned_name) < 2 or len(cleaned_name) > 100:
        return False
    
    # Allow letters, spaces, hyphens, apostrophes, and dots
    pattern = r"^[a-zA-Z\u4e00-\u9fff\s\-'\.]+$"
    return bool(re.match(pattern, cleaned_name))


def validate_access_level(access_level: int) -> bool:
    """
    Validate access level.
    
    Args:
        access_level: Access level to validate
        
    Returns:
        bool: True if valid access level
    """
    return isinstance(access_level, int) and 0 <= access_level <= 10


def validate_department(department: str) -> bool:
    """
    Validate department name.
    
    Args:
        department: Department name to validate
        
    Returns:
        bool: True if valid department name
    """
    if not department:
        return True  # Department is optional
    
    # Department should be 2-50 characters
    cleaned = department.strip()
    return 2 <= len(cleaned) <= 50


def sanitize_string(text: str, max_length: int = None) -> str:
    """
    Sanitize string input.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    
    # Strip whitespace and normalize
    sanitized = text.strip()
    
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    
    # Truncate if needed
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def validate_image_file(filename: str) -> bool:
    """
    Validate image file extension.
    
    Args:
        filename: Image filename to validate
        
    Returns:
        bool: True if valid image file
    """
    if not filename:
        return False
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    return f'.{extension}' in allowed_extensions