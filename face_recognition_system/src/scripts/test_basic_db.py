#!/usr/bin/env python3
"""Basic database test without complex relationships."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_database, DatabaseConfig
from database.models import Person
import numpy as np


def test_basic_database():
    """Test basic database operations."""
    print("Testing basic database operations...")
    
    # Initialize database
    config = DatabaseConfig()
    config.database_url = 'sqlite:///test_basic_db.db'
    
    try:
        db_connection = init_database(config)
        
        print("✓ Database initialized")
        
        # Test direct database operations
        with db_connection.get_session() as session:
            # Create person
            person = Person(
                name="测试用户",
                employee_id="TEST123",
                email="test123@company.com",
                department="测试部门",
                access_level=3
            )
            
            session.add(person)
            session.commit()
            
            print(f"✓ Created person with ID: {person.id}")
            
            # Query person
            retrieved = session.query(Person).filter_by(employee_id="TEST123").first()
            if retrieved:
                print(f"✓ Retrieved person: {retrieved.name}")
            
            # Update person
            retrieved.access_level = 5
            session.commit()
            print("✓ Updated person")
            
            # Count persons
            count = session.query(Person).count()
            print(f"✓ Total persons: {count}")
        
        print("✅ All basic database tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists('test_basic_db.db'):
            try:
                os.remove('test_basic_db.db')
                print("✓ Cleaned up test database")
            except:
                pass


def main():
    """Run basic database tests."""
    print("Face Recognition System - Basic Database Tests")
    print("=" * 60)
    
    success = test_basic_database()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        return 0
    else:
        print("✗ Tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())