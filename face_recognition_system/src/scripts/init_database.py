#!/usr/bin/env python3
"""Database initialization script."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_database, DatabaseConfig
from database.services import get_database_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_locations(db_service):
    """Create default access locations."""
    try:
        with db_service.persons._get_db_manager() as db:
            # Check if locations already exist
            existing_locations = db.session.query(
                db.session.query(db.session.bind.execute("SELECT COUNT(*) FROM access_locations")).scalar()
            )
            
            if existing_locations > 0:
                logger.info("Access locations already exist, skipping creation")
                return
            
            # Create default locations
            locations = [
                {
                    'name': 'Main Entrance',
                    'description': 'Main building entrance',
                    'location_type': 'door',
                    'building': 'Main Building',
                    'floor': 'Ground Floor',
                    'required_access_level': 1
                },
                {
                    'name': 'Server Room',
                    'description': 'Data center server room',
                    'location_type': 'door',
                    'building': 'Main Building',
                    'floor': 'Basement',
                    'required_access_level': 8
                },
                {
                    'name': 'Executive Floor',
                    'description': 'Executive offices access',
                    'location_type': 'elevator',
                    'building': 'Main Building',
                    'floor': '10th Floor',
                    'required_access_level': 7
                }
            ]
            
            from database.models import AccessLocation
            for loc_data in locations:
                location = AccessLocation(**loc_data)
                db.session.add(location)
            
            db.session.commit()
            logger.info(f"Created {len(locations)} default access locations")
            
    except Exception as e:
        logger.error(f"Error creating default locations: {e}")


def create_sample_data(db_service):
    """Create sample data for testing."""
    try:
        # Create sample persons
        sample_persons = [
            {
                'name': 'Alice Johnson',
                'employee_id': 'EMP001',
                'email': 'alice.johnson@company.com',
                'department': 'Engineering',
                'position': 'Senior Developer',
                'access_level': 5
            },
            {
                'name': 'Bob Smith',
                'employee_id': 'EMP002',
                'email': 'bob.smith@company.com',
                'department': 'Security',
                'position': 'Security Officer',
                'access_level': 8
            },
            {
                'name': 'Carol Davis',
                'employee_id': 'EMP003',
                'email': 'carol.davis@company.com',
                'department': 'HR',
                'position': 'HR Manager',
                'access_level': 6
            }
        ]
        
        created_persons = []
        for person_data in sample_persons:
            # Check if person already exists
            existing = db_service.persons.get_person_by_employee_id(person_data['employee_id'])
            if existing:
                logger.info(f"Person {person_data['employee_id']} already exists, skipping")
                created_persons.append(existing)
                continue
            
            person = db_service.persons.create_person(**person_data)
            if person:
                created_persons.append(person)
                logger.info(f"Created sample person: {person.name}")
        
        logger.info(f"Sample data creation completed. {len(created_persons)} persons available.")
        return created_persons
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return []


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description='Initialize face recognition database')
    parser.add_argument('--database-url', 
                       default=os.getenv('DATABASE_URL', 'sqlite:///face_recognition.db'),
                       help='Database URL (default: sqlite:///face_recognition.db)')
    parser.add_argument('--drop-existing', action='store_true',
                       help='Drop existing tables before creating new ones')
    parser.add_argument('--sample-data', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Configure database
        config = DatabaseConfig()
        config.database_url = args.database_url
        config.echo = args.verbose
        
        logger.info(f"Initializing database: {config.database_url}")
        
        # Initialize database connection
        db_connection = init_database(config, create_tables=False)
        
        # Drop existing tables if requested
        if args.drop_existing:
            logger.info("Dropping existing tables...")
            db_connection.drop_tables()
        
        # Create tables
        logger.info("Creating database tables...")
        db_connection.create_tables()
        
        # Initialize database service
        db_service = get_database_service()
        
        # Initialize default data
        logger.info("Initializing default data...")
        db_service.initialize_default_data()
        
        # Create default locations
        logger.info("Creating default access locations...")
        create_default_locations(db_service)
        
        # Create sample data if requested
        if args.sample_data:
            logger.info("Creating sample data...")
            create_sample_data(db_service)
        
        logger.info("Database initialization completed successfully!")
        
        # Print summary
        with db_service.persons._get_db_manager() as db:
            person_count = db.persons.count()
            logger.info(f"Database summary:")
            logger.info(f"  - Total persons: {person_count}")
            logger.info(f"  - Database URL: {config.database_url}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()