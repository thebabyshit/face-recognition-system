#!/usr/bin/env python3
"""
Script to run the Face Recognition API server.
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api.log')
        ]
    )

def main():
    """Main function to run the API server."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        from api.app import run_server
        
        logger.info("Starting Face Recognition API server...")
        
        # Run the server
        run_server(
            host="0.0.0.0",
            port=8000,
            reload=True  # Enable auto-reload for development
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()