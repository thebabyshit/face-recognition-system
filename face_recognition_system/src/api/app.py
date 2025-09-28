"""FastAPI application for face recognition system."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from database.connection import init_database, DatabaseConfig
from database.services import get_database_service
# from recognition.recognition_service import FaceRecognitionService
from .routes import persons, faces, access, recognition, statistics, system, auth, users
from .middleware import LoggingMiddleware, ErrorHandlingMiddleware
from .rate_limiter import SmartRateLimitMiddleware
from .dependencies import get_current_user, get_db_service
from .models import SuccessResponse
from .docs import custom_openapi, get_custom_swagger_ui_html, get_custom_redoc_html, API_METADATA
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Face Recognition API...")
    
    try:
        # Initialize database
        db_config = DatabaseConfig()
        init_database(db_config)
        logger.info("Database initialized")
        
        # Initialize recognition service (disabled for now)
        # recognition_service = FaceRecognitionService()
        # app.state.recognition_service = recognition_service
        app.state.recognition_service = None
        logger.info("Recognition service initialization skipped")
        
        # Store services in app state
        app.state.db_service = get_database_service()
        
        logger.info("Face Recognition API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Recognition API...")
    
    try:
        if hasattr(app.state, 'recognition_service'):
            app.state.recognition_service.stop()
        
        logger.info("Face Recognition API shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    
    app = FastAPI(
        title=API_METADATA["title"],
        description=API_METADATA["description"],
        version=API_METADATA["version"],
        contact=API_METADATA["contact"],
        license_info=API_METADATA["license"],
        terms_of_service=API_METADATA["terms_of_service"],
        docs_url=None,  # We'll create custom docs
        redoc_url=None,  # We'll create custom redoc
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Add rate limiting middleware
    app.add_middleware(
        SmartRateLimitMiddleware,
        default_requests=100,  # 100 requests per minute for anonymous users
        default_window=60,
        authenticated_multiplier=2.0,  # 200 requests per minute for authenticated users
        admin_multiplier=5.0,  # 500 requests per minute for admin users
        whitelist=["127.0.0.1", "::1"]  # Whitelist localhost
    )
    
    # Include routers
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["user-management"])
    app.include_router(persons.router, prefix="/api/v1/persons", tags=["persons"])
    app.include_router(faces.router, prefix="/api/v1/faces", tags=["faces"])
    app.include_router(access.router, prefix="/api/v1/access", tags=["access"])
    app.include_router(recognition.router, prefix="/api/v1/recognition", tags=["recognition"])
    app.include_router(statistics.router, prefix="/api/v1/statistics", tags=["statistics"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Face Recognition System API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            # Check database connection
            db_service = get_database_service()
            
            # Check recognition service
            recognition_service = getattr(app.state, 'recognition_service', None)
            
            health_data = {
                "database": "connected",
                "recognition_service": "available" if recognition_service else "unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return {
                "status": "healthy",
                "message": "System is healthy",
                "data": health_data
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System health check failed"
            )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    # Set custom OpenAPI schema
    app.openapi = lambda: custom_openapi(app)
    
    # Add custom documentation routes
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        return get_custom_swagger_ui_html(openapi_url="/openapi.json")
    
    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc():
        return get_custom_redoc_html(openapi_url="/openapi.json")
    
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)