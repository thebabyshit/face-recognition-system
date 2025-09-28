"""Middleware for FastAPI application."""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} - "
                f"Time: {process_time:.3f}s - "
                f"Path: {request.url.path}"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed: {e} - Time: {process_time:.3f}s")
            raise

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and return consistent error responses."""
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error in {request.url.path}: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "path": str(request.url.path)
                }
            )