# Face Recognition System API Documentation

This directory contains comprehensive documentation for the Face Recognition System API.

## Files

- `API.md` - Complete API documentation
- `openapi.json` - OpenAPI 3.0 specification
- `postman_collection.json` - Postman collection for testing

## Quick Start

1. Start the API server:
   ```bash
   python src/scripts/run_api.py
   ```

2. Access the interactive documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. Import the Postman collection for API testing

## Authentication

Use the following credentials for testing:

**Admin Account:**
- Username: `admin`
- Password: `admin123`

**Regular User Account:**
- Username: `user`
- Password: `user123`

## Testing

Run the comprehensive API tests:
```bash
python src/scripts/test_api_comprehensive.py
```

Or run pytest:
```bash
pytest tests/
```

