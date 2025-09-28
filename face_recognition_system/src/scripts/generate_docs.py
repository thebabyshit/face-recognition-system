#!/usr/bin/env python3
"""
Generate API documentation files.
"""

import sys
import os
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_openapi_spec():
    """Generate OpenAPI specification file."""
    from api.app import create_app
    
    app = create_app()
    openapi_spec = app.openapi()
    
    # Save to file
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    spec_file = docs_dir / "openapi.json"
    with open(spec_file, 'w', encoding='utf-8') as f:
        json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
    
    print(f"✓ OpenAPI specification saved to {spec_file}")
    return spec_file

def generate_postman_collection():
    """Generate Postman collection from OpenAPI spec."""
    from api.app import create_app
    
    app = create_app()
    openapi_spec = app.openapi()
    
    # Basic Postman collection structure
    collection = {
        "info": {
            "name": "Face Recognition System API",
            "description": "API collection for Face Recognition System",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{access_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "base_url",
                "value": "http://localhost:8000",
                "type": "string"
            },
            {
                "key": "access_token",
                "value": "",
                "type": "string"
            }
        ],
        "item": []
    }
    
    # Add authentication folder
    auth_folder = {
        "name": "Authentication",
        "item": [
            {
                "name": "Login",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": json.dumps({
                            "username": "admin",
                            "password": "admin123"
                        }, indent=2)
                    },
                    "url": {
                        "raw": "{{base_url}}/api/v1/auth/login",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "auth", "login"]
                    }
                },
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "exec": [
                                "if (pm.response.code === 200) {",
                                "    const response = pm.response.json();",
                                "    pm.collectionVariables.set('access_token', response.access_token);",
                                "}"
                            ]
                        }
                    }
                ]
            },
            {
                "name": "Get Current User",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/auth/me",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "auth", "me"]
                    }
                }
            }
        ]
    }
    
    collection["item"].append(auth_folder)
    
    # Add other endpoint folders
    folders = {
        "Persons": "/api/v1/persons",
        "Faces": "/api/v1/faces", 
        "Recognition": "/api/v1/recognition",
        "Access Control": "/api/v1/access",
        "Statistics": "/api/v1/statistics",
        "System": "/api/v1/system"
    }
    
    for folder_name, base_path in folders.items():
        folder = {
            "name": folder_name,
            "item": [
                {
                    "name": f"List {folder_name}",
                    "request": {
                        "method": "GET",
                        "header": [],
                        "url": {
                            "raw": f"{{{{base_url}}}}{base_path}/",
                            "host": ["{{base_url}}"],
                            "path": base_path.strip("/").split("/") + [""]
                        }
                    }
                }
            ]
        }
        collection["item"].append(folder)
    
    # Save collection
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    collection_file = docs_dir / "postman_collection.json"
    with open(collection_file, 'w', encoding='utf-8') as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Postman collection saved to {collection_file}")
    return collection_file

def generate_markdown_docs():
    """Generate Markdown documentation."""
    from api.app import create_app
    
    app = create_app()
    openapi_spec = app.openapi()
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Generate API overview
    api_md = docs_dir / "API.md"
    
    with open(api_md, 'w', encoding='utf-8') as f:
        f.write("# Face Recognition System API Documentation\n\n")
        f.write(f"**Version:** {openapi_spec['info']['version']}\n\n")
        f.write(f"**Description:** {openapi_spec['info']['description']}\n\n")
        
        f.write("## Authentication\n\n")
        f.write("The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:\n\n")
        f.write("```\nAuthorization: Bearer <your_access_token>\n```\n\n")
        
        f.write("### Default Accounts\n\n")
        f.write("- **Admin**: `admin` / `admin123`\n")
        f.write("- **User**: `user` / `user123`\n\n")
        
        f.write("## Base URL\n\n")
        f.write("```\nhttp://localhost:8000\n```\n\n")
        
        f.write("## Endpoints\n\n")
        
        # Group endpoints by tags
        paths = openapi_spec.get('paths', {})
        tags = {}
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    tag = details.get('tags', ['Other'])[0]
                    if tag not in tags:
                        tags[tag] = []
                    tags[tag].append({
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'description': details.get('description', '')
                    })
        
        for tag, endpoints in tags.items():
            f.write(f"### {tag.title()}\n\n")
            for endpoint in endpoints:
                f.write(f"#### {endpoint['method']} {endpoint['path']}\n\n")
                if endpoint['summary']:
                    f.write(f"**Summary:** {endpoint['summary']}\n\n")
                if endpoint['description']:
                    f.write(f"{endpoint['description']}\n\n")
        
        f.write("## Rate Limiting\n\n")
        f.write("- Anonymous users: 100 requests/minute\n")
        f.write("- Authenticated users: 200 requests/minute\n")
        f.write("- Admin users: 500 requests/minute\n\n")
        
        f.write("## Error Responses\n\n")
        f.write("The API returns standard HTTP status codes:\n\n")
        f.write("- `200` - Success\n")
        f.write("- `201` - Created\n")
        f.write("- `400` - Bad Request\n")
        f.write("- `401` - Unauthorized\n")
        f.write("- `403` - Forbidden\n")
        f.write("- `404` - Not Found\n")
        f.write("- `422` - Validation Error\n")
        f.write("- `429` - Rate Limit Exceeded\n")
        f.write("- `500` - Internal Server Error\n\n")
    
    print(f"✓ Markdown documentation saved to {api_md}")
    return api_md

def generate_readme():
    """Generate README for documentation."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    readme_file = docs_dir / "README.md"
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("# Face Recognition System API Documentation\n\n")
        f.write("This directory contains comprehensive documentation for the Face Recognition System API.\n\n")
        
        f.write("## Files\n\n")
        f.write("- `API.md` - Complete API documentation\n")
        f.write("- `openapi.json` - OpenAPI 3.0 specification\n")
        f.write("- `postman_collection.json` - Postman collection for testing\n\n")
        
        f.write("## Quick Start\n\n")
        f.write("1. Start the API server:\n")
        f.write("   ```bash\n")
        f.write("   python src/scripts/run_api.py\n")
        f.write("   ```\n\n")
        
        f.write("2. Access the interactive documentation:\n")
        f.write("   - Swagger UI: http://localhost:8000/docs\n")
        f.write("   - ReDoc: http://localhost:8000/redoc\n\n")
        
        f.write("3. Import the Postman collection for API testing\n\n")
        
        f.write("## Authentication\n\n")
        f.write("Use the following credentials for testing:\n\n")
        f.write("**Admin Account:**\n")
        f.write("- Username: `admin`\n")
        f.write("- Password: `admin123`\n\n")
        
        f.write("**Regular User Account:**\n")
        f.write("- Username: `user`\n")
        f.write("- Password: `user123`\n\n")
        
        f.write("## Testing\n\n")
        f.write("Run the comprehensive API tests:\n")
        f.write("```bash\n")
        f.write("python src/scripts/test_api_comprehensive.py\n")
        f.write("```\n\n")
        
        f.write("Or run pytest:\n")
        f.write("```bash\n")
        f.write("pytest tests/\n")
        f.write("```\n\n")
    
    print(f"✓ Documentation README saved to {readme_file}")
    return readme_file

def main():
    """Main function."""
    print("Generating API Documentation")
    print("=" * 40)
    
    try:
        # Generate all documentation files
        generate_openapi_spec()
        generate_postman_collection()
        generate_markdown_docs()
        generate_readme()
        
        print("\n✓ All documentation generated successfully!")
        print("\nGenerated files:")
        print("  - docs/openapi.json")
        print("  - docs/postman_collection.json")
        print("  - docs/API.md")
        print("  - docs/README.md")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Documentation generation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)