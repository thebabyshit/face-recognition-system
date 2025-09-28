"""API documentation configuration and enhancements."""

from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import HTMLResponse

def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Dict: Custom OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Face Recognition System API",
        version="1.0.0",
        description="""
        ## 人脸识别系统 API

        这是一个完整的人脸识别和门禁控制系统的RESTful API。

        ### 主要功能
        - **人员管理**: 添加、更新、删除和搜索人员信息
        - **人脸特征管理**: 上传人脸图像，提取和管理人脸特征
        - **实时人脸识别**: 识别和验证人脸身份
        - **门禁控制**: 基于人脸识别的访问控制
        - **统计报告**: 访问日志分析和统计报告
        - **用户管理**: 系统用户和权限管理
        - **系统管理**: 系统配置、备份和监控

        ### 认证方式
        API使用JWT (JSON Web Token) 进行认证。获取令牌后，在请求头中包含：
        ```
        Authorization: Bearer <your_access_token>
        ```

        ### 权限系统
        - **read**: 读取权限 - 可以查看数据
        - **write**: 写入权限 - 可以创建和修改数据
        - **admin**: 管理员权限 - 可以执行所有操作

        ### 速率限制
        - 匿名用户: 100 请求/分钟
        - 认证用户: 200 请求/分钟
        - 管理员: 500 请求/分钟

        ### 默认账户
        - 管理员: `admin` / `admin123`
        - 普通用户: `user` / `user123`
        """,
        routes=app.routes,
        servers=[
            {"url": "http://localhost:8000", "description": "开发环境"},
            {"url": "https://api.example.com", "description": "生产环境"}
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT认证令牌"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "authentication",
            "description": "用户认证和令牌管理"
        },
        {
            "name": "user-management", 
            "description": "系统用户管理（仅管理员）"
        },
        {
            "name": "persons",
            "description": "人员信息管理"
        },
        {
            "name": "faces",
            "description": "人脸特征管理"
        },
        {
            "name": "recognition",
            "description": "人脸识别服务"
        },
        {
            "name": "access",
            "description": "门禁控制和访问日志"
        },
        {
            "name": "statistics",
            "description": "统计报告和数据分析"
        },
        {
            "name": "system",
            "description": "系统管理和配置"
        }
    ]
    
    # Add response examples
    add_response_examples(openapi_schema)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def add_response_examples(schema: Dict[str, Any]):
    """Add response examples to OpenAPI schema."""
    
    # Common response examples
    examples = {
        "SuccessResponse": {
            "summary": "成功响应",
            "value": {
                "status": "success",
                "message": "操作成功",
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {}
            }
        },
        "ErrorResponse": {
            "summary": "错误响应", 
            "value": {
                "status": "error",
                "message": "操作失败",
                "timestamp": "2024-01-15T10:30:00Z",
                "error_code": "VALIDATION_ERROR",
                "details": {}
            }
        },
        "UnauthorizedResponse": {
            "summary": "未授权",
            "value": {
                "detail": "Not authenticated"
            }
        },
        "ForbiddenResponse": {
            "summary": "权限不足",
            "value": {
                "detail": "Missing required permissions: write"
            }
        },
        "RateLimitResponse": {
            "summary": "请求频率超限",
            "value": {
                "detail": "Rate limit exceeded",
                "retry_after": 60
            }
        }
    }
    
    # Add examples to components
    if "components" not in schema:
        schema["components"] = {}
    if "examples" not in schema["components"]:
        schema["components"]["examples"] = {}
    
    schema["components"]["examples"].update(examples)

def get_custom_swagger_ui_html(
    openapi_url: str = "/openapi.json",
    title: str = "Face Recognition API - Swagger UI",
) -> HTMLResponse:
    """Generate custom Swagger UI HTML."""
    
    swagger_ui_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css" />
        <style>
            .swagger-ui .topbar {{ display: none }}
            .swagger-ui .info .title {{ color: #3b82f6 }}
            .swagger-ui .scheme-container {{ background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
        <script>
            const ui = SwaggerUIBundle({{
                url: '{openapi_url}',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.presets.standalone
                ],
                layout: "BaseLayout",
                deepLinking: true,
                showExtensions: true,
                showCommonExtensions: true,
                defaultModelsExpandDepth: 2,
                defaultModelExpandDepth: 2,
                docExpansion: "list",
                filter: true,
                tryItOutEnabled: true,
                requestInterceptor: function(request) {{
                    // Add custom headers or modify requests
                    return request;
                }},
                responseInterceptor: function(response) {{
                    // Handle responses
                    return response;
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=swagger_ui_html)

def get_custom_redoc_html(
    openapi_url: str = "/openapi.json",
    title: str = "Face Recognition API - ReDoc",
) -> HTMLResponse:
    """Generate custom ReDoc HTML."""
    
    redoc_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
        <style>
            body {{ margin: 0; padding: 0; }}
            redoc {{ display: block; }}
        </style>
    </head>
    <body>
        <redoc spec-url='{openapi_url}' theme='{{
            "colors": {{
                "primary": {{
                    "main": "#3b82f6"
                }}
            }},
            "typography": {{
                "fontSize": "14px",
                "lineHeight": "1.5em",
                "code": {{
                    "fontSize": "13px"
                }},
                "headings": {{
                    "fontFamily": "Montserrat, sans-serif",
                    "fontWeight": "400"
                }}
            }},
            "sidebar": {{
                "width": "300px"
            }}
        }}'></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=redoc_html)

# API documentation metadata
API_METADATA = {
    "title": "Face Recognition System API",
    "description": "Complete face recognition and access control system",
    "version": "1.0.0",
    "contact": {
        "name": "API Support",
        "email": "support@example.com",
        "url": "https://example.com/support"
    },
    "license": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    "terms_of_service": "https://example.com/terms"
}

# Common response schemas for documentation
COMMON_RESPONSES = {
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Invalid request parameters"
                }
            }
        }
    },
    401: {
        "description": "Unauthorized", 
        "content": {
            "application/json": {
                "example": {
                    "detail": "Not authenticated"
                }
            }
        }
    },
    403: {
        "description": "Forbidden",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Missing required permissions"
                }
            }
        }
    },
    404: {
        "description": "Not Found",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Resource not found"
                }
            }
        }
    },
    422: {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "field"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    429: {
        "description": "Rate Limit Exceeded",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Rate limit exceeded",
                    "retry_after": 60
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Internal server error"
                }
            }
        }
    }
}