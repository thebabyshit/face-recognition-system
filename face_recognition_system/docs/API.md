# Face Recognition System API Documentation

**Version:** 1.0.0

**Description:** 
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
        

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_access_token>
```

### Default Accounts

- **Admin**: `admin` / `admin123`
- **User**: `user` / `user123`

## Base URL

```
http://localhost:8000
```

## Endpoints

### Authentication

#### POST /api/v1/auth/login

**Summary:** Login

User login endpoint.
Authenticates user and returns JWT tokens.

#### POST /api/v1/auth/refresh

**Summary:** Refresh Token

Refresh access token using refresh token.

#### GET /api/v1/auth/me

**Summary:** Get Current User Info

Get current user information.

#### POST /api/v1/auth/logout

**Summary:** Logout

User logout endpoint.
In a production system, this would invalidate the token.

#### POST /api/v1/auth/register

**Summary:** Register User

Register a new user.
Only admin users can register new users.

#### GET /api/v1/auth/permissions

**Summary:** Get Available Permissions

Get list of available permissions.

#### GET /api/v1/auth/validate-token

**Summary:** Validate Token

Validate current token and return user info.

### User-Management

#### GET /api/v1/users/

**Summary:** List Users

List all users.
Only admin users can access this endpoint.

#### POST /api/v1/users/

**Summary:** Create User

Create a new user.
Only admin users can create new users.

#### GET /api/v1/users/{user_id}

**Summary:** Get User

Get user by ID.
Only admin users can access this endpoint.

#### PUT /api/v1/users/{user_id}

**Summary:** Update User

Update user information.
Only admin users can update users.

#### DELETE /api/v1/users/{user_id}

**Summary:** Delete User

Delete user.
Only admin users can delete users.

#### POST /api/v1/users/{user_id}/change-password

**Summary:** Change User Password

Change user password.
Only admin users can change other users' passwords.

#### GET /api/v1/users/{user_id}/sessions

**Summary:** Get User Sessions

Get active sessions for a user.
Only admin users can view user sessions.

### Persons

#### GET /api/v1/persons/

**Summary:** List Persons

List persons with optional filtering and pagination.
Returns a paginated list of persons with optional filters.

#### POST /api/v1/persons/

**Summary:** Create Person

Create a new person.
Creates a new person record with the provided information.

### Faces

#### POST /api/v1/faces/upload

**Summary:** Upload Face Image

Upload a face image for a person.
Processes the uploaded image, detects faces, extracts features,
and stores them in the database.

#### POST /api/v1/faces/upload-file

**Summary:** Upload Face File

Upload a face image file for a person.
Processes the uploaded file, detects faces, extracts features,
and stores them in the database.

#### GET /api/v1/faces/person/{person_id}

**Summary:** Get Person Faces

Get all face features for a person.
Returns all face features associated with the specified person.

#### GET /api/v1/faces/{feature_id}

**Summary:** Get Face Feature

Get a specific face feature by ID.
Returns detailed information about a face feature.

#### PUT /api/v1/faces/{feature_id}

**Summary:** Update Face Feature

Update a face feature.
Updates the specified face feature with the provided data.

#### DELETE /api/v1/faces/{feature_id}

**Summary:** Delete Face Feature

Delete a face feature.
Soft deletes (deactivates) or hard deletes a face feature.

### Access

#### POST /api/v1/access/attempt

**Summary:** Process Access Attempt

Process an access attempt.
Validates access permissions and logs the attempt.

#### GET /api/v1/access/logs

**Summary:** Get Access Logs

Get access logs with filtering and pagination.
Returns a paginated list of access logs with optional filters.

#### GET /api/v1/access/logs/{log_id}

**Summary:** Get Access Log

Get a specific access log by ID.
Returns detailed information about an access log entry.

#### GET /api/v1/access/person/{person_id}/logs

**Summary:** Get Person Access Logs

Get access logs for a specific person.
Returns access history for the specified person.

#### GET /api/v1/access/location/{location_id}/logs

**Summary:** Get Location Access Logs

Get access logs for a specific location.
Returns access history for the specified location.

#### GET /api/v1/access/stats/summary

**Summary:** Get Access Summary

Get access statistics summary.
Returns summary statistics for access attempts.

#### GET /api/v1/access/stats/hourly

**Summary:** Get Hourly Access Stats

Get hourly access statistics.
Returns access attempts grouped by hour for a specific date.

### Recognition

#### POST /api/v1/recognition/identify

**Summary:** Identify Face

Identify a person from a face image.
Processes the provided image and returns recognition results.

#### GET /api/v1/recognition/status

**Summary:** Get Recognition Status

Get face recognition service status.
Returns information about the recognition service health and statistics.

### Statistics

#### GET /api/v1/statistics/dashboard

**Summary:** Get Dashboard Stats

Get dashboard statistics.
Returns key metrics for the main dashboard.

#### GET /api/v1/statistics/access-trends

**Summary:** Get Access Trends

Get access trends over time.
Returns access statistics grouped by time periods.

#### GET /api/v1/statistics/department-stats

**Summary:** Get Department Statistics

Get statistics by department.
Returns access statistics grouped by department.

#### GET /api/v1/statistics/person-activity

**Summary:** Get Person Activity Stats

Get person activity statistics.
Returns access statistics for individual persons.

#### GET /api/v1/statistics/location-stats

**Summary:** Get Location Statistics

Get statistics by location.
Returns access statistics grouped by location.

#### GET /api/v1/statistics/recognition-performance

**Summary:** Get Recognition Performance Stats

Get face recognition performance statistics.
Returns statistics about recognition accuracy and performance.

#### GET /api/v1/statistics/system-health

**Summary:** Get System Health Stats

Get system health statistics.
Returns various system health metrics and status information.

#### POST /api/v1/statistics/generate-report

**Summary:** Generate Custom Report

Generate a custom report.
Creates a detailed report based on specified parameters.

### System

#### GET /api/v1/system/health

**Summary:** Health Check

System health check endpoint.
Returns basic system health information.

#### GET /api/v1/system/info

**Summary:** Get System Info

Get system information.
Returns detailed system information and statistics.

#### GET /api/v1/system/logs

**Summary:** Get System Logs

Get system logs.
Returns system logs with optional filtering.

#### POST /api/v1/system/maintenance/cleanup

**Summary:** Cleanup Old Data

Clean up old system data.
Removes old logs and inactive records based on retention policy.

#### POST /api/v1/system/maintenance/optimize

**Summary:** Optimize Database

Optimize database performance.
Performs database maintenance operations like VACUUM and ANALYZE.

#### POST /api/v1/system/backup/create

**Summary:** Create Backup

Create system backup.
Creates a backup of the database and system data.

#### GET /api/v1/system/backup/list

**Summary:** List Backups

List available backups.
Returns a list of available system backups.

#### GET /api/v1/system/config

**Summary:** Get System Config

Get system configuration.
Returns current system configuration settings.

#### PUT /api/v1/system/config

**Summary:** Update System Config

Update system configuration.
Updates system configuration settings.

#### POST /api/v1/system/restart

**Summary:** Restart Service

Restart system components.
Restarts specified system components.

### Other

#### GET /

**Summary:** Root

Root endpoint.

#### GET /health

**Summary:** Health Check

Health check endpoint.

## Rate Limiting

- Anonymous users: 100 requests/minute
- Authenticated users: 200 requests/minute
- Admin users: 500 requests/minute

## Error Responses

The API returns standard HTTP status codes:

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

