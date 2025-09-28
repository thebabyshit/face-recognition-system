# Face Recognition System Deployment Script for Windows
param(
    [Parameter(Position=0)]
    [string]$Command = "deploy",
    [Parameter(Position=1)]
    [string]$Service = ""
)

# Configuration
$ProjectName = "face-recognition-system"
$DockerComposeFile = "docker-compose.yml"
$EnvFile = ".env"

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Docker is installed
function Test-Docker {
    Write-Info "Checking Docker installation..."
    
    try {
        $dockerVersion = docker --version
        $dockerComposeVersion = docker-compose --version
        Write-Success "Docker and Docker Compose are installed"
        Write-Info "Docker: $dockerVersion"
        Write-Info "Docker Compose: $dockerComposeVersion"
        return $true
    }
    catch {
        Write-Error "Docker or Docker Compose is not installed. Please install Docker Desktop first."
        return $false
    }
}

# Create necessary directories
function New-ProjectDirectories {
    Write-Info "Creating necessary directories..."
    
    $directories = @(
        "data\models",
        "data\faces", 
        "data\backups",
        "logs",
        "config",
        "nginx\ssl",
        "nginx\logs",
        "monitoring\grafana\dashboards",
        "monitoring\grafana\datasources",
        "monitoring\rules",
        "sql"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Generate SSL certificates (self-signed for development)
function New-SSLCertificates {
    Write-Info "Generating SSL certificates..."
    
    if (!(Test-Path "nginx\ssl\cert.pem") -or !(Test-Path "nginx\ssl\key.pem")) {
        try {
            # Check if OpenSSL is available
            $opensslPath = Get-Command openssl -ErrorAction SilentlyContinue
            if ($opensslPath) {
                & openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
                    -keyout nginx\ssl\key.pem `
                    -out nginx\ssl\cert.pem `
                    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
                Write-Success "SSL certificates generated using OpenSSL"
            }
            else {
                # Use PowerShell to create self-signed certificate
                $cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "cert:\LocalMachine\My"
                $certPath = "nginx\ssl\cert.pem"
                $keyPath = "nginx\ssl\key.pem"
                
                # Export certificate
                Export-Certificate -Cert $cert -FilePath $certPath -Type CERT
                
                Write-Success "SSL certificates generated using PowerShell"
                Write-Warning "For production, please use proper SSL certificates"
            }
        }
        catch {
            Write-Warning "Could not generate SSL certificates. Using HTTP only."
            # Create dummy files to prevent errors
            "# Dummy cert file" | Out-File -FilePath "nginx\ssl\cert.pem"
            "# Dummy key file" | Out-File -FilePath "nginx\ssl\key.pem"
        }
    }
    else {
        Write-Info "SSL certificates already exist"
    }
}

# Create environment file
function New-EnvironmentFile {
    Write-Info "Creating environment file..."
    
    if (!(Test-Path $EnvFile)) {
        # Generate random passwords
        $dbPassword = -join ((1..16) | ForEach {Get-Random -input ([char[]]([char]'a'..[char]'z') + [char[]]([char]'A'..[char]'Z') + [char[]]([char]'0'..[char]'9'))})
        $jwtSecret = -join ((1..64) | ForEach {Get-Random -input ([char[]]([char]'a'..[char]'z') + [char[]]([char]'A'..[char]'Z') + [char[]]([char]'0'..[char]'9'))})
        $grafanaPassword = -join ((1..12) | ForEach {Get-Random -input ([char[]]([char]'a'..[char]'z') + [char[]]([char]'A'..[char]'Z') + [char[]]([char]'0'..[char]'9'))})
        
        $envContent = @"
# Database Configuration
POSTGRES_DB=face_recognition
POSTGRES_USER=face_user
POSTGRES_PASSWORD=$dbPassword
DATABASE_URL=postgresql://face_user:$dbPassword@postgres:5432/face_recognition

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# JWT Configuration
JWT_SECRET_KEY=$jwtSecret
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost
NODE_ENV=production

# Security Configuration
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost,http://127.0.0.1

# File Upload Configuration
MAX_UPLOAD_SIZE=50MB
UPLOAD_PATH=/app/data/faces

# Model Configuration
MODEL_PATH=/app/data/models
FACE_DETECTION_CONFIDENCE=0.7
FACE_RECOGNITION_THRESHOLD=0.6

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=$grafanaPassword

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
"@
        
        $envContent | Out-File -FilePath $EnvFile -Encoding UTF8
        Write-Success "Environment file created"
    }
    else {
        Write-Info "Environment file already exists"
    }
}

# Create database initialization script
function New-DatabaseInit {
    Write-Info "Creating database initialization script..."
    
    $initScript = @'
-- Face Recognition System Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'face_user') THEN
        CREATE ROLE face_user LOGIN PASSWORD 'face_password';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE face_recognition TO face_user;
GRANT ALL ON SCHEMA public TO face_user;

-- Create initial tables will be handled by SQLAlchemy migrations
'@
    
    $initScript | Out-File -FilePath "sql\init.sql" -Encoding UTF8
    Write-Success "Database initialization script created"
}

# Pull Docker images
function Get-DockerImages {
    Write-Info "Pulling Docker images..."
    try {
        docker-compose pull
        Write-Success "Docker images pulled"
    }
    catch {
        Write-Error "Failed to pull Docker images: $_"
        return $false
    }
    return $true
}

# Build custom images
function Build-DockerImages {
    Write-Info "Building custom Docker images..."
    try {
        docker-compose build --no-cache
        Write-Success "Docker images built"
    }
    catch {
        Write-Error "Failed to build Docker images: $_"
        return $false
    }
    return $true
}

# Start services
function Start-Services {
    Write-Info "Starting services..."
    try {
        docker-compose up -d
        Write-Success "Services started"
    }
    catch {
        Write-Error "Failed to start services: $_"
        return $false
    }
    return $true
}

# Wait for services to be ready
function Wait-ForServices {
    Write-Info "Waiting for services to be ready..."
    
    # Wait for database
    Write-Info "Waiting for database..."
    $timeout = 60
    $elapsed = 0
    do {
        try {
            docker-compose exec -T postgres pg_isready -U face_user -d face_recognition 2>$null
            if ($LASTEXITCODE -eq 0) { break }
        }
        catch { }
        Start-Sleep 2
        $elapsed += 2
    } while ($elapsed -lt $timeout)
    
    if ($elapsed -ge $timeout) {
        Write-Error "Database failed to start within $timeout seconds"
        return $false
    }
    Write-Success "Database is ready"
    
    # Wait for Redis
    Write-Info "Waiting for Redis..."
    $elapsed = 0
    do {
        try {
            docker-compose exec -T redis redis-cli ping 2>$null
            if ($LASTEXITCODE -eq 0) { break }
        }
        catch { }
        Start-Sleep 2
        $elapsed += 2
    } while ($elapsed -lt $timeout)
    
    if ($elapsed -ge $timeout) {
        Write-Error "Redis failed to start within $timeout seconds"
        return $false
    }
    Write-Success "Redis is ready"
    
    # Wait for backend
    Write-Info "Waiting for backend API..."
    $elapsed = 0
    do {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) { break }
        }
        catch { }
        Start-Sleep 5
        $elapsed += 5
    } while ($elapsed -lt $timeout)
    
    if ($elapsed -ge $timeout) {
        Write-Warning "Backend API may not be ready, but continuing..."
    } else {
        Write-Success "Backend API is ready"
    }
    
    # Wait for frontend
    Write-Info "Waiting for frontend..."
    $elapsed = 0
    do {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) { break }
        }
        catch { }
        Start-Sleep 5
        $elapsed += 5
    } while ($elapsed -lt $timeout)
    
    if ($elapsed -ge $timeout) {
        Write-Warning "Frontend may not be ready, but continuing..."
    } else {
        Write-Success "Frontend is ready"
    }
    
    return $true
}

# Run database migrations
function Invoke-DatabaseMigrations {
    Write-Info "Running database migrations..."
    try {
        docker-compose exec backend python src/scripts/init_database.py
        Write-Success "Database migrations completed"
    }
    catch {
        Write-Warning "Database migrations may have failed: $_"
    }
}

# Create initial admin user
function New-AdminUser {
    Write-Info "Creating initial admin user..."
    
    $pythonScript = @'
from src.database.services import get_database_service
from src.auth.rbac import RoleBasedAccessControl
from werkzeug.security import generate_password_hash
import sys

try:
    db = get_database_service()
    rbac = RoleBasedAccessControl()
    
    # Create admin user
    admin_data = {
        'username': 'admin',
        'email': 'admin@localhost',
        'password_hash': generate_password_hash('admin123'),
        'is_active': True
    }
    
    # Check if admin user already exists
    existing_user = db.users.get_user_by_username('admin')
    if not existing_user:
        user_id = db.users.create_user(admin_data)
        rbac.assign_role_to_user(user_id, 'super_admin')
        print('Admin user created successfully')
    else:
        print('Admin user already exists')
        
except Exception as e:
    print(f'Error creating admin user: {e}')
    sys.exit(1)
'@
    
    try {
        docker-compose exec backend python -c $pythonScript
        Write-Success "Initial admin user created (username: admin, password: admin123)"
    }
    catch {
        Write-Warning "Failed to create admin user: $_"
    }
}

# Show service status
function Show-ServiceStatus {
    Write-Info "Service Status:"
    docker-compose ps
    
    Write-Host ""
    Write-Info "Service URLs:"
    Write-Host "  Frontend:    http://localhost:3000" -ForegroundColor Cyan
    Write-Host "  Backend API: http://localhost:8000/api" -ForegroundColor Cyan
    Write-Host "  API Docs:    http://localhost:8000/api/docs" -ForegroundColor Cyan
    Write-Host "  Grafana:     http://localhost:3001 (admin/admin123)" -ForegroundColor Cyan
    Write-Host "  Prometheus:  http://localhost:9090" -ForegroundColor Cyan
    Write-Host ""
    Write-Info "Default Admin Credentials:"
    Write-Host "  Username: admin" -ForegroundColor Yellow
    Write-Host "  Password: admin123" -ForegroundColor Yellow
    Write-Host ""
    Write-Warning "Please change the default passwords in production!"
}

# Stop services
function Stop-Services {
    Write-Info "Stopping services..."
    try {
        docker-compose down
        Write-Success "Services stopped"
    }
    catch {
        Write-Error "Failed to stop services: $_"
    }
}

# Clean up (remove containers and volumes)
function Remove-AllData {
    Write-Warning "This will remove all containers and data volumes!"
    $confirmation = Read-Host "Are you sure? (y/N)"
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        Write-Info "Cleaning up..."
        try {
            docker-compose down -v --remove-orphans
            docker system prune -f
            Write-Success "Cleanup completed"
        }
        catch {
            Write-Error "Cleanup failed: $_"
        }
    }
    else {
        Write-Info "Cleanup cancelled"
    }
}

# Show logs
function Show-ServiceLogs {
    param([string]$ServiceName)
    
    if ($ServiceName) {
        docker-compose logs -f $ServiceName
    }
    else {
        docker-compose logs -f
    }
}

# Backup data
function Backup-SystemData {
    Write-Info "Creating backup..."
    
    $backupDir = "backups\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    try {
        # Backup database
        docker-compose exec -T postgres pg_dump -U face_user face_recognition | Out-File -FilePath "$backupDir\database.sql" -Encoding UTF8
        
        # Backup uploaded files
        $containerId = docker-compose ps -q backend
        docker cp "${containerId}:/app/data" $backupDir
        
        # Create archive
        Compress-Archive -Path $backupDir -DestinationPath "$backupDir.zip"
        Remove-Item -Path $backupDir -Recurse -Force
        
        Write-Success "Backup created: $backupDir.zip"
    }
    catch {
        Write-Error "Backup failed: $_"
    }
}

# Main script logic
function Main {
    param([string]$Command, [string]$Service)
    
    switch ($Command.ToLower()) {
        "deploy" {
            Write-Info "Starting Face Recognition System deployment..."
            if (!(Test-Docker)) { return }
            New-ProjectDirectories
            New-SSLCertificates
            New-EnvironmentFile
            New-DatabaseInit
            if (!(Get-DockerImages)) { return }
            if (!(Build-DockerImages)) { return }
            if (!(Start-Services)) { return }
            if (!(Wait-ForServices)) { return }
            Invoke-DatabaseMigrations
            New-AdminUser
            Show-ServiceStatus
            Write-Success "Deployment completed successfully!"
        }
        "start" {
            Write-Info "Starting services..."
            Start-Services
            Wait-ForServices
            Show-ServiceStatus
        }
        "stop" {
            Stop-Services
        }
        "restart" {
            Stop-Services
            Start-Sleep 5
            Start-Services
            Wait-ForServices
            Show-ServiceStatus
        }
        "status" {
            Show-ServiceStatus
        }
        "logs" {
            Show-ServiceLogs $Service
        }
        "backup" {
            Backup-SystemData
        }
        "cleanup" {
            Remove-AllData
        }
        "help" {
            Write-Host "Usage: .\deploy.ps1 [command] [service]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  deploy    - Full deployment (default)"
            Write-Host "  start     - Start services"
            Write-Host "  stop      - Stop services"
            Write-Host "  restart   - Restart services"
            Write-Host "  status    - Show service status"
            Write-Host "  logs      - Show logs (optionally for specific service)"
            Write-Host "  backup    - Create data backup"
            Write-Host "  cleanup   - Remove all containers and volumes"
            Write-Host "  help      - Show this help message"
        }
        default {
            Write-Error "Unknown command: $Command"
            Write-Host "Use '.\deploy.ps1 help' for usage information"
        }
    }
}

# Run main function
Main -Command $Command -Service $Service