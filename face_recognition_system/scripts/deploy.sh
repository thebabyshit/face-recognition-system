#!/bin/bash

# Face Recognition System Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="face-recognition-system"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are installed"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p data/{models,faces,backups}
    mkdir -p logs
    mkdir -p config
    mkdir -p nginx/{ssl,logs}
    mkdir -p monitoring/{grafana/{dashboards,datasources},rules}
    mkdir -p sql
    
    log_success "Directories created"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certs() {
    log_info "Generating SSL certificates..."
    
    if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Create environment file
create_env_file() {
    log_info "Creating environment file..."
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Database Configuration
POSTGRES_DB=face_recognition
POSTGRES_USER=face_user
POSTGRES_PASSWORD=face_password_$(openssl rand -hex 8)
DATABASE_URL=postgresql://face_user:face_password_$(openssl rand -hex 8)@postgres:5432/face_recognition

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# JWT Configuration
JWT_SECRET_KEY=$(openssl rand -hex 32)
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
NEXT_PUBLIC_API_URL=https://localhost
NODE_ENV=production

# Security Configuration
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=https://localhost,https://127.0.0.1

# File Upload Configuration
MAX_UPLOAD_SIZE=50MB
UPLOAD_PATH=/app/data/faces

# Model Configuration
MODEL_PATH=/app/data/models
FACE_DETECTION_CONFIDENCE=0.7
FACE_RECOGNITION_THRESHOLD=0.6

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin123

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
EOF
        
        log_success "Environment file created"
    else
        log_info "Environment file already exists"
    fi
}

# Create database initialization script
create_db_init() {
    log_info "Creating database initialization script..."
    
    cat > "sql/init.sql" << 'EOF'
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
EOF
    
    log_success "Database initialization script created"
}

# Pull Docker images
pull_images() {
    log_info "Pulling Docker images..."
    docker-compose pull
    log_success "Docker images pulled"
}

# Build custom images
build_images() {
    log_info "Building custom Docker images..."
    docker-compose build --no-cache
    log_success "Docker images built"
}

# Start services
start_services() {
    log_info "Starting services..."
    docker-compose up -d
    log_success "Services started"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for database
    log_info "Waiting for database..."
    until docker-compose exec -T postgres pg_isready -U face_user -d face_recognition; do
        sleep 2
    done
    log_success "Database is ready"
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    until docker-compose exec -T redis redis-cli ping; do
        sleep 2
    done
    log_success "Redis is ready"
    
    # Wait for backend
    log_info "Waiting for backend API..."
    until curl -f http://localhost:8000/api/health &>/dev/null; do
        sleep 5
    done
    log_success "Backend API is ready"
    
    # Wait for frontend
    log_info "Waiting for frontend..."
    until curl -f http://localhost:3000 &>/dev/null; do
        sleep 5
    done
    log_success "Frontend is ready"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    docker-compose exec backend python src/scripts/init_database.py
    log_success "Database migrations completed"
}

# Create initial admin user
create_admin_user() {
    log_info "Creating initial admin user..."
    docker-compose exec backend python -c "
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
"
    log_success "Initial admin user created (username: admin, password: admin123)"
}

# Show service status
show_status() {
    log_info "Service Status:"
    docker-compose ps
    
    echo ""
    log_info "Service URLs:"
    echo "  Frontend:    https://localhost"
    echo "  Backend API: https://localhost/api"
    echo "  API Docs:    https://localhost/api/docs"
    echo "  Grafana:     http://localhost:3001 (admin/admin123)"
    echo "  Prometheus:  http://localhost:9090"
    echo ""
    log_info "Default Admin Credentials:"
    echo "  Username: admin"
    echo "  Password: admin123"
    echo ""
    log_warning "Please change the default passwords in production!"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped"
}

# Clean up (remove containers and volumes)
cleanup() {
    log_warning "This will remove all containers and data volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show logs
show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Backup data
backup_data() {
    log_info "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U face_user face_recognition > "$BACKUP_DIR/database.sql"
    
    # Backup uploaded files
    docker cp $(docker-compose ps -q backend):/app/data "$BACKUP_DIR/"
    
    # Create archive
    tar -czf "$BACKUP_DIR.tar.gz" -C backups "$(basename "$BACKUP_DIR")"
    rm -rf "$BACKUP_DIR"
    
    log_success "Backup created: $BACKUP_DIR.tar.gz"
}

# Main script
main() {
    case "${1:-deploy}" in
        "deploy")
            log_info "Starting Face Recognition System deployment..."
            check_docker
            create_directories
            generate_ssl_certs
            create_env_file
            create_db_init
            pull_images
            build_images
            start_services
            wait_for_services
            run_migrations
            create_admin_user
            show_status
            log_success "Deployment completed successfully!"
            ;;
        "start")
            log_info "Starting services..."
            start_services
            wait_for_services
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            start_services
            wait_for_services
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "backup")
            backup_data
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    - Full deployment (default)"
            echo "  start     - Start services"
            echo "  stop      - Stop services"
            echo "  restart   - Restart services"
            echo "  status    - Show service status"
            echo "  logs      - Show logs (optionally for specific service)"
            echo "  backup    - Create data backup"
            echo "  cleanup   - Remove all containers and volumes"
            echo "  help      - Show this help message"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"