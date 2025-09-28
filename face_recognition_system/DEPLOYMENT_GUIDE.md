# 人脸识别系统部署指南

## 系统要求

### 硬件要求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **内存**: 最少 8GB RAM，推荐 16GB 或更多
- **存储**: 最少 50GB 可用空间，推荐 SSD
- **GPU**: 可选，NVIDIA GPU 用于加速推理（推荐）

### 软件要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Docker**: Docker Desktop 4.0+ 
- **Docker Compose**: 2.0+
- **PowerShell**: Windows PowerShell 5.1+ 或 PowerShell Core 7.0+

## 快速部署 (Windows)

### 1. 安装 Docker Desktop
1. 下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. 启动 Docker Desktop 并确保其正在运行
3. 在命令行中验证安装：
   ```cmd
   docker --version
   docker-compose --version
   ```

### 2. 克隆项目
```cmd
git clone <repository-url>
cd face_recognition_system
```

### 3. 一键部署
```cmd
deploy.bat
```

或者使用 PowerShell：
```powershell
.\scripts\deploy.ps1 deploy
```

### 4. 访问系统
部署完成后，您可以访问：
- **前端界面**: http://localhost:3000
- **API 文档**: http://localhost:8000/api/docs
- **监控面板**: http://localhost:3001 (Grafana)

**默认登录凭据**:
- 用户名: `admin`
- 密码: `admin123`

## 详细部署步骤

### 1. 环境准备

#### Windows 环境
```cmd
# 检查 Docker 安装
docker --version
docker-compose --version

# 检查 PowerShell 版本
powershell -Command "$PSVersionTable.PSVersion"
```

#### Linux 环境
```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

### 2. 项目配置

#### 创建环境变量文件
系统会自动创建 `.env` 文件，您也可以手动创建：

```env
# 数据库配置
POSTGRES_DB=face_recognition
POSTGRES_USER=face_user
POSTGRES_PASSWORD=your_secure_password

# JWT 配置
JWT_SECRET_KEY=your_jwt_secret_key

# API 配置
API_HOST=0.0.0.0
API_PORT=8000

# 前端配置
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### SSL 证书配置
对于生产环境，请替换自签名证书：
```cmd
# 将您的证书文件放置在以下位置
nginx\ssl\cert.pem    # SSL 证书
nginx\ssl\key.pem     # 私钥文件
```

### 3. 服务部署

#### 使用 Docker Compose
```cmd
# 拉取镜像
docker-compose pull

# 构建自定义镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

#### 服务说明
- **postgres**: PostgreSQL 数据库
- **redis**: Redis 缓存服务
- **backend**: Python FastAPI 后端服务
- **frontend**: Next.js 前端应用
- **nginx**: 反向代理服务器
- **prometheus**: 监控数据收集
- **grafana**: 监控数据可视化

### 4. 数据库初始化

```cmd
# 运行数据库迁移
docker-compose exec backend python src/scripts/init_database.py

# 创建管理员用户
docker-compose exec backend python -c "
from src.database.services import get_database_service
from src.auth.rbac import RoleBasedAccessControl
from werkzeug.security import generate_password_hash

db = get_database_service()
rbac = RoleBasedAccessControl()

admin_data = {
    'username': 'admin',
    'email': 'admin@localhost',
    'password_hash': generate_password_hash('admin123'),
    'is_active': True
}

user_id = db.users.create_user(admin_data)
rbac.assign_role_to_user(user_id, 'super_admin')
print('Admin user created')
"
```

## 服务管理

### 启动和停止服务

#### Windows (PowerShell)
```powershell
# 启动服务
.\scripts\deploy.ps1 start

# 停止服务
.\scripts\deploy.ps1 stop

# 重启服务
.\scripts\deploy.ps1 restart

# 查看状态
.\scripts\deploy.ps1 status

# 查看日志
.\scripts\deploy.ps1 logs

# 查看特定服务日志
.\scripts\deploy.ps1 logs backend
```

#### Linux/macOS
```bash
# 启动服务
./scripts/deploy.sh start

# 停止服务
./scripts/deploy.sh stop

# 重启服务
./scripts/deploy.sh restart

# 查看状态
./scripts/deploy.sh status

# 查看日志
./scripts/deploy.sh logs backend
```

### 服务端口说明
- **3000**: 前端应用 (Next.js)
- **8000**: 后端 API (FastAPI)
- **5432**: PostgreSQL 数据库
- **6379**: Redis 缓存
- **80/443**: Nginx 反向代理
- **9090**: Prometheus 监控
- **3001**: Grafana 仪表板

## 配置说明

### 数据库配置
```yaml
# docker-compose.yml
postgres:
  environment:
    POSTGRES_DB: face_recognition
    POSTGRES_USER: face_user
    POSTGRES_PASSWORD: your_password
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

### 后端配置
```yaml
backend:
  environment:
    - DATABASE_URL=postgresql://user:pass@postgres:5432/db
    - REDIS_URL=redis://redis:6379/0
    - JWT_SECRET_KEY=your_secret_key
  volumes:
    - ./data:/app/data
    - ./logs:/app/logs
```

### 前端配置
```yaml
frontend:
  environment:
    - NEXT_PUBLIC_API_URL=http://localhost:8000
    - NODE_ENV=production
```

## 监控和日志

### Prometheus 监控
访问 http://localhost:9090 查看系统指标：
- API 响应时间
- 数据库连接数
- 内存和CPU使用率
- 自定义业务指标

### Grafana 仪表板
访问 http://localhost:3001 查看可视化监控：
- 默认用户名: `admin`
- 默认密码: `admin123`

### 日志管理
```cmd
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs frontend
docker-compose logs postgres

# 实时跟踪日志
docker-compose logs -f backend
```

## 备份和恢复

### 数据备份
```cmd
# 使用部署脚本备份
.\scripts\deploy.ps1 backup

# 手动备份数据库
docker-compose exec postgres pg_dump -U face_user face_recognition > backup.sql

# 备份上传的文件
docker cp $(docker-compose ps -q backend):/app/data ./data_backup
```

### 数据恢复
```cmd
# 恢复数据库
docker-compose exec -T postgres psql -U face_user face_recognition < backup.sql

# 恢复文件数据
docker cp ./data_backup $(docker-compose ps -q backend):/app/data
```

## 性能优化

### 硬件优化
1. **使用 SSD**: 提高数据库和文件 I/O 性能
2. **增加内存**: 提高缓存效率和并发处理能力
3. **GPU 加速**: 使用 NVIDIA GPU 加速人脸识别推理

### 软件优化
1. **数据库调优**:
   ```sql
   -- 调整 PostgreSQL 配置
   shared_buffers = 256MB
   effective_cache_size = 1GB
   work_mem = 4MB
   ```

2. **Redis 缓存**:
   ```yaml
   redis:
     command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
   ```

3. **应用优化**:
   ```yaml
   backend:
     environment:
       - API_WORKERS=4  # 根据 CPU 核心数调整
       - WORKER_CONNECTIONS=1000
   ```

## 安全配置

### SSL/TLS 配置
1. 获取有效的 SSL 证书（Let's Encrypt 或商业证书）
2. 更新 nginx 配置使用 HTTPS
3. 强制 HTTP 重定向到 HTTPS

### 防火墙配置
```cmd
# Windows 防火墙规则
netsh advfirewall firewall add rule name="Face Recognition HTTP" dir=in action=allow protocol=TCP localport=80
netsh advfirewall firewall add rule name="Face Recognition HTTPS" dir=in action=allow protocol=TCP localport=443
```

### 密码安全
1. 更改默认管理员密码
2. 使用强密码策略
3. 定期轮换密钥和密码

## 故障排除

### 常见问题

#### 1. Docker 服务启动失败
```cmd
# 检查 Docker 状态
docker info

# 重启 Docker Desktop
# Windows: 重启 Docker Desktop 应用

# 检查端口占用
netstat -an | findstr :8000
```

#### 2. 数据库连接失败
```cmd
# 检查数据库容器状态
docker-compose ps postgres

# 查看数据库日志
docker-compose logs postgres

# 测试数据库连接
docker-compose exec postgres psql -U face_user -d face_recognition -c "SELECT 1;"
```

#### 3. 前端无法访问后端
```cmd
# 检查网络连接
docker-compose exec frontend curl http://backend:8000/api/health

# 检查环境变量
docker-compose exec frontend env | grep API_URL
```

#### 4. 内存不足
```cmd
# 检查容器资源使用
docker stats

# 增加 Docker Desktop 内存限制
# Settings -> Resources -> Memory
```

### 日志分析
```cmd
# 查看错误日志
docker-compose logs backend | findstr ERROR
docker-compose logs frontend | findstr ERROR

# 查看访问日志
docker-compose logs nginx | findstr "GET\|POST"
```

## 生产环境部署

### 环境差异
1. **域名和证书**: 使用真实域名和有效 SSL 证书
2. **数据库**: 使用外部托管数据库服务
3. **文件存储**: 使用对象存储服务 (AWS S3, Azure Blob)
4. **负载均衡**: 使用外部负载均衡器
5. **监控**: 集成企业监控系统

### 部署清单
- [ ] 更新环境变量文件
- [ ] 配置有效 SSL 证书
- [ ] 设置防火墙规则
- [ ] 配置备份策略
- [ ] 设置监控告警
- [ ] 更改默认密码
- [ ] 测试所有功能
- [ ] 准备回滚计划

## 维护和更新

### 定期维护
1. **系统更新**: 定期更新 Docker 镜像
2. **数据备份**: 自动化数据备份
3. **日志清理**: 定期清理旧日志文件
4. **性能监控**: 监控系统性能指标

### 版本更新
```cmd
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose up -d
```

## 支持和帮助

### 获取帮助
```cmd
# 查看部署脚本帮助
.\scripts\deploy.ps1 help

# 查看服务状态
.\scripts\deploy.ps1 status

# 查看日志
.\scripts\deploy.ps1 logs
```

### 联系支持
如果遇到问题，请提供以下信息：
1. 操作系统版本
2. Docker 版本
3. 错误日志
4. 系统配置信息

---

**注意**: 这是一个完整的企业级人脸识别系统，请确保在部署前仔细阅读所有配置要求和安全建议。