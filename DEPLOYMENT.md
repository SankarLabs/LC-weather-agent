# Weather Agent Deployment Guide

This guide covers deploying the Weather Agent application with UI and containerization.

## 📋 Prerequisites

- Docker Desktop or Docker Engine 20.10+
- Docker Compose v2.0+
- At least 2GB RAM available
- OpenAI API key
- OpenWeatherMap API key

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd my-agent-project

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or your preferred editor
```

**Required environment variables:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

### 2. Development Deployment

For local development with hot reload:

```bash
# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Or with Redis for memory persistence
docker-compose -f docker-compose.dev.yml --profile with-redis up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

**Access points:**
- **Streamlit UI**: http://localhost:8501
- **FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Production Deployment

For production-ready deployment:

```bash
# Start production services
docker-compose up -d

# Or with monitoring stack
docker-compose --profile monitoring up -d

# Or with nginx reverse proxy
docker-compose --profile production up -d
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Weather APIs  │
│   (Port 8501)   │◄──►│   (Port 8000)   │◄──►│   (External)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       
         │                       │                       
         ▼                       ▼                       
┌─────────────────┐    ┌─────────────────┐              
│     Redis       │    │   LangChain     │              
│   (Port 6379)   │◄──►│   Agent Core    │              
└─────────────────┘    └─────────────────┘              
```

## 🐳 Docker Services

### Core Services

| Service | Description | Port | Health Check |
|---------|-------------|------|--------------|
| `weather-api` | FastAPI backend | 8000 | `/health` |
| `weather-ui` | Streamlit interface | 8501 | `/_stcore/health` |
| `redis` | Memory & caching | 6379 | `redis-cli ping` |

### Optional Services (Profiles)

| Service | Profile | Description | Port |
|---------|---------|-------------|------|
| `nginx` | production | Reverse proxy | 80, 443 |
| `prometheus` | monitoring | Metrics collection | 9090 |
| `grafana` | monitoring | Metrics visualization | 3000 |

## 🔧 Configuration Options

### Environment Variables

Create `.env` file with these variables:

```bash
# Required API Keys
OPENAI_API_KEY=sk-...
OPENWEATHER_API_KEY=your_key

# Optional LLM providers
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# LangSmith Tracing (Recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=weather-agent-prod

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
MAX_RETRIES=3
REQUEST_TIMEOUT=30

# Redis Settings (if using external Redis)
REDIS_URL=redis://localhost:6379

# Monitoring (optional)
GRAFANA_PASSWORD=secure_password
```

### Docker Compose Overrides

Create `docker-compose.override.yml` for custom configurations:

```yaml
version: '3.8'

services:
  weather-api:
    environment:
      - CUSTOM_SETTING=value
    volumes:
      - ./custom-config:/app/config

  weather-ui:
    ports:
      - "8502:8501"  # Custom port
```

## 🚀 Deployment Commands

### Basic Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart specific service
docker-compose restart weather-api

# Update and restart
docker-compose pull && docker-compose up -d
```

### Health Checks

```bash
# Check all services
docker-compose ps

# Test API health
curl http://localhost:8000/health

# Test Streamlit health
curl http://localhost:8501/_stcore/health

# Test Redis
docker-compose exec redis redis-cli ping
```

### Scaling

```bash
# Scale API service
docker-compose up -d --scale weather-api=3

# Scale with load balancer
docker-compose --profile production up -d --scale weather-api=3
```

## 🌐 Production Deployment Options

### 1. Single Server Deployment

**Requirements:**
- Ubuntu 20.04+ / CentOS 8+ / Amazon Linux 2
- 4GB RAM, 2 CPU cores
- Docker & Docker Compose installed

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone <repo-url>
cd my-agent-project
cp .env.example .env
# Edit .env with production values
docker-compose --profile production up -d
```

### 2. AWS ECS Deployment

```bash
# Install AWS CLI and ECS CLI
pip install awscli ecs-cli

# Configure ECS cluster
ecs-cli configure --cluster weather-agent --region us-west-2
ecs-cli up --keypair your-keypair --capability-iam --size 2 --instance-type t3.medium

# Deploy services
ecs-cli compose --file docker-compose.yml service up
```

### 3. Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests:

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/secret.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/ingress.yml
```

### 4. Cloud Run / Container Services

For serverless deployment:

```bash
# Build and push to registry
docker build -t gcr.io/project-id/weather-agent:latest .
docker push gcr.io/project-id/weather-agent:latest

# Deploy to Cloud Run
gcloud run deploy weather-agent --image gcr.io/project-id/weather-agent:latest --platform managed
```

## 🔒 Security Configuration

### Production Security Checklist

- [ ] Set strong Redis password
- [ ] Use HTTPS (TLS certificates)
- [ ] Configure firewall rules
- [ ] Set up container security scanning
- [ ] Enable API rate limiting
- [ ] Configure CORS properly
- [ ] Use secrets management
- [ ] Enable audit logging

### SSL/TLS Setup

Create SSL certificates and update nginx configuration:

```bash
# Using Let's Encrypt
sudo certbot --nginx -d your-domain.com

# Or use existing certificates
cp your-cert.pem nginx/ssl/
cp your-key.pem nginx/ssl/
```

## 📊 Monitoring & Observability

### Prometheus Metrics

Enable monitoring profile:

```bash
docker-compose --profile monitoring up -d
```

**Access:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Application Logs

```bash
# View application logs
docker-compose logs -f weather-api
docker-compose logs -f weather-ui

# Export logs to file
docker-compose logs --no-color > app.log

# Real-time log monitoring
docker-compose logs -f --tail=100
```

### Health Monitoring

Set up automated health checks:

```bash
#!/bin/bash
# health-check.sh
curl -f http://localhost:8000/health || echo "API down"
curl -f http://localhost:8501/_stcore/health || echo "UI down"
```

## 🚨 Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Check environment variables
docker-compose exec weather-api env | grep API_KEY

# Solution: Update .env file and restart
docker-compose restart weather-api
```

**2. Memory Issues**
```bash
# Check container memory usage
docker stats

# Solution: Increase Docker memory limit or add swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**3. Port Conflicts**
```bash
# Check port usage
sudo netstat -tulpn | grep :8000

# Solution: Change ports in docker-compose.yml
```

**4. Container Won't Start**
```bash
# Check container logs
docker-compose logs weather-api

# Check container status
docker-compose ps

# Rebuild container
docker-compose build --no-cache weather-api
```

### Debug Mode

Enable debug logging:

```bash
# Add to .env
LOG_LEVEL=DEBUG
LANGCHAIN_VERBOSE=true

# Restart services
docker-compose restart
```

## 🔄 Updates & Maintenance

### Application Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# Or rolling update
docker-compose up -d --force-recreate --no-deps weather-api
```

### Backup & Recovery

```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./backup/

# Backup application data
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/ .env

# Restore Redis data
docker-compose down
docker cp ./backup/dump.rdb $(docker-compose ps -q redis):/data/
docker-compose up -d
```

## 📈 Performance Tuning

### Resource Optimization

```yaml
# In docker-compose.yml
services:
  weather-api:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          memory: 256M
```

### Caching Strategy

- Redis TTL: 600 seconds (weather data)
- Memory window: 10 conversations
- API rate limiting: 100 requests/minute

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoint
ab -n 1000 -c 10 http://localhost:8000/health

# Test with curl
for i in {1..100}; do
  curl -s http://localhost:8000/health > /dev/null
  echo "Request $i completed"
done
```

## 🎯 Next Steps

1. **Set up CI/CD pipeline** - Automate deployments
2. **Configure monitoring alerts** - Set up notifications
3. **Implement rate limiting** - Protect against abuse
4. **Add database persistence** - Store conversation history
5. **Set up backup automation** - Scheduled backups
6. **Configure CDN** - Improve global performance

## 📞 Support

- **Documentation**: Check README.md for usage details
- **Issues**: Report bugs via GitHub Issues
- **Logs**: Check application logs for error details
- **Health**: Use `/health` endpoints for status checks

---

**Built with ❤️ using Docker, FastAPI, Streamlit, and LangChain**