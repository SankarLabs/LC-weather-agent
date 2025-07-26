#!/bin/bash

# Weather Agent Deployment Script
# Usage: ./scripts/deploy.sh [dev|prod|monitoring]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Prerequisites check passed!"
}

check_env_file() {
    log_info "Checking environment configuration..."
    
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_warning "Please edit .env file with your API keys before continuing."
            read -p "Press Enter when ready to continue..."
        else
            log_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    # Check for required environment variables
    source .env
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        log_error "OPENAI_API_KEY is not set in .env file"
        exit 1
    fi
    
    if [ -z "$OPENWEATHER_API_KEY" ] || [ "$OPENWEATHER_API_KEY" = "your_openweather_api_key_here" ]; then
        log_error "OPENWEATHER_API_KEY is not set in .env file"
        exit 1
    fi
    
    log_success "Environment configuration is valid!"
}

deploy_development() {
    log_info "Deploying development environment..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    
    # Build and start services
    docker-compose -f docker-compose.dev.yml build
    docker-compose -f docker-compose.dev.yml up -d
    
    log_success "Development environment deployed!"
    log_info "Access points:"
    log_info "  - Streamlit UI: http://localhost:8501"
    log_info "  - FastAPI: http://localhost:8000"
    log_info "  - API Docs: http://localhost:8000/docs"
}

deploy_production() {
    log_info "Deploying production environment..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose build
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check health
    check_health
    
    log_success "Production environment deployed!"
    log_info "Access points:"
    log_info "  - Streamlit UI: http://localhost:8501"
    log_info "  - FastAPI: http://localhost:8000"
    log_info "  - API Docs: http://localhost:8000/docs"
    log_info "  - Redis: localhost:6379"
}

deploy_monitoring() {
    log_info "Deploying with monitoring stack..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Create monitoring directories
    mkdir -p monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    
    # Create basic Prometheus config if it doesn't exist
    if [ ! -f "monitoring/prometheus.yml" ]; then
        create_prometheus_config
    fi
    
    # Build and start services with monitoring
    docker-compose --profile monitoring build
    docker-compose --profile monitoring up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 60
    
    # Check health
    check_health
    
    log_success "Production environment with monitoring deployed!"
    log_info "Access points:"
    log_info "  - Streamlit UI: http://localhost:8501"
    log_info "  - FastAPI: http://localhost:8000"
    log_info "  - API Docs: http://localhost:8000/docs"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Grafana: http://localhost:3000 (admin/admin)"
}

create_prometheus_config() {
    log_info "Creating Prometheus configuration..."
    
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'weather-api'
    static_configs:
      - targets: ['weather-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
}

check_health() {
    log_info "Checking service health..."
    
    # Check API health
    for i in {1..10}; do
        if curl -sf http://localhost:8000/health > /dev/null; then
            log_success "API is healthy"
            break
        elif [ $i -eq 10 ]; then
            log_error "API health check failed after 10 attempts"
            return 1
        else
            log_info "Waiting for API to be ready... (attempt $i/10)"
            sleep 5
        fi
    done
    
    # Check Streamlit health
    for i in {1..10}; do
        if curl -sf http://localhost:8501/_stcore/health > /dev/null; then
            log_success "Streamlit UI is healthy"
            break
        elif [ $i -eq 10 ]; then
            log_error "Streamlit health check failed after 10 attempts"
            return 1
        else
            log_info "Waiting for Streamlit to be ready... (attempt $i/10)"
            sleep 5
        fi
    done
}

stop_services() {
    log_info "Stopping all services..."
    
    # Stop development services
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    
    # Stop production services
    docker-compose down 2>/dev/null || true
    
    # Stop monitoring services
    docker-compose --profile monitoring down 2>/dev/null || true
    
    log_success "All services stopped!"
}

view_logs() {
    log_info "Viewing logs..."
    docker-compose logs -f
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    log_info "\nContainer resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

show_help() {
    echo "Weather Agent Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev         Deploy development environment"
    echo "  prod        Deploy production environment"
    echo "  monitoring  Deploy with monitoring stack"
    echo "  stop        Stop all services"
    echo "  logs        View service logs"
    echo "  status      Show service status"
    echo "  health      Check service health"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev                 # Deploy for development"
    echo "  $0 prod                # Deploy for production"
    echo "  $0 monitoring          # Deploy with Prometheus/Grafana"
    echo "  $0 stop                # Stop all services"
}

# Main script
main() {
    case "$1" in
        "dev")
            check_prerequisites
            check_env_file
            deploy_development
            ;;
        "prod")
            check_prerequisites
            check_env_file
            deploy_production
            ;;
        "monitoring")
            check_prerequisites
            check_env_file
            deploy_monitoring
            ;;
        "stop")
            stop_services
            ;;
        "logs")
            view_logs
            ;;
        "status")
            show_status
            ;;
        "health")
            check_health
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        "")
            log_error "No command specified. Use 'help' for usage information."
            exit 1
            ;;
        *)
            log_error "Unknown command: $1. Use 'help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"