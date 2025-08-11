# Weather Agent - LangChain Implementation

A comprehensive weather information agent built with LangChain that provides real-time weather data, forecasts, alerts, and personalized recommendations through natural language conversations.

## 🌟 Features

- **🌤️ Current Weather Conditions** - Real-time weather data for any location worldwide
- **📅 Weather Forecasts** - Up to 5-day detailed weather predictions  
- **⚠️ Weather Alerts** - Severe weather warnings and advisories
- **📍 Location Services** - Geocoding and coordinate resolution
- **🤖 Natural Conversations** - Memory-enabled chat with context preservation
- **💡 Smart Recommendations** - Weather-based suggestions for clothing, activities, and travel
- **🛡️ Production Ready** - Comprehensive error handling, caching, and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ 
- OpenAI API key
- OpenWeatherMap API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd my-agent-project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  #  On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your_openai_api_key
   # OPENWEATHER_API_KEY=your_openweather_api_key
   ```

5. **Test the installation:**
   ```bash
   python weather_agent_demo.py
   ```

## 📖 Usage

### Interactive Mode

Run the weather agent in interactive mode:

```bash
python weather_agent_demo.py
```

**Example queries:**
- "What's the weather like in London?"
- "Give me a 5-day forecast for New York"
- "Is it going to rain in Tokyo tomorrow?"
- "What should I wear in Paris today?"
- "Compare weather in Berlin and Madrid"

### Demo Mode

Run with predefined queries:

```bash
python weather_agent_demo.py demo
```

### Programmatic Usage

```python
from agents.weather_agent import create_weather_agent
from utils.config import get_config

# Create agent
config = get_config()
agent = create_weather_agent(config=config)

# Query weather
response = agent.query("What's the weather in London?")
print(response["response"])
```

## 🏗️ Architecture

### Components

- **Weather Tools** (`tools/weather_tools.py`)
  - `WeatherTool` - Current weather conditions
  - `ForecastTool` - Weather forecasts  
  - `LocationTool` - Geocoding and location resolution
  - `AlertTool` - Weather alerts and warnings

- **Agent System** (`agents/weather_agent.py`)
  - OpenAI Tools Agent with weather-specific prompts
  - Conversation memory with context preservation
  - Performance monitoring and health checks

- **Data Processing** (`utils/weather_formatter.py`)
  - Temperature unit conversions
  - Weather condition interpretation
  - Structured response formatting
  - Recommendation generation

- **Error Handling** (`utils/resilience.py`)
  - Circuit breaker pattern for API resilience
  - Retry logic with exponential backoff
  - Comprehensive input validation
  - Graceful degradation strategies

### Data Flow

```
User Query → Agent → Tool Selection → Weather API → Data Processing → Formatted Response
     ↑                                                                        ↓
     └─────────────────── Memory & Context ←──────────────────────────────────┘
```

## ⚙️ Configuration

Configuration is managed through environment variables and the `utils/config.py` module:

### Required Environment Variables

```bash
# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

# Optional Configuration
MODEL_NAME=gpt-3.5-turbo          # LLM model to use
TEMPERATURE=0.1                   # Model temperature
MAX_ITERATIONS=5                  # Agent max iterations
CACHE_DURATION=600               # Weather cache duration (seconds)
MEMORY_WINDOW_SIZE=10            # Conversation memory size
```

### Configuration Classes

```python
from utils.config import WeatherAgentConfig

config = WeatherAgentConfig(
    openai_api_key="your_key",
    openweather_api_key="your_weather_key",
    temperature=0.1,
    max_iterations=5
)
```

## 🧪 Testing

### Run All Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=tools --cov=agents --cov=utils --cov-report=html

# Run specific test file
python -m pytest tests/test_weather_tools.py -v
```

### Test Categories

- **Unit Tests** - Individual component testing with mocks
- **Integration Tests** - End-to-end workflow validation  
- **Performance Tests** - Response time and memory usage
- **Security Tests** - Input validation and error handling

### Manual Testing

```bash
# Test configuration
python -c "from utils.config import validate_environment; validate_environment()"

# Test weather tools
python tools/weather_tools.py

# Test agent
python agents/weather_agent.py
```

## 📊 Monitoring

### Health Checks

```python
agent = create_weather_agent()
health = agent.health_check()
print(health["status"])  # "healthy" or "unhealthy"
```

### Performance Metrics

```python
metrics = agent.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Error rate: {metrics['error_rate_percent']:.1f}%")
print(f"Avg response time: {metrics['avg_response_time']:.2f}s")
```

### Logging

Structured logging is configured throughout the application:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 🔒 Security

### Input Validation

All user inputs are validated and sanitized:

```python
from utils.resilience import InputValidator

# Location validation
clean_location = InputValidator.validate_location(user_input)

# Unit validation  
valid_units = InputValidator.validate_units(units)
```

### API Key Management

- Store API keys in environment variables
- Never commit keys to version control
- Use `.env` files for local development
- Implement key rotation for production

### Error Handling

Comprehensive error handling prevents information leakage:

```python
try:
    result = agent.query(user_input)
except WeatherAgentError as e:
    # Log error securely
    logger.error(f"Weather error: {e.severity.value}")
    # Return user-friendly message
    return "Weather service temporarily unavailable"
```

## 🚀 Production Deployment

### Environment Setup

1. **Production Configuration:**
   ```python
   from utils.config import get_config
   config = get_config(production=True)
   ```

2. **Environment Variables:**
   ```bash
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ENABLE_MONITORING=true
   API_RATE_LIMIT=1000
   ```

### Performance Optimization

- **Caching**: Weather data cached for 10 minutes (current) / 1 hour (forecast)
- **Rate Limiting**: Built-in API rate limiting and usage monitoring
- **Connection Pooling**: Efficient HTTP connection management
- **Memory Management**: Conversation window size limits

### Monitoring Setup

```python
from agents.weather_agent import WeatherAgentMonitor

monitor = WeatherAgentMonitor()
# Monitor tracks:
# - Request count and error rates
# - Response times
# - API usage patterns
```

### Scaling Considerations

- **Stateless Design**: Agent can be scaled horizontally
- **Database Integration**: Add persistent storage for memory
- **Load Balancing**: Distribute requests across multiple instances
- **Caching Layer**: Use Redis for distributed caching

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
ERROR: Missing required environment variables: ['OPENAI_API_KEY']
```
**Solution:** Ensure API keys are set in `.env` file

**2. Weather API Timeout**
```bash
Error: Weather service timeout for location 'London'
```
**Solution:** Check internet connection and API service status

**3. Location Not Found**
```bash
Error: Location 'XYZ' not found
```
**Solution:** Use more specific location names or coordinates

**4. Memory Issues**
```bash
Memory overflow in long conversations
```
**Solution:** Adjust `MEMORY_WINDOW_SIZE` in configuration

### Debug Mode

Enable verbose logging for debugging:

```bash
export VERBOSE=true
python weather_agent_demo.py
```

### Health Checks

Regular health monitoring:

```python
health = agent.health_check()
if health["status"] != "healthy":
    print(f"Issue: {health['error']}")
    # Check individual components
    for component, status in health["components"].items():
        if status != "healthy":
            print(f"Problem with {component}")
```

## 📚 API Reference

### Weather Agent

```python
class WeatherAgent:
    def query(self, user_input: str) -> Dict[str, Any]
    def reset_conversation(self) -> None
    def get_conversation_history(self) -> List[BaseMessage]
    def get_metrics(self) -> Dict[str, Any]
    def health_check(self) -> Dict[str, Any]
```

### Weather Tools

```python
class WeatherTool(BaseTool):
    name = "current_weather"
    def _run(self, location: str, units: str = "metric") -> str
    
class ForecastTool(BaseTool):
    name = "weather_forecast" 
    def _run(self, location: str, days: int = 5, units: str = "metric") -> str
```

### Configuration

```python
class WeatherAgentConfig(BaseModel):
    openai_api_key: str
    openweather_api_key: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_iterations: int = 5
    cache_duration: int = 600
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Run tests before committing
python -m pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - Framework for LLM applications
- **OpenWeatherMap** - Weather data API
- **OpenAI** - Language model services
- **LangChain Context Engineering** - Template foundation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/weather-agent/issues)
- **Documentation**: [Project Wiki](https://github.com/your-org/weather-agent/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/weather-agent/discussions)

---

**Built with ❤️ using LangChain Context Engineering principles**