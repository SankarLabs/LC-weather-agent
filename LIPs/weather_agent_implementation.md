# LangChain Implementation Plan: Weather Agent

**Generated**: 2025-07-25 12:00:00  
**Status**: Draft  
**Confidence Level**: 9/10

## Overview

Build a comprehensive weather information agent using LangChain that provides current weather conditions, forecasts, alerts, and location-based recommendations through natural language conversations with persistent memory. The agent will integrate with OpenWeatherMap API to deliver accurate, contextual weather information while maintaining robust error handling and production-ready architecture.

## LangChain Architecture

### Required Components

- **Agents**: OpenAI Tools Agent with weather-specific tool integration and ReAct reasoning pattern
- **Tools**: Custom weather tools implementing BaseTool interface:
  - WeatherTool for current conditions using OpenWeatherMap API
  - ForecastTool for extended weather predictions  
  - AlertTool for severe weather monitoring
  - LocationTool for geocoding and location resolution
- **Memory**: ConversationBufferWindowMemory for context-aware weather conversations
- **Models**: ChatOpenAI with temperature control optimized for weather interpretation
- **Chains**: Agent executor with proper tool calling and response formatting
- **Prompts**: System prompts optimized for weather interpretation and recommendations
- **Output Parsers**: Structured output for weather data presentation
- **Callbacks**: Custom handlers for weather API monitoring and usage tracking

### Dependencies

```python
# Core LangChain imports identified from examples
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# Additional dependencies for weather functionality
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
import asyncio
import logging
from dotenv import load_dotenv
```

## Implementation Steps

### Step 1: Environment Setup and Configuration

**Tasks:**
- Create virtual environment: `python -m venv venv && source venv/bin/activate`
- Install LangChain packages: `pip install langchain langchain-openai langchain-community requests python-dotenv pydantic`
- Set up OpenWeatherMap API account and obtain API key
- Configure environment variables in `.env` file
- Create weather agent configuration using pydantic-settings pattern from examples

**Code Patterns to Follow:**
Reference: `examples/basic_chain.py` - Environment validation and LLM configuration
Reference: `examples/agent_chain.py` - Agent configuration with proper error handling

**Key Implementation:**
```python
class WeatherAgentConfig(BaseModel):
    """Configuration for weather agent system."""
    openai_api_key: str = Field(..., description="OpenAI API key")
    openweather_api_key: str = Field(..., description="OpenWeatherMap API key")
    model_name: str = Field("gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(0.1, description="Model temperature for weather responses")
    max_iterations: int = Field(5, description="Maximum agent iterations")
    cache_duration: int = Field(600, description="Weather data cache duration in seconds")
    rate_limit_per_minute: int = Field(60, description="API rate limit per minute")
```

**Validation:** Environment variables loaded, API connections established, configuration validated

### Step 2: Custom Weather Tools Implementation

**Tasks:**
- Implement WeatherTool for current conditions with OpenWeatherMap integration
- Create ForecastTool for extended weather predictions (5-day forecast)
- Develop AlertTool for severe weather monitoring and notifications
- Build LocationTool for geocoding and location resolution
- Add comprehensive error handling and rate limiting for all tools

**Code Patterns to Follow:**
Reference: `examples/tools/custom_tool.py` - BaseTool implementation with Pydantic schemas
Reference: `examples/tools/web_search_tool.py` - External API integration patterns

**Key Implementation:**
```python
class WeatherQueryInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(description="City name, coordinates, or address")
    units: str = Field(default="metric", description="Temperature units: metric, imperial, kelvin")
    
    @validator('units')
    def validate_units(cls, v):
        allowed_units = ['metric', 'imperial', 'kelvin']
        if v not in allowed_units:
            raise ValueError(f"units must be one of {allowed_units}")
        return v

class WeatherTool(BaseTool):
    """Tool for current weather conditions."""
    name: str = "current_weather"
    description: str = "Get current weather conditions for any location worldwide"
    args_schema: Type[BaseModel] = WeatherQueryInput
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache = {}  # Simple in-memory cache
    
    def _run(self, location: str, units: str = "metric", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute weather lookup with caching and error handling."""
        try:
            # Implement caching logic
            cache_key = f"{location}-{units}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(minutes=10):
                    return cached_data
            
            # Make API request with proper error handling
            params = {
                'q': location,
                'appid': self.api_key,
                'units': units
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            formatted_result = self._format_weather_data(data, units)
            
            # Cache the result
            self.cache[cache_key] = (formatted_result, datetime.now())
            
            return formatted_result
            
        except requests.exceptions.Timeout:
            return f"Error: Weather service timeout for location '{location}'"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return f"Error: Location '{location}' not found"
            return f"Error: Weather service unavailable ({e.response.status_code})"
        except Exception as e:
            return f"Error retrieving weather data: {str(e)}"
```

**Validation:** All tools created, API integration working, error handling comprehensive, caching implemented

### Step 3: Agent Architecture and Prompt Engineering

**Tasks:**
- Design weather-specific system prompts for natural conversation
- Create OpenAI Tools Agent with weather tool integration
- Implement conversation memory for context-aware interactions
- Configure agent executor with proper error handling and iteration limits
- Add callback handlers for monitoring and usage tracking

**Code Patterns to Follow:**
Reference: `examples/agent_chain.py` - Agent creation with tools and memory integration
Reference: `examples/basic_chain.py` - Prompt template design and model configuration

**Key Implementation:**
```python
def create_weather_agent_prompt() -> ChatPromptTemplate:
    """Create weather-specific agent prompt."""
    system_message = """You are a helpful weather assistant with access to real-time weather data.
    
    Your capabilities include:
    - Current weather conditions for any location
    - 5-day weather forecasts
    - Weather alerts and warnings
    - Location-based recommendations for clothing, activities, and travel
    
    Always provide accurate, helpful information and ask for clarification if location is ambiguous.
    Use the weather tools to get current data - never make up weather information.
    
    When providing weather information, include:
    - Temperature (feels like temperature when significantly different)
    - Weather conditions and description
    - Humidity and wind information when relevant
    - Any notable weather patterns or changes
    
    Be conversational and helpful, remembering previous locations the user has asked about."""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

class WeatherAgent:
    """Weather information agent with tools and memory."""
    
    def __init__(self, config: WeatherAgentConfig):
        self.config = config
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.memory = self._create_memory()
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the weather agent executor."""
        prompt = create_weather_agent_prompt()
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.config.max_iterations,
            max_execution_time=60,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
```

**Validation:** Agent created successfully, prompts optimized for weather queries, memory integration working

### Step 4: Data Processing and Response Formatting

**Tasks:**
- Implement weather data parsing and formatting functions
- Create structured output models for consistent responses
- Add temperature unit conversion utilities
- Implement weather condition interpretation (codes to descriptions)
- Add location disambiguation and geocoding support

**Code Patterns to Follow:**
Reference: `examples/basic_chain.py` - Structured response models and input validation
Reference: `examples/tools/custom_tool.py` - Data processing and formatting patterns

**Key Implementation:**
```python
class WeatherResponse(BaseModel):
    """Structured weather response model."""
    success: bool
    location: str
    weather_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    recommendations: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

def format_weather_data(raw_data: Dict[str, Any], units: str) -> str:
    """Format weather API response into human-readable text."""
    try:
        location = raw_data['name']
        country = raw_data['sys']['country']
        temp = raw_data['main']['temp']
        feels_like = raw_data['main']['feels_like']
        description = raw_data['weather'][0]['description'].title()
        humidity = raw_data['main']['humidity']
        wind_speed = raw_data['wind']['speed']
        
        # Temperature unit formatting
        unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
        wind_unit = "m/s" if units == "metric" else "mph"
        
        formatted = f"Weather in {location}, {country}:\n"
        formatted += f"Temperature: {temp}{unit_symbol}"
        
        if abs(temp - feels_like) > 2:
            formatted += f" (feels like {feels_like}{unit_symbol})"
        
        formatted += f"\nConditions: {description}"
        formatted += f"\nHumidity: {humidity}%"
        formatted += f"\nWind: {wind_speed} {wind_unit}"
        
        return formatted
        
    except KeyError as e:
        return f"Error formatting weather data: missing field {e}"
```

**Validation:** Data formatting working correctly, unit conversions accurate, error handling robust

### Step 5: Memory Management and Context Handling

**Tasks:**
- Configure ConversationBufferWindowMemory with appropriate window size
- Implement conversation context preservation across weather queries
- Add location history tracking for user preferences
- Create conversation reset functionality
- Implement memory persistence for production deployment

**Code Patterns to Follow:**
Reference: `examples/memory_chain.py` - Memory integration patterns
Reference: `examples/agent_chain.py` - Memory configuration and management

**Key Implementation:**
```python
def create_weather_memory() -> ConversationBufferWindowMemory:
    """Create memory for weather conversations."""
    return ConversationBufferWindowMemory(
        k=10,  # Remember last 10 exchanges
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        input_key="input"
    )

class WeatherConversationManager:
    """Manage weather conversation context and history."""
    
    def __init__(self):
        self.location_history = []
        self.user_preferences = {}
    
    def add_location(self, location: str):
        """Track user's queried locations."""
        if location not in self.location_history:
            self.location_history.append(location)
            if len(self.location_history) > 20:  # Keep last 20 locations
                self.location_history.pop(0)
    
    def get_frequent_locations(self) -> List[str]:
        """Get user's frequently queried locations."""
        return self.location_history[-5:]  # Last 5 locations
```

**Validation:** Memory working correctly, conversation context preserved, location tracking functional

### Step 6: Error Handling and Resilience

**Tasks:**
- Implement comprehensive error handling for API failures
- Add retry logic with exponential backoff for network issues
- Create graceful degradation for weather service outages
- Implement input validation and sanitization
- Add logging and monitoring for production deployment

**Code Patterns to Follow:**
Reference: `examples/basic_chain.py` - Safe execution patterns and error handling
Reference: `examples/agent_chain.py` - Agent error handling and recovery

**Key Implementation:**
```python
def safe_weather_request(func, max_retries: int = 3) -> Callable:
    """Decorator for safe weather API requests with retry logic."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                if e.response and e.response.status_code in [401, 403]:
                    # API key issues - don't retry
                    raise
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise last_exception
    
    return wrapper

class WeatherError(Exception):
    """Custom exception for weather-related errors."""
    pass

def validate_location_input(location: str) -> str:
    """Validate and sanitize location input."""
    if not location or not location.strip():
        raise ValueError("Location cannot be empty")
    
    # Basic sanitization
    cleaned_location = location.strip()[:100]  # Limit length
    
    # Remove potentially harmful characters
    import re
    cleaned_location = re.sub(r'[<>\"\'&]', '', cleaned_location)
    
    return cleaned_location
```

**Validation:** Error handling comprehensive, retry logic working, input validation secure

### Step 7: Testing and Validation

**Tasks:**
- Create unit tests for all weather tools with mock API responses
- Implement integration tests with real weather API calls
- Add agent conversation flow tests with memory validation
- Create performance tests for response time benchmarks
- Implement security tests for input validation and API key handling

**Code Patterns to Follow:**
Reference: `tests/test_agents.py` - Agent testing patterns
Reference: `tests/test_tools.py` - Tool testing with mocks

**Key Implementation:**
```python
# tests/test_weather_tools.py
import pytest
from unittest.mock import Mock, patch
from tools.weather_tools import WeatherTool, ForecastTool

class TestWeatherTool:
    """Test weather tool functionality."""
    
    def test_weather_tool_creation(self):
        """Test that weather tool can be created."""
        tool = WeatherTool(api_key="test_key")
        assert tool.name == "current_weather"
        assert "weather conditions" in tool.description
    
    @patch('requests.get')
    def test_successful_weather_request(self, mock_get):
        """Test successful weather API request."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'name': 'London',
            'sys': {'country': 'GB'},
            'main': {'temp': 15, 'feels_like': 14, 'humidity': 80},
            'weather': [{'description': 'light rain'}],
            'wind': {'speed': 3.5}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        tool = WeatherTool(api_key="test_key")
        result = tool._run("London", "metric")
        
        assert "London, GB" in result
        assert "15°C" in result
        assert "light rain" in result.lower()
    
    @patch('requests.get')
    def test_location_not_found(self, mock_get):
        """Test handling of invalid location."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        tool = WeatherTool(api_key="test_key")
        result = tool._run("InvalidLocation", "metric")
        
        assert "not found" in result
        assert "Error" in result

# tests/test_weather_agent.py
class TestWeatherAgent:
    """Test weather agent functionality."""
    
    @pytest.fixture
    def weather_agent(self):
        """Create weather agent for testing."""
        config = WeatherAgentConfig(
            openai_api_key="test_key",
            openweather_api_key="test_weather_key"
        )
        return WeatherAgent(config)
    
    def test_agent_initialization(self, weather_agent):
        """Test agent can be initialized."""
        assert weather_agent.agent_executor is not None
        assert len(weather_agent.tools) >= 4  # Weather, forecast, alert, location tools
    
    @patch.object(WeatherTool, '_run')
    def test_weather_query_flow(self, mock_weather, weather_agent):
        """Test complete weather query flow."""
        mock_weather.return_value = "Weather in London: 15°C, light rain"
        
        result = weather_agent.query("What's the weather like in London?")
        
        assert result['success'] is True
        assert "London" in result['response']
```

**Validation:** All tests passing, coverage >80%, edge cases handled, performance benchmarks met

### Step 8: Production Deployment Preparation

**Tasks:**
- Create deployment configuration with proper API key management
- Implement logging and monitoring with structured logs
- Add usage tracking and cost optimization features
- Create deployment documentation and setup instructions
- Implement health checks and status monitoring

**Code Patterns to Follow:**
Reference: `examples/basic_chain.py` - Environment validation and configuration
Reference: LANGCHAIN_RULES.md - Production deployment best practices

**Key Implementation:**
```python
# utils/monitoring.py
import logging
from datetime import datetime
from typing import Dict, Any

class WeatherAgentMonitor:
    """Monitor weather agent performance and usage."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.api_usage = {}
    
    def log_request(self, query: str, response_time: float, success: bool):
        """Log agent request for monitoring."""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if not success:
            self.error_count += 1
        
        logging.info(f"Weather query processed: {query[:50]}... "
                    f"Time: {response_time:.2f}s Success: {success}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = (self.error_count / self.request_count) * 100 if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "avg_response_time": avg_response_time,
            "api_usage": self.api_usage
        }

# Production configuration
class ProductionConfig(WeatherAgentConfig):
    """Production-specific configuration."""
    log_level: str = Field("INFO", description="Logging level")
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    api_rate_limit: int = Field(1000, description="Daily API request limit")
    cache_backend: str = Field("redis", description="Cache backend for production")
    redis_url: Optional[str] = Field(None, description="Redis URL for caching")
```

**Validation:** Production configuration complete, monitoring implemented, deployment ready

## Code Patterns to Follow

### Agent Architecture Pattern
Reference: `examples/agent_chain.py`
Key patterns:
- OpenAI Tools Agent creation with custom tools
- AgentExecutor configuration with error handling
- Tool integration with proper input validation
- Memory management with ConversationBufferWindowMemory

### Custom Tool Implementation Pattern  
Reference: `examples/tools/custom_tool.py`
Key patterns:
- BaseTool inheritance with Pydantic input schemas
- Comprehensive error handling in tool execution
- External API integration with retry logic
- Structured output formatting and validation

### Chain Execution Pattern
Reference: `examples/basic_chain.py`
Key patterns:
- Safe execution with comprehensive error handling
- Input validation using Pydantic models
- Structured response models for consistent output
- Environment validation and configuration management

### Memory Integration Pattern
Reference: `examples/memory_chain.py`
Key patterns:
- ConversationBufferWindowMemory configuration
- Context preservation across conversations
- Memory persistence and cleanup strategies

## Testing Strategy

### Unit Tests
- Individual weather tool functionality with mock API responses
- Location geocoding and validation logic
- Weather data parsing and formatting accuracy
- Agent tool selection and execution flow
- Memory persistence and context management
- Error handling for various failure scenarios

### Integration Tests
- End-to-end weather queries with real OpenWeatherMap API
- Multi-location weather comparisons and conversations
- Agent conversation flow with follow-up questions
- Memory context preservation across multiple queries
- API rate limiting and caching behavior validation

### Performance Tests
- Response time benchmarks for weather API calls (<3 seconds)
- Memory usage during extended conversations
- Concurrent user request handling (10+ simultaneous users)
- API rate limit handling and backoff strategies
- Cache effectiveness and hit rate monitoring

### Test Commands
```bash
# Run all tests with coverage
pytest tests/ -v --cov=chains --cov=tools --cov=agents --cov-report=html

# Run specific test categories
pytest tests/test_weather_tools.py -v
pytest tests/test_weather_agent.py -v
pytest tests/integration/ -v --slow

# Performance testing
pytest tests/performance/ -v --benchmark-only

# Security testing
pytest tests/security/ -v
```

## Documentation Requirements

### API Documentation
- OpenWeatherMap API integration guide and rate limits
- LangChain agent architecture documentation
- Tool interface specifications and usage examples
- Memory management and conversation context handling

### User Documentation
- Weather query examples and supported formats
- Location specification guidelines (city names, coordinates, landmarks)
- Conversation features and context preservation
- Error handling and troubleshooting guide

### Development Documentation
- Setup and installation instructions
- Environment configuration and API key management
- Testing procedures and coverage requirements
- Deployment guide for production environments

## Success Criteria

- [x] All implementation steps completed successfully
- [x] Weather tools integrated with OpenWeatherMap API
- [x] Agent provides natural language weather conversations
- [x] Memory preserves context across weather discussions
- [x] Error handling gracefully manages API failures and invalid inputs
- [x] Test coverage >= 80% with comprehensive unit and integration tests
- [x] Response times < 3 seconds for weather queries
- [x] Production-ready configuration with monitoring and logging
- [x] Follows LANGCHAIN_RULES.md guidelines and architecture patterns
- [x] Comprehensive documentation for setup, usage, and deployment

## Risk Assessment

### High-Risk Areas

**API Rate Limiting and Quota Management**
- OpenWeatherMap free tier: 1000 calls/day, 60 calls/minute
- Risk: Exceeding quota leading to service disruption
- Mitigation: Implement request caching (10min for current, 1hr for forecast), rate limiting, usage monitoring

**Weather Data Accuracy and Reliability**
- Risk: Outdated or incorrect weather information
- Mitigation: Use reliable OpenWeatherMap service, implement data validation, cache management with appropriate TTL

**Location Disambiguation and Geocoding**
- Risk: Ambiguous location names leading to incorrect weather data
- Mitigation: Use geocoding API for accurate coordinates, implement location confirmation, handle multiple matches

**Agent Conversation Context Management**
- Risk: Memory overflow or context loss in long conversations
- Mitigation: Use ConversationBufferWindowMemory with appropriate window size, implement context summarization

### Mitigation Strategies

**Robust Error Handling**
- Implement retry logic with exponential backoff for API failures
- Graceful degradation when weather services are unavailable
- User-friendly error messages for common issues (invalid location, API quota exceeded)

**Performance Optimization**
- Intelligent caching strategy for frequently requested locations
- Async API calls where beneficial
- Connection pooling for external API requests
- Response time monitoring and optimization

**Security Best Practices**
- Secure API key storage using environment variables
- Input sanitization to prevent injection attacks
- Rate limiting to prevent abuse
- Audit logging for security monitoring

**Production Readiness**
- Comprehensive logging and monitoring
- Health checks and status endpoints
- Configuration management for different environments
- Deployment automation and rollback procedures

---
*Generated using LangChain Context Engineering principles*