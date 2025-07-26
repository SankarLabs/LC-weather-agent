# LangChain Weather Agent Implementation

## FEATURE:
Build a comprehensive weather information agent using LangChain that can:
- Provide current weather conditions for any location worldwide
- Deliver detailed weather forecasts (5-day, hourly predictions)
- Send weather alerts and severe weather warnings
- Offer location-based weather recommendations (clothing, activities, travel)
- Support natural language queries with conversation memory
- Provide weather comparisons between multiple locations
- Include weather trend analysis and seasonal insights
- Integrate with user preferences for personalized weather updates

## LANGCHAIN COMPONENTS:
- **Agents**: OpenAI Tools Agent with weather-specific tool integration
- **Tools**: 
  - WeatherTool for current conditions using OpenWeatherMap API
  - ForecastTool for extended weather predictions
  - AlertTool for severe weather monitoring
  - LocationTool for geocoding and location resolution
- **Memory**: ConversationBufferWindowMemory for context-aware weather conversations 
- **LLMs**: ChatOpenAI with temperature control for natural weather responses
- **Chains**: Agent executor with proper tool calling and response formatting
- **Prompts**: System prompts optimized for weather interpretation and recommendations
- **Output Parsers**: Structured output for weather data presentation
- **Callbacks**: Custom handlers for weather API monitoring and usage tracking

## EXAMPLES:
Refer to these example files and follow their patterns:

- `examples/agent_chain.py` - Agent architecture with tool integration patterns
- `examples/basic_chain.py` - Chain creation and error handling fundamentals
- `examples/memory_chain.py` - Conversation memory for follow-up weather queries
- `examples/tools/custom_tool.py` - Custom tool implementation patterns
- `examples/tools/web_search_tool.py` - External API integration best practices
- `tests/test_agents.py` - Agent testing with mock tools and real API validation
- `tests/test_tools.py` - Tool testing patterns for external API integrations

Follow these patterns for:
- Proper agent initialization and tool registration
- API key management and secure external service calls
- Error handling for network failures and API limits
- Conversation flow with weather context preservation
- Input validation for location queries and weather requests

## DOCUMENTATION:
Essential LangChain and Weather API documentation to reference:

- **LangChain Agents**: https://python.langchain.com/docs/modules/agents
- **Custom Tools**: https://python.langchain.com/docs/modules/agents/tools/custom_tools
- **Tool Calling**: https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
- **Agent Executor**: https://python.langchain.com/docs/modules/agents/concepts#agentexecutor
- **Memory Integration**: https://python.langchain.com/docs/modules/memory/agent_with_memory
- **OpenWeatherMap API**: https://openweathermap.org/api
- **Weather API Documentation**: https://openweathermap.org/current
- **Geocoding API**: https://openweathermap.org/api/geocoding-api

## TESTING REQUIREMENTS:
Implement comprehensive tests covering:

1. **Unit Tests**:
   - Individual weather tool functionality (current, forecast, alerts)
   - Location geocoding and coordinate resolution
   - Weather data parsing and formatting
   - Agent tool selection and execution logic
   - Memory persistence across weather conversations
   - Error handling for invalid locations and API failures

2. **Integration Tests**:
   - End-to-end weather queries with real API calls
   - Multi-location weather comparisons
   - Conversation flow with weather follow-up questions
   - Alert system integration and notification delivery
   - Weather recommendation generation based on conditions

3. **Performance Tests**:
   - Response time benchmarks for weather API calls
   - Memory usage during extended weather conversations
   - API rate limit handling and request optimization
   - Concurrent user request handling

## OTHER CONSIDERATIONS:

### API Integration:
- Use OpenWeatherMap API for reliable weather data (free tier: 1000 calls/day)
- Implement proper API key rotation and usage monitoring
- Handle API rate limits with exponential backoff strategies
- Cache weather data appropriately (current: 10min, forecast: 1hr)
- Support fallback weather services for high availability

### Error Handling:
- Graceful handling of invalid location queries
- Network timeout and connection error recovery
- API quota exceeded scenarios with user notifications
- Malformed weather data parsing with sensible defaults
- Location disambiguation for ambiguous queries

### Weather Data Processing:
- Convert temperature units based on user preference or location
- Interpret weather codes into human-readable descriptions
- Calculate feels-like temperature and weather indices
- Process wind speed, humidity, and pressure data
- Handle timezone conversions for accurate local weather

### Conversation Features:
- Maintain weather conversation context across sessions
- Support follow-up questions about previously queried locations
- Remember user weather preferences and frequent locations
- Provide proactive weather updates for saved locations
- Smart location suggestions based on conversation history

### Security:
- Secure API key storage and environment variable management
- Input sanitization for location queries to prevent injection
- Rate limiting to prevent API abuse
- User data privacy for location and preference storage
- Audit logging for weather data access and usage

### Performance Optimization:
- Implement intelligent caching for frequently requested locations
- Batch processing for multiple location weather requests
- Async API calls for improved response times
- Optimize memory usage for long weather conversations
- Monitor and alert on API performance degradation

### User Experience:
- Natural language processing for various weather query formats
- Smart location detection from user input (city names, coordinates, landmarks)
- Weather visualization suggestions for complex data
- Personalized weather recommendations based on user activity
- Context-aware responses that reference previous weather discussions

### Monitoring and Analytics:
- Track weather API usage and cost optimization
- Monitor popular weather queries and location patterns
- Log weather alert effectiveness and user engagement
- Performance metrics for response times and accuracy
- User satisfaction tracking for weather recommendations

### Environment Setup:
Include comprehensive setup instructions for:
- OpenWeatherMap API key registration and configuration
- Environment variable setup for weather service credentials
- Dependencies installation (requests, pydantic, langchain packages)
- Testing configuration with mock weather data
- Development vs production API endpoint configuration

The weather agent should provide accurate, timely, and contextually relevant weather information while maintaining natural conversation flow and robust error handling for production deployment.