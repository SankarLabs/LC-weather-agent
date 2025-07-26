# LangChain Tools Examples

This directory contains custom tool implementations for LangChain agents. These examples demonstrate best practices for creating robust, production-ready tools that integrate seamlessly with agent workflows.

## 📁 Files in this Directory

### `custom_tool.py`
Comprehensive examples of custom tool development including:
- **TextAnalysisTool**: Sentiment analysis, readability, keyword extraction, summarization
- **FileOperationTool**: File system operations (read, write, delete, list)
- **DataProcessingTool**: JSON/CSV data parsing, filtering, aggregation, transformation

### `web_search_tool.py`
Web search tool implementations featuring:
- **WebSearchTool**: General web search with multiple providers
- **NewsSearchTool**: News-specific search with date filtering
- **AcademicSearchTool**: Academic paper search with field filtering
- Rate limiting, caching, and error handling patterns

## 🎯 Key Patterns Demonstrated

### 1. Tool Structure
All tools follow the LangChain `BaseTool` pattern:
```python
class CustomTool(BaseTool):
    name: str = "tool_name"
    description: str = "Clear description of what the tool does"
    args_schema: Type[BaseModel] = InputSchema
    
    def _run(self, arg1: str, arg2: int) -> str:
        # Tool implementation
        pass
    
    async def _arun(self, arg1: str, arg2: int) -> str:
        # Async implementation (optional)
        pass
```

### 2. Input Validation
Using Pydantic models for robust input validation:
```python
class ToolInput(BaseModel):
    query: str = Field(description="The search query")
    limit: int = Field(default=5, ge=1, le=20, description="Number of results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
```

### 3. Error Handling
Comprehensive error handling patterns:
```python
def _run(self, query: str) -> str:
    try:
        result = self.execute_operation(query)
        return f"Success: {result}"
    except SpecificException as e:
        logger.error(f"Specific error: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Tool execution failed: {str(e)}"
```

### 4. Rate Limiting
For external API tools:
```python
class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        # Rate limiting logic
        pass
```

### 5. Caching
For expensive operations:
```python
class SearchCache:
    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        self.cache = {}
        # Caching implementation
```

## 🚀 Usage Patterns

### Basic Tool Usage
```python
# Create and test a tool
tool = CustomTool()
result = tool._run("test input")
print(result)
```

### Integration with Agents
```python
from langchain.agents import create_react_agent, AgentExecutor

# Create tools list
tools = [
    WebSearchTool(),
    FileOperationTool(),
    DataProcessingTool()
]

# Create agent with tools
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Use the agent
result = agent_executor.invoke({"input": "Search for Python tutorials"})
```

### Tool Composition
```python
def create_research_toolkit():
    """Create a comprehensive research toolkit."""
    return [
        WebSearchTool(),
        NewsSearchTool(),
        AcademicSearchTool(),
        TextAnalysisTool(),
        FileOperationTool()
    ]
```

## 🧪 Testing Your Tools

### Unit Testing Pattern
```python
def test_custom_tool():
    tool = CustomTool()
    
    # Test normal operation
    result = tool._run("valid input")
    assert "Success" in result
    
    # Test error handling
    result = tool._run("")
    assert "Error" in result
```

### Integration Testing
```python
def test_tool_in_agent():
    tools = [CustomTool()]
    agent = create_test_agent(tools)
    
    result = agent.invoke({"input": "test query"})
    assert result["success"]
```

## 🔧 Development Guidelines

### When Creating New Tools

1. **Clear Purpose**: Each tool should have a single, well-defined purpose
2. **Robust Input Validation**: Use Pydantic models to validate all inputs
3. **Comprehensive Error Handling**: Handle all possible error scenarios gracefully
4. **Good Documentation**: Include clear descriptions and examples
5. **Testing**: Write tests for both success and failure cases

### Tool Naming Conventions
- Use descriptive names: `WebSearchTool` not `SearchTool`
- Use consistent suffixes: `Tool` for all tools
- Keep descriptions under 200 characters
- Make descriptions action-oriented: "Search the web for..." not "A tool that searches..."

### Input Schema Best Practices
- Use descriptive field names
- Include helpful descriptions for each field
- Set reasonable defaults where appropriate
- Add validators for complex validation rules
- Use appropriate types (str, int, List, Dict, etc.)

## 📚 External Dependencies

Some tools may require additional packages:

```bash
# For web search tools
pip install requests

# For data processing tools  
pip install pandas numpy

# For file processing tools
pip install python-magic

# For academic search
pip install scholarly arxiv

# For rate limiting
pip install ratelimit
```

## 🔒 Security Considerations

### API Keys
- Never hardcode API keys in tool code
- Use environment variables: `os.getenv("API_KEY")`
- Validate API keys before making requests
- Implement proper key rotation procedures

### Input Sanitization
- Always validate and sanitize user inputs
- Prevent injection attacks in search queries
- Limit input lengths to reasonable bounds
- Filter out potentially harmful characters

### Output Safety
- Sanitize outputs before returning to users
- Don't expose internal error details to end users
- Implement content filtering for inappropriate results
- Rate limit to prevent abuse

## 🎛️ Configuration Management

### Environment Variables
Tools should use environment variables for configuration:

```python
# .env file
SEARCH_API_KEY=your_api_key_here
SEARCH_RATE_LIMIT=60
CACHE_TTL_MINUTES=30
MAX_RESULTS=20

# In your tool
class WebSearchTool(BaseTool):
    def __init__(self):
        self.api_key = os.getenv("SEARCH_API_KEY")
        self.rate_limit = int(os.getenv("SEARCH_RATE_LIMIT", 60))
        self.cache_ttl = int(os.getenv("CACHE_TTL_MINUTES", 30))
```

### Tool Configuration Classes
```python
from pydantic import BaseModel

class ToolConfig(BaseModel):
    api_key: str
    rate_limit: int = 60
    cache_enabled: bool = True
    timeout_seconds: int = 30
    
class ConfigurableTool(BaseTool):
    def __init__(self, config: ToolConfig):
        self.config = config
        super().__init__()
```

## 📊 Monitoring and Logging

### Logging Best Practices
```python
import logging

logger = logging.getLogger(__name__)

def _run(self, query: str) -> str:
    logger.info(f"Tool {self.name} executing with query: {query[:100]}")
    
    try:
        result = self.execute_query(query)
        logger.info(f"Tool {self.name} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Tool {self.name} failed: {e}")
        return f"Error: {str(e)}"
```

### Performance Monitoring
```python
import time
from typing import Dict, Any

def _run(self, query: str) -> str:
    start_time = time.time()
    
    try:
        result = self.execute_query(query)
        execution_time = time.time() - start_time
        
        # Log performance metrics
        logger.info(f"Tool execution time: {execution_time:.2f}s")
        
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Tool failed after {execution_time:.2f}s: {e}")
        return f"Error: {str(e)}"
```

## 🔄 Advanced Patterns

### Tool Chaining
Create tools that can work together:

```python
class MultiStepTool(BaseTool):
    def __init__(self):
        self.search_tool = WebSearchTool()
        self.analysis_tool = TextAnalysisTool()
    
    def _run(self, query: str) -> str:
        # Step 1: Search for information
        search_results = self.search_tool._run(query)
        
        # Step 2: Analyze the results
        analysis = self.analysis_tool._run(search_results, "summary")
        
        return f"Search Results:\n{search_results}\n\nAnalysis:\n{analysis}"
```

### Conditional Tool Execution
```python
def _run(self, query: str, analysis_type: str) -> str:
    if analysis_type == "sentiment":
        return self._analyze_sentiment(query)
    elif analysis_type == "keywords":
        return self._extract_keywords(query)
    else:
        # Default behavior
        return self._general_analysis(query)
```

### Tool State Management
```python
class StatefulTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.state = {}
        self.history = []
    
    def _run(self, command: str) -> str:
        # Update state based on command
        self.history.append(command)
        
        if command.startswith("remember"):
            key, value = command.split(" ", 2)[1:]
            self.state[key] = value
            return f"Remembered: {key} = {value}"
        
        elif command.startswith("recall"):
            key = command.split(" ", 1)[1]
            return f"{key} = {self.state.get(key, 'not found')}"
```

## 🧩 Integration Examples

### With Memory Chains
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def create_tool_with_memory():
    memory = ConversationBufferMemory()
    
    class MemoryAwareTool(BaseTool):
        def __init__(self, memory):
            super().__init__()
            self.memory = memory
        
        def _run(self, query: str) -> str:
            # Access conversation history
            history = self.memory.load_memory_variables({})
            
            # Use history to provide context-aware responses
            context = history.get("history", "")
            return f"Based on our conversation: {context}\nResponse to '{query}': ..."
```

### With RAG Systems
```python
def create_rag_aware_tool(vectorstore):
    class RAGTool(BaseTool):
        def __init__(self, vectorstore):
            super().__init__()
            self.vectorstore = vectorstore
        
        def _run(self, query: str) -> str:
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(query, k=3)
            
            # Use retrieved context in tool operation
            context = "\n".join([doc.page_content for doc in docs])
            return f"Based on knowledge base:\n{context}\n\nAnswer: ..."
```

## 🚀 Production Deployment

### Containerization
```dockerfile
# Dockerfile for tools
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY tools/ ./tools/
ENV PYTHONPATH=/app

CMD ["python", "-m", "tools.web_search_tool"]
```

### Health Checks
```python
def health_check() -> Dict[str, Any]:
    """Check tool health and dependencies."""
    try:
        # Test external APIs
        test_search = WebSearchTool()
        test_result = test_search._run("test query")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "tools_available": ["web_search", "file_operations", "data_processing"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

### Scaling Considerations
- Use connection pooling for external APIs
- Implement proper caching strategies
- Consider async implementations for I/O heavy tools
- Monitor resource usage and set appropriate limits

## 📈 Performance Optimization

### Caching Strategies
```python
from functools import lru_cache
import redis

# In-memory caching
@lru_cache(maxsize=100)
def cached_operation(query: str) -> str:
    return expensive_operation(query)

# Redis caching
class RedisCachedTool(BaseTool):
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def _run(self, query: str) -> str:
        # Check cache first
        cached_result = self.redis_client.get(f"tool_result:{query}")
        if cached_result:
            return cached_result.decode('utf-8')
        
        # Execute operation
        result = self.execute_operation(query)
        
        # Cache result
        self.redis_client.setex(f"tool_result:{query}", 3600, result)
        return result
```

### Async Implementation
```python
import asyncio
import aiohttp

class AsyncWebSearchTool(BaseTool):
    async def _arun(self, query: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.search.com?q={query}") as response:
                data = await response.json()
                return self._process_results(data)
```

## 🔍 Debugging Tools

### Debug Mode
```python
class DebuggableTool(BaseTool):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
    
    def _run(self, query: str) -> str:
        if self.debug:
            print(f"DEBUG: Tool input: {query}")
            print(f"DEBUG: Tool state: {self.__dict__}")
        
        result = self.execute_operation(query)
        
        if self.debug:
            print(f"DEBUG: Tool output: {result}")
        
        return result
```

## 📝 Documentation Standards

### Tool Docstrings
```python
class WellDocumentedTool(BaseTool):
    """
    A well-documented tool that demonstrates proper documentation standards.
    
    This tool performs X operation and is useful for Y scenarios.
    It integrates with Z external service and handles A, B, C error cases.
    
    Examples:
        >>> tool = WellDocumentedTool()
        >>> result = tool._run("example input")
        >>> print(result)
        "Expected output format"
    
    Note:
        This tool requires API_KEY environment variable to be set.
        Rate limiting: 100 requests per minute.
        Cache TTL: 30 minutes.
    """
    
    name: str = "well_documented_tool"
    description: str = """Performs specific operation with clear description.
    Input should be a string query. Returns formatted results."""
```

## 🎯 Next Steps

To extend this tools collection:

1. **Identify needs** in your specific use case
2. **Follow the patterns** established in existing tools
3. **Write comprehensive tests** for new tools
4. **Document thoroughly** with examples and usage patterns
5. **Consider edge cases** and error scenarios
6. **Test integration** with agents and chains

## 📚 Additional Resources

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Python Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [API Design Best Practices](https://restfulapi.net/)

Remember: Good tools are the foundation of powerful agents. Invest time in making them robust, well-tested, and thoroughly documented!