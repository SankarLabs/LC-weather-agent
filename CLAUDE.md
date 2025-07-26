# CLAUDE.md - LangChain Context Engineering Guide

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with LangChain applications in this repository.

## 🔄 LangChain Core Principles

**IMPORTANT: These principles apply to ALL LangChain development:**

### LangChain Development Workflow
- **Always start with INITIAL.md** - Define chain/agent requirements before generating LIPs (LangChain Implementation Plans)
- **Use the LIP pattern**: INITIAL.md → `/generate-langchain-lip INITIAL.md` → `/execute-langchain-lip LIPs/filename.md`
- **Follow validation loops** - Each LIP must include comprehensive testing with mock providers
- **Context is King** - Include ALL necessary LangChain patterns, examples, and documentation

### Research Methodology for LangChain Applications
- **Web search extensively** - Always research LangChain patterns and best practices
- **Study official documentation** - python.langchain.com is the authoritative source
- **Pattern extraction** - Identify reusable chain architectures and agent patterns
- **Gotcha documentation** - Document async patterns, memory management, and streaming issues

## 📚 Project Overview & Architecture

This is a LangChain Context Engineering template - a comprehensive framework for building production-ready LangChain applications using systematic context engineering principles.

### Architecture Structure
```
langchain-context-engineering/
├── .claude/                    # Enhanced Claude Code configuration
│   └── settings.local.json    # Comprehensive permissions & project settings
├── LIPs/                      # LangChain Implementation Plans
│   ├── templates/             # Plan templates for chains, agents, tools
│   └── EXAMPLE_rag_chain.md   # Example implementation plan
├── examples/                  # LangChain code examples (critical for patterns)
│   ├── README.md              # Examples documentation
│   ├── basic_chain.py         # Chain creation, error handling, validation
│   ├── rag_chain.py          # RAG implementation patterns
│   ├── agent_chain.py        # Agent-based chain patterns
│   ├── memory_chain.py       # Conversation memory patterns
│   ├── streaming_chain.py    # Streaming response patterns
│   ├── tools/                # Custom tool implementations
│   │   ├── README.md         # Tools documentation
│   │   ├── custom_tool.py    # Custom tool patterns
│   │   ├── web_search_tool.py # Web search integration
│   │   └── vector_tool.py    # Vector store operations
│   └── tests/                # Example-specific test patterns
│       ├── README.md         # Testing documentation
│       ├── test_chains.py    # Chain testing patterns
│       ├── test_agents.py    # Agent testing patterns
│       ├── test_tools.py     # Tool testing patterns
│       └── conftest.py       # Pytest fixtures and configuration
├── tests/                    # Repository-level test patterns
├── chains/                   # Production chain implementations
│   ├── __init__.py
│   ├── base_chain.py        # Base chain with common patterns
│   ├── rag_chains.py        # RAG chain implementations
│   └── agent_chains.py      # Agent chain implementations
├── tools/                    # Production tool implementations
│   ├── __init__.py
│   ├── base_tool.py         # Base tool with common patterns
│   └── custom_tools.py      # Custom tool implementations
├── memory/                   # Memory management implementations
│   ├── __init__.py
│   ├── persistent_memory.py # Persistent memory patterns
│   └── context_memory.py    # Context-aware memory
├── utils/                    # Utility functions and helpers
│   ├── __init__.py
│   ├── validation.py        # Input/output validation
│   ├── settings.py          # Environment configuration
│   └── logging_config.py    # Structured logging setup
├── LANGCHAIN_RULES.md        # Comprehensive LangChain development guidelines
├── INITIAL.md               # Template for feature requests
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Modern Python project configuration
```

## 🚀 Development Setup Commands

```bash
# Modern Python environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with development extras
pip install -e ".[dev]"  # If using pyproject.toml
# OR traditional installation
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Environment configuration
cp .env.example .env
# Edit .env with required API keys

# Verification commands
python examples/basic_chain.py                    # Test basic functionality
python -c "import langchain; print(f'LangChain {langchain.__version__} installed successfully')"
python -c "from langchain_core.messages import HumanMessage; print('Core imports working')"

# Testing commands with comprehensive coverage
pytest examples/tests/ -v                         # Run example tests with verbose output
pytest tests/ -v --cov=chains --cov=tools        # Run all tests with coverage
pytest tests/test_chains.py::TestBasicChain      # Run specific test class
pytest --cov=. --cov-report=html                 # Generate HTML coverage report
pytest --tb=short --maxfail=3                    # Short traceback, stop after 3 failures
pytest -m "not slow"                             # Skip slow integration tests
pytest --durations=10                            # Show 10 slowest tests

# Development quality commands
black . --check                                   # Check code formatting
black .                                          # Format code
ruff check .                                     # Fast linting (replaces flake8)
ruff check . --fix                               # Auto-fix lint issues
mypy chains/ tools/ utils/                       # Type checking
pre-commit run --all-files                       # Run all pre-commit hooks

# LangChain specific validation
python -m langchain_core.utils.check_package_version  # Verify package compatibility
langchain --version                               # Check LangChain CLI version
```

## 🧱 LangChain Structure & Modularity

### File Organization Standards
- **Never create files longer than 300 lines** - Split into focused modules when approaching limit
- **Organize LangChain code into clearly separated modules** grouped by responsibility:
  - `chains/` - Chain implementations with proper composition patterns
  - `tools/` - Tool implementations using `BaseTool` interface
  - `memory/` - Memory management and persistence patterns
  - `utils/` - Shared utilities, validation, and configuration
- **Use clear, consistent imports** - Import from langchain packages appropriately
- **Use pydantic-settings and python-dotenv** for environment variables
- **Never hardcode sensitive information** - Always use .env files for API keys

### LangChain Component Development Standards

#### Chain Composition Patterns
```python
# Follow structured chain patterns with proper typing
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import asyncio

class ChainInput(BaseModel):
    """Standardized input validation for all chains."""
    query: str = Field(..., description="User query", min_length=1)
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

class ChainOutput(BaseModel):
    """Standardized output format for all chains."""
    response: str = Field(..., description="Chain response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    success: bool = Field(default=True, description="Success status")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")

def create_basic_chain(llm: ChatOpenAI) -> Any:
    """Create a basic chain with proper composition."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Respond concisely and accurately."),
        ("human", "{query}")
    ])
    
    return (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )
```

#### Agent Architecture Patterns
```python
# Agent patterns with proper tool integration
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Type

class AgentDependencies(BaseModel):
    """Dependencies for agent execution."""
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    verbose: bool = Field(default=False, description="Enable verbose logging")

def create_agent_with_tools(
    llm: ChatOpenAI,
    tools: List[BaseTool],
    system_prompt: str,
    dependencies: AgentDependencies
) -> AgentExecutor:
    """Create an agent with proper tool integration and error handling."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=dependencies.max_iterations,
        verbose=dependencies.verbose,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
```

#### Memory Management Patterns
```python
# Memory patterns with persistence and context awareness
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

class MemoryConfig(BaseModel):
    """Configuration for memory management."""
    memory_type: str = Field(default="buffer_window", description="Type of memory to use")
    max_token_limit: int = Field(default=2000, description="Maximum tokens for summary memory")
    window_size: int = Field(default=10, description="Window size for buffer memory")
    session_id: str = Field(..., description="Session identifier for memory persistence")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for persistence")

def create_memory(config: MemoryConfig, llm: ChatOpenAI) -> BaseMemory:
    """Create memory with proper configuration and persistence."""
    
    # Use Redis for persistence if available
    if config.redis_url:
        message_history = RedisChatMessageHistory(
            url=config.redis_url,
            session_id=config.session_id
        )
    else:
        message_history = None
    
    if config.memory_type == "summary_buffer":
        return ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=config.max_token_limit,
            chat_memory=message_history,
            return_messages=True,
            output_key="output",
            input_key="input"
        )
    else:
        return ConversationBufferWindowMemory(
            k=config.window_size,
            chat_memory=message_history,
            return_messages=True,
            output_key="output",
            input_key="input"
        )
```

#### Tool Development Standards
```python
# Custom tool patterns with proper validation and error handling
from langchain_core.tools import BaseTool
from pydantic import Field
from typing import Optional, Type, Any
import httpx
import asyncio

class CustomToolInput(BaseModel):
    """Input schema for custom tools."""
    query: str = Field(..., description="The search query or input")
    max_results: int = Field(default=5, description="Maximum number of results")

class WebSearchTool(BaseTool):
    """Example custom tool with proper async implementation."""
    
    name: str = "web_search"
    description: str = "Search the web for current information"
    args_schema: Type[BaseModel] = CustomToolInput
    
    api_key: str = Field(..., description="API key for search service")
    base_url: str = Field(default="https://api.example.com", description="Base URL for API")
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Async implementation of the tool."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query, "limit": max_results},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return f"Error in web search: {str(e)}"
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query, max_results))
```

## ⚙️ Environment Configuration

### Required Environment Variables
```bash
# Core LLM API Keys
OPENAI_API_KEY=your_openai_api_key           # Required for OpenAI models
ANTHROPIC_API_KEY=your_anthropic_api_key     # Optional for Claude models
GOOGLE_API_KEY=your_google_api_key           # Optional for Gemini models
AZURE_OPENAI_API_KEY=your_azure_key          # Optional for Azure OpenAI

# LangSmith Tracing (Highly Recommended)
LANGCHAIN_TRACING_V2=true                    # Enable detailed tracing
LANGCHAIN_API_KEY=your_langsmith_key         # LangSmith API key
LANGCHAIN_PROJECT=your_project_name          # Project name for organized tracing
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # LangSmith endpoint

# Vector Store Configuration
PINECONE_API_KEY=your_pinecone_key_here     # Pinecone vector store
PINECONE_ENVIRONMENT=your_pinecone_env      # Pinecone environment
CHROMA_PERSIST_DIRECTORY=./chroma_db        # Local Chroma persistence
WEAVIATE_URL=http://localhost:8080          # Weaviate instance URL

# Memory & Persistence
REDIS_URL=redis://localhost:6379            # Redis for chat history persistence
POSTGRES_URL=postgresql://user:pass@localhost/db  # PostgreSQL for advanced persistence

# Search & External APIs
SERPAPI_API_KEY=your_serpapi_key            # SerpAPI for web search
TAVILY_API_KEY=your_tavily_key              # Tavily search API
BRAVE_API_KEY=your_brave_key                # Brave search API

# Testing Configuration
PYTEST_FAST_MODE=false                       # Skip slow integration tests
PYTEST_ALLOW_API_TESTS=true                  # Allow tests that make API calls
PYTEST_DEBUG=false                           # Debug mode for tests
LANGCHAIN_TEST_MODE=true                     # Use test configurations

# Application Configuration
LOG_LEVEL=INFO                               # Logging level
ENVIRONMENT=development                      # Environment (development/staging/production)
MAX_RETRIES=3                               # Maximum retries for API calls
REQUEST_TIMEOUT=30                          # Request timeout in seconds
```

### Settings Configuration with pydantic-settings
```python
# utils/settings.py - Robust configuration management
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, validator
from dotenv import load_dotenv
from typing import Optional, List
import os

class LangChainSettings(BaseSettings):
    """Comprehensive LangChain application settings."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    default_model: str = Field(default="gpt-3.5-turbo", description="Default LLM model")
    temperature: float = Field(default=0.7, description="Default temperature")
    max_tokens: int = Field(default=1000, description="Default max tokens")
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="langchain-project", description="LangSmith project name")
    
    # Vector Store Configuration
    vector_store: str = Field(default="chroma", description="Default vector store")
    chroma_persist_directory: str = Field(default="./chroma_db", description="Chroma persistence directory")
    
    # Memory Configuration
    redis_url: Optional[str] = Field(default=None, description="Redis URL for persistence")
    memory_type: str = Field(default="buffer_window", description="Default memory type")
    
    # Application Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Application environment")
    max_retries: int = Field(default=3, description="Maximum API retries")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    @validator('openai_api_key', 'anthropic_api_key', pre=True)
    def validate_api_keys(cls, v):
        """Ensure at least one API key is provided."""
        return v
    
    def __init__(self, **kwargs):
        """Load environment variables before initialization."""
        load_dotenv()
        super().__init__(**kwargs)
        
        # Validate that at least one LLM API key is available
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("At least one LLM API key must be provided (OpenAI or Anthropic)")

def get_settings() -> LangChainSettings:
    """Get application settings with caching."""
    if not hasattr(get_settings, '_settings'):
        get_settings._settings = LangChainSettings()
    return get_settings._settings

# Example usage in chain creation
def create_llm():
    """Create LLM instance with proper configuration."""
    settings = get_settings()
    
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries
        )
    elif settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model="claude-3-sonnet-20240229",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries
        )
    else:
        raise ValueError("No valid LLM API key found")
```

## 🧪 Testing & Reliability Standards

### Comprehensive Testing Patterns
```python
# tests/conftest.py - Test configuration and fixtures
import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.callbacks import BaseCallbackHandler
from utils.settings import get_settings

@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    return FakeListLLM(responses=["Test response", "Another response"])

@pytest.fixture
def settings():
    """Test settings configuration."""
    return get_settings()

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = MagicMock()
    mock.similarity_search.return_value = [
        MagicMock(page_content="Test document content", metadata={"source": "test.txt"})
    ]
    return mock

# tests/test_chains.py - Chain testing patterns
import pytest
from chains.base_chain import ChainInput, ChainOutput, create_basic_chain
from langchain_core.language_models.fake import FakeListLLM

class TestBasicChain:
    """Test cases for basic chain functionality."""
    
    def test_chain_creation(self, mock_llm):
        """Test that chain can be created successfully."""
        chain = create_basic_chain(mock_llm)
        assert chain is not None
    
    @pytest.mark.asyncio
    async def test_chain_execution(self, mock_llm):
        """Test chain execution with valid input."""
        chain = create_basic_chain(mock_llm)
        
        input_data = ChainInput(query="What is the capital of France?")
        result = await chain.ainvoke({"query": input_data.query})
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_input_validation(self):
        """Test input validation with invalid data."""
        with pytest.raises(ValueError):
            ChainInput(query="")  # Empty query should fail
    
    @pytest.mark.slow
    def test_chain_with_real_llm(self):
        """Integration test with real LLM (marked as slow)."""
        from utils.settings import create_llm
        
        llm = create_llm()
        chain = create_basic_chain(llm)
        
        result = chain.invoke({"query": "What is 2+2?"})
        assert "4" in result
```

### Error Handling & Resilience Patterns
```python
# utils/validation.py - Comprehensive validation and error handling
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
import time
import logging
from pydantic import ValidationError
import asyncio

logger = logging.getLogger(__name__)

def safe_chain_execution(
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """Decorator for safe chain execution with retry logic."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    execution_time = time.time() - start_time
                    logger.info(f"Chain executed successfully in {execution_time:.2f}s")
                    
                    return ChainOutput(
                        response=result,
                        success=True,
                        execution_time=execution_time,
                        metadata={"attempts": attempt + 1}
                    )
                    
                except asyncio.TimeoutError as e:
                    last_exception = e
                    logger.warning(f"Chain execution timeout on attempt {attempt + 1}")
                    
                except ValidationError as e:
                    logger.error(f"Validation error: {e}")
                    return ChainOutput(
                        response="",
                        success=False,
                        error=f"Validation error: {str(e)}",
                        execution_time=time.time() - start_time
                    )
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Chain execution failed on attempt {attempt + 1}: {e}")
                    
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff_factor ** attempt)
            
            # All retries failed
            execution_time = time.time() - start_time
            return ChainOutput(
                response="",
                success=False,
                error=f"Chain execution failed after {max_retries} attempts: {str(last_exception)}",
                execution_time=execution_time,
                metadata={"attempts": max_retries}
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return async wrapper if function is async, sync wrapper otherwise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def validate_environment() -> Dict[str, bool]:
    """Validate that required environment variables are set."""
    settings = get_settings()
    
    validation_results = {
        "llm_available": bool(settings.openai_api_key or settings.anthropic_api_key),
        "tracing_configured": bool(settings.langchain_tracing_v2 and settings.langchain_api_key),
        "vector_store_available": True,  # Chroma works without external deps
        "memory_persistence": bool(settings.redis_url),
    }
    
    logger.info(f"Environment validation: {validation_results}")
    return validation_results
```

## 🔍 Research & Development Standards

### LangChain Best Practices Research
- **Study LangChain documentation extensively** - python.langchain.com has comprehensive guides
- **Follow LangChain Expression Language (LCEL)** patterns for chain composition
- **Use proper async/await patterns** throughout the application
- **Implement comprehensive observability** with LangSmith tracing
- **Study community patterns** on GitHub and LangChain Hub

### Common LangChain Gotchas & Solutions
- **Memory management complexity** - Use appropriate memory types and implement persistence
- **Token limit exceeded errors** - Implement proper context management and summarization
- **Async/sync pattern mixing** - Be consistent with async patterns throughout chains
- **Vector store performance** - Index optimization and proper embedding strategies
- **Agent infinite loops** - Implement proper stopping criteria and iteration limits
- **Streaming implementation issues** - Use proper callback handlers and async generators

## 🚫 Anti-Patterns to Always Avoid

- ❌ Don't skip comprehensive testing - Always test chains with mock and real LLMs
- ❌ Don't hardcode prompts - Use PromptTemplate and proper template management
- ❌ Don't ignore memory management - Implement appropriate memory strategies
- ❌ Don't skip error handling - Use comprehensive retry and fallback mechanisms
- ❌ Don't forget observability - Always implement proper logging and tracing
- ❌ Don't mix async/sync patterns inconsistently - Choose one pattern and stick to it
- ❌ Don't skip input/output validation - Use Pydantic models for all interfaces
- ❌ Don't ignore token limits - Implement proper context management strategies

## 🎯 Implementation Standards

### Quality Checklist for LangChain Applications
- [ ] **Environment Configuration** - Proper settings management with pydantic-settings
- [ ] **Input/Output Validation** - Pydantic models for all chain interfaces
- [ ] **Error Handling** - Comprehensive retry logic and graceful degradation
- [ ] **Testing Coverage** - Unit tests, integration tests, and performance tests
- [ ] **Observability** - LangSmith tracing and structured logging
- [ ] **Memory Management** - Appropriate memory types with persistence
- [ ] **Security** - API key management and input sanitization
- [ ] **Performance** - Async patterns and optimization strategies
- [ ] **Documentation** - Comprehensive docstrings and usage examples

### Code Review Standards
- **Architecture Review** - Proper separation of concerns and modular design
- **Performance Review** - Efficient prompt management and context handling
- **Security Review** - Safe handling of API keys and user inputs
- **Testing Review** - Comprehensive test coverage and edge case handling
- **Documentation Review** - Clear documentation and usage examples

This enhanced CLAUDE.md provides comprehensive guidance for building production-ready LangChain applications with proper error handling, testing, observability, and security practices.