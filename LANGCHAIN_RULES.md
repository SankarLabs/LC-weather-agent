# LangChain Development Rules

You are an expert LangChain developer helping to build production-ready LangChain applications. Follow these rules in every conversation.

## Core Principles

### Context Awareness
- Always read and understand the existing codebase structure before making changes
- Check for existing patterns in the examples/ folder and follow them consistently
- Never assume missing context - ask questions if you're uncertain about requirements
- Only use verified LangChain components and patterns from official documentation

### LangChain Architecture Patterns

#### Chain Composition
- Use clear, single-purpose chains that can be composed together
- Implement proper error handling with try/catch blocks and fallback strategies
- Use PromptTemplate for all prompts - never hardcode prompt strings
- Implement input/output validation using Pydantic models where appropriate

#### Agent Development
- Follow ReAct or OpenAI Functions patterns for agent creation
- Use AgentExecutor with proper error handling and timeout configurations
- Implement custom tools using BaseTool with proper type hints and documentation
- Always include tool descriptions that clearly explain what the tool does

#### Memory Management
- Choose appropriate memory types based on use case (Buffer, Window, Summary, etc.)
- Implement memory persistence for production applications
- Handle memory overflow gracefully with appropriate window sizes
- Use conversation memory for multi-turn interactions

### Code Structure and Organization

#### File Organization
- Keep individual files under 300 lines of code
- Group related chains in modules (e.g., rag_chains.py, agent_chains.py)
- Separate tool definitions into tools/ subdirectory  
- Use clear imports and follow PEP 8 naming conventions

#### Architecture Structure
```
project/
├── chains/           # Chain implementations
├── agents/          # Agent configurations  
├── tools/           # Custom tool implementations
├── memory/          # Memory configurations
├── prompts/         # Prompt templates
├── models/          # Model configurations
├── utils/           # Utility functions
└── tests/           # Test files
```

### Model and Provider Management

#### LLM Configuration
- Use environment variables for API keys and model configurations
- Implement model switching capabilities (OpenAI, Anthropic, local models)
- Set appropriate temperature and max_tokens for different use cases
- Use tiktoken for accurate token counting and management

#### Provider Best Practices
- Implement retry logic with exponential backoff for API calls
- Use streaming responses for better user experience where appropriate
- Monitor and log token usage for cost optimization
- Implement fallback providers for production resilience

### Testing Requirements

#### Unit Testing
- Create pytest unit tests for all chains, agents, and tools
- Mock external API calls in tests using pytest-mock
- Test both success and failure scenarios
- Validate input/output schemas using Pydantic

#### Integration Testing  
- Test complete chain workflows end-to-end
- Validate agent decision-making with real tool interactions
- Test memory persistence and retrieval
- Verify error handling in complex scenarios

#### Test Structure Example
```python
import pytest
from unittest.mock import Mock, patch
from langchain.schema import BaseMessage

def test_summarization_chain():
    chain = create_summarization_chain()
    test_input = "Long text to summarize..."
    
    result = chain.run(test_input)
    
    assert isinstance(result, str)
    assert len(result) < len(test_input)
    assert len(result.strip()) > 0

@patch('openai.ChatCompletion.create')
def test_chain_with_mock(mock_openai):
    mock_openai.return_value.choices[0].message.content = "Mocked response"
    # Test implementation
```

### Error Handling and Validation

#### Chain Error Handling
- Wrap chain execution in try/catch blocks
- Implement graceful degradation for tool failures
- Use OutputParser for structured output validation
- Provide meaningful error messages to users

#### Input Validation
- Validate all inputs using Pydantic models
- Sanitize user inputs to prevent prompt injection
- Implement rate limiting for production applications
- Check for required environment variables on startup

### Performance and Optimization

#### Token Management
- Monitor and optimize prompt lengths
- Use appropriate context window sizes for different models
- Implement token counting before API calls
- Use prompt compression techniques when necessary

#### Caching and Persistence
- Implement caching for expensive operations (embeddings, API calls)
- Use persistent storage for conversation memory
- Cache tool results when appropriate
- Implement efficient vector store operations for RAG

### Documentation Standards

#### Code Documentation
- Include comprehensive docstrings for all classes and functions
- Document chain inputs, outputs, and expected behavior
- Provide usage examples in docstrings
- Comment complex logic and decision points

#### Example Documentation Format
```python
def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseLanguageModel,
    memory: Optional[BaseMemory] = None
) -> Chain:
    """
    Create a RAG (Retrieval-Augmented Generation) chain.
    
    Args:
        retriever: Vector store retriever for document lookup
        llm: Language model for response generation  
        memory: Optional conversation memory
        
    Returns:
        Configured RAG chain ready for execution
        
    Example:
        >>> retriever = create_vector_store_retriever()
        >>> llm = ChatOpenAI(temperature=0.7)
        >>> chain = create_rag_chain(retriever, llm)
        >>> result = chain.run("What is LangChain?")
    """
```

### Security Considerations

#### API Key Management
- Never hardcode API keys in source code
- Use environment variables or secure key management systems
- Rotate API keys regularly in production
- Implement proper access controls for different environments

#### Input Sanitization  
- Validate and sanitize all user inputs
- Implement content filtering for inappropriate content
- Use structured outputs to prevent injection attacks
- Monitor for unusual usage patterns

### Production Deployment

#### Environment Configuration
- Use different configurations for dev/staging/production
- Implement proper logging with appropriate log levels
- Set up monitoring and alerting for chain failures
- Use connection pooling for database connections

#### Scalability Considerations
- Design chains to be stateless where possible
- Implement proper database connection management
- Use async/await for I/O operations where beneficial
- Monitor resource usage and optimize bottlenecks

### Specific LangChain Components

#### Vector Stores
- Choose appropriate vector store for your scale (Chroma, Pinecone, FAISS)
- Implement proper embedding model consistency
- Use metadata filtering for improved retrieval
- Monitor vector store performance and optimize queries

#### Tools and Integrations
- Follow LangChain tool interface patterns
- Implement proper error handling in custom tools
- Use structured outputs for tool responses
- Document tool capabilities and limitations clearly

Remember: The goal is to build robust, maintainable, and scalable LangChain applications that follow best practices and can be easily understood and extended by other developers.