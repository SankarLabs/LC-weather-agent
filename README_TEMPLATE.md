# LangChain Context Engineering Template

A comprehensive template for building LangChain applications using Context Engineering principles - the discipline of providing comprehensive context to AI assistants so they can build robust, production-ready LangChain solutions.

## Quick Start

```bash
# 1. Clone this template
git clone https://github.com/your-username/langchain-context-engineering.git
cd langchain-context-engineering

# 2. Set up your environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure your environment
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, etc.)

# 4. Set up your project rules (optional - template provided)
# Edit LANGCHAIN_RULES.md to add your project-specific guidelines

# 5. Add examples (highly recommended)
# Place relevant LangChain code examples in the examples/ folder

# 6. Create your initial feature request
# Edit INITIAL.md with your LangChain feature requirements

# 7. Generate a comprehensive LangChain Implementation Plan (LIP)
# In Claude Code, run:
/generate-lip INITIAL.md

# 8. Execute the LIP to implement your feature
# In Claude Code, run:
/execute-lip LIPs/your-feature-name.md
```

## Template Structure

```
langchain-context-engineering/
├── .claude/
│   ├── commands/
│   │   ├── generate-lip.md      # Generates LangChain Implementation Plans
│   │   └── execute-lip.md       # Executes LIPs to implement features
│   └── settings.local.json      # Claude Code permissions
├── LIPs/                        # LangChain Implementation Plans
│   ├── templates/
│   │   └── lip_base.md         # Base template for LIPs
│   └── EXAMPLE_rag_chain.md    # Example of a complete LIP
├── examples/                    # Your LangChain code examples (critical!)
│   ├── README.md
│   ├── basic_chain.py          # Simple LLM chain example
│   ├── rag_chain.py            # RAG implementation example
│   ├── agent_chain.py          # Agent-based chain example
│   ├── memory_chain.py         # Conversation memory example
│   ├── tools/
│   │   ├── README.md           # Tools documentation and patterns
│   │   ├── custom_tool.py      # Custom tool implementations
│   │   └── web_search_tool.py  # Web search tool examples
│   └── tests/                  # Test patterns for LangChain
│       ├── README.md           # Testing documentation
│       ├── test_chains.py      # Chain testing patterns
│       ├── test_agents.py      # Agent testing patterns
│       └── conftest.py         # Pytest fixtures and configuration
├── LANGCHAIN_RULES.md          # Global rules for LangChain development
├── INITIAL.md                  # Template for feature requests
├── INITIAL_EXAMPLE.md          # Example LangChain feature request
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
└── README.md                  # This file
```

## What is Context Engineering for LangChain?

Context Engineering for LangChain represents a systematic approach to building LangChain applications:

**Traditional Approach:**
- Write individual LangChain chains without clear patterns
- Limited documentation and examples
- Inconsistent error handling and validation
- Like building with random LEGO pieces

**Context Engineering Approach:**
- Comprehensive system with patterns, examples, and validation
- Consistent architecture across all LangChain components
- Self-correcting implementation with testing
- Like having a detailed LEGO instruction manual with all pieces organized

### Why Context Engineering for LangChain?

- **Reduces Implementation Failures**: Most LangChain failures come from missing context about proper patterns
- **Ensures Best Practices**: Follow LangChain's recommended patterns and conventions
- **Enables Complex Applications**: Build multi-step LangChain applications with proper context
- **Self-Validating**: Built-in testing ensures your LangChain chains work correctly

## Step-by-Step Guide

### 1. Configure LangChain Rules

The `LANGCHAIN_RULES.md` file contains project-wide rules for LangChain development:

- **LangChain Patterns**: Chain composition, agent architecture, memory handling
- **Model Management**: LLM provider switching, token optimization
- **Tool Integration**: Custom tool creation, tool selection patterns
- **Error Handling**: Retry logic, fallback strategies, validation
- **Testing Requirements**: Chain testing, agent testing, integration tests

### 2. Add LangChain Examples

Place relevant LangChain code examples in the `examples/` folder:

```
examples/
├── basic_chain.py              # Simple LLM chain
├── rag_chain.py               # RAG with vector stores
├── agent_chain.py             # ReAct agent with tools
├── memory_chain.py            # Conversation memory
├── streaming_chain.py         # Streaming responses
├── multi_modal_chain.py       # Multi-modal processing
└── tools/
    ├── custom_tool.py         # Custom tool implementation
    ├── web_search_tool.py     # Web search integration
    └── database_tool.py       # Database query tool
```

### 3. Write Your Feature Request

Edit `INITIAL.md` to describe your LangChain feature:

```markdown
## FEATURE:
[Describe the LangChain application you want to build]

## LANGCHAIN COMPONENTS:
[List specific LangChain components: chains, agents, tools, memory, etc.]

## EXAMPLES:
[Reference example files and explain patterns to follow]

## DOCUMENTATION:
[Include LangChain docs, API references, integration guides]

## TESTING REQUIREMENTS:
[Specify how to test chains, agents, and integrations]

## OTHER CONSIDERATIONS:
[LangChain-specific gotchas, performance requirements, model considerations]
```

### 4. Generate LangChain Implementation Plan (LIP)

Run the generator in Claude Code:

```bash
/generate-lip INITIAL.md
```

This will:
- Analyze your codebase for LangChain patterns
- Research LangChain documentation and best practices
- Create a comprehensive implementation plan in `LIPs/your-feature-name.md`
- Include validation steps specific to LangChain

### 5. Execute the Implementation

Run the executor in Claude Code:

```bash
/execute-lip LIPs/your-feature-name.md
```

The system will:
- Read all context from the LIP
- Implement LangChain components step by step
- Run validation tests for chains and agents
- Ensure all LangChain best practices are followed

## LangChain-Specific Best Practices

### Chain Composition Patterns

```python
# Example from examples/basic_chain.py
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def create_summarization_chain():
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}\n\nSummary:"
    )
    
    llm = OpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
```

### Agent Architecture Patterns

```python
# Example from examples/agent_chain.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

def create_research_agent():
    tools = [
        web_search_tool,
        calculator_tool,
        database_query_tool
    ]
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor
```

### Memory Management Patterns

```python
# Example from examples/memory_chain.py
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

def create_conversation_chain():
    memory = ConversationBufferWindowMemory(k=5)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation
```

### Testing Patterns

```python
# Example from tests/test_chains.py
import pytest
from langchain.schema import BaseMessage

def test_summarization_chain():
    chain = create_summarization_chain()
    
    test_text = "This is a long document that needs to be summarized..."
    result = chain.run(test_text)
    
    assert len(result) < len(test_text)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
```

## Environment Configuration

### Required Environment Variables

```bash
# .env.example
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=your_project_name

# Vector Store Configuration
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env

# Optional: Other providers
COHERE_API_KEY=your_cohere_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

### Dependencies

```txt
# requirements.txt
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-anthropic>=0.1.1
langchain-community>=0.0.10
langsmith>=0.0.83
openai>=1.0.0
anthropic>=0.8.0
pinecone-client>=3.0.0
chromadb>=0.4.0
faiss-cpu>=1.7.4
tiktoken>=0.5.0
python-dotenv>=1.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
requests>=2.28.0
```

## 🔧 Setup Requirements

### Installation
```bash
# Clone the template
git clone https://github.com/your-username/langchain-context-engineering.git
cd langchain-context-engineering

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
LANGCHAIN_TRACING_V2=true  # Optional, for LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key  # Optional

# Vector Store Configuration (if using)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env

# Testing Configuration
PYTEST_FAST_MODE=false
PYTEST_ALLOW_API_TESTS=true
PYTEST_DEBUG=false
```

### Verify Installation
```bash
# Test basic functionality
python examples/basic_chain.py

# Run the test suite
pytest examples/tests/ -v

# Check specific components
python -c "import langchain; print('LangChain installed successfully')"
```

## Advanced Features

### Custom Tool Creation

The template includes patterns for creating custom tools that integrate seamlessly with LangChain agents:

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CustomSearchInput(BaseModel):
    query: str = Field(description="Search query to execute")

class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "Search for information using a custom search engine"
    args_schema: Type[BaseModel] = CustomSearchInput

    def _run(self, query: str) -> str:
        # Your custom search implementation
        return f"Search results for: {query}"
```

### Multi-Modal Chain Patterns

Examples for handling different types of input (text, images, audio):

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_multimodal_chain():
    # Pattern for processing multiple input types
    prompt = PromptTemplate(
        input_variables=["text_input", "image_description"],
        template="Analyze this text: {text_input}\nAnd this image: {image_description}\nProvide insights:"
    )
    
    return LLMChain(llm=llm, prompt=prompt)
```

### Streaming Response Patterns

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def create_streaming_chain():
    llm = OpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.7
    )
    return LLMChain(llm=llm, prompt=prompt)
```

## Validation and Testing

The template includes comprehensive testing patterns:

- **Unit Tests**: Test individual chains and components
- **Integration Tests**: Test chain compositions and agent workflows  
- **Performance Tests**: Validate response times and token usage
- **Error Handling Tests**: Ensure graceful failure handling

## Contributing

1. Follow the established patterns in the examples/ folder
2. Add comprehensive tests for new features
3. Update documentation for new LangChain components
4. Ensure all validation steps pass

## License

MIT License - feel free to use this template for your LangChain projects!