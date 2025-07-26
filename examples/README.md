# LangChain Examples Directory

This directory contains reference implementations and patterns for building LangChain applications. These examples serve as the foundation for context engineering - they show the AI assistant exactly how you prefer to structure and implement LangChain components.

## 📋 How to Use These Examples

When creating your `INITIAL.md` feature request, reference specific example files to ensure the AI follows your established patterns. For example:

```markdown
## EXAMPLES:
Reference `examples/rag_chain.py` for document processing and vector store setup patterns.
Follow the error handling approach from `examples/basic_chain.py`.
Use the memory integration pattern from `examples/memory_chain.py`.
```

## 📁 File Structure

### Core Chain Examples
- **`basic_chain.py`** - Fundamental LangChain patterns every project should follow
- **`rag_chain.py`** - Complete RAG implementation with best practices
- **`agent_chain.py`** - Agent architecture with custom tools
- **`memory_chain.py`** - Conversation memory management patterns

### Specialized Tools
- **`tools/`** - Custom tool implementations and integration patterns

### Testing Patterns
- **`tests/`** - Testing approaches for LangChain components

## 🎯 Key Patterns Demonstrated

### 1. Error Handling
All examples show proper error handling with:
- Try/catch blocks around LLM calls
- Logging for debugging and monitoring
- Graceful degradation strategies
- User-friendly error messages

### 2. Configuration Management
- Environment variable usage for API keys
- Model parameter configuration
- Flexible provider switching
- Production vs development settings

### 3. Input Validation
- Pydantic models for structured inputs
- Input sanitization and validation
- Type hints throughout
- Clear function signatures

### 4. Performance Optimization
- Token counting and optimization
- Streaming responses where appropriate
- Caching strategies for expensive operations
- Memory management for long conversations

### 5. Code Organization
- Clear separation of concerns
- Modular, reusable components
- Consistent naming conventions
- Comprehensive docstrings

## 🔧 Setup Requirements

Before running these examples, ensure you have:

```bash
# Install dependencies
pip install langchain langchain-openai langchain-community python-dotenv

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
LANGCHAIN_TRACING_V2=true  # Optional, for LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key  # Optional
```

## 🧪 Running the Examples

Each example file can be run independently:

```bash
# Basic chain example
python examples/basic_chain.py

# RAG system example
python examples/rag_chain.py

# Agent example
python examples/agent_chain.py

# Memory management example
python examples/memory_chain.py
```

## 🏗️ Architecture Patterns

### Chain Composition Pattern
```python
# From basic_chain.py
def create_chain(llm, prompt_template):
    """Standard pattern for creating reusable chains."""
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True  # For development
    )
    return chain
```

### Error Handling Pattern
```python
# From all examples
def safe_chain_execution(chain, input_data):
    """Standard error handling pattern."""
    try:
        result = chain.run(input_data)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        return {"success": False, "error": str(e)}
```

### Memory Integration Pattern
```python
# From memory_chain.py
def create_conversation_chain(llm, memory_type="buffer"):
    """Standard memory integration pattern."""
    memory = get_memory_instance(memory_type)
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return chain
```

## 📚 Learning Path

If you're new to LangChain, study the examples in this order:

1. **`basic_chain.py`** - Learn fundamental patterns
2. **`memory_chain.py`** - Understand conversation management
3. **`rag_chain.py`** - Master document retrieval systems
4. **`agent_chain.py`** - Explore agent architectures

## 🔄 Updating Examples

When adding new examples:

1. **Follow existing patterns** from current examples
2. **Include comprehensive error handling**
3. **Add proper documentation and type hints**
4. **Create corresponding test files**
5. **Update this README** with new patterns

## 🧪 Testing

Each example includes basic testing patterns. Run tests with:

```bash
pytest examples/tests/ -v
```

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Community](https://github.com/langchain-ai/langchain)
- [LangSmith for Debugging](https://smith.langchain.com/)

---

**Remember**: These examples are the foundation of your context engineering approach. The more comprehensive and well-structured they are, the better your generated implementations will be!