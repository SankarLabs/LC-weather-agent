# LangChain Testing Patterns

This directory contains comprehensive testing patterns and examples for LangChain applications. These tests demonstrate best practices for testing chains, agents, tools, and memory components.

## 📁 Test Files

### `test_chains.py`
Comprehensive testing patterns for LangChain chains:
- **Basic Chain Testing**: Prompt templates, LLM integration, execution patterns
- **RAG Chain Testing**: Document processing, vector store operations, retrieval testing
- **Memory Integration**: Testing chains with different memory types
- **Performance Testing**: Execution time, token counting, optimization
- **Error Handling**: Edge cases, API failures, malformed inputs

### `test_agents.py`
Agent testing patterns covering:
- **Tool Functionality**: Individual tool testing and validation
- **Agent Creation**: Configuration, initialization, tool integration
- **Agent Execution**: Reasoning chains, multi-step workflows, decision making
- **Error Recovery**: Tool failures, invalid actions, graceful degradation
- **Performance**: Execution time tracking, iteration limits, timeouts
- **Advanced Patterns**: Custom prompts, state persistence, async operations

### `conftest.py`
Pytest configuration and shared fixtures:
- **Mock Fixtures**: LLMs, embeddings, vector stores, external APIs
- **Environment Setup**: API key mocking, clean environments
- **File System**: Temporary directories, sample documents, test data
- **Performance Utilities**: Timing, benchmarks, load testing
- **Debugging Tools**: Log capture, debug mode, test metadata

## 🎯 Testing Strategies

### Unit Testing
```python
def test_chain_creation():
    """Test that chain is created with proper configuration."""
    chain = create_test_chain()
    assert chain is not None
    assert hasattr(chain, 'llm')
    assert hasattr(chain, 'prompt')
```

### Integration Testing  
```python
@pytest.mark.integration
def test_end_to_end_rag_workflow(mock_all_api_keys):
    """Test complete RAG workflow from documents to response."""
    rag_system = RAGSystem()
    rag_system.initialize_from_directory("test_docs/")
    response = rag_system.query("What is LangChain?")
    assert response.success
    assert len(response.sources) > 0
```

### Performance Testing
```python
@pytest.mark.slow
def test_chain_performance(performance_timer, performance_benchmarks):
    """Test chain execution meets performance requirements."""
    chain = create_test_chain()
    
    performance_timer.start()
    result = chain.run("test input")
    elapsed = performance_timer.stop()
    
    max_time = performance_benchmarks["chain_execution"]["max_time"]
    assert elapsed <= max_time
```

### Error Testing
```python
def test_chain_error_handling(error_scenarios):
    """Test chain handles various error scenarios gracefully."""
    chain = create_test_chain()
    
    for scenario_name, scenario in error_scenarios.items():
        with patch.object(chain.llm, 'predict', side_effect=scenario["error"]):
            result = safe_chain_execution(chain, {"input": "test"})
            assert result.success is False
            assert scenario["expected_message"] in result.error
```

## 🚀 Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest examples/tests/ -v

# Run specific test file
pytest examples/tests/test_chains.py -v

# Run specific test
pytest examples/tests/test_chains.py::TestBasicChains::test_summarization_chain_creation -v
```

### Test Categories
```bash
# Run only unit tests
pytest examples/tests/ -m "unit" -v

# Run only integration tests (requires API keys)
pytest examples/tests/ -m "integration" -v

# Skip slow tests
pytest examples/tests/ -m "not slow" -v

# Run tests that require API keys
pytest examples/tests/ -m "requires_api" -v
```

### Environment-Specific Testing
```bash
# Fast mode (skip slow tests)
PYTEST_FAST_MODE=true pytest examples/tests/ -v

# Allow API tests in CI
PYTEST_ALLOW_API_TESTS=true pytest examples/tests/ -v

# Debug mode with detailed output
PYTEST_DEBUG=true pytest examples/tests/ -v -s
```

### Coverage Testing
```bash
# Run with coverage
pytest examples/tests/ --cov=examples --cov-report=html --cov-report=term

# Coverage with minimum threshold
pytest examples/tests/ --cov=examples --cov-fail-under=80
```

## 🔧 Test Configuration

### Environment Variables
```bash
# API Keys (for integration tests)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Test behavior
export PYTEST_FAST_MODE="true"          # Skip slow tests
export PYTEST_ALLOW_API_TESTS="true"    # Allow API tests
export PYTEST_DEBUG="true"              # Enable debug output

# CI environment
export CI="true"                         # Indicates CI environment
```

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = examples/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --durations=10
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    requires_api: marks tests that require API keys
```

## 🧪 Mock Usage Patterns

### Mocking LLMs
```python
def test_with_mock_llm(mock_llm):
    """Test using mock LLM fixture."""
    chain = LLMChain(llm=mock_llm, prompt=test_prompt)
    result = chain.run("test input")
    
    mock_llm.predict.assert_called_once()
    assert result == "Mock LLM response"
```

### Mocking External APIs
```python
@patch('requests.get')
def test_web_search_tool(mock_get):
    """Test web search with mocked HTTP requests."""
    mock_get.return_value.json.return_value = {"results": []}
    
    tool = WebSearchTool()
    result = tool._run("test query")
    
    mock_get.assert_called_once()
    assert "results" in result.lower()
```

### Mocking File System
```python
def test_file_operations(temp_directory):
    """Test file operations with temporary directory."""
    test_file = temp_directory / "test.txt"
    test_content = "Test content"
    
    # Write test
    tool = FileOperationTool()
    result = tool._run("write", str(test_file), test_content)
    assert "Successfully wrote" in result
    
    # Read test
    result = tool._run("read", str(test_file))
    assert test_content in result
```

## 📊 Test Data Patterns

### Sample Documents
```python
def test_document_processing(sample_documents):
    """Test with predefined sample documents."""
    processor = DocumentProcessor()
    split_docs = processor.split_documents(sample_documents)
    
    assert len(split_docs) >= len(sample_documents)
    for doc in split_docs:
        assert hasattr(doc, 'page_content')
        assert hasattr(doc, 'metadata')
```

### Parameterized Testing
```python
@pytest.mark.parametrize("query,expected_type", [
    ("2 + 3", "calculation"),
    ("What is AI?", "question"),
    ("Hello there", "greeting")
])
def test_query_classification(query, expected_type):
    """Test query classification with multiple inputs."""
    classifier = QueryClassifier()
    result = classifier.classify(query)
    assert result == expected_type
```

### Edge Cases
```python
def test_edge_cases(edge_case_inputs):
    """Test handling of edge case inputs."""
    tool = TextAnalysisTool()
    
    for case_name, case_input in edge_case_inputs.items():
        result = tool._run(case_input, "sentiment")
        
        # Should handle gracefully without crashing
        assert isinstance(result, str)
        assert len(result) > 0
```

## 🔍 Debugging Tests

### Capturing Logs
```python
def test_with_log_capture(capture_logs):
    """Test with log capture for debugging."""
    chain = create_test_chain()
    result = chain.run("test input")
    
    log_contents = capture_logs.getvalue()
    assert "Chain executed" in log_contents
```

### Debug Mode
```python
def test_with_debug(debug_mode):
    """Test with debug output."""
    debug_mode.enable()
    
    debug_mode.log("Starting test")
    result = perform_test_operation()
    debug_mode.log(f"Result: {result}")
    
    logs = debug_mode.get_logs()
    assert len(logs) >= 2
```

### Test Metadata
```python
def test_with_metadata(test_metadata):
    """Test with metadata tracking."""
    test_metadata.set_test_info("test_example", __file__)
    
    # Perform test
    result = some_operation()
    
    # Check metadata
    metadata = test_metadata.to_dict()
    assert metadata["test_name"] == "test_example"
    assert metadata["duration_seconds"] > 0
```

## 🏗️ Custom Test Fixtures

### Creating Custom Fixtures
```python
@pytest.fixture
def custom_test_setup():
    """Custom fixture for specific test needs."""
    # Setup
    test_data = create_test_data()
    mock_services = setup_mock_services()
    
    yield {"data": test_data, "services": mock_services}
    
    # Teardown
    cleanup_test_data(test_data)
    teardown_mock_services(mock_services)
```

### Fixture Scopes
```python
@pytest.fixture(scope="session")
def expensive_setup():
    """Session-scoped fixture for expensive setup."""
    expensive_resource = create_expensive_resource()
    yield expensive_resource
    cleanup_expensive_resource(expensive_resource)

@pytest.fixture(scope="function")
def fresh_instance():
    """Function-scoped fixture for clean instances."""
    return create_fresh_instance()
```

## 📈 Performance Testing

### Benchmarking
```python
def test_performance_benchmark(performance_benchmarks):
    """Test against performance benchmarks."""
    operation = create_test_operation()
    
    start_time = time.time()
    result = operation.execute()
    execution_time = time.time() - start_time
    
    benchmark = performance_benchmarks["operation"]["max_time"]
    assert execution_time <= benchmark
```

### Load Testing
```python
@pytest.mark.slow
def test_load_performance(load_test_data):
    """Test performance under load."""
    queries = load_test_data.generate_queries(100)
    
    start_time = time.time()
    for query in queries:
        process_query(query)
    total_time = time.time() - start_time
    
    avg_time = total_time / len(queries)
    assert avg_time <= 0.1  # 100ms average
```

## 🔐 Security Testing

### Input Sanitization
```python
def test_input_sanitization(edge_case_inputs):
    """Test input sanitization and security."""
    tool = create_secure_tool()
    
    malicious_inputs = [
        edge_case_inputs["sql_injection_attempt"],
        edge_case_inputs["script_injection_attempt"]
    ]
    
    for malicious_input in malicious_inputs:
        result = tool._run(malicious_input)
        
        # Should sanitize or reject malicious input
        assert "error" in result.lower() or "invalid" in result.lower()
```

## 📝 Best Practices

### Test Organization
- **Group related tests** in classes
- **Use descriptive test names** that explain what is being tested
- **Follow AAA pattern**: Arrange, Act, Assert
- **One assertion per logical concept**
- **Test both success and failure paths**

### Mock Usage
- **Mock external dependencies** (APIs, databases, file systems)
- **Don't mock the code you're testing**
- **Use fixtures** for consistent mock setup
- **Verify mock interactions** when relevant
- **Reset mocks** between tests

### Test Data
- **Use fixtures** for reusable test data
- **Parametrize tests** for multiple inputs
- **Generate data** rather than hardcoding when possible
- **Clean up** test artifacts
- **Isolate tests** from each other

### Performance
- **Mark slow tests** appropriately
- **Set reasonable timeouts** 
- **Test performance requirements** explicitly
- **Use profiling** for optimization
- **Monitor resource usage**

Remember: Good tests are your safety net - they should give you confidence that your LangChain applications work correctly under all conditions! 🎯