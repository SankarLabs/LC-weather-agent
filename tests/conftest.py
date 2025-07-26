"""
Pytest configuration and shared fixtures for LangChain testing.

This file provides common fixtures, utilities, and configuration
for testing LangChain applications following best practices.

Key Components:
- Mock LLM fixtures for consistent testing
- Temporary file management for file operations
- Environment variable mocking
- Performance testing utilities
- Common test data and scenarios
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Generator
import json
import time
from datetime import datetime

# LangChain imports for fixtures
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStore

# Configure pytest
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API keys"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in item.name or item.parent.name.startswith("Test"):
            item.add_marker(pytest.mark.unit)
        
        # Mark API-requiring tests
        if "api" in item.name or "openai" in item.name.lower():
            item.add_marker(pytest.mark.requires_api)

# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.predict.return_value = "Mock LLM response"
    llm.generate.return_value = Mock(
        generations=[[Mock(text="Mock generation")]]
    )
    llm.model_name = "mock-model"
    llm.temperature = 0.7
    return llm

@pytest.fixture
def mock_chat_llm():
    """Create a mock chat LLM for testing."""
    llm = Mock()
    llm.predict.return_value = "Mock chat response"
    llm.predict_messages.return_value = AIMessage(content="Mock AI message")
    llm.generate.return_value = Mock(
        generations=[[Mock(message=AIMessage(content="Mock generation"))]]
    )
    return llm

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 1536  # Standard OpenAI embedding size
    embeddings.embed_documents.return_value = [[0.1] * 1536] * 3  # Mock multiple docs
    return embeddings

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    vector_store = Mock(spec=VectorStore)
    
    # Mock documents for search results
    mock_docs = [
        Document(
            page_content="This is a test document about LangChain.",
            metadata={"source": "test1.txt", "chunk_id": 0}
        ),
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "test2.txt", "chunk_id": 1}
        )
    ]
    
    vector_store.similarity_search.return_value = mock_docs[:2]
    vector_store.similarity_search_with_score.return_value = [
        (mock_docs[0], 0.9),
        (mock_docs[1], 0.8)
    ]
    vector_store.as_retriever.return_value = Mock()
    vector_store.add_documents.return_value = None
    
    return vector_store

# ============================================================================
# ENVIRONMENT FIXTURES
# ============================================================================

@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key environment variable."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
        yield "test-openai-key"

@pytest.fixture
def mock_anthropic_api_key():
    """Mock Anthropic API key environment variable."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}):
        yield "test-anthropic-key"

@pytest.fixture
def mock_all_api_keys():
    """Mock all common API keys."""
    mock_env = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGCHAIN_API_KEY": "test-langsmith-key"
    }
    with patch.dict(os.environ, mock_env):
        yield mock_env

@pytest.fixture
def clean_environment():
    """Provide a clean environment without API keys."""
    # Store original values
    original_env = {}
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGCHAIN_API_KEY"]
    
    for key in api_keys:
        if key in os.environ:
            original_env[key] = os.environ[key]
            del os.environ[key]
    
    yield
    
    # Restore original values
    for key, value in original_env.items():
        os.environ[key] = value

# ============================================================================
# FILE SYSTEM FIXTURES
# ============================================================================

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_documents(temp_directory):
    """Create sample documents for testing."""
    documents = []
    
    # Create sample text files
    sample_files = {
        "doc1.txt": "This is the first test document. It contains information about LangChain.",
        "doc2.txt": "This is the second test document. It discusses AI and machine learning.",
        "doc3.txt": "This is the third test document. It covers natural language processing topics."
    }
    
    for filename, content in sample_files.items():
        file_path = temp_directory / filename
        file_path.write_text(content)
        
        documents.append(Document(
            page_content=content,
            metadata={"source": str(file_path), "filename": filename}
        ))
    
    return documents

@pytest.fixture
def sample_json_file(temp_directory):
    """Create a sample JSON file for testing."""
    sample_data = [
        {"name": "Alice", "age": 30, "department": "Engineering"},
        {"name": "Bob", "age": 25, "department": "Marketing"},
        {"name": "Charlie", "age": 35, "department": "Engineering"}
    ]
    
    json_file = temp_directory / "sample_data.json"
    json_file.write_text(json.dumps(sample_data, indent=2))
    
    return json_file, sample_data

@pytest.fixture
def sample_csv_file(temp_directory):
    """Create a sample CSV file for testing."""
    csv_content = """name,age,department
Alice,30,Engineering
Bob,25,Marketing  
Charlie,35,Engineering
Diana,28,Sales"""
    
    csv_file = temp_directory / "sample_data.csv"
    csv_file.write_text(csv_content)
    
    return csv_file, csv_content

# ============================================================================
# LANGCHAIN COMPONENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_memory():
    """Create a sample conversation memory with history."""
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    
    # Add sample conversation
    memory.save_context(
        {"input": "Hello, my name is Alice"},
        {"output": "Hi Alice! Nice to meet you."}
    )
    memory.save_context(
        {"input": "I work as a software engineer"},
        {"output": "That's great! What kind of software do you work on?"}
    )
    
    return memory

@pytest.fixture
def sample_chat_history():
    """Create sample chat history messages."""
    return [
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you! How can I help you today?"),
        HumanMessage(content="I need help with LangChain"),
        AIMessage(content="I'd be happy to help you with LangChain! What specifically would you like to know?")
    ]

@pytest.fixture
def sample_documents_for_rag():
    """Create sample documents optimized for RAG testing."""
    return [
        Document(
            page_content="""
            LangChain is a framework for developing applications powered by language models.
            It enables applications that are context-aware and can reason about their environment.
            The main value props of LangChain are: Components, Chains, and Agents.
            """,
            metadata={"source": "langchain_intro", "type": "documentation", "chunk_id": 0}
        ),
        Document(
            page_content="""
            RAG (Retrieval-Augmented Generation) is a technique that combines retrieval 
            of relevant documents with language model generation. It helps reduce 
            hallucinations and provides source attribution for responses.
            """,
            metadata={"source": "rag_explanation", "type": "documentation", "chunk_id": 1}
        ),
        Document(
            page_content="""
            Vector databases store high-dimensional vectors representing semantic meanings.
            Popular options include Chroma, Pinecone, Weaviate, and FAISS. They enable
            efficient similarity search for RAG applications.
            """,
            metadata={"source": "vector_databases", "type": "documentation", "chunk_id": 2}
        )
    ]

# ============================================================================
# TESTING UTILITIES
# ============================================================================

@pytest.fixture
def performance_timer():
    """Utility fixture for timing test operations."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return PerformanceTimer()

@pytest.fixture
def assertion_helpers():
    """Provide helper functions for common assertions."""
    class AssertionHelpers:
        @staticmethod
        def assert_valid_chain_response(response):
            """Assert that a chain response is valid."""
            assert response is not None
            assert isinstance(response, (str, dict))
            if isinstance(response, dict):
                assert "success" in response or "result" in response or "output" in response
        
        @staticmethod
        def assert_valid_agent_response(response):
            """Assert that an agent response is valid."""
            assert response is not None
            assert isinstance(response, dict)
            assert "success" in response
            if response["success"]:
                assert "answer" in response or "output" in response
            else:
                assert "error" in response
        
        @staticmethod
        def assert_valid_documents(documents):
            """Assert that documents are valid."""
            assert isinstance(documents, list)
            for doc in documents:
                assert hasattr(doc, "page_content")
                assert hasattr(doc, "metadata")
                assert isinstance(doc.page_content, str)
                assert isinstance(doc.metadata, dict)
        
        @staticmethod
        def assert_execution_time(actual_time, expected_max, tolerance=0.1):
            """Assert execution time is within expected bounds."""
            assert actual_time <= expected_max + tolerance, \
                f"Execution took {actual_time}s, expected max {expected_max}s"
    
    return AssertionHelpers()

# ============================================================================
# PARAMETERIZED TEST DATA
# ============================================================================

@pytest.fixture(params=[
    "What is LangChain?",
    "How does RAG work?", 
    "Explain vector databases",
    "What are the benefits of using AI agents?"
])
def sample_queries(request):
    """Parameterized fixture providing various test queries."""
    return request.param

@pytest.fixture(params=[
    {"memory_type": "buffer", "config": {}},
    {"memory_type": "window", "config": {"k": 3}},
    {"memory_type": "summary", "config": {"max_token_limit": 500}}
])
def memory_configurations(request):
    """Parameterized fixture providing different memory configurations."""
    return request.param

@pytest.fixture(params=[
    "2 + 3",
    "10 * 5", 
    "100 / 4",
    "2 ** 8"
])
def calculation_queries(request):
    """Parameterized fixture providing calculation queries."""
    return request.param

# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================

@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_timeout": 30,  # seconds
        "max_retries": 3,
        "rate_limit_delay": 1,  # seconds between API calls
        "expected_response_time": 5  # seconds
    }

@pytest.fixture
def mock_external_apis():
    """Mock external API responses for integration testing."""
    
    # Mock OpenAI API responses
    openai_responses = {
        "chat/completions": {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock OpenAI response for testing purposes."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        },
        "embeddings": {
            "data": [
                {
                    "embedding": [0.1] * 1536,
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
    }
    
    # Mock search API responses
    search_responses = {
        "web_search": {
            "results": [
                {
                    "title": "Test Search Result 1",
                    "url": "https://example.com/result1",
                    "snippet": "This is a test search result for integration testing."
                },
                {
                    "title": "Test Search Result 2", 
                    "url": "https://example.com/result2",
                    "snippet": "Another test search result with relevant information."
                }
            ],
            "total_results": 2
        }
    }
    
    return {
        "openai": openai_responses,
        "search": search_responses
    }

# ============================================================================
# ERROR TESTING FIXTURES
# ============================================================================

@pytest.fixture
def error_scenarios():
    """Provide common error scenarios for testing."""
    return {
        "api_key_missing": {
            "error": ValueError("API key not provided"),
            "expected_message": "API key not provided"
        },
        "rate_limit_exceeded": {
            "error": Exception("Rate limit exceeded"),
            "expected_message": "Rate limit exceeded"
        },
        "invalid_input": {
            "error": ValueError("Invalid input format"),
            "expected_message": "Invalid input format"
        },
        "network_timeout": {
            "error": TimeoutError("Request timed out"),
            "expected_message": "Request timed out"
        },
        "parsing_error": {
            "error": SyntaxError("Failed to parse response"),
            "expected_message": "Failed to parse response"
        }
    }

@pytest.fixture
def edge_case_inputs():
    """Provide edge case inputs for robust testing."""
    return {
        "empty_string": "",
        "whitespace_only": "   \n\t   ",
        "very_long_string": "x" * 10000,
        "special_characters": "!@#$%^&*()[]{}|;':\",./<>?",
        "unicode_characters": "🚀 Hello 世界 🌍",
        "sql_injection_attempt": "'; DROP TABLE users; --",
        "script_injection_attempt": "<script>alert('xss')</script>",
        "null_bytes": "test\x00null",
        "mixed_encoding": "café naïve résumé"
    }

# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def performance_benchmarks():
    """Define performance benchmarks for different operations."""
    return {
        "chain_execution": {
            "max_time": 2.0,  # seconds
            "expected_avg": 0.5
        },
        "document_processing": {
            "max_time": 5.0,
            "expected_avg": 1.0
        },
        "vector_search": {
            "max_time": 1.0,
            "expected_avg": 0.2
        },
        "agent_reasoning": {
            "max_time": 10.0,
            "expected_avg": 3.0
        },
        "memory_operations": {
            "max_time": 0.5,
            "expected_avg": 0.1
        }
    }

@pytest.fixture
def load_test_data():
    """Generate data for load testing."""
    class LoadTestData:
        @staticmethod
        def generate_queries(count: int = 100):
            """Generate test queries for load testing."""
            base_queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain neural networks",
                "What are transformers in AI?",
                "How do you build a chatbot?"
            ]
            
            return [f"{query} (test {i})" for i, query in 
                   enumerate(base_queries * (count // len(base_queries) + 1))[:count]]
        
        @staticmethod
        def generate_documents(count: int = 50):
            """Generate test documents for processing."""
            base_content = [
                "This is a test document about artificial intelligence and machine learning.",
                "Natural language processing is a field of AI that focuses on human language.",
                "Deep learning uses neural networks with multiple layers.",
                "Transformers have revolutionized natural language understanding.",
                "LangChain is a framework for building LLM applications."
            ]
            
            documents = []
            for i in range(count):
                content = base_content[i % len(base_content)]
                documents.append(Document(
                    page_content=f"{content} Document {i}.",
                    metadata={"doc_id": i, "source": f"test_doc_{i}.txt"}
                ))
            
            return documents
    
    return LoadTestData()

# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    # This runs before each test
    yield
    
    # This runs after each test
    test_files = [
        "test_file.txt",
        "test_output.json",
        "test_memory.pkl",
        "test_vectorstore.db"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass  # File might be in use or already deleted

@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Session-wide cleanup for test artifacts."""
    yield
    
    # Cleanup session-wide test artifacts
    cleanup_dirs = [
        "./test_memory_data",
        "./test_chroma_db", 
        "./test_outputs"
    ]
    
    for directory in cleanup_dirs:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except OSError:
                pass

# ============================================================================
# LOGGING AND DEBUGGING FIXTURES
# ============================================================================

@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add handler to relevant loggers
    loggers = [
        logging.getLogger("langchain"),
        logging.getLogger("agent_chain"),
        logging.getLogger("rag_chain"),
        logging.getLogger("memory_chain")
    ]
    
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    # Remove handlers
    for logger in loggers:
        logger.removeHandler(handler)

@pytest.fixture
def debug_mode():
    """Enable debug mode for detailed test output."""
    class DebugMode:
        def __init__(self):
            self.enabled = False
            self.logs = []
        
        def enable(self):
            self.enabled = True
        
        def log(self, message: str):
            if self.enabled:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                log_entry = f"[{timestamp}] {message}"
                self.logs.append(log_entry)
                print(log_entry)
        
        def get_logs(self):
            return self.logs
        
        def clear_logs(self):
            self.logs = []
    
    return DebugMode()

# ============================================================================
# CUSTOM PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def test_metadata():
    """Provide metadata about the current test."""
    class TestMetadata:
        def __init__(self):
            self.start_time = datetime.now()
            self.test_name = None
            self.test_file = None
        
        def set_test_info(self, name: str, file_path: str):
            self.test_name = name
            self.test_file = file_path
        
        @property
        def duration(self):
            return datetime.now() - self.start_time
        
        def to_dict(self):
            return {
                "test_name": self.test_name,
                "test_file": self.test_file,
                "start_time": self.start_time.isoformat(),
                "duration_seconds": self.duration.total_seconds()
            }
    
    return TestMetadata()

# ============================================================================
# CONDITIONAL FIXTURES
# ============================================================================

@pytest.fixture
def skip_if_no_api_key():
    """Skip test if API key is not available."""
    def _skip_if_no_api_key(api_key_name: str = "OPENAI_API_KEY"):
        if not os.getenv(api_key_name):
            pytest.skip(f"Skipping test: {api_key_name} not available")
    
    return _skip_if_no_api_key

@pytest.fixture
def skip_if_slow():
    """Skip test if running in fast mode."""
    def _skip_if_slow():
        if os.getenv("PYTEST_FAST_MODE", "false").lower() == "true":
            pytest.skip("Skipping slow test in fast mode")
    
    return _skip_if_slow

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pytest_runtest_setup(item):
    """Setup function run before each test."""
    # Set test metadata
    test_name = item.name
    test_file = item.fspath.basename
    
    # Skip tests based on markers and environment
    if item.get_closest_marker("requires_api"):
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("PYTEST_ALLOW_API_TESTS"):
            pytest.skip("API tests require OPENAI_API_KEY or PYTEST_ALLOW_API_TESTS=true")
    
    if item.get_closest_marker("slow"):
        if os.getenv("PYTEST_FAST_MODE", "false").lower() == "true":
            pytest.skip("Skipping slow test in fast mode")

def pytest_runtest_teardown(item, nextitem):
    """Teardown function run after each test."""
    # Cleanup any remaining mocks
    try:
        from unittest.mock import patch
        patch.stopall()
    except:
        pass

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

class TestConfig:
    """Test configuration helper."""
    
    @staticmethod
    def get_test_env():
        """Get test environment configuration."""
        return {
            "is_ci": os.getenv("CI", "false").lower() == "true",
            "fast_mode": os.getenv("PYTEST_FAST_MODE", "false").lower() == "true",
            "allow_api_tests": os.getenv("PYTEST_ALLOW_API_TESTS", "false").lower() == "true",
            "debug_mode": os.getenv("PYTEST_DEBUG", "false").lower() == "true"
        }
    
    @staticmethod
    def should_run_integration_tests():
        """Determine if integration tests should run."""
        env = TestConfig.get_test_env()
        return not env["fast_mode"] and (env["allow_api_tests"] or not env["is_ci"])
    
    @staticmethod
    def get_timeout_multiplier():
        """Get timeout multiplier based on environment."""
        env = TestConfig.get_test_env()
        if env["is_ci"]:
            return 2.0  # CI environments are often slower
        return 1.0

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()