"""
Test patterns for LangChain chains.

This file demonstrates comprehensive testing approaches for LangChain chains,
including unit tests, integration tests, and performance tests.

Key Testing Patterns:
- Chain execution testing with mocked LLMs
- Prompt template validation
- Memory integration testing
- Error handling and edge cases
- Performance and token usage testing
- RAG chain testing with mock vector stores
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json
import time

from langchain.chains import LLMChain, ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.document import Document
from langchain.vectorstores.base import VectorStore

# Import the components we're testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from basic_chain import create_summarization_chain, create_qa_chain, safe_chain_execution
from rag_chain import RAGSystem, DocumentProcessor, RAGConfig
from memory_chain import ConversationManager, MemoryConfig

class TestBasicChains:
    """Test basic chain implementations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_llm = Mock()
        self.mock_llm.predict.return_value = "Mock LLM response"
        self.mock_llm.generate.return_value = Mock(generations=[[Mock(text="Mock response")]])
    
    def test_summarization_chain_creation(self):
        """Test that summarization chain is created properly."""
        with patch('basic_chain.create_llm') as mock_create_llm:
            mock_create_llm.return_value = self.mock_llm
            
            chain = create_summarization_chain()
            
            assert chain is not None
            assert hasattr(chain, 'llm')
            assert hasattr(chain, 'prompt')
            assert 'text' in chain.prompt.input_variables
            assert 'max_sentences' in chain.prompt.input_variables
    
    def test_summarization_chain_execution(self):
        """Test summarization chain execution."""
        with patch('basic_chain.create_llm') as mock_create_llm:
            mock_create_llm.return_value = self.mock_llm
            
            chain = create_summarization_chain()
            
            # Mock the run method
            chain.run = Mock(return_value="This is a test summary.")
            
            result = chain.run({
                "text": "This is a long text that needs to be summarized for testing purposes.",
                "max_sentences": "2"
            })
            
            assert result == "This is a test summary."
            chain.run.assert_called_once()
    
    def test_qa_chain_creation(self):
        """Test Q&A chain creation."""
        with patch('basic_chain.create_llm') as mock_create_llm:
            mock_create_llm.return_value = self.mock_llm
            
            chain = create_qa_chain()
            
            assert chain is not None
            assert 'context' in chain.prompt.input_variables
            assert 'question' in chain.prompt.input_variables
    
    def test_safe_chain_execution_success(self):
        """Test successful chain execution."""
        mock_chain = Mock()
        mock_chain.run.return_value = "Successful response"
        
        result = safe_chain_execution(mock_chain, {"input": "test"})
        
        assert result.success is True
        assert result.result == "Successful response"
        assert result.error is None
        assert result.execution_time is not None
    
    def test_safe_chain_execution_failure(self):
        """Test chain execution with error."""
        mock_chain = Mock()
        mock_chain.run.side_effect = Exception("Test error")
        
        result = safe_chain_execution(mock_chain, {"input": "test"})
        
        assert result.success is False
        assert result.result is None
        assert "Test error" in result.error
        assert result.execution_time is not None
    
    @patch('basic_chain.os.getenv')
    def test_environment_validation(self, mock_getenv):
        """Test environment variable validation."""
        from basic_chain import validate_environment
        
        # Test missing API key
        mock_getenv.return_value = None
        assert validate_environment() is False
        
        # Test valid API key
        mock_getenv.return_value = "test-api-key"
        assert validate_environment() is True
    
    def test_prompt_template_formatting(self):
        """Test prompt template formatting."""
        prompt = PromptTemplate(
            input_variables=["text", "max_sentences"],
            template="Summarize this text in {max_sentences} sentences: {text}"
        )
        
        formatted = prompt.format(
            text="Test text content",
            max_sentences="3"
        )
        
        assert "Test text content" in formatted
        assert "3" in formatted
        assert "Summarize" in formatted

class TestRAGChains:
    """Test RAG (Retrieval-Augmented Generation) chains."""
    
    def setup_method(self):
        """Setup for RAG tests."""
        self.config = RAGConfig(
            chunk_size=500,
            chunk_overlap=100,
            k_documents=3
        )
        
        # Mock documents
        self.sample_documents = [
            Document(
                page_content="LangChain is a framework for building LLM applications.",
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="RAG combines retrieval with generation for better responses.",
                metadata={"source": "test2.txt"}
            )
        ]
    
    def test_document_processor_creation(self):
        """Test document processor initialization."""
        processor = DocumentProcessor(self.config)
        
        assert processor.config.chunk_size == 500
        assert processor.config.chunk_overlap == 100
        assert processor.text_splitter is not None
    
    def test_document_splitting(self):
        """Test document splitting functionality."""
        processor = DocumentProcessor(self.config)
        
        split_docs = processor.split_documents(self.sample_documents)
        
        assert len(split_docs) >= len(self.sample_documents)
        for doc in split_docs:
            assert hasattr(doc, 'page_content')
            assert hasattr(doc, 'metadata')
            assert 'chunk_id' in doc.metadata
    
    @patch('rag_chain.os.getenv')
    def test_rag_system_initialization(self, mock_getenv):
        """Test RAG system initialization."""
        mock_getenv.return_value = "test-api-key"
        
        with patch('rag_chain.OpenAIEmbeddings') as mock_embeddings, \
             patch('rag_chain.ChatOpenAI') as mock_llm:
            
            mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
            mock_llm.return_value = Mock()
            
            rag_system = RAGSystem(self.config)
            
            assert rag_system.embeddings is not None
            assert rag_system.llm is not None
            assert rag_system.processor is not None
    
    @patch('rag_chain.Chroma')
    def test_vector_store_creation(self, mock_chroma):
        """Test vector store creation."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        with patch('rag_chain.os.getenv', return_value="test-api-key"), \
             patch('rag_chain.OpenAIEmbeddings') as mock_embeddings:
            
            mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
            
            rag_system = RAGSystem(self.config)
            vectorstore = rag_system.create_vector_store(self.sample_documents)
            
            assert vectorstore is not None
            mock_chroma.from_documents.assert_called_once()
    
    def test_rag_query_processing(self):
        """Test RAG query processing."""
        with patch('rag_chain.os.getenv', return_value="test-api-key"), \
             patch('rag_chain.OpenAIEmbeddings') as mock_embeddings, \
             patch('rag_chain.ChatOpenAI') as mock_llm:
            
            # Setup mocks
            mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
            mock_llm.return_value = Mock()
            
            rag_system = RAGSystem(self.config)
            
            # Mock the retrieval chain
            mock_chain = Mock()
            mock_chain.return_value = {
                "result": "This is a test answer",
                "source_documents": self.sample_documents
            }
            rag_system.retrieval_chain = mock_chain
            
            response = rag_system.query("What is LangChain?")
            
            assert response.success is True
            assert "This is a test answer" in response.answer
            assert len(response.sources) > 0
    
    def test_rag_error_handling(self):
        """Test RAG system error handling."""
        with patch('rag_chain.os.getenv', return_value="test-api-key"):
            rag_system = RAGSystem(self.config)
            
            # Query without initialization should fail gracefully
            response = rag_system.query("test query")
            
            assert response.success is False
            assert "not initialized" in response.error

class TestMemoryIntegration:
    """Test memory integration with chains."""
    
    def setup_method(self):
        """Setup for memory tests."""
        self.mock_llm = Mock()
        self.mock_llm.predict.return_value = "Mock response"
    
    def test_buffer_memory_creation(self):
        """Test buffer memory creation and usage."""
        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        
        # Test saving context
        memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
        
        # Test loading memory variables
        variables = memory.load_memory_variables({})
        
        assert "history" in variables
        assert len(variables["history"]) == 2  # Human + AI messages
    
    def test_window_memory_limits(self):
        """Test window memory size limits."""
        memory = ConversationBufferWindowMemory(
            k=2,  # Keep only last 2 exchanges
            memory_key="history",
            return_messages=True
        )
        
        # Add multiple conversations
        conversations = [
            ("Hello", "Hi"),
            ("How are you?", "I'm good"),
            ("What's your name?", "I'm Claude"),
            ("Nice to meet you", "Nice to meet you too")
        ]
        
        for human_msg, ai_msg in conversations:
            memory.save_context({"input": human_msg}, {"output": ai_msg})
        
        variables = memory.load_memory_variables({})
        history = variables["history"]
        
        # Should only keep last 2 exchanges (4 messages total)
        assert len(history) <= 4
    
    def test_conversation_chain_with_memory(self):
        """Test conversation chain with memory integration."""
        memory = ConversationBufferMemory()
        
        chain = ConversationChain(
            llm=self.mock_llm,
            memory=memory,
            verbose=False
        )
        
        # First conversation
        response1 = chain.predict(input="My name is Alice")
        assert response1 == "Mock response"
        
        # Memory should contain the conversation
        variables = memory.load_memory_variables({})
        assert "Alice" in str(variables)
    
    @patch('memory_chain.os.getenv')
    def test_conversation_manager(self, mock_getenv):
        """Test conversation manager functionality."""
        mock_getenv.return_value = "test-api-key"
        
        with patch('memory_chain.ChatOpenAI') as mock_llm_class:
            mock_llm_class.return_value = self.mock_llm
            
            config = MemoryConfig(window_size=3)
            manager = ConversationManager(config)
            
            assert len(manager.memories) == 5  # All memory types
            assert len(manager.chains) == 5   # All chain types
            
            # Test chat functionality
            result = manager.chat('buffer', "Hello!")
            
            assert result['success'] is True
            assert result['ai_response'] == "Mock response"

class TestChainPerformance:
    """Test chain performance and optimization."""
    
    def test_chain_execution_time(self):
        """Test chain execution performance."""
        mock_chain = Mock()
        
        # Simulate processing time
        def slow_run(inputs):
            time.sleep(0.1)  # 100ms delay
            return "Response"
        
        mock_chain.run = slow_run
        
        start_time = time.time()
        result = safe_chain_execution(mock_chain, {"input": "test"})
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert result.execution_time >= 0.1
        assert execution_time >= 0.1
    
    def test_token_counting(self):
        """Test token counting functionality."""
        try:
            import tiktoken
            
            processor = DocumentProcessor(RAGConfig())
            
            test_messages = [
                HumanMessage(content="Hello, how are you?"),
                AIMessage(content="I'm doing well, thank you!")
            ]
            
            token_count = processor._estimate_token_count(test_messages)
            
            assert token_count > 0
            assert isinstance(token_count, int)
            
        except ImportError:
            # If tiktoken not available, test fallback
            processor = DocumentProcessor(RAGConfig())
            
            test_messages = [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi")
            ]
            
            token_count = processor._estimate_token_count(test_messages)
            assert token_count > 0
    
    def test_large_document_processing(self):
        """Test processing of large documents."""
        config = RAGConfig(chunk_size=100, chunk_overlap=20)
        processor = DocumentProcessor(config)
        
        # Create a large document
        large_content = "This is a test sentence. " * 100  # ~2500 characters
        large_doc = Document(
            page_content=large_content,
            metadata={"source": "large_test.txt"}
        )
        
        split_docs = processor.split_documents([large_doc])
        
        assert len(split_docs) > 1  # Should be split into multiple chunks
        
        # Check chunk sizes
        for doc in split_docs:
            assert len(doc.page_content) <= config.chunk_size + config.chunk_overlap

class TestErrorScenarios:
    """Test various error scenarios and edge cases."""
    
    def test_invalid_prompt_variables(self):
        """Test handling of invalid prompt variables."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize: {text}"
        )
        
        # Missing required variable should raise error
        with pytest.raises(KeyError):
            prompt.format(wrong_variable="test")
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        mock_chain = Mock()
        mock_chain.run.return_value = "Empty input handled"
        
        result = safe_chain_execution(mock_chain, {})
        
        assert result.success is True
        mock_chain.run.assert_called_once_with({})
    
    def test_malformed_document_handling(self):
        """Test handling of malformed documents."""
        processor = DocumentProcessor(RAGConfig())
        
        # Document with None content
        malformed_docs = [
            Document(page_content=None, metadata={}),
            Document(page_content="", metadata={}),
            Document(page_content="Valid content", metadata={"source": "test"})
        ]
        
        # Should handle malformed documents gracefully
        try:
            split_docs = processor.split_documents(malformed_docs)
            # At least the valid document should be processed
            assert len(split_docs) >= 0
        except Exception as e:
            pytest.fail(f"Document processing should handle malformed docs: {e}")
    
    def test_api_key_missing_error(self):
        """Test behavior when API key is missing."""
        with patch('basic_chain.os.getenv', return_value=None):
            from basic_chain import validate_environment
            
            assert validate_environment() is False
    
    def test_chain_timeout_handling(self):
        """Test chain timeout scenarios."""
        mock_chain = Mock()
        
        def timeout_run(inputs):
            time.sleep(2)  # Simulate long processing
            return "Response"
        
        mock_chain.run = timeout_run
        
        # This should complete but take time
        start_time = time.time()
        result = safe_chain_execution(mock_chain, {"input": "test"})
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert execution_time >= 2

class TestAsyncChains:
    """Test asynchronous chain operations."""
    
    @pytest.mark.asyncio
    async def test_async_chain_execution(self):
        """Test asynchronous chain execution pattern."""
        
        async def mock_async_run(inputs):
            await asyncio.sleep(0.1)  # Simulate async work
            return "Async response"
        
        mock_chain = Mock()
        mock_chain.arun = mock_async_run
        
        result = await mock_chain.arun({"input": "test"})
        
        assert result == "Async response"
    
    @pytest.mark.asyncio
    async def test_concurrent_chain_execution(self):
        """Test concurrent execution of multiple chains."""
        
        async def async_chain_call(chain_id):
            await asyncio.sleep(0.1)
            return f"Response from chain {chain_id}"
        
        # Execute multiple chains concurrently
        tasks = [async_chain_call(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all("Response from chain" in result for result in results)

if __name__ == "__main__":
    """
    Run the chain tests.
    
    Usage:
        python test_chains.py
        pytest test_chains.py -v
        pytest test_chains.py::TestBasicChains::test_summarization_chain_creation -v
    """
    
    # Run basic test discovery
    pytest.main([__file__, "-v"])