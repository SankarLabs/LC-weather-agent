"""
Streaming Response Implementation Example

This file demonstrates how to implement streaming responses in LangChain
applications for real-time user feedback and improved user experience.

Key Patterns Demonstrated:
- Streaming chat responses
- Token-by-token output
- Progress callbacks
- Async streaming patterns
- Buffer management
- Error handling in streams
"""

import asyncio
import logging
from typing import AsyncIterator, Iterator, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingConfig(BaseModel):
    """Configuration for streaming responses."""
    model_name: str = Field("gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(1000, description="Maximum tokens")
    chunk_size: int = Field(1, description="Size of each streaming chunk")
    buffer_size: int = Field(100, description="Buffer size for streaming")


class CustomStreamingCallback(BaseCallbackHandler):
    """Custom callback handler for streaming with additional features."""
    
    def __init__(self, on_token_callback=None, on_finish_callback=None):
        self.tokens = []
        self.on_token_callback = on_token_callback
        self.on_finish_callback = on_finish_callback
        self.start_time = None
        self.token_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts, **kwargs) -> None:
        """Called when LLM starts running."""
        import time
        self.start_time = time.time()
        self.tokens = []
        self.token_count = 0
        logger.info("Streaming started")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.tokens.append(token)
        self.token_count += 1
        
        # Call custom callback if provided
        if self.on_token_callback:
            self.on_token_callback(token, self.token_count)
        
        # Print token (similar to StreamingStdOutCallbackHandler)
        print(token, end="", flush=True)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes."""
        import time
        
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            logger.info(f"\nStreaming completed: {self.token_count} tokens in {elapsed_time:.2f}s")
        
        if self.on_finish_callback:
            self.on_finish_callback(self.tokens, self.token_count)
    
    def get_full_response(self) -> str:
        """Get the complete response as a string."""
        return "".join(self.tokens)


class StreamingChain:
    """Chain that supports streaming responses."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.llm = self._create_llm()
        self.chain = self._create_chain()
    
    def _create_llm(self):
        """Create LLM with streaming capabilities."""
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=True  # Enable streaming
        )
    
    def _create_chain(self):
        """Create the streaming chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Provide detailed and informative responses."),
            ("human", "{query}")
        ])
        
        return (
            prompt
            | self.llm
            | StrOutputParser()
        )
    
    def stream(self, query: str, callback: Optional[BaseCallbackHandler] = None) -> Iterator[str]:
        """
        Stream the response token by token.
        
        Args:
            query: User query
            callback: Optional callback handler
            
        Yields:
            str: Individual tokens or chunks
        """
        # Use provided callback or default streaming callback
        if callback is None:
            callback = StreamingStdOutCallbackHandler()
        
        # Configure LLM with callback
        streaming_llm = self.llm.bind(callbacks=[callback])
        
        # Create chain with streaming LLM
        streaming_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Provide detailed and informative responses."),
                ("human", "{query}")
            ])
            | streaming_llm
            | StrOutputParser()
        )
        
        # Stream the response
        for chunk in streaming_chain.stream({"query": query}):
            yield chunk
    
    async def astream(self, query: str, callback: Optional[BaseCallbackHandler] = None) -> AsyncIterator[str]:
        """
        Async version of stream method.
        
        Args:
            query: User query
            callback: Optional callback handler
            
        Yields:
            str: Individual tokens or chunks
        """
        # Use provided callback or default streaming callback
        if callback is None:
            callback = StreamingStdOutCallbackHandler()
        
        # Configure LLM with callback
        streaming_llm = self.llm.bind(callbacks=[callback])
        
        # Create chain with streaming LLM
        streaming_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Provide detailed and informative responses."),
                ("human", "{query}")
            ])
            | streaming_llm
            | StrOutputParser()
        )
        
        # Stream the response asynchronously
        async for chunk in streaming_chain.astream({"query": query}):
            yield chunk
    
    def stream_with_callback(self, query: str, on_token=None, on_finish=None) -> str:
        """
        Stream with custom callbacks for token and completion events.
        
        Args:
            query: User query
            on_token: Callback for each token (token, count)
            on_finish: Callback for completion (all_tokens, count)
            
        Returns:
            str: Complete response
        """
        callback = CustomStreamingCallback(
            on_token_callback=on_token,
            on_finish_callback=on_finish
        )
        
        # Stream with callback
        list(self.stream(query, callback))
        
        return callback.get_full_response()


class BufferedStreamingChain:
    """Chain with buffered streaming for better performance."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.llm = self._create_llm()
        self.buffer = []
        self.buffer_size = config.buffer_size
    
    def _create_llm(self):
        """Create LLM with streaming capabilities."""
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=True
        )
    
    def stream_buffered(self, query: str, chunk_size: int = None) -> Iterator[str]:
        """
        Stream response in chunks rather than token by token.
        
        Args:
            query: User query
            chunk_size: Size of chunks to yield
            
        Yields:
            str: Chunks of the response
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        # Create callback to collect tokens in buffer
        class BufferCallback(BaseCallbackHandler):
            def __init__(self, buffer, chunk_size):
                self.buffer = buffer
                self.chunk_size = chunk_size
                self.current_chunk = ""
            
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.current_chunk += token
                
                # Yield chunk when it reaches desired size
                if len(self.current_chunk) >= self.chunk_size:
                    self.buffer.append(self.current_chunk)
                    self.current_chunk = ""
            
            def on_llm_end(self, response, **kwargs) -> None:
                # Yield remaining chunk
                if self.current_chunk:
                    self.buffer.append(self.current_chunk)
        
        # Clear buffer
        self.buffer = []
        
        # Create callback
        callback = BufferCallback(self.buffer, chunk_size)
        
        # Create streaming chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{query}")
        ])
        
        streaming_llm = self.llm.bind(callbacks=[callback])
        chain = prompt | streaming_llm | StrOutputParser()
        
        # Start streaming (this will populate the buffer via callback)
        response = chain.invoke({"query": query})
        
        # Yield chunks from buffer
        for chunk in self.buffer:
            yield chunk


def demonstrate_basic_streaming():
    """Demonstrate basic streaming functionality."""
    print("\n" + "="*60)
    print("BASIC STREAMING DEMONSTRATION")
    print("="*60)
    
    config = StreamingConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=200
    )
    
    chain = StreamingChain(config)
    
    query = "Explain the concept of quantum computing in simple terms."
    
    print(f"Query: {query}")
    print(f"Streaming response:")
    print("-" * 40)
    
    # Stream the response
    for chunk in chain.stream(query):
        pass  # Printing is handled by the callback
    
    print("\n" + "-" * 40)
    print("Streaming completed")


async def demonstrate_async_streaming():
    """Demonstrate async streaming functionality."""
    print("\n" + "="*60)
    print("ASYNC STREAMING DEMONSTRATION")
    print("="*60)
    
    config = StreamingConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=200
    )
    
    chain = StreamingChain(config)
    
    query = "What are the benefits of renewable energy?"
    
    print(f"Query: {query}")
    print(f"Async streaming response:")
    print("-" * 40)
    
    # Stream the response asynchronously
    async for chunk in chain.astream(query):
        pass  # Printing is handled by the callback
    
    print("\n" + "-" * 40)
    print("Async streaming completed")


def demonstrate_custom_callbacks():
    """Demonstrate streaming with custom callbacks."""
    print("\n" + "="*60)
    print("CUSTOM CALLBACK DEMONSTRATION")
    print("="*60)
    
    config = StreamingConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=150
    )
    
    chain = StreamingChain(config)
    
    query = "List 5 interesting facts about space exploration."
    
    print(f"Query: {query}")
    print(f"Response with token counting:")
    print("-" * 40)
    
    token_count = 0
    
    def on_token(token, count):
        nonlocal token_count
        token_count = count
        if count % 10 == 0:  # Show progress every 10 tokens
            print(f"\n[{count} tokens]", end="")
    
    def on_finish(tokens, count):
        print(f"\n\nFinished! Generated {count} tokens.")
        print(f"Average token length: {sum(len(t) for t in tokens) / len(tokens):.2f} characters")
    
    response = chain.stream_with_callback(
        query,
        on_token=on_token,
        on_finish=on_finish
    )
    
    print(f"Complete response length: {len(response)} characters")


def demonstrate_buffered_streaming():
    """Demonstrate buffered streaming for better performance."""
    print("\n" + "="*60)
    print("BUFFERED STREAMING DEMONSTRATION")
    print("="*60)
    
    config = StreamingConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=200,
        chunk_size=20  # Yield chunks of 20 characters
    )
    
    chain = BufferedStreamingChain(config)
    
    query = "Describe the process of photosynthesis step by step."
    
    print(f"Query: {query}")
    print(f"Buffered streaming response (chunks of {config.chunk_size} chars):")
    print("-" * 40)
    
    chunk_number = 0
    for chunk in chain.stream_buffered(query, chunk_size=25):
        chunk_number += 1
        print(f"[Chunk {chunk_number}] {chunk}", end="", flush=True)
        
        # Small delay to demonstrate chunking
        import time
        time.sleep(0.1)
    
    print(f"\n\nBuffered streaming completed ({chunk_number} chunks)")


def demonstrate_error_handling():
    """Demonstrate error handling in streaming."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    config = StreamingConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )
    
    chain = StreamingChain(config)
    
    # Test with potentially problematic input
    problematic_queries = [
        "",  # Empty query
        "A" * 1000,  # Very long query
        "What is 2+2?",  # Normal query for comparison
    ]
    
    for i, query in enumerate(problematic_queries, 1):
        print(f"\nTest {i}: Query length = {len(query)}")
        print(f"Query preview: {query[:50]}{'...' if len(query) > 50 else ''}")
        print("-" * 30)
        
        try:
            response_chunks = []
            for chunk in chain.stream(query):
                response_chunks.append(chunk)
            
            response = "".join(response_chunks)
            print(f"Success! Response length: {len(response)}")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    """
    Run streaming demonstrations.
    
    This demonstrates:
    1. Basic token-by-token streaming
    2. Async streaming patterns
    3. Custom callback integration
    4. Buffered streaming for performance
    5. Error handling in streaming contexts
    """
    
    try:
        # Basic streaming
        demonstrate_basic_streaming()
        
        # Async streaming
        asyncio.run(demonstrate_async_streaming())
        
        # Custom callbacks
        demonstrate_custom_callbacks()
        
        # Buffered streaming
        demonstrate_buffered_streaming()
        
        # Error handling
        demonstrate_error_handling()
        
    except KeyboardInterrupt:
        logger.info("Streaming demonstration interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error in demonstration: {e}")
        raise
    
    finally:
        logger.info("Streaming demonstration completed")