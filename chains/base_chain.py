"""
Base chain implementation with common patterns and utilities.

This module provides base classes and utilities that all production chains
should inherit from, ensuring consistency across the application.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


class BaseChainConfig(BaseModel):
    """Base configuration for all chains."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
    
    # LLM Configuration
    model_provider: str = Field(default="openai", description="LLM provider (openai, anthropic)")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens")
    
    # Execution Configuration
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class ChainInput(BaseModel):
    """Standardized input for all chains."""
    
    query: str = Field(..., min_length=1, description="User query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")


class ChainOutput(BaseModel):
    """Standardized output for all chains."""
    
    response: str = Field(..., description="Chain response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    success: bool = Field(default=True, description="Success status")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class BaseChainRunner(ABC):
    """Abstract base class for all chain runners."""
    
    def __init__(self, config: BaseChainConfig):
        self.config = config
        self._llm = None
        self._chain = None
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)
    
    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self):
        """Create LLM instance based on configuration."""
        if self.config.model_provider == "openai":
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        elif self.config.model_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.model_name or "claude-3-sonnet-20240229",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config.model_provider}")
    
    @property
    def chain(self):
        """Lazy initialization of chain."""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain
    
    @abstractmethod
    def _create_chain(self):
        """Create the chain implementation. Must be implemented by subclasses."""
        pass
    
    def _safe_execution(self, func, *args, **kwargs) -> ChainOutput:
        """Execute function with error handling and timing."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self.logger.info(f"Chain executed successfully in {execution_time:.2f}s")
            
            # Handle different result types
            if isinstance(result, str):
                response = result
                metadata = {}
            elif isinstance(result, dict):
                response = result.get("output", str(result))
                metadata = {k: v for k, v in result.items() if k != "output"}
            else:
                response = str(result)
                metadata = {}
            
            return ChainOutput(
                response=response,
                metadata=metadata,
                success=True,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Chain execution timed out after {self.config.timeout}s"
            self.logger.error(error_msg)
            
            return ChainOutput(
                response="",
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Chain execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ChainOutput(
                response="",
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def run(self, input_data: Union[ChainInput, str, Dict[str, Any]]) -> ChainOutput:
        """Run the chain with input validation and error handling."""
        
        # Normalize input
        if isinstance(input_data, str):
            input_data = ChainInput(query=input_data)
        elif isinstance(input_data, dict):
            input_data = ChainInput(**input_data)
        elif not isinstance(input_data, ChainInput):
            raise ValueError("Input must be ChainInput, string, or dict")
        
        self.logger.debug(f"Running chain with input: {input_data.query[:100]}...")
        
        # Execute chain
        return self._safe_execution(self._execute_chain, input_data)
    
    async def arun(self, input_data: Union[ChainInput, str, Dict[str, Any]]) -> ChainOutput:
        """Async version of run method."""
        
        # Normalize input
        if isinstance(input_data, str):
            input_data = ChainInput(query=input_data)
        elif isinstance(input_data, dict):
            input_data = ChainInput(**input_data)
        elif not isinstance(input_data, ChainInput):
            raise ValueError("Input must be ChainInput, string, or dict")
        
        self.logger.debug(f"Running async chain with input: {input_data.query[:100]}...")
        
        # Execute chain asynchronously
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._aexecute_chain(input_data),
                timeout=self.config.timeout
            )
            execution_time = time.time() - start_time
            
            self.logger.info(f"Async chain executed successfully in {execution_time:.2f}s")
            
            # Handle different result types
            if isinstance(result, str):
                response = result
                metadata = {}
            elif isinstance(result, dict):
                response = result.get("output", str(result))
                metadata = {k: v for k, v in result.items() if k != "output"}
            else:
                response = str(result)
                metadata = {}
            
            return ChainOutput(
                response=response,
                metadata=metadata,
                success=True,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Async chain execution timed out after {self.config.timeout}s"
            self.logger.error(error_msg)
            
            return ChainOutput(
                response="",
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Async chain execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ChainOutput(
                response="",
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    @abstractmethod
    def _execute_chain(self, input_data: ChainInput) -> Any:
        """Execute the chain synchronously. Must be implemented by subclasses."""
        pass
    
    async def _aexecute_chain(self, input_data: ChainInput) -> Any:
        """Execute the chain asynchronously. Default implementation uses sync version."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._execute_chain, input_data
        )


class SimpleChain(BaseChainRunner):
    """Simple chain implementation for basic LLM interactions."""
    
    def __init__(self, config: BaseChainConfig, system_prompt: Optional[str] = None):
        super().__init__(config)
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    def _create_chain(self):
        """Create a simple prompt -> LLM -> output parser chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        
        return (
            RunnablePassthrough()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _execute_chain(self, input_data: ChainInput) -> str:
        """Execute the simple chain."""
        return self.chain.invoke({
            "query": input_data.query,
            "context": input_data.context or {}
        })
    
    async def _aexecute_chain(self, input_data: ChainInput) -> str:
        """Execute the simple chain asynchronously."""
        return await self.chain.ainvoke({
            "query": input_data.query,
            "context": input_data.context or {}
        })


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test configuration
    config = BaseChainConfig(
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        verbose=True
    )
    
    # Test simple chain
    chain = SimpleChain(config, "You are a helpful math tutor.")
    
    # Test cases
    test_inputs = [
        "What is 2 + 2?",
        ChainInput(query="Explain the concept of derivatives in calculus"),
        {"query": "What is the quadratic formula?", "context": {"level": "high_school"}}
    ]
    
    for test_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"Testing with input: {test_input}")
        print(f"{'='*50}")
        
        result = chain.run(test_input)
        
        print(f"Success: {result.success}")
        print(f"Response: {result.response}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        if not result.success:
            print(f"Error: {result.error}")