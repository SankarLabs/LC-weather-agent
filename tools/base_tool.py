"""
Base tool implementation with common patterns and safety features.

This module provides base classes and mixins that all production tools
should inherit from, ensuring consistency and safety across the application.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import BaseTool as LangChainBaseTool

logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    """Base input schema for tools."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )


class ToolOutput(BaseModel):
    """Standardized output for all tools."""
    
    result: str = Field(..., description="Tool execution result")
    success: bool = Field(default=True, description="Success status")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SafeToolMixin:
    """Mixin providing safety features for tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._execution_count = 0
        self._total_execution_time = 0.0
    
    def _validate_input_safety(self, input_str: str) -> bool:
        """
        Validate input for basic safety.
        Override this method for tool-specific safety checks.
        """
        # Basic checks
        if not input_str or len(input_str.strip()) == 0:
            return False
        
        # Check for excessively long inputs
        if len(input_str) > 10000:
            self.logger.warning(f"Input too long: {len(input_str)} characters")
            return False
        
        # Check for dangerous patterns (customize per tool)
        dangerous_patterns = [
            'rm -rf',
            'del /f',
            'DROP TABLE',
            'DELETE FROM',
            '__import__',
            'exec(',
            'eval(',
        ]
        
        input_lower = input_str.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in input_lower:
                self.logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        
        return True
    
    def _safe_execute(self, func, *args, **kwargs) -> ToolOutput:
        """Execute function with error handling and timing."""
        start_time = time.time()
        self._execution_count += 1
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            
            self.logger.debug(f"Tool {self.name} executed successfully in {execution_time:.2f}s")
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)
            
            return ToolOutput(
                result=result,
                success=True,
                execution_time=execution_time,
                metadata={
                    "execution_count": self._execution_count,
                    "total_execution_time": self._total_execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            error_msg = f"Tool {self.name} execution failed: {str(e)}"
            
            self.logger.error(error_msg, exc_info=True)
            
            return ToolOutput(
                result="",
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "execution_count": self._execution_count,
                    "total_execution_time": self._total_execution_time
                }
            )
    
    async def _safe_aexecute(self, func, *args, **kwargs) -> ToolOutput:
        """Async version of safe_execute."""
        start_time = time.time()
        self._execution_count += 1
        
        try:
            # Execute the async function
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            
            self.logger.debug(f"Tool {self.name} executed async successfully in {execution_time:.2f}s")
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)
            
            return ToolOutput(
                result=result,
                success=True,
                execution_time=execution_time,
                metadata={
                    "execution_count": self._execution_count,
                    "total_execution_time": self._total_execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            error_msg = f"Tool {self.name} async execution failed: {str(e)}"
            
            self.logger.error(error_msg, exc_info=True)
            
            return ToolOutput(
                result="",
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "execution_count": self._execution_count,
                    "total_execution_time": self._total_execution_time
                }
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this tool."""
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._execution_count 
                if self._execution_count > 0 else 0
            )
        }


class BaseTool(SafeToolMixin, LangChainBaseTool, ABC):
    """
    Enhanced base tool class combining LangChain's BaseTool with safety features.
    
    All production tools should inherit from this class.
    """
    
    # Tool configuration
    timeout: Optional[int] = Field(default=30, description="Tool execution timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _run(self, *args, **kwargs) -> str:
        """
        Synchronous tool execution with safety wrapper.
        
        This method wraps the actual tool implementation with safety checks
        and error handling. Override _execute() instead of this method.
        """
        # Validate inputs if provided
        if args and isinstance(args[0], str):
            if not self._validate_input_safety(args[0]):
                return "Error: Input failed safety validation"
        
        # Execute with retry logic
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                output = self._safe_execute(self._execute, *args, **kwargs)
                
                if output.success:
                    return output.result
                else:
                    if attempt == self.max_retries:
                        return f"Error after {self.max_retries + 1} attempts: {output.error}"
                    last_exception = output.error
                    
            except Exception as e:
                last_exception = str(e)
                if attempt == self.max_retries:
                    return f"Error after {self.max_retries + 1} attempts: {str(e)}"
                
                # Wait before retry
                time.sleep(0.5 * (attempt + 1))
        
        return f"Tool execution failed: {last_exception}"
    
    async def _arun(self, *args, **kwargs) -> str:
        """
        Asynchronous tool execution with safety wrapper.
        
        This method wraps the actual async tool implementation with safety checks
        and error handling. Override _aexecute() instead of this method.
        """
        # Validate inputs if provided
        if args and isinstance(args[0], str):
            if not self._validate_input_safety(args[0]):
                return "Error: Input failed safety validation"
        
        # Execute with retry logic
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.timeout:
                    output = await asyncio.wait_for(
                        self._safe_aexecute(self._aexecute, *args, **kwargs),
                        timeout=self.timeout
                    )
                else:
                    output = await self._safe_aexecute(self._aexecute, *args, **kwargs)
                
                if output.success:
                    return output.result
                else:
                    if attempt == self.max_retries:
                        return f"Error after {self.max_retries + 1} attempts: {output.error}"
                    last_exception = output.error
                    
            except asyncio.TimeoutError:
                error_msg = f"Tool execution timed out after {self.timeout}s"
                if attempt == self.max_retries:
                    return f"Error after {self.max_retries + 1} attempts: {error_msg}"
                last_exception = error_msg
                
            except Exception as e:
                last_exception = str(e)
                if attempt == self.max_retries:
                    return f"Error after {self.max_retries + 1} attempts: {str(e)}"
                
                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return f"Tool execution failed: {last_exception}"
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> str:
        """
        Implement the actual tool logic here.
        
        This method should contain the core functionality of your tool.
        It will be called by _run() with safety wrappers.
        """
        pass
    
    async def _aexecute(self, *args, **kwargs) -> str:
        """
        Implement the async version of tool logic here.
        
        Default implementation runs the sync version in an executor.
        Override for true async implementation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute, *args, **kwargs)


# Example tool implementation
class EchoTool(BaseTool):
    """Simple echo tool for testing and examples."""
    
    name: str = "echo"
    description: str = "Echo back the input text. Useful for testing and debugging."
    
    def _execute(self, text: str) -> str:
        """Echo the input text."""
        return f"Echo: {text}"
    
    async def _aexecute(self, text: str) -> str:
        """Async echo implementation."""
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Async Echo: {text}"


# Example usage
if __name__ == "__main__":
    # Test the echo tool
    echo_tool = EchoTool()
    
    print("Testing EchoTool:")
    print("Sync:", echo_tool._run("Hello, World!"))
    
    # Test async
    async def test_async():
        result = await echo_tool._arun("Hello, Async World!")
        print("Async:", result)
    
    asyncio.run(test_async())
    
    # Show stats
    print("Stats:", echo_tool.get_stats())