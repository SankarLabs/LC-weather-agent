"""
Production-ready agent chain implementations.

This module provides robust agent implementations with proper tool integration,
memory management, and error handling patterns.
"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from .base_chain import BaseChainRunner, BaseChainConfig, ChainInput, ChainOutput


class AgentChainConfig(BaseChainConfig):
    """Extended configuration for agent chains."""
    
    # Agent-specific configuration
    max_iterations: int = Field(default=10, gt=0, description="Maximum agent iterations")
    max_execution_time: int = Field(default=120, gt=0, description="Maximum execution time")
    handle_parsing_errors: bool = Field(default=True, description="Handle parsing errors gracefully")
    return_intermediate_steps: bool = Field(default=True, description="Return reasoning steps")
    
    # Memory configuration
    memory_type: str = Field(default="buffer_window", description="Memory type to use")
    memory_window_size: int = Field(default=10, gt=0, description="Memory window size")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for persistence")
    
    # Tool configuration
    tool_timeout: int = Field(default=30, gt=0, description="Individual tool timeout")
    allow_dangerous_tools: bool = Field(default=False, description="Allow potentially dangerous tools")


class AgentInput(ChainInput):
    """Extended input for agent chains."""
    
    tools_to_use: Optional[List[str]] = Field(default=None, description="Specific tools to enable")
    max_iterations: Optional[int] = Field(default=None, description="Override max iterations")


class AgentOutput(ChainOutput):
    """Extended output for agent chains."""
    
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Agent reasoning steps")
    tools_used: List[str] = Field(default_factory=list, description="Tools that were used")
    iterations_used: int = Field(default=0, description="Number of iterations used")


class AgentChain(BaseChainRunner):
    """Production-ready agent chain with tool integration."""
    
    def __init__(
        self, 
        config: AgentChainConfig,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None
    ):
        super().__init__(config)
        self.agent_config = config
        self.tools = tools
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._memory = None
        self._agent_executor = None
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return """You are a helpful AI assistant with access to various tools. 
Use the tools to help answer questions and complete tasks effectively.

Guidelines:
- Always think step by step about what you need to do
- Use tools when you need external information or to perform actions
- Be precise and accurate in your responses
- If you're unsure about something, say so rather than guessing
- Break down complex tasks into smaller steps"""
    
    @property 
    def memory(self):
        """Lazy initialization of memory."""
        if self._memory is None:
            self._memory = self._create_memory()
        return self._memory
    
    def _create_memory(self):
        """Create memory instance based on configuration."""
        if self.agent_config.redis_url:
            # Use Redis for persistent memory
            message_history = RedisChatMessageHistory(
                url=self.agent_config.redis_url,
                session_id="default"  # Should be dynamic in production
            )
            
            return ConversationBufferWindowMemory(
                k=self.agent_config.memory_window_size,
                chat_memory=message_history,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
                input_key="input"
            )
        else:
            # Use in-memory storage
            return ConversationBufferWindowMemory(
                k=self.agent_config.memory_window_size,
                memory_key="chat_history", 
                return_messages=True,
                output_key="output",
                input_key="input"
            )
    
    def _create_chain(self):
        """Create the agent executor."""
        if self._agent_executor is None:
            self._agent_executor = self._create_agent_executor()
        return self._agent_executor
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and memory."""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.agent_config.verbose,
            max_iterations=self.agent_config.max_iterations,
            max_execution_time=self.agent_config.max_execution_time,
            handle_parsing_errors=self.agent_config.handle_parsing_errors,
            return_intermediate_steps=self.agent_config.return_intermediate_steps
        )
    
    def _execute_chain(self, input_data: AgentInput) -> Dict[str, Any]:
        """Execute the agent chain."""
        
        # Override config if specified in input
        if input_data.max_iterations:
            self.chain.max_iterations = input_data.max_iterations
        
        # Filter tools if specified
        if input_data.tools_to_use:
            available_tools = [
                tool for tool in self.tools 
                if tool.name in input_data.tools_to_use
            ]
            if available_tools:
                # Create temporary executor with filtered tools
                temp_executor = AgentExecutor(
                    agent=self.chain.agent,
                    tools=available_tools,
                    memory=self.memory,
                    verbose=self.agent_config.verbose,
                    max_iterations=input_data.max_iterations or self.agent_config.max_iterations,
                    max_execution_time=self.agent_config.max_execution_time,
                    handle_parsing_errors=self.agent_config.handle_parsing_errors,
                    return_intermediate_steps=self.agent_config.return_intermediate_steps
                )
                
                return temp_executor.invoke({
                    "input": input_data.query,
                    "context": input_data.context or {}
                })
        
        # Execute with all tools
        return self.chain.invoke({
            "input": input_data.query,
            "context": input_data.context or {}
        })
    
    async def _aexecute_chain(self, input_data: AgentInput) -> Dict[str, Any]:
        """Execute the agent chain asynchronously."""
        
        # Override config if specified in input
        if input_data.max_iterations:
            self.chain.max_iterations = input_data.max_iterations
        
        # Execute asynchronously
        return await self.chain.ainvoke({
            "input": input_data.query,
            "context": input_data.context or {}
        })
    
    def run(self, input_data) -> AgentOutput:
        """Run the agent with enhanced output."""
        
        # Normalize input to AgentInput
        if isinstance(input_data, str):
            input_data = AgentInput(query=input_data)
        elif isinstance(input_data, dict):
            input_data = AgentInput(**input_data)
        elif isinstance(input_data, ChainInput):
            # Convert ChainInput to AgentInput
            input_data = AgentInput(**input_data.dict())
        
        # Execute the chain
        result = self._safe_execution(self._execute_chain, input_data)
        
        # Convert to AgentOutput
        if result.success and isinstance(result.metadata, dict):
            intermediate_steps = result.metadata.get("intermediate_steps", [])
            
            # Extract tools used
            tools_used = []
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, _ = step
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
            
            return AgentOutput(
                response=result.response,
                metadata=result.metadata,
                success=result.success,
                error=result.error,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                intermediate_steps=intermediate_steps,
                tools_used=list(set(tools_used)),
                iterations_used=len(intermediate_steps)
            )
        else:
            return AgentOutput(
                response=result.response,
                metadata=result.metadata,
                success=result.success,
                error=result.error,
                execution_time=result.execution_time,
                token_usage=result.token_usage
            )
    
    async def arun(self, input_data) -> AgentOutput:
        """Run the agent asynchronously with enhanced output."""
        
        # Normalize input to AgentInput
        if isinstance(input_data, str):
            input_data = AgentInput(query=input_data)
        elif isinstance(input_data, dict):
            input_data = AgentInput(**input_data)
        elif isinstance(input_data, ChainInput):
            # Convert ChainInput to AgentInput
            input_data = AgentInput(**input_data.dict())
        
        # Execute the chain asynchronously
        result = await self.arun_base(input_data)
        
        # Convert to AgentOutput (similar to sync version)
        if result.success and isinstance(result.metadata, dict):
            intermediate_steps = result.metadata.get("intermediate_steps", [])
            
            # Extract tools used
            tools_used = []
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, _ = step
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
            
            return AgentOutput(
                response=result.response,
                metadata=result.metadata,
                success=result.success,
                error=result.error,
                execution_time=result.execution_time,
                token_usage=result.token_usage,
                intermediate_steps=intermediate_steps,
                tools_used=list(set(tools_used)),
                iterations_used=len(intermediate_steps)
            )
        else:
            return AgentOutput(
                response=result.response,
                metadata=result.metadata,
                success=result.success,
                error=result.error,
                execution_time=result.execution_time,
                token_usage=result.token_usage
            )
    
    async def arun_base(self, input_data: AgentInput) -> ChainOutput:
        """Base async run method that returns ChainOutput."""
        return await super().arun(input_data)
    
    def reset_memory(self):
        """Reset the agent's memory."""
        if self._memory:
            self.memory.clear()
            self.logger.info("Agent memory reset")
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent."""
        if tool not in self.tools:
            self.tools.append(tool)
            # Reset the agent executor to include new tool
            self._agent_executor = None
            self.logger.info(f"Added tool: {tool.name}")
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent."""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
        # Reset the agent executor
        self._agent_executor = None
        self.logger.info(f"Removed tool: {tool_name}")
    
    def list_tools(self) -> List[str]:
        """List available tool names."""
        return [tool.name for tool in self.tools]


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain.tools import Tool
    
    load_dotenv()
    
    # Create a simple calculator tool
    def calculator(expression: str) -> str:
        """Simple calculator for basic math operations."""
        try:
            # Basic safety: only allow safe characters
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    calculator_tool = Tool(
        name="calculator",
        description="Calculate mathematical expressions. Input should be a valid math expression.",
        func=calculator
    )
    
    # Create agent configuration
    config = AgentChainConfig(
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_iterations=5,
        verbose=True
    )
    
    # Create agent chain
    agent = AgentChain(
        config=config,
        tools=[calculator_tool],
        system_prompt="You are a math tutor. Use the calculator tool for computations."
    )
    
    # Test queries
    test_queries = [
        "What is 25 * 17 + 100?",
        "Calculate the square root of 144 and multiply by 3",
        "What is 15% of 250?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        result = agent.run(query)
        
        print(f"Success: {result.success}")
        print(f"Response: {result.response}")
        print(f"Tools used: {result.tools_used}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        if result.intermediate_steps:
            print("\nReasoning steps:")
            for i, step in enumerate(result.intermediate_steps, 1):
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    print(f"  {i}. Action: {action.tool} - {action.tool_input}")
                    print(f"     Result: {observation}")
        
        if not result.success:
            print(f"Error: {result.error}")