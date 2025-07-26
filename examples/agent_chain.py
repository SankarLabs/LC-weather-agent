"""
Agent Implementation Example

This file demonstrates how to create LangChain agents with custom tools.
It shows best practices for agent architecture, tool creation, and
the ReAct (Reasoning and Acting) pattern.

Key Patterns Demonstrated:
- Agent creation with ReAct pattern
- Custom tool implementation using BaseTool
- Tool integration and management
- Agent execution with error handling
- Multi-step reasoning and action chains
- Agent memory and state management
"""

import os
import logging
import json
import requests
from typing import Dict, Any, List, Optional, Type
from dotenv import load_dotenv

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool, Tool
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for agent system."""
    model_name: str = Field("gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(0.1, description="Model temperature")
    max_iterations: int = Field(5, description="Maximum agent iterations")
    max_execution_time: int = Field(60, description="Maximum execution time in seconds")
    verbose: bool = Field(True, description="Enable verbose logging")

# Custom Tools Implementation

class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    """Calculator tool for mathematical operations."""
    
    name: str = "calculator"
    description: str = "Useful for mathematical calculations. Input should be a valid mathematical expression."
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Execute the calculator tool."""
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Basic security: only allow safe characters
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, (), .) are allowed."
            
            # Evaluate the expression
            result = eval(expression)
            
            logger.info(f"Calculator: {expression} = {result}")
            return f"The result of {expression} is {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed."
        except Exception as e:
            return f"Error: Could not evaluate expression '{expression}'. {str(e)}"
    
    def _arun(self, expression: str) -> str:
        """Async version (not implemented for this example)."""
        raise NotImplementedError("CalculatorTool does not support async operations")

class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Search query to execute")

class WebSearchTool(BaseTool):
    """Web search tool (simplified mock implementation)."""
    
    name: str = "web_search"
    description: str = "Search the web for current information. Input should be a search query."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        """Execute web search (mock implementation)."""
        try:
            logger.info(f"Web search: {query}")
            
            # Mock search results (in a real implementation, use actual search API)
            mock_results = {
                "python": "Python is a high-level programming language known for its simplicity and readability.",
                "langchain": "LangChain is a framework for developing applications powered by language models.",
                "ai": "Artificial Intelligence refers to computer systems that can perform tasks typically requiring human intelligence.",
                "weather": "Current weather information varies by location. Check local weather services for accurate forecasts."
            }
            
            # Simple keyword matching for demo
            query_lower = query.lower()
            for keyword, result in mock_results.items():
                if keyword in query_lower:
                    return f"Search results for '{query}': {result}"
            
            return f"Search results for '{query}': No specific information found. This is a mock search tool."
            
        except Exception as e:
            return f"Error: Web search failed. {str(e)}"
    
    def _arun(self, query: str) -> str:
        """Async version (not implemented for this example)."""
        raise NotImplementedError("WebSearchTool does not support async operations")

class FileManagerInput(BaseModel):
    """Input schema for file manager tool."""
    action: str = Field(description="Action to perform: 'read', 'write', 'list'")
    filename: Optional[str] = Field(None, description="Name of file to operate on")
    content: Optional[str] = Field(None, description="Content to write (for write action)")

class FileManagerTool(BaseTool):
    """File management tool for reading and writing files."""
    
    name: str = "file_manager"
    description: str = """Manage files in the current directory. 
    Actions: 'read filename' to read a file, 'write filename content' to write a file, 'list' to list files."""
    args_schema: Type[BaseModel] = FileManagerInput
    
    def _run(self, action: str, filename: Optional[str] = None, content: Optional[str] = None) -> str:
        """Execute file management operations."""
        try:
            if action == "list":
                import os
                files = [f for f in os.listdir('.') if os.path.isfile(f)]
                return f"Files in current directory: {', '.join(files)}"
            
            elif action == "read":
                if not filename:
                    return "Error: Filename required for read operation"
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    return f"Content of {filename}:\n{file_content[:500]}..." if len(file_content) > 500 else f"Content of {filename}:\n{file_content}"
                except FileNotFoundError:
                    return f"Error: File '{filename}' not found"
            
            elif action == "write":
                if not filename or not content:
                    return "Error: Both filename and content required for write operation"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote content to {filename}"
            
            else:
                return f"Error: Unknown action '{action}'. Available actions: read, write, list"
                
        except Exception as e:
            return f"Error: File operation failed. {str(e)}"
    
    def _arun(self, action: str, filename: Optional[str] = None, content: Optional[str] = None) -> str:
        """Async version (not implemented for this example)."""
        raise NotImplementedError("FileManagerTool does not support async operations")

class ResearchAgent:
    """Research agent with multiple tools and memory."""
    
    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create the tools available to the agent."""
        tools = [
            CalculatorTool(),
            WebSearchTool(),
            FileManagerTool(),
        ]
        
        # Add a simple string manipulation tool using the Tool wrapper
        def string_operations(input_str: str) -> str:
            """Perform string operations like uppercase, lowercase, length, reverse."""
            try:
                parts = input_str.split(' ', 1)
                if len(parts) != 2:
                    return "Error: Format should be 'operation text', e.g., 'uppercase hello world'"
                
                operation, text = parts
                operation = operation.lower()
                
                if operation == "uppercase":
                    return f"Uppercase: {text.upper()}"
                elif operation == "lowercase":
                    return f"Lowercase: {text.lower()}"
                elif operation == "length":
                    return f"Length of '{text}': {len(text)} characters"
                elif operation == "reverse":
                    return f"Reversed: {text[::-1]}"
                elif operation == "words":
                    word_count = len(text.split())
                    return f"Word count in '{text}': {word_count} words"
                else:
                    return f"Error: Unknown operation '{operation}'. Available: uppercase, lowercase, length, reverse, words"
                    
            except Exception as e:
                return f"Error in string operation: {str(e)}"
        
        string_tool = Tool(
            name="string_operations",
            description="Perform string operations. Format: 'operation text'. Operations: uppercase, lowercase, length, reverse, words",
            func=string_operations
        )
        
        tools.append(string_tool)
        
        logger.info(f"Created {len(tools)} tools for agent: {[tool.name for tool in tools]}")
        return tools
    
    def _create_agent_prompt(self) -> PromptTemplate:
        """Create the agent prompt template."""
        template = """You are a helpful research assistant with access to various tools. 
Use the tools to help answer questions and complete tasks effectively.

TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "chat_history", "agent_scratchpad"]
        )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        try:
            # Create the agent prompt
            prompt = self._create_agent_prompt()
            
            # Create the ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=self.config.verbose,
                max_iterations=self.config.max_iterations,
                max_execution_time=self.config.max_execution_time,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            logger.info("Research agent created successfully")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent with a query.
        
        Args:
            query: User query or task
            
        Returns:
            Dict with result and metadata
        """
        try:
            logger.info(f"Agent processing query: {query[:100]}...")
            
            # Execute the agent
            result = self.agent_executor.invoke({"input": query})
            
            # Extract information
            output = {
                "success": True,
                "query": query,
                "answer": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "memory": self.memory.load_memory_variables({})
            }
            
            logger.info("Agent execution completed successfully")
            return output
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "query": query,
                "error": error_msg,
                "intermediate_steps": [],
                "memory": {}
            }
    
    def reset_memory(self):
        """Reset the agent's conversation memory."""
        self.memory.clear()
        logger.info("Agent memory reset")

def demonstrate_agent_capabilities():
    """Demonstrate the agent's capabilities with various tasks."""
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    print("\n" + "="*60)
    print("RESEARCH AGENT DEMONSTRATION")
    print("="*60)
    
    # Initialize agent
    config = AgentConfig(
        temperature=0.1,
        max_iterations=5,
        verbose=True
    )
    
    agent = ResearchAgent(config)
    
    # Test queries demonstrating different capabilities
    test_queries = [
        "What is 25 * 17 + 45?",
        "Search for information about Python programming language",
        "Create a file called 'test.txt' with the content 'Hello, LangChain Agent!'",
        "Read the file 'test.txt' that I just created",
        "Convert the text 'Hello World' to uppercase and tell me its length",
        "What files are in the current directory?",
        "Calculate the square root of 144 and then multiply it by 5"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n" + "-"*50)
        print(f"TASK {i}: {query}")
        print("-"*50)
        
        result = agent.run(query)
        
        if result["success"]:
            print(f"ANSWER: {result['answer']}")
            
            if result["intermediate_steps"]:
                print("\nREASONING STEPS:")
                for j, (action, observation) in enumerate(result["intermediate_steps"], 1):
                    print(f"  Step {j}:")
                    print(f"    Action: {action.tool} - {action.tool_input}")
                    print(f"    Result: {observation}")
        else:
            print(f"ERROR: {result['error']}")
        
        print(f"Memory: {len(result.get('memory', {}).get('chat_history', []))} messages")
    
    # Demonstrate conversation memory
    print(f"\n" + "-"*50)
    print("MEMORY DEMONSTRATION")
    print("-"*50)
    
    memory_query = "What was the result of my first calculation?"
    print(f"TASK: {memory_query}")
    
    result = agent.run(memory_query)
    
    if result["success"]:
        print(f"ANSWER: {result['answer']}")
    else:
        print(f"ERROR: {result['error']}")

def create_custom_agent_example():
    """Example of creating a custom agent with specific tools."""
    
    def create_weather_tool():
        """Create a mock weather tool."""
        def get_weather(location: str) -> str:
            """Get weather information for a location (mock implementation)."""
            mock_weather = {
                "new york": "Sunny, 72°F",
                "london": "Cloudy, 15°C",
                "tokyo": "Rainy, 18°C",
                "paris": "Partly cloudy, 20°C"
            }
            
            location_lower = location.lower()
            weather = mock_weather.get(location_lower, f"Weather data not available for {location}")
            return f"Weather in {location}: {weather}"
        
        return Tool(
            name="weather",
            description="Get current weather for a specific location",
            func=get_weather
        )
    
    def create_task_agent():
        """Create an agent specialized for task management."""
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        tools = [
            create_weather_tool(),
            CalculatorTool(),
        ]
        
        prompt = PromptTemplate(
            template="""You are a task management assistant. Help users with calculations, weather information, and organizing their tasks.

You have access to these tools:
{tools}

Use this format:
Question: {input}
Thought: I should think about what tool to use
Action: {tool_names}
Action Input: the input to the action
Observation: the result of the action
Thought: I now know what to respond
Final Answer: my response to the user

Begin!
Question: {input}
{agent_scratchpad}""",
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )
        
        agent = create_react_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=3
        )
    
    # Example usage
    task_agent = create_task_agent()
    
    print("\n" + "="*50)
    print("CUSTOM TASK AGENT EXAMPLE")
    print("="*50)
    
    custom_queries = [
        "What's the weather like in Tokyo?",
        "Calculate 15% tip on a $85 bill",
        "What's the weather in London and what's 100 divided by 4?"
    ]
    
    for query in custom_queries:
        print(f"\nQuery: {query}")
        try:
            result = task_agent.invoke({"input": query})
            print(f"Response: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    """
    Run the agent demonstrations.
    
    This demonstrates:
    1. Agent creation with multiple tools
    2. ReAct reasoning pattern
    3. Multi-step task execution
    4. Memory and conversation handling
    5. Custom tool creation
    6. Error handling in agent workflows
    """
    
    try:
        # Main agent demonstration
        demonstrate_agent_capabilities()
        
        # Custom agent example
        create_custom_agent_example()
        
    except KeyboardInterrupt:
        logger.info("Agent demonstration interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error in demonstration: {e}")
        raise
    
    finally:
        logger.info("Agent demonstration completed")