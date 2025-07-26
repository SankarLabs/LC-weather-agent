"""
Weather Agent Implementation

This module implements the main weather agent using LangChain's OpenAI Tools Agent,
following the patterns from examples/agent_chain.py.

Key Features:
- OpenAI Tools Agent with weather-specific tool integration
- Conversation memory for context-aware interactions
- Weather-optimized prompts and reasoning
- Comprehensive error handling and monitoring
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain.callbacks.base import BaseCallbackHandler

from utils.config import WeatherAgentConfig, get_config
from tools.weather_tools import create_weather_tools, WeatherError

logger = logging.getLogger(__name__)


class WeatherAgentMonitor(BaseCallbackHandler):
    """Custom callback handler for monitoring weather agent performance."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.api_usage = {}
        self.start_time = None
    
    def on_agent_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when agent starts processing."""
        self.start_time = time.time()
        self.request_count += 1
        query = inputs.get('input', '')
        logger.info(f"Weather agent started processing: {query[:50]}...")
    
    def on_agent_finish(self, finish: Any, **kwargs):
        """Called when agent finishes processing."""
        if self.start_time:
            response_time = time.time() - self.start_time
            self.response_times.append(response_time)
            logger.info(f"Weather agent finished processing in {response_time:.2f}s")
    
    def on_agent_error(self, error: Exception, **kwargs):
        """Called when agent encounters an error."""
        self.error_count += 1
        logger.error(f"Weather agent error: {error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = (self.error_count / self.request_count) * 100 if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "avg_response_time": avg_response_time,
            "api_usage": self.api_usage
        }


def create_weather_agent_prompt() -> ChatPromptTemplate:
    """
    Create weather-specific agent prompt optimized for natural conversation.
    
    Returns:
        ChatPromptTemplate: Configured prompt for weather agent
    """
    system_message = """You are a helpful weather assistant powered by real-time weather data.

Your capabilities include:
- 🌤️ Current weather conditions for any location worldwide
- 📅 Weather forecasts up to 5 days ahead
- ⚠️ Weather alerts and warnings
- 📍 Location lookup and coordinate resolution
- 👔 Weather-based recommendations for clothing, activities, and travel

Guidelines for responses:
- Always use the weather tools to get current data - never make up weather information
- Be conversational and friendly while staying informative
- When providing weather information, include key details like:
  • Temperature (and feels-like when significantly different)
  • Weather conditions and description
  • Humidity and wind information when relevant
  • Any notable weather patterns or changes
- If a location is ambiguous, ask for clarification or use the location lookup tool
- Remember previous locations the user has asked about for context
- Provide helpful recommendations based on weather conditions
- Use emojis to make responses more engaging and readable

Format your responses clearly with appropriate emojis and structure the information logically."""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])


class WeatherAgent:
    """
    Weather information agent with tools and memory.
    
    This agent provides comprehensive weather information through natural language
    conversations, maintaining context and offering personalized recommendations.
    """
    
    def __init__(self, config: Optional[WeatherAgentConfig] = None):
        """
        Initialize weather agent.
        
        Args:
            config: Optional configuration. If not provided, loads from environment.
        """
        self.config = config or get_config()
        self.monitor = WeatherAgentMonitor()
        
        # Initialize components
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.memory = self._create_memory()
        self.agent_executor = self._create_agent()
        
        logger.info("Weather agent initialized successfully")
    
    def _create_llm(self) -> ChatOpenAI:
        """Create and configure the language model."""
        try:
            llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.openai_api_key,
                request_timeout=30,
                max_retries=self.config.max_retries,
                streaming=False  # Disable streaming for simpler implementation
            )
            
            # Test the connection
            test_response = llm.invoke([{"role": "user", "content": "Hello"}])
            logger.info("LLM connection established successfully")
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
    
    def _create_tools(self) -> List:
        """Create weather tools for the agent."""
        try:
            tools = create_weather_tools(api_key=self.config.openweather_api_key)
            logger.info(f"Created {len(tools)} weather tools: {[tool.name for tool in tools]}")
            return tools
        except Exception as e:
            logger.error(f"Failed to create weather tools: {e}")
            raise
    
    def _create_memory(self) -> ConversationBufferWindowMemory:
        """Create conversation memory for context preservation."""
        memory = ConversationBufferWindowMemory(
            k=self.config.memory_window_size,
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            input_key="input"
        )
        
        logger.info(f"Created conversation memory with window size {self.config.memory_window_size}")
        return memory
    
    def _create_agent(self) -> AgentExecutor:
        """Create the weather agent executor."""
        try:
            # Create the agent prompt
            prompt = create_weather_agent_prompt()
            
            # Create the OpenAI Tools agent
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor with comprehensive configuration
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=self.config.verbose,
                max_iterations=self.config.max_iterations,
                max_execution_time=self.config.max_execution_time,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                callbacks=[self.monitor] if hasattr(self, 'monitor') else []
            )
            
            logger.info("Weather agent executor created successfully")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Failed to create weather agent: {e}")
            raise
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a weather query from the user.
        
        Args:
            user_input: User's weather question or request
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing weather query: {user_input[:100]}...")
            
            # Execute the agent
            result = self.agent_executor.invoke({"input": user_input})
            
            # Extract and format response
            response = {
                "success": True,
                "query": user_input,
                "response": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "timestamp": datetime.now().isoformat(),
                "memory_context": len(self.memory.load_memory_variables({}).get("chat_history", []))
            }
            
            logger.info("Weather query processed successfully")
            return response
            
        except WeatherError as e:
            # Weather-specific errors
            error_msg = f"Weather service error: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "query": user_input,
                "error": error_msg,
                "error_type": "weather_error",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # General errors
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "query": user_input,
                "error": error_msg,
                "error_type": "agent_error",
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_conversation(self):
        """Reset the agent's conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory reset")
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history."""
        memory_vars = self.memory.load_memory_variables({})
        return memory_vars.get("chat_history", [])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.monitor.get_metrics()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the weather agent."""
        try:
            # Test LLM connection
            llm_test = self.llm.invoke([{"role": "user", "content": "Test"}])
            llm_status = "healthy" if llm_test else "unhealthy"
            
            # Test tools
            tools_status = "healthy" if len(self.tools) == 4 else "unhealthy"
            
            # Test memory
            memory_status = "healthy" if self.memory else "unhealthy"
            
            return {
                "status": "healthy" if all([
                    llm_status == "healthy",
                    tools_status == "healthy", 
                    memory_status == "healthy"
                ]) else "unhealthy",
                "components": {
                    "llm": llm_status,
                    "tools": tools_status,
                    "memory": memory_status
                },
                "metrics": self.get_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class WeatherConversationManager:
    """Utility class for managing weather conversation context and history."""
    
    def __init__(self):
        self.location_history = []
        self.user_preferences = {}
        self.conversation_stats = {
            "queries_count": 0,
            "locations_queried": set(),
            "start_time": datetime.now()
        }
    
    def add_location(self, location: str):
        """Track user's queried locations."""
        if location and location.strip():
            location = location.strip().title()
            if location not in self.location_history:
                self.location_history.append(location)
                if len(self.location_history) > 20:  # Keep last 20 locations
                    self.location_history.pop(0)
            
            self.conversation_stats["locations_queried"].add(location)
    
    def get_frequent_locations(self) -> List[str]:
        """Get user's frequently queried locations."""
        return self.location_history[-5:]  # Last 5 locations
    
    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences."""
        self.user_preferences.update(preferences)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary statistics."""
        return {
            "total_queries": self.conversation_stats["queries_count"],
            "unique_locations": len(self.conversation_stats["locations_queried"]),
            "recent_locations": self.get_frequent_locations(),
            "session_duration": str(datetime.now() - self.conversation_stats["start_time"]),
            "user_preferences": self.user_preferences
        }


def create_weather_agent(config: Optional[WeatherAgentConfig] = None) -> WeatherAgent:
    """
    Factory function to create a weather agent.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured weather agent
    """
    return WeatherAgent(config=config)


if __name__ == "__main__":
    """Test weather agent functionality."""
    
    # Test environment setup
    try:
        from utils.config import validate_environment
        
        if not validate_environment():
            print("❌ Environment validation failed. Please set up API keys in .env file.")
            exit(1)
        
        print("✅ Environment validation passed")
        
        # Create and test weather agent
        print("Creating weather agent...")
        agent = create_weather_agent()
        
        print("✅ Weather agent created successfully")
        
        # Health check
        health = agent.health_check()
        print(f"Health check: {health['status']}")
        
        # Test query (this would require actual API keys)
        print("\\nTesting basic functionality...")
        print("Weather agent is ready for queries!")
        print("\\nExample queries you can try:")
        print("- 'What's the weather like in London?'")
        print("- 'Give me a 5-day forecast for New York'")
        print("- 'Is it going to rain in Tokyo tomorrow?'")
        
    except Exception as e:
        print(f"❌ Weather agent test failed: {e}")
        exit(1)