"""
Test suite for weather agent.

This module contains comprehensive tests for the weather agent,
following the patterns from existing test files.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from agents.weather_agent import (
    WeatherAgent,
    WeatherAgentMonitor,
    WeatherConversationManager,
    create_weather_agent_prompt,
    create_weather_agent
)
from utils.config import WeatherAgentConfig


class TestWeatherAgentConfig:
    """Test weather agent configuration."""
    
    def test_config_creation(self):
        """Test creating configuration with valid values."""
        config = WeatherAgentConfig(
            openai_api_key="test_openai_key",
            openweather_api_key="test_weather_key",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        assert config.openai_api_key == "test_openai_key"
        assert config.openweather_api_key == "test_weather_key"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            WeatherAgentConfig(
                openai_api_key="test_key",
                openweather_api_key="test_key",
                temperature=3.0
            )
        
        # Test invalid max_iterations
        with pytest.raises(ValueError, match="Max iterations must be between 1 and 20"):
            WeatherAgentConfig(
                openai_api_key="test_key", 
                openweather_api_key="test_key",
                max_iterations=25
            )


class TestWeatherAgentMonitor:
    """Test weather agent monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor for testing."""
        return WeatherAgentMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.request_count == 0
        assert monitor.error_count == 0
        assert monitor.response_times == []
    
    def test_agent_start_callback(self, monitor):
        """Test agent start callback."""
        monitor.on_agent_start({}, {"input": "What's the weather?"})
        
        assert monitor.request_count == 1
        assert monitor.start_time is not None
    
    def test_agent_finish_callback(self, monitor):
        """Test agent finish callback."""
        import time
        monitor.start_time = time.time()
        monitor.on_agent_finish({})
        
        assert len(monitor.response_times) == 1
        assert monitor.response_times[0] > 0
    
    def test_agent_error_callback(self, monitor):
        """Test agent error callback."""
        monitor.on_agent_error(Exception("Test error"))
        
        assert monitor.error_count == 1
    
    def test_get_metrics(self, monitor):
        """Test getting performance metrics."""
        # Simulate some activity
        monitor.request_count = 10
        monitor.error_count = 2
        monitor.response_times = [1.0, 2.0, 1.5]
        
        metrics = monitor.get_metrics()
        
        assert metrics["total_requests"] == 10
        assert metrics["error_count"] == 2
        assert metrics["error_rate_percent"] == 20.0
        assert metrics["avg_response_time"] == 1.5


class TestWeatherConversationManager:
    """Test conversation management functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create conversation manager for testing."""
        return WeatherConversationManager()
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.location_history == []
        assert manager.user_preferences == {}
        assert manager.conversation_stats["queries_count"] == 0
    
    def test_add_location(self, manager):
        """Test adding locations to history."""
        manager.add_location("London")
        manager.add_location("Paris")
        manager.add_location("London")  # Duplicate
        
        assert len(manager.location_history) == 2
        assert "London" in manager.location_history
        assert "Paris" in manager.location_history
        assert len(manager.conversation_stats["locations_queried"]) == 2
    
    def test_location_history_limit(self, manager):
        """Test location history size limit."""
        # Add more than 20 locations
        for i in range(25):
            manager.add_location(f"City{i}")
        
        assert len(manager.location_history) == 20
        assert "City4" not in manager.location_history  # Should be removed
        assert "City24" in manager.location_history  # Should be kept
    
    def test_get_frequent_locations(self, manager):
        """Test getting frequent locations."""
        locations = ["A", "B", "C", "D", "E", "F"]
        for loc in locations:
            manager.add_location(loc)
        
        frequent = manager.get_frequent_locations()
        assert len(frequent) == 5
        assert frequent == ["B", "C", "D", "E", "F"]  # Last 5
    
    def test_update_preferences(self, manager):
        """Test updating user preferences."""
        manager.update_preferences({"units": "imperial", "theme": "dark"})
        assert manager.user_preferences["units"] == "imperial"
        assert manager.user_preferences["theme"] == "dark"
    
    def test_conversation_summary(self, manager):
        """Test getting conversation summary."""
        manager.add_location("Tokyo")
        manager.add_location("Osaka")
        manager.update_preferences({"units": "metric"})
        
        summary = manager.get_conversation_summary()
        
        assert summary["unique_locations"] == 2
        assert "Tokyo" in summary["recent_locations"]
        assert summary["user_preferences"]["units"] == "metric"


class TestWeatherAgentPrompt:
    """Test weather agent prompt creation."""
    
    def test_prompt_creation(self):
        """Test creating weather agent prompt."""
        prompt = create_weather_agent_prompt()
        
        assert prompt is not None
        # Check that prompt contains weather-specific guidance
        prompt_str = str(prompt)
        assert "weather" in prompt_str.lower()
        assert "temperature" in prompt_str.lower()


class TestWeatherAgent:
    """Test weather agent functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WeatherAgentConfig(
            openai_api_key="test_openai_key",
            openweather_api_key="test_weather_key",
            verbose=False  # Reduce noise in tests
        )
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    def test_agent_initialization(self, mock_create_tools, mock_llm, config):
        """Test weather agent initialization."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="Test response")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock(name="weather_tool"), Mock(name="forecast_tool")]
        mock_create_tools.return_value = mock_tools
        
        # Create agent
        agent = WeatherAgent(config=config)
        
        # Verify initialization
        assert agent.config == config
        assert agent.llm == mock_llm_instance
        assert len(agent.tools) == 2
        assert agent.memory is not None
        assert agent.agent_executor is not None
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    @patch('agents.weather_agent.create_openai_tools_agent')
    @patch('agents.weather_agent.AgentExecutor')
    def test_agent_query_success(self, mock_executor_class, mock_create_agent, mock_create_tools, mock_llm, config):
        """Test successful weather query."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="Test LLM response")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock()]
        mock_create_tools.return_value = mock_tools
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "The weather in London is sunny, 20°C",
            "intermediate_steps": []
        }
        mock_executor_class.return_value = mock_executor
        
        # Create agent and test query
        agent = WeatherAgent(config=config)
        result = agent.query("What's the weather in London?")
        
        # Verify response
        assert result["success"] is True
        assert result["query"] == "What's the weather in London?"
        assert "sunny" in result["response"]
        assert "timestamp" in result
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    def test_agent_query_failure(self, mock_create_tools, mock_llm, config):
        """Test weather query failure handling."""
        # Mock LLM to raise an exception
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.side_effect = Exception("API Error")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock()]
        mock_create_tools.return_value = mock_tools
        
        # This will fail during agent creation due to LLM test
        with pytest.raises(Exception):
            WeatherAgent(config=config)
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools') 
    @patch('agents.weather_agent.create_openai_tools_agent')
    @patch('agents.weather_agent.AgentExecutor')
    def test_agent_memory_management(self, mock_executor_class, mock_create_agent, mock_create_tools, mock_llm, config):
        """Test agent memory management."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="Test response")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock()]
        mock_create_tools.return_value = mock_tools
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        
        # Create agent
        agent = WeatherAgent(config=config)
        
        # Test memory operations
        agent.reset_conversation()
        history = agent.get_conversation_history()
        assert isinstance(history, list)
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    @patch('agents.weather_agent.create_openai_tools_agent')
    @patch('agents.weather_agent.AgentExecutor')
    def test_agent_health_check(self, mock_executor_class, mock_create_agent, mock_create_tools, mock_llm, config):
        """Test agent health check."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="Healthy")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock(), Mock(), Mock(), Mock()]  # 4 tools
        mock_create_tools.return_value = mock_tools
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        
        # Create agent and test health check
        agent = WeatherAgent(config=config)
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert health["components"]["llm"] == "healthy"
        assert health["components"]["tools"] == "healthy"
        assert health["components"]["memory"] == "healthy"
        assert "metrics" in health
        assert "timestamp" in health
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    def test_agent_health_check_failure(self, mock_create_tools, mock_llm, config):
        """Test agent health check with failures."""
        # Mock LLM to fail health check
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.side_effect = Exception("LLM Error")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock()]
        mock_create_tools.return_value = mock_tools
        
        # This will fail during initialization
        with pytest.raises(Exception):
            WeatherAgent(config=config)
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    @patch('agents.weather_agent.create_openai_tools_agent')
    @patch('agents.weather_agent.AgentExecutor')
    def test_agent_metrics(self, mock_executor_class, mock_create_agent, mock_create_tools, mock_llm, config):
        """Test agent performance metrics."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="Test response")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock()]
        mock_create_tools.return_value = mock_tools
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        
        # Create agent
        agent = WeatherAgent(config=config)
        
        # Test metrics
        metrics = agent.get_metrics()
        assert "total_requests" in metrics
        assert "error_count" in metrics
        assert "avg_response_time" in metrics


class TestWeatherAgentFactory:
    """Test weather agent factory function."""
    
    @patch('agents.weather_agent.WeatherAgent')
    def test_create_weather_agent(self, mock_agent_class):
        """Test creating weather agent with factory function."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Test with no config
        agent = create_weather_agent()
        mock_agent_class.assert_called_once_with(config=None)
        
        # Test with config
        config = Mock()
        agent = create_weather_agent(config=config)
        assert mock_agent_class.call_count == 2


class TestIntegrationScenarios:
    """Integration tests for realistic weather agent scenarios."""
    
    @pytest.fixture
    def config(self):
        return WeatherAgentConfig(
            openai_api_key="test_openai_key",
            openweather_api_key="test_weather_key",
            verbose=False
        )
    
    @patch('agents.weather_agent.ChatOpenAI')
    @patch('tools.weather_tools.create_weather_tools')
    @patch('agents.weather_agent.create_openai_tools_agent')
    @patch('agents.weather_agent.AgentExecutor')
    def test_complete_weather_conversation_flow(self, mock_executor_class, mock_create_agent, mock_create_tools, mock_llm, config):
        """Test complete weather conversation workflow."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="Test response")
        mock_llm.return_value = mock_llm_instance
        
        mock_tools = [Mock() for _ in range(4)]
        mock_create_tools.return_value = mock_tools
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        # Mock different responses for different queries
        mock_executor = Mock()
        def mock_invoke(inputs):
            query = inputs["input"].lower()
            if "london" in query:
                return {"output": "Weather in London: 15°C, cloudy", "intermediate_steps": []}
            elif "forecast" in query:
                return {"output": "5-day forecast for London: Partly cloudy", "intermediate_steps": []}
            else:
                return {"output": "I can help with weather information", "intermediate_steps": []}
        
        mock_executor.invoke = mock_invoke
        mock_executor_class.return_value = mock_executor
        
        # Create agent and test conversation flow
        agent = WeatherAgent(config=config)
        
        # Test current weather query
        result1 = agent.query("What's the weather in London?")
        assert result1["success"] is True
        assert "London" in result1["response"]
        
        # Test forecast query
        result2 = agent.query("Can you give me a 5-day forecast for London?")
        assert result2["success"] is True
        assert "forecast" in result2["response"]
        
        # Verify conversation context is maintained
        history = agent.get_conversation_history()
        assert isinstance(history, list)


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v"])