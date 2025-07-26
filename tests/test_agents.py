"""
Test patterns for LangChain agents.

This file demonstrates comprehensive testing approaches for LangChain agents,
including tool testing, agent reasoning, and integration scenarios.

Key Testing Patterns:
- Agent creation and configuration testing
- Tool execution and integration testing
- Agent reasoning and decision-making validation
- Memory integration with agents
- Error handling in agent workflows
- Performance testing for agent operations
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List
import json
import time

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, Tool
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Import the components we're testing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agent_chain import ResearchAgent, AgentConfig, CalculatorTool, WebSearchTool, FileManagerTool
from tools.custom_tool import TextAnalysisTool, DataProcessingTool
from tools.web_search_tool import WebSearchTool as WebSearchToolAdvanced

class TestToolFunctionality:
    """Test individual tool functionality."""
    
    def test_calculator_tool_basic_operations(self):
        """Test calculator tool with basic operations."""
        calc_tool = CalculatorTool()
        
        # Test addition
        result = calc_tool._run("2 + 3")
        assert "5" in result
        
        # Test multiplication
        result = calc_tool._run("4 * 6")
        assert "24" in result
        
        # Test division
        result = calc_tool._run("10 / 2")
        assert "5" in result
    
    def test_calculator_tool_error_handling(self):
        """Test calculator tool error handling."""
        calc_tool = CalculatorTool()
        
        # Test division by zero
        result = calc_tool._run("10 / 0")
        assert "Error" in result
        assert "Division by zero" in result
        
        # Test invalid characters
        result = calc_tool._run("2 + evil_function()")
        assert "Error" in result
        assert "Invalid characters" in result
        
        # Test malformed expression
        result = calc_tool._run("2 + + 3")
        assert "Error" in result
    
    def test_web_search_tool_mock_results(self):
        """Test web search tool with mock results."""
        web_tool = WebSearchTool()
        
        # Test general search
        result = web_tool._run("python programming")
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Test langchain search
        result = web_tool._run("langchain")
        assert "LangChain" in result or "langchain" in result.lower()
    
    def test_file_manager_tool_operations(self):
        """Test file manager tool operations."""
        file_tool = FileManagerTool()
        
        # Test file listing
        result = file_tool._run("list")
        assert "Files" in result or "directories" in result.lower()
        
        # Test file writing and reading
        write_result = file_tool._run("write", "test_file.txt", "Test content")
        assert "Successfully wrote" in write_result
        
        read_result = file_tool._run("read", "test_file.txt")
        assert "Test content" in read_result
        
        # Cleanup
        import os
        try:
            os.remove("test_file.txt")
        except FileNotFoundError:
            pass
    
    def test_text_analysis_tool_sentiment(self):
        """Test text analysis tool sentiment analysis."""
        text_tool = TextAnalysisTool()
        
        # Test positive sentiment
        result = text_tool._run("This is a great and wonderful day!", "sentiment")
        assert "positive" in result.lower()
        
        # Test negative sentiment
        result = text_tool._run("This is terrible and awful.", "sentiment")
        assert "negative" in result.lower()
    
    def test_text_analysis_tool_keywords(self):
        """Test text analysis tool keyword extraction."""
        text_tool = TextAnalysisTool()
        
        result = text_tool._run("Python programming language is excellent for data science", "keywords")
        
        assert "Keywords" in result
        assert any(word in result.lower() for word in ["python", "programming", "language", "data", "science"])
    
    def test_data_processing_tool_json_parsing(self):
        """Test data processing tool with JSON data."""
        data_tool = DataProcessingTool()
        
        test_data = json.dumps([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ])
        
        result = data_tool._run(test_data, "parse")
        
        assert "JSON" in result
        assert "Array" in result
        assert "2 items" in result

class TestAgentCreation:
    """Test agent creation and configuration."""
    
    def setup_method(self):
        """Setup for agent tests."""
        self.mock_llm = Mock()
        self.mock_llm.predict.return_value = "Mock LLM response"
        
    @patch('agent_chain.os.getenv')
    @patch('agent_chain.ChatOpenAI')
    def test_research_agent_initialization(self, mock_llm_class, mock_getenv):
        """Test research agent initialization."""
        mock_getenv.return_value = "test-api-key"
        mock_llm_class.return_value = self.mock_llm
        
        config = AgentConfig(temperature=0.5, max_iterations=3)
        agent = ResearchAgent(config)
        
        assert agent.config.temperature == 0.5
        assert agent.config.max_iterations == 3
        assert len(agent.tools) > 0
        assert agent.agent_executor is not None
    
    @patch('agent_chain.os.getenv')
    @patch('agent_chain.ChatOpenAI')
    def test_agent_tools_creation(self, mock_llm_class, mock_getenv):
        """Test that agent creates tools correctly."""
        mock_getenv.return_value = "test-api-key"
        mock_llm_class.return_value = self.mock_llm
        
        agent = ResearchAgent()
        
        tool_names = [tool.name for tool in agent.tools]
        
        assert "calculator" in tool_names
        assert "web_search" in tool_names
        assert "file_manager" in tool_names
        assert "string_operations" in tool_names
    
    def test_agent_prompt_creation(self):
        """Test agent prompt template creation."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=self.mock_llm):
            
            agent = ResearchAgent()
            prompt = agent._create_agent_prompt()
            
            assert isinstance(prompt, PromptTemplate)
            assert "tools" in prompt.input_variables
            assert "tool_names" in prompt.input_variables
            assert "input" in prompt.input_variables

class TestAgentExecution:
    """Test agent execution and reasoning."""
    
    def setup_method(self):
        """Setup for execution tests."""
        self.mock_llm = Mock()
        
        # Mock agent responses for different scenarios
        self.mock_responses = {
            "calculation": """
            Thought: I need to calculate 25 * 4 + 10
            Action: calculator
            Action Input: 25 * 4 + 10
            Observation: The result of 25 * 4 + 10 is 110
            Thought: I now know the answer
            Final Answer: The result is 110
            """,
            "search": """
            Thought: I need to search for information about Python
            Action: web_search
            Action Input: Python programming language
            Observation: Python is a high-level programming language...
            Thought: I have the information needed
            Final Answer: Python is a high-level programming language known for its simplicity.
            """
        }
    
    @patch('agent_chain.os.getenv')
    @patch('agent_chain.ChatOpenAI')
    def test_agent_calculator_task(self, mock_llm_class, mock_getenv):
        """Test agent performing calculation task."""
        mock_getenv.return_value = "test-api-key"
        mock_llm_class.return_value = self.mock_llm
        
        # Mock the agent executor's invoke method
        with patch.object(ResearchAgent, '_create_agent') as mock_create_agent:
            mock_executor = Mock()
            mock_executor.invoke.return_value = {
                "output": "The result is 110",
                "intermediate_steps": [
                    (AgentAction(tool="calculator", tool_input="25 * 4 + 10", log=""), "110")
                ]
            }
            mock_create_agent.return_value = mock_executor
            
            agent = ResearchAgent()
            result = agent.run("Calculate 25 * 4 + 10")
            
            assert result["success"] is True
            assert "110" in result["answer"]
            assert len(result["intermediate_steps"]) > 0
    
    @patch('agent_chain.os.getenv')
    @patch('agent_chain.ChatOpenAI')
    def test_agent_search_task(self, mock_llm_class, mock_getenv):
        """Test agent performing search task."""
        mock_getenv.return_value = "test-api-key"
        mock_llm_class.return_value = self.mock_llm
        
        with patch.object(ResearchAgent, '_create_agent') as mock_create_agent:
            mock_executor = Mock()
            mock_executor.invoke.return_value = {
                "output": "Python is a programming language",
                "intermediate_steps": [
                    (AgentAction(tool="web_search", tool_input="Python", log=""), "Python info")
                ]
            }
            mock_create_agent.return_value = mock_executor
            
            agent = ResearchAgent()
            result = agent.run("Tell me about Python programming")
            
            assert result["success"] is True
            assert "Python" in result["answer"]
    
    @patch('agent_chain.os.getenv')
    @patch('agent_chain.ChatOpenAI')
    def test_agent_error_handling(self, mock_llm_class, mock_getenv):
        """Test agent error handling."""
        mock_getenv.return_value = "test-api-key"
        mock_llm_class.return_value = self.mock_llm
        
        with patch.object(ResearchAgent, '_create_agent') as mock_create_agent:
            mock_executor = Mock()
            mock_executor.invoke.side_effect = Exception("Test error")
            mock_create_agent.return_value = mock_executor
            
            agent = ResearchAgent()
            result = agent.run("This should fail")
            
            assert result["success"] is False
            assert "Test error" in result["error"]
    
    def test_agent_memory_reset(self):
        """Test agent memory reset functionality."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=self.mock_llm):
            
            agent = ResearchAgent()
            
            # Add some memory
            agent.memory.save_context({"input": "Hello"}, {"output": "Hi there"})
            
            # Check memory has content
            memory_vars = agent.memory.load_memory_variables({})
            assert len(memory_vars.get("chat_history", [])) > 0
            
            # Reset memory
            agent.reset_memory()
            
            # Check memory is cleared
            memory_vars = agent.memory.load_memory_variables({})
            assert len(memory_vars.get("chat_history", [])) == 0

class TestAgentIntegration:
    """Test agent integration with various components."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.mock_llm = Mock()
    
    @patch('agent_chain.os.getenv')
    @patch('agent_chain.ChatOpenAI')
    def test_agent_with_custom_tools(self, mock_llm_class, mock_getenv):
        """Test agent with custom tool integration."""
        mock_getenv.return_value = "test-api-key"
        mock_llm_class.return_value = self.mock_llm
        
        # Create custom tool
        def custom_operation(input_str: str) -> str:
            return f"Custom result for: {input_str}"
        
        custom_tool = Tool(
            name="custom_tool",
            description="A custom tool for testing",
            func=custom_operation
        )
        
        # Test tool works independently
        result = custom_tool.run("test input")
        assert "Custom result for: test input" == result
    
    def test_agent_tool_selection_logic(self):
        """Test agent tool selection based on query."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=self.mock_llm):
            
            agent = ResearchAgent()
            
            # Verify all expected tools are available
            tool_names = [tool.name for tool in agent.tools]
            
            # Mathematical queries should have calculator available
            assert "calculator" in tool_names
            
            # Search queries should have web_search available
            assert "web_search" in tool_names
            
            # File operations should have file_manager available
            assert "file_manager" in tool_names
    
    def test_agent_chain_of_reasoning(self):
        """Test agent's chain of reasoning through multiple steps."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=self.mock_llm):
            
            agent = ResearchAgent()
            
            # Mock a multi-step reasoning process
            with patch.object(agent.agent_executor, 'invoke') as mock_invoke:
                mock_invoke.return_value = {
                    "output": "Final answer after multiple steps",
                    "intermediate_steps": [
                        (AgentAction(tool="web_search", tool_input="query1", log=""), "result1"),
                        (AgentAction(tool="calculator", tool_input="calc1", log=""), "result2"),
                        (AgentAction(tool="web_search", tool_input="query2", log=""), "result3")
                    ]
                }
                
                result = agent.run("Complex multi-step query")
                
                assert result["success"] is True
                assert len(result["intermediate_steps"]) == 3
                assert result["intermediate_steps"][0][0].tool == "web_search"
                assert result["intermediate_steps"][1][0].tool == "calculator"

class TestAgentPerformance:
    """Test agent performance and optimization."""
    
    def test_agent_execution_time_tracking(self):
        """Test that agent tracks execution time."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent()
            
            # Mock agent executor with delay
            with patch.object(agent.agent_executor, 'invoke') as mock_invoke:
                def slow_invoke(inputs):
                    time.sleep(0.1)  # 100ms delay
                    return {"output": "Response", "intermediate_steps": []}
                
                mock_invoke.side_effect = slow_invoke
                
                start_time = time.time()
                result = agent.run("test query")
                total_time = time.time() - start_time
                
                assert result["success"] is True
                assert total_time >= 0.1
    
    def test_agent_iteration_limits(self):
        """Test agent respects iteration limits."""
        config = AgentConfig(max_iterations=2)
        
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent(config)
            
            # Verify configuration is applied
            assert agent.config.max_iterations == 2
    
    def test_agent_timeout_handling(self):
        """Test agent timeout configuration."""
        config = AgentConfig(max_execution_time=30)
        
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent(config)
            
            # Verify timeout configuration
            assert agent.config.max_execution_time == 30

class TestToolChaining:
    """Test tool chaining and composition."""
    
    def test_sequential_tool_usage(self):
        """Test using multiple tools in sequence."""
        # Create individual tools
        calc_tool = CalculatorTool()
        text_tool = TextAnalysisTool()
        
        # Test sequential usage
        calc_result = calc_tool._run("10 + 5")
        assert "15" in calc_result
        
        # Use calculation result in text analysis
        text_result = text_tool._run(f"The calculation result is {calc_result}", "sentiment")
        assert isinstance(text_result, str)
    
    def test_tool_result_formatting(self):
        """Test that tool results are properly formatted for chaining."""
        tools = [
            CalculatorTool(),
            WebSearchTool(),
            FileManagerTool()
        ]
        
        for tool in tools:
            # Each tool should return a string
            if tool.name == "calculator":
                result = tool._run("2 + 2")
            elif tool.name == "web_search":
                result = tool._run("test query")
            elif tool.name == "file_manager":
                result = tool._run("list")
            
            assert isinstance(result, str)
            assert len(result) > 0

class TestAgentErrorRecovery:
    """Test agent error recovery and resilience."""
    
    def test_tool_failure_recovery(self):
        """Test agent recovery from tool failures."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent()
            
            # Mock a tool that fails initially but succeeds on retry
            failing_tool = Mock()
            failing_tool.name = "failing_tool"
            failing_tool.description = "A tool that fails sometimes"
            failing_tool._run.side_effect = [Exception("Tool failed"), "Success on retry"]
            
            # Test that tool failure is handled gracefully
            try:
                result1 = failing_tool._run("test")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Tool failed" in str(e)
            
            # Second call should succeed
            result2 = failing_tool._run("test")
            assert result2 == "Success on retry"
    
    def test_invalid_action_handling(self):
        """Test handling of invalid agent actions."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent()
            
            # Mock an invalid action scenario
            with patch.object(agent.agent_executor, 'invoke') as mock_invoke:
                mock_invoke.side_effect = ValueError("Invalid action format")
                
                result = agent.run("Invalid query that causes parsing error")
                
                assert result["success"] is False
                assert "error" in result
    
    def test_agent_graceful_degradation(self):
        """Test agent graceful degradation when tools are unavailable."""
        # Create agent with no tools
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            config = AgentConfig()
            
            # Mock agent creation to use no tools
            with patch.object(ResearchAgent, '_create_tools', return_value=[]):
                agent = ResearchAgent(config)
                
                assert len(agent.tools) == 0

class TestAdvancedAgentPatterns:
    """Test advanced agent patterns and configurations."""
    
    def test_agent_with_custom_prompt(self):
        """Test agent with custom prompt template."""
        custom_prompt = PromptTemplate(
            template="""You are a specialized math assistant.

Available tools: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}

Solve this step by step.""",
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        
        assert "specialized math assistant" in custom_prompt.template
        assert all(var in custom_prompt.input_variables 
                  for var in ["tools", "tool_names", "input", "agent_scratchpad"])
    
    def test_agent_state_persistence(self):
        """Test agent state persistence across interactions."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent()
            
            # Simulate multiple interactions
            with patch.object(agent.agent_executor, 'invoke') as mock_invoke:
                mock_invoke.return_value = {"output": "Response", "intermediate_steps": []}
                
                # First interaction
                result1 = agent.run("Remember my name is Alice")
                assert result1["success"] is True
                
                # Second interaction should have access to memory
                result2 = agent.run("What's my name?")
                assert result2["success"] is True
    
    def test_agent_configuration_validation(self):
        """Test agent configuration validation."""
        # Test valid configuration
        valid_config = AgentConfig(
            temperature=0.5,
            max_iterations=5,
            max_execution_time=60
        )
        
        assert valid_config.temperature == 0.5
        assert valid_config.max_iterations == 5
        assert valid_config.max_execution_time == 60
        
        # Test default values
        default_config = AgentConfig()
        assert default_config.temperature == 0.1
        assert default_config.max_iterations == 5

class TestAsyncAgentOperations:
    """Test asynchronous agent operations."""
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test asynchronous tool execution."""
        
        class AsyncMockTool(BaseTool):
            name = "async_mock"
            description = "Async mock tool"
            
            def _run(self, query: str) -> str:
                return f"Sync result for {query}"
            
            async def _arun(self, query: str) -> str:
                await asyncio.sleep(0.1)  # Simulate async work
                return f"Async result for {query}"
        
        tool = AsyncMockTool()
        
        # Test sync execution
        sync_result = tool._run("test")
        assert sync_result == "Sync result for test"
        
        # Test async execution
        async_result = await tool._arun("test")
        assert async_result == "Async result for test"
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self):
        """Test concurrent agent operations."""
        
        async def mock_agent_run(agent_id: str, query: str):
            await asyncio.sleep(0.1)  # Simulate agent work
            return {"agent_id": agent_id, "result": f"Response to {query}"}
        
        # Test concurrent execution
        tasks = [
            mock_agent_run("agent1", "query1"),
            mock_agent_run("agent2", "query2"),
            mock_agent_run("agent3", "query3")
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all("result" in result for result in results)
        assert all("agent_id" in result for result in results)

class TestAgentDebugging:
    """Test agent debugging and introspection capabilities."""
    
    def test_agent_step_by_step_execution(self):
        """Test agent step-by-step execution tracking."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent()
            
            # Mock detailed execution tracking
            with patch.object(agent.agent_executor, 'invoke') as mock_invoke:
                mock_invoke.return_value = {
                    "output": "Final answer",
                    "intermediate_steps": [
                        (AgentAction(tool="calculator", tool_input="2+2", log="Calculating"), "4"),
                        (AgentAction(tool="web_search", tool_input="test", log="Searching"), "results")
                    ]
                }
                
                result = agent.run("Test query")
                
                # Verify we can track each step
                assert len(result["intermediate_steps"]) == 2
                assert result["intermediate_steps"][0][0].tool == "calculator"
                assert result["intermediate_steps"][0][1] == "4"
                assert result["intermediate_steps"][1][0].tool == "web_search"
    
    def test_agent_reasoning_trace(self):
        """Test agent reasoning trace capture."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            config = AgentConfig(verbose=True)
            agent = ResearchAgent(config)
            
            # Verify verbose mode is enabled
            assert agent.config.verbose is True
    
    def test_agent_error_diagnosis(self):
        """Test agent error diagnosis capabilities."""
        with patch('agent_chain.os.getenv', return_value="test-api-key"), \
             patch('agent_chain.ChatOpenAI', return_value=Mock()):
            
            agent = ResearchAgent()
            
            # Mock different types of errors
            error_scenarios = [
                ("Tool not found", ValueError("Unknown tool")),
                ("Parsing error", SyntaxError("Invalid format")),
                ("Timeout error", TimeoutError("Agent timeout"))
            ]
            
            for error_name, error in error_scenarios:
                with patch.object(agent.agent_executor, 'invoke', side_effect=error):
                    result = agent.run(f"Query that causes {error_name}")
                    
                    assert result["success"] is False
                    assert error_name.split()[0].lower() in result["error"].lower()

if __name__ == "__main__":
    """
    Run the agent tests.
    
    Usage:
        python test_agents.py
        pytest test_agents.py -v
        pytest test_agents.py::TestToolFunctionality::test_calculator_tool_basic_operations -v
    """
    
    # Run basic test discovery
    pytest.main([__file__, "-v"])