"""
Custom Tool Implementation Examples

This file demonstrates how to create custom tools for LangChain agents.
It shows best practices for tool design, input validation, error handling,
and integration with agent workflows.

Key Patterns Demonstrated:
- BaseTool implementation with Pydantic schemas
- Input validation and type safety
- Error handling and graceful failures
- Tool composition and chaining
- Async tool implementations
- Tool testing patterns
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from pydantic import BaseModel, Field, validator

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# Input schemas for tools

class TextAnalysisInput(BaseModel):
    """Input schema for text analysis tool."""
    text: str = Field(description="Text to analyze")
    analysis_type: str = Field(
        description="Type of analysis: 'sentiment', 'readability', 'keywords', 'summary'"
    )
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = ['sentiment', 'readability', 'keywords', 'summary']
        if v not in allowed_types:
            raise ValueError(f"analysis_type must be one of {allowed_types}")
        return v

class FileOperationInput(BaseModel):
    """Input schema for file operations."""
    operation: str = Field(description="Operation: 'read', 'write', 'delete', 'list'")
    file_path: Optional[str] = Field(None, description="Path to file (required for read/write/delete)")
    content: Optional[str] = Field(None, description="Content to write (required for write)")
    
    @validator('operation')
    def validate_operation(cls, v):
        allowed_ops = ['read', 'write', 'delete', 'list']
        if v not in allowed_ops:
            raise ValueError(f"operation must be one of {allowed_ops}")
        return v

class DataProcessingInput(BaseModel):
    """Input schema for data processing tool."""
    data: str = Field(description="JSON string or CSV data to process")
    operation: str = Field(description="Operation: 'parse', 'filter', 'aggregate', 'transform'")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Operation parameters")

# Custom Tool Implementations

class TextAnalysisTool(BaseTool):
    """Tool for analyzing text content."""
    
    name: str = "text_analyzer"
    description: str = "Analyze text for sentiment, readability, keywords, or generate summaries"
    args_schema: Type[BaseModel] = TextAnalysisInput
    
    def _run(
        self, 
        text: str, 
        analysis_type: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute text analysis."""
        try:
            if analysis_type == "sentiment":
                return self._analyze_sentiment(text)
            elif analysis_type == "readability":
                return self._analyze_readability(text)
            elif analysis_type == "keywords":
                return self._extract_keywords(text)
            elif analysis_type == "summary":
                return self._generate_summary(text)
            else:
                return f"Error: Unknown analysis type '{analysis_type}'"
                
        except Exception as e:
            return f"Error in text analysis: {str(e)}"
    
    async def _arun(
        self,
        text: str,
        analysis_type: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of text analysis."""
        # For this example, we'll just call the sync version
        # In a real implementation, you'd use async operations
        return self._run(text, analysis_type, run_manager)
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis (mock implementation)."""
        # In a real implementation, you'd use a proper sentiment analysis library
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return f"Sentiment Analysis: {sentiment} (positive: {positive_count}, negative: {negative_count})"
    
    def _analyze_readability(self, text: str) -> str:
        """Simple readability analysis."""
        words = text.split()
        sentences = text.split('.')
        
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = sum(len(word) for word in words) / max(len(words), 1)
        
        # Simple readability score
        if avg_words_per_sentence < 15 and avg_chars_per_word < 5:
            level = "Easy"
        elif avg_words_per_sentence < 20 and avg_chars_per_word < 6:
            level = "Medium"
        else:
            level = "Hard"
        
        return f"Readability: {level} (avg {avg_words_per_sentence:.1f} words/sentence, {avg_chars_per_word:.1f} chars/word)"
    
    def _extract_keywords(self, text: str) -> str:
        """Simple keyword extraction."""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        words = [word.lower().strip('.,!?";') for word in text.split()]
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return f"Top Keywords: {', '.join([f'{word} ({count})' for word, count in top_keywords])}"
    
    def _generate_summary(self, text: str) -> str:
        """Simple text summarization."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Simple extractive summarization - take first and longest sentences
        if len(sentences) <= 2:
            return f"Summary: {text}"
        
        # Get the longest sentence as likely most informative
        longest_sentence = max(sentences, key=len)
        summary_sentences = [sentences[0], longest_sentence] if sentences[0] != longest_sentence else [sentences[0]]
        
        return f"Summary: {'. '.join(summary_sentences)}."

class FileOperationTool(BaseTool):
    """Tool for file system operations."""
    
    name: str = "file_operations"
    description: str = "Perform file operations: read, write, delete, or list files"
    args_schema: Type[BaseModel] = FileOperationInput
    
    def _run(
        self,
        operation: str,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute file operations."""
        try:
            if operation == "list":
                return self._list_files()
            elif operation == "read":
                if not file_path:
                    return "Error: file_path required for read operation"
                return self._read_file(file_path)
            elif operation == "write":
                if not file_path or content is None:
                    return "Error: file_path and content required for write operation"
                return self._write_file(file_path, content)
            elif operation == "delete":
                if not file_path:
                    return "Error: file_path required for delete operation"
                return self._delete_file(file_path)
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error in file operation: {str(e)}"
    
    async def _arun(
        self,
        operation: str,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of file operations."""
        return self._run(operation, file_path, content, run_manager)
    
    def _list_files(self) -> str:
        """List files in current directory."""
        import os
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        
        return f"Files: {', '.join(files[:10])}{'...' if len(files) > 10 else ''}\nDirectories: {', '.join(dirs[:5])}{'...' if len(dirs) > 5 else ''}"
    
    def _read_file(self, file_path: str) -> str:
        """Read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Truncate if too long
            if len(content) > 1000:
                return f"File content (first 1000 chars): {content[:1000]}...\n[Total length: {len(content)} characters]"
            else:
                return f"File content: {content}"
                
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to '{file_path}'"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _delete_file(self, file_path: str) -> str:
        """Delete a file."""
        try:
            import os
            os.remove(file_path)
            return f"Successfully deleted '{file_path}'"
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except Exception as e:
            return f"Error deleting file: {str(e)}"

class DataProcessingTool(BaseTool):
    """Tool for processing structured data."""
    
    name: str = "data_processor"
    description: str = "Process JSON or CSV data with operations like parse, filter, aggregate, transform"
    args_schema: Type[BaseModel] = DataProcessingInput
    
    def _run(
        self,
        data: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute data processing operations."""
        try:
            if operation == "parse":
                return self._parse_data(data)
            elif operation == "filter":
                return self._filter_data(data, parameters or {})
            elif operation == "aggregate":
                return self._aggregate_data(data, parameters or {})
            elif operation == "transform":
                return self._transform_data(data, parameters or {})
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error in data processing: {str(e)}"
    
    async def _arun(
        self,
        data: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of data processing."""
        return self._run(data, operation, parameters, run_manager)
    
    def _parse_data(self, data: str) -> str:
        """Parse and analyze data structure."""
        try:
            # Try parsing as JSON first
            try:
                parsed = json.loads(data)
                data_type = "JSON"
                
                if isinstance(parsed, list):
                    structure = f"Array with {len(parsed)} items"
                    if parsed:
                        item_type = type(parsed[0]).__name__
                        if isinstance(parsed[0], dict):
                            keys = list(parsed[0].keys())
                            structure += f", each item has keys: {keys}"
                        else:
                            structure += f", items are of type: {item_type}"
                elif isinstance(parsed, dict):
                    structure = f"Object with keys: {list(parsed.keys())}"
                else:
                    structure = f"Single value of type: {type(parsed).__name__}"
                    
            except json.JSONDecodeError:
                # Try parsing as CSV
                lines = data.strip().split('\n')
                if len(lines) > 1 and ',' in lines[0]:
                    data_type = "CSV"
                    headers = lines[0].split(',')
                    structure = f"CSV with {len(lines)} rows and columns: {headers}"
                else:
                    data_type = "Text"
                    structure = f"Plain text with {len(lines)} lines"
            
            return f"Data Type: {data_type}\nStructure: {structure}"
            
        except Exception as e:
            return f"Error parsing data: {str(e)}"
    
    def _filter_data(self, data: str, parameters: Dict[str, Any]) -> str:
        """Filter data based on parameters."""
        try:
            parsed = json.loads(data)
            
            if not isinstance(parsed, list):
                return "Error: Filter operation requires array data"
            
            filter_key = parameters.get('key')
            filter_value = parameters.get('value')
            
            if not filter_key:
                return "Error: 'key' parameter required for filtering"
            
            filtered = []
            for item in parsed:
                if isinstance(item, dict) and item.get(filter_key) == filter_value:
                    filtered.append(item)
            
            return f"Filtered {len(parsed)} items to {len(filtered)} items where {filter_key} = {filter_value}"
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON data for filtering"
        except Exception as e:
            return f"Error filtering data: {str(e)}"
    
    def _aggregate_data(self, data: str, parameters: Dict[str, Any]) -> str:
        """Aggregate data based on parameters."""
        try:
            parsed = json.loads(data)
            
            if not isinstance(parsed, list):
                return "Error: Aggregate operation requires array data"
            
            group_by = parameters.get('group_by')
            agg_field = parameters.get('field')
            agg_func = parameters.get('function', 'count')
            
            if agg_func == 'count':
                if group_by:
                    counts = {}
                    for item in parsed:
                        if isinstance(item, dict):
                            key = item.get(group_by, 'unknown')
                            counts[key] = counts.get(key, 0) + 1
                    return f"Count by {group_by}: {counts}"
                else:
                    return f"Total count: {len(parsed)}"
            
            elif agg_func in ['sum', 'avg', 'min', 'max'] and agg_field:
                values = []
                for item in parsed:
                    if isinstance(item, dict) and agg_field in item:
                        try:
                            values.append(float(item[agg_field]))
                        except (ValueError, TypeError):
                            continue
                
                if not values:
                    return f"No numeric values found for field '{agg_field}'"
                
                if agg_func == 'sum':
                    result = sum(values)
                elif agg_func == 'avg':
                    result = sum(values) / len(values)
                elif agg_func == 'min':
                    result = min(values)
                elif agg_func == 'max':
                    result = max(values)
                
                return f"{agg_func.upper()} of {agg_field}: {result}"
            
            return "Error: Invalid aggregation parameters"
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON data for aggregation"
        except Exception as e:
            return f"Error aggregating data: {str(e)}"
    
    def _transform_data(self, data: str, parameters: Dict[str, Any]) -> str:
        """Transform data based on parameters."""
        try:
            parsed = json.loads(data)
            
            transform_type = parameters.get('type', 'keys')
            
            if transform_type == 'keys' and isinstance(parsed, list):
                # Extract unique keys from objects
                all_keys = set()
                for item in parsed:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                return f"Unique keys in data: {sorted(list(all_keys))}"
            
            elif transform_type == 'flatten' and isinstance(parsed, list):
                # Flatten nested structures
                flattened_count = 0
                for item in parsed:
                    if isinstance(item, (list, dict)):
                        flattened_count += 1
                return f"Found {flattened_count} items that could be flattened"
            
            elif transform_type == 'schema':
                # Generate schema
                def get_schema(obj):
                    if isinstance(obj, dict):
                        return {k: get_schema(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [get_schema(obj[0])] if obj else []
                    else:
                        return type(obj).__name__
                
                schema = get_schema(parsed)
                return f"Data schema: {json.dumps(schema, indent=2)}"
            
            return f"Transform type '{transform_type}' not implemented"
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON data for transformation"
        except Exception as e:
            return f"Error transforming data: {str(e)}"

# Tool testing utilities

def test_custom_tools():
    """Test all custom tools with sample data."""
    
    print("\n" + "="*50)
    print("CUSTOM TOOLS TESTING")
    print("="*50)
    
    # Test TextAnalysisTool
    print("\n1. Testing TextAnalysisTool:")
    text_tool = TextAnalysisTool()
    
    sample_text = "This is a great example of wonderful text analysis. The implementation is fantastic and works amazingly well!"
    
    analyses = ['sentiment', 'readability', 'keywords', 'summary']
    for analysis in analyses:
        result = text_tool._run(sample_text, analysis)
        print(f"   {analysis}: {result}")
    
    # Test FileOperationTool
    print("\n2. Testing FileOperationTool:")
    file_tool = FileOperationTool()
    
    # Create a test file
    test_content = "This is a test file created by the FileOperationTool."
    write_result = file_tool._run("write", "test_tool_output.txt", test_content)
    print(f"   Write: {write_result}")
    
    # Read the file
    read_result = file_tool._run("read", "test_tool_output.txt")
    print(f"   Read: {read_result}")
    
    # List files
    list_result = file_tool._run("list")
    print(f"   List: {list_result}")
    
    # Test DataProcessingTool
    print("\n3. Testing DataProcessingTool:")
    data_tool = DataProcessingTool()
    
    sample_data = json.dumps([
        {"name": "Alice", "age": 30, "department": "Engineering"},
        {"name": "Bob", "age": 25, "department": "Marketing"},
        {"name": "Charlie", "age": 35, "department": "Engineering"}
    ])
    
    # Parse data
    parse_result = data_tool._run(sample_data, "parse")
    print(f"   Parse: {parse_result}")
    
    # Filter data
    filter_result = data_tool._run(sample_data, "filter", {"key": "department", "value": "Engineering"})
    print(f"   Filter: {filter_result}")
    
    # Aggregate data
    agg_result = data_tool._run(sample_data, "aggregate", {"group_by": "department"})
    print(f"   Aggregate: {agg_result}")
    
    # Transform data
    transform_result = data_tool._run(sample_data, "transform", {"type": "keys"})
    print(f"   Transform: {transform_result}")

def create_tool_usage_example():
    """Example of using custom tools in an agent."""
    
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    print("\n" + "="*50)
    print("TOOLS IN AGENT EXAMPLE")
    print("="*50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY required for agent example")
        return
    
    # Create LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = [
        TextAnalysisTool(),
        FileOperationTool(),
        DataProcessingTool()
    ]
    
    # Create agent prompt
    prompt = PromptTemplate(
        template="""You are a helpful assistant with access to powerful tools for text analysis, file operations, and data processing.

Tools available:
{tools}

Use this format:
Question: {input}
Thought: I need to think about which tool to use
Action: {tool_names}
Action Input: the input for the action
Observation: the result of the action
Thought: I can now respond
Final Answer: my response to the user

Begin!
Question: {input}
{agent_scratchpad}""",
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3
    )
    
    # Test queries
    test_queries = [
        "Analyze the sentiment of this text: 'I love working with LangChain tools!'",
        "Create a file called 'tool_test.txt' with the content 'Custom tools are powerful!'",
        "Parse this JSON data and tell me about its structure: [{'id': 1, 'name': 'test'}]"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = agent_executor.invoke({"input": query})
            print(f"Result: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    """
    Test and demonstrate custom tools.
    
    This demonstrates:
    1. Custom tool implementation with BaseTool
    2. Input validation with Pydantic
    3. Error handling in tools
    4. Tool testing patterns
    5. Integration with agents
    """
    
    try:
        # Test individual tools
        test_custom_tools()
        
        # Test tools in agent
        create_tool_usage_example()
        
    except KeyboardInterrupt:
        print("\nTool testing interrupted by user")
        
    except Exception as e:
        print(f"Unexpected error in tool testing: {e}")
        raise
    
    finally:
        print("\nCustom tool demonstration completed")