"""
Web Search Tool Implementation Examples

This file demonstrates how to create web search tools for LangChain agents.
It shows best practices for integrating with search APIs, handling results,
and providing structured search capabilities.

Key Patterns Demonstrated:
- Web search API integration
- Search result processing and filtering
- Rate limiting and error handling
- Multiple search provider support
- Search result caching
- Structured search responses
"""

import os
import logging
import json
import time
import requests
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Input schemas for search tools

class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Search query to execute")
    num_results: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    search_type: str = Field(default="general", description="Type of search: 'general', 'news', 'images', 'academic'")
    
    @validator('search_type')
    def validate_search_type(cls, v):
        allowed_types = ['general', 'news', 'images', 'academic']
        if v not in allowed_types:
            raise ValueError(f"search_type must be one of {allowed_types}")
        return v

class NewsSearchInput(BaseModel):
    """Input schema for news search tool."""
    query: str = Field(description="News search query")
    days_back: int = Field(default=7, description="How many days back to search", ge=1, le=30)
    language: str = Field(default="en", description="Language code (en, es, fr, etc.)")
    category: Optional[str] = Field(None, description="News category filter")

class AcademicSearchInput(BaseModel):
    """Input schema for academic search tool."""
    query: str = Field(description="Academic search query")
    year_from: Optional[int] = Field(None, description="Start year for publication date")
    year_to: Optional[int] = Field(None, description="End year for publication date")
    field_of_study: Optional[str] = Field(None, description="Field of study filter")

# Data classes for search results

@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: Optional[datetime] = None
    relevance_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    provider: str
    timestamp: datetime

# Search provider implementations

class SearchProviderBase:
    """Base class for search providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        self.cache = SearchCache()
    
    def search(self, query: str, **kwargs) -> SearchResponse:
        """Execute search query."""
        raise NotImplementedError
    
    def _build_search_url(self, query: str, **params) -> str:
        """Build search URL with parameters."""
        raise NotImplementedError
    
    def _process_results(self, raw_results: Dict[str, Any]) -> List[SearchResult]:
        """Process raw API results into SearchResult objects."""
        raise NotImplementedError

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.calls.append(now)

class SearchCache:
    """Simple in-memory cache for search results."""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache = {}
    
    def get(self, key: str) -> Optional[SearchResponse]:
        """Get cached result if available and not expired."""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.info(f"Cache hit for query: {key}")
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: SearchResponse):
        """Cache search result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
        logger.info(f"Cached result for query: {key}")

class DuckDuckGoSearchProvider(SearchProviderBase):
    """DuckDuckGo search provider (no API key required)."""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, num_results: int = 5, **kwargs) -> SearchResponse:
        """Execute DuckDuckGo search."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"ddg_{query}_{num_results}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            self.rate_limiter.wait_if_needed()
            
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            raw_results = response.json()
            results = self._process_results(raw_results, num_results)
            
            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=time.time() - start_time,
                provider="DuckDuckGo",
                timestamp=datetime.now()
            )
            
            # Cache the result
            self.cache.set(cache_key, search_response)
            
            return search_response
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                provider="DuckDuckGo",
                timestamp=datetime.now()
            )
    
    def _process_results(self, raw_results: Dict[str, Any], num_results: int) -> List[SearchResult]:
        """Process DuckDuckGo API results."""
        results = []
        
        # Process web results
        for item in raw_results.get('RelatedTopics', [])[:num_results]:
            if 'Text' in item and 'FirstURL' in item:
                result = SearchResult(
                    title=item.get('Text', '')[:100] + '...' if len(item.get('Text', '')) > 100 else item.get('Text', ''),
                    url=item.get('FirstURL', ''),
                    snippet=item.get('Text', ''),
                    source='DuckDuckGo',
                    metadata={'icon': item.get('Icon', {})}
                )
                results.append(result)
        
        # If no related topics, try abstract
        if not results and raw_results.get('Abstract'):
            result = SearchResult(
                title=raw_results.get('Heading', 'Search Result'),
                url=raw_results.get('AbstractURL', ''),
                snippet=raw_results.get('Abstract', ''),
                source=raw_results.get('AbstractSource', 'DuckDuckGo')
            )
            results.append(result)
        
        return results

class MockSearchProvider(SearchProviderBase):
    """Mock search provider for testing and examples."""
    
    def __init__(self):
        super().__init__()
        self.mock_data = self._create_mock_data()
    
    def search(self, query: str, num_results: int = 5, **kwargs) -> SearchResponse:
        """Execute mock search."""
        start_time = time.time()
        
        # Simple keyword matching for demo
        query_lower = query.lower()
        matching_results = []
        
        for mock_result in self.mock_data:
            if any(keyword in mock_result['title'].lower() or 
                  keyword in mock_result['snippet'].lower() 
                  for keyword in query_lower.split()):
                matching_results.append(SearchResult(
                    title=mock_result['title'],
                    url=mock_result['url'],
                    snippet=mock_result['snippet'],
                    source='MockSearch',
                    relevance_score=0.8
                ))
        
        # Limit results
        matching_results = matching_results[:num_results]
        
        return SearchResponse(
            query=query,
            results=matching_results,
            total_results=len(matching_results),
            search_time=time.time() - start_time,
            provider="MockSearch",
            timestamp=datetime.now()
        )
    
    def _create_mock_data(self) -> List[Dict[str, str]]:
        """Create mock search data for demonstration."""
        return [
            {
                'title': 'LangChain Documentation - Official Guide',
                'url': 'https://python.langchain.com/docs/',
                'snippet': 'LangChain is a framework for developing applications powered by language models. It provides tools for building context-aware and reasoning applications.'
            },
            {
                'title': 'Python Programming Tutorial - Complete Guide',
                'url': 'https://python.org/tutorial/',
                'snippet': 'Learn Python programming from basics to advanced concepts. Comprehensive tutorial covering syntax, data structures, and best practices.'
            },
            {
                'title': 'Artificial Intelligence News and Updates',
                'url': 'https://ai-news.com/',
                'snippet': 'Latest developments in artificial intelligence, machine learning, and deep learning. Stay updated with AI research and industry trends.'
            },
            {
                'title': 'Web Development Best Practices',
                'url': 'https://webdev-guide.com/',
                'snippet': 'Modern web development techniques, frameworks, and tools. Learn HTML, CSS, JavaScript, and popular frameworks like React and Vue.'
            },
            {
                'title': 'Data Science and Machine Learning Resources',
                'url': 'https://datascience-hub.com/',
                'snippet': 'Comprehensive resources for data science, machine learning, and analytics. Tutorials, datasets, and practical examples.'
            }
        ]

# LangChain Tool Implementations

class WebSearchTool(BaseTool):
    """General web search tool."""
    
    name: str = "web_search"
    description: str = "Search the web for current information. Useful for finding recent news, facts, or general information."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def __init__(self, provider: Optional[SearchProviderBase] = None):
        super().__init__()
        self.provider = provider or self._get_default_provider()
    
    def _get_default_provider(self) -> SearchProviderBase:
        """Get the default search provider."""
        # Try to use DuckDuckGo, fall back to mock if needed
        try:
            return DuckDuckGoSearchProvider()
        except Exception:
            logger.warning("Using mock search provider")
            return MockSearchProvider()
    
    def _run(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute web search."""
        try:
            logger.info(f"Searching for: {query} (type: {search_type})")
            
            # Execute search
            response = self.provider.search(
                query=query,
                num_results=num_results,
                search_type=search_type
            )
            
            # Format results
            if not response.results:
                return f"No search results found for '{query}'"
            
            formatted_results = []
            for i, result in enumerate(response.results, 1):
                formatted_result = f"{i}. {result.title}\n   URL: {result.url}\n   {result.snippet}"
                formatted_results.append(formatted_result)
            
            result_text = f"Search results for '{query}' ({response.total_results} results, {response.search_time:.2f}s):\n\n"
            result_text += "\n\n".join(formatted_results)
            
            return result_text
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "general",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of web search."""
        # For this example, we'll just call the sync version
        # In a real implementation, you'd use async HTTP requests
        return self._run(query, num_results, search_type, run_manager)

class NewsSearchTool(BaseTool):
    """News-specific search tool."""
    
    name: str = "news_search"
    description: str = "Search for recent news articles on a specific topic. Good for current events and recent developments."
    args_schema: Type[BaseModel] = NewsSearchInput
    
    def __init__(self):
        super().__init__()
        self.provider = MockSearchProvider()  # In real implementation, use news API
    
    def _run(
        self,
        query: str,
        days_back: int = 7,
        language: str = "en",
        category: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute news search."""
        try:
            logger.info(f"Searching news for: {query} (last {days_back} days)")
            
            # Add news-specific keywords to query
            news_query = f"{query} news recent"
            if category:
                news_query += f" {category}"
            
            response = self.provider.search(query=news_query, num_results=5)
            
            if not response.results:
                return f"No recent news found for '{query}'"
            
            formatted_results = []
            for i, result in enumerate(response.results, 1):
                formatted_result = f"{i}. {result.title}\n   {result.snippet}\n   Source: {result.source}"
                formatted_results.append(formatted_result)
            
            result_text = f"Recent news for '{query}' (last {days_back} days):\n\n"
            result_text += "\n\n".join(formatted_results)
            
            return result_text
            
        except Exception as e:
            error_msg = f"News search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        days_back: int = 7,
        language: str = "en",
        category: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of news search."""
        return self._run(query, days_back, language, category, run_manager)

class AcademicSearchTool(BaseTool):
    """Academic paper search tool."""
    
    name: str = "academic_search"
    description: str = "Search for academic papers and research on a topic. Good for scholarly information and research papers."
    args_schema: Type[BaseModel] = AcademicSearchInput
    
    def __init__(self):
        super().__init__()
        self.provider = MockSearchProvider()  # In real implementation, use academic API
    
    def _run(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        field_of_study: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute academic search."""
        try:
            logger.info(f"Searching academic papers for: {query}")
            
            # Add academic-specific keywords
            academic_query = f"{query} research paper academic study"
            if field_of_study:
                academic_query += f" {field_of_study}"
            
            response = self.provider.search(query=academic_query, num_results=5)
            
            if not response.results:
                return f"No academic papers found for '{query}'"
            
            formatted_results = []
            for i, result in enumerate(response.results, 1):
                year_filter = ""
                if year_from or year_to:
                    year_filter = f" ({year_from or 'any'}-{year_to or 'present'})"
                
                formatted_result = f"{i}. {result.title}{year_filter}\n   {result.snippet}\n   Research Source: {result.source}"
                formatted_results.append(formatted_result)
            
            result_text = f"Academic research for '{query}':\n\n"
            result_text += "\n\n".join(formatted_results)
            
            return result_text
            
        except Exception as e:
            error_msg = f"Academic search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        field_of_study: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of academic search."""
        return self._run(query, year_from, year_to, field_of_study, run_manager)

# Testing and demonstration functions

def test_search_tools():
    """Test all search tools with sample queries."""
    
    print("\n" + "="*60)
    print("WEB SEARCH TOOLS TESTING")
    print("="*60)
    
    # Test WebSearchTool
    print("\n1. Testing WebSearchTool:")
    web_search = WebSearchTool()
    
    web_queries = [
        "LangChain framework",
        "Python programming best practices",
        "artificial intelligence 2024"
    ]
    
    for query in web_queries:
        print(f"\nQuery: {query}")
        result = web_search._run(query, num_results=3)
        print(f"Result:\n{result[:500]}..." if len(result) > 500 else result)
    
    # Test NewsSearchTool
    print(f"\n" + "="*50)
    print("2. Testing NewsSearchTool:")
    news_search = NewsSearchTool()
    
    news_queries = [
        "artificial intelligence developments",
        "technology trends"
    ]
    
    for query in news_queries:
        print(f"\nQuery: {query}")
        result = news_search._run(query, days_back=7)
        print(f"Result:\n{result[:500]}..." if len(result) > 500 else result)
    
    # Test AcademicSearchTool
    print(f"\n" + "="*50)
    print("3. Testing AcademicSearchTool:")
    academic_search = AcademicSearchTool()
    
    academic_queries = [
        "machine learning algorithms",
        "natural language processing"
    ]
    
    for query in academic_queries:
        print(f"\nQuery: {query}")
        result = academic_search._run(query, year_from=2020, field_of_study="computer science")
        print(f"Result:\n{result[:500]}..." if len(result) > 500 else result)

def demonstrate_search_in_agent():
    """Demonstrate search tools in an agent context."""
    
    print(f"\n" + "="*60)
    print("SEARCH TOOLS IN AGENT DEMONSTRATION")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY required for agent demonstration")
        return
    
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        
        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create search tools
        tools = [
            WebSearchTool(),
            NewsSearchTool(),
            AcademicSearchTool()
        ]
        
        # Create agent prompt
        prompt = PromptTemplate(
            template="""You are a research assistant with access to web search, news search, and academic search tools.

Tools available:
{tools}

Use this format:
Question: {input}
Thought: I need to search for information
Action: {tool_names}
Action Input: the search query
Observation: the search results
Thought: I can now provide a comprehensive answer
Final Answer: my response based on the search results

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
            "What are the latest developments in AI?",
            "Find academic research on natural language processing",
            "Search for recent news about Python programming"
        ]
        
        for query in test_queries:
            print(f"\n" + "-"*40)
            print(f"Query: {query}")
            print("-"*40)
            
            try:
                result = agent_executor.invoke({"input": query})
                print(f"Agent Response: {result['output']}")
            except Exception as e:
                print(f"Error: {e}")
    
    except ImportError:
        print("LangChain agent dependencies not available for demonstration")

if __name__ == "__main__":
    """
    Test and demonstrate web search tools.
    
    This demonstrates:
    1. Different search tool implementations
    2. Search provider patterns
    3. Rate limiting and caching
    4. Integration with LangChain agents
    5. Error handling for search operations
    """
    
    try:
        # Test individual search tools
        test_search_tools()
        
        # Demonstrate in agent context
        demonstrate_search_in_agent()
        
    except KeyboardInterrupt:
        print("\nSearch tool testing interrupted by user")
        
    except Exception as e:
        print(f"Unexpected error in search tool testing: {e}")
        raise
    
    finally:
        print("\nWeb search tool demonstration completed")