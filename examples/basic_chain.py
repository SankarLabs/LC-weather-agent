"""
Basic LangChain Chain Examples

This file demonstrates fundamental LangChain patterns that should be followed
in all implementations. It shows the core building blocks of chains, prompts,
and proper error handling.

Key Patterns Demonstrated:
- Chain creation and configuration
- Prompt template design
- Error handling and logging
- Input validation
- Model configuration
- Response processing
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChainInput(BaseModel):
    """Input validation model for chain operations."""
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to process")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(500, ge=1, le=4000, description="Maximum tokens to generate")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

class ChainResponse(BaseModel):
    """Structured response model."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    execution_time: Optional[float] = None

def create_llm(temperature: float = 0.7, max_tokens: int = 500) -> ChatOpenAI:
    """
    Create a properly configured LLM instance.
    
    Args:
        temperature: Creativity level (0.0 = deterministic, 2.0 = very creative)
        max_tokens: Maximum tokens to generate
        
    Returns:
        ChatOpenAI: Configured language model
        
    Raises:
        ValueError: If API key is not configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            request_timeout=30,  # 30 second timeout
            max_retries=3,       # Retry failed requests
        )
        
        # Test the connection
        test_response = llm.predict("Hello")
        logger.info("LLM connection established successfully")
        
        return llm
        
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        raise

def create_summarization_chain() -> LLMChain:
    """
    Create a text summarization chain.
    
    This demonstrates the basic pattern for creating LangChain chains:
    1. Define a clear prompt template
    2. Configure the LLM
    3. Combine into a chain
    4. Add error handling
    
    Returns:
        LLMChain: Configured summarization chain
    """
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["text", "max_sentences"],
        template="""Please summarize the following text in no more than {max_sentences} sentences.
Focus on the key points and main ideas.

Text to summarize:
{text}

Summary:"""
    )
    
    # Create the LLM
    llm = create_llm(temperature=0.3)  # Lower temperature for more focused summaries
    
    # Create the chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,  # Enable for development/debugging
        output_key="summary"
    )
    
    return chain

def create_qa_chain() -> LLMChain:
    """
    Create a question-answering chain with context.
    
    Returns:
        LLMChain: Configured Q&A chain
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question based on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
    )
    
    llm = create_llm(temperature=0.1)  # Very low temperature for factual accuracy
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        output_key="answer"
    )
    
    return chain

def create_creative_writing_chain() -> LLMChain:
    """
    Create a creative writing chain.
    
    Returns:
        LLMChain: Configured creative writing chain
    """
    prompt = PromptTemplate(
        input_variables=["topic", "style", "length"],
        template="""Write a {style} piece about {topic}. The piece should be approximately {length} words long.
Be creative and engaging while staying true to the specified style.

Topic: {topic}
Style: {style}
Length: {length} words

Creative piece:"""
    )
    
    llm = create_llm(temperature=0.9)  # High temperature for creativity
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        output_key="creative_text"
    )
    
    return chain

def safe_chain_execution(chain: LLMChain, input_data: Dict[str, Any]) -> ChainResponse:
    """
    Execute a chain with proper error handling and monitoring.
    
    This is the standard pattern for chain execution that should be used
    throughout the codebase.
    
    Args:
        chain: LangChain chain to execute
        input_data: Input data for the chain
        
    Returns:
        ChainResponse: Structured response with success/failure information
    """
    import time
    
    start_time = time.time()
    
    try:
        # Validate inputs if possible
        logger.info(f"Executing chain with inputs: {list(input_data.keys())}")
        
        # Execute the chain
        result = chain.run(input_data)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Chain executed successfully in {execution_time:.2f} seconds")
        
        return ChainResponse(
            success=True,
            result=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Chain execution failed: {str(e)}"
        logger.error(error_msg)
        
        return ChainResponse(
            success=False,
            error=error_msg,
            execution_time=execution_time
        )

def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if environment is properly configured
    """
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("Environment validation passed")
    return True

def demonstrate_basic_patterns():
    """Demonstrate the basic chain patterns with examples."""
    
    if not validate_environment():
        logger.error("Environment validation failed. Please check your .env file.")
        return
    
    # Example 1: Text Summarization
    print("\n" + "="*50)
    print("EXAMPLE 1: Text Summarization")
    print("="*50)
    
    summary_chain = create_summarization_chain()
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. The term "artificial intelligence" 
    is often used to describe machines that mimic "cognitive" functions that humans 
    associate with the human mind, such as "learning" and "problem solving".
    """
    
    summary_input = {
        "text": sample_text,
        "max_sentences": "3"
    }
    
    response = safe_chain_execution(summary_chain, summary_input)
    
    if response.success:
        print(f"Summary: {response.result}")
        print(f"Execution time: {response.execution_time:.2f}s")
    else:
        print(f"Error: {response.error}")
    
    # Example 2: Question Answering
    print("\n" + "="*50)
    print("EXAMPLE 2: Question Answering")
    print("="*50)
    
    qa_chain = create_qa_chain()
    
    context = """
    LangChain is a framework for developing applications powered by language models. 
    It enables applications that are context-aware and can reason about their environment. 
    LangChain provides tools for loading, processing, and indexing data so that it can 
    be used by language models.
    """
    
    qa_input = {
        "context": context,
        "question": "What is LangChain used for?"
    }
    
    response = safe_chain_execution(qa_chain, qa_input)
    
    if response.success:
        print(f"Question: {qa_input['question']}")
        print(f"Answer: {response.result}")
        print(f"Execution time: {response.execution_time:.2f}s")
    else:
        print(f"Error: {response.error}")
    
    # Example 3: Creative Writing
    print("\n" + "="*50)
    print("EXAMPLE 3: Creative Writing")
    print("="*50)
    
    creative_chain = create_creative_writing_chain()
    
    creative_input = {
        "topic": "a robot learning to paint",
        "style": "short story",
        "length": "200"
    }
    
    response = safe_chain_execution(creative_chain, creative_input)
    
    if response.success:
        print(f"Creative piece: {response.result}")
        print(f"Execution time: {response.execution_time:.2f}s")
    else:
        print(f"Error: {response.error}")

if __name__ == "__main__":
    """
    Run the basic chain examples.
    
    This demonstrates:
    1. Environment validation
    2. Chain creation
    3. Safe execution patterns
    4. Error handling
    5. Logging and monitoring
    """
    
    try:
        demonstrate_basic_patterns()
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    
    finally:
        logger.info("Basic chain examples completed")