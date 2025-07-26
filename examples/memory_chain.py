"""
Memory Chain Implementation Examples

This file demonstrates different types of memory management in LangChain.
It shows best practices for conversation memory, context preservation,
and memory optimization techniques.

Key Patterns Demonstrated:
- Different memory types (Buffer, Window, Summary)
- Memory persistence and retrieval
- Conversation context management
- Memory optimization for long conversations
- Custom memory implementations
- Memory integration with chains and agents
"""

import os
import logging
import pickle
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationSummaryMemory,
    ConversationEntityMemory
)
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory, BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryConfig(BaseModel):
    """Configuration for memory systems."""
    memory_type: str = Field("buffer", description="Type of memory to use")
    max_token_limit: int = Field(1000, description="Maximum tokens for summary memory")
    window_size: int = Field(5, description="Window size for window memory")
    persist_path: str = Field("./memory_data", description="Path for memory persistence")
    auto_save: bool = Field(True, description="Automatically save memory")

class ConversationManager:
    """Manages different types of conversation memory."""
    
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create memory instances
        self.memories = self._create_memory_instances()
        
        # Create conversation chains
        self.chains = self._create_conversation_chains()
        
        # Setup persistence
        self.persist_dir = Path(config.persist_path)
        self.persist_dir.mkdir(exist_ok=True)
    
    def _create_memory_instances(self) -> Dict[str, BaseMemory]:
        """Create different types of memory instances."""
        memories = {}
        
        # 1. Buffer Memory - Stores all conversation history
        memories['buffer'] = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        
        # 2. Window Memory - Stores only last N exchanges
        memories['window'] = ConversationBufferWindowMemory(
            k=self.config.window_size,
            memory_key="history",
            return_messages=True
        )
        
        # 3. Summary Memory - Summarizes old conversations
        memories['summary'] = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="history",
            return_messages=True
        )
        
        # 4. Summary Buffer Memory - Keeps recent + summary of old
        memories['summary_buffer'] = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.config.max_token_limit,
            memory_key="history",
            return_messages=True
        )
        
        # 5. Entity Memory - Tracks entities mentioned in conversation
        memories['entity'] = ConversationEntityMemory(
            llm=self.llm,
            memory_key="history",
            entity_key="entities"
        )
        
        logger.info(f"Created {len(memories)} memory types")
        return memories
    
    def _create_conversation_chains(self) -> Dict[str, ConversationChain]:
        """Create conversation chains with different memory types."""
        chains = {}
        
        for memory_type, memory in self.memories.items():
            # Create custom prompt for each memory type
            if memory_type == 'entity':
                # Entity memory needs special prompt handling
                prompt = PromptTemplate(
                    input_variables=["history", "entities", "input"],
                    template="""The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant entity information:
{entities}

Current conversation:
{history}
Human: {input}
AI:"""
                )
            else:
                # Standard prompt for other memory types
                prompt = PromptTemplate(
                    input_variables=["history", "input"],
                    template="""The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
                )
            
            chains[memory_type] = ConversationChain(
                llm=self.llm,
                memory=memory,
                prompt=prompt,
                verbose=True
            )
        
        logger.info(f"Created conversation chains for {len(chains)} memory types")
        return chains
    
    def chat(self, memory_type: str, message: str) -> Dict[str, Any]:
        """
        Have a conversation using specified memory type.
        
        Args:
            memory_type: Type of memory to use
            message: User message
            
        Returns:
            Dict with response and memory info
        """
        if memory_type not in self.chains:
            return {
                "success": False,
                "error": f"Unknown memory type: {memory_type}. Available: {list(self.chains.keys())}"
            }
        
        try:
            start_time = datetime.now()
            
            # Get the conversation chain
            chain = self.chains[memory_type]
            
            # Generate response
            response = chain.predict(input=message)
            
            # Get memory information
            memory_info = self._get_memory_info(memory_type)
            
            # Auto-save if enabled
            if self.config.auto_save:
                self._save_memory(memory_type)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "memory_type": memory_type,
                "user_message": message,
                "ai_response": response,
                "memory_info": memory_info,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Chat completed using {memory_type} memory in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Chat failed with {memory_type} memory: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "memory_type": memory_type,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_memory_info(self, memory_type: str) -> Dict[str, Any]:
        """Get information about the current state of memory."""
        memory = self.memories[memory_type]
        
        try:
            # Get memory variables
            memory_variables = memory.load_memory_variables({})
            
            info = {
                "type": memory_type,
                "variables": list(memory_variables.keys())
            }
            
            # Add type-specific information
            if memory_type == 'buffer':
                history = memory_variables.get('history', [])
                info.update({
                    "message_count": len(history),
                    "total_tokens": self._estimate_token_count(history)
                })
            
            elif memory_type == 'window':
                history = memory_variables.get('history', [])
                info.update({
                    "message_count": len(history),
                    "window_size": self.config.window_size,
                    "is_full": len(history) >= self.config.window_size * 2  # Human + AI pairs
                })
            
            elif memory_type in ['summary', 'summary_buffer']:
                history = memory_variables.get('history', [])
                info.update({
                    "message_count": len(history),
                    "has_summary": hasattr(memory, 'moving_summary_buffer') and bool(memory.moving_summary_buffer),
                    "max_token_limit": getattr(memory, 'max_token_limit', None)
                })
            
            elif memory_type == 'entity':
                entities = memory_variables.get('entities', {})
                info.update({
                    "entity_count": len(entities),
                    "entities": list(entities.keys())
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get memory info for {memory_type}: {e}")
            return {"type": memory_type, "error": str(e)}
    
    def _estimate_token_count(self, messages: List[BaseMessage]) -> int:
        """Estimate token count for messages."""
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            
            total_tokens = 0
            for message in messages:
                if hasattr(message, 'content'):
                    total_tokens += len(encoder.encode(message.content))
            
            return total_tokens
            
        except ImportError:
            # Rough estimation if tiktoken not available
            total_chars = sum(len(msg.content) if hasattr(msg, 'content') else 0 for msg in messages)
            return total_chars // 4  # Rough approximation
    
    def _save_memory(self, memory_type: str):
        """Save memory state to disk."""
        try:
            memory = self.memories[memory_type]
            memory_variables = memory.load_memory_variables({})
            
            save_path = self.persist_dir / f"{memory_type}_memory.json"
            
            # Convert messages to serializable format
            serializable_data = {}
            for key, value in memory_variables.items():
                if isinstance(value, list) and value and isinstance(value[0], BaseMessage):
                    # Convert messages to dict format
                    serializable_data[key] = [
                        {
                            "type": msg.__class__.__name__,
                            "content": msg.content,
                            "timestamp": datetime.now().isoformat()
                        }
                        for msg in value
                    ]
                else:
                    serializable_data[key] = value
            
            with open(save_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved {memory_type} memory to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save {memory_type} memory: {e}")
    
    def load_memory(self, memory_type: str) -> bool:
        """Load memory state from disk."""
        try:
            save_path = self.persist_dir / f"{memory_type}_memory.json"
            
            if not save_path.exists():
                logger.info(f"No saved memory found for {memory_type}")
                return False
            
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            # Reconstruct memory
            memory = self.memories[memory_type]
            
            # Load messages back
            if 'history' in saved_data and isinstance(saved_data['history'], list):
                for msg_data in saved_data['history']:
                    if msg_data['type'] == 'HumanMessage':
                        memory.chat_memory.add_user_message(msg_data['content'])
                    elif msg_data['type'] == 'AIMessage':
                        memory.chat_memory.add_ai_message(msg_data['content'])
            
            logger.info(f"Loaded {memory_type} memory from {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {memory_type} memory: {e}")
            return False
    
    def clear_memory(self, memory_type: str):
        """Clear memory for specified type."""
        try:
            memory = self.memories[memory_type]
            memory.clear()
            
            # Also remove saved file
            save_path = self.persist_dir / f"{memory_type}_memory.json"
            if save_path.exists():
                save_path.unlink()
            
            logger.info(f"Cleared {memory_type} memory")
            
        except Exception as e:
            logger.error(f"Failed to clear {memory_type} memory: {e}")
    
    def compare_memories(self, message: str) -> Dict[str, Any]:
        """Compare how different memory types handle the same message."""
        results = {}
        
        for memory_type in self.memories.keys():
            result = self.chat(memory_type, message)
            results[memory_type] = {
                "response": result.get("ai_response", ""),
                "memory_info": result.get("memory_info", {}),
                "success": result.get("success", False)
            }
        
        return results

def demonstrate_memory_types():
    """Demonstrate different memory types with conversation examples."""
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    print("\n" + "="*60)
    print("MEMORY TYPES DEMONSTRATION")
    print("="*60)
    
    # Initialize conversation manager
    config = MemoryConfig(
        window_size=3,
        max_token_limit=500,
        auto_save=True
    )
    
    manager = ConversationManager(config)
    
    # Conversation sequence to test memory
    conversation = [
        "Hi, my name is Alice and I'm a software engineer.",
        "I work mainly with Python and JavaScript.",
        "I love building web applications and AI tools.",
        "I have a cat named Whiskers who is 3 years old.",
        "What programming languages did I mention?",
        "What's my cat's name and age?",
        "Can you tell me about my profession?"
    ]
    
    # Test each memory type
    memory_types = ['buffer', 'window', 'summary_buffer']
    
    for memory_type in memory_types:
        print(f"\n" + "="*50)
        print(f"TESTING {memory_type.upper()} MEMORY")
        print("="*50)
        
        # Clear memory for clean test
        manager.clear_memory(memory_type)
        
        for i, message in enumerate(conversation, 1):
            print(f"\nTurn {i}: {message}")
            print("-" * 40)
            
            result = manager.chat(memory_type, message)
            
            if result["success"]:
                print(f"AI: {result['ai_response']}")
                
                memory_info = result['memory_info']
                print(f"Memory: {memory_info}")
            else:
                print(f"Error: {result['error']}")
        
        print(f"\nFinal memory state for {memory_type}:")
        final_info = manager._get_memory_info(memory_type)
        print(json.dumps(final_info, indent=2))

def demonstrate_entity_memory():
    """Demonstrate entity memory capabilities."""
    
    print(f"\n" + "="*50)
    print("ENTITY MEMORY DEMONSTRATION")
    print("="*50)
    
    config = MemoryConfig()
    manager = ConversationManager(config)
    
    # Clear entity memory
    manager.clear_memory('entity')
    
    # Entity-rich conversation
    entity_conversation = [
        "I work at TechCorp as a senior developer.",
        "My manager Sarah is really supportive.",
        "We're working on a project called DataViz using React and Python.",
        "The deadline is next Friday, March 15th.",
        "Tell me about my work situation."
    ]
    
    for i, message in enumerate(entity_conversation, 1):
        print(f"\nTurn {i}: {message}")
        print("-" * 30)
        
        result = manager.chat('entity', message)
        
        if result["success"]:
            print(f"AI: {result['ai_response']}")
            
            memory_info = result['memory_info']
            if 'entities' in memory_info:
                print(f"Tracked Entities: {memory_info['entities']}")
        else:
            print(f"Error: {result['error']}")

def demonstrate_memory_persistence():
    """Demonstrate memory persistence across sessions."""
    
    print(f"\n" + "="*50)
    print("MEMORY PERSISTENCE DEMONSTRATION")
    print("="*50)
    
    # Session 1: Create memories
    print("SESSION 1: Creating memories...")
    manager1 = ConversationManager()
    
    manager1.chat('buffer', "My favorite color is blue.")
    manager1.chat('buffer', "I have two dogs named Max and Luna.")
    
    print("Created memories and saved to disk.")
    
    # Session 2: Load memories
    print("\nSESSION 2: Loading memories...")
    manager2 = ConversationManager()
    
    # Load the saved memory
    if manager2.load_memory('buffer'):
        result = manager2.chat('buffer', "What do you remember about me?")
        if result["success"]:
            print(f"AI Response: {result['ai_response']}")
        else:
            print(f"Error: {result['error']}")
    else:
        print("No memories to load.")

def create_custom_memory_example():
    """Example of creating a custom memory implementation."""
    
    class TimestampedMemory(BaseMemory):
        """Custom memory that timestamps all messages."""
        
        memories: List[Dict[str, Any]] = []
        memory_key: str = "history"
        
        @property
        def memory_variables(self) -> List[str]:
            return [self.memory_key]
        
        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            # Return formatted history with timestamps
            formatted_history = []
            for mem in self.memories:
                timestamp = mem['timestamp'].strftime("%H:%M:%S")
                formatted_history.append(f"[{timestamp}] {mem['type']}: {mem['content']}")
            
            return {self.memory_key: "\n".join(formatted_history)}
        
        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            # Save input and output with timestamps
            self.memories.append({
                "type": "Human",
                "content": inputs.get("input", ""),
                "timestamp": datetime.now()
            })
            
            self.memories.append({
                "type": "AI",
                "content": outputs.get("response", ""),
                "timestamp": datetime.now()
            })
        
        def clear(self) -> None:
            self.memories.clear()
    
    print(f"\n" + "="*50)
    print("CUSTOM MEMORY EXAMPLE")
    print("="*50)
    
    # Create chain with custom memory
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    custom_memory = TimestampedMemory()
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""Conversation history with timestamps:
{history}

Current message: {input}
Response:"""
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=custom_memory,
        verbose=True
    )
    
    # Test the custom memory
    test_messages = [
        "Hello, I'm testing custom memory!",
        "What time did I first message you?",
        "Can you show me our conversation history?"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = chain.predict(input=message)
        print(f"AI: {response}")

def create_memory_with_chain_examples():
    """Examples of integrating different memory types with chains."""
    
    print(f"\n" + "="*50)
    print("MEMORY WITH CHAIN INTEGRATION")
    print("="*50)
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Example 1: Q&A Chain with Buffer Memory
    print("\n1. Q&A Chain with Buffer Memory:")
    
    qa_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""You are a helpful assistant. Use the conversation history to provide context-aware answers.

Chat History:
{chat_history}

Question: {question}
Answer:"""
    )
    
    qa_chain = LLMChain(
        llm=llm,
        prompt=qa_prompt,
        memory=qa_memory,
        verbose=True
    )
    
    # Test Q&A chain
    qa_questions = [
        "My name is John and I work as a data scientist.",
        "What's my profession?",
        "I specialize in machine learning and NLP.",
        "What are my specializations?"
    ]
    
    for question in qa_questions:
        print(f"Q: {question}")
        answer = qa_chain.predict(question=question)
        print(f"A: {answer}\n")
    
    # Example 2: Summarization Chain with Window Memory
    print("\n2. Summarization Chain with Window Memory:")
    
    summary_memory = ConversationBufferWindowMemory(
        k=3,  # Keep last 3 exchanges
        memory_key="recent_conversation"
    )
    
    summary_prompt = PromptTemplate(
        input_variables=["recent_conversation", "text"],
        template="""Based on our recent conversation, summarize the following text:

Recent conversation:
{recent_conversation}

Text to summarize: {text}

Summary:"""
    )
    
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        memory=summary_memory,
        verbose=True
    )
    
    # Test summarization chain
    texts_to_summarize = [
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning algorithms can process vast amounts of data.",
        "Natural language processing enables computers to understand human language."
    ]
    
    for text in texts_to_summarize:
        print(f"Text: {text}")
        summary = summary_chain.predict(text=text)
        print(f"Summary: {summary}\n")

if __name__ == "__main__":
    """
    Run the memory demonstrations.
    
    This demonstrates:
    1. Different memory types and their characteristics
    2. Memory persistence across sessions
    3. Entity tracking and extraction
    4. Custom memory implementations
    5. Memory optimization strategies
    6. Conversation context management
    """
    
    try:
        # Main memory types demonstration
        demonstrate_memory_types()
        
        # Entity memory demonstration
        demonstrate_entity_memory()
        
        # Memory persistence demonstration
        demonstrate_memory_persistence()
        
        # Custom memory example
        create_custom_memory_example()
        
        # Memory with chain integration
        create_memory_with_chain_examples()
        
    except KeyboardInterrupt:
        logger.info("Memory demonstration interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error in demonstration: {e}")
        raise
    
    finally:
        logger.info("Memory demonstration completed")