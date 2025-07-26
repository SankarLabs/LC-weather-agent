#!/usr/bin/env python3
"""
Weather Agent Demo

This script demonstrates the weather agent functionality and provides
a simple command-line interface for testing the weather agent.

Usage:
    python weather_agent_demo.py

Requirements:
    - Set OPENAI_API_KEY and OPENWEATHER_API_KEY in .env file
    - Install required dependencies: pip install -r requirements.txt
"""

import os
import sys
import logging
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.weather_agent import create_weather_agent
from utils.config import validate_environment, get_config
from utils.resilience import validate_environment_safety

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print weather agent banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🌤️  Weather Agent Demo                     ║
║                                                              ║
║  Powered by LangChain + OpenWeatherMap + OpenAI             ║
║  ------------------------------------------------------------ ║
║  Ask me about weather conditions, forecasts, and more!      ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_usage_examples():
    """Print usage examples."""
    examples = """
📝 Example Queries:
   • "What's the weather like in London?"
   • "Give me a 5-day forecast for New York"
   • "Is it going to rain in Tokyo tomorrow?"
   • "What should I wear in Paris today?"
   • "Compare weather in Berlin and Madrid"
   • "Show me the coordinates for Sydney"

💡 Tips:
   • Use specific city names for better results
   • Ask for recommendations based on weather
   • Try follow-up questions - I remember our conversation!
   • Type 'quit', 'exit', or 'bye' to end the session
   • Type 'help' to see this message again
   • Type 'stats' to see performance metrics
   • Type 'health' to check system health
    """
    print(examples)


def validate_setup() -> bool:
    """
    Validate environment setup and configuration.
    
    Returns:
        True if setup is valid, False otherwise
    """
    print("🔍 Validating environment setup...")
    
    # Basic environment validation
    if not validate_environment():
        print("❌ Environment validation failed!")
        print("   Please ensure you have:")
        print("   • OPENAI_API_KEY set in your .env file")
        print("   • OPENWEATHER_API_KEY set in your .env file")
        print("\\n   Copy .env.example to .env and add your API keys")
        return False
    
    # Security validation
    safety_results = validate_environment_safety()
    if not safety_results["safe"]:
        print("❌ Environment safety validation failed!")
        for error in safety_results["errors"]:
            print(f"   • {error}")
        return False
    
    if safety_results["warnings"]:
        print("⚠️  Environment warnings:")
        for warning in safety_results["warnings"]:
            print(f"   • {warning}")
    
    print("✅ Environment validation passed!")
    return True


def create_agent():
    """
    Create and initialize the weather agent.
    
    Returns:
        Weather agent instance or None if creation fails
    """
    try:
        print("🚀 Initializing weather agent...")
        
        # Load configuration
        config = get_config(production=False)
        
        # Create agent
        agent = create_weather_agent(config=config)
        
        # Perform health check
        health = agent.health_check()
        if health["status"] != "healthy":
            print(f"❌ Agent health check failed: {health.get('error', 'Unknown error')}")
            return None
        
        print("✅ Weather agent initialized successfully!")
        return agent
        
    except Exception as e:
        print(f"❌ Failed to create weather agent: {e}")
        logger.error(f"Agent creation failed: {e}")
        return None


def process_special_commands(command: str, agent) -> bool:
    """
    Process special commands like help, stats, etc.
    
    Args:
        command: User command
        agent: Weather agent instance
        
    Returns:
        True if command was processed, False otherwise
    """
    command = command.lower().strip()
    
    if command in ['help', '?']:
        print_usage_examples()
        return True
    
    elif command in ['stats', 'metrics']:
        metrics = agent.get_metrics()
        print("\\n📊 Performance Metrics:")
        print(f"   • Total requests: {metrics['total_requests']}")
        print(f"   • Error count: {metrics['error_count']}")
        print(f"   • Error rate: {metrics['error_rate_percent']:.1f}%")
        print(f"   • Avg response time: {metrics['avg_response_time']:.2f}s")
        return True
    
    elif command in ['health', 'status']:
        health = agent.health_check()
        print(f"\\n💚 System Health: {health['status'].upper()}")
        print("   Components:")
        for component, status in health['components'].items():
            emoji = "✅" if status == "healthy" else "❌"
            print(f"   • {component}: {emoji} {status}")
        return True
    
    elif command in ['reset', 'clear']:
        agent.reset_conversation()
        print("\\n🔄 Conversation history cleared!")
        return True
    
    elif command in ['history']:
        history = agent.get_conversation_history()
        print(f"\\n📜 Conversation History ({len(history)} messages):")
        for i, msg in enumerate(history[-5:], 1):  # Show last 5 messages
            role = msg.type if hasattr(msg, 'type') else 'unknown'
            content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
            print(f"   {i}. [{role}] {content}")
        return True
    
    return False


def interactive_mode(agent):
    """
    Run interactive mode for weather queries.
    
    Args:
        agent: Weather agent instance
    """
    print("\\n🎯 Entering interactive mode...")
    print("   Type your weather questions or 'help' for examples")
    print("   Type 'quit' to exit\\n")
    
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("🌤️  You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\\n👋 Thanks for using the Weather Agent! Stay weather-aware!")
                break
            
            # Process special commands
            if process_special_commands(user_input, agent):
                continue
            
            # Process weather query
            print("🤔 Thinking...")
            
            try:
                response = agent.query(user_input)
                conversation_count += 1
                
                if response["success"]:
                    print(f"\\n🌤️  Weather Agent: {response['response']}\\n")
                    
                    # Show debug info for first few queries
                    if conversation_count <= 2 and response.get("intermediate_steps"):
                        print("🔧 Debug - Tools used:")
                        for i, (action, observation) in enumerate(response["intermediate_steps"], 1):
                            tool_name = getattr(action, 'tool', 'unknown')
                            print(f"   {i}. {tool_name}")
                        print()
                else:
                    print(f"\\n❌ Error: {response.get('error', 'Unknown error')}\\n")
                    
            except KeyboardInterrupt:
                print("\\n\\n⏸️  Query interrupted by user")
                continue
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}\\n")
                logger.error(f"Query processing error: {e}")
                
        except KeyboardInterrupt:
            print("\\n\\n👋 Goodbye!")
            break
        except EOFError:
            print("\\n\\n👋 Goodbye!")
            break


def demo_mode(agent):
    """
    Run demo mode with predefined queries.
    
    Args:
        agent: Weather agent instance
    """
    print("\\n🎪 Running demo mode with sample queries...")
    
    demo_queries = [
        "What's the current weather in London?",
        "Give me a 3-day forecast for New York",
        "What should I wear if I'm visiting Tokyo today?",
        "Look up coordinates for Sydney, Australia"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\\n--- Demo Query {i}/{len(demo_queries)} ---")
        print(f"Query: {query}")
        
        try:
            response = agent.query(query)
            
            if response["success"]:
                print(f"Response: {response['response']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error processing query: {e}")
        
        # Small delay between queries
        import time
        time.sleep(1)
    
    print("\\n✅ Demo completed!")


def main():
    """Main function."""
    print_banner()
    
    # Validate setup
    if not validate_setup():
        sys.exit(1)
    
    # Create agent
    agent = create_agent()
    if not agent:
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_mode(agent)
    else:
        print_usage_examples()
        interactive_mode(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n👋 Goodbye!")
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)