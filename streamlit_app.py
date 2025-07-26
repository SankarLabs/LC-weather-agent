"""
Streamlit Web Interface for Weather Agent
Provides a chat-based interface for interacting with the weather agent.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your weather agent
try:
    from agents.weather_agent import create_weather_agent
    from utils.config import get_config
except ImportError as e:
    st.error(f"Failed to import weather agent: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Weather Agent Chat",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .metrics-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

def create_agent():
    """Create and cache the weather agent."""
    if st.session_state.agent is None:
        try:
            with st.spinner("Initializing weather agent..."):
                config = get_config()
                agent = create_weather_agent(config=config)
                st.session_state.agent = agent
                st.session_state.agent_initialized = True
                logger.info("Weather agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize weather agent: {e}")
            st.error(f"Failed to initialize weather agent: {e}")
            st.stop()
    return st.session_state.agent

def display_message(role: str, content: str, timestamp: Optional[datetime] = None):
    """Display a chat message with proper styling."""
    if timestamp is None:
        timestamp = datetime.now()
    
    with st.chat_message(role):
        st.write(content)
        st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')}")

def get_agent_response(user_input: str) -> Dict[str, Any]:
    """Get response from the weather agent with error handling."""
    try:
        start_time = time.time()
        agent = st.session_state.agent
        
        # Show typing indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = agent.query(user_input)
        
        response_time = time.time() - start_time
        response["response_time"] = response_time
        
        return response
    except Exception as e:
        logger.error(f"Error getting agent response: {e}")
        return {
            "response": f"Sorry, I encountered an error: {str(e)}",
            "success": False,
            "error": str(e),
            "response_time": 0
        }

def display_agent_metrics():
    """Display agent performance metrics in sidebar."""
    if st.session_state.agent and st.session_state.agent_initialized:
        try:
            metrics = st.session_state.agent.get_metrics()
            health = st.session_state.agent.health_check()
            
            st.sidebar.markdown("### 📊 Agent Metrics")
            
            # Health status
            status_color = "🟢" if health["status"] == "healthy" else "🔴"
            st.sidebar.markdown(f"**Status:** {status_color} {health['status'].title()}")
            
            # Performance metrics
            if metrics:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Total Requests", metrics.get("total_requests", 0))
                    st.metric("Success Rate", f"{metrics.get('success_rate_percent', 0):.1f}%")
                with col2:
                    st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
                    st.metric("Error Rate", f"{metrics.get('error_rate_percent', 0):.1f}%")
            
        except Exception as e:
            st.sidebar.error(f"Error loading metrics: {e}")

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌤️ Weather Agent Chat</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Agent initialization
        if not st.session_state.agent_initialized:
            if st.button("🚀 Initialize Agent", type="primary"):
                create_agent()
                st.rerun()
        else:
            st.success("✅ Agent Ready")
            
            # Reset conversation
            if st.button("🔄 Reset Conversation"):
                if st.session_state.agent:
                    st.session_state.agent.reset_conversation()
                st.session_state.messages = []
                st.session_state.conversation_count = 0
                st.success("Conversation reset!")
                st.rerun()
        
        st.markdown("---")
        
        # Display metrics
        display_agent_metrics()
        
        st.markdown("---")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "What's the weather in London?",
            "5-day forecast for New York",
            "Is it going to rain today?",
            "What should I wear in Paris?",
            "Compare weather in Tokyo and Berlin"
        ]
        
        for query in example_queries:
            if st.button(f"📝 {query}", key=f"example_{query}", use_container_width=True):
                st.session_state.example_query = query
                st.rerun()
    
    # Main chat interface
    if not st.session_state.agent_initialized:
        st.info("👆 Please initialize the agent using the sidebar to start chatting!")
        return
    
    # Create agent if not already created
    if st.session_state.agent is None:
        create_agent()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(
                message["role"],
                message["content"],
                message.get("timestamp")
            )
    
    # Handle example query selection
    if hasattr(st.session_state, 'example_query'):
        user_input = st.session_state.example_query
        del st.session_state.example_query
    else:
        # Chat input
        user_input = st.chat_input("Ask me anything about weather! 🌦️")
    
    # Process user input
    if user_input:
        timestamp = datetime.now()
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Display user message
        display_message("user", user_input, timestamp)
        
        # Get and display agent response
        with st.spinner("Getting weather information..."):
            response_data = get_agent_response(user_input)
        
        # Add assistant response to chat
        assistant_timestamp = datetime.now()
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": assistant_timestamp,
            "metadata": response_data
        })
        
        # Display assistant response
        display_message("assistant", response_data["response"], assistant_timestamp)
        
        # Show response metadata if there was an error
        if not response_data.get("success", True):
            st.error(f"Error: {response_data.get('error', 'Unknown error')}")
        
        # Show response time
        if "response_time" in response_data:
            st.caption(f"⏱️ Response time: {response_data['response_time']:.2f}s")
        
        # Update conversation count
        st.session_state.conversation_count += 1
        
        # Rerun to update the display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using LangChain and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()