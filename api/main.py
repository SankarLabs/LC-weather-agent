"""
FastAPI Backend for Weather Agent
Provides REST API endpoints for the weather agent functionality.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, AsyncGenerator
import logging
import time
import asyncio
import json
from datetime import datetime
import uvicorn

# Import your weather agent
try:
    from agents.weather_agent import create_weather_agent
    from utils.config import get_config, WeatherAgentConfig
except ImportError as e:
    logging.error(f"Failed to import weather agent: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Weather Agent API",
    description="REST API for intelligent weather information and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class WeatherQuery(BaseModel):
    """Request model for weather queries."""
    query: str = Field(..., description="Natural language weather query", min_length=1)
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation context")
    include_metadata: bool = Field(default=False, description="Include response metadata")

class WeatherResponse(BaseModel):
    """Response model for weather queries."""
    response: str = Field(..., description="Weather agent response")
    success: bool = Field(..., description="Whether the query was successful")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    response_time: Optional[float] = Field(default=None, description="Response time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    error: Optional[str] = Field(default=None, description="Error message if any")

class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime: Optional[float] = Field(default=None, description="Uptime in seconds")

class AgentMetrics(BaseModel):
    """Agent metrics response model."""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    success_rate_percent: float = Field(..., description="Success rate percentage")
    error_rate_percent: float = Field(..., description="Error rate percentage")
    avg_response_time: float = Field(..., description="Average response time in seconds")
    uptime: float = Field(..., description="Uptime in seconds")

# Global variables
weather_agent = None
app_start_time = time.time()

async def get_weather_agent():
    """Dependency to get the weather agent instance."""
    global weather_agent
    if weather_agent is None:
        try:
            config = get_config()
            weather_agent = create_weather_agent(config=config)
            logger.info("Weather agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize weather agent: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize weather agent: {str(e)}"
            )
    return weather_agent

@app.on_event("startup")
async def startup_event():
    """Initialize the weather agent on startup."""
    logger.info("Starting Weather Agent API...")
    try:
        await get_weather_agent()
        logger.info("Weather Agent API started successfully")
    except Exception as e:
        logger.error(f"Failed to start Weather Agent API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Weather Agent API...")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Weather Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/query", response_model=WeatherResponse)
async def query_weather(
    query: WeatherQuery,
    agent = Depends(get_weather_agent)
) -> WeatherResponse:
    """Query the weather agent with natural language input."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {query.query[:100]}...")
        
        # Query the agent
        response_data = agent.query(query.query)
        response_time = time.time() - start_time
        
        # Prepare response
        weather_response = WeatherResponse(
            response=response_data.get("response", ""),
            success=response_data.get("success", True),
            session_id=query.session_id,
            response_time=response_time,
            metadata=response_data if query.include_metadata else None,
            error=response_data.get("error") if not response_data.get("success", True) else None
        )
        
        logger.info(f"Query processed successfully in {response_time:.2f}s")
        return weather_response
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Error processing query: {e}")
        
        return WeatherResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            success=False,
            session_id=query.session_id,
            response_time=response_time,
            error=str(e)
        )

@app.post("/query/stream")
async def query_weather_stream(
    query: WeatherQuery,
    agent = Depends(get_weather_agent)
) -> StreamingResponse:
    """Stream weather agent response for real-time updates."""
    
    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"Processing streaming query: {query.query[:100]}...")
            
            # For now, simulate streaming by chunking the response
            # In a real implementation, you'd use agent.astream() if available
            response_data = agent.query(query.query)
            response_text = response_data.get("response", "")
            
            # Stream response in chunks
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "chunk": word + " ",
                    "index": i,
                    "total_words": len(words),
                    "done": i == len(words) - 1
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Simulate typing delay
            
            # Send final completion message
            final_chunk = {
                "chunk": "",
                "done": True,
                "success": response_data.get("success", True),
                "metadata": response_data if query.include_metadata else None
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            error_chunk = {
                "chunk": "",
                "done": True,
                "success": False,
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/conversation/reset")
async def reset_conversation(
    session_id: Optional[str] = None,
    agent = Depends(get_weather_agent)
) -> Dict[str, str]:
    """Reset the conversation memory."""
    try:
        agent.reset_conversation()
        logger.info(f"Conversation reset for session: {session_id}")
        return {"message": "Conversation reset successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/history")
async def get_conversation_history(
    agent = Depends(get_weather_agent)
) -> Dict[str, Any]:
    """Get the current conversation history."""
    try:
        history = agent.get_conversation_history()
        return {
            "history": [
                {
                    "role": msg.type if hasattr(msg, 'type') else 'unknown',
                    "content": msg.content if hasattr(msg, 'content') else str(msg),
                    "timestamp": datetime.now().isoformat()
                }
                for msg in history
            ],
            "total_messages": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthCheck)
async def health_check(agent = Depends(get_weather_agent)) -> HealthCheck:
    """Health check endpoint."""
    try:
        # Get agent health
        agent_health = agent.health_check()
        uptime = time.time() - app_start_time
        
        return HealthCheck(
            status=agent_health.get("status", "unknown"),
            components=agent_health.get("components", {}),
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            components={"error": str(e)},
            uptime=time.time() - app_start_time
        )

@app.get("/metrics", response_model=AgentMetrics)
async def get_metrics(agent = Depends(get_weather_agent)) -> AgentMetrics:
    """Get agent performance metrics."""
    try:
        metrics = agent.get_metrics()
        uptime = time.time() - app_start_time
        
        return AgentMetrics(
            total_requests=metrics.get("total_requests", 0),
            successful_requests=metrics.get("successful_requests", 0),
            failed_requests=metrics.get("failed_requests", 0),
            success_rate_percent=metrics.get("success_rate_percent", 0.0),
            error_rate_percent=metrics.get("error_rate_percent", 0.0),
            avg_response_time=metrics.get("avg_response_time", 0.0),
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools(agent = Depends(get_weather_agent)) -> Dict[str, Any]:
    """List available tools in the weather agent."""
    try:
        # This would depend on your agent implementation
        # Assuming agent has a way to list tools
        tools_info = {
            "tools": [
                {
                    "name": "current_weather",
                    "description": "Get current weather conditions for a location"
                },
                {
                    "name": "weather_forecast",
                    "description": "Get weather forecast for a location"
                },
                {
                    "name": "location_search",
                    "description": "Search and resolve location coordinates"
                },
                {
                    "name": "weather_alerts",
                    "description": "Get weather alerts and warnings"
                }
            ],
            "total_tools": 4
        }
        return tools_info
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )