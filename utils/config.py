"""
Weather Agent Configuration

This module provides configuration management for the weather agent using 
pydantic-settings pattern following examples/basic_chain.py patterns.

Key Features:
- Environment variable validation
- Configuration validation
- API key management
- Development/production settings
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAgentConfig(BaseModel):
    """Configuration for weather agent system."""
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    openweather_api_key: str = Field(..., description="OpenWeatherMap API key")
    
    # LLM Configuration
    model_name: str = Field("gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(0.1, description="Model temperature for weather responses")
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    
    # Agent Configuration
    max_iterations: int = Field(5, description="Maximum agent iterations")
    max_execution_time: int = Field(60, description="Maximum execution time in seconds")
    verbose: bool = Field(True, description="Enable verbose logging")
    
    # Weather API Configuration
    cache_duration: int = Field(600, description="Weather data cache duration in seconds")
    rate_limit_per_minute: int = Field(60, description="API rate limit per minute")
    request_timeout: int = Field(10, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum API retries")
    
    # Memory Configuration
    memory_window_size: int = Field(10, description="Conversation memory window size")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @validator('max_iterations')
    def validate_max_iterations(cls, v):
        """Validate max iterations is reasonable."""
        if not 1 <= v <= 20:
            raise ValueError("Max iterations must be between 1 and 20")
        return v
    
    @validator('cache_duration')
    def validate_cache_duration(cls, v):
        """Validate cache duration is reasonable."""
        if not 60 <= v <= 3600:  # 1 minute to 1 hour
            raise ValueError("Cache duration must be between 60 and 3600 seconds")
        return v
    
    @classmethod
    def from_env(cls) -> "WeatherAgentConfig":
        """Create configuration from environment variables."""
        try:
            return cls(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
                model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
                max_execution_time=int(os.getenv("MAX_EXECUTION_TIME", "60")),
                verbose=os.getenv("VERBOSE", "true").lower() == "true",
                cache_duration=int(os.getenv("CACHE_DURATION", "600")),
                rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "10")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                memory_window_size=int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
            )
        except Exception as e:
            logger.error(f"Failed to create configuration from environment: {e}")
            raise


class ProductionConfig(WeatherAgentConfig):
    """Production-specific configuration."""
    
    # Production overrides
    verbose: bool = Field(False, description="Disable verbose logging in production")
    log_level: str = Field("INFO", description="Logging level")
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    api_rate_limit: int = Field(1000, description="Daily API request limit")
    cache_backend: str = Field("memory", description="Cache backend for production")
    redis_url: Optional[str] = Field(None, description="Redis URL for caching")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if environment is properly configured
    """
    required_vars = ["OPENAI_API_KEY", "OPENWEATHER_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please create a .env file with the following variables:")
        logger.info("OPENAI_API_KEY=your_openai_api_key")
        logger.info("OPENWEATHER_API_KEY=your_openweather_api_key")
        return False
    
    logger.info("Environment validation passed")
    return True


def get_config(production: bool = False) -> WeatherAgentConfig:
    """
    Get weather agent configuration.
    
    Args:
        production: Whether to use production configuration
        
    Returns:
        WeatherAgentConfig: Configured settings
        
    Raises:
        ValueError: If environment validation fails
    """
    if not validate_environment():
        raise ValueError("Environment validation failed")
    
    config_class = ProductionConfig if production else WeatherAgentConfig
    
    try:
        config = config_class.from_env()
        logger.info(f"Configuration loaded successfully: {config_class.__name__}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


if __name__ == "__main__":
    """Test configuration loading."""
    try:
        print("Testing configuration loading...")
        
        # Test environment validation
        if validate_environment():
            print("✅ Environment validation passed")
        else:
            print("❌ Environment validation failed")
            exit(1)
        
        # Test configuration loading
        config = get_config(production=False)
        print(f"✅ Configuration loaded: {config.model_name}")
        
        # Test production configuration
        prod_config = get_config(production=True)
        print(f"✅ Production configuration loaded: {prod_config.log_level}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        exit(1)