"""
Weather Tools Implementation

This module implements custom LangChain tools for weather functionality,
following the BaseTool patterns from examples/tools/custom_tool.py.

Key Features:
- WeatherTool for current conditions
- ForecastTool for extended predictions
- AlertTool for severe weather monitoring
- LocationTool for geocoding
- Comprehensive error handling and caching
"""

import os
import json
import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Type
from functools import wraps

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Input Schemas

class WeatherQueryInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(description="City name, coordinates, or address")
    units: str = Field(default="metric", description="Temperature units: metric, imperial, kelvin")
    
    @field_validator('units')
    @classmethod
    def validate_units(cls, v):
        allowed_units = ['metric', 'imperial', 'kelvin']
        if v not in allowed_units:
            raise ValueError(f"units must be one of {allowed_units}")
        return v
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if not v or not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip()[:100]  # Limit length


class ForecastQueryInput(BaseModel):
    """Input schema for forecast queries."""
    location: str = Field(description="City name, coordinates, or address") 
    days: int = Field(default=5, description="Number of forecast days (1-5)")
    units: str = Field(default="metric", description="Temperature units: metric, imperial, kelvin")
    
    @field_validator('days')
    @classmethod
    def validate_days(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("Days must be between 1 and 5")
        return v
    
    @field_validator('units')
    @classmethod
    def validate_units(cls, v):
        allowed_units = ['metric', 'imperial', 'kelvin']
        if v not in allowed_units:
            raise ValueError(f"units must be one of {allowed_units}")
        return v
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if not v or not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip()[:100]


class LocationQueryInput(BaseModel):
    """Input schema for location/geocoding queries."""
    location: str = Field(description="Location name to resolve coordinates for")
    limit: int = Field(default=5, description="Maximum number of results to return")
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Limit must be between 1 and 10")
        return v
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if not v or not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip()[:100]


# Error Handling Decorators

def safe_weather_request(max_retries: int = 3):
    """Decorator for safe weather API requests with retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.Timeout as e:
                    last_exception = e
                    logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                except requests.exceptions.HTTPError as e:
                    if e.response and e.response.status_code in [401, 403]:
                        # API key issues - don't retry
                        raise WeatherError(f"API authentication failed: {e}")
                    last_exception = e
                    logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            
            raise WeatherError(f"Request failed after {max_retries} attempts: {last_exception}")
        
        return wrapper
    return decorator


class WeatherError(Exception):
    """Custom exception for weather-related errors."""
    pass


# Weather Tools Implementation

class WeatherTool(BaseTool):
    """Tool for current weather conditions using OpenWeatherMap API."""
    
    name: str = "current_weather"
    description: str = """Get current weather conditions for any location worldwide. 
    Input should be a location (city name, coordinates, or address) and optional temperature units.
    Returns temperature, conditions, humidity, wind speed, and other weather details."""
    args_schema: Type[BaseModel] = WeatherQueryInput
    
    def __init__(self, api_key: str, cache_duration: int = 600):
        """
        Initialize weather tool.
        
        Args:
            api_key: OpenWeatherMap API key
            cache_duration: Cache duration in seconds (default 10 minutes)
        """
        super().__init__()
        # Store configuration as private attributes
        self._api_key = api_key
        self._base_url = "http://api.openweathermap.org/data/2.5/weather"
        self._cache = {}  # Simple in-memory cache
        self._cache_duration = cache_duration
    
    @safe_weather_request(max_retries=3)
    def _make_weather_request(self, location: str, units: str) -> Dict[str, Any]:
        """Make weather API request with error handling."""
        params = {
            'q': location,
            'appid': self._api_key,
            'units': units
        }
        
        response = requests.get(self._base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _format_weather_data(self, data: Dict[str, Any], units: str) -> str:
        """Format weather API response into human-readable text."""
        try:
            location = data['name']
            country = data['sys']['country']
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            description = data['weather'][0]['description'].title()
            humidity = data['main']['humidity']
            wind_speed = data['wind'].get('speed', 0)
            pressure = data['main'].get('pressure', 0)
            
            # Temperature unit formatting
            unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
            wind_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"
            
            formatted = f"Current weather in {location}, {country}:\\n"
            formatted += f"🌡️ Temperature: {temp}{unit_symbol}"
            
            if abs(temp - feels_like) > 2:
                formatted += f" (feels like {feels_like}{unit_symbol})"
            
            formatted += f"\\n☁️ Conditions: {description}"
            formatted += f"\\n💧 Humidity: {humidity}%"
            
            if wind_speed > 0:
                formatted += f"\\n💨 Wind: {wind_speed} {wind_unit}"
            
            if pressure > 0:
                formatted += f"\\n📊 Pressure: {pressure} hPa"
            
            return formatted
            
        except KeyError as e:
            raise WeatherError(f"Error formatting weather data: missing field {e}")
    
    def _run(
        self, 
        location: str, 
        units: str = "metric",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute weather lookup with caching and error handling."""
        try:
            # Check cache
            cache_key = f"{location.lower()}-{units}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self._cache_duration):
                    logger.info(f"Returning cached weather data for {location}")
                    return cached_data
            
            # Make API request
            logger.info(f"Fetching weather data for {location}")
            data = self._make_weather_request(location, units)
            
            # Format result
            formatted_result = self._format_weather_data(data, units)
            
            # Cache the result
            self._cache[cache_key] = (formatted_result, datetime.now())
            
            return formatted_result
            
        except WeatherError:
            # Re-raise weather errors as-is
            raise
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                return f"❌ Error: Location '{location}' not found. Please check the spelling or try a different location."
            return f"❌ Error: Weather service unavailable (HTTP {e.response.status_code if e.response else 'unknown'})"
        except Exception as e:
            logger.error(f"Unexpected error in weather lookup: {e}")
            return f"❌ Error retrieving weather data: {str(e)}"
    
    async def _arun(
        self,
        location: str,
        units: str = "metric",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of weather lookup (calls sync version for now)."""
        # For now, just call the sync version
        # In a real implementation, you'd use aiohttp for async requests
        return self._run(location, units, run_manager)


class ForecastTool(BaseTool):
    """Tool for weather forecasts using OpenWeatherMap API."""
    
    name: str = "weather_forecast"
    description: str = """Get weather forecast for any location (1-5 days ahead).
    Input should be a location and number of days (1-5).
    Returns detailed forecast with temperatures, conditions, and weather patterns."""
    args_schema: Type[BaseModel] = ForecastQueryInput
    
    def __init__(self, api_key: str, cache_duration: int = 3600):
        """
        Initialize forecast tool.
        
        Args:
            api_key: OpenWeatherMap API key
            cache_duration: Cache duration in seconds (default 1 hour)
        """
        super().__init__()
        self._api_key = api_key
        self._base_url = "http://api.openweathermap.org/data/2.5/forecast"
        self._cache = {}
        self._cache_duration = cache_duration
    
    @safe_weather_request(max_retries=3)
    def _make_forecast_request(self, location: str, units: str) -> Dict[str, Any]:
        """Make forecast API request with error handling."""
        params = {
            'q': location,
            'appid': self._api_key,
            'units': units
        }
        
        response = requests.get(self._base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _format_forecast_data(self, data: Dict[str, Any], days: int, units: str) -> str:
        """Format forecast API response into human-readable text."""
        try:
            location = data['city']['name']
            country = data['city']['country']
            forecasts = data['list']
            
            unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
            
            formatted = f"🌤️ {days}-day weather forecast for {location}, {country}:\\n\\n"
            
            # Group forecasts by day
            daily_forecasts = {}
            for forecast in forecasts[:days * 8]:  # 8 forecasts per day (3-hour intervals)
                date_str = forecast['dt_txt'].split(' ')[0]
                if date_str not in daily_forecasts:
                    daily_forecasts[date_str] = []
                daily_forecasts[date_str].append(forecast)
            
            # Format each day
            for i, (date_str, day_forecasts) in enumerate(list(daily_forecasts.items())[:days]):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                day_name = date_obj.strftime('%A, %B %d')
                
                # Get daily summary (use midday forecast if available)
                midday_forecast = None
                for f in day_forecasts:
                    hour = int(f['dt_txt'].split(' ')[1].split(':')[0])
                    if 11 <= hour <= 13:  # Around noon
                        midday_forecast = f
                        break
                
                if not midday_forecast:
                    midday_forecast = day_forecasts[len(day_forecasts)//2]  # Middle forecast
                
                temp = midday_forecast['main']['temp']
                temp_min = min(f['main']['temp_min'] for f in day_forecasts)
                temp_max = max(f['main']['temp_max'] for f in day_forecasts)
                description = midday_forecast['weather'][0]['description'].title()
                
                formatted += f"📅 {day_name}\\n"
                formatted += f"   🌡️ {temp_min:.0f}{unit_symbol} - {temp_max:.0f}{unit_symbol} (midday: {temp:.0f}{unit_symbol})\\n"
                formatted += f"   ☁️ {description}\\n"
                
                if i < len(daily_forecasts) - 1:
                    formatted += "\\n"
            
            return formatted
            
        except (KeyError, IndexError) as e:
            raise WeatherError(f"Error formatting forecast data: {e}")
    
    def _run(
        self,
        location: str,
        days: int = 5,
        units: str = "metric",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute forecast lookup with caching and error handling."""
        try:
            # Check cache
            cache_key = f"forecast-{location.lower()}-{days}-{units}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self._cache_duration):
                    logger.info(f"Returning cached forecast data for {location}")
                    return cached_data
            
            # Make API request
            logger.info(f"Fetching forecast data for {location} ({days} days)")
            data = self._make_forecast_request(location, units)
            
            # Format result
            formatted_result = self._format_forecast_data(data, days, units)
            
            # Cache the result
            self._cache[cache_key] = (formatted_result, datetime.now())
            
            return formatted_result
            
        except WeatherError:
            raise
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                return f"❌ Error: Location '{location}' not found for forecast. Please check the spelling."
            return f"❌ Error: Forecast service unavailable (HTTP {e.response.status_code if e.response else 'unknown'})"
        except Exception as e:
            logger.error(f"Unexpected error in forecast lookup: {e}")
            return f"❌ Error retrieving forecast data: {str(e)}"
    
    async def _arun(
        self,
        location: str,
        days: int = 5,
        units: str = "metric",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of forecast lookup."""
        return self._run(location, days, units, run_manager)


class LocationTool(BaseTool):
    """Tool for geocoding and location resolution using OpenWeatherMap API."""
    
    name: str = "location_lookup"
    description: str = """Look up location coordinates and get detailed location information.
    Input should be a location name. Returns coordinates, country, state/region info."""
    args_schema: Type[BaseModel] = LocationQueryInput
    
    def __init__(self, api_key: str, cache_duration: int = 86400):  # 24 hours
        """
        Initialize location tool.
        
        Args:
            api_key: OpenWeatherMap API key
            cache_duration: Cache duration in seconds (default 24 hours)
        """
        super().__init__()
        self._api_key = api_key
        self._base_url = "http://api.openweathermap.org/geo/1.0/direct"
        self._cache = {}
        self._cache_duration = cache_duration
    
    @safe_weather_request(max_retries=3)
    def _make_geocoding_request(self, location: str, limit: int) -> List[Dict[str, Any]]:
        """Make geocoding API request with error handling."""
        params = {
            'q': location,
            'limit': limit,
            'appid': self._api_key
        }
        
        response = requests.get(self._base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _format_location_data(self, locations: List[Dict[str, Any]]) -> str:
        """Format location data into human-readable text."""
        if not locations:
            return "❌ No locations found. Please try a different search term."
        
        formatted = f"📍 Found {len(locations)} location(s):\\n\\n"
        
        for i, loc in enumerate(locations, 1):
            name = loc.get('name', 'Unknown')
            country = loc.get('country', 'Unknown')
            state = loc.get('state', '')
            lat = loc.get('lat', 0)
            lon = loc.get('lon', 0)
            
            formatted += f"{i}. {name}"
            if state:
                formatted += f", {state}"
            formatted += f", {country}\\n"
            formatted += f"   📍 Coordinates: {lat:.4f}, {lon:.4f}\\n"
            
            if i < len(locations):
                formatted += "\\n"
        
        return formatted
    
    def _run(
        self,
        location: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute location lookup with caching and error handling."""
        try:
            # Check cache
            cache_key = f"location-{location.lower()}-{limit}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self._cache_duration):
                    logger.info(f"Returning cached location data for {location}")
                    return cached_data
            
            # Make API request
            logger.info(f"Looking up location: {location}")
            locations = self._make_geocoding_request(location, limit)
            
            # Format result
            formatted_result = self._format_location_data(locations)
            
            # Cache the result
            self._cache[cache_key] = (formatted_result, datetime.now())
            
            return formatted_result
            
        except WeatherError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in location lookup: {e}")
            return f"❌ Error looking up location: {str(e)}"
    
    async def _arun(
        self,
        location: str,
        limit: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of location lookup."""
        return self._run(location, limit, run_manager)


class AlertTool(BaseTool):
    """Tool for weather alerts and warnings (mock implementation for now)."""
    
    name: str = "weather_alerts"
    description: str = """Check for weather alerts and warnings for a location.
    Input should be a location name. Returns any active weather alerts, warnings, or advisories."""
    args_schema: Type[BaseModel] = LocationQueryInput
    
    def __init__(self, api_key: str):
        """Initialize alert tool."""
        super().__init__()
        self._api_key = api_key
    
    def _run(
        self,
        location: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Check for weather alerts (mock implementation)."""
        # NOTE: This is a simplified implementation
        # In a real system, you'd use a proper weather alerts API
        logger.info(f"Checking weather alerts for {location}")
        
        try:
            # For now, return a mock response
            # In production, integrate with weather alerts API
            return f"⚠️ Weather alerts for {location}:\\n\\nNo active weather alerts or warnings at this time.\\n\\nNote: This is a basic implementation. For critical weather information, please check official weather services."
            
        except Exception as e:
            logger.error(f"Error checking weather alerts: {e}")
            return f"❌ Error checking weather alerts: {str(e)}"
    
    async def _arun(
        self,
        location: str,
        limit: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of weather alerts check."""
        return self._run(location, limit, run_manager)


# Tool Factory

def create_weather_tools(api_key: str) -> List[BaseTool]:
    """
    Create all weather tools with the given API key.
    
    Args:
        api_key: OpenWeatherMap API key
        
    Returns:
        List of weather tools
    """
    return [
        WeatherTool(api_key=api_key),
        ForecastTool(api_key=api_key),
        LocationTool(api_key=api_key),
        AlertTool(api_key=api_key)
    ]


if __name__ == "__main__":
    """Test weather tools functionality."""
    # This would require actual API keys to test
    print("Weather tools module loaded successfully!")
    print("Available tools:")
    
    tools = create_weather_tools("test_api_key")
    for tool in tools:
        print(f"- {tool.name}: {tool.description.split('.')[0]}")