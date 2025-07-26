"""
Weather Data Processing and Response Formatting

This module provides utilities for processing weather data and formatting responses,
following the patterns from examples/basic_chain.py and examples/tools/custom_tool.py.

Key Features:
- Structured response models for consistent output
- Temperature unit conversion utilities
- Weather condition interpretation
- Location disambiguation and geocoding support
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)


class WeatherResponse(BaseModel):
    """Structured weather response model for consistent API responses."""
    
    success: bool = Field(description="Whether the request was successful")
    location: str = Field(description="Location name")
    weather_data: Optional[Dict[str, Any]] = Field(default=None, description="Raw weather data")
    formatted_response: Optional[str] = Field(default=None, description="Human-readable response") 
    error: Optional[str] = Field(default=None, description="Error message if failed")
    recommendations: Optional[List[str]] = Field(default=None, description="Weather-based recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    data_source: str = Field(default="OpenWeatherMap", description="Data source")
    units: str = Field(default="metric", description="Temperature units used")


class TemperatureConverter:
    """Utility class for temperature unit conversions."""
    
    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32
    
    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5/9
    
    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin."""
        return celsius + 273.15
    
    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15
    
    @staticmethod
    def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between units.
        
        Args:
            temp: Temperature value
            from_unit: Source unit ('celsius', 'fahrenheit', 'kelvin')
            to_unit: Target unit ('celsius', 'fahrenheit', 'kelvin')
            
        Returns:
            Converted temperature
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit == to_unit:
            return temp
        
        # Convert to Celsius first
        if from_unit == 'fahrenheit':
            temp_c = TemperatureConverter.fahrenheit_to_celsius(temp)
        elif from_unit == 'kelvin':
            temp_c = TemperatureConverter.kelvin_to_celsius(temp)
        else:  # celsius
            temp_c = temp
        
        # Convert from Celsius to target
        if to_unit == 'fahrenheit':
            return TemperatureConverter.celsius_to_fahrenheit(temp_c)
        elif to_unit == 'kelvin':
            return TemperatureConverter.celsius_to_kelvin(temp_c)
        else:  # celsius
            return temp_c


class WeatherConditionInterpreter:
    """Utility class for interpreting weather conditions and codes."""
    
    # Weather condition mappings for recommendations
    CONDITION_RECOMMENDATIONS = {
        'clear': {
            'clothing': ['Comfortable clothing', 'Sunglasses recommended'],
            'activities': ['Great for outdoor activities', 'Perfect for walking or sports'],
            'travel': ['Excellent travel conditions']
        },
        'clouds': {
            'clothing': ['Light layers recommended'],
            'activities': ['Good for most outdoor activities'],
            'travel': ['Good travel conditions']
        },
        'rain': {
            'clothing': ['Umbrella or raincoat essential', 'Waterproof shoes recommended'],
            'activities': ['Indoor activities preferred', 'Avoid outdoor sports'],
            'travel': ['Drive carefully, expect delays']
        },
        'snow': {
            'clothing': ['Warm winter clothing', 'Waterproof boots essential'],
            'activities': ['Winter sports weather', 'Avoid unnecessary travel'],
            'travel': ['Winter driving conditions, check road reports']
        },
        'thunderstorm': {
            'clothing': ['Stay indoors if possible', 'Waterproof gear if going out'],
            'activities': ['Stay indoors', 'Avoid metal objects and high places'],
            'travel': ['Avoid travel if possible, severe weather conditions']
        },
        'fog': {
            'clothing': ['Normal clothing'],
            'activities': ['Reduced visibility activities'],
            'travel': ['Drive slowly, use fog lights']
        }
    }
    
    @staticmethod
    def get_condition_category(description: str) -> str:
        """
        Categorize weather condition from description.
        
        Args:
            description: Weather condition description
            
        Returns:
            Category string
        """
        description = description.lower()
        
        if 'clear' in description or 'sunny' in description:
            return 'clear'
        elif 'cloud' in description or 'overcast' in description:
            return 'clouds'
        elif 'rain' in description or 'drizzle' in description:
            return 'rain'
        elif 'snow' in description or 'blizzard' in description:
            return 'snow'
        elif 'thunder' in description or 'storm' in description:
            return 'thunderstorm'
        elif 'fog' in description or 'mist' in description:
            return 'fog'
        else:
            return 'other'
    
    @staticmethod
    def get_recommendations(condition: str, temperature: Optional[float] = None) -> Dict[str, List[str]]:
        """
        Get recommendations based on weather condition and temperature.
        
        Args:
            condition: Weather condition description
            temperature: Temperature in Celsius (optional)
            
        Returns:
            Dictionary with clothing, activities, and travel recommendations
        """
        category = WeatherConditionInterpreter.get_condition_category(condition)
        recommendations = WeatherConditionInterpreter.CONDITION_RECOMMENDATIONS.get(
            category, 
            {'clothing': ['Check current conditions'], 'activities': ['Use your judgment'], 'travel': ['Normal conditions']}
        ).copy()
        
        # Add temperature-based recommendations
        if temperature is not None:
            if temperature < 0:
                recommendations['clothing'].insert(0, 'Very cold - heavy winter clothing essential')
            elif temperature < 10:
                recommendations['clothing'].insert(0, 'Cold - warm clothing recommended')
            elif temperature > 30:
                recommendations['clothing'].insert(0, 'Hot - light, breathable clothing')
                recommendations['activities'].append('Stay hydrated, avoid midday sun')
        
        return recommendations


class LocationProcessor:
    """Utility class for processing and validating location data."""
    
    @staticmethod
    def clean_location_name(location: str) -> str:
        """
        Clean and standardize location name.
        
        Args:
            location: Raw location string
            
        Returns:
            Cleaned location string
        """
        if not location:
            return ""
        
        # Remove extra whitespace and special characters
        cleaned = re.sub(r'[<>\"\'&]', '', location.strip())
        
        # Limit length
        cleaned = cleaned[:100]
        
        # Capitalize properly
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())
        
        return cleaned
    
    @staticmethod
    def parse_coordinates(location: str) -> Optional[Dict[str, float]]:
        """
        Parse coordinates from location string.
        
        Args:
            location: Location string that might contain coordinates
            
        Returns:
            Dictionary with lat/lon if coordinates found, None otherwise
        """
        # Look for coordinate patterns like "40.7128,-74.0060" or "40.7128, -74.0060"
        coord_pattern = r'^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)$'
        match = re.match(coord_pattern, location.strip())
        
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                
                # Validate coordinate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return {"lat": lat, "lon": lon}
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def format_location_display(location_data: Dict[str, Any]) -> str:
        """
        Format location data for display.
        
        Args:
            location_data: Location data from geocoding API
            
        Returns:
            Formatted location string
        """
        try:
            name = location_data.get('name', 'Unknown')
            country = location_data.get('country', '')
            state = location_data.get('state', '')
            
            if state and country:
                return f"{name}, {state}, {country}"
            elif country:
                return f"{name}, {country}"
            else:
                return name
                
        except Exception as e:
            logger.error(f"Error formatting location display: {e}")
            return str(location_data.get('name', 'Unknown Location'))


class WeatherDataFormatter:
    """Main formatter class for weather data and responses."""
    
    @staticmethod
    def format_current_weather(
        data: Dict[str, Any], 
        units: str = "metric",
        include_recommendations: bool = True
    ) -> WeatherResponse:
        """
        Format current weather data into structured response.
        
        Args:
            data: Raw weather data from API
            units: Temperature units
            include_recommendations: Whether to include recommendations
            
        Returns:
            Structured weather response
        """
        try:
            location = data['name']
            country = data['sys']['country']
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            description = data['weather'][0]['description'].title()
            humidity = data['main']['humidity']
            wind_speed = data['wind'].get('speed', 0)
            pressure = data['main'].get('pressure', 0)
            
            # Format temperature symbols
            unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
            wind_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"
            
            # Build formatted response
            formatted_parts = [
                f"🌤️ **Current weather in {location}, {country}**",
                "",
                f"🌡️ **Temperature:** {temp:.1f}{unit_symbol}"
            ]
            
            if abs(temp - feels_like) > 2:
                formatted_parts.append(f"   *(feels like {feels_like:.1f}{unit_symbol})*")
            
            formatted_parts.extend([
                f"☁️ **Conditions:** {description}",
                f"💧 **Humidity:** {humidity}%"
            ])
            
            if wind_speed > 0:
                formatted_parts.append(f"💨 **Wind:** {wind_speed:.1f} {wind_unit}")
            
            if pressure > 0:
                formatted_parts.append(f"📊 **Pressure:** {pressure} hPa")
            
            formatted_response = "\\n".join(formatted_parts)
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                temp_celsius = temp if units == "metric" else TemperatureConverter.fahrenheit_to_celsius(temp) if units == "imperial" else TemperatureConverter.kelvin_to_celsius(temp)
                recs = WeatherConditionInterpreter.get_recommendations(description, temp_celsius)
                
                rec_parts = ["\\n**💡 Recommendations:**"]
                if recs.get('clothing'):
                    rec_parts.append(f"👕 **Clothing:** {', '.join(recs['clothing'])}")
                if recs.get('activities'):
                    rec_parts.append(f"🏃 **Activities:** {', '.join(recs['activities'])}")
                if recs.get('travel'):
                    rec_parts.append(f"🚗 **Travel:** {', '.join(recs['travel'])}")
                
                formatted_response += "\\n" + "\\n".join(rec_parts)
                recommendations = [item for sublist in recs.values() for item in sublist]
            
            return WeatherResponse(
                success=True,
                location=f"{location}, {country}",
                weather_data=data,
                formatted_response=formatted_response,
                recommendations=recommendations,
                units=units
            )
            
        except KeyError as e:
            logger.error(f"Missing required field in weather data: {e}")
            return WeatherResponse(
                success=False,
                location="Unknown",
                error=f"Invalid weather data format: missing field {e}"
            )
        except Exception as e:
            logger.error(f"Error formatting weather data: {e}")
            return WeatherResponse(
                success=False,
                location="Unknown", 
                error=f"Error formatting weather data: {str(e)}"
            )
    
    @staticmethod
    def format_forecast_data(
        data: Dict[str, Any],
        days: int = 5,
        units: str = "metric"
    ) -> WeatherResponse:
        """
        Format forecast data into structured response.
        
        Args:
            data: Raw forecast data from API
            days: Number of days to include
            units: Temperature units
            
        Returns:
            Structured forecast response
        """
        try:
            location = data['city']['name']
            country = data['city']['country']
            forecasts = data['list']
            
            unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
            
            formatted_parts = [
                f"📅 **{days}-day weather forecast for {location}, {country}**",
                ""
            ]
            
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
                
                day_parts = [
                    f"**📅 {day_name}**",
                    f"   🌡️ {temp_min:.0f}{unit_symbol} - {temp_max:.0f}{unit_symbol} *(midday: {temp:.0f}{unit_symbol})*",
                    f"   ☁️ {description}"
                ]
                
                if i < len(list(daily_forecasts.items())[:days]) - 1:
                    day_parts.append("")
                
                formatted_parts.extend(day_parts)
            
            formatted_response = "\\n".join(formatted_parts)
            
            return WeatherResponse(
                success=True,
                location=f"{location}, {country}",
                weather_data=data,
                formatted_response=formatted_response,
                units=units
            )
            
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error formatting forecast data: {e}")
            return WeatherResponse(
                success=False,
                location="Unknown",
                error=f"Error formatting forecast data: {str(e)}"
            )


# Utility functions for quick access
def format_weather_data(raw_data: Dict[str, Any], units: str = "metric") -> str:
    """
    Quick utility to format weather data into human-readable text.
    
    Args:
        raw_data: Raw weather data from API
        units: Temperature units
        
    Returns:
        Formatted weather string
    """
    response = WeatherDataFormatter.format_current_weather(raw_data, units)
    return response.formatted_response or response.error or "Error formatting weather data"


def get_weather_recommendations(condition: str, temperature: Optional[float] = None) -> List[str]:
    """
    Quick utility to get weather recommendations.
    
    Args:
        condition: Weather condition description
        temperature: Temperature in Celsius (optional)
        
    Returns:
        List of recommendations
    """
    recs = WeatherConditionInterpreter.get_recommendations(condition, temperature)
    return [item for sublist in recs.values() for item in sublist]


if __name__ == "__main__":
    """Test weather data formatting functionality."""
    
    # Test temperature conversion
    print("Testing temperature conversion:")
    print(f"32°F = {TemperatureConverter.fahrenheit_to_celsius(32):.1f}°C")
    print(f"0°C = {TemperatureConverter.celsius_to_fahrenheit(0):.1f}°F")
    print(f"273.15K = {TemperatureConverter.kelvin_to_celsius(273.15):.1f}°C")
    
    # Test weather condition interpretation
    print("\\nTesting weather recommendations:")
    recs = WeatherConditionInterpreter.get_recommendations("light rain", 15)
    for category, items in recs.items():
        print(f"{category.title()}: {', '.join(items)}")
    
    # Test location processing
    print("\\nTesting location processing:")
    print(f"Cleaned: {LocationProcessor.clean_location_name('  new YORK city  ')}")
    coords = LocationProcessor.parse_coordinates("40.7128, -74.0060")
    print(f"Coordinates: {coords}")
    
    print("\\n✅ Weather data formatting utilities working correctly!")