"""
Test suite for weather data formatting utilities.

This module contains tests for weather data processing and formatting functions.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from utils.weather_formatter import (
    WeatherResponse,
    TemperatureConverter,
    WeatherConditionInterpreter,
    LocationProcessor,
    WeatherDataFormatter,
    format_weather_data,
    get_weather_recommendations
)


class TestWeatherResponse:
    """Test WeatherResponse model."""
    
    def test_successful_response(self):
        """Test creating successful weather response."""
        response = WeatherResponse(
            success=True,
            location="London, UK",
            formatted_response="Weather in London: 15°C, cloudy"
        )
        
        assert response.success is True
        assert response.location == "London, UK"
        assert "15°C" in response.formatted_response
        assert response.data_source == "OpenWeatherMap"
        assert response.units == "metric"
    
    def test_error_response(self):
        """Test creating error weather response."""
        response = WeatherResponse(
            success=False,
            location="Unknown",
            error="Location not found"
        )
        
        assert response.success is False
        assert response.error == "Location not found"


class TestTemperatureConverter:
    """Test temperature conversion utilities."""
    
    def test_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        assert TemperatureConverter.celsius_to_fahrenheit(0) == 32.0
        assert TemperatureConverter.celsius_to_fahrenheit(100) == 212.0
        assert abs(TemperatureConverter.celsius_to_fahrenheit(25) - 77.0) < 0.1
    
    def test_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        assert TemperatureConverter.fahrenheit_to_celsius(32) == 0.0
        assert TemperatureConverter.fahrenheit_to_celsius(212) == 100.0
        assert abs(TemperatureConverter.fahrenheit_to_celsius(77) - 25.0) < 0.1
    
    def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        assert TemperatureConverter.celsius_to_kelvin(0) == 273.15
        assert TemperatureConverter.celsius_to_kelvin(-273.15) == 0.0
    
    def test_kelvin_to_celsius(self):
        """Test Kelvin to Celsius conversion."""
        assert TemperatureConverter.kelvin_to_celsius(273.15) == 0.0
        assert TemperatureConverter.kelvin_to_celsius(373.15) == 100.0
    
    def test_convert_temperature(self):
        """Test general temperature conversion."""
        # Same unit
        assert TemperatureConverter.convert_temperature(25, "celsius", "celsius") == 25
        
        # Celsius to Fahrenheit
        result = TemperatureConverter.convert_temperature(25, "celsius", "fahrenheit")
        assert abs(result - 77.0) < 0.1
        
        # Fahrenheit to Celsius
        result = TemperatureConverter.convert_temperature(77, "fahrenheit", "celsius")
        assert abs(result - 25.0) < 0.1
        
        # Celsius to Kelvin
        result = TemperatureConverter.convert_temperature(0, "celsius", "kelvin")
        assert result == 273.15


class TestWeatherConditionInterpreter:
    """Test weather condition interpretation."""
    
    def test_get_condition_category(self):
        """Test weather condition categorization."""
        assert WeatherConditionInterpreter.get_condition_category("clear sky") == "clear"
        assert WeatherConditionInterpreter.get_condition_category("sunny") == "clear"
        assert WeatherConditionInterpreter.get_condition_category("cloudy") == "clouds"
        assert WeatherConditionInterpreter.get_condition_category("light rain") == "rain"
        assert WeatherConditionInterpreter.get_condition_category("snow") == "snow"
        assert WeatherConditionInterpreter.get_condition_category("thunderstorm") == "thunderstorm"
        assert WeatherConditionInterpreter.get_condition_category("fog") == "fog"
        assert WeatherConditionInterpreter.get_condition_category("unknown condition") == "other"
    
    def test_get_recommendations_clear_weather(self):
        """Test recommendations for clear weather."""
        recs = WeatherConditionInterpreter.get_recommendations("clear sky", 20)
        
        assert "clothing" in recs
        assert "activities" in recs
        assert "travel" in recs
        assert any("sunglasses" in item.lower() for item in recs["clothing"])
        assert any("outdoor" in item.lower() for item in recs["activities"])
    
    def test_get_recommendations_rain(self):
        """Test recommendations for rainy weather."""
        recs = WeatherConditionInterpreter.get_recommendations("light rain", 15)
        
        assert any("umbrella" in item.lower() or "raincoat" in item.lower() for item in recs["clothing"])
        assert any("indoor" in item.lower() for item in recs["activities"])
    
    def test_get_recommendations_with_temperature(self):
        """Test recommendations with temperature considerations."""
        # Cold weather
        recs = WeatherConditionInterpreter.get_recommendations("clear", -5)
        assert any("very cold" in item.lower() for item in recs["clothing"])
        
        # Hot weather
        recs = WeatherConditionInterpreter.get_recommendations("clear", 35)
        assert any("light" in item.lower() for item in recs["clothing"])
        assert any("hydrated" in item.lower() for item in recs["activities"])


class TestLocationProcessor:
    """Test location processing utilities."""
    
    def test_clean_location_name(self):
        """Test location name cleaning."""
        assert LocationProcessor.clean_location_name("  new york  ") == "New York"
        assert LocationProcessor.clean_location_name("LONDON") == "London"
        assert LocationProcessor.clean_location_name("san francisco") == "San Francisco"
        assert LocationProcessor.clean_location_name("") == ""
        
        # Test special character removal
        assert LocationProcessor.clean_location_name("New<York>") == "NewYork"
        assert LocationProcessor.clean_location_name('City"Name') == "CityName"
    
    def test_clean_location_length_limit(self):
        """Test location name length limiting."""
        long_name = "a" * 150
        cleaned = LocationProcessor.clean_location_name(long_name)
        assert len(cleaned) <= 100
    
    def test_parse_coordinates(self):
        """Test coordinate parsing."""
        # Valid coordinates
        coords = LocationProcessor.parse_coordinates("40.7128,-74.0060")
        assert coords == {"lat": 40.7128, "lon": -74.0060}
        
        coords = LocationProcessor.parse_coordinates("40.7128, -74.0060")
        assert coords == {"lat": 40.7128, "lon": -74.0060}
        
        # Invalid coordinates
        assert LocationProcessor.parse_coordinates("New York") is None
        assert LocationProcessor.parse_coordinates("invalid,coords") is None
        assert LocationProcessor.parse_coordinates("91.0,0.0") is None  # Invalid latitude
        assert LocationProcessor.parse_coordinates("0.0,181.0") is None  # Invalid longitude
    
    def test_format_location_display(self):
        """Test location display formatting."""
        # Full location data
        location_data = {
            "name": "New York",
            "state": "NY",
            "country": "US"
        }
        result = LocationProcessor.format_location_display(location_data)
        assert result == "New York, NY, US"
        
        # No state
        location_data = {
            "name": "London",
            "country": "UK"
        }
        result = LocationProcessor.format_location_display(location_data)
        assert result == "London, UK"
        
        # Only name
        location_data = {"name": "Paris"}
        result = LocationProcessor.format_location_display(location_data)
        assert result == "Paris"
        
        # Empty data
        location_data = {}
        result = LocationProcessor.format_location_display(location_data)
        assert result == "Unknown Location"


class TestWeatherDataFormatter:
    """Test weather data formatting."""
    
    def test_format_current_weather_success(self):
        """Test successful current weather formatting."""
        weather_data = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 80,
                "pressure": 1013
            },
            "weather": [{"description": "light rain"}],
            "wind": {"speed": 3.5}
        }
        
        response = WeatherDataFormatter.format_current_weather(weather_data, "metric", True)
        
        assert response.success is True
        assert response.location == "London, GB"
        assert "15.5°C" in response.formatted_response
        assert "feels like 14.2°C" in response.formatted_response
        assert "light rain" in response.formatted_response.lower()
        assert "80%" in response.formatted_response
        assert "3.5 m/s" in response.formatted_response
        assert response.recommendations is not None
        assert len(response.recommendations) > 0
    
    def test_format_current_weather_imperial_units(self):
        """Test weather formatting with imperial units."""
        weather_data = {
            "name": "New York",
            "sys": {"country": "US"},
            "main": {
                "temp": 68.0,
                "feels_like": 70.0,
                "humidity": 65
            },
            "weather": [{"description": "partly cloudy"}],
            "wind": {"speed": 5.0}
        }
        
        response = WeatherDataFormatter.format_current_weather(weather_data, "imperial", False)
        
        assert response.success is True
        assert "68.0°F" in response.formatted_response
        assert "mph" in response.formatted_response
        assert response.recommendations is None or len(response.recommendations) == 0
    
    def test_format_current_weather_missing_data(self):
        """Test weather formatting with missing data."""
        incomplete_data = {
            "name": "Test City",
            "sys": {"country": "TC"}
            # Missing main weather data
        }
        
        response = WeatherDataFormatter.format_current_weather(incomplete_data)
        
        assert response.success is False
        assert "Invalid weather data format" in response.error
    
    def test_format_forecast_data_success(self):
        """Test successful forecast formatting."""
        forecast_data = {
            "city": {"name": "Paris", "country": "FR"},
            "list": [
                {
                    "dt_txt": "2023-12-01 12:00:00",
                    "main": {"temp": 16, "temp_min": 12, "temp_max": 18},
                    "weather": [{"description": "sunny"}]
                },
                {
                    "dt_txt": "2023-12-01 15:00:00",
                    "main": {"temp": 17, "temp_min": 12, "temp_max": 18},
                    "weather": [{"description": "sunny"}]
                },
                {
                    "dt_txt": "2023-12-02 12:00:00",
                    "main": {"temp": 14, "temp_min": 10, "temp_max": 16},
                    "weather": [{"description": "cloudy"}]
                }
            ]
        }
        
        response = WeatherDataFormatter.format_forecast_data(forecast_data, 2, "metric")
        
        assert response.success is True
        assert response.location == "Paris, FR"
        assert "forecast" in response.formatted_response.lower()
        assert "sunny" in response.formatted_response.lower()
        assert "cloudy" in response.formatted_response.lower()
    
    def test_format_forecast_data_invalid(self):
        """Test forecast formatting with invalid data."""
        invalid_data = {"invalid": "data"}
        
        response = WeatherDataFormatter.format_forecast_data(invalid_data)
        
        assert response.success is False
        assert "Error formatting forecast data" in response.error


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_format_weather_data_function(self):
        """Test format_weather_data utility function."""
        weather_data = {
            "name": "Tokyo",
            "sys": {"country": "JP"},
            "main": {"temp": 22, "feels_like": 24, "humidity": 75},
            "weather": [{"description": "clear"}],
            "wind": {"speed": 2.1}
        }
        
        result = format_weather_data(weather_data, "metric")
        
        assert "Tokyo, JP" in result
        assert "22°C" in result
        assert "clear" in result.lower()
    
    def test_get_weather_recommendations_function(self):
        """Test get_weather_recommendations utility function."""
        recommendations = get_weather_recommendations("rain", 10)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("umbrella" in rec.lower() or "raincoat" in rec.lower() for rec in recommendations)


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_extreme_temperatures(self):
        """Test handling of extreme temperatures."""
        # Very hot
        recs = WeatherConditionInterpreter.get_recommendations("clear", 50)
        assert any("hot" in item.lower() for item in recs["clothing"])
        
        # Very cold
        recs = WeatherConditionInterpreter.get_recommendations("clear", -30)
        assert any("very cold" in item.lower() for item in recs["clothing"])
    
    def test_missing_optional_fields(self):
        """Test handling of missing optional weather fields."""
        minimal_data = {
            "name": "Test City",
            "sys": {"country": "TC"},
            "main": {"temp": 20, "feels_like": 20, "humidity": 50},
            "weather": [{"description": "clear"}],
            "wind": {}  # Empty wind data
        }
        
        response = WeatherDataFormatter.format_current_weather(minimal_data)
        
        assert response.success is True
        assert "Test City, TC" in response.formatted_response
        # Should handle missing wind speed gracefully
    
    def test_unicode_location_names(self):
        """Test handling of unicode characters in location names."""
        unicode_location = "北京"  # Beijing in Chinese
        cleaned = LocationProcessor.clean_location_name(unicode_location)
        assert cleaned == "北京"
        
        # Test with mixed characters
        mixed_location = "São Paulo"
        cleaned = LocationProcessor.clean_location_name(mixed_location)
        assert cleaned == "São Paulo"


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v"])