"""
Test suite for weather tools.

This module contains comprehensive tests for all weather tools,
following the patterns from existing test files.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from tools.weather_tools import (
    WeatherTool,
    ForecastTool,
    LocationTool,
    AlertTool,
    create_weather_tools,
    WeatherError,
    WeatherQueryInput,
    ForecastQueryInput,
    LocationQueryInput
)


class TestWeatherQueryInput:
    """Test weather query input validation."""
    
    def test_valid_input(self):
        """Test valid weather query input."""
        input_data = WeatherQueryInput(location="London", units="metric")
        assert input_data.location == "London"
        assert input_data.units == "metric"
    
    def test_default_units(self):
        """Test default units are set correctly."""
        input_data = WeatherQueryInput(location="Paris")
        assert input_data.units == "metric"
    
    def test_invalid_units(self):
        """Test invalid units raise validation error."""
        with pytest.raises(ValueError, match="units must be one of"):
            WeatherQueryInput(location="London", units="invalid")
    
    def test_empty_location(self):
        """Test empty location raises validation error."""
        with pytest.raises(ValueError, match="Location cannot be empty"):
            WeatherQueryInput(location="", units="metric")
    
    def test_location_length_limit(self):
        """Test location length is limited."""
        long_location = "a" * 150
        input_data = WeatherQueryInput(location=long_location)
        assert len(input_data.location) == 100


class TestForecastQueryInput:
    """Test forecast query input validation."""
    
    def test_valid_input(self):
        """Test valid forecast query input."""
        input_data = ForecastQueryInput(location="Tokyo", days=3, units="imperial")
        assert input_data.location == "Tokyo"
        assert input_data.days == 3
        assert input_data.units == "imperial"
    
    def test_invalid_days(self):
        """Test invalid days raise validation error."""
        with pytest.raises(ValueError, match="Days must be between 1 and 5"):
            ForecastQueryInput(location="Tokyo", days=10)
        
        with pytest.raises(ValueError, match="Days must be between 1 and 5"):
            ForecastQueryInput(location="Tokyo", days=0)


class TestLocationQueryInput:
    """Test location query input validation."""
    
    def test_valid_input(self):
        """Test valid location query input."""
        input_data = LocationQueryInput(location="New York", limit=3)
        assert input_data.location == "New York"
        assert input_data.limit == 3
    
    def test_invalid_limit(self):
        """Test invalid limit raises validation error."""
        with pytest.raises(ValueError, match="Limit must be between 1 and 10"):
            LocationQueryInput(location="Berlin", limit=15)


class TestWeatherTool:
    """Test weather tool functionality."""
    
    @pytest.fixture
    def weather_tool(self):
        """Create weather tool for testing."""
        return WeatherTool(api_key="test_api_key")
    
    def test_tool_creation(self, weather_tool):
        """Test that weather tool can be created."""
        assert weather_tool.name == "current_weather"
        assert "weather conditions" in weather_tool.description.lower()
        assert weather_tool._api_key == "test_api_key"
    
    @patch('requests.get')
    def test_successful_weather_request(self, mock_get, weather_tool):
        """Test successful weather API request."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'name': 'London',
            'sys': {'country': 'GB'},
            'main': {
                'temp': 15.5,
                'feels_like': 14.2,
                'humidity': 80,
                'pressure': 1013
            },
            'weather': [{'description': 'light rain'}],
            'wind': {'speed': 3.5}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the tool
        result = weather_tool._run("London", "metric")
        
        # Verify result
        assert "London, GB" in result
        assert "15.5°C" in result
        assert "light rain" in result.lower()
        assert "80%" in result
        assert "3.5 m/s" in result
        
        # Verify API call
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "q=London" in str(kwargs.get('params', {})) or "London" in str(args)
    
    @patch('requests.get')
    def test_location_not_found(self, mock_get, weather_tool):
        """Test handling of invalid location."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = weather_tool._run("InvalidLocation", "metric")
        
        assert "not found" in result.lower()
        assert "error" in result.lower()
    
    @patch('requests.get')
    def test_api_timeout(self, mock_get, weather_tool):
        """Test handling of API timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = weather_tool._run("London", "metric")
        
        assert "timeout" in result.lower()
        assert "error" in result.lower()
    
    def test_caching_functionality(self, weather_tool):
        """Test weather data caching."""
        # Mock successful response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'name': 'Paris',
                'sys': {'country': 'FR'},
                'main': {'temp': 20, 'feels_like': 19, 'humidity': 65},
                'weather': [{'description': 'clear sky'}],
                'wind': {'speed': 2.1}
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # First call should make API request
            result1 = weather_tool._run("Paris", "metric")
            assert mock_get.call_count == 1
            
            # Second call should use cache
            result2 = weather_tool._run("Paris", "metric")
            assert mock_get.call_count == 1  # No additional API call
            assert result1 == result2
    
    def test_cache_expiration(self, weather_tool):
        """Test cache expiration functionality."""
        # Set very short cache duration for testing
        weather_tool._cache_duration = 1
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'name': 'Berlin',
                'sys': {'country': 'DE'},
                'main': {'temp': 18, 'feels_like': 17, 'humidity': 70},
                'weather': [{'description': 'partly cloudy'}],
                'wind': {'speed': 1.8}
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # First call
            weather_tool._run("Berlin", "metric")
            assert mock_get.call_count == 1
            
            # Wait for cache to expire
            import time
            time.sleep(1.1)
            
            # Second call should make new API request
            weather_tool._run("Berlin", "metric")
            assert mock_get.call_count == 2


class TestForecastTool:
    """Test forecast tool functionality."""
    
    @pytest.fixture
    def forecast_tool(self):
        """Create forecast tool for testing."""
        return ForecastTool(api_key="test_api_key")
    
    def test_tool_creation(self, forecast_tool):
        """Test that forecast tool can be created."""
        assert forecast_tool.name == "weather_forecast"
        assert "forecast" in forecast_tool.description.lower()
        assert forecast_tool._api_key == "test_api_key"
    
    @patch('requests.get')
    def test_successful_forecast_request(self, mock_get, forecast_tool):
        """Test successful forecast API request."""
        # Mock forecast response
        mock_response = Mock()
        mock_response.json.return_value = {
            'city': {'name': 'Madrid', 'country': 'ES'},
            'list': [
                {
                    'dt_txt': '2023-12-01 12:00:00',
                    'main': {'temp': 16, 'temp_min': 12, 'temp_max': 18},
                    'weather': [{'description': 'sunny'}]
                },
                {
                    'dt_txt': '2023-12-02 12:00:00',
                    'main': {'temp': 14, 'temp_min': 10, 'temp_max': 16},
                    'weather': [{'description': 'cloudy'}]
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = forecast_tool._run("Madrid", 2, "metric")
        
        assert "Madrid, ES" in result
        assert "forecast" in result.lower()
        assert "sunny" in result.lower()
        assert "cloudy" in result.lower()


class TestLocationTool:
    """Test location tool functionality."""
    
    @pytest.fixture
    def location_tool(self):
        """Create location tool for testing."""
        return LocationTool(api_key="test_api_key")
    
    def test_tool_creation(self, location_tool):
        """Test that location tool can be created."""
        assert location_tool.name == "location_lookup"
        assert "location" in location_tool.description.lower()
        assert location_tool._api_key == "test_api_key"
    
    @patch('requests.get')
    def test_successful_location_request(self, mock_get, location_tool):
        """Test successful location lookup."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'name': 'Sydney',
                'country': 'AU',
                'state': 'NSW',
                'lat': -33.8688,
                'lon': 151.2093
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = location_tool._run("Sydney", 1)
        
        assert "Sydney" in result
        assert "AU" in result
        assert "NSW" in result
        assert "-33.8688" in result
        assert "151.2093" in result
    
    @patch('requests.get')
    def test_no_locations_found(self, mock_get, location_tool):
        """Test when no locations are found."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = location_tool._run("NonexistentPlace", 1)
        
        assert "no locations found" in result.lower()


class TestAlertTool:
    """Test alert tool functionality."""
    
    @pytest.fixture
    def alert_tool(self):
        """Create alert tool for testing."""
        return AlertTool(api_key="test_api_key")
    
    def test_tool_creation(self, alert_tool):
        """Test that alert tool can be created."""
        assert alert_tool.name == "weather_alerts"
        assert "alerts" in alert_tool.description.lower()
        assert alert_tool._api_key == "test_api_key"
    
    def test_alert_response(self, alert_tool):
        """Test alert tool response."""
        result = alert_tool._run("TestCity", 5)
        
        assert "TestCity" in result
        assert "alerts" in result.lower()
        # Since this is a mock implementation, check for expected mock response
        assert "no active weather alerts" in result.lower()


class TestWeatherToolsFactory:
    """Test weather tools factory function."""
    
    def test_create_weather_tools(self):
        """Test creating all weather tools."""
        tools = create_weather_tools("test_api_key")
        
        assert len(tools) == 4
        assert any(tool.name == "current_weather" for tool in tools)
        assert any(tool.name == "weather_forecast" for tool in tools)
        assert any(tool.name == "location_lookup" for tool in tools)
        assert any(tool.name == "weather_alerts" for tool in tools)
        
        # Verify all tools have the correct API key
        for tool in tools:
            assert hasattr(tool, '_api_key')
            assert tool._api_key == "test_api_key"


class TestWeatherError:
    """Test weather error handling."""
    
    def test_weather_error_creation(self):
        """Test creating weather error."""
        error = WeatherError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestIntegrationScenarios:
    """Integration tests for realistic weather scenarios."""
    
    @pytest.fixture
    def weather_tool(self):
        return WeatherTool(api_key="test_api_key")
    
    @patch('requests.get')
    def test_complete_weather_workflow(self, mock_get, weather_tool):
        """Test complete weather data workflow."""
        # Mock a realistic weather response
        mock_response = Mock()
        mock_response.json.return_value = {
            'name': 'Tokyo',
            'sys': {'country': 'JP'},
            'main': {
                'temp': 22.5,
                'feels_like': 24.1,
                'humidity': 85,
                'pressure': 1008
            },
            'weather': [{'description': 'moderate rain'}],
            'wind': {'speed': 4.2}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = weather_tool._run("Tokyo", "metric")
        
        # Verify comprehensive weather information
        assert "Tokyo, JP" in result
        assert "22.5°C" in result
        assert "feels like 24.1°C" in result
        assert "moderate rain" in result.lower()
        assert "85%" in result
        assert "4.2 m/s" in result
        assert "1008 hPa" in result
    
    @patch('requests.get')
    def test_api_failure_recovery(self, mock_get, weather_tool):
        """Test API failure and recovery scenarios."""
        # First call fails
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        
        result1 = weather_tool._run("London", "metric")
        assert "error" in result1.lower()
        
        # Second call succeeds (simulating recovery)
        mock_response = Mock()
        mock_response.json.return_value = {
            'name': 'London',
            'sys': {'country': 'GB'},
            'main': {'temp': 15, 'feels_like': 14, 'humidity': 75},
            'weather': [{'description': 'clear sky'}],
            'wind': {'speed': 2.1}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.side_effect = None
        mock_get.return_value = mock_response
        
        result2 = weather_tool._run("London", "metric")
        assert "London, GB" in result2
        assert "clear sky" in result2.lower()


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v"])