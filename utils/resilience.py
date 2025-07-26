"""
Error Handling and Resilience Utilities

This module provides comprehensive error handling and resilience patterns
for the weather agent, following the patterns from examples/basic_chain.py
and examples/agent_chain.py.

Key Features:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Graceful degradation strategies
- Input validation and sanitization
- Comprehensive logging and monitoring
"""

import logging
import time
import asyncio
from typing import Any, Callable, Dict, Optional, Union, List
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherAgentError(Exception):
    """Base exception for weather agent errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message)
        self.severity = severity
        self.timestamp = datetime.now()
        self.metadata = kwargs


class APIError(WeatherAgentError):
    """API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.status_code = status_code


class ValidationError(WeatherAgentError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorSeverity.LOW, **kwargs)
        self.field = field


class ConfigurationError(WeatherAgentError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorSeverity.CRITICAL, **kwargs)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API resilience.
    
    Prevents cascading failures by monitoring error rates and temporarily
    blocking requests when error threshold is exceeded.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting to close circuit
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise WeatherAgentError(
                    f"Circuit breaker is OPEN. Try again after {self.timeout} seconds.",
                    ErrorSeverity.HIGH
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time >= timedelta(seconds=self.timeout)
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class InputValidator:
    """Utility class for input validation and sanitization."""
    
    @staticmethod
    def validate_location(location: str) -> str:
        """
        Validate and sanitize location input.
        
        Args:
            location: Raw location input
            
        Returns:
            Sanitized location string
            
        Raises:
            ValidationError: If location is invalid
        """
        if not location or not isinstance(location, str):
            raise ValidationError("Location cannot be empty or non-string", field="location")
        
        # Remove leading/trailing whitespace
        location = location.strip()
        
        if not location:
            raise ValidationError("Location cannot be empty after trimming", field="location")
        
        if len(location) > 100:
            raise ValidationError("Location name too long (max 100 characters)", field="location")
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>\"\'&\\]', '', location)
        
        # Check for SQL injection patterns
        suspicious_patterns = ['drop', 'delete', 'insert', 'update', 'select', '--', ';']
        lower_location = sanitized.lower()
        for pattern in suspicious_patterns:
            if pattern in lower_location:
                raise ValidationError(f"Location contains suspicious content: {pattern}", field="location")
        
        return sanitized
    
    @staticmethod
    def validate_units(units: str) -> str:
        """
        Validate temperature units.
        
        Args:
            units: Temperature units
            
        Returns:
            Validated units string
            
        Raises:
            ValidationError: If units are invalid
        """
        if not units or not isinstance(units, str):
            return "metric"  # Default
        
        units = units.lower().strip()
        valid_units = ['metric', 'imperial', 'kelvin']
        
        if units not in valid_units:
            raise ValidationError(f"Invalid units. Must be one of: {valid_units}", field="units")
        
        return units
    
    @staticmethod
    def validate_days(days: Union[int, str]) -> int:
        """
        Validate forecast days parameter.
        
        Args:
            days: Number of forecast days
            
        Returns:
            Validated days integer
            
        Raises:
            ValidationError: If days is invalid
        """
        try:
            days = int(days)
        except (ValueError, TypeError):
            raise ValidationError("Days must be a valid integer", field="days")
        
        if not 1 <= days <= 5:
            raise ValidationError("Days must be between 1 and 5", field="days")
        
        return days


class GracefulDegradation:
    """Handles graceful degradation when services are unavailable."""
    
    @staticmethod
    def get_fallback_weather_response(location: str, error: str) -> Dict[str, Any]:
        """
        Generate fallback response when weather service is unavailable.
        
        Args:
            location: Location that was queried
            error: Error message
            
        Returns:
            Fallback response dictionary
        """
        return {
            "success": False,
            "location": location,
            "error": error,
            "fallback_message": (
                f"🌤️ Weather service temporarily unavailable for {location}. "
                f"Please try again in a few minutes or check local weather services. "
                f"We apologize for the inconvenience."
            ),
            "suggestions": [
                "Check your local weather app",
                "Visit weather.com or your national weather service",
                "Try again in a few minutes"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_limited_service_response(location: str, available_services: List[str]) -> Dict[str, Any]:
        """
        Generate response when only limited services are available.
        
        Args:
            location: Location that was queried
            available_services: List of available services
            
        Returns:
            Limited service response
        """
        return {
            "success": True,
            "location": location,
            "limited_service": True,
            "available_services": available_services,
            "message": (
                f"⚠️ Limited weather service available for {location}. "
                f"Available: {', '.join(available_services)}. "
                f"Some features may be temporarily unavailable."
            ),
            "timestamp": datetime.now().isoformat()
        }


class ErrorReporter:
    """Centralized error reporting and logging."""
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = []
        self.max_stored_errors = 100
    
    def report_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        user_query: str = None
    ):
        """
        Report and log an error with context.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            user_query: User query that caused the error
        """
        error_type = type(error).__name__
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Store error details
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context or {},
            "user_query": user_query,
            "severity": getattr(error, 'severity', ErrorSeverity.MEDIUM).value
        }
        
        # Add to recent errors (with size limit)
        self.last_errors.append(error_details)
        if len(self.last_errors) > self.max_stored_errors:
            self.last_errors.pop(0)
        
        # Log error with appropriate level
        severity = getattr(error, 'severity', ErrorSeverity.MEDIUM)
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error_type}: {error}", extra=error_details)
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {error_type}: {error}", extra=error_details)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {error_type}: {error}", extra=error_details)
        else:
            logger.info(f"LOW SEVERITY: {error_type}: {error}", extra=error_details)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of reported errors."""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "recent_errors": self.last_errors[-10:],  # Last 10 errors
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }


# Global error reporter instance
error_reporter = ErrorReporter()


def safe_execute(
    func: Callable,
    *args,
    fallback_result: Any = None,
    context: Dict[str, Any] = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        fallback_result: Result to return if function fails
        context: Additional context for error reporting
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or fallback result
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_reporter.report_error(e, context=context)
        return fallback_result


def validate_environment_safety() -> Dict[str, Any]:
    """
    Validate environment for security and safety.
    
    Returns:
        Validation results dictionary
    """
    results = {
        "safe": True,
        "warnings": [],
        "errors": []
    }
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "OPENWEATHER_API_KEY"]
    for var in required_vars:
        import os
        if not os.getenv(var):
            results["errors"].append(f"Missing required environment variable: {var}")
            results["safe"] = False
    
    # Check for debug mode in production
    import os
    if os.getenv("DEBUG", "").lower() == "true":
        results["warnings"].append("Debug mode is enabled - not recommended for production")
    
    # Check for insecure configurations
    if os.getenv("VERBOSE", "").lower() == "true":
        results["warnings"].append("Verbose mode enabled - may expose sensitive information")
    
    return results


if __name__ == "__main__":
    """Test error handling and resilience utilities."""
    
    # Test input validation
    print("Testing input validation:")
    try:
        valid_location = InputValidator.validate_location("New York")
        print(f"✅ Valid location: {valid_location}")
        
        InputValidator.validate_location("")  # Should raise error
    except ValidationError as e:
        print(f"✅ Caught validation error: {e}")
    
    # Test circuit breaker
    print("\\nTesting circuit breaker:")
    cb = CircuitBreaker(failure_threshold=2, timeout=1)
    
    def failing_function():
        raise Exception("Simulated failure")
    
    for i in range(3):
        try:
            cb.call(failing_function)
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
    
    # Test error reporter
    print("\\nTesting error reporter:")
    try:
        raise APIError("Test API error", status_code=500)
    except APIError as e:
        error_reporter.report_error(e, context={"test": True})
        summary = error_reporter.get_error_summary()
        print(f"Error summary: {summary['total_errors']} total errors")
    
    # Test environment validation
    print("\\nTesting environment validation:")
    env_results = validate_environment_safety()
    print(f"Environment safe: {env_results['safe']}")
    if env_results['warnings']:
        print(f"Warnings: {env_results['warnings']}")
    if env_results['errors']:
        print(f"Errors: {env_results['errors']}")
    
    print("\\n✅ Error handling and resilience utilities working correctly!")