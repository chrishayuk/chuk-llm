# tests/core/test_errors.py
"""
Unit tests for LLM error handling, retry logic, and error mapping
"""
import pytest
import asyncio
import time
from typing import Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

# Note: These imports will need to be adjusted based on your actual structure
# For now, creating mock implementations since the error classes don't exist yet

from enum import Enum


class ErrorSeverity(Enum):
    RECOVERABLE = "recoverable"      # Can retry
    PERMANENT = "permanent"          # Don't retry
    RATE_LIMITED = "rate_limited"    # Retry with backoff


class LLMError(Exception):
    """Base exception for LLM errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.PERMANENT, 
                 provider: str = None, model: str = None, **kwargs):
        super().__init__(message)
        self.severity = severity
        self.provider = provider
        self.model = model
        self.metadata = kwargs


class RateLimitError(LLMError):
    """Rate limit exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.RATE_LIMITED, **kwargs)
        self.retry_after = retry_after


class ModelNotFoundError(LLMError):
    """Model not found or no access"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorSeverity.PERMANENT, **kwargs)


class APIError(LLMError):
    """General API error"""
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.RECOVERABLE, **kwargs)
        self.status_code = status_code


class ConfigurationError(LLMError):
    """Configuration-related error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorSeverity.PERMANENT, **kwargs)


class ProviderErrorMapper:
    """Maps provider-specific errors to unified LLM errors"""
    
    @staticmethod
    def map_openai_error(error: Exception, provider: str, model: str) -> LLMError:
        if hasattr(error, 'status_code'):
            if error.status_code == 429:
                return RateLimitError(
                    str(error), 
                    retry_after=getattr(error, 'retry_after', None),
                    provider=provider, 
                    model=model
                )
            elif error.status_code == 404:
                return ModelNotFoundError(str(error), provider=provider, model=model)
            elif error.status_code in [400, 401, 403]:
                return ConfigurationError(str(error), provider=provider, model=model)
            else:
                return APIError(str(error), status_code=error.status_code, provider=provider, model=model)
        return LLMError(str(error), provider=provider, model=model)
    
    @staticmethod
    def map_anthropic_error(error: Exception, provider: str, model: str) -> LLMError:
        if hasattr(error, 'status_code'):
            if error.status_code == 429:
                return RateLimitError(str(error), provider=provider, model=model)
            elif error.status_code == 404:
                return ModelNotFoundError(str(error), provider=provider, model=model)
            else:
                return APIError(str(error), status_code=error.status_code, provider=provider, model=model)
        return LLMError(str(error), provider=provider, model=model)


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    retryable_severities: tuple[ErrorSeverity, ...] = (ErrorSeverity.RECOVERABLE, ErrorSeverity.RATE_LIMITED)
):
    """Decorator for automatic retry with exponential backoff"""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except LLMError as e:
                    last_error = e
                    
                    # Don't retry permanent errors
                    if e.severity not in retryable_severities:
                        raise
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        raise
                    
                    # Calculate backoff time
                    if e.severity == ErrorSeverity.RATE_LIMITED and hasattr(e, 'retry_after') and e.retry_after is not None:
                        wait_time = e.retry_after
                    else:
                        wait_time = min(backoff_factor ** (attempt + 1), max_backoff)
                    
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    # Convert unknown errors to LLMError
                    mapped_error = LLMError(str(e), ErrorSeverity.RECOVERABLE)
                    if mapped_error.severity in retryable_severities and attempt < max_retries:
                        last_error = mapped_error
                        wait_time = min(backoff_factor ** (attempt + 1), max_backoff)
                        await asyncio.sleep(wait_time)
                        continue
                    raise mapped_error
                    
            raise last_error
        return wrapper
    return decorator


class TestLLMErrors:
    """Test suite for LLM error classes"""
    
    def test_base_llm_error_creation(self):
        """Test basic LLM error creation"""
        error = LLMError("Test error", provider="openai", model="gpt-4")
        
        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.PERMANENT
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.metadata == {}
    
    def test_llm_error_with_metadata(self):
        """Test LLM error with additional metadata"""
        error = LLMError(
            "Test error", 
            severity=ErrorSeverity.RECOVERABLE,
            provider="anthropic",
            model="claude-3",
            request_id="req_123",
            custom_field="custom_value"
        )
        
        assert error.severity == ErrorSeverity.RECOVERABLE
        assert error.provider == "anthropic"
        assert error.model == "claude-3"
        assert error.metadata["request_id"] == "req_123"
        assert error.metadata["custom_field"] == "custom_value"
    
    def test_rate_limit_error_creation(self):
        """Test RateLimitError creation"""
        error = RateLimitError("Rate limit exceeded", retry_after=60, provider="openai")
        
        assert str(error) == "Rate limit exceeded"
        assert error.severity == ErrorSeverity.RATE_LIMITED
        assert error.retry_after == 60
        assert error.provider == "openai"
    
    def test_model_not_found_error_creation(self):
        """Test ModelNotFoundError creation"""
        error = ModelNotFoundError("Model not found", provider="openai", model="gpt-5")
        
        assert str(error) == "Model not found"
        assert error.severity == ErrorSeverity.PERMANENT
        assert error.provider == "openai"
        assert error.model == "gpt-5"
    
    def test_api_error_creation(self):
        """Test APIError creation"""
        error = APIError("Internal server error", status_code=500, provider="anthropic")
        
        assert str(error) == "Internal server error"
        assert error.severity == ErrorSeverity.RECOVERABLE
        assert error.status_code == 500
        assert error.provider == "anthropic"
    
    def test_configuration_error_creation(self):
        """Test ConfigurationError creation"""
        error = ConfigurationError("Invalid API key", provider="openai")
        
        assert str(error) == "Invalid API key"
        assert error.severity == ErrorSeverity.PERMANENT
        assert error.provider == "openai"
    
    def test_error_inheritance(self):
        """Test that all errors inherit from LLMError"""
        rate_limit = RateLimitError("Rate limit")
        model_not_found = ModelNotFoundError("Model not found")
        api_error = APIError("API error")
        config_error = ConfigurationError("Config error")
        
        assert isinstance(rate_limit, LLMError)
        assert isinstance(model_not_found, LLMError)
        assert isinstance(api_error, LLMError)
        assert isinstance(config_error, LLMError)


class TestProviderErrorMapper:
    """Test suite for provider error mapping"""
    
    def test_openai_rate_limit_mapping(self):
        """Test mapping OpenAI rate limit error"""
        class MockOpenAIError(Exception):
            def __init__(self):
                self.status_code = 429
                self.retry_after = 30
            
            def __str__(self):
                return "Rate limit exceeded"
        
        original_error = MockOpenAIError()
        mapped_error = ProviderErrorMapper.map_openai_error(original_error, "openai", "gpt-4")
        
        assert isinstance(mapped_error, RateLimitError)
        assert mapped_error.severity == ErrorSeverity.RATE_LIMITED
        assert mapped_error.retry_after == 30
        assert mapped_error.provider == "openai"
        assert mapped_error.model == "gpt-4"
    
    def test_openai_model_not_found_mapping(self):
        """Test mapping OpenAI model not found error"""
        class MockOpenAIError(Exception):
            def __init__(self):
                self.status_code = 404
            
            def __str__(self):
                return "Model not found"
        
        original_error = MockOpenAIError()
        mapped_error = ProviderErrorMapper.map_openai_error(original_error, "openai", "gpt-5")
        
        assert isinstance(mapped_error, ModelNotFoundError)
        assert mapped_error.severity == ErrorSeverity.PERMANENT
        assert mapped_error.provider == "openai"
        assert mapped_error.model == "gpt-5"
    
    def test_openai_configuration_error_mapping(self):
        """Test mapping OpenAI configuration errors"""
        for status_code in [400, 401, 403]:
            class MockOpenAIError(Exception):
                def __init__(self, code):
                    self.status_code = code
                    self.code = code  # Store code for __str__ method
                
                def __str__(self):
                    return f"Configuration error {self.code}"
            
            original_error = MockOpenAIError(status_code)
            mapped_error = ProviderErrorMapper.map_openai_error(original_error, "openai", "gpt-4")
            
            assert isinstance(mapped_error, ConfigurationError)
            assert mapped_error.severity == ErrorSeverity.PERMANENT
    
    def test_openai_api_error_mapping(self):
        """Test mapping OpenAI API errors"""
        class MockOpenAIError(Exception):
            def __init__(self):
                self.status_code = 500
            
            def __str__(self):
                return "Internal server error"
        
        original_error = MockOpenAIError()
        mapped_error = ProviderErrorMapper.map_openai_error(original_error, "openai", "gpt-4")
        
        assert isinstance(mapped_error, APIError)
        assert mapped_error.severity == ErrorSeverity.RECOVERABLE
        assert mapped_error.status_code == 500
    
    def test_openai_unknown_error_mapping(self):
        """Test mapping unknown OpenAI errors"""
        original_error = Exception("Unknown error")
        mapped_error = ProviderErrorMapper.map_openai_error(original_error, "openai", "gpt-4")
        
        assert isinstance(mapped_error, LLMError)
        assert mapped_error.severity == ErrorSeverity.PERMANENT
        assert mapped_error.provider == "openai"
        assert mapped_error.model == "gpt-4"
    
    def test_anthropic_error_mapping(self):
        """Test mapping Anthropic errors"""
        class MockAnthropicError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
            
            def __str__(self):
                return f"Anthropic error {self.status_code}"
        
        # Rate limit
        rate_limit_error = MockAnthropicError(429)
        mapped = ProviderErrorMapper.map_anthropic_error(rate_limit_error, "anthropic", "claude-3")
        assert isinstance(mapped, RateLimitError)
        
        # Model not found
        not_found_error = MockAnthropicError(404)
        mapped = ProviderErrorMapper.map_anthropic_error(not_found_error, "anthropic", "claude-3")
        assert isinstance(mapped, ModelNotFoundError)
        
        # API error
        api_error = MockAnthropicError(500)
        mapped = ProviderErrorMapper.map_anthropic_error(api_error, "anthropic", "claude-3")
        assert isinstance(mapped, APIError)


class TestRetryDecorator:
    """Test suite for retry decorator"""
    
    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        """Test that successful calls don't retry"""
        call_count = 0
        
        @with_retry(max_retries=3)
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_function()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_recoverable_error(self):
        """Test retry on recoverable errors"""
        call_count = 0
        
        @with_retry(max_retries=2, backoff_factor=0.01)  # Fast backoff for testing
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Temporary error")
            return "success"
        
        result = await failing_function()
        
        assert result == "success"
        assert call_count == 3  # Initial call + 2 retries
    
    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        """Test no retry on permanent errors"""
        call_count = 0
        
        @with_retry(max_retries=3)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ModelNotFoundError("Model not found")
        
        with pytest.raises(ModelNotFoundError):
            await failing_function()
        
        assert call_count == 1  # No retries for permanent errors
    
    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that retries stop after max attempts"""
        call_count = 0
        
        @with_retry(max_retries=2, backoff_factor=0.01)
        async def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise APIError("Always fails")
        
        with pytest.raises(APIError):
            await always_failing_function()
        
        assert call_count == 3  # Initial call + 2 retries
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_retry_after(self):
        """Test rate limit retry uses retry_after"""
        call_count = 0
        retry_times = []
        
        @with_retry(max_retries=2, backoff_factor=2.0)
        async def rate_limited_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Should use retry_after instead of exponential backoff
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "success"
        
        start_time = time.time()
        result = await rate_limited_function()
        total_time = time.time() - start_time
        
        assert result == "success"
        assert call_count == 3
        # Should be faster than exponential backoff (which would be 0.01 + 2.0 + 4.0)
        assert total_time < 1.0
    
    @pytest.mark.asyncio
    async def test_retry_delays_exist(self):
        """Test that retry delays are applied"""
        call_times = []
        
        @with_retry(max_retries=2, backoff_factor=0.05, max_backoff=1.0)
        async def failing_function():
            call_times.append(time.time())
            raise APIError("Temporary error")
        
        with pytest.raises(APIError):
            await failing_function()
        
        # Should have 3 calls (initial + 2 retries)
        assert len(call_times) == 3
        
        # Calculate delays
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # Both delays should exist and be reasonable
        assert delay1 >= 0.04   # Should be at least the backoff time
        assert delay2 >= 0.001  # Should have some delay
        
        # Total time should be reasonable
        total_time = call_times[-1] - call_times[0]
        assert total_time >= 0.05  # At least the sum of delays
        assert total_time < 1.0    # But not too long
    
    @pytest.mark.asyncio
    async def test_max_backoff_limit(self):
        """Test that backoff doesn't exceed max_backoff"""
        call_times = []
        
        @with_retry(max_retries=2, backoff_factor=10.0, max_backoff=0.05)
        async def failing_function():
            call_times.append(time.time())
            raise APIError("Temporary error")
        
        start_time = time.time()
        
        with pytest.raises(APIError):
            await failing_function()
        
        total_time = time.time() - start_time
        
        # Even with high backoff_factor, should be limited by max_backoff
        assert total_time < 0.5  # Much less than 10.0 + 100.0
    
    @pytest.mark.asyncio
    async def test_custom_retryable_severities(self):
        """Test custom retryable severities"""
        call_count = 0
        
        # Only retry RATE_LIMITED errors
        @with_retry(max_retries=2, backoff_factor=0.01, retryable_severities=(ErrorSeverity.RATE_LIMITED,))
        async def selective_retry_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIError("Recoverable error")  # Normally retryable
            elif call_count == 2:
                raise RateLimitError("Rate limited")  # Should be retried
            return "success"
        
        # Should not retry APIError with custom severities
        with pytest.raises(APIError):
            await selective_retry_function()
        
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_non_llm_error_conversion(self):
        """Test that non-LLM errors are converted and retried"""
        call_count = 0
        
        @with_retry(max_retries=2, backoff_factor=0.01)
        async def generic_error_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Generic error")
            return "success"
        
        result = await generic_error_function()
        
        assert result == "success"
        assert call_count == 3  # Should retry generic errors
    
    @pytest.mark.asyncio
    async def test_retry_preserves_original_error(self):
        """Test that the original error is preserved when retries are exhausted"""
        original_error = APIError("Original error message", status_code=503)
        
        @with_retry(max_retries=1, backoff_factor=0.01)
        async def failing_function():
            raise original_error
        
        with pytest.raises(APIError) as exc_info:
            await failing_function()
        
        # Should be the same error object
        assert exc_info.value is original_error
        assert exc_info.value.status_code == 503


class TestErrorIntegration:
    """Integration tests for error handling"""
    
    @pytest.mark.asyncio
    async def test_error_mapping_with_retry(self):
        """Test error mapping combined with retry logic"""
        call_count = 0
        
        class MockClient:
            @with_retry(max_retries=2, backoff_factor=0.01)
            async def api_call(self):
                nonlocal call_count
                call_count += 1
                
                # Simulate different provider errors
                if call_count == 1:
                    # First call: rate limit (should retry)
                    class MockError(Exception):
                        def __init__(self):
                            self.status_code = 429
                            self.retry_after = 0.01  # Very short for testing
                        
                        def __str__(self):
                            return "Rate limit"
                    
                    error = MockError()
                    raise ProviderErrorMapper.map_openai_error(error, "openai", "gpt-4")
                
                elif call_count == 2:
                    # Second call: temporary API error (should retry)
                    class MockError(Exception):
                        def __init__(self):
                            self.status_code = 503
                        
                        def __str__(self):
                            return "Service unavailable"
                    
                    error = MockError()
                    raise ProviderErrorMapper.map_openai_error(error, "openai", "gpt-4")
                
                else:
                    # Third call: success
                    return {"response": "Success", "tool_calls": []}
        
        client = MockClient()
        result = await client.api_call()
        
        assert result["response"] == "Success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_permanent_error_no_retry(self):
        """Test that permanent errors are not retried"""
        call_count = 0
        
        class MockClient:
            @with_retry(max_retries=3, backoff_factor=0.01)
            async def api_call(self):
                nonlocal call_count
                call_count += 1
                
                # Simulate model not found error
                class MockError(Exception):
                    def __init__(self):
                        self.status_code = 404
                    
                    def __str__(self):
                        return "Model not found"
                
                error = MockError()
                raise ProviderErrorMapper.map_openai_error(error, "openai", "invalid-model")
        
        client = MockClient()
        
        with pytest.raises(ModelNotFoundError):
            await client.api_call()
        
        assert call_count == 1  # No retries for permanent errors


# Fixtures
@pytest.fixture
def mock_openai_rate_limit_error():
    """Mock OpenAI rate limit error"""
    class MockError(Exception):
        def __init__(self):
            self.status_code = 429
            self.retry_after = 30
        
        def __str__(self):
            return "Rate limit exceeded"
    
    return MockError()


@pytest.fixture
def mock_anthropic_api_error():
    """Mock Anthropic API error"""
    class MockError(Exception):
        def __init__(self):
            self.status_code = 500
        
        def __str__(self):
            return "Internal server error"
    
    return MockError()


# Parametrized tests
@pytest.mark.parametrize("severity,should_retry", [
    (ErrorSeverity.RECOVERABLE, True),
    (ErrorSeverity.RATE_LIMITED, True),
    (ErrorSeverity.PERMANENT, False),
])
@pytest.mark.asyncio
async def test_retry_behavior_by_severity(severity, should_retry):
    """Test retry behavior for different error severities"""
    call_count = 0
    
    @with_retry(max_retries=2, backoff_factor=0.01)
    async def test_function():
        nonlocal call_count
        call_count += 1
        raise LLMError("Test error", severity=severity)
    
    with pytest.raises(LLMError):
        await test_function()
    
    if should_retry:
        assert call_count == 3  # Initial + 2 retries
    else:
        assert call_count == 1  # No retries


@pytest.mark.parametrize("status_code,expected_error_type", [
    (400, ConfigurationError),
    (401, ConfigurationError),
    (403, ConfigurationError),
    (404, ModelNotFoundError),
    (429, RateLimitError),
    (500, APIError),
    (503, APIError),
])
def test_openai_error_mapping_by_status_code(status_code, expected_error_type):
    """Test OpenAI error mapping for different status codes"""
    class MockError(Exception):
        def __init__(self, code):
            self.status_code = code
            self.code = code  # Store for __str__ method
        
        def __str__(self):
            return f"Error {self.code}"
    
    original_error = MockError(status_code)
    mapped_error = ProviderErrorMapper.map_openai_error(original_error, "openai", "gpt-4")
    
    assert isinstance(mapped_error, expected_error_type)
    assert mapped_error.provider == "openai"
    assert mapped_error.model == "gpt-4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])