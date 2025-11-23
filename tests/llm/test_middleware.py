"""Comprehensive tests for middleware.py module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import time
from typing import Any, Dict

from chuk_llm.llm.middleware import (
    Middleware,
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    MiddlewareStack,
    PaymentGuardMiddleware,
)


class TestMiddlewareABC:
    """Tests for Middleware abstract base class."""

    def test_middleware_is_abstract(self):
        """Test that Middleware cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Middleware()


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    def test_init_default(self):
        """Test initialization with defaults."""
        middleware = LoggingMiddleware()
        assert middleware.logger is not None
        assert middleware.log_level == 20  # logging.INFO

    def test_init_custom_logger(self):
        """Test initialization with custom logger."""
        import logging
        custom_logger = logging.getLogger("custom")
        middleware = LoggingMiddleware(logger=custom_logger, log_level=logging.DEBUG)
        assert middleware.logger == custom_logger
        assert middleware.log_level == logging.DEBUG

    @pytest.mark.asyncio
    async def test_process_request(self):
        """Test process_request."""
        middleware = LoggingMiddleware()
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test"}]

        result_messages, result_tools, result_kwargs = await middleware.process_request(
            messages, tools, stream=True
        )

        assert result_messages == messages
        assert result_tools == tools
        assert result_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_process_request_no_tools(self):
        """Test process_request without tools."""
        middleware = LoggingMiddleware()
        messages = [{"role": "user", "content": "Hello"}]

        result_messages, result_tools, result_kwargs = await middleware.process_request(
            messages, None
        )

        assert result_messages == messages
        assert result_tools is None

    @pytest.mark.asyncio
    async def test_process_response_non_streaming(self):
        """Test process_response for non-streaming."""
        middleware = LoggingMiddleware()
        response = {"response": "Hello there", "tool_calls": []}

        result = await middleware.process_response(response, duration=1.5, is_streaming=False)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_response_streaming(self):
        """Test process_response for streaming."""
        middleware = LoggingMiddleware()
        response = {"response": "stream completed"}

        result = await middleware.process_response(response, duration=2.3, is_streaming=True)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_error(self):
        """Test process_error."""
        middleware = LoggingMiddleware()
        error = ValueError("Test error")

        result = await middleware.process_error(error, duration=0.5)
        assert result == error


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    def test_init(self):
        """Test initialization."""
        middleware = MetricsMiddleware()
        assert middleware.metrics["total_requests"] == 0
        assert middleware.metrics["total_errors"] == 0
        assert middleware.metrics["total_duration"] == 0.0
        assert middleware.metrics["streaming_requests"] == 0
        assert middleware.metrics["tool_requests"] == 0

    @pytest.mark.asyncio
    async def test_process_request_basic(self):
        """Test process_request increments counter."""
        middleware = MetricsMiddleware()
        messages = [{"role": "user", "content": "Hello"}]

        await middleware.process_request(messages)
        assert middleware.metrics["total_requests"] == 1
        assert middleware.metrics["streaming_requests"] == 0
        assert middleware.metrics["tool_requests"] == 0

    @pytest.mark.asyncio
    async def test_process_request_with_streaming(self):
        """Test process_request with streaming."""
        middleware = MetricsMiddleware()
        messages = [{"role": "user", "content": "Hello"}]

        await middleware.process_request(messages, stream=True)
        assert middleware.metrics["total_requests"] == 1
        assert middleware.metrics["streaming_requests"] == 1

    @pytest.mark.asyncio
    async def test_process_request_with_tools(self):
        """Test process_request with tools."""
        middleware = MetricsMiddleware()
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test"}]

        await middleware.process_request(messages, tools)
        assert middleware.metrics["total_requests"] == 1
        assert middleware.metrics["tool_requests"] == 1

    @pytest.mark.asyncio
    async def test_process_response(self):
        """Test process_response updates duration."""
        middleware = MetricsMiddleware()
        response = {"response": "Hello"}

        await middleware.process_response(response, duration=1.5)
        assert middleware.metrics["total_duration"] == 1.5

    @pytest.mark.asyncio
    async def test_process_error(self):
        """Test process_error increments error count."""
        middleware = MetricsMiddleware()
        error = ValueError("Test")

        await middleware.process_error(error, duration=0.5)
        assert middleware.metrics["total_errors"] == 1
        assert middleware.metrics["total_duration"] == 0.5

    def test_get_metrics(self):
        """Test get_metrics returns copy with calculated fields."""
        middleware = MetricsMiddleware()
        middleware.metrics["total_requests"] = 10
        middleware.metrics["total_errors"] = 2
        middleware.metrics["total_duration"] = 15.0

        metrics = middleware.get_metrics()
        assert metrics["total_requests"] == 10
        assert metrics["total_errors"] == 2
        assert metrics["average_duration"] == 1.5
        assert metrics["error_rate"] == 0.2

    def test_get_metrics_zero_requests(self):
        """Test get_metrics with zero requests."""
        middleware = MetricsMiddleware()
        metrics = middleware.get_metrics()
        assert "average_duration" not in metrics
        assert "error_rate" not in metrics


class TestCachingMiddleware:
    """Tests for CachingMiddleware."""

    def test_init_default(self):
        """Test initialization with defaults."""
        middleware = CachingMiddleware()
        assert middleware.ttl == 300
        assert middleware.max_size == 100
        assert middleware.cache == {}

    def test_init_custom(self):
        """Test initialization with custom values."""
        middleware = CachingMiddleware(ttl=600, max_size=50)
        assert middleware.ttl == 600
        assert middleware.max_size == 50

    def test_get_cache_key(self):
        """Test _get_cache_key generates consistent key."""
        middleware = CachingMiddleware()
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test"}]

        key1 = middleware._get_cache_key(messages, tools)
        key2 = middleware._get_cache_key(messages, tools)
        assert key1 == key2
        assert isinstance(key1, str)

    def test_get_cache_key_different_messages(self):
        """Test _get_cache_key generates different keys for different messages."""
        middleware = CachingMiddleware()
        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "Goodbye"}]

        key1 = middleware._get_cache_key(messages1, None)
        key2 = middleware._get_cache_key(messages2, None)
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_process_request_no_cache(self):
        """Test process_request without cached data."""
        middleware = CachingMiddleware()
        messages = [{"role": "user", "content": "Hello"}]

        result_messages, result_tools, result_kwargs = await middleware.process_request(
            messages, None
        )

        assert result_messages == messages
        assert result_tools is None
        assert "_cached_response" not in result_kwargs

    @pytest.mark.asyncio
    async def test_process_request_with_cache(self):
        """Test process_request with cached data."""
        middleware = CachingMiddleware()
        messages = [{"role": "user", "content": "Hello"}]
        cache_key = middleware._get_cache_key(messages, None)

        # Add to cache
        middleware.cache[cache_key] = {
            "response": {"response": "Cached response"},
            "timestamp": time.time()
        }

        result_messages, result_tools, result_kwargs = await middleware.process_request(
            messages, None
        )

        assert "_cached_response" in result_kwargs
        assert result_kwargs["_cached_response"]["response"] == "Cached response"

    @pytest.mark.asyncio
    async def test_process_request_expired_cache(self):
        """Test process_request with expired cache."""
        middleware = CachingMiddleware(ttl=1)
        messages = [{"role": "user", "content": "Hello"}]
        cache_key = middleware._get_cache_key(messages, None)

        # Add expired entry
        middleware.cache[cache_key] = {
            "response": {"response": "Old response"},
            "timestamp": time.time() - 10
        }

        result_messages, result_tools, result_kwargs = await middleware.process_request(
            messages, None
        )

        assert "_cached_response" not in result_kwargs

    @pytest.mark.asyncio
    async def test_process_request_streaming_skips_cache(self):
        """Test that streaming requests skip cache."""
        middleware = CachingMiddleware()
        messages = [{"role": "user", "content": "Hello"}]
        cache_key = middleware._get_cache_key(messages, None)

        # Add to cache
        middleware.cache[cache_key] = {
            "response": {"response": "Cached"},
            "timestamp": time.time()
        }

        result_messages, result_tools, result_kwargs = await middleware.process_request(
            messages, None, stream=True
        )

        # Should not use cache for streaming
        assert "_cached_response" not in result_kwargs

    @pytest.mark.asyncio
    async def test_process_response_non_streaming(self):
        """Test process_response for non-streaming."""
        middleware = CachingMiddleware()
        response = {"response": "Hello"}

        result = await middleware.process_response(response, duration=1.0, is_streaming=False)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_response_streaming(self):
        """Test process_response for streaming."""
        middleware = CachingMiddleware()
        response = {"response": "Stream done"}

        result = await middleware.process_response(response, duration=1.0, is_streaming=True)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_response_with_error(self):
        """Test process_response with error (should not cache)."""
        middleware = CachingMiddleware()
        response = {"error": "Something failed"}

        result = await middleware.process_response(response, duration=1.0, is_streaming=False)
        assert result == response


class TestPaymentGuardMiddleware:
    """Tests for PaymentGuardMiddleware."""

    @staticmethod
    def _create_payment_guard():
        """Create concrete PaymentGuardMiddleware for testing."""
        class ConcretePaymentGuard(PaymentGuardMiddleware):
            async def process_request(self, messages, tools=None, **kwargs):
                return messages, tools, kwargs

        return ConcretePaymentGuard()

    def test_middleware_needs_process_request(self):
        """Test that PaymentGuardMiddleware needs process_request to be instantiated."""
        middleware = self._create_payment_guard()
        assert middleware is not None

    @pytest.mark.asyncio
    async def test_process_response_no_error(self):
        """Test process_response with normal response."""
        middleware = self._create_payment_guard()
        response = {"response": "Hello there"}

        result = await middleware.process_response(response, duration=1.0)
        assert result == response
        assert not result.get("error")

    @pytest.mark.asyncio
    async def test_process_response_quota_exceeded(self):
        """Test detection of quota exceeded."""
        middleware = self._create_payment_guard()
        response = {"response": "Quota exceeded for this account"}

        result = await middleware.process_response(response, duration=1.0)
        assert result["error"] is True
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_process_response_payment_required(self):
        """Test detection of payment required."""
        middleware = self._create_payment_guard()
        response = {"response": "Payment required to continue"}

        result = await middleware.process_response(response, duration=1.0)
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_process_response_insufficient_balance(self):
        """Test detection of insufficient balance."""
        middleware = self._create_payment_guard()
        response = {"response": "Insufficient balance in your account"}

        result = await middleware.process_response(response, duration=1.0)
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_process_response_credit_depleted(self):
        """Test detection of credit depleted."""
        middleware = self._create_payment_guard()
        response = {"response": "Credit depleted, please add funds"}

        result = await middleware.process_response(response, duration=1.0)
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_process_response_already_has_error(self):
        """Test that existing error is preserved."""
        middleware = self._create_payment_guard()
        response = {"error": True, "error_message": "Existing error"}

        result = await middleware.process_response(response, duration=1.0)
        assert result["error"] is True
        assert result["error_message"] == "Existing error"

    @pytest.mark.asyncio
    async def test_process_response_non_dict(self):
        """Test with non-dict response."""
        middleware = self._create_payment_guard()
        response = "just a string"

        result = await middleware.process_response(response, duration=1.0)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_response_case_insensitive(self):
        """Test billing message detection is case insensitive."""
        middleware = self._create_payment_guard()
        response = {"response": "QUOTA EXCEEDED for user"}

        result = await middleware.process_response(response, duration=1.0)
        assert result["error"] is True


class TestMiddlewareStack:
    """Tests for MiddlewareStack."""

    def test_init(self):
        """Test initialization."""
        middleware1 = LoggingMiddleware()
        middleware2 = MetricsMiddleware()
        stack = MiddlewareStack([middleware1, middleware2])

        assert len(stack.middlewares) == 2
        assert stack.middlewares[0] == middleware1
        assert stack.middlewares[1] == middleware2

    def test_init_empty(self):
        """Test initialization with empty list."""
        stack = MiddlewareStack([])
        assert stack.middlewares == []

    @pytest.mark.asyncio
    async def test_process_request_single(self):
        """Test process_request with single middleware."""
        middleware = LoggingMiddleware()
        stack = MiddlewareStack([middleware])
        messages = [{"role": "user", "content": "Hello"}]

        result_messages, result_tools, result_kwargs = await stack.process_request(messages)
        assert result_messages == messages

    @pytest.mark.asyncio
    async def test_process_request_multiple(self):
        """Test process_request with multiple middlewares."""
        middleware1 = LoggingMiddleware()
        middleware2 = MetricsMiddleware()
        stack = MiddlewareStack([middleware1, middleware2])
        messages = [{"role": "user", "content": "Hello"}]

        result_messages, result_tools, result_kwargs = await stack.process_request(messages)
        assert result_messages == messages
        assert middleware2.metrics["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_process_request_empty_stack(self):
        """Test process_request with empty stack."""
        stack = MiddlewareStack([])
        messages = [{"role": "user", "content": "Hello"}]

        result_messages, result_tools, result_kwargs = await stack.process_request(messages)
        assert result_messages == messages

    @pytest.mark.asyncio
    async def test_process_response_single(self):
        """Test process_response with single middleware."""
        middleware = LoggingMiddleware()
        stack = MiddlewareStack([middleware])
        response = {"response": "Hello"}

        result = await stack.process_response(response, duration=1.0)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_response_multiple(self):
        """Test process_response with multiple middlewares (reversed)."""
        middleware1 = LoggingMiddleware()
        middleware2 = MetricsMiddleware()
        stack = MiddlewareStack([middleware1, middleware2])
        response = {"response": "Hello"}

        result = await stack.process_response(response, duration=1.5)
        assert result == response
        assert middleware2.metrics["total_duration"] == 1.5

    @pytest.mark.asyncio
    async def test_process_response_empty_stack(self):
        """Test process_response with empty stack."""
        stack = MiddlewareStack([])
        response = {"response": "Hello"}

        result = await stack.process_response(response, duration=1.0)
        assert result == response

    @pytest.mark.asyncio
    async def test_process_stream_chunk(self):
        """Test process_stream_chunk."""
        middleware = LoggingMiddleware()
        stack = MiddlewareStack([middleware])
        chunk = {"response": "Hello"}

        result = await stack.process_stream_chunk(chunk, chunk_index=0, total_duration=0.5)
        assert result == chunk

    @pytest.mark.asyncio
    async def test_process_error(self):
        """Test process_error."""
        middleware = MetricsMiddleware()
        stack = MiddlewareStack([middleware])
        error = ValueError("Test error")

        result = await stack.process_error(error, duration=0.5)
        assert result == error
        assert middleware.metrics["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self):
        """Test full request-response cycle."""
        metrics = MetricsMiddleware()
        logging_mw = LoggingMiddleware()
        stack = MiddlewareStack([logging_mw, metrics])

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test"}]

        # Process request
        result_messages, result_tools, result_kwargs = await stack.process_request(
            messages, tools, stream=False
        )
        assert result_messages == messages
        assert metrics.metrics["total_requests"] == 1
        assert metrics.metrics["tool_requests"] == 1

        # Process response
        response = {"response": "Hello there"}
        result = await stack.process_response(response, duration=2.0)
        assert result == response
        assert metrics.metrics["total_duration"] == 2.0

    @pytest.mark.asyncio
    async def test_middleware_execution_order(self):
        """Test that middlewares execute in correct order."""
        order = []

        class OrderMiddleware1(Middleware):
            async def process_request(self, messages, tools=None, **kwargs):
                order.append("m1_req")
                return messages, tools, kwargs

            async def process_response(self, response, duration, is_streaming=False):
                order.append("m1_resp")
                return response

        class OrderMiddleware2(Middleware):
            async def process_request(self, messages, tools=None, **kwargs):
                order.append("m2_req")
                return messages, tools, kwargs

            async def process_response(self, response, duration, is_streaming=False):
                order.append("m2_resp")
                return response

        m1 = OrderMiddleware1()
        m2 = OrderMiddleware2()
        stack = MiddlewareStack([m1, m2])

        messages = [{"role": "user", "content": "Hi"}]
        await stack.process_request(messages)
        await stack.process_response({"response": "Hello"}, duration=1.0)

        # Request should be m1, m2; response should be m2, m1 (reversed)
        assert order == ["m1_req", "m2_req", "m2_resp", "m1_resp"]

    @pytest.mark.asyncio
    async def test_caching_integration(self):
        """Test caching middleware integration."""
        caching = CachingMiddleware()
        stack = MiddlewareStack([caching])
        messages = [{"role": "user", "content": "Hello"}]

        # First request - no cache
        result_messages, result_tools, result_kwargs = await stack.process_request(messages)
        assert "_cached_response" not in result_kwargs

        # Manually add to cache
        cache_key = caching._get_cache_key(messages, None)
        caching.cache[cache_key] = {
            "response": {"response": "Cached"},
            "timestamp": time.time()
        }

        # Second request - should get cache
        result_messages, result_tools, result_kwargs = await stack.process_request(messages)
        assert "_cached_response" in result_kwargs
        assert result_kwargs["_cached_response"]["response"] == "Cached"

    @pytest.mark.asyncio
    async def test_payment_guard_integration(self):
        """Test payment guard middleware integration."""
        # Create concrete PaymentGuard
        class ConcretePaymentGuard(PaymentGuardMiddleware):
            async def process_request(self, messages, tools=None, **kwargs):
                return messages, tools, kwargs

        payment_guard = ConcretePaymentGuard()
        stack = MiddlewareStack([payment_guard])

        response = {"response": "Quota exceeded"}
        result = await stack.process_response(response, duration=1.0)
        assert result["error"] is True
        assert "error_message" in result
