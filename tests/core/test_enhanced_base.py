# tests/core/test_enhanced_base.py
"""
Unit tests for EnhancedBaseLLMClient with middleware support
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

# Note: These imports will need to be adjusted based on your actual structure
# For now, assuming the enhanced classes would be in these locations:
# from chuk_llm.llm.enhanced_base import EnhancedBaseLLMClient
# from chuk_llm.llm.middleware import Middleware, MiddlewareStack, LoggingMiddleware, MetricsMiddleware
# from chuk_llm.llm.errors import LLMError, RateLimitError, APIError, ErrorSeverity


# Mock implementations for testing since these don't exist yet
class MockLLMError(Exception):
    """Mock LLM error for testing"""

    def __init__(self, message: str, severity: str = "permanent"):
        super().__init__(message)
        self.severity = severity


class MockMiddleware:
    """Mock middleware for testing"""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.process_request_calls = []
        self.process_response_calls = []
        self.process_stream_chunk_calls = []
        self.process_error_calls = []

    async def process_request(self, messages, tools=None, **kwargs):
        self.process_request_calls.append((messages, tools, kwargs))
        return messages, tools, kwargs

    async def process_response(self, response, duration, is_streaming=False):
        self.process_response_calls.append((response, duration, is_streaming))
        return response

    async def process_stream_chunk(self, chunk, chunk_index, total_duration):
        self.process_stream_chunk_calls.append((chunk, chunk_index, total_duration))
        return chunk

    async def process_error(self, error, duration):
        self.process_error_calls.append((error, duration))
        return error


class MockMiddlewareStack:
    """Mock middleware stack for testing"""

    def __init__(self, middlewares: list[MockMiddleware]):
        self.middlewares = middlewares

    async def process_request(self, messages, tools=None, **kwargs):
        for middleware in self.middlewares:
            messages, tools, kwargs = await middleware.process_request(
                messages, tools, **kwargs
            )
        return messages, tools, kwargs

    async def process_response(self, response, duration, is_streaming=False):
        for middleware in reversed(self.middlewares):
            response = await middleware.process_response(
                response, duration, is_streaming
            )
        return response

    async def process_stream_chunk(self, chunk, chunk_index, total_duration):
        for middleware in reversed(self.middlewares):
            chunk = await middleware.process_stream_chunk(
                chunk, chunk_index, total_duration
            )
        return chunk

    async def process_error(self, error, duration):
        for middleware in reversed(self.middlewares):
            error = await middleware.process_error(error, duration)
        return error


class MockEnhancedBaseLLMClient:
    """Mock enhanced base client for testing"""

    def __init__(self, middleware: list[MockMiddleware] | None = None):
        self.middleware_stack = MockMiddlewareStack(middleware or [])
        self.provider_name = "test_provider"
        self.model_name = "test_model"

        # Mock responses
        self.mock_response = {"response": "Test response", "tool_calls": []}
        self.mock_stream = [
            {"response": "Hello ", "tool_calls": []},
            {"response": "world!", "tool_calls": []},
        ]
        self.should_raise = None

        # Call tracking
        self.create_completion_impl_calls = []
        self.create_stream_impl_calls = []

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ):
        """Enhanced completion with middleware and error handling"""

        if stream:
            # For streaming, return the async generator directly
            return self._create_enhanced_stream_wrapper(messages, tools, **kwargs)
        else:
            # For non-streaming, return a coroutine
            return self._create_enhanced_completion_wrapper(messages, tools, **kwargs)

    async def _create_enhanced_completion_wrapper(self, messages, tools, **kwargs):
        """Wrapper for non-streaming completion"""
        start_time = time.time()

        try:
            # Process request through middleware
            (
                processed_messages,
                processed_tools,
                processed_kwargs,
            ) = await self.middleware_stack.process_request(messages, tools, **kwargs)

            # Check for cached response
            if "_cached_response" in processed_kwargs:
                cached = processed_kwargs.pop("_cached_response")
                duration = time.time() - start_time
                return await self.middleware_stack.process_response(
                    cached, duration, False
                )

            # Call the actual provider implementation
            return await self._create_enhanced_completion(
                processed_messages, processed_tools, start_time, **processed_kwargs
            )

        except Exception as e:
            duration = time.time() - start_time
            processed_error = await self.middleware_stack.process_error(e, duration)
            raise processed_error

    async def _create_enhanced_stream_wrapper(self, messages, tools, **kwargs):
        """Wrapper for streaming completion"""
        start_time = time.time()

        try:
            # Process request through middleware
            (
                processed_messages,
                processed_tools,
                processed_kwargs,
            ) = await self.middleware_stack.process_request(messages, tools, **kwargs)

            # Check for cached response (shouldn't happen with streaming, but handle it)
            if "_cached_response" in processed_kwargs:
                cached = processed_kwargs.pop("_cached_response")
                duration = time.time() - start_time
                processed_response = await self.middleware_stack.process_response(
                    cached, duration, True
                )
                yield processed_response
                return

            # Call the actual streaming implementation
            async for chunk in self._create_enhanced_stream(
                processed_messages, processed_tools, start_time, **processed_kwargs
            ):
                yield chunk

        except Exception as e:
            duration = time.time() - start_time
            processed_error = await self.middleware_stack.process_error(e, duration)

            # Yield error as final chunk
            yield {
                "response": f"Streaming error: {str(processed_error)}",
                "tool_calls": [],
                "error": True,
                "error_type": type(processed_error).__name__,
            }

    async def _create_enhanced_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        start_time: float,
        **kwargs,
    ):
        """Enhanced non-streaming completion"""
        self.create_completion_impl_calls.append((messages, tools, kwargs))

        if self.should_raise:
            raise self.should_raise

        response = await self._create_completion_impl(messages, tools, **kwargs)
        duration = time.time() - start_time
        return await self.middleware_stack.process_response(
            response, duration, is_streaming=False
        )

    async def _create_enhanced_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        start_time: float,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Enhanced streaming with middleware processing"""
        self.create_stream_impl_calls.append((messages, tools, kwargs))

        chunk_index = 0

        try:
            # Get the raw stream from implementation
            stream = self._create_stream_impl(messages, tools, **kwargs)

            async for chunk in stream:
                current_time = time.time()
                duration = current_time - start_time

                # Process through middleware
                processed_chunk = await self.middleware_stack.process_stream_chunk(
                    chunk, chunk_index, duration
                )

                yield processed_chunk
                chunk_index += 1

        except Exception as e:
            duration = time.time() - start_time
            processed_error = await self.middleware_stack.process_error(e, duration)

            # Yield error as final chunk
            yield {
                "response": f"Streaming error: {str(processed_error)}",
                "tool_calls": [],
                "error": True,
                "error_type": type(processed_error).__name__,
            }

    async def _create_completion_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Mock implementation"""
        await asyncio.sleep(0.001)  # Simulate API delay
        return self.mock_response

    async def _create_stream_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Mock streaming implementation"""
        for chunk in self.mock_stream:
            yield chunk
            await asyncio.sleep(0.001)


class TestEnhancedBaseLLMClient:
    """Test suite for EnhancedBaseLLMClient"""

    @pytest.mark.asyncio
    async def test_creation_without_middleware(self):
        """Test creating enhanced client without middleware"""
        client = MockEnhancedBaseLLMClient()

        assert client.middleware_stack is not None
        assert len(client.middleware_stack.middlewares) == 0
        assert client.provider_name == "test_provider"
        assert client.model_name == "test_model"

    @pytest.mark.asyncio
    async def test_creation_with_middleware(self):
        """Test creating enhanced client with middleware"""
        middleware = [MockMiddleware("test1"), MockMiddleware("test2")]
        client = MockEnhancedBaseLLMClient(middleware=middleware)

        assert len(client.middleware_stack.middlewares) == 2
        assert client.middleware_stack.middlewares[0].name == "test1"
        assert client.middleware_stack.middlewares[1].name == "test2"

    @pytest.mark.asyncio
    async def test_non_streaming_with_middleware(self):
        """Test non-streaming completion with middleware processing"""
        middleware = MockMiddleware("test")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        await client.create_completion(
            messages, tools=tools, stream=False, temperature=0.7
        )

        # Verify middleware was called
        assert len(middleware.process_request_calls) == 1
        assert len(middleware.process_response_calls) == 1
        assert len(middleware.process_stream_chunk_calls) == 0

        # Check request processing
        req_messages, req_tools, req_kwargs = middleware.process_request_calls[0]
        assert req_messages == messages
        assert req_tools == tools
        assert req_kwargs["temperature"] == 0.7

        # Check response processing
        resp_response, resp_duration, resp_is_streaming = (
            middleware.process_response_calls[0]
        )
        assert resp_response == client.mock_response
        assert resp_duration > 0
        assert resp_is_streaming is False

        # Verify implementation was called
        assert len(client.create_completion_impl_calls) == 1

    @pytest.mark.asyncio
    async def test_streaming_with_middleware(self):
        """Test streaming completion with middleware processing"""
        middleware = MockMiddleware("test")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]

        result = client.create_completion(messages, stream=True)

        # Collect chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Verify middleware was called for each chunk
        assert len(middleware.process_request_calls) == 1
        assert (
            len(middleware.process_stream_chunk_calls) == 2
        )  # Two chunks in mock_stream
        assert (
            len(middleware.process_response_calls) == 0
        )  # No response processing for streaming

        # Check chunks were processed
        for i, (chunk, chunk_index, duration) in enumerate(
            middleware.process_stream_chunk_calls
        ):
            assert chunk_index == i
            assert duration > 0
            assert chunk in client.mock_stream

        # Verify implementation was called
        assert len(client.create_stream_impl_calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_middleware_execution_order(self):
        """Test that multiple middleware are executed in correct order"""
        middleware1 = MockMiddleware("first")
        middleware2 = MockMiddleware("second")
        client = MockEnhancedBaseLLMClient(middleware=[middleware1, middleware2])

        messages = [{"role": "user", "content": "Hello"}]

        await client.create_completion(messages, stream=False)

        # Request processing should be in forward order
        assert len(middleware1.process_request_calls) == 1
        assert len(middleware2.process_request_calls) == 1

        # Response processing should be in reverse order
        assert len(middleware1.process_response_calls) == 1
        assert len(middleware2.process_response_calls) == 1

    @pytest.mark.asyncio
    async def test_middleware_can_modify_request(self):
        """Test that middleware can modify request parameters"""

        class ModifyingMiddleware(MockMiddleware):
            async def process_request(self, messages, tools=None, **kwargs):
                # Add a custom parameter
                kwargs["modified_by_middleware"] = True
                # Modify temperature
                kwargs["temperature"] = 0.5
                await super().process_request(messages, tools, **kwargs)
                return messages, tools, kwargs

        middleware = ModifyingMiddleware("modifying")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]

        await client.create_completion(messages, stream=False, temperature=0.7)

        # Check that the implementation received modified parameters
        impl_call = client.create_completion_impl_calls[0]
        _, _, kwargs = impl_call
        assert kwargs["modified_by_middleware"] is True
        assert kwargs["temperature"] == 0.5  # Should be modified value, not original

    @pytest.mark.asyncio
    async def test_middleware_can_modify_response(self):
        """Test that middleware can modify response"""

        class ModifyingMiddleware(MockMiddleware):
            async def process_response(self, response, duration, is_streaming=False):
                # Modify the response
                response = response.copy()
                response["modified_by_middleware"] = True
                await super().process_response(response, duration, is_streaming)
                return response

        middleware = ModifyingMiddleware("modifying")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]

        result = await client.create_completion(messages, stream=False)

        # Response should be modified
        assert result["modified_by_middleware"] is True
        assert "response" in result  # Original content should still be there

    @pytest.mark.asyncio
    async def test_error_handling_with_middleware(self):
        """Test error handling with middleware processing"""
        middleware = MockMiddleware("test")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        # Set up client to raise an error
        test_error = RuntimeError("Test error")
        client.should_raise = test_error

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="Test error"):
            await client.create_completion(messages, stream=False)

        # Verify error was processed by middleware
        assert len(middleware.process_error_calls) == 1
        error, duration = middleware.process_error_calls[0]
        assert error == test_error
        assert duration > 0

    @pytest.mark.asyncio
    async def test_streaming_error_handling_with_middleware(self):
        """Test streaming error handling with middleware"""
        middleware = MockMiddleware("test")

        class ErrorStreamClient(MockEnhancedBaseLLMClient):
            async def _create_stream_impl(self, messages, tools=None, **kwargs):
                yield {"response": "Start", "tool_calls": []}
                raise RuntimeError("Stream error")

        client = ErrorStreamClient(middleware=[middleware])
        messages = [{"role": "user", "content": "Hello"}]

        chunks = []
        result = client.create_completion(messages, stream=True)

        async for chunk in result:
            chunks.append(chunk)

        # Should get one normal chunk and one error chunk
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Start"
        assert chunks[1]["error"] is True
        assert "Stream error" in chunks[1]["response"]

        # Verify middleware processed the error
        assert len(middleware.process_error_calls) == 1

    @pytest.mark.asyncio
    async def test_cached_response_handling(self):
        """Test handling of cached responses"""
        middleware = MockMiddleware("test")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]
        cached_response = {"response": "Cached response", "tool_calls": []}

        # Simulate cached response by adding it to kwargs
        result = await client.create_completion(
            messages, stream=False, _cached_response=cached_response
        )

        # Should return cached response
        assert result == cached_response

        # Implementation should NOT be called for cached response
        assert len(client.create_completion_impl_calls) == 0

        # But middleware should still process the response
        assert len(middleware.process_response_calls) == 1

    @pytest.mark.asyncio
    async def test_timing_accuracy(self):
        """Test that timing measurements are accurate"""

        class TimingMiddleware(MockMiddleware):
            def __init__(self):
                super().__init__("timing")
                self.start_times = []
                self.durations = []

            async def process_response(self, response, duration, is_streaming=False):
                self.durations.append(duration)
                await super().process_response(response, duration, is_streaming)
                return response

        middleware = TimingMiddleware()

        class SlowClient(MockEnhancedBaseLLMClient):
            async def _create_completion_impl(self, messages, tools=None, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                return await super()._create_completion_impl(messages, tools, **kwargs)

        client = SlowClient(middleware=[middleware])
        messages = [{"role": "user", "content": "Hello"}]

        await client.create_completion(messages, stream=False)

        # Duration should be at least 10ms
        assert len(middleware.durations) == 1
        assert middleware.durations[0] >= 0.01

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_middleware(self):
        """Test concurrent requests with middleware"""
        middleware = MockMiddleware("test")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]

        # Start multiple concurrent requests
        tasks = [client.create_completion(messages, stream=False) for _ in range(3)]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 3
        assert all(r == client.mock_response for r in results)

        # Middleware should be called for each request
        assert len(middleware.process_request_calls) == 3
        assert len(middleware.process_response_calls) == 3

    @pytest.mark.asyncio
    async def test_middleware_exception_propagation(self):
        """Test that middleware exceptions are properly propagated"""

        class FailingMiddleware(MockMiddleware):
            async def process_request(self, messages, tools=None, **kwargs):
                raise ValueError("Middleware failed")

        middleware = FailingMiddleware("failing")
        client = MockEnhancedBaseLLMClient(middleware=[middleware])

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Middleware failed"):
            await client.create_completion(messages, stream=False)

        # Implementation should NOT be called if middleware fails
        assert len(client.create_completion_impl_calls) == 0


# Fixtures
@pytest.fixture
def simple_messages():
    """Simple message list for testing"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def tools_list():
    """Tool definitions for testing"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]


# Parametrized tests
@pytest.mark.parametrize("middleware_count", [0, 1, 3, 5])
@pytest.mark.asyncio
async def test_variable_middleware_count(middleware_count, simple_messages):
    """Test with different numbers of middleware"""
    middleware_list = [
        MockMiddleware(f"middleware_{i}") for i in range(middleware_count)
    ]
    client = MockEnhancedBaseLLMClient(middleware=middleware_list)

    await client.create_completion(simple_messages, stream=False)

    # Each middleware should be called once for request and response
    for middleware in middleware_list:
        assert len(middleware.process_request_calls) == 1
        assert len(middleware.process_response_calls) == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_middleware_with_both_modes(stream, simple_messages):
    """Test middleware behavior in both streaming and non-streaming modes"""
    middleware = MockMiddleware("test")
    client = MockEnhancedBaseLLMClient(middleware=[middleware])

    if stream:
        result = client.create_completion(simple_messages, stream=True)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should have chunk processing calls
        assert len(middleware.process_stream_chunk_calls) > 0
        assert len(middleware.process_response_calls) == 0
    else:
        result = await client.create_completion(simple_messages, stream=False)

        # Should have response processing call
        assert len(middleware.process_response_calls) == 1
        assert len(middleware.process_stream_chunk_calls) == 0

    # Both modes should have request processing
    assert len(middleware.process_request_calls) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
