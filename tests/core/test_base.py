# tests/core/test_base.py
"""
Unit tests for BaseLLMClient abstract base class
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from chuk_llm.llm.core.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """Concrete implementation of BaseLLMClient for testing"""

    def __init__(self, mock_response=None, mock_stream=None, should_raise=None):
        self.mock_response = mock_response or {
            "response": "Test response",
            "tool_calls": [],
        }
        self.mock_stream = mock_stream or [
            {"response": "Hello ", "tool_calls": []},
            {"response": "world", "tool_calls": []},
        ]
        self.should_raise = should_raise
        self.call_count = 0
        self.last_call_args = None
        self.last_call_kwargs = None

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]] | Any:
        # Record call details
        self.call_count += 1
        self.last_call_args = (messages, tools)
        self.last_call_kwargs = kwargs

        if self.should_raise:
            raise self.should_raise

        if stream:
            return self._mock_stream_response()
        else:
            return self._mock_regular_response()

    async def _mock_stream_response(self):
        """Mock streaming response"""
        for chunk in self.mock_stream:
            yield chunk
            await asyncio.sleep(0.001)  # Small delay to simulate real streaming

    async def _mock_regular_response(self):
        """Mock non-streaming response"""
        await asyncio.sleep(0.001)  # Small delay to simulate API call
        return self.mock_response


class TestBaseLLMClient:
    """Test suite for BaseLLMClient abstract base class"""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseLLMClient cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLLMClient()

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated"""
        client = MockLLMClient()
        assert isinstance(client, BaseLLMClient)
        assert hasattr(client, "create_completion")

    @pytest.mark.asyncio
    async def test_non_streaming_completion(self):
        """Test non-streaming completion call"""
        mock_response = {"response": "Hello, world!", "tool_calls": []}
        client = MockLLMClient(mock_response=mock_response)

        messages = [{"role": "user", "content": "Hello"}]

        result = client.create_completion(messages, stream=False)

        # Should return a coroutine for non-streaming
        assert asyncio.iscoroutine(result) or hasattr(result, "__await__")

        # Await the result
        response = await result

        assert response == mock_response
        assert client.call_count == 1
        assert client.last_call_args == (messages, None)
        assert client.last_call_kwargs == {}

    @pytest.mark.asyncio
    async def test_streaming_completion(self):
        """Test streaming completion call"""
        mock_stream = [
            {"response": "Hello ", "tool_calls": []},
            {"response": "world!", "tool_calls": []},
        ]
        client = MockLLMClient(mock_stream=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]

        result = client.create_completion(messages, stream=True)

        # Should return an async iterator for streaming
        assert hasattr(result, "__aiter__")

        # Collect chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert chunks == mock_stream
        assert client.call_count == 1
        assert client.last_call_args == (messages, None)

    @pytest.mark.asyncio
    async def test_completion_with_tools(self):
        """Test completion with tools parameter"""
        client = MockLLMClient()

        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
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

        await client.create_completion(messages, tools=tools, stream=False)

        assert client.last_call_args == (messages, tools)

    @pytest.mark.asyncio
    async def test_completion_with_kwargs(self):
        """Test completion with additional keyword arguments"""
        client = MockLLMClient()

        messages = [{"role": "user", "content": "Hello"}]

        await client.create_completion(
            messages, stream=False, temperature=0.7, max_tokens=100, custom_param="test"
        )

        expected_kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "custom_param": "test",
        }

        assert client.last_call_kwargs == expected_kwargs

    @pytest.mark.asyncio
    async def test_error_handling_non_streaming(self):
        """Test error handling in non-streaming mode"""
        error = RuntimeError("API Error")
        client = MockLLMClient(should_raise=error)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="API Error"):
            await client.create_completion(messages, stream=False)

    @pytest.mark.asyncio
    async def test_error_handling_streaming(self):
        """Test error handling in streaming mode"""
        error = RuntimeError("Streaming Error")
        client = MockLLMClient(should_raise=error)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="Streaming Error"):
            result = client.create_completion(messages, stream=True)
            async for _chunk in result:
                pass  # This should raise the error

    @pytest.mark.asyncio
    async def test_multiple_calls_increment_counter(self):
        """Test that multiple calls increment the call counter"""
        client = MockLLMClient()
        messages = [{"role": "user", "content": "Hello"}]

        await client.create_completion(messages, stream=False)
        await client.create_completion(messages, stream=False)

        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_streaming_chunks_arrive_sequentially(self):
        """Test that streaming chunks arrive in the correct order"""
        mock_stream = [
            {"response": "First ", "tool_calls": []},
            {"response": "second ", "tool_calls": []},
            {"response": "third", "tool_calls": []},
        ]
        client = MockLLMClient(mock_stream=mock_stream)

        messages = [{"role": "user", "content": "Count"}]
        result = client.create_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk["response"])

        assert chunks == ["First ", "second ", "third"]

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self):
        """Test streaming with tool calls"""
        mock_stream = [
            {
                "response": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {"response": "Based on the weather data...", "tool_calls": []},
        ]
        client = MockLLMClient(mock_stream=mock_stream)

        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        result = client.create_completion(messages, tools=tools, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["tool_calls"] != []
        assert chunks[1]["response"] == "Based on the weather data..."

    def test_method_signature_compatibility(self):
        """Test that the create_completion method has the expected signature"""
        import inspect

        sig = inspect.signature(BaseLLMClient.create_completion)
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "self" in params
        assert "messages" in params

        # Check optional parameters
        tools_param = sig.parameters.get("tools")
        assert tools_param is not None
        assert tools_param.default is None

        # Check keyword-only parameters
        stream_param = sig.parameters.get("stream")
        assert stream_param is not None
        assert stream_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert stream_param.default is False

        # Check **kwargs
        assert any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    @pytest.mark.asyncio
    async def test_concurrent_streaming_calls(self):
        """Test that multiple streaming calls can run concurrently"""
        client = MockLLMClient()
        messages = [{"role": "user", "content": "Hello"}]

        # Start multiple streaming calls concurrently
        task1 = asyncio.create_task(
            self._collect_stream(client.create_completion(messages, stream=True))
        )
        task2 = asyncio.create_task(
            self._collect_stream(client.create_completion(messages, stream=True))
        )
        task3 = asyncio.create_task(
            self._collect_stream(client.create_completion(messages, stream=True))
        )

        # Wait for all to complete
        results = await asyncio.gather(task1, task2, task3)

        # All should complete successfully
        assert len(results) == 3
        assert all(len(result) > 0 for result in results)
        assert client.call_count == 3

    async def _collect_stream(self, stream):
        """Helper to collect all chunks from a stream"""
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        """Test behavior with empty messages list"""
        client = MockLLMClient()

        # Should not raise an error - let the implementation decide
        result = await client.create_completion([], stream=False)
        assert result is not None

    @pytest.mark.asyncio
    async def test_none_tools_parameter(self):
        """Test that None tools parameter is handled correctly"""
        client = MockLLMClient()
        messages = [{"role": "user", "content": "Hello"}]

        # Explicitly pass None for tools
        await client.create_completion(messages, tools=None, stream=False)

        assert client.last_call_args == (messages, None)


# Fixtures for common test data
@pytest.fixture
def simple_messages():
    """Simple message list for testing"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def messages_with_tools():
    """Messages and tools for function calling tests"""
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    return messages, tools


@pytest.fixture
def complex_mock_response():
    """Complex mock response with tool calls"""
    return {
        "response": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA"}',
                },
            }
        ],
    }


# Parametrized tests for different scenarios
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_completion_modes(stream, simple_messages):
    """Test both streaming and non-streaming modes"""
    client = MockLLMClient()

    result = client.create_completion(simple_messages, stream=stream)

    if stream:
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) > 0
    else:
        response = await result
        assert "response" in response
        assert "tool_calls" in response


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.asyncio
async def test_tools_parameter_variations(tools, simple_messages):
    """Test different tools parameter values"""
    client = MockLLMClient()

    await client.create_completion(simple_messages, tools=tools, stream=False)

    assert client.last_call_args == (simple_messages, tools)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
