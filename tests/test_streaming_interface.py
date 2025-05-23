# tests/test_streaming_interface.py
"""
Universal tests for streaming interface compliance across all providers.
"""
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

# Test the interface compliance for all providers
@pytest.mark.parametrize("provider_class,provider_name", [
    ("chuk_llm.llm.providers.openai_client.OpenAILLMClient", "openai"),
    ("chuk_llm.llm.providers.anthropic_client.AnthropicLLMClient", "anthropic"), 
    ("chuk_llm.llm.providers.gemini_client.GeminiLLMClient", "gemini"),
    ("chuk_llm.llm.providers.groq_client.GroqAILLMClient", "groq"),
    ("chuk_llm.llm.providers.ollama_client.OllamaLLMClient", "ollama"),
])
def test_streaming_interface_compliance(provider_class, provider_name):
    """Test that all providers follow the correct streaming interface."""
    
    # Skip if provider not available
    try:
        module_path, class_name = provider_class.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        ClientClass = getattr(module, class_name)
    except (ImportError, AttributeError):
        pytest.skip(f"Provider {provider_name} not available")
    
    # Create client instance (with minimal config)
    if provider_name == "gemini":
        client = ClientClass(api_key="fake-key")
    elif provider_name == "ollama":
        client = ClientClass(model="test-model")
    else:
        client = ClientClass(api_key="fake-key")
    
    messages = [{"role": "user", "content": "test"}]
    
    # Test streaming interface
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, '__aiter__'), f"{provider_name} streaming should return async generator"
    assert not hasattr(stream_result, '__await__'), f"{provider_name} streaming should not be awaitable"
    
    # Test non-streaming interface  
    non_stream_result = client.create_completion(messages, stream=False)
    assert hasattr(non_stream_result, '__await__'), f"{provider_name} non-streaming should be awaitable"
    assert not hasattr(non_stream_result, '__aiter__'), f"{provider_name} non-streaming should not be async generator"

# Test that the base interface is correctly defined
def test_base_interface():
    """Test that BaseLLMClient has the correct interface."""
    from chuk_llm.llm.providers.base import BaseLLMClient
    
    # Check that create_completion is not async
    import inspect
    create_completion_method = getattr(BaseLLMClient, 'create_completion')
    assert not inspect.iscoroutinefunction(create_completion_method), \
        "BaseLLMClient.create_completion should not be async"
    
    # Check method signature
    sig = inspect.signature(create_completion_method)
    assert 'stream' in sig.parameters, "create_completion should have stream parameter"
    assert sig.parameters['stream'].default is False, "stream should default to False"

# Test streaming behavior patterns
@pytest.mark.asyncio
async def test_streaming_behavior_pattern():
    """Test common streaming behavior patterns."""
    
    # Mock a streaming client
    class MockStreamingClient:
        def create_completion(self, messages, stream=False, **kwargs):
            if stream:
                return self._stream_completion()
            else:
                return self._regular_completion()
        
        async def _stream_completion(self):
            """Mock streaming that yields multiple chunks."""
            yield {"response": "Hello", "tool_calls": []}
            yield {"response": " there", "tool_calls": []}
            yield {"response": "!", "tool_calls": []}
        
        async def _regular_completion(self):
            """Mock non-streaming response."""
            return {"response": "Hello there!", "tool_calls": []}
    
    client = MockStreamingClient()
    messages = [{"role": "user", "content": "test"}]
    
    # Test streaming
    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " there"
    assert chunks[2]["response"] == "!"
    
    # Test non-streaming
    non_stream_result = client.create_completion(messages, stream=False)
    result = await non_stream_result
    assert result["response"] == "Hello there!"

# Test error handling in streaming
@pytest.mark.asyncio 
async def test_streaming_error_handling():
    """Test that streaming properly handles errors."""
    
    class MockErrorClient:
        def create_completion(self, messages, stream=False, **kwargs):
            if stream:
                return self._error_stream()
            else:
                return self._error_completion()
        
        async def _error_stream(self):
            """Mock streaming that yields error."""
            yield {"response": "Starting...", "tool_calls": []}
            yield {"response": "Error occurred", "tool_calls": [], "error": True}
        
        async def _error_completion(self):
            """Mock non-streaming error."""
            return {"response": "Error occurred", "tool_calls": [], "error": True}
    
    client = MockErrorClient()
    messages = [{"role": "user", "content": "test"}]
    
    # Test streaming error
    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[1].get("error") is True
    
    # Test non-streaming error
    non_stream_result = client.create_completion(messages, stream=False)
    result = await non_stream_result
    assert result.get("error") is True

# Test tool calls in streaming
@pytest.mark.asyncio
async def test_streaming_tool_calls():
    """Test that tool calls work correctly in streaming."""
    
    class MockToolClient:
        def create_completion(self, messages, stream=False, tools=None, **kwargs):
            if stream:
                return self._stream_with_tools()
            else:
                return self._completion_with_tools()
        
        async def _stream_with_tools(self):
            """Mock streaming with tool calls."""
            yield {"response": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}
            ]}
            yield {"response": "Tool result processed", "tool_calls": []}
        
        async def _completion_with_tools(self):
            """Mock non-streaming with tool calls."""
            return {
                "response": "Tool result processed", 
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}
                ]
            }
    
    client = MockToolClient()
    messages = [{"role": "user", "content": "test"}]
    tools = [{"function": {"name": "test_tool"}}]
    
    # Test streaming with tools
    stream_result = client.create_completion(messages, stream=True, tools=tools)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert len(chunks[0]["tool_calls"]) == 1
    assert chunks[0]["tool_calls"][0]["function"]["name"] == "test_tool"
    
    # Test non-streaming with tools
    non_stream_result = client.create_completion(messages, stream=False, tools=tools)
    result = await non_stream_result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"