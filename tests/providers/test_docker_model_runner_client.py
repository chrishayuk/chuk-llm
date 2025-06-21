# tests/providers/test_docker_model_runner_client.py
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub the `openai` SDK before importing the adapter.
# ---------------------------------------------------------------------------

openai_mod = types.ModuleType("openai")
sys.modules["openai"] = openai_mod

# Mock OpenAI response classes for Docker Model Runner
class MockDockerChatCompletionChunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [types.SimpleNamespace(
            delta=types.SimpleNamespace(
                content=content,
                tool_calls=tool_calls
            )
        )]
        self.model = "ai/smollm2"
        self.id = "chatcmpl-docker-test"

class MockDockerChatCompletion:
    def __init__(self, content="Test response from Docker Model Runner", tool_calls=None):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(
                content=content,
                tool_calls=tool_calls
            )
        )]
        self.model = "ai/smollm2"
        self.id = "chatcmpl-docker-test"

# Mock AsyncOpenAI client for Docker Model Runner
class MockDockerAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create = AsyncMock()
        # Store the base_url for verification
        self.base_url = kwargs.get('base_url', 'http://localhost:12434/engines/llama.cpp/v1')

# Mock sync OpenAI client for Docker Model Runner
class MockDockerOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create = MagicMock()
        # Store the base_url for verification
        self.base_url = kwargs.get('base_url', 'http://localhost:12434/engines/llama.cpp/v1')

openai_mod.AsyncOpenAI = MockDockerAsyncOpenAI
openai_mod.OpenAI = MockDockerOpenAI

# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.openai_client import OpenAILLMClient  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def docker_client():
    """Create a Docker Model Runner client instance."""
    return OpenAILLMClient(
        model="ai/smollm2",
        api_key="docker-model-runner",
        api_base="http://localhost:12434/engines/llama.cpp/v1"
    )

@pytest.fixture
def docker_llama_client():
    """Create a Docker Model Runner client with Llama model."""
    return OpenAILLMClient(
        model="ai/llama3.2",
        api_key="docker-model-runner",
        api_base="http://localhost:12434/engines/llama.cpp/v1"
    )

@pytest.fixture
def docker_code_client():
    """Create a Docker Model Runner client with CodeLlama model."""
    return OpenAILLMClient(
        model="ai/codellama",
        api_key="docker-model-runner",
        api_base="http://localhost:12434/engines/llama.cpp/v1"
    )

# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_create_completion_non_stream(docker_client):
    """Test non-streaming completion with Docker Model Runner."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is Python programming?"},
    ]
    
    # Mock the async client response
    mock_response = MockDockerChatCompletion(content="Python is a high-level programming language.")
    docker_client.async_client.chat.completions.create.return_value = mock_response
    
    # Get the awaitable result
    result_awaitable = docker_client.create_completion(messages, stream=False)
    assert hasattr(result_awaitable, '__await__')
    
    # Await the result
    result = await result_awaitable
    
    # Verify response format
    assert result["response"] == "Python is a high-level programming language."
    assert result["tool_calls"] == []
    
    # Verify the API was called correctly
    docker_client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = docker_client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "ai/smollm2"
    assert call_kwargs["stream"] is False
    assert len(call_kwargs["messages"]) == 2

@pytest.mark.asyncio
async def test_docker_different_models(docker_llama_client, docker_code_client):
    """Test Docker Model Runner with different models."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Test Llama model
    mock_llama_response = MockDockerChatCompletion(content="Hello from Llama!")
    mock_llama_response.model = "ai/llama3.2"
    docker_llama_client.async_client.chat.completions.create.return_value = mock_llama_response
    
    result = await docker_llama_client.create_completion(messages, stream=False)
    assert result["response"] == "Hello from Llama!"
    
    # Test CodeLlama model
    mock_code_response = MockDockerChatCompletion(content="# Python code example")
    mock_code_response.model = "ai/codellama"
    docker_code_client.async_client.chat.completions.create.return_value = mock_code_response
    
    result = await docker_code_client.create_completion(messages, stream=False)
    assert result["response"] == "# Python code example"

@pytest.mark.asyncio
async def test_docker_api_configuration(docker_client):
    """Test that Docker Model Runner is configured with correct endpoint."""
    # Verify the client is configured with Docker Model Runner endpoint
    assert docker_client.async_client.base_url == "http://localhost:12434/engines/llama.cpp/v1"
    
    # Test that fake API key works
    messages = [{"role": "user", "content": "Test"}]
    mock_response = MockDockerChatCompletion(content="Test response")
    docker_client.async_client.chat.completions.create.return_value = mock_response
    
    result = await docker_client.create_completion(messages, stream=False)
    assert result["response"] == "Test response"

# ---------------------------------------------------------------------------
# Streaming tests  
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_create_completion_stream(docker_client):
    """Test streaming completion with Docker Model Runner."""
    messages = [{"role": "user", "content": "Count from 1 to 5"}]
    
    # Mock streaming response
    async def mock_docker_stream():
        yield MockDockerChatCompletionChunk(content="1")
        yield MockDockerChatCompletionChunk(content=", 2") 
        yield MockDockerChatCompletionChunk(content=", 3")
        yield MockDockerChatCompletionChunk(content=", 4")
        yield MockDockerChatCompletionChunk(content=", 5")
    
    docker_client.async_client.chat.completions.create.return_value = mock_docker_stream()
    
    # Get the async generator directly
    stream_result = docker_client.create_completion(messages, stream=True)
    assert hasattr(stream_result, '__aiter__')
    assert not hasattr(stream_result, '__await__')
    
    # Collect chunks
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 5
    assert chunks[0]["response"] == "1"
    assert chunks[1]["response"] == ", 2"
    assert chunks[2]["response"] == ", 3"
    assert chunks[3]["response"] == ", 4"
    assert chunks[4]["response"] == ", 5"
    
    # Verify API call
    docker_client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = docker_client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True

@pytest.mark.asyncio
async def test_docker_streaming_with_different_models(docker_llama_client):
    """Test streaming with different Docker models."""
    messages = [{"role": "user", "content": "Write a haiku"}]
    
    # Mock streaming response from Llama
    async def mock_llama_stream():
        yield MockDockerChatCompletionChunk(content="Code flows like water")
        yield MockDockerChatCompletionChunk(content="\nThrough silicon pathways")
        yield MockDockerChatCompletionChunk(content="\nDigital poetry")
    
    docker_llama_client.async_client.chat.completions.create.return_value = mock_llama_stream()
    
    # Get streaming result
    stream_result = docker_llama_client.create_completion(messages, stream=True)
    
    # Collect chunks
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Code flows like water"
    assert chunks[1]["response"] == "\nThrough silicon pathways"
    assert chunks[2]["response"] == "\nDigital poetry"

# ---------------------------------------------------------------------------
# Docker-specific tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_fake_api_key_handling():
    """Test that Docker Model Runner works with fake API key."""
    # Test with the standard fake API key
    client = OpenAILLMClient(
        model="ai/smollm2",
        api_key="docker-model-runner",
        api_base="http://localhost:12434/engines/llama.cpp/v1"
    )
    
    messages = [{"role": "user", "content": "Test fake API key"}]
    mock_response = MockDockerChatCompletion(content="Fake API key works!")
    client.async_client.chat.completions.create.return_value = mock_response
    
    result = await client.create_completion(messages, stream=False)
    assert result["response"] == "Fake API key works!"

@pytest.mark.asyncio
async def test_docker_local_endpoint_configuration():
    """Test Docker Model Runner local endpoint configuration."""
    client = OpenAILLMClient(
        model="ai/smollm2",
        api_key="any-fake-key",
        api_base="http://localhost:12434/engines/llama.cpp/v1"
    )
    
    # Verify the endpoint is configured correctly
    assert "localhost:12434" in client.async_client.base_url
    assert "engines/llama.cpp/v1" in client.async_client.base_url

@pytest.mark.asyncio
async def test_docker_model_variations():
    """Test various Docker Hub AI models."""
    models_to_test = [
        "ai/smollm2",
        "ai/llama3.2", 
        "ai/phi3",
        "ai/gemma2",
        "ai/qwen2.5",
        "ai/mistral",
        "ai/codellama"
    ]
    
    for model in models_to_test:
        client = OpenAILLMClient(
            model=model,
            api_key="docker-model-runner",
            api_base="http://localhost:12434/engines/llama.cpp/v1"
        )
        
        messages = [{"role": "user", "content": f"Test {model}"}]
        mock_response = MockDockerChatCompletion(content=f"Response from {model}")
        mock_response.model = model
        client.async_client.chat.completions.create.return_value = mock_response
        
        result = await client.create_completion(messages, stream=False)
        assert result["response"] == f"Response from {model}"
        
        # Verify model was passed correctly
        call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == model

# ---------------------------------------------------------------------------
# Interface compliance tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_interface_compliance(docker_client):
    """Test that Docker Model Runner client follows the BaseLLMClient interface."""
    # Test non-streaming
    messages = [{"role": "user", "content": "Test interface"}]
    
    mock_response = MockDockerChatCompletion(content="Interface test response")
    docker_client.async_client.chat.completions.create.return_value = mock_response
    
    # For non-streaming, we need to await the result if it's a coroutine
    result = docker_client.create_completion(messages, stream=False)
    if asyncio.iscoroutine(result):
        result = await result
    
    assert isinstance(result, dict)
    assert "response" in result
    assert result["response"] == "Interface test response"
    
    # Test streaming returns async iterator
    async def mock_stream():
        yield MockDockerChatCompletionChunk(content="Stream test")
    
    docker_client.async_client.chat.completions.create.return_value = mock_stream()
    stream_result = docker_client.create_completion(messages, stream=True)
    assert hasattr(stream_result, "__aiter__")

# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_connection_error_handling(docker_client):
    """Test error handling when Docker Model Runner is not accessible."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock a connection error (Docker not running)
    docker_client.async_client.chat.completions.create.side_effect = Exception("Connection refused - Docker Model Runner not accessible")
    
    result_awaitable = docker_client.create_completion(messages, stream=False)
    result = await result_awaitable
    
    assert "error" in result
    assert "Connection refused" in result["response"]

@pytest.mark.asyncio
async def test_docker_streaming_error_handling(docker_client):
    """Test error handling in Docker Model Runner streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock an exception during streaming (model not available)
    async def mock_error_stream():
        yield MockDockerChatCompletionChunk(content="Starting...")
        raise Exception("Model ai/smollm2 not found - please pull the model first")
    
    docker_client.async_client.chat.completions.create.return_value = mock_error_stream()
    
    # Get streaming result
    stream_result = docker_client.create_completion(messages, stream=True)
    
    # Should yield error chunk
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Should have at least one chunk and an error chunk
    assert len(chunks) >= 1
    error_chunk = chunks[-1] 
    assert "error" in error_chunk
    assert "Model ai/smollm2 not found" in error_chunk["response"]

@pytest.mark.asyncio
async def test_docker_model_not_available_error(docker_client):
    """Test error when requested model is not available in Docker."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock model not available error
    docker_client.async_client.chat.completions.create.side_effect = Exception("Model 'ai/smollm2' not found. Available models: []")
    
    result = await docker_client.create_completion(messages, stream=False)
    
    assert "error" in result
    assert "Model 'ai/smollm2' not found" in result["response"]

# ---------------------------------------------------------------------------
# Docker-specific feature tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_no_tools_support(docker_client):
    """Test that Docker Model Runner doesn't support function calling (as per config)."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object"}
            }
        }
    ]
    
    # Docker Model Runner should handle tools gracefully but not use them
    mock_response = MockDockerChatCompletion(content="I can't access real-time weather data, but I can help you understand weather concepts.")
    docker_client.async_client.chat.completions.create.return_value = mock_response
    
    result = await docker_client.create_completion(messages, tools=tools, stream=False)
    
    # Should return text response, not tool calls
    assert result["response"] == "I can't access real-time weather data, but I can help you understand weather concepts."
    assert result["tool_calls"] == []

@pytest.mark.asyncio
async def test_docker_local_privacy_features(docker_client):
    """Test that Docker Model Runner maintains local privacy."""
    messages = [
        {"role": "user", "content": "This is sensitive information that should stay local"}
    ]
    
    mock_response = MockDockerChatCompletion(content="I understand. All processing happens locally on your machine.")
    docker_client.async_client.chat.completions.create.return_value = mock_response
    
    result = await docker_client.create_completion(messages, stream=False)
    
    # Verify the endpoint is local
    assert "localhost" in docker_client.async_client.base_url
    assert result["response"] == "I understand. All processing happens locally on your machine."

# ---------------------------------------------------------------------------
# Performance and configuration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_no_rate_limits():
    """Test that Docker Model Runner has no rate limits (local processing)."""
    client = OpenAILLMClient(
        model="ai/smollm2",
        api_key="docker-model-runner",
        api_base="http://localhost:12434/engines/llama.cpp/v1"
    )
    
    messages = [{"role": "user", "content": "Quick test"}]
    mock_response = MockDockerChatCompletion(content="Quick response")
    
    # Simulate multiple rapid requests (should not be rate limited)
    for i in range(5):
        client.async_client.chat.completions.create.return_value = mock_response
        result = await client.create_completion(messages, stream=False)
        assert result["response"] == "Quick response"

@pytest.mark.asyncio
async def test_docker_custom_parameters(docker_client):
    """Test Docker Model Runner with custom parameters."""
    messages = [{"role": "user", "content": "Creative writing test"}]
    
    mock_response = MockDockerChatCompletion(content="Creative response with high temperature")
    docker_client.async_client.chat.completions.create.return_value = mock_response
    
    # Test with custom parameters
    result = await docker_client.create_completion(
        messages, 
        stream=False,
        temperature=0.9,
        max_tokens=100
    )
    
    assert result["response"] == "Creative response with high temperature"
    
    # Verify parameters were passed
    call_kwargs = docker_client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.9
    assert call_kwargs.get("max_tokens") == 100
